# SPDX-FileCopyrightText: Andreas Mueller
# SPDX-FileCopyrightText: Joris Van den Bossche
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Original authors from Sckit-Learn:
#          Andreas Mueller
#          Joris Van den Bossche
# License: BSD

# This code originates from the Scikit-Learn library,
# it was since modified to allow GPU acceleration.
# This code is under BSD 3 clause license.
# Authors mentioned above do not endorse or promote this production.


import functools
import numbers
import timeit
import warnings
from collections import defaultdict
from contextlib import contextmanager
from itertools import chain, compress

import cudf
import cupy as np
import numpy as cpu_np
import numba
import pandas as pd
import scipy.sparse as sp_sparse
from cupyx.scipy import sparse as cu_sparse
from joblib import Parallel
from pandas.api.types import is_bool_dtype
from sklearn.base import clone
from sklearn.utils import Bunch

import cuml
from cuml.internals.global_settings import _global_settings_data
from cuml.internals.mixins import DeprecatedGetFeatureNamesMixin
from cuml.internals.validation import (
    check_is_fitted,
    check_features,
    check_array,
    check_input_features,
)

from ..preprocessing._function_transformer import FunctionTransformer
from ..utils.skl_dependencies import (
    BaseComposition,
    BaseEstimator,
    TransformerMixin,
)

_ERR_MSG_1DCOLUMN = ("1D data passed to a transformer that expects 2D data. "
                     "Try to specify the column selection as a list of one "
                     "item instead of a scalar.")


def issparse(X):
    return sp_sparse.issparse(X) or cu_sparse.issparse(X)


def _determine_key_type(key, accept_slice=True):
    """Determine the data type of key.

    Parameters
    ----------
    key : scalar, slice or array-like
        The key from which we want to infer the data type.

    accept_slice : bool, default=True
        Whether or not to raise an error if the key is a slice.

    Returns
    -------
    dtype : {'int', 'str', 'bool', None}
        Returns the data type of key.
    """
    err_msg = ("No valid specification of the columns. Only a scalar, list or "
               "slice of all integers or all strings, or boolean mask is "
               "allowed")

    dtype_to_str = {int: 'int', str: 'str', bool: 'bool', np.bool_: 'bool'}
    array_dtype_to_str = {'i': 'int', 'u': 'int', 'b': 'bool', 'O': 'str',
                          'U': 'str', 'S': 'str'}

    if key is None:
        return None
    if isinstance(key, tuple(dtype_to_str.keys())):
        try:
            return dtype_to_str[type(key)]
        except KeyError:
            raise ValueError(err_msg)
    if isinstance(key, slice):
        if not accept_slice:
            raise TypeError(
                'Only array-like or scalar are supported. '
                'A Python slice was given.'
            )
        if key.start is None and key.stop is None:
            return None
        key_start_type = _determine_key_type(key.start)
        key_stop_type = _determine_key_type(key.stop)
        if key_start_type is not None and key_stop_type is not None:
            if key_start_type != key_stop_type:
                raise ValueError(err_msg)
        if key_start_type is not None:
            return key_start_type
        return key_stop_type
    if isinstance(key, (list, tuple)):
        unique_key = set(key)
        key_type = {_determine_key_type(elt) for elt in unique_key}
        if not key_type:
            return None
        if len(key_type) != 1:
            raise ValueError(err_msg)
        return key_type.pop()
    if hasattr(key, 'dtype'):
        try:
            return array_dtype_to_str[key.dtype.kind]
        except KeyError:
            raise ValueError(err_msg)
    raise ValueError(err_msg)


def _get_column_indices(X, key):
    """Get feature column indices for input data X and key.
    """
    n_columns = X.shape[1]

    key_dtype = _determine_key_type(key)

    if isinstance(key, (list, tuple)) and not key:
        # we get an empty list
        return []
    elif key_dtype in ('bool', 'int'):
        # Convert key into positive indexes
        try:
            idx = _safe_indexing(np.arange(n_columns), key)
        except IndexError as e:
            raise ValueError(
                'all features must be in [0, {}] or [-{}, 0]'
                .format(n_columns - 1, n_columns)
            ) from e
        return np.atleast_1d(idx).tolist()
    elif key_dtype == 'str':
        try:
            all_columns = X.columns
        except AttributeError:
            raise ValueError("Specifying the columns using strings is only "
                             "supported for pandas DataFrames")
        if isinstance(key, str):
            columns = [key]
        elif isinstance(key, slice):
            start, stop = key.start, key.stop
            if start is not None:
                start = all_columns.get_loc(start)
            if stop is not None:
                # pandas indexing with strings is endpoint included
                stop = all_columns.get_loc(stop) + 1
            else:
                stop = n_columns + 1
            return list(range(n_columns)[slice(start, stop)])
        else:
            columns = list(key)

        try:
            column_indices = []
            for col in columns:
                col_idx = all_columns.get_loc(col)
                if not isinstance(col_idx, numbers.Integral):
                    raise ValueError(f"Selected columns, {columns}, are not "
                                     "unique in dataframe")
                column_indices.append(col_idx)

        except KeyError as e:
            raise ValueError(
                "A given column is not a column of the dataframe"
            ) from e

        return column_indices
    else:
        raise ValueError("No valid specification of the columns. Only a "
                         "scalar, list or slice of all integers or all "
                         "strings, or boolean mask is allowed")


def _safe_indexing(X, indices, *, axis=0):
    """Return rows, items or columns of X using indices.

    Parameters
    ----------
    X : array-like, sparse-matrix, list, dataframes, series data
        from which to sample rows, items or columns. `list` are only
        supported when `axis=0`.
    indices : bool, int, str, slice, array-like
        - If `axis=0`, boolean and integer array-like, integer slice,
          and scalar integer are supported.
        - If `axis=1`:
            - to select a single column, `indices` can be of `int` type for
              all `X` types and `str` only for dataframe. The selected subset
              will be 1D, unless `X` is a sparse matrix in which case it will
              be 2D.
            - to select multiples columns, `indices` can be one of the
              following: `list`, `array`, `slice`. The type used in
              these containers can be one of the following: `int`, 'bool' and
              `str`. However, `str` is only supported when `X` is a dataframe.
              The selected subset will be 2D.
    axis : int, default=0
        The axis along which `X` will be subsampled. `axis=0` will select
        rows while `axis=1` will select columns.

    Returns
    -------
    subset
        Subset of X on axis 0 or 1.

    Notes
    -----
    CSR, CSC, and LIL sparse matrices are supported. COO sparse matrices are
    not supported.
    """
    if indices is None:
        return X

    if axis not in (0, 1):
        raise ValueError(
            "'axis' should be either 0 (to index rows) or 1 (to index "
            " column). Got {} instead.".format(axis)
        )

    if isinstance(indices, (pd.Index, cudf.Index)):
        indices = list(indices)

    indices_dtype = _determine_key_type(indices)

    if axis == 0 and indices_dtype == 'str':
        raise ValueError(
            "String indexing is not supported with 'axis=0'"
        )

    if axis == 1 and X.ndim != 2:
        raise ValueError(
            "'X' should be a 2D NumPy array, 2D sparse matrix or pandas "
            "dataframe when indexing the columns (i.e. 'axis=1'). "
            "Got {} instead with {} dimension(s).".format(type(X), X.ndim)
        )

    if axis == 1 and indices_dtype == 'str' and not hasattr(X, 'loc'):
        raise ValueError(
            "Specifying the columns using strings is only supported for "
            "pandas DataFrames"
        )

    if hasattr(X, "iloc"):
        return _pandas_indexing(X, indices, indices_dtype, axis=axis)
    elif hasattr(X, "shape"):
        return _array_indexing(X, indices, indices_dtype, axis=axis)
    else:
        return _list_indexing(X, indices, indices_dtype)


def _array_indexing(array, key, key_dtype, axis):
    """Index an array or a sparse array"""
    if issparse(array):
        # check if we have an boolean array-likes to make the proper indexing
        if key_dtype == 'bool':
            key = np.asarray(key)
    if isinstance(key, tuple):
        key = list(key)
    if numba.cuda.is_cuda_array(array):
        array = np.asarray(array)
    return array[key] if axis == 0 else array[:, key]


def _pandas_indexing(X, key, key_dtype, axis):
    """Index a dataframe or a series"""
    if hasattr(key, 'shape'):
        # Work-around for indexing with read-only key in pandas
        # FIXME: solved in pandas 0.25
        key = key.to_numpy()
        key = key if key.flags.writeable else key.copy()
    elif isinstance(key, tuple):
        key = list(key)
    # check whether we should index with loc or iloc
    indexer = X.iloc if key_dtype == 'int' else X.loc
    return indexer[:, key] if axis else indexer[key]


def _list_indexing(X, key, key_dtype):
    """Index a Python list."""
    if np.isscalar(key) or isinstance(key, slice):
        # key is a slice or a scalar
        return X[key]
    if key_dtype == 'bool':
        # key is a boolean array-like
        return list(compress(X, key))
    # key is a integer array-like of key
    return [X[idx] for idx in key]


def _transform_one(transformer, X, y, weight, **fit_params):
    with cuml.using_output_type("cupy"):
        res = transformer.transform(X)

    # if we have a weight for this transformer, multiply output
    if weight is None:
        return res
    return res * weight


def _fit_transform_one(transformer,
                       X,
                       y,
                       weight,
                       message_clsname='',
                       message=None,
                       **fit_params):
    """
    Fits ``transformer`` to ``X`` and ``y``. The transformed result is returned
    with the fitted transformer. If ``weight`` is not ``None``, the result will
    be multiplied by ``weight``.
    """
    with _print_elapsed_time(message_clsname, message):
        with cuml.using_output_type("cupy"):
            transformer.accept_sparse = True
            if hasattr(transformer, 'fit_transform'):
                res = transformer.fit_transform(X, y, **fit_params)
            else:
                res = transformer.fit(X, y, **fit_params).transform(X)

    if weight is None:
        return res, transformer
    return res * weight, transformer


def _name_estimators(estimators):
    """Generate names for estimators."""

    names = [
        estimator
        if isinstance(estimator, str) else type(estimator).__name__.lower()
        for estimator in estimators
    ]
    namecount = defaultdict(int)
    for est, name in zip(estimators, names):
        namecount[name] += 1

    for k, v in list(namecount.items()):
        if v == 1:
            del namecount[k]

    for i in reversed(range(len(estimators))):
        name = names[i]
        if name in namecount:
            names[i] += "-%d" % namecount[name]
            namecount[name] -= 1

    return list(zip(names, estimators))


def delayed(function):
    """Decorator used to capture the arguments of a function."""
    @functools.wraps(function)
    def delayed_function(*args, **kwargs):
        return _FuncWrapper(function), args, kwargs
    return delayed_function


class _FuncWrapper:
    """"Load the global configuration before calling the function."""

    def __init__(self, function):
        self.function = function
        self.config = _global_settings_data.shared_state
        functools.update_wrapper(self, self.function)

    def __call__(self, *args, **kwargs):
        _global_settings_data.shared_state = self.config
        return self.function(*args, **kwargs)


@contextmanager
def _print_elapsed_time(source, message=None):
    """Log elapsed time to stdout when the context is exited.
    Parameters
    ----------
    source : str
        String indicating the source or the reference of the message.
    message : str, default=None
        Short message. If None, nothing will be printed.
    Returns
    -------
    context_manager
        Prints elapsed time upon exit if verbose.
    """
    if message is None:
        yield
    else:
        start = timeit.default_timer()
        yield
        print(
            _message_with_time(source, message,
                               timeit.default_timer() - start))


def _message_with_time(source, message, time):
    """Create one line message for logging purposes.
    Parameters
    ----------
    source : str
        String indicating the source or the reference of the message.
    message : str
        Short message.
    time : int
        Time in seconds.
    """
    start_message = "[%s] " % source

    # adapted from joblib.logger.short_format_time without the Windows -.1s
    # adjustment
    if time > 60:
        time_str = "%4.1fmin" % (time / 60)
    else:
        time_str = " %5.1fs" % time
    end_message = " %s, total=%s" % (message, time_str)
    dots_len = (70 - len(start_message) - len(end_message))
    return "%s%s%s" % (start_message, dots_len * '.', end_message)


class ColumnTransformer(
    DeprecatedGetFeatureNamesMixin,
    TransformerMixin,
    BaseComposition,
    BaseEstimator,
):
    """Applies transformers to columns of an array or dataframe.

    This estimator allows different columns or column subsets of the input
    to be transformed separately and the features generated by each transformer
    will be concatenated to form a single feature space.
    This is useful for heterogeneous or columnar data, to combine several
    feature extraction mechanisms or transformations into a single transformer.

    Parameters
    ----------
    transformers : list of tuples
        List of (name, transformer, columns) tuples specifying the
        transformer objects to be applied to subsets of the data:

        * name : str
            Like in Pipeline and FeatureUnion, this allows the transformer and
            its parameters to be set using ``set_params`` and searched in grid
            search.
        * transformer : {'drop', 'passthrough'} or estimator
            Estimator must support `fit` and `transform`.
            Special-cased strings 'drop' and 'passthrough' are accepted as
            well, to indicate to drop the columns or to pass them through
            untransformed, respectively.
        * columns :  str, array-like of str, int, array-like of int, \
                array-like of bool, slice or callable
            Indexes the data on its second axis. Integers are interpreted as
            positional columns, while strings can reference DataFrame columns
            by name.  A scalar string or int should be used where
            ``transformer`` expects X to be a 1d array-like (vector),
            otherwise a 2d array will be passed to the transformer.
            A callable is passed the input data `X` and can return any of the
            above. To select multiple columns by name or dtype, you can use
            :obj:`make_column_selector`.

    remainder : {'drop', 'passthrough'} or estimator, default='drop'
        By default, only the specified columns in `transformers` are
        transformed and combined in the output, and the non-specified
        columns are dropped. (default of ``'drop'``).
        By specifying ``remainder='passthrough'``, all remaining columns that
        were not specified in `transformers` will be automatically passed
        through. This subset of columns is concatenated with the output of
        the transformers.
        By setting ``remainder`` to be an estimator, the remaining
        non-specified columns will use the ``remainder`` estimator. The
        estimator must support `fit` and `transform`.
        Note that using this feature requires that the DataFrame columns
        input at `fit` and `transform` have identical order.

    sparse_threshold : float, default=0.3
        If the output of the different transformers contains sparse matrices,
        these will be stacked as a sparse matrix if the overall density is
        lower than this value. Use ``sparse_threshold=0`` to always return
        dense.  When the transformed output consists of all dense data, the
        stacked result will be dense, and this keyword will be ignored.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
        for more details.

    transformer_weights : dict, default=None
        Multiplicative weights for features per transformer. The output of the
        transformer is multiplied by these weights. Keys are transformer names,
        values the weights.

    verbose : bool, default=False
        If True, the time elapsed while fitting each transformer will be
        printed as it is completed.

    Attributes
    ----------
    transformers_ : list
        The collection of fitted transformers as tuples of
        (name, fitted_transformer, column). `fitted_transformer` can be an
        estimator, 'drop', or 'passthrough'. In case there were no columns
        selected, this will be the unfitted transformer.
        If there are remaining columns, the final element is a tuple of the
        form:
        ('remainder', transformer, remaining_columns) corresponding to the
        ``remainder`` parameter. If there are remaining columns, then
        ``len(transformers_)==len(transformers)+1``, otherwise
        ``len(transformers_)==len(transformers)``.

    named_transformers_ : :class:`~sklearn.utils.Bunch`
        Read-only attribute to access any transformer by given name.
        Keys are transformer names and values are the fitted transformer
        objects.

    sparse_output_ : bool
        Boolean flag indicating whether the output of ``transform`` is a
        sparse matrix or a dense numpy array, which depends on the output
        of the individual transformers and the `sparse_threshold` keyword.

    Notes
    -----
    The order of the columns in the transformed feature matrix follows the
    order of how the columns are specified in the `transformers` list.
    Columns of the original feature matrix that are not specified are
    dropped from the resulting transformed feature matrix, unless specified
    in the `passthrough` keyword. Those columns specified with `passthrough`
    are added at the right to the output of the transformers.

    See Also
    --------
    make_column_transformer : Convenience function for
        combining the outputs of multiple transformer objects applied to
        column subsets of the original feature space.
    make_column_selector : Convenience function for selecting
        columns based on datatype or the columns name with a regex pattern.

    Examples
    --------
    >>> import cupy as cp
    >>> from cuml.compose import ColumnTransformer
    >>> from cuml.preprocessing import Normalizer
    >>> ct = ColumnTransformer(
    ...     [("norm1", Normalizer(norm='l1'), [0, 1]),
    ...      ("norm2", Normalizer(norm='l1'), slice(2, 4))])
    >>> X = cp.array([[0., 1., 2., 2.],
    ...               [1., 1., 0., 1.]])
    >>> # Normalizer scales each row of X to unit norm. A separate scaling
    >>> # is applied for the two first and two last elements of each
    >>> # row independently.
    >>> ct.fit_transform(X)
    array([[0. , 1. , 0.5, 0.5],
           [0.5, 0.5, 0. , 1. ]])

    """
    _required_parameters = ['transformers']

    def __init__(self,
                 transformers=None,
                 *,
                 remainder='drop',
                 sparse_threshold=0.3,
                 n_jobs=None,
                 transformer_weights=None,
                 verbose=False):
        if not transformers:
            warnings.warn('Transformers are required')
        self.transformers = transformers
        self.remainder = remainder
        self.sparse_threshold = sparse_threshold
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose

    @classmethod
    def _get_param_names(cls):
        return [
            "n_jobs",
            "output_type",
            "remainder",
            "sparse_threshold",
            "transformer_weights",
            "transformers",
            "verbose",
        ]

    @property
    def _transformers(self):
        """
        Internal list of transformer only containing the name and
        transformers, dropping the columns. This is for the implementation
        of get_params via BaseComposition._get_params which expects lists
        of tuples of len 2.
        """
        if self.transformers is None:
            return []
        return [(name, trans) for name, trans, _ in self.transformers]

    @_transformers.setter
    def _transformers(self, value):
        if not self.transformers:
            self.transformers = [(name, trans, None) for name, trans in value]
        else:
            self.transformers = [
                (name, trans, col) for ((name, trans), (_, _, col))
                in zip(value, self.transformers, strict=True)]

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Returns the parameters given in the constructor as well as the
        estimators contained within the `transformers` of the
        `ColumnTransformer`.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return self._get_params('_transformers', deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of this estimator.

        Valid parameter keys can be listed with ``get_params()``. Note that you
        can directly set the parameters of the estimators contained in
        `transformers` of `ColumnTransformer`.

        Returns
        -------
        self
        """
        self._set_params('_transformers', **kwargs)
        return self

    def _iter(self, fitted=False, replace_strings=False):
        """
        Generate (name, trans, column, weight) tuples.

        If fitted=True, use the fitted transformers, else use the
        user specified transformers updated with converted column names
        and potentially appended with transformer for remainder.

        """
        if fitted:
            transformers = self.transformers_
        else:
            # interleave the validated column specifiers
            transformers = [
                (name, trans, column) for (name, trans, _), column
                in zip(self.transformers, self._columns)
            ]
            # add transformer tuple for remainder
            if self._remainder[2] is not None:
                transformers = chain(transformers, [self._remainder])
        get_weight = (self.transformer_weights or {}).get

        for name, trans, column in transformers:
            if replace_strings:
                # replace 'passthrough' with identity transformer and
                # skip in case of 'drop'
                if trans == 'passthrough':
                    with cuml.using_output_type("cupy"):
                        trans = FunctionTransformer(accept_sparse=True,
                                                    check_inverse=False)
                elif trans == 'drop':
                    continue
                elif _is_empty_column_selection(column):
                    continue

            yield (name, trans, column, get_weight(name))

    def _validate_transformers(self):
        if not self.transformers:
            return

        names, transformers, _ = zip(*self.transformers)

        # validate names
        self._validate_names(names)

        # validate estimators
        for t in transformers:
            if t in ('drop', 'passthrough'):
                continue
            if (not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not
                    hasattr(t, "transform")):
                raise TypeError("All estimators should implement fit and "
                                "transform, or can be 'drop' or 'passthrough' "
                                "specifiers. '%s' (type %s) doesn't." %
                                (t, type(t)))

    def _validate_column_callables(self, X):
        """
        Converts callable column specifications.
        """
        columns = []
        for _, _, column in self.transformers:
            if callable(column):
                column = column(X)
            columns.append(column)
        self._columns = columns

    def _validate_remainder(self, X):
        """
        Validates ``remainder`` and defines ``_remainder`` targeting
        the remaining columns.
        """
        is_transformer = ((hasattr(self.remainder, "fit")
                           or hasattr(self.remainder, "fit_transform"))
                          and hasattr(self.remainder, "transform"))
        if (self.remainder not in ('drop', 'passthrough')
                and not is_transformer):
            raise ValueError(
                "The remainder keyword needs to be one of 'drop', "
                "'passthrough', or estimator. '%s' was passed instead" %
                self.remainder)

        # Make it possible to check for reordered named columns on transform
        self._has_str_cols = any(_determine_key_type(cols) == 'str'
                                 for cols in self._columns)
        if hasattr(X, 'columns'):
            self._df_columns = X.columns

        self._n_features = X.shape[1]
        cols = []
        for columns in self._columns:
            cols.extend(_get_column_indices(X, columns))

        remaining_idx = sorted(set(range(self._n_features)) - set(cols))
        self._remainder = ('remainder', self.remainder, remaining_idx or None)

    @property
    def named_transformers_(self):
        """Access the fitted transformer by name.

        Read-only attribute to access any transformer by given name.
        Keys are transformer names and values are the fitted transformer
        objects.

        """
        # Use Bunch object to improve autocomplete
        return Bunch(**{name: trans for name, trans, _
                        in self.transformers_})

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input feature names.

        Returns
        -------
        feature_names_out : numpy.ndarray of str objects.
            Transformed feature names.
        """
        check_is_fitted(self)
        input_features = check_input_features(self, input_features)
        out = []
        for trans_name, trans, col, _ in self._iter(fitted=True):
            if trans == "drop":
                continue

            # Determine column subset to pass to trans
            if isinstance(col, slice):
                inputs = input_features[col].tolist()
            elif isinstance(col, int):
                inputs = [input_features[col]]
            elif isinstance(col, str):
                inputs = [col]
            else:
                col = cpu_np.asarray(col, dtype=object)
                if all(isinstance(c, str) for c in col):
                    inputs = col.tolist()
                elif all(isinstance(c, bool) for c in col):
                    inputs = input_features[col.astype(bool)].tolist()
                else:
                    inputs = [input_features[c] for c in col]

            if not len(inputs):
                continue

            if trans == 'passthrough':
                names = inputs
            elif not hasattr(trans, 'get_feature_names_out'):
                raise AttributeError(
                    f"Transformer {trans_name!s} (type {type(trans).__name__}) "
                    "does not provide get_feature_names_out."
                )
            else:
                names = trans.get_feature_names_out(inputs)

            out.extend([f"{trans_name}__{name}" for name in names])

        return cpu_np.array(out, dtype=object)

    def _update_fitted_transformers(self, transformers):
        # transformers are fitted; excludes 'drop' cases
        fitted_transformers = iter(transformers)
        transformers_ = []

        for name, old, column, _ in self._iter():
            if old == 'drop':
                trans = 'drop'
            elif old == 'passthrough':
                # FunctionTransformer is present in list of transformers,
                # so get next transformer, but save original string
                next(fitted_transformers)
                trans = 'passthrough'
            elif _is_empty_column_selection(column):
                trans = old
            else:
                trans = next(fitted_transformers)
            transformers_.append((name, trans, column))

        # sanity check that transformers is exhausted
        assert not list(fitted_transformers)
        self.transformers_ = transformers_

    def _validate_output(self, result):
        """
        Ensure that the output of each transformer is 2D. Otherwise
        hstack can raise an error or produce incorrect results.
        """
        names = [name for name, _, _, _ in self._iter(fitted=True,
                                                      replace_strings=True)]
        for Xs, name in zip(result, names):
            if not getattr(Xs, 'ndim', 0) == 2:
                raise ValueError(
                    "The output of the '{0}' transformer should be 2D (scipy "
                    "matrix, array, or pandas DataFrame).".format(name))

    def _log_message(self, name, idx, total):
        if not self.verbose:
            return None
        return '(%d of %d) Processing %s' % (idx, total, name)

    def _fit_transform(self, X, y, func, fitted=False):
        """
        Private function to fit and/or transform on demand.

        Return value (transformers and/or transformed X data) depends
        on the passed function.
        ``fitted=True`` ensures the fitted transformers are used.
        """
        transformers = list(
            self._iter(fitted=fitted, replace_strings=True))
        try:
            return Parallel(n_jobs=self.n_jobs)(
                delayed(func)(
                    transformer=clone(trans) if not fitted else trans,
                    X=_safe_indexing(X, column, axis=1),
                    y=y,
                    weight=weight,
                    message_clsname='ColumnTransformer',
                    message=self._log_message(name, idx, len(transformers)))
                for idx, (name, trans, column, weight) in enumerate(
                    self._iter(fitted=fitted, replace_strings=True), 1))
        except ValueError as e:
            if "Expected 2D array, got 1D array instead" in str(e):
                raise ValueError(_ERR_MSG_1DCOLUMN) from e
            else:
                raise

    @cuml.internals.mlfunc
    def fit(self, X, y=None) -> "ColumnTransformer":
        """Fit all transformers using X.

        Parameters
        ----------
        X : {array-like, dataframe} of shape (n_samples, n_features)
            Input data, of which specified subsets are used to fit the
            transformers.

        y : array-like of shape (n_samples,...), default=None
            Targets for supervised learning.

        Returns
        -------
        self : ColumnTransformer
            This estimator

        """
        # we use fit_transform to make sure to set sparse_output_ (for which we
        # need the transformed data) to have consistent output type in predict
        self.fit_transform(X, y=y)
        return self

    @cuml.internals.mlfunc(
        set_input_type=True,
        preserve_index=True,
        column_names="feature_names_out",
    )
    def fit_transform(self, X, y=None):
        """Fit all transformers, transform the data and concatenate results.

        Parameters
        ----------
        X : {array-like, dataframe} of shape (n_samples, n_features)
            Input data, of which specified subsets are used to fit the
            transformers.

        y : array-like of shape (n_samples,), default=None
            Targets for supervised learning.

        Returns
        -------
        X_t : {array-like, sparse matrix} of \
                shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers. If
            any result is a sparse matrix, everything will be converted to
            sparse matrices.

        """
        check_features(self, X, reset=True)
        self._validate_transformers()
        self._validate_column_callables(X)
        self._validate_remainder(X)

        result = self._fit_transform(X, y, _fit_transform_one)

        if not result:
            self._update_fitted_transformers([])
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        Xs, transformers = zip(*result)

        # determine if concatenated output will be sparse or not
        if any(issparse(X) for X in Xs):
            nnz = sum(X.nnz if issparse(X) else X.size for X in Xs)
            total = sum(X.shape[0] * X.shape[1] if issparse(X)
                        else X.size for X in Xs)
            density = nnz / total
            self.sparse_output_ = density < self.sparse_threshold
        else:
            self.sparse_output_ = False

        self._update_fitted_transformers(transformers)
        self._validate_output(Xs)

        return self._hstack(list(Xs))

    @cuml.internals.mlfunc(preserve_index=True, column_names="feature_names_out")
    def transform(self, X):
        """Transform X separately by each transformer, concatenate results.

        Parameters
        ----------
        X : {array-like, dataframe} of shape (n_samples, n_features)
            The data to be transformed by subset.

        Returns
        -------
        X_t : {array-like, sparse matrix} of \
                shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers. If
            any result is a sparse matrix, everything will be converted to
            sparse matrices.

        """
        check_is_fitted(self)
        check_features(self, X)
        Xs = self._fit_transform(X, None, _transform_one, fitted=True)
        self._validate_output(Xs)

        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        return self._hstack(list(Xs))

    def _hstack(self, Xs):
        """Stacks Xs horizontally.

        This allows subclasses to control the stacking behavior, while reusing
        everything else from ColumnTransformer.

        Parameters
        ----------
        Xs : list of {array-like, sparse matrix, dataframe}
        """
        if self.sparse_output_:
            try:
                # since all columns should be numeric before stacking them
                # in a sparse matrix, `check_array` is used for the
                # dtype conversion if necessary.
                converted_Xs = [check_array(X,
                                            accept_sparse=True,
                                            ensure_all_finite=False)
                                for X in Xs]
            except ValueError as e:
                raise ValueError(
                    "For a sparse output, all columns should "
                    "be a numeric or convertible to a numeric."
                ) from e

            return cu_sparse.hstack(converted_Xs).tocsr()
        else:
            Xs = [f.toarray() if issparse(f) else f for f in Xs]
            return np.hstack(Xs)


def _is_empty_column_selection(column):
    """
    Return True if the column selection is empty (empty list or all-False
    boolean array).

    """
    if hasattr(column, 'dtype') and is_bool_dtype(column.dtype):
        return not column.any()
    elif hasattr(column, '__len__'):
        return (len(column) == 0 or
                all(isinstance(col, bool) for col in column)
                and not any(column))
    else:
        return False


def _get_transformer_list(estimators):
    """
    Construct (name, trans, column) tuples from list

    """
    transformers, columns = zip(*estimators)
    names, _ = zip(*_name_estimators(transformers))

    transformer_list = list(zip(names, transformers, columns))
    return transformer_list


def make_column_transformer(*transformers,
                            remainder='drop',
                            sparse_threshold=0.3,
                            n_jobs=None,
                            verbose=False):
    """Construct a ColumnTransformer from the given transformers.

    This is a shorthand for the ColumnTransformer constructor; it does not
    require, and does not permit, naming the transformers. Instead, they will
    be given names automatically based on their types. It also does not allow
    weighting with ``transformer_weights``.

    Parameters
    ----------
    *transformers : tuples
        Tuples of the form (transformer, columns) specifying the
        transformer objects to be applied to subsets of the data:

        * transformer : {'drop', 'passthrough'} or estimator
            Estimator must support `fit` and `transform`.
            Special-cased strings 'drop' and 'passthrough' are accepted as
            well, to indicate to drop the columns or to pass them through
            untransformed, respectively.
        * columns : str,  array-like of str, int, array-like of int, slice, \
                array-like of bool or callable
            Indexes the data on its second axis. Integers are interpreted as
            positional columns, while strings can reference DataFrame columns
            by name. A scalar string or int should be used where
            ``transformer`` expects X to be a 1d array-like (vector),
            otherwise a 2d array will be passed to the transformer.
            A callable is passed the input data `X` and can return any of the
            above. To select multiple columns by name or dtype, you can use
            :obj:`make_column_selector`.

    remainder : {'drop', 'passthrough'} or estimator, default='drop'
        By default, only the specified columns in `transformers` are
        transformed and combined in the output, and the non-specified
        columns are dropped. (default of ``'drop'``).
        By specifying ``remainder='passthrough'``, all remaining columns that
        were not specified in `transformers` will be automatically passed
        through. This subset of columns is concatenated with the output of
        the transformers.
        By setting ``remainder`` to be an estimator, the remaining
        non-specified columns will use the ``remainder`` estimator. The
        estimator must support `fit` and `transform`.

    sparse_threshold : float, default=0.3
        If the transformed output consists of a mix of sparse and dense data,
        it will be stacked as a sparse matrix if the density is lower than this
        value. Use ``sparse_threshold=0`` to always return dense.
        When the transformed output consists of all sparse or all dense data,
        the stacked result will be sparse or dense, respectively, and this
        keyword will be ignored.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See `Glossary <n_jobs>`
        for more details.

    verbose : bool, default=False
        If True, the time elapsed while fitting each transformer will be
        printed as it is completed.

    Returns
    -------
    ct : ColumnTransformer

    See Also
    --------
    ColumnTransformer : Class that allows combining the
        outputs of multiple transformer objects used on column subsets
        of the data into a single feature space.

    Examples
    --------
    >>> from cuml.preprocessing import StandardScaler, OneHotEncoder
    >>> from cuml.compose import make_column_transformer
    >>> make_column_transformer(
    ...     (StandardScaler(), ['numerical_column']),
    ...     (OneHotEncoder(), ['categorical_column']))
    ColumnTransformer(transformers=[('standardscaler', StandardScaler(...),
                                     ['numerical_column']),
                                    ('onehotencoder', OneHotEncoder(...),
                                     ['categorical_column'])])
    """
    # transformer_weights keyword is not passed through because the user
    # would need to know the automatically generated names of the transformers
    transformer_list = _get_transformer_list(transformers)
    return ColumnTransformer(transformer_list, n_jobs=n_jobs,
                             remainder=remainder,
                             sparse_threshold=sparse_threshold,
                             verbose=verbose)


class make_column_selector:
    """Create a callable to select columns to be used with
    :class:`ColumnTransformer`.

    :func:`make_column_selector` can select columns based on datatype or the
    columns name with a regex. When using multiple selection criteria, **all**
    criteria must match for a column to be selected.

    Parameters
    ----------
    pattern : str, default=None
        Name of columns containing this regex pattern will be included. If
        None, column selection will not be selected based on pattern.

    dtype_include : column dtype or list of column dtypes, default=None
        A selection of dtypes to include. For more details, see
        :meth:`pandas.DataFrame.select_dtypes`.

    dtype_exclude : column dtype or list of column dtypes, default=None
        A selection of dtypes to exclude. For more details, see
        :meth:`pandas.DataFrame.select_dtypes`.

    Returns
    -------
    selector : callable
        Callable for column selection to be used by a
        :class:`ColumnTransformer`.

    See Also
    --------
    ColumnTransformer : Class that allows combining the
        outputs of multiple transformer objects used on column subsets
        of the data into a single feature space.

    Examples
    --------
    >>> from cuml.preprocessing import StandardScaler, OneHotEncoder
    >>> from cuml.compose import make_column_transformer
    >>> from cuml.compose import make_column_selector
    >>> import cupy as cp
    >>> import cudf  # doctest: +SKIP
    >>> X = cudf.DataFrame({'city': ['London', 'London', 'Paris', 'Sallisaw'],
    ...                    'rating': [5, 3, 4, 5]})  # doctest: +SKIP
    >>> ct = make_column_transformer(
    ...       (StandardScaler(),
    ...        make_column_selector(dtype_include=cp.number)),  # rating
    ...       (OneHotEncoder(),
    ...        make_column_selector(dtype_include=object)))  # city
    >>> ct.fit_transform(X)  # doctest: +SKIP
    array([[ 0.90453403,  1.        ,  0.        ,  0.        ],
           [-1.50755672,  1.        ,  0.        ,  0.        ],
           [-0.30151134,  0.        ,  1.        ,  0.        ],
           [ 0.90453403,  0.        ,  0.        ,  1.        ]])
    """

    def __init__(self, pattern=None, *, dtype_include=None,
                 dtype_exclude=None):
        self.pattern = pattern
        self.dtype_include = dtype_include
        self.dtype_exclude = dtype_exclude

    def __call__(self, df):
        if not hasattr(df, 'iloc'):
            raise ValueError("make_column_selector can only be applied to "
                             "pandas dataframes")
        df_row = df.iloc[:1]
        if self.dtype_include is not None or self.dtype_exclude is not None:
            df_row = df_row.select_dtypes(include=self.dtype_include,
                                          exclude=self.dtype_exclude)
        cols = df_row.columns
        if self.pattern is not None:
            cols = cols[cols.str.contains(self.pattern, regex=True)]
        return cols.tolist()
