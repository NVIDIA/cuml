# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import warnings
from collections.abc import Sequence

import cudf
import cupy as cp
import cupyx.scipy.sparse as cp_sp
import numpy as np
from sklearn.base import OneToOneFeatureMixin

from cuml.common.doc_utils import generate_docstring
from cuml.internals.base import Base
from cuml.internals.interop import InteropMixin, UnsupportedOnGPU
from cuml.internals.mixins import DeprecatedGetFeatureNamesMixin
from cuml.internals.outputs import mlfunc
from cuml.internals.validation import (
    check_array,
    check_cudf,
    check_features,
    check_input_features,
    check_is_fitted,
)


def _safe_is_nan(x):
    """Check if `x` is NaN, without erroring if non-numeric"""
    try:
        return np.isnan(x)
    except (TypeError, ValueError):
        pass
    return False


def _get_diff(unique_vals, cats):
    """Equal to ``unique_vals.difference(cats)``, but for numpy arrays not sets."""
    # Since NaN's sort last and we enforce NaN is last value in cats if
    # present, we only need to check the last values.
    if _safe_is_nan(cats[-1]) and _safe_is_nan(unique_vals[-1]):
        unique_vals = unique_vals[:-1]
    return np.setdiff1d(unique_vals, cats, assume_unique=True).tolist()


def _cats_to_series(cats):
    """Coerce `cats` to a Series, but supporting `NaN` in object arrays"""
    # XXX: `cudf.Series(['a', 'b', NaN])` errors. Here we coerce NaN->None for
    # this edge case. Since we enforce NaN is last value in cats if present, we
    # only need to check the last value.
    if cats.dtype.kind == "O" and _safe_is_nan(cats[-1]):
        cats = cats.copy()
        cats[-1] = None
    return cudf.Series(cats, nan_as_null=True)


def _as_numpy(x, dtype=None):
    """Coerce an array-like `x` to a numpy array."""
    if hasattr(x, "to_numpy"):
        return x.to_numpy(dtype=dtype)
    if hasattr(x, "__cuda_array_interface__"):
        x = cp.asnumpy(x)
    return np.asarray(x, dtype=dtype)


def _compute_categories(
    X, unique=False, categories="auto", handle_unknown="error"
):
    """Compute `categories_` for an encoder.

    Parameters
    ----------
    X : cudf.DataFrame or list[cudf.Series]
        A cudf.DataFrame or list[cudf.Series] of the input X.
    unique : bool, default=False
        Whether the columns already are reduced to only their unique entries.
    categories : 'auto' or list[array-like], default='auto'
        Explicitly provided categories per-column, or 'auto' to automatically
        infer the categories from the input data.
    handle_unknown : {'error', 'ignore'}, default='error'
        If 'error', entries found in X that don't exist in explicitly provided
        categories will lead to an error. If 'ignore' no error will be raised.

    Returns
    -------
    categories_ : list[numpy.ndarray]
        A list of the categories determined per-column.
    """
    if isinstance(X, list):
        X_list = X
    else:
        X_list = [X.iloc[:, i] for i in range(X.shape[1])]
    n_features = len(X_list)

    if handle_unknown not in ("ignore", "error"):
        raise ValueError(
            "Expected `handle_unknown` to be one of ['error', 'ignore'], "
            f"got {handle_unknown!r}"
        )

    if auto := (isinstance(categories, str) and categories == "auto"):
        pass
    elif isinstance(categories, Sequence) and not isinstance(categories, str):
        if len(categories) != n_features:
            raise ValueError(
                "Shape mismatch: if categories is an array,"
                " it has to be of shape (n_features,)."
            )
    else:
        raise ValueError(
            "Expected `categories` to be 'auto' or a sequence of "
            f"array-likes, got {categories!r}"
        )

    out = []

    for i, Xi in enumerate(X_list):
        if auto:
            if not unique:
                Xi = Xi.drop_duplicates()
            # For the edge case of floating inputs, we want to ensure NaN and null
            # are treated equivalently (cudf's default). Coerce NaN to null
            # and redrop duplicates in case the input had both NaN and null.
            # This is cheaper than doing `nans_to_nulls` on the full input first.
            if Xi.dtype.kind == "f":
                Xi = Xi.nans_to_nulls().drop_duplicates()
            # cudf's object dtype uses None for NA, we want NaN everywhere
            if Xi.dtype == "object":
                Xi = Xi.astype(str)
            cats = Xi.sort_values().to_numpy()
        else:
            cats_cudf = cudf.Series(
                categories[i],
                dtype=(Xi.dtype if isinstance(Xi.dtype, np.dtype) else str),
                nan_as_null=True,
            )
            if cats_cudf.dtype == "object":
                cats_cudf = cats_cudf.astype(str)
            cats = cats_cudf.to_numpy()

            # Any null values must be the last stated category
            if cats_cudf[:-1].isnull().any():
                raise ValueError(
                    "Nan should be the last element in user"
                    f" provided categories, see categories {cats}"
                    f" in column #{i}"
                )

            if not cats_cudf.is_unique:
                raise ValueError(
                    f"In column {i}, the predefined categories"
                    " contain duplicate elements."
                )

            if handle_unknown == "error":
                if not unique:
                    Xi = Xi.drop_duplicates()
                if Xi.dtype.kind == "f":
                    Xi = Xi.nans_to_nulls()
                diff = Xi[~Xi.isin(cats_cudf)]
                if len(diff):
                    # XXX: need to repeat drop_duplicates just in case Xi had both
                    # None & NaN earlier.
                    diff = (
                        diff.drop_duplicates()
                        .sort_values()
                        .to_numpy()
                        .tolist()
                    )
                    raise ValueError(
                        f"Found unknown categories {diff} in column {i} during fit"
                    )
        out.append(cats)

    return out


class OneHotEncoder(DeprecatedGetFeatureNamesMixin, InteropMixin, Base):
    """
    Encode categorical features as a one-hot numeric array.

    The input to this transformer should be an array-like of integers or
    strings, denoting the values taken on by categorical (discrete) features.
    The features are encoded using a one-hot (aka 'one-of-K' or 'dummy')
    encoding scheme. This creates a binary column for each category and
    returns a sparse matrix or dense array (depending on the ``sparse_output``
    parameter).

    By default, the encoder derives the categories based on the unique values
    in each feature. Alternatively, you can also specify the `categories`
    manually.

    Parameters
    ----------
    categories : 'auto' or a list of array-like, default='auto'
        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column.

    drop : 'first', None, or array-like of shape (n_features,), default=None
        Specifies a methodology to use to drop one of the categories per
        feature. This is useful in situations where perfectly collinear
        features cause problems, such as when feeding the resulting data
        into an unregularized linear regression model.

        However, dropping one category breaks the symmetry of the original
        representation and can therefore induce a bias in downstream models,
        for instance for penalized linear classification or regression models.

        - None : retain all features (the default).
        - 'first' : drop the first category in each feature. If only one
          category is present, the feature will be dropped entirely.
        - array : ``drop[i]`` is the category in feature ``X[:, i]`` that
          should be dropped.

    sparse_output : bool, default=True
        When ``True``, transform returns a sparse matrix/array in CSR format.

    dtype : dtype, default=np.float32
        Desired dtype of transformed output.

    handle_unknown : {'error', 'ignore'}, default='error'
        Specifies the way unknown categories are handled during :meth:`transform`.

        - 'error' : Raise an error if an unknown category is present during transform.
        - 'ignore' : When an unknown category is encountered during
          transform, the resulting one-hot encoded columns for this feature
          will be all zeros. In the inverse transform, an unknown category
          will be denoted as None.

    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.

    output_type : {None, 'input', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting
        (in order of the features in X and corresponding with the output
        of ``transform``). This includes the category specified in ``drop``
        (if any).

    drop_idx_ : array of shape (n_features,)
        - ``drop_idx_[i]`` is the index in ``categories_[i]`` of the category
          to be dropped for feature ``i``, or ``None`` if no category is to be
          dropped.
        - ``drop_idx_ = None`` if all the transformed features will be
          retained.

    n_features_in_ : int
        Number of features seen during ``fit``.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during ``fit``. Defined only when `X`
        has feature names that are all strings.

    Examples
    --------
    >>> import cudf
    >>> from cuml.preprocessing import OneHotEncoder
    >>> X = cudf.DataFrame({"fruit": ["apple", "banana", "apple"], "group": [1, 3, 2]})
    >>> enc = OneHotEncoder().fit(X)
    >>> enc.categories_
    [array(['apple', 'banana'], dtype=object), array([1, 2, 3])]
    >>> enc.transform(X).toarray()
    array([[1., 0., 1., 0., 0.],
           [0., 1., 0., 0., 1.],
           [1., 0., 0., 1., 0.]], dtype=float32)
    >>> enc.inverse_transform([[0, 1, 1, 0, 0], [1, 0, 0, 1, 0]])
    array([['banana', 1],
           ['apple', 2]], dtype=object)
    """

    _cpu_class_path = "sklearn.preprocessing.OneHotEncoder"

    def __init__(
        self,
        *,
        categories="auto",
        drop=None,
        sparse_output=True,
        dtype=np.float32,
        handle_unknown="error",
        output_type=None,
        verbose=False,
    ):
        super().__init__(output_type=output_type, verbose=verbose)
        self.categories = categories
        self.drop = drop
        self.sparse_output = sparse_output
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    @classmethod
    def _get_param_names(cls):
        return [
            "categories",
            "drop",
            "sparse_output",
            "dtype",
            "handle_unknown",
            *super()._get_param_names(),
        ]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.categorical = True
        tags.input_tags.allow_nan = True
        return tags

    @classmethod
    def _params_from_cpu(cls, model):
        if np.dtype(model.dtype).kind not in "fb":
            raise UnsupportedOnGPU(f"`dtype={model.dtype!r}` is not supported")
        if isinstance(model.drop, str) and model.drop == "if_binary":
            raise UnsupportedOnGPU("`drop='if_binary'` is not supported")
        if model.handle_unknown in ("infrequent_if_exist", "warn"):
            raise UnsupportedOnGPU(
                f"`handle_unknown={model.handle_unknown!r}` is not supported"
            )
        if model.min_frequency is not None:
            raise UnsupportedOnGPU("`min_frequency` is not supported")
        if model.max_categories is not None:
            raise UnsupportedOnGPU("`max_categories` is not supported")
        if not (
            isinstance(model.feature_name_combiner, str)
            and model.feature_name_combiner == "concat"
        ):
            raise UnsupportedOnGPU("`feature_name_combiner` is not supported")
        return {
            "categories": model.categories,
            "drop": model.drop,
            "sparse_output": model.sparse_output,
            "dtype": model.dtype,
            "handle_unknown": model.handle_unknown,
        }

    def _params_to_cpu(self):
        categories = self.categories
        if not (isinstance(categories, str) and categories == "auto"):
            categories = [_as_numpy(c) for c in categories]

        drop = self.drop
        if not (drop is None or (isinstance(drop, str) and drop == "first")):
            drop = _as_numpy(drop, dtype=object)

        return {
            "categories": categories,
            "drop": drop,
            "sparse_output": self.sparse_output,
            "dtype": self.dtype,
            "handle_unknown": self.handle_unknown,
        }

    def _attrs_from_cpu(self, model):
        return {
            "categories_": model.categories_,
            "drop_idx_": model.drop_idx_,
            "_n_features_outs": model._n_features_outs,
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        return {
            "categories_": self.categories_,
            "drop_idx_": self.drop_idx_,
            "_n_features_outs": self._n_features_outs,
            "_infrequent_enabled": False,
            "_drop_idx_after_grouping": self.drop_idx_,
            **super()._attrs_to_cpu(model),
        }

    @mlfunc(set_input_type=True)
    @generate_docstring(y=None)
    def fit(self, X, y=None) -> "OneHotEncoder":
        """Fit OneHotEncoder to X."""
        check_features(self, X, reset=True)
        X = check_cudf(X, input_name="X")
        return self._fit(X)

    @mlfunc(
        set_input_type=True,
        preserve_index=True,
        column_names="feature_names_out",
    )
    @generate_docstring(
        y=None,
        return_values={
            "name": "X_out",
            "description": (
                "Transformed input. A sparse matrix if ``sparse_output=True``, "
                "dense otherwise."
            ),
            "type": "dense_sparse",
            "shape": "(n_samples, n_encoded_features)",
        },
    )
    def fit_transform(self, X, y=None):
        """Fit OneHotEncoder to X, then transform X."""
        check_features(self, X, reset=True)
        X = check_cudf(X, input_name="X")
        return self._fit(X).transform(X)

    def _fit(self, X, unique=False):
        categories = _compute_categories(
            X,
            unique=unique,
            categories=self.categories,
            handle_unknown=self.handle_unknown,
        )

        # Compute drop_idx_
        if self.drop is None:
            drop_idx = None
        elif isinstance(self.drop, str):
            if self.drop == "first":
                drop_idx = np.zeros(len(categories), dtype=object)
            else:
                raise ValueError(
                    "Expected `drop` to be 'first' or an array-like, "
                    f"got {self.drop!r}"
                )
        else:
            drop = _as_numpy(self.drop, dtype=object)

            if len(drop) != len(categories):
                raise ValueError(
                    "`drop` should have length equal to the number of features "
                    f"({len(categories)}), got {len(drop)}"
                )
            missing_drops = []
            drop_indices = []
            for feature, (drop_val, cat) in enumerate(zip(drop, categories)):
                if _safe_is_nan(drop_val):
                    if _safe_is_nan(cat[-1]):
                        drop_indices.append(cat.size - 1)
                    else:
                        missing_drops.append((feature, drop_val))
                else:
                    idx = np.where(cat == drop_val)[0]
                    if idx.size:
                        drop_indices.append(idx.item())
                    else:
                        missing_drops.append((feature, drop_val))

            if any(missing_drops):
                raise ValueError(
                    "The following categories were supposed to be "
                    "dropped, but were not found in the training "
                    "data.\n{}".format(
                        "\n".join(
                            [
                                "Category: {}, Feature: {}".format(c, v)
                                for c, v in missing_drops
                            ]
                        )
                    )
                )
            drop_idx = np.array(drop_indices, dtype=object)

        # Compute n_features_out per input feature
        n_features_outs = [len(cats) for cats in categories]
        if drop_idx is not None:
            for i, idx in enumerate(drop_idx):
                if idx is not None:
                    n_features_outs[i] -= 1

        # Store fitted attributes
        self.categories_ = categories
        self.drop_idx_ = drop_idx
        self._n_features_outs = n_features_outs

        return self

    @mlfunc(preserve_index=True, column_names="feature_names_out")
    @generate_docstring(
        return_values={
            "name": "X_out",
            "description": (
                "Transformed input. A sparse matrix if ``sparse_output=True``, "
                "dense otherwise."
            ),
            "type": "dense_sparse",
            "shape": "(n_samples, n_encoded_features)",
        }
    )
    def transform(self, X):
        """Transform X using one-hot encoding."""
        check_is_fitted(self)
        check_features(self, X)
        X = check_cudf(X, input_name="X")

        raw_inds = cp.zeros(X.shape, dtype="int32")
        is_masked = False
        columns_with_unknown = []
        drop_idx = self.drop_idx_

        for i in range(X.shape[1]):
            Xi = X.iloc[:, i]
            cats = self.categories_[i]

            if _safe_is_nan(cats[-1]):
                # cudf's CategoricalDtype doesn't allow encoding null values,
                # we have to handle these manually.
                codes = Xi.astype(cudf.CategoricalDtype(cats[:-1])).cat.codes
                if Xi.hasnans:
                    codes[Xi.isnull()] = len(cats) - 1
            else:
                codes = Xi.astype(cudf.CategoricalDtype(cats)).cat.codes

            if codes.has_nulls:
                if self.handle_unknown == "error":
                    present = Xi.drop_duplicates().sort_values().to_numpy()
                    diff = _get_diff(present, self.categories_[i])
                    raise ValueError(
                        f"Found unknown categories {diff} in column {i}"
                        " during transform"
                    )
                is_masked = True
                columns_with_unknown.append(i)

            if drop_idx is not None and drop_idx[i] is not None:
                is_masked = True
                if drop_idx[i] == 0:
                    codes -= 1
                else:
                    codes[codes == drop_idx[i]] = -1
                    codes[codes > drop_idx[i]] -= 1
            raw_inds[:, i] = codes.fillna(-1)

        n_samples, n_features = raw_inds.shape

        feature_indices = np.cumsum([0, *self._n_features_outs])
        indices = (raw_inds + cp.asarray(feature_indices[:-1])).ravel()

        if (
            self.handle_unknown == "ignore"
            and self.drop is not None
            and columns_with_unknown
        ):
            warnings.warn(
                "Found unknown categories in columns "
                f"{columns_with_unknown} during transform. These "
                "unknown categories will be encoded as all zeros",
            )

        if is_masked:
            mask = raw_inds != -1
            indices = indices[mask.ravel()]

            indptr = cp.zeros(n_samples + 1, dtype=int)
            cp.sum(mask, axis=1, out=indptr[1:], dtype=indptr.dtype)
            cp.cumsum(indptr[1:], out=indptr[1:])
        else:
            indptr = cp.arange(
                0, n_features * n_samples + 1, n_features, dtype=int
            )

        data = cp.ones(indptr[-1].item(), dtype=self.dtype)

        out = cp_sp.csr_matrix(
            (data, indices, indptr),
            shape=(n_samples, feature_indices[-1]),
            dtype=self.dtype,
        )
        if self.sparse_output:
            return out
        return out.toarray()

    @mlfunc(preserve_index=True, column_names="feature_names_in")
    def inverse_transform(self, X):
        """Convert the data back to the original representation.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_encoded_features)
            The transformed data.

        Returns
        -------
        X_original : array of shape (n_samples, n_features)
            Inverse transformed array.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse="csr")

        n_features_out = np.sum(self._n_features_outs)
        if X.shape[1] != n_features_out:
            raise ValueError(
                f"Shape of the passed X data is not correct. Expected "
                f"{n_features_out} columns, got {X.shape[1]}."
            )

        j = 0
        found_unknown = {}
        columns = {}

        for i, (cats, n_cols) in enumerate(
            zip(self.categories_, self._n_features_outs)
        ):
            drop_idx = None if self.drop_idx_ is None else self.drop_idx_[i]

            if len(cats) == 1 and drop_idx is not None:
                columns[i] = (
                    cudf.Series(cats[drop_idx], nan_as_null=True)
                    .repeat(X.shape[0])
                    .reset_index(drop=True)
                )
            else:
                if drop_idx is not None:
                    cats = np.delete(cats, drop_idx)
                cats = _cats_to_series(cats)
                sub = X[:, j : j + n_cols]
                labels = cp.asarray(sub.argmax(axis=1)).ravel()
                columns[i] = cats.take(labels).reset_index(drop=True)

                unknown = cp.asarray(sub.sum(axis=1) == 0).ravel()
                if unknown.any():
                    if drop_idx is not None:
                        # Treat all zeros as the dropped category
                        columns[i][unknown] = self.categories_[i][drop_idx]
                    else:
                        if self.handle_unknown == "ignore":
                            # Could be anything, fill with None later
                            found_unknown[i] = unknown
                        else:
                            all_zero_samples = cp.flatnonzero(unknown)
                            raise ValueError(
                                f"Samples {all_zero_samples} can not be inverted "
                                "when drop=None and handle_unknown='error' "
                                "because they contain all zeros"
                            )

            j += n_cols

        out = cudf.DataFrame(columns)

        for idx, mask in found_unknown.items():
            out.loc[mask, idx] = None

        return out

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

        cats = self.categories_
        input_features = check_input_features(self, input_features)

        out = []
        for i, (col, cats) in enumerate(zip(input_features, self.categories_)):
            drop_idx = None if self.drop_idx_ is None else self.drop_idx_[i]
            if drop_idx is not None:
                cats = np.delete(cats, drop_idx)
            out.extend(f"{col}_{val!s}" for val in cats)

        return np.array(out, dtype=object)


class OrdinalEncoder(OneToOneFeatureMixin, Base):
    """Encode categorical features as an integer array.

    The input to this transformer should be an array-like of integers or
    strings, denoting the values taken on by categorical (discrete) features.
    The features are converted to ordinal integers. This results in
    a single column of integers (0 to n_categories - 1) per feature.

    Parameters
    ----------
    categories : 'auto' or a list of array-like, default='auto'
        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column.

        The used categories can be found in the ``categories_`` attribute.

    dtype : number type, default=np.float64
        Desired dtype of output.

    handle_unknown : {'error', 'ignore'}, default='error'
        When set to 'error' an error will be raised in case an unknown
        categorical feature is present during transform. When set to 'ignore',
        the encoded value of unknown categories will be set to NaN. In
        :meth:`inverse_transform`, an unknown category will be denoted as None.

    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.

    output_type : {None, 'input', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during ``fit`` (in order of
        the features in X and corresponding with the output of ``transform``).
        This does not include categories that weren't seen during ``fit``.

    n_features_in_ : int
        Number of features seen during ``fit``.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during ``fit``. Defined only when `X`
        has feature names that are all strings.

    Examples
    --------
    >>> import cudf
    >>> from cuml.preprocessing import OrdinalEncoder
    >>> X = cudf.DataFrame({"fruit": ["apple", "banana", "apple"], "group": [1, 3, 2]})
    >>> enc = OrdinalEncoder(output_type="numpy").fit(X)
    >>> enc.categories_
    [array(['apple', 'banana'], dtype=object), array([1, 2, 3])]
    >>> enc.transform(X)
    array([[0., 0.],
           [1., 2.],
           [0., 1.]])
    >>> enc.inverse_transform([[1, 0], [0, 1]])
    array([['banana', 1],
           ['apple', 2]], dtype=object)
    """

    def __init__(
        self,
        *,
        categories="auto",
        dtype=np.float64,
        handle_unknown="error",
        verbose=False,
        output_type=None,
    ) -> None:
        super().__init__(verbose=verbose, output_type=output_type)
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    @classmethod
    def _get_param_names(cls):
        return [
            "categories",
            "dtype",
            "handle_unknown",
            *super()._get_param_names(),
        ]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.categorical = True
        tags.input_tags.allow_nan = True
        return tags

    @mlfunc(set_input_type=True)
    @generate_docstring(y=None)
    def fit(self, X, y=None) -> "OrdinalEncoder":
        """Fit OrdinalEncoder to X."""
        check_features(self, X, reset=True)
        X = check_cudf(X, input_name="X")
        return self._fit(X)

    @mlfunc(
        set_input_type=True,
        preserve_index=True,
        column_names="feature_names_out",
    )
    @generate_docstring(
        y=None,
        return_values={
            "name": "X_out",
            "description": "Transformed input.",
            "type": "dense",
            "shape": "(n_samples, n_features)",
        },
    )
    def fit_transform(self, X, y=None):
        """Fit OrdinalEncoder to X, then transform X."""
        check_features(self, X, reset=True)
        X = check_cudf(X, input_name="X")
        return self._fit(X).transform(X)

    def _fit(self, X, unique=False):
        self.categories_ = _compute_categories(
            X,
            unique=unique,
            categories=self.categories,
            handle_unknown=self.handle_unknown,
        )

        out_dtype = np.dtype(self.dtype)
        if self.handle_unknown == "ignore" and out_dtype.kind != "f":
            raise ValueError(
                f"When handle_unknown='ignore', the dtype parameter "
                f"should be a float dtype. Got {self.dtype}."
            )

        self._missing_indices = {
            i: len(cats) - 1
            for i, cats in enumerate(self.categories_)
            if _safe_is_nan(cats[-1])
        }

        if self._missing_indices and out_dtype.kind != "f":
            raise ValueError(
                "There are missing values in features "
                f"{list(self._missing_indices)}. Please "
                "set dtype to a float."
            )

        return self

    @mlfunc(preserve_index=True, column_names="feature_names_out")
    @generate_docstring(
        return_values={
            "name": "X_out",
            "description": "Transformed input.",
            "type": "dense",
            "shape": "(n_samples, n_features)",
        }
    )
    def transform(self, X):
        """Transform X using ordinal encoding."""
        check_is_fitted(self)
        check_features(self, X)
        X = check_cudf(X, input_name="X")

        out = cp.zeros(X.shape, dtype=self.dtype)

        for i in range(X.shape[1]):
            Xi = X.iloc[:, i]
            cats = self.categories_[i]

            if _safe_is_nan(cats[-1]):
                cats = cats[:-1]
            codes = Xi.astype(cudf.CategoricalDtype(cats)).cat.codes

            if codes.has_nulls:
                if self.handle_unknown == "error":
                    # If NaN is a known category and all nulls map to NaN in
                    # the input then there's no need to error. Otherwise error.
                    if not (
                        i in self._missing_indices
                        and Xi.hasnans
                        and not codes[Xi.notnull()].has_nulls
                    ):
                        present = Xi.drop_duplicates().sort_values().to_numpy()
                        diff = _get_diff(present, self.categories_[i])
                        raise ValueError(
                            f"Found unknown categories {diff} in column {i}"
                            " during transform"
                        )

                codes = codes.to_cupy()

            out[:, i] = codes

        return out

    @mlfunc(preserve_index=True, column_names="feature_names_in")
    def inverse_transform(self, X):
        """Convert the data back to the original representation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_encoded_features)
            The transformed data.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Inverse transformed array.
        """
        check_is_fitted(self)
        X = check_array(X, ensure_all_finite="allow-nan")

        if X.shape[1] != len(self.categories_):
            raise ValueError(
                f"Shape of the passed X data is not correct. Expected "
                f"{len(self.categories_)} columns, got {X.shape[1]}."
            )

        columns = {}
        found_unknown = {}

        for i, cats in enumerate(self.categories_):
            labels = X[:, i]
            cats = _cats_to_series(cats)

            if (
                labels.dtype.kind == "f"
                and (nan_entries := cp.isnan(labels)).any()
            ):
                if i in self._missing_indices:
                    labels = labels.copy()
                    labels[nan_entries] = self._missing_indices[i]
                elif self.handle_unknown == "ignore":
                    labels = labels.copy()
                    # Fill with an arbitrary valid label, will be replaced later
                    labels[nan_entries] = 0
                    found_unknown[i] = nan_entries
                else:
                    unknown_indices = cp.flatnonzero(nan_entries)
                    raise ValueError(
                        f"Samples {unknown_indices} can not be inverted "
                        "when handle_unknown='error' because they contain "
                        "NaN values"
                    )

            columns[i] = cats.take(labels.astype("int")).reset_index(drop=True)

        out = cudf.DataFrame(columns)

        for idx, mask in found_unknown.items():
            out.loc[mask, idx] = None

        return out
