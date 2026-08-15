# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import warnings
from typing import Optional

import cudf
import cupy as cp
import cupyx
import numpy as np
from cudf import Index

import cuml
from cuml.common.doc_utils import generate_docstring
from cuml.internals.base import Base
from cuml.internals.mixins import DeprecatedGetFeatureNamesMixin
from cuml.internals.outputs import mlfunc
from cuml.internals.validation import (
    check_features,
    check_input_features,
    check_is_fitted,
)
from cuml.preprocessing._label import LabelEncoder


class BaseEncoder(Base):
    """Base implementation for encoding categorical values, uses
    :py:class:`~cuml.preprocessing.LabelEncoder` for obtaining unique values.

    Parameters
    ----------
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {None, 'input', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.
    """

    def _set_input_type(self, value):
        if self.input_type is None:
            self.input_type = value

    def _check_input(self, X, is_categories=False):
        """If input is cupy, convert it to a DataFrame with 0 copies."""
        if isinstance(X, cp.ndarray):
            self._set_input_type("array")
            if is_categories:
                X = X.transpose()
            return cudf.DataFrame(X)
        else:
            self._set_input_type("df")
            return X

    def _check_input_fit(self, X, is_categories=False):
        """Helper function used in fit, can be overridden in subclasses."""
        return self._check_input(X, is_categories=is_categories)

    def _unique(self, inp):
        """Helper function used in fit. Can be overridden in subclasses."""

        # Default implementation passes input through directly since this is
        # performed in `LabelEncoder.fit()`
        return inp

    def _fit(self, X, need_drop: bool):
        check_features(self, X, reset=True)
        X = self._check_input_fit(X)
        if type(self.categories) is str and self.categories == "auto":
            self._features = X.columns
            self._encoders = {
                feature: LabelEncoder(
                    verbose=self.verbose,
                    output_type="cudf",
                    handle_unknown=self.handle_unknown,
                ).fit(self._unique(X[feature]))
                for feature in self._features
            }
        else:
            self.categories = self._check_input(self.categories, True)
            self._features = self.categories.columns
            if len(self._features) != X.shape[1]:
                raise ValueError(
                    "Shape mismatch: if categories is not 'auto',"
                    " it has to be of shape (n_features, _)."
                )
            self._encoders = dict()
            for feature in self._features:
                le = LabelEncoder(
                    verbose=self.verbose,
                    output_type="cudf",
                    handle_unknown=self.handle_unknown,
                )

                self._encoders[feature] = le.fit(self.categories[feature])

                if self.handle_unknown == "error":
                    if self._has_unknown(
                        X[feature],
                        cudf.Series(self._encoders[feature].classes_),
                    ):
                        msg = (
                            "Found unknown categories in column {0}"
                            " during fit".format(feature)
                        )
                        raise KeyError(msg)

        if need_drop:
            self.drop_idx_ = self._compute_drop_idx()
        self._fitted = True

    @property
    def categories_(self):
        """Returns categories used for the one hot encoding in the correct order."""
        return [self._encoders[f].classes_ for f in self._features]


class OneHotEncoder(DeprecatedGetFeatureNamesMixin, BaseEncoder):
    """
    Encode categorical features as a one-hot numeric array.
    The input to this estimator should be a :py:class:`cuDF.DataFrame` or a
    :py:class:`cupy.ndarray`, denoting the unique values taken on by categorical
    (discrete) features.  The features are encoded using a one-hot (aka 'one-of-K' or
    'dummy') encoding scheme. This creates a binary column for each category and returns
    a sparse matrix or dense array (depending on the ``sparse_output`` parameter).

    By default, the encoder derives the categories based on the unique values
    in each feature. Alternatively, you can also specify the `categories`
    manually.

    .. note:: a one-hot encoding of y labels should use a LabelBinarizer
        instead.

    Parameters
    ----------
    categories : 'auto' an cupy.ndarray or a cudf.DataFrame, default='auto'
                 Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.

        - DataFrame/ndarray : ``categories[col]`` holds the categories expected
          in the feature col.

    drop : 'first', None, a dict or a list, default=None
        Specifies a methodology to use to drop one of the categories per
        feature. This is useful in situations where perfectly collinear
        features cause problems, such as when feeding the resulting data
        into a neural network or an unregularized regression.

        - None : retain all features (the default).

        - 'first' : drop the first category in each feature. If only one
          category is present, the feature will be dropped entirely.

        - dict/list : ``drop[col]`` is the category in feature col that
          should be dropped.

    sparse_output : bool, default=True
        This feature is not fully supported by cupy
        yet, causing incorrect values when computing one hot encodings.
        See https://github.com/cupy/cupy/issues/3223

        .. versionadded:: 24.06
           `sparse` was renamed to `sparse_output`

    dtype : number type, default=np.float
        Desired datatype of transform's output.
    handle_unknown : {'error', 'ignore'}, default='error'
        Whether to raise an error or ignore if an unknown categorical feature
        is present during transform (default is to raise). When this parameter
        is set to 'ignore' and an unknown category is encountered during
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
    drop_idx_ : array of shape (n_features,)
        ``drop_idx_[i]`` is the index in ``categories_[i]`` of the category to
        be dropped for each feature. None if all the transformed features will
        be retained.
    """

    def __init__(
        self,
        *,
        categories="auto",
        drop=None,
        sparse_output=True,
        dtype=np.float32,
        handle_unknown="error",
        verbose=False,
        output_type=None,
    ):
        super().__init__(verbose=verbose, output_type=output_type)
        self.categories = categories
        self.sparse_output = sparse_output
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.drop = drop
        self.drop_idx_ = None
        self._features = None
        self._encoders = None
        self.input_type = None
        # This parameter validation should be performed in `fit` instead
        # of in the constructor. Hence the awkwark `if` clause
        if sparse_output and np.dtype(dtype) not in ["f", "d", "F", "D"]:
            raise ValueError(
                "Only float32, float64, complex64 and complex128 "
                "are supported when using sparse_output"
            )

    def _validate_keywords(self):
        if self.handle_unknown not in ("error", "ignore"):
            msg = (
                "handle_unknown should be either 'error' or 'ignore', "
                "got {0}.".format(self.handle_unknown)
            )
            raise ValueError(msg)
        # If we have both dropped columns and ignored unknown
        # values, there will be ambiguous cells. This creates difficulties
        # in interpreting the model.
        if self.drop is not None and self.handle_unknown != "error":
            raise ValueError(
                "`handle_unknown` must be 'error' when the drop parameter is "
                "specified, as both would create categories that are all "
                "zero."
            )

    def __sklearn_is_fitted__(self):
        # TODO: fix state management of this class so `check_is_fitted` works
        # without special casing
        return getattr(self, "_fitted", False)

    def _compute_drop_idx(self):
        """Helper to compute indices to drop from category to drop."""
        if self.drop is None:
            return None
        elif isinstance(self.drop, str) and self.drop == "first":
            return {feature: 0 for feature in self._encoders.keys()}
        elif isinstance(self.drop, (dict, list)):
            if isinstance(self.drop, list):
                self.drop = dict(zip(range(len(self.drop)), self.drop))
            if len(self.drop.keys()) != len(self._encoders):
                msg = (
                    "`drop` should have as many columns as the number "
                    "of features ({}), got {}"
                )
                raise ValueError(
                    msg.format(len(self._encoders), len(self.drop.keys()))
                )
            drop_idx = dict()
            for feature in self.drop.keys():
                self.drop[feature] = cudf.Series(self.drop[feature])
                if len(self.drop[feature]) != 1:
                    msg = (
                        "Trying to drop multiple values for feature {}, "
                        "this is not supported."
                    ).format(feature)
                    raise ValueError(msg)
                cats = self._encoders[feature].classes_
                drop_vals = self.drop[feature]
                cats = cudf.Series(cats)
                # Match the dtype of the drop values to the dtype of the categories
                # seen during `fit`. In particular if arrow strings and object dtypes
                # are used, then having a mix means `isin` won't work correctly.
                if drop_vals.dtype != cats.dtype:
                    drop_vals = drop_vals.astype(cats.dtype)
                if not drop_vals.isin(cats).all():
                    msg = (
                        "Some categories for feature {} were supposed "
                        "to be dropped, but were not found in the encoder "
                        "categories.".format(feature)
                    )
                    raise ValueError(msg)
                idx = cats.isin(drop_vals)
                drop_idx[feature] = cp.asarray(cats[idx].index)
            return drop_idx
        else:
            msg = (
                "Wrong input for parameter `drop`. Expected "
                "'first', None or a dict, got {}"
            )
            raise ValueError(msg.format(type(self.drop)))

    def _has_unknown(self, X_cat, encoder_cat):
        """Check if X_cat has categories that are not present in encoder_cat."""
        if X_cat.dtype != encoder_cat.dtype:
            encoder_cat = encoder_cat.astype(X_cat.dtype)
        return not X_cat.isin(encoder_cat).all()

    @generate_docstring(y=None)
    def fit(self, X, y=None):
        """Fit OneHotEncoder to X."""
        self._validate_keywords()
        self._fit(X, True)
        return self

    @generate_docstring(
        y=None,
        return_values={
            "name": "X_out",
            "description": "Transformed input.",
            "type": "sparse matrix if sparse_output=True else a 2-d array",
        },
    )
    def fit_transform(self, X, y=None):
        """
        Fit OneHotEncoder to X, then transform X.  Equivalent to fit(X).transform(X).

        """
        X = self._check_input(X)
        return self.fit(X).transform(X)

    @generate_docstring(
        return_values={
            "name": "X_out",
            "description": "Transformed input.",
            "type": "sparse matrix if sparse_output=True else a 2-d array",
        }
    )
    @mlfunc(convert_output=False)
    def transform(self, X):
        """Transform X using one-hot encoding."""
        check_is_fitted(self)
        check_features(self, X)

        X = self._check_input(X)

        cols, rows = list(), list()
        col_idx = None
        j = 0

        try:
            for feature in X.columns:
                encoder = self._encoders[feature]
                with cuml.using_output_type("cudf"):
                    col_idx = encoder.transform(X[feature])
                idx_to_keep = col_idx.notnull().to_cupy()
                col_idx = col_idx.dropna().to_cupy()

                # Simple test to auto upscale col_idx type as needed
                # First, determine the maximum value we will add assuming
                # monotonically increasing up to len(encoder.classes_)
                # Ensure we dont go negative by clamping to 0
                max_value = int(max(len(encoder.classes_) - 1, 0) + j)
                min_dtype = np.min_scalar_type(max_value)
                col_idx = col_idx.astype(min_dtype, copy=False)

                # increase indices to take previous features into account
                col_idx += j

                # Filter out rows with null values
                row_idx = cp.arange(len(X))[idx_to_keep]

                if self.drop_idx_ is not None:
                    drop_idx = self.drop_idx_[feature] + j
                    mask = cp.ones(col_idx.shape, dtype=bool)
                    mask[col_idx == drop_idx] = False
                    col_idx = col_idx[mask]
                    row_idx = row_idx[mask]
                    # account for dropped category in indices
                    col_idx[col_idx > drop_idx] -= 1
                    # account for dropped category in current cats number
                    j -= 1

                j += len(encoder.classes_)
                cols.append(col_idx)
                rows.append(row_idx)

            cols = cp.concatenate(cols)
            rows = cp.concatenate(rows)
            val = cp.ones(rows.shape[0], dtype=self.dtype)
            ohe = cupyx.scipy.sparse.coo_matrix(
                (val, (rows, cols)), shape=(len(X), j), dtype=self.dtype
            )

            if not self.sparse_output:
                ohe = ohe.toarray()

            return ohe

        except TypeError as e:
            # Append to cols to include the column that threw the error
            cols.append(col_idx)

            # Build a string showing what the types are
            input_types_str = ", ".join([str(x.dtype) for x in cols])

            raise TypeError(
                "A TypeError occurred while calculating column "
                "category indices, most likely due to integer overflow. This "
                "can occur when columns have a large difference in the number "
                "of categories, resulting in different category code dtypes "
                "for different columns."
                "Calculated column code dtypes: {}.\n"
                "Internal Error: {}".format(input_types_str, repr(e))
            )

    @mlfunc(convert_output=False)
    def inverse_transform(self, X):
        """Convert the data back to the original representation. In case unknown
        categories are encountered (all zeros in the one-hot encoding), ``None`` is used
        to represent this category.

        The return type is the same as the type of the input used by the first
        call to fit on this estimator instance.

        Parameters
        ----------
        X : array-like or sparse matrix, shape [n_samples, n_encoded_features]
            The transformed data.

        Returns
        -------
        X_tr : cudf.DataFrame or cupy.ndarray
            Inverse transformed array.
        """
        check_is_fitted(self)

        if cupyx.scipy.sparse.issparse(X):
            # cupyx.scipy.sparse 7.x does not support argmax,
            # when we upgrade cupy to 8.x, we should add a condition in the
            # if close: `and not cupyx.scipy.sparse.issparsecsc(X)`
            # and change the following line by `X = X.tocsc()`
            X = X.toarray()
        result = cudf.DataFrame(columns=self._encoders.keys())
        j = 0
        for feature in self._encoders.keys():
            feature_enc = self._encoders[feature]
            cats = cudf.Series(feature_enc.classes_)

            if self.drop is not None:
                # Remove dropped categories
                dropped_class_idx = cudf.Series(self.drop_idx_[feature])
                dropped_class_mask = cats.isin(cats[dropped_class_idx])
                if len(cats) == 1:
                    inv = cudf.Series(Index([cats[0]]).repeat(X.shape[0]))
                    result[feature] = inv
                    continue
                cats = cats[~dropped_class_mask]

            enc_size = len(cats)
            x_feature = X[:, j : j + enc_size]
            idx = cp.argmax(x_feature, axis=1)
            inv = cudf.Series(cats.iloc[idx]).reset_index(drop=True)

            if self.handle_unknown == "ignore":
                not_null_idx = x_feature.any(axis=1)
                inv.iloc[~not_null_idx] = None
            elif self.drop is not None:
                # drop will either be None or handle_unknown will be error. If
                # self.drop is not None, then we can safely assume that all of
                # the nulls in each column are the dropped value
                dropped_mask = cp.asarray(x_feature.sum(axis=1) == 0).flatten()
                if dropped_mask.any():
                    with cuml.using_output_type("cudf"):
                        inv[dropped_mask] = feature_enc.inverse_transform(
                            cudf.Series(self.drop_idx_[feature])
                        )[0]

            result[feature] = inv
            j += enc_size
        if self.input_type == "array":
            try:
                result = result.to_cupy()
            except ValueError:
                warnings.warn(
                    "The input one hot encoding contains rows with "
                    "unknown categories. Since device arrays do not "
                    "support null values, the output will be "
                    "returned as a DataFrame "
                    "instead."
                )
        return result

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
            # TODO: when `drop_idx_` is actually implemented properly, this can
            # be simplified to
            # drop_idx = None if self.drop_idx_ is None else self.drop_idx_[i]
            drop_idx = cp.asnumpy(
                None
                if self.drop_idx_ is None
                else self.drop_idx_[self._features[i]]
            ).item()
            if drop_idx is not None:
                cats = np.delete(cats, drop_idx)
            out.extend(f"{col}_{val!s}" for val in cats)

        return np.array(out, dtype=object)

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + [
            "categories",
            "drop",
            "sparse_output",
            "dtype",
            "handle_unknown",
        ]


def _slice_feat(X, i):
    if hasattr(X, "iloc"):
        return X[i]
    return X[:, i]


def _get_output(
    output_type: Optional[str],
    input_type: Optional[str],
    out: "cudf.DataFrame",
    dtype,
):
    if output_type in (None, "input"):
        if input_type == "array":
            output_type = "cupy"
        elif input_type == "df":
            output_type = "cudf"

    if output_type is None:
        output_type = "cupy"

    if output_type == "cudf":
        return out
    elif output_type == "cupy":
        return out.astype(dtype).to_cupy(na_value=np.nan)
    elif output_type == "numpy":
        return cp.asnumpy(out.to_cupy(na_value=np.nan, dtype=dtype))
    elif output_type == "pandas":
        import cudf.pandas

        if cudf.pandas.LOADED:
            return cudf.pandas.as_proxy_object(out)
        return out.to_pandas()
    else:
        raise ValueError("Unsupported output type.")


class OrdinalEncoder(BaseEncoder):
    def __init__(
        self,
        *,
        categories="auto",
        dtype=np.float64,
        handle_unknown="error",
        verbose=False,
        output_type=None,
    ) -> None:
        """Encode categorical features as an integer array.

        The input to this transformer should be an :py:class:`cudf.DataFrame` or a
        :py:class:`cupy.ndarray`, denoting the unique values taken on by categorical
        (discrete) features. The features are converted to ordinal integers. This
        results in a single column of integers (0 to n_categories - 1) per feature.

        Parameters
        ----------
        categories : 'auto' an cupy.ndarray or a cudf.DataFrame, default='auto'
                     Categories (unique values) per feature:
            - 'auto' : Determine categories automatically from the training data.
            - DataFrame/ndarray : ``categories[col]`` holds the categories expected
              in the feature col.
        handle_unknown : {'error', 'ignore'}, default='error'
            Whether to raise an error or ignore if an unknown categorical feature is
            present during transform (default is to raise). When this parameter is set
            to 'ignore' and an unknown category is encountered during transform, the
            resulting encoded value would be null when output type is cudf
            dataframe.
        verbose : int or boolean, default=False
            Sets logging level. It must be one of `cuml.common.logger.level_*`.  See
            :ref:`verbosity-levels` for more info.
        output_type : {None, 'input', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
            Return results and set estimator attributes to the indicated output
            type. If None, the output type set at the module level
            (`cuml.global_settings.output_type`) will be used. See
            :ref:`output-data-type-configuration` for more info.
        """
        super().__init__(verbose=verbose, output_type=output_type)

        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

        self.input_type = None

    @generate_docstring(y=None)
    def fit(self, X, y=None) -> "OrdinalEncoder":
        """Fit Ordinal to X."""
        self._fit(X, need_drop=False)
        return self

    @generate_docstring(
        return_values={
            "name": "X_out",
            "description": "Transformed input.",
            "type": "Type is specified by the `output_type` parameter.",
        }
    )
    @mlfunc(convert_output=False)
    def transform(self, X):
        """Transform X using ordinal encoding."""
        check_is_fitted(self)
        check_features(self, X)

        result = {}
        for feature in self._features:
            Xi = _slice_feat(X, feature)
            with cuml.using_output_type("cudf"):
                col_idx = self._encoders[feature].transform(Xi)
            result[feature] = col_idx

        r = cudf.DataFrame(result)
        return _get_output(self.output_type, self.input_type, r, self.dtype)

    @generate_docstring(
        y=None,
        return_values={
            "name": "X_out",
            "description": "Transformed input.",
            "type": "Type is specified by the `output_type` parameter.",
        },
    )
    def fit_transform(self, X, y=None):
        """Fit OrdinalEncoder to X, then transform X. Equivalent to fit(X).transform(X)."""
        X = self._check_input(X)
        return self.fit(X).transform(X)

    @mlfunc(convert_output=False)
    def inverse_transform(self, X):
        """Convert the data back to the original representation.

        Parameters
        ----------
        X : array-like or sparse matrix, shape [n_samples, n_encoded_features]
            The transformed data.

        Returns
        -------
        X_tr : Type is specified by the `output_type` parameter.
            Inverse transformed array.
        """
        check_is_fitted(self)

        result = {}
        for feature in self._features:
            Xi = _slice_feat(X, feature)
            with cuml.using_output_type("cudf"):
                inv = self._encoders[feature].inverse_transform(Xi)
            result[feature] = inv

        r = cudf.DataFrame(result)
        return _get_output(self.output_type, self.input_type, r, self.dtype)

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + [
            "categories",
            "dtype",
            "handle_unknown",
        ]
