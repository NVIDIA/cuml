# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import dask_cudf

from cuml.dask.common.base import (
    BaseEstimator,
    DelayedInverseTransformMixin,
    DelayedTransformMixin,
)
from cuml.dask.common.dask_arr_utils import to_dask_cudf


class DelayedFitTransformMixin:
    def fit_transform(self, X, delayed=True):
        """Fit the encoder to X, then transform X. Equivalent to fit(X).transform(X).

        Parameters
        ----------
        X : Dask cuDF DataFrame or CuPy backed Dask Array
            The data to encode.
        delayed : bool (default = True)
            Whether to execute as a delayed task or eager.

        Returns
        -------
        out : Dask cuDF DataFrame or CuPy backed Dask Array
            Distributed object containing the transformed data
        """
        return self.fit(X).transform(X, delayed=delayed)


class OneHotEncoder(
    BaseEstimator,
    DelayedTransformMixin,
    DelayedInverseTransformMixin,
    DelayedFitTransformMixin,
):
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
    """

    def fit(self, X):
        """Fit a multi-node multi-gpu OneHotEncoder to X.

        Parameters
        ----------
        X : Dask cuDF DataFrame or CuPy backed Dask Array
            The data to determine the categories of each feature.

        Returns
        -------
        self
        """
        from cuml.preprocessing import OneHotEncoder

        model = OneHotEncoder(**self.kwargs)

        if isinstance(X, dask_cudf.DataFrame):
            self.datatype = model._input_type = model.output_type = "cudf"
        else:
            self.datatype = model._input_type = model.output_type = "cupy"
            X = to_dask_cudf(X, client=self.client)

        X_list = self.client.compute(
            [X.iloc[:, i].drop_duplicates() for i in range(X.shape[1])],
            sync=True,
        )
        model._fit(X_list, unique=True)
        self._set_internal_model(model)

        return self

    def transform(self, X, delayed=True):
        """Transform X using one-hot encoding.

        Parameters
        ----------
        X : Dask cuDF DataFrame or CuPy backed Dask Array
            The data to encode.
        delayed : bool (default = True)
            Whether to execute as a delayed task or eager.

        Returns
        -------
        out : Dask cuDF DataFrame or CuPy backed Dask Array
            Distributed object containing the transformed input.
        """
        output_collection_type = (
            "cupy" if self.kwargs.get("sparse_output", True) else self.datatype
        )
        return self._transform(
            X,
            n_dims=2,
            delayed=delayed,
            output_dtype=self._get_internal_model().dtype,
            output_collection_type=output_collection_type,
        )

    def inverse_transform(self, X, delayed=True):
        """Convert the data back to the original representation.

        Parameters
        ----------
        X : CuPy backed Dask Array, shape [n_samples, n_encoded_features]
            The transformed data.
        delayed : bool (default = True)
            Whether to execute as a delayed task or eager.

        Returns
        -------
        X_tr : Dask cuDF DataFrame or CuPy backed Dask Array
            Distributed object containing the inverse transformed array.
        """
        dtype = self._get_internal_model().dtype
        return self._inverse_transform(
            X,
            n_dims=2,
            delayed=delayed,
            output_dtype=dtype,
            output_collection_type=self.datatype,
        )


class OrdinalEncoder(
    BaseEstimator,
    DelayedTransformMixin,
    DelayedInverseTransformMixin,
    DelayedFitTransformMixin,
):
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
    """

    def fit(self, X):
        """Fit Ordinal to X.

        Parameters
        ----------
        X : :py:class:`dask_cudf.DataFrame` or a CuPy backed :py:class:`dask.array.Array`.
            shape = (n_samples, n_features) The data to determine the categories of each
            feature.

        Returns
        -------
        self
        """
        from cuml.preprocessing import OrdinalEncoder

        model = OrdinalEncoder(**self.kwargs)

        if isinstance(X, dask_cudf.DataFrame):
            self.datatype = model._input_type = model.output_type = "cudf"
        else:
            self.datatype = model._input_type = model.output_type = "cupy"
            X = to_dask_cudf(X, client=self.client)

        X_list = self.client.compute(
            [X.iloc[:, i].drop_duplicates() for i in range(X.shape[1])],
            sync=True,
        )
        model._fit(X_list, unique=True)
        self._set_internal_model(model)

        return self

    def transform(self, X, delayed=True):
        """Transform X using ordinal encoding.

        Parameters
        ----------
        X : :py:class:`dask_cudf.DataFrame` or cupy backed dask array.  The data to
            encode.

        Returns
        -------
        X_out :
            Transformed input.
        """
        return self._transform(
            X,
            n_dims=2,
            delayed=delayed,
            output_dtype=self._get_internal_model().dtype,
            output_collection_type=self.datatype,
        )

    def inverse_transform(self, X, delayed=True):
        """Convert the data back to the original representation.

        Parameters
        ----------
        X : :py:class:`dask_cudf.DataFrame` or cupy backed dask array.
        delayed : bool (default = True)
            Whether to execute as a delayed task or eager.

        Returns
        -------
        X_tr :
            Distributed object containing the inverse transformed array.
        """
        dtype = self._get_internal_model().dtype
        return self._inverse_transform(
            X,
            n_dims=2,
            delayed=delayed,
            output_dtype=dtype,
            output_collection_type=self.datatype,
        )
