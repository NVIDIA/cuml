#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import cupyx.scipy.sparse
import numpy as np
from sklearn.base import ClassNamePrefixFeaturesOutMixin

from cuml.common.doc_utils import generate_docstring
from cuml.common.sparse import is_sparse, sparse_cov_and_mean
from cuml.internals.base import Base, get_handle
from cuml.internals.interop import InteropMixin, UnsupportedOnGPU
from cuml.internals.mixins import FMajorInputTagMixin, SparseInputTagMixin
from cuml.internals.outputs import ReflectedAttr, mlfunc
from cuml.internals.validation import (
    check_array,
    check_inputs,
    check_is_fitted,
)

from libc.stdint cimport uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t

from cuml.decomposition.common cimport paramsPCA, solver


cdef extern from "cuml/decomposition/pca.hpp" namespace "ML" nogil:

    cdef void pcaFit(handle_t& handle,
                     float *input,
                     float *components,
                     float *explained_var,
                     float *explained_var_ratio,
                     float *singular_vals,
                     float *mu,
                     float *noise_vars,
                     const paramsPCA &prms) except +

    cdef void pcaFit(handle_t& handle,
                     double *input,
                     double *components,
                     double *explained_var,
                     double *explained_var_ratio,
                     double *singular_vals,
                     double *mu,
                     double *noise_vars,
                     const paramsPCA &prms) except +

    cdef void pcaInverseTransform(handle_t& handle,
                                  float *trans_input,
                                  float *components,
                                  float *singular_vals,
                                  float *mu,
                                  float *input,
                                  const paramsPCA &prms) except +

    cdef void pcaInverseTransform(handle_t& handle,
                                  double *trans_input,
                                  double *components,
                                  double *singular_vals,
                                  double *mu,
                                  double *input,
                                  const paramsPCA &prms) except +

    cdef void pcaTransform(handle_t& handle,
                           float *input,
                           float *components,
                           float *trans_input,
                           float *singular_vals,
                           float *mu,
                           const paramsPCA &prms) except +

    cdef void pcaTransform(handle_t& handle,
                           double *input,
                           double *components,
                           double *trans_input,
                           double *singular_vals,
                           double *mu,
                           const paramsPCA &prms) except +


class PCA(InteropMixin,
          FMajorInputTagMixin,
          SparseInputTagMixin,
          ClassNamePrefixFeaturesOutMixin,
          Base):

    """
    PCA (Principal Component Analysis) is a fundamental dimensionality
    reduction technique used to combine features in X in linear combinations
    such that each new component captures the most information or variance of
    the data. N_components is usually small, say at 3, where it can be used for
    data visualization, data compression and exploratory analysis.

    cuML's PCA expects an array-like object or cuDF DataFrame, and provides 2
    algorithms Full and Jacobi. Full (default) uses a full eigendecomposition
    then selects the top K eigenvectors. The Jacobi algorithm is much faster
    as it iteratively tries to correct the top K eigenvectors, but might be
    less accurate.

    Examples
    --------

    .. code-block:: python

        >>> # Both import methods supported
        >>> from cuml import PCA
        >>> from cuml.decomposition import PCA

        >>> import cudf
        >>> import cupy as cp

        >>> gdf_float = cudf.DataFrame()
        >>> gdf_float['0'] = cp.asarray([1.0,2.0,5.0], dtype = cp.float32)
        >>> gdf_float['1'] = cp.asarray([4.0,2.0,1.0], dtype = cp.float32)
        >>> gdf_float['2'] = cp.asarray([4.0,2.0,1.0], dtype = cp.float32)

        >>> pca_float = PCA(n_components = 2)
        >>> pca_float.fit(gdf_float)
        PCA(n_components=2)

        >>> print(f'components: {pca_float.components_}') # doctest: +SKIP
        components: 0           1           2
        0  0.69225764  -0.5102837 -0.51028395
        1 -0.72165036 -0.48949987  -0.4895003
        >>> print(f'explained variance: {pca_float.explained_variance_}')
        explained variance: 0   8.510...
        1 0.489...
        dtype: float32
        >>> exp_var = pca_float.explained_variance_ratio_
        >>> print(f'explained variance ratio: {exp_var}')
        explained variance ratio: 0   0.9456...
        1 0.054...
        dtype: float32

        >>> print(f'singular values: {pca_float.singular_values_}')
        singular values: 0 4.125...
        1 0.989...
        dtype: float32
        >>> print(f'mean: {pca_float.mean_}')
        mean: 0 2.666...
        1 2.333...
        2 2.333...
        dtype: float32
        >>> trans_gdf_float = pca_float.transform(gdf_float)
        >>> print(f'Inverse: {trans_gdf_float}') # doctest: +SKIP
        Inverse: 0           1
        0   -2.8547091 -0.42891636
        1 -0.121316016  0.80743366
        2    2.9760244 -0.37851727
        >>> input_gdf_float = pca_float.inverse_transform(trans_gdf_float)
        >>> print(f'Input: {input_gdf_float}') # doctest: +SKIP
        Input: 0         1         2
        0 1.0 4.0 4.0
        1 2.0 2.0 2.0
        2 5.0 1.0 1.0

    Parameters
    ----------
    copy : boolean (default = True)
        If True, then copies data then removes mean from data. False might
        cause data to be overwritten with its mean centered version.
    iterated_power : int (default = 15)
        Used in Jacobi solver. The more iterations, the more accurate, but
        slower.
    n_components : int (default = None)
        The number of top K singular vectors / values you want.
        Must be <= number(columns). If n_components is not set, then all
        components are kept:

            ``n_components = min(n_samples, n_features)``

    svd_solver : 'full' or 'jacobi' or 'auto' (default = 'full')
        Full uses a eigendecomposition of the covariance matrix then discards
        components.
        Jacobi is much faster as it iteratively corrects, but is less accurate.
    tol : float (default = 1e-7)
        Used if algorithm = "jacobi". Smaller tolerance can increase accuracy,
        but but will slow down the algorithm's convergence.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    whiten : boolean (default = False)
        If True, de-correlates the components. This is done by dividing them by
        the corresponding singular values then multiplying by sqrt(n_samples).
        Whitening allows each component to have unit variance and removes
        multi-collinearity. It might be beneficial for downstream
        tasks like LinearRegression where correlated features cause problems.
    output_type : {None, 'input', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Attributes
    ----------
    components_ : array
        The top K components (VT.T[:,:n_components]) in U, S, VT = svd(X)
    explained_variance_ : array
        How much each component explains the variance in the data given by S**2
    explained_variance_ratio_ : array
        How much in % the variance is explained given by S**2/sum(S**2)
    singular_values_ : array
        The top K singular values. Remember all singular values >= 0
    mean_ : array
        The column wise mean of X. Used to mean - center the data first.
    noise_variance_ : float
        From Bishop 1999's Textbook. Used in later tasks like calculating the
        estimated covariance of X.

    Notes
    -----
    PCA considers linear combinations of features, specifically those that
    maximize global variance structure. This means PCA is fantastic for global
    structure analyses, but weak for local relationships. Consider UMAP or
    T-SNE for a locally important embedding.

    **Applications of PCA**

        PCA is used extensively in practice for data visualization and data
        compression. It has been used to visualize extremely large word
        embeddings like Word2Vec and GloVe in 2 or 3 dimensions, large
        datasets of everyday objects and images, and used to distinguish
        between cancerous cells from healthy cells.


    For additional docs, see `scikitlearn's PCA
    <http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_.
    """

    components_ = ReflectedAttr()
    explained_variance_ = ReflectedAttr()
    explained_variance_ratio_ = ReflectedAttr()
    singular_values_ = ReflectedAttr()
    mean_ = ReflectedAttr()

    _cpu_class_path = "sklearn.decomposition.PCA"

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + \
            ["copy", "iterated_power", "n_components", "svd_solver", "tol", "whiten"]

    @classmethod
    def _params_from_cpu(cls, model):
        if model.n_components == "mle":
            raise UnsupportedOnGPU("`n_components='mle'` is not supported")
        if model.n_components == 0:
            raise UnsupportedOnGPU("`n_components=0` is not supported")

        svd_solver = "auto" if model.svd_solver == "auto" else "full"

        # Since the solvers are different, we want to adjust the tolerances used.
        # TODO: here we only adjust the default tolerance, there's likely a better
        # conversion equation we should apply.
        if (tol := model.tol) == 0.0:
            tol = 1e-7

        if (iterated_power := model.iterated_power) == "auto":
            iterated_power = 15

        return {
            "n_components": model.n_components,
            "copy": model.copy,
            "whiten": model.whiten,
            "svd_solver": svd_solver,
            "tol": tol,
            "iterated_power": iterated_power,
        }

    def _params_to_cpu(self):
        # Since the solvers are different, we want to adjust the tolerances used.
        # TODO: here we only adjust the default tolerance, there's likely a better
        # conversion equation we should apply.
        if (tol := self.tol) == 1e-7:
            tol = 0.0

        svd_solver = "auto" if self.svd_solver == "jacobi" else self.svd_solver

        return {
            "n_components": self.n_components,
            "copy": self.copy,
            "whiten": self.whiten,
            "svd_solver": svd_solver,
            "tol": tol,
            "iterated_power": self.iterated_power,
        }

    def _attrs_from_cpu(self, model):
        return {
            "components_": cp.asarray(model.components_, order="F"),
            "explained_variance_": cp.asarray(model.explained_variance_),
            "explained_variance_ratio_": cp.asarray(model.explained_variance_ratio_),
            "singular_values_": cp.asarray(model.singular_values_),
            "mean_": cp.asarray(model.mean_),
            "n_components_": model.n_components_,
            "n_samples_": model.n_samples_,
            "noise_variance_": model.noise_variance_,
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        return {
            "components_": self.components_.get(order="A"),
            "explained_variance_": self.explained_variance_.get(),
            "explained_variance_ratio_": self.explained_variance_ratio_.get(),
            "singular_values_": self.singular_values_.get(),
            "mean_": self.mean_.get(),
            "n_components_": self.n_components_,
            "n_samples_": self.n_samples_,
            "noise_variance_": self.noise_variance_,
            **super()._attrs_to_cpu(model),
        }

    def __init__(
        self,
        *,
        copy=True,
        iterated_power=15,
        n_components=None,
        svd_solver='auto',
        tol=1e-7,
        verbose=False,
        whiten=False,
        output_type=None,
    ):
        super().__init__(verbose=verbose, output_type=output_type)
        self.copy = copy
        self.iterated_power = iterated_power
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.tol = tol
        self.whiten = whiten

    @property
    @mlfunc(convert_output=False)
    def _n_features_out(self):
        """Number of transformed output features."""
        # Exposed to support sklearn's `get_feature_names_out`
        return self.components_.shape[0]

    def _flip_sign(self, components, X):
        """Flip component signs using the components matrix."""
        max_idx = cp.abs(components).argmax(axis=1)
        signs = cp.sign(components[cp.arange(components.shape[0]), max_idx])
        signs[signs == 0] = 1
        return components * signs[:, cp.newaxis]

    def _fit_dense(self, X):
        # Initialize parameters
        cdef paramsPCA params
        params.n_components = self.n_components_
        params.n_rows = self.n_samples_
        params.n_cols = self.n_features_in_
        params.whiten = self.whiten
        params.n_iterations = self.iterated_power
        params.tol = self.tol
        if self.svd_solver in ("auto", "full"):
            params.algorithm = solver.COV_EIG_DQ
        elif self.svd_solver == "jacobi":
            params.algorithm = solver.COV_EIG_JACOBI
        else:
            raise ValueError(
                f"Expected `svd_solver` to be one of ['auto', 'full', 'jacobi'], "
                f"got {self.svd_solver!r}"
            )

        # Allocate output arrays (F-order expected by libcuml))
        components = cp.zeros(
            (self.n_components_, self.n_features_in_), dtype=X.dtype, order="F"
        )
        explained_variance = cp.zeros(self.n_components_, dtype=X.dtype)
        explained_variance_ratio = cp.zeros(self.n_components_, dtype=X.dtype)
        mean = cp.zeros(self.n_features_in_, dtype=X.dtype)
        singular_values = cp.zeros(self.n_components_, dtype=X.dtype)
        noise_variance = cp.zeros(1, dtype=X.dtype)

        cdef uintptr_t X_ptr = X.data.ptr
        cdef uintptr_t components_ptr = components.data.ptr
        cdef uintptr_t explained_variance_ptr = explained_variance.data.ptr
        cdef uintptr_t explained_variance_ratio_ptr = explained_variance_ratio.data.ptr
        cdef uintptr_t singular_values_ptr = singular_values.data.ptr
        cdef uintptr_t mean_ptr = mean.data.ptr
        cdef uintptr_t noise_variance_ptr = noise_variance.data.ptr
        cdef bint fit_float32 = (X.dtype == np.float32)
        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

        # Perform fit
        with nogil:
            if fit_float32:
                pcaFit(
                    handle_[0],
                    <float*> X_ptr,
                    <float*> components_ptr,
                    <float*> explained_variance_ptr,
                    <float*> explained_variance_ratio_ptr,
                    <float*> singular_values_ptr,
                    <float*> mean_ptr,
                    <float*> noise_variance_ptr,
                    params
                )
            else:
                pcaFit(
                    handle_[0],
                    <double*> X_ptr,
                    <double*> components_ptr,
                    <double*> explained_variance_ptr,
                    <double*> explained_variance_ratio_ptr,
                    <double*> singular_values_ptr,
                    <double*> mean_ptr,
                    <double*> noise_variance_ptr,
                    params
                )
        handle.sync()

        # Store results
        self.components_ = components
        self.explained_variance_ = explained_variance
        self.explained_variance_ratio_ = explained_variance_ratio
        self.mean_ = mean
        self.singular_values_ = singular_values
        self.noise_variance_ = float(noise_variance.item())

    def _fit_sparse(self, X):
        covariance, mean = sparse_cov_and_mean(X)

        explained_variance, components = cp.linalg.eigh(covariance, UPLO='U')

        # Reverse the eigen vector and eigen values here because cupy provides
        # them in ascending order.
        explained_variance = explained_variance[::-1]
        components = components[:, ::-1]

        if self.n_components_ < min(self.n_samples_, self.n_features_in_):
            noise_variance = float(explained_variance[self.n_components_:].mean())
        else:
            noise_variance = 0.0

        explained_variance_sum = explained_variance.sum()

        components = self._flip_sign(
            components.T[:self.n_components_, :],
            X
        )
        explained_variance = explained_variance[:self.n_components_]

        explained_variance_ratio = explained_variance / explained_variance_sum
        singular_values = cp.sqrt(
            cp.where(explained_variance < 0, 0, explained_variance) * (X.shape[0] - 1)
        )

        # Store results
        self.components_ = cp.asfortranarray(components)
        self.explained_variance_ = explained_variance
        self.explained_variance_ratio_ = explained_variance_ratio
        self.mean_ = mean
        self.singular_values_ = singular_values
        self.noise_variance_ = noise_variance

    @generate_docstring(X='dense_sparse')
    @mlfunc(set_input_type=True)
    def fit(self, X, y=None) -> "PCA":
        """
        Fit the model with X. y is currently ignored.

        """
        X = check_inputs(
            self,
            X,
            accept_sparse=["csr"],
            accept_large_sparse=True,
            dtype=("float32", "float64"),
            ensure_min_samples=2,
            ensure_min_features=2,
            order="F",
            reset=True,
        )

        n_rows, n_cols = X.shape
        self.n_samples_ = n_rows

        if self.n_components is None:
            self.n_components_ = min(n_rows, n_cols)
        elif self.n_components > n_cols:
            raise ValueError(
                f"`n_components` ({self.n_components}) must be <= than the "
                f"number of features in X ({n_cols})"
            )
        else:
            self.n_components_ = self.n_components

        if is_sparse(X):
            self._fit_sparse(X)
        else:
            self._fit_dense(X)
        return self

    @generate_docstring(X='dense_sparse',
                        return_values={'name': 'trans',
                                       'type': 'dense_sparse',
                                       'description': 'Transformed values',
                                       'shape': '(n_samples, n_components)'})
    @mlfunc(
        set_input_type=True,
        preserve_index=True,
        column_names="feature_names_out",
    )
    def fit_transform(self, X, y=None):
        """
        Fit the model with X and apply the dimensionality reduction on X.

        """
        return self.fit(X).transform(X)

    def _inverse_transform_sparse(self, X, return_sparse=False, sparse_tol=1e-10):
        components = self.components_
        if self.whiten:
            components = cp.sqrt(self.explained_variance_[:, None]) * components

        out = X @ components
        out += self.mean_

        if return_sparse:
            out[out < sparse_tol] = 0
            return cupyx.scipy.sparse.csr_matrix(out)

        return out

    def _inverse_transform_dense(self, X):
        dtype = X.dtype
        n_rows = X.shape[0]

        out = cp.zeros((n_rows, self.n_features_in_), dtype=dtype, order="F")

        cdef paramsPCA params
        params.n_components = self.n_components_
        params.n_rows = n_rows
        params.n_cols = self.n_features_in_
        params.whiten = self.whiten

        cdef uintptr_t X_ptr = X.data.ptr
        cdef uintptr_t X_inv_ptr = out.data.ptr
        cdef uintptr_t components_ptr = self.components_.data.ptr
        cdef uintptr_t singular_values_ptr = self.singular_values_.data.ptr
        cdef uintptr_t mean_ptr = self.mean_.data.ptr
        cdef bool use_float32 = dtype == np.float32
        handle = get_handle()
        cdef handle_t* h_ = <handle_t*><size_t>handle.getHandle()

        with nogil:
            if use_float32:
                pcaInverseTransform(h_[0],
                                    <float*> X_ptr,
                                    <float*> components_ptr,
                                    <float*> singular_values_ptr,
                                    <float*> mean_ptr,
                                    <float*> X_inv_ptr,
                                    params)
            else:
                pcaInverseTransform(h_[0],
                                    <double*> X_ptr,
                                    <double*> components_ptr,
                                    <double*> singular_values_ptr,
                                    <double*> mean_ptr,
                                    <double*> X_inv_ptr,
                                    params)
        handle.sync()

        return out

    @generate_docstring(X='dense_sparse',
                        return_values={'name': 'X_inv',
                                       'type': 'dense_sparse',
                                       'description': 'Transformed values',
                                       'shape': '(n_samples, n_features)'})
    @mlfunc(preserve_index=True, column_names="feature_names_in")
    def inverse_transform(
        self,
        X,
        *,
        return_sparse=False,
        sparse_tol=1e-10,
    ):
        """
        Transform data back to its original space.

        In other words, return an input X_original whose transform would be X.

        """
        check_is_fitted(self)
        X = check_array(
            X,
            accept_sparse=True,
            dtype=self.components_.dtype,
            order="F",
        )
        if X.shape[1] != self.n_components_:
            raise ValueError(
                f"X has {X.shape[1]} columns, but PCA.inverse_transform "
                f"expects {self.n_components_} (one per fitted component)."
            )
        if is_sparse(X):
            return self._inverse_transform_sparse(
                X, return_sparse=return_sparse, sparse_tol=sparse_tol
            )
        return self._inverse_transform_dense(X)

    def _transform_sparse(self, X):
        out = X @ self.components_.T
        out -= (self.mean_.reshape((1, -1)) @ self.components_.T)
        if self.whiten:
            scale = cp.sqrt(self.explained_variance_)
            min_scale = cp.finfo(scale.dtype).eps
            scale[scale < min_scale] = min_scale
            out /= scale
        return out

    def _transform_dense(self, X):
        dtype = X.dtype
        n_rows, n_cols = X.shape

        out = cp.zeros((n_rows, self.n_components_), dtype=dtype, order="F")

        cdef paramsPCA params
        params.n_components = self.n_components_
        params.n_rows = n_rows
        params.n_cols = n_cols
        params.whiten = self.whiten

        cdef uintptr_t X_ptr = X.data.ptr
        cdef uintptr_t out_ptr = out.data.ptr
        cdef uintptr_t components_ptr = self.components_.data.ptr
        cdef uintptr_t singular_values_ptr = self.singular_values_.data.ptr
        cdef uintptr_t mean_ptr = self.mean_.data.ptr
        cdef bool use_float32 = dtype == np.float32
        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

        with nogil:
            if use_float32:
                pcaTransform(
                    handle_[0],
                    <float*> X_ptr,
                    <float*> components_ptr,
                    <float*> out_ptr,
                    <float*> singular_values_ptr,
                    <float*> mean_ptr,
                    params
                )
            else:
                pcaTransform(
                    handle_[0],
                    <double*> X_ptr,
                    <double*> components_ptr,
                    <double*> out_ptr,
                    <double*> singular_values_ptr,
                    <double*> mean_ptr,
                    params
                )
        handle.sync()
        return out

    @generate_docstring(X='dense_sparse',
                        return_values={'name': 'trans',
                                       'type': 'dense_sparse',
                                       'description': 'Transformed values',
                                       'shape': '(n_samples, n_components)'})
    @mlfunc(preserve_index=True, column_names="feature_names_out")
    def transform(self, X):
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted
        from a training set.

        """
        check_is_fitted(self)

        X = check_inputs(
            self,
            X,
            accept_sparse=True,
            dtype=self.components_.dtype,
            order="F",
        )
        if is_sparse(X):
            return self._transform_sparse(X)
        return self._transform_dense(X)
