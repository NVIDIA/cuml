#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import cupyx.scipy.sparse as sp
import numpy as np

from cuml.common.doc_utils import generate_docstring
from cuml.internals.base import Base, get_handle
from cuml.internals.interop import InteropMixin, UnsupportedOnGPU
from cuml.internals.mixins import (
    FMajorInputTagMixin,
    RegressorMixin,
    SparseInputTagMixin,
)
from cuml.internals.outputs import ReflectedAttr, mlfunc
from cuml.internals.validation import check_inputs
from cuml.linear_model.base import (
    LinearPredictMixin,
    cuda_ptr,
    fit_least_squares,
)

from libc.stdint cimport uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t


cdef extern from "cuml/linear_model/glm.hpp" namespace "ML::GLM" nogil:

    cdef void ridgeFit(handle_t& handle,
                       float *input,
                       size_t n_rows,
                       size_t n_cols,
                       float *labels,
                       float *alpha,
                       int n_alpha,
                       float *coef,
                       float *intercept,
                       bool fit_intercept,
                       int algo,
                       float *sample_weight) except +

    cdef void ridgeFit(handle_t& handle,
                       double *input,
                       size_t n_rows,
                       size_t n_cols,
                       double *labels,
                       double *alpha,
                       int n_alpha,
                       double *coef,
                       double *intercept,
                       bool fit_intercept,
                       int algo,
                       double *sample_weight) except +


_SOLVER_CUML_TO_SKLEARN = {
    "auto": "auto",
    "svd": "svd",
    "eig": "cholesky",
    "lsmr": "lsqr",
}


class Ridge(InteropMixin,
            RegressorMixin,
            LinearPredictMixin,
            FMajorInputTagMixin,
            SparseInputTagMixin,
            Base):
    """Linear least squares with L2 regularization.

    Ridge extends LinearRegression by providing L2 regularization on the
    coefficients when predicting response y with a linear combination of the
    predictors in X. It can reduce the variance of the predictors, and improves
    the conditioning of the problem.

    Parameters
    ----------
    alpha : float or array of shape (n_targets,), default=1.0
        Regularization strength - must be a positive float. Larger values
        specify stronger regularization.
    fit_intercept : bool, default=True
        If True, Ridge tries to correct for the global mean of y.
        If False, the model expects that you have centered the data.
    solver : {'auto', 'eig', 'svd', 'lsmr'}, default='auto'
        The solver to use when fitting:

        - 'auto': will select 'eig' if supported, falling back to 'lsmr' if X
          is sparse, and 'svd' otherwise.

        - 'eig': uses an eigendecomposition of the covariance matrix. It is
          faster than SVD, but potentially unstable. It doesn't support
          multi-target ``y`` or sparse ``X``.

        - 'svd': uses an SVD decomposition. It's slower, but stable. It doesn't
          support sparse ``X``.

        - 'lsmr': uses ``cupyx.scipy.sparse.linalg.lsmr``, an iterative algorithm.
          It is typically the fastest, and supports all options.

    tol : float, default=1e-4
        The tolerance used by the ``lsmr`` solver. Has no impact on other solvers.
    max_iter : int, default=None
        Maximum number of iterations for the ``lsmr`` solver. Defaults to ``None``
        for no limit. Has no impact on other solvers.
    copy_X: bool, default=True
        If True, X will never be mutated. Setting to False may reduce memory
        usage, at the cost of potentially mutating X.
    output_type : {None, 'input', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        The estimated coefficients for the linear regression model.
    intercept_ : float or array, shape (n_targets,)
        The independent term. If `fit_intercept` is False, will be 0. Will be
        an array when fit on multi-target y, otherwise will be a float.
    solver_ : str
        The solver that was used at fit time.
    n_iter_ : numpy.ndarray or None, shape (n_targets,)
        The number of iterations the solver ran per-target if the ``'lsmr'``
        solver was used, or ``None`` for other solvers.

    Notes
    -----
    For additional docs, see `Scikit-learn's Ridge Regression
    <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html>`_.

    Examples
    --------
    >>> import cupy as cp
    >>> import cudf
    >>> from cuml import Ridge

    >>> X = cudf.DataFrame()
    >>> X['col1'] = cp.array([1,1,2,2], dtype = cp.float32)
    >>> X['col2'] = cp.array([1,2,2,3], dtype = cp.float32)
    >>> y = cudf.Series(cp.array([6.0, 8.0, 9.0, 11.0], dtype=cp.float32))

    >>> ridge = Ridge(alpha=1e-5).fit(X, y)
    >>> print(ridge.coef_) # doctest: +SKIP
    0 1.000...
    1 1.999...
    >>> print(ridge.intercept_) #doctest: +SKIP
    3.0...
    >>> X_new = cudf.DataFrame()
    >>> X_new['col1'] = cp.array([3,2], dtype=cp.float32)
    >>> X_new['col2'] = cp.array([5,5], dtype=cp.float32)
    >>> preds = ridge.predict(X_new)
    >>> print(preds) # doctest: +SKIP
    0 15.999...
    1 14.999...
    """
    coef_ = ReflectedAttr()
    intercept_ = ReflectedAttr()

    _cpu_class_path = "sklearn.linear_model.Ridge"

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "alpha",
            "fit_intercept",
            "solver",
            "tol",
            "max_iter",
            "copy_X",
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        if model.positive:
            raise UnsupportedOnGPU("`positive=True` is not supported")

        if model.solver == "svd":
            solver = "svd"
        elif model.solver == "lsqr":
            solver = "lsmr"
        elif model.solver == "lbfgs":
            # lbfgs only works in sklearn for positive=True, since we don't
            # support that parameter we want to fallback so sklearn
            # can error appropriately
            raise UnsupportedOnGPU(f"`solver={model.solver!r}` is not supported")
        else:
            solver = "auto"

        return {
            "alpha": model.alpha,
            "fit_intercept": model.fit_intercept,
            "solver": solver,
            "tol": model.tol,
            "max_iter": model.max_iter,
            "copy_X": model.copy_X,
        }

    def _params_to_cpu(self):
        solver = _SOLVER_CUML_TO_SKLEARN[self.solver]
        return {
            "alpha": self.alpha,
            "fit_intercept": self.fit_intercept,
            "solver": solver,
            "tol": self.tol,
            "max_iter": self.max_iter,
            "copy_X": self.copy_X,
        }

    def _attrs_from_cpu(self, model):
        solver = {"svd": "svd", "lsqr": "lsmr"}.get(model.solver_, "eig")
        return {
            "intercept_": (
                model.intercept_
                if cp.isscalar(model.intercept_)
                else cp.asarray(model.intercept_)
            ),
            "coef_": cp.asarray(model.coef_),
            "n_iter_": model.n_iter_,
            "solver_": solver,
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        if model.solver == "auto":
            solver = _SOLVER_CUML_TO_SKLEARN[self.solver_]
        else:
            solver = model.solver

        return {
            "intercept_": (
                self.intercept_
                if cp.isscalar(self.intercept_)
                else cp.asnumpy(self.intercept_)
            ),
            "coef_": cp.asnumpy(self.coef_, order="A"),
            "solver_": solver,
            "n_iter_": self.n_iter_,
            **super()._attrs_to_cpu(model),
        }

    def __init__(
        self,
        alpha=1.0,
        *,
        fit_intercept=True,
        solver="auto",
        tol=1e-4,
        max_iter=None,
        copy_X=True,
        output_type=None,
        verbose=False,
    ):
        super().__init__(verbose=verbose, output_type=output_type)
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.tol = tol
        self.max_iter = max_iter
        self.copy_X = copy_X

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = True
        return tags

    def _fit_eig(
        self,
        X,
        y,
        sample_weight,
        *,
        alpha,
        may_mutate_X,
        may_mutate_y,
        may_mutate_sample_weight,
    ):
        """Fit a Ridge regression using the Eig solver."""
        cdef int n_rows = X.shape[0]
        cdef int n_cols = X.shape[1]

        # The `eig` solver requires X be F-contiguous. Additionally, all inputs
        # are mutated when weighted or `fit_intercept=True`.
        mutates = self.fit_intercept or sample_weight is not None
        X = cp.array(X, order="F", copy=True if mutates and not may_mutate_X else None)
        if mutates and not may_mutate_y:
            y = y.copy()
        if sample_weight is not None and mutates and not may_mutate_sample_weight:
            sample_weight = sample_weight.copy()

        # Allocate outputs
        coef = cp.zeros(n_cols, dtype=X.dtype)

        cdef float intercept_f32
        cdef double intercept_f64
        cdef float alpha_f32 = alpha
        cdef double alpha_f64 = alpha

        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
        cdef uintptr_t X_ptr = X.data.ptr
        cdef uintptr_t y_ptr = y.data.ptr
        cdef uintptr_t coef_ptr = coef.data.ptr
        cdef uintptr_t sample_weight_ptr = (
            0 if sample_weight is None else sample_weight.data.ptr
        )
        cdef bool fit_intercept = self.fit_intercept
        cdef bool use_float32 = X.dtype == np.float32

        with nogil:
            if use_float32:
                ridgeFit(
                    handle_[0],
                    <float*>X_ptr,
                    n_rows,
                    n_cols,
                    <float*>y_ptr,
                    &alpha_f32,
                    1,
                    <float*>coef_ptr,
                    &intercept_f32,
                    fit_intercept,
                    1,
                    <float*>sample_weight_ptr,
                )
            else:
                ridgeFit(
                    handle_[0],
                    <double*>X_ptr,
                    n_rows,
                    n_cols,
                    <double*>y_ptr,
                    &alpha_f64,
                    1,
                    <double*>coef_ptr,
                    &intercept_f64,
                    fit_intercept,
                    1,
                    <double*>sample_weight_ptr,
                )
        handle.sync()

        if self.fit_intercept:
            intercept = intercept_f32 if use_float32 else intercept_f64
            if y.ndim == 1:
                intercept = coef.dtype.type(intercept)
            else:
                intercept = cp.array([intercept], dtype=coef.dtype)
        else:
            intercept = 0.0

        return coef, intercept, None

    @generate_docstring()
    @mlfunc(set_input_type=True)
    def fit(self, X, y, sample_weight=None) -> "Ridge":
        """
        Fit the model with X and y.
        """
        X_orig, y_orig, sample_weight_orig = X, y, sample_weight
        X, y, sample_weight = check_inputs(
            self,
            X,
            y,
            sample_weight,
            dtype=("float32", "float64"),
            ensure_min_samples=2,
            accept_sparse=True,
            accept_large_sparse=True,
            accept_multi_output=True,
            reset=True,
        )
        X_is_copy = cuda_ptr(X) != cuda_ptr(X_orig)
        y_is_copy = cuda_ptr(y) != cuda_ptr(y_orig)
        sample_weight_is_copy = cuda_ptr(sample_weight) != cuda_ptr(sample_weight_orig)

        n_targets = 1 if y.ndim == 1 else y.shape[1]

        # Validate alpha
        if cp.isscalar(self.alpha):
            alpha = self.alpha
            if self.alpha < 0.0:
                raise ValueError(f"alpha must be non-negative, got {self.alpha}")
        else:
            alpha = cp.asarray(self.alpha, dtype=X.dtype).ravel()
            if (alpha < 0).any():
                raise ValueError(f"alpha must be non-negative, got {self.alpha}")
            if alpha.shape[0] == 1:
                alpha = alpha.item()
            elif alpha.shape[0] != n_targets:
                raise ValueError(
                    f"Number of targets and number of penalties do not correspond: "
                    f"{n_targets} != {alpha.shape[0]}"
                )

        # Validate and select solver
        _SUPPORTED_SOLVERS = ["auto", "eig", "svd", "lsmr"]
        if (solver := self.solver) not in _SUPPORTED_SOLVERS:
            raise ValueError(
                f"Expected `solver` to be one of {_SUPPORTED_SOLVERS}, got {solver!r}"
            )

        if solver == "eig":
            if X.shape[1] == 1:
                raise ValueError(
                    "solver='eig' doesn't support X with 1 column, please select "
                    "solver='svd' or solver='auto' instead"
                )
            if n_targets != 1:
                raise ValueError(
                    "solver='eig' doesn't support multi-target y, please select "
                    "solver='svd' or solver='auto' instead"
                )
        elif solver == "auto":
            if sp.issparse(X):
                solver = "lsmr"
            elif X.shape[1] == 1 or n_targets != 1:
                solver = "svd"
            else:
                solver = "eig"

        if sp.issparse(X) and solver != "lsmr":
            raise ValueError(
                f"solver={solver!r} doesn't support sparse X, please select "
                "solver='lsmr' or solver='auto' instead"
            )

        # Perform fit
        if solver == "eig":
            coef, intercept, n_iter = self._fit_eig(
                X,
                y,
                sample_weight,
                alpha=alpha,
                may_mutate_X=X_is_copy or not self.copy_X,
                may_mutate_y=y_is_copy,
                may_mutate_sample_weight=sample_weight_is_copy,
            )
        else:
            coef, intercept, n_iter = fit_least_squares(
                X,
                y,
                sample_weight=sample_weight,
                fit_intercept=self.fit_intercept,
                alpha=alpha,
                tol=self.tol,
                max_iter=self.max_iter,
                may_mutate_X=X_is_copy or not self.copy_X,
                may_mutate_y=y_is_copy,
                solver=solver,
            )

        if y.ndim == 2 and y.shape[1] == 1:
            coef = coef.ravel()

        self.coef_ = coef
        self.intercept_ = intercept
        self.n_iter_ = n_iter
        self.solver_ = solver

        return self
