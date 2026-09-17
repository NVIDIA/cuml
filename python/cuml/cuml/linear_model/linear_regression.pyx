# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import enum
import warnings

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

    cdef void olsFit(handle_t& handle,
                     float *input,
                     size_t n_rows,
                     size_t n_cols,
                     float *labels,
                     float *coef,
                     float *intercept,
                     bool fit_intercept,
                     int algo,
                     float *sample_weight) except +

    cdef void olsFit(handle_t& handle,
                     double *input,
                     size_t n_rows,
                     size_t n_cols,
                     double *labels,
                     double *coef,
                     double *intercept,
                     bool fit_intercept,
                     int algo,
                     double *sample_weight) except +


class Algo(enum.IntEnum):
    """The libcuml solver algorithm"""
    SVD = 0
    EIG = 1
    QR = 2
    SVD_QR = 3

    @classmethod
    def parse(cls, name):
        out = {
            "svd": cls.SVD,
            "eig": cls.EIG,
            "qr": cls.QR,
            "svd-qr": cls.SVD_QR,
            "svd-jacobi": cls.SVD
        }.get(name)
        if out is None:
            raise ValueError(f"algorithm {name!r} is not supported")
        return out


class LinearRegression(InteropMixin,
                       LinearPredictMixin,
                       RegressorMixin,
                       FMajorInputTagMixin,
                       SparseInputTagMixin,
                       Base):
    """
    Ordinary least squares Linear Regression.

    Parameters
    ----------
    algorithm : {'auto', 'eig', 'svd', 'lsmr', 'qr', 'svd-qr', 'svd-jacobi'}, default='auto'
        The algorithm to use when fitting:

        - 'auto': will select 'eig' if supported, falling back to 'lsmr' if X
          is sparse, and 'svd' otherwise.

        - 'eig': uses an eigendecomposition of the covariance matrix. It is
          faster than SVD, but potentially unstable. It doesn't support
          multi-target ``y`` or sparse ``X``.

        - 'svd' or 'svd-jacobi': uses an SVD decomposition. It's slower, but
          stable. It doesn't support sparse ``X``.

        - 'lsmr': uses ``cupyx.scipy.sparse.linalg.lsmr``, an iterative
          algorithm. It supports all input types and is typically very fast.

        - 'qr': uses QR decomposition and solves ``Rx = Q^T y``. It's faster
          than SVD, but doesn't support multi-target ``y`` or sparse ``X``.

        - 'svd-qr': computes SVD decomposition using QR algorithm. It's the
          slowest option. It doesn't support multi-target ``y`` or sparse ``X``.

    fit_intercept : boolean (default = True)
        If True, LinearRegression tries to correct for the global mean of y.
        If False, the model expects that you have centered the data.
    copy_X : boolean, default=True
        If True, X will never be mutated. Setting to False may reduce memory
        usage, at the cost of potentially mutating X.
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
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        The estimated coefficients for the linear regression model.
    intercept_ : float or array, shape (n_targets,)
        The independent term. If `fit_intercept` is False, will be 0. Will be
        an array when fit on multi-target y, otherwise will be a float.

    Notes
    -----
    LinearRegression suffers from multicollinearity (when columns are
    correlated with each other), and variance explosions from outliers.
    Consider using :class:`Ridge` to fix the multicollinearity problem, and
    consider maybe first :class:`DBSCAN` to remove the outliers, or
    statistical analysis to filter possible outliers.

    **Applications of LinearRegression**

        LinearRegression is used in regression tasks where one wants to predict
        say sales or house prices. It is also used in extrapolation or time
        series tasks, dynamic systems modelling and many other machine learning
        tasks. This model should be first tried if the machine learning problem
        is a regression task (predicting a continuous variable).

    For additional information, see scikit-learn's documentation for
    :class:`sklearn.linear_model.LinearRegression`.

    For an additional example see `the OLS notebook
    <https://github.com/NVIDIA/cuml/blob/main/notebooks/linear_regression_demo.ipynb>`__.

    Examples
    --------
    >>> import cupy as cp
    >>> from cuml.linear_model import LinearRegression
    >>> X = cp.array([[1, 1], [1, 2], [2, 2], [2, 3]], dtype=cp.float32)
    >>> y = cp.array([6.0, 8.0, 9.0, 11.0], dtype=cp.float32)
    >>> model = LinearRegression().fit(X, y)

    >>> X_test = cp.array([[3, 5], [2, 5]], dtype=cp.float32)
    >>> model.predict(X_test)  # doctest: +SKIP
    array([16.      , 14.999999], dtype=float32)
    """
    coef_ = ReflectedAttr()
    intercept_ = ReflectedAttr()

    _cpu_class_path = "sklearn.linear_model.LinearRegression"

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "algorithm",
            "fit_intercept",
            "copy_X",
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        if model.positive:
            raise UnsupportedOnGPU("`positive=True` is not supported")

        return {
            "fit_intercept": model.fit_intercept,
            "copy_X": model.copy_X,
        }

    def _params_to_cpu(self):
        return {
            "fit_intercept": self.fit_intercept,
            "copy_X": self.copy_X,
        }

    def _attrs_from_cpu(self, model):
        return {
            "intercept_": (
                model.intercept_
                if cp.isscalar(model.intercept_)
                else cp.asarray(model.intercept_)
            ),
            "coef_": cp.asarray(model.coef_),
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        return {
            "intercept_": (
                self.intercept_
                if cp.isscalar(self.intercept_)
                else cp.asnumpy(self.intercept_)
            ),
            "coef_": cp.asnumpy(self.coef_, order="A"),
            **super()._attrs_to_cpu(model),
        }

    def __init__(
        self,
        *,
        algorithm="auto",
        fit_intercept=True,
        copy_X=True,
        verbose=False,
        output_type=None
    ):
        super().__init__(verbose=verbose, output_type=output_type)
        self.algorithm = algorithm
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X

    def _fit_libcuml(
        self,
        X,
        y,
        sample_weight=None,
        may_mutate_X=False,
        may_mutate_y=False,
        may_mutate_sample_weight=False,
    ):
        """Fit a LinearRegression using a libcuml solver"""
        cdef int algo = (
            Algo.EIG if self.algorithm == "auto" else Algo.parse(self.algorithm)
        )

        # All libcuml solvers require F-ordered X, and mutate the inputs.
        X = cp.array(X, order="F", copy=None if may_mutate_X else True)
        if not may_mutate_y:
            y = y.copy()
        if sample_weight is not None and not may_mutate_sample_weight:
            sample_weight = sample_weight.copy()

        coef = cp.zeros(
            (X.shape[1],) if y.ndim == 1 else (1, X.shape[1]),
            dtype=X.dtype,
        )

        cdef size_t n_rows = X.shape[0]
        cdef size_t n_cols = X.shape[1]
        cdef uintptr_t X_ptr = X.data.ptr
        cdef uintptr_t y_ptr = y.data.ptr
        cdef uintptr_t sample_weight_ptr = (
            0 if sample_weight is None else sample_weight.data.ptr
        )
        cdef uintptr_t coef_ptr = coef.data.ptr
        cdef bool is_float32 = X.dtype == np.float32
        cdef float intercept_f32
        cdef double intercept_f64
        # Always use 2 streams to expose concurrency in the eig computation
        handle = get_handle(n_streams=2)
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
        cdef bool fit_intercept = self.fit_intercept

        with nogil:
            if is_float32:
                olsFit(
                    handle_[0],
                    <float*>X_ptr,
                    n_rows,
                    n_cols,
                    <float*>y_ptr,
                    <float*>coef_ptr,
                    &intercept_f32,
                    fit_intercept,
                    algo,
                    <float*>sample_weight_ptr,
                )
            else:
                olsFit(
                    handle_[0],
                    <double*>X_ptr,
                    n_rows,
                    n_cols,
                    <double*>y_ptr,
                    <double*>coef_ptr,
                    &intercept_f64,
                    fit_intercept,
                    algo,
                    <double*>sample_weight_ptr,
                )
        handle.sync()

        if self.fit_intercept:
            intercept = intercept_f32 if is_float32 else intercept_f64
            if y.ndim == 1:
                intercept = X.dtype.type(intercept)
            else:
                intercept = cp.full((1,), intercept, dtype=X.dtype)
        else:
            intercept = 0.0

        return coef, intercept

    @generate_docstring()
    @mlfunc(set_input_type=True)
    def fit(self, X, y, sample_weight=None) -> "LinearRegression":
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

        if self.algorithm not in (
            "auto", "eig", "svd", "lsmr", "qr", "svd-qr", "svd-jacobi"
        ):
            raise ValueError(f"algorithm={self.algorithm!r} is not supported")

        # Determine the solver to use
        fallback_reason = None
        if self.algorithm == "lsmr":
            solver = "lsmr"
        elif sp.issparse(X):
            solver = "lsmr"
            fallback_reason = "sparse X"
        elif X.shape[1] == 1:
            solver = "svd"
            fallback_reason = "single-column X"
        elif y.ndim == 2 and y.shape[1] > 1:
            solver = "svd"
            fallback_reason = "multi-column y"
        else:
            solver = "libcuml"

        # Warn if falling back to a different solver
        if fallback_reason is not None and self.algorithm not in ("auto", solver):
            warnings.warn(
                (
                    f"Falling back to `algorithm={solver!r}` as `algorithm="
                    f"{self.algorithm!r}` doesn't support {fallback_reason}."
                ),
                UserWarning,
            )

        if solver == "libcuml":
            coef, intercept = self._fit_libcuml(
                X,
                y,
                sample_weight,
                may_mutate_X=X_is_copy or not self.copy_X,
                may_mutate_y=y_is_copy,
                may_mutate_sample_weight=sample_weight_is_copy,
            )
        else:
            coef, intercept, _ = fit_least_squares(
                X,
                y,
                sample_weight=sample_weight,
                fit_intercept=self.fit_intercept,
                alpha=0.0,
                may_mutate_X=X_is_copy or not self.copy_X,
                may_mutate_y=y_is_copy,
                solver=solver,
            )

        self.coef_ = coef
        self.intercept_ = intercept

        return self

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = True
        return tags
