# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import numpy as np

from cuml.common.doc_utils import generate_docstring
from cuml.internals.base import Base, get_handle
from cuml.internals.mixins import FMajorInputTagMixin
from cuml.internals.outputs import ReflectedAttr, mlfunc
from cuml.internals.validation import check_inputs, check_is_fitted

from libc.stdint cimport uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t

__all__ = ("fit_cd", "CD")


cdef extern from "cuml/solvers/solver.hpp" namespace "ML::Solver" nogil:
    cdef int cdFit(
        handle_t& handle,
        float *input,
        int n_rows,
        int n_cols,
        float *labels,
        float *coef,
        float *intercept,
        bool fit_intercept,
        int epochs,
        int loss,
        float alpha,
        float l1_ratio,
        bool shuffle,
        float tol,
        float *sample_weight
    ) except +

    cdef int cdFit(
        handle_t& handle,
        double *input,
        int n_rows,
        int n_cols,
        double *labels,
        double *coef,
        double *intercept,
        bool fit_intercept,
        int epochs,
        int loss,
        double alpha,
        double l1_ratio,
        bool shuffle,
        double tol,
        double *sample_weight
    ) except +

    cdef void cdPredict(handle_t& handle,
                        const float *input,
                        int n_rows,
                        int n_cols,
                        const float *coef,
                        float intercept,
                        float *preds,
                        int loss) except +

    cdef void cdPredict(handle_t& handle,
                        const double *input,
                        int n_rows,
                        int n_cols,
                        const double *coef,
                        double intercept,
                        double *preds,
                        int loss) except +


def fit_cd(
    estimator,
    X,
    y,
    sample_weight=None,
    *,
    loss="squared_loss",
    double alpha=0.0001,
    double l1_ratio=0.15,
    bool fit_intercept=True,
    int max_iter=1000,
    double tol=1e-3,
    bool shuffle=True,
):
    """Fit a linear model using coordinate descent.

    Parameters
    ----------
    estimator : Base
        The estimator being fit.
    X : array-like, shape=(n_samples, n_features)
        The training data.
    y : array-like, shape=(n_samples,)
        The target values.
    sample_weight : None or array-like, shape=(n_samples,)
        The sample weights.
    **kwargs
        Remaining keyword arguments match the hyperparameters
        to ``CD``, see the ``CD`` docs for more information.

    Returns
    -------
    coef : cupy.ndarray, shape=(n_features,)
        The fit coefficients
    intercept : float
        The fit intercept, or 0 if `fit_intercept=False`
    n_iter : int
        The number of iterations the solver ran for.
    """
    handle = get_handle()

    # Process and validate parameters
    if loss != "squared_loss":
        raise ValueError(f"{loss=!r} is not supported")

    if alpha < 0.0:
        raise ValueError(f"Expected alpha >= 0, got {alpha}")

    # Process and validate input arrays
    X, y, sample_weight = check_inputs(
        estimator,
        X,
        y,
        sample_weight,
        dtype=("float32", "float64"),
        order="F",
        ensure_min_samples=2,
        reset=True,
    )

    # Allocate outputs
    coef = cp.zeros(X.shape[1], dtype=X.dtype)

    cdef int n_rows = X.shape[0]
    cdef int n_cols = X.shape[1]
    cdef uintptr_t X_ptr = X.data.ptr
    cdef uintptr_t y_ptr = y.data.ptr
    cdef uintptr_t sample_weight_ptr = (
        0 if sample_weight is None else sample_weight.data.ptr
    )
    cdef uintptr_t coef_ptr = coef.data.ptr

    cdef float intercept_f32
    cdef double intercept_f64
    cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
    cdef bool is_float32 = X.dtype == np.float32
    cdef int n_iter

    # Perform fit
    with nogil:
        if is_float32:
            n_iter = cdFit(
                handle_[0],
                <float*>X_ptr,
                n_rows,
                n_cols,
                <float*>y_ptr,
                <float*>coef_ptr,
                &intercept_f32,
                fit_intercept,
                max_iter,
                0,
                <float>alpha,
                <float>l1_ratio,
                shuffle,
                <float>tol,
                <float*>sample_weight_ptr
            )
        else:
            n_iter = cdFit(
                handle_[0],
                <double*>X_ptr,
                n_rows,
                n_cols,
                <double*>y_ptr,
                <double*>coef_ptr,
                &intercept_f64,
                fit_intercept,
                max_iter,
                0,
                alpha,
                l1_ratio,
                shuffle,
                tol,
                <double*>sample_weight_ptr
            )
    handle.sync()

    intercept = intercept_f32 if is_float32 else intercept_f64

    return coef, intercept, n_iter


class CD(FMajorInputTagMixin, Base):
    """
    Coordinate Descent (CD) is a very common optimization algorithm that
    minimizes along coordinate directions to find the minimum of a function.

    cuML's CD algorithm accepts a numpy matrix or a cuDF DataFrame as the
    input dataset.algorithm The CD algorithm currently works with linear
    regression and ridge, lasso, and elastic-net penalties.

    Parameters
    ----------
    loss : 'squared_loss'
        Only 'squared_loss' is supported right now.
        'squared_loss' uses linear regression in its predict step.
    alpha: float (default = 0.0001)
        The constant value which decides the degree of regularization.
        'alpha = 0' is equivalent to an ordinary least square, solved by the
        LinearRegression object.
    l1_ratio: float (default = 0.15)
        The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1. For
        l1_ratio = 0 the penalty is an L2 penalty.
        For l1_ratio = 1 it is an L1 penalty. For 0 < l1_ratio < 1,
        the penalty is a combination of L1 and L2.
    fit_intercept : boolean (default = True)
       If True, the model tries to correct for the global mean of y.
       If False, the model expects that you have centered the data.
    max_iter : int (default = 1000)
        The number of times the model should iterate through the entire
        dataset during training
    tol : float (default = 1e-3)
       The tolerance for the optimization: if the updates are smaller than tol,
       solver stops.
    shuffle : boolean (default = True)
       If set to 'True', a random coefficient is updated every iteration rather
       than looping over features sequentially by default.
       This (setting to 'True') often leads to significantly faster convergence
       especially when tol is higher than 1e-4.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {None, 'input', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Examples
    --------
    >>> import cupy as cp
    >>> import cudf
    >>> from cuml.solvers import CD

    >>> cd = CD(alpha=0.0)

    >>> X = cudf.DataFrame()
    >>> X['col1'] = cp.array([1,1,2,2], dtype=cp.float32)
    >>> X['col2'] = cp.array([1,2,2,3], dtype=cp.float32)

    >>> y = cudf.Series(cp.array([6.0, 8.0, 9.0, 11.0], dtype=cp.float32))

    >>> cd.fit(X,y)
    CD(alpha=0.0)
    >>> print(cd.coef_) # doctest: +SKIP
    0 1.001...
    1 1.998...
    dtype: float32
    >>> print(cd.intercept_) # doctest: +SKIP
    3.00...
    >>> X_new = cudf.DataFrame()
    >>> X_new['col1'] = cp.array([3,2], dtype=cp.float32)
    >>> X_new['col2'] = cp.array([5,5], dtype=cp.float32)

    >>> preds = cd.predict(X_new)
    >>> print(preds) # doctest: +SKIP
    0 15.997...
    1 14.995...
    dtype: float32
    """
    coef_ = ReflectedAttr()

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "loss",
            "alpha",
            "l1_ratio",
            "fit_intercept",
            "max_iter",
            "tol",
            "shuffle",
        ]

    def __init__(self, *, loss='squared_loss', alpha=0.0001, l1_ratio=0.15,
                 fit_intercept=True, max_iter=1000, tol=1e-3, shuffle=True,
                 output_type=None, verbose=False):

        super().__init__(verbose=verbose, output_type=output_type)

        self.loss = loss
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle

    @generate_docstring()
    @mlfunc(set_input_type=True)
    def fit(self, X, y, sample_weight=None) -> "CD":
        """
        Fit the model with X and y.
        """
        coef, intercept, n_iter = fit_cd(
            self,
            X,
            y,
            sample_weight=sample_weight,
            loss=self.loss,
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            shuffle=self.shuffle,
        )
        self.coef_ = coef
        self.intercept_ = intercept
        self.n_iter_ = n_iter

        return self

    @generate_docstring(return_values={'name': 'preds',
                                       'type': 'dense',
                                       'description': 'Predicted values',
                                       'shape': '(n_samples, 1)'})
    @mlfunc(preserve_index=True)
    def predict(self, X):
        """
        Predicts the y for X.
        """
        check_is_fitted(self)

        X = check_inputs(
            self,
            X,
            dtype=self.coef_.dtype,
        )
        preds = cp.zeros(X.shape[0], dtype=self.coef_.dtype, order="F")

        cdef int n_rows = X.shape[0]
        cdef int n_cols = X.shape[1]
        cdef uintptr_t X_ptr = X.data.ptr
        cdef uintptr_t preds_ptr = preds.data.ptr
        cdef uintptr_t coef_ptr = self.coef_.data.ptr
        cdef double intercept = self.intercept_
        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
        cdef bool is_float32 = self.coef_.dtype == np.float32

        with nogil:
            if is_float32:
                cdPredict(
                    handle_[0],
                    <float*>X_ptr,
                    n_rows,
                    n_cols,
                    <float*>coef_ptr,
                    <float>intercept,
                    <float*>preds_ptr,
                    0,
                )
            else:
                cdPredict(
                    handle_[0],
                    <double*>X_ptr,
                    n_rows,
                    n_cols,
                    <double*>coef_ptr,
                    intercept,
                    <double*>preds_ptr,
                    0,
                )
        handle.sync()

        return preds
