#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp

from cuml.common.doc_utils import generate_docstring
from cuml.internals import ReflectedAttr, logger, mlfunc
from cuml.internals.base import Base, get_handle
from cuml.internals.mixins import RegressorMixin
from cuml.internals.validation import (
    check_array,
    check_inputs,
    check_is_fitted,
)
from cuml.linear_model.base import cuda_ptr

from libc.stdint cimport uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t

from cuml.internals cimport logger


cdef extern from "cuml/solvers/lars.hpp" namespace "ML::Solver::Lars" nogil:

    cdef void larsFit[math_t](
        const handle_t& handle, math_t* X, int n_rows, int n_cols,
        const math_t* y, math_t* beta, int* active_idx, math_t* alphas,
        int* n_active, math_t* gram, int max_iter, math_t* coef_path,
        logger.level_enum verbosity, int ld_X, int ld_G, math_t epsilon) except +

    cdef void larsPredict[math_t](
        const handle_t& handle, const math_t* X, int n_rows, int n_cols,
        int ld_X, const math_t* beta, int n_active, int* active_idx,
        math_t intercept, math_t* preds) except +


class Lars(RegressorMixin, Base):
    """
    Least Angle Regression

    Least Angle Regression (LAR or LARS) is a model selection algorithm. It
    builds up the model using the following algorithm:

    1. We start with all the coefficients equal to zero.
    2. At each step we select the predictor that has the largest absolute
       correlation with the residual.
    3. We take the largest step possible in the direction which is equiangular
       with all the predictors selected so far. The largest step is determined
       such that using this step a new predictor will have as much correlation
       with the residual as any of the currently active predictors.
    4. Stop if `max_iter` reached or all the predictors are used, or if the
       correlation between any unused predictor and the residual is lower than
       a tolerance.

    The solver is based on [1]_. The equations referred in the comments
    correspond to the equations in the paper.

    .. note:: This algorithm assumes that the offset is removed from `X` and
        `y`, and each feature is normalized:

        .. math::

            sum_i y_i = 0, sum_i x_{i,j} = 0,sum_i x_{i,j}^2=1 \
            for j=0..n_{col}-1

    Parameters
    ----------
    fit_intercept : boolean (default = True)
        If True, Lars tries to correct for the global mean of y.
        If False, the model expects that you have centered the data.
    copy_X : boolean (default = True)
        The solver permutes the columns of X. Set `copy_X` to True to prevent
        changing the input data.
    fit_path : boolean (default = True)
        Whether to return all the coefficients along the regularization path
        in the `coef_path_` attribute.
    precompute : bool, 'auto', or array-like with shape = (n_features, \
            n_features). (default = 'auto')
        Whether to precompute the gram matrix. The user can provide the gram
        matrix as an argument.
    n_nonzero_coefs : int (default 500)
        The maximum number of coefficients to fit. This gives an upper limit of
        how many features we select for prediction. This number is also an
        upper limit of the number of iterations.
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
    alphas_ : array, shape (n_alphas + 1,)
        The maximum correlation at each step.
    active_ : array, shape (n_alphas,)
        The indices of the active variables at the end of the path.
    beta_ : array, shape (n_alphas,)
        The active regression coefficients (same as `coef_` but zeros omitted).
    coef_path_ : array, shape (n_features, n_alphas + 1)
        The coefficients along the regularization path. Stored only if
        `fit_path` is True. Note that we only store coefficients for indices
        in the active set (i.e. :py:`coef_path_[:,-1] == coef_[active_]`)
    coef_ : array, shape (n_features,)
        The estimated coefficients for the regression model.
    intercept_ : float
        The independent term. If `fit_intercept_` is False, will be 0.
    n_iter_ : int
        The number of iterations taken by the solver.

    Notes
    -----
    For additional information, see `scikitlearn's OLS documentation
    <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html>`__.

    References
    ----------
    .. [1] `B. Efron, T. Hastie, I. Johnstone, R Tibshirani, Least Angle
       Regression The Annals of Statistics (2004) Vol 32, No 2, 407-499
       <http://statweb.stanford.edu/~tibs/ftp/lars.pdf>`_

    """

    alphas_ = ReflectedAttr()
    active_ = ReflectedAttr()
    beta_ = ReflectedAttr()
    coef_path_ = ReflectedAttr()
    coef_ = ReflectedAttr()

    def __init__(
        self,
        *,
        fit_intercept=True,
        copy_X=True,
        fit_path=True,
        n_nonzero_coefs=500,
        eps=None,
        precompute="auto",
        verbose=False,
        output_type=None,
    ):
        super().__init__(verbose=verbose, output_type=output_type)
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.eps = eps
        self.fit_path = fit_path
        self.n_nonzero_coefs = n_nonzero_coefs  # this corresponds to max_iter
        self.precompute = precompute

    @classmethod
    def _get_param_names(cls):
        return [
            "fit_intercept",
            "copy_X",
            "fit_path",
            "n_nonzero_coefs",
            "precompute",
            "eps",
            *super()._get_param_names(),
        ]

    def _calc_gram(self, X):
        """
        Return the gram matrix, or None if it is not applicable.
        """
        n_rows, n_cols = X.shape

        precompute = self.precompute
        if isinstance(precompute, str):
            if precompute == "auto":
                precompute = n_cols < n_rows
            else:
                raise ValueError(f"Invalid value for precompute: {precompute!r}")

        # `precompute` should be True, False, or an array-like now
        if precompute is True:
            try:
                gram = cp.dot(X.T, X)
            except MemoryError:
                logger.debug(
                    "Not enough memory to store the gram matrix. "
                    "Proceeding without it."
                )
                gram = None
        elif precompute is False:
            gram = None
        else:
            gram = check_array(precompute, order="F", dtype=X.dtype)
            if gram.shape != (n_cols, n_cols):
                raise ValueError(
                    f"Expected `precompute` of shape {(n_cols, n_cols)}, "
                    f"got shape {gram.shape}"
                )
        return gram

    @generate_docstring(y="dense_anydtype")
    @mlfunc(set_input_type=True)
    def fit(self, X, y) -> "Lars":
        """
        Fit the model with X and y.

        """
        orig_X_ptr = cuda_ptr(X)
        X, y = check_inputs(
            self,
            X,
            y,
            dtype=("float32", "float64"),
            order="F",
            ensure_min_samples=2,
            reset=True,
        )
        gram = self._calc_gram(X)

        if self.fit_intercept:
            y_mean = y.mean()
            y = y - y_mean
        else:
            y_mean = X.dtype.type(0.0)

        if gram is None and self.copy_X and X.data.ptr == orig_X_ptr:
            # Without gram matrix, the solver will permute columns of X
            X = X.copy()

        cdef int max_iter = self.n_nonzero_coefs

        # Allocate outputs
        beta = cp.zeros(max_iter, dtype=X.dtype)
        active = cp.zeros(max_iter, dtype=cp.int32)
        alphas = cp.zeros(max_iter + 1, dtype=X.dtype)
        if self.fit_path:
            try:
                coef_path = cp.zeros(
                    (X.shape[1], max_iter + 1),
                    dtype=X.dtype,
                    order="F",
                )
            except MemoryError as err:
                raise MemoryError("Not enough memory to store coef_path_. "
                                  "Try to decrease n_nonzero_coefs or set "
                                  "fit_path=False.") from err
        else:
            coef_path = None

        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
        cdef uintptr_t X_ptr = X.data.ptr
        cdef uintptr_t y_ptr = y.data.ptr
        cdef uintptr_t gram_ptr = <uintptr_t>NULL if gram is None else gram.data.ptr
        cdef int n_rows = X.shape[0]
        cdef int n_cols = X.shape[1]
        cdef uintptr_t beta_ptr = beta.data.ptr
        cdef uintptr_t active_idx_ptr = active.data.ptr
        cdef uintptr_t alphas_ptr = alphas.data.ptr
        cdef uintptr_t coef_path_ptr = (
            <uintptr_t>NULL if coef_path is None else coef_path.data.ptr
        )
        cdef int n_active
        cdef bool use_float32 = X.dtype == cp.float32
        cdef double eps = cp.finfo(float).eps if self.eps is None else self.eps
        cdef logger.level_enum verbose_level = <logger.level_enum> self._verbose_level

        with nogil:
            if use_float32:
                larsFit(
                    handle_[0],
                    <float*> X_ptr,
                    n_rows,
                    n_cols,
                    <float*> y_ptr,
                    <float*> beta_ptr,
                    <int*> active_idx_ptr,
                    <float*> alphas_ptr,
                    &n_active,
                    <float*> gram_ptr,
                    max_iter,
                    <float*> coef_path_ptr,
                    verbose_level,
                    n_rows,
                    n_cols,
                    <float> eps,
                )
            else:
                larsFit(
                    handle_[0],
                    <double*> X_ptr,
                    n_rows,
                    n_cols,
                    <double*> y_ptr,
                    <double*> beta_ptr,
                    <int*> active_idx_ptr,
                    <double*> alphas_ptr,
                    &n_active,
                    <double*> gram_ptr,
                    max_iter,
                    <double*> coef_path_ptr,
                    verbose_level,
                    n_rows,
                    n_cols,
                    <double> eps,
                )
        handle.sync()

        active = active[:n_active]
        beta = beta[:n_active]
        alphas = alphas[:n_active + 1]
        if coef_path is not None:
            coef_path = coef_path[:, :n_active + 1]

        coef = cp.zeros(n_cols, dtype=X.dtype)
        coef[active] = beta

        if self.fit_intercept:
            intercept = y_mean
        else:
            intercept = X.dtype.type(0.0)

        self.alphas_ = alphas
        self.active_ = active
        self.beta_ = beta
        self.coef_path_ = coef_path
        self.coef_ = coef
        self.intercept_ = intercept
        self.n_iter_ = n_active

        return self

    @generate_docstring(
        return_values={
            "name": "preds",
            "type": "dense",
            "description": "Predicted values",
            "shape": "(n_samples,)",
        }
    )
    @mlfunc(preserve_index=True)
    def predict(self, X):
        """Predicts `y` values for `X`."""
        check_is_fitted(self)

        X = check_inputs(
            self,
            X,
            dtype=self.coef_.dtype,
            order="F",
        )
        cdef int n_rows = X.shape[0]
        cdef int n_cols = X.shape[1]
        preds = cp.zeros(n_rows, dtype=X.dtype)

        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
        cdef uintptr_t X_ptr = X.data.ptr
        cdef int n_active = self.active_.shape[0]
        cdef double intercept = self.intercept_
        cdef uintptr_t beta_ptr = self.beta_.data.ptr
        cdef uintptr_t active_idx_ptr = self.active_.data.ptr
        cdef bool use_float32 = self.coef_.dtype == cp.float32
        cdef uintptr_t preds_ptr = preds.data.ptr

        with nogil:
            if use_float32:
                larsPredict(
                    handle_[0],
                    <float*> X_ptr,
                    n_rows,
                    n_cols,
                    n_rows,
                    <float*> beta_ptr,
                    n_active,
                    <int*> active_idx_ptr,
                    <float> intercept,
                    <float*> preds_ptr,
                )
            else:
                larsPredict(
                    handle_[0],
                    <double*> X_ptr,
                    n_rows,
                    n_cols,
                    n_rows,
                    <double*> beta_ptr,
                    n_active,
                    <int*> active_idx_ptr,
                    <double> intercept,
                    <double*> preds_ptr,
                )
        handle.sync()

        return preds
