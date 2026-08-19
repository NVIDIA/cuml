# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import cupyx.scipy.sparse as sp
import numpy as np

from cuml.common.classification import process_class_weight
from cuml.common.doc_utils import generate_docstring
from cuml.internals.base import Base, get_handle
from cuml.internals.outputs import ReflectedAttr, mlfunc
from cuml.internals.validation import (
    check_array,
    check_inputs,
    check_is_fitted,
)
from cuml.metrics import accuracy_score

from libc.stdint cimport uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t


cdef extern from "cuml/linear_model/glm.hpp" namespace "ML::GLM" nogil:
    void qnFit[T, I](
        const handle_t& cuml_handle,
        const qn_params& pams,
        T *X,
        bool X_col_major,
        T *y,
        I N,
        I D,
        I C,
        T *w0,
        T *f,
        int *num_iters,
        T *sample_weight) except +

    void qnFitSparse[T, I](
        const handle_t& cuml_handle,
        const qn_params& pams,
        T *X_values,
        I *X_cols,
        I *X_row_ids,
        I X_nnz,
        T *y,
        I N,
        I D,
        I C,
        T *w0,
        T *f,
        int *num_iters,
        T *sample_weight) except +


SUPPORTED_LOSSES = {
    "sigmoid": Loss.LOGISTIC,
    "logistic": Loss.LOGISTIC,
    "softmax": Loss.SOFTMAX,
    "normal": Loss.SQUARED,
    "l2": Loss.SQUARED,
    "l1": Loss.ABS,
    "svc_l1": Loss.SVC_L1,
    "svc_l2": Loss.SVC_L2,
    "svr_l1": Loss.SVR_L1,
    "svr_l2": Loss.SVR_L2,
}


cdef void init_qn_params(
    qn_params &params,
    int n_classes,
    loss,
    bool fit_intercept,
    double l1_strength,
    double l2_strength,
    int max_iter,
    double tol,
    delta,
    int linesearch_max_iter,
    int lbfgs_memory,
    bool penalty_normalized,
    level_enum verbose,
):
    """Initialize a `qn_params` from the corresponding python parameters."""
    if loss == "logistic":
        # Automatically promote logistic -> softmax for multiclass problems
        loss = "softmax" if n_classes > 2 else "sigmoid"

    # Validate hyperparameters
    if (loss_type := SUPPORTED_LOSSES.get(loss)) is None:
        raise ValueError(f"{loss=!r} is unsupported")

    if loss_type == Loss.SOFTMAX:
        if n_classes <= 2:
            raise ValueError(
                f"loss='softmax' requires n_classes > 2 (got {n_classes})"
            )
    elif loss_type in {Loss.LOGISTIC, Loss.SVC_L1, Loss.SVC_L2}:
        if n_classes != 2:
            raise ValueError(
                f"loss={loss!r} requires n_classes == 2 (got {n_classes})"
            )
    elif n_classes != 0:
        raise ValueError(
            f"loss={loss!r} does not support classification (got {n_classes=})"
        )

    params.loss = loss_type
    params.penalty_l1 = l1_strength
    params.penalty_l2 = l2_strength
    params.grad_tol = tol
    params.change_tol = delta if delta is not None else tol * 0.01
    params.max_iter = max_iter
    params.linesearch_max_iter = linesearch_max_iter
    params.lbfgs_memory = lbfgs_memory
    params.verbose = <int>verbose
    params.fit_intercept = fit_intercept
    params.penalty_normalized = penalty_normalized


def fit_qn(
    estimator,
    X,
    y,
    sample_weight=None,
    *,
    loss="l2",
    class_weight=None,
    bool fit_intercept=True,
    double l1_strength=0.0,
    double l2_strength=0.0,
    int max_iter=1000,
    double tol=1e-4,
    delta=None,
    int linesearch_max_iter=50,
    int lbfgs_memory=5,
    bool penalty_normalized=True,
    init_coef=None,
    level_enum verbose=level_enum.warn,
    return_classes=False,
):
    """Fit a linear model using a Quasi-newton method.

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
    class_weight : dict or 'balanced', default=None
        Weights associated per-classes, or None for uniform weights. If 'balanced',
        weights inversely proportional to the class frequencies will be used.
    return_classes : bool, default=False
        Whether to preprocess `y` and return the classes. Defaults to False.
    **kwargs
        Remaining keyword arguments match the hyperparameters
        to ``QN``, see the ``QN`` docs for more information.

    Returns
    -------
    coef : cupy.ndarray, shape=(1, n_features) or (n_classes, n_features)
        The fit coefficients.
    intercept : cupy.ndarray, shape=(1,) or (n_classes,)
        Intercept added to the decision function.
    n_iter : int
        The number of iterations taken by the solver.
    objective : float
        The value of the objective function.
    classes : numpy.ndarray, shape=(n_classes,)
        The classes. Only returned if ``return_classes=True``.
    """
    handle = get_handle()

    out = check_inputs(
        estimator,
        X,
        y,
        sample_weight,
        dtype=("float32", "float64"),
        accept_sparse="csr",
        ensure_min_samples=2,
        y_dtype=(None if return_classes else ...),
        return_classes=return_classes,
        reset=True,
    )
    if return_classes:
        X, y, sample_weight, classes = out
        n_classes = len(classes)
    else:
        X, y, sample_weight = out
        n_classes = 0

    if class_weight is not None:
        if not return_classes:
            raise ValueError("class_weights is only supported for classifiers")
        _, sample_weight = process_class_weight(
            classes,
            y,
            class_weight=class_weight,
            sample_weight=sample_weight,
            dtype=X.dtype
        )
    if return_classes:
        # Coerce `y` to X dtype for classifier only _after_ processing class weights
        y = y.astype(X.dtype, copy=False)

    # Validate and process hyperparameters
    cdef qn_params params
    init_qn_params(
        params,
        n_classes=n_classes,
        loss=loss,
        fit_intercept=fit_intercept,
        l1_strength=l1_strength,
        l2_strength=l2_strength,
        max_iter=max_iter,
        tol=tol,
        delta=delta,
        linesearch_max_iter=linesearch_max_iter,
        lbfgs_memory=lbfgs_memory,
        penalty_normalized=penalty_normalized,
        verbose=verbose,
    )

    coef_shape = (
        ((X.shape[1] + 1) if fit_intercept else X.shape[1]),
        (n_classes if n_classes > 2 else 1),
    )

    if init_coef is None:
        coef = cp.zeros(coef_shape, dtype=X.dtype, order="C")
    else:
        coef = check_array(init_coef, dtype=X.dtype, order="C")
        if coef.shape != coef_shape:
            raise ValueError(f"Expected coef.shape == ({coef_shape}), got {coef.shape}")

    cdef bool sparse_X = sp.issparse(X)
    cdef int n_rows = X.shape[0]
    cdef int n_cols = X.shape[1]
    cdef uintptr_t X_ptr, X_indices_ptr, X_indptr_ptr
    cdef int X_nnz
    cdef bool X_is_col_major
    if sparse_X:
        X_ptr = X.data.data.ptr
        X_indices_ptr = X.indices.data.ptr
        X_indptr_ptr = X.indptr.data.ptr
        X_nnz = X.nnz
    else:
        X_ptr = X.data.ptr
        X_is_col_major = X.flags["F_CONTIGUOUS"]

    cdef uintptr_t y_ptr = y.data.ptr
    cdef uintptr_t coef_ptr = coef.data.ptr
    cdef uintptr_t sample_weight_ptr = (
        0 if sample_weight is None else sample_weight.data.ptr
    )
    cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
    cdef float objective_f32
    cdef double objective_f64
    cdef int n_iter
    cdef bool use_float32 = X.dtype == np.float32

    with nogil:
        if sparse_X:
            if use_float32:
                qnFitSparse[float, int](
                    handle_[0],
                    params,
                    <float*> X_ptr,
                    <int*> X_indices_ptr,
                    <int*> X_indptr_ptr,
                    X_nnz,
                    <float*> y_ptr,
                    n_rows,
                    n_cols,
                    n_classes or 1,
                    <float*> coef_ptr,
                    &objective_f32,
                    &n_iter,
                    <float*> sample_weight_ptr
                )
            else:
                qnFitSparse[double, int](
                    handle_[0],
                    params,
                    <double*> X_ptr,
                    <int*> X_indices_ptr,
                    <int*> X_indptr_ptr,
                    X_nnz,
                    <double*> y_ptr,
                    n_rows,
                    n_cols,
                    n_classes or 1,
                    <double*> coef_ptr,
                    &objective_f64,
                    &n_iter,
                    <double*> sample_weight_ptr
                )
        else:
            if use_float32:
                qnFit[float, int](
                    handle_[0],
                    params,
                    <float*> X_ptr,
                    X_is_col_major,
                    <float*> y_ptr,
                    n_rows,
                    n_cols,
                    n_classes or 1,
                    <float*> coef_ptr,
                    &objective_f32,
                    &n_iter,
                    <float*> sample_weight_ptr
                )
            else:
                qnFit[double, int](
                    handle_[0],
                    params,
                    <double*> X_ptr,
                    X_is_col_major,
                    <double*> y_ptr,
                    n_rows,
                    n_cols,
                    n_classes or 1,
                    <double*> coef_ptr,
                    &objective_f64,
                    &n_iter,
                    <double*> sample_weight_ptr
                )

    handle.sync()

    objective = objective_f32 if use_float32 else objective_f64

    if fit_intercept:
        intercept = coef[-1]
        coef = coef[:-1].T
    else:
        if n_classes <= 2:
            intercept = cp.zeros(shape=1, dtype=X.dtype)
        else:
            intercept = cp.zeros(shape=n_classes, dtype=X.dtype)
        coef = coef.T

    if return_classes:
        return coef, intercept, n_iter, objective, classes
    else:
        return coef, intercept, n_iter, objective


class QN(Base):
    """
    Quasi-Newton methods are used to either find zeroes or local maxima
    and minima of functions, and used by this class to optimize a cost
    function.

    Two algorithms are implemented underneath cuML's QN class, and which one
    is executed depends on the following rule:

      * Orthant-Wise Limited Memory Quasi-Newton (OWL-QN) if there is l1
        regularization

      * Limited Memory BFGS (L-BFGS) otherwise.

    Parameters
    ----------
    loss: 'sigmoid', 'softmax', 'l1', 'l2', 'svc_l1', 'svc_l2', 'svr_l1', \
        'svr_l2' (default = 'sigmoid').
        'sigmoid' loss used for single class logistic regression;
        'softmax' loss used for multiclass logistic regression;
        'l1'/'l2' loss used for regression.
    fit_intercept: boolean (default = True)
        If True, the model tries to correct for the global mean of y.
        If False, the model expects that you have centered the data.
    l1_strength: float (default = 0.0)
        l1 regularization strength (if non-zero, will run OWL-QN, else L-BFGS).
        Use `penalty_normalized` to control whether the solver divides this
        by the sample size.
    l2_strength: float (default = 0.0)
        l2 regularization strength.
        Use `penalty_normalized` to control whether the solver divides this
        by the sample size.
    max_iter: int (default = 1000)
        Maximum number of iterations taken for the solvers to converge.
    tol: float (default = 1e-4)
        The training process will stop if

        `norm(current_loss_grad) <= tol * max(current_loss, tol)`.

        This differs slightly from the `gtol`-controlled stopping condition in
        `scipy.optimize.minimize(method='L-BFGS-B')
        <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html>`_:

        `norm(current_loss_projected_grad) <= gtol`.

        Note, `sklearn.LogisticRegression()
        <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_
        uses the sum of softmax/logistic loss over the input data, whereas cuML
        uses the average. As a result, Scikit-learn's loss is usually
        `sample_size` times larger than cuML's.
        To account for the differences you may divide the `tol` by the sample
        size; this would ensure that the cuML solver does not stop earlier than
        the Scikit-learn solver.
    delta: Optional[float] (default = None)
        The training process will stop if

        `abs(current_loss - previous_loss) <= delta * max(current_loss, tol)`.

        When `None`, it's set to `tol * 0.01`; when `0`, the check is disabled.
        Given the current step `k`, parameter `previous_loss` here is the loss
        at the step `k - p`, where `p` is a small positive integer set
        internally.

        Note, this parameter corresponds to `ftol` in
        `scipy.optimize.minimize(method='L-BFGS-B')
        <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html>`_,
        which is set by default to a minuscule `2.2e-9` and is not exposed in
        `sklearn.LogisticRegression()
        <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_.
        This condition is meant to protect the solver against doing vanishingly
        small linesearch steps or zigzagging.
        You may choose to set `delta = 0` to make sure the cuML solver does
        not stop earlier than the Scikit-learn solver.
    linesearch_max_iter: int (default = 50)
        Max number of linesearch iterations per outer iteration of the
        algorithm.
    lbfgs_memory: int (default = 5)
        Rank of the lbfgs inverse-Hessian approximation. Method will use
        O(lbfgs_memory * D) memory.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {None, 'input', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.
    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
    penalty_normalized : bool, default=True
        When set to True, l1 and l2 parameters are divided by the sample size.
        This flag can be used to achieve a behavior compatible with other
        implementations, such as sklearn's.

    Attributes
    ----------
    coef_ : array, shape (n_classes, n_features)
        The estimated coefficients for the linear regression model.
    intercept_ : array (n_classes, 1)
        The independent term. If `fit_intercept` is False, will be 0.

    Notes
    -----
    This class contains implementations of two popular Quasi-Newton methods:

        - Limited-memory Broyden Fletcher Goldfarb Shanno (L-BFGS) [Nocedal,
          Wright - Numerical Optimization (1999)]

        - `Orthant-wise limited-memory quasi-newton (OWL-QN)
          [Andrew, Gao - ICML 2007]
          <https://www.microsoft.com/en-us/research/publication/scalable-training-of-l1-regularized-log-linear-models/>`_

    Examples
    --------
    >>> import cupy as cp
    >>> from cuml.solvers import QN
    >>> X = cp.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    >>> y = cp.array([0, 0, 1, 1])
    >>> solver = QN(loss="sigmoid").fit(X, y)
    >>> solver.predict(X)
    array([0, 0, 1, 1], dtype=int32)
    """

    coef_ = ReflectedAttr()
    intercept_ = ReflectedAttr()

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "loss",
            "fit_intercept",
            "l1_strength",
            "l2_strength",
            "max_iter",
            "tol",
            "linesearch_max_iter",
            "lbfgs_memory",
            "warm_start",
            "delta",
            "penalty_normalized"
        ]

    def __init__(
        self,
        *,
        loss="sigmoid",
        fit_intercept=True,
        l1_strength=0.0,
        l2_strength=0.0,
        max_iter=1000,
        tol=1e-4,
        delta=None,
        linesearch_max_iter=50,
        lbfgs_memory=5,
        verbose=False,
        output_type=None,
        warm_start=False,
        penalty_normalized=True,
    ):
        super().__init__(verbose=verbose, output_type=output_type)

        self.loss = loss
        self.fit_intercept = fit_intercept
        self.l1_strength = l1_strength
        self.l2_strength = l2_strength
        self.max_iter = max_iter
        self.tol = tol
        self.delta = delta
        self.linesearch_max_iter = linesearch_max_iter
        self.lbfgs_memory = lbfgs_memory
        self.warm_start = warm_start
        self.penalty_normalized = penalty_normalized

    @generate_docstring(X="dense_sparse")
    @mlfunc(set_input_type=True)
    def fit(self, X, y, sample_weight=None) -> "QN":
        """
        Fit the model with X and y.
        """
        is_classifier = (
            self.loss in {"logistic", "sigmoid", "softmax", "svc_l1", "svc_l2"}
        )

        if self.warm_start and hasattr(self, "coef_"):
            init_coef = self.coef_.T
            if self.fit_intercept:
                init_coef = cp.concatenate([init_coef, self.intercept_[None, :]])
        else:
            init_coef = None

        out = fit_qn(
            self,
            X,
            y,
            sample_weight=sample_weight,
            loss=self.loss,
            fit_intercept=self.fit_intercept,
            l1_strength=self.l1_strength,
            l2_strength=self.l2_strength,
            max_iter=self.max_iter,
            tol=self.tol,
            delta=self.delta,
            linesearch_max_iter=self.linesearch_max_iter,
            lbfgs_memory=self.lbfgs_memory,
            penalty_normalized=self.penalty_normalized,
            init_coef=init_coef,
            verbose=self._verbose_level,
            return_classes=is_classifier,
        )
        if is_classifier:
            coef, intercept, n_iter, objective, classes = out
            n_classes = len(classes)
        else:
            coef, intercept, n_iter, objective = out
            n_classes = 0

        self.coef_ = coef
        self.intercept_ = intercept
        self.n_classes_ = n_classes
        self.n_iter_ = n_iter
        self.objective = objective

        return self

    @generate_docstring(X="dense_sparse")
    @mlfunc(preserve_index=True)
    def predict(self, X):
        """Predicts the y for X."""
        check_is_fitted(self)

        X = check_inputs(
            self,
            X,
            dtype=self.coef_.dtype,
            accept_sparse=True,
        )

        out = X @ self.coef_.T
        out += self.intercept_

        if out.ndim > 1 and out.shape[1] == 1:
            out = out.reshape(-1)

        if self.n_classes_ >= 2:
            if out.ndim == 1:
                out = (out > 0).astype(np.int32)
            else:
                out = cp.argmax(out, axis=1)

        return out

    @mlfunc(convert_output=False)
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
