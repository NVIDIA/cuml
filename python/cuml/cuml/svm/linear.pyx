# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp

from cuml.common.classification import process_class_weight
from cuml.internals.base import get_handle
from cuml.internals.validation import check_inputs

from libc.stdint cimport uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t

from cuml.internals.logger cimport level_enum

__all__ = ("fit",)


cdef extern from "cuml/svm/linear.hpp" namespace "ML::SVM::linear" nogil:

    cdef enum Penalty "ML::SVM::linear::Params::Penalty":
        L1 "ML::SVM::linear::Params::L1"
        L2 "ML::SVM::linear::Params::L2"

    cdef enum Loss "ML::SVM::linear::Params::Loss":
        HINGE "ML::SVM::linear::Params::HINGE"
        SQUARED_HINGE "ML::SVM::linear::Params::SQUARED_HINGE"
        EPSILON_INSENSITIVE "ML::SVM::linear::Params::EPSILON_INSENSITIVE"
        SQUARED_EPSILON_INSENSITIVE "ML::SVM::linear::Params::SQUARED_EPSILON_INSENSITIVE"

    cdef struct Params:
        Penalty penalty
        Loss loss
        bool fit_intercept
        bool penalized_intercept
        int max_iter
        int linesearch_max_iter
        int lbfgs_memory
        level_enum verbose
        double C
        double grad_tol
        double change_tol
        double epsilon

    cdef int cpp_fit "ML::SVM::linear::fit"[T](
        const handle_t& handle,
        const Params& params,
        const size_t nRows,
        const size_t nCols,
        const int nClasses,
        const T* classes,
        const T* X,
        const T* y,
        const T* sampleWeight,
        T* w,
    ) except +


def fit(
    estimator,
    X,
    y,
    sample_weight=None,
    *,
    is_classifier=False,
    class_weight=None,
    n_streams=0,
    penalty,
    loss,
    fit_intercept,
    penalized_intercept,
    max_iter,
    linesearch_max_iter,
    lbfgs_memory,
    C,
    tol,
    epsilon,
    level_enum verbose,
):
    """Perform a Linear SVR or SVC fit.

    Parameters
    ----------
    estimator : Base
        The estimator being fit.
    X : array-like, shape = (n_samples, n_features)
        The training vectors
    y : array-like, shape = (n_samples,)
        The target values
    sample_weight : None or array-like, shape = (n_samples,), default=None
        The sample weights
    is_classifier : bool, default=False
        Whether this is a classifier model. Defaults to False.
    class_weight : dict or 'balanced', default=None
        When fitting a classifier, weights associated per-classes, or None for
        uniform weights. If 'balanced', weights inversely proportional to the
        class frequencies will be used.
    n_streams : int, default=0
        The number of streams to use when fitting a classifier.
    **kwargs
        Remaining common hyperparameters for LinearSVC/LinearSVR, see their
        docstrings for details. These are required keyword-only to ensure
        they're properly plumbed through at all callsites.

    Returns
    -------
    coef_ : cp.ndarray, shape = (1, n_features) or (n_classes, n_features)
        The fitted coefficients. Has 1 row for regression and binary
        classification, or n_classes rows for multi-class classification.
    intercept_ : float or cp.ndarray, shape = (1,) or (n_classes,)
        The fitted intercept. If `fit_intercept=False`, returns 0.0 (matching
        sklearn behavior). Otherwise returns a 1-element array for regression
        and binary classification, or `n_classes` elements for multi-class
        classification.
    n_iter_ : int
        The maximum number of iterations run across all classes.
    classes_ : numpy.ndarray, shape=(n_classes,)
        The classes (if ``is_classifier=True``), `None` otherwise.
    """
    penalties = {"l1": Penalty.L1, "l2": Penalty.L2}
    if is_classifier:
        losses = {"hinge": Loss.HINGE, "squared_hinge": Loss.SQUARED_HINGE}
    else:
        losses = {
            "epsilon_insensitive": Loss.EPSILON_INSENSITIVE,
            "squared_epsilon_insensitive": Loss.SQUARED_EPSILON_INSENSITIVE,
        }

    # Process parameters
    cdef Params params

    if penalty not in penalties:
        raise ValueError(
            f"Expected penalty to be one of {list(penalties)}, got {penalty!r}"
        )
    else:
        params.penalty = penalties[penalty]

    if loss not in losses:
        raise ValueError(
            f"Expected loss to be one of {list(losses)}, got {loss!r}"
        )
    else:
        params.loss = losses[loss]

    params.fit_intercept = fit_intercept
    params.penalized_intercept = penalized_intercept
    params.max_iter = max_iter
    params.linesearch_max_iter = linesearch_max_iter
    params.lbfgs_memory = lbfgs_memory
    params.C = C
    params.epsilon = epsilon
    params.grad_tol = tol
    params.change_tol = 0.1 * tol
    params.verbose = verbose

    # Validate and normalize inputs
    out = check_inputs(
        estimator,
        X,
        y,
        sample_weight,
        dtype=("float32", "float64"),
        order="F",
        y_dtype=(None if is_classifier else ...),
        return_classes=is_classifier,
        reset=True,
    )

    # Extract dimensions
    cdef size_t n_rows = out[0].shape[0]
    cdef size_t n_cols = out[0].shape[1]
    cdef int n_classes
    n_coefs = n_cols + int(fit_intercept)

    if is_classifier:
        X, y, sample_weight, classes = out
        n_classes = len(classes)
        if n_classes == 1:
            raise ValueError(
                "This solver needs samples of at least 2 classes in the data, but "
                "the data contains only 1 class"
            )
        _, sample_weight = process_class_weight(
            classes,
            y,
            class_weight=class_weight,
            sample_weight=sample_weight,
            dtype=X.dtype,
        )
        y = y.astype(X.dtype, copy=False)
        class_codes = cp.arange(n_classes, dtype=X.dtype)
        w_shape = (1 if n_classes == 2 else n_classes, n_coefs)
    else:
        X, y, sample_weight = out
        n_classes = 0
        classes = class_codes = None
        w_shape = n_coefs

    # Allocate output arrays
    w = cp.empty(shape=w_shape, dtype=X.dtype, order="F")

    handle = get_handle(n_streams=n_streams)
    cdef handle_t *handle_ = <handle_t*><size_t>handle.getHandle()
    cdef bool is_float32 = X.dtype == cp.float32
    cdef uintptr_t X_ptr = X.data.ptr
    cdef uintptr_t y_ptr = y.data.ptr
    cdef uintptr_t sample_weight_ptr = 0 if sample_weight is None else sample_weight.data.ptr
    cdef uintptr_t classes_ptr = 0 if class_codes is None else class_codes.data.ptr
    cdef uintptr_t w_ptr = w.data.ptr
    cdef int n_iter

    # Perform fit
    with nogil:
        if is_float32:
            n_iter = cpp_fit[float](
                handle_[0],
                params,
                n_rows,
                n_cols,
                n_classes,
                <const float*>classes_ptr,
                <const float*>X_ptr,
                <const float*>y_ptr,
                <const float*>sample_weight_ptr,
                <float*>w_ptr,
            )
        else:
            n_iter = cpp_fit[double](
                handle_[0],
                params,
                n_rows,
                n_cols,
                n_classes,
                <const double*>classes_ptr,
                <const double*>X_ptr,
                <const double*>y_ptr,
                <const double*>sample_weight_ptr,
                <double*>w_ptr,
            )
    handle.sync()

    # Decompose `w` into coef and intercept
    if fit_intercept:
        if w.ndim == 2:
            coef = w[:, :-1]
            intercept = w[:, -1:].flatten()
        else:
            coef = w[:-1]
            intercept = w[-1:]
    else:
        coef = w
        intercept = 0.0

    return coef, intercept, n_iter, classes
