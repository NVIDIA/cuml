# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import cupy as cp
import cupyx.scipy.sparse
import cupyx.scipy.sparse.linalg
import numpy as np

from cuml.common.doc_utils import generate_docstring
from cuml.internals.outputs import mlfunc
from cuml.internals.validation import check_inputs, check_is_fitted


class LinearPredictMixin:
    @generate_docstring(
        return_values={
            "name": "preds",
            "type": "dense",
            "description": "Predicted values",
            "shape": "(n_samples, 1)",
        }
    )
    @mlfunc(preserve_index=True)
    def predict(self, X):
        """
        Predicts `y` values for `X`.
        """
        check_is_fitted(self)

        X = check_inputs(
            self,
            X,
            dtype=self.coef_.dtype,
            order=None,
            accept_sparse=True,
        )

        out = X @ self.coef_.T
        out += self.intercept_
        return out


class LinearClassifierMixin:
    @generate_docstring(
        X="dense_sparse",
        return_values={
            "name": "scores",
            "type": "dense",
            "description": "Confidence scores",
            "shape": "(n_samples,) or (n_samples, n_classes)",
        },
    )
    @mlfunc(preserve_index=True)
    def decision_function(self, X):
        """Predict confidence scores for samples."""
        check_is_fitted(self)

        X = check_inputs(
            self,
            X,
            dtype=self.coef_.dtype,
            order=None,
            accept_sparse=True,
        )

        out = X @ self.coef_.T
        out += self.intercept_

        if out.ndim > 1 and out.shape[1] == 1:
            out = out.reshape(-1)

        return out


def center_and_scale(
    X,
    y,
    sample_weight=None,
    fit_intercept=True,
    may_mutate_X=False,
    may_mutate_y=False,
):
    """Common preprocessing for X and y for fitting a linear model.

    Performs centering and scaling of X and y.

    Parameters
    ----------
    X : dense or sparse array, shape (n_samples, n_features)
        The features.
    y : dense array, shape (n_samples,) or (n_samples, n_targets)
        The targets.
    sample_weight : cp.ndarray or None
        The sample weights.
    fit_intercept : bool
        Whether to fit an intercept.
    may_mutate_X : bool
        Whether to allow mutating X inplace to save memory when possible.
    may_mutate_y : bool
        Whether to allow mutating y inplace to save memory when possible.

    Returns
    -------
    X : cupy.ndarray or cupyx.scipy.sparse.sp_matrix, shape (n_samples, n_features)
        Rescaled by sample weights. Also centered if dense.
    y : cupy.ndarray, shape (n_samples, 1) or (n_samples, n_targets)
        Rescaled by sample weights and centered.
    X_offset : cupy.ndarray or None, shape (n_features,)
        The per-column mean of X, or None if ``fit_intercept=False``.
    y_offset : cupy.ndarray or None, shape (n_targets,)
        The per-column mean of y, or None if ``fit_intercept=False``.
    sample_weight_sqrt : cupy.ndarray or None, shape (n_samples,)
        The sqrt of the ``sample_weight``, or None if unweighted.
    """
    X_is_sparse = cupyx.scipy.sparse.issparse(X)

    # Ensure 2D
    if X.ndim == 1:
        X = X[:, None]
    if y.ndim == 1:
        y = y[:, None]

    if fit_intercept:
        if sample_weight is not None:
            # Offset by weighted mean
            den = sample_weight.sum()
            if X_is_sparse:
                X_offset = (
                    X.multiply(sample_weight[:, None]).sum(axis=0).ravel()
                    / den
                )
            else:
                X_offset = (X * sample_weight[:, None]).sum(axis=0) / den
            y_offset = (y * sample_weight[:, None]).sum(axis=0) / den
        else:
            # Offset by mean
            X_offset = X.mean(axis=0).ravel()
            y_offset = y.mean(axis=0)

        # Subtract offset, reusing existing buffers when possible
        if not X_is_sparse:
            # Don't offset sparse X since that would remove sparsity.
            # Instead that's handled later in the solvers.
            X = cp.subtract(
                X,
                X_offset,
                out=X if may_mutate_X else None,
            )
            may_mutate_X = True
        y = cp.subtract(y, y_offset, out=y if may_mutate_y else None)
        may_mutate_y = True
    else:
        X_offset = y_offset = None

    if sample_weight is not None:
        sqrt_weight = cp.sqrt(sample_weight)
        # Multiply by sqrt(weight), reusing existing buffers when possible
        if X_is_sparse:
            X = X.multiply(sqrt_weight[:, None])
        else:
            X = cp.multiply(
                X,
                sqrt_weight[:, None],
                out=X if may_mutate_X else None,
            )
        y = cp.multiply(
            y, sqrt_weight[:, None], out=y if may_mutate_y else None
        )
    else:
        sqrt_weight = None

    return X, y, X_offset, y_offset, sqrt_weight


def cuda_ptr(X):
    """Returns a pointer to a backing device array, or None if not a device array"""
    if (interface := getattr(X, "__cuda_array_interface__", None)) is not None:
        return interface["data"][0]
    return None


_ridge_transform = cp.ElementwiseKernel(
    "T x, T s, T alpha",
    "T out",
    "out = s < 1e-10 ? 0 : x * s / (s * s + alpha)",
    "_ridge_transform",
)
_ridge_transform_zero_alpha = cp.ElementwiseKernel(
    "T x, T s",
    "T out",
    "out = s < 1e-10 ? 0 : x / s",
    "_ridge_transform_zero_alpha",
)


def fit_least_squares(
    X,
    y,
    sample_weight=None,
    *,
    fit_intercept=True,
    alpha=0.0,
    solver="svd",
    tol=1e-6,
    max_iter=None,
    may_mutate_X=False,
    may_mutate_y=False,
):
    """Fit a (possibly regularized) least-squares problem.

    Parameters
    ----------
    X : cp.ndarray or cupyx.scipy.sparse.sp_matrix, shape (n_samples, n_features)
        The features.
    y : cp.ndarray array, shape (n_samples,) or (n_samples, n_targets)
        The targets.
    sample_weight : cp.ndarray or None
        The sample weights.
    fit_intercept : bool
        Whether to fit an intercept.
    alpha : float or cp.ndarray
        Ridge regularization strength. Must be a non-negative float. Defaults
        to 0 for no regularization (a LinearRegression).
    solver : {'svd', 'lsmr'}
        The solver to use.
    tol : float
        Tolerance, used by the LSMR solver.
    max_iter : int or None
        Maximum number of iterations, used by the LSMR solver.
    may_mutate_X : bool
        Whether to allow mutating X inplace to save memory when possible.
    may_mutate_y : bool
        Whether to allow mutating y inplace to save memory when possible.

    Returns
    -------
    coef : cp.ndarray, shape (n_features,) or (n_targets, n_features)
        The fitted coefficients. Returns a 1D array if y is 1D, 2D otherwise.
    intercept : float or cp.ndarray, shape (n_targets,)
        The intercept. A scalar if y is 1D, otherwise an array.
    n_iter : np.ndarray or None
        The number of solver iterations ran per-target if using the LSMR
        solver, None otherwise.
    """
    y_1d = y.ndim == 1

    X, y, X_offset, y_offset, sqrt_weight = center_and_scale(
        X,
        y,
        sample_weight=sample_weight,
        fit_intercept=fit_intercept,
        may_mutate_X=may_mutate_X,
        may_mutate_y=may_mutate_y,
    )

    # Normalize alpha to a cupy array of shape (n_targets,)
    if cp.isscalar(alpha):
        alpha = cp.full(y.shape[1], alpha, dtype=X.dtype)

    if solver == "svd":
        # Solve using SVD method
        u, s, vh = cp.linalg.svd(X, full_matrices=False)
        if (alpha == 0).all():
            # Small optimization in the case of all-zero alpha
            temp = _ridge_transform_zero_alpha(u.T.dot(y), s[:, None])
        else:
            temp = _ridge_transform(u.T.dot(y), s[:, None], alpha)
        coef = vh.T.dot(temp).T
        n_iter = None
    elif solver == "lsmr":
        if cupyx.scipy.sparse.issparse(X) and fit_intercept:
            # To keep sparsity, sparse inputs aren't already centered when
            # fitting an intercept. We handle removing the offset within the
            # fit via a LinearOperator.
            if sqrt_weight is None:
                A = cupyx.scipy.sparse.linalg.LinearOperator(
                    shape=X.shape,
                    matvec=lambda w: X.dot(w) - w.dot(X_offset),
                    rmatvec=lambda w: X.T.dot(w) - X_offset * w.sum(),
                )
            else:
                A = cupyx.scipy.sparse.linalg.LinearOperator(
                    shape=X.shape,
                    matvec=lambda w: X.dot(w) - sqrt_weight * w.dot(X_offset),
                    rmatvec=lambda w: X.T.dot(w)
                    - X_offset * w.dot(sqrt_weight),
                )
        else:
            A = X

        coef = cp.empty((y.shape[1], X.shape[1]), dtype=X.dtype)
        n_iter = np.empty(y.shape[1], dtype=np.int32)
        damp = cp.sqrt(alpha)

        for i in range(y.shape[1]):
            b = y[:, i]
            info = cupyx.scipy.sparse.linalg.lsmr(
                A,
                b,
                damp=damp[i],
                atol=tol,
                btol=tol,
                maxiter=max_iter,
            )
            coef[i] = info[0]
            n_iter[i] = info[2]
    else:
        raise ValueError(f"Unsupported solver={solver!r}")

    if fit_intercept:
        intercept = y_offset - cp.dot(X_offset, coef.T)
        if y_1d:
            intercept = coef.dtype.type(intercept.item())
    else:
        intercept = 0.0
    if y_1d:
        coef = coef.ravel()

    return coef, intercept, n_iter
