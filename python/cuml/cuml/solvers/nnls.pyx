#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import numpy as np

import cuml.internals.nvtx as nvtx
from cuml.internals.base import get_handle
from cuml.internals.outputs import mlfunc

from libc.stdint cimport int64_t, uint8_t, uintptr_t
from pylibraft.common.handle cimport handle_t

__all__ = ("nnls",)

_NVTX_DOMAIN = "cuml_python"
_NVTX_CATEGORY = "solvers.nnls"


cdef extern from "cuml/solvers/nnls.hpp" namespace "ML::Solver" nogil:
    cdef cppclass NnlsBatchedParams:
        int max_iter
        double tol
        NnlsBatchedParams() except +

    cdef void nnlsBatched(
        handle_t& handle,
        const float* A,
        int64_t m,
        int64_t n,
        const float* B,
        int64_t n_problems,
        const uint8_t* masks,
        float* X,
        float* rnorm,
        const NnlsBatchedParams& params,
    ) except +

    cdef void nnlsBatched(
        handle_t& handle,
        const double* A,
        int64_t m,
        int64_t n,
        const double* B,
        int64_t n_problems,
        const uint8_t* masks,
        double* X,
        double* rnorm,
        const NnlsBatchedParams& params,
    ) except +


def _nnls(
    A,
    B,
    masks=None,
    *,
    int max_iter=0,
    double tol=1e-6,
    bint compute_rnorm=True,
):
    """Backend of :func:`nnls`: solve a batch of NNLS problems sharing ``A``.

    This performs no validation or conversion. It must be called from within
    a cuml internal context (e.g. from a function decorated with ``mlfunc``)
    so that cupy operations and the solver run on the same stream.

    Parameters
    ----------
    A : cupy.ndarray, shape (m, n)
        Coefficient matrix. Must be F-contiguous with dtype ``float32`` or
        ``float64``, with ``m >= 1`` and ``n >= 1``.
    B : cupy.ndarray, shape (m, n_problems)
        Right-hand sides, one problem per column. Must be F-contiguous with
        the same dtype as ``A``, with ``n_problems >= 1``.
    masks : cupy.ndarray, shape (n, n_problems), optional
        Active coefficients of each problem: ``masks[j, p]`` is nonzero iff
        ``A[:, j]`` may be used by problem ``p``. Must be F-contiguous with
        dtype ``uint8``. ``None`` makes every column active in every problem.
    max_iter : int, default=0
        Iteration cap per problem; ``0`` selects ``3 * n + 1``.
    tol : float, default=1e-6
        Absolute tolerance on the projected gradient.
    compute_rnorm : bool, default=True
        Whether to compute the residual norms.

    Returns
    -------
    X : cupy.ndarray, shape (n, n_problems)
        The solutions, F-contiguous with the dtype of ``A``.
    rnorm : cupy.ndarray, shape (n_problems,) or None
        The residual norms with the dtype of ``A``, or ``None`` if
        ``compute_rnorm=False``.
    """
    cdef int64_t m = A.shape[0]
    cdef int64_t n = A.shape[1]
    cdef int64_t n_problems = B.shape[1]

    X = cp.zeros((n, n_problems), dtype=A.dtype, order="F")
    rnorm = cp.zeros(n_problems, dtype=A.dtype) if compute_rnorm else None

    cdef NnlsBatchedParams params
    params.max_iter = max_iter
    params.tol = tol

    cdef uintptr_t A_ptr = A.data.ptr
    cdef uintptr_t B_ptr = B.data.ptr
    cdef uintptr_t masks_ptr = 0 if masks is None else masks.data.ptr
    cdef uintptr_t X_ptr = X.data.ptr
    cdef uintptr_t rnorm_ptr = 0 if rnorm is None else rnorm.data.ptr
    cdef bint is_float32 = A.dtype == np.float32

    handle = get_handle()
    cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

    with nogil:
        if is_float32:
            nnlsBatched(
                handle_[0],
                <const float*>A_ptr,
                m,
                n,
                <const float*>B_ptr,
                n_problems,
                <const uint8_t*>masks_ptr,
                <float*>X_ptr,
                <float*>rnorm_ptr,
                params,
            )
        else:
            nnlsBatched(
                handle_[0],
                <const double*>A_ptr,
                m,
                n,
                <const double*>B_ptr,
                n_problems,
                <const uint8_t*>masks_ptr,
                <double*>X_ptr,
                <double*>rnorm_ptr,
                params,
            )
    handle.sync()

    return X, rnorm


@mlfunc(convert_output=False)
def nnls(
    A,
    b,
    masks=None,
    b_index=None,
    *,
    max_iter=None,
    tol=1e-6,
    compute_rnorm=True,
):
    """Solve a non-negative least squares problem.

    Finds ``x >= 0`` minimizing ``|| A @ x - b ||_2``, an analogue of
    :func:`scipy.optimize.nnls` on the GPU. Many problems that share ``A``
    can be solved at once: ``b`` may hold several right-hand sides, and
    ``masks`` restricts each problem to a subset of the columns of ``A``::

        argmin_x  || A[:, masks[:, j]] @ x - b[:, b_index[j]] ||_2
        subject to  x >= 0

    The Gram matrix ``A.T @ A`` and the projections ``A.T @ b`` are computed
    once and shared by all problems, so solving a batch is much cheaper than
    solving the problems one at a time. The solver is an active-set method
    that keeps the state of one problem in GPU shared memory, which limits it
    to problems with ``min(m, n)`` up to a few thousand (depending on the GPU
    and the dtype); larger problems raise an error.

    Parameters
    ----------
    A : array-like, shape (m, n)
        The coefficient matrix. A NumPy array, a CuPy array, or any object
        implementing ``__cuda_array_interface__``. The data is not modified.
    b : array-like, shape (m,) or (m, n_targets)
        The right-hand side(s).
    masks : array-like of bool or uint8, shape (n, n_problems), optional
        Which columns of ``A`` each problem may use. ``masks[i, j]`` is
        nonzero iff column ``i`` of ``A`` is active in problem ``j``. Columns
        that are not active get a coefficient of exactly zero. A 1D mask of
        shape ``(n,)`` describes a single problem. By default all columns are
        active in all problems. An F-contiguous (column-major) ``uint8`` array
        is used without copying.
    b_index : array-like of int, shape (n_problems,), optional
        The column of ``b`` to use in each problem, so that distinct problems
        can share a right-hand side. By default problem ``j`` uses column
        ``j`` of ``b``.
    max_iter : int, optional
        The maximum number of active-set iterations per problem. By default
        ``3 * n + 1``, which is enough for the method to converge.
    tol : float, default=1e-6
        Absolute tolerance on the projected gradient
        ``A.T @ (A @ x - b)``: a problem is solved once the gradient is
        nonnegative, within ``tol``, at the coefficients that are zero. The
        tolerance is not relative to the magnitude of the data, so for data
        of very large or very small scale, rescale the data or adjust
        ``tol``. In single precision, a tolerance below the rounding error of
        the gradient is raised to that error.
    compute_rnorm : bool, default=True
        Whether to compute the residual norms. Skipping them saves one pass
        over ``A`` per problem.

    Returns
    -------
    x : cupy.ndarray, shape (n,) or (n, n_problems)
        The solution(s), all entries non-negative. This is 1D only if ``b`` is
        1D, ``b_index`` is not given and ``masks`` (if given) describes a
        single problem.
    rnorm : float, cupy.ndarray or None
        The residual norm(s) ``|| A @ x - b ||_2``: a float if ``x`` is 1D,
        and an array of shape ``(n_problems,)`` otherwise.
        ``None`` if ``compute_rnorm=False``.

    Notes
    -----
    ``A`` and ``b`` are converted to ``float64`` if either has that dtype, and
    to ``float32`` otherwise.

    Examples
    --------
    >>> import cupy as cp
    >>> from cuml.solvers import nnls
    >>> A = cp.array([[1, 0], [1, 0], [0, 1]], dtype=cp.float32)
    >>> b = cp.array([2, 1, 1], dtype=cp.float32)
    >>> x, rnorm = nnls(A, b)
    """
    with nvtx.annotate(
        message="nnls",
        domain=_NVTX_DOMAIN,
        category=_NVTX_CATEGORY,
    ):
        with nvtx.annotate(
            message="nnls.prepare_inputs",
            domain=_NVTX_DOMAIN,
            category=_NVTX_CATEGORY,
        ):
            A = cp.asarray(A)
            b = cp.asarray(b)

            if A.ndim != 2:
                raise ValueError(f"Expected 2-D array for A, got {A.ndim}-D")
            if b.ndim not in (1, 2):
                raise ValueError(
                    f"Expected 1-D or 2-D array for b, got {b.ndim}-D"
                )
            m, n = A.shape
            if m < 1 or n < 1:
                raise ValueError(
                    f"Found array with shape {A.shape} for A, while at least "
                    "one row and one column are required."
                )
            if b.shape[0] != m:
                raise ValueError(
                    f"Incompatible dimensions: A has {m} rows, "
                    f"b has {b.shape[0]}"
                )

            dtype = (
                np.float64 if np.float64 in (A.dtype, b.dtype) else np.float32
            )
            A = cp.asfortranarray(A.astype(dtype, copy=False))
            B = b.reshape(m, 1) if b.ndim == 1 else b

            n_problems = None
            if masks is not None:
                masks = cp.asarray(masks)
                if masks.ndim == 1:
                    masks = masks.reshape(-1, 1)
                if masks.ndim != 2:
                    raise ValueError(
                        f"Expected 1-D or 2-D masks, got {masks.ndim}-D"
                    )
                if masks.shape[0] != n:
                    raise ValueError(
                        f"Expected masks with {n} rows (one per column of A), "
                        f"got {masks.shape[0]}"
                    )
                n_problems = masks.shape[1]
                if masks.dtype not in (np.bool_, np.uint8):
                    masks = masks != 0
                masks = cp.asfortranarray(masks.astype(np.uint8, copy=False))

            # Gather the right-hand side of each problem on the device
            if b_index is not None:
                b_index = (
                    cp.asarray(b_index).astype(cp.int64, copy=False).ravel()
                )
                if n_problems is not None and b_index.shape[0] != n_problems:
                    raise ValueError(
                        f"Expected b_index of length {n_problems}, "
                        f"got {b_index.shape[0]}"
                    )
                if b_index.size and (
                    int(b_index.max()) >= B.shape[1] or int(b_index.min()) < 0
                ):
                    raise ValueError(
                        f"b_index entries must lie in [0, {B.shape[1]})"
                    )
                B = B[:, b_index]
            elif n_problems is not None and B.shape[1] != n_problems:
                raise ValueError(
                    f"Expected b with {n_problems} columns (one per problem "
                    f"in masks), got {B.shape[1]}"
                )
            B = cp.asfortranarray(B.astype(dtype, copy=False))

        with nvtx.annotate(
            message="nnls.solve",
            domain=_NVTX_DOMAIN,
            category=_NVTX_CATEGORY,
        ):
            X, rnorm = _nnls(
                A,
                B,
                masks,
                max_iter=0 if max_iter is None else int(max_iter),
                tol=float(tol),
                compute_rnorm=compute_rnorm,
            )

        single = masks is None or masks.shape[1] == 1
        if b.ndim == 1 and b_index is None and single:
            X = X[:, 0]
            rnorm = None if rnorm is None else float(rnorm[0])
        return X, rnorm
