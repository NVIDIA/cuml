#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import math

import cupy as cp
import cupyx.scipy.sparse as cp_sp
import scipy.sparse as sp

from cuml.common.kernel_utils import cuda_kernel_factory

__all__ = (
    "is_sparse",
    "csr_row_normalize_l1",
    "csr_row_normalize_l2",
    "csr_diag_mul",
    "sparse_gram",
    "sparse_cov_and_mean",
)


def is_sparse(X):
    """Returns whether X is sparse matrix.

    Parameters
    ----------
    X : array-like, sparse-matrix

    Returns
    -------
    is_sparse : bool
        Whether the input is sparse.
    """
    return sp.issparse(X) or cp_sp.issparse(X)


def csr_row_normalize_l1(X, inplace=True):
    """Row normalize for csr matrix using the l1 norm"""
    if not inplace:
        X = X.copy()

    kernel = cuda_kernel_factory(
        """
        ({0} *data, {1} *indptr, int n_samples) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;

            if (tid >= n_samples) return;
            {0} sum = 0.0;

            for ({1} i = indptr[tid]; i < indptr[tid+1]; i++) {
                sum += fabs(data[i]);
            }

            if (sum == 0) return;

            for ({1} i = indptr[tid]; i < indptr[tid+1]; i++) {
                data[i] /= sum;
            }
        }
        """,
        (X.dtype, X.indptr.dtype),
        "csr_row_normalize_l1",
    )
    kernel(
        (math.ceil(X.shape[0] / 32),),
        (32,),
        (X.data, X.indptr, X.shape[0]),
    )
    return X


def csr_row_normalize_l2(X, inplace=True):
    """Row normalize for csr matrix using the l2 norm"""
    if not inplace:
        X = X.copy()

    kernel = cuda_kernel_factory(
        """
        ({0} *data, {1} *indptr, int n_samples) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;

            if (tid >= n_samples) return;
            {0} sum = 0.0;

            for ({1} i = indptr[tid]; i < indptr[tid+1]; i++) {
                sum += (data[i] * data[i]);
            }

            if (sum == 0) return;

            sum = sqrt(sum);

            for ({1} i = indptr[tid]; i < indptr[tid+1]; i++) {
                data[i] /= sum;
            }
        }
        """,
        (X.dtype, X.indptr.dtype),
        "csr_row_normalize_l2",
    )
    kernel(
        (math.ceil(X.shape[0] / 32),),
        (32,),
        (X.data, X.indptr, X.shape[0]),
    )
    return X


def csr_diag_mul(X, y, inplace=True):
    """Multiply a sparse X matrix with diagonal matrix y"""
    if not inplace:
        X = X.copy()
    # grab underlying dense ar from y
    y = y.data[0]
    X.data *= y[X.indices]
    return X


def sparse_gram(X, transpose=False, tri=False):
    """Computes `X.T.dot(X)` or `X.dot(X.T)` for sparse matrices.

    Parameters
    ----------
    X : cupyx.scipy.sparse.sp_matrix
        A sparse matrix input.
    transpose : bool
        If True, computes `X.dot(X.T)` instead of `X.T.dot(X)`.
    tri : bool, default=False
        If True, returns an upper-triangular matrix.

    Returns
    -------
    gram : cupy.ndarray
    """
    if transpose:
        X = X.T

    X = X.tocsr()
    X.sum_duplicates()

    # Workaround for cupy #7699 (fixed in cupy 14) and cupy #10033. cupy < 14
    # may try to allocate large amounts for sparse matrix multiplication
    # (spGEMM). cupy >= 14 may still run into this issue, but may also silently
    # return all-0 outputs for large enough matrices. For now we workaround the
    # issue entirely and use a handrolled kernel.
    out = cp.zeros((X.shape[1], X.shape[1]), dtype=X.data.dtype)
    # sanity check, `cupyx.scipy.sparse` already enforces this
    assert X.indices.dtype == X.indptr.dtype
    compute_gram = cuda_kernel_factory(
        """
        ({1} *indptr, {1} *indices, {0} *data, int n_rows, int n_cols, {0} *out) {
            int row = blockIdx.x;
            int col = threadIdx.x;

            if (row >= n_rows) return;

            {1} start = indptr[row];
            {1} end = indptr[row + 1];

            for ({1} idx1 = start; idx1 < end; idx1++) {
                {0} data1 = data[idx1];
                {1} index1 = indices[idx1];
                for ({1} idx2 = idx1 + col; idx2 < end; idx2 += blockDim.x) {
                    {0} data2 = data[idx2];
                    {1} index2 = indices[idx2];
                    atomicAdd(&out[index1 * n_cols + index2], data1 * data2);
                }
            }
        }
        """,
        (X.dtype, X.indices.dtype),
        "compute_gram",
    )
    compute_gram(
        (X.shape[0],),
        (128,),
        (
            X.indptr,
            X.indices,
            X.data,
            X.shape[0],
            X.shape[1],
            out,
        ),
    )

    if not tri:
        reflect_upper = cuda_kernel_factory(
            """
            ({0} *out, int N) {
                int rid = blockDim.x * blockIdx.x + threadIdx.x;
                int cid = blockDim.y * blockIdx.y + threadIdx.y;

                if (rid >= N || cid >= N || rid <= cid) return;
                out[rid * N + cid] = out[cid * N + rid];
            }
            """,
            (X.dtype,),
            "reflect_upper",
        )
        reflect_upper(
            (math.ceil(X.shape[1] / 32),) * 2,
            (32, 32),
            (out, X.shape[1]),
        )
    return out


def sparse_cov_and_mean(X, ddof=1):
    """Computes the covariance and mean of X.

    Parameters
    ----------
    X : cupyx.scipy.sparse, shape=(n_samples, n_features)
        The input data.
    ddof : int, default=1
        Delta degrees of freedom. The divisor used in calculations is
        ``n_samples - ddof``.

    Returns
    -------
    covariance: cp.ndarray, shape=(n_features, n_features)
        The covariance of X, in the form of E(XX) - E(X)E(X).
    mean: cp.ndarray, shape=(n_features,)
        The mean of X, equivalent to X.mean(axis=0).
    """
    X = X.tocsr()
    X.sum_duplicates()

    C = sparse_gram(X, tri=True)

    x_mean = X.mean(axis=0).reshape(-1)
    finish_cov = cuda_kernel_factory(
        """
        ({0} *C, {0} *x_mean, int n_samples, int n_cols, int ddof) {
            int rid = blockDim.x * blockIdx.x + threadIdx.x;
            int cid = blockDim.y * blockIdx.y + threadIdx.y;

            if (rid >= n_cols || cid >= n_cols || rid < cid) return;

            C[cid * n_cols + rid] -= (n_samples * x_mean[rid] * x_mean[cid]);
            C[cid * n_cols + rid] /= (n_samples - ddof);

            if (rid > cid) {
                C[rid * n_cols + cid] = C[cid * n_cols + rid];
            }
        }
        """,
        (X.dtype,),
        "finish_cov",
    )
    finish_cov(
        (math.ceil(X.shape[1] / 32),) * 2,
        (32, 32),
        (C, x_mean, X.shape[0], X.shape[1], ddof),
    )
    return C, x_mean
