#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import math

import cupy as cp
import cupyx.scipy.sparse

from cuml.common.kernel_utils import cuda_kernel_factory


def csr_mean_variance_axis0(X):
    """Compute mean and variance on the axis 0 of a CSR matrix

    Parameters
    ----------
    X : sparse CSR matrix
        Input array

    Returns
    -------
    mean and variance
    """
    X = X.tocsc()
    means, variances, _ = _csc_mean_variance_axis0(X)
    return means, variances


def csc_mean_variance_axis0(X):
    """Compute mean and variance on the axis 0 of a CSC matrix

    Parameters
    ----------
    X : sparse CSC matrix
        Input array

    Returns
    -------
    mean and variance
    """
    means, variances, _ = _csc_mean_variance_axis0(X)
    return means, variances


def _csc_mean_variance_axis0(X):
    """Compute mean, variance and nans count on the axis 0 of a CSC matrix

    Parameters
    ----------
    X : sparse CSC matrix
        Input array

    Returns
    -------
    mean, variance, nans count
    """
    n_samples, n_features = X.shape

    means = cp.empty(n_features)
    variances = cp.empty(n_features)
    counts_nan = cp.empty(n_features)

    start = X.indptr[0]
    for i, end in enumerate(X.indptr[1:]):
        col = X.data[start:end]

        _count_zeros = n_samples - col.size
        _count_nans = (col != col).sum()

        _mean = cp.nansum(col) / (n_samples - _count_nans)
        _variance = cp.nansum((col - _mean) ** 2)
        _variance += _count_zeros * (_mean**2)
        _variance /= n_samples - _count_nans

        means[i] = _mean
        variances[i] = _variance
        counts_nan[i] = _count_nans

        start = end
    return means, variances, counts_nan


_PERFORM_EXPANSION_KERNEL = """
({0} *indptr,
 {0} *indices,
 {1} *data,
 {2} *expanded_indptr,
 {2} *expanded_indices,
 {1} *expanded_data,
 long long n_rows,
 long long n_cols,
 int interaction_only,
 int is_degree_2) {
    {0} row_i = blockIdx.x * blockDim.x + threadIdx.x;
    {0} inrow_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (row_i >= n_rows) return;

    {2} expanded_index = expanded_indptr[row_i] + inrow_idx;
    if (expanded_index >= expanded_indptr[row_i + 1]) return;

    {0} row_starts = indptr[row_i];
    {0} row_ends = indptr[row_i + 1];

    {0} i_ptr = row_starts;
    {0} j_ptr = -1;
    {0} k_ptr = inrow_idx;

    if (is_degree_2) {
        j_ptr = inrow_idx;
        for ({0} i = row_starts; i < row_ends; i++) {
            {0} diff = row_ends - i - interaction_only;
            if (j_ptr >= diff) {
                j_ptr -= diff;
            } else {
                i_ptr = i;
                break;
            }
        }
        j_ptr += i_ptr + interaction_only;
    } else {
        for ({0} i = row_starts; i < row_ends; i++) {
            for ({0} j = i + interaction_only; j < row_ends; j++) {
                {0} diff = row_ends - j - interaction_only;
                if (k_ptr >= diff) {
                    k_ptr -= diff;
                } else {
                    j_ptr = j;
                    i_ptr = i;
                    break;
                }
            }
            if (j_ptr != -1) break;
        }
        k_ptr += j_ptr + interaction_only;
    }

    // Always use long long here to avoid overflow
    long long i = indices[i_ptr];
    long long j = indices[j_ptr];

    if (is_degree_2) {
        expanded_indices[expanded_index] = (
            interaction_only
            ? n_cols * i - (i*i + 3 * i) / 2 - 1 + j
            : n_cols * i - (i*i + i) / 2 + j
        );
        expanded_data[expanded_index] = data[i_ptr] * data[j_ptr];
    } else {
        long long k = indices[k_ptr];
        expanded_indices[expanded_index] = (
            interaction_only
            ? (3*i*n_cols*n_cols - 3*i*i*n_cols + i*i*i + 11*i - 3*j*j - 9*j)/6
              + i*i - 2*i*n_cols + n_cols*j - n_cols + k
            : (3*i*n_cols*n_cols - 3*i*i*n_cols + i*i*i - i - 3*j*j - 3*j)/6
              + n_cols*j + k
        );
        expanded_data[expanded_index] = data[i_ptr] * data[j_ptr] * data[k_ptr];
    }
}
"""


def csr_polynomial_expansion(X, interaction_only, degree):
    """Apply polynomial expansion on CSR matrix

    Parameters
    ----------
    X : cupyx.scipy.sparse.csr_matrix
        Input matrix.
    interaction_only : bool
        Whether to produce only interaction features.
    degree : {2, 3}
        The polynomial degree.

    Returns
    -------
    X_t : cupy.scipy.sparse.csr_matrix or None
        New polynomial-expanded matrix. If there are no columns in the expanded
        matrix, returns None instead.
    """
    assert degree in (2, 3)
    assert X.format == "csr"
    assert X.indices.dtype == X.indptr.dtype

    n_cols = X.shape[1]
    interaction_only = int(bool(interaction_only))

    if degree == 2:
        expanded_dimensionality = int(
            (n_cols**2 + n_cols) / 2 - interaction_only * n_cols
        )
    else:
        expanded_dimensionality = int(
            (n_cols**3 + 3 * n_cols**2 + 2 * n_cols) / 6
            - interaction_only * n_cols**2
        )
    if expanded_dimensionality == 0:
        return None

    nnz = cp.diff(X.indptr)
    if degree == 2:
        nnz = (nnz**2 + nnz) / 2 - interaction_only * nnz
    else:
        nnz = (nnz**3 + 3 * nnz**2 + 2 * nnz) / 6 - interaction_only * nnz**2
    nnz_max = int(nnz.max())
    nnz_sum = int(nnz.sum())

    # Use int32 for output indices when possible, falling back to int64
    out_ind_dtype = "int32" if nnz_sum <= cp.iinfo("int32").max else "int64"
    expanded_data = cp.empty(shape=nnz_sum, dtype=X.data.dtype)
    expanded_indices = cp.empty(shape=nnz_sum, dtype=out_ind_dtype)
    expanded_indptr = cp.empty(shape=X.shape[0] + 1, dtype=out_ind_dtype)
    expanded_indptr[0] = X.indptr[0]
    nnz.cumsum(out=expanded_indptr[1:])
    del nnz

    perform_expansion = cuda_kernel_factory(
        _PERFORM_EXPANSION_KERNEL,
        (X.indptr.dtype, X.data.dtype, expanded_indptr.dtype),
        "perform_expansion",
    )
    perform_expansion(
        (math.ceil(X.indptr.shape[0] / 32), math.ceil(nnz_max / 32)),
        (32, 32),
        (
            X.indptr,
            X.indices,
            X.data,
            expanded_indptr,
            expanded_indices,
            expanded_data,
            X.shape[0],
            X.shape[1],
            interaction_only,
            degree == 2,
        ),
    )
    return cupyx.scipy.sparse.csr_matrix(
        (expanded_data, expanded_indices, expanded_indptr),
        shape=(X.shape[0], expanded_dimensionality),
    )
