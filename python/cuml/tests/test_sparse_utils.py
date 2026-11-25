# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import cupyx.scipy.sparse as cp_sp
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.utils.sparsefuncs_fast import (
    inplace_csr_row_normalize_l1,
    inplace_csr_row_normalize_l2,
)

from cuml.common.sparse import (
    csr_row_normalize_l1,
    csr_row_normalize_l2,
    sparse_cov_and_mean,
    sparse_gram,
)


@pytest.mark.parametrize(
    "norm, ref_norm",
    [
        (csr_row_normalize_l1, inplace_csr_row_normalize_l1),
        (csr_row_normalize_l2, inplace_csr_row_normalize_l2),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("seed, shape", [(10, (10, 5)), (123, (500, 12))])
def test_csr_norms(norm, ref_norm, dtype, seed, shape):
    X = np.random.RandomState(seed).randn(*shape).astype(dtype)
    X_csr = sp.csr_matrix(X)
    X_csr_gpu = cp_sp.csr_matrix(X_csr)

    norm(X_csr_gpu)
    ref_norm(X_csr)

    # checks that array have been changed inplace
    assert cp.any(cp.not_equal(X_csr_gpu.todense(), cp.array(X)))

    cp.testing.assert_array_almost_equal(X_csr_gpu.todense(), X_csr.todense())


@pytest.mark.parametrize("n_cols", [1, 100, 1000])
@pytest.mark.parametrize("n_rows", [2, 100, 1000])
@pytest.mark.parametrize("density", [0.2, 0.4, 0.6])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("format", ["csr", "coo"])
@pytest.mark.parametrize("transpose", [True, False])
def test_sparse_gram(n_cols, n_rows, density, dtype, format, transpose):
    X = cp_sp.random(
        n_rows,
        n_cols,
        density=density,
        format=format,
        dtype=dtype,
        random_state=42,
    )
    X_dense = X.todense()
    sol = X_dense.dot(X_dense.T) if transpose else X_dense.T.dot(X_dense)
    res = sparse_gram(X, transpose=transpose)
    assert res.dtype == dtype
    cp.testing.assert_allclose(
        res, sol, atol=1e-2 if res.dtype == "f4" else 1e-4
    )


@pytest.mark.parametrize("n_cols", [1, 100, 1000])
@pytest.mark.parametrize("n_rows", [2, 100, 1000])
@pytest.mark.parametrize("density", [0.2, 0.4, 0.6])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("format", ["csr", "coo"])
@pytest.mark.parametrize("ddof", [0, 1])
def test_sparse_cov_and_mean(n_cols, n_rows, density, dtype, format, ddof):
    X = 10 * cp_sp.random(
        n_rows,
        n_cols,
        density=density,
        format=format,
        dtype=dtype,
        random_state=42,
    )
    cov_sol = cp.cov(X.todense(), ddof=ddof, rowvar=False)
    mean_sol = X.mean(axis=0).reshape(-1)
    cov, mean = sparse_cov_and_mean(X, ddof=ddof)
    assert cov.dtype == dtype
    assert mean.dtype == dtype
    cp.testing.assert_allclose(cov, cov_sol, atol=1e-4)
    cp.testing.assert_allclose(mean, mean_sol, atol=1e-4)
