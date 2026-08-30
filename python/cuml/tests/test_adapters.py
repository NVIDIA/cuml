#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import platform

import cupy as cp
import cupyx as cpx
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.utils._mask import _get_mask as sk_get_mask

from cuml.testing.test_preproc_utils import assert_allclose
from cuml.thirdparty_adapters.adapters import _get_mask as cu_get_mask
from cuml.thirdparty_adapters.adapters import (
    _masked_column_mean,
    _masked_column_median,
    _masked_column_mode,
)
from cuml.thirdparty_adapters.sparsefuncs_fast import (
    _csc_mean_variance_axis0,
    csc_mean_variance_axis0,
    csr_mean_variance_axis0,
)

IS_ARM = platform.processor() == "aarch64"


@pytest.fixture(scope="session", params=["zero", "one", "nan"])
def mask_dataset(request, random_seed):
    cp.random.seed(random_seed)
    randint = cp.random.randint(30, size=(500, 20))
    randint = randint.astype(cp.float64)
    if request.param == "zero":
        mask_value = 0
    elif request.param == "one":
        mask_value = 1
    else:
        mask_value = cp.nan
    random_loc = cp.random.choice(
        randint.size, int(randint.size * 0.3), replace=False
    )
    randint.ravel()[random_loc] = mask_value
    return mask_value, randint.get(), randint


@pytest.fixture(scope="session", params=["cupy-csr", "cupy-csc"])
def sparse_random_dataset(request, random_seed):
    cp.random.seed(random_seed)
    X = cp.random.rand(100, 10)
    random_loc = cp.random.choice(X.size, int(X.size * 0.3), replace=False)
    X.ravel()[random_loc] = 0
    if request.param == "cupy-csr":
        X_sparse = cpx.scipy.sparse.csr_matrix(X)
    elif request.param == "cupy-csc":
        X_sparse = cpx.scipy.sparse.csc_matrix(X)
    return X.get(), X, X_sparse.get(), X_sparse


def test_csr_mean_variance_axis0(failure_logger, sparse_random_dataset):
    X_np, _, _, X_sparse = sparse_random_dataset
    if X_sparse.format != "csr":
        pytest.skip("Skip non CSR matrices")

    means, variances = csr_mean_variance_axis0(X_sparse)

    ref_means = np.nanmean(X_np, axis=0)
    ref_variances = np.nanvar(X_np, axis=0)

    assert_allclose(means, ref_means)
    assert_allclose(variances, ref_variances)


def test_csc_mean_variance_axis0(failure_logger, sparse_random_dataset):
    X_np, _, _, X_sparse = sparse_random_dataset
    if X_sparse.format != "csc":
        pytest.skip("Skip non CSC matrices")

    means, variances = csc_mean_variance_axis0(X_sparse)

    ref_means = np.nanmean(X_np, axis=0)
    ref_variances = np.nanvar(X_np, axis=0)

    assert_allclose(means, ref_means)
    assert_allclose(variances, ref_variances)


def test__csc_mean_variance_axis0(failure_logger, sparse_random_dataset):
    X_np, _, _, X_sparse = sparse_random_dataset
    if X_sparse.format != "csc":
        pytest.skip("Skip non CSC matrices")

    means, variances, counts_nan = _csc_mean_variance_axis0(X_sparse)

    ref_means = np.nanmean(X_np, axis=0)
    ref_variances = np.nanvar(X_np, axis=0)
    ref_counts_nan = np.isnan(X_np).sum(axis=0)

    assert_allclose(means, ref_means)
    assert_allclose(variances, ref_variances)
    assert_allclose(counts_nan, ref_counts_nan)


def test_get_mask(failure_logger, mask_dataset):
    mask_value, X_np, X = mask_dataset
    cu_mask = cu_get_mask(X, value_to_mask=mask_value)
    sk_mask = sk_get_mask(X_np, value_to_mask=mask_value)
    assert_allclose(cu_mask, sk_mask)


def test_get_mask_nan_string_on_host_object_array():
    X = np.array([1.0, np.nan, "present"], dtype=object)

    np.testing.assert_array_equal(
        cu_get_mask(X, value_to_mask="NaN"), [False, True, False]
    )


@pytest.mark.parametrize(
    ("function", "expected"),
    [
        (_masked_column_mean, [3.0, 5.0]),
        (_masked_column_median, [3.0, 5.0]),
    ],
)
@pytest.mark.parametrize("missing_value", [pd.NA, "NaN"])
def test_masked_column_numeric_na_sentinel(function, expected, missing_value):
    X = cp.array([[1.0, cp.nan], [3.0, 4.0], [5.0, 6.0]])

    result = function(X, missing_value)

    np.testing.assert_allclose(result.get(), expected)


def test_masked_column_median(failure_logger, mask_dataset):
    mask_value, X_np, X = mask_dataset
    median = _masked_column_median(X, mask_value).get()
    mask = ~sk_get_mask(X_np, value_to_mask=mask_value)
    n_columns = X.shape[1]
    for i in range(n_columns):
        column_mask = mask[:, i]
        column_median = np.median(X_np[:, i][column_mask])
        assert column_median == median[i]


def test_masked_column_mean(failure_logger, mask_dataset):
    mask_value, X_np, X = mask_dataset
    mean = _masked_column_mean(X, mask_value).get()
    mask = ~sk_get_mask(X_np, value_to_mask=mask_value)
    n_columns = X.shape[1]
    for i in range(n_columns):
        column_mask = mask[:, i]
        column_mean = np.mean(X_np[:, i][column_mask])
        assert column_mean == mean[i]


def test_masked_column_mode(failure_logger, mask_dataset):
    mask_value, X_np, X = mask_dataset
    mode = _masked_column_mode(X, mask_value).get()
    mask = ~sk_get_mask(X_np, value_to_mask=mask_value)
    n_columns = X.shape[1]
    for i in range(n_columns):
        column_mask = mask[:, i]
        column_mode = stats.mode(X_np[:, i][column_mask], keepdims=True)[0][0]
        assert column_mode == mode[i]
