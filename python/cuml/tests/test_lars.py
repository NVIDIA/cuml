# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
import numpy as np
import pytest
import sklearn
from sklearn.datasets import make_regression
from sklearn.linear_model import Lars as skLars

from cuml.linear_model import Lars as cuLars
from cuml.testing.datasets import make_regression_dataset
from cuml.testing.utils import (
    array_equal,
    quality_param,
    stress_param,
    unit_param,
)


def normalize_data(X, y):
    y_mean = np.mean(y)
    y = y - y_mean
    x_mean = np.mean(X, axis=0)
    x_scale = np.sqrt(np.var(X, axis=0) * X.shape[0])
    x_scale[x_scale == 0] = 1
    X = (X - x_mean) / x_scale
    return X, y, x_mean, x_scale, y_mean


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "nrows", [unit_param(500), quality_param(5000), stress_param(90000)]
)
@pytest.mark.parametrize(
    "column_info",
    [
        unit_param([1, 1]),
        unit_param([20, 10]),
        quality_param([100, 50]),
        stress_param([1000, 500]),
    ],
)
@pytest.mark.parametrize("precompute", [True, False, "precompute"])
def test_lars_model(datatype, nrows, column_info, precompute):
    ncols, n_info = column_info
    X_train, X_test, y_train, y_test = make_regression_dataset(
        datatype, nrows, ncols, n_info
    )

    if precompute == "precompute":
        # Apply normalization manually, because the solver expects normalized
        # input data
        X_train, y_train, x_mean, x_scale, y_mean = normalize_data(
            X_train, y_train
        )
        y_test = y_test - y_mean
        X_test = (X_test - x_mean) / x_scale

    if precompute == "precompute":
        precompute = np.dot(X_train.T, X_train)

    params = {"precompute": precompute}

    # Initialization of cuML's LARS
    culars = cuLars(**params)

    # fit and predict cuml LARS
    culars.fit(X_train, y_train)

    cu_score_train = culars.score(X_train, y_train)
    cu_score_test = culars.score(X_test, y_test)

    if nrows < 500000:
        # sklearn model initialization, fit and predict
        sklars = skLars(**params)
        sklars.fit(X_train, y_train)
        # Set tolerance to include the 95% confidence interval around
        # scikit-learn accuracy.
        accuracy_target = sklars.score(X_test, y_test)
        tol = 1.96 * np.sqrt(accuracy_target * (1.0 - accuracy_target) / 100.0)
        if tol < 0.002:
            tol = 0.002  # We allow at least 0.2% tolerance
        print(cu_score_train, cu_score_test, accuracy_target, tol)
        assert cu_score_train >= sklars.score(X_train, y_train) - tol
        assert cu_score_test >= accuracy_target - tol
    else:
        assert cu_score_test > 0.95


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "nrows", [unit_param(500), quality_param(5000), stress_param(500000)]
)
@pytest.mark.parametrize(
    "column_info",
    [
        unit_param([20, 10]),
        quality_param([100, 50]),
        stress_param([1000, 500]),
    ],
)
@pytest.mark.parametrize("precompute", [True, False])
def test_lars_collinear(datatype, nrows, column_info, precompute):
    # Assume at least 4GB memory
    max_gpu_memory = pytest.max_gpu_memory or 4

    ncols, n_info = column_info
    if nrows == 500000 and ncols == 1000 and max_gpu_memory < 32:
        if pytest.adapt_stress_test:
            nrows = nrows * max_gpu_memory // 32
        else:
            pytest.skip(
                "Insufficient GPU memory for this test."
                "Re-run with 'CUML_ADAPT_STRESS_TESTS=True'"
            )

    X_train, X_test, y_train, y_test = make_regression_dataset(
        datatype, nrows, ncols, n_info
    )
    n_duplicate = min(ncols, 100)
    X_train = np.concatenate((X_train, X_train[:, :n_duplicate]), axis=1)
    X_test = np.concatenate((X_test, X_test[:, :n_duplicate]), axis=1)

    params = {"precompute": precompute, "n_nonzero_coefs": ncols + n_duplicate}
    culars = cuLars(**params)
    culars.fit(X_train, y_train)

    assert culars.score(X_train, y_train) > 0.85
    assert culars.score(X_test, y_test) > 0.85


@pytest.mark.skipif(
    sklearn.__version__ >= "1.0",
    reason="discrepancies on coefficients with sklearn 1.2",
)
@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "params",
    [
        {"precompute": True},
        {"precompute": False},
        {"n_nonzero_coefs": 5},
        {"n_nonzero_coefs": 2},
        {"n_nonzero_coefs": 2, "fit_intercept": False},
    ],
)
def test_lars_attributes(datatype, params):
    X, y = make_regression(
        n_samples=20000, n_features=8, n_informative=5, random_state=0
    )
    X = X.astype(datatype)
    y = y.astype(datatype)

    culars = cuLars(**params)
    culars.fit(X, y)

    sklars = skLars(**params)
    sklars.fit(X, y)

    assert culars.score(X, y) >= sklars.score(X, y) - 0.01

    limit_max_iter = "n_nonzero_coefs" in params
    if limit_max_iter:
        n_iter_tol = 0
    else:
        n_iter_tol = 2

    assert abs(culars.n_iter_ - sklars.n_iter_) <= n_iter_tol

    tol = 1e-4 if params.pop("fit_intercept", True) else 1e-1
    n = min(culars.n_iter_, sklars.n_iter_)
    assert array_equal(
        culars.alphas_[:n], sklars.alphas_[:n], unit_tol=tol, total_tol=1e-4
    )
    assert array_equal(culars.active_[:n], sklars.active_[:n])

    if limit_max_iter:
        assert array_equal(culars.coef_, sklars.coef_)

        if hasattr(sklars, "coef_path_"):
            assert array_equal(
                culars.coef_path_,
                sklars.coef_path_[sklars.active_],
                unit_tol=1e-3,
            )
            assert culars.coef_path_.shape[1] == culars.n_iter_ + 1

        intercept_diff = abs(culars.intercept_ - sklars.intercept_)
        if abs(sklars.intercept_) > 1e-6:
            intercept_diff /= sklars.intercept_
            assert intercept_diff <= 1e-3


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
def test_lars_copy_X(datatype):
    X, y = make_regression(
        n_samples=20000,
        n_features=8,
        n_informative=5,
        random_state=0,
    )
    X = cp.asarray(X, dtype=datatype, order="F")
    y = cp.asarray(y, dtype=datatype, order="F")

    X0 = cp.copy(X)
    culars1 = cuLars(precompute=False, copy_X=True)
    culars1.fit(X, y)
    # Test that array was not changed
    assert cp.all(X0 == X)

    # We make a copy of X during preprocessing, we should preprocess
    # in place if copy_X is false to save memory. Afterwards we can enable
    # the following test:
    # culars2 = cuLars(precompute=False, copy_X=False)
    # culars2.fit(X, y)
    # Test that array was changed i.e. no unnecessary copies were made
    # assert cp.any(X0 != X)
    #
    # assert abs(culars1.score(X, y) - culars2.score(X, y)) < 1e-9
