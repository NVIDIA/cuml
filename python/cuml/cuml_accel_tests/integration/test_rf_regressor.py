#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from cuml.accel.profilers import profile


@pytest.fixture(scope="module")
def regression_data():
    # Create a synthetic regression dataset.
    X, y = make_regression(
        n_samples=300,
        n_features=20,
        n_informative=10,
        noise=0.1,
        random_state=42,
    )
    return X, y


@pytest.mark.parametrize("n_estimators", [10, 50, 100])
def test_rf_n_estimators_reg(regression_data, n_estimators):
    X, y = regression_data
    reg = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    reg.fit(X, y)
    _ = r2_score(y, reg.predict(X))


@pytest.mark.parametrize("criterion", ["squared_error", "absolute_error"])
def test_rf_criterion_reg(regression_data, criterion):
    X, y = regression_data
    reg = RandomForestRegressor(
        criterion=criterion, n_estimators=50, random_state=42
    )
    reg.fit(X, y)
    score1 = r2_score(y, reg.predict(X))
    assert isinstance(score1, float)
    score2 = reg.score(X, y)
    assert isinstance(score2, float)


@pytest.mark.parametrize("max_depth", [None, 5, 10])
def test_rf_max_depth_reg(regression_data, max_depth):
    X, y = regression_data
    reg = RandomForestRegressor(
        max_depth=max_depth, n_estimators=50, random_state=42
    )
    reg.fit(X, y)
    _ = r2_score(y, reg.predict(X))


@pytest.mark.parametrize("min_samples_split", [2, 5, 10])
def test_rf_min_samples_split_reg(regression_data, min_samples_split):
    X, y = regression_data
    reg = RandomForestRegressor(
        min_samples_split=min_samples_split, n_estimators=50, random_state=42
    )
    reg.fit(X, y)
    _ = r2_score(y, reg.predict(X))


@pytest.mark.parametrize("min_samples_leaf", [1, 2, 4])
def test_rf_min_samples_leaf_reg(regression_data, min_samples_leaf):
    X, y = regression_data
    reg = RandomForestRegressor(
        min_samples_leaf=min_samples_leaf, n_estimators=50, random_state=42
    )
    reg.fit(X, y)
    _ = r2_score(y, reg.predict(X))


@pytest.mark.parametrize("bootstrap", [False, True])
def test_rf_sample_weight_reg(bootstrap):
    zero_weight_X = np.linspace(-3.0, -1.0, 72).reshape(-1, 1)
    one_weight_X = np.linspace(1.0, 3.0, 72).reshape(-1, 1)
    X = np.vstack([zero_weight_X, one_weight_X])
    y = np.array([-100.0] * len(zero_weight_X) + [4.0] * len(one_weight_X))
    sample_weight = np.array(
        [0.0] * len(zero_weight_X) + [1.0] * len(one_weight_X)
    )
    probe = np.linspace(-3.5, 3.5, 17).reshape(-1, 1)

    reg = RandomForestRegressor(
        n_estimators=3,
        bootstrap=bootstrap,
        max_depth=3,
        max_features=1.0,
        random_state=42,
    )
    reg.fit(X, y, sample_weight=sample_weight)

    expected = np.full(probe.shape[0], 4.0)
    np.testing.assert_allclose(reg.predict(probe), expected)
    assert reg.score(X, y, sample_weight=sample_weight) == pytest.approx(
        r2_score(y, reg.predict(X), sample_weight=sample_weight)
    )


@pytest.mark.parametrize("min_weight_fraction_leaf", [0.0, 0.1])
def test_rf_min_weight_fraction_leaf_reg(
    regression_data, min_weight_fraction_leaf
):
    X, y = regression_data
    reg = RandomForestRegressor(
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        n_estimators=50,
        random_state=42,
    )
    reg.fit(X, y)
    _ = r2_score(y, reg.predict(X))


@pytest.mark.parametrize("max_features", ["sqrt", "log2", 0.5, 5])
def test_rf_max_features_reg(regression_data, max_features):
    X, y = regression_data
    reg = RandomForestRegressor(
        max_features=max_features, n_estimators=50, random_state=42
    )
    reg.fit(X, y)
    _ = r2_score(y, reg.predict(X))


@pytest.mark.parametrize("max_leaf_nodes", [None, 10, 20])
def test_rf_max_leaf_nodes_reg(regression_data, max_leaf_nodes):
    X, y = regression_data
    reg = RandomForestRegressor(
        max_leaf_nodes=max_leaf_nodes, n_estimators=50, random_state=42
    )
    reg.fit(X, y)
    _ = r2_score(y, reg.predict(X))


@pytest.mark.parametrize("min_impurity_decrease", [0.0, 0.1])
def test_rf_min_impurity_decrease_reg(regression_data, min_impurity_decrease):
    X, y = regression_data
    reg = RandomForestRegressor(
        min_impurity_decrease=min_impurity_decrease,
        n_estimators=50,
        random_state=42,
    )
    reg.fit(X, y)
    _ = r2_score(y, reg.predict(X))


@pytest.mark.parametrize("bootstrap", [True, False])
def test_rf_bootstrap_reg(regression_data, bootstrap):
    X, y = regression_data
    reg = RandomForestRegressor(
        bootstrap=bootstrap, n_estimators=50, random_state=42
    )
    reg.fit(X, y)
    _ = r2_score(y, reg.predict(X))


@pytest.mark.parametrize("n_jobs", [1, -1])
def test_rf_n_jobs_reg(regression_data, n_jobs):
    X, y = regression_data
    reg = RandomForestRegressor(
        n_jobs=n_jobs, n_estimators=50, random_state=42
    )
    reg.fit(X, y)
    _ = r2_score(y, reg.predict(X))


@pytest.mark.parametrize("verbose", [0, 1])
def test_rf_verbose_reg(regression_data, verbose):
    X, y = regression_data
    reg = RandomForestRegressor(
        verbose=verbose, n_estimators=50, random_state=42
    )
    reg.fit(X, y)
    _ = r2_score(y, reg.predict(X))


@pytest.mark.parametrize("warm_start", [False, True])
def test_rf_warm_start_reg(regression_data, warm_start):
    X, y = regression_data
    reg = RandomForestRegressor(
        warm_start=warm_start, n_estimators=50, random_state=42
    )
    reg.fit(X, y)
    _ = r2_score(y, reg.predict(X))


@pytest.mark.parametrize("ccp_alpha", [0.0, 0.1])
def test_rf_ccp_alpha_reg(regression_data, ccp_alpha):
    X, y = regression_data
    reg = RandomForestRegressor(
        ccp_alpha=ccp_alpha, n_estimators=50, random_state=42
    )
    reg.fit(X, y)
    _ = r2_score(y, reg.predict(X))


@pytest.mark.parametrize("max_samples", [None, 0.8, 50])
def test_rf_max_samples_reg(regression_data, max_samples):
    X, y = regression_data
    reg = RandomForestRegressor(
        max_samples=max_samples,
        bootstrap=True,
        n_estimators=50,
        random_state=42,
    )
    reg.fit(X, y)
    _ = r2_score(y, reg.predict(X))


@pytest.mark.parametrize("n_targets", [1, 2])
def test_oob_score(n_targets: int) -> None:
    X, y = make_regression(
        n_samples=300, n_features=20, n_targets=n_targets, random_state=42
    )
    y = y.astype(np.int32)
    reg = RandomForestRegressor(
        oob_score=True,
        n_estimators=50,
        random_state=42,
    )
    with profile(quiet=True) as stats:
        reg.fit(X, y)
        predictions = reg.predict(X)
        score = reg.score(X, y)

    for method in ("fit", "predict", "score"):
        calls = stats.method_calls[f"RandomForestRegressor.{method}"]
        assert calls.gpu_calls == 1
    assert predictions.shape == reg.oob_prediction_.shape == y.shape
    assert score == pytest.approx(r2_score(y, predictions))
    assert reg.oob_score_ == pytest.approx(r2_score(y, reg.oob_prediction_))
