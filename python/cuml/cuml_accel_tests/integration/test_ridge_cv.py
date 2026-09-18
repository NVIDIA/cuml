#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

ALPHAS = [0.1, 1.0, 10.0, 100.0]


@pytest.fixture(scope="module")
def regression_data():
    X, y = make_regression(
        n_samples=500,
        n_features=20,
        n_informative=10,
        noise=0.1,
        random_state=42,
    )
    X = StandardScaler().fit_transform(X)
    return X, y


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_ridge_cv_gcv_runs_on_gpu(regression_data, fit_intercept):
    X, y = regression_data
    model = RidgeCV(alphas=ALPHAS, fit_intercept=fit_intercept)
    model.fit(X, y)
    # cv=None (GCV) is accelerated on GPU
    assert model._gpu is not None
    y_pred = model.predict(X)
    assert r2_score(y, y_pred) > 0.5


def test_ridge_cv_selects_best_alpha(regression_data):
    X, y = regression_data
    model = RidgeCV(alphas=ALPHAS).fit(X, y)
    assert model._gpu is not None

    # The grid fit should pick the alpha that maximizes the CV score, i.e. the
    # same one we'd get by scoring each alpha on its own.
    scores = {a: RidgeCV(alphas=[a]).fit(X, y).best_score_ for a in ALPHAS}
    assert model.alpha_ == max(scores, key=scores.get)
    assert model.best_score_ == pytest.approx(scores[model.alpha_])


def test_ridge_cv_explicit_cv_falls_back_to_cpu(regression_data):
    X, y = regression_data
    model = RidgeCV(alphas=ALPHAS, cv=5)
    model.fit(X, y)
    # An explicit cv is not supported on GPU, should fall back to CPU
    assert model._gpu is None
    y_pred = model.predict(X)
    assert r2_score(y, y_pred) > 0.5


def test_ridge_cv_custom_scoring_falls_back_to_cpu(regression_data):
    X, y = regression_data
    model = RidgeCV(alphas=ALPHAS, scoring="neg_mean_squared_error")
    model.fit(X, y)
    # Custom scoring is not supported on GPU, should fall back to CPU
    assert model._gpu is None
    y_pred = model.predict(X)
    assert r2_score(y, y_pred) > 0.5


def test_ridge_cv_multi_output_runs_on_gpu():
    X, y = make_regression(
        n_samples=400, n_features=15, n_targets=3, noise=0.1, random_state=0
    )
    model = RidgeCV(alphas=ALPHAS).fit(X, y)
    assert model._gpu is not None
    assert model.coef_.shape == (3, 15)
    np.testing.assert_array_equal(np.asarray(model.predict(X)).shape, (400, 3))
