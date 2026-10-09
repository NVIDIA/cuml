# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import sklearn.linear_model
from sklearn.datasets import make_regression

import cuml

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


def _make(n_samples=200, n_features=20, n_targets=1, dtype=np.float64):
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(n_features // 2, 1),
        n_targets=n_targets,
        noise=1.0,
        random_state=42,
    )
    return X.astype(dtype), y.astype(dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("gcv_mode", [None, "auto", "svd", "eigen"])
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize(
    "n_samples, n_features",
    [(200, 20), (40, 80)],  # n > p and n < p
)
def test_gcv_matches_sklearn(
    dtype, gcv_mode, fit_intercept, n_samples, n_features
):
    X, y = _make(n_samples=n_samples, n_features=n_features, dtype=dtype)

    cu = cuml.RidgeCV(
        alphas=ALPHAS, fit_intercept=fit_intercept, gcv_mode=gcv_mode
    ).fit(X, y)
    sk = sklearn.linear_model.RidgeCV(
        alphas=ALPHAS, fit_intercept=fit_intercept, gcv_mode=gcv_mode
    ).fit(X, y)

    # cuML computes the GCV in the input dtype, while sklearn always upcasts to
    # float64. In float32 the forward error is ~kappa_eff * eps_f32, so we use
    # per-quantity tolerances (~10x over measured worst case). Note `coef_` is
    # bound by `atol`, not `rtol`: make_regression leaves ~half the true
    # coefficients near zero, where a normal float32 absolute error is a large
    # relative error.
    if dtype == np.float32:
        coef_tol = dict(atol=1e-3, rtol=1e-3)
        intercept_tol = dict(atol=1e-3)
        predict_tol = dict(atol=5e-3, rtol=1e-3)
        best_score_tol = dict(rtol=1e-2)
    else:
        coef_tol = dict(atol=1e-6, rtol=1e-5)
        intercept_tol = dict(atol=1e-6, rtol=1e-5)
        predict_tol = dict(atol=1e-5, rtol=1e-4)
        best_score_tol = dict(rtol=1e-5)

    assert cu.alpha_ == pytest.approx(sk.alpha_)
    np.testing.assert_allclose(cu.coef_, sk.coef_, **coef_tol)
    np.testing.assert_allclose(cu.intercept_, sk.intercept_, **intercept_tol)
    np.testing.assert_allclose(
        cu.best_score_, sk.best_score_, **best_score_tol
    )
    np.testing.assert_allclose(cu.predict(X), sk.predict(X), **predict_tol)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_gcv_sample_weight_matches_sklearn(fit_intercept):
    X, y = _make(n_samples=200, n_features=20)
    sw = np.random.default_rng(0).uniform(0.5, 2.0, size=X.shape[0])

    cu = cuml.RidgeCV(alphas=ALPHAS, fit_intercept=fit_intercept).fit(
        X, y, sample_weight=sw
    )
    sk = sklearn.linear_model.RidgeCV(
        alphas=ALPHAS, fit_intercept=fit_intercept
    ).fit(X, y, sample_weight=sw)

    assert cu.alpha_ == pytest.approx(sk.alpha_)
    np.testing.assert_allclose(cu.coef_, sk.coef_, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(
        cu.predict(X), sk.predict(X), atol=1e-5, rtol=1e-4
    )


@pytest.mark.parametrize("alpha_per_target", [False, True])
def test_gcv_multi_output_matches_sklearn(alpha_per_target):
    X, y = _make(n_samples=300, n_features=20, n_targets=3)

    cu = cuml.RidgeCV(alphas=ALPHAS, alpha_per_target=alpha_per_target).fit(
        X, y
    )
    sk = sklearn.linear_model.RidgeCV(
        alphas=ALPHAS, alpha_per_target=alpha_per_target
    ).fit(X, y)

    np.testing.assert_allclose(cu.alpha_, sk.alpha_)
    assert cu.coef_.shape == (3, 20)
    np.testing.assert_allclose(cu.coef_, sk.coef_, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(
        cu.predict(X), sk.predict(X), atol=1e-5, rtol=1e-4
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gcv_dtype_and_shapes(dtype):
    X, y = _make(n_samples=120, n_features=10, n_targets=3, dtype=dtype)

    # 1D y
    m = cuml.RidgeCV(alphas=ALPHAS).fit(X, y[:, 0])
    assert m.coef_.shape == (10,)
    assert m.coef_.dtype == dtype
    assert np.isscalar(m.intercept_) or m.intercept_.shape == ()

    # multi-output y
    m = cuml.RidgeCV(alphas=ALPHAS).fit(X, y)
    assert m.coef_.shape == (3, 10)
    assert m.coef_.dtype == dtype
    assert m.intercept_.shape == (3,)

    # no intercept
    m = cuml.RidgeCV(alphas=ALPHAS, fit_intercept=False).fit(X, y[:, 0])
    assert m.intercept_ == 0.0


def test_store_cv_results_shape():
    X, y = _make(n_samples=150, n_features=12, n_targets=1)
    m = cuml.RidgeCV(alphas=ALPHAS, store_cv_results=True).fit(X, y)
    assert m.cv_results_.shape == (150, len(ALPHAS))

    X, y = _make(n_samples=150, n_features=12, n_targets=3)
    m = cuml.RidgeCV(alphas=ALPHAS, store_cv_results=True).fit(X, y)
    assert m.cv_results_.shape == (150, 3, len(ALPHAS))


@pytest.mark.parametrize("cv", [3, 5])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_kfold_path_matches_sklearn(cv, fit_intercept):
    X, y = _make(n_samples=300, n_features=15)

    cu = cuml.RidgeCV(alphas=ALPHAS, cv=cv, fit_intercept=fit_intercept).fit(
        X, y
    )
    sk = sklearn.linear_model.RidgeCV(
        alphas=ALPHAS, cv=cv, fit_intercept=fit_intercept
    ).fit(X, y)

    assert cu.alpha_ == pytest.approx(sk.alpha_)
    np.testing.assert_allclose(
        cu.predict(X), sk.predict(X), atol=1e-2, rtol=1e-2
    )


def test_invalid_alphas_gcv():
    X, y = _make()

    est = cuml.RidgeCV(alphas=[0.0, 1.0])
    with pytest.raises(ValueError, match="strictly positive"):
        est.fit(X, y)


def test_incompatible_options_with_cv():
    X, y = _make()
    with pytest.raises(ValueError, match="store_cv_results"):
        cuml.RidgeCV(alphas=ALPHAS, cv=5, store_cv_results=True).fit(X, y)
    with pytest.raises(ValueError, match="alpha_per_target"):
        cuml.RidgeCV(alphas=ALPHAS, cv=5, alpha_per_target=True).fit(X, y)


def test_custom_scoring_not_implemented():
    X, y = _make()

    est = cuml.RidgeCV(alphas=ALPHAS, scoring="r2")
    with pytest.raises(NotImplementedError):
        est.fit(X, y)


def test_sklearn_roundtrip():
    X, y = _make(n_samples=200, n_features=20)
    cu = cuml.RidgeCV(alphas=ALPHAS).fit(X, y)

    sk = cu.as_sklearn()
    assert isinstance(sk, sklearn.linear_model.RidgeCV)
    np.testing.assert_allclose(cu.predict(X), sk.predict(X), rtol=1e-5)

    cu2 = cuml.RidgeCV.from_sklearn(sk)
    np.testing.assert_allclose(cu2.predict(X), sk.predict(X), rtol=1e-4)
