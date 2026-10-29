# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import cupyx.scipy.sparse as cp_sp
import hypothesis.strategies as st
import numpy as np
import pytest
import scipy.sparse as sp
import sklearn.linear_model
from hypothesis import assume, example, given
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

import cuml
from cuml.internals.param_validation import InvalidParameterError


def reg_example(**kwargs):
    # Merge in common options
    opts = {
        "fit_intercept": True,
        "weighted": False,
        "n_targets": 1,
        "n_samples": 500,
        "n_features": 20,
        "dtype": np.float32,
        **kwargs,
    }
    return example(**opts)


@given(
    solver=st.sampled_from(["eig", "svd", "lsmr"]),
    fit_intercept=st.booleans(),
    weighted=st.booleans(),
    n_targets=st.integers(min_value=1, max_value=3),
    n_samples=st.integers(min_value=500, max_value=5000),
    n_features=st.integers(min_value=20, max_value=100),
    dtype=st.sampled_from([np.float32, np.float64]),
)
@reg_example(solver="eig")
@reg_example(solver="eig", dtype=np.float64)
@reg_example(solver="eig", fit_intercept=False)
@reg_example(solver="eig", weighted=True)
@reg_example(solver="svd")
@reg_example(solver="svd", dtype=np.float64)
@reg_example(solver="svd", n_targets=3)
@reg_example(solver="svd", fit_intercept=False)
@reg_example(solver="svd", weighted=True)
def test_ridge_regression(
    solver,
    fit_intercept,
    weighted,
    n_targets,
    n_samples,
    n_features,
    dtype,
):
    # eig solver doesn't support multi-target
    assume(n_targets == 1 or solver != "eig")

    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        n_targets=n_targets,
        random_state=42,
    )
    X = X.astype(dtype, copy=False)
    y = y.astype(dtype, copy=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=42
    )
    if weighted:
        sample_weight = (
            np.random.default_rng(42)
            .uniform(0.5, 1, len(X_train))
            .astype(dtype)
        )
    else:
        sample_weight = None

    opts = {"fit_intercept": fit_intercept, "alpha": 15.2, "tol": 1e-6}

    cu_model = cuml.Ridge(solver=solver, **opts)
    cu_model.fit(X_train, y_train, sample_weight=sample_weight)
    cu_pred = cu_model.predict(X_test)

    sk_model = sklearn.linear_model.Ridge(**opts)
    sk_model.fit(X_train, y_train, sample_weight=sample_weight)
    sk_pred = sk_model.predict(X_test)

    # Check predictions are close
    np.testing.assert_allclose(cu_pred, sk_pred, atol=1e-2)


@pytest.mark.parametrize("solver", ["eig", "svd", "lsmr"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ridge_coef_and_intercept_shape(solver, dtype):
    n_samples, n_features, n_targets = 33, 7, 3
    X, y = make_regression(
        n_targets=n_targets,
        n_features=n_features,
        n_samples=n_samples,
        random_state=42,
    )
    X = X.astype(dtype)
    y = y.astype(dtype)

    # 1D y
    model = cuml.Ridge(solver=solver).fit(X, y[:, 0])
    assert model.coef_.shape == (n_features,)
    assert model.coef_.dtype == dtype
    assert model.intercept_.shape == ()
    assert model.intercept_.dtype == dtype

    # 2D y with 1 column
    model = cuml.Ridge(solver=solver).fit(X, y[:, 0, None])
    assert model.coef_.shape == (n_features,)
    assert model.coef_.dtype == dtype
    assert model.intercept_.shape == (1,)
    assert model.intercept_.dtype == dtype

    if solver != "eig":
        # 2D y with 3 columns
        model = cuml.Ridge(solver=solver).fit(X, y)
        assert model.coef_.shape == (n_targets, n_features)
        assert model.intercept_.shape == (n_targets,)

    # No intercept
    model = cuml.Ridge(solver=solver, fit_intercept=False).fit(X, y[:, 0])
    assert model.coef_.shape == (n_features,)
    assert model.intercept_ == 0.0
    assert isinstance(model.intercept_, float)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("n_targets", [0, 1, 3])
def test_ridge_sparse(dtype, fit_intercept, weighted, n_targets):
    n_samples, n_features = 3000, 500
    rng = np.random.default_rng(42)
    coef = rng.random((n_features, n_targets) if n_targets else n_features)
    if fit_intercept:
        intercept = rng.uniform(-10, 10, size=n_targets or ())
    else:
        intercept = 0.0
    X = sp.rand(
        n_samples, n_features, density=0.2, random_state=42, dtype=dtype
    )
    y = (X.dot(coef) + intercept).astype(dtype)

    if weighted:
        sample_weight = rng.uniform(0.5, 1, size=n_samples).astype(dtype)
    else:
        sample_weight = None

    cu_model = cuml.Ridge(tol=1e-6, fit_intercept=fit_intercept)
    cu_model.fit(X, y, sample_weight=sample_weight)
    cu_pred = cu_model.predict(X)

    sk_model = sklearn.linear_model.Ridge(
        solver="lsqr", fit_intercept=fit_intercept
    )
    sk_model.fit(X, y, sample_weight=sample_weight)
    sk_pred = sk_model.predict(X).reshape(cu_pred.shape)

    # Check shapes and dtypes
    assert cu_model.coef_.dtype == dtype
    assert cu_model.coef_.shape == (
        (n_targets, n_features) if n_targets > 1 else (n_features,)
    )
    if fit_intercept:
        assert cu_model.intercept_.dtype == dtype
        assert cu_model.intercept_.shape == (
            (n_targets,) if n_targets > 0 else ()
        )
    else:
        assert cu_model.intercept_ == 0.0
    assert cu_pred.shape == (
        (n_samples, n_targets) if n_targets > 1 else (n_samples,)
    )

    # Check predictions are close
    np.testing.assert_allclose(cu_pred, sk_pred, atol=1e-2)

    # Check fit metadata
    assert isinstance(cu_model.n_iter_, np.ndarray)
    assert cu_model.n_iter_.shape == (n_targets or 1,)
    assert cu_model.solver_ == "lsmr"


def test_ridge_and_least_squares_equal_when_alpha_is_0():
    X, y = make_regression(n_samples=5, n_features=4, random_state=42)

    ridge = cuml.Ridge(alpha=0.0, fit_intercept=False).fit(X, y)
    ols = cuml.LinearRegression(fit_intercept=False).fit(X, y)

    np.testing.assert_allclose(ridge.coef_, ols.coef_)


@pytest.mark.parametrize("train_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("test_dtype", [np.float32, np.float64])
def test_ridge_predict_dtype(train_dtype, test_dtype):
    X, y = make_regression(random_state=42)
    model = cuml.Ridge().fit(X.astype(train_dtype), y.astype(train_dtype))
    out = model.predict(X.astype(test_dtype))
    assert out.dtype == train_dtype


@pytest.mark.parametrize("arr_type", [np.array, cp.array])
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize(
    "solver, n_alpha, n_targets",
    [
        ("eig", 1, 1),
        ("svd", 1, 1),
        ("svd", 1, 3),
        ("svd", 3, 3),
        ("lsmr", 3, 3),
    ],
)
def test_ridge_alpha_array(
    arr_type, fit_intercept, solver, n_alpha, n_targets
):
    X, y = make_regression(
        n_samples=95, n_features=21, n_targets=n_targets, random_state=42
    )
    alpha = np.array([2.0, 50.0, 100.0][:n_alpha])

    sk_model = sklearn.linear_model.Ridge(
        alpha=alpha, fit_intercept=fit_intercept
    )
    sk_model.fit(X, y)
    cu_model = cuml.Ridge(
        solver=solver,
        alpha=arr_type(alpha),
        fit_intercept=fit_intercept,
        tol=1e-6,
    )
    cu_model.fit(X, y)
    atol = 1e-3 if solver == "lsmr" else 1e-4
    np.testing.assert_allclose(cu_model.coef_, sk_model.coef_, atol=atol)
    np.testing.assert_allclose(
        cu_model.intercept_, sk_model.intercept_, atol=atol
    )


def test_ridge_invalid_alpha():
    X, y = make_regression(n_targets=3, random_state=42)

    model = cuml.Ridge(alpha=np.array([2.0, 3.0]))
    with pytest.raises(
        ValueError,
        match="Number of targets and number of penalties do not correspond: 3 != 2",
    ):
        model.fit(X, y)

    model = cuml.Ridge(alpha=-0.5)
    with pytest.raises(
        InvalidParameterError,
        match=r"The 'alpha' parameter of Ridge must be a float in the range \[0.0, inf\)",
    ):
        model.fit(X, y)

    model = cuml.Ridge(alpha=np.array([0.1, 10.0, -2.0]))
    with pytest.raises(ValueError, match="alpha must be non-negative"):
        model.fit(X, y)


def test_eig_solver_invalid():
    model = cuml.Ridge(solver="eig")

    X, y = make_regression(n_features=1, random_state=42)
    with pytest.raises(ValueError, match="doesn't support X with 1 column"):
        model.fit(X, y)

    X, y = make_regression(n_targets=3, random_state=42)
    with pytest.raises(ValueError, match="doesn't support multi-target y"):
        model.fit(X, y)


@pytest.mark.parametrize("solver", ["eig", "svd"])
def test_invalid_sparse_solver(solver):
    X, y = make_regression(random_state=42)
    X = sp.csr_matrix(X)
    model = cuml.Ridge(solver=solver)
    with pytest.raises(
        ValueError, match=f"solver={solver!r} doesn't support sparse X"
    ):
        model.fit(X, y)


def test_solver_auto():
    # 1 feature defaults to svd
    X, y = make_regression(n_features=1, random_state=42)
    model = cuml.Ridge().fit(X, y)
    assert model.solver_ == "svd"

    # multi-target defaults to svd
    X, y = make_regression(n_targets=3, random_state=42)
    model = cuml.Ridge().fit(X, y)
    assert model.solver_ == "svd"

    # sparse defaults to lsmr
    model = cuml.Ridge().fit(sp.csr_matrix(X), y)
    assert model.solver_ == "lsmr"
    model = cuml.Ridge().fit(cp_sp.csr_matrix(cp.asarray(X)), y)
    assert model.solver_ == "lsmr"

    # otherwise defaults to eig
    X, y = make_regression(random_state=42)
    model = cuml.Ridge().fit(X, y)
    assert model.solver_ == "eig"


@pytest.mark.parametrize("solver", ["eig", "svd", "lsmr"])
def test_more_features_than_samples(solver):
    X, y = make_regression(n_features=200, n_samples=100, random_state=42)

    cu_model = cuml.Ridge(solver=solver, tol=1e-6).fit(X, y)
    cu_preds = cu_model.predict(X)

    sk_model = sklearn.linear_model.Ridge().fit(X, y)
    sk_preds = sk_model.predict(X)

    np.testing.assert_allclose(cu_preds, sk_preds, atol=1e-2)


@given(
    solver=st.sampled_from(["svd", "eig", "lsmr"]),
    n_targets=st.integers(min_value=1, max_value=2),
    fit_intercept=st.booleans(),
    weighted=st.booleans(),
)
@example(solver="eig", n_targets=1, fit_intercept=True, weighted=False)
@example(solver="svd", n_targets=1, fit_intercept=True, weighted=False)
@example(solver="svd", n_targets=2, fit_intercept=True, weighted=False)
@example(solver="svd", n_targets=2, fit_intercept=False, weighted=True)
def test_ridge_input_mutation(solver, n_targets, fit_intercept, weighted):
    """Check that `Ridge.fit`:
    - Never mutates y and sample_weight
    - Only sometimes mutates X
    """
    # solver='eig' doesn't support n_targets > 1
    assume(n_targets == 1 or solver != "eig")

    X, y = make_regression(n_targets=n_targets, random_state=42)
    if weighted:
        sample_weight = (
            cp.random.default_rng(42)
            .uniform(0.5, 1, y.shape[0])
            .astype("float32")
        )
        sample_weight_orig = sample_weight.copy()
    else:
        sample_weight = None

    # The eig solver expects fortran-ordered inputs, and will always copy C
    # ordered inputs. Mutation can only happen for F-ordered inputs.
    # The other solvers accept either, but we'll only use C for simplicity.
    X = cp.asarray(X, order="F", dtype="float32")
    y = cp.asarray(y, order="F", dtype="float32")
    X_orig = X.copy()
    y_orig = y.copy()

    params = {"solver": solver, "fit_intercept": fit_intercept}

    # Default never mutates inputs
    cuml.Ridge(**params).fit(X, y, sample_weight=sample_weight)
    cp.testing.assert_array_equal(X, X_orig)
    cp.testing.assert_array_equal(y, y_orig)
    if weighted:
        cp.testing.assert_array_equal(sample_weight, sample_weight_orig)

    cuml.Ridge(copy_X=False, **params).fit(X, y, sample_weight=sample_weight)
    # y and sample_weight are never mutated
    cp.testing.assert_array_equal(y, y_orig)
    if weighted:
        cp.testing.assert_array_equal(sample_weight, sample_weight_orig)
    # The interface doesn't actually care if X is mutated if copy_X=False,
    # but if our solvers stop mutating (and we can avoid a copy) it'd be good
    # to notice. Asserting the current behavior here for now.
    if fit_intercept or weighted:
        assert not cp.array_equal(X, X_orig)
    else:
        # All other options don't mutate
        cp.testing.assert_array_equal(X, X_orig)
