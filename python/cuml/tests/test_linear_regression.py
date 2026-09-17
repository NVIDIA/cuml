# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import cupy as cp
import numpy as np
import pytest
import scipy.sparse
import sklearn.linear_model
from hypothesis import assume, example, given
from hypothesis import strategies as st
from hypothesis import target
from sklearn.linear_model import LinearRegression as skLinearRegression
from sklearn.model_selection import train_test_split

import cuml
from cuml import LinearRegression as cuLinearRegression
from cuml.testing.datasets import (
    is_cuml_compatible_dataset,
    is_sklearn_compatible_dataset,
    make_regression,
    make_regression_dataset,
    regression_datasets,
    small_regression_dataset,
    split_datasets,
    standard_regression_datasets,
)
from cuml.testing.strategies import dataset_dtypes
from cuml.testing.utils import array_difference, array_equal


@given(
    datatype=dataset_dtypes(),
    algorithm=st.sampled_from(["eig", "svd", "lsmr"]),
    nrows=st.integers(min_value=1000, max_value=5000),
    column_info=st.sampled_from([[20, 10], [100, 50]]),
    ntargets=st.integers(min_value=1, max_value=2),
)
@example(
    datatype=np.float32,
    algorithm="eig",
    nrows=1000,
    column_info=[20, 10],
    ntargets=1,
)
@example(
    datatype=np.float64,
    algorithm="svd",
    nrows=5000,
    column_info=[100, 50],
    ntargets=2,
)
@example(
    datatype=np.float32,
    algorithm="lsmr",
    nrows=3000,
    column_info=[50, 20],
    ntargets=3,
)
def test_linear_regression_model(
    datatype, algorithm, nrows, column_info, ntargets
):
    assume(ntargets == 1 or algorithm in ("svd", "lsmr"))

    ncols, n_info = column_info
    X_train, X_test, y_train, y_test = make_regression_dataset(
        datatype, nrows, ncols, n_info, n_targets=ntargets
    )

    # Initialization of cuML's linear regression model
    cuols = cuLinearRegression(fit_intercept=True, algorithm=algorithm)

    # fit and predict cuml linear regression model
    cuols.fit(X_train, y_train)
    cuols_predict = cuols.predict(X_test)

    # sklearn linear regression model initialization, fit and predict
    skols = skLinearRegression(fit_intercept=True)
    skols.fit(X_train, y_train)

    skols_predict = skols.predict(X_test)

    assert array_equal(skols_predict, cuols_predict, 1e-1, with_sign=True)


@given(
    ntargets=st.integers(min_value=1, max_value=2),
    datatype=dataset_dtypes(),
    algorithm=st.sampled_from(["eig", "svd", "qr", "svd-qr", "lsmr"]),
    fit_intercept=st.booleans(),
    distribution=st.sampled_from(["lognormal", "exponential", "uniform"]),
)
@example(
    ntargets=1,
    datatype=np.float32,
    algorithm="eig",
    fit_intercept=True,
    distribution="uniform",
)
@example(
    ntargets=2,
    datatype=np.float64,
    algorithm="svd",
    fit_intercept=False,
    distribution="lognormal",
)
@example(
    ntargets=2,
    datatype=np.float32,
    algorithm="lsmr",
    fit_intercept=True,
    distribution="uniform",
)
def test_weighted_linear_regression(
    ntargets, datatype, algorithm, fit_intercept, distribution
):
    nrows, ncols, n_info = 1000, 20, 10
    max_weight = 10
    noise = 20

    assume(ntargets == 1 or algorithm in ("svd", "lsmr"))

    X_train, X_test, y_train, y_test = make_regression_dataset(
        datatype, nrows, ncols, n_info, noise=noise, n_targets=ntargets
    )

    # set weight per sample to be from 1 to max_weight
    if distribution == "uniform":
        wt = np.random.randint(1, high=max_weight, size=len(X_train))
        wt_test = np.random.randint(1, high=max_weight, size=len(X_test))
    elif distribution == "exponential":
        wt = np.random.exponential(size=len(X_train))
        wt_test = np.random.exponential(size=len(X_test))
    else:
        wt = np.random.lognormal(size=len(X_train))
        wt_test = np.random.lognormal(size=len(X_test))

    # Initialization of cuML's linear regression model
    cuols = cuLinearRegression(
        fit_intercept=fit_intercept, algorithm=algorithm
    )

    # fit and predict cuml linear regression model
    cuols.fit(X_train, y_train, sample_weight=wt)
    cuols_predict = cuols.predict(X_test)

    # sklearn linear regression model initialization, fit and predict
    skols = skLinearRegression(fit_intercept=fit_intercept)
    skols.fit(X_train, y_train, sample_weight=wt)

    skols_predict = skols.predict(X_test)

    assert array_equal(skols_predict, cuols_predict, 1e-1, with_sign=True)

    # Compare weighted scores
    sk_score = skols.score(X_test, y_test, sample_weight=wt_test)
    cu_score = cuols.score(X_test, y_test, sample_weight=wt_test)
    np.testing.assert_almost_equal(cu_score, sk_score)


def test_linear_regression_single_column():
    """Test that linear regression can be run on single column with more than
    46340 rows"""
    model = cuml.LinearRegression()
    model.fit(cp.random.rand(46341)[:, None], cp.random.rand(46341))


# The assumptions required to have this test pass are relatively strong.
# It should be possible to relax assumptions once #4963 is resolved.
# See also: test_linear_regression_model_default_generalized
@given(
    split_datasets(
        standard_regression_datasets(
            n_samples=st.just(1000),
            n_targets=st.integers(1, 10),
        ),
        test_sizes=st.just(0.2),
    )
)
@example(small_regression_dataset(np.float32))
@example(small_regression_dataset(np.float64))
def test_linear_regression_model_default(dataset):
    X_train, X_test, y_train, _ = dataset

    # Filter datasets based on required assumptions
    assume(is_sklearn_compatible_dataset(X_train, X_test, y_train))
    assume(is_cuml_compatible_dataset(X_train, X_test, y_train))

    # Initialization of cuML's linear regression model
    cuols = cuLinearRegression()

    # fit and predict cuml linear regression model
    cuols.fit(X_train, y_train)
    cuols_predict = cuols.predict(X_test)

    # sklearn linear regression model initialization and fit
    skols = skLinearRegression()
    skols.fit(X_train, y_train)

    skols_predict = skols.predict(X_test)

    target(float(array_difference(skols_predict, cuols_predict)))
    assert array_equal(skols_predict, cuols_predict, 1e-1, with_sign=True)


# TODO: Replace test_linear_regression_model_default with this test once #4963
# is resolved.
@pytest.mark.skip(reason="https://github.com/NVIDIA/cuml/issues/4963")
@given(split_datasets(regression_datasets()))
@example(small_regression_dataset(np.float32))
@example(small_regression_dataset(np.float64))
def test_linear_regression_model_default_generalized(dataset):
    X_train, X_test, y_train, _ = dataset

    # Filter datasets based on required assumptions
    assume(is_sklearn_compatible_dataset(X_train, X_test, y_train))
    assume(is_cuml_compatible_dataset(X_train, X_test, y_train))

    # Initialization of cuML's linear regression model
    cuols = cuLinearRegression()

    # fit and predict cuml linear regression model
    cuols.fit(X_train, y_train)
    cuols_predict = cuols.predict(X_test)

    # sklearn linear regression model initialization and fit
    skols = skLinearRegression()
    skols.fit(X_train, y_train)

    skols_predict = skols.predict(X_test)

    target(float(array_difference(skols_predict, cuols_predict)))
    assert array_equal(skols_predict, cuols_predict, 1e-1, with_sign=True)


@given(train_dtype=dataset_dtypes(), test_dtype=dataset_dtypes())
@example(train_dtype=np.float32, test_dtype=np.float32)
@example(train_dtype=np.float32, test_dtype=np.float64)
@example(train_dtype=np.float64, test_dtype=np.float32)
@example(train_dtype=np.float64, test_dtype=np.float64)
def test_linreg_predict_mixed_dtypes(train_dtype, test_dtype):
    X, y = make_regression(
        n_samples=50, n_features=10, n_informative=5, random_state=0
    )
    X = X.astype(train_dtype)
    y = y.astype(train_dtype)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )

    clf = cuLinearRegression()
    clf.fit(X_train, y_train)
    clf.predict(X_test.astype(test_dtype))


@pytest.mark.filterwarnings("ignore:Changing solver.*:UserWarning")
@given(
    algo=st.sampled_from(["eig", "qr", "svd", "svd-qr", "lsmr"]),
    n_targets=st.integers(min_value=1, max_value=2),
    fit_intercept=st.booleans(),
    weighted=st.booleans(),
)
@example(algo="eig", n_targets=1, fit_intercept=True, weighted=False)
@example(algo="qr", n_targets=1, fit_intercept=True, weighted=False)
@example(algo="svd-qr", n_targets=1, fit_intercept=True, weighted=False)
@example(algo="svd", n_targets=1, fit_intercept=True, weighted=False)
@example(algo="svd", n_targets=2, fit_intercept=False, weighted=True)
@example(algo="lsmr", n_targets=1, fit_intercept=True, weighted=False)
@example(algo="lsmr", n_targets=2, fit_intercept=False, weighted=False)
def test_linear_regression_input_mutation(
    algo, n_targets, fit_intercept, weighted
):
    """Check that `LinearRegression.fit`:
    - Never mutates y and sample_weight
    - Only sometimes mutates X
    """
    # Only "svd" and "lsmr" support n_targets > 1. While we do fallback
    # automatically (with a warning), there's no need to have hypothesis
    # explore those cases.
    assume(n_targets == 1 or algo in ("lsmr", "svd"))

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

    # The solvers expected fortran-ordered inputs, and will always copy C
    # ordered inputs. Mutation can only happen for F-ordered inputs.
    X = cp.asarray(X, order="F", dtype="float32")
    y = cp.asarray(y, order="F", dtype="float32")
    X_orig = X.copy()
    y_orig = y.copy()

    params = {"algorithm": algo, "fit_intercept": fit_intercept}

    # Default never mutates inputs
    cuml.LinearRegression(**params).fit(X, y, sample_weight=sample_weight)
    cp.testing.assert_allclose(X, X_orig)
    cp.testing.assert_allclose(y, y_orig)
    if weighted:
        cp.testing.assert_allclose(sample_weight, sample_weight_orig)

    cuml.LinearRegression(copy_X=False, **params).fit(
        X, y, sample_weight=sample_weight
    )
    # y and sample_weight are never mutated
    cp.testing.assert_allclose(y, y_orig)
    if weighted:
        cp.testing.assert_allclose(sample_weight, sample_weight_orig)
    # The interface doesn't actually care if X is mutated if copy_X=False,
    # but if our solvers stop mutating (and we can avoid a copy) it'd be good
    # to notice. Asserting the current behavior here for now.
    if n_targets == 1 and algo in ["eig", "qr", "svd", "svd-qr"]:
        # `eig` sometimes mutates and sometimes doesn't, the others always do
        if algo != "eig":
            assert not cp.array_equal(X, X_orig)
    elif algo in ("svd", "lsmr") and (fit_intercept or weighted):
        # The cupy solvers also mutates in this case
        assert not cp.array_equal(X, X_orig)
    else:
        # All other options don't mutate
        cp.testing.assert_array_equal(X, X_orig)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("n_targets", [0, 1, 3])
def test_linear_regression_sparse(dtype, fit_intercept, weighted, n_targets):
    n_samples, n_features = 3000, 500
    rng = np.random.default_rng(42)
    coef = rng.random((n_features, n_targets) if n_targets else n_features)
    if fit_intercept:
        intercept = rng.uniform(-10, 10, size=n_targets or ())
    else:
        intercept = 0.0
    X = scipy.sparse.rand(
        n_samples, n_features, density=0.2, random_state=42, dtype=dtype
    )
    y = (X.dot(coef) + intercept).astype(dtype)

    if weighted:
        sample_weight = rng.uniform(0.5, 1, size=n_samples).astype(dtype)
    else:
        sample_weight = None

    cu_model = cuml.LinearRegression(fit_intercept=fit_intercept)
    cu_model.fit(X, y, sample_weight=sample_weight)
    cu_pred = cu_model.predict(X)

    sk_model = sklearn.linear_model.LinearRegression(
        fit_intercept=fit_intercept
    )
    sk_model.fit(X, y, sample_weight=sample_weight)
    sk_pred = sk_model.predict(X).reshape(cu_pred.shape)

    # Check shapes and dtypes
    assert cu_model.coef_.dtype == dtype
    assert cu_model.coef_.shape == (
        (n_targets, n_features) if n_targets >= 1 else (n_features,)
    )
    if fit_intercept:
        assert cu_model.intercept_.dtype == dtype
        assert cu_model.intercept_.shape == (
            (n_targets,) if n_targets > 0 else ()
        )
    else:
        assert cu_model.intercept_ == 0.0
    assert cu_pred.shape == (
        (n_samples, n_targets) if n_targets >= 1 else (n_samples,)
    )

    # Check predictions are close
    np.testing.assert_allclose(cu_pred, sk_pred, atol=1e-2)
