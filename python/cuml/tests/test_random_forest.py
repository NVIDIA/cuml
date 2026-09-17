# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import json
import math
import os
import random
import warnings

import cudf
import cupy as cp
import numpy as np
import pytest
import treelite
from cudf.pandas import LOADED as cudf_pandas_active
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_iris,
    make_classification,
    make_regression,
)
from sklearn.ensemble import RandomForestClassifier as skrfc
from sklearn.ensemble import RandomForestRegressor as skrfr
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    mean_tweedie_deviance,
)
from sklearn.metrics import r2_score as sk_r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

import cuml
from cuml.ensemble import RandomForestClassifier as curfc
from cuml.ensemble import RandomForestRegressor as curfr
from cuml.ensemble.randomforest_common import compute_max_features
from cuml.metrics import r2_score
from cuml.testing.utils import quality_param, stress_param, unit_param


@pytest.fixture(
    scope="session",
    params=[
        unit_param({"n_samples": 350, "n_features": 20, "n_informative": 10}),
        quality_param(
            {"n_samples": 5000, "n_features": 200, "n_informative": 80}
        ),
        stress_param(
            {"n_samples": 500000, "n_features": 400, "n_informative": 180}
        ),
    ],
)
def small_clf(request):
    X, y = make_classification(
        n_samples=request.param["n_samples"],
        n_features=request.param["n_features"],
        n_clusters_per_class=1,
        n_informative=request.param["n_informative"],
        random_state=123,
        n_classes=2,
    )
    return X, y


@pytest.fixture(
    scope="session",
    params=[
        unit_param({"n_samples": 350, "n_features": 30, "n_informative": 15}),
        quality_param(
            {"n_samples": 5000, "n_features": 200, "n_informative": 80}
        ),
        stress_param(
            {"n_samples": 500000, "n_features": 400, "n_informative": 180}
        ),
    ],
)
def mclass_clf(request):
    X, y = make_classification(
        n_samples=request.param["n_samples"],
        n_features=request.param["n_features"],
        n_clusters_per_class=1,
        n_informative=request.param["n_informative"],
        random_state=123,
        n_classes=10,
    )
    return X, y


@pytest.fixture(
    scope="session",
    params=[
        unit_param({"n_samples": 500, "n_features": 20, "n_informative": 10}),
        quality_param(
            {"n_samples": 5000, "n_features": 200, "n_informative": 50}
        ),
        stress_param(
            {"n_samples": 500000, "n_features": 400, "n_informative": 100}
        ),
    ],
)
def large_clf(request):
    X, y = make_classification(
        n_samples=request.param["n_samples"],
        n_features=request.param["n_features"],
        n_clusters_per_class=1,
        n_informative=request.param["n_informative"],
        random_state=123,
        n_classes=2,
    )
    return X, y


@pytest.fixture(
    scope="session",
    params=[
        unit_param({"n_samples": 1500, "n_features": 20, "n_informative": 10}),
        quality_param(
            {"n_samples": 12000, "n_features": 200, "n_informative": 100}
        ),
        stress_param(
            {"n_samples": 500000, "n_features": 500, "n_informative": 350}
        ),
    ],
)
def large_reg(request):
    X, y = make_regression(
        n_samples=request.param["n_samples"],
        n_features=request.param["n_features"],
        n_informative=request.param["n_informative"],
        random_state=123,
    )
    return X, y


special_reg_params = [
    unit_param(
        {
            "mode": "unit",
            "n_samples": 500,
            "n_features": 20,
            "n_informative": 10,
        }
    ),
    quality_param(
        {
            "mode": "quality",
            "n_samples": 500,
            "n_features": 20,
            "n_informative": 10,
        }
    ),
    quality_param({"mode": "quality", "n_features": 200, "n_informative": 50}),
    stress_param(
        {
            "mode": "stress",
            "n_samples": 500,
            "n_features": 20,
            "n_informative": 10,
        }
    ),
    stress_param({"mode": "stress", "n_features": 200, "n_informative": 50}),
    stress_param(
        {
            "mode": "stress",
            "n_samples": 1000,
            "n_features": 400,
            "n_informative": 100,
        }
    ),
]


@pytest.fixture(scope="session", params=special_reg_params)
def special_reg(request):
    if request.param["mode"] == "quality":
        X, y = fetch_california_housing(return_X_y=True)
    else:
        X, y = make_regression(
            n_samples=request.param["n_samples"],
            n_features=request.param["n_features"],
            n_informative=request.param["n_informative"],
            random_state=123,
        )
    return X, y


def test_default_parameters():
    X = np.array([[1.0, 2.0]], dtype=np.float32)
    y_reg = np.array([1.0], dtype=np.float32)
    y_clf = np.array([1], dtype=np.int32)

    reg = curfr()
    reg_params = reg.get_params()
    reg.fit(X, y_reg)

    clf = curfc()
    clf_params = clf.get_params()
    clf.fit(X, y_clf)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        curfr(max_depth=16).fit(X, y_reg).get_params()
        curfr(max_depth=None).fit(X, y_reg).get_params()
        curfc(max_depth=16).fit(X, y_clf).get_params()
        curfc(max_depth=None).fit(X, y_clf).get_params()

    # Different default max_features
    assert reg_params["max_features"] == 1.0
    assert clf_params["max_features"] == "sqrt"

    # Different default split_criterion
    assert reg_params["split_criterion"] == "mse"
    assert clf_params["split_criterion"] == "gini"

    assert reg_params["max_depth"] is None
    assert clf_params["max_depth"] is None

    # Drop differing params
    for name in ["max_features", "split_criterion"]:
        reg_params.pop(name)
        clf_params.pop(name)
    clf_params.pop("class_weight")

    # The rest are the same
    assert reg_params == clf_params


@pytest.mark.parametrize("estimator", [curfc, curfr])
@pytest.mark.parametrize(
    "n_streams,error_type",
    [
        (0, ValueError),
        (-1, ValueError),
        (1.5, TypeError),
        ("1", TypeError),
        (None, TypeError),
        (True, TypeError),
    ],
)
def test_rf_invalid_n_streams(estimator, n_streams, error_type):
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([0, 1], dtype=np.int32)
    if estimator is curfr:
        y = y.astype(np.float32)

    with pytest.raises(
        error_type,
        match="n_streams must be a positive integer",
    ):
        estimator(n_streams=n_streams).fit(X, y)


def _zero_weight_two_cluster_data(datatype, n_zero=8, n_one=8):
    zero_weight_X = np.linspace(-3.0, -1.0, n_zero, dtype=datatype).reshape(
        -1, 1
    )
    one_weight_X = np.linspace(1.0, 3.0, n_one, dtype=datatype).reshape(-1, 1)
    X = np.vstack([zero_weight_X, one_weight_X])
    y = np.array(
        [0] * len(zero_weight_X) + [1] * len(one_weight_X),
        dtype=np.int32,
    )
    sample_weight = y.astype(np.float64)
    probe = np.linspace(-3.5, 3.5, 17, dtype=datatype).reshape(-1, 1)
    return X, y, sample_weight, probe


def _sample_weight_rf_params(X, bootstrap):
    return dict(
        n_estimators=3,
        bootstrap=bootstrap,
        max_depth=3,
        max_features=1.0,
        n_bins=X.shape[0],
        min_samples_leaf=1,
        min_samples_split=2,
        random_state=0,
        n_streams=1,
    )


def _cuml_preds(model, X):
    return cp.asnumpy(cp.asarray(model.predict(X)))


@pytest.mark.parametrize(
    "estimator", [curfc, curfr], ids=["classifier", "regressor"]
)
@pytest.mark.parametrize("input_type", ["numpy", "cupy"])
@pytest.mark.parametrize("datatype", [np.float32, np.float64])
def test_rf_fit_input_order_parity(datatype, input_type, estimator):
    X, y = make_classification(
        n_samples=128,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        n_clusters_per_class=1,
        n_classes=3,
        class_sep=2.0,
        random_state=7,
    )
    xp = cp if input_type == "cupy" else np
    X_c = xp.asarray(X, dtype=datatype, order="C")
    X_f = xp.asarray(X, dtype=datatype, order="F")

    params = dict(
        n_estimators=3,
        bootstrap=False,
        max_depth=4,
        max_features=1.0,
        n_bins=16,
        random_state=11,
        n_streams=1,
    )
    c_model = estimator(**params).fit(X_c, y)
    f_model = estimator(**params).fit(X_f, y)

    X_pred = np.array(X, dtype=datatype, order="C")
    c_preds = _cuml_preds(c_model, X_pred)
    f_preds = _cuml_preds(f_model, X_pred)

    np.testing.assert_allclose(c_preds, f_preds, rtol=1e-6, atol=1e-6)
    assert c_model.as_treelite().num_feature == X.shape[1]


def _sklearn_fit_params(cuml_model):
    params = cuml_model._params_to_cpu()
    if params["max_samples"] == 1.0:
        params["max_samples"] = None
    return params


@pytest.mark.parametrize("bootstrap", [False, True])
@pytest.mark.parametrize("datatype", [np.float32, np.float64])
def test_rf_classifier_sample_weight_zero_distribution_matches_sklearn(
    datatype, bootstrap
):
    X, y, sample_weight, probe = _zero_weight_two_cluster_data(datatype)
    cuml_model = curfc(**_sample_weight_rf_params(X, bootstrap))
    sk_model = skrfc(**_sklearn_fit_params(cuml_model))

    cuml_model.fit(X, y, sample_weight=sample_weight)
    sk_model.fit(X, y, sample_weight=sample_weight)

    cuml_preds = _cuml_preds(cuml_model, probe)
    np.testing.assert_array_equal(cuml_preds, sk_model.predict(probe))
    np.testing.assert_array_equal(
        cuml_preds, np.ones(probe.shape[0], dtype=np.int32)
    )

    score = cuml_model.score(X, y, sample_weight=sample_weight)
    assert score == pytest.approx(
        accuracy_score(y, sk_model.predict(X), sample_weight=sample_weight)
    )


@pytest.mark.parametrize("bootstrap", [False, True])
@pytest.mark.parametrize("datatype", [np.float32, np.float64])
def test_rf_regressor_sample_weight_zero_distribution_matches_sklearn(
    datatype, bootstrap
):
    X, y, sample_weight, probe = _zero_weight_two_cluster_data(datatype)
    cuml_model = curfr(**_sample_weight_rf_params(X, bootstrap))
    sk_model = skrfr(**_sklearn_fit_params(cuml_model))

    cuml_model.fit(X, y, sample_weight=sample_weight)
    sk_model.fit(X, y, sample_weight=sample_weight)

    cuml_preds = _cuml_preds(cuml_model, probe)
    np.testing.assert_allclose(
        cuml_preds, sk_model.predict(probe), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        cuml_preds,
        np.ones(probe.shape[0]),
        rtol=1e-6,
        atol=1e-6,
    )

    score = cuml_model.score(X, y, sample_weight=sample_weight)
    assert score == pytest.approx(
        sk_r2_score(y, sk_model.predict(X), sample_weight=sample_weight),
        abs=1e-6,
    )


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
def test_rf_regressor_sample_weight_min_samples_leaf_matches_sklearn(datatype):
    X = np.array([[0.0], [1.0], [2.0]], dtype=datatype)
    y = np.array([0.0, 0.0, 10.0], dtype=datatype)
    sample_weight = np.array([1.0, 1.0, 1000.0], dtype=np.float64)

    cuml_model = curfr(
        n_estimators=1,
        bootstrap=False,
        max_depth=2,
        max_features=1.0,
        n_bins=3,
        min_samples_leaf=2,
        min_samples_split=2,
        random_state=0,
        n_streams=1,
    )
    sk_model = skrfr(**_sklearn_fit_params(cuml_model))

    cuml_model.fit(X, y, sample_weight=sample_weight)
    sk_model.fit(X, y, sample_weight=sample_weight)

    np.testing.assert_allclose(
        _cuml_preds(cuml_model, X), sk_model.predict(X), rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize(
    "class_weight,match",
    [
        pytest.param("not-balanced", "class_weight", id="invalid-string"),
        pytest.param(
            {0: 1.0, 2: 1.0}, "not in class_weight", id="missing-class"
        ),
    ],
)
def test_rf_classifier_invalid_class_weight_raises(class_weight, match):
    X = np.array([[0.0], [1.0]], dtype=np.float32)
    y = np.array([0, 1], dtype=np.int32)
    clf = curfc(
        n_estimators=1,
        max_depth=1,
        n_bins=2,
        n_streams=1,
        class_weight=class_weight,
    )

    with pytest.raises(ValueError, match=match):
        clf.fit(X, y)


@pytest.mark.parametrize(
    "class_weight",
    [
        pytest.param({0: 0.0, 1: 1.0}, id="zero-class-zero"),
        pytest.param("balanced", id="balanced"),
    ],
)
@pytest.mark.parametrize("bootstrap", [False, True])
@pytest.mark.parametrize("datatype", [np.float32, np.float64])
def test_rf_classifier_class_weight_matches_sklearn(
    datatype, bootstrap, class_weight
):
    n_zero, n_one = (12, 4) if class_weight == "balanced" else (8, 8)
    X, y, _, probe = _zero_weight_two_cluster_data(
        datatype, n_zero=n_zero, n_one=n_one
    )
    sample_weight = compute_sample_weight(class_weight, y).astype(np.float64)
    params = _sample_weight_rf_params(X, bootstrap)

    class_weight_model = curfc(**params, class_weight=class_weight)
    sample_weight_model = curfc(**params)
    sk_model = skrfc(**_sklearn_fit_params(class_weight_model))

    class_weight_model.fit(X, y)
    sample_weight_model.fit(X, y, sample_weight=sample_weight)
    sk_model.fit(X, y)

    class_weight_preds = _cuml_preds(class_weight_model, probe)
    np.testing.assert_array_equal(
        class_weight_preds, _cuml_preds(sample_weight_model, probe)
    )

    if class_weight == "balanced":
        assert np.unique(sample_weight).size == 2
        np.testing.assert_array_equal(
            _cuml_preds(class_weight_model, X), sk_model.predict(X)
        )
    else:
        np.testing.assert_array_equal(
            class_weight_preds, sk_model.predict(probe)
        )
        np.testing.assert_array_equal(
            class_weight_preds, np.ones(probe.shape[0], dtype=np.int32)
        )


@pytest.mark.parametrize("max_depth", [2, 4])
@pytest.mark.parametrize(
    "split_criterion", ["poisson", "gamma", "inverse_gaussian"]
)
def test_tweedie_convergence(max_depth, split_criterion):
    np.random.seed(33)
    bootstrap = None
    max_features = 1.0
    n_estimators = 1
    min_impurity_decrease = 1e-5
    n_datapoints = 1000
    tweedie = {
        "poisson": {"power": 1, "gen": np.random.poisson, "args": [0.01]},
        "gamma": {"power": 2, "gen": np.random.gamma, "args": [2.0]},
        "inverse_gaussian": {
            "power": 3,
            "gen": np.random.wald,
            "args": [0.1, 2.0],
        },
    }
    # generating random dataset with tweedie distribution
    X = np.random.random((n_datapoints, 4)).astype(np.float32)
    y = tweedie[split_criterion]["gen"](
        *tweedie[split_criterion]["args"], size=n_datapoints
    ).astype(np.float32)

    tweedie_preds = (
        curfr(
            split_criterion=split_criterion,
            max_depth=max_depth,
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
        )
        .fit(X, y)
        .predict(X)
    )
    mse_preds = (
        curfr(
            split_criterion=2,
            max_depth=max_depth,
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
        )
        .fit(X, y)
        .predict(X)
    ).squeeze()
    # y should not be non-positive for mean_poisson_deviance
    mask = mse_preds > 0
    mse_tweedie_deviance = mean_tweedie_deviance(
        y[mask], mse_preds[mask], power=tweedie[split_criterion]["power"]
    )
    tweedie_tweedie_deviance = mean_tweedie_deviance(
        y[mask], tweedie_preds[mask], power=tweedie[split_criterion]["power"]
    )

    # model trained on tweedie data with tweedie criterion must perform
    # no worse on tweedie loss, allowing tiny numeric variation.
    deviance_tolerance = 1e-5
    assert (
        mse_tweedie_deviance + deviance_tolerance >= tweedie_tweedie_deviance
    )


@pytest.mark.parametrize(
    "max_samples", [unit_param(1.0), quality_param(0.90), stress_param(0.95)]
)
@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("max_features", [1.0, "log2", "sqrt"])
@pytest.mark.skipif(
    cudf_pandas_active,
    reason="cudf.pandas causes sklearn RF estimators crashes sometimes. "
    "Issue: https://github.com/NVIDIA/cuml/issues/5991",
)
def test_rf_classification(small_clf, datatype, max_samples, max_features):
    X, y = small_clf
    X = X.astype(datatype)
    y = y.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )
    # Initialize, fit and predict using cuML's
    # random forest classification model
    cuml_model = curfc(
        max_features=max_features,
        max_samples=max_samples,
        n_bins=16,
        split_criterion=0,
        min_samples_leaf=2,
        random_state=123,
        n_streams=1,
        n_estimators=40,
        max_leaves=-1,
    )
    cuml_model.fit(X_train, y_train)

    preds = cuml_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    if X.shape[0] < 500000:
        sk_model = skrfc(
            n_estimators=40,
            min_samples_split=2,
            max_features=max_features,
            random_state=10,
        )
        sk_model.fit(X_train, y_train)
        sk_preds = sk_model.predict(X_test)
        sk_acc = accuracy_score(y_test, sk_preds)

        # Observed: mean=0.026, range=[0.000, 0.043], stderr=0.002
        assert acc >= (sk_acc - 0.07)


@pytest.mark.parametrize(
    "max_samples", [unit_param(1.0), quality_param(0.90), stress_param(0.95)]
)
@pytest.mark.parametrize("datatype", [np.float32, np.float64])
def test_rf_classification_unorder(
    small_clf, datatype, max_samples, max_features=1, a=2, b=5
):
    X, y = small_clf
    X = X.astype(datatype)
    y = y.astype(np.int32)
    # affine transformation
    y = a * y + b
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )

    # Initialize, fit and predict using cuML's
    # random forest classification model
    cuml_model = curfc(
        max_features=max_features,
        max_samples=max_samples,
        n_bins=16,
        split_criterion=0,
        min_samples_leaf=2,
        random_state=123,
        n_streams=1,
        n_estimators=40,
        max_leaves=-1,
    )
    cuml_model.fit(X_train, y_train)

    preds = cuml_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    if X.shape[0] < 500000:
        sk_model = skrfc(
            n_estimators=40,
            min_samples_split=2,
            max_features=max_features,
            random_state=10,
        )
        sk_model.fit(X_train, y_train)
        sk_preds = sk_model.predict(X_test)
        sk_acc = accuracy_score(y_test, sk_preds)
        # Increased tolerance to 0.10 to account for RNG variance from bias fix
        # Variance analysis (1000 runs): mean=0.9140, range=[0.8429,0.9714]
        # Worst-case diff from sklearn baseline (0.9143): ~0.071
        assert acc >= (sk_acc - 0.10)


@pytest.mark.parametrize(
    "max_samples", [unit_param(1.0), quality_param(0.90), stress_param(0.95)]
)
@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "max_features,n_bins",
    [
        (1.0, 16),
        (1.0, 11),
        ("log2", 100),
        ("sqrt", 100),
        (1.0, 17),
        (1.0, 32),
    ],
)
@pytest.mark.skipif(
    cudf_pandas_active,
    reason="cudf.pandas causes sklearn RF estimators crashes sometimes. "
    "Issue: https://github.com/NVIDIA/cuml/issues/5991",
)
def test_rf_regression(
    special_reg, datatype, max_features, max_samples, n_bins
):
    X, y = special_reg
    X = X.astype(datatype)
    y = y.astype(datatype)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )

    # Initialize and fit using cuML's random forest regression model
    cuml_model = curfr(
        max_features=max_features,
        max_samples=max_samples,
        n_bins=n_bins,
        split_criterion=2,
        min_samples_leaf=2,
        random_state=123,
        n_streams=1,
        n_estimators=50,
        max_leaves=-1,
    )
    cuml_model.fit(X_train, y_train)
    preds = cuml_model.predict(X_test)

    r2 = r2_score(y_test, preds)
    # Initialize, fit and predict using sklearn's random forest regression model
    if X.shape[0] < 1000:  # mode != "stress"
        sk_model = skrfr(
            n_estimators=50,
            min_samples_split=2,
            max_features=max_features,
            random_state=10,
        )
        sk_model.fit(X_train, y_train)
        sk_preds = sk_model.predict(X_test)
        sk_r2 = r2_score(y_test, sk_preds)
        # Observed: mean=0.020, range=[0.001, 0.037], stderr=0.002
        assert r2 >= (sk_r2 - 0.07)


@pytest.mark.skipif(
    cudf_pandas_active,
    reason="cudf.pandas causes sklearn RF estimators crashes sometimes. "
    "Issue: https://github.com/NVIDIA/cuml/issues/5991",
)
@pytest.mark.parametrize("datatype", [np.float32, np.float64])
def test_rf_classification_seed(small_clf, datatype):
    X, y = small_clf
    X = X.astype(datatype)
    y = y.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )

    for i in range(8):
        seed = random.randint(100, 10**5)

        cu_class = curfc(random_state=seed, n_streams=1)
        cu_class.fit(X_train, y_train)
        preds_orig = cu_class.predict(X_test)
        acc_orig = accuracy_score(y_test, preds_orig)

        cu_class2 = curfc(random_state=seed, n_streams=1)
        cu_class2.fit(X_train, y_train)
        preds_rerun = cu_class2.predict(X_test)
        acc_rerun = accuracy_score(y_test, preds_rerun)

        assert acc_orig == acc_rerun
        assert (preds_orig == preds_rerun).all()


@pytest.mark.parametrize(
    "datatype", [(np.float64, np.float32), (np.float32, np.float64)]
)
@pytest.mark.filterwarnings("ignore:To use pickling(.*)::cuml[.*]")
@pytest.mark.skipif(
    cudf_pandas_active,
    reason="cudf.pandas causes sklearn RF estimators crashes sometimes. "
    "Issue: https://github.com/NVIDIA/cuml/issues/5991",
)
def test_rf_classification_fit_and_predict_dtypes_differ(small_clf, datatype):
    X, y = small_clf
    X = X.astype(datatype[0])
    y = y.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )
    X_test = X_test.astype(datatype[1])

    cuml_model = curfc()
    cuml_model.fit(X_train, y_train)

    preds = cuml_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    if X.shape[0] < 500000:
        sk_model = skrfc(random_state=10)
        sk_model.fit(X_train, y_train)
        sk_preds = sk_model.predict(X_test)
        sk_acc = accuracy_score(y_test, sk_preds)
        # Observed: mean=0.043, range=[0.043, 0.043], stderr=0.000
        assert acc >= (sk_acc - 0.10)


@pytest.mark.parametrize(
    "datatype", [(np.float64, np.float32), (np.float32, np.float64)]
)
@pytest.mark.filterwarnings("ignore:To use pickling(.*)::cuml[.*]")
@pytest.mark.skipif(
    cudf_pandas_active,
    reason="cudf.pandas causes sklearn RF estimators crashes sometimes. "
    "Issue: https://github.com/NVIDIA/cuml/issues/5991",
)
def test_rf_regression_fit_and_predict_dtypes_differ(large_reg, datatype):
    X, y = large_reg
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )
    X_train = X_train.astype(datatype[0])
    y_train = y_train.astype(datatype[0])
    X_test = X_test.astype(datatype[1])
    y_test = y_test.astype(datatype[1])

    cuml_model = curfr()
    cuml_model.fit(X_train, y_train)
    preds = cuml_model.predict(X_test)
    r2 = r2_score(y_test, preds)
    if X.shape[0] < 500000:
        sk_model = skrfr(random_state=10)
        sk_model.fit(X_train, y_train)
        sk_preds = sk_model.predict(X_test)
        sk_r2 = r2_score(y_test, sk_preds)
        # Observed: mean=-0.004, range=[-0.005, -0.003], stderr=0.000
        # cuML outperforms sklearn on this test
        assert r2 >= (sk_r2 - 0.09)


def check_predict_proba(test_proba, baseline_proba, y_test, rel_err):
    y_proba = np.zeros(np.shape(baseline_proba))
    for count, _class in enumerate(y_test):
        y_proba[count, _class] = 1
    baseline_mse = mean_squared_error(y_proba, baseline_proba)
    test_mse = mean_squared_error(y_proba, test_proba)
    # using relative error is more stable when changing decision tree
    # parameters, column or class count
    assert test_mse <= baseline_mse * (1.0 + rel_err)


def rf_classification(
    datatype, array_type, max_features, max_samples, fixture
):
    X, y = fixture
    X = X.astype(datatype[0])
    y = y.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )
    X_test = X_test.astype(datatype[1])

    # Initialize, fit and predict using cuML's
    # random forest classification model
    cuml_model = curfc(
        max_features=max_features,
        max_samples=max_samples,
        n_bins=16,
        split_criterion=0,
        min_samples_leaf=2,
        random_state=999,
        n_estimators=40,
        max_leaves=-1,
        n_streams=1,
    )
    if array_type == "dataframe":
        X_train_df = cudf.DataFrame(X_train)
        y_train_df = cudf.Series(y_train)
        X_test_df = cudf.DataFrame(X_test)
        cuml_model.fit(X_train_df, y_train_df)
        cu_proba_gpu = cuml_model.predict_proba(X_test_df).to_numpy()
        cu_preds_gpu = cuml_model.predict(X_test_df).to_numpy()
    else:
        cuml_model.fit(X_train, y_train)
        cu_proba_gpu = cuml_model.predict_proba(X_test)
        cu_preds_gpu = cuml_model.predict(X_test)
    np.testing.assert_array_equal(
        cu_preds_gpu, np.argmax(cu_proba_gpu, axis=1)
    )
    cu_acc_gpu = accuracy_score(y_test, cu_preds_gpu)

    # sklearn random forest classification model
    # initialization, fit and predict
    if y.size < 500000:
        sk_model = skrfc(
            n_estimators=40,
            min_samples_split=2,
            max_features=max_features,
            random_state=10,
        )
        sk_model.fit(X_train, y_train)
        sk_preds = sk_model.predict(X_test)
        sk_acc = accuracy_score(y_test, sk_preds)
        sk_proba = sk_model.predict_proba(X_test)
        # Observed: mean=0.024, range=[-0.014, 0.043], stderr=0.002
        assert cu_acc_gpu >= (sk_acc - 0.08)
        check_predict_proba(cu_proba_gpu, sk_proba, y_test, 0.12)


@pytest.mark.parametrize("datatype", [(np.float32, np.float64)])
@pytest.mark.parametrize("array_type", ["dataframe", "numpy"])
@pytest.mark.skipif(
    cudf_pandas_active,
    reason="cudf.pandas causes sklearn RF estimators crashes sometimes. "
    "Issue: https://github.com/NVIDIA/cuml/issues/5991",
)
def test_rf_classification_multi_class(mclass_clf, datatype, array_type):
    rf_classification(datatype, array_type, 1.0, 1.0, mclass_clf)


@pytest.mark.parametrize("datatype", [(np.float32, np.float64)])
@pytest.mark.parametrize("max_samples", [unit_param(1.0), stress_param(0.95)])
@pytest.mark.parametrize("max_features", [1.0, "log2", "sqrt"])
@pytest.mark.skipif(
    cudf_pandas_active,
    reason="cudf.pandas causes sklearn RF estimators crashes sometimes. "
    "Issue: https://github.com/NVIDIA/cuml/issues/5991",
)
def test_rf_classification_proba(
    small_clf, datatype, max_samples, max_features
):
    rf_classification(datatype, "numpy", max_features, max_samples, small_clf)


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "nvforest_layout", ["depth_first", "breadth_first", "layered"]
)
@pytest.mark.skipif(
    cudf_pandas_active,
    reason="cudf.pandas causes sklearn RF estimators crashes sometimes. "
    "Issue: https://github.com/NVIDIA/cuml/issues/5991",
)
def test_rf_classification_sparse(small_clf, datatype, nvforest_layout):
    num_trees = 50

    X, y = small_clf
    X = X.astype(datatype)
    y = y.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )

    # Initialize, fit and predict using cuML's
    # random forest classification model
    cuml_model = curfc(
        n_bins=16,
        split_criterion=0,
        min_samples_leaf=2,
        random_state=123,
        n_streams=1,
        n_estimators=num_trees,
        max_leaves=-1,
        max_depth=40,
    )
    cuml_model.fit(X_train, y_train)
    preds = cuml_model.predict(X_test, layout=nvforest_layout)
    acc = accuracy_score(y_test, preds)
    np.testing.assert_almost_equal(acc, cuml_model.score(X_test, y_test))

    nvforest_model = cuml_model.as_nvforest(layout=nvforest_layout)

    with cuml.using_output_type("numpy"):
        nvforest_model_preds = cp.asnumpy(nvforest_model.predict(X_test))
        nvforest_model_acc = accuracy_score(y_test, nvforest_model_preds)
        assert acc == nvforest_model_acc

    tl_model = cuml_model.as_treelite()
    assert num_trees == tl_model.num_tree
    assert X.shape[1] == tl_model.num_feature

    if X.shape[0] < 500000:
        sk_model = skrfc(
            n_estimators=50,
            max_depth=40,
            min_samples_split=2,
            random_state=10,
        )
        sk_model.fit(X_train, y_train)
        sk_preds = sk_model.predict(X_test)
        sk_acc = accuracy_score(y_test, sk_preds)
        # Observed: mean=0.000, range=[0.000, 0.000], stderr=0.000
        # Perfect match with sklearn on sparse layout tests
        assert acc >= (sk_acc - 0.07)


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "nvforest_layout", ["depth_first", "breadth_first", "layered"]
)
@pytest.mark.skipif(
    cudf_pandas_active,
    reason="cudf.pandas causes sklearn RF estimators crashes sometimes. "
    "Issue: https://github.com/NVIDIA/cuml/issues/5991",
)
def test_rf_regression_sparse(special_reg, datatype, nvforest_layout):
    num_trees = 50

    X, y = special_reg
    X = X.astype(datatype)
    y = y.astype(datatype)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )

    # Initialize and fit using cuML's random forest regression model
    cuml_model = curfr(
        n_bins=16,
        split_criterion=2,
        min_samples_leaf=2,
        random_state=123,
        n_streams=1,
        n_estimators=num_trees,
        max_leaves=-1,
        max_depth=40,
    )
    cuml_model.fit(X_train, y_train)

    preds = cuml_model.predict(X_test, layout=nvforest_layout)
    r2 = r2_score(y_test, preds)

    nvforest_model = cuml_model.as_nvforest(layout=nvforest_layout)

    nvforest_model_preds = cp.reshape(
        nvforest_model.predict(X_test), np.shape(y_test)
    )
    nvforest_model_r2 = r2_score(cp.asarray(y_test), nvforest_model_preds)
    assert r2 == nvforest_model_r2

    tl_model = cuml_model.as_treelite()
    assert num_trees == tl_model.num_tree
    assert X.shape[1] == tl_model.num_feature

    # Initialize, fit and predict using
    # sklearn's random forest regression model
    if X.shape[0] < 1000:  # mode != "stress":
        sk_model = skrfr(
            n_estimators=50,
            max_depth=40,
            min_samples_split=2,
            random_state=10,
        )
        sk_model.fit(X_train, y_train)
        sk_preds = sk_model.predict(X_test)
        sk_r2 = r2_score(y_test, sk_preds)
        # Observed: mean=0.025, range=[0.021, 0.029], stderr=0.001
        assert r2 >= (sk_r2 - 0.08)


@pytest.mark.parametrize("max_features", [1.0, "log2", "sqrt"])
@pytest.mark.parametrize("max_depth", [10, 13, 16])
@pytest.mark.parametrize("n_estimators", [10, 20, 100])
@pytest.mark.parametrize("n_bins", [8, 9, 10])
def test_create_classification_model(
    max_features, max_depth, n_estimators, n_bins
):
    # random forest classification model
    cuml_model = curfc(
        max_features=max_features,
        n_bins=n_bins,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    params = cuml_model.get_params()
    cuml_model2 = curfc()
    cuml_model2.set_params(**params)
    verfiy_params = cuml_model2.get_params()
    assert params["max_features"] == verfiy_params["max_features"]
    assert params["max_depth"] == verfiy_params["max_depth"]
    assert params["n_estimators"] == verfiy_params["n_estimators"]
    assert params["n_bins"] == verfiy_params["n_bins"]


def test_unlimited_max_depth_classifier():
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)

    clf = curfc(n_estimators=10, max_depth=None, random_state=42)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == len(y)

    params = clf.get_params()
    assert params["max_depth"] is None
    clf2 = curfc()
    clf2.set_params(**params)
    assert clf2.get_params()["max_depth"] is None

    shallow = curfc(n_estimators=10, max_depth=2, random_state=42)
    shallow.fit(X, y)
    assert accuracy_score(y, preds) >= accuracy_score(y, shallow.predict(X))


def test_unlimited_max_depth_regressor():
    X, y = make_regression(n_samples=500, n_features=10, random_state=42)

    reg = curfr(n_estimators=10, max_depth=None, random_state=42)
    reg.fit(X, y)
    assert len(reg.predict(X)) == len(y)

    params = reg.get_params()
    assert params["max_depth"] is None
    reg2 = curfr()
    reg2.set_params(**params)
    assert reg2.get_params()["max_depth"] is None


@pytest.mark.parametrize("n_estimators", [10, 20, 100])
@pytest.mark.parametrize("n_bins", [8, 9, 10])
def test_multiple_fits_classification(large_clf, n_estimators, n_bins):
    datatype = np.float32
    X, y = large_clf
    X = X.astype(datatype)
    y = y.astype(np.int32)
    cuml_model = curfc(n_bins=n_bins, n_estimators=n_estimators, max_depth=10)

    # Calling multiple fits
    cuml_model.fit(X, y)

    cuml_model.fit(X, y)

    # Check if params are still intact
    params = cuml_model.get_params()
    assert params["n_estimators"] == n_estimators
    assert params["n_bins"] == n_bins


@pytest.mark.parametrize(
    "column_info",
    [
        unit_param([100, 50]),
        quality_param([200, 100]),
        stress_param([500, 350]),
    ],
)
@pytest.mark.parametrize(
    "nrows", [unit_param(500), quality_param(5000), stress_param(500000)]
)
@pytest.mark.parametrize("n_estimators", [10, 20, 100])
@pytest.mark.parametrize("n_bins", [8, 9, 10])
def test_multiple_fits_regression(column_info, nrows, n_estimators, n_bins):
    datatype = np.float32
    ncols, n_info = column_info
    X, y = make_regression(
        n_samples=nrows,
        n_features=ncols,
        n_informative=n_info,
        random_state=123,
    )
    X = X.astype(datatype)
    y = y.astype(np.int32)
    cuml_model = curfr(n_bins=n_bins, n_estimators=n_estimators, max_depth=10)

    # Calling multiple fits
    cuml_model.fit(X, y)

    cuml_model.fit(X, y)

    cuml_model.fit(X, y)

    # Check if params are still intact
    params = cuml_model.get_params()
    assert params["n_estimators"] == n_estimators
    assert params["n_bins"] == n_bins


def test_rf_nbins_small(small_clf):
    X, y = small_clf
    X = X.astype(np.float32)
    y = y.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )
    # Initialize, fit and predict using cuML's
    # random forest classification model
    cuml_model = curfc()

    # display warning when nbins less than samples
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cuml_model.fit(X_train[0:3, :], y_train[0:3])
        assert (
            "The number of bins, `n_bins` is greater than "
            "the number of samples used for training. "
            "Changing `n_bins` to number of training samples."
            in str(w[-1].message)
        )


def test_rf_regression_with_identical_labels():
    X = np.array([[-1, 0], [0, 1], [2, 0], [0, 3], [-2, 0]], dtype=np.float32)
    y = np.array([1, 1, 1, 1, 1], dtype=np.float32)
    # Degenerate case: all labels are identical.
    # RF Regressor must not create any split. It must yield an empty tree
    # with only the root node.
    model = curfr(
        max_features=1.0,
        n_bins=5,
        bootstrap=False,
        split_criterion="mse",
        min_samples_leaf=1,
        min_samples_split=2,
        random_state=0,
        n_streams=1,
        n_estimators=1,
        max_depth=1,
    )
    model.fit(X, y)
    trees = json.loads(model.as_treelite().dump_as_json())["trees"]
    assert len(trees) == 1
    assert len(trees[0]["nodes"]) == 1
    assert trees[0]["nodes"][0] == {
        "node_id": 0,
        "leaf_value": 1.0,
        "data_count": 5,
    }


def test_rf_regressor_gtil_integration(tmpdir):
    X, y = make_regression(n_samples=10000, random_state=0)
    X, y = X.astype(np.float32), y.astype(np.float32)
    clf = curfr(max_depth=3, random_state=0, n_estimators=10)
    clf.fit(X, y)
    expected_pred = clf.predict(X).reshape((-1, 1, 1))

    checkpoint_path = os.path.join(tmpdir, "checkpoint.tl")
    clf.as_treelite().serialize(checkpoint_path)

    tl_model = treelite.Model.deserialize(checkpoint_path)
    out_pred = treelite.gtil.predict(tl_model, X)
    np.testing.assert_almost_equal(out_pred, expected_pred, decimal=4)


def test_rf_binary_classifier_gtil_integration(tmpdir):
    X, y = load_breast_cancer(return_X_y=True)
    X, y = X.astype(np.float32), y.astype(np.int32)
    clf = curfc(max_depth=3, random_state=0, n_estimators=10)
    clf.fit(X, y)
    expected_pred = clf.predict_proba(X).reshape((-1, 1, 2))

    checkpoint_path = os.path.join(tmpdir, "checkpoint.tl")
    clf.as_treelite().serialize(checkpoint_path)

    tl_model = treelite.Model.deserialize(checkpoint_path)
    out_pred = treelite.gtil.predict(tl_model, X)
    np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)


def test_rf_multiclass_classifier_gtil_integration(tmpdir):
    X, y = load_iris(return_X_y=True)
    X, y = X.astype(np.float32), y.astype(np.int32)
    clf = curfc(max_depth=3, random_state=0, n_estimators=10)
    clf.fit(X, y)
    expected_prob = clf.predict_proba(X).reshape((X.shape[0], 1, -1))

    checkpoint_path = os.path.join(tmpdir, "checkpoint.tl")
    clf.as_treelite().serialize(checkpoint_path)

    tl_model = treelite.Model.deserialize(checkpoint_path)
    out_prob = treelite.gtil.predict(tl_model, X, pred_margin=True)
    np.testing.assert_almost_equal(out_prob, expected_prob, decimal=5)


def test_classifier_predict_log_proba():
    X, y = make_classification(random_state=42)
    model = curfc(random_state=42).fit(X, y)
    proba = model.predict_proba(X)
    sol = np.log(proba)
    log_proba = model.predict_log_proba(X)
    np.testing.assert_allclose(log_proba, sol, rtol=1e-5)


@pytest.mark.parametrize(
    "estimator, make_data",
    [
        (curfc, make_classification),
        (curfr, make_regression),
    ],
)
def test_rf_min_samples_split_with_small_float(estimator, make_data):
    # Check that min_samples leaf is works with a small float
    # Non-regression test for gh-4613
    X, y = make_data(random_state=0)
    clf = estimator(min_samples_split=0.0001, random_state=0, n_estimators=2)

    # Capture and verify expected warning
    warning_msg = (
        "The number of bins, `n_bins` is greater than the number of samples "
        "used for training"
    )
    with pytest.warns(UserWarning, match=warning_msg):
        clf.fit(X, y)


@pytest.mark.parametrize(
    "max_features, sol",
    [
        (2, 0.02),
        (0.5, 0.5),
        ("sqrt", math.sqrt(100) / 100),
        ("log2", math.log2(100) / 100),
        (None, 1.0),
    ],
)
def test_max_features(max_features, sol):
    res = compute_max_features(max_features, 100)
    assert res == sol


def test_rf_feature_sampling_retries_until_valid_split():
    n_samples = 128
    n_features = 32
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int32)
    y[n_samples // 2 :] = 1
    X[:, 0] = y

    for random_state in range(8):
        clf = curfc(
            n_estimators=1,
            bootstrap=False,
            max_depth=None,
            max_features=1,
            n_bins=4,
            n_streams=1,
            random_state=random_state,
        )
        clf.fit(X, y)
        cuml_acc = accuracy_score(y, clf.predict(X))

        sk_clf = skrfc(
            n_estimators=1,
            bootstrap=False,
            max_depth=None,
            max_features=1,
            random_state=random_state,
        )
        sk_clf.fit(X, y)
        sk_acc = accuracy_score(y, sk_clf.predict(X))

        assert sk_acc == 1.0
        assert cuml_acc == sk_acc


def test_rf_predict_returns_int():
    X, y = make_classification()

    # Capture and verify expected warning
    warning_msg = (
        "The number of bins, `n_bins` is greater than the number of samples "
        "used for training"
    )
    with pytest.warns(UserWarning, match=warning_msg):
        clf = cuml.ensemble.RandomForestClassifier().fit(X, y)

    pred = clf.predict(X)
    assert pred.dtype == np.int64


def test_ensemble_estimator_length():
    X, y = make_classification()
    clf = cuml.ensemble.RandomForestClassifier(n_estimators=3)

    # Capture and verify expected warning
    warning_msg = (
        "The number of bins, `n_bins` is greater than the number of samples "
        "used for training"
    )
    with pytest.warns(UserWarning, match=warning_msg):
        clf.fit(X, y)

    assert len(clf) == 3


def test_rf_feature_importance_classifier():
    """Test feature importance for Random Forest Classifier.

    With shuffle=False, the first n_informative features are guaranteed to be
    informative, and the rest are noise. A correctly working Random Forest should
    always rank the informative features highest.
    """
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        shuffle=False,
        random_state=42,
    )
    X = X.astype(np.float32)
    y = y.astype(np.int32)

    cu_clf = curfc(n_estimators=50, max_depth=8, random_state=42)
    cu_clf.fit(X, y)

    assert hasattr(cu_clf, "feature_importances_")
    cu_importances = cu_clf.feature_importances_

    assert len(cu_importances) == X.shape[1]
    assert np.all(cu_importances >= 0)
    assert np.abs(np.sum(cu_importances) - 1.0) < 1e-5  # Should sum to 1

    top_features = set(np.argsort(cu_importances)[-3:])
    assert top_features == {0, 1, 2}, (
        f"Top 3 features {top_features} should be {{0, 1, 2}}. "
        f"Importances: {cu_importances}"
    )


def test_rf_feature_importance_regressor():
    """Test feature importance for Random Forest Regressor.

    With shuffle=False, the first n_informative features are guaranteed to be
    informative, and the rest are noise. A correctly working Random Forest should
    always rank the informative features highest.
    """
    X, y = make_regression(
        n_samples=1000,
        n_features=10,
        n_informative=3,
        noise=0.1,
        shuffle=False,  # First 3 features are informative, rest are noise
        random_state=42,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    cu_reg = curfr(n_estimators=50, max_depth=8, random_state=42)
    cu_reg.fit(X, y)

    assert hasattr(cu_reg, "feature_importances_")
    cu_importances = cu_reg.feature_importances_

    assert len(cu_importances) == X.shape[1]
    assert np.all(cu_importances >= 0)
    assert np.abs(np.sum(cu_importances) - 1.0) < 1e-5  # Should sum to 1

    top_features = set(np.argsort(cu_importances)[-3:])
    assert top_features == {0, 1, 2}, (
        f"Top 3 features {top_features} should be {{0, 1, 2}}. "
        f"Importances: {cu_importances}"
    )


def test_rf_feature_importance_exact_match_with_fixed_trees():
    """Test that feature importances are reproducible with fixed parameters.

    This test creates a simple dataset and verifies that the
    feature importance calculation produces consistent results.
    """
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 4).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int32)
    y[::5] = 1 - y[::5]

    cu_rf = curfc(
        n_estimators=5,
        max_depth=3,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features=1.0,
        bootstrap=False,
        random_state=42,
    )
    cu_rf.fit(X, y)

    cu_importances = cu_rf.feature_importances_

    if cu_importances.sum() > 0:
        assert np.allclose(cu_importances.sum(), 1.0, rtol=1e-5), (
            f"Feature importances don't sum to 1: {cu_importances.sum()}"
        )

        top_features = np.argsort(cu_importances)[-2:]
        assert 0 in top_features, (
            f"Feature 0 not in top features. Importances: {cu_importances}"
        )

    cu_rf2 = curfc(
        n_estimators=5,
        max_depth=3,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features=1.0,
        bootstrap=False,
        random_state=42,
    )
    cu_rf2.fit(X, y)

    cu_importances2 = cu_rf2.feature_importances_

    # With same parameters and no randomness, should get identical importances
    assert np.allclose(cu_importances, cu_importances2, rtol=1e-5), (
        f"Importances not reproducible:\n1st run: {cu_importances}\n2nd run: {cu_importances2}"
    )


def test_rf_feature_importance_consistency():
    """Test that feature importances are consistent across multiple runs."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_repeated=0,
        random_state=42,
        shuffle=False,
    )

    X = X.astype(np.float32)
    y = y.astype(np.int32)

    # Train the same model multiple times
    importances_list = []
    for i in range(3):
        rf = curfc(
            n_estimators=10,
            max_depth=5,
            min_samples_split=5,
            max_features="sqrt",
            bootstrap=True,
            random_state=42,
        )
        rf.fit(X, y)
        importances_list.append(rf.feature_importances_)

    # All runs should produce identical importances
    for i in range(1, len(importances_list)):
        assert np.allclose(
            importances_list[0], importances_list[i], rtol=1e-5
        ), (
            f"Run {i} produced different importances:\n{importances_list[0]}\nvs\n{importances_list[i]}"
        )


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
def test_rf_oob_score_classifier(datatype):
    """Test OOB score computation for RandomForestClassifier"""
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42,
    )
    X = X.astype(datatype)
    y = y.astype(np.int32)

    # Train with OOB score
    clf = curfc(
        n_estimators=50,
        oob_score=True,
        bootstrap=True,
        max_samples=0.8,
        random_state=42,
    )
    clf.fit(X, y)

    # Check OOB attributes exist
    assert hasattr(clf, "oob_score_")
    assert hasattr(clf, "oob_decision_function_")

    # Check OOB score is reasonable (between 0 and 1)
    assert 0.0 <= clf.oob_score_ <= 1.0

    # Check OOB decision function shape
    assert clf.oob_decision_function_.shape == (
        len(X),
        3,
    )

    # Compare with sklearn (should be similar but not exactly the same due to implementation differences)
    sk_clf = skrfc(
        n_estimators=50,
        oob_score=True,
        bootstrap=True,
        max_samples=0.8,
        random_state=42,
    )
    sk_clf.fit(X, y)

    # OOB scores should be reasonably close (within 0.2)
    assert abs(clf.oob_score_ - sk_clf.oob_score_) < 0.2


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
def test_rf_oob_score_regressor(datatype):
    """Test OOB score computation for RandomForestRegressor"""
    X, y = make_regression(
        n_samples=500, n_features=20, n_informative=15, random_state=42
    )
    X = X.astype(datatype)
    y = y.astype(datatype)

    # Train with OOB score
    reg = curfr(
        n_estimators=50,
        oob_score=True,
        bootstrap=True,
        max_samples=0.8,
        random_state=42,
    )
    reg.fit(X, y)

    # Check OOB attributes exist
    assert hasattr(reg, "oob_score_")
    assert hasattr(reg, "oob_prediction_")

    # Check OOB score is reasonable (R² score, typically between -1 and 1, but for good models > 0)
    assert -1.0 <= reg.oob_score_ <= 1.0
    assert reg.oob_score_ > 0

    # Check OOB prediction shape
    assert reg.oob_prediction_.shape == (len(X),)


def test_rf_oob_score_disabled():
    """Test that OOB score attributes don't exist when oob_score=False"""
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)

    clf = curfc(n_estimators=10, oob_score=False, random_state=42)

    # Capture and verify expected warning
    warning_msg = (
        "The number of bins, `n_bins` is greater than the number of samples "
        "used for training"
    )
    with pytest.warns(UserWarning, match=warning_msg):
        clf.fit(X, y)

    # OOB attributes should not exist
    assert not hasattr(clf, "oob_score_")
    assert not hasattr(clf, "oob_decision_function_")


def test_rf_oob_without_bootstrap():
    """Test OOB score with bootstrap=False (should raise ValueError)"""
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)

    clf = curfc(
        n_estimators=10,
        oob_score=True,
        bootstrap=False,
        random_state=42,
    )

    # Should raise ValueError when oob_score=True but bootstrap=False
    with pytest.raises(
        ValueError,
        match="Out of bag estimation only available if bootstrap=True",
    ):
        clf.fit(X, y)


def test_rf_oob_score_binary_classification():
    """Test OOB score on binary classification"""
    X, y = load_breast_cancer(return_X_y=True)
    X = X.astype(np.float32)

    clf = curfc(n_estimators=30, oob_score=True, max_depth=10, random_state=42)
    clf.fit(X, y)

    # OOB score should be reasonably high for this easy dataset
    assert clf.oob_score_ > 0.85

    # OOB decision function should have 2 classes
    assert clf.oob_decision_function_.shape[1] == 2


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("n_features", [10, 20, 50])
@pytest.mark.skipif(
    cudf_pandas_active,
    reason="cudf.pandas causes sklearn RF estimators crashes sometimes. "
    "Issue: https://github.com/NVIDIA/cuml/issues/5991",
)
def test_rf_feature_zero_bias(datatype, n_features):
    """
    Test for feature 0 sampling bias in RandomForest.

    Creates a dataset where ONLY feature 0 predicts the target,
    and all other features are pure noise. This tests whether
    feature 0 is being properly sampled during tree building.

    Regression test for: https://github.com/NVIDIA/cuml/issues/7422
    """
    n_samples = 5000

    # Create dataset where only feature 0 is predictive
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(datatype)
    # Target depends ONLY on feature 0
    y = (X[:, 0] > 0).astype(np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train scikit-learn model
    sk_model = skrfc(
        n_estimators=100,
        max_features="sqrt",
        max_depth=5,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1,
    )
    sk_model.fit(X_train, y_train)
    sk_pred = sk_model.predict(X_test)
    sk_acc = accuracy_score(y_test, sk_pred)

    # Train cuML model
    cuml_model = curfc(
        n_estimators=100,
        max_features="sqrt",
        max_depth=5,
        min_samples_split=10,
        random_state=42,
    )
    cuml_model.fit(X_train, y_train)
    cuml_pred = cuml_model.predict(X_test)
    cuml_acc = accuracy_score(y_test, cuml_pred)

    # cuML should achieve similar accuracy to sklearn
    # If feature 0 is severely under-sampled, accuracy will be much lower
    # Observed: mean=0.003, range=[0.001, 0.005], stderr=0.000
    assert cuml_acc >= (sk_acc - 0.10)
