#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import json

import cudf
import cupy as cp
import numpy as np
import pandas as pd
import pytest
import treelite
from cudf.pandas import LOADED as cudf_pandas_active
from hypothesis import HealthCheck, assume, example, given, settings
from hypothesis import strategies as st
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier as sklrfc
from sklearn.ensemble import RandomForestRegressor as sklrfr
from sklearn.exceptions import NotFittedError

import cuml
from cuml.ensemble import RandomForestClassifier as curfc
from cuml.ensemble import RandomForestRegressor as curfr
from cuml.explainer.tree_shap import TreeExplainer
from cuml.testing.utils import as_type

shap = pytest.importorskip("shap")


def make_classification_with_categorical(
    *,
    n_samples,
    n_features,
    n_categorical,
    n_informative,
    n_redundant,
    n_repeated,
    n_classes,
    random_state,
    numeric_dtype=np.float32,
):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        n_classes=n_classes,
        random_state=random_state,
        n_clusters_per_class=min(2, n_features),
    )
    X, y = X.astype(numeric_dtype), y.astype(numeric_dtype)

    # Turn some columns into categorical, by taking quartiles
    n = np.atleast_1d(y).shape[0]
    X = pd.DataFrame({f"f{i}": X[:, i] for i in range(n_features)})
    for i in range(n_categorical):
        column = f"f{i}"
        n_bins = min(4, n)
        X[column] = pd.qcut(X[column], n_bins, labels=range(n_bins))
    # make sure each target exists
    y[0:n_classes] = range(n_classes)

    assert len(np.unique(y)) == n_classes
    return X, y


def make_regression_with_categorical(
    *,
    n_samples,
    n_features,
    n_categorical,
    n_informative,
    random_state,
    numeric_dtype=np.float32,
    n_targets=1,
):
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_targets=n_targets,
        random_state=random_state,
    )
    X, y = X.astype(numeric_dtype), y.astype(numeric_dtype)

    # Turn some columns into categorical, by taking quartiles
    n = np.atleast_1d(y).shape[0]
    X = pd.DataFrame({f"f{i}": X[:, i] for i in range(n_features)})
    for i in range(n_categorical):
        column = f"f{i}"
        n_bins = min(4, n)
        X[column] = pd.qcut(X[column], n_bins, labels=range(n_bins))
    return X, y


def count_categorical_split(tl_model):
    model_dump = json.loads(tl_model.dump_as_json(pretty_print=False))
    count = 0
    for tree in model_dump["trees"]:
        for node in tree["nodes"]:
            if (
                "node_type" in node
                and node["node_type"] == "categorical_test_node"
            ):
                count += 1
    return count


@pytest.mark.parametrize(
    "objective",
    [
        "reg:squarederror",
        "reg:squaredlogerror",
        "reg:pseudohubererror",
    ],
)
def test_xgb_regressor(objective):
    xgb = pytest.importorskip("xgboost")
    pytest.importorskip(
        "numpy", minversion="2.0", reason="Test fails on numpy < 2"
    )

    n_samples = 100
    X, y = make_regression(
        n_samples=n_samples,
        n_features=8,
        n_informative=8,
        n_targets=1,
        random_state=2021,
    )
    # Ensure that the label exceeds -1
    y += (-0.5) - np.min(y)
    assert np.all(y > -1)
    X, y = X.astype(np.float32), y.astype(np.float32)
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        "objective": objective,
        "base_score": 0.5,
        "seed": 0,
        "max_depth": 6,
        "tree_method": "hist",
        "device": "cuda",
    }
    num_round = 10
    xgb_model = xgb.train(
        params, dtrain, num_boost_round=num_round, evals=[(dtrain, "train")]
    )
    tl_model = treelite.frontend.from_xgboost(xgb_model)

    # Insert NaN randomly into X
    X_test = X.copy()
    n_nan = int(np.floor(X.size * 0.1))
    rng = np.random.default_rng(seed=0)
    index_nan = rng.choice(X.size, size=n_nan, replace=False)
    X_test.ravel()[index_nan] = np.nan

    explainer = TreeExplainer(model=tl_model)
    out = explainer.shap_values(X_test)

    ref_explainer = shap.explainers.Tree(model=xgb_model)
    correct_out = ref_explainer.shap_values(X_test)
    np.testing.assert_almost_equal(out, correct_out, decimal=4)
    np.testing.assert_almost_equal(
        explainer.expected_value, ref_explainer.expected_value, decimal=4
    )


@pytest.mark.parametrize(
    "objective,n_classes",
    [
        ("binary:logistic", 2),
        ("binary:hinge", 2),
        ("binary:logitraw", 2),
        ("count:poisson", 4),
        ("rank:pairwise", 5),
        ("rank:ndcg", 5),
        ("rank:map", 2),
        ("multi:softmax", 5),
        ("multi:softprob", 5),
    ],
    ids=[
        "binary:logistic",
        "binary:hinge",
        "binary:logitraw",
        "count:poisson",
        "rank:pairwise",
        "rank:ndcg",
        "rank:map",
        "multi:softmax",
        "multi:softprob",
    ],
)
def test_xgb_classifier(objective, n_classes):
    xgb = pytest.importorskip("xgboost")
    pytest.importorskip(
        "numpy", minversion="2.0", reason="Test fails on numpy < 2"
    )

    n_samples = 100
    X, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=8,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        random_state=2021,
    )
    X, y = X.astype(np.float32), y.astype(np.float32)
    num_round = 10
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        "objective": objective,
        "base_score": 0.5,
        "seed": 0,
        "max_depth": 6,
        "tree_method": "hist",
        "device": "cuda",
    }
    if objective.startswith("rank:"):
        dtrain.set_group([10] * 10)
    if n_classes > 2 and objective.startswith("multi:"):
        params["num_class"] = n_classes
    xgb_model = xgb.train(params, dtrain=dtrain, num_boost_round=num_round)

    # Insert NaN randomly into X
    X_test = X.copy()
    n_nan = int(np.floor(X.size * 0.1))
    rng = np.random.default_rng(seed=0)
    index_nan = rng.choice(X.size, size=n_nan, replace=False)
    X_test.ravel()[index_nan] = np.nan

    explainer = TreeExplainer(model=xgb_model)
    out = explainer.shap_values(X_test)

    ref_explainer = shap.explainers.Tree(model=xgb_model)
    correct_out = ref_explainer.shap_values(X_test)
    np.testing.assert_almost_equal(out, correct_out, decimal=5)
    np.testing.assert_almost_equal(
        explainer.expected_value, ref_explainer.expected_value, decimal=5
    )


def test_degenerate_cases():
    n_samples = 100
    cuml_model = curfr(
        max_features=1.0,
        max_samples=0.1,
        n_bins=128,
        min_samples_leaf=2,
        random_state=123,
        n_streams=1,
        n_estimators=10,
        max_leaves=-1,
        max_depth=16,
    )
    # Attempt to import un-fitted model
    with pytest.raises(NotFittedError):
        TreeExplainer(model=cuml_model)

    # Depth 0 trees
    rng = np.random.default_rng(seed=0)
    X = rng.standard_normal(size=(n_samples, 8), dtype=np.float32)
    y = np.ones(shape=(n_samples,), dtype=np.float32)
    cuml_model.fit(X, y)
    explainer = TreeExplainer(model=cuml_model)
    out = explainer.shap_values(X)
    # Since the output is always 1.0 no matter the input, SHAP values for all
    # features are zero, as feature values don't have any effect on the output.
    # The bias (expected_value) is 1.0.
    assert np.all(out == 0)
    assert explainer.expected_value == 1.0


@pytest.mark.parametrize("input_type", ["numpy", "cupy", "cudf"])
def test_cuml_rf_regressor(input_type):
    n_samples = 100
    X, y = make_regression(
        n_samples=n_samples,
        n_features=8,
        n_informative=8,
        n_targets=1,
        random_state=2021,
    )
    X, y = X.astype(np.float32), y.astype(np.float32)
    if input_type == "cupy":
        X, y = cp.array(X), cp.array(y)
    elif input_type == "cudf":
        X, y = cudf.DataFrame(X), cudf.Series(y)
    cuml_model = curfr(
        max_features=1.0,
        max_samples=0.1,
        n_bins=128,
        min_samples_leaf=2,
        random_state=123,
        n_streams=1,
        n_estimators=10,
        max_leaves=-1,
        max_depth=16,
    )
    cuml_model.fit(X, y)
    pred = cuml_model.predict(X).squeeze()

    explainer = TreeExplainer(model=cuml_model)
    out = explainer.shap_values(X)
    if input_type == "cupy":
        pred = pred.get()
        out = out.get()
        expected_value = explainer.expected_value.get()
    elif input_type == "cudf":
        pred = pred.to_numpy()
        out = out.get()
        expected_value = explainer.expected_value.get()
    else:
        expected_value = explainer.expected_value
    # SHAP values should add up to predicted score
    shap_sum = np.sum(out, axis=1) + expected_value
    np.testing.assert_almost_equal(shap_sum, pred, decimal=4)


@pytest.mark.parametrize("input_type", ["numpy", "cupy", "cudf"])
@pytest.mark.parametrize("n_classes", [2, 5])
def test_cuml_rf_classifier(n_classes, input_type):
    n_samples = 100
    X, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=8,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        random_state=2021,
    )
    X, y = X.astype(np.float32), y.astype(np.float32)
    if input_type == "cupy":
        X, y = cp.array(X), cp.array(y)
    elif input_type == "cudf":
        X, y = cudf.DataFrame(X), cudf.Series(y)
    cuml_model = curfc(
        max_features=1.0,
        max_samples=0.1,
        n_bins=128,
        min_samples_leaf=2,
        random_state=123,
        n_streams=1,
        n_estimators=10,
        max_leaves=-1,
        max_depth=16,
    )
    cuml_model.fit(X, y)
    pred = cuml_model.predict_proba(X)

    explainer = TreeExplainer(model=cuml_model)
    out = explainer.shap_values(X)
    if input_type == "cupy":
        pred = pred.get()
        out = out.get()
        expected_value = explainer.expected_value.get()
    elif input_type == "cudf":
        pred = pred.to_numpy()
        out = out.get()
        expected_value = explainer.expected_value.get()
    else:
        expected_value = explainer.expected_value
    # SHAP values should add up to predicted score
    expected_value = expected_value.reshape(1, -1)
    shap_sum = np.sum(out, axis=1) + np.tile(expected_value, (n_samples, 1))
    np.testing.assert_almost_equal(shap_sum, pred, decimal=4)


def test_sklearn_rf_regressor():
    n_samples = 100
    X, y = make_regression(
        n_samples=n_samples,
        n_features=8,
        n_informative=8,
        n_targets=1,
        random_state=2021,
    )
    X, y = X.astype(np.float32), y.astype(np.float32)
    skl_model = sklrfr(
        max_features=1.0,
        max_samples=0.1,
        min_samples_leaf=2,
        random_state=123,
        n_estimators=10,
        max_depth=16,
    )
    skl_model.fit(X, y)

    explainer = TreeExplainer(model=skl_model)
    out = explainer.shap_values(X)

    ref_explainer = shap.explainers.Tree(model=skl_model)
    correct_out = ref_explainer.shap_values(X)
    np.testing.assert_almost_equal(out, correct_out, decimal=5)
    np.testing.assert_almost_equal(
        explainer.expected_value, ref_explainer.expected_value, decimal=5
    )


@pytest.mark.parametrize("n_classes", [2, 3, 5])
def test_sklearn_rf_classifier(n_classes):
    n_samples = 100
    X, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=8,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        random_state=2021,
    )
    X, y = X.astype(np.float32), y.astype(np.float32)
    skl_model = sklrfc(
        max_features=1.0,
        max_samples=0.1,
        min_samples_leaf=2,
        random_state=123,
        n_estimators=10,
        max_depth=16,
    )
    skl_model.fit(X, y)

    explainer = TreeExplainer(model=skl_model)
    out = explainer.shap_values(X)

    ref_explainer = shap.explainers.Tree(model=skl_model)
    correct_out = np.array(ref_explainer.shap_values(X))
    expected_value = ref_explainer.expected_value
    np.testing.assert_almost_equal(out, correct_out, decimal=5)
    np.testing.assert_almost_equal(
        explainer.expected_value, expected_value, decimal=5
    )


@pytest.mark.xfail(
    reason=(
        "Treelite does not yet support XGBoost models with categoricals "
        "(https://github.com/NVIDIA/cuml/issues/8055)"
    )
)
def test_xgb_toy_categorical():
    xgb = pytest.importorskip("xgboost")

    X = pd.DataFrame(
        {
            "dummy": np.zeros(5, dtype=np.float32),
            "x": np.array([0, 1, 2, 3, 4], dtype=np.int32),
        }
    )
    y = np.array([0, 0, 1, 1, 1], dtype=np.float32)
    X["x"] = X["x"].astype("category")
    dtrain = xgb.DMatrix(X, y, enable_categorical=True)
    params = {
        "tree_method": "hist",
        "device": "cuda",
        "eval_metric": "error",
        "objective": "binary:logistic",
        "max_depth": 2,
        "min_child_weight": 0,
        "lambda": 0,
    }
    xgb_model = xgb.train(
        params, dtrain, num_boost_round=1, evals=[(dtrain, "train")]
    )
    explainer = TreeExplainer(model=xgb_model)
    out = explainer.shap_values(X)

    ref_out = xgb_model.predict(dtrain, pred_contribs=True)
    np.testing.assert_almost_equal(out, ref_out[:, :-1], decimal=5)
    np.testing.assert_almost_equal(
        explainer.expected_value, ref_out[0, -1], decimal=5
    )


@pytest.mark.xfail(
    reason=(
        "Treelite does not yet support XGBoost models with categoricals "
        "(https://github.com/NVIDIA/cuml/issues/8055)"
    )
)
@pytest.mark.parametrize("n_classes", [2, 3])
def test_xgb_classifier_with_categorical(n_classes):
    xgb = pytest.importorskip("xgboost")

    n_samples = 100
    n_features = 8
    X, y = make_classification_with_categorical(
        n_samples=n_samples,
        n_features=n_features,
        n_categorical=4,
        n_informative=n_features,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        random_state=2022,
    )

    dtrain = xgb.DMatrix(X, y, enable_categorical=True)
    params = {
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": 6,
        "base_score": 0.5,
        "seed": 0,
    }
    if n_classes == 2:
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"
    else:
        params["objective"] = "multi:softprob"
        params["eval_metric"] = "mlogloss"
        params["num_class"] = n_classes
    xgb_model = xgb.train(
        params, dtrain, num_boost_round=10, evals=[(dtrain, "train")]
    )
    assert (
        count_categorical_split(treelite.frontend.from_xgboost(xgb_model)) > 0
    )

    # Insert NaN randomly into X
    X_test = X.values.copy()
    n_nan = int(np.floor(X.size * 0.1))
    rng = np.random.default_rng(seed=0)
    index_nan = rng.choice(X.size, size=n_nan, replace=False)
    X_test.ravel()[index_nan] = np.nan

    explainer = TreeExplainer(model=xgb_model)
    out = explainer.shap_values(X_test)

    dtest = xgb.DMatrix(X_test)
    ref_out = xgb_model.predict(
        dtest, pred_contribs=True, validate_features=False
    )
    if n_classes == 2:
        ref_out, ref_expected_value = ref_out[:, :-1], ref_out[0, -1]
    else:
        ref_out = ref_out.transpose((0, 2, 1))
        ref_out, ref_expected_value = ref_out[:, :-1, :], ref_out[0, -1, :]
    np.testing.assert_almost_equal(out, ref_out, decimal=5)
    np.testing.assert_almost_equal(
        explainer.expected_value, ref_expected_value, decimal=5
    )


@pytest.mark.xfail(
    reason=(
        "Treelite does not yet support XGBoost models with categoricals "
        "(https://github.com/NVIDIA/cuml/issues/8055)"
    )
)
def test_xgb_regressor_with_categorical():
    xgb = pytest.importorskip("xgboost")

    n_samples = 100
    n_features = 8
    X, y = make_regression_with_categorical(
        n_samples=n_samples,
        n_features=n_features,
        n_categorical=4,
        n_informative=n_features,
        random_state=2022,
    )

    dtrain = xgb.DMatrix(X, y, enable_categorical=True)
    params = {
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": 6,
        "base_score": 0.5,
        "seed": 0,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
    }
    xgb_model = xgb.train(
        params, dtrain, num_boost_round=10, evals=[(dtrain, "train")]
    )
    assert (
        count_categorical_split(treelite.frontend.from_xgboost(xgb_model)) > 0
    )

    explainer = TreeExplainer(model=xgb_model)
    out = explainer.shap_values(X).get()

    ref_out = xgb_model.predict(dtrain, pred_contribs=True)
    ref_out, ref_expected_value = ref_out[:, :-1], ref_out[0, -1]
    np.testing.assert_almost_equal(out, ref_out, decimal=5)
    np.testing.assert_almost_equal(
        explainer.expected_value, ref_expected_value, decimal=5
    )


def test_lightgbm_regressor_with_categorical():
    lgb = pytest.importorskip("lightgbm")

    n_samples = 100
    n_features = 8
    n_categorical = 8
    X, y = make_regression_with_categorical(
        n_samples=n_samples,
        n_features=n_features,
        n_categorical=n_categorical,
        n_informative=n_features,
        random_state=2022,
    )

    dtrain = lgb.Dataset(
        X, label=y, categorical_feature=list(range(n_categorical))
    )
    params = {
        "num_leaves": 64,
        "seed": 0,
        "objective": "regression",
        "metric": "rmse",
        "min_data_per_group": 1,
    }
    lgb_model = lgb.train(
        params,
        dtrain,
        num_boost_round=10,
        valid_sets=[dtrain],
        valid_names=["train"],
    )
    assert (
        count_categorical_split(treelite.frontend.from_lightgbm(lgb_model)) > 0
    )

    explainer = TreeExplainer(model=lgb_model)
    out = explainer.shap_values(X)

    ref_explainer = shap.explainers.Tree(model=lgb_model)
    ref_out = ref_explainer.shap_values(X)
    np.testing.assert_almost_equal(out, ref_out, decimal=5)
    np.testing.assert_almost_equal(
        explainer.expected_value, ref_explainer.expected_value, decimal=5
    )


@pytest.mark.parametrize("n_classes", [2, 3])
def test_lightgbm_classifier_with_categorical(n_classes):
    lgb = pytest.importorskip("lightgbm")

    n_samples = 100
    n_features = 8
    n_categorical = 8
    X, y = make_classification_with_categorical(
        n_samples=n_samples,
        n_features=n_features,
        n_categorical=n_categorical,
        n_informative=n_features,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        random_state=2022,
    )

    dtrain = lgb.Dataset(
        X, label=y, categorical_feature=list(range(n_categorical))
    )
    params = {"num_leaves": 64, "seed": 0, "min_data_per_group": 1}
    if n_classes == 2:
        params["objective"] = "binary"
        params["metric"] = "binary_logloss"
    else:
        params["objective"] = "multiclass"
        params["metric"] = "multi_logloss"
        params["num_class"] = n_classes
    lgb_model = lgb.train(
        params,
        dtrain,
        num_boost_round=10,
        valid_sets=[dtrain],
        valid_names=["train"],
    )
    assert (
        count_categorical_split(treelite.frontend.from_lightgbm(lgb_model)) > 0
    )

    # Insert NaN randomly into X
    X_test = X.values.astype(np.float64).copy()
    n_nan = int(np.floor(X.size * 0.1))
    rng = np.random.default_rng(seed=0)
    index_nan = rng.choice(X.size, size=n_nan, replace=False)
    X_test.ravel()[index_nan] = np.nan

    explainer = TreeExplainer(model=lgb_model)
    out = explainer.shap_values(X_test)

    ref_explainer = shap.explainers.Tree(model=lgb_model)
    ref_out = ref_explainer.shap_values(X_test)
    ref_expected_value = ref_explainer.expected_value
    np.testing.assert_almost_equal(
        explainer.expected_value, ref_expected_value, decimal=5
    )
    if not cudf_pandas_active:
        # cudf.pandas causes small numerical issues in the output here,
        # possibly due to something in shap or lightgbm
        np.testing.assert_almost_equal(out, ref_out, decimal=5)


def learn_model(draw, X, y, task, learner, n_estimators, n_targets):
    # for lgbm or xgb return the booster or sklearn object?
    use_sklearn_estimator = draw(st.booleans())
    if learner == "xgb":
        try:
            import xgboost as xgb
        except ImportError:
            assume(False)
            return None, None
        if task == "regression":
            objective = draw(
                st.sampled_from(["reg:squarederror", "reg:pseudohubererror"])
            )
            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                tree_method="hist",
                device="cuda",
                objective=objective,
                enable_categorical=True,
                verbosity=0,
            ).fit(X, y)
        elif task == "classification":
            valid_objectives = [
                "binary:logistic",
                "binary:hinge",
                "binary:logitraw",
                "count:poisson",
            ]
            if n_targets > 2:
                valid_objectives += [
                    "rank:pairwise",
                    "rank:ndcg",
                    "rank:map",
                    "multi:softmax",
                    "multi:softprob",
                ]

            objective = draw(st.sampled_from(valid_objectives))
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                tree_method="hist",
                device="cuda",
                objective=objective,
                enable_categorical=True,
                verbosity=0,
            ).fit(X, y)
        else:
            raise ValueError(f"Unknown task: {task}")
        pred = model.predict(X, output_margin=True)
        if not use_sklearn_estimator:
            model = model.get_booster()
        return model, pred
    elif learner == "rf":
        if task == "regression":
            model = cuml.ensemble.RandomForestRegressor(
                n_estimators=n_estimators
            )
            model.fit(X, y)
            pred = model.predict(X)
        elif task == "classification":
            model = cuml.ensemble.RandomForestClassifier(
                n_estimators=n_estimators
            )
            model.fit(X, y)
            pred = model.predict_proba(X)
        else:
            raise ValueError(f"Unknown task: {task}")
        return model, pred
    elif learner == "skl_rf":
        if task == "regression":
            model = sklrfr(n_estimators=n_estimators)
            model.fit(X, y)
            pred = model.predict(X)
        elif task == "classification":
            model = sklrfc(n_estimators=n_estimators)
            model.fit(X, y)
            pred = model.predict_proba(X)
        else:
            raise ValueError(f"Unknown task: {task}")
        return model, pred
    elif learner == "lgbm":
        try:
            import lightgbm as lgb
        except ImportError:
            assume(False)
            return None, None
        if task == "regression":
            model = lgb.LGBMRegressor(n_estimators=n_estimators).fit(X, y)
        elif task == "classification":
            model = lgb.LGBMClassifier(n_estimators=n_estimators).fit(X, y)
        else:
            raise ValueError(f"Unknown task: {task}")
        pred = model.predict(X, raw_score=True)
        if not use_sklearn_estimator:
            model = model.booster_
        return model, pred
    else:
        raise ValueError(f"Unknown learner: {learner}")


@st.composite
def shap_strategy(draw):
    task = draw(st.sampled_from(["regression", "classification"]))

    n_estimators = draw(st.integers(1, 16))
    n_samples = draw(st.integers(2, 100))
    n_features = draw(st.integers(2, 100))
    learner = draw(st.sampled_from(["xgb", "rf", "skl_rf", "lgbm"]))

    # TODO: Treelite does not yet support XGBoost models with categoricals.
    # (https://github.com/NVIDIA/cuml/issues/8055). Can drop this assume when
    # that's fixed.
    assume(learner != "xgb")

    supports_categorical = learner in ["xgb", "lgbm"]
    supports_nan = learner in ["xgb", "lgbm"]
    if task == "classification":
        n_targets = draw(st.integers(2, 5))
    else:
        n_targets = 1
    n_targets = min(n_targets, n_features)
    n_targets = min(n_targets, n_samples)

    has_categoricals = draw(st.booleans()) and supports_categorical
    dtype = draw(st.sampled_from([np.float32, np.float64]))
    if has_categoricals:
        n_categorical = draw(st.integers(1, n_features))
    else:
        n_categorical = 0

    has_nan = not has_categoricals and supports_nan

    # Filter issues and invalid examples here
    if task == "classification" and learner == "rf":
        # No way to predict_proba with RandomForestClassifier
        # trained on 64-bit data
        # https://github.com/NVIDIA/cuml/issues/4663
        assume(dtype == np.float32)
    if task == "regression" and learner == "skl_rf":
        # multi-output regression not working
        # https://github.com/dmlc/treelite/issues/375
        assume(n_targets == 1)

    # treelite considers a binary classification model to have
    # n_classes=1, which produces an unexpected output shape
    # in the shap values
    if task == "classification" and learner == "skl_rf":
        assume(n_targets > 2)

    # ensure we get some variation in test datasets
    dataset_seed = draw(st.integers(1, 5))
    if task == "classification":
        X, y = make_classification_with_categorical(
            n_samples=n_samples,
            n_features=n_features,
            n_categorical=n_categorical,
            n_informative=n_features,
            n_redundant=0,
            n_repeated=0,
            random_state=dataset_seed,
            n_classes=n_targets,
            numeric_dtype=dtype,
        )
    else:
        X, y = make_regression_with_categorical(
            n_samples=n_samples,
            n_features=n_features,
            n_categorical=n_categorical,
            n_informative=n_features,
            random_state=dataset_seed,
            numeric_dtype=dtype,
            n_targets=n_targets,
        )

    if has_nan:
        # set about half the first column to nan
        X.iloc[np.random.randint(0, n_samples, n_samples // 2), 0] = np.nan

    assert len(X.select_dtypes(include="category").columns) == n_categorical

    model, preds = learn_model(
        draw, X, y, task, learner, n_estimators, n_targets
    )

    if isinstance(preds, pd.DataFrame):
        preds = preds.to_numpy()

    # convert any DataFrame categorical columns to numeric
    return X.astype(dtype), y.astype(dtype), model, preds


def check_efficiency(expected_value, pred, shap_values):
    # shap values add up to prediction
    if len(shap_values.shape) <= 2:
        assert np.allclose(
            np.sum(shap_values, axis=-1) + expected_value, pred, 1e-3, 1e-3
        )
    else:
        n_targets = shap_values.shape[-1]
        for i in range(n_targets):
            assert np.allclose(
                np.sum(shap_values[:, :, i], axis=-1) + expected_value[i],
                pred[:, i],
                1e-3,
                1e-3,
            )


def check_efficiency_interactions(expected_value, pred, shap_values):
    # shap values add up to prediction
    if len(shap_values.shape) <= 3:
        assert np.allclose(
            np.sum(shap_values, axis=(-2, -1)) + expected_value,
            pred,
            1e-3,
            1e-3,
        )
    else:
        n_targets = shap_values.shape[0]
        for i in range(n_targets):
            assert np.allclose(
                np.sum(shap_values[i], axis=(-2, -1)) + expected_value[i],
                pred[:, i],
                1e-3,
                1e-3,
            )


# Generating input data/models can be time consuming and triggers
# hypothesis HealthCheck
@settings(
    deadline=None,
    max_examples=20,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@example(
    params=(
        pd.DataFrame(np.ones((10, 5), dtype=np.float32)),
        np.ones(10, dtype=np.float32),
        curfr(max_features=1.0, random_state=0, n_streams=1, n_bins=10).fit(
            np.ones((10, 5), dtype=np.float32), np.ones(10, dtype=np.float32)
        ),
        np.ones(10, dtype=np.float32),
    ),
    interactions_method="shapley-interactions",
)
@given(
    shap_strategy(),
    st.sampled_from(["shapley-interactions", "shapley-taylor"]),
)
def test_with_hypothesis(params, interactions_method):
    X, y, model, preds = params
    explainer = TreeExplainer(model=model)
    shap_values = explainer.shap_values(X)
    shap_interactions = explainer.shap_interaction_values(
        X, method=interactions_method
    )
    check_efficiency(explainer.expected_value, preds, shap_values)
    check_efficiency_interactions(
        explainer.expected_value, preds, shap_interactions
    )

    # Interventional
    explainer = TreeExplainer(
        model=model, data=X.sample(n=15, replace=True, random_state=0)
    )
    interventional_shap_values = explainer.shap_values(X)
    check_efficiency(
        explainer.expected_value, preds, interventional_shap_values
    )


def test_wrong_inputs():
    X = np.array([[0.0, 2.0], [1.0, 0.5]])
    y = np.array([0, 1])
    model = cuml.ensemble.RandomForestRegressor().fit(X, y)

    # background/X different dtype
    with pytest.raises(
        ValueError, match="Expected background data to have the same dtype"
    ):
        explainer = TreeExplainer(model=model, data=X.astype(np.float32))
        explainer.shap_values(X)

    # background/X different number columns
    with pytest.raises(RuntimeError):
        explainer = TreeExplainer(model=model, data=X[:, 0:1])
        explainer.shap_values(X)

    with pytest.raises(
        ValueError,
        match="Interventional algorithm not"
        " supported for interactions. Please"
        " specify data as None in constructor.",
    ):
        explainer = TreeExplainer(model=model, data=X.astype(np.float32))
        explainer.shap_interaction_values(X)

    with pytest.raises(ValueError, match="Unknown interactions method."):
        explainer = TreeExplainer(model=model)
        explainer.shap_interaction_values(X, method="asdasd")


def test_different_algorithms_different_output():
    # ensure different algorithms are actually being called
    rng = np.random.RandomState(3)
    X = rng.normal(size=(100, 10))
    y = rng.normal(size=100)
    model = cuml.ensemble.RandomForestRegressor().fit(X, y)
    interventional_explainer = TreeExplainer(model=model, data=X)
    explainer = TreeExplainer(model=model)
    assert not np.all(
        explainer.shap_values(X) == interventional_explainer.shap_values(X)
    )
    assert not np.all(
        explainer.shap_interaction_values(X, method="shapley-interactions")
        == explainer.shap_interaction_values(X, method="shapley-taylor")
    )


@settings(deadline=None)
@example(input_type="numpy")
@given(st.sampled_from(["numpy", "cupy", "cudf", "pandas"]))
def test_input_types(input_type):
    # simple test to not crash on different input data-frames
    X = np.array([[0.0, 2.0], [1.0, 0.5]])
    y = np.array([0, 1])
    X, y = as_type(input_type, X, y)
    model = cuml.ensemble.RandomForestRegressor().fit(X, y)
    explainer = TreeExplainer(model=model)
    explainer.shap_values(X)
