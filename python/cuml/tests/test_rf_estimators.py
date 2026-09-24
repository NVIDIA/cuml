# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Tests for RandomForest.estimators_ GPU-backed proxy objects."""

import cupy as cp
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier as skRFC
from sklearn.ensemble import RandomForestRegressor as skRFR

from cuml.ensemble import RandomForestClassifier as cuRFC
from cuml.ensemble import RandomForestRegressor as cuRFR
from cuml.internals.base import Base

pytestmark = pytest.mark.filterwarnings(
    "ignore:The default value of 'max_depth':FutureWarning"
)


@pytest.fixture(scope="module")
def clf_data():
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        n_classes=2,
        random_state=42,
    )
    return X.astype(np.float32), y.astype(np.int32)


@pytest.fixture(scope="module")
def multiclass_data():
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        n_classes=5,
        n_clusters_per_class=1,
        random_state=42,
    )
    return X.astype(np.float32), y.astype(np.int32)


@pytest.fixture(scope="module")
def reg_data():
    X, y = make_regression(
        n_samples=500,
        n_features=10,
        n_informative=5,
        random_state=42,
    )
    return X.astype(np.float32), y.astype(np.float32)


# ---------------------------------------------------------------------------
# GPUTree structural attributes
# ---------------------------------------------------------------------------


def test_tree_topology_shapes(clf_data):
    X, y = clf_data
    rf = cuRFC(n_estimators=3, max_depth=5, random_state=42)
    rf.fit(cp.asarray(X), cp.asarray(y))

    tree = rf.estimators_[0].tree_

    assert tree.node_count > 0
    assert tree.n_features == X.shape[1]
    assert tree.n_outputs == 1
    assert tree.children_left.shape == (tree.node_count,)
    assert tree.children_right.shape == (tree.node_count,)
    assert tree.feature.shape == (tree.node_count,)
    assert tree.threshold.shape == (tree.node_count,)


def test_leaf_internal_consistency(clf_data):
    X, y = clf_data
    rf = cuRFC(n_estimators=3, max_depth=5, random_state=42)
    rf.fit(cp.asarray(X), cp.asarray(y))

    tree = rf.estimators_[0].tree_
    is_leaf = tree.children_left == -1

    np.testing.assert_array_equal(
        tree.children_left[is_leaf], tree.children_right[is_leaf]
    )
    assert np.all(tree.feature[is_leaf] == -2)
    assert np.all(tree.feature[~is_leaf] >= 0)
    assert np.all(tree.children_left[~is_leaf] >= 0)
    assert np.all(tree.children_right[~is_leaf] >= 0)


def test_n_node_samples_parent_equals_children(clf_data):
    X, y = clf_data
    rf = cuRFC(n_estimators=3, max_depth=5, random_state=42)
    rf.fit(cp.asarray(X), cp.asarray(y))

    tree = rf.estimators_[0].tree_
    is_leaf = tree.children_left == -1

    assert tree.n_node_samples[0] > 0
    for i in range(tree.node_count):
        if not is_leaf[i]:
            left = tree.children_left[i]
            right = tree.children_right[i]
            assert tree.n_node_samples[i] == (
                tree.n_node_samples[left] + tree.n_node_samples[right]
            )


def test_estimators_count(clf_data):
    X, y = clf_data
    rf = cuRFC(n_estimators=7, max_depth=4, random_state=42)
    rf.fit(cp.asarray(X), cp.asarray(y))
    assert len(rf.estimators_) == 7


def test_estimators_attribute_error_before_fit():
    rf = cuRFC(n_estimators=3, max_depth=4, random_state=42)
    with pytest.raises(AttributeError, match="no attribute"):
        rf.estimators_


def test_max_depth_respected(clf_data):
    X, y = clf_data
    rf = cuRFC(n_estimators=3, max_depth=4, random_state=42)
    rf.fit(cp.asarray(X), cp.asarray(y))
    for est in rf.estimators_:
        assert est.tree_.max_depth <= 4


# ---------------------------------------------------------------------------
# GPUTree predict / apply
# ---------------------------------------------------------------------------


def test_tree_predict_matches_predict_proba(clf_data):
    """tree_.predict() returns same probabilities as estimator predict_proba."""
    X, y = clf_data
    rf = cuRFC(n_estimators=3, max_depth=5, random_state=42)
    rf.fit(cp.asarray(X), cp.asarray(y))

    est = rf.estimators_[0]
    tree_pred = est.tree_.predict(X)
    est_proba = est.predict_proba(X)
    np.testing.assert_allclose(tree_pred, est_proba, rtol=1e-5)


def test_tree_apply_returns_valid_leaf_ids(clf_data):
    """tree_.apply() returns leaf node IDs that are actually leaves."""
    X, y = clf_data
    rf = cuRFC(n_estimators=3, max_depth=5, random_state=42)
    rf.fit(cp.asarray(X), cp.asarray(y))

    tree = rf.estimators_[0].tree_
    leaf_ids = tree.apply(X)
    is_leaf = tree.children_left == -1

    assert leaf_ids.shape == (X.shape[0],)
    assert np.all(is_leaf[leaf_ids])


def test_tree_apply_matches_estimator_apply(clf_data):
    """tree_.apply() matches estimator-level apply()."""
    X, y = clf_data
    rf = cuRFC(n_estimators=3, max_depth=5, random_state=42)
    rf.fit(cp.asarray(X), cp.asarray(y))

    est = rf.estimators_[0]
    np.testing.assert_array_equal(est.tree_.apply(X), est.apply(X))


def test_tree_predict_multiclass(multiclass_data):
    """tree_.predict() works for multi-class problems."""
    X, y = multiclass_data
    rf = cuRFC(n_estimators=3, max_depth=5, random_state=42)
    rf.fit(cp.asarray(X), cp.asarray(y))

    tree_pred = rf.estimators_[0].tree_.predict(X[:10])
    assert tree_pred.shape == (10, 5)
    np.testing.assert_allclose(tree_pred.sum(axis=1), 1.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# Node value reconstruction and impurity
# ---------------------------------------------------------------------------


def test_value_parent_equals_children_sum(clf_data):
    """value[parent] == value[left] + value[right] for all internal nodes."""
    X, y = clf_data
    rf = cuRFC(n_estimators=3, max_depth=5, random_state=42)
    rf.fit(cp.asarray(X), cp.asarray(y))

    tree = rf.estimators_[0].tree_
    for i in range(tree.node_count):
        if tree.children_left[i] != -1:
            left = tree.children_left[i]
            right = tree.children_right[i]
            np.testing.assert_allclose(
                tree.value[i], tree.value[left] + tree.value[right], rtol=1e-5
            )


def test_value_sum_equals_n_node_samples(clf_data):
    """For classification: sum(value[i]) == n_node_samples[i]."""
    X, y = clf_data
    rf = cuRFC(n_estimators=3, max_depth=5, random_state=42)
    rf.fit(cp.asarray(X), cp.asarray(y))

    for est in rf.estimators_:
        tree = est.tree_
        for i in range(tree.node_count):
            np.testing.assert_allclose(
                tree.value[i].sum(),
                tree.n_node_samples[i],
                rtol=1e-5,
                err_msg=f"node {i}",
            )


def test_impurity_bounds(clf_data):
    """Impurity in [0, 1] for all nodes; pure leaves have impurity 0."""
    X, y = clf_data
    rf = cuRFC(n_estimators=3, max_depth=5, random_state=42)
    rf.fit(cp.asarray(X), cp.asarray(y))

    tree = rf.estimators_[0].tree_
    assert np.all(tree.impurity >= 0)
    assert np.all(tree.impurity <= 1)

    # Pure leaves (single class) should have impurity == 0
    is_leaf = tree.children_left == -1
    for i in range(tree.node_count):
        if is_leaf[i]:
            v = tree.value[i, 0, :]
            if np.count_nonzero(v) <= 1:
                assert tree.impurity[i] == 0.0


def test_from_sklearn_roundtrip_values(clf_data):
    """from_sklearn roundtrip: cuml estimators_ predictions match sklearn's."""
    X, y = clf_data
    sk_rf = skRFC(n_estimators=5, max_depth=5, random_state=42)
    sk_rf.fit(X, y)

    cu_rf = cuRFC.from_sklearn(sk_rf)
    for i in range(5):
        sk_proba = sk_rf.estimators_[i].predict_proba(X[:20])
        cu_proba = cu_rf.estimators_[i].predict_proba(X[:20])
        np.testing.assert_allclose(cu_proba, sk_proba, rtol=1e-4)


# ---------------------------------------------------------------------------
# GPUDecisionTreeClassifier full API
# ---------------------------------------------------------------------------


def test_isinstance_classifier(clf_data):
    X, y = clf_data
    rf = cuRFC(n_estimators=3, max_depth=5, random_state=42)
    rf.fit(cp.asarray(X), cp.asarray(y))

    est = rf.estimators_[0]

    assert isinstance(est, Base)
    assert hasattr(est, "tree_")
    assert hasattr(est, "classes_")
    assert hasattr(est, "n_classes_")
    assert hasattr(est, "n_features_in_")
    assert hasattr(est, "n_outputs_")


def test_estimator_repr(clf_data):
    X, y = clf_data
    rf = cuRFC(n_estimators=3, max_depth=5, random_state=42)
    rf.fit(cp.asarray(X), cp.asarray(y))

    r = repr(rf.estimators_[0])
    assert "GPUDecisionTreeClassifier" in r
    assert "split_criterion" in r


def test_classifier_predict_returns_classes(clf_data):
    X, y = clf_data
    rf = cuRFC(n_estimators=3, max_depth=5, random_state=42)
    rf.fit(cp.asarray(X), cp.asarray(y))

    est = rf.estimators_[0]
    pred = est.predict(X)
    assert pred.shape == (X.shape[0],)
    assert set(pred).issubset(set(cp.asnumpy(rf.classes_)))


def test_classifier_predict_proba_shape_and_sum(clf_data):
    X, y = clf_data
    rf = cuRFC(n_estimators=3, max_depth=5, random_state=42)
    rf.fit(cp.asarray(X), cp.asarray(y))

    proba = rf.estimators_[0].predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)


def test_classifier_self_consistency(clf_data):
    """Forest predict_proba == mean of individual estimator predict_probas."""
    X, y = clf_data
    rf = cuRFC(n_estimators=10, max_depth=5, random_state=42)
    rf.fit(cp.asarray(X), cp.asarray(y))

    forest_proba = cp.asnumpy(rf.predict_proba(cp.asarray(X)))
    est_probas = np.array([est.predict_proba(X) for est in rf.estimators_])
    mean_proba = est_probas.mean(axis=0)
    np.testing.assert_allclose(forest_proba, mean_proba, rtol=1e-4)


def test_classifier_feature_importances(clf_data):
    X, y = clf_data
    rf = cuRFC(n_estimators=3, max_depth=5, random_state=42)
    rf.fit(cp.asarray(X), cp.asarray(y))

    imp = rf.estimators_[0].feature_importances_
    assert imp.shape == (X.shape[1],)
    assert np.all(imp >= 0)
    np.testing.assert_allclose(imp.sum(), 1.0, rtol=1e-5)


def test_multiclass_predict_proba(multiclass_data):
    X, y = multiclass_data
    rf = cuRFC(n_estimators=3, max_depth=5, random_state=42)
    rf.fit(cp.asarray(X), cp.asarray(y))

    proba = rf.estimators_[0].predict_proba(X)
    assert proba.shape == (X.shape[0], 5)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# GPUDecisionTreeRegressor
# ---------------------------------------------------------------------------


def test_isinstance_regressor(reg_data):
    X, y = reg_data
    rf = cuRFR(n_estimators=3, max_depth=5, random_state=42)
    rf.fit(cp.asarray(X), cp.asarray(y))

    est = rf.estimators_[0]

    assert isinstance(est, Base)
    assert hasattr(est, "tree_")
    assert hasattr(est, "n_features_in_")


def test_regressor_predict_shape(reg_data):
    X, y = reg_data
    rf = cuRFR(n_estimators=3, max_depth=5, random_state=42)
    rf.fit(cp.asarray(X), cp.asarray(y))

    pred = rf.estimators_[0].predict(X)
    assert pred.shape == (X.shape[0],)
    assert pred.dtype in (np.float32, np.float64)


def test_regressor_self_consistency(reg_data):
    """Forest predict == mean of individual estimator predictions."""
    X, y = reg_data
    rf = cuRFR(n_estimators=10, max_depth=5, random_state=42)
    rf.fit(cp.asarray(X), cp.asarray(y))

    forest_pred = cp.asnumpy(rf.predict(cp.asarray(X))).ravel()
    est_preds = np.array([est.predict(X) for est in rf.estimators_])
    mean_pred = est_preds.mean(axis=0)
    np.testing.assert_allclose(forest_pred, mean_pred, rtol=1e-4)


def test_regressor_from_sklearn_roundtrip(reg_data):
    X, y = reg_data
    sk_rf = skRFR(n_estimators=5, max_depth=5, random_state=42)
    sk_rf.fit(X, y)

    cu_rf = cuRFR.from_sklearn(sk_rf)
    for i in range(5):
        sk_pred = sk_rf.estimators_[i].predict(X[:20])
        cu_pred = cu_rf.estimators_[i].predict(X[:20])
        np.testing.assert_allclose(cu_pred, sk_pred, rtol=1e-4)


def test_regressor_tree_structure(reg_data):
    X, y = reg_data
    rf = cuRFR(n_estimators=3, max_depth=5, random_state=42)
    rf.fit(cp.asarray(X), cp.asarray(y))

    tree = rf.estimators_[0].tree_
    is_leaf = tree.children_left == -1

    assert tree.node_count > 0
    assert np.all(tree.feature[is_leaf] == -2)
    assert np.all(tree.feature[~is_leaf] >= 0)
    assert tree.n_node_samples[0] > 0


# ---------------------------------------------------------------------------
# sklearn source-of-truth tests
# ---------------------------------------------------------------------------


def test_as_sklearn_roundtrip_classifier(clf_data):
    """as_sklearn() roundtrip: exported estimators_ match our proxy."""
    X, y = clf_data
    rf = cuRFC(n_estimators=5, max_depth=5, random_state=42)
    rf.fit(cp.asarray(X), cp.asarray(y))

    sk_model = rf.as_sklearn()
    for i in range(5):
        sk_proba = sk_model.estimators_[i].predict_proba(X[:20])
        cu_proba = rf.estimators_[i].predict_proba(X[:20])
        np.testing.assert_allclose(cu_proba, sk_proba, rtol=1e-4)


def test_as_sklearn_roundtrip_regressor(reg_data):
    """as_sklearn() roundtrip: exported estimators_ match our proxy."""
    X, y = reg_data
    rf = cuRFR(n_estimators=5, max_depth=5, random_state=42)
    rf.fit(cp.asarray(X), cp.asarray(y))

    sk_model = rf.as_sklearn()
    for i in range(5):
        sk_pred = sk_model.estimators_[i].predict(X[:20])
        cu_pred = rf.estimators_[i].predict(X[:20])
        np.testing.assert_allclose(cu_pred, sk_pred, rtol=1e-4)


def test_from_sklearn_roundtrip_classifier_apply(clf_data):
    """from_sklearn roundtrip: apply() matches sklearn's tree."""
    X, y = clf_data
    sk_rf = skRFC(n_estimators=5, max_depth=5, random_state=42)
    sk_rf.fit(X, y)

    cu_rf = cuRFC.from_sklearn(sk_rf)
    for i in range(5):
        sk_leaf = sk_rf.estimators_[i].apply(X[:20])
        cu_leaf = cu_rf.estimators_[i].apply(X[:20])
        np.testing.assert_array_equal(cu_leaf, sk_leaf)


def test_deep_tree_classifier(clf_data):
    """Deep trees work correctly (depth 10+)."""
    X, y = clf_data
    rf = cuRFC(n_estimators=3, max_depth=12, random_state=42)
    rf.fit(cp.asarray(X), cp.asarray(y))

    est = rf.estimators_[0]
    proba = est.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
    assert est.tree_.max_depth <= 12


def test_estimators_cache_invalidated_on_refit(clf_data):
    """Re-fitting clears the estimators_ cache."""
    X, y = clf_data
    rf = cuRFC(n_estimators=3, max_depth=4, random_state=42)
    rf.fit(cp.asarray(X), cp.asarray(y))
    old_est = rf.estimators_

    rf.fit(cp.asarray(X), cp.asarray(y))
    new_est = rf.estimators_
    assert old_est is not new_est


# ---------------------------------------------------------------------------
# Ecosystem integration tests
# ---------------------------------------------------------------------------


def test_skforecast_pattern(clf_data):
    """Mimics skforecast's access: estimators_[i].tree_.predict(X)."""
    X, y = clf_data
    rf = cuRFC(n_estimators=3, max_depth=5, random_state=42)
    rf.fit(cp.asarray(X), cp.asarray(y))

    for i in range(len(rf.estimators_)):
        tree_pred = rf.estimators_[i].tree_.predict(X[:10])
        assert tree_pred.shape[0] == 10
        assert tree_pred.ndim == 2
        np.testing.assert_allclose(tree_pred.sum(axis=1), 1.0, rtol=1e-5)


def test_skforecast_pattern_regressor(reg_data):
    """Mimics skforecast's access for regression: estimators_[i].tree_.predict(X)."""
    X, y = reg_data
    rf = cuRFR(n_estimators=3, max_depth=5, random_state=42)
    rf.fit(cp.asarray(X), cp.asarray(y))

    for i in range(len(rf.estimators_)):
        tree_pred = rf.estimators_[i].tree_.predict(X[:10])
        assert tree_pred.shape[0] == 10


def test_decision_path(clf_data):
    """decision_path returns a sparse matrix with correct shape."""
    X, y = clf_data
    rf = cuRFC(n_estimators=3, max_depth=5, random_state=42)
    rf.fit(cp.asarray(X), cp.asarray(y))

    tree = rf.estimators_[0].tree_
    path = tree.decision_path(X[:10])

    assert path.shape == (10, tree.node_count)
    # Each sample must visit the root
    assert np.all(path[:, 0].toarray() == 1)
    # Each sample visits at least one leaf
    is_leaf = tree.children_left == -1
    for i in range(10):
        visited = path[i].toarray().ravel()
        assert np.any(visited[is_leaf])
