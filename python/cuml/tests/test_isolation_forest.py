# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""
Pytest tests for cuML's IsolationForest implementation.

These tests are designed to be:
1. Complete - covering all API methods and parameters
2. Fast-running - using small datasets for unit tests
3. Based on sklearn IsolationForest test patterns
4. Validating sklearn compatibility
"""

import pickle

import cupy as cp
import numpy as np
import pytest
import treelite
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest as skIsolationForest

from cuml import IsolationForest as cuIsolationForest
from cuml.internals.interop import UnsupportedOnGPU
from cuml.testing.utils import stress_param, unit_param

# =============================================================================
# Fixtures for test data
# =============================================================================


@pytest.fixture(scope="module")
def synthetic_data_small():
    """Small dataset for fast unit tests."""
    rng = np.random.RandomState(42)
    # Normal data centered at origin
    X_normal = rng.randn(100, 4).astype(np.float32)
    # Outliers far from origin
    X_outliers = rng.uniform(low=-6, high=6, size=(10, 4)).astype(np.float32)
    X = np.vstack([X_normal, X_outliers])
    # Labels: 1 for normal, -1 for outliers (sklearn convention)
    y_true = np.array([1] * 100 + [-1] * 10)
    return X, y_true


@pytest.fixture(scope="module")
def anomaly_data():
    """Dataset large enough for every max_samples setting under test."""
    rng = np.random.RandomState(7)
    X = rng.randn(2000, 8).astype(np.float32)
    X[:30] += 6.0
    return X


@pytest.fixture(scope="module")
def blobs_data():
    """Blob dataset for clustering-like anomaly detection."""
    X, _ = make_blobs(
        n_samples=200,
        n_features=4,
        centers=2,
        cluster_std=1.0,
        random_state=42,
    )
    return X.astype(np.float32)


# =============================================================================
# Basic functionality tests
# =============================================================================


def test_fit_returns_self(synthetic_data_small):
    """fit() should return self for method chaining."""
    X, _ = synthetic_data_small
    clf = cuIsolationForest(n_estimators=10, random_state=42)
    result = clf.fit(X)
    assert result is clf


def test_fit_predict(synthetic_data_small):
    """fit_predict() should work correctly."""
    X, _ = synthetic_data_small
    clf = cuIsolationForest(n_estimators=10, random_state=42)
    predictions = clf.fit_predict(X)
    assert predictions.shape == (X.shape[0],)
    assert set(np.unique(predictions)).issubset({-1, 1})


def test_predict_returns_correct_shape(synthetic_data_small):
    """predict() should return array of shape (n_samples,)."""
    X, _ = synthetic_data_small
    clf = cuIsolationForest(n_estimators=10, random_state=42)
    clf.fit(X)
    predictions = clf.predict(X)
    assert predictions.shape == (X.shape[0],)


def test_predict_returns_valid_labels(synthetic_data_small):
    """predict() should return only -1 (anomaly) or 1 (normal)."""
    X, _ = synthetic_data_small
    clf = cuIsolationForest(n_estimators=10, random_state=42)
    clf.fit(X)
    predictions = clf.predict(X)
    unique_labels = set(np.unique(predictions))
    assert unique_labels.issubset({-1, 1})


def test_score_samples_returns_correct_shape(synthetic_data_small):
    """score_samples() should return array of shape (n_samples,)."""
    X, _ = synthetic_data_small
    clf = cuIsolationForest(n_estimators=10, random_state=42)
    clf.fit(X)
    scores = clf.score_samples(X)
    assert scores.shape == (X.shape[0],)


def test_decision_function_returns_correct_shape(synthetic_data_small):
    """decision_function() should return array of shape (n_samples,)."""
    X, _ = synthetic_data_small
    clf = cuIsolationForest(n_estimators=10, random_state=42)
    clf.fit(X)
    df = clf.decision_function(X)
    assert df.shape == (X.shape[0],)


def test_n_features_in_set_after_fit(synthetic_data_small):
    """n_features_in_ should be set after fit."""
    X, _ = synthetic_data_small
    clf = cuIsolationForest(n_estimators=10, random_state=42)
    clf.fit(X)
    assert clf.n_features_in_ == X.shape[1]


# =============================================================================
# Parameter handling tests
# =============================================================================


def test_default_parameters():
    """Default parameters should match expected values."""
    clf = cuIsolationForest()
    assert clf.n_estimators == 100
    assert clf.max_samples == "auto"
    assert clf.max_depth is None
    assert clf.max_features == 1.0
    assert clf.bootstrap is False
    assert clf.contamination == "auto"


@pytest.mark.parametrize("n_estimators", [10, 50, 100])
def test_n_estimators_parameter(blobs_data, n_estimators):
    """n_estimators parameter should be respected."""
    clf = cuIsolationForest(n_estimators=n_estimators, random_state=42)
    clf.fit(blobs_data)
    predictions = clf.predict(blobs_data)
    assert predictions.shape[0] == blobs_data.shape[0]


@pytest.mark.parametrize("max_samples", [32, 64, 128, 256])
def test_max_samples_int(blobs_data, max_samples):
    """Integer max_samples should subsample correctly."""
    clf = cuIsolationForest(
        n_estimators=10, max_samples=max_samples, random_state=42
    )
    clf.fit(blobs_data)
    predictions = clf.predict(blobs_data)
    assert predictions.shape[0] == blobs_data.shape[0]


@pytest.mark.parametrize("max_samples", [0.5, 0.8, 1.0])
def test_max_samples_float(blobs_data, max_samples):
    """Float max_samples should work as fraction."""
    clf = cuIsolationForest(
        n_estimators=10, max_samples=max_samples, random_state=42
    )
    clf.fit(blobs_data)
    predictions = clf.predict(blobs_data)
    assert predictions.shape[0] == blobs_data.shape[0]


def test_max_samples_auto(blobs_data):
    """max_samples='auto' should use min(256, n_samples)."""
    clf = cuIsolationForest(
        n_estimators=10, max_samples="auto", random_state=42
    )
    clf.fit(blobs_data)
    predictions = clf.predict(blobs_data)
    assert predictions.shape[0] == blobs_data.shape[0]


@pytest.mark.parametrize(
    "max_samples",
    [0, -1, 0.0, -0.1, 1.1, 1.5, "invalid", True],
)
def test_invalid_max_samples_raises(blobs_data, max_samples):
    """Invalid max_samples values should raise a clear ValueError."""
    clf = cuIsolationForest(
        n_estimators=10, max_samples=max_samples, random_state=42
    )

    with pytest.raises(ValueError, match="max_samples"):
        clf.fit(blobs_data)


def test_max_samples_larger_than_input_warns_and_sets_attribute(blobs_data):
    """Large integer max_samples should warn, cap, and set max_samples_."""
    clf = cuIsolationForest(
        n_estimators=10,
        max_samples=blobs_data.shape[0] + 1,
        random_state=42,
    )

    with pytest.warns(UserWarning, match="greater than"):
        clf.fit(blobs_data)

    assert clf.max_samples_ == blobs_data.shape[0]


def test_max_samples_attribute(blobs_data):
    """max_samples_ should expose the resolved number of rows per tree."""
    clf = cuIsolationForest(
        n_estimators=10, max_samples=0.5, random_state=42
    ).fit(blobs_data)

    assert clf.max_samples_ == int(0.5 * blobs_data.shape[0])


@pytest.mark.parametrize("max_depth", [2, 4, 8, None])
def test_max_depth_parameter(blobs_data, max_depth):
    """max_depth parameter should be respected."""
    clf = cuIsolationForest(
        n_estimators=10, max_depth=max_depth, random_state=42
    )
    clf.fit(blobs_data)
    predictions = clf.predict(blobs_data)
    assert predictions.shape[0] == blobs_data.shape[0]


def test_deep_max_depth_global_memory(blobs_data):
    """Deep global-memory trees should fit and score."""
    clf = cuIsolationForest(
        n_estimators=10, max_samples=64, max_depth=17, random_state=42
    )
    clf.fit(blobs_data)

    scores = clf.score_samples(blobs_data)
    predictions = clf.predict(blobs_data)

    assert scores.shape == (blobs_data.shape[0],)
    assert predictions.shape == (blobs_data.shape[0],)
    assert set(np.unique(predictions)).issubset({-1, 1})


@pytest.mark.parametrize(
    "max_features, expected_features",
    [
        (1, 1),
        (2, 2),
        (0.5, 2),
        (0.8, 3),
        (1.0, 4),
    ],
)
def test_max_features_parameter(blobs_data, max_features, expected_features):
    """max_features parameter should be respected."""
    clf = cuIsolationForest(
        n_estimators=10, max_features=max_features, random_state=42
    )
    clf.fit(blobs_data)
    # Each tree may only split on the features sampled for it, so the number
    # of distinct split features per tree is the resolved `max_features`.
    tl_model = clf.as_treelite()
    for tree_id in range(tl_model.num_tree):
        split_index = tl_model.get_tree_accessor(tree_id).get_field(
            "split_index"
        )
        # Leaves carry ``split_index == -1``.
        used = set(split_index[split_index >= 0].tolist())
        assert len(used) == expected_features
    predictions = clf.predict(blobs_data)
    assert predictions.shape[0] == blobs_data.shape[0]


@pytest.mark.parametrize("max_features", [0, -1, 1.1, "invalid", True])
def test_invalid_max_features_raises(blobs_data, max_features):
    """Invalid max_features values should raise during fit."""
    clf = cuIsolationForest(
        n_estimators=10, max_features=max_features, random_state=42
    )

    with pytest.raises(ValueError, match="max_features"):
        clf.fit(blobs_data)


@pytest.mark.parametrize("bootstrap", [True, False])
def test_bootstrap_parameter(blobs_data, bootstrap):
    """Both sklearn bootstrap sampling modes should fit and predict."""
    clf = cuIsolationForest(
        n_estimators=10, bootstrap=bootstrap, random_state=42
    )
    clf.fit(blobs_data)
    predictions = clf.predict(blobs_data)
    assert predictions.shape[0] == blobs_data.shape[0]


def test_unfitted_sklearn_conversion_preserves_parameters():
    """Unfitted estimators may still be converted in either direction."""
    cu_model = cuIsolationForest(n_estimators=7, random_state=42)
    sk_model = cu_model.as_sklearn()
    assert isinstance(sk_model, skIsolationForest)
    assert sk_model.n_estimators == 7

    roundtrip = cuIsolationForest.from_sklearn(sk_model)
    assert roundtrip.n_estimators == 7
    assert roundtrip.random_state == 42


def test_fitted_from_sklearn_is_explicitly_unsupported(blobs_data):
    """Importing a fitted sklearn model must not silently drop the forest."""
    sk_model = skIsolationForest(n_estimators=5, random_state=42).fit(
        blobs_data
    )
    with pytest.raises(UnsupportedOnGPU, match="fitted"):
        cuIsolationForest.from_sklearn(sk_model)


def test_as_sklearn_respects_max_depth(anomaly_data):
    """A configured ``max_depth`` must carry over to the reconstructed
    estimators, and truncated trees (leaves holding many samples) must
    still score identically."""
    cu_model = cuIsolationForest(
        n_estimators=20, max_samples=256, max_depth=4, random_state=11
    ).fit(anomaly_data)
    sk_model = cu_model.as_sklearn()
    assert all(est.max_depth == 4 for est in sk_model.estimators_)
    cu_scores = np.asarray(
        cu_model.score_samples(anomaly_data), dtype=np.float64
    )
    np.testing.assert_allclose(
        cu_scores, sk_model.score_samples(anomaly_data), atol=1e-5
    )


def test_as_sklearn_after_failed_fit_raises(blobs_data):
    """A failed fit sets ``n_features_in_`` before raising, which makes the
    model look fitted to ``InteropMixin``; conversion must still fail
    loudly rather than deserialize a missing forest."""
    cu_model = cuIsolationForest(max_features=0)
    with pytest.raises(ValueError, match="max_features"):
        cu_model.fit(blobs_data)
    with pytest.raises(RuntimeError, match="not been fitted"):
        cu_model.as_sklearn()


@pytest.mark.parametrize(
    "params",
    [
        {"n_estimators": 100, "max_samples": 256},
        {"n_estimators": 50, "max_samples": 128, "max_features": 0.5},
        {"n_estimators": 50, "max_samples": "auto", "contamination": 0.05},
        {"n_estimators": 20, "max_samples": 150, "bootstrap": True},
    ],
)
def test_as_sklearn_scoring_parity(anomaly_data, params):
    """A converted fitted model scores like the cuML model it came from."""
    cu_model = cuIsolationForest(random_state=7, **params).fit(anomaly_data)
    sk_model = cu_model.as_sklearn()
    assert isinstance(sk_model, skIsolationForest)

    cu_scores = np.asarray(
        cu_model.score_samples(anomaly_data), dtype=np.float64
    )
    np.testing.assert_allclose(
        sk_model.score_samples(anomaly_data), cu_scores, rtol=0, atol=1e-5
    )
    cu_decision = np.asarray(
        cu_model.decision_function(anomaly_data), dtype=np.float64
    )
    np.testing.assert_allclose(
        sk_model.decision_function(anomaly_data),
        cu_decision,
        rtol=0,
        atol=1e-5,
    )
    np.testing.assert_array_equal(
        sk_model.predict(anomaly_data),
        np.asarray(cu_model.predict(anomaly_data)),
    )


def test_as_sklearn_float64_parity(anomaly_data):
    """Conversion of a float64 fit matches to float64 precision."""
    X = anomaly_data.astype(np.float64)
    cu_model = cuIsolationForest(n_estimators=20, random_state=1).fit(X)
    sk_model = cu_model.as_sklearn()
    cu_scores = np.asarray(cu_model.score_samples(X), dtype=np.float64)
    np.testing.assert_allclose(
        sk_model.score_samples(X), cu_scores, rtol=0, atol=1e-10
    )


def test_as_sklearn_populates_fitted_attributes(blobs_data):
    """The converted model carries the attributes a sklearn fit would set."""
    cu_model = cuIsolationForest(
        n_estimators=10, max_samples=64, random_state=0
    ).fit(blobs_data)
    sk_model = cu_model.as_sklearn()

    assert sk_model.max_samples_ == 64
    assert sk_model.offset_ == pytest.approx(float(cu_model.offset_))
    assert sk_model.n_features_in_ == blobs_data.shape[1]
    assert len(sk_model.estimators_) == 10
    assert len(sk_model.estimators_features_) == 10
    # The private fit caches sklearn scoring reads must exist and align.
    assert isinstance(sk_model._average_path_length_per_tree, tuple)
    assert isinstance(sk_model._decision_path_lengths, tuple)
    assert len(sk_model._average_path_length_per_tree) == 10
    for est, lengths in zip(
        sk_model.estimators_, sk_model._decision_path_lengths
    ):
        assert est.tree_.node_count == len(lengths)
        assert est.tree_.n_node_samples[0] == 64

    # Sample indices are not recorded by cuML, so the sklearn property
    # backed by `_seeds` stays unavailable rather than returning wrong ones.
    with pytest.raises(AttributeError):
        sk_model.estimators_samples_


def test_as_sklearn_pickle_roundtrip(blobs_data):
    """The converted model survives pickling with identical behavior."""
    cu_model = cuIsolationForest(n_estimators=10, random_state=0).fit(
        blobs_data
    )
    sk_model = cu_model.as_sklearn()
    restored = pickle.loads(pickle.dumps(sk_model))
    np.testing.assert_array_equal(
        restored.score_samples(blobs_data), sk_model.score_samples(blobs_data)
    )
    np.testing.assert_array_equal(
        restored.predict(blobs_data), sk_model.predict(blobs_data)
    )


def test_as_sklearn_constant_data():
    """Degenerate single-node trees convert and score identically."""
    X = np.ones((300, 4), dtype=np.float32)
    cu_model = cuIsolationForest(
        n_estimators=5, max_samples=32, random_state=0
    ).fit(X)
    sk_model = cu_model.as_sklearn()
    np.testing.assert_allclose(
        sk_model.score_samples(X),
        np.asarray(cu_model.score_samples(X), dtype=np.float64),
        rtol=0,
        atol=1e-5,
    )


def test_sync_attrs_to_cpu_populates_target(blobs_data):
    """The sync path `cuml.accel` relies on fills a target sklearn model."""
    cu_model = cuIsolationForest(n_estimators=10, random_state=0).fit(
        blobs_data
    )
    target = skIsolationForest(**cu_model._params_to_cpu())
    cu_model._sync_attrs_to_cpu(target)
    np.testing.assert_allclose(
        target.score_samples(blobs_data),
        np.asarray(cu_model.score_samples(blobs_data), dtype=np.float64),
        rtol=0,
        atol=1e-5,
    )


def test_invert_average_path_length_roundtrip():
    """Recovered counts invert sklearn's average path length exactly."""
    from sklearn.ensemble._iforest import _average_path_length

    from cuml.ensemble.isolation_forest import _invert_average_path_length

    for n in (1, 2, 3, 10, 256, 5000):
        value = float(_average_path_length(np.asarray([n]))[0])
        assert _invert_average_path_length(value) == n


def test_invert_average_path_length_fails_loudly():
    """Values matching no count, or more than one, raise a clear error."""
    from sklearn.ensemble._iforest import _average_path_length

    from cuml.ensemble.isolation_forest import _invert_average_path_length

    with pytest.raises(ValueError, match="negative"):
        _invert_average_path_length(-0.5)
    # Between apl(4) and apl(5), far from both: no candidate matches.
    with pytest.raises(ValueError, match="no integer count"):
        _invert_average_path_length(1.95)
    # At large counts adjacent values collapse within tolerance: ambiguous.
    midpoint = float(_average_path_length(np.asarray([50000, 50001])).mean())
    with pytest.raises(ValueError, match="more than one"):
        _invert_average_path_length(midpoint)


def test_contamination_float_sets_score_quantile_offset(blobs_data):
    """Float contamination should set offset_ from training score quantile."""
    contamination = 0.1
    clf = cuIsolationForest(
        n_estimators=25, contamination=contamination, random_state=42
    )
    clf.fit(blobs_data)

    scores = np.asarray(clf.score_samples(blobs_data))
    decision = np.asarray(clf.decision_function(blobs_data))
    predictions = np.asarray(clf.predict(blobs_data))
    expected_offset = np.percentile(scores, 100.0 * contamination)

    assert clf.offset_ == pytest.approx(expected_offset, rel=1e-6, abs=1e-6)
    np.testing.assert_allclose(decision, scores - clf.offset_)
    np.testing.assert_array_equal(predictions, np.where(decision < 0, -1, 1))


@pytest.mark.parametrize("contamination", [0.0, -0.1, 0.51, 1.0, "invalid"])
def test_invalid_contamination_raises(blobs_data, contamination):
    """contamination should match sklearn's accepted range."""
    clf = cuIsolationForest(
        n_estimators=10, contamination=contamination, random_state=42
    )

    with pytest.raises(ValueError, match="contamination"):
        clf.fit(blobs_data)


# =============================================================================
# Data type tests
# =============================================================================


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_float_dtypes(dtype):
    """Should handle float32 and float64."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 4).astype(dtype)

    clf = cuIsolationForest(n_estimators=10, random_state=42)
    clf.fit(X)

    scores = clf.score_samples(X)
    assert scores.shape == (X.shape[0],)

    predictions = clf.predict(X)
    assert predictions.shape == (X.shape[0],)


def test_refit_replaces_model():
    """Refitting should replace the fitted model, including across dtypes."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 4)
    clf = cuIsolationForest(n_estimators=10, random_state=42)

    for dtype in (np.float32, np.float64):
        X_dtype = X.astype(dtype)
        clf.fit(X_dtype)
        scores = np.asarray(clf.score_samples(X_dtype))
        assert scores.shape == (X.shape[0],)
        assert scores.dtype == dtype


# =============================================================================
# Anomaly detection quality tests
# =============================================================================


def test_outliers_score_lower_sklearn_convention(synthetic_data_small):
    """
    Outliers should have lower (more negative) scores than inliers.
    In sklearn convention: more negative = more anomalous.
    """
    X, y_true = synthetic_data_small
    clf = cuIsolationForest(n_estimators=50, random_state=42)
    clf.fit(X)

    scores = clf.score_samples(X)
    scores = np.asarray(scores)

    inlier_mask = y_true == 1
    outlier_mask = y_true == -1
    mean_inlier_score = np.mean(scores[inlier_mask])
    mean_outlier_score = np.mean(scores[outlier_mask])

    assert mean_outlier_score < mean_inlier_score, (
        f"Outliers should have lower scores than inliers. "
        f"Got inlier={mean_inlier_score:.4f}, outlier={mean_outlier_score:.4f}"
    )


def test_outliers_detected_by_predict(synthetic_data_small):
    """predict() should detect at least some outliers."""
    X, y_true = synthetic_data_small
    clf = cuIsolationForest(n_estimators=50, random_state=42)
    clf.fit(X)

    predictions = clf.predict(X)
    predictions = np.asarray(predictions)

    n_detected = np.sum(predictions == -1)
    assert n_detected > 0, "Should detect at least some anomalies"


def test_extreme_outliers_detected():
    """Extreme outliers should be reliably detected."""
    rng = np.random.RandomState(42)
    X_normal = rng.randn(200, 4).astype(np.float32)
    X_outlier = np.array([[100, 100, 100, 100]], dtype=np.float32)
    X = np.vstack([X_normal, X_outlier])

    clf = cuIsolationForest(n_estimators=100, random_state=42)
    clf.fit(X)

    scores = clf.score_samples(X)
    scores = np.asarray(scores)

    outlier_score = scores[-1]
    normal_scores = scores[:-1]
    assert outlier_score < np.percentile(normal_scores, 5), (
        "Extreme outlier should have very low score"
    )


# =============================================================================
# sklearn compatibility tests
# =============================================================================


def test_api_compatibility(blobs_data):
    """cuML IF should have the same API as sklearn IF."""
    cu_clf = cuIsolationForest(n_estimators=10, random_state=42)
    sk_clf = skIsolationForest(n_estimators=10, random_state=42)

    cu_clf.fit(blobs_data)
    sk_clf.fit(blobs_data)

    cu_preds = cu_clf.predict(blobs_data)
    sk_preds = sk_clf.predict(blobs_data)
    assert cu_preds.shape == sk_preds.shape

    cu_scores = cu_clf.score_samples(blobs_data)
    sk_scores = sk_clf.score_samples(blobs_data)
    assert cu_scores.shape == sk_scores.shape

    cu_df = cu_clf.decision_function(blobs_data)
    sk_df = sk_clf.decision_function(blobs_data)
    assert cu_df.shape == sk_df.shape


def test_score_convention_matches_sklearn(synthetic_data_small):
    """
    Score convention should match sklearn:
    - More negative scores = more anomalous
    - Outliers should have negative decision_function values
    """
    X, y_true = synthetic_data_small
    clf = cuIsolationForest(n_estimators=50, random_state=42)
    clf.fit(X)

    df = clf.decision_function(X)
    df = np.asarray(df)

    outlier_mask = y_true == -1
    inlier_mask = y_true == 1
    mean_outlier_df = np.mean(df[outlier_mask])
    mean_inlier_df = np.mean(df[inlier_mask])
    assert mean_outlier_df < mean_inlier_df


def test_score_decision_predict_consistency(synthetic_data_small):
    """decision_function and predict should be derived from score_samples."""
    X, _ = synthetic_data_small
    clf = cuIsolationForest(n_estimators=50, random_state=42)
    clf.fit(X)

    scores = np.asarray(clf.score_samples(X))
    decision = np.asarray(clf.decision_function(X))
    predictions = np.asarray(clf.predict(X))

    np.testing.assert_allclose(decision, scores - clf.offset_)
    np.testing.assert_array_equal(predictions, np.where(decision < 0, -1, 1))


def test_similar_results_to_sklearn(blobs_data):
    """
    cuML IF should produce similar (not identical) results to sklearn.

    Note: Results won't be identical due to different RNG implementations,
    tree building algorithms, and GPU vs CPU computation.
    But the relative ordering should be similar.
    """
    X = blobs_data

    cu_clf = cuIsolationForest(n_estimators=100, random_state=42)
    sk_clf = skIsolationForest(n_estimators=100, random_state=42)

    cu_clf.fit(X)
    sk_clf.fit(X)

    cu_scores = np.asarray(cu_clf.score_samples(X))
    sk_scores = sk_clf.score_samples(X)

    correlation = np.corrcoef(cu_scores, sk_scores)[0, 1]
    assert correlation > 0.5, (
        f"Score correlation with sklearn should be > 0.5, got {correlation:.3f}"
    )


# =============================================================================
# Treelite / nvForest export tests
# =============================================================================


def test_as_treelite_metadata(blobs_data):
    """as_treelite() should return a valid Treelite IF model."""
    n_estimators = 10
    clf = cuIsolationForest(n_estimators=n_estimators, random_state=42)
    clf.fit(blobs_data)

    tl_model = clf.as_treelite()
    assert isinstance(tl_model, treelite.Model)
    assert tl_model.num_tree == n_estimators
    assert tl_model.num_feature == blobs_data.shape[1]


def test_as_treelite_serialize_roundtrip(blobs_data):
    """Treelite-exported IF models should serialize and deserialize."""
    clf = cuIsolationForest(n_estimators=10, random_state=42)
    clf.fit(blobs_data)

    tl_model = clf.as_treelite()
    roundtrip_model = treelite.Model.deserialize_bytes(
        tl_model.serialize_bytes()
    )

    assert roundtrip_model.num_tree == tl_model.num_tree
    assert roundtrip_model.num_feature == tl_model.num_feature


def test_as_nvforest_loads(blobs_data):
    """as_nvforest() should load the Treelite IF model for GPU inference."""
    X = cp.asarray(blobs_data)
    clf = cuIsolationForest(n_estimators=10, random_state=42)
    clf.fit(X)

    nvforest_model = clf.as_nvforest()
    avg_path_lengths = cp.asarray(nvforest_model.predict(X))

    assert "ForestInferenceRegressor" in type(nvforest_model).__name__
    assert avg_path_lengths.shape[0] == X.shape[0]
    assert bool(cp.all(cp.isfinite(avg_path_lengths)))


def test_score_samples_single_sample():
    """c(1)=0 should produce the neutral -0.5 score."""
    X = cp.asarray([[1.0, 2.0]], dtype=cp.float32)
    clf = cuIsolationForest(n_estimators=2, random_state=42).fit(X)

    cp.testing.assert_allclose(
        cp.asarray(clf.score_samples(X)), cp.asarray([-0.5], dtype=cp.float32)
    )


def test_treelite_export_before_fit_raises():
    """Treelite and nvForest export should require a fitted model."""
    clf = cuIsolationForest()

    with pytest.raises(RuntimeError, match="not been fitted"):
        clf.as_treelite()

    with pytest.raises(RuntimeError, match="not been fitted"):
        clf.as_nvforest()


# =============================================================================
# Determinism tests
# =============================================================================


def test_same_random_state_same_results(blobs_data):
    """Same random_state should produce identical results."""
    clf1 = cuIsolationForest(n_estimators=20, random_state=42)
    clf2 = cuIsolationForest(n_estimators=20, random_state=42)

    clf1.fit(blobs_data)
    clf2.fit(blobs_data)

    scores1 = np.asarray(clf1.score_samples(blobs_data))
    scores2 = np.asarray(clf2.score_samples(blobs_data))

    np.testing.assert_array_almost_equal(
        scores1,
        scores2,
        decimal=5,
        err_msg="Same random_state should produce identical scores",
    )


def test_different_random_state_different_results(blobs_data):
    """Different random_state should produce different results."""
    clf1 = cuIsolationForest(n_estimators=20, random_state=42)
    clf2 = cuIsolationForest(n_estimators=20, random_state=123)

    clf1.fit(blobs_data)
    clf2.fit(blobs_data)

    scores1 = np.asarray(clf1.score_samples(blobs_data))
    scores2 = np.asarray(clf2.score_samples(blobs_data))

    assert not np.allclose(scores1, scores2), (
        "Different random_state should produce different scores"
    )


def test_none_random_state_uses_numpy_global_state(blobs_data):
    """random_state=None should draw its seed from NumPy's global RNG."""
    random_state = np.random.get_state()
    try:
        np.random.seed(42)
        scores1 = np.asarray(
            cuIsolationForest(n_estimators=20)
            .fit(blobs_data)
            .score_samples(blobs_data)
        )

        np.random.seed(42)
        scores2 = np.asarray(
            cuIsolationForest(n_estimators=20)
            .fit(blobs_data)
            .score_samples(blobs_data)
        )

        np.random.seed(123)
        scores3 = np.asarray(
            cuIsolationForest(n_estimators=20)
            .fit(blobs_data)
            .score_samples(blobs_data)
        )
    finally:
        np.random.set_state(random_state)

    np.testing.assert_array_equal(scores1, scores2)
    assert not np.allclose(scores1, scores3)


# =============================================================================
# Edge case tests
# =============================================================================


def test_small_dataset():
    """Should handle small datasets."""
    rng = np.random.RandomState(42)
    X = rng.randn(20, 3).astype(np.float32)

    clf = cuIsolationForest(n_estimators=5, max_samples=10, random_state=42)
    clf.fit(X)
    scores = clf.score_samples(X)
    assert scores.shape == (X.shape[0],)


def test_single_feature():
    """Should handle single-feature data."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 1).astype(np.float32)

    clf = cuIsolationForest(n_estimators=10, random_state=42)
    clf.fit(X)
    scores = clf.score_samples(X)
    assert scores.shape == (X.shape[0],)


def test_many_features():
    """Should handle high-dimensional data."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 50).astype(np.float32)

    clf = cuIsolationForest(n_estimators=10, random_state=42)
    clf.fit(X)
    scores = clf.score_samples(X)
    assert scores.shape == (X.shape[0],)


def test_predict_before_fit_raises():
    """predict() before fit() should raise an error."""
    clf = cuIsolationForest()
    X = np.random.randn(10, 3).astype(np.float32)

    with pytest.raises(RuntimeError, match="not been fitted"):
        clf.predict(X)


def test_score_samples_before_fit_raises():
    """score_samples() before fit() should raise an error."""
    clf = cuIsolationForest()
    X = np.random.randn(10, 3).astype(np.float32)

    with pytest.raises(RuntimeError, match="not been fitted"):
        clf.score_samples(X)


def test_feature_mismatch_raises(blobs_data):
    """Predicting with wrong number of features should raise."""
    clf = cuIsolationForest(n_estimators=10, random_state=42)
    clf.fit(blobs_data)

    X_wrong = np.random.randn(10, blobs_data.shape[1] + 1).astype(np.float32)

    with pytest.raises(ValueError, match="features"):
        clf.predict(X_wrong)


# =============================================================================
# Performance-oriented tests (parameterized by test level)
# =============================================================================


@pytest.mark.parametrize("nrows", [unit_param(200), stress_param(50000)])
@pytest.mark.parametrize("ncols", [unit_param(10), stress_param(200)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_isolation_forest_scaling(nrows, ncols, dtype):
    """Test IF scales with data size."""
    rng = np.random.RandomState(42)
    X = rng.randn(nrows, ncols).astype(dtype)

    clf = cuIsolationForest(n_estimators=20, random_state=42)
    clf.fit(X)

    scores = clf.score_samples(X)
    assert scores.shape == (nrows,)

    predictions = clf.predict(X)
    assert predictions.shape == (nrows,)
    assert set(np.unique(predictions)).issubset({-1, 1})


@pytest.mark.parametrize("n_estimators", [unit_param(10), stress_param(50)])
def test_n_estimators_scaling(blobs_data, n_estimators):
    """Test IF with different numbers of estimators."""
    clf = cuIsolationForest(n_estimators=n_estimators, random_state=42)
    clf.fit(blobs_data)

    scores = clf.score_samples(blobs_data)
    predictions = clf.predict(blobs_data)

    assert scores.shape == (blobs_data.shape[0],)
    assert predictions.shape == (blobs_data.shape[0],)


# =============================================================================
# Integration tests
# =============================================================================


def test_full_workflow(synthetic_data_small):
    """Test complete anomaly detection workflow."""
    X, y_true = synthetic_data_small

    # Split data (don't use outliers in training for realistic scenario)
    X_train = X[y_true == 1]  # Only normal points for training
    X_test = X

    # Train model
    clf = cuIsolationForest(n_estimators=50, random_state=42)
    clf.fit(X_train)

    # Get predictions and scores
    predictions = clf.predict(X_test)
    scores = clf.score_samples(X_test)
    df = clf.decision_function(X_test)

    # Basic sanity checks
    assert predictions.shape == X_test.shape[:1]
    assert scores.shape == X_test.shape[:1]
    assert df.shape == X_test.shape[:1]

    # Check that model detected something
    predictions = np.asarray(predictions)
    n_anomalies = np.sum(predictions == -1)
    assert n_anomalies > 0, "Should detect some anomalies"


def test_train_on_pure_data_detect_outliers():
    """Train on pure data, then detect injected outliers."""
    rng = np.random.RandomState(42)

    # Pure training data (no outliers)
    X_train = rng.randn(500, 4).astype(np.float32)

    # Test data with injected outliers
    X_test_normal = rng.randn(100, 4).astype(np.float32)
    X_test_outliers = np.array(
        [
            [10, 10, 10, 10],
            [-10, -10, -10, -10],
            [5, -5, 5, -5],
        ],
        dtype=np.float32,
    )
    X_test = np.vstack([X_test_normal, X_test_outliers])

    # Train and predict
    clf = cuIsolationForest(n_estimators=100, random_state=42)
    clf.fit(X_train)
    predictions = clf.predict(X_test)
    predictions = np.asarray(predictions)

    # The injected outliers should be detected
    outlier_predictions = predictions[-3:]
    detected_outliers = np.sum(outlier_predictions == -1)
    assert detected_outliers >= 2, (
        f"Should detect at least 2 of 3 injected outliers, got {detected_outliers}"
    )
