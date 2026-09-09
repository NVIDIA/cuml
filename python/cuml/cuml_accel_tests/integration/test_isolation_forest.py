# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest

CPUIsolationForest = IsolationForest._cpu_class


@pytest.fixture(scope="module")
def blobs_with_outliers():
    X, _ = make_blobs(
        n_samples=200,
        centers=1,
        cluster_std=0.5,
        random_state=42,
    )
    rng = np.random.RandomState(42)
    outliers = rng.uniform(low=-10, high=10, size=(20, X.shape[1]))
    return np.vstack([X, outliers])


def test_isolation_forest_fit_predict_agreement(blobs_with_outliers):
    X = blobs_with_outliers
    params = {"n_estimators": 100, "random_state": 0}

    expected = CPUIsolationForest(**params).fit(X)
    result = IsolationForest(**params).fit(X)

    assert result._gpu is not None

    expected_labels = expected.predict(X)
    result_labels = result.predict(X)
    assert set(np.unique(result_labels)) <= {-1, 1}
    assert np.mean(expected_labels == result_labels) >= 0.9


def test_isolation_forest_fit_sample_weight_falls_back_to_cpu(
    blobs_with_outliers,
):
    # cuml.ensemble.IsolationForest.fit() has no sample_weight parameter,
    # so a non-None sample_weight cannot be honored on GPU.
    X = blobs_with_outliers
    sample_weight = np.ones(len(X))

    result = IsolationForest(n_estimators=10, random_state=0).fit(
        X, sample_weight=sample_weight
    )

    assert result._gpu is None


def test_isolation_forest_fit_predict_sample_weight_falls_back_to_cpu(
    blobs_with_outliers,
):
    # Same as above, for fit_predict().
    X = blobs_with_outliers
    sample_weight = np.ones(len(X))

    result = IsolationForest(n_estimators=10, random_state=0)
    result.fit_predict(X, sample_weight=sample_weight)

    assert result._gpu is None


def test_isolation_forest_gpu_methods_and_attrs(blobs_with_outliers):
    """GPU methods and synchronized sklearn attributes remain equivalent."""
    X = blobs_with_outliers
    result = IsolationForest(n_estimators=50, random_state=0).fit(X)
    assert result._gpu is not None

    scores = result.score_samples(X)
    decisions = result.decision_function(X)
    assert scores.dtype == np.float64
    assert decisions.dtype == np.float64
    np.testing.assert_allclose(decisions, scores - result._gpu.offset_)

    assert len(result.estimators_) == 50
    assert len(result.estimators_features_) == 50
    assert result.offset_ == pytest.approx(float(result._gpu.offset_))
    assert result._max_samples == result.max_samples_
    np.testing.assert_allclose(
        result._cpu.decision_function(X), decisions, atol=1e-5
    )

    with pytest.raises(AttributeError):
        result.estimators_samples_


def test_isolation_forest_pickle_roundtrip(blobs_with_outliers):
    """A GPU-fitted proxy pickles through its synchronized CPU estimator."""
    X = blobs_with_outliers
    result = IsolationForest(n_estimators=20, random_state=0).fit(X)
    expected_scores = result.score_samples(X)
    expected_predictions = result.predict(X)

    restored = pickle.loads(pickle.dumps(result))

    np.testing.assert_allclose(
        restored.score_samples(X), expected_scores, atol=1e-5
    )
    np.testing.assert_array_equal(restored.predict(X), expected_predictions)
