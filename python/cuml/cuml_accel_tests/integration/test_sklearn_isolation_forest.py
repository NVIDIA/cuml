# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle

import numpy as np
import pytest
import scipy.sparse
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest

from cuml.accel import is_proxy
from cuml.ensemble import IsolationForest as CumlIsolationForest

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


def test_isolation_forest_is_a_proxy():
    assert is_proxy(IsolationForest)
    assert IsolationForest._cpu_class is CPUIsolationForest
    assert (
        IsolationForest._gpu_class._cpu_class_path
        == "sklearn.ensemble.IsolationForest"
    )
    assert CumlIsolationForest._cpu_class_path == (
        "sklearn.ensemble.IsolationForest"
    )


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


def test_isolation_forest_decision_function_and_score_samples(
    blobs_with_outliers,
):
    X = blobs_with_outliers
    result = IsolationForest(n_estimators=100, random_state=0).fit(X)
    assert result._gpu is not None

    scores = result.score_samples(X)
    decision = result.decision_function(X)
    assert scores.shape == decision.shape == (len(X),)
    # decision_function is score_samples shifted by a constant offset.
    np.testing.assert_allclose(
        decision, scores - (scores - decision)[0], atol=1e-4
    )


def test_isolation_forest_not_implemented_attributes_give_friendly_error(
    blobs_with_outliers,
):
    result = IsolationForest(n_estimators=50, random_state=0).fit(
        blobs_with_outliers
    )
    assert result._gpu is not None

    with pytest.raises(AttributeError, match="not yet implemented"):
        result.offset_

    with pytest.raises(AttributeError, match="not yet implemented"):
        result.estimators_


def test_isolation_forest_pickle_after_gpu_fit_does_not_crash(
    blobs_with_outliers,
):
    result = IsolationForest(n_estimators=50, random_state=0).fit(
        blobs_with_outliers
    )
    assert result._gpu is not None

    # Must not raise UnsupportedOnCPU: pickling falls back to whatever
    # the CPU estimator has synced, the not-implemented attributes are
    # simply absent from the unpickled copy.
    restored = pickle.loads(pickle.dumps(result))
    assert type(restored) is CPUIsolationForest


def test_isolation_forest_falls_back_on_nan_input(blobs_with_outliers):
    X = blobs_with_outliers.copy()
    X[0, 0] = np.nan

    result = IsolationForest(n_estimators=50, random_state=0).fit(X)
    assert result._gpu is None
    assert type(result._cpu) is CPUIsolationForest


def test_isolation_forest_falls_back_on_sparse_input(blobs_with_outliers):
    sparse_X = scipy.sparse.csr_matrix(blobs_with_outliers)

    result = IsolationForest(n_estimators=50, random_state=0).fit(sparse_X)
    assert result._gpu is None
    assert type(result._cpu) is CPUIsolationForest