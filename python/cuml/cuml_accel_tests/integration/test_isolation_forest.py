# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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


def test_isolation_forest_gpu_fit_attrs_available_after_conversion(
    blobs_with_outliers,
):
    # Regression test: conversion of a fitted GPU IsolationForest back to a
    # CPU estimator is now supported (landed in #8483, tracked by #8420).
    # Accessing fit attributes on a GPU-fitted proxy must trigger that
    # conversion and expose the real synced values instead of raising.
    X = blobs_with_outliers
    result = IsolationForest(n_estimators=50, random_state=0).fit(X)
    assert result._gpu is not None

    gpu_scores = result.decision_function(X)  # dispatched to GPU

    assert len(result.estimators_) == 50
    assert result.offset_ == pytest.approx(float(result._gpu.offset_))

    np.testing.assert_allclose(
        result._cpu.decision_function(X), gpu_scores, atol=1e-5
    )
