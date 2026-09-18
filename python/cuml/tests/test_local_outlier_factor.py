# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for cuML's LocalOutlierFactor implementation."""

import pickle

import numpy as np
import pytest
from sklearn.neighbors import LocalOutlierFactor as skLocalOutlierFactor

from cuml.neighbors import LocalOutlierFactor as cuLocalOutlierFactor


@pytest.fixture(scope="module")
def outlier_data():
    """Gaussian bulk with a shifted cluster of outliers."""
    rng = np.random.RandomState(7)
    X = rng.randn(4000, 8).astype(np.float32)
    X[:40] += 5.0
    return X


@pytest.fixture(scope="module")
def query_data():
    rng = np.random.RandomState(11)
    X = rng.randn(500, 8).astype(np.float32)
    X[:5] += 5.0
    return X


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_neighbors", [5, 20, 50])
def test_negative_outlier_factor_matches_sklearn(
    outlier_data, dtype, n_neighbors
):
    X = outlier_data.astype(dtype)
    sk_model = skLocalOutlierFactor(n_neighbors=n_neighbors).fit(X)
    cu_model = cuLocalOutlierFactor(
        n_neighbors=n_neighbors, output_type="numpy"
    ).fit(X)

    np.testing.assert_allclose(
        cu_model.negative_outlier_factor_,
        sk_model.negative_outlier_factor_,
        atol=1e-4,
    )
    assert cu_model.n_neighbors_ == sk_model.n_neighbors_
    assert cu_model.n_samples_fit_ == sk_model.n_samples_fit_
    assert cu_model.offset_ == sk_model.offset_ == -1.5


def test_fit_predict_matches_sklearn(outlier_data):
    sk_labels = skLocalOutlierFactor(n_neighbors=15).fit_predict(outlier_data)
    cu_labels = cuLocalOutlierFactor(
        n_neighbors=15, output_type="numpy"
    ).fit_predict(outlier_data)
    np.testing.assert_array_equal(cu_labels, sk_labels)


def test_contamination_offset_matches_sklearn(outlier_data):
    sk_model = skLocalOutlierFactor(n_neighbors=15, contamination=0.02).fit(
        outlier_data
    )
    cu_model = cuLocalOutlierFactor(
        n_neighbors=15, contamination=0.02, output_type="numpy"
    ).fit(outlier_data)
    np.testing.assert_allclose(cu_model.offset_, sk_model.offset_, atol=1e-4)


@pytest.mark.parametrize("contamination", [0.0, -0.1, 0.6, "invalid"])
def test_invalid_contamination_raises(outlier_data, contamination):
    model = cuLocalOutlierFactor(contamination=contamination)
    with pytest.raises(ValueError, match="contamination"):
        model.fit(outlier_data)


def test_novelty_scoring_matches_sklearn(outlier_data, query_data):
    sk_model = skLocalOutlierFactor(n_neighbors=15, novelty=True).fit(
        outlier_data
    )
    cu_model = cuLocalOutlierFactor(
        n_neighbors=15, novelty=True, output_type="numpy"
    ).fit(outlier_data)

    np.testing.assert_allclose(
        cu_model.score_samples(query_data),
        sk_model.score_samples(query_data),
        atol=1e-4,
    )
    np.testing.assert_allclose(
        cu_model.decision_function(query_data),
        sk_model.decision_function(query_data),
        atol=1e-4,
    )
    np.testing.assert_array_equal(
        cu_model.predict(query_data), sk_model.predict(query_data)
    )


def test_mode_guards_match_sklearn_semantics(outlier_data):
    outlier_model = cuLocalOutlierFactor(n_neighbors=10).fit(outlier_data)
    with pytest.raises(AttributeError, match="novelty=True"):
        outlier_model.predict(outlier_data)
    with pytest.raises(AttributeError, match="novelty=True"):
        outlier_model.score_samples(outlier_data)

    novelty_model = cuLocalOutlierFactor(n_neighbors=10, novelty=True)
    with pytest.raises(AttributeError, match="novelty=False"):
        novelty_model.fit_predict(outlier_data)


def test_duplicated_points(outlier_data):
    """Exact duplicates exercise the self-removal path and the lrd epsilon."""
    X = np.vstack([outlier_data[:200]] * 3).astype(np.float64)
    sk_model = skLocalOutlierFactor(n_neighbors=10).fit(X)
    cu_model = cuLocalOutlierFactor(n_neighbors=10, output_type="numpy").fit(X)
    np.testing.assert_allclose(
        cu_model.negative_outlier_factor_,
        sk_model.negative_outlier_factor_,
        atol=1e-3,
    )


def test_n_neighbors_clamped_to_n_samples(outlier_data):
    X = outlier_data[:10]
    cu_model = cuLocalOutlierFactor(n_neighbors=50, output_type="numpy").fit(X)
    sk_model = skLocalOutlierFactor(n_neighbors=50).fit(X)
    assert cu_model.n_neighbors_ == sk_model.n_neighbors_ == 9


def test_pickle_roundtrip(outlier_data, query_data):
    cu_model = cuLocalOutlierFactor(
        n_neighbors=15, novelty=True, output_type="numpy"
    ).fit(outlier_data)
    loaded = pickle.loads(pickle.dumps(cu_model))
    np.testing.assert_allclose(
        loaded.score_samples(query_data),
        cu_model.score_samples(query_data),
    )
    assert loaded.get_params() == cu_model.get_params()


def test_get_set_params_roundtrip():
    model = cuLocalOutlierFactor(n_neighbors=7, contamination=0.1)
    params = model.get_params()
    assert params["n_neighbors"] == 7
    assert params["contamination"] == 0.1
    clone = cuLocalOutlierFactor(**params)
    assert clone.get_params() == params
