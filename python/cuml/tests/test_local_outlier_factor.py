# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import pickle

import cudf
import cupy as cp
import numpy as np
import pandas as pd
import pytest
import scipy.sparse
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


@pytest.mark.parametrize(
    "metric, p",
    [
        ("manhattan", 2),
        ("chebyshev", 2),
        ("cosine", 2),
        ("minkowski", 3),
    ],
)
def test_metrics_match_sklearn(outlier_data, metric, p):
    sk_model = skLocalOutlierFactor(metric=metric, p=p).fit(outlier_data)
    cu_model = cuLocalOutlierFactor(
        metric=metric, p=p, output_type="numpy"
    ).fit(outlier_data)
    np.testing.assert_allclose(
        cu_model.negative_outlier_factor_,
        sk_model.negative_outlier_factor_,
        atol=1e-4,
    )


def test_fit_predict_matches_sklearn(outlier_data):
    sk_labels = skLocalOutlierFactor(n_neighbors=15).fit_predict(outlier_data)
    cu_labels = cuLocalOutlierFactor(
        n_neighbors=15, output_type="numpy"
    ).fit_predict(outlier_data)
    np.testing.assert_array_equal(cu_labels, sk_labels)


def test_contamination_matches_sklearn(outlier_data):
    sk_model = skLocalOutlierFactor(n_neighbors=15, contamination=0.02)
    cu_model = cuLocalOutlierFactor(
        n_neighbors=15, contamination=0.02, output_type="numpy"
    )
    cu_labels = cu_model.fit_predict(outlier_data)
    sk_labels = sk_model.fit_predict(outlier_data)
    np.testing.assert_allclose(cu_model.offset_, sk_model.offset_, atol=1e-4)
    np.testing.assert_array_equal(cu_labels, sk_labels)


@pytest.mark.parametrize("contamination", [0.0, -0.1, 0.6, "invalid", None])
def test_invalid_contamination_raises(outlier_data, contamination):
    model = cuLocalOutlierFactor(contamination=contamination)
    with pytest.raises(ValueError, match="contamination"):
        model.fit(outlier_data)


def test_novelty_matches_sklearn(outlier_data, query_data):
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


def test_methods_availability_follows_novelty():
    # Same as sklearn: methods of the other mode are not exposed at all
    outlier_model = cuLocalOutlierFactor()
    assert hasattr(outlier_model, "fit_predict")
    for method in ["predict", "decision_function", "score_samples"]:
        assert not hasattr(outlier_model, method)

    novelty_model = cuLocalOutlierFactor(novelty=True)
    assert not hasattr(novelty_model, "fit_predict")
    for method in ["predict", "decision_function", "score_samples"]:
        assert hasattr(novelty_model, method)


@pytest.mark.parametrize("novelty", [False, True])
def test_sparse_input(outlier_data, query_data, novelty):
    X = outlier_data.copy()
    X[X < -0.5] = 0
    X = scipy.sparse.csr_matrix(X)
    sk_model = skLocalOutlierFactor(novelty=novelty).fit(X)
    cu_model = cuLocalOutlierFactor(novelty=novelty, output_type="numpy").fit(
        X
    )
    np.testing.assert_allclose(
        cu_model.negative_outlier_factor_,
        sk_model.negative_outlier_factor_,
        atol=1e-4,
    )
    if novelty:
        Q = query_data.copy()
        Q[Q < -0.5] = 0
        Q = scipy.sparse.csr_matrix(Q)
        np.testing.assert_allclose(
            cu_model.score_samples(Q), sk_model.score_samples(Q), atol=1e-4
        )


def test_output_types(outlier_data, query_data):
    index = pd.RangeIndex(100, 100 + len(outlier_data))
    X = cudf.DataFrame(outlier_data, index=index)
    model = cuLocalOutlierFactor().fit(X)
    assert isinstance(model.negative_outlier_factor_, cudf.Series)
    labels = cuLocalOutlierFactor().fit_predict(X)
    assert isinstance(labels, cudf.Series)
    assert labels.index.to_pandas().equals(index)

    index = pd.RangeIndex(100, 100 + len(query_data))
    Q = pd.DataFrame(query_data, index=index)
    model = cuLocalOutlierFactor(novelty=True).fit(outlier_data)
    labels = model.predict(Q)
    assert isinstance(labels, pd.Series)
    assert labels.index.equals(index)
    assert isinstance(model.score_samples(cp.asarray(query_data)), cp.ndarray)


@pytest.mark.parametrize("n_neighbors", [8, 20])
def test_tied_distances(n_neighbors):
    # On a regular grid most neighbors are tied, so cuml and sklearn do not
    # select the same ones and the scores can only agree loosely. No point
    # should be an outlier in both cases.
    grid = np.arange(40, dtype=np.float32)
    X = np.stack(np.meshgrid(grid, grid), axis=-1).reshape(-1, 2)
    sk_model = skLocalOutlierFactor(n_neighbors=n_neighbors).fit(X)
    cu_model = cuLocalOutlierFactor(
        n_neighbors=n_neighbors, output_type="numpy"
    )
    assert (cu_model.fit_predict(X) == 1).all()
    np.testing.assert_allclose(
        cu_model.negative_outlier_factor_,
        sk_model.negative_outlier_factor_,
        atol=0.1,
    )


@pytest.mark.parametrize("n_neighbors", [10, 40])
def test_duplicated_points(n_neighbors):
    # Each point is repeated 30 times, more than n_neighbors in the first
    # case, so the reachability distances are all zero
    rng = np.random.RandomState(0)
    X = np.repeat(rng.randn(50, 4).astype(np.float32), 30, axis=0)
    sk_model = skLocalOutlierFactor(n_neighbors=n_neighbors).fit(X)
    cu_model = cuLocalOutlierFactor(
        n_neighbors=n_neighbors, output_type="numpy"
    ).fit(X)
    np.testing.assert_allclose(
        cu_model.negative_outlier_factor_,
        sk_model.negative_outlier_factor_,
        atol=1e-4,
    )


def test_duplicates_warning():
    # The neighbors of [1, 1] are all copies of the same point, which have an
    # infinite density, so its score explodes. sklearn warns in that case.
    X = np.array([[0, 0]] * 20 + [[1, 1]], dtype=np.float32)
    msg = "Duplicate values are leading to incorrect results"
    with pytest.warns(UserWarning, match=msg):
        skLocalOutlierFactor(n_neighbors=5).fit(X)
    with pytest.warns(UserWarning, match=msg):
        cuLocalOutlierFactor(n_neighbors=5).fit(X)


def test_n_neighbors_larger_than_n_samples(outlier_data):
    X = outlier_data[:10]
    cu_model = cuLocalOutlierFactor(n_neighbors=50, output_type="numpy")
    with pytest.warns(UserWarning, match="n_neighbors will be set to"):
        cu_model.fit(X)
    with pytest.warns(UserWarning, match="n_neighbors will be set to"):
        sk_model = skLocalOutlierFactor(n_neighbors=50).fit(X)
    assert cu_model.n_neighbors_ == sk_model.n_neighbors_ == 9
    np.testing.assert_allclose(
        cu_model.negative_outlier_factor_,
        sk_model.negative_outlier_factor_,
        atol=1e-4,
    )


def test_pickle(outlier_data, query_data):
    cu_model = cuLocalOutlierFactor(novelty=True, output_type="numpy").fit(
        outlier_data
    )
    loaded = pickle.loads(pickle.dumps(cu_model))
    np.testing.assert_array_equal(
        loaded.score_samples(query_data), cu_model.score_samples(query_data)
    )
