# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import numpy as np
import pytest
from sklearn import cluster
from sklearn.datasets import make_blobs

from cuml.cluster import AgglomerativeClustering
from cuml.metrics import adjusted_rand_score, pairwise_distances


@pytest.mark.parametrize("connectivity", ["knn", "pairwise"])
def test_duplicate_distances(connectivity):
    X = cp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])

    cuml_agg = AgglomerativeClustering(
        n_clusters=2,
        metric="euclidean",
        linkage="single",
        connectivity=connectivity,
    )

    sk_agg = cluster.AgglomerativeClustering(
        n_clusters=2, metric="euclidean", linkage="single"
    )

    cuml_agg.fit(X)
    sk_agg.fit(X.get())

    assert adjusted_rand_score(cuml_agg.labels_, sk_agg.labels_) == 1.0


@pytest.mark.parametrize("n_samples", [100, 1000])
@pytest.mark.parametrize("n_features", [25, 50])
@pytest.mark.parametrize("n_clusters", [1, 2, 10, 50])
@pytest.mark.parametrize("c", [3, 5, 15])
@pytest.mark.parametrize("connectivity", ["knn", "pairwise"])
@pytest.mark.parametrize("compute_distances", [True, False])
def test_single_linkage_sklearn_compare(
    n_samples,
    n_features,
    n_clusters,
    c,
    connectivity,
    compute_distances,
):
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=1.0,
        random_state=42,
    )

    common = {
        "n_clusters": n_clusters,
        "compute_distances": compute_distances,
        "metric": "euclidean",
        "linkage": "single",
    }

    cu_model = AgglomerativeClustering(
        c=c, connectivity=connectivity, **common
    ).fit(X)
    sk_model = cluster.AgglomerativeClustering(**common).fit(X)

    # Cluster assignments should be exact, even though the actual
    # labels may differ
    assert adjusted_rand_score(cu_model.labels_, sk_model.labels_) == 1.0
    assert cu_model.n_connected_components_ == sk_model.n_connected_components_
    assert cu_model.n_leaves_ == sk_model.n_leaves_
    assert cu_model.n_clusters_ == sk_model.n_clusters_
    # The children in the tree may differ, just compare shapes
    assert cu_model.children_.shape == sk_model.children_.shape

    if compute_distances:
        # Since children_ may differ, we compare:
        # - the expected shape
        # - distances between leaf nodes
        assert cu_model.distances_.shape == (n_samples - 1,)
        left, right = cu_model.children_.T
        mask = (left < n_samples) & (right < n_samples)
        res = cu_model.distances_[mask]
        sol = pairwise_distances(X)[left[mask], right[mask]]
        np.testing.assert_allclose(res, sol, atol=1e-3)
    else:
        assert not hasattr(cu_model, "distances_")


def test_invalid_inputs():
    X, _ = make_blobs()

    with pytest.raises(ValueError):
        AgglomerativeClustering(metric="doesntexist").fit(X)

    with pytest.raises(ValueError):
        AgglomerativeClustering(linkage="doesntexist").fit(X)

    with pytest.raises(ValueError):
        AgglomerativeClustering(connectivity="doesntexist").fit(X)

    with pytest.raises(ValueError):
        AgglomerativeClustering(n_clusters=0).fit(X)

    with pytest.raises(ValueError):
        AgglomerativeClustering(n_clusters=500).fit(X)
