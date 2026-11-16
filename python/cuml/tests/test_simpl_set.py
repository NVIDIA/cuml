# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from importlib.metadata import version as package_version

import cupy as cp
import numba
import numpy as np
import pytest
from packaging.version import Version
from umap.umap_ import find_ab_params
from umap.umap_ import fuzzy_simplicial_set as ref_fuzzy_simplicial_set
from umap.umap_ import simplicial_set_embedding as ref_simplicial_set_embedding

from cuml.datasets import make_blobs
from cuml.manifold.umap import fuzzy_simplicial_set as cu_fuzzy_simplicial_set
from cuml.manifold.umap import (
    simplicial_set_embedding as cu_simplicial_set_embedding,
)
from cuml.metrics import trustworthiness
from cuml.neighbors import NearestNeighbors


def correctness_dense(a, b, rtol=0.1, threshold=0.95):
    n_elms = a.size
    n_correct = (cp.abs(a - b) <= (rtol * cp.abs(b))).sum()
    correctness = n_correct / n_elms
    return correctness >= threshold


def correctness_sparse(a, b, atol=0.1, rtol=0.2, threshold=0.95):
    n_ref_zeros = (a == 0).sum()
    n_ref_non_zero_elms = a.size - n_ref_zeros
    n_correct = (cp.abs(a - b) <= (atol + rtol * cp.abs(b))).sum()
    correctness = (n_correct - n_ref_zeros) / n_ref_non_zero_elms
    return correctness >= threshold


UMAP_VERSION = Version(package_version("umap-learn"))
UMAP_NUMBA_REGRESSION = Version(numba.__version__) >= Version(
    "0.62.0"
) and UMAP_VERSION < Version("0.5.12")

umap_numba_regression = pytest.mark.xfail(
    UMAP_NUMBA_REGRESSION,
    reason="Upstream regression in umap<0.5.12 with numba >= 0.62.0",
    strict=True,
)


@pytest.mark.parametrize(
    "n_rows", [pytest.param(800, marks=[umap_numba_regression]), 5000]
)
@pytest.mark.parametrize("n_features", [8, 32])
@pytest.mark.parametrize("n_neighbors", [8, 16])
@pytest.mark.parametrize("precomputed_nearest_neighbors", [False, True])
def test_fuzzy_simplicial_set(
    n_rows, n_features, n_neighbors, precomputed_nearest_neighbors
):
    # TODO: remove once upstream `numba` issues are sorted out
    if UMAP_NUMBA_REGRESSION:
        if (
            n_rows == 5000
            and n_features == 32
            and n_neighbors == 8
            and not precomputed_nearest_neighbors
        ):
            pytest.xfail(
                "Upstream regression in umap<0.5.12 with numba >= 0.62.0"
            )

    n_clusters = 30
    random_state = 42
    metric = "euclidean"

    X, _ = make_blobs(
        n_samples=n_rows,
        centers=n_clusters,
        n_features=n_features,
        random_state=random_state,
    )

    if precomputed_nearest_neighbors:
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        nn.fit(X)
        knn_dists, knn_indices = nn.kneighbors(
            X, n_neighbors=n_neighbors, return_distance=True
        )
        cu_fss_graph = cu_fuzzy_simplicial_set(
            X,
            n_neighbors=n_neighbors,
            random_state=random_state,
            metric=metric,
            knn_indices=knn_indices,
            knn_dists=knn_dists,
        )

        knn_indices = knn_indices.get()
        knn_dists = knn_dists.get()
        # umap-learn assumes exact self-distances and a self entry in column 0.
        # Normalize its inputs without changing the arrays passed to cuML.
        self_neighbors = knn_indices == np.arange(n_rows)[:, None]
        knn_dists[self_neighbors] = 0.0
        order = np.argsort(
            np.where(self_neighbors, -1.0, knn_dists), axis=1, kind="stable"
        )
        knn_indices = np.take_along_axis(knn_indices, order, axis=1)
        knn_dists = np.take_along_axis(knn_dists, order, axis=1)
        ref_fss_graph = ref_fuzzy_simplicial_set(
            X.get(),
            n_neighbors=n_neighbors,
            random_state=random_state,
            metric=metric,
            knn_indices=knn_indices,
            knn_dists=knn_dists,
        )[0]
    else:
        cu_fss_graph = cu_fuzzy_simplicial_set(
            X,
            n_neighbors=n_neighbors,
            random_state=random_state,
            metric=metric,
        )

        ref_fss_graph = ref_fuzzy_simplicial_set(
            X.get(),
            n_neighbors=n_neighbors,
            random_state=random_state,
            metric=metric,
        )[0]

    cu_fss_graph = cu_fss_graph.todense()
    ref_fss_graph = cp.array(ref_fss_graph.todense())
    assert correctness_sparse(
        ref_fss_graph, cu_fss_graph, atol=0.1, rtol=0.2, threshold=0.95
    )


@pytest.mark.parametrize("self_distance", [1e-5, 0.02])
@pytest.mark.parametrize("local_connectivity", [0.0, 0.5, 1.0, 1.5, 8.5])
def test_fuzzy_simplicial_set_nonzero_self_distances(
    self_distance, local_connectivity
):
    n_rows = n_neighbors = 8
    X = cp.zeros((n_rows, 1), dtype=cp.float32)
    indices = (
        cp.arange(n_rows)[:, None] + cp.arange(n_neighbors)[None, :]
    ) % n_rows
    distances = cp.tile(
        cp.array(
            [0.0, 1e-4, 1e-3, 0.01, 0.03, 0.05, 0.08, 0.1],
            dtype=cp.float32,
        ),
        (n_rows, 1),
    )
    # Include an all-zero neighborhood alongside small positive distances.
    distances[0] = 0.0
    expected = cu_fuzzy_simplicial_set(
        X,
        n_neighbors,
        knn_indices=indices,
        knn_dists=distances,
        local_connectivity=local_connectivity,
    )

    rounded_distances = distances.copy()
    rounded_distances[:, 0] = self_distance
    order = cp.argsort(rounded_distances, axis=1)
    rounded_indices = cp.take_along_axis(indices, order, axis=1)
    rounded_distances = cp.take_along_axis(rounded_distances, order, axis=1)
    indices_before = rounded_indices.copy()
    distances_before = rounded_distances.copy()
    actual = cu_fuzzy_simplicial_set(
        X,
        n_neighbors,
        knn_indices=rounded_indices,
        knn_dists=rounded_distances,
        local_connectivity=local_connectivity,
    )

    cp.testing.assert_allclose(
        actual.toarray(), expected.toarray(), rtol=1e-5, atol=1e-6
    )
    cp.testing.assert_array_equal(rounded_indices, indices_before)
    cp.testing.assert_array_equal(rounded_distances, distances_before)


@pytest.mark.parametrize("n_rows", [800, 5000])
@pytest.mark.parametrize("n_features", [8, 32])
@pytest.mark.parametrize("n_neighbors", [8, 16])
@pytest.mark.parametrize("n_components", [2, 5])
def test_simplicial_set_embedding(
    n_rows, n_features, n_neighbors, n_components
):
    n_clusters = 30
    random_state = 42
    metric = "euclidean"
    initial_alpha = 1.0
    a, b = find_ab_params(1.0, 0.1)
    gamma = 1.0
    negative_sample_rate = 5
    n_epochs = 500
    init = "random"
    metric = "euclidean"
    metric_kwds = {}
    densmap = False
    densmap_kwds = {}
    output_dens = False
    output_metric = "euclidean"
    output_metric_kwds = {}

    X, _ = make_blobs(
        n_samples=n_rows,
        centers=n_clusters,
        n_features=n_features,
        random_state=random_state,
    )
    X = X.get()

    ref_fss_graph = ref_fuzzy_simplicial_set(
        X, n_neighbors=n_neighbors, random_state=random_state, metric=metric
    )[0]
    ref_embedding = ref_simplicial_set_embedding(
        X,
        graph=ref_fss_graph,
        n_components=n_components,
        initial_alpha=initial_alpha,
        a=a,
        b=b,
        gamma=gamma,
        negative_sample_rate=negative_sample_rate,
        n_epochs=n_epochs,
        init=init,
        random_state=np.random.RandomState(random_state),
        metric=metric,
        metric_kwds=metric_kwds,
        densmap=densmap,
        densmap_kwds=densmap_kwds,
        output_dens=output_dens,
        output_metric=output_metric,
        output_metric_kwds=output_metric_kwds,
    )[0]

    cu_fss_graph = cu_fuzzy_simplicial_set(
        X, n_neighbors=n_neighbors, random_state=random_state, metric=metric
    )
    cu_embedding = cu_simplicial_set_embedding(
        X,
        graph=cu_fss_graph,
        n_components=n_components,
        initial_alpha=initial_alpha,
        a=a,
        b=b,
        gamma=gamma,
        negative_sample_rate=negative_sample_rate,
        n_epochs=n_epochs,
        init=init,
        random_state=random_state,
        metric=metric,
        metric_kwds=metric_kwds,
        output_metric=output_metric,
        output_metric_kwds=output_metric_kwds,
    )

    assert isinstance(cu_embedding, type(X))

    ref_t_score = trustworthiness(X, ref_embedding, n_neighbors=n_neighbors)
    t_score = trustworthiness(X, cu_embedding, n_neighbors=n_neighbors)
    abs_tol = 0.05
    assert t_score >= ref_t_score - abs_tol
