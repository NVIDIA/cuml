# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
import cupyx.scipy.sparse as cp_sp
import numpy as np
import pytest
import scipy.sparse as sp
from hypothesis import assume, example, given
from hypothesis import strategies as st
from sklearn.cluster import SpectralClustering as skSpectralClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import kneighbors_graph

from cuml.cluster import SpectralClustering
from cuml.testing.strategies import dataset_dtypes


@pytest.mark.parametrize(
    "affinity",
    [
        "precomputed",
        "nearest_neighbors",
    ],
)
@pytest.mark.parametrize(
    "n_samples,n_clusters",
    [
        (1000, 3),
        (2000, 4),
        (3000, 5),
    ],
)
def test_spectral_clustering_adjusted_rand_score(
    affinity, n_samples, n_clusters
):
    X, y_true = make_blobs(
        n_samples=n_samples,
        n_features=10,
        centers=n_clusters,
        cluster_std=0.3,
        random_state=42,
    )
    X = X.astype(np.float32)

    shared_params = {
        "n_clusters": n_clusters,
        "affinity": affinity,
        "random_state": 42,
        "n_neighbors": 10,
    }

    if affinity == "precomputed":
        knn_graph = kneighbors_graph(
            X,
            n_neighbors=10,
            mode="connectivity",
            include_self=True,
        )
        knn_graph = 0.5 * (knn_graph + knn_graph.T)

        sk_spectral = skSpectralClustering(**shared_params)
        y_sklearn = sk_spectral.fit_predict(knn_graph)

        cuml_spectral = SpectralClustering(**shared_params)
        y_cuml_gpu = cuml_spectral.fit_predict(knn_graph)
        y_cuml = cp.asnumpy(y_cuml_gpu)
    else:
        sk_spectral = skSpectralClustering(**shared_params)
        y_sklearn = sk_spectral.fit_predict(X)

        X_gpu = cp.asarray(X)
        cuml_spectral = SpectralClustering(**shared_params)
        y_cuml_gpu = cuml_spectral.fit_predict(X_gpu)
        y_cuml = cp.asnumpy(y_cuml_gpu)

    ari_sklearn = adjusted_rand_score(y_true, y_sklearn)
    ari_cuml = adjusted_rand_score(y_true, y_cuml)
    ari_sklearn_cuml = adjusted_rand_score(y_sklearn, y_cuml)

    min_ari = 0.8
    assert ari_sklearn > min_ari
    assert ari_cuml > min_ari
    assert ari_sklearn_cuml > min_ari


@pytest.mark.parametrize(
    "input_type,expected_type",
    [
        ("numpy", np.ndarray),
        ("cupy", cp.ndarray),
    ],
)
def test_output_type_handling(input_type, expected_type):
    n_samples = 1000
    n_clusters = 3

    X_np, y_true = make_blobs(
        n_samples=n_samples,
        n_features=10,
        centers=n_clusters,
        cluster_std=0.3,
        random_state=42,
    )
    X_np = X_np.astype(np.float32)
    X = X_np if input_type == "numpy" else cp.asarray(X_np)

    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        n_neighbors=30,
        random_state=42,
    )
    model.fit(X)
    assert isinstance(model.labels_, expected_type)
    assert model.labels_.shape == (n_samples,)

    out = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        n_neighbors=30,
        random_state=42,
    ).fit_predict(X)
    assert isinstance(out, expected_type)
    assert out.shape == (n_samples,)


@pytest.mark.parametrize("n_components", [None, 3, 5])
@pytest.mark.parametrize("n_init", [1, 5, 10])
@pytest.mark.parametrize("eigen_tol", ["auto", 0.0, 1e-4])
def test_hyperparameters(n_components, n_init, eigen_tol):
    n_samples = 500
    n_clusters = 4

    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=10,
        centers=n_clusters,
        cluster_std=0.5,
        random_state=42,
    )
    X = cp.asarray(X, dtype=cp.float32)

    model = SpectralClustering(
        n_clusters=n_clusters,
        n_components=n_components,
        n_init=n_init,
        eigen_tol=eigen_tol,
        affinity="nearest_neighbors",
        n_neighbors=15,
        random_state=42,
    )

    labels = model.fit_predict(X)
    assert labels.shape == (n_samples,)
    assert len(cp.unique(labels)) <= n_clusters


def test_reproducibility_with_random_state():
    n_samples = 500
    n_clusters = 4
    random_state = 42

    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=10,
        centers=n_clusters,
        cluster_std=0.5,
        random_state=0,
    )
    X = cp.asarray(X, dtype=cp.float32)

    results = []
    for _ in range(3):
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=15,
            random_state=random_state,
            n_init=10,
        )
        labels = model.fit_predict(X)
        results.append(labels)

    for i in range(1, len(results)):
        cp.testing.assert_array_equal(results[0], results[i])


@pytest.mark.parametrize(
    "converter",
    [
        pytest.param(lambda x: x.toarray(), id="numpy"),
        pytest.param(lambda x: cp.asarray(x.toarray()), id="cupy"),
        pytest.param(sp.coo_matrix, id="scipy_coo"),
        pytest.param(sp.csr_matrix, id="scipy_csr"),
        pytest.param(sp.csc_matrix, id="scipy_csc"),
        pytest.param(cp_sp.coo_matrix, id="cupy_coo"),
        pytest.param(cp_sp.csr_matrix, id="cupy_csr"),
        pytest.param(cp_sp.csc_matrix, id="cupy_csc"),
    ],
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_precomputed_matrix_formats(converter, dtype):
    n_samples = 1000
    n_clusters = 3

    X, y_true = make_blobs(
        n_samples=n_samples,
        n_features=10,
        centers=n_clusters,
        cluster_std=0.3,
        random_state=42,
    )

    knn_graph = kneighbors_graph(
        X,
        n_neighbors=30,
        mode="connectivity",
        include_self=True,
    )
    knn_graph = 0.5 * (knn_graph + knn_graph.T)

    affinity_matrix = converter(knn_graph).astype(dtype)

    model = SpectralClustering(
        n_clusters=n_clusters, affinity="precomputed", random_state=42
    )
    labels = model.fit_predict(affinity_matrix)

    labels_np = (
        cp.asnumpy(labels)
        if hasattr(labels, "get") or isinstance(labels, cp.ndarray)
        else labels
    )

    ari = adjusted_rand_score(y_true, labels_np)

    assert labels_np.shape == (n_samples,)

    min_ari = 0.8
    assert ari > min_ari


@given(
    n_samples=st.integers(min_value=100, max_value=2000),
    n_features=st.integers(min_value=2, max_value=20),
    n_clusters=st.integers(min_value=2, max_value=8),
    n_neighbors=st.integers(min_value=3, max_value=15),
    affinity=st.sampled_from(["nearest_neighbors", "precomputed"]),
    dtype=dataset_dtypes(),
)
@example(
    n_samples=150,
    n_features=15,
    n_clusters=3,
    n_neighbors=10,
    affinity="nearest_neighbors",
    dtype=np.float32,
)
@example(
    n_samples=200,
    n_features=20,
    n_clusters=4,
    n_neighbors=12,
    affinity="precomputed",
    dtype=np.float64,
)
def test_spectral_clustering_output_shape_type_and_label_count(
    n_samples, n_features, n_clusters, n_neighbors, affinity, dtype
):
    # Ensure parameters are compatible for spectral clustering
    # The eigensolver needs sufficient structure to converge
    # At least 20 samples per cluster for stability
    assume(n_samples >= 20 * n_clusters)
    # But not too dense
    assume(n_neighbors <= n_samples // 5)
    # Need sufficient dimensions for separation
    assume(n_features >= 2 * n_clusters)

    # Generate classification data with cluster structure
    # Use n_clusters as the number of classes to create natural clustering
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=1.0,
        random_state=42,
    )
    X = X.astype(dtype)

    # Filter out datasets with inf, nan, or extremely large values
    # (make_blobs should always produce finite values, but check anyway)
    assume(np.isfinite(X).all())
    assume(np.abs(X).max() < 1e10)

    if affinity == "precomputed":
        knn_graph = kneighbors_graph(
            X,
            n_neighbors=n_neighbors,
            mode="connectivity",
            include_self=True,
        )
        X = (0.5 * (knn_graph + knn_graph.T)).toarray().astype(dtype)

    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity=affinity,
        n_neighbors=n_neighbors,
        random_state=42,
    )

    labels = model.fit_predict(X)

    assert labels.shape == (n_samples,)
    assert labels.dtype in [np.int32, np.int64]

    n_unique = len(np.unique(labels))
    assert n_unique >= 1
    assert n_unique <= n_clusters


def test_spectral_clustering_precomputed_not_square():
    X = np.ones((10, 5))
    with pytest.raises(ValueError, match="X must be 2-dimensional and square"):
        SpectralClustering(affinity="precomputed").fit(X)


def test_spectral_clustering_convergence_failure():
    """Ensure spectral clustering raises a convergence error for unsuitable input.

    When provided with data lacking clustering structure, the eigensolver
    should fail and raise a RuntimeError indicating lack of convergence.
    """
    X = np.random.default_rng(42).normal(size=(100, 100)).astype("float32")

    model = SpectralClustering(
        n_clusters=3,
        affinity="precomputed",
        n_neighbors=10,
        random_state=42,
    )

    # Expect a RuntimeError about convergence failure
    with pytest.raises(RuntimeError, match="failed to converge"):
        model.fit_predict(X)
