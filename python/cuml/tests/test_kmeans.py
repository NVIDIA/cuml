# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import random

import cupy as cp
import numpy as np
import pytest
import sklearn
from sklearn import cluster
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

import cuml
import cuml.internals.logger as logger
from cuml.datasets import make_blobs
from cuml.testing.datasets import make_pattern
from cuml.testing.utils import (
    array_equal,
    quality_param,
    stress_param,
    unit_param,
)

dataset_names = ["blobs", "noisy_circles", "noisy_moons", "varied", "aniso"]


@pytest.fixture
def get_data_consistency_test():
    cluster_std = 1.0
    nrows = 1000
    ncols = 50
    nclusters = 8

    X, y = make_blobs(
        nrows,
        ncols,
        nclusters,
        cluster_std=cluster_std,
        shuffle=False,
        random_state=0,
    )
    return X, y


@pytest.fixture
def random_state():
    random_state = random.randint(0, 10**6)
    with logger.set_level(logger.level_enum.debug):
        logger.debug("Random seed: {}".format(random_state))
    return random_state


@pytest.mark.xfail
def test_n_init_cluster_consistency(random_state):
    nclusters = 8
    X, y = get_data_consistency_test()

    cuml_kmeans = cuml.KMeans(
        init="k-means++",
        n_clusters=nclusters,
        n_init=10,
        random_state=random_state,
        output_type="numpy",
    )

    cuml_kmeans.fit(X)
    initial_clusters = cuml_kmeans.cluster_centers_

    cuml_kmeans = cuml.KMeans(
        init="k-means++",
        n_clusters=nclusters,
        n_init=10,
        random_state=random_state,
        output_type="numpy",
    )

    cuml_kmeans.fit(X)

    assert array_equal(initial_clusters, cuml_kmeans.cluster_centers_)


@pytest.mark.parametrize("nrows", [1000, 10000])
@pytest.mark.parametrize("ncols", [25])
@pytest.mark.parametrize("nclusters", [2, 5])
def test_traditional_kmeans_plus_plus_init(
    nrows, ncols, nclusters, random_state
):
    # Using fairly high variance between points in clusters
    cluster_std = 1.0

    X, y = make_blobs(
        int(nrows),
        ncols,
        nclusters,
        cluster_std=cluster_std,
        shuffle=False,
        random_state=0,
    )

    cuml_kmeans = cuml.KMeans(
        init="k-means++",
        n_clusters=nclusters,
        n_init=10,
        random_state=random_state,
        output_type="numpy",
    )

    cuml_kmeans.fit(X)
    cu_score = cuml_kmeans.score(X)

    kmeans = cluster.KMeans(
        random_state=random_state, n_clusters=nclusters, n_init=10
    )
    kmeans.fit(cp.asnumpy(X))
    sk_score = kmeans.score(cp.asnumpy(X))

    if cu_score < sk_score:
        cp.testing.assert_allclose(cu_score, sk_score, atol=0.1, rtol=1e-2)


@pytest.mark.parametrize("nrows", [100, 500])
@pytest.mark.parametrize("ncols", [25])
@pytest.mark.parametrize("nclusters", [5, 10])
@pytest.mark.parametrize("max_weight", [10])
def test_weighted_kmeans(nrows, ncols, nclusters, max_weight, random_state):
    # Using fairly high variance between points in clusters
    cluster_std = 1.0
    np.random.seed(random_state)

    # set weight per sample to be from 1 to max_weight
    wt = np.random.randint(1, high=max_weight, size=nrows)

    X, y = make_blobs(
        nrows,
        ncols,
        nclusters,
        cluster_std=cluster_std,
        shuffle=False,
        random_state=0,
    )

    cuml_kmeans = cuml.KMeans(
        init="k-means++",
        n_clusters=nclusters,
        n_init=10,
        random_state=random_state,
        output_type="numpy",
    )

    cuml_kmeans.fit(X, sample_weight=wt)
    cu_score = cuml_kmeans.score(X)

    sk_kmeans = cluster.KMeans(
        random_state=random_state, n_clusters=nclusters, n_init=10
    )
    sk_kmeans.fit(cp.asnumpy(X), sample_weight=wt)
    sk_score = sk_kmeans.score(cp.asnumpy(X))

    if cu_score < sk_score:
        relative_tolerance = 0.1  # allow 10% difference
        diff = abs(cu_score - sk_score)
        avg_score = (abs(cu_score) + abs(sk_score)) / 2.0
        assert diff / avg_score <= relative_tolerance


@pytest.mark.parametrize("nrows", [1000, 10000])
@pytest.mark.parametrize("ncols", [25])
@pytest.mark.parametrize("nclusters", [2, 5])
@pytest.mark.parametrize("cluster_std", [1.0, 0.1, 0.01])
def test_kmeans_clusters_blobs(
    nrows, ncols, nclusters, random_state, cluster_std
):
    X, y = make_blobs(
        int(nrows),
        ncols,
        nclusters,
        cluster_std=cluster_std,
        shuffle=False,
        random_state=0,
    )

    # Set n_init to 2 to improve stability of k-means|| initialization
    # See https://github.com/NVIDIA/cuml/issues/5530 for details
    cuml_kmeans = cuml.KMeans(
        init="k-means||",
        n_clusters=nclusters,
        random_state=random_state,
        output_type="numpy",
        n_init=2,
    )

    preds = cuml_kmeans.fit_predict(X)

    assert adjusted_rand_score(cp.asnumpy(preds), cp.asnumpy(y)) >= 0.99


@pytest.mark.parametrize("name", dataset_names)
@pytest.mark.parametrize("nrows", [unit_param(1000), quality_param(5000)])
def test_kmeans_sklearn_comparison(name, nrows, random_state):
    default_base = {
        "quantile": 0.3,
        "eps": 0.3,
        "damping": 0.9,
        "preference": -200,
        "n_neighbors": 10,
        "n_clusters": 3,
    }

    pat = make_pattern(name, nrows)

    params = default_base.copy()
    params.update(pat[1])

    cuml_kmeans = cuml.KMeans(
        n_clusters=params["n_clusters"],
        output_type="numpy",
        init="k-means++",
        random_state=random_state,
        n_init=10,
    )

    X, y = pat[0]

    X = StandardScaler().fit_transform(X)

    cu_y_pred = cuml_kmeans.fit_predict(X)
    cu_score = adjusted_rand_score(cu_y_pred, y)
    kmeans = cluster.KMeans(
        random_state=random_state,
        n_clusters=params["n_clusters"],
        n_init=10,
    )
    sk_y_pred = kmeans.fit_predict(X)
    sk_score = adjusted_rand_score(sk_y_pred, y)

    assert sk_score - 1e-2 <= cu_score <= sk_score + 1e-2


@pytest.mark.parametrize("name", dataset_names)
@pytest.mark.parametrize(
    "nrows", [unit_param(500), quality_param(5000), stress_param(500000)]
)
def test_kmeans_sklearn_comparison_default(name, nrows, random_state):
    default_base = {
        "quantile": 0.3,
        "eps": 0.3,
        "damping": 0.9,
        "preference": -200,
        "n_neighbors": 10,
        "n_clusters": 3,
    }

    pat = make_pattern(name, nrows)

    params = default_base.copy()
    params.update(pat[1])

    cuml_kmeans = cuml.KMeans(
        n_clusters=params["n_clusters"],
        random_state=random_state,
        n_init=10,
        output_type="numpy",
    )

    X, y = pat[0]

    X = StandardScaler().fit_transform(X)

    cu_y_pred = cuml_kmeans.fit_predict(X)
    cu_score = adjusted_rand_score(cu_y_pred, y)
    kmeans = cluster.KMeans(
        random_state=random_state,
        n_clusters=params["n_clusters"],
        n_init=10,
    )
    sk_y_pred = kmeans.fit_predict(X)
    sk_score = adjusted_rand_score(sk_y_pred, y)

    assert sk_score - 1e-2 <= cu_score <= sk_score + 1e-2


@pytest.mark.parametrize(
    "max_iter, oversampling_factor, max_samples_per_batch, init",
    [
        (100, 0.5, 1 << 10, "preset"),
        (1000, 1.0, 1 << 15, "preset"),
        (500, 1.5, 1 << 5, "k-means||"),
        (1000, 1.0, 1 << 10, "random"),
    ],
)
@pytest.mark.parametrize(
    "n_clusters", [unit_param(10), unit_param(100), stress_param(1000)]
)
def test_all_kmeans_params(
    n_clusters,
    max_iter,
    init,
    oversampling_factor,
    max_samples_per_batch,
    random_state,
):
    np.random.seed(0)
    X = np.random.rand(1000, 10)

    if init == "preset":
        init = np.random.rand(n_clusters, 10)

    model = cuml.KMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        init=init,
        random_state=random_state,
        oversampling_factor=oversampling_factor,
        max_samples_per_batch=max_samples_per_batch,
        output_type="cupy",
        n_init=1,
    )
    model.fit(X)
    assert hasattr(model, "labels_")

    # Check that can clone and refit
    model2 = sklearn.clone(model)
    assert not hasattr(model2, "labels_")
    model2.fit(X)
    assert hasattr(model2, "labels_")


@pytest.mark.parametrize(
    "nrows", [unit_param(500), quality_param(5000), stress_param(500000)]
)
@pytest.mark.parametrize("ncols", [10, 30])
@pytest.mark.parametrize(
    "nclusters", [unit_param(5), quality_param(10), stress_param(50)]
)
def test_score(nrows, ncols, nclusters, random_state):
    X, y = make_blobs(
        int(nrows),
        ncols,
        nclusters,
        cluster_std=1.0,
        shuffle=False,
        random_state=0,
    )

    cuml_kmeans = cuml.KMeans(
        init="k-means||",
        n_clusters=nclusters,
        random_state=random_state,
        output_type="numpy",
        n_init=1,
    )

    cuml_kmeans.fit(X)

    actual_score = cuml_kmeans.score(X)
    predictions = cuml_kmeans.predict(X)

    centers = cuml_kmeans.cluster_centers_

    expected_score = 0.0
    for idx, label in enumerate(predictions):
        x = X[idx, :]
        y = cp.array(centers[label, :], dtype=cp.float32)

        sq_euc_dist = cp.sum(cp.square((x - y)))
        expected_score += sq_euc_dist

    expected_score *= -1

    cp.testing.assert_allclose(
        actual_score, expected_score, atol=0.1, rtol=5e-3
    )


@pytest.mark.parametrize("nrows", [100])
@pytest.mark.parametrize("ncols", [25])
@pytest.mark.parametrize("nclusters", [5])
@pytest.mark.parametrize("max_weight", [10])
def test_fit_transform_weighted_kmeans(
    nrows, ncols, nclusters, max_weight, random_state
):
    # Using fairly high variance between points in clusters
    cluster_std = 1.0
    np.random.seed(random_state)

    # set weight per sample to be from 1 to max_weight
    wt = np.random.randint(1, high=max_weight, size=nrows)

    X, y = make_blobs(
        nrows,
        ncols,
        nclusters,
        cluster_std=cluster_std,
        shuffle=False,
        random_state=0,
    )

    cuml_kmeans = cuml.KMeans(
        init="k-means++",
        n_clusters=nclusters,
        n_init=10,
        random_state=random_state,
        output_type="numpy",
    )

    cuml_transf = cuml_kmeans.fit_transform(X, sample_weight=wt)
    cu_score = cuml_kmeans.score(X)

    sk_kmeans = cluster.KMeans(random_state=random_state, n_clusters=nclusters)
    sk_transf = sk_kmeans.fit_transform(cp.asnumpy(X), sample_weight=wt)
    sk_score = sk_kmeans.score(cp.asnumpy(X))

    # we fail if cuML's score is significantly worse than sklearn's
    if cu_score < sk_score:
        relative_tolerance = 0.1  # allow 10% difference
        diff = abs(cu_score - sk_score)
        avg_score = (abs(cu_score) + abs(sk_score)) / 2.0
        assert diff / avg_score <= relative_tolerance

    assert sk_transf.shape == cuml_transf.shape


def test_kmeans_empty_x():
    """Check that a nice error happens if X is empty, rather than a segfault"""
    model = cuml.KMeans()

    X = np.empty(shape=(0, 10))
    y = np.ones(shape=0)
    with pytest.raises(ValueError, match=r"Found array with 0 sample\(s\)"):
        model.fit(X, y)

    X = np.empty(shape=(10, 0))
    y = np.ones(shape=10)
    with pytest.raises(ValueError, match=r"Found array with 0 feature\(s\)"):
        model.fit(X, y)


def test_kmeans_n_samples_less_than_n_clusters():
    model = cuml.KMeans(n_clusters=8)

    X = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    with pytest.raises(
        ValueError, match="n_samples=2 should be >= n_clusters=8"
    ):
        model.fit(X)


def test_kmeans_init_wrong_shape():
    X = np.ones(shape=(20, 10))

    # init not compatible with X
    model = cuml.KMeans(n_init=1, init=X[:8, :2], n_clusters=8)
    with pytest.raises(
        ValueError,
        match=(
            r"The shape of the initial centers .* does not match "
            r"the number of features of the data"
        ),
    ):
        model.fit(X)

    # init not compatible with n_clusters
    model = cuml.KMeans(n_init=1, init=X[:2], n_clusters=8)
    with pytest.raises(
        ValueError,
        match=(
            r"The shape of the initial centers .* does not match "
            r"the number of clusters"
        ),
    ):
        model.fit(X)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("device_buffer_samples", [256, 1024, 5000])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("init_method", ["explicit", "default"])
def test_kmeans_device_buffer_samples_host_path(
    dtype, device_buffer_samples, weighted, init_method
):
    """The single-GPU host-streaming fit path should agree with the device
    path on identical inputs.
    """
    n_rows = 4000
    n_cols = 16
    n_clusters = 8

    X_dev, _ = make_blobs(
        n_samples=n_rows,
        n_features=n_cols,
        centers=n_clusters,
        cluster_std=0.5,
        random_state=42,
        dtype=dtype,
    )
    X_host = cp.asnumpy(X_dev).astype(dtype)

    if weighted:
        sample_weight_host = np.linspace(0.5, 1.5, num=n_rows, dtype=dtype)
        sample_weight_dev = cp.asarray(sample_weight_host)
    else:
        sample_weight_host = None
        sample_weight_dev = None

    common_kwargs = dict(
        n_clusters=n_clusters,
        n_init=1,
        tol=1e-3,
        random_state=42,
    )
    if init_method == "explicit":
        rng = np.random.RandomState(1234)
        init_idx = rng.choice(n_rows, size=n_clusters, replace=False)
        common_kwargs["init"] = X_host[init_idx]
    else:
        common_kwargs["init_size"] = n_rows

    host_model = cuml.KMeans(
        device_buffer_samples=device_buffer_samples, **common_kwargs
    )
    host_model.fit(X_host, sample_weight=sample_weight_host)

    dev_model = cuml.KMeans(device_buffer_samples=0, **common_kwargs)
    dev_model.fit(X_dev, sample_weight=sample_weight_dev)

    host_labels = cp.asnumpy(cp.asarray(host_model.labels_))
    dev_labels = cp.asnumpy(cp.asarray(dev_model.labels_))
    assert host_labels.shape == (n_rows,)
    assert dev_labels.shape == (n_rows,)

    if init_method == "explicit":
        np.testing.assert_allclose(
            cp.asnumpy(cp.asarray(host_model.cluster_centers_)),
            cp.asnumpy(cp.asarray(dev_model.cluster_centers_)),
            atol=0.1,
            rtol=0.1,
        )
    np.testing.assert_allclose(
        float(host_model.inertia_),
        float(dev_model.inertia_),
        rtol=1e-3,
    )
    assert adjusted_rand_score(dev_labels, host_labels) >= 0.97


def test_get_feature_names_out():
    X, _ = make_blobs(n_features=5)
    cu_model = cuml.KMeans().fit(X)
    sk_model = sklearn.cluster.KMeans().fit(X.get())
    res = cu_model.get_feature_names_out()
    sol = sk_model.get_feature_names_out()
    np.testing.assert_array_equal(res, sol)
