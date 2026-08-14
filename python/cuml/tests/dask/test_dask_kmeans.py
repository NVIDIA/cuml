# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import cudf
import cupy as cp
import dask.array as da
import dask_cudf
import numpy as np
import pytest
from raft_dask.common.comms import Comms
from sklearn.datasets import make_blobs as sk_make_blobs
from sklearn.metrics import adjusted_rand_score as sk_adjusted_rand_score

from cuml.dask.cluster import KMeans
from cuml.dask.common.dask_arr_utils import to_dask_cudf
from cuml.metrics import adjusted_rand_score
from cuml.testing.utils import quality_param, stress_param, unit_param


def _make_skewed_kmeans_data(client, n_small_rows=2):
    workers = list(client.scheduler_info()["workers"].keys())
    if len(workers) < 2:
        pytest.skip("Need at least 2 workers to test skewed partitions")

    n_clusters = 4
    X_np = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 10.0],
            [10.1, 10.0],
            [20.0, 20.0],
            [20.1, 20.0],
        ],
        dtype=np.float32,
    )

    worker_info = Comms(comms_p2p=False, client=client).worker_info(workers)
    rank0_worker = next(
        worker for worker, info in worker_info.items() if info["rank"] == 0
    )
    other_worker = next(worker for worker in workers if worker != rank0_worker)

    X_small = cudf.DataFrame(X_np[:n_small_rows])
    X_large = cudf.DataFrame(X_np[n_small_rows:])
    small_f = client.scatter(X_small, workers=[rank0_worker])
    large_f = client.scatter(X_large, workers=[other_worker])
    X = dask_cudf.from_delayed([small_f, large_f], meta=X_small.iloc[:0])

    return X, X_np, n_clusters


@pytest.mark.mg
@pytest.mark.parametrize(
    "nrows", [unit_param(1e3), quality_param(1e5), stress_param(5e6)]
)
@pytest.mark.parametrize("ncols", [10, 30])
@pytest.mark.parametrize(
    "nclusters", [unit_param(5), quality_param(10), stress_param(50)]
)
@pytest.mark.parametrize(
    "n_parts", [unit_param(None), quality_param(7), stress_param(50)]
)
@pytest.mark.parametrize("delayed_predict", [True, False])
@pytest.mark.parametrize("input_type", ["dataframe", "array"])
def test_end_to_end(
    nrows, ncols, nclusters, n_parts, delayed_predict, input_type, client
):
    from cuml.dask.cluster import KMeans
    from cuml.dask.datasets import make_blobs

    X, y = make_blobs(
        n_samples=int(nrows),
        n_features=ncols,
        centers=nclusters,
        n_parts=n_parts,
        cluster_std=0.01,
        random_state=10,
    )

    if input_type == "dataframe":
        X_train = to_dask_cudf(X)
        y_train = to_dask_cudf(y)
    elif input_type == "array":
        X_train, y_train = X, y

    model = KMeans(
        init="k-means||",
        n_clusters=nclusters,
        random_state=10,
        n_init="auto",
    )

    dask_fit_predict_labels = model.fit_predict(X_train)
    dask_predict_labels = model.predict(X_train, delayed=delayed_predict)

    n_workers = len(list(client.has_what().keys()))

    # Verifying we are grouping partitions. This should be changed soon.
    if n_parts is not None:
        parts_len = n_parts
    else:
        parts_len = n_workers

    if input_type == "dataframe":
        assert dask_predict_labels.npartitions == parts_len
        pred_labels = dask_predict_labels.compute().values
        fit_pred_labels = dask_fit_predict_labels.compute().values
        labels = y_train.compute().values
    elif input_type == "array":
        assert len(dask_predict_labels.chunks[0]) == parts_len
        pred_labels = cp.array(dask_predict_labels.compute())
        fit_pred_labels = cp.array(dask_fit_predict_labels.compute())
        labels = cp.squeeze(y_train.compute())

    assert pred_labels.shape[0] == nrows
    assert cp.max(pred_labels) == nclusters - 1
    assert cp.min(pred_labels) == 0

    # Assert fit_predict(X) and fit(X).predict(X) have same result
    cp.testing.assert_array_equal(pred_labels, fit_pred_labels)

    score = adjusted_rand_score(labels, pred_labels)

    assert 1.0 == score


@pytest.mark.mg
@pytest.mark.parametrize("nrows_per_part", [quality_param(1e7)])
@pytest.mark.parametrize("ncols", [quality_param(256)])
@pytest.mark.parametrize("nclusters", [quality_param(5)])
def test_large_data_no_overflow(nrows_per_part, ncols, nclusters, client):
    from cuml.dask.cluster import KMeans as cumlKMeans
    from cuml.dask.datasets import make_blobs

    n_parts = len(list(client.has_what().keys()))

    X, y = make_blobs(
        n_samples=nrows_per_part * n_parts,
        n_features=ncols,
        centers=nclusters,
        n_parts=n_parts,
        cluster_std=0.01,
        random_state=10,
    )

    X_train, y_train = X, y

    X.compute_chunk_sizes().persist()

    cumlModel = cumlKMeans(
        init="k-means||", n_clusters=nclusters, random_state=10, n_init="auto"
    )

    cumlModel.fit(X_train)
    n_predict = int(X_train.shape[0] / 4)
    cumlLabels = cumlModel.predict(X_train[:n_predict, :], delayed=False)

    cumlPred = cp.array(cumlLabels.compute())
    labels = cp.squeeze(y_train.compute()[:n_predict])

    print(str(cumlPred))
    print(str(labels))

    assert 1.0 == adjusted_rand_score(labels, cumlPred)


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [500])
@pytest.mark.parametrize("ncols", [5])
@pytest.mark.parametrize("nclusters", [3, 10])
@pytest.mark.parametrize("n_parts", [1, 5])
def test_weighted_kmeans(nrows, ncols, nclusters, n_parts, client):
    cluster_std = 10000.0
    from cuml.dask.cluster import KMeans as cumlKMeans
    from cuml.dask.datasets import make_blobs

    # Using fairly high variance between points in clusters
    wt = cp.array([0.00001 for j in range(nrows)])

    bound = nclusters * 100000

    # Open the space really large
    centers = cp.random.uniform(-bound, bound, size=(nclusters, ncols))

    X_cudf, y = make_blobs(
        n_samples=nrows,
        n_features=ncols,
        centers=centers,
        n_parts=n_parts,
        cluster_std=cluster_std,
        shuffle=False,
        verbose=False,
        random_state=10,
    )

    # Choose one sample from each label and increase its weight
    for i in range(nclusters):
        wt[cp.argmax(cp.array(y.compute()) == i).item()] = 5000.0

    cumlModel = cumlKMeans(
        verbose=0,
        init="k-means||",
        n_clusters=nclusters,
        random_state=10,
        n_init="auto",
    )

    chunk_parts = int(nrows / n_parts)
    sample_weights = da.from_array(wt, chunks=(chunk_parts,))
    cumlModel.fit(X_cudf, sample_weight=sample_weights)

    X = X_cudf.compute()

    labels_ = cumlModel.predict(X_cudf).compute()
    cluster_centers_ = cumlModel.cluster_centers_

    for i in range(nrows):
        label = labels_[i]
        actual_center = cluster_centers_[label]

        diff = sum(abs(X[i] - actual_center))

        # The large weight should be the centroid
        if wt[i] > 1.0:
            assert diff < 1.0

        # Otherwise it should be pretty far away
        else:
            assert diff > 1000.0


@pytest.mark.mg
@pytest.mark.parametrize(
    "nrows", [unit_param(5e3), quality_param(1e5), stress_param(1e6)]
)
@pytest.mark.parametrize(
    "ncols", [unit_param(10), quality_param(30), stress_param(50)]
)
@pytest.mark.parametrize("nclusters", [1, 10, 30])
@pytest.mark.parametrize(
    "n_parts", [unit_param(None), quality_param(7), stress_param(50)]
)
@pytest.mark.parametrize("input_type", ["dataframe", "array"])
def test_transform(nrows, ncols, nclusters, n_parts, input_type, client):
    from cuml.dask.cluster import KMeans as cumlKMeans
    from cuml.dask.datasets import make_blobs

    X, y = make_blobs(
        n_samples=int(nrows),
        n_features=ncols,
        centers=nclusters,
        n_parts=n_parts,
        cluster_std=0.01,
        shuffle=False,
        random_state=10,
    )
    y = y.astype("int64")

    if input_type == "dataframe":
        X_train = to_dask_cudf(X)
        y_train = to_dask_cudf(y)
        labels = y_train.compute().values
    elif input_type == "array":
        X_train, y_train = X, y
        labels = cp.squeeze(y_train.compute())

    cumlModel = cumlKMeans(
        init="k-means||",
        n_clusters=nclusters,
        random_state=10,
        n_init="auto",
    )

    cumlModel.fit(X_train)

    xformed = cumlModel.transform(X_train).compute()
    if input_type == "dataframe":
        xformed = cp.array(
            xformed if len(xformed.shape) == 1 else xformed.to_cupy()
        )

    if nclusters == 1:
        # series shape is (nrows,) not (nrows, 1) but both are valid
        # and equivalent for this test
        assert xformed.shape in [(nrows, nclusters), (nrows,)]
    else:
        assert xformed.shape == (nrows, nclusters)

    # The argmin of the transformed values should be equal to the labels
    # reshape is a quick manner of dealing with (nrows,) is not (nrows, 1)
    xformed_labels = cp.argmin(
        xformed.reshape((int(nrows), int(nclusters))), axis=1
    )

    assert sk_adjusted_rand_score(
        cp.asnumpy(labels), cp.asnumpy(xformed_labels)
    )


@pytest.mark.mg
@pytest.mark.parametrize(
    "nrows", [unit_param(1e3), quality_param(1e5), stress_param(5e6)]
)
@pytest.mark.parametrize("ncols", [10, 30])
@pytest.mark.parametrize(
    "nclusters", [unit_param(5), quality_param(10), stress_param(50)]
)
@pytest.mark.parametrize(
    "n_parts", [unit_param(None), quality_param(7), stress_param(50)]
)
@pytest.mark.parametrize("input_type", ["dataframe", "array"])
def test_score(nrows, ncols, nclusters, n_parts, input_type, client):
    from cuml.dask.cluster import KMeans as cumlKMeans
    from cuml.dask.datasets import make_blobs

    X, y = make_blobs(
        n_samples=int(nrows),
        n_features=ncols,
        centers=nclusters,
        n_parts=n_parts,
        cluster_std=0.01,
        shuffle=False,
        random_state=10,
    )

    if input_type == "dataframe":
        X_train = to_dask_cudf(X)
        y_train = to_dask_cudf(y)
        y = y_train
    elif input_type == "array":
        X_train, y_train = X, y

    cumlModel = cumlKMeans(
        init="k-means||",
        n_clusters=nclusters,
        random_state=10,
        n_init="auto",
    )

    cumlModel.fit(X_train)

    actual_score = cumlModel.score(X_train)

    local_model = cumlModel.get_combined_model()
    expected_score = local_model.score(X_train.compute())

    np.testing.assert_allclose(actual_score, expected_score, atol=9e-3)
    # The score is -1 * inertia. Scoring the training data should result in
    # -1 * model.inertia_
    np.testing.assert_allclose(actual_score, -cumlModel.inertia_, atol=9e-3)


@pytest.mark.mg
def test_nclusters_exceeds_n_samples(client):
    """Test that n_clusters > n_samples raises a clear ValueError."""
    from cuml.dask.cluster import KMeans
    from cuml.dask.datasets import make_blobs

    # Use fewer data points than clusters
    n_clusters = 11
    n_samples = 10

    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=5,
        centers=5,
        n_parts=2,
        random_state=10,
    )

    model = KMeans(
        n_clusters=n_clusters,
        random_state=10,
    )

    with pytest.raises(
        ValueError, match="n_samples=10 should be >= n_clusters=11"
    ):
        model.fit(X)


@pytest.mark.mg
@pytest.mark.parametrize("n_clusters", [0, -1])
def test_invalid_nclusters_raises(client, n_clusters):
    from cuml.dask.datasets import make_blobs

    X, _ = make_blobs(
        n_samples=10,
        n_features=5,
        centers=2,
        n_parts=2,
        random_state=10,
    )

    model = KMeans(n_clusters=n_clusters, random_state=10)

    with pytest.raises(
        ValueError,
        match=f"n_clusters={n_clusters} should be a positive integer",
    ):
        model.fit(X)


@pytest.mark.mg
@pytest.mark.parametrize(
    "kwargs, match",
    [
        (
            {"init": "k-means++"},
            r"init='k-means\+\+' is not supported for KMeansMG",
        ),
        (
            {"oversampling_factor": 0},
            "oversampling_factor=0 is not supported for KMeansMG",
        ),
    ],
)
def test_unsupported_mg_init_params_raise(client, kwargs, match):
    from cuml.dask.datasets import make_blobs

    X, _ = make_blobs(
        n_samples=10,
        n_features=5,
        centers=2,
        n_parts=2,
        random_state=10,
    )

    model = KMeans(n_clusters=2, random_state=10, **kwargs)

    with pytest.raises(ValueError, match=match):
        model.fit(X)


@pytest.mark.mg
def test_partition_with_fewer_rows_than_clusters(client):
    """KMeansMG validates n_clusters against global, not per-rank, rows."""
    X, X_np, n_clusters = _make_skewed_kmeans_data(client)

    init = X_np[[0, 2, 4, 5]]
    model = KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=1,
        random_state=10,
    )

    model.fit(X)

    assert model.cluster_centers_.shape == (n_clusters, X_np.shape[1])
    assert len(model.labels_.compute()) == X_np.shape[0]


@pytest.mark.mg
def test_random_init_partition_too_small_raises(client):
    X, _, n_clusters = _make_skewed_kmeans_data(client, n_small_rows=1)

    model = KMeans(
        n_clusters=n_clusters,
        init="random",
        random_state=10,
    )

    with pytest.raises(
        ValueError,
        match="init='random' requires rank 0 to sample up to 2",
    ):
        model.fit(X)


@pytest.mark.mg
@pytest.mark.parametrize("n_parts", [2, 8])
@pytest.mark.parametrize("device_buffer_samples", [0, 64])
@pytest.mark.parametrize("weighted", [False, True])
def test_out_of_core_host_fit(
    client, n_parts, device_buffer_samples, weighted
):
    n_rows, n_cols, n_clusters = 4000, 16, 5
    X_np, y_np = sk_make_blobs(
        n_samples=n_rows,
        n_features=n_cols,
        centers=n_clusters,
        cluster_std=0.1,
        random_state=10,
    )
    X_np = X_np.astype(np.float32)

    chunk = int(np.ceil(n_rows / n_parts))
    X = da.from_array(X_np, chunks=(chunk, n_cols))

    fit_kwargs = {}
    if weighted:
        rng = np.random.RandomState(42)
        wt = rng.uniform(0.5, 2.0, size=n_rows).astype(np.float32)
        fit_kwargs["sample_weight"] = da.from_array(wt, chunks=(chunk,))

    model = KMeans(
        init="k-means||",
        n_clusters=n_clusters,
        random_state=10,
        n_init="auto",
        init_size=n_rows,
        device_buffer_samples=device_buffer_samples,
    )

    labels = model.fit_predict(X, **fit_kwargs).compute()
    labels = cp.asnumpy(cp.asarray(labels)).reshape(-1)

    assert labels.shape[0] == n_rows
    assert model.cluster_centers_.shape == (n_clusters, n_cols)
    assert sk_adjusted_rand_score(y_np, labels) >= 0.99


@pytest.mark.mg
@pytest.mark.parametrize("device_buffer_samples", [0, 128])
def test_out_of_core_host_matches_device(client, device_buffer_samples):
    """The host out-of-core fit and the in-core device fit should agree on
    identical data (same clustering and inertia)."""
    from sklearn.datasets import make_blobs as sk_make_blobs
    from sklearn.metrics import adjusted_rand_score as sk_adjusted_rand_score

    n_rows, n_cols, n_clusters, n_parts = 3000, 8, 6, 6
    X_np, _ = sk_make_blobs(
        n_samples=n_rows,
        n_features=n_cols,
        centers=n_clusters,
        cluster_std=0.1,
        random_state=10,
    )
    X_np = X_np.astype(np.float32)
    chunk = int(np.ceil(n_rows / n_parts))

    rng = np.random.RandomState(10)
    wt = rng.uniform(0.5, 2.0, size=n_rows).astype(np.float32)

    common = dict(
        init="k-means||",
        n_clusters=n_clusters,
        random_state=10,
        n_init="auto",
    )

    X_host = da.from_array(X_np, chunks=(chunk, n_cols))
    wt_host = da.from_array(wt, chunks=(chunk,))
    host_model = KMeans(device_buffer_samples=device_buffer_samples, **common)
    host_labels = cp.asnumpy(
        cp.asarray(
            host_model.fit_predict(X_host, sample_weight=wt_host).compute()
        )
    ).reshape(-1)

    X_dev = da.from_array(cp.asarray(X_np), chunks=(chunk, n_cols))
    wt_dev = da.from_array(cp.asarray(wt), chunks=(chunk,))
    dev_model = KMeans(**common)
    dev_labels = cp.asnumpy(
        cp.asarray(
            dev_model.fit_predict(X_dev, sample_weight=wt_dev).compute()
        )
    ).reshape(-1)

    assert sk_adjusted_rand_score(host_labels, dev_labels) >= 0.99
    np.testing.assert_allclose(
        float(host_model.inertia_), float(dev_model.inertia_), rtol=1e-2
    )
