#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import gc
import math

import cudf
import cupy as cp
import cupyx
import numpy as np
import pytest
import sklearn
import sklearn.datasets
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import isspmatrix_csr
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors as skKNN

import cuml
from cuml.datasets import make_blobs
from cuml.internals import logger  # noqa: F401
from cuml.metrics import pairwise_distances as cuPW
from cuml.neighbors import NearestNeighbors as cuKNN
from cuml.testing.utils import (
    array_equal,
    quality_param,
    stress_param,
    unit_param,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:((.|\n)*)#4020((.|\n)*):UserWarning:cuml[.*]"
)


def predict(neigh_ind, _y, n_neighbors):
    import scipy.stats as stats

    neigh_ind = neigh_ind.astype(np.int32)
    if isinstance(_y, cp.ndarray):
        _y = _y.get()
    if isinstance(neigh_ind, cp.ndarray):
        neigh_ind = neigh_ind.get()

    ypred, count = stats.mode(_y[neigh_ind], axis=1)
    return ypred.ravel(), count.ravel() * 1.0 / n_neighbors


def valid_metrics(algo="brute", cuml_algo=None):
    cuml_algo = algo if cuml_algo is None else cuml_algo
    cuml_metrics = cuml.neighbors.VALID_METRICS[cuml_algo]
    sklearn_metrics = sklearn.neighbors.VALID_METRICS[algo]
    ret = [value for value in cuml_metrics if value in sklearn_metrics]
    ret.sort()
    ret.remove("haversine")  # This is tested on its own
    return ret


def valid_metrics_sparse(algo="brute", cuml_algo=None):
    """
    The list of sparse prims in scikit-learn / scipy does not
    include sparse inputs for all of the metrics we support in cuml
    (even metrics which are implicitly sparse, such as jaccard and dice,
    which accume boolean inputs). To maintain high test coverage for all
    metrics supported by Scikit-learn, we take the union of both
    dense and sparse metrics. This way, a sparse input can just be converted
    to dense form for Scikit-learn.
    """
    cuml_algo = algo if cuml_algo is None else cuml_algo
    cuml_metrics = cuml.neighbors.VALID_METRICS_SPARSE[cuml_algo]
    sklearn_metrics = set(sklearn.neighbors.VALID_METRICS_SPARSE[algo])
    sklearn_metrics.update(sklearn.neighbors.VALID_METRICS[algo])
    ret = [value for value in cuml_metrics if value in sklearn_metrics]
    ret.sort()
    return ret


def metric_p_combinations():
    for metric in valid_metrics():
        yield metric, 2
        if metric in ("minkowski", "lp"):
            yield metric, 3


@pytest.mark.parametrize("datatype", ["dataframe", "numpy"])
@pytest.mark.parametrize("metric_p", list(metric_p_combinations()))
@pytest.mark.parametrize("nrows", [1000, stress_param(10000)])
def test_self_neighboring(datatype, metric_p, nrows):
    """Test that searches using an indexed vector itself return sensible
    results for that vector

    For L2-derived metrics, this specifically exercises the slow high-precision
    mode used to correct for approximation errors in L2 computation during NN
    searches.
    """
    ncols = 1000
    n_clusters = 10
    n_neighbors = 3

    metric, p = metric_p

    X, y = make_blobs(
        n_samples=nrows, centers=n_clusters, n_features=ncols, random_state=0
    )

    if datatype == "dataframe":
        X = cudf.DataFrame(X)

    knn_cu = cuKNN(metric=metric, n_neighbors=n_neighbors)
    knn_cu.fit(X)
    neigh_dist, neigh_ind = knn_cu.kneighbors(
        X,
        n_neighbors=n_neighbors,
        return_distance=True,
        two_pass_precision=True,
    )

    if datatype == "dataframe":
        assert isinstance(neigh_ind, cudf.DataFrame)
        neigh_ind = neigh_ind.to_numpy()
        neigh_dist = neigh_dist.to_numpy()
    else:
        assert isinstance(neigh_ind, cp.ndarray)
        neigh_ind = neigh_ind.get()
        neigh_dist = neigh_dist.get()

    neigh_ind = neigh_ind[:, 0]
    neigh_dist = neigh_dist[:, 0]

    assert_array_equal(
        neigh_ind,
        np.arange(0, neigh_dist.shape[0]),
    )
    assert_allclose(
        neigh_dist,
        np.zeros(neigh_dist.shape, dtype=neigh_dist.dtype),
        atol=1e-4,
    )


@pytest.mark.parametrize(
    "nrows,ncols,n_neighbors,n_clusters",
    [
        (500, 128, 10, 2),
        (4301, 128, 10, 2),
        (1000, 128, 50, 2),
        (2233, 1024, 2, 10),
        stress_param(10000, 1024, 50, 10),
    ],
)
@pytest.mark.parametrize(
    "algo,datatype",
    [("brute", "dataframe"), ("ivfflat", "numpy"), ("ivfpq", "dataframe")],
)
def test_neighborhood_predictions(
    nrows, ncols, n_neighbors, n_clusters, datatype, algo
):
    X, y = make_blobs(
        n_samples=nrows, centers=n_clusters, n_features=ncols, random_state=0
    )

    if datatype == "dataframe":
        X = cudf.DataFrame(X)

    knn_cu = cuKNN(algorithm=algo)
    knn_cu.fit(X)
    neigh_ind = knn_cu.kneighbors(
        X, n_neighbors=n_neighbors, return_distance=False
    )
    del knn_cu
    gc.collect()

    if datatype == "dataframe":
        assert isinstance(neigh_ind, cudf.DataFrame)
        neigh_ind = neigh_ind.to_numpy()
    else:
        assert isinstance(neigh_ind, cp.ndarray)

    labels, probs = predict(neigh_ind, y, n_neighbors)

    assert array_equal(labels, y)


@pytest.mark.parametrize(
    "nlist,nrows,ncols,n_neighbors",
    [
        (4, 10000, 128, 8),
        (8, 100, 512, 8),
        (8, 10000, 512, 16),
    ],
)
def test_ivfflat_pred(nrows, ncols, n_neighbors, nlist):
    algo_params = {"nlist": nlist, "nprobe": nlist * 0.5}

    X, y = make_blobs(
        n_samples=nrows, centers=5, n_features=ncols, random_state=0
    )

    knn_cu = cuKNN(algorithm="ivfflat", algo_params=algo_params)
    knn_cu.fit(X)
    neigh_ind = knn_cu.kneighbors(
        X, n_neighbors=n_neighbors, return_distance=False
    )
    del knn_cu
    gc.collect()

    labels, probs = predict(neigh_ind, y, n_neighbors)

    assert array_equal(labels, y)


@pytest.mark.parametrize("nlist", [8])
@pytest.mark.parametrize("M", [32])
@pytest.mark.parametrize("n_bits", [4])
@pytest.mark.parametrize("usePrecomputedTables", [False, True])
@pytest.mark.parametrize("nrows", [4000])
@pytest.mark.parametrize("ncols", [64, 512])
@pytest.mark.parametrize("n_neighbors", [8])
def test_ivfpq_pred(
    nrows, ncols, n_neighbors, nlist, M, n_bits, usePrecomputedTables
):
    if ncols == 512:
        pytest.skip("https://github.com/NVIDIA/cuml/issues/5603")
    algo_params = {
        "nlist": nlist,
        "nprobe": int(nlist * 0.2),
        "M": M,
        "n_bits": n_bits,
        "usePrecomputedTables": usePrecomputedTables,
    }

    X, y = make_blobs(
        n_samples=nrows, centers=5, n_features=ncols, random_state=0
    )

    knn_cu = cuKNN(algorithm="ivfpq", algo_params=algo_params)
    knn_cu.fit(X)

    neigh_ind = knn_cu.kneighbors(
        X, n_neighbors=n_neighbors, return_distance=False
    )

    del knn_cu
    gc.collect()

    labels, probs = predict(neigh_ind, y, n_neighbors)
    assert array_equal(labels, y)


@pytest.mark.parametrize(
    "algo, metric",
    [
        (algo, metric)
        for algo in ["brute", "ivfflat", "ivfpq"]
        for metric in [
            "l2",
            "euclidean",
            "sqeuclidean",
            "cosine",
            "correlation",
        ]
        if metric in cuml.neighbors.VALID_METRICS[algo]
    ],
)
def test_ann_distances_metrics(algo, metric, request):
    X, y = make_blobs(n_samples=500, centers=2, n_features=128, random_state=0)

    cu_knn = cuKNN(algorithm=algo, metric=metric)
    cu_knn.fit(X)
    cu_dist, cu_ind = cu_knn.kneighbors(
        X, n_neighbors=10, return_distance=True
    )
    del cu_knn
    gc.collect()

    X = X.get()
    sk_knn = skKNN(metric=metric)
    sk_knn.fit(X)
    sk_dist, sk_ind = sk_knn.kneighbors(
        X, n_neighbors=10, return_distance=True
    )
    request.applymarker(
        pytest.mark.xfail(
            not (algo == "brute" and metric in ("cosine", "correlation")),
            reason=f"arrays not equal with {algo=} and {metric=}",
        )
    )
    assert bool(array_equal(sk_dist, cu_dist))


def test_return_dists():
    n_samples = 50
    n_feats = 50
    k = 5

    X, y = make_blobs(n_samples=n_samples, n_features=n_feats, random_state=0)

    knn_cu = cuKNN()
    knn_cu.fit(X)

    ret = knn_cu.kneighbors(X, k, return_distance=False)
    assert not isinstance(ret, tuple)
    assert ret.shape == (n_samples, k)

    ret = knn_cu.kneighbors(X, k, return_distance=True)
    assert isinstance(ret, tuple)
    assert len(ret) == 2


@pytest.mark.parametrize("input_type", ["dataframe", "ndarray"])
@pytest.mark.parametrize(
    "nrows", [unit_param(500), quality_param(5000), stress_param(70000)]
)
@pytest.mark.parametrize("n_feats", [unit_param(3), stress_param(1000)])
@pytest.mark.parametrize("k", [unit_param(3), stress_param(50)])
@pytest.mark.parametrize("metric", valid_metrics())
def test_knn_separate_index_search(input_type, nrows, n_feats, k, metric):
    X, _ = make_blobs(n_samples=nrows, n_features=n_feats, random_state=0)

    X_index = X[:100]
    X_search = X[101:]

    p = 5  # Testing 5-norm of the minkowski metric only
    knn_sk = skKNN(metric=metric, p=p)  # Testing
    knn_sk.fit(X_index.get())
    D_sk, I_sk = knn_sk.kneighbors(X_search.get(), k)

    X_orig = X_index

    if input_type == "dataframe":
        X_index = cudf.DataFrame(X_index)
        X_search = cudf.DataFrame(X_search)

    knn_cu = cuKNN(metric=metric, p=p)
    knn_cu.fit(X_index)
    D_cuml, I_cuml = knn_cu.kneighbors(X_search, k)

    if input_type == "dataframe":
        assert isinstance(D_cuml, cudf.DataFrame)
        assert isinstance(I_cuml, cudf.DataFrame)
        D_cuml_np = D_cuml.to_numpy()
        I_cuml_np = I_cuml.to_numpy()
    else:
        assert isinstance(D_cuml, cp.ndarray)
        assert isinstance(I_cuml, cp.ndarray)
        D_cuml_np = D_cuml.get()
        I_cuml_np = I_cuml.get()

    with cuml.using_output_type("numpy"):
        # Assert the cuml model was properly reverted
        np.testing.assert_allclose(
            knn_cu._fit_X, X_orig.get(), atol=1e-3, rtol=1e-3
        )

    if metric == "braycurtis":
        diff = D_cuml_np - D_sk
        # Braycurtis has a few differences, but this is computed by FAISS.
        # So long as the indices all match below, the small discrepancy
        # should be okay.
        assert len(diff[diff > 1e-2]) / X_search.shape[0] < 0.06
    else:
        np.testing.assert_allclose(D_cuml_np, D_sk, atol=1e-3, rtol=1e-3)
    assert I_cuml_np.all() == I_sk.all()


@pytest.mark.parametrize("input_type", ["dataframe", "ndarray"])
@pytest.mark.parametrize("nrows", [unit_param(500), stress_param(70000)])
@pytest.mark.filterwarnings(
    "ignore:algorithm='rbc' requires sqrt\\(n_samples\\) >= n_neighbors.*:UserWarning"
)
@pytest.mark.parametrize("n_feats", [unit_param(3), stress_param(1000)])
@pytest.mark.parametrize(
    "k", [unit_param(3), unit_param(35), stress_param(50)]
)
@pytest.mark.parametrize("metric", valid_metrics())
def test_knn_x_none(input_type, nrows, n_feats, k, metric):
    X, _ = make_blobs(n_samples=nrows, n_features=n_feats, random_state=0)

    p = 5  # Testing 5-norm of the minkowski metric only
    knn_sk = skKNN(metric=metric, p=p)  # Testing
    knn_sk.fit(X.get())
    D_sk, I_sk = knn_sk.kneighbors(X=None, n_neighbors=k)

    X_orig = X

    if input_type == "dataframe":
        X = cudf.DataFrame(X)

    knn_cu = cuKNN(metric=metric, p=p, output_type="numpy")
    knn_cu.fit(X)
    D_cuml, I_cuml = knn_cu.kneighbors(X=None, n_neighbors=k)

    # Assert the cuml model was properly reverted
    cp.testing.assert_allclose(knn_cu._fit_X, X_orig, atol=1e-5, rtol=1e-4)

    # Allow a max relative diff of 10% and absolute diff of 1%
    cp.testing.assert_allclose(D_cuml, D_sk, atol=5e-2, rtol=1e-1)
    assert I_cuml.all() == I_sk.all()


def test_knn_fit_twice():
    """
    Test that fitting a model twice does not fail.
    This is necessary since the NearestNeighbors class
    needs to free Cython allocated heap memory when
    fit() is called more than once.
    """

    n_samples = 1000
    n_feats = 50
    k = 5

    X, y = make_blobs(n_samples=n_samples, n_features=n_feats, random_state=0)

    knn_cu = cuKNN()
    knn_cu.fit(X)
    knn_cu.fit(X)

    knn_cu.kneighbors(X, k)

    del knn_cu


def check_knn_graph(
    X, k, mode, metric, p, input_type, output_type, as_instance, include_self
):
    if as_instance:
        sparse_sk = sklearn.neighbors.kneighbors_graph(
            X.get(),
            k,
            mode=mode,
            metric=metric,
            p=p,
            include_self=include_self,
        )
    else:
        knn_sk = skKNN(metric=metric, p=p)
        knn_sk.fit(X.get())
        sparse_sk = knn_sk.kneighbors_graph(X.get(), k, mode=mode)

    if input_type == "dataframe":
        X = cudf.DataFrame(X)

    with cuml.using_output_type(output_type):
        if as_instance:
            sparse_cu = cuml.neighbors.kneighbors_graph(
                X, k, mode=mode, metric=metric, p=p, include_self=include_self
            )
        else:
            knn_cu = cuKNN(metric=metric, p=p)
            knn_cu.fit(X)
            sparse_cu = knn_cu.kneighbors_graph(X, k, mode=mode)

    assert np.array_equal(sparse_sk.data.shape, sparse_cu.data.shape)
    assert np.array_equal(sparse_sk.indices.shape, sparse_cu.indices.shape)
    assert np.array_equal(sparse_sk.indptr.shape, sparse_cu.indptr.shape)
    assert np.array_equal(sparse_sk.toarray().shape, sparse_cu.toarray().shape)

    if output_type == "cupy":
        assert np.allclose(
            sparse_sk.toarray(),
            np.asarray(sparse_cu.toarray().get()),
            atol=1e-4,
        )

    if output_type == "cupy" or output_type is None:
        assert cupyx.scipy.sparse.isspmatrix_csr(sparse_cu)
    else:
        assert isspmatrix_csr(sparse_cu)


@pytest.mark.parametrize(
    "input_type,mode,output_type,as_instance",
    [
        ("dataframe", "connectivity", "cupy", True),
        ("dataframe", "connectivity", None, True),
        ("dataframe", "distance", "numpy", True),
        ("ndarray", "connectivity", "cupy", False),
        ("ndarray", "distance", "numpy", False),
    ],
)
@pytest.mark.parametrize("nrows", [unit_param(100), stress_param(1000)])
@pytest.mark.parametrize("n_feats", [unit_param(5), stress_param(100)])
@pytest.mark.parametrize("p", [2, 5])
@pytest.mark.parametrize(
    "k", [unit_param(3), unit_param(35), stress_param(30)]
)
@pytest.mark.parametrize("metric", valid_metrics())
def test_knn_graph(
    input_type, mode, output_type, as_instance, nrows, n_feats, p, k, metric
):
    X, _ = make_blobs(n_samples=nrows, n_features=n_feats, random_state=0)

    check_knn_graph(
        X, k, mode, metric, p, input_type, output_type, as_instance, "auto"
    )


def test_knn_graph_duplicate_point():
    X = cp.array([[1, 5], [1, 5], [7, 3], [9, 6], [10, 1]])

    check_knn_graph(
        X, 2, "connectivity", "euclidean", 2, "ndarray", "cupy", True, False
    )


@pytest.mark.parametrize("algorithm", ["brute", "rbc", "ivfpq", "ivfflat"])
def test_knn_graph_algorithm(algorithm):
    n_features = 3 if algorithm == "rbc" else 5
    X, _ = make_blobs(n_samples=100, n_features=n_features, random_state=42)

    sk_knn = skKNN(n_neighbors=5).fit(X.get())
    cu_knn = cuKNN(n_neighbors=5, algorithm=algorithm).fit(X)

    cu_graph = cu_knn.kneighbors_graph(X)
    assert cupyx.scipy.sparse.isspmatrix_csr(cu_graph)
    np.testing.assert_array_equal(cu_graph.sum(axis=1).get(), 5)

    if algorithm in ("brute", "rbc"):
        # These algorithms are exact, can check for exact equality
        sk_graph = sk_knn.kneighbors_graph(X.get())
        np.testing.assert_array_equal(
            sk_graph.toarray(),
            cu_graph.toarray().get(),
        )
    else:
        # Approximate algorithms may have differences. Instead of checking
        # exact equality, here we check that the graph found for the nearest 5
        # neighbors is a subset of the graph found for nearest 20.
        sk_graph = sk_knn.kneighbors_graph(X.get(), 20)
        assert ((sk_graph - cu_graph.get()) < 0).sum() == 0


@pytest.mark.parametrize("n_features", [2, 3, 5])
@pytest.mark.parametrize("radius", [2.5, None])
@pytest.mark.parametrize("self_query", [True, False])
def test_radius_neighbors_graph(n_features, radius, self_query):
    X, _ = make_blobs(n_samples=500, n_features=n_features, random_state=42)
    X_train, X_query = X[:400], X[400:]

    sk_model = skKNN(radius=1.5).fit(X_train.get())
    cu_model = cuKNN(radius=1.5).fit(X_train)

    sk_graph = sk_model.radius_neighbors_graph(
        None if self_query else X_query.get(), radius=radius
    )
    cu_graph = cu_model.radius_neighbors_graph(
        None if self_query else X_query, radius=radius
    )

    assert cupyx.scipy.sparse.isspmatrix_csr(cu_graph)
    np.testing.assert_array_equal(
        sk_graph.toarray(),
        cu_graph.toarray().get(),
    )


def test_radius_neighbors_graph_errors():
    X, _ = make_blobs(n_samples=100, random_state=42)
    X_sparse = cupyx.scipy.sparse.random(
        100, 5, format="csr", density=0.5, random_state=42
    )

    # Unsupported sparse inputs
    model = cuKNN().fit(X)
    model_sparse = cuKNN().fit(X_sparse)

    with pytest.raises(TypeError, match="doesn't support sparse inputs"):
        model.radius_neighbors_graph(X_sparse)

    with pytest.raises(TypeError, match="doesn't support sparse inputs"):
        model_sparse.radius_neighbors_graph(X)

    # Invalid radius
    with pytest.raises(ValueError, match="Expected `radius > 0`, got -2"):
        model.radius_neighbors_graph(radius=-2)

    # Unsupported metric
    model = cuKNN(metric="linf").fit(X)
    with pytest.raises(ValueError, match="doesn't support metric='linf'"):
        model.radius_neighbors_graph()


@pytest.mark.parametrize(
    "distance_dims", [("euclidean", 2), ("euclidean", 3), ("haversine", 2)]
)
@pytest.mark.parametrize("n_neighbors", [4, 25])
@pytest.mark.parametrize("nrows", [unit_param(10000), stress_param(70000)])
def test_nearest_neighbors_rbc(distance_dims, n_neighbors, nrows):
    distance, dims = distance_dims

    X, y = make_blobs(
        n_samples=nrows,
        centers=25,
        shuffle=True,
        n_features=dims,
        cluster_std=3.0,
        random_state=42,
    )

    knn_cu = cuKNN(metric=distance, algorithm="rbc")
    knn_cu.fit(X)

    query_rows = int(nrows / 2)

    rbc_d, rbc_i = knn_cu.kneighbors(
        X[:query_rows, :], n_neighbors=n_neighbors
    )

    if distance == "euclidean":
        # Need to use unexpanded euclidean distance
        pw_dists = cuPW(X, metric="l2")
        brute_i = cp.argsort(pw_dists, axis=1)[:query_rows, :n_neighbors]
        brute_d = cp.sort(pw_dists, axis=1)[:query_rows, :n_neighbors]
    else:
        knn_cu_brute = cuKNN(metric=distance, algorithm="brute")
        knn_cu_brute.fit(X)

        brute_d, brute_i = knn_cu_brute.kneighbors(
            X[:query_rows, :], n_neighbors=n_neighbors
        )

    cp.testing.assert_allclose(brute_d, rbc_d, atol=1e-3, rtol=1e-3)

    # All the distances match so allow a couple mismatched indices
    # through from potential non-determinism in exact matching
    # distances
    assert (
        len(brute_i[brute_i != rbc_i]) <= 3 if distance != "haversine" else 10
    )


@pytest.mark.parametrize("metric", valid_metrics_sparse())
@pytest.mark.parametrize(
    "nrows,ncols,density,n_neighbors,batch_size_index,batch_size_query",
    [
        (1, 10, 0.8, 1, 10, 10),
        (10, 35, 0.8, 4, 10, 20000),
        (40, 35, 0.5, 4, 20000, 10),
        (35, 35, 0.8, 4, 20000, 20000),
    ],
)
@pytest.mark.filterwarnings("ignore:(.*)converted(.*)::")
def test_nearest_neighbors_sparse(
    metric,
    nrows,
    ncols,
    density,
    n_neighbors,
    batch_size_index,
    batch_size_query,
):
    if nrows == 1 and n_neighbors > 1:
        return

    a = cupyx.scipy.sparse.random(
        nrows, ncols, format="csr", density=density, random_state=35
    )
    b = cupyx.scipy.sparse.random(
        nrows, ncols, format="csr", density=density, random_state=38
    )

    if metric == "jaccard":
        a = a.astype("bool").astype("float32")
        b = b.astype("bool").astype("float32")

    nn = cuKNN(
        metric=metric,
        p=2.0,
        n_neighbors=n_neighbors,
        algorithm="brute",
        output_type="numpy",
        algo_params={
            "batch_size_index": batch_size_index,
            "batch_size_query": batch_size_query,
        },
    )
    nn.fit(a)

    cuD, cuI = nn.kneighbors(b)

    if metric not in sklearn.neighbors.VALID_METRICS_SPARSE["brute"]:
        a = a.todense()
        b = b.todense()

    sknn = skKNN(
        metric=metric,
        p=2.0,
        n_neighbors=n_neighbors,
        algorithm="brute",
        n_jobs=-1,
    )
    sk_X = a.get()
    sknn.fit(sk_X)

    skD, skI = sknn.kneighbors(b.get())

    # For some reason, this will occasionally fail w/ a single
    # mismatched element in CI. Allowing the single mismatch for now.
    cp.testing.assert_allclose(cuD, skD, atol=1e-5, rtol=1e-5)

    # Jaccard & Chebyshev have a high potential for mismatched indices
    # due to duplicate distances. We can ignore the indices in this case.
    if metric not in ["jaccard", "chebyshev"]:
        # The actual neighbors returned in the presence of duplicate distances
        # is non-deterministic. If we got to this point, the distances all
        # match between cuml and sklearn. We set a reasonable threshold
        # (.5% in this case) to allow differences from non-determinism.
        diffs = abs(cuI - skI)
        assert (len(diffs[diffs > 0]) / len(np.ravel(skI))) <= 0.005


@pytest.mark.parametrize("n_neighbors", [1, 5, 6])
def test_haversine(n_neighbors):
    hoboken_nj = [40.745255, -74.034775]
    port_hueneme_ca = [34.155834, -119.202789]
    auburn_ny = [42.933334, -76.566666]
    league_city_tx = [29.499722, -95.089722]
    tallahassee_fl = [30.455000, -84.253334]
    aurora_il = [41.763889, -88.29001]

    data = np.array(
        [
            hoboken_nj,
            port_hueneme_ca,
            auburn_ny,
            league_city_tx,
            tallahassee_fl,
            aurora_il,
        ]
    )

    data = data * math.pi / 180

    pw_dists = pairwise_distances(data, metric="haversine")

    cunn = cuKNN(
        metric="haversine", n_neighbors=n_neighbors, algorithm="brute"
    )

    dists, inds = cunn.fit(data).kneighbors(data)

    argsort = np.argsort(pw_dists, axis=1)

    for i in range(pw_dists.shape[0]):
        cpu_ordered = pw_dists[i, argsort[i]]
        cp.testing.assert_allclose(
            cpu_ordered[:n_neighbors], dists[i], atol=1e-4, rtol=1e-4
        )


@pytest.mark.xfail(raises=RuntimeError)
def test_haversine_fails_high_dimensions():
    data = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])

    cunn = cuKNN(metric="haversine", n_neighbors=2, algorithm="brute")

    cunn.fit(data).kneighbors(data)


def test_n_jobs_parameter_passthrough():
    """Check that `n_jobs` is supported for compatibility with imbalanced-learn"""
    cunn = cuKNN(n_jobs=1)
    assert cunn.n_jobs == 1
    cunn.set_params(n_jobs=12)
    assert cunn.n_jobs == 12


@pytest.mark.parametrize(
    "metric, p, effective_metric",
    [
        ("minkowski", 1, "manhattan"),
        ("minkowski", 2, "euclidean"),
        ("minkowski", 3, "minkowski"),
        ("minkowski", np.inf, "chebyshev"),
        ("sqeuclidean", 2, "sqeuclidean"),
    ],
)
@pytest.mark.parametrize("use_params", [False, True])
def test_effective_metric_and_params(metric, p, effective_metric, use_params):
    kws = {"p": 1000, "metric_params": {"p": p}} if use_params else {"p": p}
    X, _ = sklearn.datasets.make_blobs(random_state=42)
    cunn = cuKNN(metric=metric, **kws).fit(X)
    sknn = skKNN(metric=metric, p=p).fit(X)
    assert cunn.effective_metric_ == effective_metric
    assert cunn.effective_metric_ == sknn.effective_metric_
    assert cunn.effective_metric_params_.get(
        "p"
    ) == sknn.effective_metric_params_.get("p")

    sol, _ = sknn.kneighbors(n_neighbors=5)
    res, _ = cunn.kneighbors(n_neighbors=5)
    np.testing.assert_allclose(sol, res, atol=1e-4)
