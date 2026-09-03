# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import copy
import platform
from importlib.metadata import version as package_version

import cupy as cp
import cupyx.scipy.sparse as cp_sp
import joblib
import numba
import numpy as np
import pytest
import scipy.sparse as sp
import umap
from packaging.version import Version
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.manifold import trustworthiness
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import KDTree, NearestNeighbors, kneighbors_graph

import cuml
from cuml.internals import GraphBasedDimRedCallback
from cuml.manifold.umap import UMAP as cuUMAP
from cuml.manifold.utils import extract_knn_graph
from cuml.metrics import pairwise_distances
from cuml.testing.utils import (
    array_equal,
    quality_param,
    stress_param,
    unit_param,
)

dataset_names = ["iris", "digits", "wine", "blobs"]

UMAP_VERSION = Version(package_version("umap-learn"))
UMAP_NUMBA_REGRESSION = Version(numba.__version__) >= Version(
    "0.62.0"
) and UMAP_VERSION < Version("0.5.12")


@pytest.mark.parametrize(
    "nrows", [unit_param(500), quality_param(5000), stress_param(500000)]
)
@pytest.mark.parametrize(
    "n_feats", [unit_param(20), quality_param(100), stress_param(1000)]
)
@pytest.mark.parametrize("build_algo", ["brute_force_knn", "nn_descent"])
def test_blobs_cluster(nrows, n_feats, build_algo):
    data, labels = datasets.make_blobs(
        n_samples=nrows, n_features=n_feats, centers=5, random_state=0
    )
    embedding = cuUMAP(build_algo=build_algo).fit_transform(data)

    if nrows < 500000:
        score = adjusted_rand_score(labels, KMeans(5).fit_predict(embedding))
        assert score == 1.0


@pytest.mark.parametrize(
    "nrows",
    [
        pytest.param(
            500,
            marks=pytest.mark.unit,
        ),
        quality_param(5000),
        stress_param(500000),
    ],
)
@pytest.mark.parametrize(
    "n_feats", [unit_param(10), quality_param(100), stress_param(1000)]
)
@pytest.mark.parametrize("build_algo", ["brute_force_knn", "nn_descent"])
def test_umap_fit_transform_score(nrows, n_feats, build_algo):
    n_samples = nrows
    n_features = n_feats

    data, labels = make_blobs(
        n_samples=n_samples, n_features=n_features, centers=10, random_state=42
    )

    model = umap.UMAP(n_neighbors=10, min_dist=0.1)
    cuml_model = cuUMAP(n_neighbors=10, min_dist=0.01, build_algo=build_algo)

    embedding = model.fit_transform(data)
    cuml_embedding = cuml_model.fit_transform(data)

    assert not np.isnan(embedding).any()
    assert not np.isnan(cuml_embedding).any()

    if nrows < 500000:
        cuml_score = adjusted_rand_score(
            labels, KMeans(10).fit_predict(cuml_embedding)
        )
        score = adjusted_rand_score(labels, KMeans(10).fit_predict(embedding))

        assert array_equal(score, cuml_score, 1e-2, with_sign=True)


def test_supervised_umap_trustworthiness_on_iris():
    iris = datasets.load_iris()
    data = iris.data
    embedding = cuUMAP(
        n_neighbors=10, random_state=0, min_dist=0.01
    ).fit_transform(data, iris.target)
    trust = trustworthiness(iris.data, embedding, n_neighbors=10)
    assert trust >= 0.97


def test_semisupervised_umap_trustworthiness_on_iris():
    iris = datasets.load_iris()
    data = iris.data
    target = iris.target.copy()
    target[25:75] = -1
    embedding = cuUMAP(
        n_neighbors=10, random_state=0, min_dist=0.01
    ).fit_transform(data, target)

    trust = trustworthiness(iris.data, embedding, n_neighbors=10)
    assert trust >= 0.97


def test_umap_trustworthiness_on_iris():
    iris = datasets.load_iris()
    data = iris.data
    embedding = cuUMAP(
        n_neighbors=10, min_dist=0.01, random_state=0
    ).fit_transform(data)
    trust = trustworthiness(iris.data, embedding, n_neighbors=10)
    assert trust >= 0.97


@pytest.mark.parametrize("target_metric", ["categorical", "euclidean"])
def test_umap_transform_on_iris(target_metric):
    iris = datasets.load_iris()

    iris_selection = np.random.RandomState(42).choice(
        [True, False], 150, replace=True, p=[0.75, 0.25]
    )
    data = iris.data[iris_selection]

    fitter = cuUMAP(
        n_neighbors=10,
        init="random",
        n_epochs=800,
        min_dist=0.01,
        random_state=42,
        target_metric=target_metric,
    )
    fitter.fit(data)
    new_data = iris.data[~iris_selection]
    embedding = fitter.transform(new_data)

    assert not np.isnan(embedding).any()

    trust = trustworthiness(new_data, embedding, n_neighbors=10)
    assert trust >= 0.85


@pytest.mark.parametrize("input_type", ["cupy", "scipy"])
@pytest.mark.parametrize("xform_method", ["fit", "fit_transform"])
@pytest.mark.parametrize("target_metric", ["categorical", "euclidean"])
def test_umap_transform_on_digits_sparse(
    target_metric, input_type, xform_method
):
    digits = datasets.load_digits()

    digits_selection = np.random.RandomState(42).choice(
        [True, False], 1797, replace=True, p=[0.75, 0.25]
    )

    sparse = cp_sp if input_type == "cupy" else sp

    data = sparse.csr_matrix(sp.csr_matrix(digits.data[digits_selection]))

    fitter = cuUMAP(
        n_neighbors=15,
        init="random",
        n_epochs=0,
        min_dist=0.01,
        random_state=42,
        target_metric=target_metric,
    )

    new_data = sparse.csr_matrix(sp.csr_matrix(digits.data[~digits_selection]))

    if xform_method == "fit":
        fitter.fit(data)
        embedding = fitter.transform(new_data)
    else:
        embedding = fitter.fit_transform(new_data)

    if input_type == "cupy":
        embedding = embedding.get()

    trust = trustworthiness(
        digits.data[~digits_selection], embedding, n_neighbors=15
    )
    assert trust >= 0.96


@pytest.mark.parametrize("target_metric", ["categorical", "euclidean"])
def test_umap_transform_on_digits(target_metric):
    digits = datasets.load_digits()

    digits_selection = np.random.RandomState(42).choice(
        [True, False], 1797, replace=True, p=[0.75, 0.25]
    )
    data = digits.data[digits_selection]

    fitter = cuUMAP(
        n_neighbors=15,
        init="random",
        n_epochs=0,
        min_dist=0.01,
        random_state=42,
        target_metric=target_metric,
    )
    fitter.fit(data)

    new_data = digits.data[~digits_selection]

    embedding = fitter.transform(new_data)
    trust = trustworthiness(
        digits.data[~digits_selection], embedding, n_neighbors=15
    )
    assert trust >= 0.96


@pytest.mark.parametrize("target_metric", ["categorical", "euclidean"])
@pytest.mark.parametrize("name", dataset_names)
def test_umap_fit_transform_trust(name, target_metric):
    if name == "iris":
        iris = datasets.load_iris()
        data = iris.data
        labels = iris.target

    elif name == "digits":
        digits = datasets.load_digits(n_class=5)
        data = digits.data
        labels = digits.target

    elif name == "wine":
        wine = datasets.load_wine()
        data = wine.data
        labels = wine.target
    else:
        data, labels = make_blobs(
            n_samples=500, n_features=10, centers=10, random_state=42
        )

    model = umap.UMAP(
        n_neighbors=10, min_dist=0.01, target_metric=target_metric
    )
    cuml_model = cuUMAP(
        n_neighbors=10, min_dist=0.01, target_metric=target_metric
    )
    embedding = model.fit_transform(data)
    cuml_embedding = cuml_model.fit_transform(data)

    trust = trustworthiness(data, embedding, n_neighbors=10)
    cuml_trust = trustworthiness(data, cuml_embedding, n_neighbors=10)

    assert array_equal(trust, cuml_trust, 1e-1, with_sign=True)


@pytest.mark.parametrize("target_metric", ["categorical", "euclidean"])
@pytest.mark.parametrize("name", [unit_param("digits")])
@pytest.mark.parametrize("nrows", [quality_param(5000), stress_param(500000)])
@pytest.mark.parametrize("n_feats", [quality_param(100), stress_param(1000)])
@pytest.mark.parametrize("should_downcast", [True])
@pytest.mark.parametrize("input_type", ["dataframe", "ndarray"])
@pytest.mark.parametrize("build_algo", ["brute_force_knn", "nn_descent"])
def test_umap_data_formats(
    input_type,
    should_downcast,
    nrows,
    n_feats,
    name,
    target_metric,
    build_algo,
):
    dtype = np.float32 if not should_downcast else np.float64
    n_samples = nrows
    n_feats = n_feats

    if name == "digits":
        # use the digits dataset for unit test
        digits = datasets.load_digits(n_class=9)
        X = digits["data"].astype(dtype)

    else:
        X, y = datasets.make_blobs(
            n_samples=n_samples, n_features=n_feats, random_state=0
        )

    umap = cuUMAP(
        n_neighbors=3,
        n_components=2,
        target_metric=target_metric,
        build_algo=build_algo,
    )

    embeds = umap.fit_transform(X)
    assert type(embeds) is np.ndarray


@pytest.mark.parametrize("target_metric", ["categorical", "euclidean"])
@pytest.mark.filterwarnings("ignore:(.*)connected(.*):UserWarning:sklearn[.*]")
@pytest.mark.parametrize("build_algo", ["brute_force_knn", "nn_descent"])
def test_umap_fit_transform_score_default(target_metric, build_algo):
    n_samples = 500
    n_features = 20

    data, labels = make_blobs(
        n_samples=n_samples, n_features=n_features, centers=10, random_state=42
    )

    model = umap.UMAP(target_metric=target_metric)
    cuml_model = cuUMAP(target_metric=target_metric, build_algo=build_algo)

    embedding = model.fit_transform(data)
    cuml_embedding = cuml_model.fit_transform(data)

    cuml_score = adjusted_rand_score(
        labels, KMeans(10).fit_predict(cuml_embedding)
    )
    score = adjusted_rand_score(labels, KMeans(10).fit_predict(embedding))

    assert array_equal(score, cuml_score, 1e-2, with_sign=True)


@pytest.mark.parametrize("build_algo", ["brute_force_knn", "nn_descent"])
def test_umap_fit_transform_against_fit_and_transform(build_algo):
    n_samples = 500
    n_features = 20

    data, labels = make_blobs(
        n_samples=n_samples, n_features=n_features, centers=10, random_state=42
    )

    """
    First test the default option does not hash the input
    """

    cuml_model = cuUMAP(build_algo=build_algo)

    ft_embedding = cuml_model.fit_transform(data)
    fit_embedding_same_input = cuml_model.transform(data)

    assert joblib.hash(ft_embedding) != joblib.hash(fit_embedding_same_input)

    """
    Next, test explicitly enabling feature hashes the input
    """

    cuml_model = cuUMAP(hash_input=True)

    ft_embedding = cuml_model.fit_transform(data)
    fit_embedding_same_input = cuml_model.transform(data)

    assert joblib.hash(ft_embedding) == joblib.hash(fit_embedding_same_input)

    fit_embedding_diff_input = cuml_model.transform(data[1:])
    assert joblib.hash(ft_embedding) != joblib.hash(fit_embedding_diff_input)


@pytest.mark.parametrize(
    "n_components,random_state",
    [
        unit_param(2, None),
        unit_param(2, 8),
        unit_param(2, np.random.RandomState(42)),
        unit_param(21, None),
        unit_param(21, np.random.RandomState(42)),
        unit_param(25, 8),
        unit_param(50, None),
        stress_param(50, 8),
    ],
)
def test_umap_fit_transform_reproducibility(n_components, random_state):
    n_samples = 8000
    n_features = 200

    if random_state is None:
        n_components *= 2

    data, labels = make_blobs(
        n_samples=n_samples, n_features=n_features, centers=10, random_state=42
    )

    def get_embedding(n_components, random_state):
        reducer = cuUMAP(
            init="random",
            n_components=n_components,
            random_state=random_state,
            build_algo="brute_force_knn",
        )
        return reducer.fit_transform(data)

    state = copy.deepcopy(random_state)
    cuml_embedding1 = get_embedding(n_components, state)
    state = copy.deepcopy(random_state)
    cuml_embedding2 = get_embedding(n_components, state)

    assert not np.isnan(cuml_embedding1).any()
    assert not np.isnan(cuml_embedding2).any()

    # Reproducibility threshold raised until intermittent failure is fixed
    # Ref: https://github.com/NVIDIA/cuml/issues/1903
    mean_diff = np.mean(np.abs(cuml_embedding1 - cuml_embedding2))
    if random_state is not None:
        assert mean_diff == 0.0
    else:
        assert mean_diff > 0.5


# n_components values cover the three sequential-kernel dispatch paths in
# call_optimize_sequential_kernel:
#   - 2  : per-thread register kernel (low end)
#   - 21 : per-thread register kernel (boundary, n_components <= 21)
#   - 50 : per-warp component-parallel kernel (n_components >  21)
@pytest.mark.parametrize("n_components", [2, 21, 50])
@pytest.mark.parametrize("init", ["random", "spectral"])
def test_umap_force_serial_epochs_reproducibility(n_components, init):
    n_samples = 8000
    n_features = 200

    data, _ = make_blobs(
        n_samples=n_samples, n_features=n_features, centers=10, random_state=42
    )
    data = data.astype(np.float32)

    def get_embedding():
        return cuUMAP(
            init=init,
            n_components=n_components,
            random_state=42,
            force_serial_epochs=True,
            build_algo="brute_force_knn",
        ).fit_transform(data)

    cuml_embedding1 = get_embedding()
    cuml_embedding2 = get_embedding()

    assert not np.isnan(cuml_embedding1).any()
    assert not np.isnan(cuml_embedding2).any()

    np.testing.assert_array_equal(cuml_embedding1, cuml_embedding2)


@pytest.mark.parametrize(
    "n_components,random_state",
    [
        unit_param(2, None),
        unit_param(2, 8),
        unit_param(2, np.random.RandomState(42)),
        unit_param(21, None),
        unit_param(25, 8),
        unit_param(25, np.random.RandomState(42)),
        unit_param(50, None),
        stress_param(50, 8),
    ],
)
def test_umap_transform_reproducibility(n_components, random_state):
    n_samples = 5000
    n_features = 200

    if random_state is None:
        n_components *= 2

    data, labels = make_blobs(
        n_samples=n_samples, n_features=n_features, centers=10, random_state=42
    )

    selection = np.random.RandomState(42).choice(
        [True, False], n_samples, replace=True, p=[0.5, 0.5]
    )
    fit_data = data[selection]
    transform_data = data[~selection]

    def get_embedding(n_components, random_state):
        reducer = cuUMAP(
            init="random",
            n_components=n_components,
            random_state=random_state,
            build_algo="brute_force_knn",
        )
        reducer.fit(fit_data)
        return reducer.transform(transform_data)

    state = copy.deepcopy(random_state)
    cuml_embedding1 = get_embedding(n_components, state)
    state = copy.deepcopy(random_state)
    cuml_embedding2 = get_embedding(n_components, state)

    assert not np.isnan(cuml_embedding1).any()
    assert not np.isnan(cuml_embedding2).any()

    # Reproducibility threshold raised until intermittent failure is fixed
    # Ref: https://github.com/NVIDIA/cuml/issues/1903
    mean_diff = np.mean(np.abs(cuml_embedding1 - cuml_embedding2))
    if random_state is not None:
        assert mean_diff == 0.0
    else:
        assert mean_diff > 0.5


def test_umap_fit_transform_trustworthiness_with_consistency_enabled():
    iris = datasets.load_iris()
    data = iris.data
    algo = cuUMAP(
        n_neighbors=10,
        min_dist=0.01,
        init="random",
        random_state=42,
    )
    embedding = algo.fit_transform(data)
    trust = trustworthiness(iris.data, embedding, n_neighbors=10)
    assert trust >= 0.97


def test_umap_transform_trustworthiness_with_consistency_enabled():
    iris = datasets.load_iris()
    data = iris.data
    selection = np.random.RandomState(42).choice(
        [True, False], data.shape[0], replace=True, p=[0.5, 0.5]
    )
    fit_data = data[selection]
    transform_data = data[~selection]
    model = cuUMAP(
        n_neighbors=10,
        min_dist=0.01,
        init="random",
        random_state=42,
    )
    model.fit(fit_data)
    embedding = model.transform(transform_data)
    trust = trustworthiness(transform_data, embedding, n_neighbors=10)
    assert trust >= 0.92


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"a": 0.5, "b": 2.0},
        {"a": 0.5},
        {"b": 0.5},
        {"min_dist": 0.01, "spread": 2.0},
    ],
)
def test_a_b_values_match(kwargs):
    X, _ = make_blobs(n_samples=50, random_state=42)
    cu_model = cuml.UMAP(**kwargs).fit(X)
    sk_model = umap.UMAP(**kwargs).fit(X)

    np.testing.assert_allclose(cu_model._a, sk_model._a)
    np.testing.assert_allclose(cu_model._b, sk_model._b)


def test_fit_fewer_rows_than_n_neighbors():
    X, _ = make_blobs(n_samples=10, random_state=42)
    model = cuml.UMAP(n_neighbors=15)
    with pytest.warns(UserWarning, match="larger than n_samples"):
        model.fit(X)
    assert model._n_neighbors == 10


@pytest.mark.parametrize("in_mem_type", ["device", "host"])
@pytest.mark.parametrize("in_dtype", ["float32", "float64"])
@pytest.mark.parametrize("indices_dtype", ["float32", "float64"])
@pytest.mark.parametrize("k", [10, 20])
@pytest.mark.parametrize(
    "kind, transform",
    [
        ("csr", None),
        ("csc", None),
        ("coo", None),
        ("csr", "canonical"),
        ("csc", "canonical"),
        ("coo", "canonical"),
        ("csr", "nonzero"),
        ("csc", "nonzero"),
        ("coo", "nonzero"),
        ("pairwise", None),
        ("tuple", None),
    ],
)
def test_extract_knn_graph(
    in_mem_type, in_dtype, indices_dtype, k, kind, transform
):
    X, _ = datasets.make_blobs(30, 10, centers=5, random_state=42)
    if kind == "pairwise":
        knn_info = pairwise_distances(X).astype(in_dtype)
        if in_mem_type == "device":
            knn_info = cp.asarray(knn_info)
    elif kind == "tuple":
        nn = NearestNeighbors(n_neighbors=20).fit(X)
        distances, indices = nn.kneighbors(X, return_distance=True)
        if in_mem_type == "device":
            indices = cp.asarray(indices)
            distances = cp.asarray(distances)
        knn_info = (indices, distances.astype(in_dtype))
    else:
        knn_info = kneighbors_graph(X, 20, include_self=True, mode="distance")
        if in_mem_type == "device":
            knn_info = cp_sp.csr_matrix(knn_info)
        knn_info = knn_info.asformat(kind).astype(in_dtype)
        if transform is not None:
            # Canonicalize input matrix
            if hasattr(knn_info, "sort_indices"):
                knn_info.sort_indices()
        if transform == "nonzero":
            knn_info.eliminate_zeros()

    sol = kneighbors_graph(X, k, include_self=True, mode="distance")

    # mem_type=None doesn't coerce mem_type
    indices, distances = extract_knn_graph(
        knn_info, 30, k, indices_dtype=indices_dtype, mem_type=None
    )
    assert indices.dtype == indices_dtype
    assert distances.dtype == "float32"
    xp = cp if in_mem_type == "device" else np
    assert isinstance(indices, xp.ndarray)
    assert isinstance(distances, xp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(distances), sol.data, atol=1e-5)
    np.testing.assert_array_equal(cp.asnumpy(indices), sol.indices)

    indices, distances = extract_knn_graph(
        knn_info, 30, k, indices_dtype=indices_dtype, mem_type="host"
    )
    assert indices.dtype == indices_dtype
    assert distances.dtype == "float32"
    np.testing.assert_allclose(distances, sol.data, atol=1e-5)
    np.testing.assert_array_equal(indices, sol.indices)

    indices, distances = extract_knn_graph(
        knn_info, 30, k, indices_dtype=indices_dtype, mem_type="device"
    )
    assert indices.dtype == indices_dtype
    assert distances.dtype == "float32"
    np.testing.assert_allclose(distances.get(), sol.data, atol=1e-5)
    np.testing.assert_array_equal(indices.get(), sol.indices)


def test_extract_knn_graph_errors():
    X, _ = datasets.make_blobs(30, 10, centers=5, random_state=42)

    # tuple: invalid shape
    knn_info = (np.ones((3, 4)), np.ones((4, 3)))
    with pytest.raises(
        ValueError, match="Expected indices and distances to have shape"
    ):
        extract_knn_graph(knn_info, 3, 2)
    knn_info = (np.ones((2, 4)), np.ones((2, 4)))
    with pytest.raises(
        ValueError, match="Expected indices and distances to have shape"
    ):
        extract_knn_graph(knn_info, 3, 2)

    # tuple: n_neighbors > original_n_neighbors
    knn_info = tuple(
        reversed(
            NearestNeighbors(n_neighbors=20)
            .fit(X)
            .kneighbors(X, return_distance=True)
        )
    )
    with pytest.raises(
        ValueError,
        match=(
            "Precomputed KNN data has 20 neighbors per sample, but "
            "n_neighbors=25 was specified"
        ),
    ):
        extract_knn_graph(knn_info, 30, 25)

    # tuple: expected self references
    knn_info = tuple(
        reversed(
            NearestNeighbors(n_neighbors=20)
            .fit(X)
            .kneighbors(return_distance=True)
        )
    )
    with pytest.raises(
        ValueError,
        match="Expected indices and distances to include self references",
    ):
        extract_knn_graph(knn_info, 30, 20)

    # graph: invalid shape
    knn_info = sp.random(29, 29, random_state=42, density=0.5)
    with pytest.raises(ValueError, match="Expected a sparse array of shape"):
        extract_knn_graph(knn_info, 30, 10)
    knn_info = sp.random(30, 29, random_state=42, density=0.5)
    with pytest.raises(ValueError, match="Expected a sparse array of shape"):
        extract_knn_graph(knn_info, 30, 10)

    # graph: invalid nnz
    knn_info = sp.csr_matrix(
        (
            np.array([0.5, 0.5, 0.5, 0.5]),
            (np.array([0, 0, 1, 1]), np.array([1, 2, 0, 2])),
        ),
        shape=(3, 3),
    )
    with pytest.raises(
        ValueError, match="Precomputed KNN graph has 4 nonzero elements"
    ):
        extract_knn_graph(knn_info, 3, 3)

    # graph: n_neighbors > original_n_neighbors
    knn_info = kneighbors_graph(X, 20, include_self=True, mode="distance")
    with pytest.raises(
        ValueError,
        match=(
            "Precomputed KNN data has 20 neighbors per sample, but "
            "n_neighbors=25 was specified"
        ),
    ):
        extract_knn_graph(knn_info, 30, 25)

    # pairwise: invalid shape
    knn_info = np.ones((5, 6))
    with pytest.raises(ValueError, match="Expected a dense array of shape"):
        extract_knn_graph(knn_info, 5, 5)
    knn_info = np.ones((5, 5))
    with pytest.raises(ValueError, match="Expected a dense array of shape"):
        extract_knn_graph(knn_info, 6, 5)

    # pairwise: n_neighbors > n_samples
    knn_info = np.ones((5, 5))
    with pytest.raises(
        ValueError,
        match="Precomputed KNN data requires n_samples >= n_neighbors",
    ):
        extract_knn_graph(knn_info, 5, 10)


@pytest.mark.parametrize(
    "kind, mem_type",
    [
        ("cupy", None),
        ("cupy", "device"),
        ("numpy", None),
        ("numpy", "host"),
    ],
)
def test_extract_knn_graph_zero_distance_rows(kind, mem_type):
    X = np.array([[0, 1, 2], [3, 1, 3], [5, 1, 2], [0, 1, 2]], dtype="float32")
    nn = cuml.neighbors.NearestNeighbors().fit(X)
    with cuml.using_output_type(kind):
        graph = nn.kneighbors_graph(X, 3, mode="distance")
    graph.sort_indices()

    # Hardcoded indices to ease testing - it's tricky to get `kneighbors` to
    # return self references first otherwise.
    sol_indices = np.array([0, 3, 1, 1, 2, 3, 2, 1, 3, 3, 0, 1])
    sol_distances = nn.kneighbors(X, 3)[0].flatten()

    indices, distances = extract_knn_graph(graph, 4, 3, mem_type=mem_type)
    np.testing.assert_allclose(cp.asnumpy(distances), sol_distances, atol=1e-5)
    np.testing.assert_array_equal(cp.asnumpy(indices), sol_indices)


@pytest.mark.parametrize(
    "kind, mem_type",
    [
        ("cupy", None),
        ("cupy", "device"),
        ("numpy", None),
        ("numpy", "host"),
    ],
)
def test_extract_knn_graph_from_kneighbors_graph_zero_copy(kind, mem_type):
    """When converting from the output of `kneighbors_graph`, no copy should be
    needed if not changing mem_type, as no sorting is required"""
    X, _ = datasets.make_blobs(30, 10, centers=5, random_state=42)
    with cuml.using_output_type(kind):
        graph = cuml.neighbors.kneighbors_graph(
            X, 20, include_self=True, mode="distance"
        )

    def assert_same_memory(x, y):
        assert type(x) is type(y)
        x_ptr = x.data.ptr if isinstance(x, cp.ndarray) else x.ctypes.data
        y_ptr = y.data.ptr if isinstance(y, cp.ndarray) else y.ctypes.data
        assert x_ptr == y_ptr

    indices, distances = extract_knn_graph(
        graph, 30, 20, mem_type=mem_type, indices_dtype=graph.indices.dtype
    )
    assert_same_memory(indices, graph.indices)
    assert_same_memory(distances, graph.data)


@pytest.mark.parametrize("n_neighbors", [5, 15])
@pytest.mark.parametrize("build_algo", ["brute_force_knn", "nn_descent"])
@pytest.mark.parametrize("data_on_gpu", [True, False])
def test_umap_knn_graph(n_neighbors, build_algo, data_on_gpu):
    data, labels = datasets.make_blobs(
        n_samples=2000, n_features=10, centers=5, random_state=0
    )
    data = data.astype(np.float32)

    def fit_transform_embed(knn_graph=None):
        model = cuUMAP(
            random_state=42,
            init="random",
            n_neighbors=n_neighbors,
            build_algo=build_algo,
        )
        embd = model.fit_transform(
            cp.array(data) if data_on_gpu else data,
            knn_graph=knn_graph,
        )

        return embd.get() if data_on_gpu else embd

    def transform_embed(knn_graph=None):
        model = cuUMAP(
            random_state=42,
            init="random",
            n_neighbors=n_neighbors,
            build_algo=build_algo,
        )
        model.fit(
            cp.array(data) if data_on_gpu else data,
            knn_graph=knn_graph,
        )
        embd = model.transform(cp.array(data) if data_on_gpu else data)
        return embd.get() if data_on_gpu else embd

    def test_trustworthiness(embedding):
        trust = trustworthiness(data, embedding, n_neighbors=n_neighbors)
        assert trust >= 0.92

    def test_equality(e1, e2):
        mean_diff = np.mean(np.abs(e1 - e2))
        assert mean_diff < 1.0

    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(data)
    knn_graph = neigh.kneighbors_graph(data, mode="distance")

    embedding1 = fit_transform_embed(None)
    embedding2 = fit_transform_embed(knn_graph.tocsr())
    embedding3 = fit_transform_embed(knn_graph.tocoo())
    embedding4 = fit_transform_embed(knn_graph.tocsc())
    embedding5 = transform_embed(knn_graph.tocsr())
    embedding6 = transform_embed(knn_graph.tocoo())
    embedding7 = transform_embed(knn_graph.tocsc())

    test_trustworthiness(embedding1)
    test_trustworthiness(embedding2)
    test_trustworthiness(embedding3)
    test_trustworthiness(embedding4)
    test_trustworthiness(embedding5)
    test_trustworthiness(embedding6)
    test_trustworthiness(embedding7)

    test_equality(embedding2, embedding3)
    test_equality(embedding3, embedding4)
    test_equality(embedding5, embedding6)
    test_equality(embedding6, embedding7)


@pytest.mark.parametrize(
    "precomputed_type", ["knn_graph", "tuple", "pairwise"]
)
@pytest.mark.parametrize(
    "sparse_input,build_algo",
    [
        (False, "brute_force_knn"),
        (True, "brute_force_knn"),
        (False, "nn_descent"),
    ],
)
def test_umap_precomputed_knn(precomputed_type, sparse_input, build_algo):
    data, labels = make_blobs(
        n_samples=2000, n_features=10, centers=5, random_state=0
    )
    data = data.astype(np.float32)

    if sparse_input:
        sparsification = np.random.choice(
            [0.0, 1.0], p=[0.1, 0.9], size=data.shape
        )
        data = np.multiply(data, sparsification)
        data = sp.csr_matrix(data)

    n_neighbors = 8

    if precomputed_type == "knn_graph":
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(data)
        precomputed_knn = nn.kneighbors_graph(data, mode="distance")
    elif precomputed_type == "tuple":
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(data)
        precomputed_knn = nn.kneighbors(data, return_distance=True)
        precomputed_knn = (precomputed_knn[1], precomputed_knn[0])
    elif precomputed_type == "pairwise":
        precomputed_knn = pairwise_distances(data)

    model = cuUMAP(
        n_neighbors=n_neighbors,
        precomputed_knn=precomputed_knn,
        build_algo=build_algo,
    )
    embedding = model.fit_transform(data)
    trust = trustworthiness(data, embedding, n_neighbors=n_neighbors)
    assert trust >= 0.92


def correctness_sparse(a, b, atol=0.1, rtol=0.2, threshold=0.95):
    n_ref_zeros = (a == 0).sum()
    n_ref_non_zero_elms = a.size - n_ref_zeros
    n_correct = (cp.abs(a - b) <= (atol + rtol * cp.abs(b))).sum()
    correctness = (n_correct - n_ref_zeros) / n_ref_non_zero_elms
    return correctness >= threshold


@pytest.mark.parametrize("n_rows", [200, 800])
@pytest.mark.parametrize("n_features", [8, 32])
@pytest.mark.parametrize("n_neighbors", [8, 16])
@pytest.mark.filterwarnings(
    "ignore:Spectral initialisation failed.*:UserWarning"
)
@pytest.mark.filterwarnings(
    "ignore:Graph is not fully connected.*:UserWarning"
)
@pytest.mark.xfail(
    UMAP_NUMBA_REGRESSION,
    reason="Upstream regression in umap<0.5.12 with numba >= 0.62.0",
    strict=True,
)
def test_fuzzy_simplicial_set(n_rows, n_features, n_neighbors):
    n_clusters = 30
    random_state = 42

    X, _ = make_blobs(
        n_samples=n_rows,
        centers=n_clusters,
        n_features=n_features,
        random_state=random_state,
    )

    model = cuUMAP(n_neighbors=n_neighbors)
    model.fit(X)
    cu_fss_graph = model.graph_

    model = umap.UMAP(n_neighbors=n_neighbors)
    model.fit(X)
    ref_fss_graph = model.graph_

    cu_fss_graph = cp.array(cu_fss_graph.todense())
    ref_fss_graph = cp.array(ref_fss_graph.todense())
    assert correctness_sparse(
        ref_fss_graph, cu_fss_graph, atol=0.1, rtol=0.2, threshold=0.95
    )


@pytest.mark.parametrize(
    "metric,build_algo,supported",
    [
        ("l2", "brute_force_knn", True),
        ("euclidean", "brute_force_knn", True),
        ("sqeuclidean", "brute_force_knn", True),
        ("l1", "brute_force_knn", True),
        ("manhattan", "brute_force_knn", True),
        ("minkowski", "brute_force_knn", True),
        ("chebyshev", "brute_force_knn", True),
        ("cosine", "brute_force_knn", True),
        ("correlation", "brute_force_knn", True),
        ("jaccard", "brute_force_knn", False),
        ("hamming", "brute_force_knn", True),
        ("canberra", "brute_force_knn", True),
        ("l2", "nn_descent", True),
        ("euclidean", "nn_descent", True),
        ("sqeuclidean", "nn_descent", True),
        ("l1", "nn_descent", False),
        ("manhattan", "nn_descent", False),
        ("minkowski", "nn_descent", False),
        ("chebyshev", "nn_descent", False),
        ("cosine", "nn_descent", True),
        ("correlation", "nn_descent", False),
        ("jaccard", "nn_descent", False),
        ("hamming", "nn_descent", False),
        ("canberra", "nn_descent", False),
    ],
)
@pytest.mark.filterwarnings(
    "ignore:gradient function is not yet implemented.*:UserWarning"
)
def test_umap_distance_metrics_fit_transform_trust(
    metric, build_algo, supported
):
    data, labels = make_blobs(
        n_samples=500, n_features=64, centers=5, random_state=42
    )

    if metric == "jaccard":
        data = data >= 0

    umap_model = umap.UMAP(
        n_neighbors=10, min_dist=0.01, metric=metric, init="random"
    )
    cuml_model = cuUMAP(
        n_neighbors=10,
        min_dist=0.01,
        metric=metric,
        init="random",
        build_algo=build_algo,
    )
    if not supported:
        with pytest.raises(NotImplementedError):
            cuml_model.fit_transform(data)
        return

    umap_embedding = umap_model.fit_transform(data)

    cuml_embedding = cuml_model.fit_transform(data)

    umap_trust = trustworthiness(
        data, umap_embedding, n_neighbors=10, metric=metric
    )
    cuml_trust = trustworthiness(
        data, cuml_embedding, n_neighbors=10, metric=metric
    )

    assert array_equal(umap_trust, cuml_trust, 0.05, with_sign=True)


@pytest.mark.parametrize(
    "metric,supported,umap_learn_supported",
    [
        ("l2", True, False),
        ("euclidean", True, True),
        ("sqeuclidean", True, False),
        ("l1", True, True),
        ("manhattan", True, True),
        ("minkowski", True, True),
        ("chebyshev", True, True),
        ("cosine", True, True),
        ("correlation", True, True),
        ("jaccard", True, True),
        ("hamming", True, True),
        ("canberra", True, True),
    ],
)
@pytest.mark.filterwarnings(
    "ignore:gradient function is not yet implemented.*:UserWarning"
)
def test_umap_distance_metrics_fit_transform_trust_on_sparse_input(
    metric, supported, umap_learn_supported
):
    if metric == "jaccard":
        n_features = 1000
    else:
        n_features = 64

    data, labels = make_blobs(
        n_samples=1000, n_features=n_features, centers=5, random_state=42
    )

    data_selection = np.random.RandomState(42).choice(
        [True, False], 1000, replace=True, p=[0.75, 0.25]
    )

    if metric == "jaccard":
        data = data >= 0

    new_data = sp.csr_matrix(data[~data_selection])

    if umap_learn_supported:
        umap_model = umap.UMAP(
            n_neighbors=10, min_dist=0.01, metric=metric, init="random"
        )
        umap_embedding = umap_model.fit_transform(new_data)
        umap_trust = trustworthiness(
            data[~data_selection],
            umap_embedding,
            n_neighbors=10,
            metric=metric,
        )

    cuml_model = cuUMAP(
        n_neighbors=10, min_dist=0.01, metric=metric, init="random"
    )

    if not supported:
        with pytest.raises(NotImplementedError):
            cuml_model.fit_transform(new_data)
        return

    cuml_embedding = cuml_model.fit_transform(new_data)
    cuml_trust = trustworthiness(
        data[~data_selection], cuml_embedding, n_neighbors=10, metric=metric
    )

    if umap_learn_supported:
        assert array_equal(umap_trust, cuml_trust, 0.05, with_sign=True)


@pytest.mark.parametrize("num_clusters", [3, 5])
@pytest.mark.parametrize("fit_then_transform", [False, True])
@pytest.mark.parametrize("metric", ["l2", "sqeuclidean", "cosine"])
@pytest.mark.parametrize("device_ids", [None, "all"])
@pytest.mark.parametrize("build_algo", ["brute_force_knn", "nn_descent"])
def test_umap_trustworthiness_on_batch_nnd(
    num_clusters, fit_then_transform, metric, device_ids, build_algo
):
    digits = datasets.load_digits()

    cuml_model = cuUMAP(
        n_neighbors=10,
        min_dist=0.01,
        build_algo=build_algo,
        build_kwds={"knn_n_clusters": num_clusters},
        metric=metric,
        device_ids=device_ids,
    )

    if fit_then_transform:
        cuml_model.fit(digits.data)
        cuml_embedding = cuml_model.transform(digits.data)
    else:
        cuml_embedding = cuml_model.fit_transform(digits.data)

    cuml_trust = trustworthiness(
        digits.data, cuml_embedding, n_neighbors=10, metric=metric
    )

    assert cuml_trust > 0.9


@pytest.mark.parametrize(
    "n_components,random_state",
    [
        unit_param(2, 8),
        unit_param(2, np.random.RandomState(42)),
        unit_param(21, np.random.RandomState(42)),
        unit_param(25, 8),
        stress_param(50, 8),
    ],
)
@pytest.mark.parametrize("num_clusters", [1, 4, 7])
@pytest.mark.parametrize("do_gpu_input", [False, True])
def test_umap_fit_transform_batch_brute_force_reproducibility(
    n_components, random_state, num_clusters, do_gpu_input
):
    n_samples = 8000
    n_features = 200

    data, labels = make_blobs(
        n_samples=n_samples, n_features=n_features, centers=10, random_state=42
    )
    if do_gpu_input:
        data = cp.asarray(data)

    def get_embedding(n_components, random_state):
        reducer = cuUMAP(
            init="random",
            n_components=n_components,
            random_state=random_state,
            build_algo="brute_force_knn",
            build_kwds={"knn_n_clusters": num_clusters},
        )
        return reducer.fit_transform(data)

    state = copy.deepcopy(random_state)
    cuml_embedding1 = get_embedding(n_components, state)
    state = copy.deepcopy(random_state)
    cuml_embedding2 = get_embedding(n_components, state)

    if do_gpu_input:
        cuml_embedding1 = cuml_embedding1.get()
        cuml_embedding2 = cuml_embedding2.get()

    assert not np.isnan(cuml_embedding1).any()
    assert not np.isnan(cuml_embedding2).any()

    np.testing.assert_array_equal(cuml_embedding1, cuml_embedding2)


def test_callback():
    class Callback(GraphBasedDimRedCallback):
        preprocess_event, epoch_event, train_event = False, 0, False

        def __init__(self, skip_init=False):
            if not skip_init:
                super().__init__()

        def check(self):
            assert self.preprocess_event
            assert self.epoch_event > 10
            assert self.train_event

        def on_preprocess_end(self, embeddings):
            self.preprocess_event = True

        def on_epoch_end(self, embeddings):
            self.epoch_event += 1

        def on_train_end(self, embeddings):
            self.train_event = True

    digits = datasets.load_digits()

    callback = Callback()
    reducer = cuUMAP(n_components=2, callback=callback)
    reducer.fit(digits.data)
    callback.check()


def test_umap_small_fit_large_transform():
    data, _ = make_blobs(
        n_samples=10_000, n_features=8, centers=5, random_state=0
    )
    train, infer = data[:1000], data[1000:]

    model = cuUMAP(build_algo="brute_force_knn", init="random")
    model.fit(train)
    embeddings = model.transform(infer)

    trust = trustworthiness(infer, embeddings, n_neighbors=10)
    assert trust >= 0.9


@pytest.mark.parametrize("n_neighbors", [5, 15])
@pytest.mark.parametrize("n_components", [2, 5])
@pytest.mark.parametrize("random_state", [None, 42])
@pytest.mark.parametrize("force_serial_epochs", [True, False, None])
def test_umap_outliers(
    n_neighbors, n_components, random_state, force_serial_epochs, request
):
    if force_serial_epochs is False:
        request.node.add_marker(
            pytest.mark.xfail(
                reason="With current heuristics, UMAP may produce outliers "
                "on GPUs with high SM counts when force_serial_epochs is False."
            )
        )
    if random_state is None:
        n_rows = 50_000
        build_algo = "nn_descent"
        init = "spectral"
    else:
        n_rows = 100_000
        build_algo = "brute_force_knn"
        init = "random"

    # This dataset was specifically chosen because UMAP produces outliers
    # on this dataset before the outlier fix.
    data, _ = make_moons(n_samples=n_rows, noise=0.0, random_state=42)
    data = data.astype(np.float32)

    gpu_umap = cuUMAP(
        build_algo=build_algo,
        init=init,
        n_neighbors=n_neighbors,
        n_components=n_components,
        random_state=random_state,
        force_serial_epochs=force_serial_epochs,
    )
    gpu_umap_embeddings = gpu_umap.fit_transform(data)

    # Ideally, this threshold should be determined by running the CPU UMAP implementation
    # and comparing the max and min values of the resulting embedding. However, running CPU UMAP
    # with this data size using spectral initialization is too slow to run repetitively in CI.
    # Instead, we hardwire a locally determined threshold for this dataset.
    if random_state is None:
        threshold = 50.0
    else:
        threshold = 25.0

    assert np.all(
        (gpu_umap_embeddings >= -threshold)
        & (gpu_umap_embeddings <= threshold)
    )


# Cases:
#   - (2, *, None): auto-enable for both init kinds.
#   - (25/50, random, True): per-warp sequential kernel (cpl=1 and
#     cpl=2).
#   - (513, spectral, None): dispatches to the parallel
#     batch kernel for n_components > 512.
#   - (513, random, True): explicit force_serial_epochs=True past the serial
#     kernel's upper bound must raise error at the Python layer
@pytest.mark.parametrize(
    "n_components,init,force_serial_epochs,expect_raises",
    [
        (2, "random", None, False),
        (2, "spectral", None, False),
        (25, "random", True, False),
        (50, "random", True, False),
        (513, "spectral", None, False),
        (513, "random", True, True),
    ],
)
def test_umap_force_serial_epochs_dispatch(
    n_components, init, force_serial_epochs, expect_raises
):
    data, _ = make_moons(n_samples=5000, noise=0.0, random_state=42)
    data = data.astype(np.float32)

    model = cuUMAP(
        n_components=n_components,
        n_neighbors=15,
        init=init,
        force_serial_epochs=force_serial_epochs,
        random_state=42,
        build_algo="brute_force_knn",
    )

    if expect_raises:
        with pytest.raises(ValueError, match="force_serial_epochs=True"):
            model.fit(data)
        return

    embedding = model.fit_transform(data)
    assert embedding.shape == (data.shape[0], n_components)
    assert np.isfinite(embedding).all()

    # Skip for the high-n_components case that falls back to the parallel batch kernel.
    if n_components <= 50:
        assert np.all(np.abs(embedding) <= 50.0)


@pytest.mark.parametrize("precomputed_type", ["tuple", "knn_graph"])
@pytest.mark.parametrize("k_provided,k_requested", [(15, 10), (20, 8)])
def test_umap_precomputed_knn_trimming(
    precomputed_type, k_provided, k_requested
):
    """
    Test that precomputed KNN data with more neighbors than requested
    is automatically trimmed instead of raising an error.
    """
    data, labels = make_blobs(
        n_samples=500, n_features=10, centers=5, random_state=0
    )
    data = data.astype(np.float32)

    # Build KNN graph with more neighbors than we'll request
    nn = NearestNeighbors(n_neighbors=k_provided)
    nn.fit(data)

    if precomputed_type == "tuple":
        distances, indices = nn.kneighbors(data, return_distance=True)
        precomputed_knn = (indices, distances)
    elif precomputed_type == "knn_graph":
        precomputed_knn = nn.kneighbors_graph(data, mode="distance")

    # This should work now - the excess neighbors should be trimmed
    model = cuUMAP(
        n_neighbors=k_requested,
        precomputed_knn=precomputed_knn,
        random_state=42,
        init="random",
    )
    embedding = model.fit_transform(data)

    # Verify the embedding is valid
    assert embedding.shape == (data.shape[0], 2)
    assert not np.isnan(embedding).any()

    # Verify trustworthiness with the requested number of neighbors
    trust = trustworthiness(data, embedding, n_neighbors=k_requested)
    assert trust >= 0.85


@pytest.mark.parametrize("precomputed_type", ["tuple", "knn_graph"])
def test_umap_precomputed_knn_insufficient_neighbors(precomputed_type):
    """
    Test that precomputed KNN data with fewer neighbors than requested
    raises an appropriate error.
    """
    data, labels = make_blobs(
        n_samples=500, n_features=10, centers=5, random_state=0
    )
    data = data.astype(np.float32)

    k_provided = 5
    k_requested = 10

    # Build KNN graph with fewer neighbors than we'll request
    nn = NearestNeighbors(n_neighbors=k_provided)
    nn.fit(data)

    if precomputed_type == "tuple":
        distances, indices = nn.kneighbors(data, return_distance=True)
        precomputed_knn = (indices, distances)
    elif precomputed_type == "knn_graph":
        precomputed_knn = nn.kneighbors_graph(data, mode="distance")

    # This should raise an error during fit
    model = cuUMAP(
        n_neighbors=k_requested,
        precomputed_knn=precomputed_knn,
        random_state=42,
        init="random",
    )
    with pytest.raises(
        ValueError, match=f".*at least {k_requested} neighbors.*"
    ):
        model.fit(data)


@pytest.mark.parametrize("input_type", ["numpy", "cupy"])
def test_umap_custom_init(input_type):
    n_samples = 500
    n_features = 20
    n_components = 2

    data, _ = make_blobs(
        n_samples=n_samples, n_features=n_features, centers=5, random_state=42
    )
    data = data.astype(np.float32)

    # Custom initial positions
    init_pos = (
        np.random.RandomState(42)
        .randn(n_samples, n_components)
        .astype(np.float32)
    )

    if input_type == "cupy":
        init_pos = cp.array(init_pos)

    model = cuUMAP(
        n_neighbors=10,
        init=init_pos,
        n_epochs=0,
        learning_rate=0,
        random_state=42,
    )

    # Should return the init_pos since learning_rate=0
    embedding = model.fit_transform(data)

    assert array_equal(embedding, init_pos)


def test_umap_custom_init_errors():
    n_samples = 100
    data, _ = make_blobs(n_samples=n_samples, n_features=10, random_state=42)
    data = data.astype(np.float32)

    # Wrong number of samples
    init_wrong_samples = np.zeros((n_samples + 1, 2), dtype=np.float32)
    model = cuUMAP(init=init_wrong_samples)
    with pytest.raises(ValueError, match="Expected `init` with shape"):
        model.fit(data)

    # Wrong number of components
    init_wrong_components = np.zeros((n_samples, 3), dtype=np.float32)
    model = cuUMAP(init=init_wrong_components, n_components=2)
    with pytest.raises(ValueError, match="Expected `init` with shape"):
        model.fit(data)


@pytest.mark.xfail(
    UMAP_NUMBA_REGRESSION,
    reason="Upstream regression in umap<0.5.12 with numba >= 0.62.0",
    strict=True,
)
def test_umap_sigmas_rhos():
    """Test that sigmas and rhos are correctly produced and valid."""
    n_samples = 200
    n_features = 10
    n_neighbors = 15
    X, _ = make_blobs(
        n_samples=n_samples, n_features=n_features, random_state=42
    )
    X = X.astype(np.float32)

    # Compute KNN once to use for both implementations
    # This ensures both cuML and umap-learn use identical neighbors
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(X)
    knn_dists, knn_indices = knn.kneighbors(X)

    # Test Dense with precomputed KNN
    cu_model = cuUMAP(n_neighbors=n_neighbors, random_state=42)
    cu_model.fit(X, knn_graph=(knn_indices, knn_dists))

    # Compare with umap-learn using the same precomputed KNN
    ref_model = umap.UMAP(
        n_neighbors=n_neighbors,
        random_state=42,
        precomputed_knn=(knn_indices, knn_dists),
    )
    ref_model.fit(X)

    # With identical KNN inputs, sigmas and rhos should match closely
    np.testing.assert_allclose(
        cu_model._rhos.get(), ref_model._rhos, atol=1e-3
    )

    # Sigmas are more sensitive to numerical differences
    np.testing.assert_allclose(
        cu_model._sigmas.get(), ref_model._sigmas, atol=5e-2
    )


@pytest.mark.xfail(
    UMAP_NUMBA_REGRESSION and (platform.machine() == "x86_64"),
    reason="Upstream regression in umap<0.5.12 with numba >= 0.62.0",
    strict=True,
)
def test_inverse_transform():
    """Test cuML UMAP inverse_transform produces valid output."""
    X, _ = make_blobs(n_samples=200, n_features=10, centers=4, random_state=42)
    X = X.astype(np.float32)

    random_state = 42
    umap_model = cuUMAP(
        n_neighbors=15, n_components=2, random_state=random_state
    )
    embedding = umap_model.fit_transform(X)

    def to_np(arr):
        return cp.asnumpy(arr) if hasattr(arr, "get") else np.asarray(arr)

    embedding_np = to_np(embedding)

    # Inverse transform the entire embedding
    inv = umap_model.inverse_transform(embedding_np)
    inv_np = to_np(inv)

    # 1. Basic checks
    assert inv_np.shape == (X.shape[0], X.shape[1])
    assert np.all(np.isfinite(inv_np))

    # 2. Neighborhood preservation check (KDTree check like upstream)
    # Reconstructed high-D points should have similar neighbors as their low-D counterparts
    highd_tree = KDTree(X)
    lowd_tree = KDTree(embedding_np)

    overlaps = []
    for i in range(X.shape[0]):
        # Find neighbors in low-D space for the original point
        query_point_low = embedding_np[i : i + 1]
        near_indices_low = lowd_tree.query(
            query_point_low, k=10, return_distance=False
        )[0]

        # Find neighbors of reconstructed point in high-D space
        query_point_high = inv_np[i : i + 1]
        near_indices_high = highd_tree.query(
            query_point_high, k=15, return_distance=False
        )[0]

        # Check for intersection of neighborhoods
        intersection = np.intersect1d(near_indices_low, near_indices_high)
        overlaps.append(intersection.shape[0])

    # We expect a significant average overlap.
    mean_overlap = np.mean(overlaps)
    assert mean_overlap >= 6.5, (
        f"Average neighborhood overlap too low: {mean_overlap:.2f}. "
        "Reconstructed points do not preserve local structure well."
    )

    # Ensure most points have at least some overlap
    overlap_count = np.sum(np.array(overlaps) >= 2)
    overlap_ratio = overlap_count / X.shape[0]
    assert overlap_ratio >= 0.95, (
        f"Only {overlap_ratio:.2%} of points have >= 2 common neighbors. "
        "Neighborhood preservation is poor for too many points."
    )

    # 3. Trustworthiness check
    # Check if the high-D reconstruction preserves the neighborhood structure of the low-D embedding
    trust = trustworthiness(embedding_np, inv_np, n_neighbors=10)
    assert trust > 0.95, (
        f"Trustworthiness of reconstruction is too low: {trust:.2f}"
    )

    # 4. Comparison to Reference implementation quality
    ref_model = umap.UMAP(
        n_neighbors=15, n_components=2, random_state=random_state
    )
    ref_model.fit(X)
    ref_inv = ref_model.inverse_transform(embedding_np)
    ref_inv_np = to_np(ref_inv)

    nn = NearestNeighbors(n_neighbors=1).fit(X)
    dists_cu, _ = nn.kneighbors(inv_np)
    dists_ref, _ = nn.kneighbors(ref_inv_np)

    mean_dist_cu = np.mean(dists_cu)
    mean_dist_ref = np.mean(dists_ref)

    # cuML reconstruction should be comparable to reference in terms of distance to training data
    assert mean_dist_cu < mean_dist_ref * 2.0, (
        f"cuML reconstruction quality ({mean_dist_cu:.4f}) is significantly worse "
        f"than reference ({mean_dist_ref:.4f})"
    )


def test_inverse_transform_sparse_error():
    """Test that inverse_transform raises error for sparse input data."""
    # Create sparse data
    X_sparse = sp.random(100, 20, density=0.3, format="csr", random_state=42)
    X_sparse = X_sparse.astype(np.float32)

    # Fit UMAP on sparse data
    umap_model = cuUMAP(
        n_neighbors=10,
        n_components=2,
        random_state=42,
        verbose=False,
    )
    umap_model.fit(X_sparse)

    # inverse_transform should raise ValueError for sparse input
    embedding = umap_model.embedding_
    embedding_np = (
        cp.asnumpy(embedding)
        if isinstance(embedding, cp.ndarray)
        else np.asarray(embedding)
    )

    with pytest.raises(ValueError, match="sparse"):
        umap_model.inverse_transform(embedding_np[:10])


def test_inverse_transform_dimension_mismatch():
    """Test that inverse_transform raises error for wrong embedding dimension."""
    X, _ = make_blobs(
        n_samples=100,
        n_features=10,
        centers=3,
        random_state=42,
    )
    X = X.astype(np.float32)

    # Fit UMAP with 2 components
    umap_model = cuUMAP(
        n_neighbors=10,
        n_components=2,
        random_state=42,
        verbose=False,
    )
    umap_model.fit(X)

    # Try to inverse_transform with wrong number of components
    wrong_embedding = np.random.rand(10, 3).astype(np.float32)

    with pytest.raises(ValueError, match="components"):
        umap_model.inverse_transform(wrong_embedding)


def test_get_feature_names_out():
    X, _ = make_blobs(n_features=5, random_state=42)
    cu_model = cuUMAP(n_components=2).fit(X)
    res = cu_model.get_feature_names_out()
    sol = np.array(["umap0", "umap1"], dtype=object)
    np.testing.assert_array_equal(res, sol)
