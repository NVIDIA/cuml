# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect

import cupy as cp
import numpy as np
import pandas as pd
import pytest
import scipy.sparse
from hdbscan import HDBSCAN as ContribHDBSCAN
from sklearn.cluster import HDBSCAN
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score, pairwise_distances

from cuml.accel import is_proxy
from cuml.cluster import HDBSCAN as CumlHDBSCAN
from cuml.internals.interop import UnsupportedOnGPU

CPUHDBSCAN = HDBSCAN._cpu_class
COPY_DEFAULT_WARNS = (
    inspect.signature(CPUHDBSCAN).parameters["copy"].default == "warn"
)


@pytest.fixture(scope="module")
def blobs():
    return make_blobs(n_samples=100, centers=3, random_state=42)[0]


def _gpu_params(**kwargs):
    model = HDBSCAN(copy=False, **kwargs)
    return model._gpu_class._params_from_cpu(model._cpu)


def _assert_cpu_fallback(model):
    assert model._gpu is None
    assert type(model._cpu) is CPUHDBSCAN


def _with_nan(X):
    X = X.copy()
    X[0, 0] = np.nan
    return X


def test_hdbscan_implementations_are_distinct_proxies():
    assert is_proxy(HDBSCAN)
    assert is_proxy(ContribHDBSCAN)
    assert HDBSCAN is not ContribHDBSCAN
    assert HDBSCAN._cpu_class is CPUHDBSCAN
    assert HDBSCAN._gpu_class._cpu_class_path == "sklearn.cluster.HDBSCAN"
    assert ContribHDBSCAN._gpu_class._cpu_class_path == "hdbscan.HDBSCAN"
    assert CumlHDBSCAN._cpu_class_path == "hdbscan.HDBSCAN"
    assert inspect.signature(HDBSCAN) == inspect.signature(CPUHDBSCAN)
    assert inspect.signature(ContribHDBSCAN) == inspect.signature(
        ContribHDBSCAN._cpu_class
    )
    assert set(HDBSCAN().get_params()) != set(ContribHDBSCAN().get_params())


@pytest.mark.skipif(
    not COPY_DEFAULT_WARNS,
    reason="This sklearn version does not use copy='warn'",
)
def test_hdbscan_copy_default_warning(blobs):
    msg = (
        r"The default value of `copy` will change from False to True in 1.10."
    )
    with pytest.warns(FutureWarning, match=msg):
        model = HDBSCAN().fit(blobs)

    assert model._gpu is not None


@pytest.mark.parametrize(
    "kwargs,name,expected",
    [
        ({}, "min_samples", 4),
        ({"min_samples": 2}, "min_samples", 1),
        ({"min_cluster_size": 20, "min_samples": 8}, "min_samples", 7),
        ({"min_samples": 1024}, "min_samples", 1023),
        ({}, "max_cluster_size", 0),
        ({"max_cluster_size": 25}, "max_cluster_size", 25),
    ],
)
def test_hdbscan_parameter_translation(kwargs, name, expected):
    assert _gpu_params(**kwargs)[name] == expected


@pytest.mark.parametrize("min_samples", [1, 1025])
def test_hdbscan_min_samples_gpu_bound(min_samples):
    with pytest.raises(UnsupportedOnGPU):
        _gpu_params(min_samples=min_samples)


def test_hdbscan_min_samples_semantics():
    X, _ = make_moons(n_samples=200, noise=0.12, random_state=1)
    params = {
        "min_cluster_size": 50,
        "min_samples": 4,
        "copy": False,
    }

    expected = CPUHDBSCAN(**params).fit_predict(X)
    off_by_one = CPUHDBSCAN(**{**params, "min_samples": 5}).fit_predict(X)
    assert adjusted_rand_score(expected, off_by_one) <= 0.1

    result = HDBSCAN(**params)
    labels = result.fit_predict(X)

    assert result._gpu.min_samples == 3
    expected_score = adjusted_rand_score(expected, labels)
    off_by_one_score = adjusted_rand_score(off_by_one, labels)
    assert expected_score > off_by_one_score


@pytest.mark.parametrize(
    "n_samples,min_cluster_size,min_samples,method,algorithm",
    [
        (100, 5, None, "eom", "auto"),
        (1_000, 15, 8, "leaf", "brute"),
        (10_000, 30, 15, "eom", "kd_tree"),
    ],
)
def test_hdbscan_cpu_gpu_agreement(
    n_samples, min_cluster_size, min_samples, method, algorithm
):
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=8,
        centers=5,
        cluster_std=0.7,
        random_state=42,
    )
    params = {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "cluster_selection_method": method,
        "algorithm": algorithm,
        "copy": False,
    }

    expected = CPUHDBSCAN(**params).fit(X)
    result = HDBSCAN(**params).fit(X)

    assert result._gpu is not None
    assert adjusted_rand_score(expected.labels_, result.labels_) >= 0.95


def test_hdbscan_fit_predict_and_dbscan_clustering():
    X, _ = make_moons(n_samples=500, noise=0.08, random_state=42)
    X = pd.DataFrame(X, columns=["x", "y"])
    params = {"min_cluster_size": 10, "min_samples": 6, "copy": False}

    expected = CPUHDBSCAN(**params).fit(X)
    result = HDBSCAN(**params)
    labels = result.fit_predict(X)

    assert result._gpu is not None
    assert adjusted_rand_score(expected.labels_, labels) >= 0.95

    # Accessing fitted attributes synchronizes a valid sklearn model without
    # dropping the GPU model.
    assert labels.dtype == np.dtype(np.intp)
    assert result.probabilities_.dtype == np.dtype(np.float64)
    assert labels.shape == result.probabilities_.shape == (len(X),)
    np.testing.assert_array_equal(result.feature_names_in_, X.columns)
    assert result.n_features_in_ == X.shape[1]
    assert result._cpu._raw_data.dtype == np.dtype(np.float64)
    assert result._cpu._min_samples == result.min_samples
    # sklearn uses this private fitted attribute in its HDBSCAN helpers. It
    # must be available through the proxy, not only on the synchronized model.
    linkage_tree = result._single_linkage_tree_
    assert linkage_tree.dtype.names == (
        "left_node",
        "right_node",
        "value",
        "cluster_size",
    )
    np.testing.assert_array_equal(
        linkage_tree, result._cpu._single_linkage_tree_
    )

    expected_cut = expected.dbscan_clustering(
        cut_distance=0.25, min_cluster_size=5
    )
    result_cut = result.dbscan_clustering(
        cut_distance=0.25, min_cluster_size=5
    )
    assert adjusted_rand_score(expected_cut, result_cut) >= 0.95
    assert result._gpu is not None

    device_labels = HDBSCAN(copy=False).fit_predict(cp.asarray(X.to_numpy()))
    assert isinstance(device_labels, np.ndarray)


def test_hdbscan_single_cluster():
    X, _ = make_blobs(
        n_samples=500,
        centers=1,
        cluster_std=0.25,
        random_state=42,
    )
    params = {
        "min_cluster_size": 10,
        "allow_single_cluster": True,
        "copy": False,
    }

    expected = CPUHDBSCAN(**params).fit_predict(X)
    result = HDBSCAN(**params).fit_predict(X)

    clustered = result >= 0
    assert np.unique(result[clustered]).size == 1
    assert clustered.sum() >= params["min_cluster_size"]

    # ARI treats every noise point as belonging to one ordinary second
    # cluster, making it misleading for a single cluster plus noise. Compare
    # the per-sample cluster/noise decision directly instead.
    assert np.mean((expected >= 0) == (result >= 0)) >= 0.9


def test_hdbscan_parameter_updates(blobs):
    model = HDBSCAN(copy=False).fit(blobs)

    model.set_params(
        min_samples=8,
        max_cluster_size=25,
        algorithm="brute",
        leaf_size=25,
        n_jobs=2,
    )
    assert model._gpu is not None
    assert model._gpu.min_samples == 7
    assert model._gpu.max_cluster_size == 25
    assert model.fit(blobs)._gpu is not None

    model.set_params(store_centers="centroid")
    assert model._gpu is None
    model.fit(blobs)
    _assert_cpu_fallback(model)


@pytest.mark.parametrize(
    "kwargs,transform,expected_label",
    [
        pytest.param({"metric": "manhattan"}, lambda X: X, None, id="metric"),
        pytest.param(
            {"metric": "minkowski", "metric_params": {"p": 2}},
            lambda X: X,
            None,
            id="metric-params",
        ),
        pytest.param(
            {"store_centers": "centroid"},
            lambda X: X,
            None,
            id="centers",
        ),
        pytest.param({}, scipy.sparse.csr_matrix, None, id="sparse-input"),
        pytest.param(
            {"metric": "precomputed"},
            pairwise_distances,
            None,
            id="precomputed-input",
        ),
        pytest.param({}, _with_nan, -3, id="nonfinite-input"),
    ],
)
def test_hdbscan_fallback_uses_sklearn(
    blobs, kwargs, transform, expected_label
):
    model = HDBSCAN(copy=False, **kwargs).fit(transform(blobs))
    _assert_cpu_fallback(model)
    assert model.labels_.shape == (len(blobs),)
    if kwargs.get("store_centers"):
        assert model.centroids_.shape[1] == blobs.shape[1]
    if expected_label is not None:
        assert model.labels_[0] == expected_label


def test_hdbscan_data_dependent_min_samples_falls_back(blobs):
    model = HDBSCAN(min_samples=101, copy=False)

    with pytest.raises(ValueError, match="min_samples .* must be at most"):
        model.fit(blobs)

    _assert_cpu_fallback(model)


def test_hdbscan_min_samples_one_falls_back(blobs):
    params = {"min_samples": 1, "copy": False}
    expected = CPUHDBSCAN(**params).fit_predict(blobs)

    model = HDBSCAN(**params)
    result = model.fit_predict(blobs)

    _assert_cpu_fallback(model)
    np.testing.assert_array_equal(result, expected)


def test_hdbscan_complex_input_uses_sklearn_validation(blobs):
    with pytest.raises(ValueError, match="Complex data not supported"):
        HDBSCAN(copy=False).fit(blobs.astype(np.complex128))
