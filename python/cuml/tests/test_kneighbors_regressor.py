# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import cudf
import cupy as cp
import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor as skKNN
from sklearn.utils.validation import check_random_state

from cuml.neighbors import KNeighborsRegressor as cuKNN
from cuml.testing.utils import array_equal


def test_kneighbors_regressor(
    n_samples=40, n_features=5, n_test_pts=10, n_neighbors=3, random_state=0
):
    # Test k-neighbors regression
    rng = np.random.RandomState(random_state)
    X = 2 * rng.rand(n_samples, n_features) - 1
    y = np.sqrt((X**2).sum(1))
    y /= y.max()

    y_target = y[:n_test_pts]

    knn = cuKNN(n_neighbors=n_neighbors)
    knn.fit(X, y)
    epsilon = 1e-5 * (2 * rng.rand(1, n_features) - 1)
    y_pred = knn.predict(X[:n_test_pts] + epsilon)
    assert np.all(abs(y_pred - y_target) < 0.3)


def test_kneighborsRegressor_multioutput_uniform_weight():
    # Test k-neighbors in multi-output regression with uniform weight
    rng = check_random_state(0)
    n_features = 5
    n_samples = 40
    n_output = 4

    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples, n_output)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    knn = cuKNN()
    knn.fit(X_train, y_train)

    neigh_idx = knn.kneighbors(X_test, return_distance=False).astype(np.int32)

    y_pred_idx = np.array([np.mean(y_train[idx], axis=0) for idx in neigh_idx])

    y_pred = knn.predict(X_test)

    assert y_pred.shape[0] == y_test.shape[0]
    assert y_pred_idx.shape == y_test.shape
    np.testing.assert_array_almost_equal(y_pred, y_pred_idx)


@pytest.mark.parametrize("datatype", ["dataframe", "numpy"])
@pytest.mark.parametrize("nrows", [1000, 10000])
@pytest.mark.parametrize("ncols", [50, 100])
@pytest.mark.parametrize("n_neighbors", [2, 5, 10])
@pytest.mark.parametrize("n_clusters", [2, 5, 10])
def test_score(nrows, ncols, n_neighbors, n_clusters, datatype):
    # Using make_blobs here to check averages and neighborhoods
    X, y = make_blobs(
        n_samples=nrows,
        centers=n_clusters,
        cluster_std=0.01,
        n_features=ncols,
        random_state=0,
    )

    X = X.astype(np.float32)
    y = y.astype(np.float32)

    if datatype == "dataframe":
        X = cudf.DataFrame(X)
        y = cudf.DataFrame(y.reshape(nrows, 1))

    knn_cu = cuKNN(n_neighbors=n_neighbors)
    knn_cu.fit(X, y)

    assert knn_cu.score(X, y) >= 0.9999


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_predict_and_score_dtype(dtype):
    # Using make_blobs here to check averages and neighborhoods
    X, y = make_blobs(
        n_samples=1000,
        centers=2,
        cluster_std=0.01,
        n_features=50,
        random_state=0,
    )

    X = X.astype(dtype)
    y = y.astype(dtype)

    knn_cu = cuKNN(n_neighbors=5)
    knn_cu.fit(X, y)
    assert knn_cu.predict(X).dtype == dtype
    assert knn_cu.score(X, y) >= 0.9999


@pytest.mark.parametrize("input_type", ["cudf", "numpy", "cupy"])
@pytest.mark.parametrize("output_type", ["cudf", "numpy", "cupy"])
def test_predict_multioutput(input_type, output_type):
    X = np.array([[0, 0, 1, 0], [1, 0, 1, 0]]).astype(np.float32)
    y = np.array([[15, 2], [5, 4]]).astype(np.int32)

    if input_type == "cudf":
        X = cudf.DataFrame(X)
        y = cudf.DataFrame(y)
    elif input_type == "cupy":
        X = cp.asarray(X)
        y = cp.asarray(y)

    knn_cu = cuKNN(n_neighbors=1, output_type=output_type)
    knn_cu.fit(X, y)

    p = knn_cu.predict(X)

    if output_type == "cudf":
        assert isinstance(p, cudf.DataFrame)
    elif output_type == "numpy":
        assert isinstance(p, np.ndarray)
    elif output_type == "cupy":
        assert isinstance(p, cp.ndarray)

    assert array_equal(p.astype(np.int32), y)


@pytest.mark.parametrize(
    "weights",
    [
        "uniform",
        "distance",
        lambda d: 1.0 / (1.0 + d),  # Custom callable
    ],
)
@pytest.mark.parametrize("n_neighbors", [3, 5, 10])
def test_weights_predict(weights, n_neighbors):
    """Test KNN regressor predict with uniform, distance, and callable weights."""
    X, y = make_regression(
        n_samples=500,
        n_features=20,
        n_informative=15,
        random_state=42,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    # Split data
    X_train, X_test = X[:400], X[400:]
    y_train = y[:400]

    # cuML model
    knn_cu = cuKNN(n_neighbors=n_neighbors, weights=weights)
    knn_cu.fit(X_train, y_train)
    pred_cu = knn_cu.predict(X_test)

    # scikit-learn model for comparison
    knn_sk = skKNN(n_neighbors=n_neighbors, weights=weights)
    knn_sk.fit(X_train, y_train)
    pred_sk = knn_sk.predict(X_test)

    # Results should match scikit-learn
    if isinstance(pred_cu, cp.ndarray):
        pred_cu = cp.asnumpy(pred_cu)
    np.testing.assert_allclose(pred_cu, pred_sk, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_weights_multioutput(weights):
    """Test weights parameter with multioutput regression."""
    rng = check_random_state(42)
    n_samples = 100
    n_features = 5
    n_outputs = 3

    X = rng.rand(n_samples, n_features).astype(np.float32)
    y = rng.rand(n_samples, n_outputs).astype(np.float32)

    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    # cuML model
    knn_cu = cuKNN(n_neighbors=5, weights=weights)
    knn_cu.fit(X_train, y_train)

    # Test predict
    pred = knn_cu.predict(X_test)
    assert pred.shape == y_test.shape

    # Compare with sklearn
    knn_sk = skKNN(n_neighbors=5, weights=weights)
    knn_sk.fit(X_train, y_train)
    pred_sk = knn_sk.predict(X_test)

    if isinstance(pred, cp.ndarray):
        pred = cp.asnumpy(pred)
    np.testing.assert_allclose(pred, pred_sk, rtol=1e-4, atol=1e-4)


def test_invalid_weights():
    """Test that invalid weights parameter raises appropriate error."""
    X = np.array([[0, 0], [1, 1]]).astype(np.float32)
    y = np.array([0.0, 1.0]).astype(np.float32)

    # Invalid string value should raise error during fit
    knn_cu = cuKNN(n_neighbors=1, weights="invalid")

    with pytest.raises(ValueError, match="weights must be"):
        knn_cu.fit(X, y)
