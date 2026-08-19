# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from sklearn.cluster import DBSCAN, HDBSCAN, KMeans, SpectralClustering
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    NearestNeighbors,
)
from sklearn.preprocessing import StandardScaler, TargetEncoder


def test_kmeans():
    X, y_true = make_blobs(n_samples=100, centers=3, random_state=42)
    clf = KMeans().fit(X)
    clf.predict(X)


def test_dbscan():
    X, y_true = make_blobs(n_samples=100, centers=3, random_state=42)
    clf = DBSCAN().fit(X)
    clf.labels_


def test_hdbscan():
    X, y_true = make_blobs(n_samples=100, centers=3, random_state=42)
    clf = HDBSCAN(copy=False).fit(X)
    clf.labels_


def test_spectral_clustering():
    X, y_true = make_blobs(n_samples=100, centers=3, random_state=42)
    X = X.astype("float32")
    sc = SpectralClustering(
        n_clusters=3, affinity="nearest_neighbors", random_state=42
    ).fit(X)
    sc.labels_


def test_pca():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    pca = PCA().fit(X)
    pca.transform(X)


def test_truncated_svd():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    svd = TruncatedSVD().fit(X)
    svd.transform(X)


def test_standard_scaler():
    import numpy as np

    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    scaler = StandardScaler().fit(X)

    # Check fitted attributes exist
    assert hasattr(scaler, "mean_")
    assert hasattr(scaler, "var_")
    assert hasattr(scaler, "scale_")
    assert scaler.mean_.shape == (X.shape[1],)
    assert scaler.var_.shape == (X.shape[1],)
    assert scaler.scale_.shape == (X.shape[1],)

    # Transform and check shape
    X_transformed = scaler.transform(X)
    assert X_transformed.shape == X.shape

    # Check that transformed data has mean ≈ 0 and std ≈ 1
    assert np.allclose(X_transformed.mean(axis=0), 0, atol=1e-7)
    assert np.allclose(X_transformed.std(axis=0), 1, atol=1e-7)

    # Check inverse transform
    X_inverse = scaler.inverse_transform(X_transformed)
    assert X_inverse.shape == X.shape
    assert np.allclose(X_inverse, X, atol=1e-6)


def test_linear_regression():
    X, y = make_regression(
        n_samples=100, n_features=20, noise=0.1, random_state=42
    )
    lr = LinearRegression().fit(X, y)
    lr.predict(X)


def test_logistic_regression():
    X, y = make_classification(
        n_samples=100, n_features=20, n_classes=2, random_state=42
    )
    clf = LogisticRegression().fit(X, y)
    clf.predict(X)


def test_elastic_net():
    X, y = make_regression(
        n_samples=100, n_features=20, noise=0.1, random_state=42
    )
    enet = ElasticNet().fit(X, y)
    enet.predict(X)


def test_ridge():
    X, y = make_regression(
        n_samples=100, n_features=20, noise=0.1, random_state=42
    )
    ridge = Ridge().fit(X, y)
    ridge.predict(X)


def test_lasso():
    X, y = make_regression(
        n_samples=100, n_features=20, noise=0.1, random_state=42
    )
    lasso = Lasso().fit(X, y)
    lasso.predict(X)


def test_tsne():
    X, _ = make_blobs(n_samples=100, centers=3, n_features=20, random_state=42)
    tsne = TSNE()
    tsne.fit_transform(X)


def test_spectral_embedding():
    X, _ = make_blobs(n_samples=100, centers=3, n_features=20, random_state=42)
    se = SpectralEmbedding(n_components=2, random_state=42)
    se.fit_transform(X)


def test_nearest_neighbors():
    X, _ = make_blobs(n_samples=100, centers=3, n_features=20, random_state=42)
    nn = NearestNeighbors().fit(X)
    distances, indices = nn.kneighbors(X)
    assert distances.shape == (100, 5)
    assert indices.shape == (100, 5)


def test_k_neighbors_classifier():
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        n_classes=3,
        random_state=42,
        n_informative=6,
    )
    for weights in ["uniform", "distance"]:
        for metric in ["euclidean", "manhattan"]:
            knn = KNeighborsClassifier().fit(X, y)
            knn.predict(X)


def test_k_neighbors_regressor():
    X, y = make_regression(
        n_samples=100, n_features=20, noise=0.1, random_state=42
    )
    for weights in ["uniform", "distance"]:
        for metric in ["euclidean", "manhattan"]:
            knr = KNeighborsRegressor().fit(X, y)
            knr.predict(X)


def test_target_encoder():
    import numpy as np

    X = np.array(
        [
            ["cat", "small"],
            ["dog", "medium"],
            ["cat", "large"],
            ["dog", "small"],
            ["cat", "medium"],
            ["dog", "large"],
            ["cat", "small"],
            ["dog", "medium"],
        ]
    )

    y = np.array([1.0, 2.5, 1.5, 3.0, 1.2, 2.8, 1.1, 2.6])

    encoder = TargetEncoder()
    X_encoded = encoder.fit_transform(X, y)

    assert X_encoded.shape == X.shape

    X_new = np.array([["cat", "small"], ["dog", "large"], ["cat", "medium"]])
    X_new_encoded = encoder.transform(X_new)
    assert X_new_encoded.shape == X_new.shape

    encoder_smooth = TargetEncoder(smooth="auto")
    X_encoded_smooth = encoder_smooth.fit_transform(X, y)
    assert X_encoded_smooth.shape == X.shape
