# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hdbscan
import numpy as np
import pytest
import umap
from packaging.version import Version
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_classification, make_regression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.neighbors import (
    KernelDensity,
    KNeighborsClassifier,
    KNeighborsRegressor,
    NearestNeighbors,
)
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR

ort = pytest.importorskip("onnxruntime")
skl2onnx = pytest.importorskip("skl2onnx")
ORT_VERSION = Version(ort.__version__)

convert_sklearn = skl2onnx.convert_sklearn
FloatTensorType = skl2onnx.common.data_types.FloatTensorType

# Which estimators are supported and not is also mentioned in the cuml.accel docs,
# make sure to update the docs if you make changes here.
xfail_unsupported = pytest.mark.xfail(
    reason="not supported by skl2onnx", strict=True
)
xfail_skl2onnx_bug = pytest.mark.xfail(
    reason="skl2onnx conversion error", strict=True
)


@pytest.fixture(scope="module")
def classification_data():
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=5, random_state=42
    )
    X = X.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def regression_data():
    X, y = make_regression(
        n_samples=500, n_features=10, n_informative=5, random_state=42
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


classifiers = [
    pytest.param(
        RandomForestClassifier(n_estimators=20, max_depth=8, random_state=42),
        marks=(xfail_skl2onnx_bug if ORT_VERSION < Version("1.28.0") else ()),
        id="RandomForestClassifier",
    ),
    pytest.param(KNeighborsClassifier(), id="KNeighborsClassifier"),
    pytest.param(LinearSVC(dual="auto"), id="LinearSVC"),
    pytest.param(
        SVC(kernel="linear"),
        id="SVC",
        marks=pytest.mark.filterwarnings(
            "ignore:Attribute `prob[A|B]_` was deprecated:FutureWarning"
        ),
    ),
    pytest.param(
        OneVsOneClassifier(LinearSVC(dual="auto")),
        marks=xfail_skl2onnx_bug,
        id="OneVsOneClassifier",
    ),
    pytest.param(
        OneVsRestClassifier(LinearSVC(dual="auto")),
        marks=xfail_skl2onnx_bug,
        id="OneVsRestClassifier",
    ),
]

regressors = [
    pytest.param(
        RandomForestRegressor(n_estimators=20, max_depth=8, random_state=42),
        id="RandomForestRegressor",
    ),
    pytest.param(KNeighborsRegressor(), id="KNeighborsRegressor"),
    pytest.param(LinearSVR(dual="auto"), id="LinearSVR"),
    pytest.param(SVR(kernel="linear"), id="SVR"),
]

transformers = [
    pytest.param(PCA(n_components=5), id="PCA"),
    pytest.param(TruncatedSVD(n_components=5), id="TruncatedSVD"),
    pytest.param(
        KMeans(n_clusters=3, random_state=42, n_init=10), id="KMeans"
    ),
    pytest.param(
        NearestNeighbors(n_neighbors=5),
        marks=xfail_skl2onnx_bug,
        id="NearestNeighbors",
    ),
]

unsupported = [
    pytest.param(DBSCAN(eps=0.5), marks=xfail_unsupported, id="DBSCAN"),
    pytest.param(
        SpectralEmbedding(n_components=2),
        marks=xfail_unsupported,
        id="SpectralEmbedding",
    ),
    pytest.param(TSNE(n_components=2), marks=xfail_unsupported, id="TSNE"),
    pytest.param(KernelDensity(), marks=xfail_unsupported, id="KernelDensity"),
    pytest.param(hdbscan.HDBSCAN(), marks=xfail_unsupported, id="HDBSCAN"),
    pytest.param(umap.UMAP(), marks=xfail_unsupported, id="UMAP"),
]


def to_onnx(model, n_features):
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    options = {}
    if hasattr(model, "predict_proba"):
        options["zipmap"] = False
    return convert_sklearn(model, initial_types=initial_type, options=options)


def onnx_predict(onnx_model, X):
    sess = ort.InferenceSession(onnx_model.SerializeToString())
    input_name = sess.get_inputs()[0].name
    return sess.run(None, {input_name: X})


@pytest.mark.parametrize("estimator", classifiers)
def test_onnx_classifier(estimator, classification_data):
    X_train, X_test, y_train, _ = classification_data

    estimator.fit(X_train, y_train)
    accel_preds = np.asarray(estimator.predict(X_test))

    onnx_model = to_onnx(estimator, X_test.shape[1])
    onnx_results = onnx_predict(onnx_model, X_test)
    onnx_preds = np.asarray(onnx_results[0])

    match_rate = np.mean(accel_preds == onnx_preds)
    assert match_rate >= 0.99, f"Prediction match rate {match_rate:.4f} < 0.99"


@pytest.mark.parametrize("estimator", regressors)
def test_onnx_regressor(estimator, regression_data):
    X_train, X_test, y_train, _ = regression_data

    estimator.fit(X_train, y_train)
    accel_preds = np.asarray(estimator.predict(X_test)).flatten()

    onnx_model = to_onnx(estimator, X_test.shape[1])
    onnx_results = onnx_predict(onnx_model, X_test)
    onnx_preds = np.asarray(onnx_results[0]).flatten()

    max_diff = np.max(np.abs(accel_preds - onnx_preds))
    assert max_diff < 1e-2, f"Max prediction diff {max_diff:.6e} >= 1e-2"


@pytest.mark.parametrize("estimator", transformers + unsupported)
def test_onnx_transformer(estimator, classification_data):
    X_train, X_test, _, _ = classification_data

    name = type(estimator).__name__
    estimator.fit(X_train)

    onnx_model = to_onnx(estimator, X_test.shape[1])
    onnx_results = onnx_predict(onnx_model, X_test)

    if name == "NearestNeighbors":
        accel_dist, accel_idx = estimator.kneighbors(X_test)
        accel_dist = np.asarray(accel_dist).astype(np.float32)
        onnx_dist = np.asarray(onnx_results[1]).astype(np.float32)
        onnx_idx = np.asarray(onnx_results[0])
        np.testing.assert_array_equal(accel_idx, onnx_idx)
        np.testing.assert_allclose(accel_dist, onnx_dist, atol=1e-5)
    elif name == "KMeans":
        accel_out = np.asarray(estimator.predict(X_test))
        onnx_out = np.asarray(onnx_results[0])
        np.testing.assert_array_equal(accel_out, onnx_out)
    else:
        accel_out = np.asarray(estimator.transform(X_test))
        onnx_out = np.asarray(onnx_results[0])
        np.testing.assert_allclose(accel_out, onnx_out, atol=1e-5, rtol=1e-5)
