# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Define accelerated bindings to upstream estimator providers."""

from ..backends import get_backend
from .base import Provider
from .hdbscan import PROVIDER as HDBSCAN_PROVIDER
from .sklearn import PROVIDER as SKLEARN_PROVIDER
from .umap import PROVIDER as UMAP_PROVIDER

# Supported sklearn bindings; UMAP and HDBSCAN are selected separately below.
_SKLEARN_ESTIMATORS = (
    "DBSCAN",
    "ElasticNet",
    "EmpiricalCovariance",
    "IncrementalPCA",
    "KMeans",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    "KernelDensity",
    "KernelRidge",
    "LabelBinarizer",
    "LabelEncoder",
    "Lasso",
    "LedoitWolf",
    "LinearRegression",
    "LinearSVC",
    "LinearSVR",
    "LogisticRegression",
    "MaxAbsScaler",
    "MinMaxScaler",
    "NearestNeighbors",
    "OneHotEncoder",
    "PCA",
    "PolynomialFeatures",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "Ridge",
    "SVC",
    "SVR",
    "SpectralClustering",
    "SpectralEmbedding",
    "StandardScaler",
    "TSNE",
    "TargetEncoder",
    "TruncatedSVD",
)

CATALOG = {
    **{
        name: SKLEARN_PROVIDER.estimator_spec(name)
        for name in _SKLEARN_ESTIMATORS
    },
    "UMAP": UMAP_PROVIDER.estimator_spec("UMAP"),
    # Accelerate standalone hdbscan, not sklearn.cluster.HDBSCAN.
    "HDBSCAN": HDBSCAN_PROVIDER.estimator_spec("HDBSCAN"),
}

PROVIDER = Provider(get_backend("cuml.accel"), CATALOG)
