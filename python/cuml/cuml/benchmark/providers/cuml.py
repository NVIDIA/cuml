# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Define the provider for native single-GPU cuML estimators."""

from ..backends import get_backend
from .base import EstimatorSpec, Provider

CATALOG = {
    "KMeans": EstimatorSpec("cuml.cluster", "KMeans", "cuml"),
    "DBSCAN": EstimatorSpec("cuml.cluster", "DBSCAN", "cuml"),
    "SpectralClustering": EstimatorSpec(
        "cuml.cluster", "SpectralClustering", "cuml"
    ),
    "AgglomerativeClustering": EstimatorSpec(
        "cuml.cluster", "AgglomerativeClustering", "cuml"
    ),
    "EmpiricalCovariance": EstimatorSpec(
        "cuml.covariance", "EmpiricalCovariance", "cuml"
    ),
    "LedoitWolf": EstimatorSpec("cuml.covariance", "LedoitWolf", "cuml"),
    "PCA": EstimatorSpec("cuml.decomposition", "PCA", "cuml"),
    "IncrementalPCA": EstimatorSpec(
        "cuml.decomposition", "IncrementalPCA", "cuml"
    ),
    "TruncatedSVD": EstimatorSpec(
        "cuml.decomposition", "TruncatedSVD", "cuml"
    ),
    "GaussianRandomProjection": EstimatorSpec(
        "cuml.random_projection", "GaussianRandomProjection", "cuml"
    ),
    "SparseRandomProjection": EstimatorSpec(
        "cuml.random_projection", "SparseRandomProjection", "cuml"
    ),
    "NearestNeighbors": EstimatorSpec(
        "cuml.neighbors", "NearestNeighbors", "cuml"
    ),
    "KNeighborsClassifier": EstimatorSpec(
        "cuml.neighbors", "KNeighborsClassifier", "cuml"
    ),
    "KNeighborsRegressor": EstimatorSpec(
        "cuml.neighbors", "KNeighborsRegressor", "cuml"
    ),
    "KernelDensity": EstimatorSpec("cuml.neighbors", "KernelDensity", "cuml"),
    "LinearRegression": EstimatorSpec(
        "cuml.linear_model", "LinearRegression", "cuml"
    ),
    "LogisticRegression": EstimatorSpec(
        "cuml.linear_model", "LogisticRegression", "cuml"
    ),
    "ElasticNet": EstimatorSpec("cuml.linear_model", "ElasticNet", "cuml"),
    "Ridge": EstimatorSpec("cuml.linear_model", "Ridge", "cuml"),
    "Lasso": EstimatorSpec("cuml.linear_model", "Lasso", "cuml"),
    "KernelRidge": EstimatorSpec("cuml.kernel_ridge", "KernelRidge", "cuml"),
    "RandomForestClassifier": EstimatorSpec(
        "cuml.ensemble", "RandomForestClassifier", "cuml"
    ),
    "RandomForestRegressor": EstimatorSpec(
        "cuml.ensemble", "RandomForestRegressor", "cuml"
    ),
    "TSNE": EstimatorSpec("cuml.manifold", "TSNE", "cuml"),
    "SpectralEmbedding": EstimatorSpec(
        "cuml.manifold", "SpectralEmbedding", "cuml"
    ),
    "SVC": EstimatorSpec("cuml.svm", "SVC", "cuml"),
    "SVR": EstimatorSpec("cuml.svm", "SVR", "cuml"),
    "LinearSVC": EstimatorSpec("cuml.svm", "LinearSVC", "cuml"),
    "LinearSVR": EstimatorSpec("cuml.svm", "LinearSVR", "cuml"),
    "MultinomialNB": EstimatorSpec(
        "cuml.naive_bayes", "MultinomialNB", "cuml"
    ),
    "BernoulliNB": EstimatorSpec("cuml.naive_bayes", "BernoulliNB", "cuml"),
    "ComplementNB": EstimatorSpec("cuml.naive_bayes", "ComplementNB", "cuml"),
    "CategoricalNB": EstimatorSpec(
        "cuml.naive_bayes", "CategoricalNB", "cuml"
    ),
    "GaussianNB": EstimatorSpec("cuml.naive_bayes", "GaussianNB", "cuml"),
    "StandardScaler": EstimatorSpec(
        "cuml.preprocessing", "StandardScaler", "cuml"
    ),
    "MinMaxScaler": EstimatorSpec(
        "cuml.preprocessing", "MinMaxScaler", "cuml"
    ),
    "MaxAbsScaler": EstimatorSpec(
        "cuml.preprocessing", "MaxAbsScaler", "cuml"
    ),
    "Normalizer": EstimatorSpec("cuml.preprocessing", "Normalizer", "cuml"),
    "RobustScaler": EstimatorSpec(
        "cuml.preprocessing", "RobustScaler", "cuml"
    ),
    "PolynomialFeatures": EstimatorSpec(
        "cuml.preprocessing", "PolynomialFeatures", "cuml"
    ),
    "Binarizer": EstimatorSpec("cuml.preprocessing", "Binarizer", "cuml"),
    "KBinsDiscretizer": EstimatorSpec(
        "cuml.preprocessing", "KBinsDiscretizer", "cuml"
    ),
    "PowerTransformer": EstimatorSpec(
        "cuml.preprocessing", "PowerTransformer", "cuml"
    ),
    "QuantileTransformer": EstimatorSpec(
        "cuml.preprocessing", "QuantileTransformer", "cuml"
    ),
    "OneHotEncoder": EstimatorSpec(
        "cuml.preprocessing", "OneHotEncoder", "cuml"
    ),
    "OrdinalEncoder": EstimatorSpec(
        "cuml.preprocessing", "OrdinalEncoder", "cuml"
    ),
    "LabelEncoder": EstimatorSpec(
        "cuml.preprocessing", "LabelEncoder", "cuml"
    ),
    "LabelBinarizer": EstimatorSpec(
        "cuml.preprocessing", "LabelBinarizer", "cuml"
    ),
    "TargetEncoder": EstimatorSpec(
        "cuml.preprocessing", "TargetEncoder", "cuml"
    ),
    "UMAP": EstimatorSpec("cuml.manifold", "UMAP", "cuml"),
    "HDBSCAN": EstimatorSpec("cuml.cluster", "HDBSCAN", "cuml"),
}

PROVIDER = Provider(get_backend("cuml"), CATALOG)
