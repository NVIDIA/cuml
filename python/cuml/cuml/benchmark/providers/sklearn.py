# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Define the provider for scikit-learn CPU estimators."""

from ..backends import get_backend
from .base import EstimatorSpec, Provider

CATALOG = {
    "KMeans": EstimatorSpec("sklearn.cluster", "KMeans", "scikit-learn"),
    "DBSCAN": EstimatorSpec("sklearn.cluster", "DBSCAN", "scikit-learn"),
    "SpectralClustering": EstimatorSpec(
        "sklearn.cluster", "SpectralClustering", "scikit-learn"
    ),
    "AgglomerativeClustering": EstimatorSpec(
        "sklearn.cluster", "AgglomerativeClustering", "scikit-learn"
    ),
    "EmpiricalCovariance": EstimatorSpec(
        "sklearn.covariance", "EmpiricalCovariance", "scikit-learn"
    ),
    "LedoitWolf": EstimatorSpec(
        "sklearn.covariance", "LedoitWolf", "scikit-learn"
    ),
    "PCA": EstimatorSpec("sklearn.decomposition", "PCA", "scikit-learn"),
    "IncrementalPCA": EstimatorSpec(
        "sklearn.decomposition", "IncrementalPCA", "scikit-learn"
    ),
    "TruncatedSVD": EstimatorSpec(
        "sklearn.decomposition", "TruncatedSVD", "scikit-learn"
    ),
    "GaussianRandomProjection": EstimatorSpec(
        "sklearn.random_projection", "GaussianRandomProjection", "scikit-learn"
    ),
    "SparseRandomProjection": EstimatorSpec(
        "sklearn.random_projection", "SparseRandomProjection", "scikit-learn"
    ),
    "NearestNeighbors": EstimatorSpec(
        "sklearn.neighbors", "NearestNeighbors", "scikit-learn"
    ),
    "KNeighborsClassifier": EstimatorSpec(
        "sklearn.neighbors", "KNeighborsClassifier", "scikit-learn"
    ),
    "KNeighborsRegressor": EstimatorSpec(
        "sklearn.neighbors", "KNeighborsRegressor", "scikit-learn"
    ),
    "KernelDensity": EstimatorSpec(
        "sklearn.neighbors", "KernelDensity", "scikit-learn"
    ),
    "LinearRegression": EstimatorSpec(
        "sklearn.linear_model", "LinearRegression", "scikit-learn"
    ),
    "LogisticRegression": EstimatorSpec(
        "sklearn.linear_model", "LogisticRegression", "scikit-learn"
    ),
    "ElasticNet": EstimatorSpec(
        "sklearn.linear_model", "ElasticNet", "scikit-learn"
    ),
    "Ridge": EstimatorSpec("sklearn.linear_model", "Ridge", "scikit-learn"),
    "Lasso": EstimatorSpec("sklearn.linear_model", "Lasso", "scikit-learn"),
    "KernelRidge": EstimatorSpec(
        "sklearn.kernel_ridge", "KernelRidge", "scikit-learn"
    ),
    "RandomForestClassifier": EstimatorSpec(
        "sklearn.ensemble", "RandomForestClassifier", "scikit-learn"
    ),
    "RandomForestRegressor": EstimatorSpec(
        "sklearn.ensemble", "RandomForestRegressor", "scikit-learn"
    ),
    "TSNE": EstimatorSpec("sklearn.manifold", "TSNE", "scikit-learn"),
    "SpectralEmbedding": EstimatorSpec(
        "sklearn.manifold", "SpectralEmbedding", "scikit-learn"
    ),
    "SVC": EstimatorSpec("sklearn.svm", "SVC", "scikit-learn"),
    "SVR": EstimatorSpec("sklearn.svm", "SVR", "scikit-learn"),
    "LinearSVC": EstimatorSpec("sklearn.svm", "LinearSVC", "scikit-learn"),
    "LinearSVR": EstimatorSpec("sklearn.svm", "LinearSVR", "scikit-learn"),
    "MultinomialNB": EstimatorSpec(
        "sklearn.naive_bayes", "MultinomialNB", "scikit-learn"
    ),
    "BernoulliNB": EstimatorSpec(
        "sklearn.naive_bayes", "BernoulliNB", "scikit-learn"
    ),
    "ComplementNB": EstimatorSpec(
        "sklearn.naive_bayes", "ComplementNB", "scikit-learn"
    ),
    "CategoricalNB": EstimatorSpec(
        "sklearn.naive_bayes", "CategoricalNB", "scikit-learn"
    ),
    "GaussianNB": EstimatorSpec(
        "sklearn.naive_bayes", "GaussianNB", "scikit-learn"
    ),
    "StandardScaler": EstimatorSpec(
        "sklearn.preprocessing", "StandardScaler", "scikit-learn"
    ),
    "MinMaxScaler": EstimatorSpec(
        "sklearn.preprocessing", "MinMaxScaler", "scikit-learn"
    ),
    "MaxAbsScaler": EstimatorSpec(
        "sklearn.preprocessing", "MaxAbsScaler", "scikit-learn"
    ),
    "Normalizer": EstimatorSpec(
        "sklearn.preprocessing", "Normalizer", "scikit-learn"
    ),
    "RobustScaler": EstimatorSpec(
        "sklearn.preprocessing", "RobustScaler", "scikit-learn"
    ),
    "PolynomialFeatures": EstimatorSpec(
        "sklearn.preprocessing", "PolynomialFeatures", "scikit-learn"
    ),
    "Binarizer": EstimatorSpec(
        "sklearn.preprocessing", "Binarizer", "scikit-learn"
    ),
    "KBinsDiscretizer": EstimatorSpec(
        "sklearn.preprocessing", "KBinsDiscretizer", "scikit-learn"
    ),
    "PowerTransformer": EstimatorSpec(
        "sklearn.preprocessing", "PowerTransformer", "scikit-learn"
    ),
    "QuantileTransformer": EstimatorSpec(
        "sklearn.preprocessing", "QuantileTransformer", "scikit-learn"
    ),
    "OneHotEncoder": EstimatorSpec(
        "sklearn.preprocessing", "OneHotEncoder", "scikit-learn"
    ),
    "OrdinalEncoder": EstimatorSpec(
        "sklearn.preprocessing", "OrdinalEncoder", "scikit-learn"
    ),
    "LabelEncoder": EstimatorSpec(
        "sklearn.preprocessing", "LabelEncoder", "scikit-learn"
    ),
    "LabelBinarizer": EstimatorSpec(
        "sklearn.preprocessing", "LabelBinarizer", "scikit-learn"
    ),
    "TargetEncoder": EstimatorSpec(
        "sklearn.preprocessing", "TargetEncoder", "scikit-learn"
    ),
    "HDBSCAN": EstimatorSpec("sklearn.cluster", "HDBSCAN", "scikit-learn"),
}

PROVIDER = Provider(get_backend("cpu"), CATALOG)
