# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Define the provider for distributed cuML estimators."""

from ..backends import get_backend
from .base import EstimatorSpec, Provider

CATALOG = {
    "KMeans": EstimatorSpec("cuml.dask.cluster", "KMeans", "cuml"),
    "DBSCAN": EstimatorSpec("cuml.dask.cluster", "DBSCAN", "cuml"),
    "PCA": EstimatorSpec("cuml.dask.decomposition", "PCA", "cuml"),
    "TruncatedSVD": EstimatorSpec(
        "cuml.dask.decomposition", "TruncatedSVD", "cuml"
    ),
    "NearestNeighbors": EstimatorSpec(
        "cuml.dask.neighbors", "NearestNeighbors", "cuml"
    ),
    "KNeighborsClassifier": EstimatorSpec(
        "cuml.dask.neighbors", "KNeighborsClassifier", "cuml"
    ),
    "KNeighborsRegressor": EstimatorSpec(
        "cuml.dask.neighbors", "KNeighborsRegressor", "cuml"
    ),
    "LinearRegression": EstimatorSpec(
        "cuml.dask.linear_model", "LinearRegression", "cuml"
    ),
    "LogisticRegression": EstimatorSpec(
        "cuml.dask.linear_model", "LogisticRegression", "cuml"
    ),
    "ElasticNet": EstimatorSpec(
        "cuml.dask.linear_model", "ElasticNet", "cuml"
    ),
    "Lasso": EstimatorSpec("cuml.dask.linear_model", "Lasso", "cuml"),
    "Ridge": EstimatorSpec("cuml.dask.linear_model", "Ridge", "cuml"),
    "RandomForestClassifier": EstimatorSpec(
        "cuml.dask.ensemble", "RandomForestClassifier", "cuml"
    ),
    "RandomForestRegressor": EstimatorSpec(
        "cuml.dask.ensemble", "RandomForestRegressor", "cuml"
    ),
    "MultinomialNB": EstimatorSpec(
        "cuml.dask.naive_bayes", "MultinomialNB", "cuml"
    ),
    "OneHotEncoder": EstimatorSpec(
        "cuml.dask.preprocessing", "OneHotEncoder", "cuml"
    ),
    "OrdinalEncoder": EstimatorSpec(
        "cuml.dask.preprocessing", "OrdinalEncoder", "cuml"
    ),
    "LabelEncoder": EstimatorSpec(
        "cuml.dask.preprocessing", "LabelEncoder", "cuml"
    ),
    "LabelBinarizer": EstimatorSpec(
        "cuml.dask.preprocessing", "LabelBinarizer", "cuml"
    ),
}

PROVIDER = Provider(get_backend("cuml.dask"), CATALOG)
