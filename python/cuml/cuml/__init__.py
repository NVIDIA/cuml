#
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# If libcuml was installed as a wheel, we must request it to load the library symbols.
# Otherwise, we assume that the library was installed in a system path that ld can find.
try:
    import libcuml
except ModuleNotFoundError:
    pass
else:
    libcuml.load_library()
    del libcuml

import cuml.accel
import cuml.feature_extraction
from cuml._version import __git_commit__, __version__
from cuml.cluster.agglomerative import AgglomerativeClustering
from cuml.cluster.dbscan import DBSCAN
from cuml.cluster.hdbscan import HDBSCAN
from cuml.cluster.kmeans import KMeans
from cuml.covariance.empirical_covariance import EmpiricalCovariance
from cuml.covariance.ledoit_wolf import LedoitWolf
from cuml.datasets.arima import make_arima
from cuml.datasets.blobs import make_blobs
from cuml.datasets.classification import make_classification
from cuml.datasets.regression import make_regression
from cuml.decomposition.incremental_pca import IncrementalPCA
from cuml.decomposition.pca import PCA
from cuml.decomposition.tsvd import TruncatedSVD
from cuml.ensemble.isolation_forest import IsolationForest
from cuml.ensemble.randomforestclassifier import RandomForestClassifier
from cuml.ensemble.randomforestregressor import RandomForestRegressor
from cuml.explainer.kernel_shap import KernelExplainer
from cuml.explainer.permutation_shap import PermutationExplainer
from cuml.explainer.tree_shap import TreeExplainer
from cuml.internals.base import Base
from cuml.internals.global_settings import (
    GlobalSettings,
    _global_settings_data,
)
from cuml.internals.outputs import set_global_output_type, using_output_type
from cuml.kernel_ridge.kernel_ridge import KernelRidge
from cuml.linear_model.elastic_net import ElasticNet
from cuml.linear_model.lars import Lars
from cuml.linear_model.lasso import Lasso
from cuml.linear_model.linear_regression import LinearRegression
from cuml.linear_model.logistic_regression import LogisticRegression
from cuml.linear_model.mbsgd_classifier import MBSGDClassifier
from cuml.linear_model.mbsgd_regressor import MBSGDRegressor
from cuml.linear_model.ridge import Ridge
from cuml.manifold.t_sne import TSNE
from cuml.manifold.umap import UMAP
from cuml.metrics import accuracy_score, adjusted_rand_score, r2_score
from cuml.model_selection import train_test_split
from cuml.naive_bayes.naive_bayes import MultinomialNB
from cuml.neighbors.kernel_density import KernelDensity
from cuml.neighbors.kneighbors_classifier import KNeighborsClassifier
from cuml.neighbors.kneighbors_regressor import KNeighborsRegressor
from cuml.neighbors.nearest_neighbors import NearestNeighbors
from cuml.preprocessing._label import LabelEncoder
from cuml.random_projection.random_projection import (
    GaussianRandomProjection,
    SparseRandomProjection,
    johnson_lindenstrauss_min_dim,
)
from cuml.solvers.cd import CD
from cuml.solvers.qn import QN
from cuml.solvers.sgd import SGD
from cuml.svm import SVC, SVR, LinearSVC, LinearSVR
from cuml.tsa import stationarity
from cuml.tsa.arima import ARIMA
from cuml.tsa.auto_arima import AutoARIMA
from cuml.tsa.holtwinters import ExponentialSmoothing


def _setup_cupy():
    """One-time setup calls for cupy interop"""
    import copyreg

    import cupy as cp
    from rmm.allocators.cupy import rmm_cupy_allocator

    # Enable rmm_cupy_allocator
    cp.cuda.set_allocator(rmm_cupy_allocator)

    # TODO: this is a workaround for https://github.com/cupy/cupy/issues/10084
    # It can be conditionally done once the cupy fix is out (see
    # https://github.com/rapidsai/cuml/issues/8364).
    copyreg.dispatch_table[cp.ndarray] = lambda x: (
        cp.array,
        (x.get(order="A"),),
    )


_setup_cupy()
del _setup_cupy


def __getattr__(name):
    if name == "global_settings":
        try:
            return _global_settings_data.settings
        except AttributeError:
            _global_settings_data.settings = GlobalSettings()
            return _global_settings_data.settings

    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = [
    # Modules
    "common",
    "feature_extraction",
    "metrics",
    "multiclass",
    "naive_bayes",
    "preprocessing",
    "explainer",
    # Classes
    "AgglomerativeClustering",
    "ARIMA",
    "AutoARIMA",
    "Base",
    "CD",
    "DBSCAN",
    "ElasticNet",
    "EmpiricalCovariance",
    "ExponentialSmoothing",
    "GaussianRandomProjection",
    "Handle",
    "HDBSCAN",
    "IncrementalPCA",
    "IsolationForest",
    "KernelDensity",
    "KernelExplainer",
    "KernelRidge",
    "KMeans",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    "Lars",
    "Lasso",
    "LedoitWolf",
    "LinearRegression",
    "LinearSVC",
    "LinearSVR",
    "LogisticRegression",
    "MBSGDClassifier",
    "MBSGDRegressor",
    "NearestNeighbors",
    "PCA",
    "PermutationExplainer",
    "QN",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "Ridge",
    "SGD",
    "SparseRandomProjection",
    "SVC",
    "SVR",
    "TruncatedSVD",
    "TreeExplainer",
    "TSNE",
    "UMAP",
    # Functions
    "johnson_lindenstrauss_min_dim",
    "make_arima",
    "make_blobs",
    "make_classification",
    "make_regression",
    "stationarity",
]
