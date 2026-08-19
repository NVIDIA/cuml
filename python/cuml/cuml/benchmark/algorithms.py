#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import functools
import shutil
import sys
import tempfile
import warnings
from importlib import import_module

import numpy as np
import sklearn
import sklearn.cluster
import sklearn.covariance
import sklearn.decomposition
import sklearn.ensemble
import sklearn.impute
import sklearn.kernel_ridge
import sklearn.linear_model
import sklearn.manifold
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.preprocessing
import sklearn.random_projection
import sklearn.svm
from sklearn import metrics
from sklearn.impute import SimpleImputer as skSimpleImputer

# Supports both package and standalone execution
try:
    from cuml.benchmark.bench_helper_funcs import (
        _training_data_to_numpy,
        fit,
        fit_kneighbors,
        fit_predict,
        fit_score_samples,
        fit_transform,
        transform,
    )
    from cuml.benchmark.gpu_check import is_cuml_available
except ImportError:
    if not any("cuml/benchmark" in p for p in sys.path):
        raise
    from bench_helper_funcs import (  # noqa: E402
        _training_data_to_numpy,
        fit,
        fit_kneighbors,
        fit_predict,
        fit_score_samples,
        fit_transform,
        transform,
    )
    from gpu_check import is_cuml_available  # noqa: E402

# Conditional GPU imports
cuml = None
cuml_metrics = None
treelite = None


# GPU-specific helper functions (only available when GPU libs present)
_build_cpu_skl_classifier = None
_build_mnmg_umap = None

# cuML preprocessing classes (fallback to sklearn if not available)
Binarizer = sklearn.preprocessing.Binarizer
KBinsDiscretizer = sklearn.preprocessing.KBinsDiscretizer
KernelCenterer = sklearn.preprocessing.KernelCenterer
LabelBinarizer = sklearn.preprocessing.LabelBinarizer
LabelEncoder = sklearn.preprocessing.LabelEncoder
MaxAbsScaler = sklearn.preprocessing.MaxAbsScaler
MinMaxScaler = sklearn.preprocessing.MinMaxScaler
MissingIndicator = sklearn.impute.MissingIndicator
Normalizer = sklearn.preprocessing.Normalizer
OneHotEncoder = sklearn.preprocessing.OneHotEncoder
OrdinalEncoder = sklearn.preprocessing.OrdinalEncoder
PolynomialFeatures = sklearn.preprocessing.PolynomialFeatures
PowerTransformer = sklearn.preprocessing.PowerTransformer
QuantileTransformer = sklearn.preprocessing.QuantileTransformer
RobustScaler = sklearn.preprocessing.RobustScaler
SimpleImputer = skSimpleImputer
StandardScaler = sklearn.preprocessing.StandardScaler
TargetEncoder = sklearn.preprocessing.TargetEncoder

if is_cuml_available():
    import cuml as _cuml
    import cuml.covariance
    import cuml.decomposition
    import cuml.kernel_ridge
    import cuml.metrics as _cuml_metrics
    import cuml.naive_bayes
    from cuml.preprocessing import (
        Binarizer,
        KBinsDiscretizer,
        KernelCenterer,
        LabelBinarizer,
        LabelEncoder,
        MaxAbsScaler,
        MinMaxScaler,
        MissingIndicator,
        Normalizer,
        OneHotEncoder,
        OrdinalEncoder,
        PolynomialFeatures,
        PowerTransformer,
        QuantileTransformer,
        RobustScaler,
        SimpleImputer,
        StandardScaler,
        TargetEncoder,
    )

    cuml = _cuml
    cuml_metrics = _cuml_metrics

    # Import GPU-specific helper functions (support package + standalone execution)
    try:
        from cuml.benchmark.bench_helper_funcs import _build_mnmg_umap
    except ImportError:
        from bench_helper_funcs import _build_mnmg_umap  # noqa: E402

# Optional treelite import
try:
    import treelite as _treelite

    treelite = _treelite
except ImportError:
    treelite = None

# Optional umap-learn import
try:
    from umap import UMAP
except ImportError:
    UMAP = None

# Optional hdbscan import
try:
    from hdbscan import HDBSCAN
except ImportError:
    HDBSCAN = None


try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = None
    XGBRegressor = None


class AlgorithmPair:
    """
    Wraps a cuML algorithm and (optionally) a cpu-based algorithm
    (typically scikit-learn, but does not need to be as long as it offers
    `fit` and `predict` or `transform` methods).
    Provides mechanisms to run each version with default arguments.
    If no CPU-based version of the algorithm is available, pass None for the
    cpu_class when instantiating

    Parameters
    ----------
    cpu_class : class
       Class for CPU version of algorithm. Set to None if not available.
    cuml_class : class
       Class for cuML algorithm. Can be None for CPU-only algorithms.
    shared_args : dict
       Arguments passed to both implementations's initializer
    cuml_args : dict
       Arguments *only* passed to cuml's initializer
    cpu_args dict
       Arguments *only* passed to sklearn's initializer
    accepts_labels : boolean
       If True, the fit methods expects both X and y
       inputs. Otherwise, it expects only an X input.
    data_prep_hook : function (data -> data)
       Optional function to run on input data before passing to fit
    accuracy_function : function (y_test, y_pred)
       Function that returns a scalar representing accuracy
    bench_func : custom function to perform fit/predict/transform
                 calls.
    """

    def __init__(
        self,
        cpu_class,
        cuml_class,
        shared_args,
        cuml_args={},
        cpu_args={},
        name=None,
        accepts_labels=True,
        cpu_data_prep_hook=None,
        cuml_data_prep_hook=None,
        accuracy_function=None,
        bench_func=fit,
        setup_cpu_func=None,
        setup_cuml_func=None,
    ):
        if name:
            self.name = name
        elif cuml_class is not None:
            self.name = cuml_class.__name__
        elif cpu_class is not None:
            self.name = cpu_class.__name__
        else:
            self.name = "Unknown"
        self.accepts_labels = accepts_labels
        self.bench_func = bench_func
        self.setup_cpu_func = setup_cpu_func
        self.setup_cuml_func = setup_cuml_func
        self.cpu_class = cpu_class
        self.cuml_class = cuml_class
        self.shared_args = shared_args
        self.cuml_args = cuml_args
        self.cpu_args = cpu_args
        self.cpu_data_prep_hook = cpu_data_prep_hook
        self.cuml_data_prep_hook = cuml_data_prep_hook
        self.accuracy_function = accuracy_function
        self._tmpdir = None

    @property
    def tmpdir(self):
        if self._tmpdir is None:
            self._tmpdir = tempfile.mkdtemp()
        return self._tmpdir

    def cleanup(self):
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None

    def __del__(self):
        self.cleanup()

    def __str__(self):
        return "AlgoPair:%s" % (self.name)

    def run_cpu(self, data, bench_args={}, **override_setup_args):
        """Runs the cpu-based algorithm's fit method on specified data"""
        if self.cpu_class is None:
            raise ValueError("No CPU implementation for %s" % self.name)

        all_args = {**self.shared_args, **self.cpu_args}
        all_args = {**all_args, **override_setup_args}

        if "cpu_setup_result" not in all_args:
            cpu_obj = self.cpu_class(**all_args)
        else:
            cpu_obj = all_args["cpu_setup_result"]
        if self.cpu_data_prep_hook:
            data = self.cpu_data_prep_hook(data)
        if self.accepts_labels:
            self.bench_func(cpu_obj, data[0], data[1], **bench_args)
        else:
            self.bench_func(cpu_obj, data[0], **bench_args)

        return cpu_obj

    def run_cuml(self, data, bench_args={}, **override_setup_args):
        """Runs the cuml-based algorithm's fit method on specified data"""
        if self.cuml_class is None:
            raise ValueError("No cuML implementation for %s" % self.name)

        all_args = {**self.shared_args, **self.cuml_args}
        all_args = {**all_args, **override_setup_args}

        if "cuml_setup_result" not in all_args:
            cuml_obj = self.cuml_class(**all_args)
        else:
            cuml_obj = all_args["cuml_setup_result"]
        if self.cuml_data_prep_hook:
            data = self.cuml_data_prep_hook(data)
        if self.accepts_labels:
            self.bench_func(cuml_obj, data[0], data[1], **bench_args)
        else:
            self.bench_func(cuml_obj, data[0], **bench_args)

        return cuml_obj

    def setup_cpu(self, data, **override_args):
        all_args = {**self.shared_args, **self.cpu_args}
        all_args = {**all_args, **override_args}
        if self.setup_cpu_func is not None:
            return {
                "cpu_setup_result": self.setup_cpu_func(
                    self.cpu_class, data, all_args, self.tmpdir
                )
            }
        else:
            return all_args

    def setup_cuml(self, data, **override_args):
        all_args = {**self.shared_args, **self.cuml_args}
        all_args = {**all_args, **override_args}
        if self.setup_cuml_func is not None:
            return {
                "cuml_setup_result": self.setup_cuml_func(
                    self.cuml_class, data, all_args, self.tmpdir
                )
            }
        else:
            return all_args

    def has_cpu(self):
        """Check if this algorithm has a CPU implementation."""
        return self.cpu_class is not None

    def has_cuml(self):
        """Check if this algorithm has a cuML implementation."""
        return self.cuml_class is not None


def _labels_to_int_hook(data):
    """Helper function converting labels to int32"""
    return data[0], data[1].astype(np.int32)


def _treelite_format_hook(data):
    """Helper function converting data into treelite format"""
    data = _training_data_to_numpy(data[0], data[1])
    return data[0], data[1]


def _numpy_format_hook(data):
    """Helper function converting data into numpy array"""
    return _training_data_to_numpy(data[0], data[1])


def _labels_as_features_hook(data):
    """Helper function using labels as input for target-only transformers."""
    if len(data) >= 4:
        return data[1], None, data[3], None
    return data[1], None


def _as_array_for_matmul(X):
    """Convert dataframe-like inputs to an array while preserving device type."""
    if hasattr(X, "to_cupy"):
        return X.to_cupy()
    if hasattr(X, "to_numpy"):
        return X.to_numpy()
    return X


def _linear_kernel(X, Y=None):
    """Compute a linear kernel matrix for KernelCenterer benchmarks."""
    X_arr = _as_array_for_matmul(X)
    Y_arr = X_arr if Y is None else _as_array_for_matmul(Y)
    return X_arr @ Y_arr.T


def _linear_kernel_hook(data):
    """Helper function converting feature data into a square kernel matrix."""
    K_train = _linear_kernel(data[0])
    if len(data) >= 4 and data[2] is not None:
        K_test = _linear_kernel(data[2], data[0])
        return K_train, None, K_test, None
    return K_train, None


@functools.cache
def all_algorithms():
    """Returns all defined AlgorithmPair objects.

    Each AlgorithmPair contains both CPU (sklearn) and GPU (cuML) implementations.
    When cuML is not installed, cuml_class will be None but cpu_class will still
    be available for CPU-only benchmarking.
    """
    # Get cuML classes and metrics if available
    if is_cuml_available():
        cuml_KMeans = cuml.cluster.KMeans
        cuml_AgglomerativeClustering = cuml.cluster.AgglomerativeClustering
        cuml_SpectralClustering = cuml.cluster.SpectralClustering
        cuml_EmpiricalCovariance = cuml.covariance.EmpiricalCovariance
        cuml_LedoitWolf = cuml.covariance.LedoitWolf
        cuml_PCA = cuml.PCA
        cuml_IncrementalPCA = cuml.decomposition.IncrementalPCA
        cuml_TruncatedSVD = cuml.decomposition.tsvd.TruncatedSVD
        cuml_GaussianRandomProjection = (
            cuml.random_projection.GaussianRandomProjection
        )
        cuml_SparseRandomProjection = (
            cuml.random_projection.SparseRandomProjection
        )
        cuml_NearestNeighbors = cuml.neighbors.NearestNeighbors
        cuml_KernelDensity = cuml.neighbors.KernelDensity
        cuml_DBSCAN = cuml.DBSCAN
        cuml_HDBSCAN = cuml.cluster.HDBSCAN
        cuml_LinearRegression = cuml.linear_model.LinearRegression
        cuml_ElasticNet = cuml.linear_model.ElasticNet
        cuml_Lasso = cuml.linear_model.Lasso
        cuml_Ridge = cuml.linear_model.Ridge
        cuml_KernelRidge = cuml.kernel_ridge.KernelRidge
        cuml_LogisticRegression = cuml.linear_model.LogisticRegression
        cuml_RandomForestClassifier = cuml.ensemble.RandomForestClassifier
        cuml_RandomForestRegressor = cuml.ensemble.RandomForestRegressor
        cuml_TSNE = cuml.manifold.TSNE
        cuml_SVC = cuml.svm.SVC
        cuml_SVR = cuml.svm.SVR
        cuml_LinearSVC = cuml.svm.LinearSVC
        cuml_LinearSVR = cuml.svm.LinearSVR
        cuml_KNeighborsClassifier = cuml.neighbors.KNeighborsClassifier
        cuml_KNeighborsRegressor = cuml.neighbors.KNeighborsRegressor
        cuml_GaussianNB = cuml.naive_bayes.GaussianNB
        cuml_MultinomialNB = cuml.naive_bayes.MultinomialNB
        cuml_BernoulliNB = cuml.naive_bayes.BernoulliNB
        cuml_ComplementNB = cuml.naive_bayes.ComplementNB
        cuml_CategoricalNB = cuml.naive_bayes.CategoricalNB
        cuml_TargetEncoder = TargetEncoder
        cuml_OneHotEncoder = OneHotEncoder
        cuml_OrdinalEncoder = OrdinalEncoder
        cuml_MissingIndicator = MissingIndicator
        cuml_KernelCenterer = KernelCenterer
        cuml_UMAP = cuml.manifold.UMAP
        accuracy_fn = cuml_metrics.accuracy_score
        r2_fn = cuml_metrics.r2_score
        trustworthiness_fn = cuml_metrics.trustworthiness
    else:
        cuml_EmpiricalCovariance = None
        cuml_KMeans = cuml_PCA = cuml_TruncatedSVD = None
        cuml_AgglomerativeClustering = cuml_SpectralClustering = None
        cuml_LedoitWolf = cuml_IncrementalPCA = None
        cuml_GaussianRandomProjection = cuml_SparseRandomProjection = None
        cuml_NearestNeighbors = cuml_DBSCAN = cuml_HDBSCAN = None
        cuml_LinearRegression = cuml_ElasticNet = cuml_Lasso = cuml_Ridge = (
            None
        )
        cuml_KernelRidge = None
        cuml_LogisticRegression = None
        cuml_RandomForestClassifier = cuml_RandomForestRegressor = None
        cuml_TSNE = cuml_SVC = cuml_SVR = cuml_LinearSVC = cuml_LinearSVR = (
            None
        )
        cuml_KNeighborsClassifier = cuml_KNeighborsRegressor = None
        cuml_KernelDensity = None
        cuml_GaussianNB = cuml_MultinomialNB = None
        cuml_BernoulliNB = cuml_ComplementNB = cuml_CategoricalNB = None
        cuml_TargetEncoder = cuml_OneHotEncoder = cuml_OrdinalEncoder = None
        cuml_MissingIndicator = cuml_KernelCenterer = cuml_UMAP = None
        accuracy_fn = metrics.accuracy_score
        r2_fn = metrics.r2_score
        trustworthiness_fn = None

    algorithms = [
        AlgorithmPair(
            sklearn.cluster.KMeans,
            cuml_KMeans,
            shared_args=dict(
                init="k-means++", n_clusters=8, max_iter=300, n_init=1
            ),
            cuml_args=dict(oversampling_factor=0)
            if is_cuml_available()
            else {},
            name="KMeans",
            accepts_labels=False,
            accuracy_function=metrics.homogeneity_score,
        ),
        AlgorithmPair(
            sklearn.cluster.AgglomerativeClustering,
            cuml_AgglomerativeClustering,
            shared_args=dict(
                n_clusters=8, metric="euclidean", linkage="single"
            ),
            name="AgglomerativeClustering",
            accepts_labels=False,
        ),
        AlgorithmPair(
            sklearn.cluster.SpectralClustering,
            cuml_SpectralClustering,
            shared_args=dict(
                n_clusters=8,
                affinity="nearest_neighbors",
                n_neighbors=10,
                n_init=1,
                random_state=42,
            ),
            name="SpectralClustering",
            accepts_labels=False,
        ),
        AlgorithmPair(
            sklearn.covariance.EmpiricalCovariance,
            cuml_EmpiricalCovariance,
            shared_args=dict(),
            name="EmpiricalCovariance",
            accepts_labels=False,
        ),
        AlgorithmPair(
            sklearn.covariance.LedoitWolf,
            cuml_LedoitWolf,
            shared_args=dict(),
            name="LedoitWolf",
            accepts_labels=False,
        ),
        AlgorithmPair(
            sklearn.decomposition.PCA,
            cuml_PCA,
            shared_args=dict(n_components=10),
            name="PCA",
            accepts_labels=False,
        ),
        AlgorithmPair(
            sklearn.decomposition.IncrementalPCA,
            cuml_IncrementalPCA,
            shared_args=dict(n_components=10),
            name="IncrementalPCA",
            accepts_labels=False,
        ),
        AlgorithmPair(
            sklearn.decomposition.TruncatedSVD,
            cuml_TruncatedSVD,
            shared_args=dict(n_components=10),
            name="tSVD",
            accepts_labels=False,
        ),
        AlgorithmPair(
            sklearn.random_projection.GaussianRandomProjection,
            cuml_GaussianRandomProjection,
            shared_args=dict(n_components=10),
            name="GaussianRandomProjection",
            accepts_labels=False,
        ),
        AlgorithmPair(
            sklearn.random_projection.SparseRandomProjection,
            cuml_SparseRandomProjection,
            shared_args=dict(n_components=10),
            name="SparseRandomProjection",
            accepts_labels=False,
        ),
        AlgorithmPair(
            sklearn.neighbors.NearestNeighbors,
            cuml_NearestNeighbors,
            shared_args=dict(n_neighbors=64),
            cpu_args=dict(algorithm="brute", n_jobs=-1),
            cuml_args={},
            name="NearestNeighbors",
            accepts_labels=False,
            bench_func=fit_kneighbors,
        ),
        AlgorithmPair(
            sklearn.neighbors.KernelDensity,
            cuml_KernelDensity,
            shared_args=dict(kernel="gaussian", bandwidth=1.0),
            name="KernelDensity",
            accepts_labels=False,
            bench_func=fit_score_samples,
        ),
        AlgorithmPair(
            sklearn.cluster.DBSCAN,
            cuml_DBSCAN,
            shared_args=dict(eps=3, min_samples=2),
            cpu_args=dict(algorithm="brute"),
            name="DBSCAN",
            accepts_labels=False,
        ),
        AlgorithmPair(
            sklearn.linear_model.LinearRegression,
            cuml_LinearRegression,
            shared_args={},
            name="LinearRegression",
            accepts_labels=True,
            accuracy_function=metrics.r2_score,
        ),
        AlgorithmPair(
            sklearn.linear_model.ElasticNet,
            cuml_ElasticNet,
            shared_args={"alpha": 0.1, "l1_ratio": 0.5},
            name="ElasticNet",
            accepts_labels=True,
            accuracy_function=metrics.r2_score,
        ),
        AlgorithmPair(
            sklearn.linear_model.Lasso,
            cuml_Lasso,
            shared_args={},
            name="Lasso",
            accepts_labels=True,
            accuracy_function=metrics.r2_score,
        ),
        AlgorithmPair(
            sklearn.linear_model.Ridge,
            cuml_Ridge,
            shared_args={},
            name="Ridge",
            accepts_labels=True,
            accuracy_function=metrics.r2_score,
        ),
        AlgorithmPair(
            sklearn.kernel_ridge.KernelRidge,
            cuml_KernelRidge,
            shared_args=dict(),
            name="KernelRidge",
            accepts_labels=True,
            accuracy_function=r2_fn,
        ),
        AlgorithmPair(
            sklearn.linear_model.LogisticRegression,
            cuml_LogisticRegression,
            shared_args=dict(),  # Use default solvers
            name="LogisticRegression",
            accepts_labels=True,
            accuracy_function=metrics.accuracy_score,
        ),
        AlgorithmPair(
            sklearn.ensemble.RandomForestClassifier,
            cuml_RandomForestClassifier,
            shared_args={},
            cpu_args={"n_jobs": -1},
            name="RandomForestClassifier",
            accepts_labels=True,
            cpu_data_prep_hook=_labels_to_int_hook,
            cuml_data_prep_hook=_labels_to_int_hook
            if is_cuml_available()
            else None,
            accuracy_function=metrics.accuracy_score,
        ),
        AlgorithmPair(
            sklearn.ensemble.RandomForestRegressor,
            cuml_RandomForestRegressor,
            shared_args={},
            cpu_args={"n_jobs": -1},
            name="RandomForestRegressor",
            accepts_labels=True,
            accuracy_function=metrics.r2_score,
        ),
        AlgorithmPair(
            XGBClassifier,
            XGBClassifier,
            shared_args={"tree_method": "hist", "n_estimators": 100},
            cpu_args={"n_jobs": -1},
            cuml_args={"device": "cuda"},
            name="xgboost-classification",
            accepts_labels=True,
            cpu_data_prep_hook=_labels_to_int_hook,
            cuml_data_prep_hook=_labels_to_int_hook
            if is_cuml_available()
            else None,
            accuracy_function=metrics.accuracy_score,
        ),
        AlgorithmPair(
            XGBRegressor,
            XGBRegressor,
            shared_args={"tree_method": "hist", "n_estimators": 100},
            cpu_args={"n_jobs": -1},
            cuml_args={"device": "cuda"},
            name="xgboost-regression",
            accepts_labels=True,
            accuracy_function=metrics.r2_score,
        ),
        AlgorithmPair(
            sklearn.manifold.TSNE,
            cuml_TSNE,
            shared_args=dict(),
            name="TSNE",
            accepts_labels=False,
        ),
        AlgorithmPair(
            sklearn.svm.SVC,
            cuml_SVC,
            shared_args={"kernel": "rbf"},
            cuml_args={},
            name="SVC-RBF",
            accepts_labels=True,
            accuracy_function=accuracy_fn,
        ),
        AlgorithmPair(
            sklearn.svm.SVC,
            cuml_SVC,
            shared_args={"kernel": "linear"},
            cuml_args={},
            name="SVC-Linear",
            accepts_labels=True,
            accuracy_function=accuracy_fn,
        ),
        AlgorithmPair(
            sklearn.svm.SVR,
            cuml_SVR,
            shared_args={"kernel": "rbf"},
            cuml_args={},
            name="SVR-RBF",
            accepts_labels=True,
            accuracy_function=r2_fn,
        ),
        AlgorithmPair(
            sklearn.svm.SVR,
            cuml_SVR,
            shared_args={"kernel": "linear"},
            cuml_args={},
            name="SVR-Linear",
            accepts_labels=True,
            accuracy_function=r2_fn,
        ),
        AlgorithmPair(
            sklearn.svm.LinearSVC,
            cuml_LinearSVC,
            shared_args={},
            cuml_args={},
            name="LinearSVC",
            accepts_labels=True,
            accuracy_function=accuracy_fn,
        ),
        AlgorithmPair(
            sklearn.svm.LinearSVR,
            cuml_LinearSVR,
            shared_args={},
            cuml_args={},
            name="LinearSVR",
            accepts_labels=True,
            accuracy_function=r2_fn,
        ),
        AlgorithmPair(
            sklearn.neighbors.KNeighborsClassifier,
            cuml_KNeighborsClassifier,
            shared_args={},
            cuml_args={},
            name="KNeighborsClassifier",
            accepts_labels=True,
            accuracy_function=accuracy_fn,
            bench_func=fit_predict,
        ),
        AlgorithmPair(
            sklearn.neighbors.KNeighborsRegressor,
            cuml_KNeighborsRegressor,
            shared_args={},
            cuml_args={},
            name="KNeighborsRegressor",
            accepts_labels=True,
            accuracy_function=r2_fn,
            bench_func=fit_predict,
        ),
        AlgorithmPair(
            sklearn.naive_bayes.MultinomialNB,
            cuml_MultinomialNB,
            shared_args={},
            cuml_args={},
            name="MultinomialNB",
            accepts_labels=True,
            accuracy_function=accuracy_fn,
        ),
        AlgorithmPair(
            sklearn.naive_bayes.BernoulliNB,
            cuml_BernoulliNB,
            shared_args={},
            cuml_args={},
            name="BernoulliNB",
            accepts_labels=True,
            accuracy_function=accuracy_fn,
        ),
        AlgorithmPair(
            sklearn.naive_bayes.ComplementNB,
            cuml_ComplementNB,
            shared_args={},
            cuml_args={},
            name="ComplementNB",
            accepts_labels=True,
            accuracy_function=accuracy_fn,
        ),
        AlgorithmPair(
            sklearn.naive_bayes.CategoricalNB,
            cuml_CategoricalNB,
            shared_args={},
            cuml_args={},
            name="CategoricalNB",
            accepts_labels=True,
            accuracy_function=accuracy_fn,
        ),
        AlgorithmPair(
            sklearn.naive_bayes.GaussianNB,
            cuml_GaussianNB,
            shared_args={},
            cuml_args={},
            name="GaussianNB",
            accepts_labels=True,
            accuracy_function=accuracy_fn,
        ),
        # Preprocessing
        AlgorithmPair(
            sklearn.preprocessing.StandardScaler,
            StandardScaler,
            shared_args=dict(),
            name="StandardScaler",
            accepts_labels=False,
            bench_func=fit_transform,
        ),
        AlgorithmPair(
            sklearn.preprocessing.MinMaxScaler,
            MinMaxScaler,
            shared_args=dict(),
            name="MinMaxScaler",
            accepts_labels=False,
            bench_func=fit_transform,
        ),
        AlgorithmPair(
            sklearn.preprocessing.MaxAbsScaler,
            MaxAbsScaler,
            shared_args=dict(),
            name="MaxAbsScaler",
            accepts_labels=False,
            bench_func=fit_transform,
        ),
        AlgorithmPair(
            sklearn.preprocessing.Normalizer,
            Normalizer,
            shared_args=dict(),
            name="Normalizer",
            accepts_labels=False,
            bench_func=fit_transform,
        ),
        AlgorithmPair(
            skSimpleImputer,
            SimpleImputer,
            shared_args=dict(),
            name="SimpleImputer",
            accepts_labels=False,
            bench_func=fit_transform,
        ),
        AlgorithmPair(
            sklearn.preprocessing.RobustScaler,
            RobustScaler,
            shared_args=dict(),
            name="RobustScaler",
            accepts_labels=False,
            bench_func=fit_transform,
        ),
        AlgorithmPair(
            sklearn.preprocessing.PolynomialFeatures,
            PolynomialFeatures,
            shared_args=dict(),
            name="PolynomialFeatures",
            accepts_labels=False,
            bench_func=fit_transform,
        ),
        AlgorithmPair(
            sklearn.preprocessing.Binarizer,
            Binarizer,
            shared_args=dict(),
            name="Binarizer",
            accepts_labels=False,
            bench_func=fit_transform,
        ),
        AlgorithmPair(
            sklearn.preprocessing.KBinsDiscretizer,
            KBinsDiscretizer,
            shared_args=dict(),
            name="KBinsDiscretizer",
            accepts_labels=False,
            bench_func=fit_transform,
        ),
        AlgorithmPair(
            sklearn.preprocessing.PowerTransformer,
            PowerTransformer,
            shared_args=dict(),
            name="PowerTransformer",
            accepts_labels=False,
            bench_func=fit_transform,
        ),
        AlgorithmPair(
            sklearn.preprocessing.QuantileTransformer,
            QuantileTransformer,
            shared_args=dict(),
            name="QuantileTransformer",
            accepts_labels=False,
            bench_func=fit_transform,
        ),
        AlgorithmPair(
            sklearn.preprocessing.TargetEncoder,
            cuml_TargetEncoder,
            shared_args=dict(smooth=0.0),
            cpu_args=dict(cv=4, random_state=42),
            cuml_args=dict(
                n_folds=4,
                seed=42,
                split_method="interleaved",
                multi_feature_mode="independent",
            ),
            name="TargetEncoder",
            accepts_labels=True,
            bench_func=fit_transform,
        ),
        AlgorithmPair(
            sklearn.preprocessing.OneHotEncoder,
            cuml_OneHotEncoder,
            shared_args=dict(sparse_output=False, handle_unknown="ignore"),
            name="OneHotEncoder",
            accepts_labels=False,
            bench_func=fit_transform,
        ),
        AlgorithmPair(
            sklearn.preprocessing.OrdinalEncoder,
            cuml_OrdinalEncoder,
            shared_args=dict(),
            name="OrdinalEncoder",
            accepts_labels=False,
            bench_func=fit_transform,
        ),
        AlgorithmPair(
            sklearn.impute.MissingIndicator,
            cuml_MissingIndicator,
            shared_args=dict(features="all", sparse=False),
            name="MissingIndicator",
            accepts_labels=False,
            bench_func=fit_transform,
        ),
        AlgorithmPair(
            sklearn.preprocessing.KernelCenterer,
            cuml_KernelCenterer,
            shared_args=dict(),
            name="KernelCenterer",
            accepts_labels=False,
            cpu_data_prep_hook=_linear_kernel_hook,
            cuml_data_prep_hook=_linear_kernel_hook,
            bench_func=fit_transform,
        ),
        AlgorithmPair(
            sklearn.preprocessing.LabelEncoder,
            LabelEncoder,
            shared_args=dict(),
            name="LabelEncoder",
            accepts_labels=False,
            cpu_data_prep_hook=_labels_as_features_hook,
            cuml_data_prep_hook=_labels_as_features_hook,
            bench_func=fit_transform,
        ),
        AlgorithmPair(
            sklearn.preprocessing.LabelBinarizer,
            LabelBinarizer,
            shared_args=dict(),
            name="LabelBinarizer",
            accepts_labels=False,
            cpu_data_prep_hook=_labels_as_features_hook,
            cuml_data_prep_hook=_labels_as_features_hook,
            bench_func=fit_transform,
        ),
    ]

    # Add HDBSCAN if available
    if HDBSCAN is not None:
        algorithms.append(
            AlgorithmPair(
                HDBSCAN,
                cuml_HDBSCAN,
                shared_args={},
                cpu_args={"core_dist_n_jobs": -1},
                name="HDBSCAN",
                accepts_labels=False,
            )
        )

    # Add UMAP if available
    if UMAP is not None:
        algorithms.extend(
            [
                AlgorithmPair(
                    UMAP,
                    cuml_UMAP,
                    shared_args=dict(n_neighbors=5, n_epochs=500),
                    name="UMAP-Unsupervised",
                    accepts_labels=False,
                    accuracy_function=trustworthiness_fn,
                ),
                AlgorithmPair(
                    UMAP,
                    cuml_UMAP,
                    shared_args=dict(n_neighbors=5, n_epochs=500),
                    name="UMAP-Supervised",
                    accepts_labels=True,
                    accuracy_function=trustworthiness_fn,
                ),
            ]
        )

    # Add cuML-only MBSGD algorithms with no direct sklearn equivalents.
    if is_cuml_available():
        algorithms.extend(
            [
                AlgorithmPair(
                    None,
                    cuml.linear_model.MBSGDClassifier,
                    shared_args={},
                    cuml_args=dict(eta0=0.005, epochs=100),
                    name="MBSGDClassifier",
                    accepts_labels=True,
                    accuracy_function=cuml_metrics.accuracy_score,
                ),
                AlgorithmPair(
                    None,
                    cuml.linear_model.MBSGDRegressor,
                    shared_args={},
                    cuml_args=dict(eta0=0.005, epochs=100),
                    name="MBSGDRegressor",
                    accepts_labels=True,
                    accuracy_function=cuml_metrics.r2_score,
                ),
            ]
        )

    # Add MNMG (multi-node multi-GPU) algorithms if cuML.dask is available
    if is_cuml_available():
        try:
            import_module("cuml.dask")
        except ImportError:
            warnings.warn(
                "Not all dependencies required for `cuml.dask` are installed, the "
                "dask algorithms will be skipped"
            )
        else:
            algorithms.extend(
                [
                    AlgorithmPair(
                        None,
                        cuml.dask.neighbors.KNeighborsClassifier,
                        shared_args={},
                        cuml_args={},
                        name="MNMG.KNeighborsClassifier",
                        bench_func=fit_predict,
                        accepts_labels=True,
                        accuracy_function=cuml_metrics.accuracy_score,
                    ),
                    AlgorithmPair(
                        None,
                        cuml.dask.cluster.KMeans,
                        shared_args=dict(n_clusters=8, max_iter=300, n_init=1),
                        cpu_args=dict(init="k-means++"),
                        cuml_args=dict(init="scalable-k-means++"),
                        name="MNMG.KMeans",
                        bench_func=fit_predict,
                        accepts_labels=False,
                        accuracy_function=metrics.homogeneity_score,
                    ),
                    AlgorithmPair(
                        None,
                        cuml.dask.cluster.DBSCAN,
                        shared_args=dict(eps=3, min_samples=2),
                        cpu_args=dict(algorithm="brute"),
                        name="MNMG.DBSCAN",
                        bench_func=fit_predict,
                        accepts_labels=False,
                    ),
                    AlgorithmPair(
                        None,
                        cuml.dask.manifold.UMAP,
                        shared_args=dict(n_neighbors=5, n_epochs=500),
                        name="MNMG.UMAP-Unsupervised",
                        bench_func=transform,
                        setup_cuml_func=_build_mnmg_umap,
                        accepts_labels=False,
                        accuracy_function=cuml_metrics.trustworthiness,
                    ),
                    AlgorithmPair(
                        None,
                        cuml.dask.manifold.UMAP,
                        shared_args=dict(n_neighbors=5, n_epochs=500),
                        name="MNMG.UMAP-Supervised",
                        bench_func=transform,
                        setup_cuml_func=_build_mnmg_umap,
                        accepts_labels=True,
                        accuracy_function=cuml_metrics.trustworthiness,
                    ),
                    AlgorithmPair(
                        None,
                        cuml.dask.neighbors.NearestNeighbors,
                        shared_args=dict(n_neighbors=64),
                        cpu_args=dict(algorithm="brute", n_jobs=-1),
                        cuml_args={},
                        name="MNMG.NearestNeighbors",
                        accepts_labels=False,
                        bench_func=fit_kneighbors,
                    ),
                    AlgorithmPair(
                        None,
                        cuml.dask.decomposition.TruncatedSVD,
                        shared_args=dict(n_components=10),
                        name="MNMG.tSVD",
                        accepts_labels=False,
                    ),
                    AlgorithmPair(
                        None,
                        cuml.dask.decomposition.PCA,
                        shared_args=dict(n_components=10),
                        name="MNMG.PCA",
                        accepts_labels=False,
                    ),
                    AlgorithmPair(
                        None,
                        cuml.dask.linear_model.LinearRegression,
                        shared_args={},
                        name="MNMG.LinearRegression",
                        bench_func=fit_predict,
                        accepts_labels=True,
                        accuracy_function=metrics.r2_score,
                    ),
                    AlgorithmPair(
                        None,
                        cuml.dask.linear_model.Lasso,
                        shared_args={},
                        name="MNMG.Lasso",
                        bench_func=fit_predict,
                        accepts_labels=True,
                        accuracy_function=metrics.r2_score,
                    ),
                    AlgorithmPair(
                        None,
                        cuml.dask.linear_model.ElasticNet,
                        shared_args={"alpha": 0.1, "l1_ratio": 0.5},
                        name="MNMG.ElasticNet",
                        bench_func=fit_predict,
                        accepts_labels=True,
                        accuracy_function=metrics.r2_score,
                    ),
                    AlgorithmPair(
                        None,
                        cuml.dask.linear_model.Ridge,
                        shared_args={},
                        name="MNMG.Ridge",
                        bench_func=fit_predict,
                        accepts_labels=True,
                        accuracy_function=metrics.r2_score,
                    ),
                    AlgorithmPair(
                        None,
                        cuml.dask.neighbors.KNeighborsRegressor,
                        shared_args={},
                        cuml_args={},
                        name="MNMG.KNeighborsRegressor",
                        bench_func=fit_predict,
                        accepts_labels=True,
                        accuracy_function=cuml_metrics.r2_score,
                    ),
                ]
            )

    return algorithms


def algorithm_by_name(name):
    """Returns the algorithm pair with the name 'name' (case-insensitive)"""
    algos = all_algorithms()
    return next((a for a in algos if a.name.lower() == name.lower()), None)
