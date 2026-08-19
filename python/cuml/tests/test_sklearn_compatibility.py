#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import pytest
from sklearn.utils import estimator_checks

from cuml.cluster import (
    DBSCAN,
    HDBSCAN,
    AgglomerativeClustering,
    KMeans,
    SpectralClustering,
)
from cuml.compose import ColumnTransformer
from cuml.covariance import EmpiricalCovariance, LedoitWolf
from cuml.decomposition import PCA, IncrementalPCA, TruncatedSVD
from cuml.ensemble import (
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from cuml.feature_extraction.text import TfidfTransformer
from cuml.kernel_ridge import KernelRidge
from cuml.linear_model import (
    ElasticNet,
    Lars,
    Lasso,
    LinearRegression,
    LogisticRegression,
    MBSGDClassifier,
    MBSGDRegressor,
    Ridge,
)
from cuml.manifold import TSNE, UMAP, SpectralEmbedding
from cuml.multiclass import OneVsOneClassifier, OneVsRestClassifier
from cuml.naive_bayes import (
    BernoulliNB,
    CategoricalNB,
    ComplementNB,
    GaussianNB,
    MultinomialNB,
)
from cuml.neighbors import (
    KernelDensity,
    KNeighborsClassifier,
    KNeighborsRegressor,
    NearestNeighbors,
)
from cuml.preprocessing import (
    Binarizer,
    FunctionTransformer,
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
from cuml.random_projection import (
    GaussianRandomProjection,
    SparseRandomProjection,
)
from cuml.svm import SVC, SVR, LinearSVC, LinearSVR
from cuml.testing.utils import get_all_base_subclasses

# Skip these tests as parameterize_with_checks does not support
# strict_xfail in older versions of scikit-learn.
pytest.importorskip("sklearn", minversion="1.8")


ESTIMATORS = [
    GaussianRandomProjection(n_components=2),
    SparseRandomProjection(n_components=2),
    DBSCAN(),
    HDBSCAN(),
    AgglomerativeClustering(),
    KernelRidge(),
    GaussianNB(),
    ComplementNB(),
    CategoricalNB(),
    BernoulliNB(),
    MultinomialNB(),
    UMAP(n_neighbors=5),
    TSNE(),
    TruncatedSVD(),
    IncrementalPCA(),
    PCA(),
    SVR(),
    SVC(),
    LinearSVR(),
    LinearSVC(),
    NearestNeighbors(),
    KNeighborsRegressor(),
    KNeighborsClassifier(),
    KernelDensity(),
    EmpiricalCovariance(),
    LedoitWolf(),
    Lars(),
    Ridge(),
    ElasticNet(),
    Lasso(),
    LinearRegression(),
    IsolationForest(),
    RandomForestClassifier(),
    RandomForestRegressor(),
    KMeans(),
    SpectralClustering(),
    LogisticRegression(),
    StandardScaler(),
]


_MODULE_TO_IGNORE = {
    "dask",
    "accel",
    "solvers",
    "tsa",
    "explainer",
    "fil",
    "experimental",
    "benchmark",
    "tests",
}


def _all_cuml_estimators():
    """Discover all public cuml estimator classes (subclasses of Base)."""
    return {
        cls
        for cls in get_all_base_subclasses().values()
        if not (cls.__name__.endswith("MG") or "Base" in cls.__name__)
        and not getattr(cls, "__abstractmethods__", None)
        and not any(
            part in _MODULE_TO_IGNORE for part in cls.__module__.split(".")
        )
    }


EXCLUDED = {
    # Linear model
    MBSGDClassifier: "Not yet tested for sklearn compat",
    MBSGDRegressor: "Not yet tested for sklearn compat",
    # Meta-estimators
    OneVsRestClassifier: "Meta-estimator, requires an inner estimator",
    OneVsOneClassifier: "Meta-estimator, requires an inner estimator",
    # Manifold
    SpectralEmbedding: "Not yet tested for sklearn compat",
    # Feature extraction
    TfidfTransformer: "Not yet tested for sklearn compat",
    # Preprocessing (cuml-native)
    LabelEncoder: "Not yet tested for sklearn compat",
    TargetEncoder: "Not yet tested for sklearn compat",
    LabelBinarizer: "Not yet tested for sklearn compat",
    OneHotEncoder: "Not yet tested for sklearn compat",
    OrdinalEncoder: "Not yet tested for sklearn compat",
    # Preprocessing (vendored sklearn)
    MinMaxScaler: "Vendored sklearn preprocessing, not yet tested",
    MaxAbsScaler: "Vendored sklearn preprocessing, not yet tested",
    RobustScaler: "Vendored sklearn preprocessing, not yet tested",
    Normalizer: "Vendored sklearn preprocessing, not yet tested",
    Binarizer: "Vendored sklearn preprocessing, not yet tested",
    KernelCenterer: "Vendored sklearn preprocessing, not yet tested",
    PolynomialFeatures: "Vendored sklearn preprocessing, not yet tested",
    PowerTransformer: "Vendored sklearn preprocessing, not yet tested",
    QuantileTransformer: "Vendored sklearn preprocessing, not yet tested",
    KBinsDiscretizer: "Vendored sklearn preprocessing, not yet tested",
    SimpleImputer: "Vendored sklearn preprocessing, not yet tested",
    MissingIndicator: "Vendored sklearn preprocessing, not yet tested",
    FunctionTransformer: "Vendored sklearn preprocessing, not yet tested",
    # Compose
    ColumnTransformer: "Vendored __init__ defaults transformers=None, breaking set_params/get_params",
}


XFAILS = {
    IsolationForest: {
        "check_estimators_unfitted": (
            "Unfitted methods raise RuntimeError instead of NotFittedError"
        ),
        "check_estimators_pickle": (
            "Pickling does not preserve the fitted model state"
        ),
    },
    KMeans: {
        "check_sample_weight_equivalence_on_dense_data": "Sample weights not equal to repeating data",
    },
    LogisticRegression: {
        "check_sample_weight_equivalence_on_dense_data": "Sample weights not equal to repeating data",
        "check_sample_weight_equivalence_on_sparse_data": "Sample weights not equal to repeating data",
        "check_class_weight_classifiers": "LogisticRegression does not handle class weights properly",
    },
    Ridge: {
        "check_non_transformer_estimators_n_iter": "Ridge `n_iter_` may be `None`",
    },
    RandomForestClassifier: {
        "check_sample_weight_equivalence_on_dense_data": (
            "RandomForest uses quantile-binned splits, so sample weighting is "
            "not equivalent to duplicating rows"
        ),
    },
    RandomForestRegressor: {
        "check_sample_weight_equivalence_on_dense_data": (
            "RandomForest uses quantile-binned splits, so sample weighting is "
            "not equivalent to duplicating rows"
        ),
    },
    LinearSVC: {
        "check_sample_weight_equivalence_on_dense_data": "Sample weights not equal to repeating data",
    },
    LinearSVR: {
        "check_sample_weight_equivalence_on_dense_data": "Sample weights not equal to repeating data",
    },
    SVC: {
        "check_sample_weight_equivalence_on_dense_data": "Sample weights not equal to repeating data",
        "check_sample_weight_equivalence_on_sparse_data": "Sample weights not equal to repeating data",
    },
    SVR: {
        "check_sample_weight_equivalence_on_dense_data": "Sample weights not equal to repeating data",
        "check_sample_weight_equivalence_on_sparse_data": "Sample weights not equal to repeating data",
    },
    TSNE: {
        "check_dont_overwrite_parameters": "TSNE only supports n_components = 2",
        "check_methods_sample_order_invariance": "TSNE results depend on sample order",
        "check_methods_subset_invariance": "TSNE results depend on data subset",
        "check_fit2d_predict1d": "TSNE only supports n_components = 2",
    },
    UMAP: {
        "check_transformer_data_not_an_array": (
            "UMAP does not have consistent fit_transform and transform outputs"
        ),
        "check_methods_sample_order_invariance": "UMAP results depend on sample order",
        "check_transformer_general": "UMAP does not have consistent fit_transform and transform outputs",
        "check_methods_subset_invariance": "UMAP results depend on data subset",
        "check_transformer_preserve_dtypes": "UMAP returns float32 embeddings",
    },
    Lasso: {
        "check_sample_weight_equivalence_on_sparse_data": "Sample weights not equal to repeating data",
    },
    ElasticNet: {
        "check_sample_weight_equivalence_on_sparse_data": "Sample weights not equal to repeating data",
    },
    StandardScaler: {
        "check_no_attributes_set_in_init": "Vendored __init__ sets copy/with_mean/with_std as attributes",
        "check_do_not_raise_errors_in_init_or_set_params": "StandardScaler(**params) raises an exception",
    },
}


# Sanity check that xfails listed have at least one estimator instance
if missing := set(XFAILS).difference((type(est) for est in ESTIMATORS)):
    raise ValueError(
        f"xfails defined for {missing}, but that estimator isn't tested!"
    )


@estimator_checks.parametrize_with_checks(
    ESTIMATORS,
    expected_failed_checks=lambda est: XFAILS.get(type(est), {}),
    xfail_strict=True,
)
@pytest.mark.filterwarnings(
    "ignore:ValueError occurred during set_params.*:UserWarning"
)
@pytest.mark.filterwarnings(
    "ignore:TypeError occurred during set_params.*:UserWarning"
)
@pytest.mark.filterwarnings(
    "ignore:perplexity.*should be less than n_samples.*:UserWarning"
)
@pytest.mark.filterwarnings(
    "ignore:Estimator's parameters changed after set_params raised ValueError:UserWarning"
)
@pytest.mark.filterwarnings("ignore:Changing solver to 'svd'.*:UserWarning")
@pytest.mark.filterwarnings("ignore:The number of bins.*:UserWarning")
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_sklearn_compatible_estimator(estimator, check):
    if isinstance(estimator, RandomForestRegressor) and (
        estimator_checks._get_check_estimator_ids(check)
        == "check_regressor_data_not_an_array"
    ):
        pytest.skip(
            "Predictions from repeated fits are nondeterministic; see "
            "https://github.com/rapidsai/cuml/issues/8457"
        )
    check(estimator)


def test_all_estimators_covered():
    all_estimators = _all_cuml_estimators()
    tested = {type(est) for est in ESTIMATORS}
    excluded = set(EXCLUDED)

    overlap = tested & excluded
    assert not overlap, "Estimators both tested and excluded: " + ", ".join(
        c.__name__ for c in sorted(overlap, key=lambda c: c.__name__)
    )

    uncovered = all_estimators - tested - excluded
    assert not uncovered, (
        "Estimators not in ESTIMATORS or EXCLUDED: "
        + ", ".join(
            c.__name__ for c in sorted(uncovered, key=lambda c: c.__name__)
        )
        + ". Add them to ESTIMATORS or EXCLUDED with a reason."
    )

    stale = excluded - all_estimators
    assert not stale, (
        "EXCLUDED contains classes not found by discovery: "
        + ", ".join(
            c.__name__ for c in sorted(stale, key=lambda c: c.__name__)
        )
    )
