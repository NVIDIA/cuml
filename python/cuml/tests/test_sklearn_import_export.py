# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
import numpy as np
import pytest
import scipy
import scipy.sparse
import sklearn
import sklearn.kernel_ridge
import sklearn.preprocessing
import sklearn.svm
import umap
from numpy.testing import assert_allclose
from packaging.version import Version
from sklearn.cluster import KMeans as SkKMeans
from sklearn.cluster import SpectralClustering as SkSpectralClustering
from sklearn.datasets import (
    make_blobs,
    make_classification,
    make_multilabel_classification,
    make_regression,
)
from sklearn.decomposition import PCA as SkPCA
from sklearn.decomposition import TruncatedSVD as SkTruncatedSVD
from sklearn.linear_model import ElasticNet as SkElasticNet
from sklearn.linear_model import Lasso as SkLasso
from sklearn.linear_model import LinearRegression as SkLinearRegression
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.linear_model import Ridge as SkRidge
from sklearn.manifold import TSNE as SkTSNE
from sklearn.manifold import trustworthiness
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

import cuml
from cuml.cluster import DBSCAN, KMeans, SpectralClustering
from cuml.decomposition import PCA, TruncatedSVD
from cuml.internals.interop import UnsupportedOnCPU, UnsupportedOnGPU
from cuml.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from cuml.manifold import TSNE, SpectralEmbedding
from cuml.preprocessing import TargetEncoder
from cuml.testing.utils import array_equal

###############################################################################
#                              Helper functions                               #
###############################################################################


SKLEARN_18 = Version(sklearn.__version__) >= Version("1.8.0.dev0")


def _filter_scipy_lbfgsb_deprecation(func):
    if Version("1.16") <= Version(scipy.__version__) < Version("1.18"):
        return pytest.mark.filterwarnings(
            "ignore:.*The `disp` and `iprint` options.*:DeprecationWarning"
        )(func)
    return func


@pytest.fixture
def random_state():
    return 42


def assert_roundtrip_consistency(original, roundtrip, exclude=()):
    original_params = original.get_params()
    roundtrip_params = roundtrip.get_params()

    # Remove parameters that are not guaranteed to be equivalent
    for name in [*exclude, "handle", "output_type", "verbose"]:
        original_params.pop(name, None)
        roundtrip_params.pop(name, None)

    assert original_params == roundtrip_params

    # Next check attributes are consistent. We don't check for equality
    # since that might change. We do check for consistency in attribute
    # availability, as well as dtype/shape/order of array values
    attrs = [
        name
        for name in dir(original)
        if name.endswith("_")
        and not name.startswith("_")
        and not name == "feature_names_in_"
        and name not in exclude
        and hasattr(original, name)
    ]
    with cuml.using_output_type("cupy"):
        for name in attrs:
            orig_attr = getattr(original, name)
            roundtrip_attr = getattr(roundtrip, name)
            if isinstance(orig_attr, cp.ndarray):
                assert orig_attr.shape == roundtrip_attr.shape, name
                assert orig_attr.dtype == roundtrip_attr.dtype, name
                assert (
                    orig_attr.flags["C_CONTIGUOUS"]
                    == roundtrip_attr.flags["C_CONTIGUOUS"]
                ), name
                assert (
                    orig_attr.flags["F_CONTIGUOUS"]
                    == roundtrip_attr.flags["F_CONTIGUOUS"]
                ), name


def assert_estimator_roundtrip(
    cuml_model, sklearn_class, X, y=None, transform=False, exclude_params=()
):
    """
    Generic assertion helper to test round-trip conversion:
    fit original custom model
    convert to sklearn
    convert back to custom model
    compare predictions or transform outputs
    """
    # Fit original model
    if y is not None:
        cuml_model.fit(X, y)
    else:
        cuml_model.fit(X)

    # Convert to sklearn model
    sklearn_model = cuml_model.as_sklearn()
    assert isinstance(sklearn_model, sklearn_class)
    check_is_fitted(sklearn_model)

    # Convert back
    roundtrip_model = type(cuml_model).from_sklearn(sklearn_model)

    # Ensure params/attrs roundtrip
    assert_roundtrip_consistency(
        cuml_model, roundtrip_model, exclude=exclude_params
    )

    # Ensure roundtrip model is fitted
    check_is_fitted(roundtrip_model)

    # Compare predictions or transforms
    if transform:
        original_output = cuml_model.transform(X)
        roundtrip_output = roundtrip_model.transform(X)
        assert array_equal(original_output, roundtrip_output)
    else:
        # For predict methods
        if hasattr(cuml_model, "predict"):
            original_pred = cuml_model.predict(X)
            roundtrip_pred = roundtrip_model.predict(X)
            assert array_equal(original_pred, roundtrip_pred)
        # For models that only produce labels_ or similar attributes (e.g., clustering)
        elif hasattr(cuml_model, "labels_"):
            assert array_equal(cuml_model.labels_, roundtrip_model.labels_)
        else:
            # If we get here, need a custom handling for that type
            raise NotImplementedError(
                "No known method to compare outputs of this model."
            )

    # Check that the scikit-learn estimator can be fitted again which checks
    # that the hyper-parameter translation from cuml to scikit-learn works
    # Has to happen after comparing predictions/transforms as refitting might
    # change cluster IDs and the like
    if y is not None:
        sklearn_model.fit(X, y)
    else:
        sklearn_model.fit(X)


###############################################################################
#                                     Tests                                   #
###############################################################################


def test_basic_roundtrip():
    km = SkKMeans(n_clusters=13)
    ckm = KMeans.from_sklearn(km)

    assert ckm.n_clusters == 13


def test_kmeans(random_state):
    # Using sklearn directly for demonstration
    X, _ = make_blobs(
        n_samples=50, n_features=2, centers=3, random_state=random_state
    )
    original = KMeans(n_clusters=3, random_state=random_state)
    assert_estimator_roundtrip(original, SkKMeans, X)


def test_dbscan(random_state):
    X, _ = make_blobs(
        n_samples=50, n_features=2, centers=3, random_state=random_state
    )
    original = DBSCAN(eps=0.5, min_samples=5)
    # DBSCAN assigns labels_ after fit
    original.fit(X)
    sklearn_model = original.as_sklearn()
    roundtrip_model = DBSCAN.from_sklearn(sklearn_model)
    assert array_equal(original.labels_, roundtrip_model.labels_)


def test_spectral_clustering(random_state):
    X, _ = make_blobs(
        n_samples=100, n_features=10, centers=3, random_state=random_state
    )
    X = X.astype(np.float32)
    original = SpectralClustering(
        n_clusters=3,
        affinity="nearest_neighbors",
        n_neighbors=10,
        random_state=random_state,
    )
    assert_estimator_roundtrip(original, SkSpectralClustering, X)


def test_pca(random_state):
    X = np.random.RandomState(random_state).rand(50, 5)
    original = PCA(n_components=2)
    assert_estimator_roundtrip(original, SkPCA, X, transform=True)


def test_truncated_svd(random_state):
    X = np.random.RandomState(random_state).rand(50, 5)
    original = TruncatedSVD(n_components=2)
    assert_estimator_roundtrip(original, SkTruncatedSVD, X, transform=True)


def test_linear_regression(random_state):
    X, y = make_regression(
        n_samples=50, n_features=5, noise=0.1, random_state=random_state
    )
    original = LinearRegression()
    assert_estimator_roundtrip(original, SkLinearRegression, X, y)


@_filter_scipy_lbfgsb_deprecation
def test_logistic_regression(random_state):
    # Ignore SciPy 1.16/1.17 deprecation warnings emitted through sklearn's
    # default LogisticRegression L-BFGS-B path.
    X, y = make_classification(
        n_samples=50, n_features=5, n_informative=3, random_state=random_state
    )
    original = LogisticRegression(C=5.0, max_iter=500)
    assert_estimator_roundtrip(original, SkLogisticRegression, X, y)


def test_elasticnet(random_state):
    X, y = make_regression(
        n_samples=50, n_features=5, noise=0.1, random_state=random_state
    )
    original = ElasticNet(alpha=0.1)
    assert_estimator_roundtrip(original, SkElasticNet, X, y)


def test_ridge(random_state):
    X, y = make_regression(
        n_samples=50, n_features=5, noise=0.1, random_state=random_state
    )
    original = Ridge(alpha=1.0)
    assert_estimator_roundtrip(original, SkRidge, X, y)


def test_lasso(random_state):
    X, y = make_regression(
        n_samples=50, n_features=5, noise=0.1, random_state=random_state
    )
    original = Lasso(alpha=0.1)
    assert_estimator_roundtrip(original, SkLasso, X, y)


@pytest.mark.filterwarnings(
    "ignore:Starting from version 22.04, the default method of TSNE is 'fft'."
)
def test_tsne(random_state):
    # TSNE is a bit tricky as it is non-deterministic. For test simplicity:
    X = np.random.RandomState(random_state).rand(50, 5)
    original = TSNE(n_components=2, random_state=random_state)
    original.fit(X)
    sklearn_model = original.as_sklearn()
    roundtrip_model = TSNE.from_sklearn(sklearn_model)
    # Since TSNE is non-deterministic, exact match is unlikely.
    # We can at least check output dimensions are the same.
    original_embedding = original.embedding_
    sklearn_embedding = sklearn_model.embedding_
    roundtrip_embedding = roundtrip_model.embedding_

    assert array_equal(original_embedding, sklearn_embedding)
    assert array_equal(original_embedding, roundtrip_embedding)


def test_tsne_from_sklearn_numeric_learning_rate():
    model = TSNE.from_sklearn(
        SkTSNE(learning_rate=100.0, early_exaggeration=10.0)
    )

    assert model.learning_rate == 100.0
    assert model.learning_rate_method == "none"
    assert model.early_exaggeration == 10.0


def test_spectral_embedding(random_state):
    X, _ = make_blobs(
        n_samples=100, centers=3, n_features=20, random_state=random_state
    )
    original = SpectralEmbedding(n_components=2, random_state=random_state)
    original.fit(X)
    sklearn_model = original.as_sklearn()
    roundtrip_model = SpectralEmbedding.from_sklearn(sklearn_model)
    assert_roundtrip_consistency(original, roundtrip_model)

    original_embedding = original.embedding_
    sklearn_embedding = sklearn_model.embedding_
    roundtrip_embedding = roundtrip_model.embedding_

    assert array_equal(original_embedding, sklearn_embedding)
    assert array_equal(original_embedding, roundtrip_embedding)


@pytest.mark.parametrize("alpha_kind", ["scalar", "numpy", "cupy"])
def test_kernel_ridge(random_state, alpha_kind):
    X, y = make_regression(
        n_samples=50,
        n_features=10,
        n_targets=2,
        noise=0.1,
        random_state=random_state,
    )
    if alpha_kind == "scalar":
        cu_alpha = sk_alpha = 1
    elif alpha_kind == "numpy":
        cu_alpha = sk_alpha = np.array([1.0, 2.0])
    elif alpha_kind == "cupy":
        cu_alpha = cp.array([1.0, 2.0])
        sk_alpha = cu_alpha.get()

    cu_model = cuml.KernelRidge(alpha=cu_alpha).fit(X, y)
    sk_model = sklearn.kernel_ridge.KernelRidge(alpha=sk_alpha).fit(X, y)

    sk_model2 = cu_model.as_sklearn()
    cu_model2 = cuml.KernelRidge.from_sklearn(sk_model)

    # Can infer on converted models
    assert sk_model2.score(X, y) > 0.7
    assert cu_model2.score(X, y) > 0.7

    # Can refit on converted models
    cu_model2.fit(X, y)
    sk_model2.fit(X, y)

    # Refit models have similar results
    assert sk_model2.score(X, y) > 0.7
    assert cu_model2.score(X, y) > 0.7


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("kernel", ["rbf", "precomputed"])
def test_svr(random_state, sparse, kernel):
    X, y = make_regression(n_samples=100, random_state=random_state)
    if kernel == "precomputed":
        if sparse:
            pytest.skip("precomputed kernel not supported with sparse")
        K = X @ X.T  # Linear kernel matrix
        original = cuml.SVR(kernel="precomputed")
        assert_estimator_roundtrip(original, sklearn.svm.SVR, K, y)
    else:
        if sparse:
            X = scipy.sparse.coo_matrix(X)
        original = cuml.SVR()
        assert_estimator_roundtrip(original, sklearn.svm.SVR, X, y)

        # Check inference works after conversion
        cu_model = cuml.SVR(kernel="linear").fit(X, y)
        sk_model = sklearn.svm.SVR(kernel="linear").fit(X, y)

        sk_score = cu_model.as_sklearn().score(X, y)
        assert sk_score > 0.7

        cu_model_from_sklearn = cuml.SVR.from_sklearn(sk_model)

        cu_score = cu_model_from_sklearn.score(X, y)
        assert cu_score > 0.7

        # Check n_support is set correctly
        assert (
            cu_model_from_sklearn.n_support_
            == cu_model_from_sklearn.support_vectors_.shape[0]
        )


@pytest.mark.parametrize("sparse", [False, True])
def test_svc(random_state, sparse):
    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=3, random_state=random_state
    )
    if sparse:
        X = scipy.sparse.coo_matrix(X)
    original = cuml.SVC()
    assert_estimator_roundtrip(original, sklearn.svm.SVC, X, y)

    # Check inference works after conversion.
    cu_model = cuml.SVC().fit(X, y)
    sk_model2 = cu_model.as_sklearn()
    sk_score = sk_model2.score(X, y)
    assert sk_score > 0.7

    sk_model = sklearn.svm.SVC().fit(X, y)
    cu_model2 = cuml.SVC.from_sklearn(sk_model)

    cu_score = cu_model2.score(X, y)
    assert cu_score > 0.7

    # Check n_support_ is correctly set
    assert cu_model2.n_support_ == cu_model2.support_vectors_.shape[0]


def test_svc_precomputed(random_state):
    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=3, random_state=random_state
    )
    K = X @ X.T  # Linear kernel matrix
    original = cuml.SVC(kernel="precomputed")
    assert_estimator_roundtrip(original, sklearn.svm.SVC, K, y)


@pytest.mark.parametrize("kind", ["SVC", "SVR"])
def test_svm_big_sparse(random_state, kind):
    """SVC/SVR models are only sparse if fit with a sparse matrix AND storing
    `support_vectors_` as dense exceeds a limit (currently 1 GiB). Here we
    create a large enough input matrix to ensure a sparse repr, and check that
    the conversion is plumbed through properly"""
    n_samples = 1024
    noise_cols = 300000
    noise_density = 0.0001
    n_noise = int(noise_cols * n_samples * noise_density)

    if kind == "SVR":
        X, y = make_regression(
            n_samples=n_samples, n_features=4, random_state=random_state
        )
        model = cuml.SVR(kernel="linear")
    else:
        X, y = make_classification(
            n_samples=n_samples, n_features=4, random_state=random_state
        )
        model = cuml.SVC()

    rng = np.random.default_rng(random_state)
    rows = rng.integers(0, n_samples, size=n_noise)
    cols = rng.integers(0, noise_cols, size=n_noise)
    data = rng.uniform(size=n_noise)

    X = scipy.sparse.hstack(
        [
            scipy.sparse.csr_matrix(X),
            scipy.sparse.coo_matrix(
                (data, (rows, cols)), shape=(n_samples, noise_cols)
            ).asformat("csr"),
        ]
    )

    model.fit(X, y)

    sk_model = model.as_sklearn()
    assert scipy.sparse.issparse(sk_model.support_vectors_)

    score = sk_model.score(X[:100], y[:100])
    assert score > 0.7


def test_svc_multiclass_unsupported(random_state):
    X, y = make_classification(
        n_samples=50,
        n_features=10,
        n_classes=3,
        n_informative=5,
        random_state=random_state,
    )
    cu_model = cuml.SVC().fit(X, y)
    sk_model = sklearn.svm.SVC().fit(X, y)

    with pytest.raises(UnsupportedOnGPU):
        cuml.SVC.from_sklearn(sk_model)

    with pytest.raises(UnsupportedOnCPU):
        cu_model.as_sklearn()


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("supervised", [False, True])
def test_umap(random_state, sparse, supervised):
    n_neighbors = 10
    X, y = make_blobs(n_samples=200, random_state=random_state)
    X = X.astype("float32")
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    if sparse:
        X_train = scipy.sparse.csr_matrix(X_train)
        X_test = scipy.sparse.csr_matrix(X_test)
    if not supervised:
        y_train = None

    cu_model = cuml.UMAP(n_neighbors=n_neighbors, hash_input=True).fit(
        X_train, y_train
    )
    sk_model = umap.UMAP(n_neighbors=n_neighbors).fit(X_train, y_train)

    sk_model2 = cu_model.as_sklearn()
    cu_model2 = cuml.UMAP.from_sklearn(sk_model)

    # Ensure params/attrs roundtrip
    assert_roundtrip_consistency(cu_model, cu_model2, exclude=["build_algo"])

    # Can infer on converted models
    np.testing.assert_array_equal(
        sk_model2.transform(X_train), sk_model2.embedding_
    )
    with pytest.raises(NotImplementedError):
        # Can't currently infer on new data in umap.UMAP.transform implementation
        assert isinstance(sk_model2.transform(X_test), np.ndarray)

    np.testing.assert_array_equal(
        cu_model2.transform(X_train), cu_model2.embedding_
    )
    assert isinstance(cu_model2.transform(X_test), np.ndarray)

    # Can refit on converted models
    cu_model2.fit(X_train, y_train)
    sk_model2.fit(X_train, y_train)

    # Refit embeddings have similar scores
    cu_trust1 = trustworthiness(
        X_train, cu_model.embedding_, n_neighbors=n_neighbors
    )
    cu_trust2 = trustworthiness(
        X_train, cu_model2.embedding_, n_neighbors=n_neighbors
    )
    np.testing.assert_allclose(cu_trust1, cu_trust2, atol=0.05)

    sk_trust1 = trustworthiness(
        X_train, sk_model.embedding_, n_neighbors=n_neighbors
    )
    sk_trust2 = trustworthiness(
        X_train, sk_model2.embedding_, n_neighbors=n_neighbors
    )
    np.testing.assert_allclose(sk_trust1, sk_trust2, atol=0.05)


@pytest.mark.parametrize("sparse", [False, True])
def test_nearest_neighbors(random_state, sparse):
    if sparse:
        X = scipy.sparse.rand(
            50,
            20,
            density=0.25,
            rng=random_state,
            dtype="float32",
            format="csr",
        )
    else:
        X = np.random.default_rng(random_state).random(
            (50, 20), dtype="float32"
        )

    cu_model = cuml.NearestNeighbors(metric="minkowski", n_neighbors=10).fit(X)
    sk_model = sklearn.neighbors.NearestNeighbors(n_neighbors=10).fit(X)

    sk_model2 = cu_model.as_sklearn()
    cu_model2 = cuml.NearestNeighbors.from_sklearn(sk_model)

    # Ensure params/attrs roundtrip
    assert_roundtrip_consistency(cu_model, cu_model2)

    def assert_kneighbors_close(m1, m2):
        inds1, dists1 = m1.kneighbors(X)
        inds2, dists2 = m2.kneighbors(X)
        np.testing.assert_allclose(inds1, inds2, atol=1e-7)
        np.testing.assert_allclose(dists1, dists2, atol=1e-4)

    # Can infer on converted models
    assert_kneighbors_close(sk_model, sk_model2)
    assert_kneighbors_close(cu_model, cu_model2)

    # Can refit on converted models
    cu_model2.fit(X)
    sk_model2.fit(X)

    # Refit models have similar results
    assert_kneighbors_close(sk_model, sk_model2)
    assert_kneighbors_close(cu_model, cu_model2)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("n_targets", [1, 3])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_kneighbors_regressor(random_state, sparse, n_targets, weights):
    X, y = make_regression(
        100, 50, n_targets=n_targets, random_state=random_state
    )
    X = X.astype("float32")
    if sparse:
        X[X < -0.5] = 0
        X = scipy.sparse.csr_matrix(X)

    cu_model = cuml.KNeighborsRegressor(
        metric="minkowski", n_neighbors=10, weights=weights
    ).fit(X, y)
    sk_model = sklearn.neighbors.KNeighborsRegressor(
        n_neighbors=10, weights=weights
    ).fit(X, y)

    sk_model2 = cu_model.as_sklearn()
    cu_model2 = cuml.KNeighborsRegressor.from_sklearn(sk_model)

    # Ensure params/attrs roundtrip
    assert_roundtrip_consistency(cu_model, cu_model2)

    # Verify weights parameter is preserved
    assert cu_model2.weights == weights
    assert sk_model2.weights == weights

    # Can infer on converted models
    assert_allclose(sk_model.predict(X), sk_model2.predict(X), atol=1e-3)
    assert_allclose(cu_model.predict(X), cu_model2.predict(X), atol=1e-3)

    # Can refit on converted models
    cu_model2.fit(X, y)
    sk_model2.fit(X, y)

    # Refit models have similar results
    assert_allclose(sk_model.predict(X), sk_model2.predict(X), atol=1e-3)
    assert_allclose(cu_model.predict(X), cu_model2.predict(X), atol=1e-3)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("n_labels", [1, 3])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_kneighbors_classifier(random_state, sparse, n_labels, weights):
    if n_labels > 1:
        X, y = make_multilabel_classification(
            100,
            50,
            n_labels=n_labels,
            random_state=random_state,
            sparse=sparse,
        )
    else:
        X, y = make_classification(100, 50, random_state=random_state)
        if sparse:
            X[X < -0.5] = 0
            X = scipy.sparse.csr_matrix(X)

    X = X.astype("float32")

    cu_model = cuml.KNeighborsClassifier(
        metric="minkowski", n_neighbors=10, weights=weights
    ).fit(X, y)
    sk_model = sklearn.neighbors.KNeighborsClassifier(
        n_neighbors=10, weights=weights
    ).fit(X, y)

    sk_model2 = cu_model.as_sklearn()
    cu_model2 = cuml.KNeighborsClassifier.from_sklearn(sk_model)

    # `classes_` attribute transfers properly
    if n_labels > 1:
        assert all(isinstance(c, np.ndarray) for c in sk_model2.classes_)
        assert all(isinstance(c, np.ndarray) for c in cu_model2.classes_)
    else:
        assert isinstance(sk_model2.classes_, np.ndarray)
        assert isinstance(cu_model2.classes_, np.ndarray)

    # Ensure params/attrs roundtrip
    assert_roundtrip_consistency(cu_model, cu_model2)

    # Verify weights parameter is preserved
    assert cu_model2.weights == weights
    assert sk_model2.weights == weights

    # Can infer on converted models
    np.testing.assert_array_equal(sk_model.predict(X), sk_model2.predict(X))
    np.testing.assert_array_equal(cu_model.predict(X), cu_model2.predict(X))

    # Can refit on converted models
    cu_model2.fit(X, y)
    sk_model2.fit(X, y)

    # Refit models have similar results
    np.testing.assert_array_equal(sk_model.predict(X), sk_model2.predict(X))
    np.testing.assert_array_equal(cu_model.predict(X), cu_model2.predict(X))


def test_kneighbors_regressor_callable_weights_unsupported(random_state):
    """Test that callable weights raise an error during sklearn -> cuml conversion"""
    X, y = make_regression(100, 50, random_state=random_state)
    X = X.astype("float32")

    def custom_weights(distances):
        return np.exp(-distances)

    sk_model = sklearn.neighbors.KNeighborsRegressor(
        n_neighbors=10, weights=custom_weights
    ).fit(X, y)

    with pytest.raises(
        UnsupportedOnGPU, match="Callable weights are not supported"
    ):
        cuml.KNeighborsRegressor.from_sklearn(sk_model)


def test_kneighbors_classifier_callable_weights_unsupported(random_state):
    """Test that callable weights raise an error during sklearn -> cuml conversion"""
    X, y = make_classification(100, 50, random_state=random_state)
    X = X.astype("float32")

    def custom_weights(distances):
        return np.exp(-distances)

    sk_model = sklearn.neighbors.KNeighborsClassifier(
        n_neighbors=10, weights=custom_weights
    ).fit(X, y)

    with pytest.raises(
        UnsupportedOnGPU, match="Callable weights are not supported"
    ):
        cuml.KNeighborsClassifier.from_sklearn(sk_model)


@pytest.mark.parametrize("kernel", ["gaussian", "tophat"])
@pytest.mark.parametrize("bandwidth", ["scott", 5.0])
@pytest.mark.parametrize("metric", ["l1", "l2"])
def test_kernel_density(random_state, kernel, bandwidth, metric):
    opts = {"kernel": kernel, "bandwidth": bandwidth, "metric": metric}

    X, _ = make_blobs(random_state=random_state)

    cu_model = cuml.KernelDensity(**opts).fit(X)
    sk_model = sklearn.neighbors.KernelDensity(**opts).fit(X)

    sk_model2 = cu_model.as_sklearn()
    cu_model2 = cuml.KernelDensity.from_sklearn(sk_model)

    # Ensure params/attrs roundtrip
    assert_roundtrip_consistency(cu_model, cu_model2)

    def assert_kde_close(m1, m2):
        scores1 = m1.score_samples(X)
        scores2 = m2.score_samples(X)
        np.testing.assert_allclose(scores1, scores2)

    # Can infer on converted models
    assert_kde_close(sk_model, sk_model2)
    assert_kde_close(cu_model, cu_model2)

    # Can refit on converted models
    cu_model2.fit(X)
    sk_model2.fit(X)

    # Refit models have similar results
    assert_kde_close(sk_model, sk_model2)
    assert_kde_close(cu_model, cu_model2)


@pytest.mark.parametrize("oob_score", [False, True])
def test_random_forest_classifier(random_state, oob_score):
    X, y = make_classification(
        n_samples=200, n_features=5, n_informative=3, random_state=random_state
    )

    cu_model = cuml.RandomForestClassifier(
        oob_score=oob_score,
        max_depth=None,
    ).fit(X, y)
    sk_model = sklearn.ensemble.RandomForestClassifier(
        oob_score=oob_score,
        max_depth=None,
    ).fit(X, y)

    sk_model2 = cu_model.as_sklearn()
    cu_model2 = cuml.RandomForestClassifier.from_sklearn(sk_model)

    # `classes_` attribute transfers properly
    assert isinstance(sk_model2.classes_, np.ndarray)
    assert isinstance(cu_model2.classes_, np.ndarray)
    assert (sk_model2.classes_ == cu_model2.classes_).all()
    for estimator in sk_model2.estimators_:
        estimator_classes = np.arange(
            sk_model2.n_classes_, dtype=estimator.classes_.dtype
        )
        assert isinstance(estimator.classes_, np.ndarray)
        assert (estimator.classes_ == estimator_classes).all()
        assert estimator.n_classes_ == sk_model2.n_classes_

    # Ensure params/attrs roundtrip
    # Exclude classes_ due to dtype differences between implementations
    # Exclude feature_importances_ because sklearn can't compute them from
    # treelite-exported models
    assert_roundtrip_consistency(
        cu_model, cu_model2, exclude=("classes_", "feature_importances_")
    )

    # Verify OOB attributes are present when oob_score=True
    if oob_score:
        assert hasattr(cu_model, "oob_score_")
        assert hasattr(cu_model2, "oob_score_")
        assert hasattr(sk_model2, "oob_score_")
        assert cu_model.oob_score_ == sk_model2.oob_score_
        assert cu_model2.oob_score_ == sk_model.oob_score_
        assert isinstance(sk_model2.oob_decision_function_, np.ndarray)
        with cuml.using_output_type("cupy"):
            assert isinstance(cu_model.oob_decision_function_, cp.ndarray)
            assert isinstance(cu_model2.oob_decision_function_, cp.ndarray)

    # Can infer on converted models
    assert sk_model2.score(X, y) > 0.7
    assert cu_model2.score(X, y) > 0.7

    # Can refit on converted models
    cu_model2.fit(X, y)
    sk_model2.fit(X, y)

    # Refit models have similar results
    assert sk_model2.score(X, y) > 0.7
    assert cu_model2.score(X, y) > 0.7


def test_random_forest_classifier_as_sklearn_estimator_classes(random_state):
    X, y = make_classification(
        n_samples=200, n_features=5, n_informative=3, random_state=random_state
    )
    y = np.where(y == 0, 10, 20)
    common_params = {"n_estimators": 3, "max_depth": None}

    cu_model = cuml.RandomForestClassifier(**common_params).fit(X, y)
    sk_model = sklearn.ensemble.RandomForestClassifier(**common_params).fit(
        X, y
    )
    sk_model2 = cu_model.as_sklearn()

    np.testing.assert_array_equal(sk_model2.classes_, sk_model.classes_)
    for estimator, sk_estimator in zip(
        sk_model2.estimators_, sk_model.estimators_, strict=True
    ):
        np.testing.assert_array_equal(
            estimator.classes_, sk_estimator.classes_
        )
        assert estimator.n_classes_ == sk_estimator.n_classes_


@pytest.mark.parametrize("oob_score", [False, True])
def test_random_forest_regressor(random_state, oob_score):
    X, y = make_regression(n_samples=200, random_state=random_state)
    X = X.astype("float32")

    cu_model = cuml.RandomForestRegressor(
        oob_score=oob_score,
        max_depth=None,
    ).fit(X, y)
    sk_model = sklearn.ensemble.RandomForestRegressor(
        oob_score=oob_score,
        max_depth=None,
    ).fit(X, y)

    sk_model2 = cu_model.as_sklearn()
    cu_model2 = cuml.RandomForestRegressor.from_sklearn(sk_model)

    # Ensure params/attrs roundtrip
    # Exclude feature_importances_ because sklearn can't compute them from
    # treelite-exported models
    assert_roundtrip_consistency(
        cu_model, cu_model2, exclude=("feature_importances_",)
    )

    # Verify OOB attributes are present when oob_score=True
    if oob_score:
        assert hasattr(cu_model, "oob_score_")
        assert hasattr(cu_model2, "oob_score_")
        assert hasattr(sk_model2, "oob_score_")
        assert cu_model.oob_score_ == sk_model2.oob_score_
        assert cu_model2.oob_score_ == sk_model.oob_score_
        assert isinstance(sk_model2.oob_prediction_, np.ndarray)
        with cuml.using_output_type("cupy"):
            assert isinstance(cu_model.oob_prediction_, cp.ndarray)
            assert isinstance(cu_model2.oob_prediction_, cp.ndarray)

    # Can infer on converted models
    assert sk_model2.score(X, y) > 0.7
    assert cu_model2.score(X, y) > 0.7

    # Can refit on converted models
    cu_model2.fit(X, y)
    sk_model2.fit(X, y)

    # Refit models have similar results
    assert sk_model2.score(X, y) > 0.7
    assert cu_model2.score(X, y) > 0.7


@pytest.mark.parametrize("prediction_data", [False, True])
@pytest.mark.parametrize("gen_min_span_tree", [False, True])
@pytest.mark.skipif(SKLEARN_18, reason="hdbscan requires sklearn < 1.8.0")
# Ignore FutureWarning from third-party hdbscan package calling
# sklearn.utils.validation.check_array with deprecated 'force_all_finite'
# parameter. This is not in cuml's control. Note: this will break when
# sklearn 1.8 removes the deprecated parameter entirely - hdbscan will
# need to be updated at that point.
@pytest.mark.filterwarnings(
    "ignore:'force_all_finite' was renamed to "
    "'ensure_all_finite':FutureWarning:sklearn"
)
def test_hdbscan(random_state, prediction_data, gen_min_span_tree):
    hdbscan = pytest.importorskip("hdbscan")

    X, y = make_blobs(random_state=random_state)

    cu_model = cuml.HDBSCAN(
        prediction_data=prediction_data, gen_min_span_tree=gen_min_span_tree
    ).fit(X, y)
    sk_model = hdbscan.HDBSCAN(
        prediction_data=prediction_data, gen_min_span_tree=gen_min_span_tree
    ).fit(X, y)

    sk_model2 = cu_model.as_sklearn()
    cu_model2 = cuml.HDBSCAN.from_sklearn(sk_model)

    # Ensure params/attrs roundtrip
    assert_roundtrip_consistency(cu_model, cu_model2)

    # tree attributes all available
    for attr in ["single_linkage_tree_", "condensed_tree_"]:
        assert getattr(cu_model2, attr) is not None
        assert getattr(sk_model2, attr) is not None

    # minimum_spanning_tree_ available if generated
    if gen_min_span_tree:
        assert getattr(cu_model2, "minimum_spanning_tree_") is not None
        assert getattr(sk_model2, "minimum_spanning_tree_") is not None
    else:
        assert not hasattr(cu_model2, "minimum_spanning_tree_")
        assert not hasattr(sk_model2, "minimum_spanning_tree_")

    # prediction data available if generated
    if prediction_data:
        assert getattr(cu_model2, "prediction_data_") is not None
        assert getattr(sk_model2, "prediction_data_") is not None
    else:
        assert not hasattr(cu_model2, "prediction_data_")
        assert not hasattr(sk_model2, "prediction_data_")

        # Generate prediction data
        cu_model2.generate_prediction_data()
        sk_model2.generate_prediction_data()

    # Can infer on converted models
    cu_labels, cu_probs = cuml.cluster.hdbscan.approximate_predict(
        cu_model2, X
    )
    sk_labels, sk_probs = hdbscan.approximate_predict(sk_model2, X)
    assert accuracy_score(cu_labels, cu_model2.labels_) > 0.9
    assert accuracy_score(sk_labels, sk_model2.labels_) > 0.9

    # Can refit on converted models
    cu_model2.fit(X, y)
    sk_model2.fit(X, y)


def test_linear_svr(random_state):
    X, y = make_regression(n_samples=100, random_state=random_state)
    original = cuml.LinearSVR(loss="squared_epsilon_insensitive", penalty="l2")
    assert_estimator_roundtrip(original, sklearn.svm.LinearSVR, X, y)

    # Check inference works after conversion
    cu_model = cuml.LinearSVR(
        loss="squared_epsilon_insensitive", penalty="l2"
    ).fit(X, y)
    sk_model = sklearn.svm.LinearSVR(loss="squared_epsilon_insensitive").fit(
        X, y
    )

    sk_score = cu_model.as_sklearn().score(X, y)
    assert sk_score > 0.7

    cu_score = cuml.LinearSVR.from_sklearn(sk_model).score(X, y)
    assert cu_score > 0.7


def test_linear_svc(random_state):
    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=3, random_state=random_state
    )
    original = cuml.LinearSVC()
    assert_estimator_roundtrip(original, sklearn.svm.LinearSVC, X, y)

    # Check inference works after conversion
    cu_model = cuml.LinearSVC().fit(X, y)
    sk_model = sklearn.svm.LinearSVC().fit(X, y)

    cu_model2 = cuml.LinearSVC.from_sklearn(sk_model)
    sk_model2 = cu_model.as_sklearn()

    cu_score = cu_model2.score(X, y)
    assert cu_score > 0.7

    sk_score = sk_model2.score(X, y)
    assert sk_score > 0.7


def test_target_encoder(random_state):
    # Create simple categorical data
    X = np.array(
        [
            ["cat"],
            ["dog"],
            ["cat"],
            ["dog"],
            ["bird"],
            ["cat"],
            ["dog"],
            ["bird"],
        ]
    )
    y = np.array([1.0, 2.0, 1.5, 2.5, 3.0, 1.2, 2.2, 3.2])

    original = TargetEncoder(
        multi_feature_mode="independent",
        n_folds=2,
        smooth=1.0,
        split_method="continuous",
    )
    original.fit(X, y)

    sklearn_model = original.as_sklearn()
    roundtrip_model = TargetEncoder.from_sklearn(sklearn_model)

    assert_roundtrip_consistency(original, roundtrip_model)

    # Use separate test data (not training data) because cuML returns
    # CV-based encodings for training data while sklearn returns global encodings
    X_test = np.array([["cat"], ["bird"], ["dog"], ["cat"]])
    original_output = original.transform(X_test)
    sklearn_output = sklearn_model.transform(X_test)
    roundtrip_output = roundtrip_model.transform(X_test)

    assert array_equal(original_output, sklearn_output)
    assert array_equal(original_output, roundtrip_output)


def test_label_encoder():
    y = np.array(["a", "b", "b", "a"])
    cu_model = cuml.preprocessing.LabelEncoder().fit(y)
    sk_model = sklearn.preprocessing.LabelEncoder().fit(y)

    cu_model2 = cuml.preprocessing.LabelEncoder.from_sklearn(sk_model)
    sk_model2 = cu_model.as_sklearn()

    roundtrip = cuml.preprocessing.LabelEncoder.from_sklearn(sk_model2)
    assert_roundtrip_consistency(cu_model, roundtrip)

    np.testing.assert_array_equal(cu_model.classes_, sk_model2.classes_)
    np.testing.assert_array_equal(cu_model.classes_, roundtrip.classes_)

    sol = np.array([0, 1, 1, 0])
    cu_out = cu_model2.transform(y)
    sk_out = sk_model2.transform(y)

    np.testing.assert_array_equal(cu_out, sol)
    np.testing.assert_array_equal(sk_out, sol)


def test_label_binarizer():
    y = np.array(["a", "b", "c", "a"])
    cu_model = cuml.preprocessing.LabelBinarizer().fit(y)
    sk_model = sklearn.preprocessing.LabelBinarizer().fit(y)

    cu_model2 = cuml.preprocessing.LabelBinarizer.from_sklearn(sk_model)
    sk_model2 = cu_model.as_sklearn()

    roundtrip = cuml.preprocessing.LabelBinarizer.from_sklearn(sk_model2)
    assert_roundtrip_consistency(cu_model, roundtrip)

    np.testing.assert_array_equal(cu_model.classes_, sk_model2.classes_)
    np.testing.assert_array_equal(cu_model.classes_, roundtrip.classes_)
    assert roundtrip.y_type_ == cu_model.y_type_
    assert roundtrip.sparse_input_ == cu_model.sparse_input_

    sol = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    cu_out = cu_model2.transform(y)
    sk_out = sk_model2.transform(y)

    np.testing.assert_array_equal(cu_out, sol)
    np.testing.assert_array_equal(sk_out, sol)
