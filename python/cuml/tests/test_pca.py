# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
import cupyx
import numpy as np
import pytest
import scipy.sparse
from sklearn import datasets
from sklearn.datasets import make_blobs, make_multilabel_classification
from sklearn.decomposition import PCA as skPCA
from sklearn.exceptions import NotFittedError

from cuml import PCA as cuPCA
from cuml.testing.utils import (
    array_equal,
    quality_param,
    stress_param,
    unit_param,
)


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("input_type", ["ndarray"])
@pytest.mark.parametrize("sparse", [True, False])
def test_pca_fit(datatype, input_type, sparse):
    if sparse:
        X = 100 * scipy.sparse.random(
            200,
            100,
            density=0.1,
            dtype=cp.float32,
            random_state=10,
        )
    else:
        X, _ = make_multilabel_classification(
            n_samples=500,
            n_classes=2,
            n_labels=1,
            allow_unlabeled=False,
            random_state=1,
        )

    sol = skPCA(n_components=2).fit(X)
    res = cuPCA(n_components=2).fit(X)

    np.testing.assert_allclose(
        res.singular_values_, sol.singular_values_, atol=1e-3
    )
    np.testing.assert_allclose(res.components_, sol.components_, atol=1e-3)
    np.testing.assert_allclose(
        res.explained_variance_, sol.explained_variance_, atol=1e-3
    )
    np.testing.assert_allclose(
        res.explained_variance_ratio_, sol.explained_variance_ratio_, atol=1e-3
    )
    np.testing.assert_allclose(res.mean_, sol.mean_, atol=1e-3)
    np.testing.assert_allclose(
        res.noise_variance_, sol.noise_variance_, atol=1e-3
    )
    assert isinstance(res.noise_variance_, float)


@pytest.mark.parametrize("n_samples", [200])
@pytest.mark.parametrize("n_features", [100, 300])
@pytest.mark.parametrize("sparse", [True, False])
def test_pca_defaults(n_samples, n_features, sparse):
    # FIXME: Disable the case True-300-200 due to flaky test
    if sparse and n_features == 300 and n_samples == 200:
        pytest.xfail("Skipping the case True-300-200 due to flaky test")

    if sparse:
        X = cupyx.scipy.sparse.random(
            n_samples,
            n_features,
            density=0.03,
            dtype=cp.float32,
            random_state=10,
        )
    else:
        X, Y = make_multilabel_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=2,
            n_labels=1,
            random_state=1,
        )
    cupca = cuPCA()
    cupca.fit(X)
    curesult = cupca.transform(X)

    if sparse:
        X = X.toarray().get()
    skpca = skPCA()
    skpca.fit(X)
    skresult = skpca.transform(X)

    assert skpca.svd_solver == cupca.svd_solver
    assert cupca.components_.shape[0] == skpca.components_.shape[0]
    assert curesult.shape == skresult.shape
    assert array_equal(curesult, skresult, 1e-3, with_sign=True)


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("input_type", ["ndarray"])
@pytest.mark.parametrize(
    "name", [unit_param(None), quality_param("iris"), stress_param("blobs")]
)
def test_pca_fit_then_transform(datatype, input_type, name):
    # Assume at least 4GB memory
    max_gpu_memory = pytest.max_gpu_memory or 4

    blobs_n_samples = 500000
    if name == "blobs" and max_gpu_memory < 32:
        if pytest.adapt_stress_test:
            blobs_n_samples = int(blobs_n_samples * max_gpu_memory / 32)
        else:
            pytest.skip(
                "Insufficient GPU memory for this test."
                "Re-run with 'CUML_ADAPT_STRESS_TESTS=True'"
            )

    if name == "blobs":
        X, y = make_blobs(
            n_samples=blobs_n_samples, n_features=1000, random_state=0
        )

    elif name == "iris":
        iris = datasets.load_iris()
        X = iris.data

    else:
        X, Y = make_multilabel_classification(
            n_samples=500,
            n_classes=2,
            n_labels=1,
            allow_unlabeled=False,
            random_state=1,
        )

    if name != "blobs":
        skpca = skPCA(n_components=2)
        skpca.fit(X)
        Xskpca = skpca.transform(X)

    cupca = cuPCA(n_components=2)

    cupca.fit(X)
    X_cupca = cupca.transform(X)

    if name != "blobs":
        assert array_equal(X_cupca, Xskpca, 1e-3, with_sign=True)
        assert Xskpca.shape[0] == X_cupca.shape[0]
        assert Xskpca.shape[1] == X_cupca.shape[1]


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("input_type", ["ndarray"])
@pytest.mark.parametrize(
    "name", [unit_param(None), quality_param("iris"), stress_param("blobs")]
)
def test_pca_fit_transform(datatype, input_type, name):
    # Assume at least 4GB memory
    max_gpu_memory = pytest.max_gpu_memory or 4

    blobs_n_samples = 500000

    if name == "blobs" and max_gpu_memory < 32:
        if pytest.adapt_stress_test:
            blobs_n_samples = int(blobs_n_samples * max_gpu_memory / 32)
        else:
            pytest.skip(
                "Insufficient GPU memory for this test."
                "Re-run with 'CUML_ADAPT_STRESS_TESTS=True'"
            )

    if name == "blobs":
        X, y = make_blobs(
            n_samples=blobs_n_samples, n_features=1000, random_state=0
        )

    elif name == "iris":
        iris = datasets.load_iris()
        X = iris.data

    else:
        X, Y = make_multilabel_classification(
            n_samples=500,
            n_classes=2,
            n_labels=1,
            allow_unlabeled=False,
            random_state=1,
        )

    if name != "blobs":
        skpca = skPCA(n_components=2)
        Xskpca = skpca.fit_transform(X)

    cupca = cuPCA(n_components=2)

    X_cupca = cupca.fit_transform(X)

    if name != "blobs":
        assert array_equal(X_cupca, Xskpca, 1e-3, with_sign=True)
        assert Xskpca.shape[0] == X_cupca.shape[0]
        assert Xskpca.shape[1] == X_cupca.shape[1]


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("input_type", ["ndarray"])
@pytest.mark.parametrize(
    "name", [unit_param(None), quality_param("quality"), stress_param("blobs")]
)
@pytest.mark.parametrize("nrows", [unit_param(500), quality_param(5000)])
def test_pca_inverse_transform(datatype, input_type, name, nrows):
    if name == "blobs":
        pytest.skip("fails when using blobs dataset")
        X, y = make_blobs(n_samples=500000, n_features=1000, random_state=0)

    else:
        rng = np.random.RandomState(0)
        n, p = nrows, 3
        X = rng.randn(n, p)  # spherical data
        X[:, 1] *= 0.00001  # make middle component relatively small
        X += [3, 4, 2]  # make a large mean

    cupca = cuPCA(n_components=2)

    X_cupca = cupca.fit_transform(X)

    input_gdf = cupca.inverse_transform(X_cupca)
    assert array_equal(input_gdf, X, 5e-5, with_sign=True)


@pytest.mark.parametrize("nrows", [4000, 7000])
@pytest.mark.parametrize("ncols", [2500, stress_param(20000)])
@pytest.mark.parametrize("whiten", [True, False])
@pytest.mark.parametrize("return_sparse", [True, False])
@pytest.mark.parametrize("cupy_input", [True, False])
def test_sparse_pca_inputs(nrows, ncols, whiten, return_sparse, cupy_input):
    # Assume at least 4GB memory
    max_gpu_memory = pytest.max_gpu_memory or 4

    if ncols == 20000 and max_gpu_memory < 48:
        if pytest.adapt_stress_test:
            ncols = int(ncols * max_gpu_memory / 48)
        else:
            pytest.skip(
                "Insufficient GPU memory for this test."
                "Re-run with 'CUML_ADAPT_STRESS_TESTS=True'"
            )

    if return_sparse:
        pytest.skip("Loss of information in converting to cupy sparse csr")

    X = cupyx.scipy.sparse.random(
        nrows, ncols, density=0.07, dtype=cp.float32, random_state=10
    )
    if not (cupy_input):
        X = X.get()

    p_sparse = cuPCA(n_components=ncols, whiten=whiten)

    p_sparse.fit(X)
    t_sparse = p_sparse.transform(X)
    i_sparse = p_sparse.inverse_transform(
        t_sparse, return_sparse=return_sparse
    )

    if return_sparse:
        assert isinstance(i_sparse, cupyx.scipy.sparse.csr_matrix)

        assert array_equal(
            i_sparse.todense(), X.todense(), 1e-1, with_sign=True
        )
    else:
        if cupy_input:
            assert isinstance(i_sparse, cp.ndarray)

        assert array_equal(i_sparse, X.todense(), 1e-1, with_sign=True)


@pytest.mark.parametrize(
    "n_samples, n_features",
    [
        pytest.param(9, 20, id="n_samples <= n_components"),
        pytest.param(20, 10, id="n_features <= n_components"),
    ],
)
def test_noise_variance_zero(n_samples, n_features):
    X, _ = make_blobs(
        n_samples=n_samples, n_features=n_features, random_state=0
    )
    cupca = cuPCA(n_components=10)
    cupca.fit(X)
    assert cupca.noise_variance_ == 0


def test_exceptions():
    with pytest.raises(NotFittedError):
        X = cp.random.random((10, 10))
        cuPCA().transform(X)

    with pytest.raises(NotFittedError):
        X = cp.random.random((10, 10))
        cuPCA().inverse_transform(X)


def test_get_feature_names_out():
    X, _ = make_blobs(n_features=5)
    cu_model = cuPCA(n_components=2).fit(X)
    sk_model = skPCA(n_components=2).fit(X)
    res = cu_model.get_feature_names_out()
    sol = sk_model.get_feature_names_out()
    np.testing.assert_array_equal(res, sol)
