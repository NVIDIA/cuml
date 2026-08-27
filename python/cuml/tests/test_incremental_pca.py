#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import cupyx
import numpy as np
import pytest
from sklearn.decomposition import IncrementalPCA as skIPCA
from sklearn.exceptions import NotFittedError

from cuml.datasets import make_blobs
from cuml.decomposition import IncrementalPCA as cuIPCA
from cuml.decomposition.incremental_pca import _svd_flip
from cuml.testing.utils import array_equal


@pytest.mark.parametrize(
    "nrows, ncols, n_components, sparse_input, density, sparse_format,"
    " batch_size_divider, whiten",
    [
        (500, 15, 2, True, 0.4, "csr", 5, True),
        (5000, 25, 12, False, 0.07, "csc", 10, False),
        (5000, 15, None, True, 0.4, "csc", 5, False),
        (500, 25, 2, False, 0.07, "csr", 10, False),
        (5000, 25, 12, False, 0.07, "csr", 10, True),
        (500, 2500, 9, False, 0.07, "csr", 50, True),
        (500, 250, 14, True, 0.07, "csr", 1, True),
    ],
)
def test_fit(
    nrows,
    ncols,
    n_components,
    sparse_input,
    density,
    sparse_format,
    batch_size_divider,
    whiten,
):
    if sparse_input:
        X = cupyx.scipy.sparse.random(
            nrows,
            ncols,
            density=density,
            random_state=10,
            format=sparse_format,
        )
    else:
        X, _ = make_blobs(n_samples=nrows, n_features=ncols, random_state=10)

    cu_ipca = cuIPCA(
        n_components=n_components,
        whiten=whiten,
        batch_size=int(nrows / batch_size_divider),
    )
    cu_ipca.fit(X)
    cu_t = cu_ipca.transform(X)
    cu_inv = cu_ipca.inverse_transform(cu_t)

    sk_ipca = skIPCA(
        n_components=n_components,
        whiten=whiten,
        batch_size=int(nrows / batch_size_divider),
    )
    if sparse_input:
        X = X.get()
    else:
        X = cp.asnumpy(X)
    sk_ipca.fit(X)
    sk_t = sk_ipca.transform(X)
    sk_inv = sk_ipca.inverse_transform(sk_t)

    assert array_equal(cu_inv, sk_inv, 1e-3, with_sign=True)


@pytest.mark.parametrize(
    "nrows, ncols, n_components, density, batch_size_divider, whiten",
    [
        (500, 15, 2, 0.07, 5, False),
        (500, 15, 2, 0.07, 5, True),
        (5000, 25, 12, 0.07, 10, False),
        (5000, 15, 2, 0.4, 5, True),
        (500, 25, 12, 0.4, 10, False),
        (5000, 4, 2, 0.1, 100, False),
    ],
)
def test_partial_fit(
    nrows, ncols, n_components, density, batch_size_divider, whiten
):
    X, _ = make_blobs(n_samples=nrows, n_features=ncols, random_state=10)

    cu_ipca = cuIPCA(n_components=n_components, whiten=whiten)

    sample_size = int(nrows / batch_size_divider)
    for i in range(0, nrows, sample_size):
        cu_ipca.partial_fit(X[i : i + sample_size].copy())

    cu_t = cu_ipca.transform(X)
    cu_inv = cu_ipca.inverse_transform(cu_t)

    sk_ipca = skIPCA(n_components=n_components, whiten=whiten)

    X = cp.asnumpy(X)

    for i in range(0, nrows, sample_size):
        sk_ipca.partial_fit(X[i : i + sample_size].copy())

    sk_t = sk_ipca.transform(X)
    sk_inv = sk_ipca.inverse_transform(sk_t)

    assert array_equal(cu_inv, sk_inv, 1e-3, with_sign=True)


@pytest.mark.parametrize("whiten", [False, True])
@pytest.mark.parametrize("method", ["fit", "partial_fit"])
def test_sklearn_parity_attrs(method, whiten):
    """Check fitted-attribute parity with sklearn for fit() and partial_fit()."""
    nrows, ncols, n_components = 1000, 8, 4
    X, _ = make_blobs(
        n_samples=nrows, n_features=ncols, random_state=0, dtype="float64"
    )
    X_np = cp.asnumpy(X)

    cu_ipca = cuIPCA(n_components=n_components, whiten=whiten, batch_size=100)
    sk_ipca = skIPCA(n_components=n_components, whiten=whiten, batch_size=100)

    if method == "fit":
        cu_ipca.fit(X)
        sk_ipca.fit(X_np)
    else:
        batch = 100
        for i in range(0, nrows, batch):
            cu_ipca.partial_fit(X[i : i + batch])
            sk_ipca.partial_fit(X_np[i : i + batch])

    assert array_equal(
        cu_ipca.components_, sk_ipca.components_, 1e-3, with_sign=True
    )
    assert array_equal(
        cu_ipca.explained_variance_,
        sk_ipca.explained_variance_,
        1e-3,
        with_sign=True,
    )
    assert array_equal(
        cu_ipca.explained_variance_ratio_,
        sk_ipca.explained_variance_ratio_,
        1e-3,
        with_sign=True,
    )
    assert array_equal(
        cu_ipca.singular_values_,
        sk_ipca.singular_values_,
        1e-3,
        with_sign=True,
    )
    assert array_equal(cu_ipca.mean_, sk_ipca.mean_, 1e-3, with_sign=True)
    assert array_equal(cu_ipca.var_, sk_ipca.var_, 1e-3, with_sign=True)
    assert int(cu_ipca.n_samples_seen_) == int(sk_ipca.n_samples_seen_)
    assert cu_ipca.n_components_ == sk_ipca.n_components_


def test_exceptions():
    X = cupyx.scipy.sparse.eye(10)
    ipca = cuIPCA()
    with pytest.raises(TypeError):
        ipca.partial_fit(X)

    X = X.toarray()
    ipca = cuIPCA()
    with pytest.raises(NotFittedError):
        ipca.transform(X)

    with pytest.raises(NotFittedError):
        ipca.inverse_transform(X)

    with pytest.raises(ValueError):
        cuIPCA(n_components=8).fit(X[:5])

    with pytest.raises(ValueError):
        cuIPCA(n_components=8).fit(X[:, :5])


@pytest.mark.parametrize("batch_size", [None, 50])
def test_partial_fit_then_sparse_transform(batch_size):
    X_dense, _ = make_blobs(
        n_samples=200, n_features=10, random_state=0, dtype="float64"
    )
    X_sparse = cupyx.scipy.sparse.csr_matrix(X_dense)

    ipca = cuIPCA(n_components=4, batch_size=batch_size)
    for i in range(0, 200, 50):
        ipca.partial_fit(X_dense[i : i + 50])

    result = ipca.transform(X_sparse)
    assert result.shape == (200, 4)


def test_svd_flip():
    x = cp.array(range(-10, 80)).reshape((9, 10))
    u, s, v = cp.linalg.svd(x, full_matrices=False)
    u_true, v_true = _svd_flip(u, v, flip_signs_based_on_U=True)
    reco_true = cp.dot(u_true * s, v_true)
    u_false, v_false = _svd_flip(u, v, flip_signs_based_on_U=False)
    reco_false = cp.dot(u_false * s, v_false)

    assert array_equal(reco_true, x)
    assert array_equal(reco_false, x)


def test_get_feature_names_out():
    X, _ = make_blobs(n_features=5)
    cu_model = cuIPCA(n_components=2).fit(X)
    sk_model = skIPCA(n_components=2).fit(X.get())
    res = cu_model.get_feature_names_out()
    sol = sk_model.get_feature_names_out()
    np.testing.assert_array_equal(res, sol)
