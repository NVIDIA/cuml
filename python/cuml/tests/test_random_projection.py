# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import cupy as cp
import cupyx.scipy.sparse as cp_sp
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import sklearn.random_projection
from scipy.spatial.distance import pdist

from cuml.random_projection import (
    GaussianRandomProjection,
    SparseRandomProjection,
)

classes = [GaussianRandomProjection, SparseRandomProjection]


def random_array(m, n, dtype="float32", sparse=False, random_state=42):
    rng = np.random.default_rng(random_state)
    if sparse:
        return sp.random(m, n, format="csr", dtype=dtype, rng=rng)
    return rng.random((m, n), dtype=dtype)


def asdense(X):
    return X.toarray() if sp.issparse(X) else X


@pytest.mark.parametrize("cls", classes)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sparse", [True, False])
def test_random_projection(cls, dtype, sparse):
    model = cls(eps=0.2, random_state=42)
    X = random_array(800, 3000, dtype=dtype, sparse=sparse)
    X2 = model.fit_transform(X)
    if sparse and cls is SparseRandomProjection:
        assert sp.issparse(X2)
    else:
        assert isinstance(X2, np.ndarray)

    original_pdist = pdist(asdense(X))
    embedded_pdist = pdist(asdense(X2))

    # check JL lemma
    assert np.all(((1.0 - model.eps) * original_pdist) <= embedded_pdist)
    assert np.all(embedded_pdist <= ((1.0 + model.eps) * original_pdist))


def test_gaussian_random_matrix():
    """Test statistical properties of gaussian random matrix"""
    n_components = 100
    n_features = 1000
    X = random_array(100, n_features, random_state=42)
    model = GaussianRandomProjection(
        n_components=n_components, random_state=42
    ).fit(X)
    A = model.components_

    np.testing.assert_array_almost_equal(0.0, np.mean(A), 2)
    np.testing.assert_array_almost_equal(
        np.var(A, ddof=1), 1 / n_components, 1
    )


@pytest.mark.parametrize("density", [0.3, 1.0])
def test_sparse_random_matrix(density):
    """Test statistical properties of sparse random matrix"""
    n_components = 100
    n_features = 500
    s = 1 / density

    X = random_array(100, n_features, random_state=42)
    model = SparseRandomProjection(
        n_components=n_components,
        density=density,
        random_state=42,
    ).fit(X)

    A = asdense(model.components_)

    values = np.unique(A)
    expected = np.sqrt(s) / np.sqrt(n_components)
    if density == 1.0:
        np.testing.assert_allclose(values, [-expected, expected])
    else:
        np.testing.assert_allclose(values, [-expected, 0, expected])

    np.testing.assert_almost_equal(np.mean(A == 0.0), 1 - 1 / s, decimal=2)
    np.testing.assert_almost_equal(np.mean(A > 0), 1 / (2 * s), decimal=2)
    np.testing.assert_almost_equal(np.mean(A < 0), 1 / (2 * s), decimal=2)
    np.testing.assert_almost_equal(
        np.var(A == 0.0, ddof=1), (1 - 1 / s) * 1 / s, decimal=2
    )
    np.testing.assert_almost_equal(
        np.var(A > 0, ddof=1), (1 - 1 / (2 * s)) * 1 / (2 * s), decimal=2
    )
    np.testing.assert_almost_equal(
        np.var(A < 0, ddof=1), (1 - 1 / (2 * s)) * 1 / (2 * s), decimal=2
    )


@pytest.mark.parametrize("cls", classes)
def test_n_components_auto(cls):
    X = random_array(10, 1000)
    model = cls(n_components="auto", random_state=42, eps=0.5).fit(X)

    assert model.n_components_ == 110
    assert model.components_.shape == (110, 1000)
    if cls is SparseRandomProjection:
        np.testing.assert_almost_equal(model.density_, 0.03, 2)


@pytest.mark.parametrize("cls", classes)
def test_n_components_auto_too_large_target_dimension_error(cls):
    X = random_array(1000, 100)
    model = cls(n_components="auto", eps=0.1)
    expected_msg = (
        "eps=0.1 and n_samples=1000 lead to a target dimension"
        " of 5920 which is larger than the original space with"
        " n_features=100"
    )
    with pytest.raises(ValueError, match=expected_msg):
        model.fit(X)


def test_sparse_output_representation():
    dense = random_array(10, 1000)
    sparse = random_array(10, 1000, sparse=True)

    # Outputs are sparse for sparse inputs, dense for dense inputs
    model = SparseRandomProjection(n_components=10, random_state=42).fit(dense)
    assert sp.issparse(model.transform(sparse))
    assert isinstance(model.transform(dense), np.ndarray)

    # dense_output forces all outputs to be dense
    model = SparseRandomProjection(
        n_components=10, dense_output=True, random_state=42
    )
    model.fit(dense)
    assert sp.issparse(model.components_)
    assert isinstance(model.transform(dense), np.ndarray)
    assert isinstance(model.transform(sparse), np.ndarray)


@pytest.mark.parametrize("cls", classes)
@pytest.mark.parametrize("sparse", [False, True])
def test_random_seed_consistency(cls, sparse):
    X = random_array(10, 1000, sparse=sparse)

    model1 = cls(n_components=5, random_state=42).fit(X)
    t1 = model1.transform(X)
    model2 = cls(n_components=5, random_state=42).fit(X)
    t2 = model2.transform(X)
    np.testing.assert_array_equal(
        asdense(model1.components_), asdense(model2.components_)
    )
    # Due to https://github.com/cupy/cupy/issues/9323 only sparse @ sparse or
    # dense @ dense outputs are exactly reproducible. All other combinations
    # result in close but not identical outputs. For now we document this and
    # relax the test constraint.
    if (cls is SparseRandomProjection) != sparse:
        # Mix of sparse and dense, check outputs are close
        np.testing.assert_allclose(asdense(t1), asdense(t2), rtol=1e-3)
    else:
        # Both dense or sparse, can check exactly
        np.testing.assert_array_equal(asdense(t1), asdense(t2))


@pytest.mark.parametrize("cls", classes)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_components_and_output_dtype(cls, dtype):
    X = random_array(10, 100, dtype=dtype)
    model = cls(random_state=42, n_components=5)
    transformed = model.fit_transform(X)
    assert model.components_.dtype == dtype
    assert transformed.dtype == dtype


@pytest.mark.parametrize("cls", classes)
def test_output_type_dense_inputs(cls):
    X_np = random_array(10, 100)
    X_cp = cp.asarray(X_np)
    X_pd = pd.DataFrame(
        X_np,
        columns=[f"x{i}" for i in range(100)],
        index=np.arange(10, 30, 2),
    )

    out = cls(random_state=42, n_components=5).fit_transform(X_np)
    assert isinstance(out, np.ndarray)

    out = cls(random_state=42, n_components=5).fit_transform(X_cp)
    assert isinstance(out, cp.ndarray)

    out = cls(random_state=42, n_components=5).fit_transform(X_pd)
    assert isinstance(out, pd.DataFrame)
    assert (out.index == X_pd.index).all()


@pytest.mark.parametrize("cls", classes)
def test_output_type_sparse_inputs(cls):
    X_cpu = random_array(10, 100, sparse=True)
    X_gpu = cp_sp.csr_matrix(X_cpu)

    model = cls(random_state=42, n_components=5)
    out = model.fit_transform(X_cpu)
    if cls is SparseRandomProjection:
        assert sp.issparse(out)
        assert sp.issparse(model.components_)
    else:
        assert isinstance(out, np.ndarray)
        assert isinstance(model.components_, np.ndarray)

    model = cls(random_state=42, n_components=5)
    out = model.fit_transform(X_gpu)
    if cls is SparseRandomProjection:
        assert cp_sp.issparse(out)
        assert cp_sp.issparse(model.components_)
    else:
        assert isinstance(out, cp.ndarray)
        assert isinstance(model.components_, cp.ndarray)


@pytest.mark.parametrize(
    "cu_cls, sk_cls",
    [
        (
            GaussianRandomProjection,
            sklearn.random_projection.GaussianRandomProjection,
        ),
        (
            SparseRandomProjection,
            sklearn.random_projection.SparseRandomProjection,
        ),
    ],
)
def test_get_feature_names_out(cu_cls, sk_cls):
    X = random_array(10, 100)
    cu_model = cu_cls(n_components=5, random_state=42).fit(X)
    sk_model = sk_cls(n_components=5, random_state=42).fit(X)
    res = cu_model.get_feature_names_out()
    sol = sk_model.get_feature_names_out()
    np.testing.assert_array_equal(res, sol)
