# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import sklearn
from packaging.version import Version
from sklearn.datasets import make_blobs
from sklearn.preprocessing import (
    LabelBinarizer,
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    PolynomialFeatures,
    StandardScaler,
)

SKLEARN_161 = Version(sklearn.__version__) >= Version("1.6.1")


def test_standard_scaler():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    model = StandardScaler().fit(X)

    assert model.mean_.shape == (X.shape[1],)
    assert model.var_.shape == (X.shape[1],)
    assert model.scale_.shape == (X.shape[1],)

    # Transform and check shape
    X_transformed = model.transform(X)
    assert X_transformed.shape == X.shape

    # Check that transformed data has ~0 mean and ~1 std
    np.testing.assert_allclose(X_transformed.mean(axis=0), 0, atol=1e-7)
    np.testing.assert_allclose(X_transformed.std(axis=0), 1, atol=1e-7)

    # Check inverse transform
    X_inverse = model.inverse_transform(X_transformed)
    assert X_inverse.shape == X.shape
    np.testing.assert_allclose(X_inverse, X, atol=1e-6)


def test_standard_scaler_sparse_with_mean():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    X[X < 0] = 0
    X = sp.csr_matrix(X)

    tags = StandardScaler(with_mean=True).__sklearn_tags__()
    assert not tags.input_tags.sparse

    with pytest.raises(ValueError, match="Cannot center sparse matrices"):
        StandardScaler(with_mean=True).fit(X)

    tags = StandardScaler(with_mean=False).__sklearn_tags__()
    # scikit-learn/scikit-learn#30187 fixed stale sparse input tags after
    # 1.6.0. Drop this version guard once cuML requires scikit-learn >= 1.6.1.
    assert tags.input_tags.sparse is SKLEARN_161

    model = StandardScaler(with_mean=False).fit(X)
    out = model.transform(X)
    assert sp.issparse(out)
    assert out.shape == X.shape


def test_min_max_scaler():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    model = MinMaxScaler().fit(X)

    assert model.min_.shape == (X.shape[1],)
    assert model.scale_.shape == (X.shape[1],)
    assert model.data_min_.shape == (X.shape[1],)
    assert model.data_max_.shape == (X.shape[1],)
    assert model.data_range_.shape == (X.shape[1],)

    # Transform and check shape
    X_transformed = model.transform(X)
    assert X_transformed.shape == X.shape

    # Check that transformed data is scaled appropriately
    assert (X_transformed.min(axis=0) >= 0).all()
    assert (X_transformed.max(axis=0) <= 1).all()

    # Check inverse transform
    X_inverse = model.inverse_transform(X_transformed)
    assert X_inverse.shape == X.shape
    np.testing.assert_allclose(X_inverse, X, atol=1e-6)


def test_max_abs_scaler():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    model = MaxAbsScaler().fit(X)

    assert model.scale_.shape == (X.shape[1],)
    assert model.max_abs_.shape == (X.shape[1],)

    # Transform and check shape
    X_transformed = model.transform(X)
    assert X_transformed.shape == X.shape

    # Check that transformed data is scaled appropriately
    assert (np.abs(X_transformed) <= 1).all()

    # Check inverse transform
    X_inverse = model.inverse_transform(X_transformed)
    assert X_inverse.shape == X.shape
    np.testing.assert_allclose(X_inverse, X, atol=1e-6)


@pytest.mark.parametrize("cls", [StandardScaler, MinMaxScaler, MaxAbsScaler])
def test_scaler_partial_fit(cls):
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)

    model = cls().fit(X)

    model2 = cls()
    model2.partial_fit(X[:25])
    assert model2.n_samples_seen_ == 25
    model2.partial_fit(X[25:])
    assert model2.n_samples_seen_ == X.shape[0]

    sol = model.transform(X)
    res = model2.transform(X)
    np.testing.assert_allclose(sol, res)


def test_polynomial_features():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)

    model = PolynomialFeatures().fit(X)
    assert isinstance(model.powers_, np.ndarray)
    out = model.transform(X)
    assert isinstance(out, np.ndarray)

    model.set_output(transform="pandas")
    out_df = model.transform(X)
    assert isinstance(out_df, pd.DataFrame)


@pytest.mark.parametrize("sparse_output", [True, False])
def test_one_hot_encoder(sparse_output):
    X = np.array([["a", 1], ["b", 1], ["a", 2], ["a", 3]], dtype=object)

    model = OneHotEncoder(sparse_output=sparse_output).fit(X)
    np.testing.assert_array_equal(model.categories_[0], ["a", "b"])
    np.testing.assert_array_equal(model.categories_[1], [1, 2, 3])

    sol = np.array(
        [
            [1, 0, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1],
        ]
    )
    Xt = model.transform(X)
    if sparse_output:
        assert sp.issparse(Xt)
        np.testing.assert_array_equal(Xt.toarray(), sol)
    else:
        np.testing.assert_array_equal(Xt, sol)

    inv_X = model.inverse_transform(Xt)
    np.testing.assert_array_equal(X, inv_X)


@pytest.mark.parametrize(
    "kind, dtype",
    [
        ("list", None),
        ("numpy", "O"),
        ("numpy", "S"),
        ("pandas", "O"),
        ("pandas", "S"),
    ],
)
@pytest.mark.parametrize("auto", [True, False])
def test_one_hot_encoder_bytes_inputs(kind, dtype, auto):
    X = [[b"a", b"x"], [b"b", b"x"], [b"a", b"y"]]
    if kind == "numpy":
        X = np.array(X, dtype=dtype)
    elif kind == "pandas":
        X = pd.DataFrame(X).astype(dtype)

    if auto:
        cats = "auto"
    else:
        cats = [[b"a", b"b"], [b"x", b"y"]]
        if kind == "numpy":
            cats = [np.array(c, dtype=dtype) for c in cats]
        elif kind == "pandas":
            cats = [pd.Series(c, dtype=dtype) for c in cats]

    enc = OneHotEncoder(sparse_output=False, categories=cats)
    if not auto and dtype == "O":
        # XXX: exception raised by sklearn, we don't really care what it is
        with pytest.raises(ValueError):
            enc.fit(X)
    else:
        enc.fit(X)
        np.testing.assert_array_equal(enc.categories_[0], [b"a", b"b"])
        np.testing.assert_array_equal(enc.categories_[1], [b"x", b"y"])
        Xt = enc.transform(X)
        sol = np.array([[1, 0, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1]])
        np.testing.assert_array_equal(Xt, sol)


def test_label_encoder():
    y = np.array(["a", "b", "a", "b"])
    enc = LabelEncoder()
    y2 = enc.fit_transform(y)
    np.testing.assert_array_equal(y2, np.array([0, 1, 0, 1]))
    np.testing.assert_array_equal(enc.classes_, np.array(["a", "b"]))
    y3 = enc.inverse_transform(y2)
    np.testing.assert_array_equal(y3, y)


def test_label_binarizer():
    y = np.array(["a", "b", "a", "c"])
    enc = LabelBinarizer()

    y2 = enc.fit_transform(y)
    sol = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]])
    np.testing.assert_array_equal(y2, sol)

    np.testing.assert_array_equal(enc.classes_, np.array(["a", "b", "c"]))
    assert enc.classes_.dtype == y.dtype
    assert enc.y_type_ == "multiclass"

    y3 = enc.transform(np.array(["a", "d"]))
    sol = np.array([[1, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(y3, sol)

    y4 = enc.inverse_transform(y2)
    np.testing.assert_array_equal(y4, y)


@pytest.mark.parametrize("sparse", [True, False])
def test_label_binarizer_multilabel_indicator(sparse):
    y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if sparse:
        y = sp.csr_matrix(y)
    enc = LabelBinarizer().fit(y)
    np.testing.assert_array_equal(enc.classes_, np.array([0, 1, 2]))
    assert enc.y_type_ == "multilabel-indicator"

    y2 = enc.inverse_transform(y)
    if sparse:
        assert isinstance(y2, sp.csr_matrix)
        np.testing.assert_array_equal(y.toarray(), y2.toarray())
    else:
        assert isinstance(y2, np.ndarray)
        np.testing.assert_array_equal(y, y2)
