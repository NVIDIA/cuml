#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import math

import cupy as cp
import cupyx.scipy.sparse as cp_sp
import numpy as np
import pytest
from numpy.testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
    assert_raises,
)
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB as skBNB
from sklearn.naive_bayes import CategoricalNB as skCNB
from sklearn.naive_bayes import ComplementNB as skComplementNB
from sklearn.naive_bayes import GaussianNB as skGNB
from sklearn.naive_bayes import MultinomialNB as skNB

from cuml.datasets import make_classification
from cuml.naive_bayes import (
    BernoulliNB,
    CategoricalNB,
    ComplementNB,
    GaussianNB,
    MultinomialNB,
)


@pytest.mark.parametrize("x_dtype", [cp.int32, cp.int64])
@pytest.mark.parametrize("y_dtype", [cp.int32, cp.int64])
def test_sparse_integral_dtypes_work(x_dtype, y_dtype, sparse_text_dataset):
    """These require a bit of special casing since cupyx doesn't support
    integral spasre matrices"""
    X, y = sparse_text_dataset

    X = X.astype(x_dtype)
    y = y.astype(y_dtype)

    model = MultinomialNB().fit(X, y)
    out = model.predict(X)
    assert out.dtype == y.dtype


@pytest.mark.parametrize("x_dtype", [cp.float32, cp.float64, cp.int32])
@pytest.mark.parametrize("y_dtype", [cp.int32, cp.int64])
def test_multinomial_basic_fit_predict_dense_numpy(
    x_dtype, y_dtype, sparse_text_dataset
):
    """
    Cupy Test
    """
    X, y = sparse_text_dataset
    n_rows = 500
    n_cols = 10000

    X = cp_sp.csr_matrix(X, dtype="float32")[:n_rows, :n_cols]
    y = y[:n_rows].astype(y_dtype)

    model = MultinomialNB()
    model.fit(np.ascontiguousarray(cp.asnumpy(X.todense()).astype(x_dtype)), y)

    y_hat = model.predict(X).get()

    modelsk = skNB()
    modelsk.fit(X.get(), y.get())
    y_sk = model.predict(X.get())

    assert_allclose(y_hat, y_sk)


@pytest.mark.parametrize("x_dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("y_dtype", [cp.int32, cp.float32, cp.float64])
def test_multinomial_partial_fit(x_dtype, y_dtype, sparse_text_dataset):
    chunk_size = 500

    X, y = sparse_text_dataset

    X = cp_sp.coo_matrix(X, dtype=x_dtype)
    y = y.astype(y_dtype)

    X = X.tocsr()

    model = MultinomialNB()

    classes = np.unique(y)

    total_fit = 0

    for i in range(math.ceil(X.shape[0] / chunk_size)):
        upper = i * chunk_size + chunk_size
        if upper > X.shape[0]:
            upper = -1

        if upper > 0:
            x = X[i * chunk_size : upper]
            y_c = y[i * chunk_size : upper]
        else:
            x = X[i * chunk_size :]
            y_c = y[i * chunk_size :]

        model.partial_fit(x, y_c, classes=classes)

        total_fit += upper - (i * chunk_size)

        if upper == -1:
            break

    y_hat = model.predict(X)

    y_hat = cp.asnumpy(y_hat)
    y = cp.asnumpy(y)

    assert accuracy_score(y, y_hat) >= 0.924


@pytest.mark.parametrize(
    "model_cls,sk_cls",
    [
        (MultinomialNB, skNB),
        (BernoulliNB, skBNB),
        (ComplementNB, skComplementNB),
        (GaussianNB, skGNB),
        (CategoricalNB, skCNB),
    ],
)
def test_partial_fit_new_class_in_later_batch(model_cls, sk_cls):
    """A partial_fit call after the first must label-encode `y` against
    the already-established `classes_`, not just the unique labels
    present in that batch, or updates silently land on the wrong class
    row. Regression test for
    https://github.com/NVIDIA/cuml/issues/7203
    """
    X1 = cp.array([[0, 1], [1, 0]], dtype=cp.float32)
    y1 = cp.array([0, 1], dtype=cp.int32)
    X_add = cp.array([[1, 1]], dtype=cp.float32)
    y_add = cp.array([2], dtype=cp.int32)
    classes = cp.array([0, 1, 2], dtype=cp.int32)

    model = model_cls()
    model.partial_fit(X1, y1, classes=classes)
    # Second call omits `classes=`; class 2 is unseen until this batch.
    model.partial_fit(X_add, y_add)

    sk_model = sk_cls()
    sk_model.partial_fit(X1.get(), y1.get(), classes=classes.get())
    sk_model.partial_fit(X_add.get(), y_add.get())

    X_test = cp.array([[0, 1], [1, 0], [1, 1]], dtype=cp.float32)
    y_hat = cp.asnumpy(model.predict(X_test))
    y_sk = sk_model.predict(X_test.get())

    assert_array_equal(y_hat, y_sk)


@pytest.mark.parametrize("x_dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("y_dtype", [cp.int32, cp.int64])
def test_multinomial(x_dtype, y_dtype, sparse_text_dataset):
    X, y = sparse_text_dataset

    cu_X = cp_sp.coo_matrix(X, dtype=x_dtype)
    cu_y = y.astype(y_dtype)

    cu_X = cu_X.tocsr()

    y = y.get()

    cuml_model = MultinomialNB()
    sk_model = skNB()

    cuml_model.fit(cu_X, cu_y)
    sk_model.fit(X, y)

    cuml_log_proba = cuml_model.predict_log_proba(cu_X).get()
    sk_log_proba = sk_model.predict_log_proba(X)
    cuml_proba = cuml_model.predict_proba(cu_X).get()
    sk_proba = sk_model.predict_proba(X)
    cuml_score = cuml_model.score(cu_X, cu_y)
    sk_score = sk_model.score(X, y)

    y_hat = cuml_model.predict(cu_X)
    y_hat = cp.asnumpy(y_hat)
    cu_y = cp.asnumpy(cu_y)

    THRES = 1e-4
    assert_allclose(cuml_log_proba, sk_log_proba, atol=1e-2, rtol=1e-2)
    assert_allclose(cuml_proba, sk_proba, atol=1e-6, rtol=1e-2)
    assert sk_score - THRES <= cuml_score <= sk_score + THRES
    assert accuracy_score(y, y_hat) >= 0.924


@pytest.mark.parametrize("x_dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("y_dtype", [cp.int32, cp.int64])
@pytest.mark.parametrize("is_sparse", [True, False])
def test_bernoulli(x_dtype, y_dtype, is_sparse, sparse_text_dataset):
    X, y = sparse_text_dataset
    n_rows = 500
    n_cols = 20000

    X = cp_sp.coo_matrix(X, dtype=x_dtype)
    y = y.astype(y_dtype)

    X = X.tocsr()[:n_rows, :n_cols]
    y = y[:n_rows]
    if not is_sparse:
        X = X.todense()

    sk_model = skBNB()
    cuml_model = BernoulliNB()

    sk_model.fit(X.get(), y.get())
    cuml_model.fit(X, y)

    sk_score = sk_model.score(X.get(), y.get())
    cuml_score = cuml_model.score(X, y)
    cuml_proba = cuml_model.predict_log_proba(X).get()
    sk_proba = sk_model.predict_log_proba(X.get())

    THRES = 1e-3

    assert_array_equal(sk_model.class_count_, cuml_model.class_count_.get())
    assert_allclose(
        sk_model.class_log_prior_, cuml_model.class_log_prior_.get(), 1e-6
    )
    assert_allclose(cuml_proba, sk_proba, atol=1e-2, rtol=1e-2)
    assert sk_score - THRES <= cuml_score <= sk_score + THRES


@pytest.mark.parametrize("x_dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("y_dtype", [cp.int32, cp.float32, cp.float64])
def test_bernoulli_partial_fit(x_dtype, y_dtype, sparse_text_dataset):
    chunk_size = 500
    n_rows = 1500

    X, y = sparse_text_dataset

    X = cp_sp.coo_matrix(X, dtype=x_dtype)
    y = y.astype(y_dtype)[:n_rows]

    X = X.tocsr()[:n_rows]

    model = BernoulliNB()
    modelsk = skBNB()

    classes = np.unique(y)

    for i in range(math.ceil(X.shape[0] / chunk_size)):
        upper = i * chunk_size + chunk_size
        if upper > X.shape[0]:
            upper = -1

        if upper > 0:
            x = X[i * chunk_size : upper]
            y_c = y[i * chunk_size : upper]
        else:
            x = X[i * chunk_size :]
            y_c = y[i * chunk_size :]

        model.partial_fit(x, y_c, classes=classes)
        modelsk.partial_fit(x.get(), y_c.get(), classes=classes.get())
        if upper == -1:
            break

    y_hat = model.predict(X).get()
    y_sk = modelsk.predict(X.get())

    assert_allclose(y_hat, y_sk)


@pytest.mark.parametrize("x_dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("y_dtype", [cp.int32, cp.int64])
@pytest.mark.parametrize("is_sparse", [True, False])
@pytest.mark.parametrize("norm", [True, False])
def test_complement(x_dtype, y_dtype, is_sparse, norm, sparse_text_dataset):
    X, y = sparse_text_dataset
    n_rows = 500
    n_cols = 20000

    X = cp_sp.coo_matrix(X, dtype=x_dtype)
    y = y.astype(y_dtype)

    X = X.tocsr()[:n_rows, :n_cols]
    y = y[:n_rows]
    if not is_sparse:
        X = X.todense()

    sk_model = skComplementNB(norm=norm)
    cuml_model = ComplementNB(norm=norm)

    sk_model.fit(X.get(), y.get())
    cuml_model.fit(X, y)

    sk_score = sk_model.score(X.get(), y.get())
    cuml_score = cuml_model.score(X, y)
    cuml_proba = cuml_model.predict_log_proba(X).get()
    sk_proba = sk_model.predict_log_proba(X.get())

    THRES = 1e-3

    assert_array_equal(sk_model.class_count_, cuml_model.class_count_.get())
    assert_allclose(
        sk_model.class_log_prior_, cuml_model.class_log_prior_.get(), 1e-6
    )
    assert_allclose(cuml_proba, sk_proba, atol=1e-2, rtol=1e-2)
    assert sk_score - THRES <= cuml_score <= sk_score + THRES


@pytest.mark.parametrize("x_dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("y_dtype", [cp.int32, cp.float32, cp.float64])
@pytest.mark.parametrize("norm", [True, False])
def test_complement_partial_fit(x_dtype, y_dtype, norm):
    chunk_size = 500
    n_rows, n_cols = 1500, 100
    weights = [0.6, 0.2, 0.15, 0.05]
    rtol = 1.5e-3 if x_dtype == cp.float32 else 1e-6

    X, y = make_classification(
        n_rows,
        n_cols,
        n_classes=len(weights),
        weights=weights,
        dtype=x_dtype,
        n_informative=9,
        random_state=2,
    )
    X -= X.min(0)  # Make all inputs positive
    y = y.astype(y_dtype)

    model = ComplementNB(norm=norm)
    modelsk = skComplementNB(norm=norm)

    classes = np.unique(y)

    for i in range(math.ceil(X.shape[0] / chunk_size)):
        upper = i * chunk_size + chunk_size
        if upper > X.shape[0]:
            upper = -1

        if upper > 0:
            x = X[i * chunk_size : upper]
            y_c = y[i * chunk_size : upper]
        else:
            x = X[i * chunk_size :]
            y_c = y[i * chunk_size :]

        model.partial_fit(x, y_c, classes=classes)
        modelsk.partial_fit(x.get(), y_c.get(), classes=classes.get())
        if upper == -1:
            break

    y_hat = model.predict_proba(X).get()
    y_sk = modelsk.predict_proba(X.get())

    assert_allclose(y_hat, y_sk, rtol=rtol)


def test_gaussian_basic():
    # Data is just 6 separable points in the plane
    X = cp.array(
        [
            [-2, -1, -1],
            [-1, -1, -1],
            [-1, -2, -1],
            [1, 1, 1],
            [1, 2, 1],
            [2, 1, 1],
        ],
        dtype=cp.float32,
    )
    y = cp.array([1, 1, 1, 2, 2, 2])

    skclf = skGNB()
    skclf.fit(X.get(), y.get())

    clf = GaussianNB()
    clf.fit(X, y)

    assert_array_almost_equal(clf.theta_.get(), skclf.theta_, 6)
    assert_array_almost_equal(clf.sigma_.get(), skclf.var_, 6)

    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)
    y_pred_log_proba = clf.predict_log_proba(X)
    y_pred_proba_sk = skclf.predict_proba(X.get())
    y_pred_log_proba_sk = skclf.predict_log_proba(X.get())

    assert_array_equal(y_pred.get(), y.get())
    assert_array_almost_equal(y_pred_proba.get(), y_pred_proba_sk, 8)
    assert_allclose(
        y_pred_log_proba.get(), y_pred_log_proba_sk, atol=1e-2, rtol=1e-2
    )


@pytest.mark.parametrize("x_dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize(
    "y_dtype", [cp.int32, cp.int64, cp.float32, cp.float64]
)
@pytest.mark.parametrize("is_sparse", [True, False])
def test_gaussian_fit_predict(
    x_dtype, y_dtype, is_sparse, sparse_text_dataset
):
    """
    Cupy Test
    """

    X, y = sparse_text_dataset
    model = GaussianNB()
    n_rows = 500
    n_cols = 50000
    X = cp_sp.csr_matrix(X, dtype=x_dtype)[:n_rows, :n_cols]

    if is_sparse:
        y = y.astype(y_dtype)[:n_rows]
        model.fit(X, y)
    else:
        X = X.todense()
        y = y[:n_rows].astype(y_dtype)
        model.fit(np.ascontiguousarray(cp.asnumpy(X).astype(x_dtype)), y)

    y_hat = model.predict(X)
    y_hat = cp.asnumpy(y_hat)
    y = cp.asnumpy(y)

    assert accuracy_score(y, y_hat) >= 0.99


def test_gaussian_partial_fit(sparse_text_dataset):
    chunk_size = 250
    n_rows = 1500
    n_cols = 60000
    x_dtype, y_dtype = cp.float32, cp.int32

    X, y = sparse_text_dataset

    X = cp_sp.csr_matrix(X, dtype=x_dtype)[:n_rows, :n_cols]
    y = y.astype(y_dtype)[:n_rows]

    model = GaussianNB()

    classes = np.unique(y)

    total_fit = 0

    for i in range(math.ceil(X.shape[0] / chunk_size)):
        upper = i * chunk_size + chunk_size
        if upper > X.shape[0]:
            upper = -1

        if upper > 0:
            x = X[i * chunk_size : upper]
            y_c = y[i * chunk_size : upper]
        else:
            x = X[i * chunk_size :]
            y_c = y[i * chunk_size :]

        model.partial_fit(x, y_c, classes=classes)

        total_fit += upper - (i * chunk_size)
        if upper == -1:
            break

    y_hat = model.predict(X)

    y_hat = cp.asnumpy(y_hat)
    y = cp.asnumpy(y)
    assert accuracy_score(y, y_hat) >= 0.99

    # Test whether label mismatch between target y and classes raises an Error
    assert_raises(
        ValueError, GaussianNB().partial_fit, X, y, classes=cp.array([0, 1])
    )
    # Raise because classes is required on first call of partial_fit
    assert_raises(ValueError, GaussianNB().partial_fit, X, y)


@pytest.mark.parametrize("priors", [None, "balanced", "unbalanced"])
@pytest.mark.parametrize("var_smoothing", [1e-5, 1e-7, 1e-9])
def test_gaussian_parameters(priors, var_smoothing, sparse_text_dataset):
    x_dtype = cp.float32
    y_dtype = cp.int32
    nrows = 150
    ncols = 20000

    X, y = sparse_text_dataset

    X = cp.asarray(X[:nrows, :ncols].todense(), dtype=x_dtype)
    y = y[:nrows].astype(y_dtype)

    if priors == "balanced":
        priors = cp.array([1 / 20] * 20)
    elif priors == "unbalanced":
        priors = cp.linspace(0.01, 0.09, 20)

    model = GaussianNB(priors=priors, var_smoothing=var_smoothing)
    model_sk = skGNB(
        priors=priors.get() if priors is not None else None,
        var_smoothing=var_smoothing,
    )
    model.fit(X, y)
    model_sk.fit(X.get(), y.get())

    y_hat = model.predict(X)
    y_hat_sk = model_sk.predict(X.get())
    y_hat = cp.asnumpy(y_hat)
    y = cp.asnumpy(y)

    assert_allclose(model.epsilon_.get(), model_sk.epsilon_, rtol=1e-4)
    assert_array_equal(y_hat, y_hat_sk)


@pytest.mark.filterwarnings("ignore:X dtype is not int32.*:UserWarning")
@pytest.mark.filterwarnings(
    "ignore:Changing the sparsity structure of a csr_matrix is expensive.*:scipy.sparse._base.SparseEfficiencyWarning"
)
@pytest.mark.parametrize("x_dtype", [cp.int32, cp.float32, cp.float64])
@pytest.mark.parametrize("y_dtype", [cp.int32, cp.int64])
@pytest.mark.parametrize("is_sparse", [True, False])
def test_categorical(x_dtype, y_dtype, is_sparse, sparse_text_dataset):
    if x_dtype == cp.int32 and is_sparse:
        pytest.skip("Sparse matrices with integers dtype are not supported")
    X, y = sparse_text_dataset
    n_rows = 500
    n_cols = 400

    X = cp_sp.csr_matrix(X, dtype=cp.float32)[:n_rows, :n_cols]
    y = y.astype(y_dtype)[:n_rows]

    if not is_sparse:
        X = X.todense()
    X = X.astype(x_dtype)
    cuml_model = CategoricalNB()
    cuml_model.fit(X, y)
    cuml_score = cuml_model.score(X, y)
    cuml_proba = cuml_model.predict_log_proba(X).get()

    X = X.todense().get() if is_sparse else X.get()
    y = y.get()
    sk_model = skCNB()
    sk_model.fit(X, y)
    sk_score = sk_model.score(X, y)
    sk_proba = sk_model.predict_log_proba(X)

    THRES = 1e-3

    assert_array_equal(sk_model.class_count_, cuml_model.class_count_.get())
    assert_allclose(
        sk_model.class_log_prior_, cuml_model.class_log_prior_.get(), 1e-6
    )
    assert_allclose(cuml_proba, sk_proba, atol=1e-2, rtol=1e-2)
    assert sk_score - THRES <= cuml_score <= sk_score + THRES


@pytest.mark.filterwarnings("ignore:X dtype is not int32.*:UserWarning")
@pytest.mark.filterwarnings(
    "ignore:Changing the sparsity structure of a csr_matrix is expensive.*:scipy.sparse._base.SparseEfficiencyWarning"
)
@pytest.mark.parametrize("x_dtype", [cp.int32, cp.float32, cp.float64])
@pytest.mark.parametrize("y_dtype", [cp.int32, cp.int64])
@pytest.mark.parametrize("is_sparse", [True, False])
def test_categorical_partial_fit(
    x_dtype, y_dtype, is_sparse, sparse_text_dataset
):
    if x_dtype == cp.int32 and is_sparse:
        pytest.skip("Sparse matrices with integers dtype are not supported")
    n_rows = 5000
    n_cols = 500
    chunk_size = 1000
    expected_score = 0.9852

    X, y = sparse_text_dataset

    X = cp_sp.csr_matrix(X, dtype="float32")[:n_rows, :n_cols]
    if is_sparse:
        X.data = X.data.astype(x_dtype)
    else:
        X = X.todense().astype(x_dtype)
    y = y.astype(y_dtype)[:n_rows]

    model = CategoricalNB()

    classes = np.unique(y)
    for i in range(math.ceil(X.shape[0] / chunk_size)):
        upper = i * chunk_size + chunk_size
        if upper > X.shape[0]:
            upper = -1

        if upper > 0:
            x = X[i * chunk_size : upper]
            y_c = y[i * chunk_size : upper]
        else:
            x = X[i * chunk_size :]
            y_c = y[i * chunk_size :]
        model.partial_fit(x, y_c, classes=classes)
        if upper == -1:
            break

    cuml_score = model.score(X, y)
    THRES = 1e-4
    assert expected_score - THRES <= cuml_score <= expected_score + THRES


@pytest.mark.filterwarnings("ignore:X dtype is not int32.*:UserWarning")
@pytest.mark.filterwarnings(
    "ignore:Changing the sparsity structure of a csr_matrix is expensive.*:scipy.sparse._base.SparseEfficiencyWarning"
)
@pytest.mark.parametrize("class_prior", [None, "balanced", "unbalanced"])
@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.5])
@pytest.mark.parametrize("fit_prior", [False, True])
@pytest.mark.parametrize("is_sparse", [False, True])
def test_categorical_parameters(
    class_prior, alpha, fit_prior, is_sparse, sparse_text_dataset
):
    x_dtype = cp.float32
    y_dtype = cp.int32
    nrows = 2000
    ncols = 500

    X, y = sparse_text_dataset

    X = cp_sp.csr_matrix(X, dtype=x_dtype)[:nrows, :ncols]
    if not is_sparse:
        X = X.todense()
    y = y.astype(y_dtype)[:nrows]

    if class_prior == "balanced":
        class_prior = np.array([1 / 20] * 20)
    elif class_prior == "unbalanced":
        class_prior = np.linspace(0.01, 0.09, 20)

    model = CategoricalNB(
        class_prior=class_prior, alpha=alpha, fit_prior=fit_prior
    )
    model_sk = skCNB(class_prior=class_prior, alpha=alpha, fit_prior=fit_prior)
    model.fit(X, y)
    y_hat = model.predict(X).get()
    y_log_prob = model.predict_log_proba(X).get()

    X = X.todense().get() if is_sparse else X.get()
    model_sk.fit(X, y.get())
    y_hat_sk = model_sk.predict(X)
    y_log_prob_sk = model_sk.predict_log_proba(X)

    assert_allclose(y_log_prob, y_log_prob_sk, rtol=1e-4, atol=1e-10)
    assert_array_equal(y_hat, y_hat_sk)
