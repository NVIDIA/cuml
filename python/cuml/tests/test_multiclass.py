# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import numpy as np
import pytest
from sklearn import multiclass as sk_multiclass
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from cuml import SVC as cuSVC
from cuml import LogisticRegression as cuLog
from cuml import multiclass as cu_multiclass
from cuml.testing.datasets import make_classification_dataset


class _DeviceOnlyClassifier(BaseEstimator):
    def fit(self, X, y):
        self.dev_ = isinstance(X, cp.ndarray) and isinstance(y, cp.ndarray)
        self.rows_ = X.shape[0]
        self.bin_ = bool(cp.all((y == 0) | (y == 1)))
        return self


@pytest.mark.parametrize(
    ("cls", "n_est", "n_rows"),
    [
        (cu_multiclass.OneVsRestClassifier, 4, 8),
        (cu_multiclass.OneVsOneClassifier, 6, 4),
    ],
)
def test_multiclass_device_fit(cls, n_est, n_rows):
    X = cp.asarray(
        [
            [-4.0, -1.0],
            [-3.0, -1.0],
            [-1.0, 1.0],
            [0.0, 1.0],
            [2.0, 2.0],
            [3.0, 2.0],
            [5.0, 3.0],
            [6.0, 3.0],
        ],
        dtype=cp.float32,
    )
    y = cp.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=cp.int32)

    m = cls(_DeviceOnlyClassifier()).fit(X, y)

    assert len(m.estimators_) == n_est
    assert all(est.dev_ for est in m.estimators_)
    assert all(est.rows_ == n_rows for est in m.estimators_)
    assert all(est.bin_ for est in m.estimators_)


@pytest.mark.parametrize("num_classes", [2, 3])
@pytest.mark.parametrize(
    ("cu_cls", "sk_cls"),
    [
        (
            cu_multiclass.OneVsRestClassifier,
            sk_multiclass.OneVsRestClassifier,
        ),
        (
            cu_multiclass.OneVsOneClassifier,
            sk_multiclass.OneVsOneClassifier,
        ),
    ],
)
def test_multiclass_sklearn_parity(cu_cls, sk_cls, num_classes):
    Xtr, Xte, ytr, _ = make_classification_dataset(
        datatype=np.float32,
        nrows=400,
        ncols=10,
        n_info=4,
        num_classes=num_classes,
    )
    ytr = ytr.astype(np.float32)

    cu = cu_cls(cuLog()).fit(Xtr, ytr)
    sk = sk_cls(cuLog()).fit(Xtr, ytr)

    np.testing.assert_array_equal(cu.predict(Xte), sk.predict(Xte))
    np.testing.assert_allclose(
        cu.decision_function(Xte),
        sk.decision_function(Xte),
        rtol=1e-4,
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "cls",
    [
        cu_multiclass.OneVsRestClassifier,
        cu_multiclass.OneVsOneClassifier,
    ],
)
def test_multiclass_not_fitted(cls):
    m = cls(cuLog())
    X = cp.zeros((2, 2), dtype=cp.float32)

    with pytest.raises(NotFittedError):
        _ = m.classes_

    with pytest.raises(NotFittedError):
        m.predict(X)

    with pytest.raises(NotFittedError):
        m.decision_function(X)


def test_ovo_single_class():
    X = cp.arange(12, dtype=cp.float32).reshape(6, 2)
    y = cp.full(6, 7, dtype=cp.int32)

    with pytest.raises(
        ValueError,
        match="only one class is present",
    ):
        cu_multiclass.OneVsOneClassifier(cuLog()).fit(X, y)


def test_ovr_single_class():
    X = cp.arange(12, dtype=cp.float32).reshape(6, 2)
    y = cp.full(6, 7, dtype=cp.int32)

    with pytest.warns(UserWarning, match="Label not 7"):
        cls = cu_multiclass.OneVsRestClassifier(cuLog()).fit(X, y)

    assert len(cls.estimators_) == 1
    assert bool(cp.all(cls.predict(X) == 7))
    assert bool(cp.all(cls.decision_function(X) == 0))
    assert cls.decision_function(X).dtype.kind == "f"


@pytest.mark.parametrize(
    "classifier_cls",
    [
        cu_multiclass.OneVsRestClassifier,
        cu_multiclass.OneVsOneClassifier,
    ],
)
def test_sample_weight(classifier_cls):
    X = np.arange(12, dtype=np.float64).reshape(6, 2)
    y = np.repeat(np.arange(3), 2)
    sample_weight = np.array([0.1, 0.1, 10.0, 10.0, 0.5, 0.5])
    X_test = X[[0, 3, 5]]
    y_test = y[[0, 3, 5]]

    model = classifier_cls(cuSVC(kernel="rbf")).fit(
        X, y, sample_weight=sample_weight
    )

    np.testing.assert_array_equal(model.predict(X_test), [1, 1, 2])
    assert model.score(X_test, y_test) == pytest.approx(2 / 3)


def test_svc_multiclass_integration():
    X = cp.asarray(
        [
            [-4, -4],
            [-3, -4],
            [-4, -3],
            [0, 4],
            [1, 4],
            [0, 5],
            [4, -4],
            [5, -4],
            [4, -3],
        ],
        dtype=cp.float32,
    )
    y = cp.asarray([2, 2, 2, 4, 4, 4, 9, 9, 9], dtype=cp.int32)

    model = cuSVC(C=10, decision_function_shape="ovo").fit(X, y)

    cp.testing.assert_array_equal(model.predict(X), y)
    assert model.support_.size > 0
    assert model.intercept_.size == 3
    assert model.n_iter_.size == 3


@pytest.mark.parametrize("strategy", ["ovr", "ovo"])
@pytest.mark.parametrize("nrows", [1000])
@pytest.mark.parametrize("num_classes", [3])
@pytest.mark.parametrize("column_info", [[10, 4]])
def test_logistic_regression(
    strategy, nrows, num_classes, column_info, dtype=np.float32
):
    ncols, n_info = column_info

    X_train, X_test, y_train, y_test = make_classification_dataset(
        datatype=dtype,
        nrows=nrows,
        ncols=ncols,
        n_info=n_info,
        num_classes=num_classes,
    )
    y_train = y_train.astype(dtype)
    y_test = y_test.astype(dtype)
    culog = cuLog()

    if strategy == "ovo":
        cls = cu_multiclass.OneVsOneClassifier(culog)
    else:
        cls = cu_multiclass.OneVsRestClassifier(culog)

    cls.fit(X_train, y_train)
    test_score = cls.score(X_test, y_test)
    assert test_score > 0.7
