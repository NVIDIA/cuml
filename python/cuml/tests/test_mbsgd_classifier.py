# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import cupy as cp
import numpy as np
import pytest
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import cuml
from cuml.datasets import make_classification
from cuml.testing.utils import quality_param, stress_param, unit_param


@pytest.fixture(
    scope="module",
    params=[
        unit_param([500, 20, 10, np.float32]),
        unit_param([500, 20, 10, np.float64]),
        quality_param([5000, 100, 50, np.float32]),
        quality_param([5000, 100, 50, np.float64]),
        stress_param([500000, 1000, 500, np.float32]),
        stress_param([500000, 1000, 500, np.float64]),
    ],
    ids=[
        "500-20-10-f32",
        "500-20-10-f64",
        "5000-100-50-f32",
        "5000-100-50-f64",
        "500000-1000-500-f32",
        "500000-1000-500-f64",
    ],
)
def make_dataset(request):
    nrows, ncols, n_info, datatype = request.param
    X, y = make_classification(
        n_samples=nrows,
        n_informative=n_info,
        n_features=ncols,
        random_state=10,
    )
    X = X.astype(datatype)
    y = y.astype(datatype)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=10
    )

    y_train = y_train.astype(datatype)
    y_test = y_test.astype(datatype)

    return nrows, X_train, X_test, y_train, y_test


@pytest.mark.parametrize(
    # Grouped those tests to reduce the total number of individual tests
    # while still keeping good coverage of the different features of MBSGD
    ("lrate", "penalty", "loss"),
    [
        ("constant", None, "log"),
        ("invscaling", "l2", "hinge"),
        ("adaptive", "l1", "squared_loss"),
        ("constant", "elasticnet", "hinge"),
    ],
)
@pytest.mark.filterwarnings("ignore:Maximum::sklearn[.*]")
def test_mbsgd_classifier_vs_skl(lrate, penalty, loss, make_dataset):
    nrows, X_train, X_test, y_train, y_test = make_dataset

    if nrows < 500000:
        cu_mbsgd_classifier = cuml.MBSGDClassifier(
            learning_rate=lrate,
            eta0=0.005,
            epochs=100,
            fit_intercept=True,
            batch_size=2,
            tol=0.0,
            penalty=penalty,
        )

        cu_mbsgd_classifier.fit(X_train, y_train)
        cu_pred = cu_mbsgd_classifier.predict(X_test)
        cu_acc = accuracy_score(cp.asnumpy(cu_pred), cp.asnumpy(y_test))

        skl_sgd_classifier = SGDClassifier(
            learning_rate=lrate,
            eta0=0.005,
            max_iter=100,
            fit_intercept=True,
            tol=0.0,
            penalty=penalty,
            random_state=0,
        )

        skl_sgd_classifier.fit(cp.asnumpy(X_train), cp.asnumpy(y_train))
        skl_pred = skl_sgd_classifier.predict(cp.asnumpy(X_test))
        skl_acc = accuracy_score(skl_pred, cp.asnumpy(y_test))
        assert cu_acc >= skl_acc - 0.08


@pytest.mark.parametrize(
    # Grouped those tests to reduce the total number of individual tests
    # while still keeping good coverage of the different features of MBSGD
    ("lrate", "penalty", "loss"),
    [
        ("constant", None, "log"),
        ("invscaling", "l2", "hinge"),
        ("adaptive", "l1", "squared_loss"),
        ("constant", "elasticnet", "hinge"),
    ],
)
def test_mbsgd_classifier(lrate, penalty, loss, make_dataset):
    nrows, X_train, X_test, y_train, y_test = make_dataset

    model = cuml.MBSGDClassifier(
        learning_rate=lrate,
        eta0=0.005,
        epochs=100,
        fit_intercept=True,
        batch_size=nrows / 100,
        tol=0.0,
        penalty=penalty,
    )
    # Fitted attributes don't exist before fit
    assert not hasattr(model, "coef_")
    assert not hasattr(model, "intercept_")

    model.fit(X_train, y_train)

    # Fitted attributes exist and have correct types after fit
    assert isinstance(model.coef_, type(X_train))
    assert isinstance(model.intercept_, float)
    assert isinstance(model.classes_, np.ndarray)

    cu_pred = model.predict(X_test)
    cu_acc = accuracy_score(cp.asnumpy(cu_pred), cp.asnumpy(y_test))

    assert cu_acc > 0.7


def test_mbsgd_classifier_default(make_dataset):
    nrows, X_train, X_test, y_train, y_test = make_dataset

    cu_mbsgd_classifier = cuml.MBSGDClassifier(batch_size=nrows / 10)

    cu_mbsgd_classifier.fit(X_train, y_train)
    cu_pred = cu_mbsgd_classifier.predict(X_test)
    cu_acc = accuracy_score(cp.asnumpy(cu_pred), cp.asnumpy(y_test))

    assert cu_acc >= 0.69


def test_mbsgd_multiclass_errors():
    X, y = make_classification(random_state=42, n_classes=4, n_informative=4)
    model = cuml.MBSGDClassifier()
    with pytest.raises(
        ValueError,
        match="binary classification is supported, got 4 classes",
    ):
        model.fit(X, y)
