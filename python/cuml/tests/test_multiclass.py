# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np
import pytest

from cuml import SVC as cuSVC
from cuml import LogisticRegression as cuLog
from cuml import multiclass as cu_multiclass
from cuml.testing.datasets import make_classification_dataset


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
