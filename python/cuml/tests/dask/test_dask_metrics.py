#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from itertools import chain, permutations

import cupy as cp
import dask.array as da
import numpy as np
import pytest
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from cuml.dask.metrics import confusion_matrix
from cuml.testing.utils import stress_param


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.mark.mg
@pytest.mark.parametrize("chunks", ["auto", 2, 1])
def test_confusion_matrix(client, chunks):
    y_true = da.from_array(cp.array([2, 0, 2, 2, 0, 1]), chunks=chunks)
    y_pred = da.from_array(cp.array([0, 0, 2, 2, 0, 2]), chunks=chunks)
    cm = confusion_matrix(y_true, y_pred)
    ref = cp.array([[2, 0, 0], [0, 0, 1], [1, 0, 2]])
    cp.testing.assert_array_equal(cm, ref)


@pytest.mark.mg
@pytest.mark.parametrize("chunks", ["auto", 2, 1])
def test_confusion_matrix_binary(client, chunks):
    y_true = da.from_array(cp.array([0, 1, 0, 1]), chunks=chunks)
    y_pred = da.from_array(cp.array([1, 1, 1, 0]), chunks=chunks)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    ref = cp.array([0, 2, 1, 1])
    cp.testing.assert_array_equal(ref, cp.array([tn, fp, fn, tp]))


@pytest.mark.mg
@pytest.mark.parametrize("n_samples", [50, 3000, stress_param(500000)])
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("problem_type", ["binary", "multiclass"])
def test_confusion_matrix_random(rng, n_samples, dtype, problem_type, client):
    upper_range = 2 if problem_type == "binary" else 1000

    np_y_true = rng.randint(0, upper_range, n_samples).astype(dtype)
    np_y_pred = rng.randint(0, upper_range, n_samples).astype(dtype)
    y_true = da.from_array(cp.asarray(np_y_true))
    y_pred = da.from_array(cp.asarray(np_y_pred))

    cm = confusion_matrix(y_true, y_pred)
    ref = sk_confusion_matrix(np_y_true, np_y_pred)
    cp.testing.assert_array_almost_equal(ref, cm, decimal=4)


@pytest.mark.mg
@pytest.mark.parametrize(
    "normalize, expected_results",
    [
        ("true", 0.333333333),
        ("pred", 0.333333333),
        ("all", 0.1111111111),
        (None, 2),
    ],
)
def test_confusion_matrix_normalize(normalize, expected_results, client):
    y_test = da.from_array(cp.array([0, 1, 2] * 6))
    y_pred = da.from_array(cp.array(list(chain(*permutations([0, 1, 2])))))
    cm = confusion_matrix(y_test, y_pred, normalize=normalize)
    cp.testing.assert_allclose(cm, cp.array(expected_results))


@pytest.mark.mg
@pytest.mark.parametrize("labels", [(0, 1), (2, 1), (2, 1, 4, 7), (2, 20)])
def test_confusion_matrix_multiclass_subset_labels(rng, labels, client):
    np_y_true = rng.randint(0, 3, 10).astype(np.int32)
    np_y_pred = rng.randint(0, 3, 10).astype(np.int32)
    y_true = da.from_array(cp.asarray(np_y_true))
    y_pred = da.from_array(cp.asarray(np_y_pred))

    ref = sk_confusion_matrix(np_y_true, np_y_pred, labels=labels)
    labels = cp.array(labels, dtype=np.int32)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cp.testing.assert_array_almost_equal(ref, cm, decimal=4)


@pytest.mark.mg
@pytest.mark.parametrize("n_samples", [50, 3000, stress_param(500000)])
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("weights_dtype", ["int", "float"])
def test_confusion_matrix_random_weights(
    rng, n_samples, dtype, weights_dtype, client
):
    np_y_true = rng.randint(0, 10, n_samples).astype(dtype)
    np_y_pred = rng.randint(0, 10, n_samples).astype(dtype)
    y_true = da.from_array(cp.asarray(np_y_true))
    y_pred = da.from_array(cp.asarray(np_y_pred))

    if weights_dtype == "int":
        sample_weight = np.random.RandomState(0).randint(0, 10, n_samples)
    else:
        sample_weight = np.random.RandomState(0).rand(n_samples)

    ref = sk_confusion_matrix(
        np_y_true, np_y_pred, sample_weight=sample_weight
    )

    sample_weight = cp.array(sample_weight)
    sample_weight = da.from_array(sample_weight)

    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    cp.testing.assert_array_almost_equal(ref, cm, decimal=4)
