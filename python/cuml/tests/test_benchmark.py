# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import time

import cudf
import cupy as cp
import numpy as np
import pandas as pd
import pytest
from sklearn import metrics

from cuml.benchmark import algorithms, datagen
from cuml.benchmark.bench_helper_funcs import (
    _training_data_to_numpy,
    fit,
    fit_predict,
)
from cuml.benchmark.runners import (
    AccuracyComparisonRunner,
    SpeedupComparisonRunner,
    run_variations,
)

pytestmark = pytest.mark.skip


@pytest.mark.parametrize("dataset", ["blobs", "regression", "classification"])
def test_data_generators(dataset):
    data = datagen.gen_data(dataset, "numpy", n_samples=100, n_features=10)
    assert isinstance(data[0], np.ndarray)
    assert data[0].shape[0] == 100


@pytest.mark.parametrize(
    "input_type", ["numpy", "cudf", "pandas", "cupy", "cupy-c"]
)
def test_data_generator_types(input_type):
    X, *_ = datagen.gen_data("blobs", input_type, n_samples=100, n_features=10)
    if input_type == "numpy":
        assert isinstance(X, np.ndarray)
    elif input_type == "cudf":
        assert isinstance(X, cudf.DataFrame)
    elif input_type == "pandas":
        assert isinstance(X, pd.DataFrame)
    elif input_type in ("cupy", "cupy-c"):
        assert isinstance(X, cp.ndarray)
    else:
        assert False


def test_data_generator_split():
    X_train, y_train, X_test, y_test = datagen.gen_data(
        "blobs", "numpy", n_samples=100, n_features=10, test_fraction=0.20
    )
    assert X_train.shape == (100, 10)
    assert X_test.shape == (25, 10)


def test_run_variations():
    algo = algorithms.algorithm_by_name("LogisticRegression")

    res = run_variations(
        [algo],
        dataset_name="classification",
        bench_rows=[100, 200],
        bench_dims=[10, 20],
    )
    assert res.shape[0] == 4
    assert (res.n_samples == 100).sum() == 2
    assert (res.n_features == 20).sum() == 2


def test_speedup_runner():
    class MockAlgo:
        def __init__(self, t):
            self.t = t

        def fit(self, X, y):
            time.sleep(self.t)
            return

        def predict(self, X):
            nr = X.shape[0]
            res = np.zeros(nr)
            res[0 : int(nr / 5.0)] = 1.0
            return res

    class FastMockAlgo(MockAlgo):
        def __init__(self):
            MockAlgo.__init__(self, 0.1)

    class SlowMockAlgo(MockAlgo):
        def __init__(self):
            MockAlgo.__init__(self, 2)

    pair = algorithms.AlgorithmPair(
        SlowMockAlgo,
        FastMockAlgo,
        shared_args={},
        name="Mock",
        bench_func=fit_predict,
        accuracy_function=metrics.accuracy_score,
    )

    runner = SpeedupComparisonRunner([20], [5], dataset_name="zeros")
    results = runner.run(pair)[0]

    expected_speedup = SlowMockAlgo().t / FastMockAlgo().t

    assert results["speedup"] == pytest.approx(expected_speedup, 0.4)


def test_multi_reps():
    class CountingAlgo:
        tot_reps = 0

        def fit(self, X, y):
            CountingAlgo.tot_reps += 1

    pair = algorithms.AlgorithmPair(
        CountingAlgo,
        CountingAlgo,
        shared_args={},
        bench_func=fit,
        name="Counting",
    )

    runner = AccuracyComparisonRunner(
        [20], [5], dataset_name="zeros", test_fraction=0.20, n_reps=4
    )
    runner.run(pair)

    # Double the n_reps since it is used in cpu and cuml versions
    assert CountingAlgo.tot_reps == 8


def test_accuracy_runner():
    # Set up data that should deliver accuracy of 0.20 if all goes right
    class MockAlgo:
        def fit(self, X, y):
            return

        def predict(self, X):
            nr = X.shape[0]
            res = np.zeros(nr)
            res[0 : int(nr / 5.0)] = 1.0
            return res

    pair = algorithms.AlgorithmPair(
        MockAlgo,
        MockAlgo,
        shared_args={},
        name="Mock",
        bench_func=fit_predict,
        accuracy_function=metrics.accuracy_score,
    )

    runner = AccuracyComparisonRunner(
        [20], [5], dataset_name="zeros", test_fraction=0.20
    )
    results = runner.run(pair)[0]

    assert results["cuml_acc"] == pytest.approx(0.80)


# Only test a few algorithms (which collectively span several types)
# to reduce runtime burden
# skipping UMAP-Supervised due to issue
# https://github.com/NVIDIA/cuml/issues/4243
@pytest.mark.parametrize(
    "algo_name",
    [
        "DBSCAN",
        "LogisticRegression",
        "ElasticNet",
        "xgboost-classification",
        "xgboost-regression",
    ],
)
def test_real_algos_runner(algo_name):
    pair = algorithms.algorithm_by_name(algo_name)

    if algo_name in ["xgboost-classification", "xgboost-regression"]:
        pytest.importorskip("xgboost")

    # Use appropriate dataset for regression algorithms
    if algo_name in ["ElasticNet", "xgboost-regression"]:
        dataset = "regression"
    else:
        dataset = "classification"

    runner = AccuracyComparisonRunner(
        [50], [5], dataset_name=dataset, test_fraction=0.20
    )
    results = runner.run(pair)[0]
    print(results)
    assert results["cuml_acc"] is not None


@pytest.mark.parametrize(
    "input_type", ["numpy", "cudf", "pandas", "cupy", "cupy-c"]
)
def test_training_data_to_numpy(input_type):
    X, y, *_ = datagen.gen_data(
        "blobs", input_type, n_samples=100, n_features=10
    )
    X_np, y_np = _training_data_to_numpy(X, y)
    assert isinstance(X_np, np.ndarray)
    assert isinstance(y_np, np.ndarray)
