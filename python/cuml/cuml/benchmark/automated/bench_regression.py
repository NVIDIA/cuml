#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from .. import datagen
from .utils.utils import bench_step  # noqa: F401
from .utils.utils import _benchmark_algo, fixture_generation_helper

#
# Core tests
#


@pytest.fixture(
    **fixture_generation_helper(
        {"n_samples": [1000, 10000], "n_features": [5, 400]}
    )
)
def regression1(request):
    data = datagen.gen_data(
        "regression",
        "cupy",
        n_samples=request.param["n_samples"],
        n_features=request.param["n_features"],
    )
    return data, {"dataset_type": "regression", **request.param}


@pytest.fixture(
    **fixture_generation_helper(
        {"n_samples": [500, 4000], "n_features": [5, 400]}
    )
)
def regression2(request):
    data = datagen.gen_data(
        "regression",
        "cupy",
        n_samples=request.param["n_samples"],
        n_features=request.param["n_features"],
    )
    return data, {"dataset_type": "regression", **request.param}


def bench_linear_regression(
    gpubenchmark,
    bench_step,
    regression1,  # noqa: F811
):
    _benchmark_algo(gpubenchmark, "LinearRegression", bench_step, regression1)


def bench_lasso(gpubenchmark, bench_step, regression1):  # noqa: F811
    _benchmark_algo(gpubenchmark, "Lasso", bench_step, regression1)


def bench_elastic(gpubenchmark, bench_step, regression1):  # noqa: F811
    _benchmark_algo(gpubenchmark, "ElasticNet", bench_step, regression1)


def bench_ridge(gpubenchmark, bench_step, regression1):  # noqa: F811
    _benchmark_algo(gpubenchmark, "Ridge", bench_step, regression1)


def bench_ridge_cv(gpubenchmark, bench_step, regression1):  # noqa: F811
    _benchmark_algo(gpubenchmark, "RidgeCV", bench_step, regression1)


def bench_knnregressor(gpubenchmark, bench_step, regression1):  # noqa: F811
    _benchmark_algo(
        gpubenchmark, "KNeighborsRegressor", bench_step, regression1
    )


def bench_svr_rbf(gpubenchmark, bench_step, regression1):  # noqa: F811
    _benchmark_algo(gpubenchmark, "SVR-RBF", bench_step, regression1)


def bench_svr_linear(gpubenchmark, bench_step, regression2):  # noqa: F811
    _benchmark_algo(gpubenchmark, "SVR-Linear", bench_step, regression2)


def bench_xgboost_regression(
    gpubenchmark,
    bench_step,
    regression1,  # noqa: F811
):
    pytest.importorskip("xgboost")
    _benchmark_algo(
        gpubenchmark, "xgboost-regression", bench_step, regression1
    )
