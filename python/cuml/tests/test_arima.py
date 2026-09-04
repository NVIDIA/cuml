#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

###############################################################################
#                             How these tests work                            #
###############################################################################
#
# This test file contains some unit tests and an integration test.
#
# The units tests use the same parameters with cuML and the reference
# implementation to compare strict parity of specific components.
#
# The integration tests compare that, when fitting and forecasting separately,
# our implementation performs better or approximately as good as the reference
# (it mostly serves to test that we don't have any regression)
#
# Note that there are significant differences between our implementation and
# the reference, and perfect parity cannot be expected for integration tests.

import os
import warnings

import cudf
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime
from sklearn.model_selection import train_test_split

import cuml.tsa.arima as arima
from cuml.testing.utils import stress_param

sm = pytest.importorskip("statsmodels.api")

pytestmark = pytest.mark.filterwarnings(
    "ignore:`cuml.tsa.ARIMA`, along with the entire `cuml.tsa` module, was "
    "deprecated:FutureWarning"
)

###############################################################################
#                                  Test data                                  #
###############################################################################


class ARIMAData:
    """Contains a dataset name and associated metadata"""

    def __init__(
        self,
        batch_size,
        n_obs,
        n_test,
        dataset,
        tolerance_integration,
        n_exog=0,
        dataset_exog=None,
    ):
        self.batch_size = batch_size
        self.n_obs = n_obs
        self.n_test = n_test
        self.dataset = dataset
        self.tolerance_integration = tolerance_integration

        self.n_exog = n_exog
        self.dataset_exog = dataset_exog

        self.n_train = n_obs - n_test


# ARIMA(1,0,1) with intercept
test_101c = ARIMAData(
    batch_size=8,
    n_obs=15,
    n_test=2,
    dataset="long_term_arrivals_by_citizenship",
    tolerance_integration=0.01,
)

# ARIMA(0,0,2) with intercept
test_002c = ARIMAData(
    batch_size=7,
    n_obs=20,
    n_test=2,
    dataset="net_migrations_auckland_by_age",
    tolerance_integration=0.05,
)

# ARIMA(0,1,0) with intercept
test_010c = ARIMAData(
    batch_size=4,
    n_obs=17,
    n_test=2,
    dataset="cattle",
    tolerance_integration=0.01,
)

# ARIMA(1,1,0)
test_110 = ARIMAData(
    batch_size=1,
    n_obs=137,
    n_test=5,
    dataset="police_recorded_crime",
    tolerance_integration=0.01,
)

# ARIMA(0,1,1) with intercept
test_011c = ARIMAData(
    batch_size=16,
    n_obs=28,
    n_test=2,
    dataset="deaths_by_region",
    tolerance_integration=0.05,
)

# ARIMA(0,1,1) with intercept (exogenous variables)
test_011c_exog = ARIMAData(
    batch_size=16,
    n_obs=28,
    n_test=2,
    dataset="endog_deaths_by_region_exog",
    tolerance_integration=0.05,
    n_exog=2,
    dataset_exog="exog_deaths_by_region_exog",
)

# ARIMA(1,2,1) with intercept
test_121c = ARIMAData(
    batch_size=2,
    n_obs=137,
    n_test=10,
    dataset="population_estimate",
    tolerance_integration=0.07,
)

# ARIMA(1,1,1) with intercept (missing observations)
test_111c_missing = ARIMAData(
    batch_size=2,
    n_obs=137,
    n_test=10,
    dataset="population_estimate_missing",
    tolerance_integration=0.01,
)

# ARIMA(1,0,1)(1,1,1)_4
test_101_111_4 = ARIMAData(
    batch_size=3,
    n_obs=101,
    n_test=10,
    dataset="alcohol",
    tolerance_integration=0.09,
)

# ARIMA(5,1,0)
test_510 = ARIMAData(
    batch_size=3,
    n_obs=101,
    n_test=10,
    dataset="alcohol",
    tolerance_integration=0.02,
)

# ARIMA(1,1,1)(2,0,0)_4 with intercept
test_111_200_4c = ARIMAData(
    batch_size=14,
    n_obs=123,
    n_test=10,
    dataset="hourly_earnings_by_industry",
    tolerance_integration=0.01,
)

# ARIMA(1,1,1)(2,0,0)_4 with intercept (missing observations)
test_111_200_4c_missing = ARIMAData(
    batch_size=14,
    n_obs=123,
    n_test=10,
    dataset="hourly_earnings_by_industry_missing",
    tolerance_integration=0.01,
)

# ARIMA(1,1,1)(2,0,0)_4 with intercept
# (missing observations and exogenous variables)
test_111_200_4c_missing_exog = ARIMAData(
    batch_size=14,
    n_obs=123,
    n_test=10,
    dataset="endog_hourly_earnings_by_industry_missing_exog",
    tolerance_integration=0.01,
    n_exog=2,
    dataset_exog="exog_hourly_earnings_by_industry_missing_exog",
)

# ARIMA(1,1,2)(0,1,2)_4
test_112_012_4 = ARIMAData(
    batch_size=2,
    n_obs=179,
    n_test=10,
    dataset="passenger_movements",
    tolerance_integration=0.005,
)

# ARIMA(1,1,1)(1,1,1)_12
test_111_111_12 = ARIMAData(
    batch_size=12,
    n_obs=279,
    n_test=20,
    dataset="guest_nights_by_region",
    tolerance_integration=0.001,
)

# ARIMA(1,1,1)(1,1,1)_12 (missing observations)
test_111_111_12_missing = ARIMAData(
    batch_size=12,
    n_obs=279,
    n_test=20,
    dataset="guest_nights_by_region_missing",
    tolerance_integration=0.03,
)

# ARIMA(1,1,1)(1,1,1)_12 (missing obs, exogenous variables, intercept)
test_111_111_12c_missing_exog = ARIMAData(
    batch_size=12,
    n_obs=279,
    n_test=20,
    dataset="endog_guest_nights_by_region_missing_exog",
    tolerance_integration=0.001,
    n_exog=2,
    dataset_exog="exog_guest_nights_by_region_missing_exog",
)

# Dictionary matching a test case to a tuple of model parameters
# (a test case could be used with different models)
# (p, d, q, P, D, Q, s, k) -> ARIMAData
test_data = [
    # ((1, 0, 1, 0, 0, 0, 0, 1), test_101c),
    ((0, 0, 2, 0, 0, 0, 0, 1), test_002c),
    ((0, 1, 0, 0, 0, 0, 0, 1), test_010c),
    ((1, 1, 0, 0, 0, 0, 0, 0), test_110),
    ((0, 1, 1, 0, 0, 0, 0, 1), test_011c),
    ((0, 1, 1, 0, 0, 0, 0, 1), test_011c_exog),
    ((1, 2, 1, 0, 0, 0, 0, 1), test_121c),
    ((1, 1, 1, 0, 0, 0, 0, 1), test_111c_missing),
    ((1, 0, 1, 1, 1, 1, 4, 0), test_101_111_4),
    ((5, 1, 0, 0, 0, 0, 0, 0), test_510),
    # Skip due to update to Scipy 1.15
    # ((1, 1, 1, 2, 0, 0, 4, 1), test_111_200_4c),
    # ((1, 1, 1, 2, 0, 0, 4, 1), test_111_200_4c_missing),
    ((1, 1, 1, 2, 0, 0, 4, 1), test_111_200_4c_missing_exog),
    ((1, 1, 2, 0, 1, 2, 4, 0), test_112_012_4),
    stress_param((1, 1, 1, 1, 1, 1, 12, 0), test_111_111_12),
    stress_param((1, 1, 1, 1, 1, 1, 12, 0), test_111_111_12_missing),
    stress_param((1, 0, 1, 1, 1, 1, 12, 1), test_111_111_12c_missing_exog),
]

# Dictionary for lazy-loading of datasets
# (name, dtype) -> (pandas dataframe, cuDF dataframe)
lazy_data = {}

# Dictionary for lazy-evaluation of reference fits
# (p, d, q, P, D, Q, s, k, name, dtype) -> SARIMAXResults
lazy_ref_fit = {}


def extract_order(tup):
    """Extract the order from a tuple of parameters"""
    p, d, q, P, D, Q, s, k = tup
    return (p, d, q), (P, D, Q, s), k


data_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ts_datasets"
)


def get_dataset(data, dtype):
    """Load a dataset with a given dtype or return a previously loaded dataset"""
    key = (data.dataset, np.dtype(dtype).name)
    if key not in lazy_data:
        y = pd.read_csv(
            os.path.join(data_path, "{}.csv".format(data.dataset)),
            usecols=range(1, data.batch_size + 1),
            dtype=dtype,
        )
        y_train, y_test = train_test_split(
            y, test_size=data.n_test, shuffle=False
        )
        y_train_cudf = cudf.from_pandas(y_train).fillna(np.nan)
        y_test_cudf = cudf.from_pandas(y_test)
        if data.dataset_exog is not None:
            exog = pd.read_csv(
                os.path.join(data_path, "{}.csv".format(data.dataset_exog)),
                usecols=range(1, data.n_exog * data.batch_size + 1),
                dtype=dtype,
            )
            exog_past, exog_fut = train_test_split(
                exog, test_size=data.n_test, shuffle=False
            )
            exog_past_cudf = cudf.from_pandas(exog_past).fillna(np.nan)
            exog_fut_cudf = cudf.from_pandas(exog_fut)
        else:
            exog_past, exog_past_cudf, exog_fut, exog_fut_cudf = [None] * 4
        lazy_data[key] = (
            y_train,
            y_train_cudf,
            y_test,
            y_test_cudf,
            exog_past,
            exog_past_cudf,
            exog_fut,
            exog_fut_cudf,
        )
    return lazy_data[key]


def get_ref_fit(data, order, seasonal_order, intercept, dtype):
    """Compute a reference fit of a dataset with the given parameters and dtype
    or return a previously computed fit
    """
    y_train, _, _, _, exog_past, *_ = get_dataset(data, dtype)
    key = (
        order
        + seasonal_order
        + (intercept, data.dataset, np.dtype(dtype).name)
    )
    batch_size = y_train.shape[1]
    if key not in lazy_ref_fit:
        ref_model = [
            sm.tsa.SARIMAX(
                endog=y_train[y_train.columns[i]],
                exog=exog_past[
                    exog_past.columns[data.n_exog * i : data.n_exog * (i + 1)]
                ]
                if exog_past is not None
                else None,
                order=order,
                seasonal_order=seasonal_order,
                trend="c" if intercept else "n",
            )
            for i in range(batch_size)
        ]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            lazy_ref_fit[key] = [model.fit(disp=0) for model in ref_model]
    return lazy_ref_fit[key]


###############################################################################
#                              Utility functions                              #
###############################################################################


def mase(y_train, y_test, y_fc, s):
    y_train_np = y_train.to_numpy()
    y_test_np = y_test.to_numpy()
    y_fc_np = y_fc

    diff = np.abs(y_train_np[s:] - y_train_np[:-s])
    scale = np.nanmean(diff, axis=0)

    error = np.abs(y_fc_np - y_test_np).mean(axis=0)
    return np.mean(error / scale)


def fill_interpolation(df_in):
    np_arr = df_in.to_numpy()
    for ib in range(np_arr.shape[1]):
        n = len(np_arr)
        start, end = -1, 0
        while start < n - 1:
            if not np.isnan(np_arr[start + 1, ib]):
                start += 1
                end = start + 1
            elif end < n and np.isnan(np_arr[end, ib]):
                end += 1
            else:
                if start == -1:
                    np_arr[:end, ib] = np_arr[end, ib]
                elif end == n:
                    np_arr[start + 1 :, ib] = np_arr[start, ib]
                else:
                    for j in range(start + 1, end):
                        coef = (j - start) / (end - start)
                        np_arr[j, ib] = (1.0 - coef) * np_arr[
                            start, ib
                        ] + coef * np_arr[end, ib]
                start = end
                end = start + 1
    return pd.DataFrame(np_arr, columns=df_in.columns)


###############################################################################
#                                    Tests                                    #
###############################################################################


@pytest.mark.parametrize("key, data", test_data)
@pytest.mark.parametrize("dtype", [np.float64])
def test_integration(key, data, dtype):
    """Full integration test: estimate, fit, forecast"""
    if data.dataset == "endog_hourly_earnings_by_industry_missing_exog":
        pytest.xfail(
            reason="test_111_200_4c_missing_exog dataset causes issues on some GPU architectures (see https://github.com/NVIDIA/cuml/issues/6209)"
        )
    if data.dataset == "passenger_movements":
        pytest.xfail(
            reason="test_112_012_4 dataset causes issues on some GPU architectures (see https://github.com/NVIDIA/cuml/issues/6209)"
        )
    order, seasonal_order, intercept = extract_order(key)
    s = max(1, seasonal_order[3])

    (
        y_train,
        y_train_cudf,
        y_test,
        _,
        _,
        exog_past_cudf,
        exog_fut,
        exog_fut_cudf,
    ) = get_dataset(data, dtype)

    # Get fit reference model
    ref_fits = get_ref_fit(data, order, seasonal_order, intercept, dtype)

    # Create and fit cuML model
    cuml_model = arima.ARIMA(
        endog=y_train_cudf,
        exog=exog_past_cudf,
        order=order,
        seasonal_order=seasonal_order,
        fit_intercept=intercept,
        output_type="numpy",
    )
    cuml_model.fit()

    # Predict
    y_fc_cuml = cuml_model.forecast(data.n_test, exog=exog_fut)
    y_fc_ref = np.zeros((data.n_test, data.batch_size))
    for i in range(data.batch_size):
        y_fc_ref[:, i] = (
            ref_fits[i]
            .get_prediction(
                data.n_train,
                data.n_obs - 1,
                exog=None
                if data.n_exog == 0
                else exog_fut[
                    exog_fut.columns[data.n_exog * i : data.n_exog * (i + 1)]
                ],
            )
            .predicted_mean
        )

    # Compare results: MASE must be better or within the tolerance margin
    mase_ref = mase(y_train, y_test, y_fc_ref, s)
    mase_cuml = mase(y_train, y_test, y_fc_cuml, s)
    assert mase_cuml < mase_ref * (1.0 + data.tolerance_integration)


def _statsmodels_to_cuml(
    ref_fits, cuml_model, order, seasonal_order, intercept, dtype
):
    """Utility function to transfer the parameters from a statsmodels'
    SARIMAXResults object to a cuML ARIMA object.

    .. note:: be cautious with the intercept, it is not always equivalent
        in statsmodels and cuML models (it depends on the order).

    """
    nb = cuml_model.batch_size
    N = cuml_model.complexity
    x = np.zeros(nb * N, dtype=np.float64)

    for ib in range(nb):
        x[ib * N : (ib + 1) * N] = ref_fits[ib].params[:N]

    cuml_model.unpack(x)


def _predict_common(
    key,
    data,
    dtype,
    start,
    end,
    num_steps=None,
    level=None,
    simple_differencing=True,
):
    """Utility function used by test_predict and test_forecast to avoid
    code duplication.
    """
    order, seasonal_order, intercept = extract_order(key)

    _, y_train_cudf, _, _, _, exog_cudf, exog_fut, exog_fut_cudf = get_dataset(
        data, dtype
    )

    # Get fit reference model
    ref_fits = get_ref_fit(data, order, seasonal_order, intercept, dtype)

    # Create cuML model
    cuml_model = arima.ARIMA(
        endog=y_train_cudf,
        exog=exog_cudf,
        order=order,
        seasonal_order=seasonal_order,
        fit_intercept=intercept,
        output_type="numpy",
        simple_differencing=simple_differencing,
    )

    # Feed the parameters to the cuML model
    _statsmodels_to_cuml(
        ref_fits, cuml_model, order, seasonal_order, intercept, dtype
    )

    # Predict or forecast
    # Reference (statsmodels)
    ref_preds = np.zeros((end - start, data.batch_size))
    for i in range(data.batch_size):
        ref_preds[:, i] = (
            ref_fits[i]
            .get_prediction(
                start,
                end - 1,
                exog=(
                    None
                    if data.n_exog == 0 or end <= data.n_train
                    else exog_fut[
                        exog_fut.columns[
                            data.n_exog * i : data.n_exog * (i + 1)
                        ]
                    ]
                ),
            )
            .predicted_mean
        )
    if level is not None:
        ref_lower = np.zeros((end - start, data.batch_size))
        ref_upper = np.zeros((end - start, data.batch_size))
        for i in range(data.batch_size):
            temp_pred = ref_fits[i].get_forecast(
                num_steps,
                exog=(
                    None
                    if data.n_exog == 0
                    else exog_fut[
                        exog_fut.columns[
                            data.n_exog * i : data.n_exog * (i + 1)
                        ]
                    ]
                ),
            )
            ci = temp_pred.summary_frame(alpha=1 - level)
            ref_lower[:, i] = ci["mean_ci_lower"].to_numpy()
            ref_upper[:, i] = ci["mean_ci_upper"].to_numpy()
    # cuML
    if num_steps is None:
        cuml_pred = cuml_model.predict(
            start,
            end,
            exog=None
            if data.n_exog == 0 or end <= data.n_train
            else exog_fut_cudf,
        )
    elif level is not None:
        cuml_pred, cuml_lower, cuml_upper = cuml_model.forecast(
            num_steps, level, exog=exog_fut_cudf
        )
    else:
        cuml_pred = cuml_model.forecast(num_steps, exog=exog_fut_cudf)

    # Compare results
    np.testing.assert_allclose(cuml_pred, ref_preds, rtol=0.002, atol=0.01)
    if level is not None:
        np.testing.assert_allclose(
            cuml_lower, ref_lower, rtol=0.005, atol=0.01
        )
        np.testing.assert_allclose(
            cuml_upper, ref_upper, rtol=0.005, atol=0.01
        )


@pytest.mark.parametrize("key, data", test_data)
@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("simple_differencing", [True, False])
def test_predict_in(key, data, dtype, simple_differencing):
    """Test in-sample prediction against statsmodels (with the same values
    for the model parameters)
    """
    _predict_common(
        key,
        data,
        dtype,
        data.n_train // 2,
        data.n_obs,
        simple_differencing=simple_differencing,
    )


@pytest.mark.parametrize("key, data", test_data)
@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("simple_differencing", [True, False])
def test_predict_inout(key, data, dtype, simple_differencing):
    """Test in- and ouf-of-sample prediction against statsmodels (with the
    same values for the model parameters)
    """
    _predict_common(
        key,
        data,
        dtype,
        data.n_train // 2,
        data.n_train,
        simple_differencing=simple_differencing,
    )


@pytest.mark.parametrize("key, data", test_data)
@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("simple_differencing", [True, False])
def test_forecast(key, data, dtype, simple_differencing):
    """Test out-of-sample forecasting against statsmodels (with the same
    values for the model parameters)
    """
    _predict_common(
        key,
        data,
        dtype,
        data.n_train,
        data.n_obs,
        data.n_test,
        simple_differencing=simple_differencing,
    )


@pytest.mark.parametrize("key, data", test_data)
@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("level", [0.5, 0.95])
def test_intervals(key, data, dtype, level):
    """Test forecast confidence intervals against statsmodels (with the same
    values for the model parameters)
    """
    _predict_common(
        key, data, dtype, data.n_train, data.n_obs, data.n_test, level
    )


@pytest.mark.parametrize("key, data", test_data)
@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("simple_differencing", [True, False])
def test_loglikelihood(key, data, dtype, simple_differencing):
    """Test loglikelihood against statsmodels (with the same values for the
    model parameters)
    """
    order, seasonal_order, intercept = extract_order(key)

    _, y_train_cudf, _, _, _, exog_past_cudf, *_ = get_dataset(data, dtype)

    # Get fit reference model
    ref_fits = get_ref_fit(data, order, seasonal_order, intercept, dtype)

    # Create cuML model
    cuml_model = arima.ARIMA(
        endog=y_train_cudf,
        exog=exog_past_cudf,
        order=order,
        seasonal_order=seasonal_order,
        fit_intercept=intercept,
        simple_differencing=simple_differencing,
    )

    # Feed the parameters to the cuML model
    _statsmodels_to_cuml(
        ref_fits, cuml_model, order, seasonal_order, intercept, dtype
    )

    # Compute loglikelihood
    cuml_llf = cuml_model.llf
    ref_llf = np.array([ref_fit.llf for ref_fit in ref_fits])

    # Compare results
    np.testing.assert_allclose(cuml_llf, ref_llf, rtol=0.01, atol=0.01)


@pytest.mark.parametrize("key, data", test_data)
@pytest.mark.parametrize("dtype", [np.float64])
def test_gradient(key, data, dtype):
    """
    Test batched gradient implementation against scipy non-batched
    gradient.

    .. note:: it doesn't test that the loglikelihood is correct!
    """
    order, seasonal_order, intercept = extract_order(key)
    p, _, q = order
    P, _, Q, _ = seasonal_order
    h = 1e-8

    _, y_train_cudf, _, _, _, exog_past_cudf, *_ = get_dataset(data, dtype)

    # Create cuML model
    cuml_model = arima.ARIMA(
        endog=y_train_cudf,
        exog=exog_past_cudf,
        order=order,
        seasonal_order=seasonal_order,
        fit_intercept=intercept,
    )

    N = cuml_model.complexity

    # Get an estimate of the parameters and pack them into a vector
    cuml_model._estimate_x0()
    x = cuml_model.pack()

    # Compute the batched loglikelihood gradient
    batched_grad = cuml_model._loglike_grad(x, h)

    # Iterate over the batch to compute a reference gradient
    scipy_grad = np.zeros(N * data.batch_size)
    for i in range(data.batch_size):
        # Create a model with only the current series
        model_i = arima.ARIMA(
            endog=y_train_cudf[y_train_cudf.columns[i]],
            exog=None
            if exog_past_cudf is None
            else exog_past_cudf[
                exog_past_cudf.columns[data.n_exog * i : data.n_exog * (i + 1)]
            ],
            order=order,
            seasonal_order=seasonal_order,
            fit_intercept=intercept,
        )

        def f(x):
            return model_i._loglike(x)

        scipy_grad[N * i : N * (i + 1)] = approx_fprime(
            x[N * i : N * (i + 1)], f, h
        )

    # Compare
    np.testing.assert_allclose(batched_grad, scipy_grad, rtol=0.001, atol=0.01)


@pytest.mark.parametrize("key, data", test_data)
@pytest.mark.parametrize("dtype", [np.float64])
def test_start_params(key, data, dtype):
    """Test starting parameters against statsmodels"""
    order, seasonal_order, intercept = extract_order(key)

    y_train, y_train_cudf, _, _, exog_past, exog_past_cudf, *_ = get_dataset(
        data, dtype
    )

    # fillna for reference to match cuML initial estimation strategy
    y_train_nona = fill_interpolation(y_train)

    # Convert to numpy to avoid misaligned indices
    if exog_past is not None:
        exog_past_np = exog_past.to_numpy()

    # Create models
    cuml_model = arima.ARIMA(
        endog=y_train_cudf,
        exog=exog_past_cudf,
        order=order,
        seasonal_order=seasonal_order,
        fit_intercept=intercept,
    )
    ref_model = [
        sm.tsa.SARIMAX(
            endog=y_train_nona[y_train_nona.columns[i]],
            exog=exog_past_np[:, i * data.n_exog : (i + 1) * data.n_exog]
            if data.n_exog
            else None,
            order=order,
            seasonal_order=seasonal_order,
            trend="c" if intercept else "n",
        )
        for i in range(data.batch_size)
    ]

    # Estimate reference starting parameters
    N = cuml_model.complexity
    nb = data.batch_size
    x_ref = np.zeros(N * nb, dtype=dtype)
    for ib in range(nb):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            x_ref[ib * N : (ib + 1) * N] = ref_model[ib].start_params[:N]

    # Estimate cuML starting parameters
    cuml_model._estimate_x0()
    x_cuml = cuml_model.pack()

    # Compare results
    np.testing.assert_allclose(x_cuml, x_ref, rtol=0.001, atol=0.01)
