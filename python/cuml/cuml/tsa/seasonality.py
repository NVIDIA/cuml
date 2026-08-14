# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import cupy as cp
import numpy as np

from cuml.internals import mlfunc
from cuml.internals.validation import check_array
from cuml.tsa._deprecation import deprecated_tsa_api


@deprecated_tsa_api("cuml.tsa.seasonality.seas_test")
@mlfunc
def seas_test(y, s):
    """
    Perform Wang, Smith & Hyndman's test to decide whether seasonal
    differencing is needed

    .. deprecated:: 26.08
        ``cuml.tsa.seasonality.seas_test`` is deprecated and will be removed
        in the cuML 26.12 release.

    Parameters
    ----------
    y : dataframe or array-like (device or host)
        The time series data, assumed to have each time series in columns.
        Acceptable formats: cuDF DataFrame, cuDF Series, NumPy ndarray,
        Numba device ndarray, cuda array interface compliant array like CuPy.
    s: integer
        Seasonal period (s > 1)

    Returns
    -------
    stationarity : List[bool]
        For each series in the batch, whether it needs seasonal differencing
    """
    # TODO: our own implementation of STL
    from statsmodels.tsa.seasonal import STL

    if s <= 1:
        raise ValueError(
            "ERROR: Invalid period for the seasonal differencing test: {}".format(
                s
            )
        )

    y = check_array(
        y,
        dtype=("float32", "float64"),
        mem_type="host",
        ensure_all_finite=False,
        input_name="y",
    )
    n_obs, batch_size = y.shape

    threshold = 0.64
    results = []
    for i in range(batch_size):
        stlfit = STL(y[:, i], s).fit()
        seasonal = stlfit.seasonal
        residual = stlfit.resid
        heuristics = max(
            0, min(1, 1 - np.var(residual) / np.var(residual + seasonal))
        )
        results.append(heuristics > threshold)

    return cp.asarray(results)
