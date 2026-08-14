# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import cupy as cp
import numpy as np

from cuml.internals import get_handle, mlfunc
from cuml.internals.validation import check_array
from cuml.tsa._deprecation import deprecated_tsa_api

from libc.stdint cimport uintptr_t
from libcpp cimport bool as boolcpp
from pylibraft.common.handle cimport handle_t


cdef extern from "cuml/tsa/stationarity.h" namespace "ML" nogil:
    int cpp_kpss "ML::Stationarity::kpss_test" (
        const handle_t& handle,
        const float* d_y,
        boolcpp* results,
        int batch_size,
        int n_obs,
        int d, int D, int s,
        float pval_threshold) except +

    int cpp_kpss "ML::Stationarity::kpss_test" (
        const handle_t& handle,
        const double* d_y,
        boolcpp* results,
        int batch_size,
        int n_obs,
        int d, int D, int s,
        double pval_threshold) except +


@deprecated_tsa_api("cuml.tsa.stationarity.kpss_test")
@mlfunc
def kpss_test(
    y,
    int d=0,
    int D=0,
    int s=0,
    double pval_threshold=0.05,
):
    """
    Perform the KPSS stationarity test on the data differenced according
    to the given order

    .. deprecated:: 26.08
        ``cuml.tsa.stationarity.kpss_test`` is deprecated and will be removed
        in the cuML 26.12 release.

    Parameters
    ----------
    y : dataframe or array-like (device or host)
        The time series data, assumed to have each time series in columns.
        Acceptable formats: cuDF DataFrame, cuDF Series, NumPy ndarray,
        Numba device ndarray, cuda array interface compliant array like CuPy.
    d: integer
        Order of simple differencing
    D: integer
        Order of seasonal differencing
    s: integer
        Seasonal period if D > 0
    pval_threshold : float
        The p-value threshold above which a series is considered stationary.

    Returns
    -------
    stationarity : List[bool]
        A list of the stationarity test result for each series in the batch
    """
    d_y = check_array(
        y,
        dtype=("float32", "float64"),
        order="F",
        input_name="y",
        ensure_all_finite=False,
    )
    cdef int n_obs = d_y.shape[0]
    cdef int batch_size = d_y.shape[1]

    cdef uintptr_t d_y_ptr = d_y.data.ptr

    handle = get_handle()
    cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

    results = cp.empty(batch_size, dtype=bool)
    cdef uintptr_t d_results = results.data.ptr

    # Call C++ function
    if d_y.dtype == np.float32:
        cpp_kpss(
            handle_[0],
            <float*> d_y_ptr,
            <boolcpp*> d_results,
            batch_size,
            n_obs,
            d,
            D,
            s,
            <float> pval_threshold
        )
    elif d_y.dtype == np.float64:
        cpp_kpss(
            handle_[0],
            <double*> d_y_ptr,
            <boolcpp*> d_results,
            batch_size,
            n_obs,
            d,
            D,
            s,
            pval_threshold
        )

    return results
