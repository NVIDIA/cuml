#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np

from cuml.internals import get_handle
from cuml.internals.validation import check_array

from libc.stdint cimport uintptr_t
from pylibraft.common.handle cimport handle_t


cdef extern from "cuml/metrics/metrics.hpp" namespace "ML::Metrics" nogil:
    double c_kl_divergence "ML::Metrics::kl_divergence"(
        const handle_t &handle,
        const double *y,
        const double *y_hat,
        int n) except +
    float c_kl_divergence "ML::Metrics::kl_divergence"(
        const handle_t &handle,
        const float *y,
        const float *y_hat,
        int n) except +


def kl_divergence(P, Q):
    """
    Calculates the "Kullback-Leibler" Divergence
    The KL divergence tells us how well the probability distribution Q
    approximates the probability distribution P
    It is often also used as a 'distance metric' between two probability
    distributions (not symmetric)

    Parameters
    ----------
    P : Dense array of probabilities corresponding to distribution P
        shape = (n_samples, 1)
        Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
        ndarray, cuda array interface compliant array like CuPy.

    Q : Dense array of probabilities corresponding to distribution Q
        shape = (n_samples, 1)
        Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
        ndarray, cuda array interface compliant array like CuPy.

    Returns
    -------
    float
        The KL Divergence value
    """
    handle = get_handle()
    cdef handle_t *handle_ = <handle_t*> <size_t> handle.getHandle()

    P_m = check_array(
        P,
        ensure_2d=False,
        order='C',
        dtype=[np.float32, np.float64],
        input_name='P',
    )
    if P_m.ndim == 2 and P_m.shape[1] != 1:
        raise ValueError(
            "P must have shape (n_samples,) or (n_samples, 1), got "
            f"{P_m.shape}"
        )
    P_m = P_m.ravel()
    dtype_p = P_m.dtype

    Q_m = check_array(
        Q,
        ensure_2d=False,
        order='C',
        dtype=[dtype_p],
        input_name='Q',
    )
    if Q_m.ndim == 2 and Q_m.shape[1] != 1:
        raise ValueError(
            "Q must have shape (n_samples,) or (n_samples, 1), got "
            f"{Q_m.shape}"
        )
    Q_m = Q_m.ravel()

    cdef int n_features_p = P_m.shape[0]
    if Q_m.shape[0] != n_features_p:
        raise ValueError(
            "Incompatible dimension for P and Q arrays: "
            f"P.shape == ({n_features_p},) while Q.shape == ({Q_m.shape[0]},)"
        )

    cdef uintptr_t d_P_ptr = P_m.data.ptr
    cdef uintptr_t d_Q_ptr = Q_m.data.ptr

    if (dtype_p == np.float32):
        res = c_kl_divergence(handle_[0],
                              <float*> d_P_ptr,
                              <float*> d_Q_ptr,
                              n_features_p)
    else:
        res = c_kl_divergence(handle_[0],
                              <double*> d_P_ptr,
                              <double*> d_Q_ptr,
                              n_features_p)

    return res
