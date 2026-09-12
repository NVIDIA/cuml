#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp

from cuml.internals import get_handle

from libc.stddef cimport size_t
from libc.stdint cimport int64_t, uint64_t, uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t


cdef extern from "cuml/datasets/make_blobs.hpp" namespace "ML" nogil:
    void cpp_make_blobs "ML::Datasets::make_blobs" (
        const handle_t& handle,
        float* out,
        int64_t* labels,
        int64_t n_rows,
        int64_t n_cols,
        int64_t n_clusters,
        bool row_major,
        const float* centers,
        const float* cluster_std,
        const float cluster_std_scalar,
        bool shuffle,
        float center_box_min,
        float center_box_max,
        uint64_t seed) except +

    void cpp_make_blobs "ML::Datasets::make_blobs" (
        const handle_t& handle,
        double* out,
        int64_t* labels,
        int64_t n_rows,
        int64_t n_cols,
        int64_t n_clusters,
        bool row_major,
        const double* centers,
        const double* cluster_std,
        const double cluster_std_scalar,
        bool shuffle,
        double center_box_min,
        double center_box_max,
        uint64_t seed) except +


def make_blobs(
    n_samples,
    n_features,
    n_centers,
    centers,
    cluster_std,
    center_box_min,
    center_box_max,
    shuffle,
    random_state,
    order,
    dtype,
):
    dtype = cp.dtype(dtype)

    h = get_handle()
    cdef handle_t* h_ptr = <handle_t*><size_t>h.getHandle()

    X = cp.empty((n_samples, n_features), dtype=dtype, order=order)
    y = cp.empty(n_samples, dtype=cp.int64)

    cdef uintptr_t x_p = X.data.ptr
    cdef uintptr_t y_p = y.data.ptr
    cdef uintptr_t ctr_p = 0
    if centers is not None:
        ctr_p = centers.data.ptr

    cdef bool row_c = order == "C"

    if dtype == cp.dtype("float32"):
        cpp_make_blobs(
            h_ptr[0],
            <float*>x_p,
            <int64_t*>y_p,
            <int64_t>n_samples,
            <int64_t>n_features,
            <int64_t>n_centers,
            row_c,
            <const float*>ctr_p,
            <const float*>0,
            <float>cluster_std,
            <bool>shuffle,
            <float>center_box_min,
            <float>center_box_max,
            <uint64_t>random_state,
        )
    elif dtype == cp.dtype("float64"):
        cpp_make_blobs(
            h_ptr[0],
            <double*>x_p,
            <int64_t*>y_p,
            <int64_t>n_samples,
            <int64_t>n_features,
            <int64_t>n_centers,
            row_c,
            <const double*>ctr_p,
            <const double*>0,
            <double>cluster_std,
            <bool>shuffle,
            <double>center_box_min,
            <double>center_box_max,
            <uint64_t>random_state,
        )
    else:
        raise ValueError("RAFT make_blobs only supports float32 and float64.")

    return X, y.astype(dtype, copy=False)
