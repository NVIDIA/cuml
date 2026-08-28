#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np

from cython.operator cimport dereference as deref
from libc.stdint cimport uintptr_t
from libcpp.vector cimport vector

from cuml.common.opg_data_utils_mg cimport (
    PartDescriptor,
    RankSizePair,
    doubleData_t,
    floatData_t,
)


def build_data_t(parts):
    """
    Build a ``vector<floatData_t*>`` or ``vector<doubleData_t*>``

    Parameters
    ----------
    parts: list[cp.ndarray]
        A list of cupy arrays, all with the same dtype (float32 or float64).

    Returns
    -------
    ptr: int
        Pointer to a ``vector<floatData_t*>`` or ``vector<doubleData_t*>``.
    """
    cdef vector[floatData_t *] *data_f32
    cdef vector[doubleData_t *] *data_f64

    if parts[0].dtype == np.float32:
        data_f32 = new vector[floatData_t *]()
        for part in parts:
            data_f32.push_back(
                new floatData_t(<float*><uintptr_t>(part.data.ptr), part.size)
            )
        return <uintptr_t> data_f32

    elif parts[0].dtype == np.float64:
        data_f64 = new vector[doubleData_t *]()
        for part in parts:
            data_f64.push_back(
                new doubleData_t(<double*><uintptr_t>(part.data.ptr), part.size)
            )
        return <uintptr_t> data_f64

    raise TypeError("Arrays passed must be np.float32 or np.float64")


def free_data_t(uintptr_t data_ptr, dtype):
    """
    Free a ``vector<floatData_t*>`` or ``vector<doubleData_t*>``

    Parameters
    ----------
    data_ptr: int
        A pointer to a ``vector<floatData_t*>`` or ``vector<doubleData_t*>``.
    dtype: dtype
        The dtype (float32 or float64).
    """
    cdef vector[floatData_t*] *d32
    cdef vector[doubleData_t*] *d64
    cdef floatData_t* ptr_32
    cdef doubleData_t* ptr_64

    if dtype == np.float32:
        d32 = <vector[floatData_t*]*> data_ptr
        for x_i in range(d32.size()):
            ptr_32 = d32.at(x_i)
            del ptr_32
        del d32
    else:
        d64 = <vector[doubleData_t*]*> data_ptr
        for x_i in range(d64.size()):
            ptr_64 = d64.at(x_i)
            del ptr_64
        del d64


def build_rank_size_pair(parts_to_sizes):
    """
    Build a ``vector<rankSizePair*>`` mapping ranks to partition sizes

    Parameters
    ----------
    parts_to_sizes: list[tuple[int, int]]
        List of tuples in the format: [(rank,size)]

    Returns
    --------
    ptr: int
        Pointer to a ``vector<RankSizePair*>``
    """
    cdef vector[RankSizePair*] *rsp_vec = new vector[RankSizePair*]()

    for (rank, size) in parts_to_sizes:
        rsp_vec.push_back(new RankSizePair(rank, size))

    return <uintptr_t>rsp_vec


def free_rank_size_pair(uintptr_t rank_size_ptr):
    """
    Free a ``vector<RankSizePair*>``

    Parameters
    ----------
    rank_size_ptr: int
        Pointer to a ``vector<RankSizePair*>``
    """
    cdef vector[RankSizePair*] *rsp_vec = <vector[RankSizePair*]*>rank_size_ptr
    cdef RankSizePair *rsp_ptr

    for x_i in range(rsp_vec.size()):
        rsp_ptr = rsp_vec.at(x_i)
        del rsp_ptr
    del rsp_vec


def build_part_descriptor(m, n, uintptr_t rank_size_ptr, rank):
    """
    Build a ``PartDescriptor``

    Parameters
    ----------
    m: int
        Total number of rows across all workers
    n: int
        Number of cols
    rank_size_ptr: int
        Pointer to a ``vector<RankSizePair*>``.
    rank: int
        Rank to be mapped

    Returns
    --------
    ptr: int
        Pointer to a ``PartDescriptor``.
    """
    cdef vector[RankSizePair *] *rsp_vec = <vector[RankSizePair *]*>rank_size_ptr

    cdef PartDescriptor *descriptor = new PartDescriptor(
        <size_t>m,
        <size_t>n,
        <vector[RankSizePair*]>deref(rsp_vec),
        <int>rank,
    )
    return <uintptr_t>descriptor


def free_part_descriptor(uintptr_t descriptor_ptr):
    """
    Free a ``PartDescriptor``

    Parameters
    ----------
    descriptor_ptr: int
        Pointer to a ``PartDescriptor``.
    """
    cdef PartDescriptor *desc_c = <PartDescriptor*>descriptor_ptr
    del desc_c
