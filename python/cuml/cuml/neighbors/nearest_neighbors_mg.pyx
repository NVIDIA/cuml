#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp

from cuml.internals import logger, mlfunc
from cuml.internals.validation import check_array
from cuml.neighbors import NearestNeighbors

from cython.operator cimport dereference as deref
from libc.stdint cimport int64_t, uintptr_t
from libcpp cimport bool
from libcpp.vector cimport vector
from pylibraft.common.handle cimport handle_t

from cuml.common.opg_data_utils_mg cimport (
    PartDescriptor,
    RankSizePair,
    floatData_t,
    int64Data_t,
)


cdef extern from "cuml/neighbors/knn_mg.hpp" namespace "ML::KNN::opg" nogil:

    cdef void knn(
        handle_t &handle,
        vector[int64Data_t*] *out_I,
        vector[floatData_t*] *out_D,
        vector[floatData_t*] &idx_data,
        PartDescriptor &idx_desc,
        vector[floatData_t*] &query_data,
        PartDescriptor &query_desc,
        bool rowMajorIndex,
        bool rowMajorQuery,
        int k,
        size_t batch_size,
        bool verbose
    ) except +


def _build_part_inputs(arrays, parts_to_ranks, m, n, local_rank):
    cupy_arrays = [
        check_array(array, order="F", dtype="float32") for array in arrays
    ]

    cdef vector[floatData_t*] *local_parts = new vector[floatData_t*]()
    for arr in cupy_arrays:
        local_parts.push_back(
            new floatData_t(<float*><uintptr_t>arr.data.ptr, arr.size)
        )

    cdef vector[RankSizePair*] parts_to_ranks_vec
    for (rank, size) in parts_to_ranks:
        parts_to_ranks_vec.push_back(new RankSizePair(rank, size))

    cdef PartDescriptor *descriptor = new PartDescriptor(
        m, n, parts_to_ranks_vec, local_rank
    )

    return cupy_arrays, <uintptr_t>local_parts, <uintptr_t>descriptor


class NearestNeighborsMG(NearestNeighbors):
    """
    Multi-node multi-GPU Nearest Neighbors kneighbors query.

    NOTE: This implementation of NearestNeighbors is meant to be
    used with an initialized cumlCommunicator instance inside an
    existing distributed system. Refer to the Dask NearestNeighbors
     implementation in `cuml.dask.neighbors.nearest_neighbors`.

    The end-user API for multi-node multi-GPU NearestNeighbors is
    `cuml.dask.neighbors.NearestNeighbors`
    """
    def __init__(self, *, handle, batch_size=2000000, **kwargs):
        self.handle = handle
        self.batch_size = batch_size
        super().__init__(**kwargs)

    @mlfunc(array_arg=None)
    def kneighbors(
        self,
        index,
        index_parts_to_ranks,
        index_nrows,
        query,
        query_parts_to_ranks,
        query_nrows,
        ncols,
        rank,
        n_neighbors,
    ):
        """
        Query the kneighbors of an index

        Parameters
        ----------
        index: [__cuda_array_interface__] of local index partitions
        index_parts_to_ranks: mappings of index partitions to ranks
        index_nrows: number of index rows
        query: [__cuda_array_interface__] of local query partitions
        query_parts_to_ranks: mappings of query partitions to ranks
        query_nrows: number of query rows
        ncols: number of columns
        rank: rank of current worker
        n_neighbors: number of nearest neighbors to query

        Returns
        -------
        predictions : indices and distances
        """
        # Detect type
        self.get_out_type(index, query)

        self.n_neighbors = self.n_neighbors if n_neighbors is None else n_neighbors

        # Build input arrays and descriptors for native code interfacing
        input = self.gen_local_input(
            index, index_parts_to_ranks, index_nrows, query,
            query_parts_to_ranks, query_nrows, ncols, rank)

        query_arrays = input['arrays']['query']
        local_query_rows = [x.shape[0] for x in query_arrays]

        # Build indices and distances outputs for native code interfacing
        result = self.alloc_local_output(local_query_rows, self.n_neighbors)

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        is_verbose = logger.should_log_for(logger.level_enum.debug)

        # Launch distributed operations
        knn(
            handle_[0],
            <vector[int64Data_t*]*><uintptr_t>result['indices'],
            <vector[floatData_t*]*><uintptr_t>result['distances'],
            deref(<vector[floatData_t*]*><uintptr_t>
                  input['index']['local_parts']),
            deref(<PartDescriptor*><uintptr_t>input['index']['desc']),
            deref(<vector[floatData_t*]*><uintptr_t>
                  input['query']['local_parts']),
            deref(<PartDescriptor*><uintptr_t>input['query']['desc']),
            <bool>False,  # column-major index
            <bool>False,  # column-major query
            <int>self.n_neighbors,
            <size_t>self.batch_size,
            <bool>is_verbose
        )
        self.handle.sync()

        # Release memory
        self.free_mem(input, result)

        return result['arrays']['distances'], result['arrays']['indices']

    def get_out_type(self, index, query):
        if len(index) > 0:
            self._set_output_type(index[0])
        if len(query) > 0:
            self._set_output_type(query[0])

    @staticmethod
    def gen_local_input(index, index_parts_to_ranks, index_nrows,
                        query, query_parts_to_ranks, query_nrows,
                        ncols, rank):
        index_dask = [d[0] if isinstance(d, (list, tuple))
                      else d for d in index]

        index_arrays, index_local_parts, index_desc = _build_part_inputs(
            index_dask, index_parts_to_ranks, index_nrows, ncols, rank
        )

        query_arrays, query_local_parts, query_desc = _build_part_inputs(
            query, query_parts_to_ranks, query_nrows, ncols, rank
        )

        return {
            'index': {
                'local_parts': <uintptr_t>index_local_parts,
                'desc': <uintptr_t>index_desc
            },
            'query': {
                'local_parts': <uintptr_t>query_local_parts,
                'desc': <uintptr_t>query_desc
            },
            'arrays': {
                'index': index_arrays,
                'query': query_arrays
            },
        }

    @staticmethod
    def gen_local_labels(index, dtype):
        cdef vector[vector[int*]] *out_local_parts_i32
        cdef vector[vector[float*]] *out_local_parts_f32

        outputs = [d[1] for d in index]
        n_out = len(outputs)

        if dtype == 'int32':
            out_local_parts_i32 = new vector[vector[int*]](<int>n_out)
        elif dtype == 'float32':
            out_local_parts_f32 = new vector[vector[float*]](<int>n_out)
        else:
            raise ValueError('Wrong dtype')

        output_arrays = []
        for i, arr in enumerate(outputs):
            arr = check_array(
                arr,
                dtype=dtype,
                order="F",
                ensure_2d=False,
            )
            n_cols = arr.shape[1] if arr.ndim != 1 else 1
            for j in range(n_cols):
                col = arr[:, j] if n_cols != 1 else arr
                output_arrays.append(col)
                if dtype == 'int32':
                    out_local_parts_i32.at(i).push_back(
                        <int*><uintptr_t>col.data.ptr
                    )
                else:
                    out_local_parts_f32.at(i).push_back(
                        <float*><uintptr_t>col.data.ptr
                    )

        return {
            'labels':
                <uintptr_t>out_local_parts_i32 if dtype == 'int32'
                else <uintptr_t>out_local_parts_f32,
            'arrays': output_arrays,
            'dtype': dtype
        }

    @staticmethod
    def alloc_local_output(local_query_rows, n_neighbors):
        cdef vector[int64Data_t*] *indices_local_parts = new vector[int64Data_t*]()
        cdef vector[floatData_t*] *distances_local_parts = new vector[floatData_t*]()

        indices_arrays = []
        distances_arrays = []
        for n_rows in local_query_rows:
            indices = cp.zeros(
                shape=(n_rows, n_neighbors), order="C", dtype='int64'
            )
            distances = cp.zeros(
                shape=(n_rows, n_neighbors), order="C", dtype='float32'
            )
            indices_arrays.append(indices)
            distances_arrays.append(distances)

            indices_local_parts.push_back(
                new int64Data_t(
                    <int64_t*><uintptr_t>indices.data.ptr,
                    n_rows * n_neighbors
                )
            )

            distances_local_parts.push_back(
                new floatData_t(
                    <float*><uintptr_t>distances.data.ptr,
                    n_rows * n_neighbors,
                )
            )

        return {
            'indices': <uintptr_t>indices_local_parts,
            'distances': <uintptr_t>distances_local_parts,
            'arrays': {
                'indices': indices_arrays,
                'distances': distances_arrays,
            }
        }

    @staticmethod
    def free_mem(input, result=None, labels=None):
        cdef floatData_t *f_ptr
        cdef vector[floatData_t*] *f_lp
        cdef PartDescriptor *desc_ptr
        cdef vector[vector[int*]] *labels_i32
        cdef vector[vector[float*]] *labels_f32

        for input_type in ['index', 'query']:
            ilp = input[input_type]['local_parts']
            f_lp = <vector[floatData_t *]*><uintptr_t>ilp
            for i in range(f_lp.size()):
                f_ptr = f_lp.at(i)
                del f_ptr
            del f_lp

            desc_ptr = <PartDescriptor *><uintptr_t>input[input_type]['desc']
            del desc_ptr

        cdef int64Data_t *i64_ptr
        cdef vector[int64Data_t*] *i64_lp

        if result is not None:
            f_lp = <vector[floatData_t *]*><uintptr_t>result['distances']
            for i in range(f_lp.size()):
                f_ptr = f_lp.at(i)
                del f_ptr
            del f_lp

            i64_lp = <vector[int64Data_t *]*><uintptr_t>result['indices']
            for i in range(i64_lp.size()):
                i64_ptr = i64_lp.at(i)
                del i64_ptr
            del i64_lp

        if labels is not None:
            if labels['dtype'] == "int32":
                labels_i32 = <vector[vector[int*]]*><uintptr_t>labels['labels']
                del labels_i32
            else:
                labels_f32 = <vector[vector[float*]]*><uintptr_t>labels['labels']
                del labels_f32
