#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp

from cuml.internals import logger, mlfunc
from cuml.neighbors.nearest_neighbors_mg import NearestNeighborsMG

from cython.operator cimport dereference as deref
from libc.stdint cimport uintptr_t
from libcpp cimport bool
from libcpp.vector cimport vector
from pylibraft.common.handle cimport handle_t

from cuml.common.opg_data_utils_mg cimport PartDescriptor, floatData_t


cdef extern from "cuml/neighbors/knn_mg.hpp" namespace "ML::KNN::opg" nogil:

    cdef void knn_regress(
        handle_t &handle,
        vector[floatData_t*] *out,
        vector[floatData_t*] &idx_data,
        PartDescriptor &idx_desc,
        vector[floatData_t*] &query_data,
        PartDescriptor &query_desc,
        vector[vector[float*]] &y,
        bool rowMajorIndex,
        bool rowMajorQuery,
        int k,
        int n_outputs,
        size_t batch_size,
        bool verbose
    ) except +


class KNeighborsRegressorMG(NearestNeighborsMG):
    """
    Multi-node Multi-GPU K-Nearest Neighbors Regressor Model.

    K-Nearest Neighbors Regressor is an instance-based learning technique,
    that keeps training samples around for prediction, rather than trying
    to learn a generalizable set of model parameters.
    """
    @mlfunc(array_arg=None)
    def predict(
        self,
        index,
        index_parts_to_ranks,
        index_nrows,
        query,
        query_parts_to_ranks,
        query_nrows,
        ncols,
        n_outputs,
        rank,
    ):
        """
        Predict outputs for a query from previously stored index
        and index labels.
        The process is done in a multi-node multi-GPU fashion.

        Parameters
        ----------
        index: [__cuda_array_interface__] of local index partitions
        index_parts_to_ranks: mappings of index partitions to ranks
        index_nrows: number of index rows
        query: [__cuda_array_interface__] of local query partitions
        query_parts_to_ranks: mappings of query partitions to ranks
        query_nrows: number of query rows
        ncols: number of columns
        n_outputs: number of outputs columns
        rank: rank of current worker

        Returns
        -------
        predictions : labels
        """
        # Detect type
        self.get_out_type(index, query)

        # Build input arrays and descriptors for native code interfacing
        input = self.gen_local_input(
            index, index_parts_to_ranks, index_nrows, query,
            query_parts_to_ranks, query_nrows, ncols, rank)

        # Build input labels arrays and descriptors for native code interfacing
        labels = self.gen_local_labels(index, dtype='float32')

        local_query_rows = [x.shape[0] for x in input['arrays']['query']]

        # Build labels output array for native code interfacing
        cdef vector[floatData_t*] out_result_local_parts
        outputs = []
        for n_rows in local_query_rows:
            output = cp.zeros(shape=(n_rows, n_outputs), order="C", dtype='float32')
            outputs.append(output)
            out_result_local_parts.push_back(
                new floatData_t(
                    <float*><uintptr_t>output.data.ptr,
                    n_rows * n_outputs
                )
            )

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        is_verbose = logger.should_log_for(logger.level_enum.debug)

        # Launch distributed operations
        knn_regress(
            handle_[0],
            &out_result_local_parts,
            deref(<vector[floatData_t*]*><uintptr_t>
                  input['index']['local_parts']),
            deref(<PartDescriptor*><uintptr_t>input['index']['desc']),
            deref(<vector[floatData_t*]*><uintptr_t>
                  input['query']['local_parts']),
            deref(<PartDescriptor*><uintptr_t>input['query']['desc']),
            deref(<vector[vector[float*]]*><uintptr_t>labels['labels']),
            <bool>False,  # column-major index
            <bool>False,  # column-major query
            <int>self.n_neighbors,
            <int>n_outputs,
            <size_t>self.batch_size,
            <bool>is_verbose
        )
        self.handle.sync()

        # Release memory
        self.free_mem(input, labels=labels)
        cdef floatData_t *f_ptr
        for i in range(out_result_local_parts.size()):
            f_ptr = out_result_local_parts.at(i)
            del f_ptr

        return outputs
