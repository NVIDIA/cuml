#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp

from cuml.internals import logger, mlfunc
from cuml.internals.validation import check_array
from cuml.neighbors.nearest_neighbors_mg import NearestNeighborsMG

from cython.operator cimport dereference as deref
from libc.stdint cimport uintptr_t
from libcpp cimport bool
from libcpp.vector cimport vector
from pylibraft.common.handle cimport handle_t

from cuml.common.opg_data_utils_mg cimport (
    PartDescriptor,
    floatData_t,
    intData_t,
)


cdef extern from "cuml/neighbors/knn_mg.hpp" namespace "ML::KNN::opg" nogil:

    cdef void knn_classify(
        handle_t &handle,
        vector[intData_t*] *out,
        vector[vector[float*]] *probas,
        vector[floatData_t*] &idx_data,
        PartDescriptor &idx_desc,
        vector[floatData_t*] &query_data,
        PartDescriptor &query_desc,
        vector[vector[int*]] &y,
        vector[int*] &uniq_labels,
        vector[int] &n_unique,
        bool rowMajorIndex,
        bool rowMajorQuery,
        bool probas_only,
        int k,
        size_t batch_size,
        bool verbose
    ) except +


class KNeighborsClassifierMG(NearestNeighborsMG):
    """
    Multi-node Multi-GPU K-Nearest Neighbors Classifier Model.

    K-Nearest Neighbors Classifier is an instance-based learning technique,
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
        uniq_labels,
        n_unique,
        ncols,
        rank,
    ):
        """
        Predict labels for a query from previously stored index
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
        uniq_labels: array of arrays of possible labels for columns
        n_unique: array with number of possible labels for each columns
        ncols: number of columns
        rank: rank of current worker
        n_neighbors: number of nearest neighbors to query

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
        labels = self.gen_local_labels(index, 'int32')

        local_query_rows = [x.shape[0] for x in input['arrays']['query']]

        # Build uniq_labels_vec vector for native code interfacing
        uniq_labels_d = check_array(
            uniq_labels, dtype="int32", order="C", ensure_2d=False
        )
        cdef int* ptr = <int*><uintptr_t>uniq_labels_d.data.ptr
        cdef vector[int*] uniq_labels_vec
        for i in range(uniq_labels_d.shape[0]):
            uniq_labels_vec.push_back(<int*>ptr)
            ptr += <int>uniq_labels_d.shape[1]

        # Build n_unique_vec vector for native code interfacing
        cdef vector[int] n_unique_vec
        for uniq_label in n_unique:
            n_unique_vec.push_back(uniq_label)

        n_outputs = len(n_unique)

        # Build labels output array for native code interfacing
        cdef vector[intData_t*] out_result_local_parts
        outputs = []
        for n_rows in local_query_rows:
            output = cp.zeros(shape=(n_rows, n_outputs), order="C", dtype='int32')
            outputs.append(output)
            out_result_local_parts.push_back(
                new intData_t(
                    <int*><uintptr_t>output.data.ptr,
                    n_rows * n_outputs
                )
            )

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        is_verbose = logger.should_log_for(logger.level_enum.debug)
        knn_classify(
            handle_[0],
            &out_result_local_parts,
            <vector[vector[float*]]*>0,
            deref(<vector[floatData_t*]*><uintptr_t>
                  input['index']['local_parts']),
            deref(<PartDescriptor*><uintptr_t>input['index']['desc']),
            deref(<vector[floatData_t*]*><uintptr_t>
                  input['query']['local_parts']),
            deref(<PartDescriptor*><uintptr_t>input['query']['desc']),
            deref(<vector[vector[int*]]*><uintptr_t>labels['labels']),
            uniq_labels_vec,
            n_unique_vec,
            False,  # column-major index
            False,  # column-major query
            False,
            <int>self.n_neighbors,
            <size_t>self.batch_size,
            is_verbose
        )

        self.handle.sync()

        # Release memory
        self.free_mem(input, labels=labels)
        cdef intData_t *i_ptr
        for i in range(out_result_local_parts.size()):
            i_ptr = out_result_local_parts.at(i)
            del i_ptr

        return outputs

    @mlfunc(array_arg=None)
    def predict_proba(self, index, index_parts_to_ranks, index_nrows,
                      query, query_parts_to_ranks, query_nrows,
                      uniq_labels, n_unique, ncols, rank) -> tuple:
        """
        Predict labels for a query from previously stored index
        and index labels.
        The process is done in a multi-node multi-GPU fashion.

        Parameters
        ----------
        index: [__cuda_array_interface__] of local index and labels partitions
        index_parts_to_ranks: mappings of index partitions to ranks
        index_nrows: number of total index rows
        query: [__cuda_array_interface__] of local query partitions
        query_parts_to_ranks: mappings of query partitions to ranks
        query_nrows: number of total query rows
        uniq_labels: array of labels of a column
        n_unique: array with number of possible labels for each columns
        ncols: number of columns
        rank: int rank of current worker

        Returns
        -------
        predictions : labels, indices, distances
        """
        # Detect type
        self.get_out_type(index, query)

        # Build input arrays and descriptors for native code interfacing
        input = self.gen_local_input(
            index, index_parts_to_ranks, index_nrows, query,
            query_parts_to_ranks, query_nrows, ncols, rank)

        # Build input labels arrays and descriptors for native code interfacing
        labels = self.gen_local_labels(index, dtype='int32')

        # Build uniq_labels_vec vector for native code interfacing
        uniq_labels_d = check_array(
            uniq_labels, dtype="int32", order="C", ensure_2d=False,
        )
        cdef int* ptr = <int*><uintptr_t>uniq_labels_d.data.ptr
        cdef vector[int*] uniq_labels_vec
        for i in range(uniq_labels_d.shape[0]):
            uniq_labels_vec.push_back(<int*>ptr)
            ptr += <int>uniq_labels_d.shape[1]

        # Build n_unique_vec vector for native code interfacing
        cdef vector[int] n_unique_vec
        for uniq_label in n_unique:
            n_unique_vec.push_back(uniq_label)

        local_query_rows = [x.shape[0] for x in input['arrays']['query']]
        n_outputs = len(n_unique)

        # Build probas output array for native code interfacing
        outputs = [[] for i in range(n_outputs)]
        cdef vector[vector[float*]] probas_local_parts
        probas_local_parts.resize(len(local_query_rows))
        for query_idx, n_rows in enumerate(local_query_rows):
            for target_idx, n_classes in enumerate(n_unique):
                output = cp.zeros(
                    shape=(n_rows, n_classes), order="C", dtype='float32'
                )
                outputs[target_idx].append(output)
                probas_local_parts.at(query_idx).push_back(
                    <float*><uintptr_t>output.data.ptr
                )

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        is_verbose = logger.should_log_for(logger.level_enum.debug)

        # Launch distributed operations
        knn_classify(
            handle_[0],
            <vector[intData_t*]*>0,
            &probas_local_parts,
            deref(<vector[floatData_t*]*><uintptr_t>
                  input['index']['local_parts']),
            deref(<PartDescriptor*><uintptr_t>input['index']['desc']),
            deref(<vector[floatData_t*]*><uintptr_t>
                  input['query']['local_parts']),
            deref(<PartDescriptor*><uintptr_t>input['query']['desc']),
            deref(<vector[vector[int*]]*><uintptr_t>labels['labels']),
            uniq_labels_vec,
            n_unique_vec,
            False,  # column-major index
            False,  # column-major query
            True,
            <int>self.n_neighbors,
            <size_t>self.batch_size,
            is_verbose
        )
        self.handle.sync()

        # Release memory
        self.free_mem(input, labels=labels)
        return tuple(outputs)
