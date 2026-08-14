#
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import cupyx.scipy.sparse as cp_sp
import numpy as np

import cuml.common.opg_data_utils_mg as opg
from cuml.internals.outputs import mlfunc
from cuml.internals.validation import check_inputs
from cuml.linear_model import LogisticRegression

from cython.operator cimport dereference as deref
from libc.stdint cimport int64_t, uintptr_t
from libcpp cimport bool
from libcpp.vector cimport vector
from pylibraft.common.handle cimport handle_t

from cuml.common.opg_data_utils_mg cimport (
    PartDescriptor,
    doubleData_t,
    floatData_t,
)
from cuml.solvers.qn cimport init_qn_params, qn_params


cdef extern from "cuml/linear_model/qn_mg.hpp" namespace "ML::GLM::opg" nogil:
    cdef void qnFit(
        handle_t& handle,
        vector[floatData_t *] input_data,
        PartDescriptor &input_desc,
        vector[floatData_t *] labels,
        float *coef,
        const qn_params& pams,
        bool X_col_major,
        bool standardization,
        int n_classes,
        float *f,
        int *num_iters) except +

    cdef void qnFit(
        handle_t& handle,
        vector[doubleData_t *] input_data,
        PartDescriptor &input_desc,
        vector[doubleData_t *] labels,
        double *coef,
        const qn_params& pams,
        bool X_col_major,
        bool standardization,
        int n_classes,
        double *f,
        int *num_iters) except +

    cdef vector[float] getUniquelabelsMG(
        const handle_t& handle,
        PartDescriptor &input_desc,
        vector[floatData_t*] labels) except+

    cdef vector[double] getUniquelabelsMG(
        const handle_t& handle,
        PartDescriptor &input_desc,
        vector[doubleData_t*] labels) except+

    cdef void qnFitSparse(
        handle_t& handle,
        vector[floatData_t *] input_values,
        int *input_cols,
        int *input_row_ids,
        int X_nnz,
        PartDescriptor &input_desc,
        vector[floatData_t *] labels,
        float *coef,
        const qn_params& pams,
        bool standardization,
        int n_classes,
        float *f,
        int *num_iters) except +

    cdef void qnFitSparse(
        handle_t& handle,
        vector[doubleData_t *] input_values,
        int *input_cols,
        int *input_row_ids,
        int X_nnz,
        PartDescriptor &input_desc,
        vector[doubleData_t *] labels,
        double *coef,
        const qn_params& pams,
        bool standardization,
        int n_classes,
        double *f,
        int *num_iters) except +

    cdef void qnFitSparse(
        handle_t& handle,
        vector[floatData_t *] input_values,
        int64_t *input_cols,
        int64_t *input_row_ids,
        int64_t X_nnz,
        PartDescriptor &input_desc,
        vector[floatData_t *] labels,
        float *coef,
        const qn_params& pams,
        bool standardization,
        int n_classes,
        float *f,
        int *num_iters) except +

    cdef void qnFitSparse(
        handle_t& handle,
        vector[doubleData_t *] input_values,
        int64_t *input_cols,
        int64_t *input_row_ids,
        int64_t X_nnz,
        PartDescriptor &input_desc,
        vector[doubleData_t *] labels,
        double *coef,
        const qn_params& pams,
        bool standardization,
        int n_classes,
        double *f,
        int *num_iters) except +


class LogisticRegressionMG(LogisticRegression):
    def __init__(self, *, handle, standardization=False, _convert_index=False, **kwargs):
        super().__init__(**kwargs)
        self.handle = handle
        self.standardization = standardization
        self._convert_index = _convert_index

    @mlfunc(convert_output=False)
    def fit(
        self,
        input_data,
        n_rows,
        n_cols,
        parts_rank_size,
        rank,
    ):
        # Validate and process input data
        if len(input_data) != 1:
            raise ValueError(
                f"Currently support only one (X, y) pair in the list. "
                f"Received {len(input_data)} pairs."
            )
        X, y = input_data[0]

        self._set_output_type(X)
        X, y = check_inputs(
            self,
            X,
            y,
            dtype=("float32", "float64"),
            order="C",
            accept_sparse="csr",
            accept_large_sparse=True,
            reset=True,
        )
        self.dtype = X.dtype

        cdef uintptr_t rank_to_sizes = opg.build_rank_size_pair(parts_rank_size)
        cdef uintptr_t input_desc = opg.build_part_descriptor(
            n_rows, n_cols, rank_to_sizes, rank
        )

        cdef bool X_is_f32 = X.dtype == np.float32
        cdef bool X_is_sparse = cp_sp.issparse(X)
        cdef bool X_index_is_i32 = False
        cdef uintptr_t y_ptr = opg.build_data_t([y])
        cdef uintptr_t X_ptr = 0, X_indices_ptr = 0, X_indptr_ptr = 0
        cdef int64_t X_nnz = 0

        if X_is_sparse:
            indices = X.indices
            indptr = X.indptr
            if self._convert_index:
                indices = indices.astype(self._convert_index, copy=False)
                indptr = indptr.astype(self._convert_index, copy=False)
            self.index_dtype = indices.dtype
            X_ptr = opg.build_data_t([X.data])
            X_indices_ptr = indices.data.ptr
            X_indptr_ptr = indptr.data.ptr
            X_nnz = X.nnz
            X_index_is_i32 = self.index_dtype == np.int32
        else:
            X_ptr = opg.build_data_t([X])

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        # Determine the number of classes in y
        if X.dtype == np.float32:
            classes = np.asarray(
                getUniquelabelsMG(
                    handle_[0],
                    deref(<PartDescriptor*>input_desc),
                    deref(<vector[floatData_t*]*>y_ptr)
                ),
                dtype=X.dtype,
            )
        else:
            classes = np.asarray(
                getUniquelabelsMG(
                    handle_[0],
                    deref(<PartDescriptor*>input_desc),
                    deref(<vector[doubleData_t*]*>y_ptr)
                ),
                dtype=X.dtype,
            )
        classes.sort()
        cdef int n_classes = len(classes)

        # QN's classification losses expect labels to be dense class indices
        # in [0, n_classes). Keep the original values in ``classes`` for
        # predictions, but encode the solver-facing labels using the global
        # class ordering shared by all ranks.
        y = cp.searchsorted(cp.asarray(classes, dtype=X.dtype), y).astype(
            X.dtype, copy=False
        )
        opg.free_data_t(y_ptr, X.dtype)
        y_ptr = opg.build_data_t([y])

        # Validate and initialize parameters
        l1_strength, l2_strength = self._get_l1_l2_strength()
        cdef qn_params params
        init_qn_params(
            params,
            n_classes=n_classes,
            loss=("softmax" if n_classes > 2 else "sigmoid"),
            fit_intercept=self.fit_intercept,
            l1_strength=l1_strength,
            l2_strength=l2_strength,
            max_iter=self.max_iter,
            tol=self.tol,
            delta=None,
            linesearch_max_iter=self.linesearch_max_iter,
            lbfgs_memory=self.lbfgs_memory,
            penalty_normalized=self.penalty_normalized,
            verbose=self._verbose_level,
        )

        # Allocate outputs
        coef_n_cols = n_classes if n_classes > 2 else 1
        coef_n_rows = n_cols + 1 if self.fit_intercept else n_cols
        coef = cp.zeros((coef_n_rows, coef_n_cols), dtype=X.dtype, order="C")

        # Prepare for fit
        cdef uintptr_t coef_ptr = coef.data.ptr
        cdef bool standardization = self.standardization
        cdef float objective_f32
        cdef double objective_f64
        cdef int n_iter

        # Perform fit
        with nogil:
            if not X_is_sparse:
                if X_is_f32:
                    qnFit(
                        handle_[0],
                        deref(<vector[floatData_t*]*>X_ptr),
                        deref(<PartDescriptor*>input_desc),
                        deref(<vector[floatData_t*]*>y_ptr),
                        <float*>coef_ptr,
                        params,
                        False,
                        standardization,
                        n_classes,
                        &objective_f32,
                        &n_iter
                    )
                else:
                    qnFit(
                        handle_[0],
                        deref(<vector[doubleData_t*]*>X_ptr),
                        deref(<PartDescriptor*>input_desc),
                        deref(<vector[doubleData_t*]*>y_ptr),
                        <double*>coef_ptr,
                        params,
                        False,
                        standardization,
                        n_classes,
                        &objective_f64,
                        &n_iter
                    )
            else:
                if X_is_f32:
                    if X_index_is_i32:
                        qnFitSparse(
                            handle_[0],
                            deref(<vector[floatData_t*]*>X_ptr),
                            <int*>X_indices_ptr,
                            <int*>X_indptr_ptr,
                            <int>X_nnz,
                            deref(<PartDescriptor*>input_desc),
                            deref(<vector[floatData_t*]*>y_ptr),
                            <float*>coef_ptr,
                            params,
                            standardization,
                            n_classes,
                            &objective_f32,
                            &n_iter,
                        )
                    else:
                        qnFitSparse(
                            handle_[0],
                            deref(<vector[floatData_t*]*>X_ptr),
                            <int64_t *>X_indices_ptr,
                            <int64_t *>X_indptr_ptr,
                            X_nnz,
                            deref(<PartDescriptor*>input_desc),
                            deref(<vector[floatData_t*]*>y_ptr),
                            <float*>coef_ptr,
                            params,
                            standardization,
                            n_classes,
                            &objective_f32,
                            &n_iter,
                        )
                else:
                    if X_index_is_i32:
                        qnFitSparse(
                            handle_[0],
                            deref(<vector[doubleData_t*]*>X_ptr),
                            <int*>X_indices_ptr,
                            <int*>X_indptr_ptr,
                            <int>X_nnz,
                            deref(<PartDescriptor*>input_desc),
                            deref(<vector[doubleData_t*]*>y_ptr),
                            <double*>coef_ptr,
                            params,
                            standardization,
                            n_classes,
                            &objective_f64,
                            &n_iter,
                        )
                    else:
                        qnFitSparse(
                            handle_[0],
                            deref(<vector[doubleData_t*]*>X_ptr),
                            <int64_t *>X_indices_ptr,
                            <int64_t *>X_indptr_ptr,
                            X_nnz,
                            deref(<PartDescriptor*>input_desc),
                            deref(<vector[doubleData_t*]*>y_ptr),
                            <double*>coef_ptr,
                            params,
                            standardization,
                            n_classes,
                            &objective_f64,
                            &n_iter,
                        )

        self.handle.sync()

        # Release memory resources
        opg.free_data_t(X_ptr, X.dtype)
        opg.free_data_t(y_ptr, X.dtype)
        opg.free_rank_size_pair(rank_to_sizes)
        opg.free_part_descriptor(input_desc)

        # Postprocess coef into coef_ and intercept_
        if self.fit_intercept:
            intercept = coef[-1]
            coef = coef[:-1].T
        else:
            if n_classes <= 2:
                intercept = cp.zeros(shape=1, dtype=X.dtype)
            else:
                intercept = cp.zeros(shape=n_classes, dtype=X.dtype)
            coef = coef.T

        # Store fitted attributes
        self.coef_ = coef
        self.intercept_ = intercept
        self.n_iter_ = np.array([n_iter])
        self.classes_ = classes

        return self
