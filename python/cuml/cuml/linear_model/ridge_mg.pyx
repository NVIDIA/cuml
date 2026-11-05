#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import cupy as cp

from cuml.internals.param_validation import Float, Options
from cuml.linear_model import Ridge
from cuml.linear_model.base_mg import MGFitMixin

from cython.operator cimport dereference as deref
from libc.stdint cimport uintptr_t
from libcpp cimport bool
from libcpp.vector cimport vector
from pylibraft.common.handle cimport handle_t

from cuml.common.opg_data_utils_mg cimport (
    PartDescriptor,
    doubleData_t,
    floatData_t,
)


cdef extern from "cuml/linear_model/ridge_mg.hpp" namespace "ML::Ridge::opg" nogil:

    cdef void fit(handle_t& handle,
                  vector[floatData_t *] input_data,
                  PartDescriptor &input_desc,
                  vector[floatData_t *] labels,
                  float *alpha,
                  int n_alpha,
                  float *coef,
                  float *intercept,
                  bool fit_intercept,
                  int algo,
                  bool verbose) except +

    cdef void fit(handle_t& handle,
                  vector[doubleData_t *] input_data,
                  PartDescriptor &input_desc,
                  vector[doubleData_t *] labels,
                  double *alpha,
                  int n_alpha,
                  double *coef,
                  double *intercept,
                  bool fit_intercept,
                  int algo,
                  bool verbose) except +


class RidgeMG(MGFitMixin, Ridge):
    # The multi-GPU implementation supports a subset of Ridge's options
    _parameter_constraints = {
        "alpha": [Float(ge=0)],
        "solver": [Options("auto", "eig", "svd")],
    }

    def _fit(
        self,
        uintptr_t X_ptr,
        uintptr_t y_ptr,
        n_cols,
        dtype,
        uintptr_t input_desc_ptr,
    ):
        solver = self.solver
        if solver == "eig":
            if n_cols == 1:
                raise ValueError(
                    "solver='eig' doesn't support X with 1 column, please select "
                    "solver='svd' instead"
                )
        elif solver == "auto":
            solver = "svd" if n_cols == 1 else "eig"

        cdef int algo = {"svd": 0, "eig": 1}[solver]

        coef = cp.zeros(n_cols, dtype=dtype)
        cdef uintptr_t coef_ptr = coef.data.ptr

        cdef float intercept_f32
        cdef double intercept_f64
        cdef float alpha_f32 = self.alpha
        cdef double alpha_f64 = self.alpha
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        cdef bool use_float32 = dtype == cp.float32

        if use_float32:
            fit(handle_[0],
                deref(<vector[floatData_t*]*>X_ptr),
                deref(<PartDescriptor*>input_desc_ptr),
                deref(<vector[floatData_t*]*>y_ptr),
                &alpha_f32,
                1,
                <float*>coef_ptr,
                <float*>&intercept_f32,
                <bool>self.fit_intercept,
                algo,
                False)
        else:
            fit(handle_[0],
                deref(<vector[doubleData_t*]*>X_ptr),
                deref(<PartDescriptor*>input_desc_ptr),
                deref(<vector[doubleData_t*]*>y_ptr),
                &alpha_f64,
                1,
                <double*>coef_ptr,
                <double*>&intercept_f64,
                <bool>self.fit_intercept,
                algo,
                False)

        self.handle.sync()

        self.solver_ = solver
        self.intercept_ = (intercept_f32 if use_float32 else intercept_f64)
        self.coef_ = coef
