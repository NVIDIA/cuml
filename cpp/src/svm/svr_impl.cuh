/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/** @file svr_impl.cuh
 * @brief Implementation of the stateless C++ functions to fit an SVM regressor.
 */

#include "kernelcache.cuh"
#include "smosolver.cuh"
#include "svc_impl.cuh"

#include <cuml/matrix/kernel_params.hpp>
#include <cuml/svm/svm_model.h>
#include <cuml/svm/svm_parameter.h>

#include <raft/core/handle.hpp>
#include <raft/linalg/unary_op.cuh>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>

#include <cublas_v2.h>
#include <cuvs/distance/distance.hpp>
#include <cuvs/distance/grammian.hpp>
#include <matrix/kernel_params.hpp>

#include <iostream>

namespace ML {
namespace SVM {

template <typename math_t, typename MatrixViewType>
int svrFitX(const raft::handle_t& handle,
            MatrixViewType matrix,
            int n_rows,
            int n_cols,
            math_t* y,
            const SvmParameter& param,
            ML::matrix::KernelParams& kernel_params,
            SvmModel<math_t>& model,
            const math_t* sample_weight)
{
  ASSERT(n_cols > 0, "Parameter n_cols: number of columns cannot be less than one");
  ASSERT(n_rows > 0, "Parameter n_rows: number of rows cannot be less than one");

  // KernelCache could use multiple streams, not implemented currently
  // See Issue #948.
  // ML::detail::streamSyncer _(handle_impl.getImpl());
  const raft::handle_t& handle_impl = handle;

  cudaStream_t stream = handle_impl.get_stream().get();

  bool is_precomputed = kernel_params.kernel == ML::matrix::KernelType::PRECOMPUTED;

  // For precomputed kernels, we don't need to create a cuvs kernel
  cuvs::distance::kernels::GramMatrixBase<math_t>* kernel = nullptr;
  cuvs::distance::kernels::KernelParams cuvs_params       = ML::matrix::to_cuvs(kernel_params);
  if (!is_precomputed) {
    kernel = cuvs::distance::kernels::KernelFactory<math_t>::create(cuvs_params);
  }

  SmoSolver<math_t> smo(handle_impl, param, cuvs_params.kernel, kernel, is_precomputed);
  smo.Solve(matrix,
            n_rows,
            n_cols,
            y,
            sample_weight,
            &(model.dual_coefs),
            &(model.n_support),
            &(model.support_matrix),
            &(model.support_idx),
            &(model.b),
            param.max_iter,
            param.max_outer_iter);
  model.n_cols = n_cols;
  if (kernel) delete kernel;
  return smo.GetNIter();
}

template <typename math_t>
int svrFit(const raft::handle_t& handle,
           math_t* X,
           int n_rows,
           int n_cols,
           math_t* y,
           const SvmParameter& param,
           ML::matrix::KernelParams& kernel_params,
           SvmModel<math_t>& model,
           const math_t* sample_weight)
{
  auto dense_view = raft::make_device_strided_matrix_view<math_t, int, raft::layout_f_contiguous>(
    X, n_rows, n_cols, 0);
  return svrFitX(handle, dense_view, n_rows, n_cols, y, param, kernel_params, model, sample_weight);
}

template <typename math_t>
int svrFitSparse(const raft::handle_t& handle,
                 int* indptr,
                 int* indices,
                 math_t* data,
                 int n_rows,
                 int n_cols,
                 int nnz,
                 math_t* y,
                 const SvmParameter& param,
                 ML::matrix::KernelParams& kernel_params,
                 SvmModel<math_t>& model,
                 const math_t* sample_weight)
{
  auto csr_structure_view = raft::make_device_compressed_structure_view<int, int, int>(
    indptr, indices, n_rows, n_cols, nnz);
  auto csr_matrix_view = raft::make_device_csr_matrix_view(data, csr_structure_view);
  return svrFitX(
    handle, csr_matrix_view, n_rows, n_cols, y, param, kernel_params, model, sample_weight);
}

};  // end namespace SVM
};  // end namespace ML
