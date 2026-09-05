/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/** @file svc_impl.cuh
 * @brief Implementation of the stateless C++ functions to fit an SVM
 * classifier, and predict with it.
 */

#include "kernelcache.cuh"
#include "smosolver.cuh"

#include <cuml/common/checked_arithmetic.hpp>
#include <cuml/matrix/kernel_params.hpp>
#include <cuml/svm/svm_model.h>
#include <cuml/svm/svm_parameter.h>

#include <raft/core/handle.hpp>
#include <raft/label/classlabels.cuh>
#include <raft/linalg/gemv.cuh>

#include <rmm/aligned.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

#include <cublas_v2.h>
#include <cuvs/distance/distance.hpp>
#include <cuvs/distance/grammian.hpp>
#include <matrix/kernel_params.hpp>

#include <iostream>

namespace ML {
namespace SVM {

/**
 * @brief Compute decision function values for a batch: y_batch = K @ dual_coefs
 *
 * @param[in] handle       cuML handle
 * @param[in] K            Kernel matrix (batch_size x n_support or n_support x batch_size if
 * transposed)
 * @param[in] dual_coefs   Dual coefficients (size n_support)
 * @param[in] batch_size   Number of samples in the batch
 * @param[in] n_support    Number of support vectors
 * @param[in] transpose    Whether K is transposed (n_support x batch_size instead of batch_size x
 * n_support)
 * @param[out] y_batch     Output decision values for the batch (size batch_size)
 * @param[in] stream       CUDA stream
 */
template <typename math_t>
void computeBatchDecisionFunction(const raft::handle_t& handle,
                                  const math_t* K,
                                  const math_t* dual_coefs,
                                  int batch_size,
                                  int n_support,
                                  bool transpose,
                                  math_t* y_batch,
                                  cudaStream_t stream)
{
  math_t one  = 1;
  math_t null = 0;
  raft::linalg::gemv(handle,
                     transpose,
                     transpose ? n_support : batch_size,
                     transpose ? batch_size : n_support,
                     &one,
                     K,
                     transpose ? n_support : batch_size,
                     dual_coefs,
                     1,
                     &null,
                     y_batch,
                     1,
                     stream);
}

/**
 * @brief Apply bias and convert to class labels or decision function values.
 *
 * Computes: preds = (y + b) for decision function, or class labels based on sign(y + b)
 *
 * @param[in] handle      cuML handle
 * @param[in] y           Decision function values before bias (size n_rows)
 * @param[in] n_rows      Number of samples
 * @param[in] b           Bias term
 * @param[in] labels      Class labels (size 2, used only if predict_class=true)
 * @param[in] predict_class Whether to output class labels (true) or decision values (false)
 * @param[out] preds      Output predictions (size n_rows)
 * @param[in] stream      CUDA stream
 */
template <typename math_t>
void applyPrediction(const raft::handle_t& handle,
                     const math_t* y,
                     int n_rows,
                     math_t b,
                     const math_t* labels,
                     bool predict_class,
                     math_t* preds,
                     cudaStream_t stream)
{
  if (predict_class) {
    // Look up the label based on the value of the decision function: f(x) = sign(y(x) + b)
    raft::linalg::unaryOp(
      preds,
      y,
      n_rows,
      [labels, b] __device__(math_t val) { return val + b < 0 ? labels[0] : labels[1]; },
      stream);
  } else {
    // Calculate the value of the decision function: f(x) = y(x) + b
    raft::linalg::unaryOp(preds, y, n_rows, [b] __device__(math_t val) { return val + b; }, stream);
  }
  handle.sync_stream(stream);
}

template <typename math_t, typename MatrixViewType>
int svcFitX(const raft::handle_t& handle,
            MatrixViewType matrix,
            int n_rows,
            int n_cols,
            math_t* labels,
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
  {
    rmm::device_uvector<math_t> unique_labels(0, stream);
    model.n_classes = raft::label::getUniquelabels(unique_labels, labels, n_rows, stream);
    rmm::device_async_resource_ref rmm_alloc = rmm::mr::get_current_device_resource_ref();
    model.unique_labels = (math_t*)rmm_alloc.allocate(stream, model.n_classes * sizeof(math_t));
    raft::copy(model.unique_labels, unique_labels.data(), model.n_classes, stream);
    handle_impl.sync_stream(stream);
  }

  ASSERT(model.n_classes == 2, "Only binary classification is implemented at the moment");

  rmm::device_uvector<math_t> y(n_rows, stream);
  raft::label::getOvrlabels(
    labels, n_rows, model.unique_labels, model.n_classes, y.data(), 1, stream);

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
            y.data(),
            sample_weight,
            &(model.dual_coefs),
            &(model.n_support),
            &(model.support_matrix),
            &(model.support_idx),
            &(model.b),
            param.max_iter,
            param.max_outer_iter);
  model.n_cols = n_cols;
  handle_impl.sync_stream(stream);
  if (kernel != nullptr) { delete kernel; }
  return smo.GetNIter();
}

template <typename math_t>
int svcFit(const raft::handle_t& handle,
           math_t* input,
           int n_rows,
           int n_cols,
           math_t* labels,
           const SvmParameter& param,
           ML::matrix::KernelParams& kernel_params,
           SvmModel<math_t>& model,
           const math_t* sample_weight)
{
  auto dense_view = raft::make_device_strided_matrix_view<math_t, int, raft::layout_f_contiguous>(
    input, n_rows, n_cols, 0);
  return svcFitX(
    handle, dense_view, n_rows, n_cols, labels, param, kernel_params, model, sample_weight);
}

template <typename math_t>
int svcFitSparse(const raft::handle_t& handle,
                 int* indptr,
                 int* indices,
                 math_t* data,
                 int n_rows,
                 int n_cols,
                 int nnz,
                 math_t* labels,
                 const SvmParameter& param,
                 ML::matrix::KernelParams& kernel_params,
                 SvmModel<math_t>& model,
                 const math_t* sample_weight)
{
  auto csr_structure_view = raft::make_device_compressed_structure_view<int, int, int>(
    indptr, indices, n_rows, n_cols, nnz);
  auto csr_matrix_view = raft::make_device_csr_matrix_view(data, csr_structure_view);
  return svcFitX(
    handle, csr_matrix_view, n_rows, n_cols, labels, param, kernel_params, model, sample_weight);
}

template <typename math_t, typename MatrixViewType>
void svcPredictX(const raft::handle_t& handle,
                 MatrixViewType matrix,
                 int n_rows,
                 int n_cols,
                 ML::matrix::KernelParams& kernel_params,
                 const SvmModel<math_t>& model,
                 math_t* preds,
                 math_t buffer_size,
                 bool predict_class)
{
  bool is_precomputed = kernel_params.kernel == ML::matrix::KernelType::PRECOMPUTED;

  // For precomputed kernels, n_cols is the number of training samples
  // (since input is K[test, train] of shape n_rows x n_train)
  if (!is_precomputed) {
    ASSERT(n_cols == model.n_cols, "Parameter n_cols: shall be the same that was used for fitting");
  }
  // We might want to query the available memory before selecting the batch size.
  // We will need n_batch * n_support floats for the kernel matrix K.
  int n_batch = n_rows;
  // Limit the memory size of the prediction buffer
  buffer_size = buffer_size * 1024 * 1024;
  if ((size_t)n_batch * model.n_support * sizeof(math_t) > buffer_size) {
    n_batch = buffer_size / (model.n_support * sizeof(math_t));
    if (n_batch < 1) n_batch = 1;
  }

  const raft::handle_t& handle_impl = handle;
  cudaStream_t stream               = handle_impl.get_stream().get();

  rmm::device_uvector<math_t> K(checked_mul<std::size_t>(n_batch, model.n_support), stream);
  rmm::device_uvector<math_t> y(n_rows, stream);
  if (model.n_support == 0) {
    RAFT_CUDA_TRY(
      cudaMemsetAsync(y.data(), 0, checked_mul<std::size_t>(n_rows, sizeof(math_t)), stream));
  }

  // Handle precomputed kernel prediction separately
  if (is_precomputed) {
    // Precomputed kernels only work with dense input
    if constexpr (!isDenseType<MatrixViewType>()) {
      ASSERT(false, "Precomputed kernels are not supported for sparse input");
    } else {
      // For precomputed kernels, matrix contains K[test, train] of shape (n_rows, n_cols)
      // where n_cols = n_train. We need to extract K[test, support_indices].
      if (model.n_support > 0) {
        const math_t* matrix_data = getDenseData(matrix);
        int* support_idx          = model.support_idx;
        int n_support             = model.n_support;

        // Process in batches
        for (int i = 0; i < n_rows; i += n_batch) {
          int batch_size = std::min(n_batch, n_rows - i);
          int n_elems    = batch_size * n_support;

          // Extract columns: K[row, col] = matrix[src_row, support_idx[col]]
          // Input matrix is column-major (F-contiguous): matrix[row, col] = matrix[row + col *
          // n_rows] Output K must also be column-major for gemv: K[row, col] = K[row + col *
          // batch_size]
          thrust::counting_iterator<int> iter(0);
          thrust::transform(
            thrust::cuda::par.on(stream),
            iter,
            iter + n_elems,
            K.data(),
            [matrix_data, support_idx, n_rows, batch_start = i, batch_size] __device__(int tid) {
              int row     = tid % batch_size;  // Column-major: row changes fast
              int col     = tid / batch_size;
              int src_col = support_idx[col];
              int src_row = batch_start + row;
              // Column-major: matrix[row, col] = matrix[row + col * n_rows]
              return matrix_data[src_row + src_col * n_rows];
            });

          // Compute y = K @ dual_coefs
          computeBatchDecisionFunction(handle_impl,
                                       K.data(),
                                       model.dual_coefs,
                                       batch_size,
                                       n_support,
                                       false,
                                       y.data() + i,
                                       stream);
        }
      }

      // Apply prediction (class labels or decision function)
      applyPrediction(
        handle_impl, y.data(), n_rows, model.b, model.unique_labels, predict_class, preds, stream);
      return;
    }
  }

  cuvs::distance::kernels::GramMatrixBase<math_t>* kernel =
    cuvs::distance::kernels::KernelFactory<math_t>::create(ML::matrix::to_cuvs(kernel_params));

  /*
    // kernel computation:
    //////////////////////////////////
    Dense input, dense support:
      * just multiply, expanded L2 norm for RBF
    Sparse Input, dense support
      * row ptr copy/shift for input csr, expanded L2 norm for RBF
    Dense input, sparse support
      * transpose kernel compute, expanded L2 norm for RBF
    Sparse input, sparse support
      * row ptr copy/shift for input csr
  */

  // store matrix dot product (l2 norm) for RBF kernels if applicable
  rmm::device_uvector<math_t> l2_input(0, stream);
  rmm::device_uvector<math_t> l2_support(0, stream);
  bool is_csr_input = !isDenseType<MatrixViewType>();

  bool is_csr_support   = model.support_matrix.data != nullptr && model.support_matrix.nnz >= 0;
  bool is_dense_support = model.support_matrix.data != nullptr && !is_csr_support;

  // Unfortunately we need runtime support for both types
  raft::device_matrix_view<math_t, int, raft::layout_stride> dense_support_matrix_view;
  if (is_dense_support) {
    dense_support_matrix_view =
      raft::make_device_strided_matrix_view<math_t, int, raft::layout_f_contiguous>(
        model.support_matrix.data, model.n_support, n_cols, 0);
  }
  auto csr_structure_view =
    is_csr_support
      ? raft::make_device_compressed_structure_view<int, int, int>(model.support_matrix.indptr,
                                                                   model.support_matrix.indices,
                                                                   model.n_support,
                                                                   n_cols,
                                                                   model.support_matrix.nnz)
      : raft::make_device_compressed_structure_view<int, int, int>(nullptr, nullptr, 0, 0, 0);
  auto csr_support_matrix_view =
    is_csr_support
      ? raft::make_device_csr_matrix_view<math_t, int, int, int>(model.support_matrix.data,
                                                                 csr_structure_view)
      : raft::make_device_csr_matrix_view<math_t, int, int, int>(nullptr, csr_structure_view);

  bool transpose_kernel = is_csr_support && !is_csr_input;
  if (model.n_support > 0 && static_cast<cuvs::distance::kernels::KernelType>(
                               kernel_params.kernel) == cuvs::distance::kernels::RBF) {
    l2_input.resize(n_rows, stream);
    l2_support.resize(model.n_support, stream);
    ML::SVM::matrixRowNorm(handle, matrix, l2_input.data(), raft::linalg::NormType::L2Norm);
    if (model.n_support > 0)
      if (is_csr_support) {
        ML::SVM::matrixRowNorm(
          handle, csr_support_matrix_view, l2_support.data(), raft::linalg::NormType::L2Norm);
      } else {
        ML::SVM::matrixRowNorm(
          handle, dense_support_matrix_view, l2_support.data(), raft::linalg::NormType::L2Norm);
      }
  }

  // additional row pointer information needed for batched CSR access
  // copy matrix row pointer to host to compute partial nnz on the fly
  std::vector<int> host_indptr;
  rmm::device_uvector<int> indptr_batched(0, stream);
  if (model.n_support > 0 && is_csr_input) {
    host_indptr.resize(n_rows + 1);
    indptr_batched.resize(n_batch + 1, stream);
    copyIndptrToHost(matrix, host_indptr.data(), stream);
  }

  // We process the input data batchwise:
  //  - calculate the kernel values K[x_batch, x_support]
  //  - calculate y(x_batch) = K[x_batch, x_support] * dual_coeffs
  for (int i = 0; i < n_rows && model.n_support > 0; i += n_batch) {
    if (i + n_batch >= n_rows) { n_batch = n_rows - i; }
    math_t* l2_input1 = l2_input.data() != nullptr ? l2_input.data() + i : nullptr;
    math_t* l2_input2 = l2_support.data();

    auto batch_matrix =
      getMatrixBatch(matrix, n_batch, i, host_indptr.data(), indptr_batched.data(), stream);

    if (transpose_kernel) {
      KernelOp(
        handle_impl, kernel, csr_support_matrix_view, batch_matrix, K.data(), l2_input2, l2_input1);
    } else if (is_csr_support) {
      KernelOp(
        handle_impl, kernel, batch_matrix, csr_support_matrix_view, K.data(), l2_input1, l2_input2);
    } else {
      KernelOp(handle_impl,
               kernel,
               batch_matrix,
               dense_support_matrix_view,
               K.data(),
               l2_input1,
               l2_input2);
    }

    computeBatchDecisionFunction(handle_impl,
                                 K.data(),
                                 model.dual_coefs,
                                 n_batch,
                                 model.n_support,
                                 transpose_kernel,
                                 y.data() + i,
                                 stream);

  }  // end of loop

  // Apply prediction (class labels or decision function)
  applyPrediction(
    handle_impl, y.data(), n_rows, model.b, model.unique_labels, predict_class, preds, stream);
  delete kernel;
}

template <typename math_t>
void svcPredict(const raft::handle_t& handle,
                math_t* input,
                int n_rows,
                int n_cols,
                ML::matrix::KernelParams& kernel_params,
                const SvmModel<math_t>& model,
                math_t* preds,
                math_t buffer_size,
                bool predict_class)
{
  auto dense_view = raft::make_device_strided_matrix_view<math_t, int, raft::layout_f_contiguous>(
    input, n_rows, n_cols, 0);
  svcPredictX(
    handle, dense_view, n_rows, n_cols, kernel_params, model, preds, buffer_size, predict_class);
}

template <typename math_t>
void svcPredictSparse(const raft::handle_t& handle,
                      int* indptr,
                      int* indices,
                      math_t* data,
                      int n_rows,
                      int n_cols,
                      int nnz,
                      ML::matrix::KernelParams& kernel_params,
                      const SvmModel<math_t>& model,
                      math_t* preds,
                      math_t buffer_size,
                      bool predict_class)
{
  auto csr_structure_view = raft::make_device_compressed_structure_view<int, int, int>(
    indptr, indices, n_rows, n_cols, nnz);
  auto csr_matrix_view = raft::make_device_csr_matrix_view(data, csr_structure_view);
  svcPredictX(handle,
              csr_matrix_view,
              n_rows,
              n_cols,
              kernel_params,
              model,
              preds,
              buffer_size,
              predict_class);
}

template <typename math_t>
void svmFreeBuffers(const raft::handle_t& handle, SvmModel<math_t>& m)
{
  cudaStream_t stream                      = handle.get_stream().get();
  rmm::device_async_resource_ref rmm_alloc = rmm::mr::get_current_device_resource_ref();
  if (m.dual_coefs) rmm_alloc.deallocate(stream, m.dual_coefs, m.n_support * sizeof(math_t));
  if (m.support_idx) rmm_alloc.deallocate(stream, m.support_idx, m.n_support * sizeof(int));
  if (m.support_matrix.indptr) {
    rmm_alloc.deallocate(stream, m.support_matrix.indptr, (m.n_support + 1) * sizeof(int));
    m.support_matrix.indptr = nullptr;
  }
  if (m.support_matrix.indices) {
    rmm_alloc.deallocate(stream, m.support_matrix.indices, m.support_matrix.nnz * sizeof(int));
    m.support_matrix.indices = nullptr;
  }
  if (m.support_matrix.data) {
    if (m.support_matrix.nnz == -1) {
      rmm_alloc.deallocate(stream, m.support_matrix.data, m.n_support * m.n_cols * sizeof(math_t));
    } else {
      rmm_alloc.deallocate(stream, m.support_matrix.data, m.support_matrix.nnz * sizeof(math_t));
    }
  }
  m.support_matrix.nnz = -1;
  if (m.unique_labels) rmm_alloc.deallocate(stream, m.unique_labels, m.n_classes * sizeof(math_t));
  m.dual_coefs    = nullptr;
  m.support_idx   = nullptr;
  m.unique_labels = nullptr;
}

};  // end namespace SVM
};  // end namespace ML
