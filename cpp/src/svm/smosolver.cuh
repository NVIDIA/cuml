/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "smosolver.h"

// #TODO: Replace with public header when ready
#include "kernelcache.cuh"
#include "results.cuh"
#include "smo_sets.cuh"
#include "smoblocksolve.cuh"
#include "workingset.cuh"
#include "ws_util.cuh"

#include <raft/core/handle.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/linalg/gemv.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/sparse/linalg/norm.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sequence.h>

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>

namespace ML {
namespace SVM {

template <typename math_t>
void SmoSolver<math_t>::GetNonzeroDeltaAlpha(const math_t* vec,
                                             int n_ws,
                                             const int* idx,
                                             math_t* nz_vec,
                                             int* n_nz,
                                             int* nz_idx,
                                             cudaStream_t stream)
{
  thrust::device_ptr<math_t> vec_ptr(const_cast<math_t*>(vec));
  thrust::device_ptr<math_t> nz_vec_ptr(nz_vec);
  thrust::device_ptr<int> idx_ptr(const_cast<int*>(idx));
  thrust::device_ptr<int> nz_idx_ptr(nz_idx);
  auto nonzero                   = [] __device__(math_t a) { return a != 0; };
  thrust::device_ptr<int> nz_end = thrust::copy_if(
    thrust::cuda::par.on(stream), idx_ptr, idx_ptr + n_ws, vec_ptr, nz_idx_ptr, nonzero);
  *n_nz = nz_end - nz_idx_ptr;
  thrust::copy_if(thrust::cuda::par.on(stream), vec_ptr, vec_ptr + n_ws, nz_vec_ptr, nonzero);
}

/**
 * @brief Solve the quadratic optimization problem.
 *
 * The output arrays (dual_coefs, support_matrix, idx) will be allocated on the
 * device, they should be unallocated on entry.
 *
 * @param [in] matrix training vectors in matrix format(MLCommon::Matrix::Matrix),
 * size [n_rows x * n_cols]
 * @param [in] n_rows number of rows (training vectors)
 * @param [in] n_cols number of columns (features)
 * @param [in] y labels (values +/-1), size [n_rows]
 * @param [in] sample_weight device array of sample weights (or nullptr if not
 *     applicable)
 * @param [out] dual_coefs size [n_support] on exit
 * @param [out] n_support number of support vectors
 * @param [out] support_matrix support vectors in matrix format, size [n_support, n_cols]
 * @param [out] idx the original training set indices of the support vectors, size [n_support]
 * @param [out] b scalar constant for the decision function
 * @param [in] max_iter: maximum number of total iterations (default -1 for no limit)
 * @param [in] max_outer_iter maximum number of outer iteration (default 100 * n_rows)
 * @param [in] max_inner_iter maximum number of inner iterations (default 10000)
 */
template <typename math_t>
template <typename MatrixViewType>
void SmoSolver<math_t>::Solve(MatrixViewType matrix,
                              int n_rows,
                              int n_cols,
                              math_t* y,
                              const math_t* sample_weight,
                              math_t** dual_coefs,
                              int* n_support,
                              SupportStorage<math_t>* support_matrix,
                              int** idx,
                              math_t* b,
                              int max_iter,
                              int max_outer_iter,
                              int max_inner_iter)
{
  constexpr const int SMO_WS_SIZE = 1024;
  // Prepare data structures for SMO
  WorkingSet<math_t> ws(handle, stream, n_rows, SMO_WS_SIZE, svmType);
  n_ws = ws.GetSize();
  Initialize(&y, sample_weight, n_rows, n_cols);
  KernelCache<math_t, MatrixViewType> cache(handle,
                                            matrix,
                                            n_rows,
                                            n_cols,
                                            n_ws,
                                            kernel,
                                            kernel_type,
                                            cache_size,
                                            svmType,
                                            1 << 30,  // kernel_tile_byte_limit
                                            1 << 30,  // dense_extract_byte_limit
                                            is_precomputed);

  // Init counters
  max_outer_iter        = GetDefaultMaxIter(n_train, max_outer_iter);
  n_outer_iter          = 0;
  n_iter                = 0;
  diff_prev             = 0;
  n_small_diff          = 0;
  n_increased_diff      = 0;
  report_increased_diff = true;
  bool keep_going       = true;

  rmm::device_uvector<math_t> nz_da(n_ws, stream);
  rmm::device_uvector<int> nz_da_idx(n_ws, stream);

  while (keep_going) {
    RAFT_CUDA_TRY(cudaMemsetAsync(delta_alpha.data(), 0, n_ws * sizeof(math_t), stream));
    raft::common::nvtx::push_range("SmoSolver::ws_select");
    ws.Select(f.data(), alpha.data(), y, C_vec.data());
    raft::common::nvtx::pop_range();
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    raft::common::nvtx::push_range("SmoSolver::Kernel");

    cache.InitWorkingSet(ws.GetIndices());

    math_t* cacheTile = cache.getSquareTileWithoutCaching();

    raft::common::nvtx::pop_range();
    raft::common::nvtx::push_range("SmoSolver::SmoBlockSolve");
    int max_iter_this_block =
      (max_iter == -1 ? max_inner_iter : std::min(max_inner_iter, max_iter - n_iter));
    SmoBlockSolve<math_t, SMO_WS_SIZE><<<1, n_ws, 0, stream>>>(y,
                                                               n_train,
                                                               alpha.data(),
                                                               n_ws,
                                                               delta_alpha.data(),
                                                               f.data(),
                                                               cacheTile,
                                                               cache.getKernelIndices(true),
                                                               C_vec.data(),
                                                               tol,
                                                               return_buff.data(),
                                                               max_iter_this_block,
                                                               svmType);

    RAFT_CUDA_TRY(cudaPeekAtLastError());

    raft::update_host(host_return_buff, return_buff.data(), 2, stream);
    raft::common::nvtx::pop_range();
    raft::common::nvtx::push_range("SmoSolver::UpdateF");
    raft::common::nvtx::push_range("SmoSolver::UpdateF::getNnzDaRows");
    int nnz_da;
    GetNonzeroDeltaAlpha(delta_alpha.data(),
                         n_ws,
                         cache.getKernelIndices(false),
                         nz_da.data(),
                         &nnz_da,
                         nz_da_idx.data(),
                         stream);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    // The following should be performed only for elements with nonzero delta_alpha
    bool made_progress = nnz_da > 0;
    if (made_progress) {
      auto batch_descriptor = cache.InitFullTileBatching(nz_da_idx.data(), nnz_da);

      while (cache.getNextBatchKernel(batch_descriptor)) {
        raft::common::nvtx::pop_range();
        raft::common::nvtx::push_range("SmoSolver::UpdateF::updateBatch");
        // do (partial) update
        UpdateF(f.data() + batch_descriptor.offset,
                batch_descriptor.batch_size,
                nz_da.data(),
                nnz_da,
                batch_descriptor.kernel_data);
        RAFT_CUDA_TRY(cudaPeekAtLastError());
      }
    } else {
      cache.FinishWorkingSet();
    }
    handle.sync_stream(stream);
    raft::common::nvtx::pop_range();
    raft::common::nvtx::pop_range();  // ("SmoSolver::UpdateF");

    math_t diff = host_return_buff[0];
    keep_going  = CheckStoppingCondition(diff);
    n_iter += host_return_buff[1];
    n_outer_iter++;
    if ((max_iter != -1 && n_iter >= max_iter) || n_outer_iter >= max_outer_iter) {
      keep_going = false;
    }
    if (keep_going && !made_progress) {
      const char* advice = std::is_same<math_t, float>::value
                             ? " Try using float64 input or reducing the magnitude of the kernel "
                               "values."
                             : " Try rescaling the input data or adjusting the kernel parameters.";
      THROW(
        "SMO error: solver made no progress while the stopping criterion was not satisfied. "
        "This can happen when kernel values are too large for the input precision.%s",
        advice);
    }

    if (n_outer_iter % 500 == 0) {
      CUML_LOG_DEBUG("SMO iteration %d, diff %lf", n_outer_iter, (double)diff);
    }
  }

  CUML_LOG_DEBUG(
    "SMO solver finished after %d outer iterations, total inner %d"
    " iterations, and diff %lf",
    n_outer_iter,
    n_iter,
    diff_prev);

  Results<math_t, MatrixViewType> res(
    handle, matrix, n_rows, n_cols, y, C_vec.data(), svmType, is_precomputed);
  res.Get(alpha.data(), f.data(), dual_coefs, n_support, idx, support_matrix, b);

  ReleaseBuffers();
}

/**
 * @brief Update the f vector after a block solve step.
 *
 * \f[ f_i = f_i + \sum_{k\in WS} K_{i,k} * \Delta \alpha_k, \f]
 * where i = [0..n_train-1], WS is the set of workspace indices,
 * and \f$K_{i,k}\f$ is the kernel function evaluated for training vector x_i and workspace vector
 * x_k.
 *
 * @param f size [n_train]
 * @param n_rows
 * @param delta_alpha size [n_ws]
 * @param n_ws
 * @param cacheTile kernel function evaluated for the following set K[X,x_ws],
 *   size [n_rows, n_ws]
 */
template <typename math_t>
void SmoSolver<math_t>::UpdateF(
  math_t* f, int n_rows, const math_t* delta_alpha, int n_ws, const math_t* cacheTile)
{
  // multipliers used in the equation : f = 1*cachtile * delta_alpha + 1*f
  math_t one = 1;
  // #TODO: Call from public API when ready
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemv(handle.get_cublas_handle(),
                                                   CUBLAS_OP_N,
                                                   n_rows,
                                                   n_ws,
                                                   &one,
                                                   cacheTile,
                                                   n_rows,
                                                   delta_alpha,
                                                   1,
                                                   &one,
                                                   f,
                                                   1,
                                                   stream));
  if (svmType == EPSILON_SVR) {
    // SVR has doubled the number of training vectors and we need to update
    // alpha for both batches individually
    // #TODO: Call from public API when ready
    RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemv(handle.get_cublas_handle(),
                                                     CUBLAS_OP_N,
                                                     n_rows,
                                                     n_ws,
                                                     &one,
                                                     cacheTile,
                                                     n_rows,
                                                     delta_alpha,
                                                     1,
                                                     &one,
                                                     f + n_rows,
                                                     1,
                                                     stream));
  }
}

/** @brief Initialize the problem to solve.
 *
 * Both SVC and SVR are solved as a classification problem.
 * The optimization target (W) does not appear directly in the SMO
 * formulation, only its derivative through f (optimality indicator vector):
 * \f[ f_i = y_i \frac{\partial W }{\partial \alpha_i}. \f]
 *
 * The f_i values are initialized here, and updated at every solver iteration
 * when alpha changes. The update step is the same for SVC and SVR, only the
 * init step differs.
 *
 * Additionally, we zero init the dual coefficients (alpha), and initialize
 * class labels for SVR.
 *
 * @param[inout] y on entry class labels or target values,
 *    on exit device pointer to class labels
 * @param[in] sample_weight sample weights (can be nullptr, otherwise device
 *    array of size [n_rows])
 * @param[in] n_rows
 * @param[in] n_cols
 */
template <typename math_t>
void SmoSolver<math_t>::Initialize(math_t** y, const math_t* sample_weight, int n_rows, int n_cols)
{
  this->n_rows = n_rows;
  this->n_cols = n_cols;
  n_train      = (svmType == EPSILON_SVR) ? n_rows * 2 : n_rows;
  ResizeBuffers(n_train, n_cols);
  // Zero init alpha
  RAFT_CUDA_TRY(cudaMemsetAsync(alpha.data(), 0, n_train * sizeof(math_t), stream));
  InitPenalty(C_vec.data(), sample_weight, n_rows);
  // Init f (and also class labels for SVR)
  switch (svmType) {
    case C_SVC: SvcInit(*y); break;
    case EPSILON_SVR:
      SvrInit(*y, n_rows, y_label.data(), f.data());
      // We return the pointer to the class labels (the target values are
      // not needed anymore, they are incorporated in f).
      *y = y_label.data();
      break;
    default: THROW("SMO initialization not implemented SvmType=%d", svmType);
  }
}

template <typename math_t>
void SmoSolver<math_t>::InitPenalty(math_t* C_vec, const math_t* sample_weight, int n_rows)
{
  if (sample_weight == nullptr) {
    thrust::device_ptr<math_t> c_ptr(C_vec);
    thrust::fill(thrust::cuda::par.on(stream), c_ptr, c_ptr + n_train, C);
  } else {
    math_t C = this->C;
    raft::linalg::unaryOp(
      C_vec, sample_weight, n_rows, [C] __device__(math_t w) { return C * w; }, stream);
    if (n_train > n_rows) {
      // Set the same penalty parameter for the duplicate set of vectors
      raft::linalg::unaryOp(
        C_vec + n_rows, sample_weight, n_rows, [C] __device__(math_t w) { return C * w; }, stream);
    }
  }
}

/** @brief Initialize Support Vector Classification
 *
 * We would like to maximize the following quantity
 * \f[ W(\mathbf{\alpha}) = -\mathbf{\alpha}^T \mathbf{1}
 *   + \frac{1}{2} \mathbf{\alpha}^T Q \mathbf{\alpha}, \f]
 *
 * We initialize f as:
 * \f[ f_i = y_i \frac{\partial W(\mathbf{\alpha})}{\partial \alpha_i} =
 *          -y_i +   y_j \alpha_j K(\mathbf{x}_i, \mathbf{x}_j) \f]
 *
 * @param [in] y device pointer of class labels size [n_rows]
 */
template <typename math_t>
void SmoSolver<math_t>::SvcInit(const math_t* y)
{
  raft::linalg::unaryOp(f.data(), y, n_rows, [] __device__(math_t y) { return -y; }, stream);
}

/**
 * @brief Initializes the solver for epsilon-SVR.
 *
 * For regression we are optimizing the following quantity
 * \f[
 * W(\alpha^+, \alpha^-) =
 * \epsilon \sum_{i=1}^l (\alpha_i^+ + \alpha_i^-)
 * - \sum_{i=1}^l yc_i (\alpha_i^+ - \alpha_i^-)
 * + \frac{1}{2} \sum_{i,j=1}^l
 *   (\alpha_i^+ - \alpha_i^-)(\alpha_j^+ - \alpha_j^-) K(\bm{x}_i, \bm{x}_j)
 * \f]
 *
 * Then \f$ f_i = y_i \frac{\partial W(\alpha}{\partial \alpha_i} \f$
 *      \f$     = yc_i*epsilon - yr_i \f$
 *
 * Additionally we set class labels for the training vectors.
 *
 * References:
 * [1] B. Schölkopf et. al (1998): New support vector algorithms,
 *     NeuroCOLT2 Technical Report Series, NC2-TR-1998-031, Section 6
 * [2] A.J. Smola, B. Schölkopf (2004): A tutorial on support vector
 *     regression, Statistics and Computing 14, 199–222
 * [3] Orchel M. (2011) Support Vector Regression as a Classification Problem
 *     with a Priori Knowledge in the Form of Detractors,
 *     Man-Machine Interactions 2. Advances in Intelligent and Soft Computing,
 *     vol 103
 *
 * @param [in] yr device pointer with values for regression, size [n_rows]
 * @param [in] n_rows
 * @param [out] yc device pointer to classes associated to the dual
 *     coefficients, size [n_rows*2]
 * @param [out] f device pointer f size [n_rows*2]
 */
template <typename math_t>
void SmoSolver<math_t>::SvrInit(const math_t* yr, int n_rows, math_t* yc, math_t* f)
{
  // Init class labels to [1, 1, 1, ..., -1, -1, -1, ...]
  thrust::device_ptr<math_t> yc_ptr(yc);
  thrust::constant_iterator<math_t> one(1);
  thrust::copy(thrust::cuda::par.on(stream), one, one + n_rows, yc_ptr);
  thrust::constant_iterator<math_t> minus_one(-1);
  thrust::copy(thrust::cuda::par.on(stream), minus_one, minus_one + n_rows, yc_ptr + n_rows);

  // f_i = epsilon - y_i, for i \in [0..n_rows-1]
  math_t epsilon = this->epsilon;
  raft::linalg::unaryOp(
    f, yr, n_rows, [epsilon] __device__(math_t y) { return epsilon - y; }, stream);

  // f_i = -epsilon - y_i, for i \in [n_rows..2*n_rows-1]
  raft::linalg::unaryOp(
    f + n_rows, yr, n_rows, [epsilon] __device__(math_t y) { return -epsilon - y; }, stream);
}

}  // namespace SVM
}  // namespace ML
