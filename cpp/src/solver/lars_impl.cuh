/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/common/checked_arithmetic.hpp>
#include <cuml/common/export.hpp>
#include <cuml/common/logger.hpp>
#include <cuml/common/utils.hpp>
#include <cuml/solvers/lars.hpp>

#include <raft/core/handle.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/cholesky_r1_update.cuh>
#include <raft/util/cache_util.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cub/cub.cuh>
#include <cuda/functional>

#include <iostream>
#include <limits>
#include <numeric>
#include <vector>
// #TODO: Replace with public header when ready
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/linalg/gemv.cuh>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/linalg/unary_op.cuh>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace ML {
namespace Solver {
namespace Lars {

enum class LarsFitStatus { kOk, kCollinear, kError, kStop };

/**
 * @brief Select the largest element from the inactive working set.
 *
 * The inactive set consist of cor[n_active..n-1]. This function returns the
 * index of the most correlated element. The value of the largest element is
 * returned in cj.
 *
 * The correlation value is checked for numeric error and convergence, and the
 * return status indicates whether training should continue.
 *
 * @param n_active number of active elements (n_active <= n )
 * @param n number of elements in vector cor
 * @param correlation device array of correlations, size [n]
 * @param cj host pointer to return the value of the largest element
 * @param wokspace buffer, size >= n_cols
 * @param max_idx host pointer the index of the max correlation is returned here
 * @param indices host pointer of feature column indices, size [n_cols]
 * @param n_iter iteration counter
 * @param stream CUDA stream
 *
 * @return fit status
 */
template <typename math_t, typename idx_t = int>
LarsFitStatus selectMostCorrelated(idx_t n_active,
                                   idx_t n,
                                   math_t* correlation,
                                   math_t* cj,
                                   rmm::device_uvector<math_t>& workspace,
                                   idx_t* max_idx,
                                   idx_t n_rows,
                                   idx_t* indices,
                                   idx_t n_iter,
                                   cudaStream_t stream)
{
  const idx_t align_bytes = 16 * sizeof(math_t);
  // We might need to start a few elements earlier to ensure that the unary
  // op has aligned access for vectorized load.
  int start = raft::alignDown<idx_t>(n_active, align_bytes) / sizeof(math_t);
  raft::linalg::unaryOp(
    workspace.data(), correlation + start, n, [] __device__(math_t a) { return abs(a); }, stream);
  thrust::device_ptr<math_t> ptr(workspace.data() + n_active - start);
  auto max_ptr = thrust::max_element(thrust::cuda::par.on(stream), ptr, ptr + n - n_active);
  raft::update_host(cj, max_ptr.get(), 1, stream);
  raft::interruptible::synchronize(stream);

  *max_idx = n_active + (max_ptr - ptr);  // the index of the maximum element

  CUML_LOG_DEBUG(
    "Iteration %d, selected feature %d with correlation %f", n_iter, indices[*max_idx], *cj);

  if (!std::isfinite(*cj)) {
    CUML_LOG_ERROR("Correlation is not finite, aborting.");
    return LarsFitStatus::kError;
  }

  // Tolerance for early stopping. Note we intentionally use here fp32 epsilon,
  // otherwise the tolerance is too small (which could result in numeric error
  // in Cholesky rank one update if eps < 0, or exploding regression parameters
  // if eps > 0).
  const math_t tolerance = std::numeric_limits<float>::epsilon();
  if (abs(*cj) / n_rows < tolerance) {
    CUML_LOG_WARN("Reached tolarence limit with %e", abs(*cj));
    return LarsFitStatus::kStop;
  }
  return LarsFitStatus::kOk;
}

/**
 * @brief Swap two feature vectors.
 *
 * The function swaps feature column j and k or the corresponding rows and
 * and columns of the Gram matrix. The elements of the cor and indices arrays
 * are also swapped.
 *
 * @param handle cuBLAS handle
 * @param j column index
 * @param k column index
 * @param X device array of feature vectors in column major format, size
 *     [n_cols * ld_X]
 * @param n_rows number of training vectors
 * @param n_cols number of features
 * @param ld_X leading dimension of X
 * @param cor device array of correlations, size [n_cols]
 * @param indices host array of indices, size [n_cols]
 * @param G device pointer of Gram matrix (or nullptr), size [n_cols * ld_G]
 * @param ld_G leading dimension of G
 * @param stream CUDA stream
 */
template <typename math_t, typename idx_t = int>
void swapFeatures(cublasHandle_t handle,
                  idx_t j,
                  idx_t k,
                  math_t* X,
                  idx_t n_rows,
                  idx_t n_cols,
                  idx_t ld_X,
                  math_t* cor,
                  idx_t* indices,
                  math_t* G,
                  idx_t ld_G,
                  cudaStream_t stream)
{
  std::swap(indices[j], indices[k]);
  if (G) {
    // #TODO: Call from public API when ready
    RAFT_CUBLAS_TRY(
      raft::linalg::detail::cublasSwap(handle, n_cols, G + ld_G * j, 1, G + ld_G * k, 1, stream));
    // #TODO: Call from public API when ready
    RAFT_CUBLAS_TRY(
      raft::linalg::detail::cublasSwap(handle, n_cols, G + j, ld_G, G + k, ld_G, stream));
  } else {
    // Only swap X if G is nullptr. Only in that case will we use the feature
    // columns, otherwise all the necessary information is already there in G.
    // #TODO: Call from public API when ready
    RAFT_CUBLAS_TRY(
      raft::linalg::detail::cublasSwap(handle, n_rows, X + ld_X * j, 1, X + ld_X * k, 1, stream));
  }
  // swap (c[j], c[k])
  // #TODO: Call from public API when ready
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublasSwap(handle, 1, cor + j, 1, cor + k, 1, stream));
}

/**
 * @brief Move feature at idx=j into the active set.
 *
 * We have an active set with n_active elements, and an inactive set with
 * n_valid_cols - n_active elements. The matrix X [n_samples, n_features] is
 * partitioned in a way that the first n_active columns store the active set.
 * Similarly the vectors correlation and indices are partitioned in a way
 * that the first n_active elements belong to the active set:
 * - active set:  X[:,:n_active], correlation[:n_active], indices[:n_active]
 * - inactive set: X[:,n_active:], correlation[n_active:], indices[n_active:].
 *
 * This function moves the feature column X[:,idx] into the active set by
 * replacing the first inactive element with idx. The indices and correlation
 * vectors are modified accordingly. The sign array is updated with the sign
 * of correlation[n_active].
 *
 * @param handle cuBLAS handle
 * @param n_active number of active elements, will be increased by one after
 *     we move the new element j into the active set
 * @param j index of the new element (n_active <= j < n_cols)
 * @param X device array of feature vectors in column major format, size
 *     [n_cols * ld_X]
 * @param n_rows number of training vectors
 * @param n_cols number of valid features columns (ignoring those features which
 *    are detected to be collinear with the active set)
 * @param ld_X leading dimension of X
 * @param cor device array of correlations, size [n_cols]
 * @param indices host array of indices, size [n_cols]
 * @param G device pointer of Gram matrix (or nullptr), size [n_cols * ld_G]
 * @param ld_G leading dimension of G
 * @param sign device pointer to sign array, size[n]
 * @param stream CUDA stream
 */
template <typename math_t, typename idx_t = int>
void moveToActive(cublasHandle_t handle,
                  idx_t* n_active,
                  idx_t j,
                  math_t* X,
                  idx_t n_rows,
                  idx_t n_cols,
                  idx_t ld_X,
                  math_t* cor,
                  idx_t* indices,
                  math_t* G,
                  idx_t ld_G,
                  math_t* sign,
                  cudaStream_t stream)
{
  idx_t idx_free = *n_active;
  swapFeatures(handle, idx_free, j, X, n_rows, n_cols, ld_X, cor, indices, G, ld_G, stream);

  // sign[n_active] = sign(c[n_active])
  raft::linalg::unaryOp(
    sign + idx_free,
    cor + idx_free,
    1,
    [] __device__(math_t c) -> math_t {
      // return the sign of c
      return (math_t(0) < c) - (c < math_t(0));
    },
    stream);

  (*n_active)++;
}

/**
 * @brief Update the Cholesky decomposition of the Gram matrix of the active set
 *
 * G0 = X.T * X, Gram matrix without signs. We use the part that corresponds to
 * the active set, [n_A x n_A]
 *
 * At each step on the LARS path we add one column to the active set, therefore
 * the Gram matrix grows incrementally. We update the Cholesky decomposition
 * G0 = U.T * U.
 *
 * The Cholesky decomposition can use the same storage as G0, if the input
 * pointers are same.
 *
 * @param handle RAFT handle
 * @param n_active number of active elements
 * @param X device array  of feature vectors in column major format, size
 *     [n_rows * n_cols]
 * @param n_rows number of training vectors
 * @param n_cols number of features
 * @param ld_X leading dimension of X (stride of columns)
 * @param U device pointer to the Cholesky decomposition of G0,
 *     size [n_cols * ld_U]
 * @param ld_U leading dimension of U
 * @param G0 device pointer to Gram matrix G0 = X.T*X (can be nullptr),
 *     size [n_cols * ld_G].
 * @param ld_G leading dimension of G
 * @param workspace workspace for the Cholesky update
 * @param eps parameter for cheleskyRankOneUpdate
 * @param stream CUDA stream
 */
template <typename math_t, typename idx_t = int>
void updateCholesky(const raft::handle_t& handle,
                    idx_t n_active,
                    const math_t* X,
                    idx_t n_rows,
                    idx_t n_cols,
                    idx_t ld_X,
                    math_t* U,
                    idx_t ld_U,
                    const math_t* G0,
                    idx_t ld_G,
                    rmm::device_uvector<math_t>& workspace,
                    math_t eps,
                    cudaStream_t stream)
{
  const cublasFillMode_t fillmode = CUBLAS_FILL_MODE_UPPER;
  if (G0 == nullptr) {
    // Calculate the new column of G0. It is stored in U.
    math_t* G_row       = U + (n_active - 1) * ld_U;
    const math_t* X_row = X + (n_active - 1) * ld_X;
    math_t one          = 1;
    math_t zero         = 0;
    // #TODO: Call from public API when ready
    RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemv(handle.get_cublas_handle(),
                                                     CUBLAS_OP_T,
                                                     n_rows,
                                                     n_cols,
                                                     &one,
                                                     X,
                                                     n_rows,
                                                     X_row,
                                                     1,
                                                     &zero,
                                                     G_row,
                                                     1,
                                                     stream));
  } else if (G0 != U) {
    // Copy the new column of G0 into U, because the factorization works in
    // place.
    raft::copy(U + (n_active - 1) * ld_U, G0 + (n_active - 1) * ld_G, n_active, stream);
  }  // Otherwise the new data is already in place in U.

  // Update the Cholesky decomposition
  int n_work = narrow_cast<int>(workspace.size());
  if (n_work == 0) {
    // Query workspace size and allocate it
    raft::linalg::choleskyRank1Update(
      handle, U, n_active, ld_U, nullptr, &n_work, fillmode, stream);
    workspace.resize(n_work, stream);
  }
  raft::linalg::choleskyRank1Update(
    handle, U, n_active, ld_U, workspace.data(), &n_work, fillmode, stream, eps);
}

/**
 * @brief Solve for ws = S * GA^(-1) * 1_A  using a Cholesky decomposition.
 *
 * See calcEquiangularVec for more details on the formulas. In this function we
 * calculate ws = S * (S * G0 * S)^{-1} 1_A = G0^{-1} (S 1_A) = G0^{-1} sign_A.
 *
 * @param handle RAFT handle
 * @param n_active number of active elements
 * @param n_cols number of features
 * @param sign array with sign of the active set, size [n_cols]
 * @param U device pointer to the Cholesky decomposition of G0,
 *     size [n_cols * n_cols]
 * @param ld_U leading dimension of U (column stride)
 * @param ws device pointer, size [n_active]
 * @param stream CUDA stream
 */
template <typename math_t, typename idx_t = int>
void calcW0(const raft::handle_t& handle,
            idx_t n_active,
            idx_t n_cols,
            const math_t* sign,
            const math_t* U,
            idx_t ld_U,
            math_t* ws,
            cudaStream_t stream)
{
  const cublasFillMode_t fillmode = CUBLAS_FILL_MODE_UPPER;

  // First we calculate x by solving equation U.T x = sign_A.
  raft::copy(ws, sign, n_active, stream);
  math_t alpha = 1;
  // #TODO: Call from public API when ready
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublastrsm(handle.get_cublas_handle(),
                                                   CUBLAS_SIDE_LEFT,
                                                   fillmode,
                                                   CUBLAS_OP_T,
                                                   CUBLAS_DIAG_NON_UNIT,
                                                   n_active,
                                                   1,
                                                   &alpha,
                                                   U,
                                                   ld_U,
                                                   ws,
                                                   ld_U,
                                                   stream));

  // ws stores x, the solution of U.T x = sign_A. Now we solve U * ws = x
  // #TODO: Call from public API when ready
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublastrsm(handle.get_cublas_handle(),
                                                   CUBLAS_SIDE_LEFT,
                                                   fillmode,
                                                   CUBLAS_OP_N,
                                                   CUBLAS_DIAG_NON_UNIT,
                                                   n_active,
                                                   1,
                                                   &alpha,
                                                   U,
                                                   ld_U,
                                                   ws,
                                                   ld_U,
                                                   stream));
  // Now ws = G0^(-1) sign_A = S GA^{-1} 1_A.
}

/**
 * @brief Calculate A = (1_A * GA^{-1} * 1_A)^{-1/2}.
 *
 * See calcEquiangularVec for more details on the formulas.
 *
 * @param handle RAFT handle
 * @param A device pointer to store the result
 * @param n_active number of active elements
 * @param sign array with sign of the active set, size [n_cols]
 * @param ws device pointer, size [n_active]
 * @param stream CUDA stream
 */
template <typename math_t, typename idx_t = int>
void calcA(const raft::handle_t& handle,
           math_t* A,
           idx_t n_active,
           const math_t* sign,
           const math_t* ws,
           cudaStream_t stream)
{
  // Calculate sum (w) = sum(ws * sign)
  auto multiply = [] __device__(math_t w, math_t s) { return w * s; };
  raft::linalg::mapThenSumReduce(A, n_active, multiply, stream, ws, sign);
  // Calc Aa = 1 / sqrt(sum(w))
  raft::linalg::unaryOp(A, A, 1, [] __device__(math_t a) { return 1 / sqrt(a); }, stream);
}

/**
 * @brief Calculate the equiangular vector u, w and A according to [1].
 *
 * We introduce the following variables (Python like indexing):
 * - n_A number of elements in the active set
 * - S = diag(sign_A): diagonal matrix with the signs, size [n_A x n_A]
 * - X_A = X[:,:n_A] * S, column vectors of the active set size [n_A x n_A]
 * - G0 = X.T * X, Gram matrix without signs. We just use the part that
 *   corresponds to the active set, [n_A x n_A]
 * - GA = X_A.T * X_A is the Gram matrix of the active set, size [n_A x n_A]
 *   GA = S * G0[:n_A, :n_A] * S
 * - 1_A = np.ones(n_A)
 * - A = (1_A * GA^{-1} * 1_A)^{-1/2}, scalar, see eq (2.5) in [1]
 * - w = A GA^{-1} * 1_A, vector of size [n_A] see eq (2.6) in [1]
 * - ws = S * w, vector of size [n_A]
 *
 * The equiangular vector can be expressed the following way (equation 2.6):
 * u = X_A * w = X[:,:n_A] S * w = X[:,:n_A] * ws.
 *
 * The equiangular vector later appears only in an expression like X.T u, which
 * can be reformulated as X.T u = X.T X[:,:n_A] S * w = G[:n_A,:n_A] * ws.
 * If the gram matrix is given, then we do not need to calculate u, it will be
 * sufficient to calculate ws and A.
 *
 * We use Cholesky decomposition G0 = U.T * U to solve to calculate A and w
 * which depend on GA^{-1}.
 *
 * References:
 *  [1] B. Efron, T. Hastie, I. Johnstone, R Tibshirani, Least Angle Regression
 *  The Annals of Statistics (2004) Vol 32, No 2, 407-499
 *  http://statweb.stanford.edu/~tibs/ftp/lars.pdf
 *
 * @param handle RAFT handle
 * @param n_active number of active elements
 * @param X device array  of feature vectors in column major format, size
 *     [ld_X * n_cols]
 * @param n_rows number of training vectors
 * @param n_cols number of features
 * @param ld_X leading dimension of array X (column stride, ld_X >= n_rows)
 * @param sign array with sign of the active set, size [n_cols]
 * @param U device pointer to the Cholesky decomposition of G0,
 *     size [ld_U * n_cols]
 * @param ld_U leading dimension of array U (ld_U >= n_cols)
 * @param G0 device pointer to Gram matrix G0 = X.T*X (can be nullptr),
 *     size [ld_G * n_cols]. Note the difference between G0 and
 *     GA = X_A.T * X_A
 * @param ld_G leading dimension of array G0 (ld_G >= n_cols)
 * @param workspace workspace for the Cholesky update
 * @param ws device pointer, size [n_active]
 * @param A device pointer to a scalar
 * @param u_eq device pointer to the equiangular vector, only used if
 *    Gram==nullptr, size [n_rows].
 * @param eps numerical regularizaton parameter for the Cholesky decomposition
 * @param stream CUDA stream
 *
 * @return fit status
 */
template <typename math_t, typename idx_t = int>
LarsFitStatus calcEquiangularVec(const raft::handle_t& handle,
                                 idx_t n_active,
                                 math_t* X,
                                 idx_t n_rows,
                                 idx_t n_cols,
                                 idx_t ld_X,
                                 math_t* sign,
                                 math_t* U,
                                 idx_t ld_U,
                                 math_t* G0,
                                 idx_t ld_G,
                                 rmm::device_uvector<math_t>& workspace,
                                 math_t* ws,
                                 math_t* A,
                                 math_t* u_eq,
                                 math_t eps,
                                 cudaStream_t stream)
{
  // Since we added a new vector to the active set, we update the Cholesky
  // decomposition (U)
  updateCholesky(
    handle, n_active, X, n_rows, n_cols, ld_X, U, ld_U, G0, ld_G, workspace, eps, stream);

  // Calculate ws = S GA^{-1} 1_A using U
  calcW0(handle, n_active, n_cols, sign, U, ld_U, ws, stream);

  calcA(handle, A, n_active, sign, ws, stream);

  // ws *= Aa
  raft::linalg::unaryOp(ws, ws, n_active, [A] __device__(math_t w) { return (*A) * w; }, stream);

  // Check for numeric error
  math_t ws_host;
  raft::update_host(&ws_host, ws, 1, stream);
  math_t diag_host;  // U[n_active-1, n_active-1]
  raft::update_host(&diag_host, U + ld_U * (n_active - 1) + n_active - 1, 1, stream);
  handle.sync_stream(stream);
  if (diag_host < 1e-7) {
    CUML_LOG_WARN(
      "Vanising diagonal in Cholesky factorization (%e). This indicates "
      "collinear features. Dropping current regressor.",
      diag_host);
    return LarsFitStatus::kCollinear;
  }
  if (!std::isfinite(ws_host)) {
    CUML_LOG_WARN("ws=%f is not finite at iteration %d", ws_host, n_active);
    return LarsFitStatus::kError;
  }

  if (G0 == nullptr) {
    // Calculate u_eq only in the case if the Gram matrix is not stored.
    math_t one  = 1;
    math_t zero = 0;
    // #TODO: Call from public API when ready
    RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemv(handle.get_cublas_handle(),
                                                     CUBLAS_OP_N,
                                                     n_rows,
                                                     n_active,
                                                     &one,
                                                     X,
                                                     ld_X,
                                                     ws,
                                                     1,
                                                     &zero,
                                                     u_eq,
                                                     1,
                                                     stream));
  }
  return LarsFitStatus::kOk;
}

/**
 * @brief Calculate the maximum step size (gamma) in the equiangular direction.
 *
 * Let mu = X beta.T be the current prediction vector. The modified solution
 * after taking step gamma is defined as mu' = mu + gamma u. With this
 * solution the correlation of the covariates in the active set will decrease
 * equally, to a new value |c_j(gamma)| = Cmax - gamma A. At the same time
 * the correlation of the values in the inactive set changes according to the
 * following formula: c_j(gamma) = c_j - gamma a_j. We increase gamma until
 * one of correlations from the inactive set becomes equal with the
 * correlation from the active set.
 *
 * References:
 *  [1] B. Efron, T. Hastie, I. Johnstone, R Tibshirani, Least Angle Regression
 *  The Annals of Statistics (2004) Vol 32, No 2, 407-499
 *  http://statweb.stanford.edu/~tibs/ftp/lars.pdf
 *
 * @param handle RAFT handle
 * @param max_iter maximum number of iterations
 * @param n_rows number of samples
 * @param n_cols number of valid feature columns
 * @param n_active size of the active set (n_active <= max_iter <= n_cols)
 * @param cj value of the maximum correlation
 * @param A device pointer to a scalar, as defined by eq 2.5 in [1]
 * @param cor device pointer to correlation vector, size [n_active]
 * @param G device pointer to Gram matrix of the active set (without signs)
 *    size [n_active * ld_G]
 * @param ld_G leading dimension of G (ld_G >= n_cols)
 * @param X device array of training vectors in column major format,
 *     size [n_rows * n_cols]. Only used if the gram matrix is not available.
 * @param ld_X leading dimension of X (ld_X >= n_rows)
 * @param u device pointer to equiangular vector size [n_rows]. Only used if the
 *     Gram matrix G is not available.
 * @param ws device pointer to the ws vector defined in calcEquiangularVec,
 *    size [n_active]
 * @param gamma device pointer to a scalar. The max step size is returned here.
 * @param a_vec device pointer, size [n_cols]
 * @param stream CUDA stream
 */
template <typename math_t, typename idx_t = int>
void calcMaxStep(const raft::handle_t& handle,
                 idx_t max_iter,
                 idx_t n_rows,
                 idx_t n_cols,
                 idx_t n_active,
                 math_t cj,
                 const math_t* A,
                 math_t* cor,
                 const math_t* G,
                 idx_t ld_G,
                 const math_t* X,
                 idx_t ld_X,
                 const math_t* u,
                 const math_t* ws,
                 math_t* gamma,
                 math_t* a_vec,
                 cudaStream_t stream)
{
  // In the active set each element has the same correlation, whose absolute
  // value is given by Cmax.
  math_t Cmax = std::abs(cj);
  if (n_active == n_cols) {
    // Last iteration, the inactive set is empty we use equation (2.21)
    raft::linalg::unaryOp(gamma, A, 1, [Cmax] __device__(math_t A) { return Cmax / A; }, stream);
  } else {
    const int n_inactive = n_cols - n_active;
    if (G == nullptr) {
      // Calculate a = X.T[:,n_active:] * u                              (2.11)
      math_t one  = 1;
      math_t zero = 0;
      // #TODO: Call from public API when ready
      RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemv(handle.get_cublas_handle(),
                                                       CUBLAS_OP_T,
                                                       n_rows,
                                                       n_inactive,
                                                       &one,
                                                       X + n_active * ld_X,
                                                       ld_X,
                                                       u,
                                                       1,
                                                       &zero,
                                                       a_vec,
                                                       1,
                                                       stream));
    } else {
      // Calculate a = X.T[:,n_A:] * u = X.T[:, n_A:] * X[:,:n_A] * ws
      //             = G[n_A:,:n_A] * ws                                 (2.11)
      math_t one  = 1;
      math_t zero = 0;
      // #TODO: Call from public API when ready
      RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemv(handle.get_cublas_handle(),
                                                       CUBLAS_OP_N,
                                                       n_inactive,
                                                       n_active,
                                                       &one,
                                                       G + n_active,
                                                       ld_G,
                                                       ws,
                                                       1,
                                                       &zero,
                                                       a_vec,
                                                       1,
                                                       stream));
    }
    const math_t tiny = std::numeric_limits<math_t>::min();
    const math_t huge = std::numeric_limits<math_t>::max();
    //
    // gamma = min^+_{j \in inactive} {(Cmax - cor_j) / (A-a_j),
    //                                 (Cmax + cor_j) / (A+a_j)}         (2.13)
    auto map = [Cmax, A, tiny, huge] __device__(math_t c, math_t a) -> math_t {
      math_t tmp1 = (Cmax - c) / (*A - a + tiny);
      math_t tmp2 = (Cmax + c) / (*A + a + tiny);
      // We consider only positive elements while we search for the minimum
      math_t val = (tmp1 > 0) ? tmp1 : huge;
      if (tmp2 > 0 && tmp2 < val) val = tmp2;
      return val;
    };
    raft::linalg::mapThenReduce(
      gamma, n_inactive, huge, map, cuda::minimum{}, stream, cor + n_active, a_vec);
  }
}

/**
 * @brief Initialize for Lars training.
 *
 * We calculate the initial correlation, initialize the indices array, and set
 * up pointers to store the Cholesky factorization.
 *
 * @param handle RAFT handle
 * @param X device array of training vectors in column major format,
 *     size [ld_X * n_cols].
 * @param n_rows number of samples
 * @param n_cols number of valid feature columns
 * @param ld_X leading dimension of X (ld_X >= n_rows)
 * @param y device pointer to regression targets, size [n_rows]
 * @param Gram device pointer to Gram matrix (X.T * X), size [n_cols * ld_G],
 *    can be nullptr
 * @param ld_G leading dimension of G (ld_G >= n_cols)
 * @param U_buffer device buffer that will be initialized to store the Cholesky
 *    factorization. Only used if Gram is nullptr.
 * @param U device pointer to U
 * @param ld_U leading dimension of U
 * @param indices host buffer to store feature column indices
 * @param cor device pointer to correlation vector, size [n_cols]
 * @param max_iter host pointer to the maximum number of iterations
 * @param coef_path device pointer to store coefficients along the
 *    regularization path size [(max_iter + 1) * max_iter], can be nullptr
 * @param stream CUDA stream
 */
template <typename math_t, typename idx_t>
void larsInit(const raft::handle_t& handle,
              const math_t* X,
              idx_t n_rows,
              idx_t n_cols,
              idx_t ld_X,
              const math_t* y,
              math_t* Gram,
              idx_t ld_G,
              rmm::device_uvector<math_t>& U_buffer,
              math_t** U,
              idx_t* ld_U,
              std::vector<idx_t>& indices,
              rmm::device_uvector<math_t>& cor,
              int* max_iter,
              math_t* coef_path,
              cudaStream_t stream)
{
  if (n_cols < *max_iter) { *max_iter = n_cols; }
  if (Gram == nullptr) {
    const idx_t align_bytes = 256;
    *ld_U                   = raft::alignTo<idx_t>(*max_iter, align_bytes);
    try {
      U_buffer.resize(checked_mul<std::size_t>(*ld_U, *max_iter), stream);
    } catch (std::bad_alloc const&) {
      THROW(
        "Not enough GPU memory! The memory usage depends quadraticaly on the "
        "n_nonzero_coefs parameter, try to decrease it.");
    }
    *U = U_buffer.data();
  } else {
    // Set U as G. During the solution in larsFit, the Cholesky factorization
    // U will overwrite G.
    *U    = Gram;
    *ld_U = ld_G;
  }
  std::iota(indices.data(), indices.data() + n_cols, 0);

  math_t one  = 1;
  math_t zero = 0;
  // Set initial correlation to X.T * y
  // #TODO: Call from public API when ready
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemv(handle.get_cublas_handle(),
                                                   CUBLAS_OP_T,
                                                   n_rows,
                                                   n_cols,
                                                   &one,
                                                   X,
                                                   ld_X,
                                                   y,
                                                   1,
                                                   &zero,
                                                   cor.data(),
                                                   1,
                                                   stream));
  if (coef_path) {
    RAFT_CUDA_TRY(
      cudaMemsetAsync(coef_path, 0, sizeof(math_t) * (*max_iter + 1) * (*max_iter), stream));
  }
}

/**
 * @brief Update regression coefficient and correlations
 *
 * After we calculated the equiangular vector and the step size (gamma) we
 * adjust the regression coefficients here.
 *
 * See calcEquiangularVec for definition of ws.
 *
 * @param handle RAFT handle
 * @param max_iter maximum number of iterations
 * @param n_cols number of valid feature columns
 * @param n_active number of elements in the active set (n_active <= n_cols)
 * @param gamma device pointer to the maximum step size (scalar)
 * @param ws device pointer to the ws vector, size [n_cols]
 * @param cor device pointer to the correlations, size [n_cols]
 * @param a_vec device pointer to a = X.T[:,n_A:] * u, size [n_cols]
 * @param beta pointer to regression coefficients, size [max_iter]
 * @param coef_path device pointer to all the coefficients along the
 *    regularization path, size [(max_iter + 1) * max_iter]
 * @param stream CUDA stream
 */
template <typename math_t, typename idx_t>
void updateCoef(const raft::handle_t& handle,
                idx_t max_iter,
                idx_t n_cols,
                idx_t n_active,
                math_t* gamma,
                const math_t* ws,
                math_t* cor,
                math_t* a_vec,
                math_t* beta,
                math_t* coef_path,
                cudaStream_t stream)
{
  // It is sufficient to update correlations only for the inactive set.
  // cor[n_active:] -= gamma * a_vec
  int n_inactive = n_cols - n_active;
  if (n_inactive > 0) {
    raft::linalg::binaryOp(
      cor + n_active,
      cor + n_active,
      a_vec,
      n_inactive,
      [gamma] __device__(math_t c, math_t a) { return c - *gamma * a; },
      stream);
  }
  // beta[:n_active] += gamma * ws
  raft::linalg::binaryOp(
    beta,
    beta,
    ws,
    n_active,
    [gamma] __device__(math_t b, math_t w) { return b + *gamma * w; },
    stream);
  if (coef_path) { raft::copy(coef_path + n_active * max_iter, beta, n_active, stream); }
}

/**
 * @brief Train a regressor using Least Angre Regression.
 *
 * Least Angle Regression (LAR or LARS) is a model selection algorithm. It
 * builds up the model using the following algorithm:
 *
 * 1. We start with all the coefficients equal to zero.
 * 2. At each step we select the predictor that has the largest absolute
 *      correlation with the residual.
 * 3. We take the largest step possible in the direction which is equiangular
 *    with all the predictors selected so far. The largest step is determined
 *    such that using this step a new predictor will have as much correlation
 *    with the residual as any of the currently active predictors.
 * 4. Stop if max_iter reached or all the predictors are used, or if the
 *    correlation between any unused predictor and the residual is lower than
 *    a tolerance.
 *
 * The solver is based on [1]. The equations referred in the comments correspond
 * to the equations in the paper.
 *
 * Note: this algorithm assumes that the offset is removed from X and y, and
 * each feature is normalized:
 * - sum_i y_i = 0,
 * - sum_i x_{i,j} = 0, sum_i x_{i,j}^2=1 for j=0..n_col-1
 *
 * References:
 * [1] B. Efron, T. Hastie, I. Johnstone, R Tibshirani, Least Angle Regression
 * The Annals of Statistics (2004) Vol 32, No 2, 407-499
 * http://statweb.stanford.edu/~tibs/ftp/lars.pdf
 *
 * @param handle RAFT handle
 * @param X device array of training vectors in column major format,
 *     size [n_rows * n_cols]. Note that the columns of X will be permuted if
 *     the Gram matrix is not specified. It is expected that X is normalized so
 *     that each column has zero mean and unit variance.
 * @param n_rows number of training samples
 * @param n_cols number of feature columns
 * @param y device array of the regression targets, size [n_rows]. y should
 *     be normalized to have zero mean.
 * @param beta device array of regression coefficients, has to be allocated on
 *     entry, size [max_iter]
 * @param active_idx device array containing the indices of active variables.
 *     Must be allocated on entry. Size [max_iter]
 * @param alphas device array to return the maximum correlation along the
 *     regularization path. Must be allocated on entry, size [max_iter+1].
 * @param n_active host pointer to return the number of active elements (scalar)
 * @param Gram device array containing Gram matrix containing X.T * X. Can be
 *     nullptr.
 * @param max_iter maximum number of iterations, this equals with the maximum
 *    number of coefficients returned. max_iter <= n_cols.
 * @param coef_path coefficients along the regularization path are returned
 *    here. Must be nullptr, or a device array already allocated on entry.
 *    Size [max_iter * (max_iter+1)].
 * @param verbosity verbosity level
 * @param ld_X leading dimension of X (stride of columns)
 * @param ld_G leading dimension of G
 * @param eps numeric parameter for Cholesky rank one update
 */
template <typename math_t, typename idx_t>
void larsFit(const raft::handle_t& handle,
             math_t* X,
             idx_t n_rows,
             idx_t n_cols,
             const math_t* y,
             math_t* beta,
             idx_t* active_idx,
             math_t* alphas,
             idx_t* n_active,
             math_t* Gram,
             int max_iter,
             math_t* coef_path,
             rapids_logger::level_enum verbosity,
             idx_t ld_X,
             idx_t ld_G,
             math_t eps)
{
  ASSERT(n_cols > 0, "Parameter n_cols: number of columns cannot be less than one");
  ASSERT(n_rows > 0, "Parameter n_rows: number of rows cannot be less than one");
  ML::default_logger().set_level(verbosity);

  // Set default ld parameters if needed.
  if (ld_X == 0) ld_X = n_rows;
  if (Gram && ld_G == 0) ld_G = n_cols;

  cudaStream_t stream = handle.get_stream().get();

  // We will use either U_buffer.data() to store the Cholesky factorization, or
  // store it in place at Gram. Pointer U will point to the actual storage.
  rmm::device_uvector<math_t> U_buffer(0, stream);
  idx_t ld_U = 0;
  math_t* U  = nullptr;

  // Indices of elements in the active set.
  std::vector<idx_t> indices(n_cols);
  // Sign of the correlation at the time when the element was added to the
  // active set.
  rmm::device_uvector<math_t> sign(n_cols, stream);

  // Correlation between the residual mu = y - X.T*beta and columns of X
  rmm::device_uvector<math_t> cor(n_cols, stream);

  // Temporary arrays used by the solver
  rmm::device_scalar<math_t> A(stream);
  rmm::device_uvector<math_t> a_vec(n_cols, stream);
  rmm::device_scalar<math_t> gamma(stream);
  rmm::device_uvector<math_t> u_eq(n_rows, stream);
  rmm::device_uvector<math_t> ws(max_iter, stream);
  rmm::device_uvector<math_t> workspace(n_cols, stream);

  larsInit(handle,
           X,
           n_rows,
           n_cols,
           ld_X,
           y,
           Gram,
           ld_G,
           U_buffer,
           &U,
           &ld_U,
           indices,
           cor,
           &max_iter,
           coef_path,
           stream);

  // If we detect collinear features, then we will move them to the end of the
  // correlation array and mark them as invalid (simply by decreasing
  // n_valid_cols). At every iteration the solver is only working with the valid
  // columns stored at X[:,:n_valid_cols], and G[:n_valid_cols, :n_valid_cols]
  // cor[:n_valid_cols].
  int n_valid_cols = n_cols;

  *n_active = 0;
  for (int i = 0; i < max_iter; i++) {
    math_t cj;
    idx_t j;
    LarsFitStatus status = selectMostCorrelated(
      *n_active, n_valid_cols, cor.data(), &cj, workspace, &j, n_rows, indices.data(), i, stream);
    if (status != LarsFitStatus::kOk) { break; }

    moveToActive(handle.get_cublas_handle(),
                 n_active,
                 j,
                 X,
                 n_rows,
                 n_valid_cols,
                 ld_X,
                 cor.data(),
                 indices.data(),
                 Gram,
                 ld_G,
                 sign.data(),
                 stream);

    status = calcEquiangularVec(handle,
                                *n_active,
                                X,
                                n_rows,
                                n_valid_cols,
                                ld_X,
                                sign.data(),
                                U,
                                ld_U,
                                Gram,
                                ld_G,
                                workspace,
                                ws.data(),
                                A.data(),
                                u_eq.data(),
                                eps,
                                stream);

    if (status == LarsFitStatus::kError) {
      if (*n_active > 1) { CUML_LOG_WARN("Returning with last valid model."); }
      *n_active -= 1;
      break;
    } else if (status == LarsFitStatus::kCollinear) {
      // We move the current feature to the invalid set
      swapFeatures(handle.get_cublas_handle(),
                   n_valid_cols - 1,
                   *n_active - 1,
                   X,
                   n_rows,
                   n_cols,
                   ld_X,
                   cor.data(),
                   indices.data(),
                   Gram,
                   ld_G,
                   stream);
      *n_active -= 1;
      n_valid_cols--;
      continue;
    }

    calcMaxStep(handle,
                max_iter,
                n_rows,
                n_valid_cols,
                *n_active,
                cj,
                A.data(),
                cor.data(),
                Gram,
                ld_G,
                X,
                ld_X,
                u_eq.data(),
                ws.data(),
                gamma.data(),
                a_vec.data(),
                stream);

    updateCoef(handle,
               max_iter,
               n_valid_cols,
               *n_active,
               gamma.data(),
               ws.data(),
               cor.data(),
               a_vec.data(),
               beta,
               coef_path,
               stream);
  }

  if (*n_active > 0) {
    // Apply sklearn definition of alphas = cor / n_rows
    raft::linalg::unaryOp(
      alphas,
      cor.data(),
      *n_active,
      [n_rows] __device__(math_t c) { return abs(c) / n_rows; },
      stream);

    // Calculate the final correlation. We use the correlation from the last
    // iteration and apply the changed during the last LARS iteration:
    // alpha[n_active] = cor[n_active-1] - gamma * A
    math_t* gamma_ptr = gamma.data();
    math_t* A_ptr     = A.data();
    raft::linalg::unaryOp(
      alphas + *n_active,
      cor.data() + *n_active - 1,
      1,
      [gamma_ptr, A_ptr, n_rows] __device__(math_t c) {
        return abs(c - (*gamma_ptr) * (*A_ptr)) / n_rows;
      },
      stream);

    raft::update_device(active_idx, indices.data(), *n_active, stream);
  } else {
    THROW("Model is not fitted.");
  }
}

/**
 * @brief Predict with least angle regressor.
 *
 * @param handle RAFT handle
 * @param X device array of training vectors in column major format,
 *     size [n_rows * n_cols].
 * @param n_rows number of training samples
 * @param n_cols number of feature columns
 * @param ld_X leading dimension of X (stride of columns)
 * @param beta device array of regression coefficients, size [n_active]
 * @param n_active the number of regression coefficients
 * @param active_idx device array containing the indices of active variables.
 *     Only these columns of X will be used for prediction, size [n_active].
 * @param intercept
 * @param preds device array to store the predictions, size [n_rows]. Must be
 *     allocated on entry.
 */
template <typename math_t, typename idx_t>
void larsPredict(const raft::handle_t& handle,
                 const math_t* X,
                 idx_t n_rows,
                 idx_t n_cols,
                 idx_t ld_X,
                 const math_t* beta,
                 idx_t n_active,
                 idx_t* active_idx,
                 math_t intercept,
                 math_t* preds)
{
  cudaStream_t stream = handle.get_stream().get();
  rmm::device_uvector<math_t> beta_sorted(0, stream);
  rmm::device_uvector<math_t> X_active_cols(0, stream);
  auto execution_policy = handle.get_thrust_policy();

  if (n_active == 0 || n_rows == 0) return;

  if (n_active == n_cols) {
    // We make a copy of the beta coefs and sort them
    beta_sorted.resize(n_active, stream);
    rmm::device_uvector<idx_t> idx_sorted(n_active, stream);
    raft::copy(beta_sorted.data(), beta, n_active, stream);
    raft::copy(idx_sorted.data(), active_idx, n_active, stream);
    thrust::device_ptr<math_t> beta_ptr(beta_sorted.data());
    thrust::device_ptr<idx_t> idx_ptr(idx_sorted.data());
    thrust::sort_by_key(execution_policy, idx_ptr, idx_ptr + n_active, beta_ptr);
    beta = beta_sorted.data();
  } else {
    // We collect active columns of X to contiguous space
    std::size_t const active_count = checked_mul<std::size_t>(n_active, ld_X);
    X_active_cols.resize(active_count, stream);
    const int TPB       = 64;
    auto const n_blocks = narrow_cast<ML::cuda_launch_t>(
      raft::ceildiv<std::size_t>(active_count, static_cast<std::size_t>(TPB)));
    raft::cache::get_vecs<<<n_blocks, TPB, 0, stream>>>(
      X, ld_X, active_idx, n_active, X_active_cols.data());
    RAFT_CUDA_TRY(cudaGetLastError());
    X = X_active_cols.data();
  }
  // Initialize preds = intercept
  thrust::device_ptr<math_t> pred_ptr(preds);
  thrust::fill(execution_policy, pred_ptr, pred_ptr + n_rows, intercept);
  math_t one = 1;
  // #TODO: Call from public API when ready
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemv(handle.get_cublas_handle(),
                                                   CUBLAS_OP_N,
                                                   n_rows,
                                                   n_active,
                                                   &one,
                                                   X,
                                                   ld_X,
                                                   beta,
                                                   1,
                                                   &one,
                                                   preds,
                                                   1,
                                                   stream));
}
};  // namespace Lars
};  // namespace Solver
};  // namespace ML
