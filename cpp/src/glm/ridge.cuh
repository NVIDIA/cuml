/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "preprocess.cuh"

#include <raft/linalg/add.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/matrix_vector.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/power.cuh>
#include <raft/linalg/sqrt.cuh>
#include <raft/linalg/subtract.cuh>
#include <raft/linalg/svd.cuh>
#include <raft/matrix/threshold.cuh>
#include <raft/stats/mean.cuh>
#include <raft/stats/mean_center.cuh>
#include <raft/stats/stddev.cuh>
#include <raft/stats/sum.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

namespace ML {
namespace GLM {
namespace detail {

template <typename math_t>
void ridgeSolve(const raft::handle_t& handle,
                math_t* S,
                math_t* V,
                math_t* U,
                size_t n_rows,
                size_t n_cols,
                math_t* b,
                math_t* alpha,
                int n_alpha,
                math_t* w)
{
  auto stream    = handle.get_stream().get();
  auto cublasH   = handle.get_cublas_handle();
  auto cusolverH = handle.get_cusolver_dn_handle();

  // Implements this: w = V * inv(S^2 + λ*I) * S * U^T * b
  rmm::device_uvector<math_t> S_nnz_vector(n_cols, stream);
  math_t* S_nnz = S_nnz_vector.data();
  math_t alp    = math_t(1);
  math_t beta   = math_t(0);
  math_t thres  = math_t(1e-10);

  raft::matrix::zero_small_values(
    handle,
    raft::make_device_matrix_view<math_t, size_t, raft::col_major>(S, size_t(1), n_cols),
    thres);

  raft::copy(S_nnz, S, n_cols, stream);
  raft::linalg::powerScalar(S_nnz, S_nnz, math_t(2), n_cols, stream);
  raft::linalg::addScalar(S_nnz, S_nnz, alpha[0], n_cols, stream);
  raft::linalg::binary_div_skip_zero<raft::Apply::ALONG_ROWS>(
    handle,
    raft::make_device_matrix_view<math_t, size_t, raft::col_major>(S, size_t(1), n_cols),
    raft::make_device_vector_view<const math_t, size_t>(S_nnz, n_cols),
    true);

  raft::linalg::binary_mult<raft::Apply::ALONG_ROWS>(
    handle,
    raft::make_device_matrix_view<math_t, size_t, raft::col_major>(V, n_cols, n_cols),
    raft::make_device_vector_view<const math_t, size_t>(S, n_cols));
  raft::linalg::gemm(
    handle, U, n_rows, n_cols, b, S_nnz, n_cols, 1, CUBLAS_OP_T, CUBLAS_OP_N, alp, beta, stream);

  raft::linalg::gemm(
    handle, V, n_cols, n_cols, S_nnz, w, n_cols, 1, CUBLAS_OP_N, CUBLAS_OP_N, alp, beta, stream);
}

template <typename math_t>
void ridgeSVD(const raft::handle_t& handle,
              math_t* A,
              size_t n_rows,
              size_t n_cols,
              math_t* b,
              math_t* alpha,
              int n_alpha,
              math_t* w)
{
  auto stream    = handle.get_stream().get();
  auto cublasH   = handle.get_cublas_handle();
  auto cusolverH = handle.get_cusolver_dn_handle();

  ASSERT(n_cols > 0, "ridgeSVD: number of columns cannot be less than one");
  ASSERT(n_rows > 1, "ridgeSVD: number of rows cannot be less than two");

  auto U_len = n_rows * n_cols;
  auto V_len = n_cols * n_cols;

  rmm::device_uvector<math_t> S(n_cols, stream);
  rmm::device_uvector<math_t> V(V_len, stream);
  rmm::device_uvector<math_t> U(U_len, stream);

  raft::linalg::svdQR(
    handle, A, n_rows, n_cols, S.data(), U.data(), V.data(), true, true, true, stream);
  ridgeSolve(handle, S.data(), V.data(), U.data(), n_rows, n_cols, b, alpha, n_alpha, w);
}

template <typename math_t>
void ridgeEig(const raft::handle_t& handle,
              math_t* A,
              size_t n_rows,
              size_t n_cols,
              math_t* b,
              math_t* alpha,
              int n_alpha,
              math_t* w)
{
  auto stream    = handle.get_stream().get();
  auto cublasH   = handle.get_cublas_handle();
  auto cusolverH = handle.get_cusolver_dn_handle();

  ASSERT(n_cols > 1, "ridgeEig: number of columns cannot be less than two");
  ASSERT(n_rows > 1, "ridgeEig: number of rows cannot be less than two");

  auto U_len = n_rows * n_cols;
  auto V_len = n_cols * n_cols;

  rmm::device_uvector<math_t> S(n_cols, stream);
  rmm::device_uvector<math_t> V(V_len, stream);
  rmm::device_uvector<math_t> U(U_len, stream);

  raft::linalg::svdEig(handle, A, n_rows, n_cols, S.data(), U.data(), V.data(), true, stream);

  ridgeSolve(handle, S.data(), V.data(), U.data(), n_rows, n_cols, b, alpha, n_alpha, w);
}

/**
 * @brief fit a ridge regression model (l2 regularized least squares)
 * @param handle        cuml handle
 * @param input         device pointer to feature matrix n_rows x n_cols (col-major)
 * @param n_rows        number of rows of the feature matrix
 * @param n_cols        number of columns of the feature matrix
 * @param labels        device pointer to label vector of length n_rows
 * @param alpha         host pointer to parameters of the l2 regularizer
 * @param n_alpha       number of regularization parameters
 * @param coef          device pointer to hold the solution for weights of size n_cols
 * @param intercept     host pointer to hold the solution for bias term of size 1
 * @param fit_intercept if true, fit intercept
 * @param algo          specifies which solver to use (0: SVD, 1: Eigendecomposition)
 * @param sample_weight device pointer to sample weight vector of length n_rows (nullptr for uniform
 * weights) This vector is modified during the computation
 */
template <typename math_t>
void ridgeFit(const raft::handle_t& handle,
              math_t* input,
              size_t n_rows,
              size_t n_cols,
              math_t* labels,
              math_t* alpha,
              int n_alpha,
              math_t* coef,
              math_t* intercept,
              bool fit_intercept,
              int algo              = 0,
              math_t* sample_weight = nullptr)
{
  cudaStream_t stream  = handle.get_stream().get();
  auto cublas_handle   = handle.get_cublas_handle();
  auto cusolver_handle = handle.get_cusolver_dn_handle();

  ASSERT(n_cols > 0, "ridgeFit: number of columns cannot be less than one");
  ASSERT(n_rows > 1, "ridgeFit: number of rows cannot be less than two");

  rmm::device_uvector<math_t> mu_input(0, stream);
  rmm::device_uvector<math_t> mu_labels(0, stream);

  if (fit_intercept) {
    mu_input.resize(n_cols, stream);
    mu_labels.resize(1, stream);
    preProcessData(handle,
                   input,
                   n_rows,
                   n_cols,
                   labels,
                   intercept,
                   mu_input.data(),
                   mu_labels.data(),
                   fit_intercept,
                   sample_weight);
  }
  if (sample_weight != nullptr) {
    raft::linalg::sqrt(sample_weight, sample_weight, n_rows, stream);
    raft::linalg::binary_mult<raft::Apply::ALONG_COLUMNS>(
      handle,
      raft::make_device_matrix_view<math_t, size_t, raft::col_major>(input, n_rows, n_cols),
      raft::make_device_vector_view<const math_t, size_t>(sample_weight, n_rows));
    raft::linalg::map_k(
      labels,
      n_rows,
      [] __device__(math_t a, math_t b) { return a * b; },
      stream,
      labels,
      sample_weight);
  }

  if (algo == 0 || n_cols == 1) {
    ridgeSVD(handle, input, n_rows, n_cols, labels, alpha, n_alpha, coef);
  } else if (algo == 1) {
    ridgeEig(handle, input, n_rows, n_cols, labels, alpha, n_alpha, coef);
  } else if (algo == 2) {
    ASSERT(false, "ridgeFit: no algorithm with this id has been implemented");
  } else {
    ASSERT(false, "ridgeFit: no algorithm with this id has been implemented");
  }

  if (sample_weight != nullptr) {
    raft::linalg::binary_div_skip_zero<raft::Apply::ALONG_COLUMNS>(
      handle,
      raft::make_device_matrix_view<math_t, size_t, raft::col_major>(input, n_rows, n_cols),
      raft::make_device_vector_view<const math_t, size_t>(sample_weight, n_rows));
    raft::linalg::map_k(
      labels,
      n_rows,
      [] __device__(math_t a, math_t b) { return a / b; },
      stream,
      labels,
      sample_weight);
    raft::linalg::powerScalar(sample_weight, sample_weight, (math_t)2, n_rows, stream);
  }

  if (fit_intercept) {
    postProcessData(handle,
                    input,
                    n_rows,
                    n_cols,
                    labels,
                    coef,
                    intercept,
                    mu_input.data(),
                    mu_labels.data(),
                    fit_intercept);
  } else {
    *intercept = math_t(0);
  }
}
};  // namespace detail
};  // namespace GLM
};  // namespace ML
