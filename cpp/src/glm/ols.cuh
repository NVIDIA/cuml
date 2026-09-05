/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "preprocess.cuh"

#include <raft/linalg/add.cuh>
#include <raft/linalg/gemv.cuh>
#include <raft/linalg/lstsq.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/matrix_vector.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/power.cuh>
#include <raft/linalg/sqrt.cuh>
#include <raft/linalg/subtract.cuh>
#include <raft/stats/mean.cuh>
#include <raft/stats/mean_center.cuh>
#include <raft/stats/stddev.cuh>
#include <raft/stats/sum.cuh>

#include <rmm/device_uvector.hpp>

namespace ML {
namespace GLM {
namespace detail {

/**
 * @brief fit an ordinary least squares model
 * @param handle        cuml handle
 * @param input         device pointer to feature matrix n_rows x n_cols
 * @param n_rows        number of rows of the feature matrix
 * @param n_cols        number of columns of the feature matrix
 * @param labels        device pointer to label vector of length n_rows
 * @param coef          device pointer to hold the solution for weights of size n_cols
 * @param intercept     host pointer to hold the solution for bias term of size 1
 * @param fit_intercept if true, fit intercept
 * @param algo          specifies which solver to use (0: SVD, 1: Eigendecomposition, 2:
 * QR-decomposition)
 * @param sample_weight device pointer to sample weight vector of length n_rows (nullptr for uniform
 * weights) This vector is modified during the computation
 */
template <typename math_t>
void olsFit(const raft::handle_t& handle,
            math_t* input,
            size_t n_rows,
            size_t n_cols,
            math_t* labels,
            math_t* coef,
            math_t* intercept,
            bool fit_intercept,
            int algo              = 0,
            math_t* sample_weight = nullptr)
{
  cudaStream_t stream  = handle.get_stream().get();
  auto cublas_handle   = handle.get_cublas_handle();
  auto cusolver_handle = handle.get_cusolver_dn_handle();

  ASSERT(n_cols > 0, "olsFit: number of columns cannot be less than one");
  ASSERT(n_rows > 1, "olsFit: number of rows cannot be less than two");

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

  int selectedAlgo = algo;
  if (n_cols > n_rows || n_cols == 1) selectedAlgo = 0;

  raft::common::nvtx::push_range("ML::GLM::olsFit/algo-%d", selectedAlgo);
  switch (selectedAlgo) {
    case 0:
      raft::linalg::lstsqSvdJacobi(handle, input, n_rows, n_cols, labels, coef, stream);
      break;
    case 1: raft::linalg::lstsqEig(handle, input, n_rows, n_cols, labels, coef, stream); break;
    case 2: raft::linalg::lstsqQR(handle, input, n_rows, n_cols, labels, coef, stream); break;
    case 3: raft::linalg::lstsqSvdQR(handle, input, n_rows, n_cols, labels, coef, stream); break;
    default:
      ASSERT(false, "olsFit: no algorithm with this id (%d) has been implemented", algo);
      break;
  }
  raft::common::nvtx::pop_range();

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

/**
 * @brief to make predictions with a fitted ordinary least squares and ridge regression model
 * @param handle        cuml ahndle
 * @param input         device pointer to feature matrix n_rows x n_cols
 * @param n_rows        number of rows of the feature matrix
 * @param n_cols        number of columns of the feature matrix
 * @param coef          coefficients of the model
 * @param intercept     bias term of the model
 * @param preds         device pointer to store predictions of size n_rows
 */
template <typename math_t>
void gemmPredict(const raft::handle_t& handle,
                 const math_t* input,
                 size_t n_rows,
                 size_t n_cols,
                 const math_t* coef,
                 math_t intercept,
                 math_t* preds)
{
  ASSERT(n_cols > 0, "gemmPredict: number of columns cannot be less than one");
  ASSERT(n_rows > 0, "gemmPredict: number of rows cannot be less than one");

  cudaStream_t stream = handle.get_stream().get();
  math_t alpha        = math_t(1);
  math_t beta         = math_t(0);
  raft::linalg::gemm(handle,
                     input,
                     n_rows,
                     n_cols,
                     coef,
                     preds,
                     n_rows,
                     1,
                     CUBLAS_OP_N,
                     CUBLAS_OP_N,
                     alpha,
                     beta,
                     stream);

  if (intercept != math_t(0)) raft::linalg::addScalar(preds, preds, intercept, n_rows, stream);
}
};  // namespace detail
};  // namespace GLM
};  // namespace ML
