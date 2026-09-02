/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "shuffle.h"

#include <cuml/common/utils.hpp>
#include <cuml/solvers/params.hpp>

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/axpy.cuh>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/gemv.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/matrix_vector.cuh>
#include <raft/linalg/multiply.cuh>
#include <raft/linalg/power.cuh>
#include <raft/linalg/sqrt.cuh>
#include <raft/linalg/subtract.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/stats/sum.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <functions/linearReg.cuh>
#include <functions/penalty.cuh>
#include <functions/softThres.cuh>
#include <glm/preprocess.cuh>

namespace ML {
namespace Solver {

using namespace MLCommon;

namespace {

/** Epoch and iteration -related state. */
template <typename math_t>
struct ConvState {
  math_t coef;
  math_t coefMax;
  math_t diffMax;
};

/**
 * Update a single CD coefficient and the corresponding convergence criteria.
 *
 * @param[inout] coefLoc pointer to the coefficient (arr ptr + column index offset)
 * @param[in] squaredLoc pointer to the precomputed data - L2 norm of input for across rows
 * @param[inout] convStateLoc pointer to the structure holding the convergence state
 * @param[in] l1_alpha L1 regularization coef
 */
template <typename math_t>
CUML_KERNEL void __launch_bounds__(1, 1) cdUpdateCoefKernel(math_t* coefLoc,
                                                            const math_t* squaredLoc,
                                                            ConvState<math_t>* convStateLoc,
                                                            const math_t l1_alpha)
{
  auto coef    = *coefLoc;
  auto r       = coef > l1_alpha ? coef - l1_alpha : (coef < -l1_alpha ? coef + l1_alpha : 0);
  auto squared = *squaredLoc;
  r            = squared > math_t(1e-5) ? r / squared : math_t(0);
  auto diff    = raft::abs(convStateLoc->coef - r);
  if (convStateLoc->diffMax < diff) convStateLoc->diffMax = diff;
  auto absv = raft::abs(r);
  if (convStateLoc->coefMax < absv) convStateLoc->coefMax = absv;
  convStateLoc->coef = -r;
  *coefLoc           = r;
}

}  // namespace

/**
 * Fits a linear, lasso, and elastic-net regression model using Coordinate Descent solver.
 *
 * i.e. finds coefficients that minimize the following loss function:
 *
 * f(coef) = 1/2 * || labels - input * coef ||^2
 *         + 1/2 * alpha * (1 - l1_ratio) * ||coef||^2
 *         +       alpha *    l1_ratio    * ||coef||_1
 *
 *
 * @param handle
 *        Reference of raft::handle_t
 * @param input
 *        pointer to an array in column-major format (size of n_rows, n_cols)
 * @param n_rows
 *        n_samples or rows in input
 * @param n_cols
 *        n_features or columns in X
 * @param labels
 *        pointer to an array for labels (size of n_rows)
 * @param coef
 *        pointer to an array for coefficients (size of n_cols). This will be filled with
 * coefficients once the function is executed.
 * @param intercept
 *        pointer to a scalar for intercept. This will be filled
 *        once the function is executed
 * @param fit_intercept
 *        boolean parameter to control if the intercept will be fitted or not
 * @param epochs
 *        Maximum number of iterations that solver will run
 * @param loss
 *        enum to use different loss functions. Only linear regression loss functions is supported
 * right now
 * @param alpha
 *        L1 parameter
 * @param l1_ratio
 *        ratio of alpha will be used for L1. (1 - l1_ratio) * alpha will be used for L2
 * @param shuffle
 *        boolean parameter to control whether coordinates will be picked randomly or not
 * @param tol
 *        tolerance to stop the solver
 * @param sample_weight
 *        device pointer to sample weight vector of length n_rows (nullptr or uniform weights)
 *        This vector is modified during the computation
 * @return n_iter
 *        The number of iterations the solver ran for.
 */
template <typename math_t>
int cdFit(const raft::handle_t& handle,
          math_t* input,
          int n_rows,
          int n_cols,
          math_t* labels,
          math_t* coef,
          math_t* intercept,
          bool fit_intercept,
          int epochs,
          ML::loss_funct loss,
          math_t alpha,
          math_t l1_ratio,
          bool shuffle,
          math_t tol,
          math_t* sample_weight = nullptr)
{
  raft::common::nvtx::range fun_scope("ML::Solver::cdFit-%d-%d", n_rows, n_cols);
  ASSERT(n_cols > 0, "Parameter n_cols: number of columns cannot be less than one");
  ASSERT(n_rows > 1, "Parameter n_rows: number of rows cannot be less than two");
  ASSERT(loss == ML::loss_funct::SQRD_LOSS,
         "Parameter loss: Only SQRT_LOSS function is supported for now");

  cudaStream_t stream = handle.get_stream();
  rmm::device_uvector<math_t> residual(n_rows, stream);
  rmm::device_uvector<math_t> squared(n_cols, stream);
  rmm::device_uvector<math_t> mu_input(0, stream);
  rmm::device_uvector<math_t> mu_labels(0, stream);
  math_t h_sum_sw = 0;

  if (sample_weight != nullptr) {
    rmm::device_scalar<math_t> sum_sw(stream);
    raft::stats::sum<true>(sum_sw.data(), sample_weight, 1, n_rows, stream);
    raft::update_host(&h_sum_sw, sum_sw.data(), 1, stream);

    raft::linalg::multiplyScalar(
      sample_weight, sample_weight, (math_t)n_rows / h_sum_sw, n_rows, stream);
  }

  if (fit_intercept) {
    mu_input.resize(n_cols, stream);
    mu_labels.resize(1, stream);

    GLM::preProcessData(handle,
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
    raft::linalg::sqrt(handle,
                       raft::make_device_vector_view<const math_t, int>(sample_weight, n_rows),
                       raft::make_device_vector_view<math_t, int>(sample_weight, n_rows));
    raft::linalg::binary_mult<raft::Apply::ALONG_COLUMNS>(
      handle,
      raft::make_device_matrix_view<math_t, int, raft::col_major>(input, n_rows, n_cols),
      raft::make_device_vector_view<const math_t, int>(sample_weight, n_rows));
    raft::linalg::map_k(
      labels,
      n_rows,
      [] __device__(math_t a, math_t b) { return a * b; },
      stream,
      labels,
      sample_weight);
  }

  std::vector<int> ri(n_cols);
  std::mt19937 g(rand());
  initShuffle(ri, g);

  math_t l2_alpha = (1 - l1_ratio) * alpha * n_rows;
  math_t l1_alpha = l1_ratio * alpha * n_rows;

  // Precompute the residual
  /* (n_cols * n_rows) may overflow, upcast for indexing */
  raft::linalg::colNorm<raft::linalg::NormType::L2Norm, false>(
    squared.data(), input, int64_t(n_cols), int64_t(n_rows), stream);
  raft::linalg::addScalar(squared.data(), squared.data(), l2_alpha, n_cols, stream);

  raft::copy(residual.data(), labels, n_rows, stream);

  ConvState<math_t> h_convState;
  rmm::device_uvector<ConvState<math_t>> convStateBuf(1, stream);
  auto convStateLoc = convStateBuf.data();

  // Passed to gemv in host pointer mode, so cuBLAS reads them during the call.
  math_t const cublas_alpha = 1.0;
  math_t const cublas_beta  = 0.0;

  int n_iter = 0;
  while (n_iter < epochs) {
    raft::common::nvtx::range epoch_scope("ML::Solver::cdFit::epoch-%d", n_iter);
    if (n_iter > 0 && shuffle) { Solver::shuffle(ri, g); }

    RAFT_CUDA_TRY(cudaMemsetAsync(convStateLoc, 0, sizeof(ConvState<math_t>), stream));

    for (int j = 0; j < n_cols; j++) {
      raft::common::nvtx::range iter_scope("ML::Solver::cdFit::col-%d", j);
      int64_t ci            = ri[j];
      math_t* coef_loc      = coef + ci;
      math_t* squared_loc   = squared.data() + ci;
      math_t* input_col_loc = input + (ci * int64_t(n_rows));

      // remember current coef
      raft::copy(&(convStateLoc->coef), coef_loc, 1, stream);
      // calculate the residual without the contribution from column ci
      // residual[:] += coef[ci] * X[:, ci]
      raft::linalg::axpy<math_t, true>(
        handle, n_rows, coef_loc, input_col_loc, 1, residual.data(), 1, stream);

      // coef[ci] = dot(X[:, ci], residual[:])
      raft::linalg::gemv<math_t, false>(handle,
                                        false,
                                        1,
                                        n_rows,
                                        &cublas_alpha,
                                        input_col_loc,
                                        1,
                                        residual.data(),
                                        1,
                                        &cublas_beta,
                                        coef_loc,
                                        1,
                                        stream);

      // Calculate the new coefficient that minimizes f along coordinate line ci
      // coef[ci] = SoftTreshold(dot(X[:, ci], residual[:]), l1_alpha) /  dot(X[:, ci], X[:, ci]))
      // Also, update the convergence criteria.
      cdUpdateCoefKernel<math_t><<<dim3(1, 1, 1), dim3(1, 1, 1), 0, stream>>>(
        coef_loc, squared_loc, convStateLoc, l1_alpha);
      RAFT_CUDA_TRY(cudaGetLastError());

      // Restore the residual using the updated coeffecient
      raft::linalg::axpy<math_t, true>(
        handle, n_rows, &(convStateLoc->coef), input_col_loc, 1, residual.data(), 1, stream);
    }
    raft::update_host(&h_convState, convStateLoc, 1, stream);
    handle.sync_stream(stream);

    // Iteration completed, increase n_iter and check early stopping criteria
    n_iter++;
    if (h_convState.coefMax < tol || (h_convState.diffMax / h_convState.coefMax) < tol) break;
  }

  if (sample_weight != nullptr) {
    raft::linalg::binary_div_skip_zero<raft::Apply::ALONG_COLUMNS>(
      handle,
      raft::make_device_matrix_view<math_t, int, raft::col_major>(input, n_rows, n_cols),
      raft::make_device_vector_view<const math_t, int>(sample_weight, n_rows));
    raft::linalg::map_k(
      labels,
      n_rows,
      [] __device__(math_t a, math_t b) { return a / b; },
      stream,
      labels,
      sample_weight);
    raft::linalg::powerScalar(sample_weight, sample_weight, (math_t)2, n_rows, stream);
    raft::linalg::multiplyScalar(sample_weight, sample_weight, h_sum_sw / n_rows, n_rows, stream);
  }

  if (fit_intercept) {
    GLM::postProcessData(handle,
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

  return n_iter;
}

/**
 * Fits a linear, lasso, and elastic-net regression model using Coordinate Descent solver
 * @param handle
 *        cuml handle
 * @param input
 *        pointer to an array in column-major format (size of n_rows, n_cols)
 * @param n_rows
 *        n_samples or rows in input
 * @param n_cols
 *        n_features or columns in X
 * @param coef
 *        pointer to an array for coefficients (size of n_cols). Calculated in cdFit function.
 * @param intercept
 *        intercept value calculated in cdFit function
 * @param preds
 *        pointer to an array for predictions (size of n_rows). This will be fitted once functions
 * is executed.
 * @param loss
 *        enum to use different loss functions. Only linear regression loss functions is supported
 * right now.
 */
template <typename math_t>
void cdPredict(const raft::handle_t& handle,
               const math_t* input,
               int n_rows,
               int n_cols,
               const math_t* coef,
               math_t intercept,
               math_t* preds,
               ML::loss_funct loss)
{
  ASSERT(n_cols > 0, "Parameter n_cols: number of columns cannot be less than one");
  ASSERT(n_rows > 1, "Parameter n_rows: number of rows cannot be less than two");
  ASSERT(loss == ML::loss_funct::SQRD_LOSS,
         "Parameter loss: Only SQRT_LOSS function is supported for now");

  Functions::linearRegH(handle, input, n_rows, n_cols, coef, preds, intercept, handle.get_stream());
}

};  // namespace Solver
};  // namespace ML
// end namespace ML
