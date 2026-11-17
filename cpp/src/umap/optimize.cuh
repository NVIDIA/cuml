/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/common/logger.hpp>
#include <cuml/common/utils.hpp>
#include <cuml/manifold/umapparams.h>

#include <raft/linalg/add.cuh>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/multiply.cuh>
#include <raft/linalg/power.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/stats/mean.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>

namespace UMAPAlgo {

namespace Optimize {

using namespace ML;

template <typename T, int TPB_X, typename Lambda>
CUML_KERNEL void map_kernel(T* output, T* X, int n_rows, T* coef, Lambda grad)
{
  int row = (blockIdx.x * TPB_X) + threadIdx.x;
  if (row < n_rows) {
    T x         = X[row];
    T a         = coef[0];
    T b         = coef[1];
    output[row] = grad(x, a, b);
    if (isnan(output[row])) output[row] = 0.0;
  }
}

/**
 * This works on a one-dimensional set of
 * x-values.
 */
template <typename T, int TPB_X>
void f(T* input, int n_rows, T* coef, T* preds)
{
  dim3 grid(raft::ceildiv(n_rows, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  // Function: 1/1+ax^(2b)
  map_kernel<T, TPB_X><<<grid, blk>>>(preds, input, n_rows, coef, [] __device__(T x, T a, T b) {
    return 1.0 / (1 + a * pow(x, 2.0 * b));
  });
}

/**
 * Calculate the gradients for fitting parameters a and b
 * to a smooth function based on exponential decay
 */
template <typename T, int TPB_X>
void abLossGrads(
  T* input, int n_rows, const T* labels, T* coef, T* grads, UMAPParams* params, cudaStream_t stream)
{
  dim3 grid(raft::ceildiv(n_rows, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  /**
   * Calculate residuals
   */
  rmm::device_uvector<T> residuals(n_rows, stream);

  f<T, TPB_X>(input, n_rows, coef, residuals.data());
  raft::linalg::eltwiseSub(residuals.data(), residuals.data(), labels, n_rows, stream);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  /**
   * Gradient w/ respect to a
   */
  rmm::device_uvector<T> a_deriv(n_rows, stream);
  raft::copy(a_deriv.data(), input, n_rows, stream);
  map_kernel<T, TPB_X><<<grid, blk, 0, stream>>>(
    a_deriv.data(), a_deriv.data(), n_rows, coef, [] __device__ __host__(T x, T a, T b) {
      return -(pow(x, 2.0 * b)) / pow((1.0 + a * pow(x, 2.0 * b)), 2.0);
    });

  raft::linalg::eltwiseMultiply(a_deriv.data(), a_deriv.data(), residuals.data(), n_rows, stream);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  /**
   * Gradient w/ respect to b
   */
  rmm::device_uvector<T> b_deriv(n_rows, stream);
  raft::copy(b_deriv.data(), input, n_rows, stream);
  map_kernel<T, TPB_X><<<grid, blk, 0, stream>>>(
    b_deriv.data(), b_deriv.data(), n_rows, coef, [] __device__ __host__(T x, T a, T b) {
      return -(2.0 * a * pow(x, 2.0 * b) * log(x)) / pow(1 + a * pow(x, 2.0 * b), 2.0);
    });

  /**
   * Multiply partial derivs by residuals
   */
  raft::linalg::eltwiseMultiply(b_deriv.data(), b_deriv.data(), residuals.data(), n_rows, stream);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  /**
   * Finally, take the mean
   */
  raft::stats::mean<false>(grads, a_deriv.data(), 1, n_rows, false, stream);
  raft::stats::mean<false>(grads + 1, b_deriv.data(), 1, n_rows, false, stream);

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * Perform non-linear gradient descent
 */
template <typename T, int TPB_X>
void optimize_params(T* input,
                     int n_rows,
                     const T* labels,
                     T* coef,
                     UMAPParams* params,
                     cudaStream_t stream,
                     float tolerance = 1e-6,
                     int max_epochs  = 25000)
{
  // Don't really need a learning rate since
  // we aren't using stochastic GD
  float learning_rate = 1.0;

  int num_iters = 0;
  int tol_grads = 0;
  do {
    tol_grads = 0;
    rmm::device_uvector<T> grads(2, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(grads.data(), 0, 2 * sizeof(T), stream));

    abLossGrads<T, TPB_X>(input, n_rows, labels, coef, grads.data(), params, stream);

    raft::linalg::multiplyScalar(grads.data(), grads.data(), learning_rate, 2, stream);
    raft::linalg::eltwiseSub(coef, coef, grads.data(), 2, stream);

    T* grads_h = (T*)malloc(2 * sizeof(T));
    raft::update_host(grads_h, grads.data(), 2, stream);

    raft::interruptible::synchronize(stream);

    for (int i = 0; i < 2; i++) {
      if (abs(grads_h[i]) - tolerance <= 0) tol_grads += 1;
    }

    num_iters += 1;

    free(grads_h);

  } while (tol_grads < 2 && num_iters < max_epochs);
}

inline void find_params_ab(UMAPParams* params, cudaStream_t stream)
{
  float spread   = params->spread;
  float min_dist = params->min_dist;

  float step = (spread * 3.0) / 300.0;

  auto const X_uptr = std::make_unique<float[]>(300);
  auto const y_uptr = std::make_unique<float[]>(300);
  auto* const X     = X_uptr.get();
  auto* const y     = y_uptr.get();

  for (int i = 0; i < 300; i++) {
    X[i] = i * step;
    y[i] = 0.0;
    if (X[i] >= min_dist)
      y[i] = exp(-(X[i] - min_dist) / spread);
    else if (X[i] < min_dist)
      y[i] = 1.0;
  }

  rmm::device_uvector<float> X_d(300, stream);
  raft::update_device(X_d.data(), X, 300, stream);

  rmm::device_uvector<float> y_d(300, stream);
  raft::update_device(y_d.data(), y, 300, stream);
  auto const coeffs_h_uptr = std::make_unique<float[]>(2);
  auto* const coeffs_h     = coeffs_h_uptr.get();
  coeffs_h[0]              = 1.0;
  coeffs_h[1]              = 1.0;

  rmm::device_uvector<float> coeffs(2, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(coeffs.data(), 0, 2 * sizeof(float), stream));

  raft::update_device(coeffs.data(), coeffs_h, 2, stream);

  optimize_params<float, 256>(X_d.data(), 300, y_d.data(), coeffs.data(), params, stream);

  raft::update_host(&(params->a), coeffs.data(), 1, stream);
  raft::update_host(&(params->b), coeffs.data() + 1, 1, stream);

  raft::interruptible::synchronize(stream);

  CUML_LOG_DEBUG("a=%f, b=%f", params->a, params->b);
}
}  // namespace Optimize
}  // namespace UMAPAlgo
