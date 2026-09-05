/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/common/checked_arithmetic.hpp>
#include <cuml/tsa/batched_kalman.hpp>

#include <raft/core/handle.hpp>
#include <raft/linalg/add.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cub/cub.cuh>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <vector>
// #TODO: Replace with public header when ready
#include <raft/core/nvtx.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>

#include <rmm/device_uvector.hpp>

#include <linalg/batched/matrix.cuh>
#include <linalg/block.cuh>
#include <timeSeries/arima_helpers.cuh>

namespace ML {

//! Thread-local Matrix-Vector multiplication.
DI void Mv_l(int n, const double* A, const double* v, double* out)
{
  for (int i = 0; i < n; i++) {
    double sum = 0.0;
    for (int j = 0; j < n; j++) {
      sum += A[i + j * n] * v[j];
    }
    out[i] = sum;
  }
}

DI void Mv_l(int n, double alpha, const double* A, const double* v, double* out)
{
  for (int i = 0; i < n; i++) {
    double sum = 0.0;
    for (int j = 0; j < n; j++) {
      sum += A[i + j * n] * v[j];
    }
    out[i] = alpha * sum;
  }
}

//! Thread-local Matrix-Matrix multiplication.
template <bool aT = false, bool bT = false>
DI void MM_l(int n, const double* A, const double* B, double* out)
{
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.0;
      for (int k = 0; k < n; k++) {
        double Aik = aT ? A[k + i * n] : A[i + k * n];
        double Bkj = bT ? B[j + k * n] : B[k + j * n];
        sum += Aik * Bkj;
      }
      out[i + j * n] = sum;
    }
  }
}

/** Improve stability by making a covariance matrix symmetric and forcing
 * diagonal elements to be positive
 */
DI void numerical_stability(int n, double* A)
{
  // A = 0.5 * (A + A')
  for (int i = 0; i < n - 1; i++) {
    for (int j = i + 1; j < n; j++) {
      double new_val = 0.5 * (A[j * n + i] + A[i * n + j]);
      A[j * n + i]   = new_val;
      A[i * n + j]   = new_val;
    }
  }
  // Aii = abs(Aii)
  for (int i = 0; i < n; i++) {
    A[i * n + i] = abs(A[i * n + i]);
  }
}

/**
 * Kalman loop kernel. Each thread computes kalman filter for a single series
 * and stores relevant matrices in registers.
 *
 * @param[in]  rd              Dimension of the state vector
 * @param[in]  ys              Batched time series
 * @param[in]  nobs            Number of observation per series
 * @param[in]  T               Batched transition matrix.            (r x r)
 * @param[in]  Z               Batched "design" vector               (1 x r)
 * @param[in]  RQR             Batched R*Q*R'                        (r x r)
 * @param[in]  P               Batched P                             (r x r)
 * @param[in]  alpha           Batched state vector                  (r x 1)
 * @param[in]  intercept       Do we fit an intercept?
 * @param[in]  d_mu            Batched intercept                     (1)
 * @param[in]  batch_size      Batch size
 * @param[in]  d_obs_inter     Observation intercept
 * @param[in]  d_obs_inter_fut Observation intercept for forecasts
 * @param[out] d_pred          Predictions                           (nobs)
 * @param[out] d_loglike       Log-likelihood                        (1)
 * @param[in]  n_diff          d + s*D
 * @param[in]  fc_steps        Number of steps to forecast
 * @param[out] d_fc            Array to store the forecast
 * @param[in]  conf_int        Whether to compute confidence intervals
 * @param[out] d_F_fc          Batched variance of forecast errors   (fc_steps)
 */
CUML_KERNEL void batched_kalman_loop_kernel(int rd,
                                            const double* ys,
                                            int nobs,
                                            const double* T,
                                            const double* Z,
                                            const double* RQR,
                                            const double* P,
                                            const double* alpha,
                                            bool intercept,
                                            const double* d_mu,
                                            int batch_size,
                                            const double* d_obs_inter,
                                            const double* d_obs_inter_fut,
                                            double* d_pred,
                                            double* d_loglike,
                                            int n_diff,
                                            int fc_steps   = 0,
                                            double* d_fc   = nullptr,
                                            bool conf_int  = false,
                                            double* d_F_fc = nullptr)
{
  constexpr int rd_max  = 8;
  constexpr int rd2_max = rd_max * rd_max;
  int rd2               = rd * rd;
  double l_RQR[rd2_max];
  double l_T[rd2_max];
  double l_Z[rd_max];
  double l_P[rd2_max];
  double l_alpha[rd_max];
  double l_K[rd_max];
  double l_tmp[rd2_max];
  double l_TP[rd2_max];

  int bid = blockDim.x * blockIdx.x + threadIdx.x;

  if (bid < batch_size) {
    // Load global mem into registers
    {
      int b_rd_offset  = bid * rd;
      int b_rd2_offset = bid * rd2;
      for (int i = 0; i < rd2; i++) {
        l_RQR[i] = RQR[b_rd2_offset + i];
        l_T[i]   = T[b_rd2_offset + i];
        l_P[i]   = P[b_rd2_offset + i];
      }
      for (int i = 0; i < rd; i++) {
        if (n_diff > 0) l_Z[i] = Z[b_rd_offset + i];
        l_alpha[i] = alpha[b_rd_offset + i];
      }
    }

    double b_sum_logFs = 0.0;
    double b_ll_s2     = 0.0;
    int n_obs_ll       = 0;
    const double* b_ys = ys + bid * nobs;
    double* b_pred     = d_pred + bid * nobs;
    double mu          = intercept ? d_mu[bid] : 0.0;

    for (int it = 0; it < nobs; it++) {
      double _Fs, vs_it;
      bool missing;
      {
        // 1. v = y - Z*alpha
        double pred = 0.0;
        if (d_obs_inter != nullptr) { pred += d_obs_inter[bid * nobs + it]; }
        if (n_diff == 0)
          pred += l_alpha[0];
        else {
          for (int i = 0; i < rd; i++) {
            pred += l_alpha[i] * l_Z[i];
          }
        }
        b_pred[it] = pred;
        double yt  = b_ys[it];
        missing    = isnan(yt);

        if (!missing) {
          vs_it = yt - pred;

          // 2. F = Z*P*Z'
          if (n_diff == 0)
            _Fs = l_P[0];
          else {
            _Fs = 0.0;
            for (int i = 0; i < rd; i++) {
              for (int j = 0; j < rd; j++) {
                _Fs += l_P[j * rd + i] * l_Z[i] * l_Z[j];
              }
            }
          }

          if (it >= n_diff) {
            b_sum_logFs += log(_Fs);
            b_ll_s2 += vs_it * vs_it / _Fs;
            n_obs_ll++;
          }
        }
      }

      // 3. K = 1/Fs[it] * T*P*Z'
      // TP = T*P
      MM_l(rd, l_T, l_P, l_TP);
      if (!missing) {
        // K = 1/Fs[it] * TP*Z'
        double _1_Fs = 1.0 / _Fs;
        if (n_diff == 0) {
          for (int i = 0; i < rd; i++) {
            l_K[i] = _1_Fs * l_TP[i];
          }
        } else {
          Mv_l(rd, _1_Fs, l_TP, l_Z, l_K);
        }
      }

      // 4. alpha = T*alpha + K*vs[it] + c
      // tmp = T*alpha
      Mv_l(rd, l_T, l_alpha, l_tmp);
      // alpha = tmp + K*vs[it]
      for (int i = 0; i < rd; i++) {
        l_alpha[i] = l_tmp[i] + (missing ? 0.0 : l_K[i] * vs_it);
      }
      // alpha = alpha + c
      l_alpha[n_diff] += mu;

      // 5. L = T - K * Z
      // L = T (L is tmp)
      for (int i = 0; i < rd2; i++) {
        l_tmp[i] = l_T[i];
      }
      if (!missing) {
        // L = L - K * Z
        if (n_diff == 0) {
          for (int i = 0; i < rd; i++) {
            l_tmp[i] -= l_K[i];
          }
        } else {
          for (int i = 0; i < rd; i++) {
            for (int j = 0; j < rd; j++) {
              l_tmp[j * rd + i] -= l_K[i] * l_Z[j];
            }
          }
        }
      }

      // 6. P = T*P*L' + R*Q*R'
      // P = TP*L'
      MM_l<false, true>(rd, l_TP, l_tmp, l_P);
      // P = P + RQR
      for (int i = 0; i < rd2; i++) {
        l_P[i] += l_RQR[i];
      }

      // Numerical stability: enforce symmetry of P and positivity of diagonal
      numerical_stability(rd, l_P);
    }

    // Compute log-likelihood
    {
      double n_obs_ll_f = static_cast<double>(n_obs_ll);
      b_ll_s2 /= n_obs_ll_f;
      d_loglike[bid] = -.5 * (b_sum_logFs + n_obs_ll_f * (b_ll_s2 + log(2 * M_PI)));
    }

    // Forecast
    {
      double* b_fc   = fc_steps ? d_fc + bid * fc_steps : nullptr;
      double* b_F_fc = conf_int ? d_F_fc + bid * fc_steps : nullptr;
      for (int it = 0; it < fc_steps; it++) {
        double pred = 0.0;
        if (d_obs_inter_fut != nullptr) { pred += d_obs_inter_fut[bid * fc_steps + it]; }
        if (n_diff == 0)
          pred += l_alpha[0];
        else {
          for (int i = 0; i < rd; i++) {
            pred += l_alpha[i] * l_Z[i];
          }
        }
        b_fc[it] = pred;

        // alpha = T*alpha + c
        Mv_l(rd, l_T, l_alpha, l_tmp);
        for (int i = 0; i < rd; i++) {
          l_alpha[i] = l_tmp[i];
        }
        l_alpha[n_diff] += mu;

        if (conf_int) {
          if (n_diff == 0)
            b_F_fc[it] = l_P[0];
          else {
            double _Fs = 0.0;
            for (int i = 0; i < rd; i++) {
              for (int j = 0; j < rd; j++) {
                _Fs += l_P[j * rd + i] * l_Z[i] * l_Z[j];
              }
            }
            b_F_fc[it] = _Fs;
          }

          // P = T*P*T' + RR'
          // TP = T*P
          MM_l(rd, l_T, l_P, l_TP);
          // P = TP*T'
          MM_l<false, true>(rd, l_TP, l_T, l_P);
          // P = P + RR'
          for (int i = 0; i < rd2; i++) {
            l_P[i] += l_RQR[i];
          }

          // Numerical stability: enforce symmetry of P and positivity of diagonal
          numerical_stability(rd, l_P);
        }
      }
    }
  }
}

/**
 * This union allows for efficient reuse of shared memory in the Kalman
 * filter.
 */
template <typename GemmPolicy, typename GemvPolicy, typename CovPolicy, typename T>
union KalmanLoopSharedMemory {
  MLCommon::LinAlg::ReductionStorage<GemmPolicy::BlockSize, T> reduction_storage;
  MLCommon::LinAlg::GemmStorage<GemmPolicy, T> gemm_storage;
  MLCommon::LinAlg::GemvStorage<GemvPolicy, T> gemv_storage[2];
  MLCommon::LinAlg::CovStabilityStorage<CovPolicy, T> cov_stability_storage;
};

/**
 * Kalman loop kernel. Each block computes kalman filter for a single series.
 *
 * @tparam     GemmPolicy      Execution policy for GEMM
 * @tparam     GemvPolicy      Execution policy for GEMV
 * @tparam     CovPolicy       Execution policy for the covariance stability operation
 * @param[in]  d_ys            Batched time series
 * @param[in]  batch_size      Batch size
 * @param[in]  n_obs           Number of observation per series
 * @param[in]  d_T             Batched transition matrix.            (r x r)
 * @param[in]  d_Z             Batched "design" vector               (1 x r)
 * @param[in]  d_RQR           Batched R*Q*R'                        (r x r)
 * @param[in]  d_P             Batched P                             (r x r)
 * @param[in]  d_alpha         Batched state vector                  (r x 1)
 * @param[in]  d_m_tmp         Batched temporary matrix              (r x r)
 * @param[in]  d_TP            Batched temporary matrix to store TP  (r x r)
 * @param[in]  intercept       Do we fit an intercept?
 * @param[in]  d_mu            Batched intercept                     (1)
 * @param[in]  rd              State vector dimension
 * @param[in]  d_obs_inter     Observation intercept
 * @param[in]  d_obs_inter_fut Observation intercept for forecasts
 * @param[out] d_pred          Predictions                           (nobs)
 * @param[out] d_loglike       Log-likelihood                        (1)
 * @param[in]  n_diff          d + s*D
 * @param[in]  fc_steps        Number of steps to forecast
 * @param[out] d_fc            Array to store the forecast
 * @param[in]  conf_int        Whether to compute confidence intervals
 * @param[out] d_F_fc          Batched variance of forecast errors   (fc_steps)
 */
template <typename GemmPolicy, typename GemvPolicy, typename CovPolicy>
CUML_KERNEL void _batched_kalman_device_loop_large_kernel(const double* d_ys,
                                                          int batch_size,
                                                          int n_obs,
                                                          const double* d_T,
                                                          const double* d_Z,
                                                          const double* d_RQR,
                                                          double* d_P,
                                                          double* d_alpha,
                                                          double* d_m_tmp,
                                                          double* d_TP,
                                                          bool intercept,
                                                          const double* d_mu,
                                                          int rd,
                                                          const double* d_obs_inter,
                                                          const double* d_obs_inter_fut,
                                                          double* d_pred,
                                                          double* d_loglike,
                                                          int n_diff,
                                                          int fc_steps,
                                                          double* d_fc,
                                                          bool conf_int,
                                                          double* d_F_fc)
{
  int rd2 = rd * rd;

  // Dynamic shared memory allocation
  extern __shared__ char dyna_shared_mem[];
  double* shared_vec0  = (double*)dyna_shared_mem;
  double* shared_Z     = (double*)(dyna_shared_mem + rd * sizeof(double));
  double* shared_alpha = (double*)(dyna_shared_mem + 2 * rd * sizeof(double));
  double* shared_K     = (double*)(dyna_shared_mem + 3 * rd * sizeof(double));

  __shared__ KalmanLoopSharedMemory<GemmPolicy, GemvPolicy, CovPolicy, double> shared_mem;

  for (int bid = blockIdx.x; bid < batch_size; bid += gridDim.x) {
    /* Load Z and alpha to shared memory */
    for (int i = threadIdx.x; i < rd; i += GemmPolicy::BlockSize) {
      shared_Z[i]     = d_Z[bid * rd + i];
      shared_alpha[i] = d_alpha[bid * rd + i];
    }

    __syncthreads();  // ensure alpha and Z are loaded before 1.

    /* Initialization */
    double mu_       = intercept ? d_mu[bid] : 0.0;
    double sum_logFs = 0.0;
    double ll_s2     = 0.0;
    int n_obs_ll     = 0;
    int it           = 0;

    /* Skip missing observations at the start */
    if (d_obs_inter == nullptr) {
      {
        double pred0;
        if (n_diff == 0) {
          pred0 = shared_alpha[0];
        } else {
          pred0 = 0.0;
          pred0 += MLCommon::LinAlg::_block_dot<GemmPolicy::BlockSize, true>(
            rd, shared_Z, shared_alpha, shared_mem.reduction_storage);
          __syncthreads();  // necessary to reuse shared memory
        }

        for (; it < n_obs && isnan(d_ys[bid * n_obs + it]); it++) {
          if (threadIdx.x == 0) d_pred[bid * n_obs + it] = pred0;
        }
      }
    }

    /* Kalman loop */
    for (; it < n_obs; it++) {
      double vt, _F;
      bool missing;
      {
        // 1. pred = Z*alpha + obs_intercept
        //    v = y - pred
        double pred = 0.0;
        if (d_obs_inter != nullptr) { pred += d_obs_inter[bid * n_obs + it]; }
        if (n_diff == 0) {
          pred += shared_alpha[0];
        } else {
          pred += MLCommon::LinAlg::_block_dot<GemmPolicy::BlockSize, true>(
            rd, shared_Z, shared_alpha, shared_mem.reduction_storage);
          __syncthreads();  // necessary to reuse shared memory
        }
        double yt = d_ys[bid * n_obs + it];
        missing   = isnan(yt);

        if (!missing) {
          vt = yt - pred;

          // 2. F = Z*P*Z'
          if (n_diff == 0) {
            _F = (d_P + bid * rd2)[0];
          } else {
            _F = MLCommon::LinAlg::_block_xAxt<GemmPolicy::BlockSize, true, false>(
              rd, shared_Z, d_P + bid * rd2, shared_mem.reduction_storage);
            __syncthreads();  // necessary to reuse shared memory
          }
        }

        if (threadIdx.x == 0) {
          d_pred[bid * n_obs + it] = pred;

          if (it >= n_diff && !missing) {
            sum_logFs += log(_F);
            ll_s2 += vt * vt / _F;
            n_obs_ll++;
          }
        }
      }

      // 3. K = 1/Fs[it] * T*P*Z'
      // TP = T*P (also used later)
      MLCommon::LinAlg::_block_gemm<GemmPolicy>(false,
                                                false,
                                                rd,
                                                rd,
                                                rd,
                                                1.0,
                                                d_T + bid * rd2,
                                                d_P + bid * rd2,
                                                d_TP + bid * rd2,
                                                shared_mem.gemm_storage);
      __syncthreads();  // for consistency of TP
      if (!missing) {
        // K = 1/Fs[it] * TP*Z'
        double _1_Fs = 1.0 / _F;
        if (n_diff == 0) {
          MLCommon::LinAlg::_block_ax(rd, _1_Fs, d_TP + bid * rd2, shared_K);
        } else {
          MLCommon::LinAlg::_block_gemv<GemvPolicy, false>(
            rd, rd, _1_Fs, d_TP + bid * rd2, shared_Z, shared_K, shared_mem.gemv_storage[0]);
        }
      }

      // 4. alpha = T*alpha + K*vs[it] + c
      // vec1 = T*alpha
      MLCommon::LinAlg::_block_gemv<GemvPolicy, false>(
        rd, rd, 1.0, d_T + bid * rd2, shared_alpha, shared_vec0, shared_mem.gemv_storage[1]);
      __syncthreads();  // For consistency of K and vec1
      // alpha = vec1 + K*vs[it] + c
      for (int i = threadIdx.x; i < rd; i += GemmPolicy::BlockSize) {
        double c_       = (i == n_diff) ? mu_ : 0.0;
        shared_alpha[i] = shared_vec0[i] + c_ + (missing ? 0.0 : vt * shared_K[i]);
      }

      // 5. L = T - K * Z
      if (n_diff == 0) {
        for (int i = threadIdx.x; i < rd2; i += GemmPolicy::BlockSize) {
          double _KZ             = (i < rd && !missing) ? shared_K[i] : 0.0;
          d_m_tmp[bid * rd2 + i] = d_T[bid * rd2 + i] - _KZ;
        }
      } else {
        for (int i = threadIdx.x; i < rd2; i += GemmPolicy::BlockSize) {
          double _KZ             = missing ? 0.0 : shared_K[i % rd] * shared_Z[i / rd];
          d_m_tmp[bid * rd2 + i] = d_T[bid * rd2 + i] - _KZ;
        }
      }

      // 6. P = T*P*L' + R*Q*R'
      __syncthreads();  // For consistency of L
      // P = TP*L'
      MLCommon::LinAlg::_block_gemm<GemmPolicy>(false,
                                                true,
                                                rd,
                                                rd,
                                                rd,
                                                1.0,
                                                d_TP + bid * rd2,
                                                d_m_tmp + bid * rd2,
                                                d_P + bid * rd2,
                                                shared_mem.gemm_storage);
      __syncthreads();  // For consistency of P
      // tmp = P + R*Q*R'
      /// TODO: shared mem R instead of precomputed matrix?
      for (int i = threadIdx.x; i < rd2; i += GemmPolicy::BlockSize) {
        d_m_tmp[bid * rd2 + i] = d_P[bid * rd2 + i] + d_RQR[bid * rd2 + i];
      }
      __syncthreads();

      // Numerical stability: enforce symmetry of P and positivity of diagonal
      // P = 0.5 * (tmp + tmp')
      // Pii = abs(Pii)
      MLCommon::LinAlg::_block_covariance_stability<CovPolicy>(
        rd, d_m_tmp + bid * rd2, d_P + bid * rd2, shared_mem.cov_stability_storage);
      __syncthreads();
    }

    /* Forecast */
    for (int it = 0; it < fc_steps; it++) {
      // pred = Z * alpha + obs_intercept
      double pred = 0.0;
      if (d_obs_inter_fut != nullptr) { pred += d_obs_inter_fut[bid * fc_steps + it]; }
      if (n_diff == 0) {
        pred += shared_alpha[0];
      } else {
        pred += MLCommon::LinAlg::_block_dot<GemmPolicy::BlockSize, false>(
          rd, shared_Z, shared_alpha, shared_mem.reduction_storage);
        __syncthreads();  // necessary to reuse shared memory
      }
      if (threadIdx.x == 0) d_fc[bid * fc_steps + it] = pred;

      // alpha = T*alpha + c
      // vec0 = T*alpha
      MLCommon::LinAlg::_block_gemv<GemvPolicy, false>(
        rd, rd, 1.0, d_T + bid * rd2, shared_alpha, shared_vec0, shared_mem.gemv_storage[0]);
      __syncthreads();  // for consistency of v_tmp + reuse of shared mem
      // alpha = vec0 + c
      for (int i = threadIdx.x; i < rd; i += GemmPolicy::BlockSize) {
        double c_       = (i == n_diff) ? mu_ : 0.0;
        shared_alpha[i] = shared_vec0[i] + c_;
      }

      double _F;
      if (conf_int) {
        if (n_diff == 0) {
          _F = d_P[bid * rd2];
        } else {
          _F = MLCommon::LinAlg::_block_xAxt<GemmPolicy::BlockSize, false, false>(
            rd, shared_Z, d_P + bid * rd2, shared_mem.reduction_storage);
          __syncthreads();  // necessary to reuse shared memory
        }

        if (threadIdx.x == 0) d_F_fc[bid * fc_steps + it] = _F;
      }

      // P = T*P*T' + R*Q*R'
      // TP = T*P
      MLCommon::LinAlg::_block_gemm<GemmPolicy>(false,
                                                false,
                                                rd,
                                                rd,
                                                rd,
                                                1.0,
                                                d_T + bid * rd2,
                                                d_P + bid * rd2,
                                                d_TP + bid * rd2,
                                                shared_mem.gemm_storage);
      __syncthreads();  // for consistency of TP
      // P = TP * T'
      MLCommon::LinAlg::_block_gemm<GemmPolicy>(false,
                                                true,
                                                rd,
                                                rd,
                                                rd,
                                                1.0,
                                                d_TP + bid * rd2,
                                                d_T + bid * rd2,
                                                d_P + bid * rd2,
                                                shared_mem.gemm_storage);
      __syncthreads();  // for consistency of P
      // P = P + R*Q*R'
      /// TODO: shared mem R instead of precomputed matrix?
      for (int i = threadIdx.x; i < rd2; i += GemmPolicy::BlockSize) {
        d_m_tmp[bid * rd2 + i] = d_P[bid * rd2 + i] + d_RQR[bid * rd2 + i];
      }

      __syncthreads();
      // Numerical stability: enforce symmetry of P and positivity of diagonal
      // P = 0.5 * (tmp + tmp')
      // Pii = abs(Pii)
      MLCommon::LinAlg::_block_covariance_stability<CovPolicy>(
        rd, d_m_tmp + bid * rd2, d_P + bid * rd2, shared_mem.cov_stability_storage);
    }

    /* Compute log-likelihood */
    if (threadIdx.x == 0) {
      double n_obs_ll_f = static_cast<double>(n_obs_ll);
      ll_s2 /= n_obs_ll_f;
      d_loglike[bid] = -.5 * (sum_logFs + n_obs_ll_f * (ll_s2 + log(2 * M_PI)));
    }
  }
}

/**
 * Kalman loop for large matrices (r > 8).
 *
 * @param[in]  arima_mem       Pre-allocated temporary memory
 * @param[in]  d_ys            Batched time series
 * @param[in]  nobs            Number of observation per series
 * @param[in]  T               Batched transition matrix.            (r x r)
 * @param[in]  Z               Batched "design" vector               (1 x r)
 * @param[in]  RQR             Batched R*Q*R'                        (r x r)
 * @param[in]  P               Batched P                             (r x r)
 * @param[in]  alpha           Batched state vector                  (r x 1)
 * @param[in]  intercept       Do we fit an intercept?
 * @param[in]  d_mu            Batched intercept                     (1)
 * @param[in]  rd              Dimension of the state vector
 * @param[in]  d_obs_inter     Observation intercept
 * @param[in]  d_obs_inter_fut Observation intercept for forecasts
 * @param[out] d_pred          Predictions                           (nobs)
 * @param[out] d_loglike       Log-likelihood                        (1)
 * @param[in]  n_diff          d + s*D
 * @param[in]  fc_steps        Number of steps to forecast
 * @param[out] d_fc            Array to store the forecast
 * @param[in]  conf_int        Whether to compute confidence intervals
 * @param[out] d_F_fc          Batched variance of forecast errors   (fc_steps)
 */
template <typename GemmPolicy, typename GemvPolicy, typename CovPolicy>
void _batched_kalman_device_loop_large(const ARIMAMemory<double>& arima_mem,
                                       const double* d_ys,
                                       int n_obs,
                                       const MLCommon::LinAlg::Batched::Matrix<double>& T,
                                       const MLCommon::LinAlg::Batched::Matrix<double>& Z,
                                       const MLCommon::LinAlg::Batched::Matrix<double>& RQR,
                                       MLCommon::LinAlg::Batched::Matrix<double>& P,
                                       MLCommon::LinAlg::Batched::Matrix<double>& alpha,
                                       bool intercept,
                                       const double* d_mu,
                                       int rd,
                                       const double* d_obs_inter,
                                       const double* d_obs_inter_fut,
                                       double* d_pred,
                                       double* d_loglike,
                                       int n_diff,
                                       int fc_steps   = 0,
                                       double* d_fc   = nullptr,
                                       bool conf_int  = false,
                                       double* d_F_fc = nullptr)
{
  static_assert(GemmPolicy::BlockSize == GemvPolicy::BlockSize,
                "Gemm and gemv policies: block size mismatch");
  static_assert(GemmPolicy::BlockSize == CovPolicy::BlockSize,
                "Gemm and cov stability policies: block size mismatch");

  auto stream       = T.stream();
  auto cublasHandle = T.cublasHandle();
  int batch_size    = ML::narrow_cast<int>(T.batches());

  // Temporary matrices
  MLCommon::LinAlg::Batched::Matrix<double> m_tmp(rd,
                                                  rd,
                                                  batch_size,
                                                  cublasHandle,
                                                  arima_mem.m_tmp_batches,
                                                  arima_mem.m_tmp_dense,
                                                  stream,
                                                  false);
  MLCommon::LinAlg::Batched::Matrix<double> TP(
    rd, rd, batch_size, cublasHandle, arima_mem.TP_batches, arima_mem.TP_dense, stream, false);

  int grid_size          = std::min(batch_size, 65536);
  size_t shared_mem_size = 4 * rd * sizeof(double);
  _batched_kalman_device_loop_large_kernel<GemmPolicy, GemvPolicy, CovPolicy>
    <<<grid_size, GemmPolicy::BlockSize, shared_mem_size, stream>>>(d_ys,
                                                                    batch_size,
                                                                    n_obs,
                                                                    T.raw_data(),
                                                                    Z.raw_data(),
                                                                    RQR.raw_data(),
                                                                    P.raw_data(),
                                                                    alpha.raw_data(),
                                                                    m_tmp.raw_data(),
                                                                    TP.raw_data(),
                                                                    intercept,
                                                                    d_mu,
                                                                    rd,
                                                                    d_obs_inter,
                                                                    d_obs_inter_fut,
                                                                    d_pred,
                                                                    d_loglike,
                                                                    n_diff,
                                                                    fc_steps,
                                                                    d_fc,
                                                                    conf_int,
                                                                    d_F_fc);
}

/// Wrapper around functions that execute the Kalman loop (for performance)
void batched_kalman_loop(raft::handle_t& handle,
                         const ARIMAMemory<double>& arima_mem,
                         const double* ys,
                         int nobs,
                         const MLCommon::LinAlg::Batched::Matrix<double>& T,
                         const MLCommon::LinAlg::Batched::Matrix<double>& Z,
                         const MLCommon::LinAlg::Batched::Matrix<double>& RQR,
                         MLCommon::LinAlg::Batched::Matrix<double>& P0,
                         MLCommon::LinAlg::Batched::Matrix<double>& alpha,
                         bool intercept,
                         const double* d_mu,
                         const ARIMAOrder& order,
                         const double* d_obs_inter,
                         const double* d_obs_inter_fut,
                         double* d_pred,
                         double* d_loglike,
                         int fc_steps   = 0,
                         double* d_fc   = nullptr,
                         bool conf_int  = false,
                         double* d_F_fc = nullptr)
{
  const int batch_size = ML::narrow_cast<int>(T.batches());
  auto stream          = T.stream();
  int rd               = order.rd();
  int n_diff           = order.n_diff();
  dim3 numThreadsPerBlock(32, 1);
  dim3 numBlocks(raft::ceildiv<int>(batch_size, numThreadsPerBlock.x), 1);
  if (rd <= 8) {
    batched_kalman_loop_kernel<<<numBlocks, numThreadsPerBlock, 0, stream>>>(rd,
                                                                             ys,
                                                                             nobs,
                                                                             T.raw_data(),
                                                                             Z.raw_data(),
                                                                             RQR.raw_data(),
                                                                             P0.raw_data(),
                                                                             alpha.raw_data(),
                                                                             intercept,
                                                                             d_mu,
                                                                             batch_size,
                                                                             d_obs_inter,
                                                                             d_obs_inter_fut,
                                                                             d_pred,
                                                                             d_loglike,
                                                                             n_diff,
                                                                             fc_steps,
                                                                             d_fc,
                                                                             conf_int,
                                                                             d_F_fc);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  } else {
    using GemmPolicy = MLCommon::LinAlg::BlockGemmPolicy<1, 32, 1, 4, 32, 8>;
    using GemvPolicy = MLCommon::LinAlg::BlockGemvPolicy<32, 8>;
    using CovPolicy  = MLCommon::LinAlg::BlockPolicy<1, 4, 32, 8>;
    _batched_kalman_device_loop_large<GemmPolicy, GemvPolicy, CovPolicy>(arima_mem,
                                                                         ys,
                                                                         nobs,
                                                                         T,
                                                                         Z,
                                                                         RQR,
                                                                         P0,
                                                                         alpha,
                                                                         intercept,
                                                                         d_mu,
                                                                         rd,
                                                                         d_obs_inter,
                                                                         d_obs_inter_fut,
                                                                         d_pred,
                                                                         d_loglike,
                                                                         n_diff,
                                                                         fc_steps,
                                                                         d_fc,
                                                                         conf_int,
                                                                         d_F_fc);
  }
}

/**
 * Kernel to finalize the computation of confidence intervals
 *
 * @note: One block per batch member, one thread per forecast time step
 *
 * @param[in]    d_fc       Mean forecasts
 * @param[inout] d_lower    Input: F_{n+t}
 *                          Output: lower bound of the confidence intervals
 * @param[out]   d_upper    Upper bound of the confidence intervals
 * @param[in]    n_elem     Total number of elements (fc_steps * batch_size)
 * @param[in]    multiplier Coefficient associated with the confidence level
 */
CUML_KERNEL void confidence_intervals(
  const double* d_fc, double* d_lower, double* d_upper, int n_elem, double multiplier)
{
  for (int idx = threadIdx.x; idx < n_elem; idx += blockDim.x * gridDim.x) {
    double fc     = d_fc[idx];
    double margin = multiplier * sqrt(d_lower[idx]);
    d_lower[idx]  = fc - margin;
    d_upper[idx]  = fc + margin;
  }
}

void _lyapunov_wrapper(raft::handle_t& handle,
                       const ARIMAMemory<double>& arima_mem,
                       const MLCommon::LinAlg::Batched::Matrix<double>& A,
                       MLCommon::LinAlg::Batched::Matrix<double>& Q,
                       MLCommon::LinAlg::Batched::Matrix<double>& X,
                       int r)
{
  if (r <= 5) {
    auto stream       = handle.get_stream().get();
    auto cublasHandle = handle.get_cublas_handle();
    int batch_size    = ML::narrow_cast<int>(A.batches());
    int r2            = r * r;

    //
    // Use direct solution with Kronecker product
    //

    MLCommon::LinAlg::Batched::Matrix<double> I_m_AxA(r2,
                                                      r2,
                                                      batch_size,
                                                      cublasHandle,
                                                      arima_mem.I_m_AxA_batches,
                                                      arima_mem.I_m_AxA_dense,
                                                      stream,
                                                      false);
    MLCommon::LinAlg::Batched::Matrix<double> I_m_AxA_inv(r2,
                                                          r2,
                                                          batch_size,
                                                          cublasHandle,
                                                          arima_mem.I_m_AxA_inv_batches,
                                                          arima_mem.I_m_AxA_inv_dense,
                                                          stream,
                                                          false);

    MLCommon::LinAlg::Batched::_direct_lyapunov_helper(
      A, Q, X, I_m_AxA, I_m_AxA_inv, arima_mem.I_m_AxA_P, arima_mem.I_m_AxA_info, r);
  } else {
    // Note: the other Lyapunov solver is doing temporary mem allocations,
    // but when r > 5, allocation overhead shouldn't be a bottleneck
    X = MLCommon::LinAlg::Batched::b_lyapunov(A, Q);
  }
}

/// Internal Kalman filter implementation that assumes data exists on GPU.
void _batched_kalman_filter(raft::handle_t& handle,
                            const ARIMAMemory<double>& arima_mem,
                            const double* d_ys,
                            const double* d_exog,
                            int nobs,
                            const ARIMAOrder& order,
                            const MLCommon::LinAlg::Batched::Matrix<double>& Zb,
                            const MLCommon::LinAlg::Batched::Matrix<double>& Tb,
                            const MLCommon::LinAlg::Batched::Matrix<double>& Rb,
                            double* d_pred,
                            double* d_loglike,
                            const double* d_sigma2,
                            bool intercept,
                            const double* d_mu,
                            const double* d_beta,
                            int fc_steps,
                            double* d_fc,
                            const double* d_exog_fut,
                            double level,
                            double* d_lower,
                            double* d_upper)
{
  const size_t batch_size = Zb.batches();
  auto stream             = handle.get_stream().get();
  auto cublasHandle       = handle.get_cublas_handle();

  auto counting = thrust::make_counting_iterator(0);

  int n_diff = order.n_diff();
  int rd     = order.rd();
  int r      = order.r();

  // Compute observation intercept (exogenous component).
  // The observation intercept is a linear combination of the values of
  // exogenous variables for this observation.
  rmm::device_uvector<double> obs_intercept(0, stream);
  rmm::device_uvector<double> obs_intercept_fut(0, stream);
  if (order.n_exog > 0) {
    obs_intercept.resize(nobs * batch_size, stream);

    double alpha = 1.0;
    double beta  = 0.0;
    // #TODO: Call from public API when ready
    RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemmStridedBatched(cublasHandle,
                                                                   CUBLAS_OP_N,
                                                                   CUBLAS_OP_N,
                                                                   nobs,
                                                                   1,
                                                                   order.n_exog,
                                                                   &alpha,
                                                                   d_exog,
                                                                   nobs,
                                                                   nobs * order.n_exog,
                                                                   d_beta,
                                                                   order.n_exog,
                                                                   order.n_exog,
                                                                   &beta,
                                                                   obs_intercept.data(),
                                                                   nobs,
                                                                   nobs,
                                                                   batch_size,
                                                                   stream));

    if (fc_steps > 0) {
      obs_intercept_fut.resize(fc_steps * batch_size, stream);

      // #TODO: Call from public API when ready
      RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemmStridedBatched(cublasHandle,
                                                                     CUBLAS_OP_N,
                                                                     CUBLAS_OP_N,
                                                                     fc_steps,
                                                                     1,
                                                                     order.n_exog,
                                                                     &alpha,
                                                                     d_exog_fut,
                                                                     fc_steps,
                                                                     fc_steps * order.n_exog,
                                                                     d_beta,
                                                                     order.n_exog,
                                                                     order.n_exog,
                                                                     &beta,
                                                                     obs_intercept_fut.data(),
                                                                     fc_steps,
                                                                     fc_steps,
                                                                     batch_size,
                                                                     stream));
    }
  }

  MLCommon::LinAlg::Batched::Matrix<double> RQb(
    rd, 1, batch_size, cublasHandle, arima_mem.RQ_batches, arima_mem.RQ_dense, stream, true);
  double* d_RQ      = RQb.raw_data();
  const double* d_R = Rb.raw_data();
  thrust::for_each(
    thrust::cuda::par.on(stream), counting, counting + batch_size, [=] __device__(int bid) {
      double sigma2 = d_sigma2[bid];
      for (int i = 0; i < rd; i++) {
        d_RQ[bid * rd + i] = d_R[bid * rd + i] * sigma2;
      }
    });
  MLCommon::LinAlg::Batched::Matrix<double> RQR(
    rd, rd, batch_size, cublasHandle, arima_mem.RQR_batches, arima_mem.RQR_dense, stream, false);
  MLCommon::LinAlg::Batched::b_gemm(false, true, rd, rd, 1, 1.0, RQb, Rb, 0.0, RQR);

  // Durbin Koopman "Time Series Analysis" pg 138
  raft::common::nvtx::push_range("Init P");
  MLCommon::LinAlg::Batched::Matrix<double> P(
    rd, rd, batch_size, cublasHandle, arima_mem.P_batches, arima_mem.P_dense, stream, true);
  {
    double* d_P = P.raw_data();

    if (n_diff > 0) {
      // Initialize the diffuse part with a large variance
      /// TODO: pass this as a parameter
      constexpr double kappa = 1e6;
      thrust::for_each(
        thrust::cuda::par.on(stream), counting, counting + batch_size, [=] __device__(int bid) {
          double* b_P = d_P + rd * rd * bid;
          for (int i = 0; i < n_diff; i++) {
            b_P[(rd + 1) * i] = kappa;
          }
        });

      // Initialize the stationary part by solving a Lyapunov equation
      MLCommon::LinAlg::Batched::Matrix<double> Ts(
        r, r, batch_size, cublasHandle, arima_mem.Ts_batches, arima_mem.Ts_dense, stream, false);
      MLCommon::LinAlg::Batched::Matrix<double> RQRs(r,
                                                     r,
                                                     batch_size,
                                                     cublasHandle,
                                                     arima_mem.RQRs_batches,
                                                     arima_mem.RQRs_dense,
                                                     stream,
                                                     false);
      MLCommon::LinAlg::Batched::Matrix<double> Ps(
        r, r, batch_size, cublasHandle, arima_mem.Ps_batches, arima_mem.Ps_dense, stream, false);

      MLCommon::LinAlg::Batched::b_2dcopy(Tb, Ts, n_diff, n_diff, r, r);
      MLCommon::LinAlg::Batched::b_2dcopy(RQR, RQRs, n_diff, n_diff, r, r);
      // Ps = MLCommon::LinAlg::Batched::b_lyapunov(Ts, RQRs);
      _lyapunov_wrapper(handle, arima_mem, Ts, RQRs, Ps, r);
      MLCommon::LinAlg::Batched::b_2dcopy(Ps, P, 0, 0, r, r, n_diff, n_diff);
    } else {
      // Initialize by solving a Lyapunov equation
      // P = MLCommon::LinAlg::Batched::b_lyapunov(Tb, RQR);
      _lyapunov_wrapper(handle, arima_mem, Tb, RQR, P, rd);
    }
  }
  raft::common::nvtx::pop_range();

  // Initialize the state alpha by solving (I - T*) x* = c with:
  //     | mu |
  // c = | 0  |
  //     | .  |
  //     | 0  |
  // T* = T[d+s*D:, d+s*D:]
  // x* = alpha_0[d+s*D:]
  MLCommon::LinAlg::Batched::Matrix<double> alpha(rd,
                                                  1,
                                                  batch_size,
                                                  handle.get_cublas_handle(),
                                                  arima_mem.alpha_batches,
                                                  arima_mem.alpha_dense,
                                                  stream,
                                                  false);
  if (intercept) {
    // Compute I-T*
    MLCommon::LinAlg::Batched::Matrix<double> ImT(
      r, r, batch_size, cublasHandle, arima_mem.ImT_batches, arima_mem.ImT_dense, stream, false);
    const double* d_T = Tb.raw_data();
    double* d_ImT     = ImT.raw_data();
    thrust::for_each(
      thrust::cuda::par.on(stream), counting, counting + batch_size, [=] __device__(int bid) {
        const double* b_T = d_T + rd * rd * bid;
        double* b_ImT     = d_ImT + r * r * bid;
        for (int i = 0; i < r; i++) {
          for (int j = 0; j < r; j++) {
            b_ImT[r * j + i] = (i == j ? 1.0 : 0.0) - b_T[rd * (j + n_diff) + i + n_diff];
          }
        }
      });

    // For r=1, prevent I-T from being too close to [[0]] -> no solution
    if (r == 1) {
      thrust::for_each(
        thrust::cuda::par.on(stream), counting, counting + batch_size, [=] __device__(int bid) {
          if (abs(d_ImT[bid]) < 1e-3) d_ImT[bid] = raft::signPrim(d_ImT[bid]) * 1e-3;
        });
    }

    // Compute (I-T*)^-1
    MLCommon::LinAlg::Batched::Matrix<double> ImT_inv(r,
                                                      r,
                                                      batch_size,
                                                      cublasHandle,
                                                      arima_mem.ImT_inv_batches,
                                                      arima_mem.ImT_inv_dense,
                                                      stream,
                                                      false);
    MLCommon::LinAlg::Batched::Matrix<double>::inv(
      ImT, ImT_inv, arima_mem.ImT_inv_P, arima_mem.ImT_inv_info);

    // Compute (I-T*)^-1 * c -> multiply 1st column by mu
    const double* d_ImT_inv = ImT_inv.raw_data();
    double* d_alpha         = alpha.raw_data();
    thrust::for_each(
      thrust::cuda::par.on(stream), counting, counting + batch_size, [=] __device__(int bid) {
        const double* b_ImT_inv = d_ImT_inv + r * r * bid;
        double* b_alpha         = d_alpha + rd * bid;
        double mu               = d_mu[bid];
        for (int i = 0; i < n_diff; i++) {
          b_alpha[i] = 0;
        }
        for (int i = 0; i < r; i++) {
          b_alpha[i + n_diff] = b_ImT_inv[i] * mu;
        }
      });
  } else {
    // Memset alpha to 0
    RAFT_CUDA_TRY(cudaMemsetAsync(alpha.raw_data(), 0, sizeof(double) * rd * batch_size, stream));
  }

  batched_kalman_loop(handle,
                      arima_mem,
                      d_ys,
                      nobs,
                      Tb,
                      Zb,
                      RQR,
                      P,
                      alpha,
                      intercept,
                      d_mu,
                      order,
                      obs_intercept.data(),
                      obs_intercept_fut.data(),
                      d_pred,
                      d_loglike,
                      fc_steps,
                      d_fc,
                      level > 0,
                      d_lower);

  if (level > 0) {
    constexpr int TPB_conf = 256;
    int n_blocks           = raft::ceildiv<int>(fc_steps * batch_size, TPB_conf);
    confidence_intervals<<<n_blocks, TPB_conf, 0, stream>>>(
      d_fc, d_lower, d_upper, fc_steps * batch_size, sqrt(2.0) * erfinv(level));
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
}

void init_batched_kalman_matrices(raft::handle_t& handle,
                                  const double* d_ar,
                                  const double* d_ma,
                                  const double* d_sar,
                                  const double* d_sma,
                                  int nb,
                                  const ARIMAOrder& order,
                                  int rd,
                                  double* d_Z_b,
                                  double* d_R_b,
                                  double* d_T_b)
{
  raft::common::nvtx::range fun_scope(__func__);

  auto stream = handle.get_stream().get();

  // Note: Z is unused yet but kept to avoid reintroducing it later when
  // adding support for exogeneous variables
  cudaMemsetAsync(d_Z_b, 0.0, rd * nb * sizeof(double), stream);
  cudaMemsetAsync(d_R_b, 0.0, rd * nb * sizeof(double), stream);
  cudaMemsetAsync(d_T_b, 0.0, rd * rd * nb * sizeof(double), stream);

  int n_diff = order.n_diff();
  int r      = order.r();

  auto counting = thrust::make_counting_iterator(0);
  auto n_theta  = order.n_theta();
  auto n_phi    = order.n_phi();
  thrust::for_each(thrust::cuda::par.on(stream), counting, counting + nb, [=] __device__(int bid) {
    // See TSA pg. 54 for Z, R, T matrices

    // Z = [ 1 | 0 . . 0 1 0 . . 0 1 | 1 0 . . 0 ]
    //       d |         s*D         |     r
    for (int i = 0; i < order.d; i++)
      d_Z_b[bid * rd + i] = 1.0;
    for (int i = 1; i <= order.D; i++)
      d_Z_b[bid * rd + order.d + i * order.s - 1] = 1.0;
    d_Z_b[bid * rd + n_diff] = 1.0;

    //     |     0     |
    //     |     .     |  d + s*D
    //     |     0     |_ _
    // R = |     1     |
    //     |  theta_1  |  r
    //     |     .     |
    //     |theta_{r-1}|
    //
    d_R_b[bid * rd + n_diff] = 1.0;
    for (int i = 0; i < n_theta; i++) {
      d_R_b[bid * rd + n_diff + i + 1] = MLCommon::TimeSeries::reduced_polynomial<false>(
        bid, d_ma, order.q, d_sma, order.Q, order.s, i + 1);
    }

    //     | 1 | 0 .. 0 1 | 1                |  d
    //     |_ _|_ _ _ _ _ |_ _ _ _ _ _ _ _ _ |_ _
    //     |   | 0 .. 0 1 | 1                |
    //     |   | 1      0 |                  |
    //     |   |   .    . |                  |  s*D
    //     |   |    .   . |                  |
    // T = |   | 0    1 0 |                  |
    //     |_ _|_ _ _ _ _ |_ _ _ _ _ _ _ _ _ |_ _
    //     |   |          | phi_1  1         |
    //     |   |          |  .       1    0  |
    //     |   |          |  .         .     |  r
    //     |   |          |  .     0     .   |
    //     |   |          |  .             1 |
    //     |   |          | phi_r  0  . .  0 |
    //
    // (non-comprehensive example with d=1 and D=1)
    //
    double* batch_T = d_T_b + bid * rd * rd;
    // 1. Differencing component
    for (int i = 0; i < order.d; i++) {
      for (int j = i; j < order.d; j++) {
        batch_T[j * rd + i] = 1.0;
      }
    }
    for (int id = 0; id < order.d; id++) {
      batch_T[n_diff * rd + id] = 1.0;
      for (int iD = 1; iD <= order.D; iD++) {
        batch_T[(order.d + order.s * iD - 1) * rd + id] = 1.0;
      }
    }
    // 2. Seasonal differencing component
    for (int iD = 0; iD < order.D; iD++) {
      int offset = order.d + iD * order.s;
      for (int i = 0; i < order.s - 1; i++) {
        batch_T[(offset + i) * rd + offset + i + 1] = 1.0;
      }
      batch_T[(offset + order.s - 1) * rd + offset] = 1.0;
      batch_T[n_diff * rd + offset]                 = 1.0;
    }
    if (order.D == 2) { batch_T[(n_diff - 1) * rd + order.d] = 1.0; }
    // 3. Auto-Regressive component
    for (int i = 0; i < n_phi; i++) {
      batch_T[n_diff * (rd + 1) + i] = MLCommon::TimeSeries::reduced_polynomial<true>(
        bid, d_ar, order.p, d_sar, order.P, order.s, i + 1);
    }
    for (int i = 0; i < r - 1; i++) {
      batch_T[(n_diff + i + 1) * rd + n_diff + i] = 1.0;
    }

    // If rd=2 and phi_2=-1, I-TxT is singular
    if (rd == 2 && order.p == 2 && abs(batch_T[1] + 1) < 0.01) { batch_T[1] = -0.99; }
  });
}

void batched_kalman_filter(raft::handle_t& handle,
                           const ARIMAMemory<double>& arima_mem,
                           const double* d_ys,
                           const double* d_exog,
                           int nobs,
                           const ARIMAParams<double>& params,
                           const ARIMAOrder& order,
                           int batch_size,
                           double* d_loglike,
                           double* d_pred,
                           int fc_steps,
                           double* d_fc,
                           const double* d_exog_fut,
                           double level,
                           double* d_lower,
                           double* d_upper)
{
  raft::common::nvtx::range fun_scope(__func__);

  auto cublasHandle = handle.get_cublas_handle();
  auto stream       = handle.get_stream().get();

  // see (3.18) in TSA by D&K
  int rd = order.rd();

  MLCommon::LinAlg::Batched::Matrix<double> Zb(
    1, rd, batch_size, cublasHandle, arima_mem.Z_batches, arima_mem.Z_dense, stream, false);
  MLCommon::LinAlg::Batched::Matrix<double> Tb(
    rd, rd, batch_size, cublasHandle, arima_mem.T_batches, arima_mem.T_dense, stream, false);
  MLCommon::LinAlg::Batched::Matrix<double> Rb(
    rd, 1, batch_size, cublasHandle, arima_mem.R_batches, arima_mem.R_dense, stream, false);

  init_batched_kalman_matrices(handle,
                               params.ar,
                               params.ma,
                               params.sar,
                               params.sma,
                               batch_size,
                               order,
                               rd,
                               Zb.raw_data(),
                               Rb.raw_data(),
                               Tb.raw_data());

  ////////////////////////////////////////////////////////////
  // Computation

  _batched_kalman_filter(handle,
                         arima_mem,
                         d_ys,
                         d_exog,
                         nobs,
                         order,
                         Zb,
                         Tb,
                         Rb,
                         d_pred,
                         d_loglike,
                         params.sigma2,
                         static_cast<bool>(order.k),
                         params.mu,
                         params.beta,
                         fc_steps,
                         d_fc,
                         d_exog_fut,
                         level,
                         d_lower,
                         d_upper);
}

void batched_jones_transform(raft::handle_t& handle,
                             const ARIMAMemory<double>& arima_mem,
                             const ARIMAOrder& order,
                             int batch_size,
                             bool isInv,
                             const double* h_params,
                             double* h_Tparams)
{
  int N                       = order.complexity();
  auto stream                 = handle.get_stream().get();
  double* d_params            = arima_mem.d_params;
  double* d_Tparams           = arima_mem.d_Tparams;
  ARIMAParams<double> params  = {arima_mem.params_mu,
                                 arima_mem.params_beta,
                                 arima_mem.params_ar,
                                 arima_mem.params_ma,
                                 arima_mem.params_sar,
                                 arima_mem.params_sma,
                                 arima_mem.params_sigma2};
  ARIMAParams<double> Tparams = {params.mu,
                                 params.beta,
                                 arima_mem.Tparams_ar,
                                 arima_mem.Tparams_ma,
                                 arima_mem.Tparams_sar,
                                 arima_mem.Tparams_sma,
                                 arima_mem.Tparams_sigma2};

  raft::update_device(d_params, h_params, N * batch_size, stream);

  params.unpack(order, batch_size, d_params, stream);

  MLCommon::TimeSeries::batched_jones_transform(order, batch_size, isInv, params, Tparams, stream);

  Tparams.pack(order, batch_size, d_Tparams, stream);

  raft::update_host(h_Tparams, d_Tparams, N * batch_size, stream);
}

}  // namespace ML
