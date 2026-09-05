/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "hw_utils.cuh"

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

template <typename Dtype>
__device__ Dtype holtwinters_eval_device(int tid,
                                         const Dtype* ts,
                                         int n,
                                         int batch_size,
                                         int frequency,
                                         int shift,
                                         Dtype plevel,
                                         Dtype ptrend,
                                         Dtype* pseason,
                                         int pseason_width,
                                         const Dtype* start_season,
                                         const Dtype* beta,
                                         const Dtype* gamma,
                                         Dtype alpha_,
                                         Dtype beta_,
                                         Dtype gamma_,
                                         Dtype* level,
                                         Dtype* trend,
                                         Dtype* season,
                                         Dtype* xhat,
                                         bool additive_seasonal)
{
  alpha_ = bound_device(alpha_);
  beta_  = bound_device(beta_);
  gamma_ = bound_device(gamma_);

  Dtype error_ = .0;
  Dtype clevel = .0, ctrend = .0, cseason = .0;
  for (int i = 0; i < n - shift; i++) {
    int s            = i % frequency;
    Dtype pts        = ts[IDX(tid, i + shift, batch_size)];
    Dtype leveltrend = plevel + ptrend;

    // xhat
    Dtype stmp;
    if (gamma)
      stmp = i < frequency ? start_season[IDX(tid, i, batch_size)] : pseason[s * pseason_width];
    else
      stmp = (!additive_seasonal);
    Dtype xhat_ = plevel + ptrend;
    if (additive_seasonal)
      xhat_ += stmp;
    else
      xhat_ *= stmp;

    // Error
    Dtype diff = pts - xhat_;
    error_ += diff * diff;

    // Level
    if (additive_seasonal) {
      clevel = alpha_ * (pts - stmp) + (1 - alpha_) * (leveltrend);
    } else {
      Dtype stmp_eps = abs(stmp) > STMP_EPS ? stmp : STMP_EPS;
      clevel         = alpha_ * (pts / stmp_eps) + (1 - alpha_) * (leveltrend);
    }

    // Trend
    if (beta) {
      ctrend = beta_ * (clevel - plevel) + (1 - beta_) * ptrend;
      ptrend = ctrend;
    }

    // Seasonal
    if (gamma) {
      if (additive_seasonal)
        cseason = gamma_ * (pts - clevel) + (1 - gamma_) * stmp;
      else
        cseason = gamma_ * (pts / clevel) + (1 - gamma_) * stmp;
      pseason[s * pseason_width] = cseason;
    }

    plevel = clevel;

    if (level) level[IDX(tid, i, batch_size)] = clevel;
    if (trend) trend[IDX(tid, i, batch_size)] = ctrend;
    if (season) season[IDX(tid, i, batch_size)] = cseason;
    if (xhat) xhat[IDX(tid, i, batch_size)] = xhat_;
  }
  return error_;
}

template <typename Dtype>
CUML_KERNEL void holtwinters_eval_gpu_shared_kernel(const Dtype* ts,
                                                    int n,
                                                    int batch_size,
                                                    int frequency,
                                                    const Dtype* start_level,
                                                    const Dtype* start_trend,
                                                    const Dtype* start_season,
                                                    const Dtype* alpha,
                                                    const Dtype* beta,
                                                    const Dtype* gamma,
                                                    Dtype* level,
                                                    Dtype* trend,
                                                    Dtype* season,
                                                    Dtype* xhat,
                                                    Dtype* error,
                                                    bool additive_seasonal)
{
  int tid = GET_TID;
  extern __shared__ unsigned char pseason_[];
  Dtype* pseason = reinterpret_cast<Dtype*>(pseason_);

  if (tid < batch_size) {
    int shift    = 1;
    Dtype plevel = start_level[tid], ptrend = .0;
    Dtype alpha_ = alpha[tid];
    Dtype beta_  = beta ? beta[tid] : .0;
    Dtype gamma_ = gamma ? gamma[tid] : .0;

    if (gamma) {
      shift  = frequency;
      ptrend = beta ? start_trend[tid] : .0;
    } else if (beta) {
      shift  = 2;
      ptrend = start_trend[tid];
    }

    Dtype error_ = holtwinters_eval_device<Dtype>(tid,
                                                  ts,
                                                  n,
                                                  batch_size,
                                                  frequency,
                                                  shift,
                                                  plevel,
                                                  ptrend,
                                                  pseason + threadIdx.x,
                                                  blockDim.x,
                                                  start_season,
                                                  beta,
                                                  gamma,
                                                  alpha_,
                                                  beta_,
                                                  gamma_,
                                                  level,
                                                  trend,
                                                  season,
                                                  xhat,
                                                  additive_seasonal);
    if (error) error[tid] = error_;
  }
}

template <typename Dtype>
CUML_KERNEL void holtwinters_eval_gpu_global_kernel(const Dtype* ts,
                                                    int n,
                                                    int batch_size,
                                                    int frequency,
                                                    const Dtype* start_level,
                                                    const Dtype* start_trend,
                                                    const Dtype* start_season,
                                                    Dtype* pseason,
                                                    const Dtype* alpha,
                                                    const Dtype* beta,
                                                    const Dtype* gamma,
                                                    Dtype* level,
                                                    Dtype* trend,
                                                    Dtype* season,
                                                    Dtype* xhat,
                                                    Dtype* error,
                                                    bool additive_seasonal)
{
  int tid = GET_TID;

  if (tid < batch_size) {
    int shift    = 1;
    Dtype plevel = start_level[tid], ptrend = .0;
    Dtype alpha_ = alpha[tid];
    Dtype beta_  = beta ? beta[tid] : .0;
    Dtype gamma_ = gamma ? gamma[tid] : .0;

    if (gamma) {
      shift  = frequency;
      ptrend = beta ? start_trend[tid] : .0;
    } else if (beta) {
      shift  = 2;
      ptrend = start_trend[tid];
    }

    Dtype error_ = holtwinters_eval_device<Dtype>(tid,
                                                  ts,
                                                  n,
                                                  batch_size,
                                                  frequency,
                                                  shift,
                                                  plevel,
                                                  ptrend,
                                                  pseason + tid,
                                                  batch_size,
                                                  start_season,
                                                  beta,
                                                  gamma,
                                                  alpha_,
                                                  beta_,
                                                  gamma_,
                                                  level,
                                                  trend,
                                                  season,
                                                  xhat,
                                                  additive_seasonal);
    if (error) error[tid] = error_;
  }
}

// Test global and shared kernels
// https://github.com/NVIDIA/cuml/issues/890
template <typename Dtype>
void holtwinters_eval_gpu(const raft::handle_t& handle,
                          const Dtype* ts,
                          int n,
                          int batch_size,
                          int frequency,
                          const Dtype* start_level,
                          const Dtype* start_trend,
                          const Dtype* start_season,
                          const Dtype* alpha,
                          const Dtype* beta,
                          const Dtype* gamma,
                          Dtype* level,
                          Dtype* trend,
                          Dtype* season,
                          Dtype* xhat,
                          Dtype* error,
                          ML::SeasonalType seasonal)
{
  cudaStream_t stream = handle.get_stream().get();

  int total_blocks      = GET_NUM_BLOCKS(batch_size);
  int threads_per_block = GET_THREADS_PER_BLOCK(batch_size);

  // How much sm needed for shared kernel
  int sm_needed    = sizeof(Dtype) * threads_per_block * frequency;
  bool is_additive = seasonal == ML::SeasonalType::ADDITIVE;

  if (sm_needed > raft::getSharedMemPerBlock()) {
    rmm::device_uvector<Dtype> pseason(batch_size * frequency, stream);
    holtwinters_eval_gpu_global_kernel<Dtype>
      <<<total_blocks, threads_per_block, 0, stream>>>(ts,
                                                       n,
                                                       batch_size,
                                                       frequency,
                                                       start_level,
                                                       start_trend,
                                                       start_season,
                                                       pseason.data(),
                                                       alpha,
                                                       beta,
                                                       gamma,
                                                       level,
                                                       trend,
                                                       season,
                                                       xhat,
                                                       error,
                                                       is_additive);
  } else {
    holtwinters_eval_gpu_shared_kernel<Dtype>
      <<<total_blocks, threads_per_block, sm_needed, stream>>>(ts,
                                                               n,
                                                               batch_size,
                                                               frequency,
                                                               start_level,
                                                               start_trend,
                                                               start_season,
                                                               alpha,
                                                               beta,
                                                               gamma,
                                                               level,
                                                               trend,
                                                               season,
                                                               xhat,
                                                               error,
                                                               is_additive);
  }
}
