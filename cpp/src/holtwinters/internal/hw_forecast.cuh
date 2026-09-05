/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "hw_utils.cuh"

template <typename Dtype>
CUML_KERNEL void holtwinters_seasonal_forecast_kernel(Dtype* forecast,
                                                      int h,
                                                      int batch_size,
                                                      int frequency,
                                                      const Dtype* level_coef,
                                                      const Dtype* trend_coef,
                                                      const Dtype* season_coef,
                                                      bool additive)
{
  int tid = GET_TID;
  if (tid < batch_size) {
    const Dtype level = (level_coef) ? level_coef[tid] : 0.;
    const Dtype trend = (trend_coef) ? trend_coef[tid] : 0.;
    for (int i = 0; i < h; ++i) {
      const Dtype season = season_coef[tid + (i % frequency) * batch_size];
      if (additive)
        forecast[tid + i * batch_size] = level + trend * (i + 1) + season;
      else
        forecast[tid + i * batch_size] = (level + trend * (i + 1)) * season;
    }
  }
}

template <typename Dtype>
CUML_KERNEL void holtwinters_nonseasonal_forecast_kernel(
  Dtype* forecast, int h, int batch_size, const Dtype* level_coef, const Dtype* trend_coef)
{
  int tid = GET_TID;
  if (tid < batch_size) {
    const Dtype level = (level_coef) ? level_coef[tid] : 0.;
    const Dtype trend = trend_coef[tid];
    for (int i = 0; i < h; ++i)
      forecast[tid + i * batch_size] = level + trend * (i + 1);
  }
}

template <typename Dtype>
CUML_KERNEL void holtwinters_level_forecast_kernel(Dtype* forecast,
                                                   int h,
                                                   int batch_size,
                                                   const Dtype* level_coef)
{
  int tid = GET_TID;
  if (tid < batch_size) {
    const Dtype level = level_coef[tid];
    for (int i = 0; i < h; ++i)
      forecast[tid + i * batch_size] = level;
  }
}

template <typename Dtype>
void holtwinters_forecast_gpu(const raft::handle_t& handle,
                              Dtype* forecast,
                              int h,
                              int batch_size,
                              int frequency,
                              const Dtype* level_coef,
                              const Dtype* trend_coef,
                              const Dtype* season_coef,
                              ML::SeasonalType seasonal)
{
  cudaStream_t stream = handle.get_stream().get();

  int total_blocks      = GET_NUM_BLOCKS(batch_size);
  int threads_per_block = GET_THREADS_PER_BLOCK(batch_size);

  if (trend_coef == nullptr && season_coef == nullptr) {
    holtwinters_level_forecast_kernel<Dtype>
      <<<total_blocks, threads_per_block, 0, stream>>>(forecast, h, batch_size, level_coef);
  } else if (season_coef == nullptr) {
    holtwinters_nonseasonal_forecast_kernel<Dtype><<<total_blocks, threads_per_block, 0, stream>>>(
      forecast, h, batch_size, level_coef, trend_coef);
  } else {
    bool is_additive = seasonal == ML::SeasonalType::ADDITIVE;
    holtwinters_seasonal_forecast_kernel<Dtype><<<total_blocks, threads_per_block, 0, stream>>>(
      forecast, h, batch_size, frequency, level_coef, trend_coef, season_coef, is_additive);
  }
}
