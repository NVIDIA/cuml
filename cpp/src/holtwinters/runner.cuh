/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "internal/hw_decompose.cuh"
#include "internal/hw_eval.cuh"
#include "internal/hw_forecast.cuh"
#include "internal/hw_optim.cuh"

#include <cuml/common/checked_arithmetic.hpp>
#include <cuml/tsa/holtwinters_params.h>

#include <raft/util/cudart_utils.hpp>
// #TODO: Replace with public header when ready
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/linalg/transpose.cuh>

#include <rmm/device_uvector.hpp>

#include <cstdint>
#include <limits>

namespace ML {

template <typename Dtype>
void HWTranspose(const raft::handle_t& handle, Dtype* data_in, int m, int n, Dtype* data_out)
{
  ASSERT(!(!data_in || !data_out || n < 1 || m < 1), "HW error in in line %d", __LINE__);
  const raft::handle_t& handle_impl = handle;
  raft::stream_syncer _(handle_impl);
  cudaStream_t stream     = handle_impl.get_stream().get();
  cublasHandle_t cublas_h = handle_impl.get_cublas_handle();

  raft::linalg::transpose<Dtype>(handle, data_in, data_out, n, m, stream);
}

namespace detail {
// Narrow an int64_t buffer-size product back to the public `int*` output type,
// trapping if it does not fit. Computing in int64_t first prevents the signed
// overflow that would otherwise truncate to a small positive value and feed an
// undersized allocation to downstream kernels.
inline int hw_narrow_size(std::int64_t value, const char* name)
{
  if (value < 0 || value > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
    RAFT_FAIL("HoltWintersBufferSize: %s = %lld exceeds int range; reduce batch_size or n",
              name,
              static_cast<long long>(value));
  }
  return static_cast<int>(value);
}
}  // namespace detail

void HoltWintersBufferSize(int n,
                           int batch_size,
                           int frequency,
                           bool use_beta,
                           bool use_gamma,
                           int* start_leveltrend_len,
                           int* start_season_len,
                           int* components_len,
                           int* error_len,
                           int* leveltrend_coef_shift,
                           int* season_coef_shift)
{
  std::int64_t const w_len   = use_gamma ? frequency : (use_beta ? 2 : 1);
  std::int64_t const n64     = n;
  std::int64_t const bs64    = batch_size;
  std::int64_t const freq64  = frequency;
  std::int64_t const n_minus = checked_sub<std::int64_t>(n64, w_len);

  if (start_leveltrend_len) *start_leveltrend_len = batch_size;
  if (use_gamma && start_season_len) {
    *start_season_len =
      detail::hw_narrow_size(checked_mul<std::int64_t>(freq64, bs64), "start_season_len");
  }

  if (components_len) {
    *components_len =
      detail::hw_narrow_size(checked_mul<std::int64_t>(n_minus, bs64), "components_len");
  }

  if (leveltrend_coef_shift) {
    *leveltrend_coef_shift =
      detail::hw_narrow_size(checked_mul<std::int64_t>(checked_sub<std::int64_t>(n_minus, 1), bs64),
                             "leveltrend_coef_shift");
  }
  if (use_gamma && season_coef_shift) {
    *season_coef_shift = detail::hw_narrow_size(
      checked_mul<std::int64_t>(checked_sub<std::int64_t>(n_minus, freq64), bs64),
      "season_coef_shift");
  }

  if (error_len) *error_len = batch_size;
}

template <typename Dtype>
void HoltWintersDecompose(const raft::handle_t& handle,
                          const Dtype* ts,
                          int n,
                          int batch_size,
                          int frequency,
                          Dtype* start_level,
                          Dtype* start_trend,
                          Dtype* start_season,
                          int start_periods,
                          ML::SeasonalType seasonal)
{
  const raft::handle_t& handle_impl = handle;
  raft::stream_syncer _(handle_impl);
  cudaStream_t stream     = handle_impl.get_stream().get();
  cublasHandle_t cublas_h = handle_impl.get_cublas_handle();

  if (start_level != nullptr && start_trend == nullptr &&
      start_season == nullptr) {  // level decomposition
    raft::copy(start_level, ts, batch_size, stream);
  } else if (start_level != nullptr && start_trend != nullptr &&
             start_season == nullptr) {  // trend decomposition
    raft::copy(start_level, ts + batch_size, batch_size, stream);
    raft::copy(start_trend, ts + batch_size, batch_size, stream);
    const Dtype alpha = -1.;
    // #TODO: Call from public API when ready
    RAFT_CUBLAS_TRY(raft::linalg::detail::cublasaxpy(
      cublas_h, batch_size, &alpha, ts, 1, start_trend, 1, stream));
    // cublas::axpy(batch_size, (Dtype)-1., ts, start_trend);
  } else if (start_level != nullptr && start_trend != nullptr && start_season != nullptr) {
    stl_decomposition_gpu(handle_impl,
                          ts,
                          n,
                          batch_size,
                          frequency,
                          start_periods,
                          start_level,
                          start_trend,
                          start_season,
                          seasonal);
  }
}

template <typename Dtype>
void HoltWintersEval(const raft::handle_t& handle,
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
  const raft::handle_t& handle_impl = handle;
  raft::stream_syncer _(handle_impl);
  cudaStream_t stream = handle_impl.get_stream().get();

  ASSERT(!((!start_trend) != (!beta) || (!start_season) != (!gamma)),
         "HW error in in line %d",
         __LINE__);
  ASSERT(!(!alpha || !start_level), "HW error in in line %d", __LINE__);
  ASSERT(!(start_season != nullptr && frequency < 2), "HW error in in line %d", __LINE__);
  if (!(!level && !trend && !season && !xhat && !error)) {
    holtwinters_eval_gpu(handle_impl,
                         ts,
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
                         seasonal);
  }
}

// expose line search step size - https://github.com/NVIDIA/cuml/issues/886
// Also, precision errors arise in optimization. There's floating point instability,
// and epsilon majorly influences the fitting based on precision. For a summary,
// https://github.com/NVIDIA/cuml/issues/888
template <typename Dtype>
void HoltWintersOptim(const raft::handle_t& handle,
                      const Dtype* ts,
                      int n,
                      int batch_size,
                      int frequency,
                      const Dtype* start_level,
                      const Dtype* start_trend,
                      const Dtype* start_season,
                      Dtype* alpha,
                      bool optim_alpha,
                      Dtype* beta,
                      bool optim_beta,
                      Dtype* gamma,
                      bool optim_gamma,
                      Dtype epsilon,
                      Dtype* level,
                      Dtype* trend,
                      Dtype* season,
                      Dtype* xhat,
                      Dtype* error,
                      OptimCriterion* optim_result,
                      OptimParams<Dtype>* optim_params,
                      ML::SeasonalType seasonal)
{
  const raft::handle_t& handle_impl = handle;
  raft::stream_syncer _(handle_impl);
  cudaStream_t stream = handle_impl.get_stream().get();

  // default values
  OptimParams<Dtype> optim_params_;
  optim_params_.eps                   = epsilon;
  optim_params_.min_param_diff        = (Dtype)1e-8;
  optim_params_.min_error_diff        = (Dtype)1e-8;
  optim_params_.min_grad_norm         = (Dtype)1e-4;
  optim_params_.bfgs_iter_limit       = 1000;
  optim_params_.linesearch_iter_limit = 100;
  optim_params_.linesearch_tau        = (Dtype)0.5;
  optim_params_.linesearch_c          = (Dtype)0.8;
  optim_params_.linesearch_step_size  = (Dtype)-1;

  if (optim_params) {
    if (optim_params->eps > .0) optim_params_.eps = optim_params->eps;
    if (optim_params->min_param_diff >= .0)
      optim_params_.min_param_diff = optim_params->min_param_diff;
    if (optim_params->min_error_diff >= .0)
      optim_params_.min_error_diff = optim_params->min_error_diff;
    if (optim_params->min_grad_norm >= .0)
      optim_params_.min_grad_norm = optim_params->min_grad_norm;
    if (optim_params->bfgs_iter_limit > 0)
      optim_params_.bfgs_iter_limit = optim_params->bfgs_iter_limit;
    if (optim_params->linesearch_iter_limit > 0)
      optim_params_.linesearch_iter_limit = optim_params->linesearch_iter_limit;
    if (optim_params->linesearch_tau > .0)
      optim_params_.linesearch_tau = optim_params->linesearch_tau;
    if (optim_params->linesearch_c > .0) optim_params_.linesearch_c = optim_params->linesearch_c;
    if (optim_params->linesearch_step_size > 0)
      optim_params_.linesearch_step_size = optim_params->linesearch_step_size;
  }

  ASSERT(alpha && start_level, "HW error in in line %d", __LINE__);
  ASSERT(!((!start_trend) != (!beta) || (!start_season) != (!gamma)),
         "HW error in in line %d",
         __LINE__);
  ASSERT(!(start_season && frequency < 2), "HW error in in line %d", __LINE__);
  ASSERT(!(!optim_alpha && !optim_beta && !optim_gamma), "HW error in in line %d", __LINE__);
  ASSERT(!((optim_beta && !beta) || (optim_gamma && !gamma)), "HW error in in line %d", __LINE__);
  if (!(!alpha && !beta && !gamma & !level && !trend && !season && !xhat && !error)) {
    holtwinters_optim_gpu(handle_impl,
                          ts,
                          n,
                          batch_size,
                          frequency,
                          start_level,
                          start_trend,
                          start_season,
                          alpha,
                          optim_alpha,
                          beta,
                          optim_beta,
                          gamma,
                          optim_gamma,
                          level,
                          trend,
                          season,
                          xhat,
                          error,
                          optim_result,
                          seasonal,
                          optim_params_);
  }
}

template <typename Dtype>
void HoltWintersForecast(const raft::handle_t& handle,
                         Dtype* forecast,
                         int h,
                         int batch_size,
                         int frequency,
                         const Dtype* level_coef,
                         const Dtype* trend_coef,
                         const Dtype* season_coef,
                         ML::SeasonalType seasonal)
{
  const raft::handle_t& handle_impl = handle;
  raft::stream_syncer _(handle_impl);
  cudaStream_t stream = handle_impl.get_stream().get();

  ASSERT(!(!level_coef && !trend_coef && !season_coef), "HW error in in line %d", __LINE__);
  ASSERT(!(season_coef && frequency < 2), "HW error in in line %d", __LINE__);
  holtwinters_forecast_gpu(
    handle_impl, forecast, h, batch_size, frequency, level_coef, trend_coef, season_coef, seasonal);
}

// change optim_gamma to false here to test bug in Double Exponential Smoothing
// https://github.com/NVIDIA/cuml/issues/889
template <typename Dtype>
void HoltWintersFitHelper(const raft::handle_t& handle,
                          int n,
                          int batch_size,
                          int frequency,
                          int start_periods,
                          ML::SeasonalType seasonal,
                          Dtype epsilon,
                          Dtype* data,
                          Dtype* level_d,
                          Dtype* trend_d,
                          Dtype* season_d,
                          Dtype* error_d)
{
  const raft::handle_t& handle_impl = handle;
  raft::stream_syncer _(handle_impl);
  cudaStream_t stream = handle_impl.get_stream().get();

  bool optim_alpha = true, optim_beta = true, optim_gamma = true;
  // initial values for alpha, beta and gamma
  std::vector<Dtype> alpha_h(batch_size, 0.4);
  std::vector<Dtype> beta_h(batch_size, 0.3);
  std::vector<Dtype> gamma_h(batch_size, 0.3);

  int leveltrend_seed_len, season_seed_len, components_len;
  int leveltrend_coef_offset, season_coef_offset;
  int error_len;

  HoltWintersBufferSize(n,
                        batch_size,
                        frequency,
                        optim_beta,
                        optim_gamma,
                        &leveltrend_seed_len,     // = batch_size
                        &season_seed_len,         // = frequency*batch_size
                        &components_len,          // = (n-w_len)*batch_size
                        &error_len,               // = batch_size
                        &leveltrend_coef_offset,  // = (n-wlen-1)*batch_size (last row)
                        &season_coef_offset);     // = (n-wlen-frequency)*batch_size(last freq rows)

  rmm::device_uvector<Dtype> dataset_d(checked_mul<std::size_t>(batch_size, n), stream);
  rmm::device_uvector<Dtype> alpha_d(batch_size, stream);
  raft::update_device(alpha_d.data(), alpha_h.data(), batch_size, stream);
  rmm::device_uvector<Dtype> level_seed_d(leveltrend_seed_len, stream);

  rmm::device_uvector<Dtype> beta_d(0, stream);
  rmm::device_uvector<Dtype> gamma_d(0, stream);
  rmm::device_uvector<Dtype> trend_seed_d(0, stream);
  rmm::device_uvector<Dtype> start_season_d(0, stream);

  if (optim_beta) {
    beta_d.resize(batch_size, stream);
    raft::update_device(beta_d.data(), beta_h.data(), batch_size, stream);
    trend_seed_d.resize(leveltrend_seed_len, stream);
  }

  if (optim_gamma) {
    gamma_d.resize(batch_size, stream);
    raft::update_device(gamma_d.data(), gamma_h.data(), batch_size, stream);
    start_season_d.resize(season_seed_len, stream);
  }

  // Step 1: transpose the dataset (ML expects col major dataset)
  HWTranspose(handle, data, batch_size, n, dataset_d.data());

  // Step 2: Decompose dataset to get seed for level, trend and seasonal values
  HoltWintersDecompose(handle,
                       dataset_d.data(),
                       n,
                       batch_size,
                       frequency,
                       level_seed_d.data(),
                       trend_seed_d.data(),
                       start_season_d.data(),
                       start_periods,
                       seasonal);

  // Step 3: Find optimal alpha, beta and gamma values (seasonal HW)
  HoltWintersOptim(handle,
                   dataset_d.data(),
                   n,
                   batch_size,
                   frequency,
                   level_seed_d.data(),
                   trend_seed_d.data(),
                   start_season_d.data(),
                   alpha_d.data(),
                   optim_alpha,
                   beta_d.data(),
                   optim_beta,
                   gamma_d.data(),
                   optim_gamma,
                   epsilon,
                   level_d,
                   trend_d,
                   season_d,
                   (Dtype*)nullptr,
                   error_d,
                   (OptimCriterion*)nullptr,
                   (OptimParams<Dtype>*)nullptr,
                   seasonal);
}

template <typename Dtype>
void HoltWintersForecastHelper(const raft::handle_t& handle,
                               int n,
                               int batch_size,
                               int frequency,
                               int h,
                               ML::SeasonalType seasonal,
                               Dtype* level_d,
                               Dtype* trend_d,
                               Dtype* season_d,
                               Dtype* forecast_d)
{
  const raft::handle_t& handle_impl = handle;
  raft::stream_syncer _(handle_impl);
  cudaStream_t stream = handle_impl.get_stream().get();

  bool optim_beta = true, optim_gamma = true;

  int leveltrend_seed_len, season_seed_len, components_len;
  int leveltrend_coef_offset, season_coef_offset;
  int error_len;

  HoltWintersBufferSize(n,
                        batch_size,
                        frequency,
                        optim_beta,
                        optim_gamma,
                        &leveltrend_seed_len,     // = batch_size
                        &season_seed_len,         // = frequency*batch_size
                        &components_len,          // = (n-w_len)*batch_size
                        &error_len,               // = batch_size
                        &leveltrend_coef_offset,  // = (n-wlen-1)*batch_size (last row)
                        &season_coef_offset);     // = (n-wlen-frequency)*batch_size(last freq rows)

  // Step 4: Do forecast
  HoltWintersForecast(handle,
                      forecast_d,
                      h,
                      batch_size,
                      frequency,
                      level_d + leveltrend_coef_offset,
                      trend_d + leveltrend_coef_offset,
                      season_d + season_coef_offset,
                      seasonal);
}

}  // namespace ML
