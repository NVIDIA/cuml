/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/common/checked_arithmetic.hpp>
#include <cuml/common/utils.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>
// #TODO: Replace with public header when ready
#include <raft/linalg/detail/cublas_wrappers.hpp>
// #TODO: Replace with public header when ready
#include "hw_utils.cuh"

#include <raft/linalg/detail/cusolver_wrappers.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

// optimize, maybe im2col ?
// https://github.com/NVIDIA/cuml/issues/891
template <typename Dtype>
CUML_KERNEL void conv1d_kernel(const Dtype* input,
                               int batch_size,
                               const Dtype* filter,
                               int filter_size,
                               Dtype* output,
                               int output_size)
{
  const int tid = GET_TID;
  if (tid < batch_size) {
    for (int o = 0; o < output_size; ++o) {
      Dtype out = 0.;
      for (int i = 0; i < filter_size; ++i)
        out += filter[i] * input[tid + (i + o) * batch_size];
      output[tid + o * batch_size] = out;
    }
  }
}

template <typename Dtype>
void conv1d(const raft::handle_t& handle,
            const Dtype* input,
            int batch_size,
            const Dtype* filter,
            int filter_size,
            Dtype* output,
            int output_size)
{
  int total_threads = batch_size;
  conv1d_kernel<Dtype>
    <<<GET_NUM_BLOCKS(total_threads),
       GET_THREADS_PER_BLOCK(total_threads),
       0,
       handle.get_stream()>>>(input, batch_size, filter, filter_size, output, output_size);
}

// https://github.com/NVIDIA/cuml/issues/891
template <typename Dtype>
CUML_KERNEL void season_mean_kernel(const Dtype* season,
                                    int len,
                                    int batch_size,
                                    Dtype* start_season,
                                    int frequency,
                                    int half_filter_size,
                                    bool ADDITIVE_KERNEL)
{
  int tid = GET_TID;
  if (tid < batch_size) {
    Dtype mean = 0.0;
    for (int i = 0; i < frequency; ++i) {
      Dtype period_mean = 0.;
      int c             = 0;
      for (int k = i; k < len; k = k + frequency) {
        period_mean += season[k * batch_size + tid];
        c++;
      }
      int count = 1 + ((len - i - 1) / frequency);
      period_mean /= count;
      int ss_idx                              = (i + half_filter_size) % frequency;
      start_season[ss_idx * batch_size + tid] = period_mean;
      mean += period_mean;
    }
    mean /= frequency;
    for (int i = 0; i < frequency; ++i) {
      if (ADDITIVE_KERNEL)
        start_season[i * batch_size + tid] -= mean;
      else  // MULTIPLOCATIVE
        start_season[i * batch_size + tid] /= mean;
    }
  }
}

template <typename Dtype>
void season_mean(const raft::handle_t& handle,
                 const Dtype* season,
                 int len,
                 int batch_size,
                 Dtype* start_season,
                 int frequency,
                 int half_filter_size,
                 ML::SeasonalType seasonal)
{
  cudaStream_t stream = handle.get_stream();
  bool is_additive    = seasonal == ML::SeasonalType::ADDITIVE;
  season_mean_kernel<Dtype>
    <<<GET_NUM_BLOCKS(batch_size), GET_THREADS_PER_BLOCK(batch_size), 0, stream>>>(
      season, len, batch_size, start_season, frequency, half_filter_size, is_additive);
}

template <typename Dtype>
CUML_KERNEL void RinvKernel(const Dtype* A, Dtype* Rinv, int trend_len)
{
  // Inverse of R (2x2 upper triangular matrix)
  int tid = GET_TID;
  if (tid == 0) {
    Dtype a = A[0], b = A[trend_len], d = A[trend_len + 1];
    Dtype factor = 1. / (a * d);
    Rinv[0]      = factor * d;
    Rinv[1]      = 0.;
    Rinv[2]      = -factor * b;
    Rinv[3]      = factor * a;
  }
}

template <typename Dtype>
CUML_KERNEL void batched_ls_solver_kernel(
  const Dtype* B, const Dtype* rq, int batch_size, int len, Dtype* level, Dtype* trend)
{
  int tid = GET_TID;
  if (tid < batch_size) {
    Dtype level_ = 0., trend_ = 0.;
    for (int i = 0; i < len; ++i) {
      Dtype b = B[tid + i * batch_size];
      level_ += rq[2 * i] * b;
      trend_ += rq[2 * i + 1] * b;
    }
    level[tid] = level_;
    trend[tid] = trend_;
  }
}

template <typename Dtype>
void batched_ls(const raft::handle_t& handle,
                const Dtype* data,
                int trend_len,
                int batch_size,
                Dtype* level,
                Dtype* trend)
{
  cudaStream_t stream           = handle.get_stream();
  cublasHandle_t cublas_h       = handle.get_cublas_handle();
  cusolverDnHandle_t cusolver_h = handle.get_cusolver_dn_handle();

  const Dtype one  = (Dtype)1.;
  const Dtype zero = (Dtype)0.;
  int geqrf_buffer;
  int orgqr_buffer;
  int lwork_size;

  // Allocate memory
  std::vector<Dtype> A_h(2 * trend_len);

  rmm::device_uvector<Dtype> A_d(2 * trend_len, stream);
  rmm::device_uvector<Dtype> tau_d(2, stream);
  rmm::device_uvector<Dtype> Rinv_d(4, stream);
  rmm::device_uvector<Dtype> R1Qt_d(2 * trend_len, stream);
  rmm::device_scalar<int> dev_info_d(stream);

  // Prepare A
  for (int i = 0; i < trend_len; ++i) {
    A_h[i]             = (Dtype)1.;
    A_h[trend_len + i] = (Dtype)(i + 1);
  }
  raft::update_device(A_d.data(), A_h.data(), 2 * trend_len, stream);

  // #TODO: Call from public API when ready
  RAFT_CUSOLVER_TRY(raft::linalg::detail::cusolverDngeqrf_bufferSize<Dtype>(
    cusolver_h, trend_len, 2, A_d.data(), 2, &geqrf_buffer));

  // #TODO: Call from public API when ready
  RAFT_CUSOLVER_TRY(raft::linalg::detail::cusolverDnorgqr_bufferSize<Dtype>(
    cusolver_h, trend_len, 2, 2, A_d.data(), 2, tau_d.data(), &orgqr_buffer));

  lwork_size = geqrf_buffer > orgqr_buffer ? geqrf_buffer : orgqr_buffer;
  rmm::device_uvector<Dtype> lwork_d(lwork_size, stream);

  // QR decomposition of A
  // #TODO: Call from public API when ready
  RAFT_CUSOLVER_TRY(raft::linalg::detail::cusolverDngeqrf<Dtype>(cusolver_h,
                                                                 trend_len,
                                                                 2,
                                                                 A_d.data(),
                                                                 trend_len,
                                                                 tau_d.data(),
                                                                 lwork_d.data(),
                                                                 lwork_size,
                                                                 dev_info_d.data(),
                                                                 stream));

  // Single thread kernel to inverse R
  RinvKernel<Dtype><<<1, 1, 0, stream>>>(A_d.data(), Rinv_d.data(), trend_len);

  // R1QT = inv(R)*transpose(Q)
  // #TODO: Call from public API when ready
  RAFT_CUSOLVER_TRY(raft::linalg::detail::cusolverDnorgqr<Dtype>(cusolver_h,
                                                                 trend_len,
                                                                 2,
                                                                 2,
                                                                 A_d.data(),
                                                                 trend_len,
                                                                 tau_d.data(),
                                                                 lwork_d.data(),
                                                                 lwork_size,
                                                                 dev_info_d.data(),
                                                                 stream));

  // #TODO: Call from public API when ready
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemm<Dtype>(cublas_h,
                                                          CUBLAS_OP_N,
                                                          CUBLAS_OP_T,
                                                          2,
                                                          trend_len,
                                                          2,
                                                          &one,
                                                          Rinv_d.data(),
                                                          2,
                                                          A_d.data(),
                                                          trend_len,
                                                          &zero,
                                                          R1Qt_d.data(),
                                                          2,
                                                          stream));

  batched_ls_solver_kernel<Dtype>
    <<<GET_NUM_BLOCKS(batch_size), GET_THREADS_PER_BLOCK(batch_size), 0, stream>>>(
      data, R1Qt_d.data(), batch_size, trend_len, level, trend);
}

template <typename Dtype>
void stl_decomposition_gpu(const raft::handle_t& handle,
                           const Dtype* ts,
                           int n,
                           int batch_size,
                           int frequency,
                           int start_periods,
                           Dtype* start_level,
                           Dtype* start_trend,
                           Dtype* start_season,
                           ML::SeasonalType seasonal)
{
  cudaStream_t stream     = handle.get_stream();
  cublasHandle_t cublas_h = handle.get_cublas_handle();

  const int end         = ML::checked_mul<int>(start_periods, frequency);
  const int filter_size = (frequency / 2) * 2 + 1;
  if (end < filter_size) {
    RAFT_FAIL(
      "HoltWintersDecompose: start_periods*frequency (%d) must be >= filter_size (%d); "
      "increase start_periods or decrease frequency",
      end,
      filter_size);
  }
  // end >= filter_size guaranteed by the check above, so the subtraction is safe;
  // checked_sub makes the underflow guard explicit.
  const int trend_len             = ML::checked_sub<int>(end, filter_size - 1);
  std::size_t const batch_trend_n = ML::checked_mul<std::size_t>(batch_size, trend_len);

  // Set filter
  std::vector<Dtype> filter_h(filter_size, 1. / frequency);
  if (frequency % 2 == 0) {
    filter_h.front() /= 2;
    filter_h.back() /= 2;
  }

  rmm::device_uvector<Dtype> filter_d(filter_size, stream);
  raft::update_device(filter_d.data(), filter_h.data(), filter_size, stream);

  // Set Trend
  rmm::device_uvector<Dtype> trend_d(batch_trend_n, stream);
  conv1d<Dtype>(handle, ts, batch_size, filter_d.data(), filter_size, trend_d.data(), trend_len);

  rmm::device_uvector<Dtype> season_d(batch_trend_n, stream);

  const int ts_offset = (filter_size / 2) * batch_size;
  if (seasonal == ML::SeasonalType::ADDITIVE) {
    const Dtype one       = 1.;
    const Dtype minus_one = -1.;
    // #TODO: Call from public API when ready
    RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgeam<Dtype>(cublas_h,
                                                            CUBLAS_OP_N,
                                                            CUBLAS_OP_N,
                                                            trend_len,
                                                            batch_size,
                                                            &one,
                                                            ts + ts_offset,
                                                            trend_len,
                                                            &minus_one,
                                                            trend_d.data(),
                                                            trend_len,
                                                            season_d.data(),
                                                            trend_len,
                                                            stream));
  } else {
    rmm::device_uvector<Dtype> aligned_ts(batch_size * trend_len, stream);
    raft::copy(aligned_ts.data(), ts + ts_offset, batch_size * trend_len, stream);
    raft::linalg::eltwiseDivide<Dtype>(
      season_d.data(), aligned_ts.data(), trend_d.data(), trend_len * batch_size, stream);
  }

  season_mean(handle,
              season_d.data(),
              trend_len,
              batch_size,
              start_season,
              frequency,
              filter_size / 2,
              seasonal);

  batched_ls(handle, trend_d.data(), trend_len, batch_size, start_level, start_trend);
}
