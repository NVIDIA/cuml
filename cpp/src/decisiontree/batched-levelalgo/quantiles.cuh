/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "quantiles.h"

#include <cuml/common/checked_arithmetic.hpp>
#include <cuml/common/export.hpp>
#include <cuml/common/utils.hpp>

#include <raft/core/error.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/resource/comms.hpp>
#include <raft/random/rng_device.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <cuda/std/algorithm>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <thrust/unique.h>

#include <algorithm>
#include <cstdint>
#include <utility>

namespace ML {
namespace DT {

namespace detail {

// Draw global sample rows and copy values owned by this rank into a column-major sample buffer.
template <typename T>
CUML_KERNEL void sampleOwnedColumnsKernel(T* out,
                                          const T* data,
                                          const std::uint64_t* rank_row_offsets,
                                          int comm_size,
                                          std::uint64_t global_rows,
                                          int sample_count,
                                          int rank,
                                          int n_rows,
                                          int n_cols,
                                          bool row_major,
                                          std::uint64_t seed)
{
  int col        = blockIdx.x;
  int sample_idx = blockIdx.y * blockDim.x + threadIdx.x;
  if (sample_idx >= sample_count) { return; }

  std::uint64_t global_row = sample_idx;
  if (static_cast<std::uint64_t>(sample_count) != global_rows) {
    raft::random::UniformIntDistParams<std::uint64_t, std::uint64_t> uniform_int_dist_params;
    uniform_int_dist_params.start = 0;
    uniform_int_dist_params.end   = global_rows;
    uniform_int_dist_params.diff  = global_rows;
    raft::random::PCGenerator gen(seed, static_cast<uint64_t>(sample_idx), uint64_t(0));
    raft::random::custom_next(
      gen, &global_row, uniform_int_dist_params, std::uint64_t(0), std::uint64_t(0));
  }

  auto sample_end = ::cuda::std::lower_bound(
    rank_row_offsets + 1, rank_row_offsets + comm_size + 1, global_row + 1);
  int sample_rank           = static_cast<int>(sample_end - (rank_row_offsets + 1));
  std::uint64_t local_begin = rank_row_offsets[rank];
  if (sample_rank == rank) {
    int local_row = static_cast<int>(global_row - local_begin);
    out[static_cast<std::size_t>(col) * sample_count + sample_idx] =
      row_major ? data[static_cast<std::size_t>(local_row) * n_cols + col]
                : data[static_cast<std::size_t>(col) * n_rows + local_row];
  }
}

}  // namespace detail

// Convert sorted per-column samples into quantile candidates and compact duplicate candidates.
template <typename T>
CUML_KERNEL void computeQuantilesBatchedKernel(
  T* quantiles, int* n_bins, const T* sorted_data, const int max_n_bins, const int sample_count)
{
  int col           = blockIdx.x;
  T* col_quantiles  = quantiles + static_cast<int64_t>(col) * max_n_bins;
  const T* col_data = sorted_data + static_cast<int64_t>(col) * sample_count;
  double bin_width  = static_cast<double>(sample_count) / max_n_bins;

  for (int bin = threadIdx.x; bin < max_n_bins; bin += blockDim.x) {
    // get index by interpolation
    int idx            = int(round((bin + 1) * bin_width)) - 1;
    idx                = min(max(0, idx), sample_count - 1);
    col_quantiles[bin] = col_data[idx];
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    // make quantiles unique, in-place
    // thrust::seq to explicitly disable cuda dynamic parallelism here
    auto new_last = thrust::unique(thrust::seq, col_quantiles, col_quantiles + max_n_bins);
    // get the unique count
    n_bins[col] = new_last - col_quantiles;
  }

  __syncthreads();
  return;
}

template <typename T>
struct QuantileResult {
  rmm::device_uvector<T> quantiles_array;
  rmm::device_uvector<int> n_bins_array;

  Quantiles<T> view() & { return {quantiles_array.data(), n_bins_array.data()}; }
  Quantiles<T> view() && = delete;
};

/**
 * @brief Compute per-feature quantile split candidates from uniformly sampled rows.
 *
 * A deterministic global row sample is drawn once with replacement and shared across
 * feature columns. When the requested sample budget covers the global row count, all
 * rows are used.
 *
 * @tparam T Floating-point input type.
 * @param handle RAFT handle used for stream and resource access.
 * @param data Input matrix. When `row_major` is false, data is column-major with
 * shape `[n_cols, n_rows]`; when `row_major` is true, data is row-major with
 * shape `[n_rows, n_cols]`.
 * @param max_n_bins Maximum number of quantile candidates to retain per feature.
 * @param n_rows Number of local rows in `data` for this rank.
 * @param n_cols Number of columns in `data`.
 * @param oversampling_factor Multiplier applied to `max_n_bins` to choose the
 * sampled row budget per feature before sorting and quantile extraction. The
 * default of 4 is a conservative choice while still bounding memory; for fixed
 * `max_n_bins`, rank error decreases like O(1 / sqrt(oversampling_factor)), so
 * returns from increasing this are strongly diminishing.
 * @param seed User seed for deterministic sampling.
 * @param row_major Whether `data` is row-major instead of column-major.
 * @return Quantile metadata and owning device buffers for quantile values and bin counts.
 */
template <typename T>
CUML_EXPORT QuantileResult<T> computeQuantiles(const raft::handle_t& handle,
                                               const T* data,
                                               int max_n_bins,
                                               int n_rows,
                                               int n_cols,
                                               int oversampling_factor = 4,
                                               uint64_t seed           = uint64_t{0},
                                               bool row_major          = false)
{
  raft::common::nvtx::push_range("computeQuantiles");
  auto stream      = handle.get_stream();
  bool distributed = raft::resource::comms_initialized(handle) && handle.get_comms().get_size() > 1;

  RAFT_EXPECTS(max_n_bins > 0, "max_n_bins must be positive");
  RAFT_EXPECTS(distributed ? n_rows >= 0 : n_rows > 0, "n_rows must be positive");
  RAFT_EXPECTS(n_rows == 0 || data != nullptr, "data pointer must not be null");
  RAFT_EXPECTS(n_cols > 0, "n_cols must be positive");
  RAFT_EXPECTS(oversampling_factor > 0, "oversampling_factor must be positive");
  int rank      = distributed ? handle.get_comms().get_rank() : 0;
  int comm_size = distributed ? handle.get_comms().get_size() : 1;

  // Build exclusive global row offsets so sampled global row ids can be mapped to owning ranks.
  rmm::device_uvector<std::uint64_t> rank_row_offsets(comm_size + 1, stream);
  rmm::device_uvector<std::uint64_t> local_row_count(1, stream);
  auto local_rows = static_cast<std::uint64_t>(n_rows);
  RAFT_CUDA_TRY(cudaMemsetAsync(rank_row_offsets.data(), 0, sizeof(std::uint64_t), stream));
  raft::update_device(local_row_count.data(), &local_rows, 1, stream);

  if (distributed) {
    // Gather each rank's local row count so global row ids can be mapped back to rank-local rows.
    handle.get_comms().allgather(local_row_count.data(), rank_row_offsets.data() + 1, 1, stream);
    ASSERT(handle.get_comms().sync_stream(stream) == raft::comms::status_t::SUCCESS,
           "An error occurred in the distributed RF quantile row-count all-gather.");
  } else {
    raft::copy(rank_row_offsets.data() + 1, local_row_count.data(), 1, stream);
  }
  thrust::inclusive_scan(rmm::exec_policy(stream),
                         rank_row_offsets.data(),
                         rank_row_offsets.data() + comm_size + 1,
                         rank_row_offsets.data());
  std::uint64_t global_rows;
  raft::update_host(&global_rows, rank_row_offsets.data() + comm_size, 1, stream);
  handle.sync_stream(stream);
  RAFT_EXPECTS(global_rows > 0, "global row count must be positive");

  // Allocate one shared row sample for all columns and the buffers used to sort it by column.
  int sample_count = ML::narrow_cast<int>(std::min<std::uint64_t>(
    global_rows, ML::checked_mul<std::uint64_t>(max_n_bins, oversampling_factor)));

  std::size_t total_sample_values = ML::checked_mul<std::size_t>(sample_count, n_cols);
  rmm::device_uvector<T> sampled_columns(total_sample_values, stream);
  rmm::device_uvector<T> sorted_samples(total_sample_values, stream);

  int n_threads        = 256;
  auto segment_offsets = thrust::make_transform_iterator(
    thrust::make_counting_iterator<std::int64_t>(0),
    [sample_count] __host__ __device__(std::int64_t col) { return col * sample_count; });
  rmm::device_uvector<T> quantiles_array(ML::checked_mul<std::size_t>(n_cols, max_n_bins), stream);
  rmm::device_uvector<int> n_bins_array(n_cols, stream);

  // Fill this rank's owned positions in the global sample; all other positions remain zero.
  RAFT_CUDA_TRY(cudaMemsetAsync(sampled_columns.data(),
                                0,
                                ML::checked_mul<std::size_t>(sizeof(T), total_sample_values),
                                stream));
  dim3 sample_grid(n_cols, (sample_count + n_threads - 1) / n_threads);
  detail::sampleOwnedColumnsKernel<<<sample_grid, n_threads, 0, stream>>>(sampled_columns.data(),
                                                                          data,
                                                                          rank_row_offsets.data(),
                                                                          comm_size,
                                                                          global_rows,
                                                                          sample_count,
                                                                          rank,
                                                                          n_rows,
                                                                          n_cols,
                                                                          row_major,
                                                                          seed);
  RAFT_CUDA_TRY(cudaGetLastError());
  if (distributed) {
    // Every global sample position is owned by exactly one rank. A SUM all-reduce turns the sparse
    // per-rank buffers into the same dense sample buffer on all ranks without per-column gathers or
    // rank-dependent counts/displacements.
    handle.get_comms().allreduce(sampled_columns.data(),
                                 sampled_columns.data(),
                                 total_sample_values,
                                 raft::comms::op_t::SUM,
                                 stream);
    ASSERT(handle.get_comms().sync_stream(stream) == raft::comms::status_t::SUCCESS,
           "An error occurred in the distributed RF quantile sample all-reduce.");
  }

  // Sort each feature column's sampled values as an independent segment in one batched call.
  size_t temp_storage_bytes = 0;
  // Query temporary storage for the batched segmented radix sort.
  RAFT_CUDA_TRY(
    cub::DeviceSegmentedRadixSort::SortKeys(nullptr,
                                            temp_storage_bytes,
                                            sampled_columns.data(),
                                            sorted_samples.data(),
                                            static_cast<std::int64_t>(total_sample_values),
                                            static_cast<std::int64_t>(n_cols),
                                            segment_offsets,
                                            segment_offsets + 1,
                                            0,
                                            8 * sizeof(T),
                                            stream));
  rmm::device_uvector<char> d_temp_storage(temp_storage_bytes, stream);

  RAFT_CUDA_TRY(
    cub::DeviceSegmentedRadixSort::SortKeys((void*)(d_temp_storage.data()),
                                            temp_storage_bytes,
                                            sampled_columns.data(),
                                            sorted_samples.data(),
                                            static_cast<std::int64_t>(total_sample_values),
                                            static_cast<std::int64_t>(n_cols),
                                            segment_offsets,
                                            segment_offsets + 1,
                                            0,
                                            8 * sizeof(T),
                                            stream));

  // Interpolate quantile positions from the sorted samples and record the non-duplicate bin count.
  computeQuantilesBatchedKernel<<<n_cols, std::min(1024, max_n_bins), 0, stream>>>(
    quantiles_array.data(), n_bins_array.data(), sorted_samples.data(), max_n_bins, sample_count);
  RAFT_CUDA_TRY(cudaGetLastError());

  handle.sync_stream(stream);

  raft::common::nvtx::pop_range();
  return {std::move(quantiles_array), std::move(n_bins_array)};
}

}  // namespace DT
}  // namespace ML
