/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "builder_kernels.cuh"

#include <common/grid_sync.cuh>

#include <cuml/common/checked_arithmetic.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cuda_utils.cuh>

#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/tabulate_output_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>

#include <cstdint>
#include <cstdio>

namespace ML {
namespace DT {

static constexpr int TPB_DEFAULT = 128;

struct NodeSplitPartitionState {
  std::int64_t left_count;
  bool valid_row;
  bool goes_left;
};

struct NodeSplitPartitionScanOp {
  __host__ __device__ NodeSplitPartitionState operator()(const NodeSplitPartitionState& lhs,
                                                         const NodeSplitPartitionState& rhs) const
  {
    return {lhs.left_count + rhs.left_count, rhs.valid_row, rhs.goes_left};
  }
};

template <typename DataT>
static __global__ void resetLocalLeftCountsKernel(Split<DataT>* splits, std::size_t n_splits)
{
  const auto idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n_splits) { splits[idx].local_nLeft = 0; }
}

template <typename DataT, typename LabelT, int TPB>
static __global__ void countLocalLeftKernel(const Dataset<DataT, LabelT> dataset,
                                            const NodeWorkItem* work_items,
                                            Split<DataT>* splits,
                                            const WorkloadInfo* workload_info)
{
  using BlockReduce = cub::BlockReduce<std::int64_t, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  const auto workload_info_cta = workload_info[blockIdx.x];
  const auto nid               = workload_info_cta.nodeid;
  const auto work_item         = work_items[nid];
  const auto split             = splits[nid];

  std::int64_t thread_count = 0;
  const auto range_pos = std::size_t(workload_info_cta.offset_blockid) * blockDim.x + threadIdx.x;
  if (split.IsValid() && range_pos < work_item.instances.count) {
    const auto row = dataset.row_ids[work_item.instances.begin + range_pos];
    thread_count =
      dataset.value(row, split.colid) <= split.quesval ? std::int64_t{1} : std::int64_t{0};
  }

  const auto block_count = BlockReduce(temp_storage).Sum(thread_count);
  if (threadIdx.x == 0 && block_count > 0) {
    atomicAdd(reinterpret_cast<unsigned long long*>(&splits[nid].local_nLeft),
              static_cast<unsigned long long>(block_count));
  }
}

// Output side of the segmented partition scan. The scan supplies the
// inclusive left count and current row side for each logical row slot in its
// node segment; this writer uses that state to place the row into the temporary
// partition buffer.
template <typename DataT, typename LabelT, int TPB>
struct NodeSplitPartitionWriter {
  Dataset<DataT, LabelT> dataset;
  const NodeWorkItem* work_items;
  const Split<DataT>* splits;
  const WorkloadInfo* workload_info;
  std::int64_t* partition_row_ids;

  __host__ __device__ void operator()(std::ptrdiff_t index, NodeSplitPartitionState state) const
  {
    if (!state.valid_row) { return; }

    const auto slot              = std::size_t(index);
    const auto workload_info_cta = workload_info[slot / TPB];
    const auto nid               = workload_info_cta.nodeid;
    const auto work_item         = work_items[nid];
    const auto split             = splits[nid];

    const auto range_start = work_item.instances.begin;
    const auto range_pos   = std::size_t(workload_info_cta.offset_blockid) * TPB + slot % TPB;

    const auto row              = dataset.row_ids[range_start + range_pos];
    const auto rank             = state.goes_left ? std::size_t(state.left_count - std::int64_t{1})
                                                  : range_pos - std::size_t(state.left_count);
    const auto local_left_count = std::size_t(split.local_nLeft);
    const auto out_idx          = range_start + (state.goes_left ? rank : local_left_count + rank);
    partition_row_ids[out_idx]  = row;
  }
};

// Copy back only ranges for nodes that actually split. Leaf/invalid nodes keep
// their existing row-id order because the scan writer skips them too.
template <typename DataT, typename LabelT, int TPB>
static __global__ void nodeSplitCopyBackKernel(const Dataset<DataT, LabelT> dataset,
                                               const NodeWorkItem* work_items,
                                               const Split<DataT>* splits,
                                               const WorkloadInfo* workload_info,
                                               const std::int64_t* partition_row_ids)
{
  const auto workload_info_cta = workload_info[blockIdx.x];
  const auto nid               = workload_info_cta.nodeid;
  const auto work_item         = work_items[nid];
  const auto split             = splits[nid];
  if (!split.IsValid()) { return; }

  const auto range_start = work_item.instances.begin;
  const auto range_len   = work_item.instances.count;
  const auto range_pos   = std::size_t(workload_info_cta.offset_blockid) * blockDim.x + threadIdx.x;
  if (range_pos < range_len) {
    const auto idx       = range_start + range_pos;
    dataset.row_ids[idx] = partition_row_ids[idx];
  }
}

template <typename DataT, typename LabelT, int TPB>
void launchNodeSplitKernel(const Dataset<DataT, LabelT>& dataset,
                           const NodeWorkItem* work_items,
                           Split<DataT>* splits,
                           const WorkloadInfo* workload_info,
                           size_t n_blocks_dimx,
                           size_t n_work_items,
                           std::int64_t* partition_row_ids,
                           cudaStream_t builder_stream)
{
  if (n_blocks_dimx == 0) return;

  constexpr int reset_tpb = 128;
  const auto reset_grid   = raft::ceildiv(n_work_items, std::size_t{reset_tpb});
  resetLocalLeftCountsKernel<DataT>
    <<<ML::narrow_cast<ML::cuda_launch_t>(reset_grid), reset_tpb, 0, builder_stream>>>(
      splits, n_work_items);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  countLocalLeftKernel<DataT, LabelT, TPB>
    <<<ML::narrow_cast<ML::cuda_launch_t>(n_blocks_dimx), TPB, 0, builder_stream>>>(
      dataset, work_items, splits, workload_info);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Each slot corresponds to one thread lane in the tiled workload_info layout.
  // workload_info is grouped by node, so scan-by-key resets ranks at node boundaries.
  const auto n_slots = n_blocks_dimx * TPB;
  auto exec_policy   = rmm::exec_policy(builder_stream);
  auto slots_begin   = thrust::make_counting_iterator<std::size_t>(0);

  auto node_key = [workload_info] __host__ __device__(std::size_t slot) {
    return workload_info[slot / TPB].nodeid;
  };
  auto partition_state = [=] __host__ __device__(std::size_t slot) {
    const auto workload_info_cta = workload_info[slot / TPB];
    const auto nid               = workload_info_cta.nodeid;
    const auto work_item         = work_items[nid];
    const auto split             = splits[nid];
    if (!split.IsValid()) { return NodeSplitPartitionState{std::int64_t{0}, false, false}; }

    const auto range_pos = std::size_t(workload_info_cta.offset_blockid) * TPB + slot % TPB;
    if (range_pos >= work_item.instances.count) {
      return NodeSplitPartitionState{std::int64_t{0}, false, false};
    }

    const auto row       = dataset.row_ids[work_item.instances.begin + range_pos];
    const auto goes_left = dataset.value(row, split.colid) <= split.quesval;
    return NodeSplitPartitionState{goes_left ? std::int64_t{1} : std::int64_t{0}, true, goes_left};
  };

  // The scan input is a stream of per-slot partition states keyed by node id.
  // The scan output is a tabulated writer, so partition_row_ids is populated
  // during the scan rather than by a second scatter kernel.
  auto node_keys        = thrust::make_transform_iterator(slots_begin, node_key);
  auto partition_states = thrust::make_transform_iterator(slots_begin, partition_state);
  auto partition_writer =
    thrust::make_tabulate_output_iterator(NodeSplitPartitionWriter<DataT, LabelT, TPB>{
      dataset, work_items, splits, workload_info, partition_row_ids});
  thrust::inclusive_scan_by_key(exec_policy,
                                node_keys,
                                node_keys + n_slots,
                                partition_states,
                                partition_writer,
                                thrust::equal_to<std::int64_t>{},
                                NodeSplitPartitionScanOp{});

  // The original row_ids buffer remains the source during the scan, so copy back after it finishes.
  nodeSplitCopyBackKernel<DataT, LabelT, TPB><<<n_blocks_dimx, TPB, 0, builder_stream>>>(
    dataset, work_items, splits, workload_info, partition_row_ids);
}

template <typename DatasetT, typename NodeT, typename ObjectiveT>
static __global__ void buildLeafHistogramsKernel(ObjectiveT objective,
                                                 DatasetT dataset,
                                                 const NodeT* tree,
                                                 const InstanceRange* instance_ranges,
                                                 typename ObjectiveT::BinT* leaf_histograms)
{
  using BinT = typename ObjectiveT::BinT;
  extern __shared__ char shared_memory[];
  auto histogram = reinterpret_cast<BinT*>(shared_memory);
  auto node_id   = blockIdx.x;
  auto& node     = tree[node_id];
  auto range     = instance_ranges[node_id];
  if (!node.IsLeaf()) return;
  auto tid = threadIdx.x;
  for (int i = tid; i < dataset.num_outputs; i += blockDim.x) {
    histogram[i] = BinT();
  }
  __syncthreads();
  for (auto i = range.begin + tid; i < range.begin + range.count; i += blockDim.x) {
    auto row   = dataset.row_ids[i];
    auto label = dataset.labels[row];
    objective.IncrementHistogram(histogram, 1, 0, label, dataset, row);
  }
  __syncthreads();
  for (int i = tid; i < dataset.num_outputs; i += blockDim.x) {
    leaf_histograms[dataset.num_outputs * node_id + i] = histogram[i];
  }
}

template <typename DatasetT, typename NodeT, typename ObjectiveT>
void launchBuildLeafHistogramsKernel(ObjectiveT objective,
                                     DatasetT& dataset,
                                     const NodeT* tree,
                                     const InstanceRange* instance_ranges,
                                     typename ObjectiveT::BinT* leaf_histograms,
                                     int batch_size,
                                     size_t smem_size,
                                     cudaStream_t builder_stream)
{
  auto num_blocks = ML::narrow_cast<ML::cuda_launch_t>(batch_size);
  buildLeafHistogramsKernel<<<num_blocks, TPB_DEFAULT, smem_size, builder_stream>>>(
    objective, dataset, tree, instance_ranges, leaf_histograms);
}

template <typename ObjectiveT, typename DataT>
static __global__ void finalizeLeafKernel(const typename ObjectiveT::BinT* leaf_histograms,
                                          DataT* leaves,
                                          int num_outputs,
                                          int batch_size)
{
  auto node_id = int(blockIdx.x) * blockDim.x + threadIdx.x;
  if (node_id >= batch_size) return;
  ObjectiveT::SetLeafVector(
    leaf_histograms + num_outputs * node_id, num_outputs, leaves + num_outputs * node_id);
}

template <typename ObjectiveT, typename DataT>
void launchFinalizeLeafKernel(const typename ObjectiveT::BinT* leaf_histograms,
                              DataT* leaves,
                              int num_outputs,
                              int batch_size,
                              cudaStream_t builder_stream)
{
  auto num_blocks = ML::narrow_cast<ML::cuda_launch_t>(raft::ceildiv(batch_size, TPB_DEFAULT));
  finalizeLeafKernel<ObjectiveT, DataT><<<num_blocks, TPB_DEFAULT, 0, builder_stream>>>(
    leaf_histograms, leaves, num_outputs, batch_size);
}

/**
 * @brief For every threadblock, converts a pdf-histogram to a
 *        cdf-histogram inplace using inclusive block-sum-scan.
 */
template <typename BinT, int TPB>
DI BinT pdf_to_cdf(BinT* histogram, std::int64_t n_bins)
{
  // Blockscan instance preparation
  typedef cub::BlockScan<BinT, TPB> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  // variable to accumulate aggregate of sumscans of previous iterations
  BinT total_aggregate = BinT();

  for (std::int64_t tix = threadIdx.x; tix < raft::ceildiv(n_bins, std::int64_t{TPB}) * TPB;
       tix += blockDim.x) {
    BinT result;
    BinT block_aggregate;
    BinT element = tix < n_bins ? histogram[tix] : BinT();
    BlockScan(temp_storage).InclusiveSum(element, result, block_aggregate);
    __syncthreads();
    if (tix < n_bins) { histogram[tix] = result + total_aggregate; }
    total_aggregate += block_aggregate;
  }
  return total_aggregate;
}

template <typename DataT, typename LabelT, int TPB, typename ObjectiveT>
static __global__ void buildHistogramsKernel(typename ObjectiveT::BinT* histograms,
                                             std::int64_t max_n_bins,
                                             const Dataset<DataT, LabelT> dataset,
                                             const Quantiles<DataT> quantiles,
                                             const NodeWorkItem* work_items,
                                             std::int64_t colStart,
                                             const std::int64_t* column_samples,
                                             ObjectiveT objective,
                                             const WorkloadInfo* workload_info,
                                             bool use_global_memory_histogram)
{
  using BinT = typename ObjectiveT::BinT;
  extern __shared__ char smem[];

  WorkloadInfo workload_info_cta = workload_info[blockIdx.x];
  std::int64_t nid               = workload_info_cta.nodeid;
  const auto work_item           = work_items[nid];
  auto range_start               = work_item.instances.begin;
  auto range_len                 = work_item.instances.count;

  std::int64_t offset_blockid = workload_info_cta.offset_blockid;
  std::int64_t num_blocks     = workload_info_cta.num_blocks;

  std::int64_t colIndex = colStart + blockIdx.y;
  std::int64_t col      = column_samples[nid * dataset.n_sampled_cols + colIndex];
  int n_bins            = quantiles.n_bins_array[col];

  auto n_classes            = objective.NumClasses();
  auto end                  = range_start + range_len;
  auto histogram_len        = n_bins * n_classes;
  auto histograms_offset    = (std::size_t(nid) * gridDim.y + blockIdx.y) * max_n_bins * n_classes;
  auto* global_histogram    = histograms + histograms_offset;
  auto* histogram           = global_histogram;
  auto* quantiles_for_split = quantiles.quantiles_array + std::size_t(max_n_bins) * col;
  std::int64_t stride       = blockDim.x * num_blocks;
  std::int64_t tid          = threadIdx.x + offset_blockid * blockDim.x;

  if (!use_global_memory_histogram) {
    histogram              = alignPointer<BinT>(smem);
    auto* shared_quantiles = alignPointer<DataT>(histogram + histogram_len);
    quantiles_for_split    = shared_quantiles;
    for (std::int64_t i = threadIdx.x; i < histogram_len; i += blockDim.x) {
      histogram[i] = BinT();
    }
    for (std::int64_t b = threadIdx.x; b < n_bins; b += blockDim.x) {
      shared_quantiles[b] = quantiles.quantiles_array[max_n_bins * col + b];
    }
    __syncthreads();
  }

  for (auto i = range_start + tid; i < end; i += stride) {
    auto row   = dataset.row_ids[i];
    auto data  = dataset.value(row, col);
    auto label = dataset.labels[row];

    // Search bin indices so lower_bound uses 32-bit distance and advance arithmetic.
    auto bin_begin = cuda::counting_iterator<int, int>(0);
    auto bin_end   = bin_begin + n_bins;
    auto bin_it    = ::cuda::std::lower_bound(
      bin_begin, bin_end, data, [quantiles_for_split](int bin, DataT value) {
        return quantiles_for_split[bin] < value;
      });
    auto bin = bin_it == bin_end ? n_bins - 1 : *bin_it;
    objective.IncrementHistogram(histogram, n_bins, bin, label, dataset, row);
  }

  if (!use_global_memory_histogram) {
    __syncthreads();
    for (std::int64_t i = threadIdx.x; i < histogram_len; i += blockDim.x) {
      BinT::AtomicAdd(global_histogram + i, histogram[i]);
    }
  }
}

template <typename DataT, typename LabelT, int TPB, typename ObjectiveT>
static __global__ void findBestSplitsKernel(typename ObjectiveT::BinT* histograms,
                                            std::int64_t max_n_bins,
                                            const Dataset<DataT, LabelT> dataset,
                                            const Quantiles<DataT> quantiles,
                                            std::int64_t colStart,
                                            const std::int64_t* column_samples,
                                            int* mutex,
                                            volatile Split<DataT>* splits,
                                            ObjectiveT objective)
{
  using BinT                  = typename ObjectiveT::BinT;
  constexpr int n_split_warps = (TPB + raft::WarpSize - 1) / raft::WarpSize;
  __shared__ __align__(alignof(
    Split<DataT>)) unsigned char split_scratch_storage[sizeof(Split<DataT>) * n_split_warps];
  auto* split_scratch = reinterpret_cast<Split<DataT>*>(split_scratch_storage);

  std::int64_t nid = blockIdx.x;

  std::int64_t colIndex = colStart + blockIdx.y;
  std::int64_t col      = column_samples[nid * dataset.n_sampled_cols + colIndex];
  int n_bins            = quantiles.n_bins_array[col];

  auto n_classes            = objective.NumClasses();
  auto histograms_offset    = (std::size_t(nid) * gridDim.y + blockIdx.y) * max_n_bins * n_classes;
  auto* histogram           = histograms + histograms_offset;
  auto* quantiles_for_split = quantiles.quantiles_array + std::size_t(max_n_bins) * col;

  std::int64_t global_sample_count = 0;
  for (std::int64_t c = 0; c < n_classes; ++c) {
    global_sample_count +=
      static_cast<std::int64_t>(pdf_to_cdf<BinT, TPB>(histogram + n_bins * c, n_bins).Count());
  }

  __syncthreads();

  Split<DataT> sp =
    objective.Gain(histogram, quantiles_for_split, col, global_sample_count, n_bins);

  __syncthreads();

  sp.evalBestSplit(split_scratch, splits + nid, mutex + nid, quantiles_for_split, n_bins);
}

template <typename DataT, typename LabelT, int TPB, typename ObjectiveT>
void launchBuildHistogramsKernel(typename ObjectiveT::BinT* histograms,
                                 std::int64_t max_n_bins,
                                 const Dataset<DataT, LabelT>& dataset,
                                 const Quantiles<DataT>& quantiles,
                                 const NodeWorkItem* work_items,
                                 std::int64_t colStart,
                                 const std::int64_t* column_samples,
                                 ObjectiveT& objective,
                                 const WorkloadInfo* workload_info,
                                 dim3 histogram_grid,
                                 const SharedMemoryConfig& split_smem_config,
                                 cudaStream_t builder_stream)
{
  buildHistogramsKernel<DataT, LabelT, TPB, ObjectiveT>
    <<<histogram_grid, TPB, split_smem_config.histogram_dynamic_smem_size, builder_stream>>>(
      histograms,
      max_n_bins,
      dataset,
      quantiles,
      work_items,
      colStart,
      column_samples,
      objective,
      workload_info,
      split_smem_config.use_global_memory_histogram);
}

template <typename DataT, typename LabelT, int TPB, typename ObjectiveT>
void launchFindBestSplitsKernel(typename ObjectiveT::BinT* histograms,
                                std::int64_t max_n_bins,
                                const Dataset<DataT, LabelT>& dataset,
                                const Quantiles<DataT>& quantiles,
                                std::int64_t colStart,
                                const std::int64_t* column_samples,
                                int* mutex,
                                volatile Split<DataT>* splits,
                                ObjectiveT& objective,
                                dim3 split_grid,
                                cudaStream_t builder_stream)
{
  findBestSplitsKernel<DataT, LabelT, TPB, ObjectiveT><<<split_grid, TPB, 0, builder_stream>>>(
    histograms, max_n_bins, dataset, quantiles, colStart, column_samples, mutex, splits, objective);
}

}  // namespace DT
}  // namespace ML
