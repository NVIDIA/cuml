/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "kernels/builder_kernels.cuh"

#include <common/Timer.h>

#include <cuml/common/checked_arithmetic.hpp>
#include <cuml/common/pinned_host_vector.hpp>
#include <cuml/tree/decisiontree.hpp>
#include <cuml/tree/flatnode.h>

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/resource/comms.hpp>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>

#include <algorithm>
#include <cstdint>
#include <deque>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace ML {
namespace DT {

/**
 * Structure that manages the iterative batched-level training and building of nodes
 * in the host.
 */
template <typename DataT, typename LabelT>
class NodeQueue {
  using NodeT = SparseTreeNode<DataT, LabelT>;
  const DecisionTreeParams params;
  std::shared_ptr<DT::TreeMetaDataNode<DataT, LabelT>> tree;
  std::vector<InstanceRange> node_instances_;
  std::deque<NodeWorkItem> work_items_;

 public:
  NodeQueue(DecisionTreeParams params,
            size_t max_nodes,
            size_t local_sampled_rows,
            std::int64_t global_sampled_rows,
            int num_outputs)
    : params(params), tree(std::make_shared<DT::TreeMetaDataNode<DataT, LabelT>>())
  {
    tree->num_outputs = num_outputs;
    tree->sparsetree.reserve(max_nodes);
    tree->sparsetree.emplace_back(NodeT::CreateLeafNode(global_sampled_rows));
    tree->leaf_counter  = 1;
    tree->depth_counter = 0;
    node_instances_.reserve(max_nodes);
    node_instances_.emplace_back(InstanceRange{0, local_sampled_rows});
    if (this->IsExpandable(tree->sparsetree.back(), 0)) {
      work_items_.emplace_back(NodeWorkItem{0, 0, node_instances_.back()});
    }
  }

  std::shared_ptr<DT::TreeMetaDataNode<DataT, LabelT>> GetTree() { return tree; }
  const std::vector<InstanceRange>& GetInstanceRanges() { return node_instances_; }

  bool HasWork() { return work_items_.size() > 0; }

  auto Pop()
  {
    std::vector<NodeWorkItem> result;
    result.reserve(std::min(size_t(params.max_batch_size), work_items_.size()));
    while (work_items_.size() > 0 && result.size() < std::size_t(params.max_batch_size)) {
      result.emplace_back(work_items_.front());
      work_items_.pop_front();
    }
    return result;
  }

  // This node is allowed to be expanded further (if its split gain is high enough)
  bool IsExpandable(const NodeT& n, int depth)
  {
    if (depth >= params.max_depth) return false;
    if (n.InstanceCount() < params.min_samples_split) return false;
    if (params.max_leaves != -1 && tree->leaf_counter >= params.max_leaves) return false;
    return true;
  }

  template <typename SplitT>
  void Push(const std::vector<NodeWorkItem>& work_items, SplitT* h_splits)
  {
    // Update node queue based on splits
    for (std::size_t i = 0; i < work_items.size(); i++) {
      auto split        = h_splits[i];
      auto item         = work_items[i];
      auto parent_range = node_instances_.at(item.idx);
      if (!split.IsValid()) { continue; }

      if (params.max_leaves != -1 && tree->leaf_counter >= params.max_leaves) break;

      using NodeCountT            = decltype(std::declval<NodeT>().InstanceCount());
      auto const parent_count     = tree->sparsetree.at(item.idx).InstanceCount();
      auto const local_left_count = ML::narrow_cast<std::size_t>(split.local_nLeft);

      // parent
      tree->sparsetree.at(item.idx) = NodeT::CreateSplitNode(split.colid,
                                                             split.quesval,
                                                             split.best_metric_val,
                                                             int64_t(tree->sparsetree.size()),
                                                             parent_count);
      tree->leaf_counter++;
      // left
      tree->sparsetree.emplace_back(
        NodeT::CreateLeafNode(ML::narrow_cast<NodeCountT>(split.global_nLeft)));
      node_instances_.emplace_back(InstanceRange{parent_range.begin, local_left_count});

      // Do not add a work item if this child is definitely a leaf
      if (this->IsExpandable(tree->sparsetree.back(), item.depth + 1)) {
        work_items_.emplace_back(
          NodeWorkItem{tree->sparsetree.size() - 1, item.depth + 1, node_instances_.back()});
      }

      // right
      tree->sparsetree.emplace_back(NodeT::CreateLeafNode(ML::checked_sub<NodeCountT>(
        parent_count, ML::narrow_cast<NodeCountT>(split.global_nLeft))));
      node_instances_.emplace_back(
        InstanceRange{ML::checked_add<std::size_t>(parent_range.begin, local_left_count),
                      ML::checked_sub<std::size_t>(parent_range.count, local_left_count)});

      // Do not add a work item if this child is definitely a leaf
      if (this->IsExpandable(tree->sparsetree.back(), item.depth + 1)) {
        work_items_.emplace_back(
          NodeWorkItem{tree->sparsetree.size() - 1, item.depth + 1, node_instances_.back()});
      }

      // update depth
      tree->depth_counter = max(tree->depth_counter, item.depth + 1);
    }
  }
};

/**
 * Internal struct used to do all the heavy-lifting required for tree building
 */
template <typename ObjectiveT>
struct Builder {
  typedef typename ObjectiveT::DataT DataT;
  typedef typename ObjectiveT::LabelT LabelT;
  typedef typename ObjectiveT::BinT BinT;
  typedef SparseTreeNode<DataT, LabelT> NodeT;
  typedef Split<DataT> SplitT;
  typedef Dataset<DataT, LabelT> DatasetT;
  typedef Quantiles<DataT> QuantilesT;

  /** default threads per block for most kernels in here */
  static constexpr int TPB_DEFAULT = 128;
  // Tunable performance heuristic for the shared-memory histogram path. Large per-block
  // histograms, usually from large n_classes, can reduce occupancy enough that global memory is
  // faster even when the histogram fits in shared memory. 16 KiB keeps small/default histograms in
  // shared memory while avoiding the large-class shared-memory slowdown measured locally.
  static constexpr size_t tunable_split_histogram_dynamic_smem_limit_bytes = 16 * 1024;
  /** handle to get device properties */
  const raft::handle_t& handle;
  /** stream to launch kernels */
  cudaStream_t builder_stream;
  /** DT params */
  DecisionTreeParams params;
  /** input dataset */
  DatasetT dataset;
  /** quantiles */
  QuantilesT quantiles;
  /** Tree index */
  std::int64_t treeid;
  /** Seed used for randomization */
  uint64_t seed;
  /** number of nodes created in the current batch */
  std::int64_t* n_nodes;
  /** buffer of segmented histograms*/
  BinT* histograms;
  /** mutex array used for atomically updating best split */
  int* mutex;
  /** best splits for the current batch of nodes */
  SplitT* splits;
  /** current batch of nodes */
  NodeWorkItem* d_work_items;
  /** device AOS to map CTA blocks along dimx to nodes of a batch */
  WorkloadInfo* workload_info;
  /** host AOS to map CTA blocks along dimx to nodes of a batch */
  WorkloadInfo* h_workload_info;
  /** maximum CTA blocks along dimx */
  int max_blocks_dimx = 0;
  /** host array of splits */
  SplitT* h_splits;
  /** packed histogram buffer used by distributed all-reduce */
  void* packed_histograms;
  /** number of blocks used to parallelize column-wise computations */
  int n_blks_for_cols = 10;
  /** Memory alignment value */
  const size_t align_value = 512;
  std::int64_t* column_samples;
  /** temporary row IDs for row-wise out-of-place partitioning */
  std::int64_t* partition_row_ids;
  /** rmm device workspace buffer */
  rmm::device_uvector<char> d_buff;
  /** pinned host buffer to store the trained nodes */
  ML::pinned_host_vector<char> h_buff;
  /** true when a communicator with more than one rank is available */
  bool distributed;

  Builder(const raft::handle_t& handle,
          cudaStream_t s,
          std::int64_t treeid,
          uint64_t seed,
          const DecisionTreeParams& p,
          const DataT* data,
          const LabelT* labels,
          const double* sample_weight,
          std::int64_t n_rows,
          std::int64_t n_cols,
          rmm::device_uvector<std::int64_t>* row_ids,
          int n_classes,
          const QuantilesT& q,
          bool row_major = false)
    : handle(handle),
      builder_stream(s),
      treeid(treeid),
      seed(seed),
      params(p),
      dataset{data,
              labels,
              sample_weight,
              n_rows,
              n_cols,
              row_major ? n_cols : std::int64_t{1},
              row_major ? std::int64_t{1} : n_rows,
              ML::narrow_cast<std::int64_t>(row_ids->size()),
              std::max(std::int64_t{1}, std::int64_t(params.max_features * n_cols)),
              row_ids->data(),
              n_classes},
      quantiles(q),
      d_buff(0, builder_stream),
      distributed(raft::resource::comms_initialized(handle) && handle.get_comms().get_size() > 1)
  {
    max_blocks_dimx = ML::narrow_cast<int>(ML::checked_add<std::int64_t>(
      1, params.max_batch_size, dataset.n_sampled_rows / TPB_DEFAULT));
    ASSERT(q.quantiles_array != nullptr && q.n_bins_array != nullptr,
           "Currently quantiles need to be computed before this call!");
    ASSERT(n_classes >= 1, "n_classes should be at least 1");

    auto [device_workspace_size, host_workspace_size] = workspaceSize();
    d_buff.resize(device_workspace_size, builder_stream);
    h_buff.resize(host_workspace_size);
    assignWorkspace(d_buff.data(), h_buff.data());
  }

  /**
   * @brief calculates nearest aligned size of input w.r.t an `align_value`.
   *
   * @param[in] actual_size actual size in bytes of input
   * @return aligned size
   */
  size_t calculateAlignedBytes(const size_t actual_size) const
  {
    return raft::alignTo(actual_size, align_value);
  }

  size_t packedHistogramWorkspaceSize(std::size_t len_histograms) const
  {
    if (!distributed) { return 0; }

    auto const packed_count =
      ML::checked_mul<std::size_t>(reduction_buffer_size_v<BinT>, len_histograms);
    auto const packed_bytes = ML::checked_mul<std::size_t>(sizeof(double), packed_count);
    return calculateAlignedBytes(packed_bytes);
  }

  /**
   * @brief returns maximum nodes possible per tree
   * @return maximum nodes possible per tree
   */
  size_t maxNodes() const
  {
    if (params.max_depth < 13) {
      // Start with allocation for a dense tree for depth < 13
      return pow(2, (params.max_depth + 1)) - 1;
    } else {
      // Start with fixed size allocation for depth >= 13
      return 8191;
    }
  }

  /**
   * @brief calculate the workspace size required
   *
   * @return a pair of device workspace and host workspace size requirements
   */
  auto workspaceSize() const
  {
    size_t d_wsize = 0, h_wsize = 0;
    raft::common::nvtx::range fun_scope("Builder::workspaceSize @builder.cuh [batched-levelalgo]");
    auto max_batch            = params.max_batch_size;
    size_t max_len_histograms = ML::checked_mul<std::size_t>(
      max_batch, params.max_n_bins, n_blks_for_cols, dataset.num_outputs);
    auto histograms_bytes    = ML::checked_mul<std::size_t>(sizeof(BinT), max_len_histograms);
    auto mutex_bytes         = ML::checked_mul<std::size_t>(sizeof(int), max_batch);
    auto splits_bytes        = ML::checked_mul<std::size_t>(sizeof(SplitT), max_batch);
    auto work_items_bytes    = ML::checked_mul<std::size_t>(sizeof(NodeWorkItem), max_batch);
    auto workload_info_bytes = ML::checked_mul<std::size_t>(sizeof(WorkloadInfo), max_blocks_dimx);
    auto column_samples_bytes =
      ML::checked_mul<std::size_t>(sizeof(std::int64_t), max_batch, dataset.n_sampled_cols);
    auto partition_row_ids_bytes =
      ML::checked_mul<std::size_t>(sizeof(std::int64_t), dataset.n_sampled_rows);

    d_wsize += calculateAlignedBytes(sizeof(std::int64_t));     // n_nodes
    d_wsize += calculateAlignedBytes(histograms_bytes);         // histograms
    d_wsize += calculateAlignedBytes(mutex_bytes);              // mutex
    d_wsize += calculateAlignedBytes(splits_bytes);             // splits
    d_wsize += calculateAlignedBytes(work_items_bytes);         // d_work_Items
    d_wsize += calculateAlignedBytes(workload_info_bytes);      // workload_info
    d_wsize += calculateAlignedBytes(column_samples_bytes);     // column_samples
    d_wsize += calculateAlignedBytes(partition_row_ids_bytes);  // partition row IDs
    d_wsize += packedHistogramWorkspaceSize(max_len_histograms);

    // all nodes in the tree
    h_wsize += calculateAlignedBytes(workload_info_bytes);  // h_workload_info
    h_wsize += calculateAlignedBytes(splits_bytes);         // splits

    return std::make_pair(d_wsize, h_wsize);
  }

  /**
   * @brief assign workspace to the current state
   *
   * @param[in] d_wspace device buffer allocated by the user for the workspace.
   *                     Its size should be at least workspaceSize()
   * @param[in] h_wspace pinned host buffer needed to store the learned nodes
   */
  void assignWorkspace(char* d_wspace, char* h_wspace)
  {
    raft::common::nvtx::range fun_scope(
      "Builder::assignWorkspace @builder.cuh [batched-levelalgo]");
    auto max_batch            = params.max_batch_size;
    size_t max_len_histograms = ML::checked_mul<std::size_t>(
      max_batch, params.max_n_bins, n_blks_for_cols, dataset.num_outputs);
    auto histograms_bytes    = ML::checked_mul<std::size_t>(sizeof(BinT), max_len_histograms);
    auto mutex_bytes         = ML::checked_mul<std::size_t>(sizeof(int), max_batch);
    auto splits_bytes        = ML::checked_mul<std::size_t>(sizeof(SplitT), max_batch);
    auto work_items_bytes    = ML::checked_mul<std::size_t>(sizeof(NodeWorkItem), max_batch);
    auto workload_info_bytes = ML::checked_mul<std::size_t>(sizeof(WorkloadInfo), max_blocks_dimx);
    auto column_samples_bytes =
      ML::checked_mul<std::size_t>(sizeof(std::int64_t), max_batch, dataset.n_sampled_cols);
    auto partition_row_ids_bytes =
      ML::checked_mul<std::size_t>(sizeof(std::int64_t), dataset.n_sampled_rows);
    // device
    n_nodes = reinterpret_cast<std::int64_t*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(std::int64_t));
    histograms = reinterpret_cast<BinT*>(d_wspace);
    d_wspace += calculateAlignedBytes(histograms_bytes);
    mutex = reinterpret_cast<int*>(d_wspace);
    d_wspace += calculateAlignedBytes(mutex_bytes);
    splits = reinterpret_cast<SplitT*>(d_wspace);
    d_wspace += calculateAlignedBytes(splits_bytes);
    d_work_items = reinterpret_cast<NodeWorkItem*>(d_wspace);
    d_wspace += calculateAlignedBytes(work_items_bytes);
    workload_info = reinterpret_cast<WorkloadInfo*>(d_wspace);
    d_wspace += calculateAlignedBytes(workload_info_bytes);
    column_samples = reinterpret_cast<std::int64_t*>(d_wspace);
    d_wspace += calculateAlignedBytes(column_samples_bytes);
    partition_row_ids = reinterpret_cast<std::int64_t*>(d_wspace);
    d_wspace += calculateAlignedBytes(partition_row_ids_bytes);
    packed_histograms = reinterpret_cast<void*>(d_wspace);
    d_wspace += packedHistogramWorkspaceSize(max_len_histograms);

    RAFT_CUDA_TRY(cudaMemsetAsync(mutex, 0, mutex_bytes, builder_stream));

    // host
    h_workload_info = reinterpret_cast<WorkloadInfo*>(h_wspace);
    h_wspace += calculateAlignedBytes(workload_info_bytes);
    h_splits = reinterpret_cast<SplitT*>(h_wspace);
    h_wspace += calculateAlignedBytes(splits_bytes);
  }

  /**
   * @brief trains the tree, builds the nodes
   *
   * @return trained tree structure
   */
  std::shared_ptr<DT::TreeMetaDataNode<DataT, LabelT>> train()
  {
    raft::common::nvtx::range fun_scope("Builder::train @builder.cuh [batched-levelalgo]");
    MLCommon::TimerCPU timer;
    NodeQueue<DataT, LabelT> queue(params,
                                   this->maxNodes(),
                                   dataset.n_sampled_rows,
                                   this->globalSampledRows(),
                                   dataset.num_outputs);
    while (queue.HasWork()) {
      auto work_items                      = queue.Pop();
      auto [splits_host_ptr, splits_count] = doSplit(work_items);
      queue.Push(work_items, splits_host_ptr);
    }
    auto tree = queue.GetTree();
    this->SetLeafPredictions(tree, queue.GetInstanceRanges());
    tree->train_time = timer.getElapsedMilliseconds();
    return tree;
  }

 private:
  std::size_t updateWorkloadInfo(const std::vector<NodeWorkItem>& work_items)
  {
    std::size_t n_blocks_dimx = 0;  // gridDim.x required for histogram construction
    for (std::size_t i = 0; i < work_items.size(); i++) {
      auto item              = work_items[i];
      auto n_blocks_per_node = std::max<std::size_t>(
        raft::ceildiv(item.instances.count, std::size_t{TPB_DEFAULT}), std::size_t{1});

      for (std::size_t b = 0; b < n_blocks_per_node; b++) {
        auto workload_idx             = ML::checked_add<std::size_t>(n_blocks_dimx, b);
        h_workload_info[workload_idx] = {ML::narrow_cast<std::int64_t>(i),
                                         ML::narrow_cast<std::int64_t>(b),
                                         ML::narrow_cast<std::int64_t>(n_blocks_per_node)};
      }
      n_blocks_dimx = ML::checked_add<std::size_t>(n_blocks_dimx, n_blocks_per_node);
    }
    raft::update_device(workload_info, h_workload_info, n_blocks_dimx, builder_stream);
    return n_blocks_dimx;
  }

  std::int64_t globalSampledRows()
  {
    auto global_sampled_rows = static_cast<std::int64_t>(dataset.n_sampled_rows);
    if (!distributed) { return global_sampled_rows; }

    rmm::device_uvector<std::int64_t> d_sampled_rows(1, builder_stream);
    raft::update_device(d_sampled_rows.data(), &global_sampled_rows, 1, builder_stream);
    handle.get_comms().allreduce(
      d_sampled_rows.data(), d_sampled_rows.data(), 1, raft::comms::op_t::SUM, builder_stream);
    ASSERT(handle.get_comms().sync_stream(builder_stream) == raft::comms::status_t::SUCCESS,
           "An error occurred in the distributed RF sampled-row-count all-reduce.");
    raft::update_host(&global_sampled_rows, d_sampled_rows.data(), 1, builder_stream);
    handle.sync_stream(builder_stream);
    return global_sampled_rows;
  }

  auto doSplit(const std::vector<NodeWorkItem>& work_items)
  {
    raft::common::nvtx::range fun_scope("Builder::doSplit @builder.cuh [batched-levelalgo]");
    // start fresh on the number of *new* nodes created in this batch
    RAFT_CUDA_TRY(cudaMemsetAsync(n_nodes, 0, sizeof(std::int64_t), builder_stream));

    const std::int64_t original_n_sampled_cols = dataset.n_sampled_cols;
    ASSERT(original_n_sampled_cols > 0 && original_n_sampled_cols <= dataset.n_cols,
           "n_sampled_cols must be in [1, n_cols]");
    const auto sampling_round_numerator = ML::checked_sub<std::int64_t>(
      ML::checked_add<std::int64_t>(dataset.n_cols, original_n_sampled_cols), 1);
    const auto max_sampling_rounds = ML::narrow_cast<std::size_t>(
      ML::checked_div<std::int64_t>(sampling_round_numerator, original_n_sampled_cols));
    // The final split chosen for each original work item. Nodes that need
    // additional feature samples are compacted in active_items, so successful
    // splits must be copied back to their original batch position.
    std::vector<SplitT> final_splits(work_items.size());
    // Current retry batch. It starts as the full batch and shrinks to only
    // nodes whose sampled features did not produce a valid split.
    std::vector<NodeWorkItem> active_items(work_items);
    // active_items[i] maps back to the corresponding index in the original
    // work_items/final_splits arrays.
    std::vector<std::size_t> active_to_original(work_items.size());
    for (std::size_t i = 0; i < active_to_original.size(); ++i) {
      active_to_original[i] = i;
    }

    // Match sklearn's behavior of searching beyond max_features when the
    // sampled features do not yield a valid split.
    for (std::size_t round = 0; !active_items.empty() && round < max_sampling_rounds; ++round) {
      auto sample_offset     = ML::checked_mul<std::int64_t>(ML::narrow_cast<std::int64_t>(round),
                                                         original_n_sampled_cols);
      dataset.n_sampled_cols = std::min(
        original_n_sampled_cols, ML::checked_sub<std::int64_t>(dataset.n_cols, sample_offset));
      computeBestSplits(active_items, seed, sample_offset);

      std::vector<NodeWorkItem> retry_items;
      std::vector<std::size_t> retry_to_original;
      for (std::size_t i = 0; i < active_items.size(); ++i) {
        const auto original_idx    = active_to_original[i];
        final_splits[original_idx] = h_splits[i];
        if (!h_splits[i].IsValid()) {
          retry_items.push_back(active_items[i]);
          retry_to_original.push_back(original_idx);
        }
      }

      if (round + 1 >= max_sampling_rounds) { break; }
      active_items       = std::move(retry_items);
      active_to_original = std::move(retry_to_original);
    }
    dataset.n_sampled_cols = original_n_sampled_cols;

    // Partition samples once, using the valid split found for each node. Nodes
    // still without a valid split after all features have been visited remain leaves.
    auto split_copy_bytes = ML::checked_mul<std::size_t>(sizeof(SplitT), work_items.size());
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      splits, final_splits.data(), split_copy_bytes, cudaMemcpyHostToDevice, builder_stream));
    raft::update_device(d_work_items, work_items.data(), work_items.size(), builder_stream);
    const auto n_partition_blocks = this->updateWorkloadInfo(work_items);
    raft::common::nvtx::push_range("nodeSplitKernel @builder.cuh [batched-levelalgo]");
    launchNodeSplitKernel<DataT, LabelT, TPB_DEFAULT>(dataset,
                                                      d_work_items,
                                                      splits,
                                                      workload_info,
                                                      n_partition_blocks,
                                                      work_items.size(),
                                                      partition_row_ids,
                                                      builder_stream);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    raft::common::nvtx::pop_range();
    raft::update_host(h_splits, splits, work_items.size(), builder_stream);
    handle.sync_stream(builder_stream);
    return std::make_tuple(h_splits, work_items.size());
  }

  void computeBestSplits(const std::vector<NodeWorkItem>& work_items,
                         uint64_t sampling_seed,
                         std::int64_t sample_offset)
  {
    initSplit<DataT, TPB_DEFAULT>(splits, work_items.size(), builder_stream);
    auto mutex_bytes = ML::checked_mul<std::size_t>(sizeof(int), params.max_batch_size);
    RAFT_CUDA_TRY(cudaMemsetAsync(mutex, 0, mutex_bytes, builder_stream));
    raft::update_device(d_work_items, work_items.data(), work_items.size(), builder_stream);
    auto n_blocks_dimx     = this->updateWorkloadInfo(work_items);
    auto split_smem_config = computeSharedMemoryConfig();

    sampleFeatures(work_items, sampling_seed, sample_offset);

    for (std::int64_t c = 0; c < dataset.n_sampled_cols; c += n_blks_for_cols) {
      computeSplit(c, n_blocks_dimx, work_items.size(), split_smem_config);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }
    raft::update_host(h_splits, splits, work_items.size(), builder_stream);
    handle.sync_stream(builder_stream);
  }

  void sampleFeatures(const std::vector<NodeWorkItem>& work_items,
                      uint64_t sampling_seed,
                      std::int64_t sample_offset)
  {
    raft::common::nvtx::range fun_scope("feature-sampling");
    sample_features(column_samples,
                    d_work_items,
                    work_items.size(),
                    treeid,
                    sampling_seed,
                    sample_offset,
                    dataset.n_cols,
                    dataset.n_sampled_cols,
                    builder_stream);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  SharedMemoryConfig computeSharedMemoryConfig() const
  {
    // Dynamic shared memory for the histogram fast path: histogram, copied quantiles, and
    // alignment padding for the kernel's shared-memory layout.
    auto shared_histogram_size =
      ML::checked_mul<std::size_t>(params.max_n_bins, dataset.num_outputs, sizeof(BinT));
    auto shared_quantiles_size = ML::checked_mul<std::size_t>(params.max_n_bins, sizeof(DataT));
    auto histogram_dynamic_smem_size =
      ML::checked_add<std::size_t>(shared_histogram_size, shared_quantiles_size);
    auto histogram_alignment_smem_size = ML::checked_add<std::size_t>(sizeof(BinT), sizeof(DataT));
    histogram_dynamic_smem_size =
      ML::checked_add<std::size_t>(histogram_dynamic_smem_size, histogram_alignment_smem_size);
    auto cdf_scan_smem_size = sizeof(typename cub::BlockScan<BinT, TPB_DEFAULT>::TempStorage);
    auto split_scratch_smem_size =
      ML::checked_mul<std::size_t>(raft::ceildiv(TPB_DEFAULT, raft::WarpSize), sizeof(SplitT));
    auto split_static_smem_size =
      ML::checked_add<std::size_t>(cdf_scan_smem_size, split_scratch_smem_size);
    auto available_smem = size_t(handle.get_device_properties().sharedMemPerBlock);
    ASSERT(available_smem >= split_static_smem_size,
           "Not enough shared memory for RF split bookkeeping.");

    // Prefer shared memory when it fits and stays small enough for good occupancy;
    // otherwise use the global histogram path to avoid launch failure or slowdown.
    bool use_global_memory_histogram =
      histogram_dynamic_smem_size > available_smem || split_static_smem_size > available_smem ||
      histogram_dynamic_smem_size > tunable_split_histogram_dynamic_smem_limit_bytes;

    return {use_global_memory_histogram,
            use_global_memory_histogram ? 0 : histogram_dynamic_smem_size};
  }

  void allReduceHistograms(BinT* histograms_to_reduce, std::size_t len_histograms)
  {
    auto const& comm  = handle.get_comms();
    auto* packed      = reinterpret_cast<double*>(packed_histograms);
    auto packed_count = ML::checked_mul<std::size_t>(reduction_buffer_size_v<BinT>, len_histograms);

    packHistograms(histograms_to_reduce, packed, len_histograms, builder_stream);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    comm.allreduce(packed, packed, packed_count, raft::comms::op_t::SUM, builder_stream);
    ASSERT(comm.sync_stream(builder_stream) == raft::comms::status_t::SUCCESS,
           "An error occurred in the distributed RF histogram all-reduce.");

    unpackHistograms(packed, histograms_to_reduce, len_histograms, builder_stream);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  void computeSplit(std::int64_t col,
                    size_t n_blocks_dimx,
                    size_t n_work_items,
                    const SharedMemoryConfig& split_smem_config)
  {
    // if no instances to split, return
    if (n_blocks_dimx == 0) return;
    raft::common::nvtx::range fun_scope("Builder::computeSplit @builder.cuh [batched-levelalgo]");
    auto n_bins    = params.max_n_bins;
    auto n_classes = dataset.num_outputs;
    // if columns left to be processed lesser than `n_blks_for_cols`, shrink the blocks along dimy
    auto remaining_sampled_cols = dataset.n_sampled_cols - col;
    auto n_blocks_dimy          = n_blks_for_cols;
    if (remaining_sampled_cols < n_blocks_dimy) {
      n_blocks_dimy = ML::narrow_cast<int>(remaining_sampled_cols);
    }
    dim3 histogram_grid(ML::narrow_cast<ML::cuda_launch_t>(n_blocks_dimx),
                        ML::narrow_cast<ML::cuda_launch_t>(n_blocks_dimy),
                        1);
    dim3 split_grid(ML::narrow_cast<ML::cuda_launch_t>(n_work_items),
                    ML::narrow_cast<ML::cuda_launch_t>(n_blocks_dimy),
                    1);
    auto len_histograms =
      ML::checked_mul<std::size_t>(n_bins, n_classes, n_blocks_dimy, n_work_items);
    auto histograms_bytes = ML::checked_mul<std::size_t>(sizeof(BinT), len_histograms);
    RAFT_CUDA_TRY(cudaMemsetAsync(histograms, 0, histograms_bytes, builder_stream));
    // create the objective function object
    ObjectiveT objective(dataset.num_outputs,
                         params.min_samples_leaf,
                         params.split_criterion,
                         params.min_impurity_decrease);
    raft::common::nvtx::range kernel_scope("computeSplitKernels @builder.cuh [batched-levelalgo]");
    launchBuildHistogramsKernel<DataT, LabelT, TPB_DEFAULT, ObjectiveT>(histograms,
                                                                        params.max_n_bins,
                                                                        dataset,
                                                                        quantiles,
                                                                        d_work_items,
                                                                        col,
                                                                        column_samples,
                                                                        objective,
                                                                        workload_info,
                                                                        histogram_grid,
                                                                        split_smem_config,
                                                                        builder_stream);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    // Distributed RF must aggregate per-rank histograms before split scoring.
    // The split kernel then sees the same global CDF histogram on every rank.
    if (distributed) { allReduceHistograms(histograms, len_histograms); }

    launchFindBestSplitsKernel<DataT, LabelT, TPB_DEFAULT, ObjectiveT>(histograms,
                                                                       params.max_n_bins,
                                                                       dataset,
                                                                       quantiles,
                                                                       col,
                                                                       column_samples,
                                                                       mutex,
                                                                       splits,
                                                                       objective,
                                                                       split_grid,
                                                                       builder_stream);
  }

  // Set the leaf value predictions in batch
  void SetLeafPredictions(std::shared_ptr<DT::TreeMetaDataNode<DataT, LabelT>> tree,
                          const std::vector<InstanceRange>& instance_ranges)
  {
    auto vector_leaf_size =
      ML::checked_mul<std::size_t>(tree->sparsetree.size(), dataset.num_outputs);
    tree->vector_leaf.resize(vector_leaf_size);
    ASSERT(tree->sparsetree.size() == instance_ranges.size(),
           "Expected instance range for each node");
    // Reuse the split histogram and packed reduction workspaces for leaf statistics. Cap the
    // number of nodes so each leaf batch fits in those workspaces.
    auto max_leaf_nodes_in_workspace =
      ML::checked_mul<std::size_t>(params.max_batch_size, params.max_n_bins, n_blks_for_cols);
    std::size_t max_batch_size =
      std::min(std::size_t{100000}, std::min(tree->sparsetree.size(), max_leaf_nodes_in_workspace));
    auto max_leaf_values = ML::checked_mul<std::size_t>(max_batch_size, dataset.num_outputs);
    rmm::device_uvector<NodeT> d_tree(max_batch_size, builder_stream);
    rmm::device_uvector<InstanceRange> d_instance_ranges(max_batch_size, builder_stream);
    rmm::device_uvector<DataT> d_leaves(max_leaf_values, builder_stream);

    ObjectiveT objective(dataset.num_outputs, params.min_samples_leaf, params.split_criterion);
    for (std::size_t batch_begin = 0; batch_begin < tree->sparsetree.size();
         batch_begin += max_batch_size) {
      std::size_t batch_size = min(max_batch_size, tree->sparsetree.size() - batch_begin);
      raft::update_device(
        d_tree.data(), tree->sparsetree.data() + batch_begin, batch_size, builder_stream);
      raft::update_device(
        d_instance_ranges.data(), instance_ranges.data() + batch_begin, batch_size, builder_stream);

      auto leaf_histogram_count = ML::checked_mul<std::size_t>(batch_size, dataset.num_outputs);
      auto leaf_histogram_bytes = ML::checked_mul<std::size_t>(sizeof(BinT), leaf_histogram_count);
      auto leaf_batch_size      = ML::narrow_cast<int>(batch_size);
      RAFT_CUDA_TRY(cudaMemsetAsync(histograms, 0, leaf_histogram_bytes, builder_stream));
      size_t smem_size = ML::checked_mul<std::size_t>(sizeof(BinT), dataset.num_outputs);
      launchBuildLeafHistogramsKernel(objective,
                                      dataset,
                                      d_tree.data(),
                                      d_instance_ranges.data(),
                                      histograms,
                                      leaf_batch_size,
                                      smem_size,
                                      builder_stream);
      RAFT_CUDA_TRY(cudaPeekAtLastError());

      if (distributed) { allReduceHistograms(histograms, leaf_histogram_count); }

      launchFinalizeLeafKernel<ObjectiveT, DataT>(
        histograms, d_leaves.data(), dataset.num_outputs, leaf_batch_size, builder_stream);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
      auto leaf_offset = ML::checked_mul<std::size_t>(batch_begin, dataset.num_outputs);
      auto leaf_count  = ML::checked_mul<std::size_t>(batch_size, dataset.num_outputs);
      raft::update_host(
        tree->vector_leaf.data() + leaf_offset, d_leaves.data(), leaf_count, builder_stream);
    }
  }
};  // end Builder

}  // namespace DT
}  // namespace ML
