/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Shared device types and helpers for the batched Isolation Forest tree builder.
 *
 * Key design decisions:
 * - All trees built in a single kernel launch (1 block per tree)
 * - Subsamples gathered into contiguous buffer to avoid scattered reads
 * - Tree built iteratively using a stack (no recursion in CUDA)
 * - Random splits: pick random feature, then random threshold in [min, max]
 * - Leaf nodes store pre-computed path lengths (depth + c(n)) so that
 *   inference is a simple tree traversal with no per-leaf computation
 */

#pragma once

#include <cuml/common/utils.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda_runtime.h>
#include <thrust/transform.h>

#include <curand_kernel.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace ML {
namespace IsolationTree {

struct StackEntry {
  int node_idx;
  int start_idx;
  int end_idx;
  int depth;
};

/**
 * @brief Compute c(n) = 2H(n-1) - 2(n-1)/n, the expected path length in an
 * unsuccessful BST search (used for adjusting isolation depth at leaves).
 */
template <typename T>
__device__ __forceinline__ T compute_c_n(int n_samples)
{
  if (n_samples <= 1) return T(0);
  if (n_samples == 2) return T(1);
  T n = static_cast<T>(n_samples);
  return T(2) * (log(n - T(1)) + T(0.5772156649015329)) - T(2) * (n - T(1)) / n;
}

/**
 * @brief Tree node structure.
 * Internal nodes: feature_idx >= 0, threshold is split value
 * Leaf nodes: feature_idx = -1, threshold stores pre-computed path length
 *             (depth + c(n_samples)) for direct use by inference
 */
template <typename T>
struct IFNode {
  int feature_idx;
  T threshold;
  int left_child;
  int right_child;
};

/** @brief Check if pointer is on GPU (for input validation). */
inline bool is_dev_ptr(const void* ptr)
{
  cudaPointerAttributes attr;
  auto err = cudaPointerGetAttributes(&attr, ptr);
  if (err != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged;
}

__device__ __forceinline__ uint64_t curand_u64(curandState* rng_state)
{
  return (static_cast<uint64_t>(curand(rng_state)) << 32) | curand(rng_state);
}

__device__ __forceinline__ size_t sample_bounded(curandState* rng_state, size_t bound)
{
  if (bound <= 1) return 0;

  constexpr uint64_t max_uint64 = 0xffffffffffffffffULL;
  uint64_t limit                = max_uint64 - (max_uint64 % static_cast<uint64_t>(bound));
  uint64_t value                = curand_u64(rng_state);
  while (value >= limit) {
    value = curand_u64(rng_state);
  }
  return static_cast<size_t>(value % static_cast<uint64_t>(bound));
}

__device__ bool contains_sample(const size_t* samples, int n_samples, size_t candidate)
{
  for (int i = 0; i < n_samples; ++i) {
    if (samples[i] == candidate) return true;
  }
  return false;
}

__device__ bool contains_int_sample(const int* samples, int n_samples, int candidate)
{
  for (int i = 0; i < n_samples; ++i) {
    if (samples[i] == candidate) return true;
  }
  return false;
}

/**
 * @brief Build one isolation tree into global-memory node storage.
 *
 * Stack and partition indices are supplied by global RMM buffers so depth and
 * max_samples are not constrained by fixed per-block shared-memory storage.
 */
template <typename T>
__device__ void build_tree_iterative_global(const T* __restrict__ local_data,
                                            int n_samples,
                                            int n_cols,
                                            const int* feature_indices,
                                            int max_depth,
                                            int max_nodes_per_tree,
                                            curandState* rng_state,
                                            IFNode<T>* nodes,
                                            int* n_nodes_out,
                                            int* max_depth_out,
                                            int* work_indices,
                                            StackEntry* stack)
{
  int tid = threadIdx.x;
  for (int i = tid; i < n_samples; i += blockDim.x) {
    work_indices[i] = i;
  }
  __syncthreads();

  if (tid == 0) {
    int n_nodes            = 1;
    int observed_max_depth = 0;
    int stack_top          = 0;
    stack[stack_top++]     = {0, 0, n_samples, 0};

    while (stack_top > 0) {
      StackEntry entry   = stack[--stack_top];
      int node_idx       = entry.node_idx;
      int start          = entry.start_idx;
      int end            = entry.end_idx;
      int depth          = entry.depth;
      int n_node_samples = end - start;
      observed_max_depth = observed_max_depth > depth ? observed_max_depth : depth;

      if (node_idx >= max_nodes_per_tree) { continue; }

      // Stopping condition: max depth, isolated sample, or exhausted capacity.
      if (depth >= max_depth || n_node_samples <= 1 || n_nodes + 2 > max_nodes_per_tree) {
        T path_length   = static_cast<T>(depth) + compute_c_n<T>(n_node_samples);
        nodes[node_idx] = {-1, path_length, -1, -1};
        continue;
      }

      // Try every feature, starting from a random offset, before concluding
      // that this node cannot be split. This avoids prematurely stopping on
      // sparse or one-hot data when the first random feature is constant.
      int local_feature = -1;
      T min_val         = T(0);
      T max_val         = T(0);
      int feature_start = static_cast<int>(sample_bounded(rng_state, static_cast<size_t>(n_cols)));
      for (int attempt = 0; attempt < n_cols; ++attempt) {
        int candidate   = (feature_start + attempt) % n_cols;
        T candidate_min = local_data[work_indices[start] * n_cols + candidate];
        T candidate_max = candidate_min;
        for (int i = start + 1; i < end; ++i) {
          T val = local_data[work_indices[i] * n_cols + candidate];
          if (val < candidate_min) candidate_min = val;
          if (val > candidate_max) candidate_max = val;
        }
        if (candidate_min < candidate_max) {
          local_feature = candidate;
          min_val       = candidate_min;
          max_val       = candidate_max;
          break;
        }
      }

      if (local_feature < 0) {
        T path_length   = static_cast<T>(depth) + compute_c_n<T>(n_node_samples);
        nodes[node_idx] = {-1, path_length, -1, -1};
        continue;
      }

      int original_feature =
        feature_indices == nullptr ? local_feature : feature_indices[local_feature];
      float rand_frac = curand_uniform(rng_state);
      T threshold     = min_val + static_cast<T>(rand_frac) * (max_val - min_val);

      int left_end = start;
      for (int i = start; i < end; i++) {
        T val = local_data[work_indices[i] * n_cols + local_feature];
        if (val < threshold) {
          int tmp                = work_indices[left_end];
          work_indices[left_end] = work_indices[i];
          work_indices[i]        = tmp;
          left_end++;
        }
      }

      if (left_end == start || left_end == end) {
        // Numerical rounding can move the random threshold onto an endpoint.
        // Repartition with max_val so the stored split and training partition
        // remain consistent. Since min_val < max_val, both children are nonempty.
        threshold = max_val;
        left_end  = start;
        for (int i = start; i < end; ++i) {
          T val = local_data[work_indices[i] * n_cols + local_feature];
          if (val < threshold) {
            int tmp                = work_indices[left_end];
            work_indices[left_end] = work_indices[i];
            work_indices[i]        = tmp;
            ++left_end;
          }
        }
      }

      int left_child  = n_nodes;
      int right_child = n_nodes + 1;
      n_nodes += 2;

      nodes[node_idx].feature_idx = original_feature;
      nodes[node_idx].threshold   = threshold;
      nodes[node_idx].left_child  = left_child;
      nodes[node_idx].right_child = right_child;

      stack[stack_top++] = {right_child, left_end, end, depth + 1};
      stack[stack_top++] = {left_child, start, left_end, depth + 1};
    }

    *n_nodes_out   = n_nodes;
    *max_depth_out = observed_max_depth;
  }
}

template <typename T>
CUML_KERNEL void build_isolation_trees_global_kernel(const T* __restrict__ data,
                                                     size_t n_rows,
                                                     int n_cols,
                                                     int n_trees,
                                                     int max_samples,
                                                     int max_features,
                                                     int max_depth,
                                                     int max_nodes_per_tree,
                                                     bool bootstrap,
                                                     uint64_t seed,
                                                     int* __restrict__ feature_indices,
                                                     IFNode<T>* __restrict__ nodes,
                                                     int* __restrict__ tree_offsets,
                                                     int* __restrict__ tree_n_nodes,
                                                     int* __restrict__ tree_max_depth,
                                                     T* __restrict__ subsample_buffer,
                                                     size_t* __restrict__ sample_indices,
                                                     int* __restrict__ work_indices,
                                                     StackEntry* __restrict__ stack)
{
  int tree_id = blockIdx.x;
  if (tree_id >= n_trees) return;

  curandState rng_state;
  curand_init(seed, tree_id, 0, &rng_state);

  size_t tree_offset = static_cast<size_t>(tree_id) * max_nodes_per_tree;
  if (threadIdx.x == 0) { tree_offsets[tree_id] = static_cast<int>(tree_offset); }

  T* local_data = subsample_buffer + static_cast<size_t>(tree_id) * max_samples * max_features;
  size_t* tree_sample_indices = sample_indices + static_cast<size_t>(tree_id) * max_samples;
  int* tree_feature_indices   = feature_indices == nullptr
                                  ? nullptr
                                  : feature_indices + static_cast<size_t>(tree_id) * max_features;
  int* tree_work_indices      = work_indices + static_cast<size_t>(tree_id) * max_samples;
  StackEntry* tree_stack      = stack + static_cast<size_t>(tree_id) * max_nodes_per_tree;
  IFNode<T>* tree_nodes       = nodes + tree_offset;

  // Thread 0 samples source rows using sklearn IsolationForest semantics:
  // bootstrap=True samples with replacement; bootstrap=False samples without
  // replacement. Bounded rejection sampling avoids modulo bias.
  if (threadIdx.x == 0) {
    if (bootstrap) {
      for (int i = 0; i < max_samples; i++) {
        tree_sample_indices[i] = sample_bounded(&rng_state, n_rows);
      }
    } else {
      size_t start = n_rows - static_cast<size_t>(max_samples);
      for (int i = 0; i < max_samples; i++) {
        size_t j               = start + static_cast<size_t>(i);
        size_t t               = sample_bounded(&rng_state, j + 1);
        tree_sample_indices[i] = contains_sample(tree_sample_indices, i, t) ? j : t;
      }
    }

    if (tree_feature_indices != nullptr) {
      size_t start = static_cast<size_t>(n_cols - max_features);
      for (int i = 0; i < max_features; i++) {
        size_t j = start + static_cast<size_t>(i);
        int t    = static_cast<int>(sample_bounded(&rng_state, j + 1));
        tree_feature_indices[i] =
          contains_int_sample(tree_feature_indices, i, t) ? static_cast<int>(j) : t;
      }
    }
  }
  __syncthreads();

  int tid = threadIdx.x;
  for (int s = 0; s < max_samples; s++) {
    size_t src_row = tree_sample_indices[s];
    for (int f = tid; f < max_features; f += blockDim.x) {
      int src_col = tree_feature_indices == nullptr ? f : tree_feature_indices[f];
      local_data[s * max_features + f] = data[src_row + static_cast<size_t>(src_col) * n_rows];
    }
  }
  __syncthreads();

  build_tree_iterative_global(local_data,
                              max_samples,
                              max_features,
                              tree_feature_indices,
                              max_depth,
                              max_nodes_per_tree,
                              &rng_state,
                              tree_nodes,
                              tree_n_nodes + tree_id,
                              tree_max_depth + tree_id,
                              tree_work_indices,
                              tree_stack);
}

template <typename T>
__device__ T traverse_global_tree(const IFNode<T>* tree_nodes, const T* sample, int n_cols)
{
  int node_idx = 0;

  while (true) {
    const IFNode<T>& node = tree_nodes[node_idx];

    if (node.feature_idx < 0) { return node.threshold; }

    T val    = sample[node.feature_idx];
    node_idx = (val < node.threshold) ? node.left_child : node.right_child;
  }
}

template <typename T>
CUML_KERNEL void compute_path_lengths_global_kernel(const T* __restrict__ data,
                                                    size_t n_samples,
                                                    int n_cols,
                                                    const IFNode<T>* __restrict__ nodes,
                                                    const int* __restrict__ tree_offsets,
                                                    int n_trees,
                                                    T* __restrict__ path_lengths)
{
  size_t sample_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (sample_idx >= n_samples) return;

  const T* sample = data + sample_idx * n_cols;

  T total_path = T(0);
  for (int t = 0; t < n_trees; t++) {
    total_path += traverse_global_tree(nodes + tree_offsets[t], sample, n_cols);
  }

  path_lengths[sample_idx] = (n_trees > 0) ? total_path / static_cast<T>(n_trees) : T(0);
}

template <typename T>
void build_isolation_forest_global(const raft::handle_t& handle,
                                   const T* data,
                                   size_t n_rows,
                                   int n_cols,
                                   int n_trees,
                                   int max_samples,
                                   int max_features,
                                   int max_depth,
                                   int max_nodes_per_tree,
                                   bool bootstrap,
                                   uint64_t seed,
                                   int* feature_indices,
                                   IFNode<T>* nodes,
                                   int* tree_offsets,
                                   int* tree_n_nodes,
                                   int* tree_max_depth)
{
  auto stream = handle.get_stream().get();

  size_t subsample_buffer_size = static_cast<size_t>(n_trees) * max_samples * max_features;
  rmm::device_uvector<T> subsample_buffer(subsample_buffer_size, stream);
  rmm::device_uvector<size_t> sample_indices(static_cast<size_t>(n_trees) * max_samples, stream);
  rmm::device_uvector<int> work_indices(static_cast<size_t>(n_trees) * max_samples, stream);
  rmm::device_uvector<StackEntry> stack(static_cast<size_t>(n_trees) * max_nodes_per_tree, stream);

  build_isolation_trees_global_kernel<T><<<n_trees, 128, 0, stream>>>(data,
                                                                      n_rows,
                                                                      n_cols,
                                                                      n_trees,
                                                                      max_samples,
                                                                      max_features,
                                                                      max_depth,
                                                                      max_nodes_per_tree,
                                                                      bootstrap,
                                                                      seed,
                                                                      feature_indices,
                                                                      nodes,
                                                                      tree_offsets,
                                                                      tree_n_nodes,
                                                                      tree_max_depth,
                                                                      subsample_buffer.data(),
                                                                      sample_indices.data(),
                                                                      work_indices.data(),
                                                                      stack.data());

  RAFT_CUDA_TRY(cudaGetLastError());
}

template <typename T>
CUML_KERNEL void compact_global_trees_kernel(const IFNode<T>* __restrict__ nodes,
                                             int n_trees,
                                             int max_nodes_per_tree,
                                             const int* __restrict__ tree_n_nodes,
                                             const int* __restrict__ compact_offsets,
                                             IFNode<T>* __restrict__ compact_nodes)
{
  int tree_id = blockIdx.x;
  if (tree_id >= n_trees) return;
  int n          = tree_n_nodes[tree_id];
  size_t src_off = static_cast<size_t>(tree_id) * max_nodes_per_tree;
  int dst_off    = compact_offsets[tree_id];
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    compact_nodes[dst_off + i] = nodes[src_off + i];
  }
}

template <typename T>
void compact_global_isolation_forest(const raft::handle_t& handle,
                                     const IFNode<T>* d_nodes,
                                     const int* d_tree_n_nodes,
                                     const int* d_tree_max_depth,
                                     int n_trees,
                                     int max_nodes_per_tree,
                                     std::vector<IFNode<T>>& h_nodes,
                                     std::vector<int>& h_tree_offsets,
                                     std::vector<int>& h_tree_n_nodes,
                                     std::vector<int>& h_tree_max_depth)
{
  auto stream = handle.get_stream().get();

  h_tree_n_nodes.resize(n_trees);
  h_tree_max_depth.resize(n_trees);
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    h_tree_n_nodes.data(), d_tree_n_nodes, n_trees * sizeof(int), cudaMemcpyDeviceToHost, stream));
  RAFT_CUDA_TRY(cudaMemcpyAsync(h_tree_max_depth.data(),
                                d_tree_max_depth,
                                n_trees * sizeof(int),
                                cudaMemcpyDeviceToHost,
                                stream));
  handle.sync_stream(stream);

  h_tree_offsets.resize(n_trees);
  int total_nodes = 0;
  for (int t = 0; t < n_trees; ++t) {
    h_tree_offsets[t] = total_nodes;
    total_nodes += h_tree_n_nodes[t];
  }

  rmm::device_uvector<int> d_compact_offsets(n_trees, stream);
  RAFT_CUDA_TRY(cudaMemcpyAsync(d_compact_offsets.data(),
                                h_tree_offsets.data(),
                                n_trees * sizeof(int),
                                cudaMemcpyHostToDevice,
                                stream));

  rmm::device_uvector<IFNode<T>> d_compact(total_nodes, stream);
  compact_global_trees_kernel<T><<<n_trees, 128, 0, stream>>>(d_nodes,
                                                              n_trees,
                                                              max_nodes_per_tree,
                                                              d_tree_n_nodes,
                                                              d_compact_offsets.data(),
                                                              d_compact.data());
  RAFT_CUDA_TRY(cudaGetLastError());

  h_nodes.resize(total_nodes);
  RAFT_CUDA_TRY(cudaMemcpyAsync(h_nodes.data(),
                                d_compact.data(),
                                total_nodes * sizeof(IFNode<T>),
                                cudaMemcpyDeviceToHost,
                                stream));
  handle.sync_stream(stream);
}

}  // namespace IsolationTree
}  // namespace ML
