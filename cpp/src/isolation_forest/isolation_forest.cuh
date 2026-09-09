/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "isolation_tree_builder.cuh"

#include <cuml/ensemble/isolation_forest.hpp>

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/transform.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <limits>

namespace ML {

inline int compute_global_max_nodes_per_tree(int max_depth, int max_samples)
{
  ASSERT(max_depth >= 0, "max_depth must be non-negative, got %d", max_depth);
  ASSERT(max_samples > 0, "max_samples must be positive, got %d", max_samples);

  int64_t sample_bound = 2LL * static_cast<int64_t>(max_samples) - 1LL;
  int64_t depth_bound  = sample_bound;
  if (max_depth < 30) { depth_bound = (1LL << (max_depth + 1)) - 1LL; }

  int64_t bounded = std::min(sample_bound, depth_bound);
  ASSERT(bounded <= static_cast<int64_t>(std::numeric_limits<int>::max()),
         "Global-memory Isolation Forest node capacity exceeds int range.");
  return static_cast<int>(bounded);
}

template <class T>
class IsolationForest {
 private:
  IF_params params;

  void error_checking(const T* input, size_t n_rows, int n_cols) const
  {
    ASSERT((n_rows > 0), "Invalid n_rows %zu", n_rows);
    ASSERT((n_cols > 0), "Invalid n_cols %d", n_cols);
    ASSERT((params.n_estimators > 0), "n_estimators must be > 0, got %d", params.n_estimators);
    ASSERT(IsolationTree::is_dev_ptr(input), "IF Error: Expected input to be a GPU pointer");
  }

 public:
  IsolationForest(const IF_params& cfg_params) : params(cfg_params) {}

  /**
   * @brief Fit the Isolation Forest model.
   *
   * @param[in] handle raft::handle_t
   * @param[in] input Training data (n_rows x n_cols) in column-major format. Device pointer.
   * @param[in] n_rows Number of training samples
   * @param[in] n_cols Number of features
   * @param[out] model IsolationForestModel to populate
   */
  void fit(const raft::handle_t& handle,
           const T* input,
           size_t n_rows,
           int n_cols,
           IsolationForestModel<T>* model)
  {
    raft::common::nvtx::range fun_scope("IsolationForest::fit @isolation_forest.cuh");
    this->error_checking(input, n_rows, n_cols);

    int n_sampled_rows = std::min(params.max_samples,
                                  static_cast<int>(std::min(n_rows, static_cast<size_t>(INT_MAX))));
    int max_depth      = params.max_depth;
    if (max_depth <= 0) {
      max_depth = static_cast<int>(std::ceil(std::log2(static_cast<double>(n_sampled_rows))));
      max_depth = std::max(max_depth, 1);
    }

    int n_sampled_features = params.max_features;
    if (n_sampled_features <= 0 || n_sampled_features > n_cols) { n_sampled_features = n_cols; }

    model->params              = params;
    model->n_features          = n_cols;
    model->n_features_per_tree = n_sampled_features;
    model->n_samples_per_tree  = n_sampled_rows;
    model->c_normalization     = compute_c_normalization(n_sampled_rows);

    auto stream               = handle.get_stream();
    model->max_nodes_per_tree = compute_global_max_nodes_per_tree(max_depth, n_sampled_rows);
    size_t total_nodes = static_cast<size_t>(params.n_estimators) * model->max_nodes_per_tree;
    ASSERT(total_nodes <= static_cast<size_t>(std::numeric_limits<int>::max()),
           "Isolation Forest node offsets exceed int range.");
    model->global_nodes =
      rmm::device_buffer(total_nodes * sizeof(IsolationTree::IFNode<T>), stream);
    model->global_tree_offsets   = rmm::device_buffer(params.n_estimators * sizeof(int), stream);
    model->global_tree_n_nodes   = rmm::device_buffer(params.n_estimators * sizeof(int), stream);
    model->global_tree_max_depth = rmm::device_buffer(params.n_estimators * sizeof(int), stream);
    if (n_sampled_features < n_cols) {
      model->global_feature_indices = rmm::device_buffer(
        static_cast<size_t>(params.n_estimators) * n_sampled_features * sizeof(int), stream);
    } else {
      model->global_feature_indices = rmm::device_buffer{};
    }

    IsolationTree::build_isolation_forest_global(
      handle,
      input,
      n_rows,
      n_cols,
      params.n_estimators,
      n_sampled_rows,
      n_sampled_features,
      max_depth,
      model->max_nodes_per_tree,
      params.bootstrap,
      params.seed,
      n_sampled_features < n_cols ? static_cast<int*>(model->global_feature_indices.data())
                                  : nullptr,
      static_cast<IsolationTree::IFNode<T>*>(model->global_nodes.data()),
      static_cast<int*>(model->global_tree_offsets.data()),
      static_cast<int*>(model->global_tree_n_nodes.data()),
      static_cast<int*>(model->global_tree_max_depth.data()));

    handle.sync_stream();
  }

  /** @brief Compute average path lengths for each sample. Input must be row-major. */
  void compute_path_lengths(const raft::handle_t& handle,
                            const IsolationForestModel<T>* model,
                            const T* input,
                            size_t n_rows,
                            int n_cols,
                            T* avg_path_lengths) const
  {
    raft::common::nvtx::range fun_scope("IF::compute_path_lengths @isolation_forest.cuh");
    cudaStream_t stream = handle.get_stream().get();

    int threads   = 256;
    size_t blocks = (n_rows + threads - 1) / threads;

    auto* nodes        = static_cast<const IsolationTree::IFNode<T>*>(model->global_nodes.data());
    auto* tree_offsets = static_cast<const int*>(model->global_tree_offsets.data());
    IsolationTree::compute_path_lengths_global_kernel<T><<<blocks, threads, 0, stream>>>(
      input, n_rows, n_cols, nodes, tree_offsets, params.n_estimators, avg_path_lengths);

    RAFT_CUDA_TRY(cudaGetLastError());
    handle.sync_stream(stream);
  }

  /** @brief Compute anomaly scores: score = 2^(-avg_path_length / c(n)). */
  void compute_anomaly_scores(const raft::handle_t& handle,
                              const IsolationForestModel<T>* model,
                              const T* avg_path_lengths,
                              size_t n_rows,
                              T* scores) const
  {
    cudaStream_t stream = handle.get_stream().get();
    T c_n               = static_cast<T>(model->c_normalization);

    if (c_n <= T(0)) {
      thrust::fill(rmm::exec_policy(stream), scores, scores + n_rows, T(0.5));
    } else {
      thrust::transform(
        rmm::exec_policy(stream),
        avg_path_lengths,
        avg_path_lengths + n_rows,
        scores,
        [c_n] __device__(T path_length) { return std::pow(T(2), -path_length / c_n); });
    }
    handle.sync_stream(stream);
  }
};

template class IsolationForest<float>;
template class IsolationForest<double>;

template <typename T>
CompactIFForest<T> get_compact_trees(const raft::handle_t& handle,
                                     const IsolationForestModel<T>* model)
{
  int n_trees = model->params.n_estimators;

  CompactIFForest<T> result;
  std::vector<IsolationTree::IFNode<T>> raw_nodes;

  auto* d_nodes          = static_cast<const IsolationTree::IFNode<T>*>(model->global_nodes.data());
  auto* d_tree_n_nodes   = static_cast<const int*>(model->global_tree_n_nodes.data());
  auto* d_tree_max_depth = static_cast<const int*>(model->global_tree_max_depth.data());
  IsolationTree::compact_global_isolation_forest<T>(handle,
                                                    d_nodes,
                                                    d_tree_n_nodes,
                                                    d_tree_max_depth,
                                                    n_trees,
                                                    model->max_nodes_per_tree,
                                                    raw_nodes,
                                                    result.tree_offsets,
                                                    result.tree_n_nodes,
                                                    result.tree_max_depth);

  result.nodes.resize(raw_nodes.size());
  static_assert(sizeof(IFNodeCompact<T>) == sizeof(IsolationTree::IFNode<T>),
                "IFNodeCompact and IFNode must have identical layout");
  std::memcpy(result.nodes.data(), raw_nodes.data(), raw_nodes.size() * sizeof(IFNodeCompact<T>));

  return result;
}

template CUML_EXPORT CompactIFForest<float> get_compact_trees<float>(
  const raft::handle_t&, const IsolationForestModel<float>*);
template CUML_EXPORT CompactIFForest<double> get_compact_trees<double>(
  const raft::handle_t&, const IsolationForestModel<double>*);

}  // namespace ML
