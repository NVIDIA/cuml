/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "isolation_forest.cuh"

#include <cuml/ensemble/isolation_forest.hpp>

#include <raft/core/error.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>

#include <treelite/enum/task_type.h>
#include <treelite/tree.h>

#include <cstdint>
#include <vector>

namespace ML {
namespace tl = treelite;

double compute_c_normalization(int n)
{
  if (n <= 1) return 0.0;
  if (n == 2) return 1.0;

  constexpr double euler_mascheroni = 0.5772156649015329;
  double harmonic_n_minus_1         = std::log(static_cast<double>(n - 1)) + euler_mascheroni;
  return 2.0 * harmonic_n_minus_1 - 2.0 * static_cast<double>(n - 1) / static_cast<double>(n);
}

template <typename T>
tl::Tree<T, T> build_treelite_if_tree(const CompactIFForest<T>& compact, int tree_id)
{
  ASSERT(tree_id >= 0 && tree_id < static_cast<int>(compact.tree_offsets.size()),
         "Invalid isolation forest tree id.");

  int tree_offset = compact.tree_offsets[tree_id];
  int n_nodes     = compact.tree_n_nodes[tree_id];
  ASSERT(n_nodes > 0, "Cannot export an empty isolation tree.");

  tl::Tree<T, T> tl_tree;
  tl_tree.Init();

  std::vector<int> tl_node_ids(n_nodes, -1);
  tl_node_ids[0] = tl_tree.AllocNode();

  std::vector<int> stack{0};
  while (!stack.empty()) {
    int if_node_id = stack.back();
    stack.pop_back();

    const auto& if_node = compact.nodes[tree_offset + if_node_id];
    int tl_node_id      = tl_node_ids[if_node_id];

    if (if_node.feature_idx < 0) {
      tl_tree.SetLeaf(tl_node_id, static_cast<T>(if_node.threshold));
    } else {
      ASSERT(if_node.left_child >= 0 && if_node.left_child < n_nodes,
             "Invalid left child in isolation tree.");
      ASSERT(if_node.right_child >= 0 && if_node.right_child < n_nodes,
             "Invalid right child in isolation tree.");

      int tl_left_child                = tl_tree.AllocNode();
      int tl_right_child               = tl_tree.AllocNode();
      tl_node_ids[if_node.left_child]  = tl_left_child;
      tl_node_ids[if_node.right_child] = tl_right_child;

      tl_tree.SetChildren(tl_node_id, tl_left_child, tl_right_child);
      tl_tree.SetNumericalTest(tl_node_id,
                               if_node.feature_idx,
                               static_cast<T>(if_node.threshold),
                               true,
                               tl::Operator::kLT);

      stack.push_back(if_node.right_child);
      stack.push_back(if_node.left_child);
    }

    tl_tree.SetDataCount(tl_node_id, 0);
  }

  return tl_tree;
}

template <typename T>
void build_treelite_isolation_forest(TreeliteModelHandle* model_handle,
                                     const raft::handle_t& handle,
                                     const IsolationForestModel<T>* forest)
{
  ASSERT(model_handle != nullptr, "Treelite output handle cannot be null.");
  ASSERT(forest != nullptr, "Isolation Forest model cannot be null.");
  ASSERT(forest->params.n_estimators > 0, "Cannot export an empty Isolation Forest.");
  ASSERT(forest->n_features > 0, "Cannot export Isolation Forest with no features.");

  auto compact = get_compact_trees(handle, forest);
  ASSERT(static_cast<int>(compact.tree_offsets.size()) == forest->params.n_estimators,
         "Inconsistent compact Isolation Forest tree offsets.");
  ASSERT(static_cast<int>(compact.tree_n_nodes.size()) == forest->params.n_estimators,
         "Inconsistent compact Isolation Forest tree sizes.");

  auto model                          = tl::Model::Create<T, T>();
  tl::ModelPreset<T, T>& model_preset = std::get<tl::ModelPreset<T, T>>(model->variant_);

  model->task_type           = tl::TaskType::kRegressor;
  model->postprocessor       = "identity";
  model->num_target          = 1;
  model->num_class           = std::vector<std::int32_t>{1};
  model->leaf_vector_shape   = std::vector<std::int32_t>{1, 1};
  model->target_id           = std::vector<std::int32_t>(forest->params.n_estimators, 0);
  model->class_id            = std::vector<std::int32_t>(forest->params.n_estimators, 0);
  model->num_feature         = forest->n_features;
  model->average_tree_output = true;
  model->base_scores         = std::vector<double>{0.0};
  model->SetTreeLimit(forest->params.n_estimators);

#pragma omp parallel for
  for (int i = 0; i < forest->params.n_estimators; ++i) {
    model_preset.trees[i] = build_treelite_if_tree<T>(compact, i);
  }

  *model_handle = static_cast<TreeliteModelHandle>(model.release());
}

void fit(const raft::handle_t& handle,
         IsolationForestF* forest,
         const float* input,
         size_t n_rows,
         int n_cols,
         const IF_params& params,
         rapids_logger::level_enum verbosity)
{
  ML::default_logger().set_level(verbosity);
  IsolationForest<float> if_model(params);
  if_model.fit(handle, input, n_rows, n_cols, forest);
}

void fit(const raft::handle_t& handle,
         IsolationForestD* forest,
         const double* input,
         size_t n_rows,
         int n_cols,
         const IF_params& params,
         rapids_logger::level_enum verbosity)
{
  ML::default_logger().set_level(verbosity);
  IsolationForest<double> if_model(params);
  if_model.fit(handle, input, n_rows, n_cols, forest);
}

void score_samples(const raft::handle_t& handle,
                   const IsolationForestF* forest,
                   const float* input,
                   size_t n_rows,
                   int n_cols,
                   float* scores,
                   rapids_logger::level_enum verbosity)
{
  ML::default_logger().set_level(verbosity);
  IsolationForest<float> if_model(forest->params);

  // Compute average path lengths
  rmm::device_uvector<float> avg_path_lengths(n_rows, handle.get_stream());
  if_model.compute_path_lengths(handle, forest, input, n_rows, n_cols, avg_path_lengths.data());

  // Convert to anomaly scores
  if_model.compute_anomaly_scores(handle, forest, avg_path_lengths.data(), n_rows, scores);
}

void score_samples(const raft::handle_t& handle,
                   const IsolationForestD* forest,
                   const double* input,
                   size_t n_rows,
                   int n_cols,
                   double* scores,
                   rapids_logger::level_enum verbosity)
{
  ML::default_logger().set_level(verbosity);
  IsolationForest<double> if_model(forest->params);

  // Compute average path lengths
  rmm::device_uvector<double> avg_path_lengths(n_rows, handle.get_stream());
  if_model.compute_path_lengths(handle, forest, input, n_rows, n_cols, avg_path_lengths.data());

  // Convert to anomaly scores
  if_model.compute_anomaly_scores(handle, forest, avg_path_lengths.data(), n_rows, scores);
}

void predict(const raft::handle_t& handle,
             const IsolationForestF* forest,
             const float* input,
             size_t n_rows,
             int n_cols,
             int* predictions,
             float threshold,
             rapids_logger::level_enum verbosity)
{
  ML::default_logger().set_level(verbosity);
  cudaStream_t stream = handle.get_stream();

  // First compute anomaly scores
  rmm::device_uvector<float> scores(n_rows, stream);
  score_samples(handle, forest, input, n_rows, n_cols, scores.data(), verbosity);

  // Convert scores to predictions: 1 for anomaly (score > threshold), -1 for normal
  thrust::transform(rmm::exec_policy(stream),
                    scores.data(),
                    scores.data() + n_rows,
                    predictions,
                    [threshold] __device__(float score) { return score > threshold ? 1 : -1; });

  handle.sync_stream(stream);
}

void predict(const raft::handle_t& handle,
             const IsolationForestD* forest,
             const double* input,
             size_t n_rows,
             int n_cols,
             int* predictions,
             double threshold,
             rapids_logger::level_enum verbosity)
{
  ML::default_logger().set_level(verbosity);
  cudaStream_t stream = handle.get_stream();

  // First compute anomaly scores
  rmm::device_uvector<double> scores(n_rows, stream);
  score_samples(handle, forest, input, n_rows, n_cols, scores.data(), verbosity);

  // Convert scores to predictions: 1 for anomaly (score > threshold), -1 for normal
  thrust::transform(rmm::exec_policy(stream),
                    scores.data(),
                    scores.data() + n_rows,
                    predictions,
                    [threshold] __device__(double score) { return score > threshold ? 1 : -1; });

  handle.sync_stream(stream);
}

template CUML_EXPORT void build_treelite_isolation_forest<float>(
  TreeliteModelHandle*, const raft::handle_t&, const IsolationForestModel<float>*);
template CUML_EXPORT void build_treelite_isolation_forest<double>(
  TreeliteModelHandle*, const raft::handle_t&, const IsolationForestModel<double>*);

}  // namespace ML
