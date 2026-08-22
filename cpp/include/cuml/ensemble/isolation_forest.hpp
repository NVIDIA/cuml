/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/common/export.hpp>
#include <cuml/common/logger.hpp>
#include <cuml/ensemble/treelite_defs.hpp>

#include <rmm/device_buffer.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace raft {
class handle_t;
}

namespace CUML_EXPORT ML {

/**
 * @brief Isolation Forest hyperparameters
 */
struct IF_params {
  int n_estimators = 100;    ///< Number of isolation trees
  int max_samples  = 256;    ///< Samples per tree (default 256)
  int max_depth    = -1;     ///< Max depth (-1 = auto: ceil(log2(max_samples)))
  int max_features = -1;     ///< Features sampled per tree (-1 = all features)
  bool bootstrap   = false;  ///< If true, sample rows with replacement
  uint64_t seed    = 0;      ///< Random seed
};

/** @brief Trained Isolation Forest model */
template <class T>
struct IsolationForestModel {
  IF_params params;                           ///< Hyperparameters used for training
  int n_features          = 0;                ///< Number of features in training data
  int n_features_per_tree = 0;                ///< Features sampled per tree
  int n_samples_per_tree  = 0;                ///< Samples used per tree (for c(n) calculation)
  double c_normalization  = 0;                ///< Precomputed c(n) normalization constant
  int max_nodes_per_tree  = 0;                ///< Allocated node capacity per tree
  rmm::device_buffer global_feature_indices;  ///< Optional per-tree sampled feature ids
  rmm::device_buffer global_nodes;            ///< Device memory for global-memory IFNode array
  rmm::device_buffer global_tree_offsets;     ///< Per-tree base offset in global_nodes
  rmm::device_buffer global_tree_n_nodes;     ///< Per-tree used node counts
  rmm::device_buffer global_tree_max_depth;   ///< Per-tree max depth metadata
};

typedef IsolationForestModel<float> IsolationForestF;
typedef IsolationForestModel<double> IsolationForestD;

/**
 * @brief Compact node representation (matches internal IFNode layout).
 *
 * Internal nodes: feature_idx >= 0, threshold is split value
 * Leaf nodes: feature_idx = -1, threshold stores pre-computed path length
 */
template <typename T>
struct IFNodeCompact {
  int feature_idx;
  T threshold;
  int left_child;
  int right_child;
};

/**
 * @brief Compact tree data returned by get_compact_trees().
 * Contains all used nodes concatenated, with per-tree metadata.
 */
template <typename T>
struct CompactIFForest {
  std::vector<IFNodeCompact<T>> nodes;  ///< All used nodes, trees concatenated
  std::vector<int> tree_offsets;        ///< Start index in nodes[] for each tree
  std::vector<int> tree_n_nodes;        ///< Number of nodes per tree
  std::vector<int> tree_max_depth;      ///< Max depth per tree
};

/**
 * @brief Extract compact tree data from a trained model for export.
 *
 * Efficiently copies only the used nodes (~400 KB for 100 trees)
 * instead of the full padded storage (~200 MB).
 */
template <typename T>
CompactIFForest<T> get_compact_trees(const raft::handle_t& handle,
                                     const IsolationForestModel<T>* model);

/** @brief Compute c(n) = 2H(n-1) - 2(n-1)/n normalization constant. */
double compute_c_normalization(int n);

/**
 * @brief Build a Treelite regression forest from a trained Isolation Forest model.
 *
 * The exported Treelite model predicts average path length across trees. The
 * Isolation Forest anomaly score transform, s(x) = 2^(-E[h(x)] / c(n)), is
 * intentionally applied by callers so exported tree structure stays faithful
 * to the trained isolation trees.
 *
 * @param[out] model_handle Treelite model handle owned by the caller
 * @param[in]  handle       RAFT handle for GPU resources
 * @param[in]  forest       Trained Isolation Forest model
 */
template <typename T>
void build_treelite_isolation_forest(TreeliteModelHandle* model_handle,
                                     const raft::handle_t& handle,
                                     const IsolationForestModel<T>* forest);

/**
 * @brief Fit an Isolation Forest model.
 *
 * @param[in]  handle    RAFT handle for GPU resources
 * @param[out] forest    Model to populate with trained trees
 * @param[in]  input     Training data, column-major [n_rows × n_cols], device pointer
 * @param[in]  n_rows    Number of training samples
 * @param[in]  n_cols    Number of features
 * @param[in]  params    Hyperparameters (n_estimators, max_samples, max_depth, seed)
 * @param[in]  verbosity Logging level
 */
void fit(const raft::handle_t& handle,
         IsolationForestF* forest,
         const float* input,
         size_t n_rows,
         int n_cols,
         const IF_params& params,
         rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);

void fit(const raft::handle_t& handle,
         IsolationForestD* forest,
         const double* input,
         size_t n_rows,
         int n_cols,
         const IF_params& params,
         rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);

/**
 * @brief Fit an Isolation Forest and export it as a Treelite model.
 *
 *
 * @param[in]  handle          RAFT handle for GPU resources
 * @param[out] model_handle    Treelite model handle owned by the caller
 * @param[in]  input           Training data, column-major [n_rows × n_cols], device pointer
 * @param[in]  n_rows          Number of training samples
 * @param[in]  n_cols          Number of features
 * @param[in]  params          Hyperparameters (n_estimators, max_samples, max_depth, seed)
 * @param[out] c_normalization Normalization constant c(n) for the trained forest, needed to
 *                             turn average path lengths into anomaly scores
 * @param[in]  verbosity       Logging level
 */
template <typename T>
void fit_treelite(const raft::handle_t& handle,
                  TreeliteModelHandle* model_handle,
                  const T* input,
                  size_t n_rows,
                  int n_cols,
                  const IF_params& params,
                  double* c_normalization,
                  rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);

/**
 * @brief Compute anomaly scores.
 *
 * Returns scores following the original paper convention (Liu et al. 2008):
 * - Score ≈ 1.0: anomaly (isolated quickly)
 * - Score ≈ 0.5: normal (average isolation depth)
 * - Score ≈ 0.0: very normal (hard to isolate)
 *
 * @param[in]  handle    RAFT handle for GPU resources
 * @param[in]  forest    Trained Isolation Forest model
 * @param[in]  input     Test data, row-major [n_rows × n_cols], device pointer
 * @param[in]  n_rows    Number of test samples
 * @param[in]  n_cols    Number of features (must match training)
 * @param[out] scores    Anomaly scores [n_rows], device pointer
 * @param[in]  verbosity Logging level
 */
void score_samples(const raft::handle_t& handle,
                   const IsolationForestF* forest,
                   const float* input,
                   size_t n_rows,
                   int n_cols,
                   float* scores,
                   rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);

void score_samples(const raft::handle_t& handle,
                   const IsolationForestD* forest,
                   const double* input,
                   size_t n_rows,
                   int n_cols,
                   double* scores,
                   rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);

/**
 * @brief Predict anomaly labels.
 *
 * @param[in]  handle      RAFT handle for GPU resources
 * @param[in]  forest      Trained Isolation Forest model
 * @param[in]  input       Test data, row-major [n_rows × n_cols], device pointer
 * @param[in]  n_rows      Number of test samples
 * @param[in]  n_cols      Number of features (must match training)
 * @param[out] predictions Labels [n_rows]: 1 = anomaly, -1 = normal, device pointer
 * @param[in]  threshold   Score threshold (default 0.5); scores strictly above it are anomalies
 * @param[in]  verbosity   Logging level
 */
void predict(const raft::handle_t& handle,
             const IsolationForestF* forest,
             const float* input,
             size_t n_rows,
             int n_cols,
             int* predictions,
             float threshold                     = 0.5f,
             rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);

void predict(const raft::handle_t& handle,
             const IsolationForestD* forest,
             const double* input,
             size_t n_rows,
             int n_cols,
             int* predictions,
             double threshold                    = 0.5,
             rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);

}  // namespace CUML_EXPORT ML
