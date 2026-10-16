/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "algo_helper.h"
#include "flatnode.h"

#include <cuml/common/export.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace CUML_EXPORT ML {

namespace DT {

struct DecisionTreeParams {
  /**
   * Maximum tree depth. Set to INT32_MAX for unlimited depth
   * (i.e., until leaves are pure or other stopping criteria are met).
   */
  int max_depth;
  /**
   * Maximum leaf nodes per tree. Soft constraint. Unlimited, If `-1`.
   */
  int max_leaves;
  /**
   * Ratio of number of features (columns) to consider per node split.
   */
  float max_features;
  /**
   * maximum number of bins used by the split algorithm per feature.
   */
  int max_n_bins;
  /**
   * The minimum number of samples (rows) in each leaf node.
   */
  int min_samples_leaf;
  /**
   * The minimum number of samples (rows) needed to split an internal node.
   */
  int min_samples_split;
  /**
   * Node split criterion. GINI and Entropy for classification, MSE for regression.
   */
  CRITERION split_criterion;
  /**
   * A node is split if its weighted impurity decrease exceeds this value.
   * Default is 0.0.
   */
  float min_impurity_decrease = 0.0f;

  /**
   * Maximum number of nodes that can be processed in a given batch. This is
   * used only for batched-level algo
   */
  int max_batch_size;
};

/**
 * @brief Set all DecisionTreeParams members.
 * @param[in,out] params: update with tree parameters
 * @param[in] cfg_max_depth: maximum tree depth; default -1
 * @param[in] cfg_max_leaves: maximum leaves; default -1
 * @param[in] cfg_max_features: maximum number of features; default 1.0f
 * @param[in] cfg_max_n_bins: maximum number of bins; default 128
 * @param[in] cfg_min_samples_leaf: min. rows in each leaf node; default 1
 * @param[in] cfg_min_samples_split: min. rows needed to split an internal node;
 *            default 2
 * @param[in] cfg_min_impurity_decrease: split a node only if its weighted impurity
 *                                       decrease exceeds this value
 * @param[in] cfg_split_criterion: split criterion; default CRITERION_END,
 *            i.e., GINI for classification or MSE for regression
 * @param[in] cfg_max_batch_size: Maximum number of nodes that can be processed
              in a batch. This is used only for batched-level algo. Default
              value 4096.
 */
void set_tree_params(DecisionTreeParams& params,
                     int cfg_max_depth               = -1,
                     int cfg_max_leaves              = -1,
                     float cfg_max_features          = 1.0f,
                     int cfg_max_n_bins              = 128,
                     int cfg_min_samples_leaf        = 1,
                     int cfg_min_samples_split       = 2,
                     float cfg_min_impurity_decrease = 0.0f,
                     CRITERION cfg_split_criterion   = CRITERION_END,
                     int cfg_max_batch_size          = 4096);

template <class T, class L>
struct TreeMetaDataNode {
  int treeid;
  int depth_counter;
  int leaf_counter;
  double train_time;
  std::vector<T> vector_leaf;
  std::vector<SparseTreeNode<T, L>> sparsetree;
  int num_outputs;
};

/**
 * @brief Obtain high-level tree information.
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 * @param[in] tree: CPU pointer to TreeMetaDataNode
 * @return High-level tree information as string
 */
template <class T, class L>
std::string get_tree_summary_text(const TreeMetaDataNode<T, L>* tree);

/**
 * @brief Obtain detailed tree information.
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 * @param[in] tree: CPU pointer to TreeMetaDataNode
 * @return Detailed tree information as string
 */
template <class T, class L>
std::string get_tree_text(const TreeMetaDataNode<T, L>* tree);

/**
 * @brief Export tree as a JSON string
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 * @param[in] tree: CPU pointer to TreeMetaDataNode
 * @return Tree structure as JSON stsring
 */
template <class T, class L>
std::string get_tree_json(const TreeMetaDataNode<T, L>* tree);

typedef TreeMetaDataNode<float, int> TreeClassifierF;
typedef TreeMetaDataNode<double, int> TreeClassifierD;
typedef TreeMetaDataNode<float, float> TreeRegressorF;
typedef TreeMetaDataNode<double, double> TreeRegressorD;

}  // End namespace DT
}  // End namespace CUML_EXPORT ML
