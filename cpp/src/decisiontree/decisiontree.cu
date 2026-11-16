/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "decisiontree.cuh"

#include <cuml/common/export.hpp>
#include <cuml/tree/decisiontree.hpp>
#include <cuml/tree/flatnode.h>

#include <raft/core/handle.hpp>

namespace ML {
namespace DT {

void validity_check(const DecisionTreeParams params)
{
  ASSERT((params.max_depth >= 0), "Invalid max depth %d", params.max_depth);
  ASSERT((params.max_leaves == -1) || (params.max_leaves > 0),
         "Invalid max leaves %d",
         params.max_leaves);
  ASSERT((params.max_features > 0) && (params.max_features <= 1.0),
         "max_features value %f outside permitted (0, 1] range",
         params.max_features);
  ASSERT((params.max_n_bins > 0), "Invalid max_n_bins %d", params.max_n_bins);
  ASSERT((params.max_n_bins <= 1024), "max_n_bins should not be larger than 1024");
  ASSERT((params.split_criterion != 3), "MAE not supported.");
  ASSERT((params.min_samples_leaf >= 1),
         "Invalid value for min_samples_leaf %d. Should be >= 1.",
         params.min_samples_leaf);
  ASSERT((params.min_samples_split >= 2),
         "Invalid value for min_samples_split: %d. Should be >= 2.",
         params.min_samples_split);
}

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
 * @param[in] cfg_split_criterion: split criterion; default CRITERION_END,
 *            i.e., GINI for classification or MSE for regression
 * @param[in] cfg_max_batch_size: batch size for experimental backend
 */
void set_tree_params(DecisionTreeParams& params,
                     int cfg_max_depth,
                     int cfg_max_leaves,
                     float cfg_max_features,
                     int cfg_max_n_bins,
                     int cfg_min_samples_leaf,
                     int cfg_min_samples_split,
                     float cfg_min_impurity_decrease,
                     CRITERION cfg_split_criterion,
                     int cfg_max_batch_size)
{
  params.max_depth             = cfg_max_depth;
  params.max_leaves            = cfg_max_leaves;
  params.max_features          = cfg_max_features;
  params.max_n_bins            = cfg_max_n_bins;
  params.min_samples_leaf      = cfg_min_samples_leaf;
  params.min_samples_split     = cfg_min_samples_split;
  params.split_criterion       = cfg_split_criterion;
  params.min_impurity_decrease = cfg_min_impurity_decrease;
  params.max_batch_size        = cfg_max_batch_size;
  validity_check(params);
}

template <class T, class L>
std::string get_tree_summary_text(const TreeMetaDataNode<T, L>* tree)
{
  std::ostringstream oss;
  oss << " Decision Tree depth --> " << tree->depth_counter << " and n_leaves --> "
      << tree->leaf_counter << "\n"
      << " Tree Fitting - Overall time --> " << tree->train_time << " milliseconds"
      << "\n";
  return oss.str();
}

template <class T, class L>
std::string get_tree_text(const TreeMetaDataNode<T, L>* tree)
{
  std::string summary = get_tree_summary_text<T, L>(tree);
  return summary + "\n" + get_node_text<T, L>("", tree, 0, false);
}

template <class T, class L>
std::string get_tree_json(const TreeMetaDataNode<T, L>* tree)
{
  return get_node_json("", tree, 0);
}

// Functions' specializations
template CUML_EXPORT std::string get_tree_summary_text<float, int>(const TreeClassifierF* tree);
template CUML_EXPORT std::string get_tree_summary_text<double, int>(const TreeClassifierD* tree);
template CUML_EXPORT std::string get_tree_summary_text<float, float>(const TreeRegressorF* tree);
template CUML_EXPORT std::string get_tree_summary_text<double, double>(const TreeRegressorD* tree);

template CUML_EXPORT std::string get_tree_text<float, int>(const TreeClassifierF* tree);
template CUML_EXPORT std::string get_tree_text<double, int>(const TreeClassifierD* tree);
template CUML_EXPORT std::string get_tree_text<float, float>(const TreeRegressorF* tree);
template CUML_EXPORT std::string get_tree_text<double, double>(const TreeRegressorD* tree);

template CUML_EXPORT std::string get_tree_json<float, int>(const TreeClassifierF* tree);
template CUML_EXPORT std::string get_tree_json<double, int>(const TreeClassifierD* tree);
template CUML_EXPORT std::string get_tree_json<float, float>(const TreeRegressorF* tree);
template CUML_EXPORT std::string get_tree_json<double, double>(const TreeRegressorD* tree);

}  // End namespace DT
}  // End namespace ML
