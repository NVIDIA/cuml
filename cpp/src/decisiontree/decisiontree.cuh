/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "batched-levelalgo/builder.cuh"

#include <cuml/common/logger.hpp>
#include <cuml/tree/flatnode.h>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <treelite/tree.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <locale>
#include <vector>

namespace ML {

namespace tl = treelite;

namespace DT {

inline bool is_dev_ptr(const void* p)
{
  cudaPointerAttributes pointer_attr;
  cudaError_t err = cudaPointerGetAttributes(&pointer_attr, p);
  if (err == cudaSuccess) {
    return (pointer_attr.devicePointer || pointer_attr.type == cudaMemoryTypeDevice);
  } else {
    err = cudaGetLastError();
    return false;
  }
}

template <typename T>
std::string to_string_high_precision(T x)
{
  static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
                "T must be float, double, or integer");
  std::ostringstream oss;
  oss.imbue(std::locale::classic());  // use C locale
  if (std::is_floating_point<T>::value) {
    oss << std::setprecision(std::numeric_limits<T>::max_digits10) << x;
  } else {
    oss << x;
  }
  return oss.str();
}

template <class T, class L>
std::string get_node_text(const std::string& prefix,
                          const TreeMetaDataNode<T, L>* tree,
                          int idx,
                          bool isLeft)
{
  const auto& node = tree->sparsetree[idx];

  std::ostringstream oss;

  // print the value of the node
  oss << prefix.c_str();
  oss << (isLeft ? "├" : "└");

  if (node.IsLeaf()) {
    oss << "(leaf, "
        << "prediction: [";

    for (int k = 0; k < tree->num_outputs - 1; k++) {
      oss << tree->vector_leaf[idx * tree->num_outputs + k] << ", ";
    }
    oss << tree->vector_leaf[idx * tree->num_outputs + tree->num_outputs - 1];

    oss << "], best_metric_val: " << node.BestMetric() << ")";
  } else {
    oss << "("
        << "colid: " << node.ColumnId() << ", quesval: " << node.QueryValue()
        << ", best_metric_val: " << node.BestMetric() << ")";
  }

  if (!node.IsLeaf()) {
    // enter the next tree level - left and right branch
    oss << "\n"
        << get_node_text(prefix + (isLeft ? "│   " : "    "), tree, node.LeftChildId(), true)
        << "\n"
        << get_node_text(prefix + (isLeft ? "│   " : "    "), tree, node.RightChildId(), false);
  }
  return oss.str();
}

template <class T, class L>
std::string get_node_json(const std::string& prefix, const TreeMetaDataNode<T, L>* tree, int idx)
{
  const auto& node = tree->sparsetree[idx];

  std::ostringstream oss;
  if (!node.IsLeaf()) {
    oss << prefix << "{\"nodeid\": " << idx << ", \"split_feature\": " << node.ColumnId()
        << ", \"split_threshold\": " << to_string_high_precision(node.QueryValue())
        << ", \"gain\": " << to_string_high_precision(node.BestMetric());
    oss << ", \"instance_count\": " << node.InstanceCount();
    oss << ", \"yes\": " << node.LeftChildId() << ", \"no\": " << (node.RightChildId())
        << ", \"children\": [\n";
    // enter the next tree level - left and right branch
    oss << get_node_json(prefix + "  ", tree, node.LeftChildId()) << ",\n"
        << get_node_json(prefix + "  ", tree, node.RightChildId()) << "\n"
        << prefix << "]}";
  } else {
    oss << prefix << "{\"nodeid\": " << idx << ", \"leaf_value\": [";
    for (int k = 0; k < tree->num_outputs - 1; k++) {
      oss << to_string_high_precision(tree->vector_leaf[idx * tree->num_outputs + k]) << ", ";
    }
    oss << to_string_high_precision(
      tree->vector_leaf[idx * tree->num_outputs + tree->num_outputs - 1]);
    oss << "], \"instance_count\": " << node.InstanceCount();
    oss << "}";
  }
  return oss.str();
}

template <class T, class L>
tl::Tree<T, T> build_treelite_tree(const DT::TreeMetaDataNode<T, L>& rf_tree,
                                   unsigned int num_class)
{
  // First index refers to the cuml node id
  // Second refers to the tl node id
  using kv = std::pair<std::size_t, std::size_t>;
  std::vector<kv> cur_level_queue;
  std::vector<kv> next_level_queue;

  tl::Tree<T, T> tl_tree;
  tl_tree.Init();
  tl_tree.AllocNode();  // Allocate the root node

  // Track head and tail of bounded "queues" (implemented as vectors for
  // performance)
  size_t cur_front  = 0;
  size_t cur_end    = 0;
  size_t next_front = 0;
  size_t next_end   = 0;

  cur_level_queue.resize(std::max<size_t>(cur_level_queue.size(), 1));
  cur_level_queue[0] = {0, 0};
  ++cur_end;

  while (cur_front != cur_end) {
    size_t cur_level_size = cur_end - cur_front;
    next_level_queue.resize(std::max(2 * cur_level_size, next_level_queue.size()));

    for (size_t i = 0; i < cur_level_size; ++i) {
      auto cuml_node_id  = cur_level_queue[cur_front].first;
      const auto& q_node = rf_tree.sparsetree[cuml_node_id];
      auto tl_node_id    = cur_level_queue[cur_front].second;
      ++cur_front;

      if (!q_node.IsLeaf()) {
        const int cleft  = tl_tree.AllocNode();
        const int cright = tl_tree.AllocNode();
        tl_tree.SetChildren(tl_node_id, cleft, cright);

        // Push left child to next_level queue.
        next_level_queue[next_end] = {q_node.LeftChildId(), cleft};
        ++next_end;

        // Push right child to next_level queue.
        next_level_queue[next_end] = {q_node.RightChildId(), cright};
        ++next_end;

        // Set node from current level as numerical node. Children IDs known.
        tl_tree.SetNumericalTest(
          tl_node_id, q_node.ColumnId(), q_node.QueryValue(), true, tl::Operator::kLE);

      } else {
        auto leaf_begin = rf_tree.vector_leaf.begin() + cuml_node_id * rf_tree.num_outputs;
        if (num_class == 1) {
          tl_tree.SetLeaf(tl_node_id, *leaf_begin);
        } else {
          // if rf_tree.num_outputs < num_class, fill the remainder with zero
          // Most likely this happens when a binary classifier is fit with all-0 labels
          ASSERT(static_cast<unsigned int>(rf_tree.num_outputs) <= num_class,
                 "num_class too small");
          std::vector<T> leaf_vector(num_class, T(0));
          std::copy(leaf_begin, leaf_begin + rf_tree.num_outputs, leaf_vector.begin());
          tl_tree.SetLeafVector(tl_node_id, leaf_vector);
        }
      }
      tl_tree.SetDataCount(tl_node_id, q_node.InstanceCount());
    }

    cur_level_queue.swap(next_level_queue);
    cur_front  = next_front;
    cur_end    = next_end;
    next_front = 0;
    next_end   = 0;
  }
  return tl_tree;
}

class DecisionTree {
 public:
  template <class DataT, class LabelT>
  static std::shared_ptr<DT::TreeMetaDataNode<DataT, LabelT>> fit(
    const raft::handle_t& handle,
    const cudaStream_t s,
    const DataT* data,
    const std::int64_t ncols,
    const std::int64_t nrows,
    const LabelT* labels,
    rmm::device_uvector<std::int64_t>* row_ids,
    int unique_labels,
    DecisionTreeParams params,
    uint64_t seed,
    const Quantiles<DataT>& quantiles,
    std::int64_t treeid,
    const double* sample_weight = nullptr,
    bool row_major              = false)
  {
    if (params.split_criterion ==
        CRITERION::CRITERION_END) {  // Set default to GINI (classification) or MSE (regression)
      CRITERION default_criterion =
        (std::numeric_limits<LabelT>::is_integer) ? CRITERION::GINI : CRITERION::MSE;
      params.split_criterion = default_criterion;
    }
    // Dispatch objective family. The objective object switches on the criterion at runtime.
    if (not std::is_same<DataT, LabelT>::value and (params.split_criterion == CRITERION::GINI ||
                                                    params.split_criterion == CRITERION::ENTROPY)) {
      if (sample_weight != nullptr) {
        return Builder<ClassificationObjectiveFunction<DataT, LabelT, true>>(handle,
                                                                             s,
                                                                             treeid,
                                                                             seed,
                                                                             params,
                                                                             data,
                                                                             labels,
                                                                             sample_weight,
                                                                             nrows,
                                                                             ncols,
                                                                             row_ids,
                                                                             unique_labels,
                                                                             quantiles,
                                                                             row_major)
          .train();
      }
      return Builder<ClassificationObjectiveFunction<DataT, LabelT>>(handle,
                                                                     s,
                                                                     treeid,
                                                                     seed,
                                                                     params,
                                                                     data,
                                                                     labels,
                                                                     sample_weight,
                                                                     nrows,
                                                                     ncols,
                                                                     row_ids,
                                                                     unique_labels,
                                                                     quantiles,
                                                                     row_major)
        .train();
    } else if (std::is_same<DataT, LabelT>::value and
               (params.split_criterion == CRITERION::MSE ||
                params.split_criterion == CRITERION::POISSON ||
                params.split_criterion == CRITERION::GAMMA ||
                params.split_criterion == CRITERION::INVERSE_GAUSSIAN)) {
      if (sample_weight != nullptr) {
        return Builder<RegressionObjectiveFunction<DataT, LabelT, true>>(handle,
                                                                         s,
                                                                         treeid,
                                                                         seed,
                                                                         params,
                                                                         data,
                                                                         labels,
                                                                         sample_weight,
                                                                         nrows,
                                                                         ncols,
                                                                         row_ids,
                                                                         unique_labels,
                                                                         quantiles,
                                                                         row_major)
          .train();
      }
      return Builder<RegressionObjectiveFunction<DataT, LabelT>>(handle,
                                                                 s,
                                                                 treeid,
                                                                 seed,
                                                                 params,
                                                                 data,
                                                                 labels,
                                                                 sample_weight,
                                                                 nrows,
                                                                 ncols,
                                                                 row_ids,
                                                                 unique_labels,
                                                                 quantiles,
                                                                 row_major)
        .train();
    } else {
      ASSERT(false, "Unknown split criterion.");
    }
  }

};  // End DecisionTree Class

// Class specializations
template tl::Tree<float, float> build_treelite_tree<float, int>(
  const DT::TreeMetaDataNode<float, int>& rf_tree, unsigned int num_class);
template tl::Tree<double, double> build_treelite_tree<double, int>(
  const DT::TreeMetaDataNode<double, int>& rf_tree, unsigned int num_class);
template tl::Tree<float, float> build_treelite_tree<float, float>(
  const DT::TreeMetaDataNode<float, float>& rf_tree, unsigned int num_class);
template tl::Tree<double, double> build_treelite_tree<double, double>(
  const DT::TreeMetaDataNode<double, double>& rf_tree, unsigned int num_class);
}  // End namespace DT

}  // End namespace ML
