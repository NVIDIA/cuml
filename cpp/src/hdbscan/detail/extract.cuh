/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "membership.cuh"
#include "select.cuh"
#include "stabilities.cuh"
#include "utils.cuh"

#include <cuml/cluster/hdbscan.hpp>

#include <raft/label/classlabels.cuh>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/op/sort.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/for_each.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cstddef>

namespace ML {
namespace HDBSCAN {
namespace detail {
namespace Extract {

/**
 * Union-rank data structure with path compression for
 * labeling data points based on their farthest ancestors
 * under root.
 *
 * For correctness, it is important that all children are
 * visited before their parents.
 * @tparam value_idx
 */
template <typename value_idx>
class TreeUnionFind {
 public:
  TreeUnionFind(value_idx size_) : size(size_), data(size_ * 2, 0)
  {
    for (int i = 0; i < size; i++) {
      data[i * 2] = i;
    }
  }

  void perform_union(value_idx x, value_idx y)
  {
    value_idx x_root = find(x);
    value_idx y_root = find(y);

    if (data[x_root * 2 + 1] < data[y_root * 2 + 1])
      data[x_root * 2] = y_root;
    else if (data[x_root * 2 + 1] > data[y_root * 2 + 1])
      data[y_root * 2] = x_root;
    else {
      data[y_root * 2] = x_root;
      data[x_root * 2 + 1] += 1;
    }
  }

  value_idx find(value_idx x)
  {
    if (data[x * 2] != x) { data[x * 2] = find(data[x * 2]); }

    return data[x * 2];
  }

  value_idx* get_data() { return data.data(); }

 private:
  value_idx size;
  std::vector<value_idx> data;
};

template <typename value_idx, typename value_t>
void do_labelling_on_host(const raft::handle_t& handle,
                          Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
                          std::set<value_idx>& clusters,
                          value_idx n_leaves,
                          bool allow_single_cluster,
                          value_idx* labels,
                          value_t cluster_selection_epsilon)
{
  auto stream = handle.get_stream();

  std::vector<value_idx> children_h(condensed_tree.get_n_edges());
  std::vector<value_t> lambda_h(condensed_tree.get_n_edges());
  std::vector<value_idx> parent_h(condensed_tree.get_n_edges());

  raft::update_host(
    children_h.data(), condensed_tree.get_children(), condensed_tree.get_n_edges(), stream);
  raft::update_host(
    parent_h.data(), condensed_tree.get_parents(), condensed_tree.get_n_edges(), stream);
  raft::update_host(
    lambda_h.data(), condensed_tree.get_lambdas(), condensed_tree.get_n_edges(), stream);

  handle.sync_stream(stream);

  auto parents       = thrust::device_pointer_cast(condensed_tree.get_parents());
  auto thrust_policy = handle.get_thrust_policy();
  value_idx size =
    *thrust::max_element(thrust_policy, parents, parents + condensed_tree.get_n_edges());

  std::vector<value_idx> result(n_leaves);
  std::vector<value_t> parent_lambdas(size + 1, 0);

  auto union_find = TreeUnionFind<value_idx>(size + 1);

  for (int i = 0; i < condensed_tree.get_n_edges(); i++) {
    value_idx child  = children_h[i];
    value_idx parent = parent_h[i];

    if (clusters.find(child) == clusters.end()) union_find.perform_union(parent, child);

    parent_lambdas[parent_h[i]] = max(parent_lambdas[parent_h[i]], lambda_h[i]);
  }

  value_t inverse_cluster_selection_epsilon;
  if (cluster_selection_epsilon != 0.0) {
    inverse_cluster_selection_epsilon = 1 / cluster_selection_epsilon;
  }

  for (int i = 0; i < n_leaves; i++) {
    value_idx cluster = union_find.find(i);

    if (cluster < n_leaves)
      result[i] = -1;
    else if (cluster == n_leaves) {
      // TODO: Implement the cluster_selection_epsilon / epsilon_search
      if (clusters.size() == 1 && allow_single_cluster) {
        auto it              = std::find(children_h.begin(), children_h.end(), i);
        auto child_idx       = std::distance(children_h.begin(), it);
        value_t child_lambda = lambda_h[child_idx];

        if (cluster_selection_epsilon != 0) {
          if (child_lambda >= inverse_cluster_selection_epsilon) {
            result[i] = cluster - n_leaves;
          } else {
            result[i] = -1;
          }
        } else if (child_lambda >= parent_lambdas[cluster])
          result[i] = cluster - n_leaves;
        else
          result[i] = -1;
      } else {
        result[i] = -1;
      }
    } else {
      result[i] = cluster - n_leaves;
    }
  }

  raft::update_device(labels, result.data(), n_leaves, stream);
}

/*
  @brief Internal function compute inverse_label_map for CPU/GPU interop
*/
template <typename value_idx, typename value_t>
void _compute_inverse_label_map(const raft::handle_t& handle,
                                Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
                                size_t n_leaves,
                                Common::CLUSTER_SELECTION_METHOD cluster_selection_method,
                                rmm::device_uvector<value_idx>& inverse_label_map,
                                bool allow_single_cluster,
                                value_idx max_cluster_size,
                                value_t cluster_selection_epsilon)
{
  auto stream = handle.get_stream();
  rmm::device_uvector<value_t> tree_stabilities(condensed_tree.get_n_clusters(),
                                                handle.get_stream());
  Stability::compute_stabilities(handle, condensed_tree, tree_stabilities.data());
  rmm::device_uvector<int> is_cluster(condensed_tree.get_n_clusters(), handle.get_stream());

  if (max_cluster_size <= 0) max_cluster_size = n_leaves;  // negates the max cluster size

  Select::select_clusters(handle,
                          condensed_tree,
                          tree_stabilities.data(),
                          is_cluster.data(),
                          cluster_selection_method,
                          allow_single_cluster,
                          max_cluster_size,
                          cluster_selection_epsilon);

  std::vector<int> is_cluster_h(is_cluster.size());
  raft::update_host(is_cluster_h.data(), is_cluster.data(), is_cluster_h.size(), stream);
  handle.sync_stream(stream);

  std::set<value_idx> clusters;
  for (std::size_t i = 0; i < is_cluster_h.size(); i++) {
    if (is_cluster_h[i] != 0) { clusters.insert(i + n_leaves); }
  }

  std::vector<value_idx> inverse_label_map_h(clusters.size(), -1);
  value_idx i = 0;
  // creating inverse index between
  // original and final labels
  for (const value_idx cluster : clusters) {
    inverse_label_map_h[i] = cluster - n_leaves;
    i++;
  }

  // resizing is to n_clusters
  inverse_label_map.resize(clusters.size(), stream);
  raft::copy(
    inverse_label_map.data(), inverse_label_map_h.data(), inverse_label_map_h.size(), stream);
}

/**
 * Compute cluster stabilities, perform cluster selection, and
 * label the resulting clusters. In addition, probabilities
 * are computed and stabilities are normalized into scores.
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle raft handle for resource management
 * @param[in] condensed_tree a condensed hierarchy
 * @param[in] n_leaves number of data samples
 * @param[out] labels array of labels on device (size n_leaves)
 * @param[out] stabilities array of stabilities on device (size n_clusters)
 * @param[out] probabilities array of probabilities on device (size n_leaves)
 * @param[out] label_map array mapping condensed label ids to selected label ids (size
 * n_condensed_trees)
 * @param[in] cluster_selection_method method to use for cluster selection
 * @param[out] inverse_label_map array mapping final label ids to condensed label ids, used for
 * prediction APIs (size n_clusters)
 * @param[in] allow_single_cluster allows a single cluster to be returned (rather than just noise)
 * @param[in] max_cluster_size maximum number of points that can be considered in a cluster before
 * it is split into multiple sub-clusters.
 * @param[in] cluster_selection_epsilon a distance threshold. clusters below this value will be
 * merged.
 */
template <typename value_idx, typename value_t>
value_idx extract_clusters(const raft::handle_t& handle,
                           Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
                           size_t n_leaves,
                           value_idx* labels,
                           value_t* tree_stabilities,
                           value_t* probabilities,
                           value_idx* label_map,
                           Common::CLUSTER_SELECTION_METHOD cluster_selection_method,
                           rmm::device_uvector<value_idx>& inverse_label_map,
                           bool allow_single_cluster         = false,
                           value_idx max_cluster_size        = 0,
                           value_t cluster_selection_epsilon = 0.0)
{
  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  Stability::compute_stabilities(handle, condensed_tree, tree_stabilities);
  rmm::device_uvector<int> is_cluster(condensed_tree.get_n_clusters(), handle.get_stream());

  if (max_cluster_size <= 0) max_cluster_size = n_leaves;  // negates the max cluster size

  Select::select_clusters(handle,
                          condensed_tree,
                          tree_stabilities,
                          is_cluster.data(),
                          cluster_selection_method,
                          allow_single_cluster,
                          max_cluster_size,
                          cluster_selection_epsilon);

  std::vector<int> is_cluster_h(is_cluster.size());
  raft::update_host(is_cluster_h.data(), is_cluster.data(), is_cluster_h.size(), stream);
  handle.sync_stream(stream);

  std::set<value_idx> clusters;
  for (std::size_t i = 0; i < is_cluster_h.size(); i++) {
    if (is_cluster_h[i] != 0) { clusters.insert(i + n_leaves); }
  }

  std::vector<value_idx> label_map_h(condensed_tree.get_n_clusters(), -1);
  std::vector<value_idx> inverse_label_map_h(clusters.size(), -1);
  value_idx i = 0;
  // creating forward and inverse index between
  // original and final labels
  for (const value_idx cluster : clusters) {
    label_map_h[cluster - n_leaves] = i;
    inverse_label_map_h[i]          = cluster - n_leaves;
    i++;
  }

  // resizing is to n_clusters
  inverse_label_map.resize(clusters.size(), stream);
  raft::copy(
    inverse_label_map.data(), inverse_label_map_h.data(), inverse_label_map_h.size(), stream);

  raft::copy(label_map, label_map_h.data(), label_map_h.size(), stream);
  do_labelling_on_host<value_idx, value_t>(handle,
                                           condensed_tree,
                                           clusters,
                                           n_leaves,
                                           allow_single_cluster,
                                           labels,
                                           cluster_selection_epsilon);

  Membership::get_probabilities<value_idx, value_t>(handle, condensed_tree, labels, probabilities);

  return clusters.size();
}

};  // end namespace Extract
};  // end namespace detail
};  // end namespace HDBSCAN
};  // end namespace ML
