/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "kernels/condense.cuh"

#include <cuml/cluster/hdbscan.hpp>

#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/op/sort.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>

#include <vector>

namespace ML {
namespace HDBSCAN {
namespace detail {
namespace Condense {

/**
 * Helper function for BFS traversal from a given node in the hierarchy
 */
template <typename value_idx>
void bfs_from_node(value_idx bfs_root,
                   value_idx n_samples,
                   const value_idx* h_children,
                   std::vector<value_idx>& result)
{
  std::vector<value_idx> process_queue;
  process_queue.push_back(bfs_root);

  while (!process_queue.empty()) {
    // Add all nodes in current level to result
    result.insert(result.end(), process_queue.begin(), process_queue.end());

    // Filter for internal nodes (>= n_samples) and convert to hierarchy indices
    std::vector<value_idx> internal_nodes;
    for (value_idx x : process_queue) {
      if (x >= n_samples) { internal_nodes.push_back(x - n_samples); }
    }

    // Get children of all internal nodes for next level
    std::vector<value_idx> next_queue;
    for (value_idx node : internal_nodes) {
      value_idx left  = h_children[node * 2];
      value_idx right = h_children[node * 2 + 1];
      next_queue.push_back(left);
      next_queue.push_back(right);
    }
    process_queue = std::move(next_queue);
  }
}

/**
 * This implementation is based on scikit-learn’s _condense_tree implementation.
 * Performs a level-by-level BFS traversal of the dendrogram
 * and applies the following logic at each internal node:
 *
 * 1. If both children are >= min_cluster_size:
 *    - Create new cluster labels for both children
 *    - Add edges from parent to both children
 *
 * 2. If both children are < min_cluster_size:
 *    - Collapse both subtrees
 *    - Add edges from parent directly to all leaf nodes in both subtrees
 *
 * 3. If only left child is < min_cluster_size:
 *    - Right child inherits parent's label
 *    - Collapse left subtree, add edges to all its leaf nodes
 *
 * 4. If only right child is < min_cluster_size:
 *    - Left child inherits parent's label
 *    - Collapse right subtree, add edges to all its leaf nodes
 *
 * we ignore subnodes of the collapsed subtree by using a ignore array
 */
template <typename value_idx, typename value_t>
void _build_condensed_hierarchy(const raft::handle_t& handle,
                                const value_idx* children,
                                const value_t* delta,
                                const value_idx* sizes,
                                int n_leaves,
                                int min_cluster_size,
                                rmm::device_uvector<value_idx>& out_parent,
                                rmm::device_uvector<value_idx>& out_child,
                                rmm::device_uvector<value_t>& out_lambda,
                                rmm::device_uvector<value_idx>& out_size)
{
  cudaStream_t stream  = handle.get_stream().get();
  value_idx root       = 2 * (n_leaves - 1);
  value_idx n_samples  = n_leaves;
  value_idx next_label = n_samples + 1;

  // Copy data to host
  std::vector<value_idx> h_children(2 * (n_leaves - 1));
  std::vector<value_t> h_delta(n_leaves - 1);
  std::vector<value_idx> h_sizes(n_leaves - 1);

  raft::copy(h_children.data(), children, 2 * (n_leaves - 1), stream);
  raft::copy(h_delta.data(), delta, n_leaves - 1, stream);
  raft::copy(h_sizes.data(), sizes, n_leaves - 1, stream);
  handle.sync_stream(stream);

  // host side output arrays
  std::vector<value_idx> h_out_parent;
  std::vector<value_idx> h_out_child;
  std::vector<value_t> h_out_lambda;
  std::vector<value_idx> h_out_sizes;

  auto add_edge = [&](value_idx parent, value_idx child, value_t lambda, value_idx size) {
    h_out_parent.push_back(parent);
    h_out_child.push_back(child);
    h_out_lambda.push_back(lambda);
    h_out_sizes.push_back(size);
  };

  // Get BFS ordering from root
  std::vector<value_idx> node_list;
  bfs_from_node(root, n_samples, h_children.data(), node_list);

  std::vector<value_idx> relabel(root + 1, 0);
  std::vector<uint8_t> ignore(node_list.size(), 0);
  relabel[root] = n_samples;

  // Process nodes in BFS order
  for (size_t idx = 0; idx < node_list.size(); idx++) {
    value_idx node = node_list[idx];

    // Skip if already processed or is a leaf
    if (ignore[node] || node < n_samples) { continue; }

    value_idx left       = h_children[(node - n_samples) * 2];
    value_idx right      = h_children[(node - n_samples) * 2 + 1];
    value_t distance     = h_delta[node - n_samples];
    value_t lambda_value = distance > 0.0 ? 1.0 / distance : std::numeric_limits<value_t>::max();

    value_idx left_count  = left >= n_samples ? h_sizes[left - n_samples] : 1;
    value_idx right_count = right >= n_samples ? h_sizes[right - n_samples] : 1;

    if (left_count >= min_cluster_size &&
        right_count >= min_cluster_size) {  // Case 1: Both children are large enough
      relabel[left] = next_label++;
      add_edge(relabel[node], relabel[left], lambda_value, left_count);

      relabel[right] = next_label++;
      add_edge(relabel[node], relabel[right], lambda_value, right_count);
    } else if (left_count < min_cluster_size &&
               right_count < min_cluster_size) {  // Case 2: Both children are too small
      // Collapse left subtree
      std::vector<value_idx> left_descendants;
      bfs_from_node(left, n_samples, h_children.data(), left_descendants);
      for (value_idx sub_node : left_descendants) {
        if (sub_node < n_samples) { add_edge(relabel[node], sub_node, lambda_value, 1); }
        ignore[sub_node] = 1;
      }

      // Collapse right subtree
      std::vector<value_idx> right_descendants;
      bfs_from_node(right, n_samples, h_children.data(), right_descendants);
      for (value_idx sub_node : right_descendants) {
        if (sub_node < n_samples) { add_edge(relabel[node], sub_node, lambda_value, 1); }
        ignore[sub_node] = 1;
      }
    }

    else if (left_count < min_cluster_size) {  // Case 3: Only left child is too small
      relabel[right] = relabel[node];

      // Collapse left subtree
      std::vector<value_idx> left_descendants;
      bfs_from_node(left, n_samples, h_children.data(), left_descendants);
      for (value_idx sub_node : left_descendants) {
        if (sub_node < n_samples) { add_edge(relabel[node], sub_node, lambda_value, 1); }
        ignore[sub_node] = 1;
      }
    }

    else {  // Case 4: Only right child is too small
      relabel[left] = relabel[node];

      // Collapse right subtree
      std::vector<value_idx> right_descendants;
      bfs_from_node(right, n_samples, h_children.data(), right_descendants);
      for (value_idx sub_node : right_descendants) {
        if (sub_node < n_samples) { add_edge(relabel[node], sub_node, lambda_value, 1); }
        ignore[sub_node] = 1;
      }
    }
  }

  if (h_out_parent.size() > 0) {
    raft::copy(out_parent.data(), h_out_parent.data(), h_out_parent.size(), stream);
    raft::copy(out_child.data(), h_out_child.data(), h_out_child.size(), stream);
    raft::copy(out_lambda.data(), h_out_lambda.data(), h_out_lambda.size(), stream);
    raft::copy(out_size.data(), h_out_sizes.data(), h_out_sizes.size(), stream);
    handle.sync_stream(stream);
  }
}

/**
 * Condenses a binary single-linkage tree dendrogram in the Scipy hierarchy
 * format by collapsing subtrees that fall below a minimum cluster size.
 *
 * For increased parallelism, the output array sizes are held fixed but
 * the result will be sparse (e.g. zeros in place of parents who have been
 * removed / collapsed). This function accepts an empty instance of
 * `CondensedHierarchy` and invokes the `condense()` function on it to
 * convert the sparse output arrays into their dense form.
 *
 * @tparam value_idx
 * @tparam value_t
 * @tparam tpb
 * @param handle
 * @param[in] children parents/children from single-linkage dendrogram
 * @param[in] delta distances from single-linkage dendrogram
 * @param[in] sizes sizes from single-linkage dendrogram
 * @param[in] min_cluster_size any subtrees less than this size will be
 *                             collapsed.
 * @param[in] n_leaves number of actual data samples in the dendrogram
 * @param[out] condensed_tree output dendrogram. will likely no longer be
 *                            a binary tree.
 */
template <typename value_idx, typename value_t, int tpb = 256>
void build_condensed_hierarchy(const raft::handle_t& handle,
                               const value_idx* children,
                               const value_t* delta,
                               const value_idx* sizes,
                               int min_cluster_size,
                               int n_leaves,
                               Common::CondensedHierarchy<value_idx, value_t>& condensed_tree)
{
  cudaStream_t stream = handle.get_stream().get();
  auto exec_policy    = handle.get_thrust_policy();

  // Root is the last edge in the dendrogram
  value_idx root = 2 * (n_leaves - 1);

  auto d_ptr           = thrust::device_pointer_cast(children);
  value_idx n_vertices = *(thrust::max_element(exec_policy, d_ptr, d_ptr + root)) + 1;

  // Prevent potential infinite loop from labeling disconnected
  // connectivities graph.
  RAFT_EXPECTS(n_vertices == root,
               "Multiple components found in MST or MST is invalid. "
               "Cannot find single-linkage solution. Found %lld vertices "
               "total.",
               static_cast<long long>(n_vertices));

  // Allocate output arrays
  rmm::device_uvector<value_idx> out_parent((root + 1) * 2, stream);
  rmm::device_uvector<value_idx> out_child((root + 1) * 2, stream);
  rmm::device_uvector<value_t> out_lambda((root + 1) * 2, stream);
  rmm::device_uvector<value_idx> out_size((root + 1) * 2, stream);

  thrust::fill(exec_policy, out_parent.begin(), out_parent.end(), -1);
  thrust::fill(exec_policy, out_child.begin(), out_child.end(), -1);
  thrust::fill(exec_policy, out_lambda.begin(), out_lambda.end(), -1);
  thrust::fill(exec_policy, out_size.begin(), out_size.end(), -1);

  _build_condensed_hierarchy(handle,
                             children,
                             delta,
                             sizes,
                             n_leaves,
                             min_cluster_size,
                             out_parent,
                             out_child,
                             out_lambda,
                             out_size);

  condensed_tree.condense(out_parent.data(), out_child.data(), out_lambda.data(), out_size.data());
}

};  // end namespace Condense
};  // end namespace detail
};  // end namespace HDBSCAN
};  // end namespace ML
