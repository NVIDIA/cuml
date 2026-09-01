/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/common/utils.hpp>

namespace ML {
namespace HDBSCAN {
namespace detail {
namespace Condense {

template <typename value_idx, typename value_t>
__device__ inline value_t get_lambda(value_idx node, value_idx num_points, const value_t* deltas)
{
  value_t delta = deltas[node - num_points];
  return delta > value_t(0.0) ? value_t(1.0) / delta : std::numeric_limits<value_t>::max();
}

/**
 * Performs a breath-first search for a single level of the dendrogram, which
 * is a binary tree, and collapses subtrees based on the min_cluster_size. The
 * Arrays `relabel` and `ignore` are used to track state throughout subsequent
 * launches of this kernel. Nodes who's ancestors are "reassigned" inherit
 * the lambda value of their new parent.
 *
 * Note: This implementation differs from the reference implementation and
 * exposes more parallelism by having any collapsed branches directly
 * inherit the id of the persisted ancestor, rather than having to maintain
 * a synchronized monotonically increasing counter. In this version, a
 * renumbering is done afterwards, in parallel. The assumption here is that
 * a topological sort order should result from sorting the resulting
 * condensed dendrogram by (cluster size, id).
 *
 *
 * @tparam value_idx
 * @tparam value_t
 * @param frontier determines which nodes should be processed in
 *                 each iteration.
 * @param ignore Should be initialized to -1. maintains the lambda
 *               value of the new parent for each child of a subtree
 *               in the process of being collapsed. For example,
 *               ignore[5] = 0.9 means that all children of node w/
 *               id 5 should be placed in the condensed tree with
 *               parent relabel[5] and lambda=0.9.
 *
 * @param relabel relabel[0] should be initialized to root and
 *                propagated to subtrees as they are collapsed. This
 *                array stores the new parent that should be assigned
 *                for all nodes in a subtree that is in the process
 *                of being collapsed. For example, relabel[5] = 9
 *                means that node with id 5 should be assigned parent
 *                9 when ignore[5] > -1.
 * @param hierarchy binary tree dendrogram as renumbered by single-linkage
 *                  agglomerative labeling process.
 * @param deltas array of distances as constructed by the single-linkage
 *               agglomerative labeling process.
 * @param sizes  array of cluster sizes as constructed by the single-linkage
 *               agglomerative labeling process.
 * @param n_leaves number of non-cluster data points
 *
 * @param min_cluster_size while performing a bfs from the root of the
 *                         dendrogram, any subtrees below this size will
 *                         be collapsed into their parent cluster.
 *
 * @param out_parent parents array of output dendrogram. this will no longer
 *                   be a binary tree.
 * @param out_child children array of output dendrogram. this will no longer
 *                  be a binary tree.
 * @param out_lambda lambda array of output dendrogram.
 * @param out_count children cluster sizes of output dendrogram.
 */
template <typename value_idx, typename value_t>
CUML_KERNEL void condense_hierarchy_kernel(bool* frontier,
                                           bool* next_frontier,
                                           value_t* ignore,
                                           value_idx* relabel,
                                           const value_idx* children,
                                           const value_t* deltas,
                                           const value_idx* sizes,
                                           int n_leaves,
                                           int min_cluster_size,
                                           value_idx* out_parent,
                                           value_idx* out_child,
                                           value_t* out_lambda,
                                           value_idx* out_count)
{
  int node = blockDim.x * blockIdx.x + threadIdx.x;

  if (node >= n_leaves * 2 - 1 || !frontier[node]) return;

  frontier[node] = false;

  value_t subtree_lambda = ignore[node];

  bool should_ignore = subtree_lambda > -1;

  // TODO: Investigate whether this would be better done w/ an additional
  //  kernel launch

  // If node is a leaf, add it to the condensed hierarchy
  if (node < n_leaves) {
    out_parent[node * 2] = relabel[node];
    out_child[node * 2]  = node;
    out_lambda[node * 2] = subtree_lambda;
    out_count[node * 2]  = 1;
  }

  // If node is not a leaf, condense its children if necessary
  else {
    value_idx left_child  = children[(node - n_leaves) * 2];
    value_idx right_child = children[((node - n_leaves) * 2) + 1];

    // flip frontier for children
    next_frontier[left_child]  = true;
    next_frontier[right_child] = true;

    // propagate ignore down to children
    ignore[left_child]  = should_ignore ? subtree_lambda : -1;
    ignore[right_child] = should_ignore ? subtree_lambda : -1;

    relabel[left_child]  = should_ignore ? relabel[node] : relabel[left_child];
    relabel[right_child] = should_ignore ? relabel[node] : relabel[right_child];

    value_idx node_relabel = relabel[node];

    // TODO: Should be able to remove this nested conditional
    if (!should_ignore) {
      value_t lambda_value = get_lambda(node, n_leaves, deltas);

      int left_count  = left_child >= n_leaves ? sizes[left_child - n_leaves] : 1;
      int right_count = right_child >= n_leaves ? sizes[right_child - n_leaves] : 1;

      // Consume left or right child as necessary
      bool left_child_too_small  = left_count < min_cluster_size;
      bool right_child_too_small = right_count < min_cluster_size;

      // Node can "persist" to the cluster tree only if
      // both children >= min_cluster_size
      bool can_persist = !left_child_too_small && !right_child_too_small;

      relabel[left_child]  = !can_persist ? node_relabel : left_child;
      relabel[right_child] = !can_persist ? node_relabel : right_child;

      // set ignore for children. This is the node at which the "points underneath fall out"
      ignore[left_child]  = left_child_too_small ? lambda_value : -1;
      ignore[right_child] = right_child_too_small ? lambda_value : -1;

      // If both children are large enough, they should be relabeled and
      // included directly in the output hierarchy.
      if (can_persist) {
        // TODO: Could probably pull this out if this conditional becomes
        //  a bottleneck
        out_parent[node * 2] = node_relabel;
        out_child[node * 2]  = left_child;
        out_lambda[node * 2] = lambda_value;
        out_count[node * 2]  = left_count;

        out_parent[node * 2 + 1] = node_relabel;
        out_child[node * 2 + 1]  = right_child;
        out_lambda[node * 2 + 1] = lambda_value;
        out_count[node * 2 + 1]  = right_count;
      }
    }
  }
}

};  // end namespace Condense
};  // end namespace detail
};  // end namespace HDBSCAN
};  // end namespace ML
