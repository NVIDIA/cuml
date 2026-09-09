/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "kernels/select.cuh"
#include "utils.cuh"

#include <cuml/cluster/hdbscan.hpp>

#include <raft/label/classlabels.cuh>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/op/sort.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include <algorithm>

namespace ML {
namespace HDBSCAN {
namespace detail {
namespace Select {

/**
 * Given a frontier, iteratively performs a breadth-first search,
 * launching the given kernel at each level.
 * @tparam value_idx
 * @tparam Bfs_Kernel
 * @tparam tpb
 * @param[in] handle raft handle for resource reuse
 * @param[in] indptr CSR indptr of children array (size n_clusters+1)
 * @param[in] children children hierarchy array (size n_clusters)
 * @param[inout] frontier array storing which nodes need to be processed
 *               in each kernel invocation (size n_clusters)
 * @param[inout] is_cluster array of cluster selection / deselections (size n_clusters)
 * @param[in] n_clusters number of clusters
 * @param[in] bfs_kernel kernel accepting indptr, children, frontier, is_cluster, and n_clusters
 */
template <typename value_idx, typename Bfs_Kernel, int tpb = 256>
void perform_bfs(const raft::handle_t& handle,
                 const value_idx* indptr,
                 const value_idx* children,
                 int* frontier,
                 int* is_cluster,
                 int n_clusters,
                 Bfs_Kernel bfs_kernel)
{
  auto stream        = handle.get_stream().get();
  auto thrust_policy = handle.get_thrust_policy();

  rmm::device_uvector<int> next_frontier(n_clusters, stream);
  thrust::fill(thrust_policy, next_frontier.begin(), next_frontier.end(), 0);

  value_idx n_elements_to_traverse =
    thrust::reduce(thrust_policy, frontier, frontier + n_clusters, 0);

  // TODO: Investigate whether it's worth gathering the sparse frontier into
  // a dense form for purposes of uniform workload/thread scheduling

  // While frontier is not empty, perform single bfs through tree
  size_t grid = raft::ceildiv(n_clusters, tpb);

  while (n_elements_to_traverse > 0) {
    bfs_kernel<<<grid, tpb, 0, stream>>>(
      indptr, children, frontier, next_frontier.data(), is_cluster, n_clusters);

    thrust::copy(thrust_policy, next_frontier.begin(), next_frontier.end(), frontier);
    thrust::fill(thrust_policy, next_frontier.begin(), next_frontier.end(), 0);

    n_elements_to_traverse = thrust::reduce(thrust_policy, frontier, frontier + n_clusters, 0);
    handle.sync_stream(stream);
  }
}

/**
 * Computes a CSR index of parents of cluster tree. CSR index is
 * created by sorting parents by (children, sizes)
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle raft handle for resource reuse
 * @param[inout] cluster_tree cluster tree (condensed hierarchy with all nodes of size > 1)
 * @param[in] n_clusters number of clusters
 * @param[out] indptr CSR indptr of parents array after sort
 */
template <typename value_idx, typename value_t>
void parent_csr(const raft::handle_t& handle,
                Common::CondensedHierarchy<value_idx, value_t>& cluster_tree,
                value_idx* indptr)
{
  auto stream = handle.get_stream().get();

  auto parents            = cluster_tree.get_parents();
  auto children           = cluster_tree.get_children();
  auto sizes              = cluster_tree.get_sizes();
  auto cluster_tree_edges = cluster_tree.get_n_edges();
  auto n_clusters         = cluster_tree.get_n_clusters();

  if (cluster_tree_edges > 0) {
    raft::sparse::op::coo_sort(static_cast<value_idx>(0),
                               static_cast<value_idx>(0),
                               cluster_tree_edges,
                               parents,
                               children,
                               sizes,
                               stream);

    raft::sparse::convert::sorted_coo_to_csr(
      parents, cluster_tree_edges, indptr, n_clusters + 1, stream);
  } else {
    thrust::fill(handle.get_thrust_policy(), indptr, indptr + n_clusters + 1, 0);
  }
}

/**
 * Computes the excess of mass. This is a cluster selection
 * strategy that iterates upwards from the leaves of the cluster
 * tree toward the root, selecting clusters based on stabilities and size.
 * @tparam value_idx
 * @tparam value_t
 * @tparam tpb
 * @param[in] handle raft handle for resource reuse
 * @param[inout] cluster_tree condensed hierarchy containing only nodes of size > 1
 * @param[in] stability an array of nodes from the cluster tree and their
 *            corresponding stabilities
 * @param[out] is_cluster array of cluster selections / deselections (size n_clusters)
 * @param[in] n_clusters number of clusters in cluster tree
 * @param[in] max_cluster_size max number of points in a cluster before
 *            it will be deselected (and children selected)
 */
template <typename value_idx, typename value_t, int tpb = 256>
void excess_of_mass(const raft::handle_t& handle,
                    Common::CondensedHierarchy<value_idx, value_t>& cluster_tree,
                    value_t* stability,
                    int* is_cluster,
                    int n_clusters,
                    value_idx max_cluster_size,
                    bool allow_single_cluster)
{
  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  auto cluster_tree_edges = cluster_tree.get_n_edges();
  auto parents            = cluster_tree.get_parents();
  auto children           = cluster_tree.get_children();
  auto lambdas            = cluster_tree.get_lambdas();
  auto sizes              = cluster_tree.get_sizes();

  rmm::device_uvector<value_idx> cluster_sizes(n_clusters, stream);

  thrust::fill(exec_policy, cluster_sizes.data(), cluster_sizes.data() + cluster_sizes.size(), 0);

  value_idx* cluster_sizes_ptr = cluster_sizes.data();

  auto out = thrust::make_zip_iterator(cuda::std::make_tuple(parents, children, sizes));
  thrust::for_each(exec_policy,
                   out,
                   out + cluster_tree_edges,
                   [=] __device__(const cuda::std::tuple<value_idx, value_idx, value_idx>& tup) {
                     // if parent is root (0), add to cluster_sizes_ptr
                     if (cuda::std::get<0>(tup) == 0)
                       cluster_sizes_ptr[0] += cuda::std::get<2>(tup);

                     cluster_sizes_ptr[cuda::std::get<1>(tup)] = cuda::std::get<2>(tup);
                   });

  /**
   * 2. Iterate through each level from leaves back to root. Use the cluster
   *    tree CSR and warp-level reduction to sum stabilities and test whether
   *    or not current cluster should continue to be its own
   */
  std::vector<int> is_cluster_h(n_clusters, true);
  // setting the selection of root
  is_cluster_h[0] = allow_single_cluster;
  std::vector<int> frontier_h(n_clusters, false);
  std::vector<value_idx> cluster_sizes_h(n_clusters);

  rmm::device_uvector<value_idx> indptr(n_clusters + 1, stream);
  parent_csr(handle, cluster_tree, indptr.data());

  raft::update_host(cluster_sizes_h.data(), cluster_sizes.data(), cluster_sizes.size(), stream);

  std::vector<value_idx> indptr_h(indptr.size(), 0);
  if (cluster_tree_edges > 0)
    raft::update_host(indptr_h.data(), indptr.data(), indptr.size(), stream);
  handle.sync_stream(stream);

  // Loop through stabilities in "reverse topological order" (e.g. reverse sorted order)
  value_idx tree_top = allow_single_cluster ? 0 : 1;
  for (value_idx node = n_clusters - 1; node >= tree_top; node--) {
    value_t node_stability = 0.0;

    raft::update_host(&node_stability, stability + node, 1, stream);

    value_t subtree_stability = 0.0;

    if (indptr_h[node + 1] - indptr_h[node] > 0) {
      subtree_stability =
        thrust::transform_reduce(exec_policy,
                                 children + indptr_h[node],
                                 children + indptr_h[node + 1],
                                 cuda::proclaim_return_type<value_t>(
                                   [=] __device__(value_idx a) -> value_t { return stability[a]; }),
                                 0.0,
                                 cuda::std::plus<value_t>());
    }

    if (subtree_stability > node_stability || cluster_sizes_h[node] > max_cluster_size) {
      // Deselect / merge cluster with children
      raft::update_device(stability + node, &subtree_stability, 1, stream);
      is_cluster_h[node] = false;
    } else {
      // Mark children to be deselected
      frontier_h[node] = true;
    }
  }

  /**
   * 3. Perform BFS through is_cluster, propagating cluster
   * "deselection" through subtrees
   */
  rmm::device_uvector<int> cluster_propagate(n_clusters, stream);
  rmm::device_uvector<int> frontier(n_clusters, stream);

  raft::update_device(is_cluster, is_cluster_h.data(), n_clusters, stream);
  raft::update_device(frontier.data(), frontier_h.data(), n_clusters, stream);

  perform_bfs(handle,
              indptr.data(),
              children,
              frontier.data(),
              is_cluster,
              n_clusters,
              propagate_cluster_negation_kernel<value_idx>);
}

/**
 * Uses the leaves of the cluster tree as final cluster selections
 * @tparam value_idx
 * @tparam value_t
 * @tparam tpb
 * @param[in] handle raft handle for resource reuse
 * @param[inout] cluster_tree condensed hierarchy containing only nodes of size > 1
 * @param[out] is_cluster array of cluster selections / deselections (size n_clusters)
 * @param[in] n_clusters number of clusters in cluster tree
 */
template <typename value_idx, typename value_t, int tpb = 256>
void leaf(const raft::handle_t& handle,
          Common::CondensedHierarchy<value_idx, value_t>& cluster_tree,
          int* is_cluster,
          int n_clusters)
{
  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  auto parents  = cluster_tree.get_parents();
  auto children = cluster_tree.get_children();
  auto n_edges  = cluster_tree.get_n_edges();

  rmm::device_uvector<int> is_parent(n_clusters, stream);
  thrust::fill(exec_policy, is_parent.begin(), is_parent.end(), false);
  auto is_parent_op = [is_parent = is_parent.data()] __device__(auto& p) { is_parent[p] = true; };
  thrust::for_each(exec_policy, parents, parents + n_edges, is_parent_op);

  auto is_cluster_op = [is_parent = is_parent.data(), is_cluster = is_cluster] __device__(auto& c) {
    if (!is_parent[c]) { is_cluster[c] = true; }
  };
  thrust::for_each(exec_policy, children, children + n_edges, is_cluster_op);
}

/**
 * Selects clusters based on distance threshold.
 * @tparam value_idx
 * @tparam value_t
 * @tparam tpb
 * @param[in] handle raft handle for resource reuse
 * @param[in] cluster_tree condensed hierarchy with nodes of size > 1
 * @param[out] is_cluster array of cluster selections / deselections (size n_clusters)
 * @param[in] n_clusters number of clusters in cluster tree
 * @param[in] cluster_selection_epsilon distance threshold
 * @param[in] allow_single_cluster allows a single cluster with noisy datasets
 * @param[in] n_selected_clusters number of cluster selections in is_cluster
 */
template <typename value_idx, typename value_t, int tpb = 256>
void cluster_epsilon_search(const raft::handle_t& handle,
                            Common::CondensedHierarchy<value_idx, value_t>& cluster_tree,
                            int* is_cluster,
                            const int n_clusters,
                            const value_t cluster_selection_epsilon,
                            const bool allow_single_cluster,
                            const int n_selected_clusters)
{
  auto stream             = handle.get_stream().get();
  auto thrust_policy      = handle.get_thrust_policy();
  auto parents            = cluster_tree.get_parents();
  auto children           = cluster_tree.get_children();
  auto lambdas            = cluster_tree.get_lambdas();
  auto cluster_tree_edges = cluster_tree.get_n_edges();

  rmm::device_uvector<int> selected_clusters(n_selected_clusters, stream);

  // copying selected clusters by index
  thrust::copy_if(thrust_policy,
                  thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator(n_clusters),
                  is_cluster,
                  selected_clusters.data(),
                  [] __device__(auto cluster) { return cluster; });

  // sort lambdas and parents by children for epsilon search
  auto start = thrust::make_zip_iterator(cuda::std::make_tuple(parents, lambdas));
  thrust::sort_by_key(thrust_policy, children, children + cluster_tree_edges, start);
  rmm::device_uvector<value_t> eps(cluster_tree_edges, stream);
  thrust::transform(
    thrust_policy, lambdas, lambdas + cluster_tree_edges, eps.begin(), [] __device__(auto x) {
      return 1 / x;
    });

  // declare frontier and search
  rmm::device_uvector<int> frontier(n_clusters, stream);
  thrust::fill(thrust_policy, frontier.begin(), frontier.end(), false);

  auto nblocks = raft::ceildiv(n_selected_clusters, tpb);
  cluster_epsilon_search_kernel<<<nblocks, tpb, 0, stream>>>(selected_clusters.data(),
                                                             n_selected_clusters,
                                                             parents,
                                                             children,
                                                             lambdas,
                                                             cluster_tree_edges,
                                                             is_cluster,
                                                             frontier.data(),
                                                             n_clusters,
                                                             cluster_selection_epsilon,
                                                             allow_single_cluster);

  rmm::device_uvector<value_idx> indptr(n_clusters + 1, stream);
  parent_csr(handle, cluster_tree, indptr.data());

  perform_bfs(handle,
              indptr.data(),
              children,
              frontier.data(),
              is_cluster,
              n_clusters,
              propagate_cluster_negation_kernel<value_idx>);
}

/**
 * Entry point for end-to-end cluster selection logic
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle raft handle for resource reuse
 * @param[in] condensed_tree condensed hierarchy
 * @param[in] tree_stabilities stabilities array (size n_leaves from condensed hierarchy)
 * @param[out] is_cluster array of cluster selections / deselections (size n_clusters from condensed
 * hierarchy)
 * @param[in] cluster_selection_method method to use for selecting clusters
 * @param[in] allow_single_cluster whether a single cluster can be selected in noisy conditions
 * @param[in] max_cluster_size max size cluster to select before selecting children
 * @param[in] cluster_selection_epsilon distance threshold (0.0 disables distance selection)
 */
template <typename value_idx, typename value_t>
void select_clusters(const raft::handle_t& handle,
                     Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
                     value_t* tree_stabilities,
                     int* is_cluster,
                     Common::CLUSTER_SELECTION_METHOD cluster_selection_method,
                     bool allow_single_cluster,
                     value_idx max_cluster_size,
                     float cluster_selection_epsilon)
{
  auto stream        = handle.get_stream();
  auto thrust_policy = handle.get_thrust_policy();

  auto n_clusters = condensed_tree.get_n_clusters();

  auto cluster_tree = Utils::make_cluster_tree(handle, condensed_tree);

  if (cluster_selection_method == Common::CLUSTER_SELECTION_METHOD::EOM) {
    Select::excess_of_mass(handle,
                           cluster_tree,
                           tree_stabilities,
                           is_cluster,
                           n_clusters,
                           max_cluster_size,
                           allow_single_cluster);
  } else {
    thrust::fill(thrust_policy, is_cluster, is_cluster + n_clusters, false);

    if (cluster_tree.get_n_edges() > 0) {
      Select::leaf(handle, cluster_tree, is_cluster, n_clusters);
    }
  }

  auto n_selected_clusters = thrust::reduce(thrust_policy, is_cluster, is_cluster + n_clusters);

  // this variable is only used when cluster_selection_epsilon != 0.0
  auto epsilon_search = true;

  if (cluster_selection_method == Common::CLUSTER_SELECTION_METHOD::LEAF) {
    // TODO: re-enable to match reference implementation
    // It's a confirmed bug https://github.com/scikit-learn-contrib/hdbscan/issues/476

    // if no cluster leaves were found, declare root as cluster
    // if (n_selected_clusters == 0 && allow_single_cluster) {
    //   constexpr int root_is_cluster = true;
    //   raft::update_device(is_cluster, &root_is_cluster, 1, stream);
    //   epsilon_search = false;
    // }
  }

  if (cluster_selection_epsilon != 0.0 && cluster_tree.get_n_edges() > 0) {
    // no epsilon search if no clusters were selected
    if (n_selected_clusters == 0) { epsilon_search = false; }

    // this is to check when eom finds root as only cluster
    // in which case, epsilon search is cancelled
    if (cluster_selection_method == Common::CLUSTER_SELECTION_METHOD::EOM) {
      if (n_selected_clusters == 1) {
        int is_root_only_cluster = false;
        raft::update_host(&is_root_only_cluster, is_cluster, 1, stream);
        if (is_root_only_cluster && allow_single_cluster) { epsilon_search = false; }
      }
    }
    if (epsilon_search) {
      Select::cluster_epsilon_search(handle,
                                     cluster_tree,
                                     is_cluster,
                                     n_clusters,
                                     cluster_selection_epsilon,
                                     allow_single_cluster,
                                     n_selected_clusters);
    }
  }
}

};  // namespace Select
};  // namespace detail
};  // namespace HDBSCAN
};  // namespace ML
