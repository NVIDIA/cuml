/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/condense.cuh"
#include "detail/extract.cuh"
#include "detail/reachability.cuh"
#include "detail/soft_clustering.cuh"

#include <cuml/cluster/hdbscan.hpp>
#include <cuml/common/distance_type.hpp>
#include <cuml/common/logger.hpp>

#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/kvp.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/sparse/coo.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>

#include <cuvs/cluster/agglomerative.hpp>

namespace ML {
namespace HDBSCAN {

/**
 * Constructs a linkage by computing mutual reachability, mst, and
 * dendrogram. This is shared by HDBSCAN and Robust Single Linkage
 * since the two algorithms differ only in the cluster
 * selection and extraction.
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle raft handle for resource reuse
 * @param[in] X data points (size m * n)
 * @param[in] m number of rows
 * @param[in] n number of columns
 * @param[in] metric distance metric to use
 * @param[in] params hyper parameters
 * @param[in] core_dists buffer for storing core distances (size m)
 * @param[out] out output container object
 */
template <typename value_idx = int64_t, typename value_t = float>
void build_linkage(const raft::handle_t& handle,
                   const value_t* X,
                   size_t m,
                   size_t n,
                   ML::distance::DistanceType metric,
                   Common::HDBSCANParams& params,
                   value_t* core_dists,
                   Common::robust_single_linkage_output<value_idx, value_t>& out)
{
  auto stream    = handle.get_stream();
  size_t n_edges = m - 1;
  cuvs::cluster::agglomerative::helpers::linkage_graph_params::mutual_reachability_params
    linkage_params;
  // (min_samples+1) is used to account for self-loops in the KNN graph
  // and be consistent with scikit-learn-contrib.
  if (static_cast<size_t>(params.min_samples + 1) > m) {
    RAFT_LOG_WARN(
      "min_samples (%d) must be less than the number of samples in X (%zu), setting min_samples to "
      "%zu",
      params.min_samples,
      m,
      m - 1);
    linkage_params.min_samples = m;
  } else {
    linkage_params.min_samples = params.min_samples + 1;
  }
  linkage_params.alpha                       = params.alpha;
  linkage_params.all_neighbors_params.metric = static_cast<cuvs::distance::DistanceType>(metric);
  linkage_params.all_neighbors_params.overlap_factor = params.build_params.overlap_factor;
  linkage_params.all_neighbors_params.n_clusters     = params.build_params.n_clusters;

  if (params.build_algo == Common::GRAPH_BUILD_ALGO::BRUTE_FORCE_KNN) {
    auto brute_force_params = cuvs::neighbors::graph_build_params::brute_force_params{};
    brute_force_params.build_params.metric = static_cast<cuvs::distance::DistanceType>(metric);
    linkage_params.all_neighbors_params.graph_build_params = brute_force_params;
  } else if (params.build_algo == Common::GRAPH_BUILD_ALGO::NN_DESCENT) {
    auto nn_descent_params           = cuvs::neighbors::graph_build_params::nn_descent_params{};
    nn_descent_params.max_iterations = params.build_params.nn_descent_params.max_iterations;
    nn_descent_params.graph_degree   = params.build_params.nn_descent_params.graph_degree;
    nn_descent_params.intermediate_graph_degree =
      params.build_params.nn_descent_params.intermediate_graph_degree;
    nn_descent_params.termination_threshold =
      params.build_params.nn_descent_params.termination_threshold;
    if (nn_descent_params.graph_degree < static_cast<std::size_t>(linkage_params.min_samples)) {
      // linkage_params.min_samples functions as the k for the knn
      RAFT_LOG_WARN(
        "NN Descent graph_degree (%zu) must be larger than or equal to min_samples + 1 (%zu), "
        "setting graph_degree to min_samples + 1 and scaling intermediate_graph_degree accordingly "
        "to 2*graph_degree",
        static_cast<std::size_t>(nn_descent_params.graph_degree),
        static_cast<std::size_t>(linkage_params.min_samples));
      nn_descent_params.graph_degree              = linkage_params.min_samples;
      nn_descent_params.intermediate_graph_degree = nn_descent_params.graph_degree * 2;
    }
    nn_descent_params.metric = static_cast<cuvs::distance::DistanceType>(metric);
    linkage_params.all_neighbors_params.graph_build_params = nn_descent_params;
  } else {
    RAFT_FAIL("Unsupported build algo. build algo must be either brute force or nn descent");
  }

  cudaPointerAttributes attr;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&attr, X));
  bool data_on_device = attr.type == cudaMemoryTypeDevice;

  if (data_on_device) {
    cuvs::cluster::agglomerative::helpers::build_linkage(
      handle,
      raft::make_device_matrix_view<const value_t, value_idx>(X, m, n),
      linkage_params,
      static_cast<cuvs::distance::DistanceType>(metric),
      raft::make_device_coo_matrix_view<value_t, value_idx, value_idx, size_t>(
        out.get_mst_weights(),
        raft::make_device_coordinate_structure_view(
          out.get_mst_src(), out.get_mst_dst(), value_idx(m), value_idx(m), n_edges)),
      raft::make_device_matrix_view<value_idx, value_idx>(out.get_children(), n_edges, 2),
      raft::make_device_vector_view<value_t, value_idx>(out.get_deltas(), n_edges),
      raft::make_device_vector_view<value_idx, value_idx>(out.get_sizes(), n_edges),
      std::make_optional<raft::device_vector_view<value_t, value_idx>>(
        raft::make_device_vector_view<value_t, value_idx>(core_dists, m)));
  } else {
    cuvs::cluster::agglomerative::helpers::build_linkage(
      handle,
      raft::make_host_matrix_view<const value_t, value_idx>(X, m, n),
      linkage_params,
      static_cast<cuvs::distance::DistanceType>(metric),
      raft::make_device_coo_matrix_view<value_t, value_idx, value_idx, size_t>(
        out.get_mst_weights(),
        raft::make_device_coordinate_structure_view(
          out.get_mst_src(), out.get_mst_dst(), value_idx(m), value_idx(m), n_edges)),
      raft::make_device_matrix_view<value_idx, value_idx>(out.get_children(), n_edges, 2),
      raft::make_device_vector_view<value_t, value_idx>(out.get_deltas(), n_edges),
      raft::make_device_vector_view<value_idx, value_idx>(out.get_sizes(), n_edges),
      std::make_optional<raft::device_vector_view<value_t, value_idx>>(
        raft::make_device_vector_view<value_t, value_idx>(core_dists, m)));
  }
}

template <typename value_idx = int64_t, typename value_t = float>
void _fit_hdbscan(const raft::handle_t& handle,
                  const value_t* X,
                  size_t m,
                  size_t n,
                  ML::distance::DistanceType metric,
                  Common::HDBSCANParams& params,
                  value_idx* labels,
                  value_t* core_dists,
                  Common::hdbscan_output<value_idx, value_t>& out)
{
  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  int min_cluster_size = params.min_cluster_size;

  RAFT_EXPECTS(params.min_samples <= m, "min_samples must be at most the number of samples in X");

  build_linkage(handle, X, m, n, metric, params, core_dists, out);

  /**
   * Condense branches of tree according to min cluster size
   */
  detail::Condense::build_condensed_hierarchy(handle,
                                              out.get_children(),
                                              out.get_deltas(),
                                              out.get_sizes(),
                                              min_cluster_size,
                                              m,
                                              out.get_condensed_tree());

  /**
   * Extract labels from stability
   */

  rmm::device_uvector<value_t> tree_stabilities(out.get_condensed_tree().get_n_clusters(),
                                                handle.get_stream());

  rmm::device_uvector<value_idx> label_map(out.get_condensed_tree().get_n_clusters(),
                                           handle.get_stream());
  value_idx n_selected_clusters =
    detail::Extract::extract_clusters(handle,
                                      out.get_condensed_tree(),
                                      m,
                                      labels,
                                      tree_stabilities.data(),
                                      out.get_probabilities(),
                                      label_map.data(),
                                      params.cluster_selection_method,
                                      out._get_inverse_label_map(),
                                      params.allow_single_cluster,
                                      static_cast<value_idx>(params.max_cluster_size),
                                      params.cluster_selection_epsilon);

  out.set_n_clusters(n_selected_clusters);

  auto lambdas_ptr   = thrust::device_pointer_cast(out.get_condensed_tree().get_lambdas());
  value_t max_lambda = *(thrust::max_element(
    exec_policy, lambdas_ptr, lambdas_ptr + out.get_condensed_tree().get_n_edges()));

  detail::Stability::get_stability_scores(handle,
                                          labels,
                                          tree_stabilities.data(),
                                          out.get_condensed_tree().get_n_clusters(),
                                          max_lambda,
                                          m,
                                          out.get_stabilities(),
                                          label_map.data());

  /**
   * Normalize labels so they are drawn from a monotonically increasing set
   * starting at 0 even in the presence of noise (-1)
   */

  thrust::transform(exec_policy,
                    labels,
                    labels + m,
                    out.get_labels(),
                    [label_map = label_map.data()] __device__(value_idx label) {
                      if (label != -1) return label_map[label];
                      return static_cast<value_idx>(-1);
                    });
}

};  // end namespace HDBSCAN
};  // end namespace ML
