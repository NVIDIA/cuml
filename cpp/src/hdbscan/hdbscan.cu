/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/condense.cuh"
#include "detail/predict.cuh"
#include "runner.cuh"

#include <cuml/cluster/hdbscan.hpp>
#include <cuml/common/distance_type.hpp>

#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

namespace ML {

void hdbscan(const raft::handle_t& handle,
             const float* X,
             size_t m,
             size_t n,
             ML::distance::DistanceType metric,
             HDBSCAN::Common::HDBSCANParams& params,
             HDBSCAN::Common::hdbscan_output<int64_t, float>& out,
             float* core_dists)
{
  rmm::device_uvector<int64_t> labels(m, handle.get_stream());
  HDBSCAN::_fit_hdbscan(handle, X, m, n, metric, params, labels.data(), core_dists, out);
}

void build_condensed_hierarchy(const raft::handle_t& handle,
                               const int64_t* children,
                               const float* delta,
                               const int64_t* sizes,
                               int min_cluster_size,
                               int n_leaves,
                               HDBSCAN::Common::CondensedHierarchy<int64_t, float>& condensed_tree)
{
  HDBSCAN::detail::Condense::build_condensed_hierarchy(
    handle, children, delta, sizes, min_cluster_size, n_leaves, condensed_tree);
}

void _extract_clusters(const raft::handle_t& handle,
                       size_t n_leaves,
                       int n_edges,
                       int64_t* parents,
                       int64_t* children,
                       float* lambdas,
                       int64_t* sizes,
                       int64_t* labels,
                       float* probabilities,
                       HDBSCAN::Common::CLUSTER_SELECTION_METHOD cluster_selection_method,
                       bool allow_single_cluster,
                       int64_t max_cluster_size,
                       float cluster_selection_epsilon)
{
  HDBSCAN::Common::CondensedHierarchy condensed_tree(
    handle, n_leaves, n_edges, parents, children, lambdas, sizes);

  rmm::device_uvector<float> stabilities(condensed_tree.get_n_clusters(), handle.get_stream());
  rmm::device_uvector<int64_t> label_map(condensed_tree.get_n_clusters(), handle.get_stream());
  rmm::device_uvector<int64_t> inverse_label_map(0, handle.get_stream());

  HDBSCAN::detail::Extract::extract_clusters(handle,
                                             condensed_tree,
                                             n_leaves,
                                             labels,
                                             stabilities.data(),
                                             probabilities,
                                             label_map.data(),
                                             cluster_selection_method,
                                             inverse_label_map,
                                             allow_single_cluster,
                                             max_cluster_size,
                                             cluster_selection_epsilon);
}

void compute_all_points_membership_vectors(
  const raft::handle_t& handle,
  HDBSCAN::Common::CondensedHierarchy<int64_t, float>& condensed_tree,
  HDBSCAN::Common::PredictionData<int64_t, float>& prediction_data,
  const float* X,
  ML::distance::DistanceType metric,
  float* membership_vec,
  size_t batch_size)
{
  HDBSCAN::detail::Predict::all_points_membership_vectors(
    handle, condensed_tree, prediction_data, X, metric, membership_vec, batch_size);
}

void compute_membership_vector(const raft::handle_t& handle,
                               HDBSCAN::Common::CondensedHierarchy<int64_t, float>& condensed_tree,
                               HDBSCAN::Common::PredictionData<int64_t, float>& prediction_data,
                               const float* X,
                               const float* points_to_predict,
                               size_t n_prediction_points,
                               int min_samples,
                               ML::distance::DistanceType metric,
                               float* membership_vec,
                               size_t batch_size)
{
  // Note that (min_samples+1) is parsed to the approximate_predict function. This was done for the
  // core distance computation to consistent with Scikit learn Contrib.
  HDBSCAN::detail::Predict::membership_vector(handle,
                                              condensed_tree,
                                              prediction_data,
                                              X,
                                              points_to_predict,
                                              n_prediction_points,
                                              metric,
                                              min_samples + 1,
                                              membership_vec,
                                              batch_size);
}

void out_of_sample_predict(const raft::handle_t& handle,
                           HDBSCAN::Common::CondensedHierarchy<int64_t, float>& condensed_tree,
                           HDBSCAN::Common::PredictionData<int64_t, float>& prediction_data,
                           const float* X,
                           int64_t* labels,
                           const float* points_to_predict,
                           size_t n_prediction_points,
                           ML::distance::DistanceType metric,
                           int min_samples,
                           int64_t* out_labels,
                           float* out_probabilities)
{
  // Note that (min_samples+1) is parsed to the approximate_predict function. This was done for the
  // core distance computation to consistent with Scikit learn Contrib.
  HDBSCAN::detail::Predict::approximate_predict(handle,
                                                condensed_tree,
                                                prediction_data,
                                                X,
                                                labels,
                                                points_to_predict,
                                                n_prediction_points,
                                                metric,
                                                min_samples + 1,
                                                out_labels,
                                                out_probabilities);
}

namespace HDBSCAN::HELPER {

void compute_core_dists(const raft::handle_t& handle,
                        const float* X,
                        float* core_dists,
                        size_t m,
                        size_t n,
                        ML::distance::DistanceType metric,
                        int min_samples)
{
  HDBSCAN::detail::Reachability::_compute_core_dists<int64_t, float>(
    handle, X, core_dists, m, n, metric, min_samples);
}

void compute_inverse_label_map(const raft::handle_t& handle,
                               HDBSCAN::Common::CondensedHierarchy<int64_t, float>& condensed_tree,
                               size_t n_leaves,
                               HDBSCAN::Common::CLUSTER_SELECTION_METHOD cluster_selection_method,
                               rmm::device_uvector<int64_t>& inverse_label_map,
                               bool allow_single_cluster,
                               int64_t max_cluster_size,
                               float cluster_selection_epsilon)
{
  HDBSCAN::detail::Extract::_compute_inverse_label_map(handle,
                                                       condensed_tree,
                                                       n_leaves,
                                                       cluster_selection_method,
                                                       inverse_label_map,
                                                       allow_single_cluster,
                                                       max_cluster_size,
                                                       cluster_selection_epsilon);
}

}  // end namespace HDBSCAN::HELPER
};  // end namespace ML
