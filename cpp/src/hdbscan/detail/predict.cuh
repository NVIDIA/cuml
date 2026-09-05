/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "kernels/predict.cuh"
#include "reachability.cuh"

#include <cuml/common/checked_arithmetic.hpp>
#include <cuml/common/distance_type.hpp>

#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

namespace ML {
namespace HDBSCAN {
namespace detail {
namespace Predict {

/**
Find the nearest mutual reachability neighbor of a point, and  compute
the associated lambda value for the point, given the mutual reachability
distance to a nearest neighbor.
 * @tparam value_idx
 * @tparam value_t
 * @tparam tpb
 * @param[in] handle raft handle for resource reuse
 * @param[in] input_core_dists an array of core distances for all points (size m)
 * @param[in] prediction_core_dists an array of core distances for all prediction points (size
n_prediction_points)
 * @param[in] knn_dists knn distance array (size n_prediction_points * neighborhood)
 * @param[in] knn_inds knn indices array (size n_prediction_points * neighborhood)
 * @param[in] n_prediction_points number of prediction points
 * @param[in] neighborhood the neighborhood of prediction points
 * @param[out] min_mr_inds indices of points with the minimum mutual reachability distance (size
n_prediction_points)
 * @param[out] prediction_lambdas lambda values for prediction points (size n_prediction_points)
 */
template <typename value_idx, typename value_t, int tpb = 256>
void _find_neighbor_and_lambda(const raft::handle_t& handle,
                               value_t* input_core_dists,
                               value_t* prediction_core_dists,
                               value_t* knn_dists,
                               value_idx* knn_inds,
                               size_t n_prediction_points,
                               int neighborhood,
                               value_idx* min_mr_inds,
                               value_t* prediction_lambdas)
{
  auto stream      = handle.get_stream().get();
  auto exec_policy = handle.get_thrust_policy();

  // Buffer for storing the minimum mutual reachability distances
  rmm::device_uvector<value_t> min_mr_dists(n_prediction_points, stream);

  auto const n_blocks =
    narrow_cast<ML::cuda_launch_t>(raft::ceildiv(narrow_cast<int>(n_prediction_points), tpb));

  // get nearest neighbors for each prediction point in mutual reachability space
  min_mutual_reachability_kernel<<<n_blocks, tpb, 0, stream>>>(input_core_dists,
                                                               prediction_core_dists,
                                                               knn_dists,
                                                               knn_inds,
                                                               n_prediction_points,
                                                               static_cast<value_idx>(neighborhood),
                                                               min_mr_dists.data(),
                                                               min_mr_inds);

  // obtain lambda values from minimum mutual reachability distances
  raft::linalg::map_offset(
    handle,
    raft::make_device_vector_view<value_t, value_idx>(prediction_lambdas, n_prediction_points),
    [min_mr_dists = min_mr_dists.data()] __device__(auto idx) {
      value_t dist = min_mr_dists[idx];
      if (dist > 0) return (1 / dist);
      return std::numeric_limits<value_t>::max();
    });
}

/**
 Return the cluster label (of the original clustering) and membership
 probability of a new data point.
 *
 * @tparam value_idx
 * @tparam value_t
 * @tparam tpb
 * @param[in] handle raft handle for resource reuse
 * @param[in] condensed_tree condensed hierarchy
 * @param[in] prediction_data PredictionData object
 * @param[in] n_prediction_points number of prediction points
 * @param[in] min_mr_inds indices of points with the minimum mutual reachability distance (size
 n_prediction_points)
 * @param[in] prediction_lambdas lambda values for prediction points (size n_prediction_points)
 * @param[in] labels monotonic labels of all points
 * @param[out] out_labels output cluster labels
 * @param[out] out_probabilities output probabilities
 */
template <typename value_idx, typename value_t, int tpb = 256>
void _find_cluster_and_probability(const raft::handle_t& handle,
                                   Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
                                   Common::PredictionData<value_idx, value_t>& prediction_data,
                                   size_t n_prediction_points,
                                   value_idx* min_mr_inds,
                                   value_t* prediction_lambdas,
                                   value_idx* labels,
                                   value_idx* out_labels,
                                   value_t* out_probabilities)
{
  auto stream      = handle.get_stream().get();
  auto exec_policy = handle.get_thrust_policy();

  auto parents     = condensed_tree.get_parents();
  auto children    = condensed_tree.get_children();
  value_t* lambdas = condensed_tree.get_lambdas();
  auto n_edges     = condensed_tree.get_n_edges();
  auto n_leaves    = condensed_tree.get_n_leaves();

  value_t* deaths                = prediction_data.get_deaths();
  value_idx* selected_clusters   = prediction_data.get_selected_clusters();
  value_idx* index_into_children = prediction_data.get_index_into_children();

  auto const n_blocks =
    narrow_cast<ML::cuda_launch_t>(raft::ceildiv(narrow_cast<int>(n_prediction_points), tpb));

  cluster_probability_kernel<<<n_blocks, tpb, 0, stream>>>(min_mr_inds,
                                                           prediction_lambdas,
                                                           index_into_children,
                                                           labels,
                                                           deaths,
                                                           selected_clusters,
                                                           parents,
                                                           lambdas,
                                                           n_leaves,
                                                           n_prediction_points,
                                                           out_labels,
                                                           out_probabilities);
}

// Build the mutual reachability graph and obtain the nearest neighbors for the prediction points.
// The KNN and core distances of prediction points are computed here.
template <typename value_idx, typename value_t, int tpb = 256>
void _compute_knn_and_nearest_neighbor(const raft::handle_t& handle,
                                       Common::PredictionData<value_idx, value_t>& prediction_data,
                                       const value_t* X,
                                       const value_t* points_to_predict,
                                       int min_samples,
                                       size_t n_prediction_points,
                                       value_idx* min_mr_inds,
                                       value_t* prediction_lambdas,
                                       ML::distance::DistanceType metric)
{
  auto stream               = handle.get_stream().get();
  size_t m                  = prediction_data.n_rows;
  size_t n                  = prediction_data.n_cols;
  value_t* input_core_dists = prediction_data.get_core_dists();
  int neighborhood          = (min_samples - 1) * 2;

  rmm::device_uvector<value_idx> inds(neighborhood * n_prediction_points, stream);
  rmm::device_uvector<value_t> dists(neighborhood * n_prediction_points, stream);
  rmm::device_uvector<value_t> prediction_core_dists(n_prediction_points, stream);

  // perform knn
  Reachability::compute_knn(handle,
                            X,
                            inds.data(),
                            dists.data(),
                            m,
                            n,
                            points_to_predict,
                            n_prediction_points,
                            neighborhood,
                            metric);

  // Slice core distances (distances to kth nearest neighbor). The index of the neighbor is
  // consistent with Scikit-learn Contrib
  Reachability::core_distances<value_idx>(dists.data(),
                                          min_samples,
                                          neighborhood,
                                          n_prediction_points,
                                          prediction_core_dists.data(),
                                          stream);

  _find_neighbor_and_lambda(handle,
                            input_core_dists,
                            prediction_core_dists.data(),
                            dists.data(),
                            inds.data(),
                            n_prediction_points,
                            neighborhood,
                            min_mr_inds,
                            prediction_lambdas);
}

/**
 * Predict the cluster label and the probability of the label for new points.
 * The returned labels are those of the original clustering,
 * and therefore are not (necessarily) the cluster labels that would
 * be found by clustering the original data combined with
 * the prediction points, hence the 'approximate' label.
 *
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle raft handle for resource reuse
 * @param[in] condensed_tree a condensed hierarchy
 * @param[in] prediction_data PredictionData object
 * @param[in] X input data points (size m * n)
 * @param[in] labels converted monotonic labels of the input data points
 * @param[in] points_to_predict input prediction points (size n_prediction_points * n)
 * @param[in] n_prediction_points number of prediction points
 * @param[in] metric distance metric
 * @param[in] min_samples neighborhood size during training (includes self-loop)
 * @param[out] out_labels output cluster labels
 * @param[out] out_probabilities output probabilities
 */
template <typename value_idx, typename value_t, int tpb = 256>
void approximate_predict(const raft::handle_t& handle,
                         Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
                         Common::PredictionData<value_idx, value_t>& prediction_data,
                         const value_t* X,
                         value_idx* labels,
                         const value_t* points_to_predict,
                         size_t n_prediction_points,
                         ML::distance::DistanceType metric,
                         int min_samples,
                         value_idx* out_labels,
                         value_t* out_probabilities)
{
  RAFT_EXPECTS(metric == ML::distance::DistanceType::L2SqrtExpanded,
               "Currently only L2 expanded distance is supported");

  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  // Obtain lambdas for each prediction point using the closest point in mutual reachability space
  rmm::device_uvector<value_t> prediction_lambdas(n_prediction_points, stream);
  rmm::device_uvector<value_idx> min_mr_inds(n_prediction_points, stream);
  _compute_knn_and_nearest_neighbor(handle,
                                    prediction_data,
                                    X,
                                    points_to_predict,
                                    min_samples,
                                    n_prediction_points,
                                    min_mr_inds.data(),
                                    prediction_lambdas.data(),
                                    metric);

  // Using the nearest neighbor indices, find the assigned cluster label and probability
  _find_cluster_and_probability(handle,
                                condensed_tree,
                                prediction_data,
                                n_prediction_points,
                                min_mr_inds.data(),
                                prediction_lambdas.data(),
                                labels,
                                out_labels,
                                out_probabilities);
}

};  // end namespace Predict
};  // end namespace detail
};  // end namespace HDBSCAN
};  // end namespace ML
