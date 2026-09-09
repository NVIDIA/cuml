/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "kernels/soft_clustering.cuh"
#include "select.cuh"
#include "utils.cuh"

#include <cuml/cluster/hdbscan.hpp>
#include <cuml/common/distance_type.hpp>
#include <cuml/common/logger.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/label/classlabels.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/matrix/argmax.cuh>
#include <raft/matrix/copy.cuh>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/op/sort.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/fast_int_div.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <thrust/execution_policy.h>

#include <cuvs/distance/distance.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace ML {
namespace HDBSCAN {
namespace detail {
namespace Predict {

// Computing distance based membership for points in the original clustering on which the clusterer
// was trained and new points outside of the training data.
template <typename value_idx, typename value_t>
void dist_membership_vector(const raft::handle_t& handle,
                            const value_t* X,
                            size_t n_X_rows,
                            const value_t* query,
                            size_t n_queries,
                            size_t n,
                            size_t n_exemplars,
                            value_idx n_selected_clusters,
                            value_idx* exemplar_idx,
                            value_idx* exemplar_label_offsets,
                            value_t* dist_membership_vec,
                            ML::distance::DistanceType metric,
                            size_t batch_size,
                            bool softmax = false)
{
  auto stream      = handle.get_stream().get();
  auto exec_policy = handle.get_thrust_policy();

  rmm::device_uvector<value_t> exemplars_dense(n_exemplars * n, stream);

  // use the exemplar point indices to obtain the exemplar points as a dense array
  // exemplar_idx indexes into X, which has n_X_rows rows (training data)
  auto X_view =
    raft::make_device_matrix_view<const value_t, value_idx, raft::row_major>(X, n_X_rows, n);
  auto exemplars_dense_view = raft::make_device_matrix_view<value_t, value_idx, raft::row_major>(
    exemplars_dense.data(), n_exemplars, n);
  auto exemplar_idx_view =
    raft::make_device_vector_view<const value_idx, value_idx>((exemplar_idx), n_exemplars);
  raft::matrix::copy_rows(handle, X_view, exemplars_dense_view, exemplar_idx_view);

  // compute the number of batches based on the batch size
  value_idx n_batches;

  n_batches = raft::ceildiv(ML::narrow_cast<int>(n_queries), ML::narrow_cast<int>(batch_size));

  for (value_idx bid = 0; bid < n_batches; bid++) {
    value_idx batch_offset      = bid * batch_size;
    value_idx samples_per_batch = min((value_idx)batch_size, (value_idx)n_queries - batch_offset);
    rmm::device_uvector<value_t> dist(samples_per_batch * n_exemplars, stream);

    // compute the distances using the CUVS API
    cuvs::distance::pairwise_distance(
      handle,
      raft::make_device_matrix_view<const value_t, int64_t>(
        query + batch_offset * n, samples_per_batch, n),
      raft::make_device_matrix_view<const value_t, int64_t>(exemplars_dense.data(), n_exemplars, n),
      raft::make_device_matrix_view<value_t, int64_t>(dist.data(), samples_per_batch, n_exemplars),
      static_cast<cuvs::distance::DistanceType>(metric));

    // compute the minimum distances to exemplars of each cluster
    value_idx n_elements = samples_per_batch * n_selected_clusters;
    auto min_dist        = raft::make_device_vector<value_t, value_idx>(handle, n_elements);

    auto reduction_op = [dist = dist.data(),
                         batch_offset,
                         divisor = raft::util::FastIntDiv<value_idx>(n_selected_clusters),
                         n_selected_clusters,
                         n_exemplars,
                         exemplar_label_offsets] __device__(auto idx) {
      auto col   = idx % divisor;
      auto row   = idx / divisor;
      auto start = exemplar_label_offsets[col];
      auto end   = exemplar_label_offsets[col + 1];

      value_t min_val = std::numeric_limits<value_t>::max();
      for (value_idx i = start; i < end; i++) {
        if (dist[row * n_exemplars + i] < min_val) { min_val = dist[row * n_exemplars + i]; }
      }
      return min_val;
    };

    raft::linalg::map_offset(handle, min_dist.view(), reduction_op);

    // Softmax computation is ignored in distance membership
    if (softmax) {
      raft::linalg::map_offset(handle,
                               raft::make_device_vector_view<value_t, value_idx>(
                                 dist_membership_vec + batch_offset * n_selected_clusters,
                                 samples_per_batch * n_selected_clusters),
                               [min_dist = min_dist.data_handle()] __device__(auto idx) {
                                 value_t val = min_dist[idx];
                                 if (val != 0) { return value_t(exp(1.0 / val)); }
                                 return std::numeric_limits<value_t>::max();
                               });
    }

    // Transform the distances to obtain membership based on proximity to exemplars
    else {
      raft::linalg::map_offset(
        handle,
        raft::make_device_vector_view<value_t, value_idx>(
          dist_membership_vec + batch_offset * n_selected_clusters,
          samples_per_batch * n_selected_clusters),
        [min_dist = min_dist.data_handle(), n_selected_clusters] __device__(auto idx) {
          value_t val = min_dist[idx];
          if (val > 0) { return value_t(1.0 / val); }
          return std::numeric_limits<value_t>::max() / n_selected_clusters;
        });
    }
  }
  // Normalize the obtained result to sum to 1.0
  Utils::normalize(dist_membership_vec, n_selected_clusters, n_queries, stream);
}

template <typename value_idx, typename value_t, int tpb = 256>
void all_points_outlier_membership_vector(
  const raft::handle_t& handle,
  Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
  value_t* deaths,
  value_idx* selected_clusters,
  value_idx* index_into_children,
  size_t m,
  size_t n_selected_clusters,
  value_t* merge_heights,
  value_t* outlier_membership_vec,
  bool softmax)
{
  auto stream      = handle.get_stream().get();
  auto exec_policy = handle.get_thrust_policy();

  auto parents      = condensed_tree.get_parents();
  auto children     = condensed_tree.get_children();
  value_t* lambdas  = condensed_tree.get_lambdas();
  value_idx n_edges = condensed_tree.get_n_edges();
  auto n_clusters   = condensed_tree.get_n_clusters();
  auto n_leaves     = condensed_tree.get_n_leaves();

  int n_blocks = raft::ceildiv(int(m * n_selected_clusters), tpb);
  merge_height_kernel<<<n_blocks, tpb, 0, stream>>>(
    merge_heights,
    lambdas,
    index_into_children,
    parents,
    m,
    static_cast<value_idx>(n_selected_clusters),
    raft::util::FastIntDiv<value_idx>(static_cast<value_idx>(n_selected_clusters)),
    selected_clusters);

  auto leaf_max_lambdas = raft::make_device_vector<value_t, value_idx>(handle, n_leaves);

  raft::linalg::map_offset(handle,
                           leaf_max_lambdas.view(),
                           [deaths, parents, index_into_children, n_leaves] __device__(auto idx) {
                             return deaths[parents[index_into_children[idx]] - n_leaves];
                           });

  raft::linalg::matrixVectorOp<true, false>(
    outlier_membership_vec,
    merge_heights,
    leaf_max_lambdas.data_handle(),
    static_cast<value_idx>(n_selected_clusters),
    static_cast<value_idx>(m),
    [] __device__(value_t mat_in, value_t vec_in) {
      return exp(-(vec_in + 1e-8) / mat_in);
    },  //+ 1e-8 to avoid zero lambda
    stream);

  if (softmax) { Utils::softmax(handle, outlier_membership_vec, n_selected_clusters, m); }

  Utils::normalize(outlier_membership_vec, n_selected_clusters, m, stream);
}

template <typename value_idx, typename value_t, int tpb = 256>
void all_points_prob_in_some_cluster(const raft::handle_t& handle,
                                     Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
                                     value_t* deaths,
                                     value_idx* selected_clusters,
                                     value_idx* index_into_children,
                                     size_t m,
                                     value_idx n_selected_clusters,
                                     value_t* merge_heights,
                                     value_t* prob_in_some_cluster)
{
  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  value_t* lambdas = condensed_tree.get_lambdas();
  auto n_leaves    = condensed_tree.get_n_leaves();
  auto n_edges     = condensed_tree.get_n_edges();
  auto children    = condensed_tree.get_children();

  auto height_argmax = raft::make_device_vector<value_idx, value_idx>(handle, m);

  auto merge_heights_view =
    raft::make_device_matrix_view<const value_t, value_idx, raft::row_major>(
      merge_heights, ML::narrow_cast<int>(m), n_selected_clusters);

  raft::matrix::argmax(handle, merge_heights_view, height_argmax.view());

  auto prob_in_some_cluster_op = [deaths,
                                  lambdas,
                                  index_into_children,
                                  selected_clusters,
                                  n_leaves,
                                  merge_heights,
                                  height_argmax = height_argmax.data_handle(),
                                  n_selected_clusters] __device__(auto idx) {
    value_idx nearest_cluster = height_argmax[idx];
    value_t max_lambda =
      max(lambdas[index_into_children[idx]], deaths[selected_clusters[nearest_cluster] - n_leaves]);
    return merge_heights[idx * n_selected_clusters + nearest_cluster] / max_lambda;
  };
  raft::linalg::map_offset(
    handle,
    raft::make_device_vector_view<value_t, value_idx>(prob_in_some_cluster, m),
    prob_in_some_cluster_op);
}

template <typename value_idx, typename value_t, int tpb = 256>
void outlier_membership_vector(const raft::handle_t& handle,
                               Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
                               value_t* deaths,
                               value_idx* min_mr_inds,
                               value_t* prediction_lambdas,
                               value_idx* selected_clusters,
                               value_idx* index_into_children,
                               size_t n_prediction_points,
                               size_t n_selected_clusters,
                               value_t* merge_heights,
                               value_t* outlier_membership_vec,
                               bool softmax)
{
  auto stream      = handle.get_stream().get();
  auto exec_policy = handle.get_thrust_policy();

  auto parents      = condensed_tree.get_parents();
  auto children     = condensed_tree.get_children();
  value_t* lambdas  = condensed_tree.get_lambdas();
  value_idx n_edges = condensed_tree.get_n_edges();
  auto n_clusters   = condensed_tree.get_n_clusters();
  auto n_leaves     = condensed_tree.get_n_leaves();

  // Using the nearest neighbor indices, compute outlier membership
  int n_blocks = raft::ceildiv(int(n_prediction_points * n_selected_clusters), tpb);
  merge_height_kernel<<<n_blocks, tpb, 0, stream>>>(
    merge_heights,
    lambdas,
    prediction_lambdas,
    min_mr_inds,
    index_into_children,
    parents,
    n_prediction_points,
    static_cast<value_idx>(n_selected_clusters),
    raft::util::FastIntDiv<value_idx>(static_cast<value_idx>(n_selected_clusters)),
    selected_clusters);

  // fetch the max lambda of the cluster to which the nearest MR neighbor belongs in the condensed
  // hierarchy

  auto nearest_cluster_max_lambda =
    raft::make_device_vector<value_t, value_idx>(handle, n_prediction_points);
  raft::linalg::map_offset(
    handle,
    nearest_cluster_max_lambda.view(),
    [deaths, parents, index_into_children, min_mr_inds, n_leaves] __device__(auto idx) {
      return deaths[parents[index_into_children[min_mr_inds[idx]]] - n_leaves];
    });

  raft::linalg::matrixVectorOp<true, false>(
    outlier_membership_vec,
    merge_heights,
    nearest_cluster_max_lambda.data_handle(),
    n_selected_clusters,
    n_prediction_points,
    [] __device__(value_t mat_in, value_t vec_in) {
      value_t denominator = vec_in - mat_in;
      if (denominator <= 0) { denominator = 1e-8; }
      return vec_in / denominator;
    },
    stream);

  if (softmax) {
    Utils::softmax(handle, outlier_membership_vec, n_selected_clusters, n_prediction_points);
  }
  Utils::normalize(outlier_membership_vec, n_selected_clusters, n_prediction_points, stream);
}

template <typename value_idx, typename value_t, int tpb = 256>
void prob_in_some_cluster(const raft::handle_t& handle,
                          Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
                          value_t* deaths,
                          value_idx* selected_clusters,
                          value_idx* index_into_children,
                          size_t n_prediction_points,
                          value_idx n_selected_clusters,
                          value_idx* min_mr_indices,
                          value_t* merge_heights,
                          value_t* prediction_lambdas,
                          value_t* prob_in_some_cluster)
{
  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  value_t* lambdas = condensed_tree.get_lambdas();
  auto n_leaves    = condensed_tree.get_n_leaves();
  auto n_edges     = condensed_tree.get_n_edges();
  auto children    = condensed_tree.get_children();

  auto height_argmax = raft::make_device_vector<value_idx, value_idx>(handle, n_prediction_points);

  auto merge_heights_view =
    raft::make_device_matrix_view<const value_t, value_idx, raft::row_major>(
      merge_heights, ML::narrow_cast<int>(n_prediction_points), n_selected_clusters);

  raft::matrix::argmax(handle, merge_heights_view, height_argmax.view());

  auto prob_in_some_cluster_op = [prediction_lambdas,
                                  deaths,
                                  selected_clusters,
                                  n_leaves,
                                  merge_heights,
                                  height_argmax = height_argmax.data_handle(),
                                  n_selected_clusters] __device__(auto idx) {
    value_idx nearest_cluster = height_argmax[idx];
    value_t max_lambda =
      max(prediction_lambdas[idx], deaths[selected_clusters[nearest_cluster] - n_leaves]) + 1e-8;
    return merge_heights[idx * n_selected_clusters + nearest_cluster] / max_lambda;
  };
  raft::linalg::map_offset(
    handle,
    raft::make_device_vector_view<value_t, value_idx>(prob_in_some_cluster, n_prediction_points),
    prob_in_some_cluster_op);
}

/**
 * Predict soft cluster membership vectors for all points in the original dataset the clusterer was
 * trained on
 *
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle raft handle for resource reuse
 * @param[in] condensed_tree a condensed hierarchy
 * @param[in] prediction_data PredictionData object
 * @param[in] X all points (size m * n)
 * @param[in] metric distance metric
 * @param[out] membership_vec output membership vectors (size m * n_selected_clusters)
 * @param[in] batch_size batch size to be used while computing distance based memberships
 */
template <typename value_idx, typename value_t>
void all_points_membership_vectors(const raft::handle_t& handle,
                                   Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
                                   Common::PredictionData<value_idx, value_t>& prediction_data,
                                   const value_t* X,
                                   ML::distance::DistanceType metric,
                                   value_t* membership_vec,
                                   size_t batch_size)
{
  auto stream      = handle.get_stream().get();
  auto exec_policy = handle.get_thrust_policy();

  size_t m = prediction_data.n_rows;
  size_t n = prediction_data.n_cols;

  if (batch_size > m) batch_size = m;
  RAFT_EXPECTS(0 < batch_size && batch_size <= m,
               "Invalid batch_size. batch_size should be > 0 and <= the number of samples in the "
               "training data");

  auto parents    = condensed_tree.get_parents();
  auto children   = condensed_tree.get_children();
  auto lambdas    = condensed_tree.get_lambdas();
  auto n_edges    = condensed_tree.get_n_edges();
  auto n_clusters = condensed_tree.get_n_clusters();
  auto n_leaves   = condensed_tree.get_n_leaves();

  value_idx n_selected_clusters  = prediction_data.get_n_selected_clusters();
  value_t* deaths                = prediction_data.get_deaths();
  value_idx* selected_clusters   = prediction_data.get_selected_clusters();
  value_idx* index_into_children = prediction_data.get_index_into_children();
  value_idx n_exemplars          = prediction_data.get_n_exemplars();

  // Compute membership vectors only if the number of selected clusters is non-zero. This is done to
  // avoid CUDA run-time errors in raft primitives for pairwise distances and other kernel
  // invocations.
  if (n_selected_clusters > 0) {
    rmm::device_uvector<value_t> dist_membership_vec(m * n_selected_clusters, stream);

    dist_membership_vector(handle,
                           X,
                           m,
                           X,
                           m,
                           n,
                           n_exemplars,
                           n_selected_clusters,
                           prediction_data.get_exemplar_idx(),
                           prediction_data.get_exemplar_label_offsets(),
                           dist_membership_vec.data(),
                           metric,
                           batch_size);

    rmm::device_uvector<value_t> merge_heights(m * n_selected_clusters, stream);

    all_points_outlier_membership_vector(handle,
                                         condensed_tree,
                                         deaths,
                                         selected_clusters,
                                         index_into_children,
                                         m,
                                         n_selected_clusters,
                                         merge_heights.data(),
                                         membership_vec,
                                         true);

    rmm::device_uvector<value_t> prob_in_some_cluster(m, stream);
    all_points_prob_in_some_cluster(handle,
                                    condensed_tree,
                                    deaths,
                                    selected_clusters,
                                    index_into_children,
                                    m,
                                    n_selected_clusters,
                                    merge_heights.data(),
                                    prob_in_some_cluster.data());

    raft::linalg::map_offset(
      handle,
      raft::make_device_vector_view<value_t, value_idx>(membership_vec, m * n_selected_clusters),
      [dist_membership_vec = dist_membership_vec.data(), membership_vec] __device__(auto idx) {
        return dist_membership_vec[idx] * membership_vec[idx];
      });

    // Normalize to obtain probabilities conditioned on points belonging to some cluster
    Utils::normalize(membership_vec, n_selected_clusters, m, stream);

    // Multiply with probabilities of points belonging to some cluster to obtain joint distribution
    raft::linalg::matrixVectorOp<true, false>(
      membership_vec,
      membership_vec,
      prob_in_some_cluster.data(),
      n_selected_clusters,
      (value_idx)m,
      [] __device__(value_t mat_in, value_t vec_in) { return mat_in * vec_in; },
      stream);
  }
}

/**
 * Predict soft cluster membership vectors for new points (not in the training data).
 *
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle raft handle for resource reuse
 * @param[in] condensed_tree a condensed hierarchy
 * @param[in] prediction_data PredictionData object
 * @param[in] X all points (size m * n)
 * @param[in] points_to_predict input prediction points (size n_prediction_points * n)
 * @param[in] n_prediction_points number of prediction points
 * @param[in] metric distance metric
 * @param[in] min_samples neighborhood size during training (includes self-loop)
 * @param[out] membership_vec output membership vectors (size n_prediction_points *
 * n_selected_clusters)
 * @param[in] batch_size batch size to be used while computing distance based memberships
 */
template <typename value_idx, typename value_t, int tpb = 256>
void membership_vector(const raft::handle_t& handle,
                       Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
                       Common::PredictionData<value_idx, value_t>& prediction_data,
                       const value_t* X,
                       const value_t* points_to_predict,
                       size_t n_prediction_points,
                       ML::distance::DistanceType metric,
                       int min_samples,
                       value_t* membership_vec,
                       size_t batch_size)
{
  RAFT_EXPECTS(metric == ML::distance::DistanceType::L2SqrtExpanded,
               "Currently only L2 expanded distance is supported");

  auto stream      = handle.get_stream().get();
  auto exec_policy = handle.get_thrust_policy();

  size_t m                       = prediction_data.n_rows;
  size_t n                       = prediction_data.n_cols;
  value_idx n_selected_clusters  = prediction_data.get_n_selected_clusters();
  value_t* deaths                = prediction_data.get_deaths();
  value_idx* selected_clusters   = prediction_data.get_selected_clusters();
  value_idx* index_into_children = prediction_data.get_index_into_children();
  value_idx n_exemplars          = prediction_data.get_n_exemplars();
  value_t* lambdas               = condensed_tree.get_lambdas();

  if (batch_size > n_prediction_points) batch_size = n_prediction_points;
  RAFT_EXPECTS(
    0 < batch_size && batch_size <= n_prediction_points,
    "Invalid batch_size. batch_size should be > 0 and <= the number of prediction points");

  rmm::device_uvector<value_t> dist_membership_vec(n_prediction_points * n_selected_clusters,
                                                   stream);

  dist_membership_vector(handle,
                         X,
                         m,
                         points_to_predict,
                         n_prediction_points,
                         n,
                         n_exemplars,
                         n_selected_clusters,
                         prediction_data.get_exemplar_idx(),
                         prediction_data.get_exemplar_label_offsets(),
                         dist_membership_vec.data(),
                         ML::distance::DistanceType::L2SqrtExpanded,
                         batch_size);

  auto prediction_lambdas =
    raft::make_device_vector<value_t, value_idx>(handle, n_prediction_points);
  rmm::device_uvector<value_idx> min_mr_inds(n_prediction_points, stream);

  _compute_knn_and_nearest_neighbor(handle,
                                    prediction_data,
                                    X,
                                    points_to_predict,
                                    min_samples,
                                    n_prediction_points,
                                    min_mr_inds.data(),
                                    prediction_lambdas.data_handle(),
                                    metric);

  raft::linalg::map_offset(
    handle,
    prediction_lambdas.view(),
    [lambdas,
     index_into_children,
     min_mr_inds        = min_mr_inds.data(),
     prediction_lambdas = prediction_lambdas.data_handle()] __device__(auto idx) {
      value_t neighbor_lambda = lambdas[index_into_children[min_mr_inds[idx]]];
      return min(prediction_lambdas[idx], neighbor_lambda);
    });

  rmm::device_uvector<value_t> merge_heights(n_prediction_points * n_selected_clusters, stream);

  outlier_membership_vector(handle,
                            condensed_tree,
                            deaths,
                            min_mr_inds.data(),
                            prediction_lambdas.data_handle(),
                            selected_clusters,
                            index_into_children,
                            n_prediction_points,
                            n_selected_clusters,
                            merge_heights.data(),
                            membership_vec,
                            true);

  auto combine_op = [membership_vec,
                     dist_membership_vec = dist_membership_vec.data()] __device__(auto idx) {
    return pow(membership_vec[idx], 2) * pow(dist_membership_vec[idx], 0.5);
  };

  raft::linalg::map_offset(handle,
                           raft::make_device_vector_view<value_t, value_idx>(
                             membership_vec, n_prediction_points * n_selected_clusters),
                           combine_op);

  // Normalize to obtain probabilities conditioned on points belonging to some cluster
  Utils::normalize(membership_vec, n_selected_clusters, n_prediction_points, stream);

  rmm::device_uvector<value_t> prob_in_some_cluster_(n_prediction_points, stream);

  prob_in_some_cluster(handle,
                       condensed_tree,
                       deaths,
                       selected_clusters,
                       index_into_children,
                       n_prediction_points,
                       n_selected_clusters,
                       min_mr_inds.data(),
                       merge_heights.data(),
                       prediction_lambdas.data_handle(),
                       prob_in_some_cluster_.data());

  // Multiply conditional probabilities with probability of point belonging to some cluster. This
  // gives the joint distribution.
  raft::linalg::matrixVectorOp<true, false>(
    membership_vec,
    membership_vec,
    prob_in_some_cluster_.data(),
    n_selected_clusters,
    (value_idx)n_prediction_points,
    [] __device__(value_t mat_in, value_t vec_in) { return mat_in * vec_in; },
    stream);
}

};  // namespace Predict
};  // namespace detail
};  // namespace HDBSCAN
};  // namespace ML
