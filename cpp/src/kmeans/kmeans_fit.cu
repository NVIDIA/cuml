/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ml_cuda_utils.h"

#include <cuml/cluster/kmeans.hpp>
#include <cuml/cluster/kmeans_params.hpp>

#include <raft/core/handle.hpp>
#include <raft/core/host_mdspan.hpp>

#include <cuvs/cluster/kmeans.hpp>
#include <kmeans/kmeans_params.hpp>

#include <optional>
#include <vector>

namespace ML {
namespace kmeans {

template <typename value_t>
void fit_impl_device_parts(const raft::handle_t& handle,
                           const KMeansParams& params,
                           const value_t* const* X_parts,
                           const int64_t* n_samples_parts,
                           int64_t n_parts,
                           int64_t n_features,
                           const value_t* const* sample_weight_parts,
                           value_t* centroids,
                           value_t& inertia,
                           int64_t& n_iter)
{
  std::vector<raft::device_matrix_view<const value_t, int64_t>> X_views;
  X_views.reserve(n_parts);
  for (int64_t i = 0; i < n_parts; ++i) {
    X_views.push_back(raft::make_device_matrix_view<const value_t, int64_t>(
      X_parts[i], n_samples_parts[i], n_features));
  }

  std::optional<std::vector<raft::device_vector_view<const value_t, int64_t>>> sw = std::nullopt;
  if (sample_weight_parts != nullptr) {
    std::vector<raft::device_vector_view<const value_t, int64_t>> sw_views;
    sw_views.reserve(n_parts);
    for (int64_t i = 0; i < n_parts; ++i) {
      sw_views.push_back(raft::make_device_vector_view<const value_t, int64_t>(
        sample_weight_parts[i], n_samples_parts[i]));
    }
    sw = std::make_optional(std::move(sw_views));
  }

  auto centroids_view =
    raft::make_device_matrix_view<value_t, int64_t>(centroids, params.n_clusters, n_features);
  auto inertia_view = raft::make_host_scalar_view<value_t>(&inertia);
  auto n_iter_view  = raft::make_host_scalar_view<int64_t>(&n_iter);

  cuvs::cluster::kmeans::fit(
    handle, to_cuvs(params), X_views, sw, centroids_view, inertia_view, n_iter_view);
}

template <typename value_t>
void fit_impl_host_parts(const raft::handle_t& handle,
                         const KMeansParams& params,
                         const value_t* const* X_parts,
                         const int64_t* n_samples_parts,
                         int64_t n_parts,
                         int64_t n_features,
                         const value_t* const* sample_weight_parts,
                         value_t* centroids,
                         value_t& inertia,
                         int64_t& n_iter)
{
  std::vector<raft::host_matrix_view<const value_t, int64_t>> X_views;
  X_views.reserve(n_parts);
  for (int64_t i = 0; i < n_parts; ++i) {
    X_views.push_back(raft::make_host_matrix_view<const value_t, int64_t>(
      X_parts[i], n_samples_parts[i], n_features));
  }

  std::optional<std::vector<raft::host_vector_view<const value_t, int64_t>>> sw = std::nullopt;
  if (sample_weight_parts != nullptr) {
    std::vector<raft::host_vector_view<const value_t, int64_t>> sw_views;
    sw_views.reserve(n_parts);
    for (int64_t i = 0; i < n_parts; ++i) {
      sw_views.push_back(raft::make_host_vector_view<const value_t, int64_t>(sample_weight_parts[i],
                                                                             n_samples_parts[i]));
    }
    sw = std::make_optional(std::move(sw_views));
  }

  auto centroids_view =
    raft::make_device_matrix_view<value_t, int64_t>(centroids, params.n_clusters, n_features);
  auto inertia_view = raft::make_host_scalar_view<value_t>(&inertia);
  auto n_iter_view  = raft::make_host_scalar_view<int64_t>(&n_iter);

  cuvs::cluster::kmeans::fit(
    handle, to_cuvs(params), X_views, sw, centroids_view, inertia_view, n_iter_view);
}

template <typename value_t, typename idx_t>
void fit_impl_host(const raft::handle_t& handle,
                   const KMeansParams& params,
                   const value_t* X,
                   idx_t n_samples,
                   idx_t n_features,
                   const value_t* sample_weight,
                   value_t* centroids,
                   value_t& inertia,
                   idx_t& n_iter)
{
  auto inertia_view  = raft::make_host_scalar_view<value_t>(&inertia);
  auto n_samples_64  = static_cast<int64_t>(n_samples);
  auto n_features_64 = static_cast<int64_t>(n_features);
  auto X_view = raft::make_host_matrix_view<const value_t, int64_t>(X, n_samples_64, n_features_64);
  std::optional<raft::host_vector_view<const value_t, int64_t>> sw = std::nullopt;
  if (sample_weight != nullptr)
    sw = std::make_optional(
      raft::make_host_vector_view<const value_t, int64_t>(sample_weight, n_samples_64));
  auto centroids_view_64 =
    raft::make_device_matrix_view<value_t, int64_t>(centroids, params.n_clusters, n_features_64);
  int64_t n_iter_64   = 0;
  auto n_iter_view_64 = raft::make_host_scalar_view<int64_t>(&n_iter_64);

  cuvs::cluster::kmeans::fit(
    handle, to_cuvs(params), X_view, sw, centroids_view_64, inertia_view, n_iter_view_64);
  n_iter = static_cast<idx_t>(n_iter_64);
  return;
}

template <typename value_t, typename idx_t>
void fit_impl_device(const raft::handle_t& handle,
                     const KMeansParams& params,
                     const value_t* X,
                     idx_t n_samples,
                     idx_t n_features,
                     const value_t* sample_weight,
                     value_t* centroids,
                     value_t& inertia,
                     idx_t& n_iter)
{
  auto centroids_view =
    raft::make_device_matrix_view<value_t, idx_t>(centroids, params.n_clusters, n_features);
  auto inertia_view = raft::make_host_scalar_view<value_t>(&inertia);

  auto X_view = raft::make_device_matrix_view(X, n_samples, n_features);
  std::optional<raft::device_vector_view<const value_t, idx_t>> sw = std::nullopt;
  if (sample_weight != nullptr)
    sw = std::make_optional(
      raft::make_device_vector_view<const value_t, idx_t>(sample_weight, n_samples));
  auto n_iter_view = raft::make_host_scalar_view<idx_t>(&n_iter);

  cuvs::cluster::kmeans::fit(
    handle, to_cuvs(params), X_view, sw, centroids_view, inertia_view, n_iter_view);
}

void fit(const raft::handle_t& handle,
         const KMeansParams& params,
         const float* X,
         int n_samples,
         int n_features,
         const float* sample_weight,
         float* centroids,
         float& inertia,
         int& n_iter)
{
  if (ML::is_device_or_managed_type(X)) {
    fit_impl_device(
      handle, params, X, n_samples, n_features, sample_weight, centroids, inertia, n_iter);
  } else {
    fit_impl_host(
      handle, params, X, n_samples, n_features, sample_weight, centroids, inertia, n_iter);
  }
}

void fit(const raft::handle_t& handle,
         const KMeansParams& params,
         const double* X,
         int n_samples,
         int n_features,
         const double* sample_weight,
         double* centroids,
         double& inertia,
         int& n_iter)
{
  if (ML::is_device_or_managed_type(X)) {
    fit_impl_device(
      handle, params, X, n_samples, n_features, sample_weight, centroids, inertia, n_iter);
  } else {
    fit_impl_host(
      handle, params, X, n_samples, n_features, sample_weight, centroids, inertia, n_iter);
  }
}

void fit(const raft::handle_t& handle,
         const KMeansParams& params,
         const float* X,
         int64_t n_samples,
         int64_t n_features,
         const float* sample_weight,
         float* centroids,
         float& inertia,
         int64_t& n_iter)
{
  if (ML::is_device_or_managed_type(X)) {
    fit_impl_device(
      handle, params, X, n_samples, n_features, sample_weight, centroids, inertia, n_iter);
  } else {
    fit_impl_host(
      handle, params, X, n_samples, n_features, sample_weight, centroids, inertia, n_iter);
  }
}

void fit(const raft::handle_t& handle,
         const KMeansParams& params,
         const double* X,
         int64_t n_samples,
         int64_t n_features,
         const double* sample_weight,
         double* centroids,
         double& inertia,
         int64_t& n_iter)
{
  if (ML::is_device_or_managed_type(X)) {
    fit_impl_device(
      handle, params, X, n_samples, n_features, sample_weight, centroids, inertia, n_iter);
  } else {
    fit_impl_host(
      handle, params, X, n_samples, n_features, sample_weight, centroids, inertia, n_iter);
  }
}

// Detect partition residency from the first non-empty partition: an empty
// partition may carry a null data pointer that `is_device_or_managed_type`
// cannot classify, so skip past empties. An all-empty rank has no local data;
// default to the host path.
template <typename value_t>
static bool parts_on_device(const value_t* const* X_parts,
                            const int64_t* n_samples_parts,
                            int64_t n_parts)
{
  for (int64_t i = 0; i < n_parts; ++i) {
    if (n_samples_parts[i] > 0) { return ML::is_device_or_managed_type(X_parts[i]); }
  }
  return false;
}

void fit(const raft::handle_t& handle,
         const KMeansParams& params,
         const float* const* X_parts,
         const int64_t* n_samples_parts,
         int64_t n_parts,
         int64_t n_features,
         const float* const* sample_weight_parts,
         float* centroids,
         float& inertia,
         int64_t& n_iter)
{
  if (parts_on_device(X_parts, n_samples_parts, n_parts)) {
    fit_impl_device_parts(handle,
                          params,
                          X_parts,
                          n_samples_parts,
                          n_parts,
                          n_features,
                          sample_weight_parts,
                          centroids,
                          inertia,
                          n_iter);
  } else {
    fit_impl_host_parts(handle,
                        params,
                        X_parts,
                        n_samples_parts,
                        n_parts,
                        n_features,
                        sample_weight_parts,
                        centroids,
                        inertia,
                        n_iter);
  }
}

void fit(const raft::handle_t& handle,
         const KMeansParams& params,
         const double* const* X_parts,
         const int64_t* n_samples_parts,
         int64_t n_parts,
         int64_t n_features,
         const double* const* sample_weight_parts,
         double* centroids,
         double& inertia,
         int64_t& n_iter)
{
  if (parts_on_device(X_parts, n_samples_parts, n_parts)) {
    fit_impl_device_parts(handle,
                          params,
                          X_parts,
                          n_samples_parts,
                          n_parts,
                          n_features,
                          sample_weight_parts,
                          centroids,
                          inertia,
                          n_iter);
  } else {
    fit_impl_host_parts(handle,
                        params,
                        X_parts,
                        n_samples_parts,
                        n_parts,
                        n_features,
                        sample_weight_parts,
                        centroids,
                        inertia,
                        n_iter);
  }
}

};  // end namespace kmeans
};  // end namespace ML
