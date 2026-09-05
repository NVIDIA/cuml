/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/common/distance_type.hpp>
#include <cuml/neighbors/knn.hpp>

#include <raft/linalg/unary_op.cuh>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/linalg/symmetrize.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>

namespace ML {
namespace HDBSCAN {
namespace detail {
namespace Reachability {

/**
 * Extract core distances from KNN graph. This is essentially
 * performing a knn_dists[:,min_pts]
 * @tparam value_idx data type for integrals
 * @tparam value_t data type for distance
 * @tparam tpb block size for kernel
 * @param[in] knn_dists knn distance array (size n * k)
 * @param[in] min_samples this neighbor will be selected for core distances
 * @param[in] n_neighbors the number of neighbors of each point in the knn graph
 * @param[in] n number of samples
 * @param[out] out output array (size n)
 * @param[in] stream stream for which to order cuda operations
 */
template <typename value_idx, typename value_t, int tpb = 256>
void core_distances(
  value_t* knn_dists, int min_samples, int n_neighbors, size_t n, value_t* out, cudaStream_t stream)
{
  ASSERT(n_neighbors >= min_samples,
         "the size of the neighborhood should be greater than or equal to min_samples");

  auto exec_policy = rmm::exec_policy(stream);

  auto indices = thrust::make_counting_iterator<value_idx>(0);

  thrust::transform(exec_policy, indices, indices + n, out, [=] __device__(value_idx row) {
    return knn_dists[row * n_neighbors + (min_samples - 1)];
  });
}

/**
 * Wraps the brute force knn API, to be used for both training and prediction
 * @tparam value_idx data type for integrals
 * @tparam value_t data type for distance
 * @param[in] handle raft handle for resource reuse
 * @param[in] X input data points (size m * n)
 * @param[out] inds nearest neighbor indices (size n_search_items * k)
 * @param[out] dists nearest neighbor distances (size n_search_items * k)
 * @param[in] m number of rows in X
 * @param[in] n number of columns in X
 * @param[in] search_items array of items to search of dimensionality D (size n_search_items * n)
 * @param[in] n_search_items number of rows in search_items
 * @param[in] k number of nearest neighbors
 * @param[in] metric distance metric to use
 */
template <typename value_idx, typename value_t>
void compute_knn(const raft::handle_t& handle,
                 const value_t* X,
                 value_idx* inds,
                 value_t* dists,
                 size_t m,
                 size_t n,
                 const value_t* search_items,
                 size_t n_search_items,
                 int k,
                 ML::distance::DistanceType metric)
{
  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();
  std::vector<value_t*> inputs;
  inputs.push_back(const_cast<value_t*>(X));

  std::vector<int> sizes;
  sizes.push_back(m);

  // This is temporary. Once faiss is updated, we should be able to
  // pass value_idx through to knn.
  rmm::device_uvector<int64_t> int64_indices(k * n_search_items, stream);

  // perform knn
  brute_force_knn(handle,
                  inputs,
                  sizes,
                  n,
                  const_cast<value_t*>(search_items),
                  n_search_items,
                  int64_indices.data(),
                  dists,
                  k,
                  true,
                  true,
                  metric);

  // convert from current knn's 64-bit to 32-bit.
  thrust::transform(exec_policy,
                    int64_indices.data(),
                    int64_indices.data() + int64_indices.size(),
                    inds,
                    [] __device__(int64_t in) -> value_idx { return in; });
}

/*
  @brief Internal function for CPU->GPU interop
         to compute core_dists
*/
template <typename value_idx, typename value_t>
void _compute_core_dists(const raft::handle_t& handle,
                         const value_t* X,
                         value_t* core_dists,
                         size_t m,
                         size_t n,
                         ML::distance::DistanceType metric,
                         int min_samples)
{
  RAFT_EXPECTS(metric == ML::distance::DistanceType::L2SqrtExpanded,
               "Currently only L2 expanded distance is supported");

  auto stream = handle.get_stream().get();

  rmm::device_uvector<value_idx> inds(min_samples * m, stream);
  rmm::device_uvector<value_t> dists(min_samples * m, stream);

  // perform knn
  compute_knn(handle, X, inds.data(), dists.data(), m, n, X, m, min_samples, metric);

  // Slice core distances (distances to kth nearest neighbor)
  core_distances<value_idx>(dists.data(), min_samples, min_samples, m, core_dists, stream);
}

};  // end namespace Reachability
};  // end namespace detail
};  // end namespace HDBSCAN
};  // end namespace ML
