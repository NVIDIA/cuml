/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "kernels/membership.cuh"
#include "utils.cuh"

#include <cuml/cluster/hdbscan.hpp>

#include <raft/label/classlabels.cuh>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/op/sort.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <algorithm>

namespace ML {
namespace HDBSCAN {
namespace detail {
namespace Membership {

// TODO: Compute outlier scores

template <typename value_idx, typename value_t>
void get_probabilities(const raft::handle_t& handle,
                       Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
                       const value_idx* labels,
                       value_t* probabilities)
{
  auto stream      = handle.get_stream().get();
  auto exec_policy = handle.get_thrust_policy();

  auto parents    = condensed_tree.get_parents();
  auto children   = condensed_tree.get_children();
  auto lambdas    = condensed_tree.get_lambdas();
  auto n_edges    = condensed_tree.get_n_edges();
  auto n_clusters = condensed_tree.get_n_clusters();
  auto n_leaves   = condensed_tree.get_n_leaves();

  rmm::device_uvector<value_idx> sorted_parents(n_edges, stream);
  raft::copy_async(sorted_parents.data(), parents, n_edges, stream);

  rmm::device_uvector<value_idx> sorted_parents_offsets(n_clusters + 1, stream);
  Utils::parent_csr(handle, condensed_tree, sorted_parents.data(), sorted_parents_offsets.data());

  // this is to find maximum lambdas of all children under a parent
  rmm::device_uvector<value_t> deaths(n_clusters, stream);
  thrust::fill(exec_policy, deaths.begin(), deaths.end(), 0.0f);

  // CCCL has changed `num_segments` to int64_t to support larger segment sizes
  // Avoid explicitly instantiating a given overload but rely on conversion from int
  auto reduce_func = [](void* d_temp_storage,
                        size_t& temp_storage_bytes,
                        const value_t* d_in,
                        value_t* d_out,
                        int num_segments,
                        const value_idx* d_begin_offsets,
                        const value_idx* d_end_offsets,
                        cudaStream_t stream = 0) -> cudaError_t {
    return cub::DeviceSegmentedReduce::Max(d_temp_storage,
                                           temp_storage_bytes,
                                           d_in,
                                           d_out,
                                           num_segments,
                                           d_begin_offsets,
                                           d_end_offsets,
                                           stream);
  };

  Utils::cub_segmented_reduce(
    lambdas, deaths.data(), n_clusters, sorted_parents_offsets.data(), stream, reduce_func);

  // Calculate probability per point
  thrust::fill(exec_policy, probabilities, probabilities + n_leaves, 0.0f);

  probabilities_functor<value_idx, value_t> probabilities_op(
    probabilities, deaths.data(), children, lambdas, labels, n_leaves);
  thrust::for_each(exec_policy,
                   thrust::make_counting_iterator(value_idx(0)),
                   thrust::make_counting_iterator(n_edges),
                   probabilities_op);
}

};  // namespace Membership
};  // namespace detail
};  // namespace HDBSCAN
};  // namespace ML
