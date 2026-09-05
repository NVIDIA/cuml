/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/cluster/hdbscan.hpp>
#include <cuml/common/logger.hpp>

#include <raft/label/classlabels.cuh>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/op/sort.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <cuda/functional>
#include <cuda/std/tuple>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

namespace ML {
namespace HDBSCAN {
namespace Common {

struct TupleComp {
  template <typename one, typename two>
  __host__ __device__ bool operator()(const one& t1, const two& t2)
  {
    // sort first by each parent,
    if (cuda::std::get<0>(t1) < cuda::std::get<0>(t2)) return true;
    if (cuda::std::get<0>(t1) > cuda::std::get<0>(t2)) return false;

    // within each parent, sort by each child,
    if (cuda::std::get<1>(t1) < cuda::std::get<1>(t2)) return true;
    if (cuda::std::get<1>(t1) > cuda::std::get<1>(t2)) return false;

    // then sort by value in descending order
    return cuda::std::get<2>(t1) < cuda::std::get<2>(t2);
  }
};

template <typename value_idx, typename value_t>
CondensedHierarchy<value_idx, value_t>::CondensedHierarchy(const raft::handle_t& handle_,
                                                           size_t n_leaves_)
  : handle(handle_),
    n_leaves(n_leaves_),
    parents(0, handle.get_stream()),
    children(0, handle.get_stream()),
    lambdas(0, handle.get_stream()),
    sizes(0, handle.get_stream())
{
}

template <typename value_idx, typename value_t>
CondensedHierarchy<value_idx, value_t>::CondensedHierarchy(const raft::handle_t& handle_,
                                                           size_t n_leaves_,
                                                           int n_edges_,
                                                           value_idx* parents_,
                                                           value_idx* children_,
                                                           value_t* lambdas_,
                                                           value_idx* sizes_)
  : handle(handle_),
    n_leaves(n_leaves_),
    n_edges(n_edges_),
    parents(n_edges_, handle.get_stream()),
    children(n_edges_, handle.get_stream()),
    lambdas(n_edges_, handle.get_stream()),
    sizes(n_edges_, handle.get_stream())
{
  raft::copy(parents.begin(), parents_, n_edges_, handle.get_stream());
  raft::copy(children.begin(), children_, n_edges_, handle.get_stream());
  raft::copy(lambdas.begin(), lambdas_, n_edges_, handle.get_stream());
  raft::copy(sizes.begin(), sizes_, n_edges_, handle.get_stream());

  auto parents_ptr = thrust::device_pointer_cast(parents.data());

  auto parents_min_max = thrust::minmax_element(
    thrust::cuda::par.on(handle.get_stream().get()), parents_ptr, parents_ptr + n_edges);
  auto min_cluster = *parents_min_max.first;
  auto max_cluster = *parents_min_max.second;

  n_clusters = max_cluster - min_cluster + 1;

  auto sort_keys = thrust::make_zip_iterator(
    cuda::std::make_tuple(parents.begin(), children.begin(), sizes.begin()));
  auto sort_values = thrust::make_zip_iterator(cuda::std::make_tuple(lambdas.begin()));

  thrust::sort_by_key(thrust::cuda::par.on(handle.get_stream().get()),
                      sort_keys,
                      sort_keys + n_edges,
                      sort_values,
                      TupleComp());
}

template <typename value_idx, typename value_t>
CondensedHierarchy<value_idx, value_t>::CondensedHierarchy(
  const raft::handle_t& handle_,
  size_t n_leaves_,
  int n_edges_,
  int n_clusters_,
  rmm::device_uvector<value_idx>&& parents_,
  rmm::device_uvector<value_idx>&& children_,
  rmm::device_uvector<value_t>&& lambdas_,
  rmm::device_uvector<value_idx>&& sizes_)
  : handle(handle_),
    n_leaves(n_leaves_),
    n_edges(n_edges_),
    n_clusters(n_clusters_),
    parents(std::move(parents_)),
    children(std::move(children_)),
    lambdas(std::move(lambdas_)),
    sizes(std::move(sizes_))
{
}

/**
 * Populates the condensed hierarchy object with the output
 * from Condense::condense_hierarchy
 * @param full_parents
 * @param full_children
 * @param full_lambdas
 * @param full_sizes
 */
template <typename value_idx, typename value_t>
void CondensedHierarchy<value_idx, value_t>::condense(value_idx* full_parents,
                                                      value_idx* full_children,
                                                      value_t* full_lambdas,
                                                      value_idx* full_sizes,
                                                      value_idx size)
{
  auto stream = handle.get_stream().get();

  if (size == -1) size = 4 * (n_leaves - 1) + 2;

  n_edges = thrust::transform_reduce(
    thrust::cuda::par.on(stream),
    full_sizes,
    full_sizes + size,
    cuda::proclaim_return_type<value_idx>(
      [=] __device__(value_idx a) -> value_idx { return static_cast<value_idx>(a != -1); }),
    static_cast<value_idx>(0),
    cuda::std::plus<value_idx>());

  parents.resize(n_edges, stream);
  children.resize(n_edges, stream);
  lambdas.resize(n_edges, stream);
  sizes.resize(n_edges, stream);

  auto in = thrust::make_zip_iterator(
    cuda::std::make_tuple(full_parents, full_children, full_lambdas, full_sizes));

  auto out = thrust::make_zip_iterator(
    cuda::std::make_tuple(parents.data(), children.data(), lambdas.data(), sizes.data()));

  thrust::copy_if(thrust::cuda::par.on(stream),
                  in,
                  in + size,
                  out,
                  [=] __device__(cuda::std::tuple<value_idx, value_idx, value_t, value_idx> tup) {
                    return cuda::std::get<3>(tup) != -1;
                  });

  auto parents_ptr = thrust::device_pointer_cast(parents.data());

  auto parents_min_max =
    thrust::minmax_element(thrust::cuda::par.on(stream), parents_ptr, parents_ptr + n_edges);
  auto min_cluster = *parents_min_max.first;
  auto max_cluster = *parents_min_max.second;

  n_clusters = max_cluster - min_cluster + 1;

  auto sort_keys = thrust::make_zip_iterator(
    cuda::std::make_tuple(parents.begin(), children.begin(), sizes.begin()));
  auto sort_values = thrust::make_zip_iterator(cuda::std::make_tuple(lambdas.begin()));

  thrust::sort_by_key(
    thrust::cuda::par.on(stream), sort_keys, sort_keys + n_edges, sort_values, TupleComp());
}

};  // namespace Common
};  // namespace HDBSCAN
};  // namespace ML
