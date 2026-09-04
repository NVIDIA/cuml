/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/utils.cuh"

#include <cuml/cluster/hdbscan.hpp>

#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/op/sort.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <cuda/std/tuple>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cmath>
namespace ML {
namespace HDBSCAN {
namespace Common {

template <typename value_idx, typename value_t>
void PredictionData<value_idx, value_t>::allocate(const raft::handle_t& handle,
                                                  value_idx n_exemplars_,
                                                  value_idx n_selected_clusters_,
                                                  value_idx n_edges_)
{
  this->n_exemplars         = n_exemplars_;
  this->n_selected_clusters = n_selected_clusters_;
  exemplar_idx.resize(n_exemplars, handle.get_stream());
  exemplar_label_offsets.resize(n_selected_clusters + 1, handle.get_stream());
  selected_clusters.resize(n_selected_clusters, handle.get_stream());
  index_into_children.resize(n_edges_ + 1, handle.get_stream());
}

/**
 * Builds an index into the children array of the CondensedHierarchy object. This is useful for
 * constant time lookups during bottom-up tree traversals in prediction algorithms. It is therefore
 * an important feature for speed-up in comparison with Scikit-learn Contrib. This is intended for
 * internal use only and users are not expected to invoke this method.
 *
 * @tparam value_idx
 * @param[in] handle raft handle for resource reuse
 * @param[in] children children array of condensed hierarchy
 * @param[in] n_edges number of edges in children array
 * @param[out] index_into_children index into the children array (size n_edges + 1)
 */
template <typename value_idx>
void build_index_into_children(const raft::handle_t& handle,
                               value_idx* children,
                               value_idx n_edges,
                               value_idx* index_into_children)
{
  auto exec_policy = handle.get_thrust_policy();

  auto counting = thrust::make_counting_iterator<value_idx>(0);

  auto index_op = [index_into_children] __device__(auto t) {
    index_into_children[cuda::std::get<0>(t)] = cuda::std::get<1>(t);
    return;
  };
  thrust::for_each(
    exec_policy,
    thrust::make_zip_iterator(cuda::std::make_tuple(children, counting)),
    thrust::make_zip_iterator(cuda::std::make_tuple(children + n_edges, counting + n_edges)),
    index_op);
}
/**
 * Populates the PredictionData container object. Computes and stores: the indices of exemplar
 * points sorted by their cluster labels, cluster label offsets of the exemplars and the set of
 * clusters selected from the cluster tree.
 *
 * @param[in] handle raft handle for resource reuse
 * @param[in] condensed_tree a condensed hierarchy
 * @param[in] labels Final normalized labels
 * @param[in] inverse_label_map map of normalized labels to original labels (size n_clusters)
 * @param[in] n_selected_clusters number of clusters in the final clustering
 * @param[in] prediction_data PreditionData object
 */
void generate_prediction_data(const raft::handle_t& handle,
                              CondensedHierarchy<int64_t, float>& condensed_tree,
                              int64_t* labels,
                              int64_t* inverse_label_map,
                              int n_selected_clusters,
                              PredictionData<int64_t, float>& prediction_data)
{
  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  auto counting = thrust::make_counting_iterator<int>(0);

  auto parents    = condensed_tree.get_parents();
  auto children   = condensed_tree.get_children();
  auto lambdas    = condensed_tree.get_lambdas();
  auto n_edges    = condensed_tree.get_n_edges();
  auto n_clusters = condensed_tree.get_n_clusters();
  auto n_leaves   = condensed_tree.get_n_leaves();
  auto sizes      = condensed_tree.get_sizes();

  // first compute the death of each cluster in the condensed hierarchy
  rmm::device_uvector<int64_t> sorted_parents(n_edges, stream);
  raft::copy_async(sorted_parents.data(), parents, n_edges, stream);

  rmm::device_uvector<int64_t> sorted_parents_offsets(n_clusters + 1, stream);
  detail::Utils::parent_csr(
    handle, condensed_tree, sorted_parents.data(), sorted_parents_offsets.data());

  prediction_data.set_n_clusters(handle, n_clusters);

  // this is to find maximum lambdas of all children under a parent
  auto reduce_func = [](void* d_temp_storage,
                        size_t& temp_storage_bytes,
                        const float* d_in,
                        float* d_out,
                        int num_segments,
                        const int64_t* d_begin_offsets,
                        const int64_t* d_end_offsets,
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
  detail::Utils::cub_segmented_reduce(lambdas,
                                      prediction_data.get_deaths(),
                                      n_clusters,
                                      sorted_parents_offsets.data(),
                                      stream,
                                      reduce_func);

  rmm::device_uvector<int> is_leaf_cluster(n_clusters, stream);
  thrust::fill(exec_policy, is_leaf_cluster.begin(), is_leaf_cluster.end(), 1);

  auto leaf_cluster_op =
    [is_leaf_cluster = is_leaf_cluster.data(), parents, sizes, n_leaves] __device__(auto idx) {
      if (sizes[idx] > 1) { is_leaf_cluster[parents[idx] - n_leaves] = 0; }
      return;
    };

  thrust::for_each(exec_policy, counting, counting + n_edges, leaf_cluster_op);

  rmm::device_uvector<int> is_exemplar(n_leaves, stream);
  rmm::device_uvector<int> exemplar_idx(n_leaves, stream);
  rmm::device_uvector<int> exemplar_label_offsets(n_selected_clusters + 1, stream);

  // classify whether or not a point is an exemplar point using the death values
  auto exemplar_op = [is_exemplar = is_exemplar.data(),
                      lambdas,
                      is_leaf_cluster = is_leaf_cluster.data(),
                      parents,
                      children,
                      n_leaves,
                      labels,
                      deaths = prediction_data.get_deaths()] __device__(auto idx) {
    if (children[idx] < n_leaves) {
      is_exemplar[children[idx]] =
        (labels[children[idx]] != -1 && is_leaf_cluster[parents[idx] - n_leaves] &&
         lambdas[idx] == deaths[parents[idx] - n_leaves]);
      return;
    }
  };

  thrust::for_each(exec_policy, counting, counting + n_edges, exemplar_op);

  int n_exemplars = thrust::count_if(
    exec_policy, is_exemplar.begin(), is_exemplar.end(), [] __device__(auto idx) { return idx; });

  prediction_data.allocate(handle, n_exemplars, n_selected_clusters, n_edges);

  auto exemplar_idx_end_ptr = thrust::copy_if(
    exec_policy,
    counting,
    counting + n_leaves,
    prediction_data.get_exemplar_idx(),
    [is_exemplar = is_exemplar.data()] __device__(auto idx) { return is_exemplar[idx]; });

  // use the exemplar labels to fetch the set of selected clusters from the condensed hierarchy
  rmm::device_uvector<int64_t> exemplar_labels(n_exemplars, stream);

  // this uses the original, pre-normalized label by
  // using the inverse label_map to lookup the original labels from final labels
  thrust::transform(exec_policy,
                    prediction_data.get_exemplar_idx(),
                    prediction_data.get_exemplar_idx() + n_exemplars,
                    exemplar_labels.data(),
                    [labels, inverse_label_map] __device__(auto idx) {
                      auto label = labels[idx];
                      if (label != -1) { return inverse_label_map[label]; }
                      return static_cast<int64_t>(-1);
                    });

  thrust::sort_by_key(exec_policy,
                      exemplar_labels.data(),
                      exemplar_labels.data() + n_exemplars,
                      prediction_data.get_exemplar_idx());

  // this uses the final, converted values of the labels
  rmm::device_uvector<int> converted_exemplar_labels(n_exemplars, stream);
  thrust::transform(exec_policy,
                    prediction_data.get_exemplar_idx(),
                    prediction_data.get_exemplar_idx() + n_exemplars,
                    converted_exemplar_labels.data(),
                    [labels] __device__(auto idx) { return labels[idx]; });

  if (n_exemplars > 0) {
    raft::sparse::convert::sorted_coo_to_csr(converted_exemplar_labels.data(),
                                             n_exemplars,
                                             prediction_data.get_exemplar_label_offsets(),
                                             n_selected_clusters + 1,
                                             stream);

    thrust::transform(exec_policy,
                      prediction_data.get_exemplar_label_offsets(),
                      prediction_data.get_exemplar_label_offsets() + n_selected_clusters,
                      prediction_data.get_selected_clusters(),
                      [exemplar_labels = exemplar_labels.data(), n_leaves] __device__(auto idx) {
                        return exemplar_labels[idx] + n_leaves;
                      });

    // build the index into the children array for constant time lookups
    build_index_into_children(handle, children, n_edges, prediction_data.get_index_into_children());
  }
}

};  // end namespace Common
};  // end namespace HDBSCAN
};  // end namespace ML
