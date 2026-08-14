/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/common/checked_arithmetic.hpp>
#include <cuml/common/logger.hpp>
#include <cuml/neighbors/knn.hpp>

#include <raft/core/device_resources.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/operators.hpp>
#include <raft/label/classlabels.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/stats/mean_center.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>

#include <cuvs/neighbors/ball_cover.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/knn_merge_parts.hpp>
#include <ml_mg_utils.cuh>
#include <selection/knn.cuh>

#include <cstddef>
#include <sstream>
#include <vector>

namespace ML {

struct knnIndexImpl {
  std::unique_ptr<cuvs::neighbors::ivf_flat::index<float, int64_t>> ivf_flat;
  std::unique_ptr<cuvs::neighbors::ivf_pq::index<int64_t>> ivf_pq;

  std::unique_ptr<rmm::device_uvector<float>> corr_norms;
  std::unique_ptr<rmm::device_uvector<float>> corr_means;
};

knnIndex::knnIndex() : pimpl{std::make_unique<knnIndexImpl>()} {}
knnIndex::~knnIndex() = default;

void brute_force_knn(const raft::handle_t& handle,
                     std::vector<float*>& input,
                     std::vector<int>& sizes,
                     int D,
                     float* search_items,
                     int n,
                     int64_t* res_I,
                     float* res_D,
                     int k,
                     bool rowMajorIndex,
                     bool rowMajorQuery,
                     ML::distance::DistanceType metric,
                     float metric_arg,
                     std::vector<int64_t>* translations)
{
  ASSERT(input.size() == sizes.size(), "input and sizes vectors must be the same size");

  // The cuvs api doesn't support having multiple input values to search against.
  auto userStream = raft::resource::get_cuda_stream(handle);

  ASSERT(input.size() == sizes.size(), "input and sizes vectors should be the same size");

  std::vector<int64_t>* id_ranges;
  if (translations == nullptr) {
    // If we don't have explicit translations
    // for offsets of the indices, build them
    // from the local partitions
    id_ranges       = new std::vector<int64_t>();
    int64_t total_n = 0;
    for (size_t i = 0; i < input.size(); i++) {
      id_ranges->push_back(total_n);
      total_n += sizes[i];
    }
  } else {
    // otherwise, use the given translations
    id_ranges = translations;
  }

  rmm::device_uvector<int64_t> trans(id_ranges->size(), userStream);
  raft::update_device(trans.data(), id_ranges->data(), id_ranges->size(), userStream);

  rmm::device_uvector<float> all_D(0, userStream);
  rmm::device_uvector<int64_t> all_I(0, userStream);

  float* out_D   = res_D;
  int64_t* out_I = res_I;

  std::size_t const k_n = ML::checked_mul<std::size_t>(k, n);
  if (input.size() > 1) {
    std::size_t const total = ML::checked_mul<std::size_t>(input.size(), k_n);
    all_D.resize(total, userStream);
    all_I.resize(total, userStream);

    out_D = all_D.data();
    out_I = all_I.data();
  }

  // Make other streams from pool wait on main stream
  raft::resource::wait_stream_pool_on_stream(handle);

  for (size_t i = 0; i < input.size(); i++) {
    float* out_d_ptr   = out_D + (i * k_n);
    int64_t* out_i_ptr = out_I + (i * k_n);

    auto stream         = raft::resource::get_next_usable_stream(handle, i);
    auto current_handle = raft::device_resources(stream);

    // build the brute_force index (precalculates norms etc)
    std::optional<cuvs::neighbors::brute_force::index<float>> idx;
    if (rowMajorIndex) {
      idx = cuvs::neighbors::brute_force::build(
        current_handle,
        raft::make_device_matrix_view<const float, int64_t, raft::row_major>(input[i], sizes[i], D),
        static_cast<cuvs::distance::DistanceType>(metric),
        metric_arg);

    } else {
      idx = cuvs::neighbors::brute_force::build(
        current_handle,
        raft::make_device_matrix_view<const float, int64_t, raft::col_major>(input[i], sizes[i], D),
        static_cast<cuvs::distance::DistanceType>(metric),
        metric_arg);
    }

    // query the index
    if (rowMajorQuery) {
      cuvs::neighbors::brute_force::search(
        current_handle,
        *idx,
        raft::make_device_matrix_view<const float, int64_t, raft::row_major>(search_items, n, D),
        raft::make_device_matrix_view<int64_t, int64_t>(out_i_ptr, n, k),
        raft::make_device_matrix_view<float, int64_t>(out_d_ptr, n, k));
    } else {
      cuvs::neighbors::brute_force::search(
        current_handle,
        *idx,
        raft::make_device_matrix_view<const float, int64_t, raft::col_major>(search_items, n, D),
        raft::make_device_matrix_view<int64_t, int64_t>(out_i_ptr, n, k),
        raft::make_device_matrix_view<float, int64_t>(out_d_ptr, n, k));
    }
  }

  // Sync internal streams if used. We don't need to
  // sync the user stream because we'll already have
  // fully serial execution.
  raft::resource::sync_stream_pool(handle);

  if (input.size() > 1 || translations != nullptr) {
    // This is necessary for proper index translations. If there are
    // no translations or partitions to combine, it can be skipped.
    // TODO: sort out where this knn_merge_parts should live
    cuvs::neighbors::knn_merge_parts(
      handle,
      raft::make_device_matrix_view<const float, int64_t>(out_D, n * input.size(), k),
      raft::make_device_matrix_view<const int64_t, int64_t>(out_I, n * input.size(), k),
      raft::make_device_matrix_view<float, int64_t>(res_D, n, k),
      raft::make_device_matrix_view<int64_t, int64_t>(res_I, n, k),
      raft::make_device_vector_view<int64_t, int64_t>(trans.data(), trans.size()));
  }

  if (translations == nullptr) delete id_ranges;
}

void rbc_build_index(const raft::handle_t& handle,
                     std::uintptr_t& rbc_index,
                     float* X,
                     int64_t n_rows,
                     int64_t n_cols,
                     ML::distance::DistanceType metric)
{
  auto X_view        = raft::make_device_matrix_view<const float, int64_t>(X, n_rows, n_cols);
  auto rbc_index_ptr = new cuvs::neighbors::ball_cover::index<int64_t, float>(
    handle, X_view, static_cast<cuvs::distance::DistanceType>(metric));
  cuvs::neighbors::ball_cover::build(handle, *rbc_index_ptr);
  rbc_index = reinterpret_cast<std::uintptr_t>(rbc_index_ptr);
}

void rbc_knn_query(const raft::handle_t& handle,
                   const std::uintptr_t& rbc_index,
                   uint32_t k,
                   const float* search_items,
                   uint32_t n_search_items,
                   int64_t dim,
                   int64_t* out_inds,
                   float* out_dists)
{
  auto rbc_index_ptr =
    reinterpret_cast<cuvs::neighbors::ball_cover::index<int64_t, float>*>(rbc_index);
  auto query_view =
    raft::make_device_matrix_view<const float, int64_t>(search_items, n_search_items, dim);
  auto indices_view = raft::make_device_matrix_view<int64_t, int64_t>(out_inds, n_search_items, k);
  auto distances_view = raft::make_device_matrix_view<float, int64_t>(out_dists, n_search_items, k);
  cuvs::neighbors::ball_cover::knn_query(
    handle, *rbc_index_ptr, query_view, indices_view, distances_view, k);
}

void rbc_radius_neighbors_graph(const raft::handle_t& handle,
                                const std::uintptr_t& rbc_index,
                                const float* query,
                                int64_t n_query,
                                int64_t dim,
                                float radius,
                                int64_t* adj_indptr,
                                int64_t* adj_indices,
                                int64_t nnz)
{
  auto index_ptr = reinterpret_cast<cuvs::neighbors::ball_cover::index<int64_t, float>*>(rbc_index);

  cuvs::neighbors::ball_cover::eps_nn(
    handle,
    *index_ptr,
    raft::make_device_vector_view<int64_t, int64_t>(adj_indptr, n_query + 1),
    raft::make_device_vector_view<int64_t, int64_t>(adj_indices, nnz),
    raft::make_device_vector_view<int64_t, int64_t>(nullptr, 0),
    raft::make_device_matrix_view<const float, int64_t>(query, n_query, dim),
    radius);
}

void rbc_free_index(std::uintptr_t rbc_index)
{
  // Cast back to the original type and delete
  auto rbc_index_ptr =
    reinterpret_cast<cuvs::neighbors::ball_cover::index<int64_t, float>*>(rbc_index);
  delete rbc_index_ptr;
}

void approx_knn_build_index(raft::handle_t& handle,
                            knnIndex* index,
                            knnIndexParam* params,
                            ML::distance::DistanceType metric,
                            float metricArg,
                            float* index_array,
                            int n,
                            int D)
{
  index->metric    = metric;
  index->metricArg = metricArg;

  auto ivf_ft_pams = dynamic_cast<IVFFlatParam*>(params);
  auto ivf_pq_pams = dynamic_cast<IVFPQParam*>(params);
  auto stream      = raft::resource::get_cuda_stream(handle);

  // For correlation: preprocess (center + normalize), use InnerProduct, then revert
  if (metric == ML::distance::DistanceType::CorrelationExpanded) {
    index->pimpl->corr_means = std::make_unique<rmm::device_uvector<float>>(n, stream);
    index->pimpl->corr_norms = std::make_unique<rmm::device_uvector<float>>(n, stream);

    // Compute means and center data
    float normalizer = 1.0f / static_cast<float>(D);
    raft::linalg::reduce<false, true>(
      index->pimpl->corr_means->data(), index_array, D, n, 0.0f, stream);
    raft::linalg::unaryOp(index->pimpl->corr_means->data(),
                          index->pimpl->corr_means->data(),
                          n,
                          raft::mul_const_op<float>(normalizer),
                          stream);
    raft::stats::meanCenter<false, false>(
      index_array, index_array, index->pimpl->corr_means->data(), D, n, stream);

    // Compute norms and normalize
    raft::linalg::rowNorm<raft::linalg::L2Norm, false>(
      index->pimpl->corr_norms->data(), index_array, D, n, stream, raft::sqrt_op{});
    raft::linalg::matrixVectorOp<false, false>(
      index_array, index_array, index->pimpl->corr_norms->data(), D, n, raft::div_op{}, stream);

    metric = ML::distance::DistanceType::InnerProduct;
  }

  auto index_view = raft::make_device_matrix_view<const float, int64_t>(index_array, n, D);

  if (ivf_ft_pams) {
    index->nprobe = ivf_ft_pams->nprobe;
    cuvs::neighbors::ivf_flat::index_params params;
    params.metric     = static_cast<cuvs::distance::DistanceType>(metric);
    params.metric_arg = metricArg;
    params.n_lists    = ivf_ft_pams->nlist;

    index->pimpl->ivf_flat = std::make_unique<cuvs::neighbors::ivf_flat::index<float, int64_t>>(
      cuvs::neighbors::ivf_flat::build(handle, params, index_view));
  } else if (ivf_pq_pams) {
    index->nprobe = ivf_pq_pams->nprobe;
    cuvs::neighbors::ivf_pq::index_params params;
    params.metric     = static_cast<cuvs::distance::DistanceType>(metric);
    params.metric_arg = metricArg;
    params.n_lists    = ivf_pq_pams->nlist;
    params.pq_bits    = ivf_pq_pams->n_bits;
    params.pq_dim     = ivf_pq_pams->M;
    // TODO: handle ivf_pq_pams.usePrecomputedTables ?

    index->pimpl->ivf_pq = std::make_unique<cuvs::neighbors::ivf_pq::index<int64_t>>(
      cuvs::neighbors::ivf_pq::build(handle, params, index_view));
  } else {
    RAFT_FAIL("Unrecognized index type.");
  }

  // Revert user data for correlation
  if (index->metric == ML::distance::DistanceType::CorrelationExpanded) {
    raft::linalg::matrixVectorOp<false, false>(
      index_array, index_array, index->pimpl->corr_norms->data(), D, n, raft::mul_op{}, stream);
    raft::stats::meanAdd<false, false>(
      index_array, index_array, index->pimpl->corr_means->data(), D, n, stream);
  }
}

void approx_knn_search(raft::handle_t& handle,
                       float* distances,
                       int64_t* indices,
                       knnIndex* index,
                       int k,
                       float* query_array,
                       int n)
{
  auto stream = raft::resource::get_cuda_stream(handle);

  // Get dimension from index
  int D = index->pimpl->ivf_flat ? index->pimpl->ivf_flat->dim() : index->pimpl->ivf_pq->dim();

  // Temporary storage for correlation query preprocessing
  std::unique_ptr<rmm::device_uvector<float>> query_means;
  std::unique_ptr<rmm::device_uvector<float>> query_norms;

  // Preprocess queries for correlation
  if (index->metric == ML::distance::DistanceType::CorrelationExpanded) {
    query_means = std::make_unique<rmm::device_uvector<float>>(n, stream);
    query_norms = std::make_unique<rmm::device_uvector<float>>(n, stream);

    float normalizer = 1.0f / static_cast<float>(D);
    raft::linalg::reduce<false, true>(query_means->data(), query_array, D, n, 0.0f, stream);
    raft::linalg::unaryOp(
      query_means->data(), query_means->data(), n, raft::mul_const_op<float>(normalizer), stream);
    raft::stats::meanCenter<false, false>(
      query_array, query_array, query_means->data(), D, n, stream);

    raft::linalg::rowNorm<raft::linalg::L2Norm, false>(
      query_norms->data(), query_array, D, n, stream, raft::sqrt_op{});
    raft::linalg::matrixVectorOp<false, false>(
      query_array, query_array, query_norms->data(), D, n, raft::div_op{}, stream);
  }

  auto indices_view   = raft::make_device_matrix_view<int64_t, int64_t>(indices, n, k);
  auto distances_view = raft::make_device_matrix_view<float, int64_t>(distances, n, k);

  if (index->pimpl->ivf_flat) {
    auto query_view = raft::make_device_matrix_view<const float, int64_t>(
      query_array, n, index->pimpl->ivf_flat->dim());
    cuvs::neighbors::ivf_flat::search_params params;
    params.n_probes = index->nprobe;

    cuvs::neighbors::ivf_flat::search(
      handle, params, *index->pimpl->ivf_flat, query_view, indices_view, distances_view);
  } else if (index->pimpl->ivf_pq) {
    auto query_view = raft::make_device_matrix_view<const float, int64_t>(
      query_array, n, index->pimpl->ivf_pq->dim());
    cuvs::neighbors::ivf_pq::search_params params;
    params.n_probes = index->nprobe;

    cuvs::neighbors::ivf_pq::search(
      handle, params, *(index->pimpl->ivf_pq), query_view, indices_view, distances_view);
  } else {
    RAFT_FAIL("The model is not trained");
  }

  // Revert query data for correlation
  if (index->metric == ML::distance::DistanceType::CorrelationExpanded) {
    raft::linalg::matrixVectorOp<false, false>(
      query_array, query_array, query_norms->data(), D, n, raft::mul_op{}, stream);
    raft::stats::meanAdd<false, false>(query_array, query_array, query_means->data(), D, n, stream);
  }

  // perform post-processing to show the real distances
  if (index->metric == ML::distance::DistanceType::LpUnexpanded) {
    /**
     * post-processing
     */
    float p = 1.0 / index->metricArg;
    raft::linalg::unaryOp<float>(distances,
                                 distances,
                                 n * k,
                                 raft::pow_const_op<float>(p),
                                 raft::resource::get_cuda_stream(handle));
  }

  // Post-process correlation: convert inner product to correlation distance
  if (index->metric == ML::distance::DistanceType::CorrelationExpanded) {
    raft::linalg::unaryOp(
      distances, distances, n * k, [] __device__(float in) { return 1.0f - in; }, stream);
  }
}

void knn_classify(raft::handle_t& handle,
                  int* out,
                  int64_t* knn_indices,
                  std::vector<int*>& y,
                  size_t n_index_rows,
                  size_t n_query_rows,
                  int k,
                  float* sample_weight)
{
  cudaStream_t stream = handle.get_stream();

  std::vector<rmm::device_uvector<int>> uniq_labels_v;
  std::vector<int*> uniq_labels(y.size());
  std::vector<int> n_unique(y.size());

  for (std::size_t i = 0; i < y.size(); i++) {
    uniq_labels_v.emplace_back(0, stream);
    n_unique[i]    = raft::label::getUniquelabels(uniq_labels_v.back(), y[i], n_index_rows, stream);
    uniq_labels[i] = uniq_labels_v[i].data();
  }

  MLCommon::Selection::knn_classify(handle,
                                    out,
                                    knn_indices,
                                    y,
                                    n_index_rows,
                                    n_query_rows,
                                    k,
                                    uniq_labels,
                                    n_unique,
                                    sample_weight);
}

void knn_regress(raft::handle_t& handle,
                 float* out,
                 int64_t* knn_indices,
                 std::vector<float*>& y,
                 size_t n_index_rows,
                 size_t n_query_rows,
                 int k,
                 float* sample_weight)
{
  MLCommon::Selection::knn_regress(
    handle, out, knn_indices, y, n_index_rows, n_query_rows, k, sample_weight);
}

void knn_regress(raft::handle_t& handle,
                 double* out,
                 int64_t* knn_indices,
                 std::vector<double*>& y,
                 size_t n_index_rows,
                 size_t n_query_rows,
                 int k,
                 float* sample_weight)
{
  MLCommon::Selection::knn_regress(
    handle, out, knn_indices, y, n_index_rows, n_query_rows, k, sample_weight);
}

void knn_class_proba(raft::handle_t& handle,
                     std::vector<float*>& out,
                     int64_t* knn_indices,
                     std::vector<int*>& y,
                     size_t n_index_rows,
                     size_t n_query_rows,
                     int k,
                     float* sample_weight)
{
  cudaStream_t stream = handle.get_stream();

  std::vector<rmm::device_uvector<int>> uniq_labels_v;
  std::vector<int*> uniq_labels(y.size());
  std::vector<int> n_unique(y.size());

  for (std::size_t i = 0; i < y.size(); i++) {
    uniq_labels_v.emplace_back(0, stream);
    n_unique[i]    = raft::label::getUniquelabels(uniq_labels_v.back(), y[i], n_index_rows, stream);
    uniq_labels[i] = uniq_labels_v[i].data();
  }

  MLCommon::Selection::class_probs(handle,
                                   out,
                                   knn_indices,
                                   y,
                                   n_index_rows,
                                   n_query_rows,
                                   k,
                                   uniq_labels,
                                   n_unique,
                                   sample_weight);
}

};  // END NAMESPACE ML
