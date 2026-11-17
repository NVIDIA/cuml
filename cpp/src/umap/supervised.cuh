/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "fuzzy_simpl_set/runner.cuh"
#include "init_embed/runner.cuh"
#include "knn_graph/runner.cuh"
#include "optimize.cuh"
#include "simpl_set_embed/runner.cuh"

#include <cuml/common/logger.hpp>
#include <cuml/manifold/umapparams.h>
#include <cuml/neighbors/knn.hpp>

#include <raft/core/handle.hpp>
#include <raft/sparse/convert/coo.cuh>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/linalg/add.cuh>
#include <raft/sparse/linalg/norm.cuh>
#include <raft/sparse/linalg/symmetrize.cuh>
#include <raft/sparse/op/filter.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cuda_runtime.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>

namespace UMAPAlgo {

namespace Supervised {

using namespace ML;

template <int TPB_X, typename T>
CUML_KERNEL void fast_intersection_kernel(
  int* rows, int* cols, T* vals, int nnz, T* target, float unknown_dist = 1.0, float far_dist = 5.0)
{
  int row = (blockIdx.x * TPB_X) + threadIdx.x;
  if (row < nnz) {
    int i = rows[row];
    int j = cols[row];
    if (target[i] == T(-1.0) || target[j] == T(-1.0))
      vals[row] *= exp(-unknown_dist);
    else if (target[i] != target[j])
      vals[row] *= exp(-far_dist);
  }
}

template <typename T, int TPB_X>
void reset_local_connectivity(raft::sparse::COO<T>* in_coo,
                              raft::sparse::COO<T>* out_coo,
                              cudaStream_t stream  // size = nnz*2
)
{
  rmm::device_uvector<int> row_ind(in_coo->n_rows, stream);

  raft::sparse::convert::sorted_coo_to_csr(in_coo, row_ind.data(), stream);

  // Perform l_inf normalization
  raft::sparse::linalg::csr_row_normalize_max<T>(
    row_ind.data(), in_coo->vals(), in_coo->nnz, in_coo->n_rows, in_coo->vals(), stream);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  raft::sparse::linalg::coo_symmetrize<T>(
    in_coo,
    out_coo,
    [] __device__(int row, int col, T result, T transpose) {
      T prod_matrix = result * transpose;
      return result + transpose - prod_matrix;
    },
    stream);

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * Combine a fuzzy simplicial set with another fuzzy simplicial set
 * generated from categorical data using categorical distances. The target
 * data is assumed to be categorical label data (a vector of labels),
 * and this will update the fuzzy simplicial set to respect that label
 * data.
 */
template <typename value_t, typename nnz_t, int TPB_X>
void categorical_simplicial_set_intersection(raft::sparse::COO<value_t>* graph_coo,
                                             value_t* target,
                                             cudaStream_t stream,
                                             float far_dist     = 5.0,
                                             float unknown_dist = 1.0)
{
  dim3 grid(raft::ceildiv(static_cast<nnz_t>(graph_coo->nnz), static_cast<nnz_t>(TPB_X)), 1, 1);
  dim3 blk(TPB_X, 1, 1);
  fast_intersection_kernel<TPB_X, value_t><<<grid, blk, 0, stream>>>(graph_coo->rows(),
                                                                     graph_coo->cols(),
                                                                     graph_coo->vals(),
                                                                     graph_coo->nnz,
                                                                     target,
                                                                     unknown_dist,
                                                                     far_dist);
}

template <typename value_t, typename nnz_t, int TPB_X>
CUML_KERNEL void sset_intersection_kernel(int* row_ind1,
                                          int* cols1,
                                          value_t* vals1,
                                          int nnz1,
                                          int* row_ind2,
                                          int* cols2,
                                          value_t* vals2,
                                          int nnz2,
                                          int* result_ind,
                                          int* result_cols,
                                          value_t* result_vals,
                                          int nnz,
                                          value_t left_min,
                                          value_t right_min,
                                          int m,
                                          float mix_weight = 0.5)
{
  int row = (blockIdx.x * TPB_X) + threadIdx.x;

  if (row < m) {
    int start_idx_res = result_ind[row];
    int stop_idx_res  = raft::sparse::get_stop_idx(row, m, nnz, result_ind);

    int start_idx1 = row_ind1[row];
    int stop_idx1  = raft::sparse::get_stop_idx(row, m, nnz1, row_ind1);

    int start_idx2 = row_ind2[row];
    int stop_idx2  = raft::sparse::get_stop_idx(row, m, nnz2, row_ind2);

    for (int j = start_idx_res; j < stop_idx_res; j++) {
      int col = result_cols[j];

      value_t left_val = left_min;
      for (int k = start_idx1; k < stop_idx1; k++) {
        if (cols1[k] == col) { left_val = vals1[k]; }
      }

      value_t right_val = right_min;
      for (int k = start_idx2; k < stop_idx2; k++) {
        if (cols2[k] == col) { right_val = vals2[k]; }
      }

      if (left_val > left_min || right_val > right_min) {
        if (mix_weight < 0.5) {
          result_vals[j] = left_val * powf(right_val, mix_weight / (1.0 - mix_weight));
        } else {
          result_vals[j] = powf(left_val, (1.0 - mix_weight) / mix_weight) * right_val;
        }
      }
    }
  }
}

/**
 * Computes the CSR column index pointer and values
 * for the general simplicial set intersecftion.
 */
template <typename T, typename nnz_t, int TPB_X>
void general_simplicial_set_intersection(int* row1_ind,
                                         raft::sparse::COO<T>* in1,
                                         int* row2_ind,
                                         raft::sparse::COO<T>* in2,
                                         raft::sparse::COO<T>* result,
                                         float weight,
                                         cudaStream_t stream)
{
  rmm::device_uvector<int> result_ind(in1->n_rows, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(result_ind.data(), 0, in1->n_rows * sizeof(int), stream));

  int result_nnz = raft::sparse::linalg::csr_add_calc_inds<float>(row1_ind,
                                                                  in1->cols(),
                                                                  in1->vals(),
                                                                  in1->nnz,
                                                                  row2_ind,
                                                                  in2->cols(),
                                                                  in2->vals(),
                                                                  in2->nnz,
                                                                  in1->n_rows,
                                                                  result_ind.data(),
                                                                  stream);

  result->allocate(result_nnz, in1->n_rows, in1->n_cols, true, stream);

  /**
   * Element-wise sum of two simplicial sets
   */
  raft::sparse::linalg::csr_add_finalize<float>(row1_ind,
                                                in1->cols(),
                                                in1->vals(),
                                                in1->nnz,
                                                row2_ind,
                                                in2->cols(),
                                                in2->vals(),
                                                in2->nnz,
                                                in1->n_rows,
                                                result_ind.data(),
                                                result->cols(),
                                                result->vals(),
                                                stream);

  //@todo: Write a wrapper function for this
  raft::sparse::convert::csr_to_coo<int>(
    result_ind.data(), result->n_rows, result->rows(), result->nnz, stream);

  thrust::device_ptr<const T> d_ptr1 = thrust::device_pointer_cast(in1->vals());
  T min1 = *(thrust::min_element(thrust::cuda::par.on(stream), d_ptr1, d_ptr1 + in1->nnz));

  thrust::device_ptr<const T> d_ptr2 = thrust::device_pointer_cast(in2->vals());
  T min2 = *(thrust::min_element(thrust::cuda::par.on(stream), d_ptr2, d_ptr2 + in2->nnz));

  T left_min  = max(min1 / 2.0, 1e-8);
  T right_min = max(min2 / 2.0, 1e-8);

  dim3 grid(raft::ceildiv(static_cast<nnz_t>(in1->nnz), static_cast<nnz_t>(TPB_X)), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  sset_intersection_kernel<T, nnz_t, TPB_X><<<grid, blk, 0, stream>>>(row1_ind,
                                                                      in1->cols(),
                                                                      in1->vals(),
                                                                      in1->nnz,
                                                                      row2_ind,
                                                                      in2->cols(),
                                                                      in2->vals(),
                                                                      in2->nnz,
                                                                      result_ind.data(),
                                                                      result->cols(),
                                                                      result->vals(),
                                                                      result->nnz,
                                                                      left_min,
                                                                      right_min,
                                                                      in1->n_rows,
                                                                      weight);
  RAFT_CUDA_TRY(cudaGetLastError());

  dim3 grid_n(raft::ceildiv(static_cast<nnz_t>(result->nnz), static_cast<nnz_t>(TPB_X)), 1, 1);
}

template <typename T, typename nnz_t, int TPB_X>
void perform_categorical_intersection(T* y,
                                      raft::sparse::COO<T>* rgraph_coo,
                                      raft::sparse::COO<T>* final_coo,
                                      UMAPParams* params,
                                      cudaStream_t stream)
{
  float far_dist = 1.0e12;  // target weight
  if (params->target_weight < 1.0) far_dist = 2.5 * (1.0 / (1.0 - params->target_weight));

  categorical_simplicial_set_intersection<T, nnz_t, TPB_X>(rgraph_coo, y, stream, far_dist);

  raft::sparse::COO<T> comp_coo(stream);
  raft::sparse::op::coo_remove_zeros<T>(rgraph_coo, &comp_coo, stream);

  reset_local_connectivity<T, TPB_X>(&comp_coo, final_coo, stream);

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename value_idx, typename value_t, typename nnz_t, int TPB_X>
void perform_general_intersection(const raft::handle_t& handle,
                                  value_t* y,
                                  raft::sparse::COO<value_t>* rgraph_coo,
                                  raft::sparse::COO<value_t>* final_coo,
                                  UMAPParams* params,
                                  cudaStream_t stream)
{
  /**
   * Calculate kNN for Y
   */
  int knn_dims = rgraph_coo->n_rows * params->target_n_neighbors;
  rmm::device_uvector<value_idx> y_knn_indices(knn_dims, stream);
  rmm::device_uvector<value_t> y_knn_dists(knn_dims, stream);

  knn_graph<value_idx, value_t> knn_graph(rgraph_coo->n_rows, params->target_n_neighbors);
  knn_graph.knn_indices = y_knn_indices.data();
  knn_graph.knn_dists   = y_knn_dists.data();

  manifold_dense_inputs_t<value_t> y_inputs(y, nullptr, rgraph_coo->n_rows, 1);
  kNNGraph::run<value_idx, value_t, manifold_dense_inputs_t<value_t>>(
    handle, y_inputs, y_inputs, knn_graph, params->target_n_neighbors, params, stream);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  /*
  if (ML::default_logger().should_log(rapids_logger::level_enum::debug)) {
    CUML_LOG_DEBUG("Target kNN Graph");
    std::stringstream ss1, ss2;
    ss1 << raft::arr2Str(
      y_knn_indices.data(), rgraph_coo->n_rows * params->target_n_neighbors, "knn_indices", stream);
    CUML_LOG_TRACE("%s", ss1.str().c_str());
    ss2 << raft::arr2Str(
      y_knn_dists.data(), rgraph_coo->n_rows * params->target_n_neighbors, "knn_dists", stream);
    CUML_LOG_TRACE("%s", ss2.str().c_str());
  }
  */

  /**
   * Compute fuzzy simplicial set
   */
  raft::sparse::COO<value_t> ygraph_coo(stream);

  FuzzySimplSet::run<value_t, value_idx, nnz_t, TPB_X>(rgraph_coo->n_rows,
                                                       y_knn_indices.data(),
                                                       y_knn_dists.data(),
                                                       params->target_n_neighbors,
                                                       &ygraph_coo,
                                                       params,
                                                       stream);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  /*
  if (ML::default_logger().should_log(rapids_logger::level_enum::debug)) {
    CUML_LOG_DEBUG("Target Fuzzy Simplicial Set");
    std::stringstream ss;
    ss << ygraph_coo;
    CUML_LOG_DEBUG(ss.str().c_str());
  }
  */

  /**
   * Compute general simplicial set intersection.
   */
  rmm::device_uvector<int> xrow_ind(rgraph_coo->n_rows, stream);
  rmm::device_uvector<int> yrow_ind(ygraph_coo.n_rows, stream);

  RAFT_CUDA_TRY(cudaMemsetAsync(xrow_ind.data(), 0, rgraph_coo->n_rows * sizeof(int), stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(yrow_ind.data(), 0, ygraph_coo.n_rows * sizeof(int), stream));

  raft::sparse::COO<value_t> cygraph_coo(stream);
  raft::sparse::op::coo_remove_zeros<value_t>(&ygraph_coo, &cygraph_coo, stream);

  raft::sparse::convert::sorted_coo_to_csr(&cygraph_coo, yrow_ind.data(), stream);
  raft::sparse::convert::sorted_coo_to_csr(rgraph_coo, xrow_ind.data(), stream);

  raft::sparse::COO<value_t> result_coo(stream);
  general_simplicial_set_intersection<value_t, nnz_t, TPB_X>(xrow_ind.data(),
                                                             rgraph_coo,
                                                             yrow_ind.data(),
                                                             &cygraph_coo,
                                                             &result_coo,
                                                             params->target_weight,
                                                             stream);

  /**
   * Remove zeros
   */
  raft::sparse::COO<value_t> out(stream);
  raft::sparse::op::coo_remove_zeros<value_t>(&result_coo, &out, stream);

  reset_local_connectivity<value_t, TPB_X>(&out, final_coo, stream);

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}
}  // namespace Supervised
}  // namespace UMAPAlgo
