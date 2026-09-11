/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tsne_runner.cuh"

#include <cuml/manifold/tsne.h>

#include <raft/core/handle.hpp>

#include <cuvs/distance/distance.hpp>

#include <utility>

namespace ML {
template <typename tsne_input, typename value_idx, typename value_t>
std::pair<float, int> _fit(const raft::handle_t& handle,
                           tsne_input& input,
                           knn_graph<value_idx, value_t>& k_graph,
                           TSNEParams& params)
{
  TSNE_runner<tsne_input, value_idx, value_t> runner(handle, input, k_graph, params);

  return runner.run();
}

void TSNE_fit(const raft::handle_t& handle,
              float* X,
              float* Y,
              int n,
              int p,
              int64_t* knn_indices,
              float* knn_dists,
              TSNEParams& params,
              float* kl_div,
              int* n_iter)
{
  ASSERT(n > 0 && p > 0 && params.dim > 0 && params.n_neighbors > 0 && X != NULL && Y != NULL,
         "Wrong input args");

  manifold_dense_inputs_t<float> input(X, Y, n, p);
  auto k_graph = make_tsne_knn_graph<int64_t, float>(n, knn_indices, knn_dists, params);

  auto stats = _fit<manifold_dense_inputs_t<float>, knn_indices_dense_t, float>(
    handle, input, k_graph, params);

  if (kl_div) { *kl_div = stats.first; }
  if (n_iter) { *n_iter = stats.second; }
}

void TSNE_fit_sparse(const raft::handle_t& handle,
                     int* indptr,
                     int* indices,
                     float* data,
                     float* Y,
                     int nnz,
                     int n,
                     int p,
                     int* knn_indices,
                     float* knn_dists,
                     TSNEParams& params,
                     float* kl_div,
                     int* n_iter)
{
  ASSERT(n > 0 && p > 0 && params.dim > 0 && params.n_neighbors > 0 && indptr != NULL &&
           indices != NULL && data != NULL && Y != NULL,
         "Wrong input args");

  manifold_sparse_inputs_t<int, float> input(indptr, indices, data, Y, nnz, n, p);
  auto k_graph = make_tsne_knn_graph<int, float>(n, knn_indices, knn_dists, params);

  auto stats = _fit<manifold_sparse_inputs_t<int, float>, knn_indices_sparse_t, float>(
    handle, input, k_graph, params);

  if (kl_div) { *kl_div = stats.first; }
  if (n_iter) { *n_iter = stats.second; }
}

}  // namespace ML
