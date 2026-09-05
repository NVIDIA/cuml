/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runner.cuh"

#include <cuml/manifold/common.hpp>
#include <cuml/manifold/umap.hpp>
#include <cuml/manifold/umapparams.h>

#include <raft/core/handle.hpp>
#include <raft/core/host_container_policy.hpp>
#include <raft/util/cuda_utils.cuh>

#include <stdint.h>

#include <iostream>

namespace ML {
namespace UMAP {

static const int TPB_X = 256;

inline bool dispatch_to_uint64_t(int n_rows, int n_neighbors, int n_components)
{
  // The fuzzy simplicial set graph can have at most 2 * n * n_neighbors elements after
  // symmetrization and removal of zeroes
  uint64_t nnz1 = 2 * static_cast<uint64_t>(n_rows) * n_neighbors;
  // The embeddings have n * n_neighbors elements
  uint64_t nnz2 = static_cast<uint64_t>(n_rows) * n_components;
  return nnz1 > std::numeric_limits<int32_t>::max() || nnz2 > std::numeric_limits<int32_t>::max();
}

template <typename nnz_t>
inline std::unique_ptr<raft::sparse::COO<float, int>> _get_graph(
  const raft::handle_t& handle,
  float* X,  // input matrix
  float* y,  // labels
  int n,
  int d,
  knn_indices_dense_t* knn_indices,  // precomputed indices
  float* knn_dists,                  // precomputed distances
  UMAPParams* params)
{
  auto graph = std::make_unique<raft::sparse::COO<float>>(handle.get_stream().get());
  if (knn_indices != nullptr && knn_dists != nullptr) {
    CUML_LOG_DEBUG("Calling UMAP::get_graph() with precomputed KNN");

    manifold_precomputed_knn_inputs_t<knn_indices_dense_t, float> inputs(
      knn_indices, knn_dists, y, n, d, params->n_neighbors);
    if (y != nullptr) {
      UMAPAlgo::_get_graph_supervised<knn_indices_dense_t,
                                      float,
                                      manifold_precomputed_knn_inputs_t<knn_indices_dense_t, float>,
                                      nnz_t,
                                      TPB_X>(handle, inputs, params, graph.get());
    } else {
      UMAPAlgo::_get_graph<knn_indices_dense_t,
                           float,
                           manifold_precomputed_knn_inputs_t<knn_indices_dense_t, float>,
                           nnz_t,
                           TPB_X>(handle, inputs, params, graph.get());
    }
    return graph;
  } else {
    manifold_dense_inputs_t<float> inputs(X, y, n, d);
    if (y != nullptr) {
      UMAPAlgo::_get_graph_supervised<knn_indices_dense_t,
                                      float,
                                      manifold_dense_inputs_t<float>,
                                      nnz_t,
                                      TPB_X>(handle, inputs, params, graph.get());
    } else {
      UMAPAlgo::
        _get_graph<knn_indices_dense_t, float, manifold_dense_inputs_t<float>, nnz_t, TPB_X>(
          handle, inputs, params, graph.get());
    }
    return graph;
  }
}

template <typename nnz_t>
inline void _refine(const raft::handle_t& handle,
                    float* X,
                    int n,
                    int d,
                    raft::sparse::COO<float>* graph,
                    UMAPParams* params,
                    float* embeddings)
{
  CUML_LOG_DEBUG("Calling UMAP::refine() with precomputed KNN");
  manifold_dense_inputs_t<float> inputs(X, nullptr, n, d);
  UMAPAlgo::_refine<knn_indices_dense_t, float, manifold_dense_inputs_t<float>, nnz_t, TPB_X>(
    handle, inputs, params, graph, embeddings);
}

template <typename nnz_t>
inline void _init_and_refine(const raft::handle_t& handle,
                             float* X,
                             int n,
                             int d,
                             raft::sparse::COO<float>* graph,
                             UMAPParams* params,
                             float* embeddings)
{
  CUML_LOG_DEBUG("Calling UMAP::init_and_refine() with precomputed KNN");
  manifold_dense_inputs_t<float> inputs(X, nullptr, n, d);
  UMAPAlgo::
    _init_and_refine<knn_indices_dense_t, float, manifold_dense_inputs_t<float>, nnz_t, TPB_X>(
      handle, inputs, params, graph, embeddings);
}

template <typename nnz_t>
inline void _fit(const raft::handle_t& handle,
                 float* X,
                 float* y,
                 int n,
                 int d,
                 knn_indices_dense_t* knn_indices,
                 float* knn_dists,
                 UMAPParams* params,
                 std::unique_ptr<rmm::device_buffer>& embeddings,
                 raft::host_coo_matrix<float, int, int, uint64_t>& graph,
                 float* sigmas = nullptr,
                 float* rhos   = nullptr)
{
  if (knn_indices != nullptr && knn_dists != nullptr) {
    CUML_LOG_DEBUG("Calling UMAP::fit() with precomputed KNN");

    manifold_precomputed_knn_inputs_t<knn_indices_dense_t, float> inputs(
      knn_indices, knn_dists, y, n, d, params->n_neighbors);
    if (y != nullptr) {
      UMAPAlgo::_fit_supervised<knn_indices_dense_t,
                                float,
                                manifold_precomputed_knn_inputs_t<knn_indices_dense_t, float>,
                                nnz_t,
                                TPB_X>(handle, inputs, params, embeddings, graph);
    } else {
      UMAPAlgo::_fit<knn_indices_dense_t,
                     float,
                     manifold_precomputed_knn_inputs_t<knn_indices_dense_t, float>,
                     nnz_t,
                     TPB_X>(handle, inputs, params, embeddings, graph, sigmas, rhos);
    }

  } else {
    manifold_dense_inputs_t<float> inputs(X, y, n, d);
    if (y != nullptr) {
      UMAPAlgo::
        _fit_supervised<knn_indices_dense_t, float, manifold_dense_inputs_t<float>, nnz_t, TPB_X>(
          handle, inputs, params, embeddings, graph);
    } else {
      UMAPAlgo::_fit<knn_indices_dense_t, float, manifold_dense_inputs_t<float>, nnz_t, TPB_X>(
        handle, inputs, params, embeddings, graph, sigmas, rhos);
    }
  }
}

template <typename nnz_t>
inline void _fit_sparse(const raft::handle_t& handle,
                        int* indptr,
                        int* indices,
                        float* data,
                        size_t nnz,
                        float* y,
                        int n,
                        int d,
                        int* knn_indices,
                        float* knn_dists,
                        UMAPParams* params,
                        std::unique_ptr<rmm::device_buffer>& embeddings,
                        raft::host_coo_matrix<float, int, int, uint64_t>& graph)
{
  if (knn_indices != nullptr && knn_dists != nullptr) {
    CUML_LOG_DEBUG("Calling UMAP::fit_sparse() with precomputed KNN");

    manifold_precomputed_knn_inputs_t<knn_indices_sparse_t, float> inputs(
      knn_indices, knn_dists, y, n, d, params->n_neighbors);
    if (y != nullptr) {
      UMAPAlgo::_fit_supervised<knn_indices_sparse_t,
                                float,
                                manifold_precomputed_knn_inputs_t<knn_indices_sparse_t, float>,
                                nnz_t,
                                TPB_X>(handle, inputs, params, embeddings, graph);
    } else {
      UMAPAlgo::_fit<knn_indices_sparse_t,
                     float,
                     manifold_precomputed_knn_inputs_t<knn_indices_sparse_t, float>,
                     nnz_t,
                     TPB_X>(handle, inputs, params, embeddings, graph);
    }
  } else {
    manifold_sparse_inputs_t<int, float> inputs(indptr, indices, data, y, nnz, n, d);
    if (y != nullptr) {
      UMAPAlgo::_fit_supervised<knn_indices_sparse_t,
                                float,
                                manifold_sparse_inputs_t<knn_indices_sparse_t, float>,
                                nnz_t,
                                TPB_X>(handle, inputs, params, embeddings, graph);
    } else {
      UMAPAlgo::_fit<knn_indices_sparse_t,
                     float,
                     manifold_sparse_inputs_t<knn_indices_sparse_t, float>,
                     nnz_t,
                     TPB_X>(handle, inputs, params, embeddings, graph);
    }
  }
}

template <typename nnz_t>
inline void _transform(const raft::handle_t& handle,
                       float* X,
                       int n,
                       int d,
                       float* orig_X,
                       int orig_n,
                       float* embedding,
                       int embedding_n,
                       UMAPParams* params,
                       float* transformed)
{
  RAFT_EXPECTS(params->build_algo == ML::UMAPParams::graph_build_algo::BRUTE_FORCE_KNN,
               "build algo nn_descent not supported for transform()");
  manifold_dense_inputs_t<float> inputs(X, nullptr, n, d);
  manifold_dense_inputs_t<float> orig_inputs(orig_X, nullptr, orig_n, d);
  UMAPAlgo::_transform<knn_indices_dense_t, float, manifold_dense_inputs_t<float>, nnz_t, TPB_X>(
    handle, inputs, orig_inputs, embedding, embedding_n, params, transformed);
}

template <typename nnz_t>
inline void _transform_sparse(const raft::handle_t& handle,
                              int* indptr,
                              int* indices,
                              float* data,
                              size_t nnz,
                              int n,
                              int d,
                              int* orig_x_indptr,
                              int* orig_x_indices,
                              float* orig_x_data,
                              size_t orig_nnz,
                              int orig_n,
                              float* embedding,
                              int embedding_n,
                              UMAPParams* params,
                              float* transformed)
{
  RAFT_EXPECTS(params->build_algo == ML::UMAPParams::graph_build_algo::BRUTE_FORCE_KNN,
               "build algo nn_descent not supported for transform()");
  manifold_sparse_inputs_t<knn_indices_sparse_t, float> inputs(
    indptr, indices, data, nullptr, nnz, n, d);
  manifold_sparse_inputs_t<knn_indices_sparse_t, float> orig_x_inputs(
    orig_x_indptr, orig_x_indices, orig_x_data, nullptr, orig_nnz, orig_n, d);

  UMAPAlgo::
    _transform<knn_indices_sparse_t, float, manifold_sparse_inputs_t<int, float>, nnz_t, TPB_X>(
      handle, inputs, orig_x_inputs, embedding, embedding_n, params, transformed);
}

template <typename nnz_t>
inline void _inverse_transform(const raft::handle_t& handle,
                               float* inv_transformed,
                               int n,
                               int n_features,
                               float* orig_X,
                               int orig_n,
                               int* graph_rows,
                               int* graph_cols,
                               float* graph_vals,
                               int nnz,
                               float* sigmas,
                               float* rhos,
                               UMAPParams* params,
                               int n_epochs)
{
  cudaStream_t stream = handle.get_stream().get();

  // Compute epochs_per_sample from graph weights
  rmm::device_uvector<float> epochs_per_sample(nnz, stream);
  UMAPAlgo::SimplSetEmbed::Algo::make_epochs_per_sample<float, int, nnz_t>(
    graph_vals, nnz, n_epochs, epochs_per_sample.data(), stream);

  // Run inverse optimization
  UMAPAlgo::SimplSetEmbed::Algo::optimize_layout_inverse<float, nnz_t, TPB_X>(
    inv_transformed,
    n,
    orig_X,
    orig_n,
    graph_rows,
    graph_cols,
    graph_vals,
    sigmas,
    rhos,
    nnz,
    epochs_per_sample.data(),
    params->repulsion_strength,
    params,
    n_epochs,
    n_features,
    stream);
}

}  // namespace UMAP
}  // namespace ML
