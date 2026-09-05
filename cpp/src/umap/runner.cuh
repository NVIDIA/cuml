/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "fuzzy_simpl_set/runner.cuh"
#include "init_embed/runner.cuh"
#include "knn_graph/runner.cuh"
#include "optimize.cuh"
#include "simpl_set_embed/runner.cuh"
#include "supervised.cuh"

#include <common/nvtx.hpp>

#include <cuml/common/logger.hpp>
#include <cuml/manifold/common.hpp>
#include <cuml/manifold/umapparams.h>

#include <raft/core/handle.hpp>
#include <raft/core/host_coo_matrix.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/linalg/norm.cuh>
#include <raft/sparse/op/filter.cuh>
#include <raft/sparse/op/sort.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_buffer.hpp>

#include <cuda/std/tuple>
#include <cuda_runtime.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>

#include <memory>
#include <type_traits>

namespace UMAPAlgo {

// Swap this as impls change for now.
namespace FuzzySimplSetImpl = FuzzySimplSet::Naive;
namespace SimplSetEmbedImpl = SimplSetEmbed::Algo;

using namespace ML;

template <int TPB_X, typename T>
CUML_KERNEL void init_transform(int* indices,
                                T* weights,
                                int n,
                                const T* embeddings,
                                int embeddings_n,
                                int n_components,
                                T* result,
                                int n_neighbors)
{
  // row-based matrix 1 thread per row
  int row = (blockIdx.x * TPB_X) + threadIdx.x;
  int i   = row * n_neighbors;  // each thread processes one row of the dist matrix

  if (row < n) {
    for (int j = 0; j < n_neighbors; j++) {
      for (int d = 0; d < n_components; d++) {
        result[row * n_components + d] +=
          weights[i + j] * embeddings[indices[i + j] * n_components + d];
      }
    }
  }
}

/**
 * Fit exponential decay curve to find the parameters
 * a and b, which are based on min_dist and spread
 * parameters.
 */
inline void find_ab(UMAPParams* params, cudaStream_t stream)
{
  Optimize::find_params_ab(params, stream);
}

inline int get_n_epochs(UMAPParams* params, uint64_t n_rows)
{
  int n_epochs = params->n_epochs;
  if (n_epochs <= 0) {
    if (n_rows <= 10000)
      n_epochs = 500;
    else
      n_epochs = 200;
  }
  return n_epochs;
}

template <typename value_t>
inline value_t get_threshold(const raft::handle_t& handle,
                             raft::sparse::COO<value_t>& in,
                             int n_epochs)
{
  auto stream             = raft::resource::get_cuda_stream(handle);
  auto thrust_exec_policy = raft::resource::get_thrust_policy(handle);

  thrust::device_ptr<const value_t> vals_ptr = thrust::device_pointer_cast(in.vals());
  value_t max_val = *(thrust::max_element(thrust_exec_policy, vals_ptr, vals_ptr + in.nnz));

  value_t threshold = max_val / static_cast<value_t>(n_epochs);
  return threshold;
}

template <typename value_t>
void trim_graph(const raft::handle_t& handle, raft::sparse::COO<value_t>& graph, int n_epochs)
{
  auto exec = raft::resource::get_thrust_policy(handle);

  value_t threshold = get_threshold(handle, graph, n_epochs);

  auto zipped_begin =
    thrust::make_zip_iterator(cuda::std::make_tuple(graph.rows(), graph.cols(), graph.vals()));
  auto zipped_end = zipped_begin + graph.nnz;

  using row_type   = typename std::remove_pointer<decltype(graph.rows())>::type;
  using col_type   = typename std::remove_pointer<decltype(graph.cols())>::type;
  using tuple_type = cuda::std::tuple<row_type, col_type, value_t>;

  auto new_end = thrust::remove_if(
    exec, zipped_begin, zipped_end, [threshold] __host__ __device__(const tuple_type& t) {
      return cuda::std::get<2>(t) < threshold;
    });

  auto new_nnz = static_cast<decltype(graph.nnz)>(thrust::distance(zipped_begin, new_end));
  graph.nnz    = new_nnz;
}

template <typename value_t, typename value_idx, typename nnz_t>
void copy_device_graph_to_host(const raft::handle_t& handle,
                               raft::sparse::COO<value_t, value_idx, nnz_t>& device_graph,
                               raft::host_coo_matrix<float, int, int, uint64_t>& host_graph)
{
  auto stream = raft::resource::get_cuda_stream(handle);

  host_graph.initialize_sparsity(device_graph.nnz);
  raft::copy(
    host_graph.structure_view().get_rows().data(), device_graph.rows(), device_graph.nnz, stream);
  raft::copy(
    host_graph.structure_view().get_cols().data(), device_graph.cols(), device_graph.nnz, stream);
  raft::copy(host_graph.get_elements().data(), device_graph.vals(), device_graph.nnz, stream);
}

template <typename value_idx, typename value_t, typename umap_inputs, typename nnz_t, int TPB_X>
void _get_strengths(const raft::handle_t& handle,
                    const umap_inputs& inputs,
                    UMAPParams* params,
                    raft::sparse::COO<value_t>& strengths,
                    value_t* out_sigmas = nullptr,
                    value_t* out_rhos   = nullptr)
{
  cudaStream_t stream = handle.get_stream().get();

  int n_neighbors       = params->n_neighbors;
  nnz_t n_x_n_neighbors = static_cast<nnz_t>(inputs.n) * n_neighbors;

  strengths.allocate(n_x_n_neighbors, inputs.n, inputs.n, true, stream);

  std::unique_ptr<rmm::device_uvector<value_idx>> knn_indices_b = nullptr;
  std::unique_ptr<rmm::device_uvector<value_t>> knn_dists_b     = nullptr;
  knn_graph<value_idx, value_t> knn_graph(inputs.n, n_neighbors);
  /* If not given precomputed knn graph, compute it */
  if (inputs.needs_device_knn_graph()) {
    /* Allocate workspace for kNN graph */
    knn_indices_b = std::make_unique<rmm::device_uvector<value_idx>>(n_x_n_neighbors, stream);
    knn_dists_b   = std::make_unique<rmm::device_uvector<value_t>>(n_x_n_neighbors, stream);
    knn_graph.knn_indices = knn_indices_b->data();
    knn_graph.knn_dists   = knn_dists_b->data();
  }

  CUML_LOG_DEBUG("Computing KNN Graph");
  raft::common::nvtx::push_range("umap::knnGraph");
  kNNGraph::run<value_idx, value_t, umap_inputs>(
    handle, inputs, inputs, knn_graph, n_neighbors, params, stream);
  raft::common::nvtx::pop_range();

  CUML_LOG_DEBUG("Computing fuzzy simplicial set");
  raft::common::nvtx::push_range("umap::simplicial_set");
  FuzzySimplSetImpl::compute_membership_strength<value_t, value_idx, nnz_t, TPB_X>(
    inputs.n,
    knn_graph.knn_indices,
    knn_graph.knn_dists,
    n_neighbors,
    strengths,
    params,
    stream,
    out_sigmas,
    out_rhos);
  raft::common::nvtx::pop_range();
}

template <typename value_idx, typename value_t, typename umap_inputs, typename nnz_t, int TPB_X>
void _get_graph(const raft::handle_t& handle,
                const umap_inputs& inputs,
                UMAPParams* params,
                raft::sparse::COO<value_t, int>* graph,
                value_t* out_sigmas = nullptr,
                value_t* out_rhos   = nullptr)
{
  raft::common::nvtx::range fun_scope("umap::supervised::_get_graph");
  cudaStream_t stream = handle.get_stream().get();

  ML::default_logger().set_level(params->verbosity);

  /* Nested scopes used here to drop resources earlier, reducing device memory usage */
  raft::sparse::COO<value_t> fss_graph(stream);
  {
    raft::sparse::COO<value_t> strengths(stream);
    _get_strengths<value_idx, value_t, umap_inputs, nnz_t, TPB_X>(
      handle, inputs, params, strengths, out_sigmas, out_rhos);
    FuzzySimplSetImpl::symmetrize<value_t>(strengths, fss_graph, params->set_op_mix_ratio, stream);
  }  // end strengths scope

  /* Canonicalize output graph */
  raft::sparse::op::coo_sort<value_t>(&fss_graph, stream);

  raft::sparse::op::coo_remove_zeros<value_t>(&fss_graph, graph, stream);
}

template <typename value_idx, typename value_t, typename umap_inputs, typename nnz_t, int TPB_X>
void _get_graph_supervised(const raft::handle_t& handle,
                           const umap_inputs& inputs,
                           UMAPParams* params,
                           raft::sparse::COO<value_t, int>* graph)
{
  if (params->target_n_neighbors == -1) params->target_n_neighbors = params->n_neighbors;

  cudaStream_t stream = handle.get_stream().get();

  /* Nested scopes used here to drop resources earlier, reducing device memory usage */
  raft::sparse::COO<value_t> ci_graph(stream);
  {
    raft::sparse::COO<value_t> fss_graph(stream);
    _get_graph<value_idx, value_t, umap_inputs, nnz_t, TPB_X>(handle, inputs, params, &fss_graph);

    if (params->target_metric == ML::UMAPParams::MetricType::CATEGORICAL) {
      CUML_LOG_DEBUG("Performing categorical intersection");
      Supervised::perform_categorical_intersection<value_t, nnz_t, TPB_X>(
        inputs.y, &fss_graph, &ci_graph, params, stream);
    } else {
      CUML_LOG_DEBUG("Performing general intersection");
      Supervised::perform_general_intersection<value_idx, value_t, nnz_t, TPB_X>(
        handle, inputs.y, &fss_graph, &ci_graph, params, stream);
    }
  }  // end fss_graph scope

  /* Canonicalize output graph */
  raft::sparse::op::coo_sort<value_t>(&ci_graph, stream);

  raft::sparse::op::coo_remove_zeros<value_t>(&ci_graph, graph, stream);
}

template <typename value_idx, typename value_t, typename umap_inputs, typename nnz_t, int TPB_X>
void _refine(const raft::handle_t& handle,
             const umap_inputs& inputs,
             UMAPParams* params,
             raft::sparse::COO<value_t>* graph,
             value_t* embeddings)
{
  cudaStream_t stream = handle.get_stream().get();
  ML::default_logger().set_level(params->verbosity);

  int n_epochs = get_n_epochs(params, inputs.n);
  trim_graph(handle, *graph, n_epochs);

  /**
   * Run simplicial set embedding to approximate low-dimensional representation
   */
  SimplSetEmbed::run<value_t, nnz_t, TPB_X>(
    inputs.n, inputs.d, graph, params, embeddings, n_epochs, stream);
}

template <typename value_idx, typename value_t, typename umap_inputs, typename nnz_t, int TPB_X>
void _init_and_refine(const raft::handle_t& handle,
                      const umap_inputs& inputs,
                      UMAPParams* params,
                      raft::sparse::COO<value_t>* graph,
                      value_t* embeddings)
{
  cudaStream_t stream = handle.get_stream().get();
  ML::default_logger().set_level(params->verbosity);

  int n_epochs = get_n_epochs(params, inputs.n);
  trim_graph(handle, *graph, n_epochs);

  // Initialize embeddings
  InitEmbed::run<value_t, nnz_t>(
    handle, inputs.n, inputs.d, graph, params, embeddings, stream, params->init);

  // Run simplicial set embedding
  SimplSetEmbed::run<value_t, nnz_t, TPB_X>(
    inputs.n, inputs.d, graph, params, embeddings, n_epochs, stream);
}

template <typename value_idx, typename value_t, typename umap_inputs, typename nnz_t, int TPB_X>
void _fit(const raft::handle_t& handle,
          const umap_inputs& inputs,
          UMAPParams* params,
          std::unique_ptr<rmm::device_buffer>& embeddings,
          raft::host_coo_matrix<float, int, int, uint64_t>& host_graph,
          value_t* out_sigmas = nullptr,
          value_t* out_rhos   = nullptr)
{
  raft::common::nvtx::range fun_scope("umap::unsupervised::fit");

  auto stream = raft::resource::get_cuda_stream(handle);
  ML::default_logger().set_level(params->verbosity);

  int n_epochs = get_n_epochs(params, inputs.n);

  raft::sparse::COO<value_t> graph(stream.get());
  UMAPAlgo::_get_graph<value_idx, value_t, umap_inputs, nnz_t, TPB_X>(
    handle, inputs, params, &graph, out_sigmas, out_rhos);

  copy_device_graph_to_host(handle, graph, host_graph);

  trim_graph(handle, graph, n_epochs);

  // Allocate embeddings buffer just before initialization
  std::size_t embeddings_size =
    static_cast<std::size_t>(inputs.n) * params->n_components * sizeof(value_t);
  if (!embeddings) { embeddings = std::make_unique<rmm::device_buffer>(embeddings_size, stream); }
  value_t* embeddings_ptr = static_cast<value_t*>(embeddings->data());

  /**
   * Run initialization method
   */
  raft::common::nvtx::push_range("umap::embedding");
  InitEmbed::run<value_t, nnz_t>(
    handle, inputs.n, inputs.d, &graph, params, embeddings_ptr, stream.get(), params->init);

  if (params->callback) {
    params->callback->setup<value_t>(inputs.n, params->n_components);
    params->callback->on_preprocess_end(embeddings_ptr);
  }

  /**
   * Run simplicial set embedding to approximate low-dimensional representation
   */
  SimplSetEmbed::run<value_t, nnz_t, TPB_X>(
    inputs.n, inputs.d, &graph, params, embeddings_ptr, n_epochs, stream.get());
  raft::common::nvtx::pop_range();

  if (params->callback) params->callback->on_train_end(embeddings_ptr);

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename value_idx, typename value_t, typename umap_inputs, typename nnz_t, int TPB_X>
void _fit_supervised(const raft::handle_t& handle,
                     const umap_inputs& inputs,
                     UMAPParams* params,
                     std::unique_ptr<rmm::device_buffer>& embeddings,
                     raft::host_coo_matrix<float, int, int, uint64_t>& host_graph)
{
  raft::common::nvtx::range fun_scope("umap::supervised::fit");

  auto stream = handle.get_stream().get();
  ML::default_logger().set_level(params->verbosity);

  int n_epochs = get_n_epochs(params, inputs.n);

  raft::sparse::COO<value_t> graph(stream);
  UMAPAlgo::_get_graph_supervised<value_idx, value_t, umap_inputs, nnz_t, TPB_X>(
    handle, inputs, params, &graph);

  copy_device_graph_to_host(handle, graph, host_graph);

  trim_graph(handle, graph, n_epochs);

  // Allocate embeddings buffer just before initialization
  std::size_t embeddings_size =
    static_cast<std::size_t>(inputs.n) * params->n_components * sizeof(value_t);
  if (!embeddings) { embeddings = std::make_unique<rmm::device_buffer>(embeddings_size, stream); }
  value_t* embeddings_ptr = static_cast<value_t*>(embeddings->data());

  /**
   * Initialize embeddings
   */
  raft::common::nvtx::push_range("umap::supervised::fit");
  InitEmbed::run<value_t, nnz_t>(
    handle, inputs.n, inputs.d, &graph, params, embeddings_ptr, stream, params->init);

  if (params->callback) {
    params->callback->setup<value_t>(inputs.n, params->n_components);
    params->callback->on_preprocess_end(embeddings_ptr);
  }

  /**
   * Run simplicial set embedding to approximate low-dimensional representation
   */
  SimplSetEmbed::run<value_t, nnz_t, TPB_X>(
    inputs.n, inputs.d, &graph, params, embeddings_ptr, n_epochs, stream);
  raft::common::nvtx::pop_range();

  if (params->callback) params->callback->on_train_end(embeddings_ptr);

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 *
 */
template <typename value_idx, typename value_t, typename umap_inputs, typename nnz_t, int TPB_X>
void _transform(const raft::handle_t& handle,
                const umap_inputs& inputs,
                umap_inputs& orig_x_inputs,
                value_t* embedding,
                int embedding_n,
                UMAPParams* params,
                value_t* transformed)
{
  raft::common::nvtx::range fun_scope("umap::transform");
  cudaStream_t stream = handle.get_stream().get();

  ML::default_logger().set_level(params->verbosity);

  CUML_LOG_DEBUG("Running transform");

  CUML_LOG_DEBUG("Building KNN Graph");

  raft::common::nvtx::push_range("umap::knnGraph");
  std::unique_ptr<rmm::device_uvector<value_idx>> knn_indices_b = nullptr;
  std::unique_ptr<rmm::device_uvector<value_t>> knn_dists_b     = nullptr;

  int k = params->n_neighbors;

  knn_graph<value_idx, value_t> knn_graph(inputs.n, k);

  /**
   * If not given precomputed knn graph, compute it
   */

  if (inputs.needs_device_knn_graph()) {
    /**
     * Allocate workspace for kNN graph
     */
    knn_indices_b =
      std::make_unique<rmm::device_uvector<value_idx>>(static_cast<nnz_t>(inputs.n) * k, stream);
    knn_dists_b =
      std::make_unique<rmm::device_uvector<value_t>>(static_cast<nnz_t>(inputs.n) * k, stream);

    knn_graph.knn_indices = knn_indices_b->data();
    knn_graph.knn_dists   = knn_dists_b->data();
  }

  kNNGraph::run<value_idx, value_t, umap_inputs>(
    handle, orig_x_inputs, inputs, knn_graph, k, params, stream);

  raft::common::nvtx::pop_range();

  raft::common::nvtx::push_range("umap::smooth_knn");
  float adjusted_local_connectivity = max(0.0, params->local_connectivity - 1.0);

  CUML_LOG_DEBUG("Smoothing KNN distances");

  /**
   * Perform smooth_knn_dist
   */
  rmm::device_uvector<value_t> sigmas(inputs.n, stream);
  rmm::device_uvector<value_t> rhos(inputs.n, stream);
  RAFT_CUDA_TRY(
    cudaMemsetAsync(sigmas.data(), 0, static_cast<nnz_t>(inputs.n) * sizeof(value_t), stream));
  RAFT_CUDA_TRY(
    cudaMemsetAsync(rhos.data(), 0, static_cast<nnz_t>(inputs.n) * sizeof(value_t), stream));

  dim3 grid_n(raft::ceildiv(inputs.n, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  FuzzySimplSetImpl::smooth_knn_dist<value_t, value_idx, nnz_t, TPB_X>(inputs.n,
                                                                       knn_graph.knn_indices,
                                                                       knn_graph.knn_dists,
                                                                       rhos.data(),
                                                                       sigmas.data(),
                                                                       params,
                                                                       params->n_neighbors,
                                                                       adjusted_local_connectivity,
                                                                       stream);
  raft::common::nvtx::pop_range();

  /**
   * Compute graph of membership strengths
   */

  nnz_t nnz = static_cast<nnz_t>(inputs.n) * params->n_neighbors;

  dim3 grid_nnz(raft::ceildiv(nnz, static_cast<nnz_t>(TPB_X)), 1, 1);

  CUML_LOG_DEBUG("Executing fuzzy simplicial set");

  /**
   * Allocate workspace for fuzzy simplicial set.
   */

  raft::sparse::COO<value_t> graph_coo(stream, nnz, inputs.n, inputs.n);

  nnz_t to_process = static_cast<nnz_t>(graph_coo.n_rows) * params->n_neighbors;
  FuzzySimplSetImpl::compute_membership_strength_kernel<value_t, value_idx, nnz_t, TPB_X>
    <<<grid_nnz, blk, 0, stream>>>(knn_graph.knn_indices,
                                   knn_graph.knn_dists,
                                   sigmas.data(),
                                   rhos.data(),
                                   graph_coo.vals(),
                                   graph_coo.rows(),
                                   graph_coo.cols(),
                                   params->n_neighbors,
                                   to_process);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  rmm::device_uvector<nnz_t> row_ind(inputs.n, stream);
  raft::sparse::convert::sorted_coo_to_csr(&graph_coo, row_ind.data(), stream);

  rmm::device_uvector<value_t> vals_normed(graph_coo.nnz, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(vals_normed.data(), 0, graph_coo.nnz * sizeof(value_t), stream));

  CUML_LOG_DEBUG("Performing L1 normalization");

  raft::sparse::linalg::csr_row_normalize_l1<value_t>(row_ind.data(),
                                                      graph_coo.vals(),
                                                      static_cast<nnz_t>(graph_coo.nnz),
                                                      graph_coo.n_rows,
                                                      vals_normed.data(),
                                                      stream);

  init_transform<TPB_X, value_t><<<grid_n, blk, 0, stream>>>(graph_coo.cols(),
                                                             vals_normed.data(),
                                                             graph_coo.n_rows,
                                                             embedding,
                                                             embedding_n,
                                                             params->n_components,
                                                             transformed,
                                                             params->n_neighbors);

  RAFT_CUDA_TRY(cudaPeekAtLastError());

  int n_epochs = params->n_epochs;
  if (n_epochs <= 0) {
    if (inputs.n <= 10000)
      n_epochs = 100;
    else
      n_epochs = 30;
  } else {
    n_epochs /= 3;
  }

  trim_graph(handle, graph_coo, n_epochs);

  raft::common::nvtx::push_range("umap::optimization");
  CUML_LOG_DEBUG("Computing # of epochs for training each sample");

  rmm::device_uvector<value_t> epochs_per_sample(nnz, stream);

  SimplSetEmbedImpl::make_epochs_per_sample(
    graph_coo.vals(), graph_coo.nnz, n_epochs, epochs_per_sample.data(), stream);

  CUML_LOG_DEBUG("Performing optimization");

  if (params->callback) {
    params->callback->setup<value_t>(inputs.n, params->n_components);
    params->callback->on_preprocess_end(transformed);
  }

  auto original_initial_alpha = params->initial_alpha;
  params->initial_alpha       = original_initial_alpha / 4.0;

  SimplSetEmbedImpl::optimize_layout<value_t, nnz_t, TPB_X>(transformed,
                                                            inputs.n,
                                                            embedding,
                                                            orig_x_inputs.n,
                                                            graph_coo.rows(),
                                                            graph_coo.cols(),
                                                            graph_coo.nnz,
                                                            epochs_per_sample.data(),
                                                            params->repulsion_strength,
                                                            params,
                                                            n_epochs,
                                                            stream);
  params->initial_alpha = original_initial_alpha;
  raft::common::nvtx::pop_range();

  if (params->callback) params->callback->on_train_end(transformed);
}

}  // namespace UMAPAlgo
