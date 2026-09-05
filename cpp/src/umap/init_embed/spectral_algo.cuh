/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/manifold/spectral_embedding.hpp>
#include <cuml/manifold/umapparams.h>

#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/handle.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/transpose.cuh>
#include <raft/random/rng.cuh>
#include <raft/sparse/coo.hpp>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include <stdint.h>

namespace UMAPAlgo {

namespace InitEmbed {

namespace SpectralInit {

using namespace ML;

/**
 * Performs a spectral layout initialization
 */
template <typename T, typename nnz_t>
void launcher(const raft::handle_t& handle,
              nnz_t n,
              int d,
              raft::sparse::COO<float>* coo,
              UMAPParams* params,
              T* embedding)
{
  cudaStream_t stream = handle.get_stream().get();

  ASSERT(n > static_cast<nnz_t>(params->n_components),
         "Spectral layout requires n_samples > n_components");

  auto tmp_embedding =
    raft::make_device_matrix<float, int, raft::col_major>(handle, n, params->n_components);

  auto connectivity_graph_view = raft::make_device_coo_matrix_view<float, int, int, int64_t>(
    coo->vals(),
    raft::make_device_coordinate_structure_view<int, int, int64_t>(
      coo->rows(), coo->cols(), n, n, coo->nnz));

  ML::SpectralEmbedding::params spectral_params;
  spectral_params.n_neighbors    = params->n_neighbors;
  spectral_params.norm_laplacian = true;
  spectral_params.drop_first     = true;
  spectral_params.seed           = params->random_state;
  spectral_params.n_components =
    spectral_params.drop_first ? params->n_components + 1 : params->n_components;

  uint64_t seed = params->random_state;

  ML::SpectralEmbedding::transform(
    handle, spectral_params, connectivity_graph_view, tmp_embedding.view());

  raft::linalg::transpose(
    handle, tmp_embedding.data_handle(), embedding, n, params->n_components, stream);

  raft::linalg::unaryOp<T>(
    tmp_embedding.data_handle(),
    tmp_embedding.data_handle(),
    n * params->n_components,
    [=] __device__(T input) { return fabsf(input); },
    stream);

  thrust::device_ptr<T> d_ptr = thrust::device_pointer_cast(tmp_embedding.data_handle());
  T max =
    *(thrust::max_element(thrust::cuda::par.on(stream), d_ptr, d_ptr + (n * params->n_components)));

  // Reuse tmp_storage to add random noise
  raft::random::Rng r(seed);
  r.normal(tmp_embedding.data_handle(), n * params->n_components, 0.0f, 0.0001f, stream);

  raft::linalg::unaryOp<T>(
    embedding,
    embedding,
    n * params->n_components,
    [=] __device__(T input) { return (10.0f / max) * input; },
    stream);

  raft::linalg::add(
    embedding, embedding, tmp_embedding.data_handle(), n * params->n_components, stream);

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}
}  // namespace SpectralInit
}  // namespace InitEmbed
};  // namespace UMAPAlgo
