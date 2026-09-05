/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "random_algo.cuh"
#include "spectral_algo.cuh"

#include <cuml/manifold/umapparams.h>

#include <raft/sparse/coo.hpp>

namespace UMAPAlgo {

namespace InitEmbed {

using namespace ML;

template <typename T, typename nnz_t>
void run(const raft::handle_t& handle,
         int n,
         int d,
         raft::sparse::COO<float>* coo,
         UMAPParams* params,
         T* embedding,
         cudaStream_t stream,
         int algo = 0)
{
  switch (algo) {
    /**
     * Initial algo uses FAISS indices
     */
    case 0: RandomInit::launcher<T, nnz_t>(n, d, params, embedding, handle.get_stream().get()); break;

    case 1: try { SpectralInit::launcher<T, nnz_t>(handle, n, d, coo, params, embedding);
      } catch (const raft::exception& e) {
        CUML_LOG_WARN("Spectral initialization failed, using random initialization instead.");
        RandomInit::launcher<T, nnz_t>(n, d, params, embedding, handle.get_stream().get());
      }
      break;

    case 2: break;  // custom initialization case
  }
}
}  // namespace InitEmbed
};  // namespace UMAPAlgo
