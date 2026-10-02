/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/common/logger.hpp>
#include <cuml/common/utils.hpp>
#include <cuml/manifold/umapparams.h>
#include <cuml/neighbors/knn.hpp>

#include <raft/sparse/coo.hpp>
#include <raft/sparse/linalg/symmetrize.cuh>
#include <raft/sparse/op/sort.cuh>
#include <raft/stats/mean.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cuda_runtime.h>

#include <stdio.h>

#include <optional>
#include <string>

namespace UMAPAlgo {
namespace FuzzySimplSet {
namespace Naive {

using namespace ML;

static const float MAX_FLOAT = std::numeric_limits<float>::max();
static const float MIN_FLOAT = std::numeric_limits<float>::min();

static const float SMOOTH_K_TOLERANCE = 1e-5;
static const float MIN_K_DIST_SCALE   = 1e-3;

/**
 * Computes a continuous version of the distance to the kth nearest neighbor.
 * That is, this is similar to knn-distance but allows continuous k values
 * rather than requiring an integral k. In essence, we are simply computing
 * the distance such that the cardinality of fuzzy set we generate is k.
 *
 * TODO: The data needs to be in column-major format (and the indices
 * of knn_dists and knn_inds transposed) so that we can take advantage
 * of read-coalescing within each block where possible.
 *
 * @param knn_dists: Distances to nearest neighbors for each sample. Each row should
 *                   be a sorted list of distances to a given sample's nearest neighbors.
 *
 * @param n: The number of samples
 * @param mean_dist: The mean distance
 * @param sigmas: An array of size n representing the distance to the kth nearest neighbor,
 *                as suitably approximated.
 * @param rhos:  An array of size n representing the distance to the 1st nearest neighbor
 *               for each point.
 * @param n_neighbors: The number of neighbors
 *
 * @param local_connectivity: The local connectivity required -- i.e. the number of nearest
 *                            neighbors that should be assumed to be connected at a local
 *                            level. The higher this value the more connected the manifold
 *                            becomes locally. In practice, this should not be more than the
 *                            local intrinsic dimension of the manifold.
 *
 * @param n_iter The number of smoothing iterations to run
 * @param bandwidth Scale factor for log of neighbors
 *
 * Descriptions adapted from: https://github.com/lmcinnes/umap/blob/master/umap/umap_.py
 *
 */
template <typename value_t, typename nnz_t, int TPB_X>
CUML_KERNEL void smooth_knn_dist_kernel(const value_t* knn_dists,
                                        int n,
                                        float mean_dist,
                                        value_t* sigmas,
                                        value_t* rhos,  // Size of n, initialized to zeros
                                        int n_neighbors,
                                        float local_connectivity = 1.0,
                                        int n_iter               = 64,
                                        float bandwidth          = 1.0)
{
  // row-based matrix 1 thread per row
  int row = (blockIdx.x * TPB_X) + threadIdx.x;
  nnz_t i =
    static_cast<nnz_t>(row) * n_neighbors;  // each thread processes one row of the dist matrix

  if (row < n) {
    float target = __log2f(n_neighbors) * bandwidth;

    float lo  = 0.0;
    float hi  = MAX_FLOAT;
    float mid = 1.0;

    int total_nonzero = 0;
    int max_nonzero   = -1;

    int start_nonzero = -1;
    float sum         = 0.0;

    for (int idx = 0; idx < n_neighbors; idx++) {
      float cur_dist = knn_dists[i + idx];
      sum += cur_dist;

      if (cur_dist > 0.0) {
        if (start_nonzero == -1) start_nonzero = idx;
        total_nonzero++;
      }

      if (cur_dist > max_nonzero) max_nonzero = cur_dist;
    }

    if (start_nonzero == -1) {
      // All distances are zero: identical neighbors
      rhos[row]   = 0.0;
      sigmas[row] = MIN_K_DIST_SCALE * mean_dist;  // e.g. 1e-5 * mean_dist
      return;
    }

    float ith_distances_mean = sum / float(n_neighbors);
    if (total_nonzero >= local_connectivity) {
      int index           = int(floor(local_connectivity));
      float interpolation = local_connectivity - index;

      if (index > 0) {
        rhos[row] = knn_dists[i + start_nonzero + (index - 1)];

        if (interpolation > SMOOTH_K_TOLERANCE) {
          rhos[row] += interpolation * (knn_dists[i + start_nonzero + index] -
                                        knn_dists[i + start_nonzero + (index - 1)]);
        }
      } else
        rhos[row] = interpolation * knn_dists[i + start_nonzero];
    } else if (total_nonzero > 0)
      rhos[row] = max_nonzero;

    for (int iter = 0; iter < n_iter; iter++) {
      float psum = 0.0;

      for (int j = 1; j < n_neighbors; j++) {
        float d = knn_dists[i + j] - rhos[row];
        if (d > 0)
          psum += exp(-(d / mid));
        else
          psum += 1.0;
      }

      if (fabsf(psum - target) < SMOOTH_K_TOLERANCE) { break; }

      if (psum > target) {
        hi  = mid;
        mid = (lo + hi) / 2.0;
      } else {
        lo = mid;
        if (hi == MAX_FLOAT)
          mid *= 2;
        else
          mid = (lo + hi) / 2.0;
      }
    }

    sigmas[row] = mid;

    if (rhos[row] > 0.0) {
      if (sigmas[row] < MIN_K_DIST_SCALE * ith_distances_mean)
        sigmas[row] = MIN_K_DIST_SCALE * ith_distances_mean;
    } else {
      if (sigmas[row] < MIN_K_DIST_SCALE * mean_dist) sigmas[row] = MIN_K_DIST_SCALE * mean_dist;
    }
  }
}

/**
 * Construct the membership strength data for the 1-skeleton of each local
 * fuzzy simplicial set -- this is formed as a sparse matrix (COO) where each
 * row is a local fuzzy simplicial set, with a membership strength for the
 * 1-simplex to each other data point.
 *
 * @param knn_indices: the knn index matrix of size (n, k)
 * @param knn_dists: the knn distance matrix of size (n, k)
 * @param sigmas: array of size n representing distance to kth nearest neighbor
 * @param rhos: array of size n representing distance to the first nearest neighbor
 * @param vals: value_t array of size n*k
 * @param rows: value_idx array of size n
 * @param cols: value_idx array of size k
 * @param n Number of samples (rows in knn indices/distances)
 * @param n_neighbors number of columns in knn indices/distances
 *
 * Descriptions adapted from: https://github.com/lmcinnes/umap/blob/master/umap/umap_.py
 */
template <typename value_t, typename value_idx, typename nnz_t, int TPB_X>
CUML_KERNEL void compute_membership_strength_kernel(
  const value_idx* knn_indices,
  const float* knn_dists,  // nn outputs
  const value_t* sigmas,
  const value_t* rhos,  // continuous dists to nearest neighbors
  value_t* vals,
  int* rows,
  int* cols,  // result coo
  int n_neighbors,
  nnz_t to_process)
{  // model params

  // row-based matrix is best
  nnz_t idx = (blockIdx.x * static_cast<nnz_t>(TPB_X)) + threadIdx.x;

  if (idx < to_process) {
    int row = idx / n_neighbors;  // one neighbor per thread

    value_t cur_rho   = rhos[row];
    value_t cur_sigma = sigmas[row];

    value_idx cur_knn_ind = knn_indices[idx];
    value_t cur_knn_dist  = knn_dists[idx];

    if (cur_knn_ind != -1) {
      value_t val = value_t(0.0);
      if (cur_knn_ind == row)
        val = value_t(0.0);
      else if (cur_knn_dist - cur_rho <= value_t(0.0) || cur_sigma == value_t(0.0))
        val = value_t(1.0);
      else {
        val = exp(-((cur_knn_dist - cur_rho) / (cur_sigma)));

        if (val < MIN_FLOAT) val = MIN_FLOAT;
      }

      rows[idx] = row;
      cols[idx] = cur_knn_ind;
      vals[idx] = val;
    }
  }
}

/*
 * Sets up and runs the knn dist smoothing
 */
template <typename value_t, typename value_idx, typename nnz_t, int TPB_X>
void smooth_knn_dist(nnz_t n,
                     const value_idx* knn_indices,
                     const float* knn_dists,
                     value_t* rhos,
                     value_t* sigmas,
                     UMAPParams* params,
                     int n_neighbors,
                     float local_connectivity,
                     cudaStream_t stream)
{
  dim3 grid(raft::ceildiv(n, static_cast<nnz_t>(TPB_X)), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  rmm::device_uvector<value_t> dist_means_dev(n_neighbors, stream);

  raft::stats::mean<false>(
    dist_means_dev.data(), knn_dists, nnz_t{1}, n * n_neighbors, false, stream);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  value_t mean_dist = 0.0;
  raft::update_host(&mean_dist, dist_means_dev.data(), 1, stream);
  raft::interruptible::synchronize(stream);

  /**
   * Smooth kNN distances to be continuous
   */

  smooth_knn_dist_kernel<value_t, nnz_t, TPB_X><<<grid, blk, 0, stream>>>(
    knn_dists, n, mean_dist, sigmas, rhos, n_neighbors, local_connectivity);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
}

template <typename value_t, typename value_idx, typename nnz_t, int TPB_X>
void compute_membership_strength(nnz_t n,
                                 const value_idx* knn_indices,
                                 const value_t* knn_dists,
                                 int n_neighbors,
                                 raft::sparse::COO<value_t>& out,
                                 UMAPParams* params,
                                 cudaStream_t stream,
                                 value_t* out_sigmas = nullptr,
                                 value_t* out_rhos   = nullptr)
{
  /**
   * Calculate mean distance through a parallel reduction
   */

  // Use caller-provided buffers if available, otherwise allocate internal ones
  std::optional<rmm::device_uvector<value_t>> sigmas_storage;
  std::optional<rmm::device_uvector<value_t>> rhos_storage;

  value_t* sigmas;
  value_t* rhos;

  if (out_sigmas != nullptr) {
    sigmas = out_sigmas;
  } else {
    sigmas_storage.emplace(n, stream);
    sigmas = sigmas_storage->data();
  }

  if (out_rhos != nullptr) {
    rhos = out_rhos;
  } else {
    rhos_storage.emplace(n, stream);
    rhos = rhos_storage->data();
  }

  RAFT_CUDA_TRY(cudaMemsetAsync(sigmas, 0, n * sizeof(value_t), stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(rhos, 0, n * sizeof(value_t), stream));

  smooth_knn_dist<value_t, value_idx, nnz_t, TPB_X>(n,
                                                    knn_indices,
                                                    knn_dists,
                                                    rhos,
                                                    sigmas,
                                                    params,
                                                    n_neighbors,
                                                    params->local_connectivity,
                                                    stream);

  RAFT_CUDA_TRY(cudaPeekAtLastError());

  /**
   * Compute graph of membership strengths
   */
  nnz_t to_process = static_cast<nnz_t>(out.n_rows) * n_neighbors;
  dim3 grid_elm(raft::ceildiv(to_process, static_cast<nnz_t>(TPB_X)), 1, 1);
  dim3 blk_elm(TPB_X, 1, 1);

  compute_membership_strength_kernel<value_t, value_idx, nnz_t, TPB_X>
    <<<grid_elm, blk_elm, 0, stream>>>(knn_indices,
                                       knn_dists,
                                       sigmas,
                                       rhos,
                                       out.vals(),
                                       out.rows(),
                                       out.cols(),
                                       n_neighbors,
                                       to_process);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename value_t>
void symmetrize(raft::sparse::COO<value_t>& in,
                raft::sparse::COO<value_t>& out,
                float set_op_mix_ratio,
                cudaStream_t stream)
{
  /**
   * Combines all the fuzzy simplicial sets into a global
   * one via a fuzzy union. (Symmetrize knn graph).
   */
  raft::sparse::linalg::coo_symmetrize<value_t>(
    &in,
    &out,
    [set_op_mix_ratio] __device__(int row, int col, value_t result, value_t transpose) {
      value_t prod_matrix = result * transpose;
      value_t res         = set_op_mix_ratio * (result + transpose - prod_matrix) +
                    (value_t(1.0) - set_op_mix_ratio) * prod_matrix;
      return res;
    },
    stream);
}

/**
 * Given a set of X, a neighborhood size, and a measure of distance, compute
 * the fuzzy simplicial set (here represented as a fuzzy graph in the form of
 * a sparse coo matrix) associated to the data. This is done by locally
 * approximating geodesic (manifold surface) distance at each point, creating
 * a fuzzy simplicial set for each such point, and then combining all the local
 * fuzzy simplicial sets into a global one via a fuzzy union.
 *
 * @param n the number of rows/elements in X
 * @param knn_indices indexes of knn search
 * @param knn_dists distances of knn search
 * @param n_neighbors number of neighbors in knn search arrays
 * @param out The output COO sparse matrix
 * @param params UMAPParams config object
 * @param stream cuda stream to use for device operations
 */
template <typename value_t, typename value_idx, typename nnz_t, int TPB_X>
void launcher(nnz_t n,
              const value_idx* knn_indices,
              const value_t* knn_dists,
              int n_neighbors,
              raft::sparse::COO<value_t>* out,
              UMAPParams* params,
              cudaStream_t stream)
{
  /* Nested scope so `strengths` is dropped before the `coo_sort` call to
   * reduce device memory usage. */
  {
    raft::sparse::COO<value_t> strengths(stream, n * n_neighbors, n, n);

    compute_membership_strength<value_t, value_idx, nnz_t, TPB_X>(
      n, knn_indices, knn_dists, n_neighbors, strengths, params, stream);

    symmetrize<value_t>(strengths, *out, params->set_op_mix_ratio, stream);
  }  // end strengths scope

  raft::sparse::op::coo_sort<value_t>(out, stream);
}
}  // namespace Naive
}  // namespace FuzzySimplSet
};  // namespace UMAPAlgo
