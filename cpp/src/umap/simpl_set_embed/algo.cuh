/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "optimize_batch_kernel.cuh"
#include "optimize_inverse_kernel.cuh"

#include <cuml/common/checked_arithmetic.hpp>
#include <cuml/common/logger.hpp>
#include <cuml/manifold/umapparams.h>

#include <raft/linalg/init.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/op/filter.cuh>
#include <raft/sparse/op/sort.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/tuple>
#include <thrust/device_ptr.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

#include <curand.h>
#include <math.h>
#include <stdint.h>

#include <cstdlib>
#include <string>

namespace UMAPAlgo {
namespace SimplSetEmbed {
namespace Algo {

using namespace ML;

/**
 * Given a set of weights and number of epochs, generate
 * the number of epochs per sample for each weight.
 *
 * @param weights: The weights of how much we wish to sample each 1-simplex
 * @param weights_n: the size of the weights array
 * @param n_epochs: the total number of epochs we want to train for
 * @param result: an array of number of epochs per sample, one for each 1-simplex
 * @param stream cuda stream
 */
template <typename T, typename idx_t, typename nnz_t>
void make_epochs_per_sample(
  T* weights, idx_t weights_n, nnz_t n_epochs, T* result, cudaStream_t stream)
{
  thrust::device_ptr<T> d_weights = thrust::device_pointer_cast(weights);
  T weights_max =
    *(thrust::max_element(thrust::cuda::par.on(stream), d_weights, d_weights + weights_n));

  //  result = -1.0 * np.ones(
  //      weights.shape[0], dtype=np.float64
  //  )
  //  n_samples = n_epochs * (weights / weights.max())
  //  result[n_samples > 0] = (
  //      float(n_epochs) / n_samples[n_samples > 0]
  //  )

  raft::linalg::unaryOp<T>(
    result,
    weights,
    weights_n,
    [=] __device__(T input) {
      T v = n_epochs * (input / weights_max);
      if (v > 0)
        return T(n_epochs) / v;
      else
        return T(-1.0);
    },
    stream);
}

template <typename T>
void optimization_iteration_finalization(
  UMAPParams* params, T* head_embedding, T& alpha, int n, int n_epochs, uint64_t& seed)
{
  if (params->callback) params->callback->on_epoch_end(head_embedding);
  alpha = params->initial_alpha * (1.0 - (T(n) / T(n_epochs)));
  seed += 1;
}

/**
 * \brief Constructs a rounding factor used to truncate elements in a sum such that the
 * sum of the truncated elements is the same no matter what the order of the sum is.
 *
 * Algorithm 5: Reproducible Sequential Sum in 'Fast Reproducible Floating-Point
 * Summation' by Demmel and Nguyen
 *
 * In algorithm 5 the bound is calculated as $max(|v_i|) * n$.  We use maximum number of
 * edges connected to each vertex as n.
 *
 * The calculation trick is borrowed from fbcuda, which is BSD-licensed.
 */
template <typename T>
T create_rounding_factor(T max_abs, int n)
{
  T delta =
    max_abs / (static_cast<T>(1.0) - static_cast<T>(2.0) * n * std::numeric_limits<T>::epsilon());

  // Calculate ceil(log_2(delta)).
  // frexpf() calculates exp and returns `x` such that
  // delta = x * 2^exp, where `x` in (-1.0, -0.5] U [0.5, 1).
  // Because |x| < 1, exp is exactly ceil(log_2(delta)).
  int exp;
  std::frexp(delta, &exp);

  // return M = 2 ^ ceil(log_2(delta))
  return std::ldexp(static_cast<T>(1.0), exp);
}

template <typename T, typename nnz_t>
T create_gradient_rounding_factor(
  const int* head, nnz_t nnz, int n_samples, T alpha, cuda::stream_ref stream)
{
  rmm::device_uvector<T> buffer(n_samples, stream);
  // calculate the maximum number of edges connected to 1 vertex.
  thrust::reduce_by_key(rmm::exec_policy(stream),
                        head,
                        head + nnz,
                        thrust::make_constant_iterator(1u),
                        thrust::make_discard_iterator(),
                        buffer.data());
  auto ptr         = thrust::device_pointer_cast(buffer.data());
  uint32_t n_edges = *(thrust::max_element(rmm::exec_policy(stream), ptr, ptr + buffer.size()));
  T max_abs        = T(n_edges) * T(4.0) * std::abs(alpha);
  return create_rounding_factor(max_abs, n_edges);
}

template <typename nnz_t>
CUML_KERNEL void compute_degrees_kernel(const int* rows, nnz_t nnz, int* degrees)
{
  nnz_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nnz) {
    int row = rows[i];
    atomicAdd(&degrees[row], 1);
  }
}

CUML_KERNEL void check_threshold_kernel(const int* degrees,
                                        int n_vertices,
                                        int threshold,
                                        bool* flag)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_vertices) {
    if (degrees[i] > threshold) { *flag = true; }
  }
}

template <typename nnz_t, int TPB_X>
bool check_outliers(const int* rows, int m, nnz_t nnz, int threshold, cudaStream_t stream)
{
  rmm::device_uvector<int> graph_degree_head(m, stream);
  raft::linalg::zero(graph_degree_head.data(), m, stream);

  dim3 grid_nnz(raft::ceildiv(nnz, static_cast<nnz_t>(TPB_X)), 1, 1);
  dim3 blk(TPB_X, 1, 1);
  compute_degrees_kernel<<<grid_nnz, blk, 0, stream>>>(rows, nnz, graph_degree_head.data());

  rmm::device_scalar<bool> has_outlier_d(stream);
  raft::linalg::zero(has_outlier_d.data(), 1, stream);

  dim3 grid_head_n(raft::ceildiv(static_cast<nnz_t>(m), static_cast<nnz_t>(TPB_X)), 1, 1);
  check_threshold_kernel<<<grid_head_n, blk, 0, stream>>>(
    graph_degree_head.data(), m, threshold, has_outlier_d.data());
  cudaStreamSynchronize(stream);
  bool has_outlier_h = has_outlier_d.value(stream);
  return has_outlier_h;
}

/**
 * Runs gradient descent using sampling weights defined on
 * both the attraction and repulsion vectors.
 *
 * In this GD implementation, the weights being tuned are the
 * embeddings themselves, as the objective function is attracting
 * positive weights (neighbors in the 1-skeleton) and repelling
 * negative weights (non-neighbors in the 1-skeleton).
 */
template <typename T, typename nnz_t, int TPB_X>
void optimize_layout(T* head_embedding,
                     int head_n,
                     T* tail_embedding,
                     int tail_n,
                     int* head,
                     int* tail,
                     nnz_t nnz,
                     T* epochs_per_sample,
                     float gamma,
                     UMAPParams* params,
                     int n_epochs,
                     cudaStream_t stream)
{
  // Are we doing a fit or a transform?
  bool move_other = head_embedding == tail_embedding;
  T alpha         = params->initial_alpha;

  auto stream_view = cuda::stream_ref(stream);

  T rounding = create_gradient_rounding_factor<T, nnz_t>(head, nnz, head_n, alpha, stream_view);

  auto min_n                = min(head_n, tail_n);
  int threshold_for_outlier = 512;  // this is a heuristic value.
  // for smaller datasets, could be a dense point even with a smaller graph degree
  if (min_n <= 100000) {
    threshold_for_outlier = 128;
  } else if (min_n <= 1000000) {
    threshold_for_outlier = 256;
  }

  int num_chunks = 1;

  // serial kernel does not need to check for outliers
  bool has_outlier = false;
  if (!params->force_serial_epochs) {
    has_outlier = check_outliers<nnz_t, TPB_X>(head, head_n, nnz, threshold_for_outlier, stream);
    if (move_other && !has_outlier) {
      has_outlier = check_outliers<nnz_t, TPB_X>(tail, tail_n, nnz, threshold_for_outlier, stream);
    }
  }

  rmm::device_uvector<nnz_t> row_ptr(0, stream_view);
  if (params->force_serial_epochs) {
    // sorted_coo_to_csr below requires `head` to be sorted. Probe with a cheap O(nnz) scan
    // and only sort when needed.
    bool rows_sorted = thrust::is_sorted(rmm::exec_policy(stream_view), head, head + nnz);

    if (!rows_sorted) {
      CUML_LOG_INFO(
        "optimize_layout: input COO rows are not sorted; sorting "
        "(required by the force_serial_epochs path for CSR conversion).");
      raft::sparse::op::coo_sort<T, int, nnz_t>(
        head_n, tail_n, nnz, head, tail, epochs_per_sample, stream);
    }
    row_ptr.resize(static_cast<nnz_t>(head_n) + 1, stream_view);
    raft::sparse::convert::sorted_coo_to_csr(head, nnz, row_ptr.data(), head_n, stream);
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      row_ptr.data() + head_n, &nnz, sizeof(nnz_t), cudaMemcpyHostToDevice, stream));
  }

  if (!params->force_serial_epochs && (has_outlier || params->deterministic)) {
    // Shuffling is necessary when outliers may be present (i.e., dense points that undergo many
    // updates). It is critical to avoid having too many threads update the same embedding vector
    // simultaneously, as this can affect correctness. By shuffling, potential outlier points are
    // distributed across threads, rather than being processed by consecutive threads that are
    // scheduled together. This approach relies on the GPU's inability to physically schedule all
    // nnz edges at once.
    // also shuffle when want deterministic behavior to ensure that updates for the same vertex are
    // processed in different kernel launches.
    auto first = thrust::make_zip_iterator(
      cuda::std::make_tuple(thrust::device_pointer_cast(head),
                            thrust::device_pointer_cast(tail),
                            thrust::device_pointer_cast(epochs_per_sample)));

    thrust::default_random_engine rng(params->random_state);
    thrust::shuffle(first, first + nnz, rng);
    if (min_n <= 100000) {
      // related issue: https://github.com/NVIDIA/cuml/issues/7431
      // Heuristic value for forcing serial behavior to handle outliers in smaller datasets.
      // Ideally this value should depend on nnz, maximum graph degree, number of SMs and maximum
      // number of threads per SM, but it is difficult to generalize due to limited cases where this
      // happens. Increase this value if we observe outliers on smaller datasets.
      num_chunks = 4;
    }
  }

  if (!params->force_serial_epochs && has_outlier && params->deterministic) {
    // num_chunks only feeds the batch-kernel grid setup below, so this branch
    // is irrelevant when force_serial_epochs is true (which dispatches to the
    // sequential kernel that computes its own launch config).
    // For deterministic mode on datasets likely to contain outliers, the
    // heuristic below picks num_chunks. Empirically, 100000 edges per chunk
    // balances determinism and gradient-accumulation behavior on large
    // datasets. See benchmarks in https://github.com/NVIDIA/cuml/pull/7597
    if (nnz > 100000) {
      num_chunks =
        std::max(num_chunks, ML::narrow_cast<int>(raft::ceildiv(nnz, static_cast<nnz_t>(100000))));
    } else if (nnz > 10000) {
      // more fine-grained chunks for smaller number of edges
      num_chunks =
        std::max(num_chunks, ML::narrow_cast<int>(raft::ceildiv(nnz, static_cast<nnz_t>(10000))));
    }
  }

  rmm::device_uvector<T> epoch_of_next_negative_sample(nnz, stream);
  T nsr_inv = T(1.0) / params->negative_sample_rate;
  raft::linalg::unaryOp<T>(
    epoch_of_next_negative_sample.data(),
    epochs_per_sample,
    nnz,
    [=] __device__(T input) { return input * nsr_inv; },
    stream);

  rmm::device_uvector<T> epoch_of_next_sample(nnz, stream);
  raft::copy(epoch_of_next_sample.data(), epochs_per_sample, nnz, stream);

  // Buffers used to store the gradient updates to avoid conflicts
  rmm::device_uvector<T> head_buffer(0, stream_view);
  rmm::device_uvector<T> tail_buffer(0, stream_view);

  // Flags to track which vertices were modified per chunk (for sparse apply)
  rmm::device_uvector<uint32_t> head_flags(0, stream_view);
  rmm::device_uvector<uint32_t> tail_flags(0, stream_view);
  // Write to embedding directly if deterministic is not needed.
  T* d_head_buffer       = head_embedding;
  T* d_tail_buffer       = tail_embedding;
  uint32_t* d_head_flags = nullptr;
  uint32_t* d_tail_flags = nullptr;
  if (params->deterministic) {
    nnz_t n_components = params->n_components;
    head_buffer.resize(head_n * n_components, stream_view);
    RAFT_CUDA_TRY(
      cudaMemsetAsync(head_buffer.data(), '\0', sizeof(T) * head_buffer.size(), stream));

    int head_flag_words = raft::ceildiv(head_n, 32);
    head_flags.resize(head_flag_words, stream_view);
    RAFT_CUDA_TRY(
      cudaMemsetAsync(head_flags.data(), '\0', sizeof(uint32_t) * head_flag_words, stream));
    d_head_buffer = head_buffer.data();
    d_head_flags  = head_flags.data();
    // No need for tail if it's not being written.
    if (move_other) {
      tail_buffer.resize(tail_n * n_components, stream_view);
      RAFT_CUDA_TRY(
        cudaMemsetAsync(tail_buffer.data(), '\0', sizeof(T) * tail_buffer.size(), stream));
      int tail_flag_words = raft::ceildiv(tail_n, 32);
      tail_flags.resize(tail_flag_words, stream_view);
      RAFT_CUDA_TRY(
        cudaMemsetAsync(tail_flags.data(), '\0', sizeof(uint32_t) * tail_flag_words, stream));
      d_tail_flags  = tail_flags.data();
      d_tail_buffer = tail_buffer.data();
    }
  }

  // we keep the tpb and change the number of blocks we launch to handle the total number of threads
  // launched
  auto nnz_per_chunk = raft::ceildiv(nnz, static_cast<nnz_t>(num_chunks));
  dim3 grid(raft::ceildiv(nnz_per_chunk, static_cast<nnz_t>(TPB_X)), 1, 1);
  dim3 blk(TPB_X, 1, 1);
  uint64_t seed = params->random_state;

  for (int n = 0; n < n_epochs; n++) {
    if (params->force_serial_epochs) {
      call_optimize_sequential_kernel<T, nnz_t, TPB_X>(head_embedding,
                                                       d_head_buffer,
                                                       d_head_flags,
                                                       head_n,
                                                       tail_embedding,
                                                       d_tail_buffer,
                                                       d_tail_flags,
                                                       tail_n,
                                                       row_ptr.data(),
                                                       tail,
                                                       epochs_per_sample,
                                                       epoch_of_next_negative_sample.data(),
                                                       epoch_of_next_sample.data(),
                                                       alpha,
                                                       gamma,
                                                       seed,
                                                       move_other,
                                                       params,
                                                       n,
                                                       stream,
                                                       rounding);
    } else {
      call_optimize_batch_kernel<T, nnz_t, TPB_X>(head_embedding,
                                                  d_head_buffer,
                                                  d_head_flags,
                                                  head_n,
                                                  tail_embedding,
                                                  d_tail_buffer,
                                                  d_tail_flags,
                                                  tail_n,
                                                  head,
                                                  tail,
                                                  nnz,
                                                  epochs_per_sample,
                                                  epoch_of_next_negative_sample.data(),
                                                  epoch_of_next_sample.data(),
                                                  alpha,
                                                  gamma,
                                                  seed,
                                                  move_other,
                                                  params,
                                                  n,
                                                  grid,
                                                  blk,
                                                  stream,
                                                  rounding);
    }
    RAFT_CUDA_TRY(cudaGetLastError());
    optimization_iteration_finalization(params, head_embedding, alpha, n, n_epochs, seed);
  }
}

/**
 * Perform a fuzzy simplicial set embedding by minimizing
 * the fuzzy set cross entropy between the embeddings
 * and their 1-skeletons.
 */
template <typename T, typename nnz_t, int TPB_X>
void launcher(int m,
              int n,
              raft::sparse::COO<T>* in,
              UMAPParams* params,
              T* embedding,
              int n_epochs,
              cudaStream_t stream)
{
  nnz_t nnz = in->nnz;

  rmm::device_uvector<T> epochs_per_sample(nnz, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(epochs_per_sample.data(), 0, nnz * sizeof(T), stream));

  make_epochs_per_sample(in->vals(), nnz, n_epochs, epochs_per_sample.data(), stream);

  /*
  if (ML::default_logger().should_log(rapids_logger::level_enum::debug)) {
    std::stringstream ss;
    ss << raft::arr2Str(epochs_per_sample.data(), nnz, "epochs_per_sample", stream);
    CUML_LOG_TRACE(ss.str().c_str());
  }
  */

  optimize_layout<T, nnz_t, TPB_X>(embedding,
                                   m,
                                   embedding,
                                   m,
                                   in->rows(),
                                   in->cols(),
                                   in->nnz,
                                   epochs_per_sample.data(),
                                   params->repulsion_strength,
                                   params,
                                   n_epochs,
                                   stream);

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace Algo
}  // namespace SimplSetEmbed
}  // namespace UMAPAlgo
