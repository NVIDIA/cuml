/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <common/fast_int_div.cuh>

#include <cuml/common/checked_arithmetic.hpp>
#include <cuml/manifold/umapparams.h>

#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <stdint.h>

#include <cstddef>

namespace UMAPAlgo {
namespace SimplSetEmbed {
namespace Algo {

using namespace ML;

/**
 * Set a bit in the flags to mark a vertex as modified.
 */
__device__ __forceinline__ void set_vertex_modified(uint32_t* flags, int vertex_id)
{
  int word_idx = vertex_id >> 5;  // vertex_id / 32
  int bit_idx  = vertex_id & 31;  // vertex_id % 32
  atomicOr(&flags[word_idx], 1u << bit_idx);
}

/**
 * Sparse apply kernel: only updates vertices that have their bit set in the flags.
 */
template <typename T, int TPB_X = 256>
CUML_KERNEL void sparse_apply_kernel(
  T* embedding, T* buffer, uint32_t* flags, int n_vertices, int n_components)
{
  int tid           = blockIdx.x * TPB_X + threadIdx.x;
  int total_threads = gridDim.x * TPB_X;

  int n_words = (n_vertices + 31) >> 5;

  // each thread strides through the flags in word units
  for (int word_idx = tid; word_idx < n_words; word_idx += total_threads) {
    uint32_t word = flags[word_idx];
    if (word == 0) continue;  // skip if no bits set

    while (word != 0) {
      int bit_pos   = __ffs(word) - 1;            // find first set bit (1-indexed, so -1)
      int vertex_id = (word_idx << 5) + bit_pos;  // word_idx * 32 (word size) + bit_pos

      if (vertex_id < n_vertices) {
        // Apply update for this vertex
        int base = vertex_id * n_components;
        for (int d = 0; d < n_components; d++) {
          embedding[base + d] += buffer[base + d];
          buffer[base + d] = T(0.0);
        }
      }

      // Clear the first set bit just processed
      word &= ~(1u << bit_pos);
    }

    flags[word_idx] = 0;  // clear flag for next chunk
  }
}

/**
 * Update the embeddings using sparse apply using bit flags to track which vertices received
 * gradient updates.
 */
template <typename T, typename nnz_t, int TPB_X = 256>
void sparse_apply_embedding_updates(T* head_embedding,
                                    T* head_buffer,
                                    uint32_t* head_flags,
                                    int head_n,
                                    T* tail_embedding,
                                    T* tail_buffer,
                                    uint32_t* tail_flags,
                                    int tail_n,
                                    UMAPParams const* params,
                                    bool move_other,
                                    cudaStream_t stream)
{
  // flags: one thread per 32-vertex word
  int head_words = raft::ceildiv(head_n, 32);
  dim3 grid_head(raft::ceildiv(head_words, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  sparse_apply_kernel<T, TPB_X><<<grid_head, blk, 0, stream>>>(
    head_embedding, head_buffer, head_flags, head_n, params->n_components);

  if (move_other) {
    int tail_words = raft::ceildiv(tail_n, 32);
    dim3 grid_tail(raft::ceildiv(tail_words, TPB_X), 1, 1);
    sparse_apply_kernel<T, TPB_X><<<grid_tail, blk, 0, stream>>>(
      tail_embedding, tail_buffer, tail_flags, tail_n, params->n_components);
  }
}

/**
 * Calculate the squared distance between two vectors of size n
 * @{
 */
template <typename T>
DI T rdist(const T* X, const T* Y, int n)
{
  auto result = T(0.0);
  for (int i = 0; i < n; i++) {
    auto diff = T(X[i] - Y[i]);
    result += diff * diff;
  }
  return result;
}
template <typename T, int LEN>
DI T rdist(const T (&X)[LEN], const T (&Y)[LEN])
{
  auto result = T(0.0);
#pragma unroll
  for (int i = 0; i < LEN; ++i) {
    auto diff = T(X[i] - Y[i]);
    result += diff * diff;
  }
  return result;
}
/** @} */

/**
 * Clip a value to within a lower and upper bound
 */
template <typename T>
DI T clip(T val, T lb, T ub)
{
  return min(max(val, lb), ub);
}

template <typename T>
DI T repulsive_coeff(T dist_squared, T gamma, UMAPParams params)
{
  if (dist_squared <= T(0.0)) { return T(0.0); }
  auto grad_coeff = T(2.0) * gamma * params.b;
  grad_coeff /= (T(0.001) + dist_squared) * (params.a * pow(dist_squared, params.b) + T(1.0));
  return grad_coeff;
}

template <typename T>
DI T attractive_coeff(T dist_squared, UMAPParams params)
{
  if (dist_squared <= T(0.0)) { return T(0.0); }
  auto grad_coeff = T(-2.0) * params.a * params.b * pow(dist_squared, params.b - T(1.0));
  grad_coeff /= params.a * pow(dist_squared, params.b) + T(1.0);
  return grad_coeff;
}

template <typename T>
DI T truncate_gradient(T const rounding_factor, T const x)
{
  return (rounding_factor + x) - rounding_factor;
}

template <typename T>
DI T attractive_component_grad(T coeff, T diff)
{
  return clip<T>(coeff * diff, T(-4.0), T(4.0));
}

template <typename T>
DI T repulsive_component_grad(T coeff, T diff)
{
  return coeff > T(0.0) ? clip<T>(coeff * diff, T(-4.0), T(4.0)) : T(4.0);
}

/**
 * Warp-level all-reduce sum across all 32 lanes
 */
template <typename T>
DI T warp_all_reduce_sum(T val)
{
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_xor_sync(0xffffffff, val, offset);
  return val;
}

template <typename T, typename nnz_t, typename DistFn, typename ApplyFn>
DI void run_negative_sampling(int n_neg_samples,
                              uint64_t seed,
                              nnz_t edge_id,
                              MLCommon::FastIntDiv tail_n,
                              int self_loop_vertex,
                              T gamma,
                              UMAPParams const& params,
                              DistFn compute_dist,
                              ApplyFn apply_grad)
{
  raft::random::detail::PhiloxGenerator gen(seed, edge_id, 0);
  for (int p = 0; p < n_neg_samples; p++) {
    int r;
    gen.next(r);
    nnz_t t        = static_cast<nnz_t>(r % tail_n);
    T dist_squared = compute_dist(t);
    if (dist_squared <= T(0.0) && self_loop_vertex == static_cast<int>(t)) { continue; }
    apply_grad(repulsive_coeff<T>(dist_squared, gamma, params));
  }
}

template <typename T, typename nnz_t, int TPB_X, nnz_t n_components>
CUML_KERNEL void optimize_batch_kernel_reg(T const* head_embedding,
                                           T* head_buffer,
                                           uint32_t* head_flags,
                                           T const* tail_embedding,
                                           T* tail_buffer,
                                           uint32_t* tail_flags,
                                           MLCommon::FastIntDiv tail_n,
                                           const int* head,
                                           const int* tail,
                                           nnz_t nnz,
                                           T const* epochs_per_sample,
                                           T* epoch_of_next_negative_sample,
                                           T* epoch_of_next_sample,
                                           T alpha,
                                           int epoch,
                                           T gamma,
                                           uint64_t seed,
                                           bool move_other,
                                           UMAPParams params,
                                           T nsr_inv,
                                           T rounding,
                                           size_t offset = 0)
{
  size_t row =
    (static_cast<size_t>(blockIdx.x) * static_cast<size_t>(TPB_X)) + threadIdx.x + offset;
  size_t skip_size = static_cast<size_t>(blockDim.x) * gridDim.x;

  T current_reg[n_components], other_reg[n_components], grads[n_components];

  // Deterministic mode: process exactly one row per thread (no grid-stride loop)
  // Non-deterministic mode: use grid-stride loop to process multiple rows
  while (row < nnz) {
    auto _epoch_of_next_sample = epoch_of_next_sample[row];
    if (_epoch_of_next_sample > epoch) {
      if (params.deterministic) {
        // we return immediately in deterministic mode instead of continuing the grid-stride loop
        // because we launch a new kernel for the next chunk
        return;
      } else {
        row += skip_size;
        continue;
      }
    }
    auto _epochs_per_sample         = epochs_per_sample[row];
    auto epochs_per_negative_sample = _epochs_per_sample * nsr_inv;
    /**
     * Positive sample stage (attractive forces)
     */
    nnz_t j          = head[row];
    nnz_t k          = tail[row];
    T const* current = head_embedding + (j * n_components);
    T const* other   = tail_embedding + (k * n_components);

    T* cur_write = head_buffer + (j * n_components);
    T* oth_write = tail_buffer + (k * n_components);

    for (int i = 0; i < n_components; ++i) {
      current_reg[i] = current[i];
      other_reg[i]   = other[i];
    }
    auto dist_squared = rdist<T, n_components>(current_reg, other_reg);
    // Attractive force between the two vertices, since they
    // are connected by an edge in the 1-skeleton.
    auto attractive_grad_coeff = attractive_coeff<T>(dist_squared, params);
    /**
     * Apply attractive force between `current` and `other`
     * by updating their 'weights' to place them relative
     * to their weight in the 1-skeleton.
     * (update `other` embedding only if we are
     * performing unsupervised training).
     */
    for (int d = 0; d < n_components; d++) {
      auto grad_d =
        attractive_component_grad<T>(attractive_grad_coeff, current_reg[d] - other_reg[d]);
      current_reg[d] += grad_d * alpha;
      grads[d] = grad_d * alpha;
    }
    // storing gradients for negative samples back to global memory
    if (move_other) {
      for (int d = 0; d < n_components; d++) {
        raft::myAtomicAdd(oth_write + d, truncate_gradient(rounding, -grads[d]));
      }
      if (tail_flags != nullptr) { set_vertex_modified(tail_flags, k); }
    }
    epoch_of_next_sample[row] = _epoch_of_next_sample + _epochs_per_sample;
    // number of negative samples to choose
    auto _epoch_of_next_negative_sample = epoch_of_next_negative_sample[row];
    int n_neg_samples = int(T(epoch - _epoch_of_next_negative_sample) / epochs_per_negative_sample);
    /**
     * Negative sampling stage
     */
    T negative_sample_reg[n_components];
    run_negative_sampling<T, nnz_t>(
      n_neg_samples,
      (uint64_t)seed,
      (nnz_t)row,
      tail_n,
      (int)j,
      gamma,
      params,
      [&](nnz_t t) -> T {
        T const* negative_sample = tail_embedding + (t * n_components);
        for (int i = 0; i < n_components; ++i)
          negative_sample_reg[i] = negative_sample[i];
        return rdist<T, n_components>(current_reg, negative_sample_reg);
      },
      [&](T coeff) {
        for (int d = 0; d < n_components; d++) {
          auto grad_d = repulsive_component_grad<T>(coeff, current_reg[d] - negative_sample_reg[d]);
          current_reg[d] += grad_d * alpha;
          grads[d] += grad_d * alpha;
        }
      });
    // storing gradients for positive samples back to global memory
    for (int d = 0; d < n_components; d++) {
      raft::myAtomicAdd(cur_write + d, truncate_gradient(rounding, grads[d]));
    }
    if (head_flags != nullptr) { set_vertex_modified(head_flags, j); }

    epoch_of_next_negative_sample[row] =
      _epoch_of_next_negative_sample + n_neg_samples * epochs_per_negative_sample;

    if (params.deterministic) {
      return;  // Only process one row in deterministic mode
    } else {
      row += skip_size;
    }
  }
}

template <typename T, typename nnz_t, int TPB_X, bool use_shared_mem>
CUML_KERNEL void optimize_batch_kernel(T const* head_embedding,
                                       T* head_buffer,
                                       uint32_t* head_flags,
                                       T const* tail_embedding,
                                       T* tail_buffer,
                                       uint32_t* tail_flags,
                                       MLCommon::FastIntDiv tail_n,
                                       const int* head,
                                       const int* tail,
                                       nnz_t nnz,
                                       T const* epochs_per_sample,
                                       T* epoch_of_next_negative_sample,
                                       T* epoch_of_next_sample,
                                       T alpha,
                                       int epoch,
                                       T gamma,
                                       uint64_t seed,
                                       bool move_other,
                                       UMAPParams params,
                                       T nsr_inv,
                                       T rounding,
                                       size_t offset = 0)
{
  extern __shared__ T embedding_shared_mem_updates[];
  size_t row =
    (static_cast<size_t>(blockIdx.x) * static_cast<size_t>(TPB_X)) + threadIdx.x + offset;
  size_t skip_size = static_cast<size_t>(blockDim.x) * gridDim.x;

  while (row < nnz) {
    auto _epoch_of_next_sample = epoch_of_next_sample[row];
    if (_epoch_of_next_sample > epoch) {
      if (params.deterministic) {
        // we return immediately in deterministic mode instead of continuing the grid-stride loop
        // because we launch a new kernel for the next chunk
        return;
      } else {
        row += skip_size;
        continue;
      }
    }
    auto _epochs_per_sample         = epochs_per_sample[row];
    auto epochs_per_negative_sample = _epochs_per_sample * nsr_inv;
    /**
     * Positive sample stage (attractive forces)
     */
    nnz_t n_components = params.n_components;

    nnz_t j          = head[row];
    nnz_t k          = tail[row];
    T const* current = head_embedding + (j * n_components);
    T const* other   = tail_embedding + (k * n_components);

    T* cur_write = head_buffer + (j * n_components);
    T* oth_write = tail_buffer + (k * n_components);

    // for reducing access to global memory. load values from global memory, and accumulate grads
    // onto this shared memory position instead of reading from global memory every time.
    T* current_buffer{nullptr};
    // for keeping track of grads, final write to global memory
    T* grads_buffer{nullptr};
    if constexpr (use_shared_mem) {
      // n_components for thread0, then the next n_components for thread1 ...
      current_buffer = (T*)embedding_shared_mem_updates + threadIdx.x * n_components;
      // TPB_X for first component, then another TPB_X for the next component for better
      // coalescing...
      grads_buffer = (T*)embedding_shared_mem_updates + TPB_X * n_components + threadIdx.x;
    }
    auto dist_squared = rdist<T>(current, other, n_components);
    // Attractive force between the two vertices, since they
    // are connected by an edge in the 1-skeleton.
    auto attractive_grad_coeff = attractive_coeff<T>(dist_squared, params);
    /**
     * Apply attractive force between `current` and `other`
     * by updating their 'weights' to place them relative
     * to their weight in the 1-skeleton.
     * (update `other` embedding only if we are
     * performing unsupervised training).
     */
    for (int d = 0; d < n_components; d++) {
      T current_val = current[d];
      if constexpr (use_shared_mem) { current_buffer[d] = current_val; }
      auto grad_d = attractive_component_grad<T>(attractive_grad_coeff, current_val - other[d]);
      grad_d *= alpha;
      if constexpr (use_shared_mem) {
        current_buffer[d] += grad_d;
        grads_buffer[d * TPB_X] = grad_d;
      } else {
        raft::myAtomicAdd<T>((T*)cur_write + d, truncate_gradient(rounding, grad_d));
        if (move_other) {  // happens only during unsupervised training
          raft::myAtomicAdd<T>((T*)oth_write + d, truncate_gradient(rounding, -grad_d));
        }
      }
    }

    if constexpr (!use_shared_mem) {
      if (head_flags != nullptr) { set_vertex_modified(head_flags, j); }
      if (move_other && tail_flags != nullptr) { set_vertex_modified(tail_flags, k); }
    }
    // storing gradients for negative samples back to global memory
    if (use_shared_mem && move_other) {
      for (int d = 0; d < n_components; d++) {
        auto grad = grads_buffer[d * TPB_X];
        raft::myAtomicAdd<T>((T*)oth_write + d, truncate_gradient(rounding, -grad));
      }
      if (tail_flags != nullptr) { set_vertex_modified(tail_flags, k); }
    }
    epoch_of_next_sample[row] = _epoch_of_next_sample + _epochs_per_sample;
    // number of negative samples to choose
    auto _epoch_of_next_negative_sample = epoch_of_next_negative_sample[row];
    int n_neg_samples = int(T(epoch - _epoch_of_next_negative_sample) / epochs_per_negative_sample);
    /**
     * Negative sampling stage
     */
    T const* negative_sample = nullptr;
    run_negative_sampling<T, nnz_t>(
      n_neg_samples,
      (uint64_t)seed,
      (nnz_t)row,
      tail_n,
      (int)j,
      gamma,
      params,
      [&](nnz_t t) -> T {
        negative_sample = tail_embedding + (t * n_components);
        if constexpr (use_shared_mem) {
          return rdist<T>(current_buffer, negative_sample, n_components);
        } else {
          return rdist<T>(current, negative_sample, n_components);
        }
      },
      [&](T coeff) {
        for (int d = 0; d < n_components; d++) {
          T cur_d;
          if constexpr (use_shared_mem) {
            cur_d = current_buffer[d];
          } else {
            cur_d = current[d];
          }
          auto grad_d = repulsive_component_grad<T>(coeff, cur_d - negative_sample[d]);
          grad_d *= alpha;
          if constexpr (use_shared_mem) {
            current_buffer[d] += grad_d;
            grads_buffer[d * TPB_X] += grad_d;
          } else {
            raft::myAtomicAdd<T>((T*)cur_write + d, truncate_gradient(rounding, grad_d));
          }
        }
      });

    // storing gradients for positive samples back to global memory
    if constexpr (use_shared_mem) {
      for (int d = 0; d < n_components; d++) {
        raft::myAtomicAdd<T>((T*)cur_write + d,
                             truncate_gradient(rounding, grads_buffer[d * TPB_X]));
      }
      if (head_flags != nullptr) { set_vertex_modified(head_flags, j); }
    }
    epoch_of_next_negative_sample[row] =
      _epoch_of_next_negative_sample + n_neg_samples * epochs_per_negative_sample;
    if (params.deterministic) {
      return;  // Only process one row in deterministic mode
    } else {
      row += skip_size;
    }
  }
}

/**
 * Each thread processes one vertex and all its edges serially.

 * This is not fully sequential because updates accumulate in registers and the net delta is written
 to head_buffer[vertex]
 * once after the edge loop. So a thread does not see other vertices' in-flight buffer writes within
 the same epoch.

 * Best when n_components is small enough to keep 3 × N_COMPONENTS floats in
 * registers without excessive spilling.
 */
template <typename T, typename nnz_t, int TPB_X, nnz_t N_COMPONENTS>
CUML_KERNEL void optimize_sequential_kernel_vertex_per_thread(T const* head_embedding,
                                                              T* head_buffer,
                                                              uint32_t* head_flags,
                                                              T const* tail_embedding,
                                                              T* tail_buffer,
                                                              uint32_t* tail_flags,
                                                              const nnz_t* row_ptr,
                                                              const int* tail,
                                                              int num_vertices,
                                                              MLCommon::FastIntDiv tail_n,
                                                              T const* epochs_per_sample,
                                                              T* epoch_of_next_negative_sample,
                                                              T* epoch_of_next_sample,
                                                              T alpha,
                                                              int epoch,
                                                              T gamma,
                                                              uint64_t seed,
                                                              bool move_other,
                                                              UMAPParams params,
                                                              T nsr_inv,
                                                              T rounding,
                                                              int vertex_offset = 0)
{
  int tid           = blockIdx.x * blockDim.x + threadIdx.x;
  int total_threads = gridDim.x * blockDim.x;

  // each thread loops over the vertices assigned to it
  for (int vertex = tid + vertex_offset; vertex < num_vertices; vertex += total_threads) {
    // the csr indices are used to determine the edges to process for the current vertex
    const nnz_t edge_start = row_ptr[vertex];
    const nnz_t edge_end   = row_ptr[vertex + 1];

    T current_reg[N_COMPONENTS], other_reg[N_COMPONENTS], original[N_COMPONENTS];

#pragma unroll
    for (int d = 0; d < N_COMPONENTS; d++) {
      current_reg[d] = head_embedding[vertex * N_COMPONENTS + d];
      original[d]    = current_reg[d];
    }

    // the thread sequentially processes the edges for the current vertex
    for (nnz_t e = edge_start; e < edge_end; e++) {
      auto _epoch_of_next_sample = epoch_of_next_sample[e];
      if (_epoch_of_next_sample > static_cast<T>(epoch)) continue;

      auto _epochs_per_sample         = epochs_per_sample[e];
      auto epochs_per_negative_sample = _epochs_per_sample * nsr_inv;

      int k          = tail[e];
      T* other_write = tail_buffer + (k * N_COMPONENTS);

#pragma unroll
      for (int d = 0; d < N_COMPONENTS; d++) {
        other_reg[d] = tail_embedding[k * N_COMPONENTS + d];
      }

      auto dist_squared          = rdist<T, N_COMPONENTS>(current_reg, other_reg);
      auto attractive_grad_coeff = attractive_coeff<T>(dist_squared, params);

#pragma unroll
      for (int d = 0; d < N_COMPONENTS; d++) {
        auto grad_d =
          attractive_component_grad<T>(attractive_grad_coeff, current_reg[d] - other_reg[d]);
        auto step_grad = grad_d * alpha;
        current_reg[d] += step_grad;
        if (move_other) {
          raft::myAtomicAdd(other_write + d, truncate_gradient(rounding, -step_grad));
        }
      }
      if (move_other && tail_flags != nullptr) { set_vertex_modified(tail_flags, k); }

      epoch_of_next_sample[e] = _epoch_of_next_sample + _epochs_per_sample;

      auto _epoch_of_next_negative_sample = epoch_of_next_negative_sample[e];
      int n_neg_samples =
        static_cast<int>(T(epoch - _epoch_of_next_negative_sample) / epochs_per_negative_sample);

      run_negative_sampling<T, nnz_t>(
        n_neg_samples,
        seed,
        static_cast<nnz_t>(e),
        tail_n,
        vertex,
        gamma,
        params,
        [&](nnz_t t) -> T {
          T const* negative_sample = tail_embedding + (t * N_COMPONENTS);
#pragma unroll
          for (int d = 0; d < N_COMPONENTS; d++) {
            // reuse other_reg for the negative sample
            other_reg[d] = __ldg(negative_sample + d);
          }
          return rdist<T, N_COMPONENTS>(current_reg, other_reg);
        },
        [&](T coeff) {
#pragma unroll
          for (int d = 0; d < N_COMPONENTS; d++) {
            auto grad_d = repulsive_component_grad<T>(coeff, current_reg[d] - other_reg[d]);
            current_reg[d] += grad_d * alpha;
          }
        });

      epoch_of_next_negative_sample[e] =
        _epoch_of_next_negative_sample + n_neg_samples * epochs_per_negative_sample;
    }

    // Single writer per head_buffer slot, so no truncate_gradient needed.
#pragma unroll
    for (int d = 0; d < N_COMPONENTS; d++) {
      T delta = current_reg[d] - original[d];
      if (delta != T(0)) { raft::myAtomicAdd(&head_buffer[vertex * N_COMPONENTS + d], delta); }
    }
    if (head_flags != nullptr) { set_vertex_modified(head_flags, vertex); }

    // In deterministic mode, the host drives vertex chunking via vertex_offset
    // and calls sparse_apply between launches to flush buffered gradients in a
    // fixed order, so each thread processes exactly one vertex per launch.
    if (params.deterministic) break;
  }
}

/**
 * Component-parallel vertex-centric kernel for n_components above the
 * register-kernel threshold (currently > 21).
 * Each warp processes one vertex at a time. Edges for each vertex are processed sequentially,
 * but per-component work is split across lanes: lane k handles components
 * k, k+32, k+64, ...
 *
 * This is not fully sequential because updates accumulate in per-lane registers and the net delta
 is written to head_buffer[vertex]
 * once after the edge loop. So a warp does not see other vertices' in-flight buffer writes within
 the same epoch.

 * All embedding data lives in registers. The only cross-lane
 * communication is the scalar distance reduction via warp shuffles.
 * Template parameter CPL = components per lane = ceil(n_components / 32).
 */
template <typename T, typename nnz_t, int TPB_X, int CPL>
CUML_KERNEL void optimize_sequential_kernel_vertex_per_warp(T const* head_embedding,
                                                            T* head_buffer,
                                                            uint32_t* head_flags,
                                                            T const* tail_embedding,
                                                            T* tail_buffer,
                                                            uint32_t* tail_flags,
                                                            const nnz_t* row_ptr,
                                                            const int* tail,
                                                            int num_vertices,
                                                            MLCommon::FastIntDiv tail_n,
                                                            T const* epochs_per_sample,
                                                            T* epoch_of_next_negative_sample,
                                                            T* epoch_of_next_sample,
                                                            T alpha,
                                                            int epoch,
                                                            T gamma,
                                                            uint64_t seed,
                                                            bool move_other,
                                                            UMAPParams params,
                                                            T nsr_inv,
                                                            T rounding,
                                                            int vertex_offset = 0)
{
  const int n_components = params.n_components;

  const int lane_id        = threadIdx.x & 31;
  const int warp_global_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int total_warps    = (gridDim.x * blockDim.x) >> 5;

  for (int vertex = warp_global_id + vertex_offset; vertex < num_vertices; vertex += total_warps) {
    const nnz_t edge_start = row_ptr[vertex];
    const nnz_t edge_end   = row_ptr[vertex + 1];

    T current_reg[CPL], original_reg[CPL];

#pragma unroll
    for (int i = 0; i < CPL; i++) {
      int d           = lane_id + i * 32;
      T val           = (d < n_components) ? head_embedding[vertex * n_components + d] : T(0);
      current_reg[i]  = val;
      original_reg[i] = val;
    }

    for (nnz_t e = edge_start; e < edge_end; e++) {
      auto _epoch_of_next_sample = epoch_of_next_sample[e];
      if (_epoch_of_next_sample > epoch) continue;

      auto _epochs_per_sample         = epochs_per_sample[e];
      auto epochs_per_negative_sample = _epochs_per_sample * nsr_inv;

      int k          = tail[e];
      T* other_write = tail_buffer + (k * n_components);

      T other_reg[CPL];
      T partial_dist = T(0);
#pragma unroll
      for (int i = 0; i < CPL; i++) {
        int d        = lane_id + i * 32;
        other_reg[i] = (d < n_components) ? __ldg(&tail_embedding[k * n_components + d]) : T(0);
        T diff       = current_reg[i] - other_reg[i];
        partial_dist += diff * diff;
      }
      T dist_squared = warp_all_reduce_sum(partial_dist);

      auto attractive_grad_coeff = attractive_coeff<T>(dist_squared, params);

#pragma unroll
      for (int i = 0; i < CPL; i++) {
        int d = lane_id + i * 32;
        if (d < n_components) {
          T grad_d =
            attractive_component_grad<T>(attractive_grad_coeff, current_reg[i] - other_reg[i]);
          current_reg[i] += grad_d * alpha;
          if (move_other) {
            raft::myAtomicAdd(other_write + d, truncate_gradient(rounding, -(grad_d * alpha)));
          }
        }
      }
      if (move_other && tail_flags != nullptr && lane_id == 0) {
        set_vertex_modified(tail_flags, k);
      }

      // all lanes have the same value
      if (lane_id == 0) { epoch_of_next_sample[e] = _epoch_of_next_sample + _epochs_per_sample; }

      auto _epoch_of_next_negative_sample = epoch_of_next_negative_sample[e];
      int n_neg_samples =
        static_cast<int>(T(epoch - _epoch_of_next_negative_sample) / epochs_per_negative_sample);

      run_negative_sampling<T, nnz_t>(
        n_neg_samples,
        seed,
        static_cast<nnz_t>(e),
        tail_n,
        vertex,
        gamma,
        params,
        [&](nnz_t t) -> T {
          T partial_dist = T(0);
#pragma unroll
          for (int i = 0; i < CPL; i++) {
            int d        = lane_id + i * 32;
            other_reg[i] = (d < n_components) ? __ldg(&tail_embedding[t * n_components + d]) : T(0);
            T diff       = current_reg[i] - other_reg[i];
            partial_dist += diff * diff;
          }
          return warp_all_reduce_sum(partial_dist);
        },
        [&](T coeff) {
#pragma unroll
          for (int i = 0; i < CPL; i++) {
            int d = lane_id + i * 32;
            if (d < n_components) {
              T grad_d = repulsive_component_grad<T>(coeff, current_reg[i] - other_reg[i]);
              current_reg[i] += grad_d * alpha;
            }
          }
        });

      if (lane_id == 0) {
        epoch_of_next_negative_sample[e] =
          _epoch_of_next_negative_sample + n_neg_samples * epochs_per_negative_sample;
      }
    }

    // Single writer per head_buffer slot, so no truncate_gradient needed.
#pragma unroll
    for (int i = 0; i < CPL; i++) {
      int d = lane_id + i * 32;
      if (d < n_components) {
        T delta = current_reg[i] - original_reg[i];
        if (delta != T(0)) { raft::myAtomicAdd(&head_buffer[vertex * n_components + d], delta); }
      }
    }
    if (head_flags != nullptr && lane_id == 0) { set_vertex_modified(head_flags, vertex); }

    // In deterministic mode, the host drives vertex chunking via vertex_offset
    // and calls sparse_apply between launches to flush buffered gradients in a
    // fixed order, so each warp processes exactly one vertex per launch.
    if (params.deterministic) break;
  }
}

/** Constant dispatch thresholds for the sequential-epoch kernels.

 SERIAL_PER_THREAD_MAX_NC: per-thread register kernel handles n_components in
   [1, 21]. With 3 register arrays of length N_COMPONENTS (current_reg,
   other_reg, original) at N=21 that's ~63 registers worth of payload per
   thread, which allows enough registers per thread on most architectures (e.g.
 Volta/Ampere/Hopper). Revisit this threshold if a future arch requires changes to the number of
 registers.

 SERIAL_PER_WARP_MAX_CPL: per-warp kernel handles cpl = ceil(n_components/32)
   in [1, 16], i.e. n_components in (21, 512]. */
constexpr int SERIAL_PER_THREAD_MAX_NC = 21;
constexpr int SERIAL_PER_WARP_MAX_CPL  = 16;
constexpr int SERIAL_PER_WARP_MAX_NC   = SERIAL_PER_WARP_MAX_CPL * 32;  // = 512

/**
 * Dispatch wrapper for sequential kernels, based on n_components.
 *
 *  n_components <= SERIAL_PER_THREAD_MAX_NC:
 *    Per-thread register kernel (optimize_sequential_kernel_vertex_per_thread).
 *    Each thread processes its own vertices.  All embedding data
 *    lives in registers. Best for small n_components where register pressure
 *    is low.
 *
 *  n_components in (SERIAL_PER_THREAD_MAX_NC, SERIAL_PER_WARP_MAX_NC]:
 *    Component-parallel warp kernel (optimize_sequential_kernel_vertex_per_warp).
 *    Each warp cooperatively processes one vertex. Lanes split the per-
 *    component work (lane k handles components k, k+32, k+64, …).
 *    Templated on CPL (components per lane = ceil(n_components/32)), keeping
 *    all data in per-lane registers.
 */
template <typename T, typename nnz_t, int TPB_X>
void call_optimize_sequential_kernel(T* head_embedding,
                                     T* head_buffer,
                                     uint32_t* head_flags,
                                     int head_n,
                                     T* tail_embedding,
                                     T* tail_buffer,
                                     uint32_t* tail_flags,
                                     int tail_n,
                                     const nnz_t* row_ptr,
                                     const int* tail,
                                     T const* epochs_per_sample,
                                     T* epoch_of_next_negative_sample,
                                     T* epoch_of_next_sample,
                                     T alpha,
                                     T gamma,
                                     uint64_t seed,
                                     bool move_other,
                                     UMAPParams const* params,
                                     int epoch,
                                     cudaStream_t& stream,
                                     T rounding)
{
  T nsr_inv = T(1.0) / params->negative_sample_rate;

  int num_sms = 0;
  int device  = -1;
  RAFT_CUDA_TRY(cudaGetDevice(&device));
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));

  // threads_per_vertex: 1 for reg kernel (one thread = one vertex),
  //                    32 for comp kernel (one warp  = one vertex)
  auto launch_kernel = [&](auto kernel_fn, int tpb, int smem_size, int threads_per_vertex) {
    int blocks_per_sm = 0;
    RAFT_CUDA_TRY(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel_fn, tpb, smem_size));
    RAFT_EXPECTS(blocks_per_sm > 0,
                 "No active blocks for tpb=%d and smem=%d; adjust the sequential-kernel launch "
                 "configuration.",
                 tpb,
                 smem_size);
    dim3 grid(num_sms * blocks_per_sm, 1, 1);
    dim3 blk(tpb, 1, 1);

    auto do_launch = [&](int vertex_offset) {
      kernel_fn<<<grid, blk, smem_size, stream>>>(head_embedding,
                                                  head_buffer,
                                                  head_flags,
                                                  tail_embedding,
                                                  tail_buffer,
                                                  tail_flags,
                                                  row_ptr,
                                                  tail,
                                                  head_n,
                                                  tail_n,
                                                  epochs_per_sample,
                                                  epoch_of_next_negative_sample,
                                                  epoch_of_next_sample,
                                                  alpha,
                                                  epoch,
                                                  gamma,
                                                  seed,
                                                  move_other,
                                                  *params,
                                                  nsr_inv,
                                                  rounding,
                                                  vertex_offset);
    };

    if (params->deterministic) {
      int chunk_size = ML::checked_mul<int>(ML::narrow_cast<int>(grid.x), tpb / threads_per_vertex);
      RAFT_EXPECTS(chunk_size > 0, "Sequential-kernel chunk_size must be > 0.");
      for (int v_off = 0; v_off < head_n; v_off += chunk_size) {
        do_launch(v_off);
        sparse_apply_embedding_updates<T, nnz_t, TPB_X>(head_embedding,
                                                        head_buffer,
                                                        head_flags,
                                                        head_n,
                                                        tail_embedding,
                                                        tail_buffer,
                                                        tail_flags,
                                                        tail_n,
                                                        params,
                                                        move_other,
                                                        stream);
      }
    } else {
      do_launch(0);
    }
  };

  if (params->n_components <= SERIAL_PER_THREAD_MAX_NC) {
#define LAUNCH_PER_THREAD_KERNEL(NC) \
  launch_kernel(optimize_sequential_kernel_vertex_per_thread<T, nnz_t, TPB_X, NC>, TPB_X, 0, 1)
    // 3 register arrays × N_COMPONENTS = 63 registers at N=21.  This keeps occupancy reasonable on
    // most architectures. This is also heuristically a good threshold.
    static_assert(SERIAL_PER_THREAD_MAX_NC == 21,
                  "Per-thread switch enumerates 1..21; update both together.");
    switch (params->n_components) {
      case 1: LAUNCH_PER_THREAD_KERNEL(1); break;
      case 2: LAUNCH_PER_THREAD_KERNEL(2); break;
      case 3: LAUNCH_PER_THREAD_KERNEL(3); break;
      case 4: LAUNCH_PER_THREAD_KERNEL(4); break;
      case 5: LAUNCH_PER_THREAD_KERNEL(5); break;
      case 6: LAUNCH_PER_THREAD_KERNEL(6); break;
      case 7: LAUNCH_PER_THREAD_KERNEL(7); break;
      case 8: LAUNCH_PER_THREAD_KERNEL(8); break;
      case 9: LAUNCH_PER_THREAD_KERNEL(9); break;
      case 10: LAUNCH_PER_THREAD_KERNEL(10); break;
      case 11: LAUNCH_PER_THREAD_KERNEL(11); break;
      case 12: LAUNCH_PER_THREAD_KERNEL(12); break;
      case 13: LAUNCH_PER_THREAD_KERNEL(13); break;
      case 14: LAUNCH_PER_THREAD_KERNEL(14); break;
      case 15: LAUNCH_PER_THREAD_KERNEL(15); break;
      case 16: LAUNCH_PER_THREAD_KERNEL(16); break;
      case 17: LAUNCH_PER_THREAD_KERNEL(17); break;
      case 18: LAUNCH_PER_THREAD_KERNEL(18); break;
      case 19: LAUNCH_PER_THREAD_KERNEL(19); break;
      case 20: LAUNCH_PER_THREAD_KERNEL(20); break;
      case 21: LAUNCH_PER_THREAD_KERNEL(21); break;
    }

#undef LAUNCH_PER_THREAD_KERNEL
  } else {
    // Supports up to n_components = 512 (cpl = 16, since each warp lane handles cpl * 32
    // components)
    int cpl = (params->n_components + 31) / 32;  // components per lane

#define LAUNCH_PER_WARP_KERNEL(CPL_VAL) \
  launch_kernel(optimize_sequential_kernel_vertex_per_warp<T, nnz_t, TPB_X, CPL_VAL>, TPB_X, 0, 32)
    static_assert(SERIAL_PER_WARP_MAX_CPL == 16,
                  "Per-warp switch enumerates 1..16; update both together.");
    switch (cpl) {
      case 1: LAUNCH_PER_WARP_KERNEL(1); break;
      case 2: LAUNCH_PER_WARP_KERNEL(2); break;
      case 3: LAUNCH_PER_WARP_KERNEL(3); break;
      case 4: LAUNCH_PER_WARP_KERNEL(4); break;
      case 5: LAUNCH_PER_WARP_KERNEL(5); break;
      case 6: LAUNCH_PER_WARP_KERNEL(6); break;
      case 7: LAUNCH_PER_WARP_KERNEL(7); break;
      case 8: LAUNCH_PER_WARP_KERNEL(8); break;
      case 9: LAUNCH_PER_WARP_KERNEL(9); break;
      case 10: LAUNCH_PER_WARP_KERNEL(10); break;
      case 11: LAUNCH_PER_WARP_KERNEL(11); break;
      case 12: LAUNCH_PER_WARP_KERNEL(12); break;
      case 13: LAUNCH_PER_WARP_KERNEL(13); break;
      case 14: LAUNCH_PER_WARP_KERNEL(14); break;
      case 15: LAUNCH_PER_WARP_KERNEL(15); break;
      case 16: LAUNCH_PER_WARP_KERNEL(16); break;
      default:
        // Should be unreachable in practice: the Python layer rejects
        // force_serial_epochs=True with n_components > SERIAL_PER_WARP_MAX_NC,
        // and falls back to the parallel batch kernel.
        RAFT_EXPECTS(false,
                     "force_serial_epochs requires n_components <= %d, got %d (cpl=%d).",
                     SERIAL_PER_WARP_MAX_NC,
                     params->n_components,
                     cpl);
    }

#undef LAUNCH_PER_WARP_KERNEL
  }
}

/**
 * @param head_buffer: Buffer the gradient update to head_embedding when deterministic
 *                     result is required.  They are the same pointer if random seed is not
 *                     provided.
 * @param tail_buffer: Similar to head_buffer, but for tail_embedding.
 * @param head_flags: flags tracking which head vertices were modified (for sparse apply).
 * @param tail_flags: flags tracking which tail vertices were modified (for sparse apply).
 * @param head:        Row index in COO connectivity graph.
 * @param tail:        Column index in COO connectivity graph.
 * @param alpha:       Learning rate
 * @param n:           Current epoch
 * @param rounding:    Floating rounding factor used to truncate the gradient update for
 *                     deterministic result.
 */
template <typename T, typename nnz_t, int TPB_X>
void call_optimize_batch_kernel(T* head_embedding,
                                T* head_buffer,
                                uint32_t* head_flags,
                                int head_n,
                                T* tail_embedding,
                                T* tail_buffer,
                                uint32_t* tail_flags,
                                int tail_n,
                                const int* head,
                                const int* tail,
                                nnz_t nnz,
                                T const* epochs_per_sample,
                                T* epoch_of_next_negative_sample,
                                T* epoch_of_next_sample,
                                T alpha,
                                T gamma,
                                uint64_t seed,
                                bool move_other,
                                UMAPParams const* params,
                                int n,
                                dim3& grid,
                                dim3& blk,
                                cudaStream_t& stream,
                                T rounding)
{
  std::size_t requiredSize = TPB_X * params->n_components * 2;
  requiredSize *= sizeof(T);
  bool use_shared_mem = requiredSize < static_cast<std::size_t>(raft::getSharedMemPerBlock());
  T nsr_inv           = T(1.0) / params->negative_sample_rate;

  auto stream_view = cuda::stream_ref(stream);

  auto launch_kernel = [&](size_t offset = 0) {
    if (params->n_components == 2) {
      // multicore implementation with registers
      optimize_batch_kernel_reg<T, nnz_t, TPB_X, 2>
        <<<grid, blk, 0, stream>>>(head_embedding,
                                   head_buffer,
                                   head_flags,
                                   tail_embedding,
                                   tail_buffer,
                                   tail_flags,
                                   tail_n,
                                   head,
                                   tail,
                                   nnz,
                                   epochs_per_sample,
                                   epoch_of_next_negative_sample,
                                   epoch_of_next_sample,
                                   alpha,
                                   n,
                                   gamma,
                                   seed,
                                   move_other,
                                   *params,
                                   nsr_inv,
                                   rounding,
                                   offset);
    } else if (use_shared_mem) {
      // multicore implementation with shared memory
      optimize_batch_kernel<T, nnz_t, TPB_X, true>
        <<<grid, blk, requiredSize, stream>>>(head_embedding,
                                              head_buffer,
                                              head_flags,
                                              tail_embedding,
                                              tail_buffer,
                                              tail_flags,
                                              tail_n,
                                              head,
                                              tail,
                                              nnz,
                                              epochs_per_sample,
                                              epoch_of_next_negative_sample,
                                              epoch_of_next_sample,
                                              alpha,
                                              n,
                                              gamma,
                                              seed,
                                              move_other,
                                              *params,
                                              nsr_inv,
                                              rounding,
                                              offset);
    } else {
      // multicore implementation without shared memory
      optimize_batch_kernel<T, nnz_t, TPB_X, false>
        <<<grid, blk, 0, stream>>>(head_embedding,
                                   head_buffer,
                                   head_flags,
                                   tail_embedding,
                                   tail_buffer,
                                   tail_flags,
                                   tail_n,
                                   head,
                                   tail,
                                   nnz,
                                   epochs_per_sample,
                                   epoch_of_next_negative_sample,
                                   epoch_of_next_sample,
                                   alpha,
                                   n,
                                   gamma,
                                   seed,
                                   move_other,
                                   *params,
                                   nsr_inv,
                                   rounding,
                                   offset);
    }
  };

  if (params->deterministic) {
    // for deterministic mode, we enforce a sequential behavior using n_chunks. after 1 chunk
    // (including TPB_X * grid.x threads each doing an update) is processed, we apply the gradients
    // in the buffer to the resulting embedding. The next chunk is processed based on the updated
    // embedding from the previous chunk. this ensures that the updates are applied in a sequential
    // manner (which is crucial for avoiding outliers), and the resulting embedding is deterministic
    // because we use buffers.
    size_t chunk_size = static_cast<size_t>(TPB_X) * static_cast<size_t>(grid.x);
    for (size_t offset = 0; offset < static_cast<size_t>(nnz); offset += chunk_size) {
      launch_kernel(offset);
      sparse_apply_embedding_updates<T, nnz_t, TPB_X>(head_embedding,
                                                      head_buffer,
                                                      head_flags,
                                                      head_n,
                                                      tail_embedding,
                                                      tail_buffer,
                                                      tail_flags,
                                                      tail_n,
                                                      params,
                                                      move_other,
                                                      stream);
    }
  } else {
    // for non-deterministic mode, we launch the kernel once with nnz/n_chunks threads.
    // each thread strides through n_chunks updates and writes gradients immediately to the
    // resulting embedding inside the kernel, increasing the chances of a sequential update which is
    // crucial for avoiding outliers.
    launch_kernel(0);
  }
}
}  // namespace Algo
}  // namespace SimplSetEmbed
}  // namespace UMAPAlgo
