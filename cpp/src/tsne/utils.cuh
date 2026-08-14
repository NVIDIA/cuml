/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/common/logger.hpp>
#include <cuml/common/utils.hpp>

#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/random/rng.cuh>
#include <raft/stats/sum.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/device_atomics.cuh>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdlib>

/**
 * @brief Performs P + P.T.
 * @param[out] vector: The output vector you want to overwrite with randomness.
 * @param[in] minimum: The minimum value in the output vector you want.
 * @param[in] maximum: The maximum value in the output vector you want.
 * @param[in] size: The size of the output vector.
 * @param[in] stream: The GPU stream.
 * @param[in] seed: If seed < 0, then the output is pure randomness. If >= 0, then you can
 * reproduce TSNE.
 */

template <typename value_t = float>
void random_vector(value_t* vector,
                   const value_t minimum,
                   const value_t maximum,
                   const int size,
                   cudaStream_t stream,
                   long long seed = -1)
{
  if (seed < 0) {
    // Get random seed based on time of day
    struct timeval tp;
    gettimeofday(&tp, NULL);
    seed = tp.tv_sec * 1000 + tp.tv_usec;
  }
  raft::random::Rng random(seed);
  random.uniform<value_t>(vector, size, minimum, maximum, stream);
}

long start, end;
struct timeval timecheck;
double SymmetrizeTime = 0, DistancesTime = 0, NormalizeTime = 0, PerplexityTime = 0,
       BoundingBoxKernel_time = 0, ClearKernel1_time = 0, TreeBuildingKernel_time = 0,
       ClearKernel2_time = 0, SummarizationKernel_time = 0, SortKernel_time = 0, RepulsionTime = 0,
       Reduction_time = 0, attractive_time = 0, IntegrationKernel_time = 0;

// To silence warnings

#define START_TIMER                                                         \
  if (ML::default_logger().should_log(rapids_logger::level_enum::debug)) {  \
    gettimeofday(&timecheck, NULL);                                         \
    start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000; \
  }

#define END_TIMER(add_onto)                                                \
  if (ML::default_logger().should_log(rapids_logger::level_enum::debug)) { \
    gettimeofday(&timecheck, NULL);                                        \
    end = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;  \
    add_onto += (end - start);                                             \
  }

#define PRINT_TIMES                                                                              \
  if (ML::default_logger().should_log(rapids_logger::level_enum::debug)) {                       \
    double total = (SymmetrizeTime + DistancesTime + NormalizeTime + PerplexityTime +            \
                    BoundingBoxKernel_time + ClearKernel1_time + TreeBuildingKernel_time +       \
                    ClearKernel2_time + SummarizationKernel_time + SortKernel_time +             \
                    RepulsionTime + Reduction_time + attractive_time + IntegrationKernel_time) / \
                   100.0;                                                                        \
    CUML_LOG_DEBUG(                                                                              \
      "SymmetrizeTime = %.lf (%.lf)\n"                                                           \
      "DistancesTime = %.lf (%.lf)\n"                                                            \
      "NormalizeTime = %.lf (%.lf)\n"                                                            \
      "PerplexityTime = %.lf (%.lf)\n"                                                           \
      "BoundingBoxKernel_time = %.lf (%.lf)\n"                                                   \
      "ClearKernel1_time  = %.lf (%.lf)\n"                                                       \
      "TreeBuildingKernel_time  = %.lf (%.lf)\n"                                                 \
      "ClearKernel2_time  = %.lf (%.lf)\n"                                                       \
      "SummarizationKernel_time  = %.lf (%.lf)\n"                                                \
      "SortKernel_time  = %.lf (%.lf)\n"                                                         \
      "RepulsionTime  = %.lf (%.lf)\n"                                                           \
      "Reduction_time  = %.lf (%.lf)\n"                                                          \
      "attractive_time  = %.lf (%.lf)\n"                                                         \
      "IntegrationKernel_time = %.lf (%.lf)\n"                                                   \
      "TOTAL TIME = %.lf",                                                                       \
      SymmetrizeTime,                                                                            \
      SymmetrizeTime / total,                                                                    \
      DistancesTime,                                                                             \
      DistancesTime / total,                                                                     \
      NormalizeTime,                                                                             \
      NormalizeTime / total,                                                                     \
      PerplexityTime,                                                                            \
      PerplexityTime / total,                                                                    \
      BoundingBoxKernel_time,                                                                    \
      BoundingBoxKernel_time / total,                                                            \
      ClearKernel1_time,                                                                         \
      ClearKernel1_time / total,                                                                 \
      TreeBuildingKernel_time,                                                                   \
      TreeBuildingKernel_time / total,                                                           \
      ClearKernel2_time,                                                                         \
      ClearKernel2_time / total,                                                                 \
      SummarizationKernel_time,                                                                  \
      SummarizationKernel_time / total,                                                          \
      SortKernel_time,                                                                           \
      SortKernel_time / total,                                                                   \
      RepulsionTime,                                                                             \
      RepulsionTime / total,                                                                     \
      Reduction_time,                                                                            \
      Reduction_time / total,                                                                    \
      attractive_time,                                                                           \
      attractive_time / total,                                                                   \
      IntegrationKernel_time,                                                                    \
      IntegrationKernel_time / total,                                                            \
      total * 100.0);                                                                            \
  }

template <typename value_t, typename value_idx, int TPB = 256>
CUML_KERNEL void min_max_kernel(
  const value_t* Y, const value_idx n, value_t* min, value_t* max, bool find_min = true)
{
  auto tid = threadIdx.x + blockDim.x * blockIdx.x;

  typedef cub::BlockReduce<value_t, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage_min;
  __shared__ typename BlockReduce::TempStorage temp_storage_max;

  value_t thread_min, thread_max;
  if (tid < n) {
    thread_max = Y[tid];
    if (find_min) thread_min = thread_max;
  } else {
    if (find_min) thread_min = std::numeric_limits<value_t>::max();
    thread_max = std::numeric_limits<value_t>::lowest();
  }

  value_t block_min, block_max;
  if (find_min) { block_min = BlockReduce(temp_storage_min).Reduce(thread_min, cuda::minimum{}); }

  block_max = BlockReduce(temp_storage_max).Reduce(thread_max, cuda::maximum{});

  // results stored in first thread of block

  if (threadIdx.x == 0) {
    if (find_min) atomicMin(min, block_min);
    atomicMax(max, block_max);
  }
}

/**
 * CUDA kernel to compute KL divergence
 */
template <typename value_idx, typename value_t>
CUML_KERNEL void compute_kl_div_k(const value_t* Ps,
                                  const value_t* Qs,
                                  value_t* __restrict__ KL_divs,
                                  const value_idx NNZ)
{
  const auto index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= NNZ) return;
  const value_t P = Ps[index];
  const value_t Q = max(Qs[index], FLT_EPSILON);

  KL_divs[index] = P * __logf(__fdividef(max(P, FLT_EPSILON), Q));
}

/**
 * Compute KL divergence
 */
template <typename value_t>
value_t compute_kl_div(
  value_t* __restrict__ Ps, value_t* Qs, value_t* KL_divs, const size_t NNZ, cudaStream_t stream)
{
  value_t P_sum = thrust::reduce(rmm::exec_policy(stream), Ps, Ps + NNZ);
  raft::linalg::scalarMultiply(Ps, Ps, 1.0f / P_sum, NNZ, stream);

  value_t Q_sum = thrust::reduce(rmm::exec_policy(stream), Qs, Qs + NNZ);
  raft::linalg::scalarMultiply(Qs, Qs, 1.0f / Q_sum, NNZ, stream);

  const size_t block = 128;
  const size_t grid  = raft::ceildiv(NNZ, block);
  compute_kl_div_k<<<grid, block, 0, stream>>>(Ps, Qs, KL_divs, NNZ);

  return thrust::reduce(rmm::exec_policy(stream), KL_divs, KL_divs + NNZ);
}

template <typename value_t>
__device__ value_t compute_q(value_t dist, value_t dof)
{
  const value_t exponent = (dof + 1.0f) / 2.0f;
  const value_t Q        = __powf(dof / (dof + dist), exponent);
  return Q;
}
