/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/linalg/eltwise.cuh>
#include <raft/util/cudart_utils.hpp>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include <float.h>
#include <math.h>

#define restrict __restrict__

namespace ML {
namespace TSNE {

/****************************************/
/* Finds the best Gaussian bandwidth for
    each row in the dataset             */
template <typename value_idx, typename value_t>
CUML_KERNEL void sigmas_kernel(const value_t* restrict distances,
                               value_t* restrict P,
                               const float perplexity,
                               const float desired_entropy,
                               const int epochs,
                               const float tol,
                               const value_idx n,
                               const int k)
{
  // For every item in row
  const auto i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i >= n) return;

  value_t beta_min = -INFINITY, beta_max = INFINITY;
  value_t beta           = 1;
  register const auto ik = i * k;

  for (int step = 0; step < epochs; step++) {
    value_t sum_Pi = FLT_EPSILON;

    // Exponentiate to get Gaussian
    for (int j = 0; j < k; j++) {
      P[ik + j] = __expf(-distances[ik + j] * beta);
      sum_Pi += P[ik + j];
    }

    // Normalize
    value_t sum_disti_Pi = 0;
    const value_t div    = __fdividef(1.0f, sum_Pi);
    for (int j = 0; j < k; j++) {
      P[ik + j] *= div;
      sum_disti_Pi += distances[ik + j] * P[ik + j];
    }

    const value_t entropy      = __logf(sum_Pi) + beta * sum_disti_Pi;
    const value_t entropy_diff = entropy - desired_entropy;
    if (fabs(entropy_diff) <= tol) break;

    // Bisection search
    if (entropy_diff > 0) {
      beta_min = beta;
      if (isinf(beta_max))
        beta *= 2.0f;
      else
        beta = (beta + beta_max) * 0.5f;
    } else {
      beta_max = beta;
      if (isinf(beta_min))
        beta *= 0.5f;
      else
        beta = (beta + beta_min) * 0.5f;
    }
  }
}

/****************************************/
/* Finds the best Gaussian bandwidth for
    each row in the dataset             */
template <typename value_idx, typename value_t>
CUML_KERNEL void sigmas_kernel_2d(const value_t* restrict distances,
                                  value_t* restrict P,
                                  const float perplexity,
                                  const float desired_entropy,
                                  const int epochs,
                                  const float tol,
                                  const value_idx n)
{
  // For every item in row
  const auto i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i >= n) return;

  value_t beta_min = -INFINITY, beta_max = INFINITY;
  value_t beta           = 1;
  register const auto ik = i * 2;

  for (int step = 0; step < epochs; step++) {
    // Exponentiate to get Gaussian
    P[ik]                = __expf(-distances[ik] * beta);
    P[ik + 1]            = __expf(-distances[ik + 1] * beta);
    const value_t sum_Pi = FLT_EPSILON + P[ik] + P[ik + 1];

    // Normalize
    const value_t div = __fdividef(1.0f, sum_Pi);
    P[ik] *= div;
    P[ik + 1] *= div;
    const value_t sum_disti_Pi = distances[ik] * P[ik] + distances[ik + 1] * P[ik + 1];

    const value_t entropy      = __logf(sum_Pi) + beta * sum_disti_Pi;
    const value_t entropy_diff = entropy - desired_entropy;
    if (fabs(entropy_diff) <= tol) break;

    // Bisection search
    if (entropy_diff > 0) {
      beta_min = beta;
      if (isinf(beta_max))
        beta *= 2.0f;
      else
        beta = (beta + beta_max) * 0.5f;
    } else {
      beta_max = beta;
      if (isinf(beta_min))
        beta *= 0.5f;
      else
        beta = (beta + beta_min) * 0.5f;
    }
  }
}

/****************************************/
template <typename value_idx, typename value_t>
void perplexity_search(const value_t* restrict distances,
                       value_t* restrict P,
                       const float perplexity,
                       const int epochs,
                       const float tol,
                       const value_idx n,
                       const int dim,
                       const raft::handle_t& handle)
{
  const float desired_entropy = logf(perplexity);
  cudaStream_t stream         = handle.get_stream().get();

  if (dim == 2)
    sigmas_kernel_2d<<<raft::ceildiv(n, (value_idx)1024), 1024, 0, stream>>>(
      distances, P, perplexity, desired_entropy, epochs, tol, n);
  else
    sigmas_kernel<<<raft::ceildiv(n, (value_idx)1024), 1024, 0, stream>>>(
      distances, P, perplexity, desired_entropy, epochs, tol, n, dim);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  handle.sync_stream(stream);
}

/****************************************/
/* Compute attractive forces in O(uN) time.
    Uses only nearest neighbors         */
template <typename value_idx, typename value_t>
CUML_KERNEL void attractive_kernel(const value_t* restrict VAL,
                                   const value_idx* restrict COL,
                                   const value_idx* restrict ROW,
                                   const value_t* restrict Y,
                                   const value_t* restrict norm,
                                   value_t* restrict attract,
                                   value_t* restrict Qs,
                                   const value_idx NNZ,
                                   const value_idx n,
                                   const value_idx dim,
                                   const value_t dof)
{
  const auto index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= NNZ) return;
  const auto i = ROW[index], j = COL[index];

  // Euclidean distances
  // TODO: can provide any distance ie cosine
  // #862
  value_t dist = 0;
  for (int k = 0; k < dim; k++)
    dist += Y[k * n + i] * Y[k * n + j];
  dist = norm[i] + norm[j] - 2.0f * dist;

  const value_t P  = VAL[index];
  const value_t Q  = compute_q(dist, dof);
  const value_t PQ = P * Q;

  // Apply forces
  for (int k = 0; k < dim; k++) {
    raft::myAtomicAdd(&attract[k * n + i], PQ * (Y[k * n + i] - Y[k * n + j]));
  }

  if (Qs) {  // when computing KL div
    Qs[index] = Q;
  }
}

/****************************************/
/* Special case when dim == 2. Can speed
    up many calculations up             */
template <typename value_idx, typename value_t>
CUML_KERNEL void attractive_kernel_2d(const value_t* restrict VAL,
                                      const value_idx* restrict COL,
                                      const value_idx* restrict ROW,
                                      const value_t* restrict Y1,
                                      const value_t* restrict Y2,
                                      const value_t* restrict norm,
                                      value_t* restrict attract1,
                                      value_t* restrict attract2,
                                      value_t* restrict Qs,
                                      const value_idx NNZ,
                                      const value_t dof)
{
  const auto index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= NNZ) return;
  const auto i = ROW[index], j = COL[index];

  // Euclidean distances
  // TODO: can provide any distance ie cosine
  // #862
  const value_t dist = norm[i] + norm[j] - 2.0f * (Y1[i] * Y1[j] + Y2[i] * Y2[j]);

  const value_t P  = VAL[index];
  const value_t Q  = compute_q(dist, dof);
  const value_t PQ = P * Q;

  // Apply forces
  raft::myAtomicAdd(&attract1[i], PQ * (Y1[i] - Y1[j]));
  raft::myAtomicAdd(&attract2[i], PQ * (Y2[i] - Y2[j]));

  if (Qs) {  // when computing KL div
    Qs[index] = Q;
  }
}

/****************************************/
template <typename value_idx, typename value_t>
void attractive_forces(const value_t* restrict VAL,
                       const value_idx* restrict COL,
                       const value_idx* restrict ROW,
                       const value_t* restrict Y,
                       const value_t* restrict norm,
                       value_t* restrict attract,
                       value_t* restrict Qs,
                       const value_idx NNZ,
                       const value_idx n,
                       const value_idx dim,
                       const value_t dof,
                       cudaStream_t stream)
{
  RAFT_CUDA_TRY(cudaMemsetAsync(attract, 0, sizeof(value_t) * n * dim, stream));

  // TODO: Calculate Kullback-Leibler divergence
  // #863
  // For general embedding dimensions
  if (dim != 2) {
    attractive_kernel<<<raft::ceildiv(NNZ, (value_idx)1024), 1024, 0, stream>>>(
      VAL, COL, ROW, Y, norm, attract, Qs, NNZ, n, dim, dof);
  }
  // For special case dim == 2
  else {
    attractive_kernel_2d<<<raft::ceildiv(NNZ, (value_idx)1024), 1024, 0, stream>>>(
      VAL, COL, ROW, Y, Y + n, norm, attract, attract + n, Qs, NNZ, dof);
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/****************************************/
/* Computes repulsive forces in pseudo-O(N^2)
    time where many of the math ops are
    made considerably faster.           */
template <typename value_idx, typename value_t>
CUML_KERNEL void repulsive_kernel(const value_t* restrict Y,
                                  value_t* restrict repel,
                                  const value_t* restrict norm,
                                  value_t* restrict Z_sum1,
                                  value_t* restrict Z_sum2,
                                  const value_idx n,
                                  const value_idx dim,
                                  const value_t df_power,  // -(df + 1)/2)
                                  const value_t recp_df)   // 1 / df
{
  const auto j = (blockIdx.x * blockDim.x) + threadIdx.x;  // for every item in row
  const auto i = (blockIdx.y * blockDim.y) + threadIdx.y;  // for every row
  if (j >= i || i >= n || j >= n) return;

  // Euclidean distances
  // TODO: can provide any distance ie cosine
  value_t d = 0;
  for (int k = 0; k < dim; k++)
    d += Y[k * n + i] * Y[k * n + j];
  const value_t euclidean_d = -2.0f * d + norm[i] + norm[j];

  // Q and Q^2
  const value_t Q  = __powf((1.0f + euclidean_d * recp_df), df_power);
  const value_t Q2 = Q * Q;

  // Apply forces
  for (int k = 0; k < dim; k++) {
    const value_t force = Q2 * (Y[k * n + j] - Y[k * n + i]);
    raft::myAtomicAdd(&repel[k * n + i], force);
    raft::myAtomicAdd(&repel[k * n + j], force);
  }

  // Sum up Z sum
  if (i % 2 == 0)
    raft::myAtomicAdd(&Z_sum1[i], Q);
  else
    raft::myAtomicAdd(&Z_sum2[i], Q);
}

/****************************************/
/* Special case when dim == 2. Much faster
    since calculations are streamlined. */
template <typename value_idx, typename value_t>
CUML_KERNEL void repulsive_kernel_2d(const value_t* restrict Y1,
                                     const value_t* restrict Y2,
                                     value_t* restrict repel1,
                                     value_t* restrict repel2,
                                     const value_t* restrict norm,
                                     value_t* restrict Z_sum1,
                                     value_t* restrict Z_sum2,
                                     const value_idx n)
{
  const auto j = (blockIdx.x * blockDim.x) + threadIdx.x;  // for every item in row
  const auto i = (blockIdx.y * blockDim.y) + threadIdx.y;  // for every row
  if (j >= i || i >= n || j >= n) return;

  // Euclidean distances
  // TODO: can provide any distance ie cosine
  // #862
  const value_t euclidean_d = norm[i] + norm[j] - 2.0f * (Y1[i] * Y1[j] + Y2[i] * Y2[j]);
  const value_t Q           = __fdividef(1.0f, (1.0f + euclidean_d));
  const value_t Q2          = Q * Q;

  const value_t force1 = Q2 * (Y1[j] - Y1[i]);
  const value_t force2 = Q2 * (Y2[j] - Y2[i]);

  // Add forces
  raft::myAtomicAdd(&repel1[i], force1);
  raft::myAtomicAdd(&repel1[j], -force1);

  raft::myAtomicAdd(&repel2[i], force2);
  raft::myAtomicAdd(&repel2[j], -force2);

  // Sum up Z sum
  if (i % 2 == 0)
    raft::myAtomicAdd(&Z_sum1[i], Q);
  else
    raft::myAtomicAdd(&Z_sum2[i], Q);
}

/****************************************/
template <typename value_idx, typename value_t, int TPB_X = 32, int TPB_Y = 32>
value_t repulsive_forces(const value_t* restrict Y,
                         value_t* restrict repel,
                         const value_t* restrict norm,
                         value_t* restrict Z_sum,
                         const value_idx n,
                         const value_idx dim,
                         const value_t df_power,  // -(df + 1)/2)
                         const value_t recp_df,
                         cudaStream_t stream)
{
  RAFT_CUDA_TRY(cudaMemsetAsync(Z_sum, 0, sizeof(value_t) * 2 * n, stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(repel, 0, sizeof(value_t) * n * dim, stream));

  const dim3 threadsPerBlock(TPB_X, TPB_Y);
  const dim3 numBlocks(raft::ceildiv(n, (value_idx)TPB_X), raft::ceildiv(n, (value_idx)TPB_Y));

  // For general embedding dimensions
  if (dim != 2) {
    repulsive_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
      Y, repel, norm, Z_sum, Z_sum + n, n, dim, df_power, recp_df);
  }
  // For special dim == 2 case
  else {
    repulsive_kernel_2d<<<numBlocks, threadsPerBlock, 0, stream>>>(
      Y, Y + n, repel, repel + n, norm, Z_sum, Z_sum + n, n);
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Find sum(Z_sum)
  thrust::device_ptr<value_t> begin = thrust::device_pointer_cast(Z_sum);
  value_t Z = thrust::reduce(thrust::cuda::par.on(stream), begin, begin + 2 * n);
  return 1.0f / (2.0f * (Z + (value_t)n));  // Notice + n since diagonal of repulsion sums to n
}

/****************************************/
/* Applies or integrates all forces. Uses
    more gains and constrains the output
    for output stability                */
template <typename value_idx, typename value_t>
CUML_KERNEL void apply_kernel(value_t* restrict Y,
                              value_t* restrict velocity,
                              const value_t* restrict attract,
                              const value_t* restrict repel,
                              value_t* restrict means,
                              value_t* restrict gains,
                              const float Z,  // sum(Q)
                              const float learning_rate,
                              const float C,  // constant from T-Dist Degrees of Freedom
                              const float exaggeration,
                              const float momentum,
                              const value_idx SIZE,  // SIZE = n*dim
                              const value_idx n,
                              const float min_gain,
                              value_t* restrict gradient,
                              const bool check_convergence)
{
  const auto index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= SIZE) return;

  const value_t dy = C * (exaggeration * attract[index] + Z * repel[index]);
  if (check_convergence) gradient[index] = dy * dy;

  // Find new gain
  // TODO: Incorporate AadaBound (2019) or Adam
  // #864
  if (signbit(dy) != signbit(velocity[index]))
    gains[index] += 0.2f;  // Original TSNE is 0.2
  else
    gains[index] *= 0.8f;  // Original TSNE is 0.8
  if (gains[index] < min_gain) gains[index] = min_gain;

  velocity[index] = momentum * velocity[index] - learning_rate * dy * gains[index];
  Y[index] += velocity[index];

  // Add to mean
  // raft::myAtomicAdd(&means[index / n], Y[index]);
}

/****************************************/
template <typename value_idx, typename value_t, int TPB_X = 32, int TPB_Y = 32>
value_t apply_forces(value_t* restrict Y,
                     value_t* restrict velocity,
                     const value_t* restrict attract,
                     const value_t* restrict repel,
                     value_t* restrict means,
                     value_t* restrict gains,
                     const float Z,  // sum(Q)
                     const float learning_rate,
                     const float C,  // constant from T-dist
                     const float exaggeration,
                     const float momentum,
                     const value_idx dim,
                     const value_idx n,
                     const float min_gain,
                     value_t* restrict gradient,
                     const bool check_convergence,
                     cudaStream_t stream)
{
  // cudaMemset(means, 0, sizeof(float) * dim);
  if (check_convergence)
    RAFT_CUDA_TRY(cudaMemsetAsync(gradient, 0, sizeof(value_t) * n * dim, stream));

  apply_kernel<<<raft::ceildiv(n * dim, (value_idx)1024), 1024, 0, stream>>>(Y,
                                                                             velocity,
                                                                             attract,
                                                                             repel,
                                                                             means,
                                                                             gains,
                                                                             Z,
                                                                             learning_rate,
                                                                             C,
                                                                             exaggeration,
                                                                             momentum,
                                                                             n * dim,
                                                                             n,
                                                                             min_gain,
                                                                             gradient,
                                                                             check_convergence);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Find sum of gradient norms
  float gradient_norm = INFINITY;
  if (check_convergence) {
    thrust::device_ptr<value_t> begin = thrust::device_pointer_cast(gradient);
    gradient_norm = sqrtf(thrust::reduce(thrust::cuda::par.on(stream), begin, begin + n * dim));
  }

  // TODO: Subtract means
  return gradient_norm;
}

}  // namespace TSNE
}  // namespace ML
