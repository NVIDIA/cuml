/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * This code is based on https://github.com/CannyLab/tsne-cuda (licensed under
 * the BSD 3-clause license at cannylabs_tsne_license.txt), which is in turn a
 * CUDA implementation of Linderman et al.'s FIt-SNE (MIT license)
 * (https://github.com/KlugerLab/FIt-SNE).
 */

#pragma once

#include "fft_kernels.cuh"
#include "utils.cuh"

#include <common/device_utils.cuh>

#include <raft/core/copy.cuh>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/init.cuh>
#include <raft/stats/sum.cuh>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/std/functional>
#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <cufft_utils.h>
#include <stdint.h>

#include <cmath>
#include <utility>

namespace ML {
namespace TSNE {

const static int NTHREADS_1024 = 1024;
const static int NTHREADS_128  = 128;
const static int NTHREADS_32   = 32;

struct FunctionalSqrt {
  template <typename value_t>
  __host__ __device__ float operator()(const value_t& x) const
  {
    return pow(x, 0.5);
  }
};
struct FunctionalSquare {
  template <typename value_t>
  __host__ __device__ float operator()(const value_t& x) const
  {
    return x * x;
  }
};

template <typename T>
cufftResult CUFFTAPI cufft_MakePlanMany(cufftHandle plan,
                                        T rank,
                                        T* n,
                                        T* inembed,
                                        T istride,
                                        T idist,
                                        T* onembed,
                                        T ostride,
                                        T odist,
                                        cufftType type,
                                        T batch,
                                        size_t* workSize);

cufftResult CUFFTAPI cufft_MakePlanMany(cufftHandle plan,
                                        int rank,
                                        int64_t* n,
                                        int64_t* inembed,
                                        int64_t istride,
                                        int64_t idist,
                                        int64_t* onembed,
                                        int64_t ostride,
                                        int64_t odist,
                                        cufftType type,
                                        int64_t batch,
                                        size_t* workSize)
{
  return cufftMakePlanMany64(plan,
                             rank,
                             reinterpret_cast<long long int*>(n),
                             reinterpret_cast<long long int*>(inembed),
                             static_cast<long long int>(istride),
                             static_cast<long long int>(idist),
                             reinterpret_cast<long long int*>(onembed),
                             static_cast<long long int>(ostride),
                             static_cast<long long int>(odist),
                             type,
                             static_cast<long long int>(batch),
                             workSize);
}
cufftResult CUFFTAPI cufft_MakePlanMany(cufftHandle plan,
                                        int rank,
                                        int* n,
                                        int* inembed,
                                        int istride,
                                        int idist,
                                        int* onembed,
                                        int ostride,
                                        int odist,
                                        cufftType type,
                                        int batch,
                                        size_t* workSize)
{
  return cufftMakePlanMany(
    plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize);
}

template <typename value_t, typename value_idx>
std::pair<value_t, value_t> min_max(const value_t* Y, const value_idx n, cudaStream_t stream)
{
  value_t min_h, max_h;

  rmm::device_scalar<value_t> min_d(stream);
  rmm::device_scalar<value_t> max_d(stream);

  // Each copy needs its own host source, alive and unmodified until the stream
  // is synchronized below, because rmm may defer reading it until the stream
  // reaches the copy.
  value_t const min_init = std::numeric_limits<value_t>::max();
  value_t const max_init = std::numeric_limits<value_t>::lowest();
  min_d.set_value_async(min_init, stream);
  max_d.set_value_async(max_init, stream);

  auto nthreads = 256;
  auto nblocks  = raft::ceildiv(n, (value_idx)nthreads);

  min_max_kernel<<<nblocks, nthreads, 0, stream>>>(Y, n, min_d.data(), max_d.data(), true);

  min_h = min_d.value(stream);
  max_h = max_d.value(stream);

  raft::interruptible::synchronize(stream);

  return std::make_pair(std::move(min_h), std::move(max_h));
}

/**
 * @brief Fast Dimensionality reduction via TSNE using the fast Fourier transform interpolation
 * approximation.
 * @param[in] VAL: The values in the attractive forces COO matrix.
 * @param[in] COL: The column indices in the attractive forces COO matrix.
 * @param[in] ROW: The row indices in the attractive forces COO matrix.
 * @param[in] NNZ: The number of non zeros in the attractive forces COO matrix.
 * @param[in] handle: The GPU handle.
 * @param[out] Y: The final embedding (col-major).
 * @param[in] n: Number of rows in data X.
 * @param[in] params: Parameters for TSNE model.
 */
template <typename value_idx, typename value_t>
std::pair<float, int> FFT_TSNE(value_t* VAL,
                               const value_idx* COL,
                               const value_idx* ROW,
                               const value_idx NNZ,
                               const raft::handle_t& handle,
                               value_t* Y,
                               const value_idx n,
                               const TSNEParams& params)
{
  auto stream        = handle.get_stream().get();
  auto thrust_policy = handle.get_thrust_policy();
  // Fixed seeds use deterministic accumulation paths; unseeded runs keep the
  // original faster atomic/reduction paths.
  const bool deterministic = params.random_state >= 0;

  // Get device properties
  //---------------------------------------------------
  const int mp_count          = raft::getMultiProcessorCount();
  const int dev_major_version = MLCommon::getDeviceCapability().first;
  // These came from the CannyLab implementation, but I don't know how they were
  // determined. TODO check/optimize.
  const int integration_kernel_factor = dev_major_version >= 6   ? 2
                                        : dev_major_version == 5 ? 1
                                        : dev_major_version == 3 ? 2
                                                                 : 3;

  constexpr value_idx n_interpolation_points = 3;
  constexpr value_idx min_num_intervals      = 50;
  // The number of "charges" or s+2 sums i.e. number of kernel sums
  constexpr value_idx n_terms = 4;
  constexpr value_idx n_interpolation_coefficients_per_chunk =
    n_interpolation_points * n_interpolation_points * n_terms;
  value_idx n_boxes_per_dim = min_num_intervals;

  // FFTW is faster on numbers that can be written as 2^a 3^b 5^c 7^d 11^e 13^f
  // where e+f is either 0 or 1, and the other exponents are arbitrary
  int allowed_n_boxes_per_dim[20] = {25, 36, 50,  55,  60,  65,  70,  75,  80,  85,
                                     90, 96, 100, 110, 120, 130, 140, 150, 175, 200};
  if (n_boxes_per_dim < allowed_n_boxes_per_dim[19]) {
    // Round up to nearest grid point
    value_idx chosen_i = 0;
    while (allowed_n_boxes_per_dim[chosen_i] < n_boxes_per_dim)
      chosen_i++;
    n_boxes_per_dim = allowed_n_boxes_per_dim[chosen_i];
  }

  value_idx n_total_boxes = n_boxes_per_dim * n_boxes_per_dim;
  value_idx total_interpolation_points =
    n_total_boxes * n_interpolation_points * n_interpolation_points;
  value_idx n_fft_coeffs_half         = n_interpolation_points * n_boxes_per_dim;
  value_idx n_fft_coeffs              = 2 * n_interpolation_points * n_boxes_per_dim;
  value_idx n_interpolation_points_1d = n_interpolation_points * n_boxes_per_dim;

#define DB(type, name, size) rmm::device_uvector<type> name(size, stream)

  DB(value_t, repulsive_forces_device, n * 2);
  raft::linalg::zero(repulsive_forces_device.data(), repulsive_forces_device.size(), stream);
  DB(value_t, attractive_forces_device, n * 2);
  raft::linalg::zero(attractive_forces_device.data(), attractive_forces_device.size(), stream);
  DB(value_t, gains_device, n * 2);
  auto gains_device_thrust = thrust::device_pointer_cast(gains_device.data());
  thrust::fill(thrust_policy, gains_device_thrust, gains_device_thrust + (n * 2), 1.0f);
  DB(value_t, old_forces_device, n * 2);
  raft::linalg::zero(old_forces_device.data(), old_forces_device.size(), stream);
  DB(value_t, normalization_vec_device, n);
  raft::linalg::zero(normalization_vec_device.data(), normalization_vec_device.size(), stream);
  DB(value_idx, point_box_idx_device, n);
  DB(value_t, x_in_box_device, n);
  raft::linalg::zero(x_in_box_device.data(), x_in_box_device.size(), stream);
  DB(value_t, y_in_box_device, n);
  raft::linalg::zero(y_in_box_device.data(), y_in_box_device.size(), stream);
  DB(value_t, y_tilde_values, total_interpolation_points * n_terms);
  raft::linalg::zero(y_tilde_values.data(), y_tilde_values.size(), stream);
  DB(value_t, x_interpolated_values_device, n * n_interpolation_points);
  raft::linalg::zero(
    x_interpolated_values_device.data(), x_interpolated_values_device.size(), stream);
  DB(value_t, y_interpolated_values_device, n * n_interpolation_points);
  raft::linalg::zero(
    y_interpolated_values_device.data(), y_interpolated_values_device.size(), stream);
  const value_idx n_interpolation_segments = total_interpolation_points * n_terms;
  DB(uint64_t, interpolation_point_sort_keys, deterministic ? n : 0);
  DB(value_idx, sorted_interpolation_point_indices, deterministic ? n : 0);
  DB(uint64_t, interpolation_box_keys, deterministic ? n_total_boxes + 1 : 0);
  DB(value_idx, interpolation_box_offsets, deterministic ? n_total_boxes + 1 : 0);
  DB(value_idx, interpolation_chunk_counts, deterministic ? n_total_boxes + 1 : 0);
  DB(value_idx, interpolation_chunk_offsets, deterministic ? n_total_boxes + 1 : 0);
  DB(value_idx, interpolation_chunk_box_indices, deterministic ? n : 0);
  DB(value_t,
     interpolation_chunk_partials,
     deterministic ? n * n_interpolation_coefficients_per_chunk : 0);
  DB(value_t, potentialsQij_device, n * n_terms);
  raft::linalg::zero(potentialsQij_device.data(), potentialsQij_device.size(), stream);
  DB(value_t, w_coefficients_device, total_interpolation_points * n_terms);
  raft::linalg::zero(w_coefficients_device.data(), w_coefficients_device.size(), stream);
  DB(value_t, chargesQij_device, n * n_terms);
  raft::linalg::zero(chargesQij_device.data(), chargesQij_device.size(), stream);
  DB(value_idx, attractive_row_ids, deterministic ? n + 1 : 0);
  DB(value_idx, attractive_row_offsets, deterministic ? n + 1 : 0);
  DB(value_t, box_lower_bounds_device, 2 * n_total_boxes);
  raft::linalg::zero(box_lower_bounds_device.data(), box_lower_bounds_device.size(), stream);
  DB(value_t, kernel_tilde_device, n_fft_coeffs * n_fft_coeffs);
  raft::linalg::zero(kernel_tilde_device.data(), kernel_tilde_device.size(), stream);
  DB(cufftComplex,
     fft_kernel_tilde_device,
     2 * n_interpolation_points_1d * 2 * n_interpolation_points_1d);
  DB(value_t, fft_input, n_terms * n_fft_coeffs * n_fft_coeffs);
  raft::linalg::zero(fft_input.data(), fft_input.size(), stream);
  DB(cufftComplex, fft_w_coefficients, n_terms * n_fft_coeffs * (n_fft_coeffs / 2 + 1));
  DB(value_t, fft_output, n_terms * n_fft_coeffs * n_fft_coeffs);
  raft::linalg::zero(fft_output.data(), fft_output.size(), stream);

  value_t h = 1.0f / n_interpolation_points;
  value_t y_tilde_spacings[n_interpolation_points];
  y_tilde_spacings[0] = h / 2;
  for (value_idx i = 1; i < n_interpolation_points; i++) {
    y_tilde_spacings[i] = y_tilde_spacings[i - 1] + h;
  }
  value_t denominator[n_interpolation_points];
  for (value_idx i = 0; i < n_interpolation_points; i++) {
    denominator[i] = 1;
    for (value_idx j = 0; j < n_interpolation_points; j++) {
      if (i != j) { denominator[i] *= y_tilde_spacings[i] - y_tilde_spacings[j]; }
    }
  }

  DB(value_t, y_tilde_spacings_device, n_interpolation_points);
  RAFT_CUDA_TRY(cudaMemcpyAsync(y_tilde_spacings_device.data(),
                                y_tilde_spacings,
                                n_interpolation_points * sizeof(value_t),
                                cudaMemcpyHostToDevice,
                                stream));
  DB(value_t, denominator_device, n_interpolation_points);
  RAFT_CUDA_TRY(cudaMemcpyAsync(denominator_device.data(),
                                denominator,
                                n_interpolation_points * sizeof(value_t),
                                cudaMemcpyHostToDevice,
                                stream));

  if (deterministic) {
    // Precompute COO row ranges once so attractive forces can be accumulated
    // row-by-row in a fixed order each iteration. lower_bound is deterministic
    // because ROW has already been sorted by the fixed-seed COO canonicalization.
    thrust::sequence(
      thrust_policy, attractive_row_ids.begin(), attractive_row_ids.end(), value_idx{0});
    thrust::lower_bound(thrust_policy,
                        ROW,
                        ROW + NNZ,
                        attractive_row_ids.begin(),
                        attractive_row_ids.end(),
                        attractive_row_offsets.begin());
  }
  if (deterministic) {
    // Build one search key per interpolation box. lower_bound uses these keys
    // after the point sort below to recover each box's [begin, end) span without
    // atomics or scheduler-dependent grouping.
    auto box_indices = thrust::make_counting_iterator<value_idx>(0);
    thrust::transform(thrust_policy,
                      box_indices,
                      box_indices + n_total_boxes + 1,
                      interpolation_box_keys.begin(),
                      [n] __device__(value_idx box) {
                        return static_cast<uint64_t>(box) * static_cast<uint64_t>(n);
                      });
  }
#undef DB

  cufftHandle plan_kernel_tilde;
  cufftHandle plan_dft;
  cufftHandle plan_idft;

  CUFFT_TRY(cufftCreate(&plan_kernel_tilde));
  CUFFT_TRY(cufftSetStream(plan_kernel_tilde, stream));
  CUFFT_TRY(cufftCreate(&plan_dft));
  CUFFT_TRY(cufftSetStream(plan_dft, stream));
  CUFFT_TRY(cufftCreate(&plan_idft));
  CUFFT_TRY(cufftSetStream(plan_idft, stream));

  size_t work_size, work_size_dft, work_size_idft;
  value_idx fft_dimensions[2] = {n_fft_coeffs, n_fft_coeffs};
  CUFFT_TRY(cufftMakePlan2d(
    plan_kernel_tilde, fft_dimensions[0], fft_dimensions[1], CUFFT_R2C, &work_size));
  CUFFT_TRY(cufft_MakePlanMany(plan_dft,
                               2,
                               fft_dimensions,
                               NULL,
                               1,
                               n_fft_coeffs * n_fft_coeffs,
                               NULL,
                               1,
                               n_fft_coeffs * (n_fft_coeffs / 2 + 1),
                               CUFFT_R2C,
                               n_terms,
                               &work_size_dft));
  CUFFT_TRY(cufft_MakePlanMany(plan_idft,
                               2,
                               fft_dimensions,
                               NULL,
                               1,
                               n_fft_coeffs * (n_fft_coeffs / 2 + 1),
                               NULL,
                               1,
                               n_fft_coeffs * n_fft_coeffs,
                               CUFFT_C2R,
                               n_terms,
                               &work_size_idft));

  value_t momentum                             = params.pre_momentum;
  value_t learning_rate                        = params.pre_learning_rate;
  value_t exaggeration                         = params.early_exaggeration;
  constexpr value_idx interpolation_chunk_size = 256;

  value_t kl_div = 0;
  int iter       = 0;
  for (; iter < params.max_iter; iter++) {
    // Compute charges Q_ij
    {
      int num_blocks = raft::ceildiv(n, (value_idx)NTHREADS_1024);
      FFT::compute_chargesQij<<<num_blocks, NTHREADS_1024, 0, stream>>>(
        chargesQij_device.data(), Y, Y + n, n, n_terms);
    }

    if (iter == params.exaggeration_iter) {
      momentum      = params.post_momentum;
      learning_rate = params.post_learning_rate;
      exaggeration  = params.late_exaggeration;
    }

    raft::linalg::zero(w_coefficients_device.data(), w_coefficients_device.size(), stream);
    raft::linalg::zero(potentialsQij_device.data(), potentialsQij_device.size(), stream);
    // IntegrationKernel zeroes this, but if this is removed
    // then FITSNE runs in an indefinite loop
    raft::linalg::zero(attractive_forces_device.data(), attractive_forces_device.size(), stream);

    auto minmax_pair = min_max(Y, n * 2, stream);
    auto min_coord   = minmax_pair.first;
    auto max_coord   = minmax_pair.second;

    value_t box_width = (max_coord - min_coord) / static_cast<value_t>(n_boxes_per_dim);
    //// Precompute FFT

    // Left and right bounds of each box, first the lower bounds in the x
    // direction, then in the y direction
    {
      auto num_blocks = raft::ceildiv(n_total_boxes, (value_idx)NTHREADS_32);
      FFT::compute_bounds<<<num_blocks, NTHREADS_32, 0, stream>>>(box_lower_bounds_device.data(),
                                                                  box_width,
                                                                  min_coord,
                                                                  min_coord,
                                                                  n_boxes_per_dim,
                                                                  n_total_boxes);
    }

    {
      // Evaluate the kernel at the interpolation nodes and form the embedded
      // generating kernel vector for a circulant matrix.
      // Coordinates of all the equispaced interpolation points
      value_t h       = box_width / n_interpolation_points;
      auto num_blocks = raft::ceildiv(n_interpolation_points_1d * n_interpolation_points_1d,
                                      (value_idx)NTHREADS_32);
      FFT::compute_kernel_tilde<<<num_blocks, NTHREADS_32, 0, stream>>>(kernel_tilde_device.data(),
                                                                        min_coord,
                                                                        min_coord,
                                                                        h,
                                                                        n_interpolation_points_1d,
                                                                        n_fft_coeffs);
    }

    {
      // Precompute the FFT of the kernel generating matrix
      CUFFT_TRY(cufftExecR2C(
        plan_kernel_tilde, kernel_tilde_device.data(), fft_kernel_tilde_device.data()));
    }

    {
      //// Run N-body FFT
      auto num_blocks = raft::ceildiv(n, (value_idx)NTHREADS_128);
      FFT::compute_point_box_idx<<<num_blocks, NTHREADS_128, 0, stream>>>(
        point_box_idx_device.data(),
        x_in_box_device.data(),
        y_in_box_device.data(),
        Y,
        Y + n,
        box_lower_bounds_device.data(),
        min_coord,
        box_width,
        n_boxes_per_dim,
        n_total_boxes,
        n);

      // Step 1: Interpolate kernel using Lagrange polynomials and compute the w
      // coefficients.

      // Compute the interpolated values at each real point with each Lagrange
      // polynomial in the `x` direction
      num_blocks = raft::ceildiv(n * n_interpolation_points, (value_idx)NTHREADS_128);
      FFT::interpolate_device<<<num_blocks, NTHREADS_128, 0, stream>>>(
        x_interpolated_values_device.data(),
        x_in_box_device.data(),
        y_tilde_spacings_device.data(),
        denominator_device.data(),
        n_interpolation_points,
        n);

      // ...and in the `y` direction
      FFT::interpolate_device<<<num_blocks, NTHREADS_128, 0, stream>>>(
        y_interpolated_values_device.data(),
        y_in_box_device.data(),
        y_tilde_spacings_device.data(),
        denominator_device.data(),
        n_interpolation_points,
        n);

      if (deterministic) {
        // Sort points by box with point id as a tie-breaker. Each box is then
        // reduced in deterministic chunk order instead of using atomic adds, so
        // fixed-seed runs use the same coefficient accumulation order.
        auto point_indices = thrust::make_counting_iterator<value_idx>(0);
        thrust::transform(thrust_policy,
                          point_indices,
                          point_indices + n,
                          interpolation_point_sort_keys.begin(),
                          [point_box_idx = point_box_idx_device.data(), n] __device__(value_idx i) {
                            return static_cast<uint64_t>(point_box_idx[i]) *
                                     static_cast<uint64_t>(n) +
                                   static_cast<uint64_t>(i);
                          });
        thrust::sequence(thrust_policy,
                         sorted_interpolation_point_indices.begin(),
                         sorted_interpolation_point_indices.end(),
                         value_idx{0});
        thrust::sort_by_key(thrust_policy,
                            interpolation_point_sort_keys.begin(),
                            interpolation_point_sort_keys.end(),
                            sorted_interpolation_point_indices.begin());
        thrust::lower_bound(thrust_policy,
                            interpolation_point_sort_keys.begin(),
                            interpolation_point_sort_keys.end(),
                            interpolation_box_keys.begin(),
                            interpolation_box_keys.end(),
                            interpolation_box_offsets.begin());
        const auto n_boxes = n_total_boxes;
        auto box_indices   = thrust::make_counting_iterator<value_idx>(0);
        // Count how many fixed-size chunks each box needs. The exclusive scan
        // maps boxes to a compact active-chunk list while preserving box order.
        thrust::transform(thrust_policy,
                          box_indices,
                          box_indices + n_boxes + 1,
                          interpolation_chunk_counts.begin(),
                          [box_offsets = interpolation_box_offsets.data(),
                           n_boxes,
                           interpolation_chunk_size] __device__(value_idx box) {
                            if (box == n_boxes) { return value_idx{0}; }
                            const value_idx n_points = box_offsets[box + 1] - box_offsets[box];
                            return (n_points + interpolation_chunk_size - 1) /
                                   interpolation_chunk_size;
                          });
        thrust::exclusive_scan(thrust_policy,
                               interpolation_chunk_counts.begin(),
                               interpolation_chunk_counts.end(),
                               interpolation_chunk_offsets.begin());
        value_idx n_active_chunks = 0;
        raft::copy(&n_active_chunks, interpolation_chunk_offsets.data() + n_boxes, 1, stream);
        raft::resource::sync_stream(handle);

        // Expand the compact chunk offsets into chunk -> box ids. This lets the
        // partial-sum kernel run only active chunks while still deriving each
        // chunk's deterministic point range from the sorted box offsets.
        thrust::for_each(
          thrust_policy,
          box_indices,
          box_indices + n_boxes,
          [chunk_box_indices = interpolation_chunk_box_indices.data(),
           chunk_counts      = interpolation_chunk_counts.data(),
           chunk_offsets     = interpolation_chunk_offsets.data()] __device__(value_idx box) {
            const value_idx start = chunk_offsets[box];
            const value_idx count = chunk_counts[box];
            for (value_idx chunk = 0; chunk < count; ++chunk) {
              chunk_box_indices[start + chunk] = box;
            }
          });

        auto chunk_blocks = raft::ceildiv(n_active_chunks * n_interpolation_coefficients_per_chunk,
                                          (value_idx)NTHREADS_128);
        // First pass: compute fixed-order partial sums inside each box chunk.
        FFT::compute_interpolated_chunk_partials_3_4<<<chunk_blocks, NTHREADS_128, 0, stream>>>(
          interpolation_chunk_partials.data(),
          interpolation_chunk_box_indices.data(),
          interpolation_chunk_offsets.data(),
          interpolation_box_offsets.data(),
          sorted_interpolation_point_indices.data(),
          chargesQij_device.data(),
          x_interpolated_values_device.data(),
          y_interpolated_values_device.data(),
          n,
          interpolation_chunk_size,
          n_active_chunks);

        num_blocks = raft::ceildiv(n_interpolation_segments, (value_idx)NTHREADS_128);
        // Second pass: reduce those chunk partials by increasing chunk id to
        // produce the final interpolation coefficients deterministically.
        FFT::reduce_interpolated_chunk_partials_3_4<<<num_blocks, NTHREADS_128, 0, stream>>>(
          w_coefficients_device.data(),
          interpolation_chunk_partials.data(),
          interpolation_chunk_offsets.data(),
          interpolation_chunk_counts.data(),
          n_boxes_per_dim,
          n_interpolation_segments);
      } else {
        num_blocks = raft::ceildiv(n_terms * n_interpolation_points * n_interpolation_points * n,
                                   (value_idx)NTHREADS_128);
        FFT::compute_interpolated_indices<<<num_blocks, NTHREADS_128, 0, stream>>>(
          w_coefficients_device.data(),
          point_box_idx_device.data(),
          chargesQij_device.data(),
          x_interpolated_values_device.data(),
          y_interpolated_values_device.data(),
          n,
          n_interpolation_points,
          n_boxes_per_dim,
          n_terms);
      }
      // Step 2: Compute the values v_{m, n} at the equispaced nodes, multiply
      // the kernel matrix with the coefficients w
      num_blocks =
        raft::ceildiv(n_terms * n_fft_coeffs_half * n_fft_coeffs_half, (value_idx)NTHREADS_128);
      FFT::copy_to_fft_input<<<num_blocks, NTHREADS_128, 0, stream>>>(
        fft_input.data(), w_coefficients_device.data(), n_fft_coeffs, n_fft_coeffs_half, n_terms);

      // Compute fft values at interpolated nodes
      CUFFT_TRY(cufftExecR2C(plan_dft, fft_input.data(), fft_w_coefficients.data()));

      // Take the broadcasted Hadamard product of a complex matrix and a complex
      // vector.
      {
        const value_idx nn = n_fft_coeffs * (n_fft_coeffs / 2 + 1);
        auto num_blocks    = raft::ceildiv(nn * n_terms, (value_idx)NTHREADS_32);
        FFT::broadcast_column_vector<<<num_blocks, NTHREADS_32, 0, stream>>>(
          fft_w_coefficients.data(), fft_kernel_tilde_device.data(), nn, n_terms);
      }

      // Invert the computed values at the interpolated nodes.
      CUFFT_TRY(cufftExecC2R(plan_idft, fft_w_coefficients.data(), fft_output.data()));

      FFT::copy_from_fft_output<<<num_blocks, NTHREADS_128, 0, stream>>>(
        y_tilde_values.data(), fft_output.data(), n_fft_coeffs, n_fft_coeffs_half, n_terms);

      // Step 3: Compute the potentials \tilde{\phi}
      if (deterministic) {
        // Seeded runs avoid atomicAdd here: each thread owns one
        // (point, charge-term) output and accumulates its 3x3 stencil directly.
        num_blocks = raft::ceildiv(n_terms * n, (value_idx)NTHREADS_128);
        FFT::compute_potential_indices_deterministic<value_idx,
                                                     value_t,
                                                     n_terms,
                                                     n_interpolation_points>
          <<<num_blocks, NTHREADS_128, 0, stream>>>(potentialsQij_device.data(),
                                                    point_box_idx_device.data(),
                                                    y_tilde_values.data(),
                                                    x_interpolated_values_device.data(),
                                                    y_interpolated_values_device.data(),
                                                    n,
                                                    n_boxes_per_dim);
      } else {
        num_blocks = raft::ceildiv(n_terms * n_interpolation_points * n_interpolation_points * n,
                                   (value_idx)NTHREADS_128);
        FFT::compute_potential_indices<value_idx, value_t, n_terms, n_interpolation_points>
          <<<num_blocks, NTHREADS_128, 0, stream>>>(potentialsQij_device.data(),
                                                    point_box_idx_device.data(),
                                                    y_tilde_values.data(),
                                                    x_interpolated_values_device.data(),
                                                    y_interpolated_values_device.data(),
                                                    n,
                                                    n_boxes_per_dim);
      }
    }

    value_t normalization;
    {
      // Compute repulsive forces
      // Make the negative term, or F_rep in the equation 3 of the paper.
      auto num_blocks = raft::ceildiv(n, (value_idx)NTHREADS_1024);
      FFT::compute_repulsive_forces_kernel<<<num_blocks, NTHREADS_1024, 0, stream>>>(
        repulsive_forces_device.data(),
        normalization_vec_device.data(),
        Y,
        Y + n,
        potentialsQij_device.data(),
        n,
        n_terms);
      auto norm_vec_thrust = thrust::device_pointer_cast(normalization_vec_device.data());

      value_t sumQ  = thrust::reduce(thrust_policy,
                                    norm_vec_thrust,
                                    norm_vec_thrust + normalization_vec_device.size(),
                                    0.0f,
                                    cuda::std::plus<value_t>());
      normalization = sumQ - n;
    }

    // Compute attractive forces
    {
      auto num_blocks = raft::ceildiv(NNZ, (value_idx)NTHREADS_1024);
      const float dof = fmaxf(params.dim - 1, 1);  // degree of freedom
      if (iter == params.max_iter - 1) {           // last iteration
        rmm::device_uvector<value_t> tmp(NNZ, stream);
        value_t* Qs      = tmp.data();
        value_t* KL_divs = tmp.data();

        if (deterministic) {
          // Seeded runs accumulate each COO row with one owning thread instead
          // of atomically adding edge contributions into attr_forces.
          num_blocks = raft::ceildiv(n, (value_idx)NTHREADS_128);
          FFT::compute_Pij_x_Qij_deterministic_rows<<<num_blocks, NTHREADS_128, 0, stream>>>(
            attractive_forces_device.data(),
            Qs,
            VAL,
            COL,
            attractive_row_offsets.data(),
            Y,
            n,
            dof);
        } else {
          FFT::compute_Pij_x_Qij_kernel<<<num_blocks, NTHREADS_1024, 0, stream>>>(
            attractive_forces_device.data(), Qs, VAL, ROW, COL, Y, n, NNZ, dof);
        }
        kl_div = compute_kl_div(VAL, Qs, KL_divs, NNZ, stream);
      } else {
        if (deterministic) {
          // Same deterministic row-wise attractive-force accumulation as the
          // final iteration, but without storing Q values for KL divergence.
          num_blocks = raft::ceildiv(n, (value_idx)NTHREADS_128);
          FFT::compute_Pij_x_Qij_deterministic_rows<<<num_blocks, NTHREADS_128, 0, stream>>>(
            attractive_forces_device.data(),
            (value_t*)nullptr,
            VAL,
            COL,
            attractive_row_offsets.data(),
            Y,
            n,
            dof);
        } else {
          FFT::compute_Pij_x_Qij_kernel<<<num_blocks, NTHREADS_1024, 0, stream>>>(
            attractive_forces_device.data(), (value_t*)nullptr, VAL, ROW, COL, Y, n, NNZ, dof);
        }
      }
    }

    // Apply Forces
    {
      auto num_blocks = mp_count * integration_kernel_factor;

      FFT::IntegrationKernel<<<num_blocks, NTHREADS_1024, 0, stream>>>(
        Y,
        attractive_forces_device.data(),
        repulsive_forces_device.data(),
        gains_device.data(),
        old_forces_device.data(),
        learning_rate,
        normalization,
        momentum,
        exaggeration,
        n);
    }

    auto att_forces_thrust = thrust::device_pointer_cast(attractive_forces_device.data());
    auto old_forces_thrust = thrust::device_pointer_cast(old_forces_device.data());

    thrust::transform(thrust_policy,
                      old_forces_thrust,
                      old_forces_thrust + n,
                      att_forces_thrust,
                      FunctionalSquare());

    thrust::transform(thrust_policy,
                      att_forces_thrust,
                      att_forces_thrust + n,
                      att_forces_thrust + n,
                      att_forces_thrust,
                      cuda::std::plus<value_t>());

    thrust::transform(thrust_policy,
                      att_forces_thrust,
                      att_forces_thrust + attractive_forces_device.size(),
                      att_forces_thrust,
                      FunctionalSqrt());

    value_t grad_norm = thrust::reduce(thrust_policy,
                                       att_forces_thrust,
                                       att_forces_thrust + attractive_forces_device.size(),
                                       0.0f,
                                       cuda::std::plus<value_t>()) /
                        attractive_forces_device.size();
    if (grad_norm <= params.min_grad_norm) {
      CUML_LOG_DEBUG("Breaking early as `min_grad_norm` was satisfied, after %d iterations", iter);
      break;
    }
  }

  CUFFT_TRY(cufftDestroy(plan_kernel_tilde));
  CUFFT_TRY(cufftDestroy(plan_dft));
  CUFFT_TRY(cufftDestroy(plan_idft));
  return std::make_pair(kl_div, iter);
}

}  // namespace TSNE
}  // namespace ML
