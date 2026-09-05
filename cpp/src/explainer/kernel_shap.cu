/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/common/checked_arithmetic.hpp>
#include <cuml/common/utils.hpp>
#include <cuml/explainer/kernel_shap.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <curand.h>
#include <curand_kernel.h>

namespace ML {
namespace Explainer {

/*
* Kernel distributes exact part of the kernel shap dataset
* Each block scatters the data of a row of `observations` into the (number of rows of
* background) in `dataset`, based on the row of `X`.
* So, given:
* background = [[0, 1, 2],
                [3, 4, 5]]
* observation = [100, 101, 102]
* X = [[1, 0, 1],
*      [0, 1, 1]]
*
* dataset (output):
* [[100, 1, 102],
*  [100, 4, 102]
*  [0, 101, 102],
*  [3, 101, 102]]
*
*
*/

template <typename DataT, typename IdxT>
CUML_KERNEL void exact_rows_kernel(float* X,
                                   IdxT nrows_X,
                                   IdxT ncols,
                                   DataT* background,
                                   IdxT nrows_background,
                                   DataT* dataset,
                                   DataT* observation)
{
  // Each block processes one row of X. Columns are iterated over by blockDim.x at a time to ensure
  // data coelescing
  int col = threadIdx.x;
  int row = blockIdx.x * ncols;

  while (col < ncols) {
    // Load the X idx for the current column
    int curr_X = (int)X[row + col];

    // Iterate over nrows_background
    int row_idx_base = blockIdx.x * nrows_background;

    for (int r = 0; r < nrows_background; r++) {
      int row_idx = row_idx_base + r;
      if (curr_X == 0) {
        dataset[row_idx * ncols + col] = background[r * ncols + col];
      } else {
        dataset[row_idx * ncols + col] = observation[col];
      }
    }
    // Increment the column
    col += blockDim.x;
  }
}

/*
* Kernel distributes sampled part of the kernel shap dataset
* The first thread of each block calculates the sampling of `k` entries of `observation`
* to scatter into `dataset`. Afterwards each block scatters the data of a row of `X` into the
(number of rows of
* background) in `dataset`.
* So, given:
* background = [[0, 1, 2, 3],
                [5, 6, 7, 8]]
* observation = [100, 101, 102, 103]
* nsamples = [3, 2]
*
* X (output)
*      [[1, 0, 1, 1],
*       [0, 1, 1, 0]]
*
* dataset (output):
* [[100, 1, 102, 103],
*  [100, 6, 102, 103]
*  [0, 101, 102, 3],
*  [5, 101, 102, 8]]
*
*
*/
template <typename DataT, typename IdxT>
CUML_KERNEL void sampled_rows_kernel(IdxT* nsamples,
                                     float* X,
                                     IdxT nrows_X,
                                     IdxT ncols,
                                     DataT* background,
                                     IdxT nrows_background,
                                     DataT* dataset,
                                     DataT* observation,
                                     uint64_t seed)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // see what k this block will generate
  int k_blk = nsamples[blockIdx.x];

  // First k threads of block generate samples
  if (threadIdx.x < k_blk) {
    curandStatePhilox4_32_10_t state;
    curand_init((unsigned long long)seed, (unsigned long long)tid, 0, &state);
    int rand_idx = (int)(curand_uniform(&state) * ncols);

    // Since X is initialized to 0, we quickly check for collisions (if k_blk << ncols the
    // likelihood of collisions is low)
    while (atomicExch(&(X[2 * blockIdx.x * ncols + rand_idx]), 1) == 1) {
      rand_idx = (int)(curand_uniform(&state) * ncols);
    }
  }
  __syncthreads();

  // Each block processes one row of X. Columns are iterated over by blockDim.x at a time to ensure
  // data coelescing
  int col_idx = threadIdx.x;
  while (col_idx < ncols) {
    // Load the X idx for the current column
    int curr_X                                = (int)X[2 * blockIdx.x * ncols + col_idx];
    X[(2 * blockIdx.x + 1) * ncols + col_idx] = 1 - curr_X;

    int bg_row_idx_base = 2 * blockIdx.x * nrows_background;

    for (int r = 0; r < nrows_background; r++) {
      int bg_row_idx = bg_row_idx_base + r;
      if (curr_X == 0) {
        dataset[bg_row_idx * ncols + col_idx] = background[r * ncols + col_idx];
      } else {
        dataset[bg_row_idx * ncols + col_idx] = observation[col_idx];
      }
    }

    bg_row_idx_base = 2 * (blockIdx.x + 1) * nrows_background;

    for (int r = 0; r < nrows_background; r++) {
      int bg_row_idx = bg_row_idx_base + r;
      if (curr_X == 0) {
        dataset[bg_row_idx * ncols + col_idx] = observation[col_idx];
      } else {
        dataset[bg_row_idx * ncols + col_idx] = background[r * ncols + col_idx];
      }
    }

    col_idx += blockDim.x;
  }
}

template <typename DataT, typename IdxT>
void kernel_dataset_impl(const raft::handle_t& handle,
                         float* X,
                         IdxT nrows_X,
                         IdxT ncols,
                         DataT* background,
                         IdxT nrows_background,
                         DataT* dataset,
                         DataT* observation,
                         int* nsamples,
                         int len_samples,
                         int maxsample,
                         uint64_t seed)
{
  const auto& handle_impl = handle;
  cudaStream_t stream     = handle_impl.get_stream().get();

  IdxT nblks;
  IdxT nthreads;

  nthreads = min(512, ncols);
  nblks    = nrows_X - len_samples;

  if (nblks > 0) {
    exact_rows_kernel<<<nblks, nthreads, 0, stream>>>(
      X, nrows_X, ncols, background, nrows_background, dataset, observation);
  }

  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // check if random part of the dataset is needed
  if (len_samples > 0) {
    if (len_samples > nrows_X) {
      RAFT_FAIL("kernel_dataset: len_samples (%d) must be <= nrows_X (%d)", len_samples, nrows_X);
    }
    // The kernel handles one row per block, but that row also attempts
    // to modify values from the next row. This means that if the number of
    // len_samples is even, we launch 1 extra block that then attempts
    // to modify a row that is out of bounds.
    if (len_samples % 2 == 0) {
      nblks = len_samples / 2 - 1;
    } else {
      nblks = len_samples / 2;
    }
    std::size_t const row_off = checked_sub<std::size_t>(nrows_X, len_samples);
    std::size_t const x_off   = checked_mul<std::size_t>(row_off, ncols);
    std::size_t const ds_off  = checked_mul<std::size_t>(row_off, nrows_background, ncols);
    // each block does a sample and its compliment
    sampled_rows_kernel<<<nblks, nthreads, 0, stream>>>(nsamples,
                                                        &X[x_off],
                                                        len_samples,
                                                        ncols,
                                                        background,
                                                        nrows_background,
                                                        &dataset[ds_off],
                                                        observation,
                                                        seed);
  }

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

void kernel_dataset(const raft::handle_t& handle,
                    float* X,
                    int nrows_X,
                    int ncols,
                    float* background,
                    int nrows_background,
                    float* dataset,
                    float* observation,
                    int* nsamples,
                    int len_nsamples,
                    int maxsample,
                    uint64_t seed)
{
  kernel_dataset_impl(handle,
                      X,
                      nrows_X,
                      ncols,
                      background,
                      nrows_background,
                      dataset,
                      observation,
                      nsamples,
                      len_nsamples,
                      maxsample,
                      seed);
}

}  // namespace Explainer
}  // namespace ML
