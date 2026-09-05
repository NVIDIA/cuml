/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/common/utils.hpp>
#include <cuml/explainer/permutation_shap.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

namespace ML {
namespace Explainer {

template <typename DataT, typename IdxT>
CUML_KERNEL void _fused_tile_scatter_pe(DataT* dataset,
                                        const DataT* background,
                                        IdxT nrows_dataset,
                                        IdxT ncols,
                                        const DataT* obs,
                                        IdxT* idx,
                                        IdxT nrows_background,
                                        IdxT sc_size,
                                        bool row_major)
{
  // kernel that actually does the scattering as described in the
  // descriptions of `permutation_dataset` and `shap_main_effect_dataset`
  // parameter sc_size allows us to generate both the permuation_shap_dataset
  // and the main_effect_dataset with the same kernel, since they do the
  // scattering in the same manner, its just the "height" of the columns
  // generated from values that is different.
  IdxT tid = threadIdx.x + blockDim.x * blockIdx.x;

  if (tid < ncols * nrows_dataset) {
    IdxT row, col, start, end;

    if (row_major) {
      row = tid / ncols;

      // we calculate the first row where the entry of dataset will be
      // entered into background depending on its place in the index array
      col   = idx[tid % ncols];
      start = ((tid % ncols) + 1) * nrows_background;

      // each entry of the dataset will be input the same number of times
      // to the matrix, controlled by the sc_size parameter
      end = start + sc_size * nrows_background;

      // now we just need to check if this thread is between start and end
      // if it is then the value should be based on the observation obs
      // otherwise on the background dataset
      if ((start <= row && row < end)) {
        dataset[row * ncols + col] = obs[col];
      } else {
        dataset[row * ncols + col] = background[(row % nrows_background) * ncols + col];
      }

    } else {
      col = tid / nrows_dataset;
      row = tid % nrows_dataset;

      // main difference between row and col major is how do we calculate
      // the end and start and which row corresponds to each thread
      start = nrows_background + idx[col] * nrows_background;

      // calculation of end position is identical
      end = start + sc_size * nrows_background;

      if ((start <= row && row < end)) {
        dataset[tid] = obs[col];
      } else {
        dataset[tid] = background[row + nrows_background * col];
      }
    }
  }
}

template <typename DataT, typename IdxT>
void permutation_shap_dataset_impl(const raft::handle_t& handle,
                                   DataT* dataset,
                                   const DataT* background,
                                   IdxT nrows_background,
                                   IdxT ncols,
                                   const DataT* row,
                                   IdxT* idx,
                                   bool row_major)
{
  const auto& handle_impl = handle;
  cudaStream_t stream     = handle_impl.get_stream().get();

  // we calculate the number of rows in the dataset and then multiply by 2 since
  // we are adding a forward and backward permutation (see docstring in header file)
  IdxT nrows_dataset = (2 * ncols * nrows_background + nrows_background);

  constexpr IdxT nthreads = 512;

  IdxT nblks = (nrows_dataset * ncols + nthreads - 1) / nthreads;

  // each thread calculates a single element
  // for the permutation shap dataset we need the sc_size parameter to be ncols
  _fused_tile_scatter_pe<<<nblks, nthreads, 0, stream>>>(
    dataset, background, nrows_dataset, ncols, row, idx, nrows_background, ncols, row_major);

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

void permutation_shap_dataset(const raft::handle_t& handle,
                              float* dataset,
                              const float* background,
                              int nrows_bg,
                              int ncols,
                              const float* row,
                              int* idx,
                              bool row_major)
{
  permutation_shap_dataset_impl(handle, dataset, background, nrows_bg, ncols, row, idx, row_major);
}

template <typename DataT, typename IdxT>
void shap_main_effect_dataset_impl(const raft::handle_t& handle,
                                   DataT* dataset,
                                   const DataT* background,
                                   IdxT nrows_bg,
                                   IdxT ncols,
                                   const DataT* row,
                                   IdxT* idx,
                                   bool row_major)
{
  const auto& handle_impl = handle;
  cudaStream_t stream     = handle_impl.get_stream().get();

  // we calculate the number of elements in the dataset
  IdxT total_num_elements = (nrows_bg * ncols + nrows_bg) * ncols;

  constexpr IdxT nthreads = 512;

  IdxT nblks = (total_num_elements + nthreads - 1) / nthreads;

  // each thread calculates a single element
  // for the permutation shap dataset we need the sc_size parameter to be 1
  _fused_tile_scatter_pe<<<nblks, nthreads, 0, stream>>>(
    dataset, background, total_num_elements / ncols, ncols, row, idx, nrows_bg, 1, row_major);

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

void shap_main_effect_dataset(const raft::handle_t& handle,
                              float* dataset,
                              const float* background,
                              int nrows_bg,
                              int ncols,
                              const float* row,
                              int* idx,
                              bool row_major)
{
  shap_main_effect_dataset_impl(handle, dataset, background, nrows_bg, ncols, row, idx, row_major);
}

template <typename DataT, typename IdxT>
CUML_KERNEL void update_perm_shap_values_kernel(DataT* output,
                                                const DataT* input,
                                                const IdxT ncols,
                                                const IdxT* idx)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < ncols) {
    DataT result = output[idx[tid]];
    result += input[tid + 1] - input[tid];
    result += input[tid + ncols] - input[tid + ncols + 1];
    output[idx[tid]] = result;
  }
}

template <typename DataT, typename IdxT>
void update_perm_shap_values_impl(const raft::handle_t& handle,
                                  DataT* shap_values,
                                  const DataT* y_hat,
                                  const IdxT ncols,
                                  const IdxT* idx)
{
  const auto& handle_impl = handle;
  cudaStream_t stream     = handle_impl.get_stream().get();

  constexpr IdxT nthreads = 512;

  IdxT nblks = ncols / nthreads + 1;

  update_perm_shap_values_kernel<<<nblks, nthreads, 0, 0>>>(shap_values, y_hat, ncols, idx);

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

void update_perm_shap_values(const raft::handle_t& handle,
                             float* shap_values,
                             const float* y_hat,
                             const int ncols,
                             const int* idx)
{
  update_perm_shap_values_impl(handle, shap_values, y_hat, ncols, idx);
}

}  // namespace Explainer
}  // namespace ML
