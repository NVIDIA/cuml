/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/decomposition/pca.hpp>
#include <cuml/decomposition/pca_mg.hpp>
#include <cuml/prims/opg/matrix/matrix_utils.hpp>
#include <cuml/prims/opg/stats/cov.hpp>
#include <cuml/prims/opg/stats/mean.hpp>
#include <cuml/prims/opg/stats/mean_center.hpp>

#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/matrix_vector.cuh>
#include <raft/linalg/pca.cuh>
#include <raft/linalg/transpose.cuh>
#include <raft/linalg/tsvd.cuh>
#include <raft/matrix/sqrt.cuh>
#include <raft/stats/mean_center.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cstddef>

namespace ML {
namespace PCA {
namespace opg {

template <typename T>
void fit_impl(raft::handle_t& handle,
              std::vector<MLCommon::Matrix::Data<T>*>& input_data,
              MLCommon::Matrix::PartDescriptor& input_desc,
              T* components,
              T* explained_var,
              T* explained_var_ratio,
              T* singular_vals,
              T* mu,
              T* noise_vars,
              paramsPCAMG prms,
              cudaStream_t* streams,
              std::uint32_t n_streams,
              bool verbose)
{
  const auto& comm = handle.get_comms();

  MLCommon::Matrix::Data<T> mu_data{mu, prms.n_cols};

  MLCommon::Stats::opg::mean(handle, mu_data, input_data, input_desc, streams, n_streams);

  rmm::device_uvector<T> cov_data(prms.n_cols * prms.n_cols, streams[0]);
  auto cov_data_size = cov_data.size();
  MLCommon::Matrix::Data<T> cov{cov_data.data(), cov_data_size};

  MLCommon::Stats::opg::cov(handle, cov, input_data, input_desc, mu_data, true, streams, n_streams);

  auto raft_prms = to_raft_params(prms);
  raft::resources handle_stream_zero;
  raft::resource::set_cuda_stream(handle_stream_zero, streams[0]);

  raft::linalg::trunc_comp_exp_vars(
    handle_stream_zero,
    raft_prms,
    raft::make_device_matrix_view<T, std::size_t, raft::col_major>(
      cov.ptr, prms.n_cols, prms.n_cols),
    raft::make_device_matrix_view<T, std::size_t, raft::col_major>(
      components, prms.n_components, prms.n_cols),
    raft::make_device_vector_view<T, std::size_t>(explained_var, prms.n_components),
    raft::make_device_vector_view<T, std::size_t>(explained_var_ratio, prms.n_components),
    raft::make_device_scalar_view<T, std::size_t>(noise_vars),
    prms.n_rows);

  T scalar = (prms.n_rows - 1);
  raft::matrix::weighted_sqrt(handle_stream_zero,
                              raft::make_device_matrix_view<const T, std::size_t, raft::row_major>(
                                explained_var, std::size_t(1), prms.n_components),
                              raft::make_device_matrix_view<T, std::size_t, raft::row_major>(
                                singular_vals, std::size_t(1), prms.n_components),
                              raft::make_host_scalar_view(&scalar),
                              true);

  MLCommon::Stats::opg::mean_add(input_data, input_desc, mu_data, comm, streams, n_streams);

  for (std::uint32_t i = 0; i < n_streams; i++) {
    handle.sync_stream(streams[i]);
  }
  raft::linalg::sign_flip_components(handle_stream_zero,
                                     raft::make_device_matrix_view<T, std::size_t, raft::col_major>(
                                       input_data[0]->ptr, prms.n_rows, prms.n_cols),
                                     raft::make_device_matrix_view<T, std::size_t, raft::col_major>(
                                       components, prms.n_components, prms.n_cols),
                                     true,
                                     false);
}

/**
 * @brief performs MNMG fit operation for the pca
 * @input param handle: the internal cuml handle object
 * @input param rank_sizes: includes all the partition size information for the rank
 * @input param input: input data
 * @input param components: principal components of the input data
 * @output param explained_var: explained var
 * @output param explained_var_ratio: the explained var ratio
 * @output param singular_vals: singular values of the data
 * @output param mu: mean of every column in input
 * @output param noise_vars: variance of the noise
 * @input param prms: data structure that includes all the parameters from input size to algorithm
 * @input param verbose
 */
template <typename T>
void fit_impl(raft::handle_t& handle,
              std::vector<MLCommon::Matrix::Data<T>*>& input_data,
              MLCommon::Matrix::PartDescriptor& input_desc,
              T* components,
              T* explained_var,
              T* explained_var_ratio,
              T* singular_vals,
              T* mu,
              T* noise_vars,
              paramsPCAMG prms,
              bool verbose)
{
  int rank = handle.get_comms().get_rank();

  // TODO: These streams should come from raft::handle_t
  // Reference issue https://github.com/NVIDIA/cuml/issues/2470
  auto n_streams = input_desc.blocksOwnedBy(rank).size();
  cudaStream_t streams[n_streams];
  for (std::uint32_t i = 0; i < n_streams; i++) {
    RAFT_CUDA_TRY(cudaStreamCreate(&streams[i]));
  }

  if (prms.algorithm == mg_solver::COV_EIG_JACOBI || prms.algorithm == mg_solver::COV_EIG_DQ) {
    fit_impl(handle,
             input_data,
             input_desc,
             components,
             explained_var,
             explained_var_ratio,
             singular_vals,
             mu,
             noise_vars,
             prms,
             streams,
             n_streams,
             verbose);
  }

  for (std::uint32_t i = 0; i < n_streams; i++) {
    handle.sync_stream(streams[i]);
  }

  for (std::uint32_t i = 0; i < n_streams; i++) {
    RAFT_CUDA_TRY(cudaStreamDestroy(streams[i]));
  }
}

template <typename T>
void transform_impl(raft::handle_t& handle,
                    std::vector<MLCommon::Matrix::Data<T>*>& input,
                    const MLCommon::Matrix::PartDescriptor input_desc,
                    T* components,
                    std::vector<MLCommon::Matrix::Data<T>*>& trans_input,
                    T* singular_vals,
                    T* mu,
                    const paramsPCAMG prms,
                    cudaStream_t* streams,
                    std::uint32_t n_streams,
                    bool verbose)
{
  std::vector<MLCommon::Matrix::RankSizePair*> local_blocks = input_desc.partsToRanks;

  if (prms.whiten) {
    T scalar = T(sqrt(prms.n_rows - 1));
    raft::linalg::scalarMultiply(
      components, components, scalar, prms.n_cols * prms.n_components, streams[0]);
    raft::resources handle_stream_zero;
    raft::resource::set_cuda_stream(handle_stream_zero, streams[0]);
    raft::linalg::binary_div_skip_zero<raft::Apply::ALONG_ROWS>(
      handle_stream_zero,
      raft::make_device_matrix_view<T, std::size_t, raft::row_major>(
        components, prms.n_cols, prms.n_components),
      raft::make_device_vector_view<const T, std::size_t>(singular_vals, prms.n_components));
  }

  for (std::size_t i = 0; i < input.size(); i++) {
    auto si = i % n_streams;

    raft::stats::meanCenter<false, true>(
      input[i]->ptr, input[i]->ptr, mu, prms.n_cols, local_blocks[i]->size, streams[si]);

    T alpha = T(1);
    T beta  = T(0);
    raft::linalg::gemm(handle,
                       input[i]->ptr,
                       local_blocks[i]->size,
                       prms.n_cols,
                       components,
                       trans_input[i]->ptr,
                       local_blocks[i]->size,
                       prms.n_components,
                       CUBLAS_OP_N,
                       CUBLAS_OP_T,
                       alpha,
                       beta,
                       streams[si]);

    raft::stats::meanAdd<false, true>(
      input[i]->ptr, input[i]->ptr, mu, prms.n_cols, local_blocks[i]->size, streams[si]);
  }

  if (prms.whiten) {
    raft::resources handle_stream_zero;
    raft::resource::set_cuda_stream(handle_stream_zero, streams[0]);
    raft::linalg::binary_mult_skip_zero<raft::Apply::ALONG_ROWS>(
      handle_stream_zero,
      raft::make_device_matrix_view<T, std::size_t, raft::row_major>(
        components, prms.n_cols, prms.n_components),
      raft::make_device_vector_view<const T, std::size_t>(singular_vals, prms.n_components));
    T scalar = T(1 / sqrt(prms.n_rows - 1));
    raft::linalg::scalarMultiply(
      components, components, scalar, prms.n_cols * prms.n_components, streams[0]);
  }

  for (std::uint32_t i = 0; i < n_streams; i++) {
    handle.sync_stream(streams[i]);
  }
}

/**
 * @brief performs MNMG transform operation for the pca.
 * @input param handle: the internal cuml handle object
 * @input param rank_sizes: includes all the partition size information for the rank
 * @input param n_parts: number of partitions
 * @input param input: input data
 * @input param components: principal components of the input data
 * @output param trans_input: transformed input data
 * @input param singular_vals: singular values of the data
 * @input param mu: mean of every column in input
 * @input param prms: data structure that includes all the parameters from input size to algorithm
 * @input param verbose
 */
template <typename T>
void transform_impl(raft::handle_t& handle,
                    MLCommon::Matrix::RankSizePair** rank_sizes,
                    std::uint32_t n_parts,
                    MLCommon::Matrix::Data<T>** input,
                    T* components,
                    MLCommon::Matrix::Data<T>** trans_input,
                    T* singular_vals,
                    T* mu,
                    paramsPCAMG prms,
                    bool verbose)
{
  // We want to update the API of this function, and other functions with
  // regards to https://github.com/NVIDIA/cuml/issues/2471

  int rank = handle.get_comms().get_rank();

  std::vector<MLCommon::Matrix::RankSizePair*> ranksAndSizes(rank_sizes, rank_sizes + n_parts);
  std::vector<MLCommon::Matrix::Data<T>*> input_data(input, input + n_parts);
  MLCommon::Matrix::PartDescriptor input_desc(prms.n_rows, prms.n_cols, ranksAndSizes, rank);
  std::vector<MLCommon::Matrix::Data<T>*> trans_data(trans_input, trans_input + n_parts);

  // TODO: These streams should come from raft::handle_t
  auto n_streams = n_parts;
  cudaStream_t streams[n_streams];
  for (std::uint32_t i = 0; i < n_streams; i++) {
    RAFT_CUDA_TRY(cudaStreamCreate(&streams[i]));
  }

  transform_impl(handle,
                 input_data,
                 input_desc,
                 components,
                 trans_data,
                 singular_vals,
                 mu,
                 prms,
                 streams,
                 n_streams,
                 verbose);

  for (std::uint32_t i = 0; i < n_streams; i++) {
    handle.sync_stream(streams[i]);
  }

  for (std::uint32_t i = 0; i < n_streams; i++) {
    RAFT_CUDA_TRY(cudaStreamDestroy(streams[i]));
  }
}

template <typename T>
void inverse_transform_impl(raft::handle_t& handle,
                            std::vector<MLCommon::Matrix::Data<T>*>& trans_input,
                            MLCommon::Matrix::PartDescriptor trans_input_desc,
                            T* components,
                            std::vector<MLCommon::Matrix::Data<T>*>& input,
                            T* singular_vals,
                            T* mu,
                            paramsPCAMG prms,
                            cudaStream_t* streams,
                            std::uint32_t n_streams,
                            bool verbose)
{
  std::vector<MLCommon::Matrix::RankSizePair*> local_blocks = trans_input_desc.partsToRanks;

  if (prms.whiten) {
    T scalar = T(1 / sqrt(prms.n_rows - 1));
    raft::linalg::scalarMultiply(
      components, components, scalar, prms.n_cols * prms.n_components, streams[0]);
    raft::resources handle_stream_zero;
    raft::resource::set_cuda_stream(handle_stream_zero, streams[0]);
    raft::linalg::binary_mult_skip_zero<raft::Apply::ALONG_ROWS>(
      handle_stream_zero,
      raft::make_device_matrix_view<T, std::size_t, raft::row_major>(
        components, prms.n_cols, prms.n_components),
      raft::make_device_vector_view<const T, std::size_t>(singular_vals, prms.n_components));
  }

  for (std::size_t i = 0; i < local_blocks.size(); i++) {
    auto si = i % n_streams;
    T alpha = T(1);
    T beta  = T(0);

    raft::linalg::gemm(handle,
                       trans_input[i]->ptr,
                       local_blocks[i]->size,
                       prms.n_components,
                       components,
                       input[i]->ptr,
                       local_blocks[i]->size,
                       prms.n_cols,
                       CUBLAS_OP_N,
                       CUBLAS_OP_N,
                       alpha,
                       beta,
                       streams[si]);

    raft::stats::meanAdd<false, true>(
      input[i]->ptr, input[i]->ptr, mu, prms.n_cols, local_blocks[i]->size, streams[si]);
  }

  if (prms.whiten) {
    raft::resources handle_stream_zero;
    raft::resource::set_cuda_stream(handle_stream_zero, streams[0]);
    raft::linalg::binary_div_skip_zero<raft::Apply::ALONG_ROWS>(
      handle_stream_zero,
      raft::make_device_matrix_view<T, std::size_t, raft::row_major>(
        components, prms.n_cols, prms.n_components),
      raft::make_device_vector_view<const T, std::size_t>(singular_vals, prms.n_components));
    T scalar = T(sqrt(prms.n_rows - 1));
    raft::linalg::scalarMultiply(
      components, components, scalar, prms.n_cols * prms.n_components, streams[0]);
  }

  for (std::uint32_t i = 0; i < n_streams; i++) {
    handle.sync_stream(streams[i]);
  }
}

/**
 * @brief performs MNMG inverse transform operation for the pca.
 * @input param handle: the internal cuml handle object
 * @input param rank_sizes: includes all the partition size information for the rank
 * @input param n_parts: number of partitions
 * @input param trans_input: transformed input data
 * @input param components: principal components of the input data
 * @output param input: input data
 * @input param singular_vals: singular values of the data
 * @input param mu: mean of every column in input
 * @input param prms: data structure that includes all the parameters from input size to algorithm
 * @input param verbose
 */
template <typename T>
void inverse_transform_impl(raft::handle_t& handle,
                            MLCommon::Matrix::RankSizePair** rank_sizes,
                            std::uint32_t n_parts,
                            MLCommon::Matrix::Data<T>** trans_input,
                            T* components,
                            MLCommon::Matrix::Data<T>** input,
                            T* singular_vals,
                            T* mu,
                            paramsPCAMG prms,
                            bool verbose)
{
  int rank = handle.get_comms().get_rank();

  std::vector<MLCommon::Matrix::RankSizePair*> ranksAndSizes(rank_sizes, rank_sizes + n_parts);
  MLCommon::Matrix::PartDescriptor trans_desc(prms.n_rows, prms.n_components, ranksAndSizes, rank);
  std::vector<MLCommon::Matrix::Data<T>*> trans_data(trans_input, trans_input + n_parts);

  std::vector<MLCommon::Matrix::Data<T>*> input_data(input, input + n_parts);

  // TODO: These streams should come from raft::handle_t
  auto n_streams = n_parts;
  cudaStream_t streams[n_streams];
  for (std::uint32_t i = 0; i < n_streams; i++) {
    RAFT_CUDA_TRY(cudaStreamCreate(&streams[i]));
  }

  inverse_transform_impl(handle,
                         trans_data,
                         trans_desc,
                         components,
                         input_data,
                         singular_vals,
                         mu,
                         prms,
                         streams,
                         n_streams,
                         verbose);

  for (std::uint32_t i = 0; i < n_streams; i++) {
    handle.sync_stream(streams[i]);
  }

  for (std::uint32_t i = 0; i < n_streams; i++) {
    RAFT_CUDA_TRY(cudaStreamDestroy(streams[i]));
  }
}

/**
 * @brief performs MNMG fit and transform operation for the pca.
 * @input param handle: the internal cuml handle object
 * @input param rank_sizes: includes all the partition size information for the rank
 * @input param n_parts: number of partitions
 * @input param input: input data
 * @output param trans_input: transformed input data
 * @output param components: principal components of the input data
 * @output param explained_var: explained var
 * @output param explained_var_ratio: the explained var ratio
 * @output param singular_vals: singular values of the data
 * @output param mu: mean of every column in input
 * @output param noise_vars: variance of the noise
 * @input param prms: data structure that includes all the parameters from input size to algorithm
 * @input param verbose
 */
template <typename T>
void fit_transform_impl(raft::handle_t& handle,
                        MLCommon::Matrix::RankSizePair** rank_sizes,
                        std::uint32_t n_parts,
                        MLCommon::Matrix::Data<T>** input,
                        MLCommon::Matrix::Data<T>** trans_input,
                        T* components,
                        T* explained_var,
                        T* explained_var_ratio,
                        T* singular_vals,
                        T* mu,
                        T* noise_vars,
                        paramsPCAMG prms,
                        bool verbose)
{
  int rank = handle.get_comms().get_rank();

  std::vector<MLCommon::Matrix::RankSizePair*> ranksAndSizes(rank_sizes, rank_sizes + n_parts);
  std::vector<MLCommon::Matrix::Data<T>*> input_data(input, input + n_parts);
  MLCommon::Matrix::PartDescriptor input_desc(prms.n_rows, prms.n_cols, ranksAndSizes, rank);
  std::vector<MLCommon::Matrix::Data<T>*> trans_data(trans_input, trans_input + n_parts);

  // TODO: These streams should come from raft::handle_t
  auto n_streams = n_parts;
  cudaStream_t streams[n_streams];
  for (std::uint32_t i = 0; i < n_streams; i++) {
    RAFT_CUDA_TRY(cudaStreamCreate(&streams[i]));
  }

  fit_impl(handle,
           input_data,
           input_desc,
           components,
           explained_var,
           explained_var_ratio,
           singular_vals,
           mu,
           noise_vars,
           prms,
           streams,
           n_streams,
           verbose);

  transform_impl(handle,
                 input_data,
                 input_desc,
                 components,
                 trans_data,
                 singular_vals,
                 mu,
                 prms,
                 streams,
                 n_streams,
                 verbose);

  for (std::uint32_t i = 0; i < n_streams; i++) {
    handle.sync_stream(streams[i]);
  }

  for (std::uint32_t i = 0; i < n_streams; i++) {
    RAFT_CUDA_TRY(cudaStreamDestroy(streams[i]));
  }
}

void fit(raft::handle_t& handle,
         std::vector<MLCommon::Matrix::Data<float>*>& input_data,
         MLCommon::Matrix::PartDescriptor& input_desc,
         float* components,
         float* explained_var,
         float* explained_var_ratio,
         float* singular_vals,
         float* mu,
         float* noise_vars,
         paramsPCAMG prms,
         bool verbose)
{
  fit_impl(handle,
           input_data,
           input_desc,
           components,
           explained_var,
           explained_var_ratio,
           singular_vals,
           mu,
           noise_vars,
           prms,
           verbose);
}

void fit(raft::handle_t& handle,
         std::vector<MLCommon::Matrix::Data<double>*>& input_data,
         MLCommon::Matrix::PartDescriptor& input_desc,
         double* components,
         double* explained_var,
         double* explained_var_ratio,
         double* singular_vals,
         double* mu,
         double* noise_vars,
         paramsPCAMG prms,
         bool verbose)
{
  fit_impl(handle,
           input_data,
           input_desc,
           components,
           explained_var,
           explained_var_ratio,
           singular_vals,
           mu,
           noise_vars,
           prms,
           verbose);
}

void fit_transform(raft::handle_t& handle,
                   MLCommon::Matrix::RankSizePair** rank_sizes,
                   std::uint32_t n_parts,
                   MLCommon::Matrix::floatData_t** input,
                   MLCommon::Matrix::floatData_t** trans_input,
                   float* components,
                   float* explained_var,
                   float* explained_var_ratio,
                   float* singular_vals,
                   float* mu,
                   float* noise_vars,
                   paramsPCAMG prms,
                   bool verbose)
{
  fit_transform_impl(handle,
                     rank_sizes,
                     n_parts,
                     input,
                     trans_input,
                     components,
                     explained_var,
                     explained_var_ratio,
                     singular_vals,
                     mu,
                     noise_vars,
                     prms,
                     verbose);
}

void fit_transform(raft::handle_t& handle,
                   MLCommon::Matrix::RankSizePair** rank_sizes,
                   std::uint32_t n_parts,
                   MLCommon::Matrix::doubleData_t** input,
                   MLCommon::Matrix::doubleData_t** trans_input,
                   double* components,
                   double* explained_var,
                   double* explained_var_ratio,
                   double* singular_vals,
                   double* mu,
                   double* noise_vars,
                   paramsPCAMG prms,
                   bool verbose)
{
  fit_transform_impl(handle,
                     rank_sizes,
                     n_parts,
                     input,
                     trans_input,
                     components,
                     explained_var,
                     explained_var_ratio,
                     singular_vals,
                     mu,
                     noise_vars,
                     prms,
                     verbose);
}

void transform(raft::handle_t& handle,
               MLCommon::Matrix::RankSizePair** rank_sizes,
               std::uint32_t n_parts,
               MLCommon::Matrix::Data<float>** input,
               float* components,
               MLCommon::Matrix::Data<float>** trans_input,
               float* singular_vals,
               float* mu,
               paramsPCAMG prms,
               bool verbose)
{
  transform_impl(
    handle, rank_sizes, n_parts, input, components, trans_input, singular_vals, mu, prms, verbose);
}

void transform(raft::handle_t& handle,
               MLCommon::Matrix::RankSizePair** rank_sizes,
               std::uint32_t n_parts,
               MLCommon::Matrix::Data<double>** input,
               double* components,
               MLCommon::Matrix::Data<double>** trans_input,
               double* singular_vals,
               double* mu,
               paramsPCAMG prms,
               bool verbose)
{
  transform_impl(
    handle, rank_sizes, n_parts, input, components, trans_input, singular_vals, mu, prms, verbose);
}

void inverse_transform(raft::handle_t& handle,
                       MLCommon::Matrix::RankSizePair** rank_sizes,
                       std::uint32_t n_parts,
                       MLCommon::Matrix::Data<float>** trans_input,
                       float* components,
                       MLCommon::Matrix::Data<float>** input,
                       float* singular_vals,
                       float* mu,
                       paramsPCAMG prms,
                       bool verbose)
{
  inverse_transform_impl(
    handle, rank_sizes, n_parts, trans_input, components, input, singular_vals, mu, prms, verbose);
}

void inverse_transform(raft::handle_t& handle,
                       MLCommon::Matrix::RankSizePair** rank_sizes,
                       std::uint32_t n_parts,
                       MLCommon::Matrix::Data<double>** trans_input,
                       double* components,
                       MLCommon::Matrix::Data<double>** input,
                       double* singular_vals,
                       double* mu,
                       paramsPCAMG prms,
                       bool verbose)
{
  inverse_transform_impl(
    handle, rank_sizes, n_parts, trans_input, components, input, singular_vals, mu, prms, verbose);
}

}  // namespace opg
}  // namespace PCA
}  // namespace ML
