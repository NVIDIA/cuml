/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/datasets/make_regression.hpp>

#include <raft/core/handle.hpp>
#include <raft/random/make_regression.cuh>

namespace ML {
namespace Datasets {

template <typename DataT, typename IdxT>
void make_regression_helper(const raft::handle_t& handle,
                            DataT* out,
                            DataT* values,
                            IdxT n_rows,
                            IdxT n_cols,
                            IdxT n_informative,
                            DataT* coef,
                            IdxT n_targets,
                            DataT bias,
                            IdxT effective_rank,
                            DataT tail_strength,
                            DataT noise,
                            bool shuffle,
                            uint64_t seed)
{
  const auto& handle_impl            = handle;
  cudaStream_t stream                = handle_impl.get_stream().get();
  cublasHandle_t cublas_handle       = handle_impl.get_cublas_handle();
  cusolverDnHandle_t cusolver_handle = handle_impl.get_cusolver_dn_handle();

  raft::random::make_regression(handle,
                                out,
                                values,
                                n_rows,
                                n_cols,
                                n_informative,
                                stream,
                                coef,
                                n_targets,
                                bias,
                                effective_rank,
                                tail_strength,
                                noise,
                                shuffle,
                                seed);
}

void make_regression(const raft::handle_t& handle,
                     float* out,
                     float* values,
                     int64_t n_rows,
                     int64_t n_cols,
                     int64_t n_informative,
                     float* coef,
                     int64_t n_targets,
                     float bias,
                     int64_t effective_rank,
                     float tail_strength,
                     float noise,
                     bool shuffle,
                     uint64_t seed)
{
  make_regression_helper(handle,
                         out,
                         values,
                         n_rows,
                         n_cols,
                         n_informative,
                         coef,
                         n_targets,
                         bias,
                         effective_rank,
                         tail_strength,
                         noise,
                         shuffle,
                         seed);
}

void make_regression(const raft::handle_t& handle,
                     double* out,
                     double* values,
                     int64_t n_rows,
                     int64_t n_cols,
                     int64_t n_informative,
                     double* coef,
                     int64_t n_targets,
                     double bias,
                     int64_t effective_rank,
                     double tail_strength,
                     double noise,
                     bool shuffle,
                     uint64_t seed)
{
  make_regression_helper(handle,
                         out,
                         values,
                         n_rows,
                         n_cols,
                         n_informative,
                         coef,
                         n_targets,
                         bias,
                         effective_rank,
                         tail_strength,
                         noise,
                         shuffle,
                         seed);
}

void make_regression(const raft::handle_t& handle,
                     float* out,
                     float* values,
                     int n_rows,
                     int n_cols,
                     int n_informative,
                     float* coef,
                     int n_targets,
                     float bias,
                     int effective_rank,
                     float tail_strength,
                     float noise,
                     bool shuffle,
                     uint64_t seed)
{
  make_regression_helper(handle,
                         out,
                         values,
                         n_rows,
                         n_cols,
                         n_informative,
                         coef,
                         n_targets,
                         bias,
                         effective_rank,
                         tail_strength,
                         noise,
                         shuffle,
                         seed);
}

void make_regression(const raft::handle_t& handle,
                     double* out,
                     double* values,
                     int n_rows,
                     int n_cols,
                     int n_informative,
                     double* coef,
                     int n_targets,
                     double bias,
                     int effective_rank,
                     double tail_strength,
                     double noise,
                     bool shuffle,
                     uint64_t seed)
{
  make_regression_helper(handle,
                         out,
                         values,
                         n_rows,
                         n_cols,
                         n_informative,
                         coef,
                         n_targets,
                         bias,
                         effective_rank,
                         tail_strength,
                         noise,
                         shuffle,
                         seed);
}

}  // namespace Datasets
}  // namespace ML
