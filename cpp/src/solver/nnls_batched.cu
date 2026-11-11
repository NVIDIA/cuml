/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/solvers/nnls.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/solver/nnls.cuh>

#include <cstdint>
#include <optional>
#include <type_traits>

namespace ML {
namespace Solver {

namespace {

template <typename T>
void nnls_batched_impl(raft::handle_t& handle,
                       const T* A,
                       int m,
                       int n,
                       const T* B,
                       int P,
                       const std::uint8_t* masks,
                       T* X,
                       T* rnorm,
                       const NnlsBatchedParams& params)
{
  raft::common::nvtx::range fun_scope("ML::Solver::nnlsBatched(%d, %d, %d)", m, n, P);

  using index_t = std::int64_t;
  auto view     = [](auto* ptr, index_t rows, index_t cols) {
    return raft::make_device_matrix_view<std::remove_pointer_t<decltype(ptr)>,
                                             index_t,
                                             raft::col_major>(ptr, rows, cols);
  };

  std::optional<raft::device_matrix_view<const std::uint8_t, index_t, raft::col_major>> masks_view;
  if (masks != nullptr) masks_view = view(masks, n, P);
  std::optional<raft::device_vector_view<T, index_t>> rnorm_view;
  if (rnorm != nullptr) rnorm_view = raft::make_device_vector_view<T, index_t>(rnorm, P);

  raft::solver::nnls<T>(handle,
                        raft::solver::nnls_params{params.tol, params.max_iter},
                        view(A, m, n),
                        view(B, m, P),
                        view(X, n, P),
                        masks_view,
                        rnorm_view);
}

}  // namespace

void nnlsBatched(raft::handle_t& handle,
                 const float* A,
                 int m,
                 int n,
                 const float* B,
                 int n_problems,
                 const std::uint8_t* masks,
                 float* X,
                 float* rnorm,
                 const NnlsBatchedParams& params)
{
  nnls_batched_impl<float>(handle, A, m, n, B, n_problems, masks, X, rnorm, params);
}

void nnlsBatched(raft::handle_t& handle,
                 const double* A,
                 int m,
                 int n,
                 const double* B,
                 int n_problems,
                 const std::uint8_t* masks,
                 double* X,
                 double* rnorm,
                 const NnlsBatchedParams& params)
{
  nnls_batched_impl<double>(handle, A, m, n, B, n_problems, masks, X, rnorm, params);
}

}  // namespace Solver
}  // namespace ML
