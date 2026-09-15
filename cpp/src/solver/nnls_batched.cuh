/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "nnls_lawson.cuh"  // detail::nnls_lawson_batched_dispatch

#include <cuml/common/utils.hpp>
#include <cuml/solvers/nnls.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/linalg/gemm.cuh>

#include <cuda_runtime.h>

#include <cstdint>
#include <optional>

namespace ML {
namespace Solver {
namespace detail {

template <typename T>
void nnls_batched_impl(raft::handle_t& handle,
                       const T* A,
                       int m,
                       int n,
                       const T* B,
                       int P,
                       const std::uint8_t* masks,
                       T* X,
                       T* fitted,
                       const NnlsBatchedParams& params)
{
  raft::common::nvtx::range fun_scope("ML::Solver::nnlsBatched(%d, %d, %d)", m, n, P);
  ASSERT(m >= 1, "ML::Solver::nnlsBatched: m must be >= 1.");
  ASSERT(n >= 1, "ML::Solver::nnlsBatched: n must be >= 1.");
  ASSERT(P >= 1, "ML::Solver::nnlsBatched: n_problems must be >= 1.");

  cudaStream_t stream = handle.get_stream();

  // Precompute the resident Gram matrix and RHS projections once, then reuse
  // them across every problem in the batch: G = A^T A (n x n), C = A^T B (n x P).
  // A col-major (m, n) buffer viewed as row-major (n, m) is exactly A^T, so the
  // transpose is expressed through the operand layout rather than a flag.
  auto G = raft::make_device_matrix<T, int, raft::col_major>(handle, n, n);
  auto C = raft::make_device_matrix<T, int, raft::col_major>(handle, n, P);

  // gemm's mdspan overload shares one ElementType across all operands, so the
  // read-only inputs are wrapped in non-const views (gemm never writes them).
  auto* A_mut = const_cast<T*>(A);
  auto At_view =
    raft::make_device_matrix_view<T, int, raft::row_major>(A_mut, n, m);              // A^T (n x m)
  auto A_view = raft::make_device_matrix_view<T, int, raft::col_major>(A_mut, m, n);  // A   (m x n)
  auto B_view = raft::make_device_matrix_view<T, int, raft::col_major>(const_cast<T*>(B), m, P);
  raft::linalg::gemm(handle, At_view, A_view, G.view());
  raft::linalg::gemm(handle, At_view, B_view, C.view());

  // Solve every problem with the batched Lawson-Hanson kernel.  A max_iter of 0
  // selects the tight active-set cap of 3 * n + 1 outer steps.
  int max_iter = params.max_iter;
  if (max_iter <= 0) max_iter = 3 * n + 1;
  const T tol = static_cast<T>(params.tol);

  auto G_view = raft::make_const_mdspan(G.view());
  auto C_view = raft::make_const_mdspan(C.view());
  auto X_view = raft::make_device_matrix_view<T, int, raft::col_major>(X, n, P);
  std::optional<raft::device_matrix_view<const std::uint8_t, int, raft::col_major>> M_view;
  if (masks != nullptr)
    M_view = raft::make_device_matrix_view<const std::uint8_t, int, raft::col_major>(masks, n, P);
  nnls_lawson_batched_dispatch<T>(handle, G_view, C_view, M_view, X_view, max_iter, tol);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Optional fitted = A @ X  (m x P).
  if (fitted != nullptr) {
    const T one  = T(1);
    const T zero = T(0);
    raft::linalg::gemm(handle,
                       /*trans_a=*/false,
                       /*trans_b=*/false,
                       m,
                       P,
                       n,
                       &one,
                       A,
                       m,
                       X,
                       n,
                       &zero,
                       fitted,
                       m,
                       stream);
  }
}

}  // namespace detail
}  // namespace Solver
}  // namespace ML
