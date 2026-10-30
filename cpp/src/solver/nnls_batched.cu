/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nnls_batched.cuh"

#include <cuml/solvers/nnls.hpp>

#include <raft/core/handle.hpp>

namespace ML {
namespace Solver {

void nnlsBatched(raft::handle_t& handle,
                 const float* A,
                 int m,
                 int n,
                 const float* B,
                 int n_problems,
                 const std::uint8_t* masks,
                 float* X,
                 float* fitted,
                 const NnlsBatchedParams& params)
{
  detail::nnls_batched_impl<float>(handle, A, m, n, B, n_problems, masks, X, fitted, params);
}

void nnlsBatched(raft::handle_t& handle,
                 const double* A,
                 int m,
                 int n,
                 const double* B,
                 int n_problems,
                 const std::uint8_t* masks,
                 double* X,
                 double* fitted,
                 const NnlsBatchedParams& params)
{
  detail::nnls_batched_impl<double>(handle, A, m, n, B, n_problems, masks, X, fitted, params);
}

}  // namespace Solver
}  // namespace ML
