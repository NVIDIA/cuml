/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/common/export.hpp>

#include <cstdint>

namespace raft {
class handle_t;
}

namespace CUML_EXPORT ML {
namespace Solver {

/**
 * @brief Solver backend selector for the batched NNLS entry point.
 *
 * Only the Lawson-Hanson active-set method is currently exposed.  The selector
 * is kept so that additional backends can be added later without changing the
 * call signature.
 */
enum class NnlsBatchedSolver {
  LAWSON = 0  ///< Lawson-Hanson active-set (exact, best for small n_cols).
};

/**
 * Parameters for the batched NNLS solver.
 */
struct NnlsBatchedParams {
  NnlsBatchedSolver solver = NnlsBatchedSolver::LAWSON;
  int max_iter             = 0;     ///< 0 => per-solver default (3 * n_cols + 1 for Lawson).
  double tol               = 1e-6;  ///< Dual-feasibility (KKT) tolerance on the projected gradient.
};

/**
 * Solve a batch of Non-Negative Least Squares problems that share the same
 * coefficient matrix but differ by right-hand side and active-column support:
 *
 *   for p in [0, n_problems):
 *     X[:, p] = argmin_{x >= 0, x[j]=0 for masks[j,p]==0}
 *                 1/2 || A[:, support_p] x[support_p] - B[:, p] ||_2^2
 *
 * The shared matrix A stays resident; its Gram matrix G = A^T A and the RHS
 * projections C = A^T B are formed once (via cuBLAS) and reused by every
 * problem.  masks selects the active support per problem; masked-out
 * coordinates of X are pinned to zero.
 *
 * @param handle       raft handle (all work on its main stream).
 * @param A            column-major coefficient matrix, shape (m, n).
 * @param m            number of rows of A (length of each B column).
 * @param n            number of columns of A (length of each X column).
 * @param B            column-major RHS matrix, shape (m, n_problems).
 * @param n_problems   number of problems / columns of B and X.
 * @param masks        column-major uint8 matrix, shape (n, n_problems),
 *                     F-contiguous; element (j, p) lives at masks[p*n + j] and
 *                     is nonzero iff column j is active for problem p.  May be
 *                     null, meaning every column is active for every problem.
 * @param X            output solutions, column-major (n, n_problems).  Masked-out
 *                     rows are written as 0.
 * @param fitted       optional output A @ X, column-major (m, n_problems).  May
 *                     be null to skip the final gemm.
 * @param params       solver selection and per-backend knobs.
 */
void nnlsBatched(raft::handle_t& handle,
                 const float* A,
                 int m,
                 int n,
                 const float* B,
                 int n_problems,
                 const std::uint8_t* masks,
                 float* X,
                 float* fitted,
                 const NnlsBatchedParams& params);

void nnlsBatched(raft::handle_t& handle,
                 const double* A,
                 int m,
                 int n,
                 const double* B,
                 int n_problems,
                 const std::uint8_t* masks,
                 double* X,
                 double* fitted,
                 const NnlsBatchedParams& params);

}  // namespace Solver
}  // end namespace CUML_EXPORT ML
