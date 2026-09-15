/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_mdspan.hpp>               // device_matrix_view / col_major
#include <raft/core/error.hpp>                       // ASSERT
#include <raft/core/resource/cuda_stream.hpp>        // raft::resource::get_cuda_stream
#include <raft/core/resource/custom_resource.hpp>    // raft::resource::get_custom_resource
#include <raft/core/resource/device_properties.hpp>  // raft::resource::get_device_properties
#include <raft/core/resources.hpp>                   // raft::resources
#include <raft/util/cache.hpp>                       // raft::cache::lru
#include <raft/util/cuda_rt_essentials.hpp>          // RAFT_CUDA_TRY
#include <raft/util/cuda_utils.cuh>                  // raft::WarpSize
#include <raft/util/integer_utils.hpp>               // raft::div_rounding_up_unsafe
#include <raft/util/reduction.cuh>  // raft::blockReduce / blockRankedReduce / warpReduce

#include <rmm/device_uvector.hpp>  // rmm::device_uvector (L scratch)

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>

namespace ML {
namespace Solver {
namespace detail {

/**
 * Cholesky ridge added to a pivot to keep the factorisation positive-definite in
 * the presence of round-off.  It scales with the pivot magnitude and with the
 * working precision (looser for float, tighter for double).
 */
template <typename T>
__device__ inline T lawson_ridge_eps(T diag)
{
  const T rel = (sizeof(T) == 4 ? T(1e-7) : T(1e-14));
  return rel * (diag > T(0) ? diag : T(1));
}

/**
 * Threshold below which an active coordinate driven down by the line search is
 * treated as exactly zero and dropped from the active set.
 */
template <typename T>
__device__ inline T lawson_zero_eps()
{
  return sizeof(T) == 4 ? T(1e-12) : T(1e-15);
}

/**
 * Finalise a bordering Cholesky append on lane 0: form the new pivot from the
 * Gram diagonal `a22` and the accumulated dots, and, when it stays positive,
 * write the diagonal of `L` and extend the forward-solve state `y`.  Publishes
 * +1 (accepted) or -1 (non-positive pivot) through `red_val[0]`.
 */
template <typename T>
__device__ inline void lawson_finish_pivot(
  T* L, int ld, int np, T a22, const T* c, int j_star, T* y, T* red_val, T dot_ll, T dot_ly)
{
  const T d2 = a22 + lawson_ridge_eps(a22) - dot_ll;
  if (d2 > T(0)) {
    const T d                   = std::sqrt(d2);
    L[(np - 1) + (np - 1) * ld] = d;
    y[np - 1]                   = (c[j_star] - dot_ly) / d;
    red_val[0]                  = T(1);
  } else {
    red_val[0] = T(-1);
  }
}

/**
 * Compute the dynamic shared-memory footprint of the Lawson-Hanson kernel for
 * a single problem with `n` columns and a block of `BlockSize` threads.  Layout
 * (T and int arrays first, then int8):
 *   T   c[n]                   A^T b
 *   T   x[n]                   current solution
 *   T   w[n]                   gradient (also scratch for the removed-index list)
 *   T   s[n]                   trial solution (also RHS / downdate scratch)
 *   T   y[n]                   forward-solve state L^-1 c_P (maintained
 *                              incrementally; back-solved into s each iteration)
 *   T   red_val[WarpSize]      reduction scratch (also used to broadcast scalars)
 *   int red_idx[WarpSize]      reduction scratch (also used to broadcast scalars)
 *   int idx[n]                 compact list of active-set column indices
 *   int8 act[n]                1 if column is in active set, 0 otherwise
 *
 * red_val/red_idx back the RAFT block reductions (raft::blockRankedReduce needs
 * a value slot followed by an index slot per warp lane, i.e. WarpSize of each,
 * laid out contiguously with red_idx immediately after red_val); slot 0 of each
 * doubles as the scalar-broadcast channel.
 *
 * Neither the Gram matrix G nor the Cholesky factor L is staged into shared
 * memory: G is read directly from global memory (L2-cached across the grid) and
 * L lives in a per-block global scratch slab (also L2-cached).  Shared memory is
 * therefore only O(n), so occupancy is not bound by the n*n factor.
 */
template <typename T, int BlockSize>
inline std::size_t lawson_smem_bytes(int n)
{
  std::size_t bytes = 0;
  bytes += sizeof(T) * static_cast<std::size_t>(n) * 5;        // c, x, w, s, y
  bytes += sizeof(T) * raft::WarpSize;                         // red_val
  bytes += sizeof(int) * raft::WarpSize;                       // red_idx
  bytes += sizeof(int) * static_cast<std::size_t>(n);          // idx
  bytes += sizeof(std::int8_t) * static_cast<std::size_t>(n);  // act
  return bytes;
}

template <typename T>
struct LawsonSmem {
  T* c;
  T* x;
  T* w;
  T* s;
  T* y;
  T* red_val;
  int* red_idx;
  int* idx;
  std::int8_t* act;
};

template <typename T, int BlockSize>
__device__ LawsonSmem<T> lawson_smem_layout(unsigned char* smem, int n)
{
  LawsonSmem<T> L;
  // Wider arrays first (double is 8-byte aligned off the 16-byte-aligned base),
  // then the 1-byte act[]; see lawson_smem_bytes.
  L.c       = reinterpret_cast<T*>(smem);
  L.x       = L.c + n;
  L.w       = L.x + n;
  L.s       = L.w + n;
  L.y       = L.s + n;
  L.red_val = L.y + n;
  // red_idx sits immediately after red_val so a single pointer (red_val) backs
  // raft::blockRankedReduce, which expects the index slots at &shbuf[WarpSize].
  L.red_idx = reinterpret_cast<int*>(L.red_val + raft::WarpSize);
  L.idx     = L.red_idx + raft::WarpSize;
  L.act     = reinterpret_cast<std::int8_t*>(L.idx + n);
  return L;
}

/**
 * Block-wide argmax of `w[i]` restricted to indices where `act[i] == 0` and,
 * when `mask` is non-empty, column `i` is enabled by the support (`mask(i) != 0`).
 * Passing an empty `mask` view means every inactive column is eligible.  Returns
 * (max_value, argmax_index) on tid 0; the result is communicated to all threads
 * via the supplied `red_val[0]` / `red_idx[0]` slots.
 */
template <typename T, int BlockSize>
__device__ inline void block_argmax_inactive(const T* w,
                                             const std::int8_t* act,
                                             raft::device_vector_view<const std::uint8_t, int> mask,
                                             int n,
                                             T* red_val,
                                             int* red_idx)
{
  const int tid       = threadIdx.x;
  const bool has_mask = mask.data_handle() != nullptr;

  T thread_max   = -std::numeric_limits<T>::infinity();
  int thread_idx = -1;
  for (int i = tid; i < n; i += BlockSize) {
    if (act[i] == 0 && (!has_mask || mask(i) != 0)) {
      T v = w[i];
      if (v > thread_max) {
        thread_max = v;
        thread_idx = i;
      }
    }
  }

  // red_val backs the reduction scratch (red_idx is contiguous right after it);
  // ties resolve to whichever lane RAFT keeps, which is fine for the algorithm.
  auto res = raft::blockRankedReduce(thread_max, red_val, thread_idx, raft::max_op{});
  if (tid == 0) {
    red_val[0] = res.first;
    red_idx[0] = res.second;
  }
  __syncthreads();
}

/**
 * Block-wide minimum of x[idx[jj]] / (x[idx[jj]] - s[jj]) over jj in [0, np)
 * where s[jj] <= 0.  Used to compute the alpha step.  Also returns the count
 * of binding indices (those with s[jj] <= 0) in `red_idx[0]`.  If no binding
 * indices are found, alpha is +inf.
 */
template <typename T, int BlockSize>
__device__ inline void block_min_alpha(
  const T* x, const T* s, const int* idx, int np, T* red_val, int* red_idx)
{
  const int tid = threadIdx.x;

  T thread_min   = std::numeric_limits<T>::infinity();
  int thread_cnt = 0;
  for (int jj = tid; jj < np; jj += BlockSize) {
    T s_jj = s[jj];
    if (s_jj <= T(0)) {
      T x_jj  = x[idx[jj]];
      T denom = x_jj - s_jj;  // strictly positive when x_jj >= 0 and s_jj <= 0
      // Numerical safety: if denom is tiny (x_jj == 0 and s_jj == 0) treat as
      // non-binding (no contribution).
      if (denom > T(0)) {
        T alpha = x_jj / denom;
        if (alpha < thread_min) thread_min = alpha;
        ++thread_cnt;
      }
    }
  }
  // Two block reductions over the same warp scratch, run back to back: the min
  // (blockRankedReduce pads absent lanes with +inf) and the binding count (a
  // plain sum, for which blockReduce's zero padding is the correct identity).
  auto min_res = raft::blockRankedReduce(thread_min, red_val, tid, raft::min_op{});
  int n_bind = raft::blockReduce<int>(thread_cnt, reinterpret_cast<char*>(red_idx), raft::add_op{});
  if (tid == 0) {
    red_val[0] = min_res.first;
    red_idx[0] = n_bind;
  }
  __syncthreads();
}

/**
 * Block-wide minimum of `s[0..np)`.  Result on `red_val[0]`.
 */
template <typename T, int BlockSize>
__device__ inline void block_min(const T* s, int np, T* red_val)
{
  const int tid = threadIdx.x;
  T thread_min  = std::numeric_limits<T>::infinity();
  for (int i = tid; i < np; i += BlockSize) {
    T v = s[i];
    if (v < thread_min) thread_min = v;
  }
  // red_val's WarpSize index scratch (red_idx) is reserved contiguously after
  // it, so blockRankedReduce is safe even though the index is unused here.
  auto res = raft::blockRankedReduce(thread_min, red_val, tid, raft::min_op{});
  if (tid == 0) red_val[0] = res.first;
  __syncthreads();
}

/**
 * Projected gradient w = c - G x, reading the Gram matrix G directly from global
 * memory (L2-cached across the grid).  Because x is zero outside the active set,
 * only the np active columns contribute:
 *   w[j] = c[j] - sum_{kk<np} G(j, idx[kk]) * x[idx[kk]].
 * For a fixed active column the threads stride over rows j, so consecutive lanes
 * read consecutive (column-major) elements of G -- a coalesced access.  np == 0
 * yields w = c.
 */
template <typename T, int BlockSize>
__device__ inline void block_matvec_gradient(
  T* w,
  const T* c,
  raft::device_matrix_view<const T, int, raft::col_major> G,
  const int* idx,
  const T* x,
  int np,
  int n)
{
  const int tid = threadIdx.x;
  for (int j = tid; j < n; j += BlockSize) {
    T acc = c[j];
    for (int kk = 0; kk < np; ++kk) {
      const int k = idx[kk];
      acc -= G(j, k) * x[k];
    }
    w[j] = acc;
  }
  __syncthreads();
}

/**
 * Incremental "bordering" Cholesky update.  On entry L (a per-block global
 * scratch slab, column-major, leading dimension `ld`, L2-cached) holds the
 * (np-1)x(np-1) lower factor of the previously active submatrix; the newly
 * activated column is idx[np-1].  The new Gram column
 * is read directly from global memory G and the factor is extended in place:
 *   solve L_11 l = a_12   for the new row l = L(np-1, 0:np-1),
 *   new diagonal          L(np-1, np-1) = sqrt(a_22 - l . l).
 * Returns false (leaving the leading (np-1) block untouched) when the new pivot
 * would be non-positive, i.e. activating idx[np-1] breaks positive-definiteness.
 *
 * Device analogue of raft::linalg::choleskyRank1Update, which is a host/cuBLAS
 * routine and so cannot be called from within a block.  `scratch` is O(np)
 * working space (the new column / row l), and positive-definiteness is kept by a
 * per-pivot ridge on a_22 (lawson_ridge_eps) rather than a global regulariser.
 *
 * Called by the whole block: the a_12 gather from global memory uses every
 * thread for full memory throughput.  Small factors (m = np-1 <= WarpSize) take a
 * single-warp fast path -- the sequential forward solve, l.l/l.y dots and pivot
 * test run on warp 0 alone (cheap `__syncwarp`/shuffle instead of block
 * barriers).  Larger factors use a blocked (panel) forward solve: panels of
 * WarpSize columns are solved top-down, the O(m^2) update of the rows below each
 * panel is applied by the WHOLE block (coalesced column reads), only the small
 * diagonal panel solve stays on warp 0, and the closing (only O(m)) l.l/l.y dots
 * stay on warp 0 with the same warpReduce as the fast path (so the pivot is
 * bit-identical) -- block barriers are O(m / WarpSize) rather than one warp-0
 * serial section that idles the other warps.  A single closing `__syncthreads`
 * publishes the extended factor and the pivot-ok flag (`red_val[0]`).
 *
 * The forward-solve state y = L^-1 c_P is extended in the same pass: the leading
 * block of L and the prefix of c_P are unchanged, so y[0:m] is untouched and the
 * only new component is y[m] = (c[j_star] - l . y[0:m]) / L_22 (for m == 0 this
 * degenerates to c[j_star] / L_22).  It is written only when the pivot is
 * accepted, so a rejected activation leaves y intact.
 */
template <int BlockSize, typename T>
__device__ inline bool block_chol_append(T* L,
                                         int ld,
                                         int np,
                                         raft::device_matrix_view<const T, int, raft::col_major> G,
                                         const int* idx,
                                         const T* c,
                                         T* y,
                                         T* red_val,
                                         T* scratch)
{
  const int tid    = threadIdx.x;
  const int m      = np - 1;  // size of the existing factor L_11
  const int j_star = idx[np - 1];

  // a_12[i] = G(idx[i], j_star): block-wide gather from global memory.
  for (int i = tid; i < m; i += BlockSize)
    scratch[i] = G(idx[i], j_star);
  __syncthreads();

  const int lane = tid % raft::WarpSize;

  // Fast path: one warp performs the whole forward solve L_11 l = a_12 (and the
  // l.l / l.y dots) when the existing factor fits in a single panel.
  if (m <= raft::WarpSize) {
    if (tid < raft::WarpSize) {
      // Forward solve in place in scratch (pivot folded into row i).
      for (int i = 0; i < m; ++i) {
        T y_i = scratch[i] / L[i + i * ld];
        __syncwarp();
        for (int j = i + lane; j < m; j += raft::WarpSize)
          scratch[j] = (j > i) ? scratch[j] - L[j + i * ld] * y_i : y_i;
        __syncwarp();
      }

      // Store l as the new row (np-1) of L; accumulate dot_ll = l . l and
      // dot_ly = l . y[0:m] (the latter extends the forward-solve state).
      T thread_dot_ll = T(0);
      T thread_dot_ly = T(0);
      for (int j = lane; j < m; j += raft::WarpSize) {
        T lj                 = scratch[j];
        L[(np - 1) + j * ld] = lj;
        thread_dot_ll += lj * lj;
        thread_dot_ly += lj * y[j];
      }
      thread_dot_ll = raft::warpReduce(thread_dot_ll, raft::add_op{});
      thread_dot_ly = raft::warpReduce(thread_dot_ly, raft::add_op{});
      if (lane == 0)
        lawson_finish_pivot(
          L, ld, np, G(j_star, j_star), c, j_star, y, red_val, thread_dot_ll, thread_dot_ly);
    }
    __syncthreads();
    return red_val[0] > T(0);
  }

  // Blocked forward solve for larger factors: panels of WarpSize rows are solved
  // top-down, and the contribution of each solved panel to the rows below it (the
  // O(m^2) bulk) is applied by the WHOLE block, while only the small diagonal
  // panel solve stays on warp 0.  Block barriers are O(m / WarpSize) instead of a
  // single warp-0 serial section that idles the other warps.  The scheme is
  // right-looking, so the between-panel update reads column j of the panel
  // (`L[i + j*ld]`, contiguous over the target rows i) -- a coalesced global read.
  {
    constexpr int b   = raft::WarpSize;
    const int warp    = tid / raft::WarpSize;
    const int n_panel = raft::div_rounding_up_unsafe(m, b);

    for (int k = 0; k < n_panel; ++k) {
      const int lo = k * b;
      const int hi = (lo + b < m) ? (lo + b) : m;

      // Within-panel solve (warp 0), right-looking over its own columns [lo,hi);
      // scratch[lo:hi] becomes the solved l entries for this panel.
      if (warp == 0) {
        for (int i = lo; i < hi; ++i) {
          T y_i = scratch[i] / L[i + i * ld];
          __syncwarp();
          for (int j = i + 1 + lane; j < hi; j += raft::WarpSize)
            scratch[j] -= L[j + i * ld] * y_i;
          __syncwarp();
          if (lane == 0) scratch[i] = y_i;
          __syncwarp();
        }
      }
      __syncthreads();

      // Between-panel update (whole block): rows [hi,m) subtract the contribution
      // of the just-solved columns [lo,hi).  Consecutive threads own consecutive
      // target rows i, so L[i + j*ld] is coalesced; scratch[j] is the solved l_j.
      for (int i = hi + tid; i < m; i += BlockSize) {
        T acc = scratch[i];
        for (int j = lo; j < hi; ++j)
          acc -= L[i + j * ld] * scratch[j];
        scratch[i] = acc;
      }
      __syncthreads();
    }

    // Store l as the new row (np-1) of L and form the closing dots on warp 0.
    // These are only O(m) (the O(m^2) work was the forward solve above), so
    // keeping them single-warp costs little; warp 0 reuses the same warpReduce
    // as the fast path so both paths produce a bit-identical pivot.
    if (warp == 0) {
      T thread_dot_ll = T(0);
      T thread_dot_ly = T(0);
      for (int j = lane; j < m; j += raft::WarpSize) {
        T lj                 = scratch[j];
        L[(np - 1) + j * ld] = lj;
        thread_dot_ll += lj * lj;
        thread_dot_ly += lj * y[j];
      }
      thread_dot_ll = raft::warpReduce(thread_dot_ll, raft::add_op{});
      thread_dot_ly = raft::warpReduce(thread_dot_ly, raft::add_op{});
      if (lane == 0)
        lawson_finish_pivot(
          L, ld, np, G(j_star, j_star), c, j_star, y, red_val, thread_dot_ll, thread_dot_ly);
    }
  }
  __syncthreads();
  return red_val[0] > T(0);
}

/**
 * Remove active-set position `p` (0-based, in the current np-ordering) from the
 * np x np lower Cholesky factor L (per-block global scratch slab, column-major,
 * leading dimension `ld`, L2-cached), producing the (np-1)x(np-1) factor of the
 * submatrix with row/column p deleted.
 *
 * Deleting an interior row/column reduces to a positive rank-1 Cholesky update
 * of the trailing diagonal block by the below-diagonal part of column p:
 *   M M^T = L33 L33^T + l3p l3p^T,
 * applied with a sequence of Givens rotations (Golub & Van Loan, "deleting a
 * column").  `v` is O(np) scratch holding l3p and the rotated residual.
 *
 * Called by the whole block.  The compaction is split into two independent
 * pieces: the columns left of p only shift rows up within a column, so the
 * whole block drives them in parallel (one thread per column, no sync); the
 * trailing block shift L(i,j) <- L(i+1,j+1) sends every element to its
 * lower-left neighbour, so it decomposes into independent diagonals (constant
 * i-j) -- one lane walks one diagonal top-down, giving a single sync-free
 * warp-wide pass on warp 0.  The Givens sweep (an inherently sequential angle
 * recurrence) also runs on warp 0 with `__syncwarp`/shuffle.  Only a single
 * closing `__syncthreads` (no per-row block barrier) exposes the downdated
 * factor to the block.  `v` is O(np) scratch holding l3p and the rotated
 * residual.
 *
 * The forward-solve state y = L^-1 c_P is downdated in lock-step: y[0:p] is
 * unchanged, the deleted component y[p] is saved as the rotation partner, the
 * trailing y[p+1:] is compacted into y[p:], and the same Givens (c,s) that
 * retriangularise L are applied to (y[p+k], partner) so y stays L^-1 c_P for the
 * shrunken active set.  These are scalar recurrences carried in a lane-0
 * register, so they piggyback on the existing sweep at no extra sync cost.
 */
template <int BlockSize, typename T>
__device__ inline void block_chol_delete_one(T* L, int ld, int np, int p, T* y, T* v)
{
  const int tid = threadIdx.x;

  // Region 1 -- columns [0, p): drop row p by shifting rows [p, np-1) up one
  // (L(i,j) <- L(i+1,j)).  Each destination reads the row directly below it in
  // the SAME column, so a thread owning a whole column and sweeping rows
  // ascending is race-free without any sync; columns are independent, so the
  // whole block runs this in parallel (one thread per column).
  for (int j = tid; j < p; j += BlockSize)
    for (int i = p; i < np - 1; ++i)
      L[i + j * ld] = L[(i + 1) + j * ld];

  // The trailing shift (diagonal-parallel), the y compaction and the Givens
  // downdate (a sequential angle recurrence) stay on warp 0, using only
  // warp-level sync; the closing block barrier below is the only __syncthreads.
  // Region 1 (other warps) writes disjoint columns [0, p), so no barrier here.
  if (tid < raft::WarpSize) {
    const int lane = tid;
    const int q    = np - 1 - p;  // trailing block size

    // Save l3p = L(p+1 : np-1, p) before the trailing shift overwrites column p.
    for (int i = lane; i < q; i += raft::WarpSize)
      v[i] = L[(p + 1 + i) + p * ld];

    // Compact the forward-solve state y (drop y[p]); keep the old y[p] as the
    // initial Givens partner for the trailing sweep below.
    T partner = T(0);
    if (lane == 0) {
      partner = y[p];
      for (int i = p; i < np - 1; ++i)
        y[i] = y[i + 1];
    }
    __syncwarp();  // l3p captured before column p is overwritten below

    // Region 2 -- trailing block up-left shift (drop row p AND column p):
    // L(i,j) <- L(i+1, j+1).  The move sends every element to its lower-left
    // neighbour, so it splits into independent diagonals (constant i-j): one
    // lane walks one diagonal (d = i-j) from the top, reading each source before
    // this same lane later overwrites it.  Diagonals never alias across lanes,
    // so the whole shift is a single sync-free warp-wide pass.
    for (int d = lane; d < q; d += raft::WarpSize)
      for (int k = 0; k <= q - 1 - d; ++k)
        L[(p + d + k) + (p + k) * ld] = L[(p + d + k + 1) + (p + k + 1) * ld];
    __syncwarp();  // trailing block published before the Givens sweep reads it

    // Positive rank-1 update of the trailing block (rows/cols p..np-2) by v.
    for (int k = 0; k < q; ++k) {
      T c = T(0), s = T(0);
      if (lane == 0) {
        T Lkk                     = L[(p + k) + (p + k) * ld];
        T vk                      = v[k];
        T r                       = std::sqrt(Lkk * Lkk + vk * vk);
        c                         = (r > T(0)) ? (Lkk / r) : T(1);
        s                         = (r > T(0)) ? (vk / r) : T(0);
        L[(p + k) + (p + k) * ld] = r;
        // Rotate the forward-solve state with the same (c,s): the trailing
        // component y[p+k] pairs with the running partner just like L's
        // column p+k pairs with the v-column.
        T yk     = y[p + k];
        y[p + k] = c * yk + s * partner;
        partner  = c * partner - s * yk;
      }
      c = __shfl_sync(0xffffffffu, c, 0);
      s = __shfl_sync(0xffffffffu, s, 0);
      for (int t = k + 1 + lane; t < q; t += raft::WarpSize) {
        const int row         = p + t;
        T lik                 = L[row + (p + k) * ld];
        T vi                  = v[t];
        L[row + (p + k) * ld] = c * lik + s * vi;
        v[t]                  = c * vi - s * lik;
      }
      __syncwarp();
    }
  }
  __syncthreads();
}

/**
 * Back substitution for the system L^T s = y, where y = L^-1 c_P is the
 * incrementally-maintained forward-solve state.  This completes the solve of
 * L L^T s = c_P: because the forward half is kept up to date across appends and
 * downdates, each inner iteration only needs this back pass (half the sequential
 * depth of a full forward+back solve).  L is the lower triangular factor stored
 * column-major with leading dimension `ld`; only the np x np leading block
 * (lower triangle) is read, so the incrementally-updated factor (fixed ld = n,
 * current size np) can be solved without repacking.  `y` is left intact (it must
 * survive into the next iteration); the solution is written to `s`.
 *
 * Called by the whole block.  After copying y -> s (coalesced), small active
 * sets (np <= WarpSize) take a single-warp fast path -- the sequential
 * substitution on warp 0 with cheap per-row `__syncwarp` and one closing
 * `__syncthreads` -- which covers the common late-stage case with minimal
 * barriers.  Larger active sets use a blocked (panel) scheme: panels of
 * WarpSize rows are solved bottom-up, and the contribution of the already-solved
 * rows below each panel (the O(np^2) bulk) is applied by the WHOLE block, while
 * only the small diagonal panel solve stays on warp 0.  Block barriers are then
 * O(np / WarpSize) instead of one giant warp-0 serial section that idles the
 * other warps.  The pass is left-looking,
 *   x_i = (y_i - sum_{j>i} L[j,i] * x_j) / L_ii,
 * so each row reads column `i` below the diagonal (`L[j + i*ld]`) -- contiguous
 * in the column-major factor (coalesced global read) -- and reduces the partial
 * products across the warp.  `y` is left intact for the next iteration.
 */
template <int BlockSize, typename T>
__device__ inline void block_chol_backsolve(const T* L, int ld, int np, const T* y, T* s)
{
  const int tid  = threadIdx.x;
  const int lane = tid % raft::WarpSize;

  for (int j = tid; j < np; j += BlockSize)
    s[j] = y[j];
  __syncthreads();

  // Fast path: one warp solves the whole system when it fits in a single panel.
  if (np <= raft::WarpSize) {
    if (tid < raft::WarpSize) {
      for (int i = np - 1; i >= 0; --i) {
        T partial = T(0);
        for (int j = i + 1 + lane; j < np; j += raft::WarpSize)
          partial += L[j + i * ld] * s[j];
        partial = raft::warpReduce(partial, raft::add_op{});
        T x_i   = (s[i] - partial) / L[i + i * ld];
        __syncwarp();
        if (lane == 0) s[i] = x_i;
        __syncwarp();
      }
    }
    __syncthreads();
    return;
  }

  // Blocked back-substitution for larger active sets.
  constexpr int b   = raft::WarpSize;
  const int nwarps  = BlockSize / raft::WarpSize;
  const int warp    = tid / raft::WarpSize;
  const int n_panel = raft::div_rounding_up_unsafe(np, b);

  for (int k = n_panel - 1; k >= 0; --k) {
    const int lo = k * b;
    const int hi = (lo + b < np) ? (lo + b) : np;

    // Between-panel update (whole block): apply the already-solved rows below the
    // panel, s[i] -= sum_{j>=hi} L[j,i]*s[j] for i in [lo,hi).  Each panel row is
    // owned by one warp (self-contained warp reduction).  Bottom panel has none.
    if (hi < np) {
      for (int i = lo + warp; i < hi; i += nwarps) {
        T partial = T(0);
        for (int j = hi + lane; j < np; j += raft::WarpSize)
          partial += L[j + i * ld] * s[j];
        partial = raft::warpReduce(partial, raft::add_op{});
        if (lane == 0) s[i] -= partial;
      }
      __syncthreads();
    }

    // Within-panel solve (warp 0): back-substitution over [lo,hi), reducing only
    // over the panel's own columns (higher columns were applied above).
    if (warp == 0) {
      for (int i = hi - 1; i >= lo; --i) {
        T partial = T(0);
        for (int j = i + 1 + lane; j < hi; j += raft::WarpSize)
          partial += L[j + i * ld] * s[j];
        partial = raft::warpReduce(partial, raft::add_op{});
        T x_i   = (s[i] - partial) / L[i + i * ld];
        __syncwarp();
        if (lane == 0) s[i] = x_i;
        __syncwarp();
      }
    }
    __syncthreads();
  }
}

/**
 * Batched, masked Lawson-Hanson NNLS kernel -- the single solver kernel used
 * for both batched and single-problem solves.  The grid is persistent: it is
 * launched with `min(P, resident)` blocks that stride over the problems
 * (`for p = blockIdx.x; p < P; p += gridDim.x`), so the per-block global scratch
 * for the Cholesky factor L is bounded by hardware occupancy rather than by the
 * batch size.  Each block reads the Gram matrix `G = A^T A` directly from global
 * memory (shared by the whole grid, L2-cached) and its own RHS projection from
 * column `p` of `C = A^T B` and active-column support from column `p` of `masks`.
 *
 * The active-set Cholesky factor L is maintained incrementally in a per-block
 * global scratch slab (L2-cached; slab `blockIdx.x` of `L_scratch`): each outer
 * iteration appends the entering column with a bordering update
 * (block_chol_append), and the inner line search shrinks it with Givens
 * downdates (block_chol_delete_one).  Consequently G is read only once per outer
 * iteration -- the active columns for the projected gradient plus the single new
 * column for the append -- and never re-gathered inside the inner loop.
 *
 * The forward-solve state y = L^-1 c_P is maintained alongside L (extended by
 * block_chol_append, downdated by block_chol_delete_one), so the inner line
 * search only runs a back substitution (block_chol_backsolve) to get the trial
 * solution s = L^-T y instead of a full forward+back solve each iteration.
 *
 * A "non-batched" solve is simply the P == 1, empty-`masks` case: the caller
 * forms G and C = A^T b once with cuBLAS and launches this kernel with a single
 * block (see nnls_batched_impl).  Columns disabled by a problem's mask are
 * excluded from the argmax so they can never enter the solution.
 *
 * @tparam BlockSize  number of threads per block (a multiple of raft::WarpSize;
 *                    chosen at launch by LawsonBlockDispatch).
 * @param G      (n, n) Gram matrix (column-major), shared by all problems.
 * @param C      (n, P) matrix of A^T B (column-major).
 * @param masks  (n, P) column-major uint8 support; column p is problem p's
 *               support.  May be an empty view, meaning "all columns eligible".
 * @param X      (n, P) solutions (column-major); written on exit.
 * @param L_scratch  global scratch for the Cholesky factor L; `gridDim.x` slabs
 *                   of `n*n` (column-major, ld = n).  Block `blockIdx.x` owns
 *                   slab `blockIdx.x`; contents are rebuilt per problem.
 * @param max_iter  outer-iteration cap.
 * @param tol       optimality tolerance on the projected gradient.
 */
template <typename T, int BlockSize>
__global__ __launch_bounds__(BlockSize) void nnls_lawson_batched_kernel(
  raft::device_matrix_view<const T, int, raft::col_major> G,
  raft::device_matrix_view<const T, int, raft::col_major> C,
  raft::device_matrix_view<const std::uint8_t, int, raft::col_major> masks,
  raft::device_matrix_view<T, int, raft::col_major> X,
  T* L_scratch,
  int max_iter,
  T tol)
{
  const int n = G.extent(0);
  const int P = C.extent(1);

  extern __shared__ unsigned char smem_raw[];
  LawsonSmem<T> S = lawson_smem_layout<T, BlockSize>(smem_raw, n);
  // The active-set Cholesky factor L lives in global memory (per-block scratch
  // slab, L2-cached) with a fixed leading dimension n so incremental
  // append/downdate never repack it.  Each block owns the slab at blockIdx.x.
  T* L = L_scratch + static_cast<std::size_t>(blockIdx.x) * n * n;

  __shared__ int sm_n_active;
  __shared__ int sm_j_star;
  __shared__ int sm_new_n;
  __shared__ int sm_n_removed;

  const int tid = threadIdx.x;

  const int inner_budget_total = 3 * n + 1;

  // Persistent grid: gridDim.x resident blocks stride over the P problems.
  for (int p = blockIdx.x; p < P; p += gridDim.x) {
    // Column p of the (possibly empty) support, as a 1-D view for the argmax.
    auto mask_col = (masks.size() != 0)
                      ? raft::make_device_vector_view<const std::uint8_t, int>(&masks(0, p), n)
                      : raft::device_vector_view<const std::uint8_t, int>{};

    // ---- Phase 1+2: load c = C[:, p]; init x and active set ------------------
    for (int j = tid; j < n; j += BlockSize) {
      S.c[j]   = C(j, p);
      S.x[j]   = T(0);
      S.act[j] = 0;
    }
    if (tid == 0) sm_n_active = 0;
    __syncthreads();

    // ---- Phase 3: outer loop (active-set growth) -----------------------------
    // Each pass brings in the inactive column with the largest projected
    // gradient; the KKT conditions hold once that gradient drops to `tol`.
    for (int outer = 0; outer < max_iter; ++outer) {
      // Projected gradient w = c - G x, reading active columns of G from global.
      block_matvec_gradient<T, BlockSize>(S.w, S.c, G, S.idx, S.x, sm_n_active, n);

      block_argmax_inactive<T, BlockSize>(S.w, S.act, mask_col, n, S.red_val, S.red_idx);
      T max_w    = S.red_val[0];
      int j_star = S.red_idx[0];
      if (j_star < 0 || max_w <= tol) break;

      // Activate j_star (append it at the end of the compact active set).
      if (tid == 0) {
        S.act[j_star]      = 1;
        S.idx[sm_n_active] = j_star;
        sm_n_active        = sm_n_active + 1;
        sm_j_star          = j_star;
      }
      __syncthreads();

      // Incremental bordering append: extend L with the new column read from
      // global G.  A non-positive pivot means idx[np-1] breaks positive-
      // definiteness, so undo the activation and stop the outer loop.
      bool ok = block_chol_append<BlockSize>(L, n, sm_n_active, G, S.idx, S.c, S.y, S.red_val, S.s);
      if (!ok) {
        if (tid == 0) {
          sm_n_active      = sm_n_active - 1;
          S.act[sm_j_star] = 0;
        }
        __syncthreads();
        break;
      }

      // Inner loop: solve the unconstrained active-set problem and, while any
      // coordinate is negative, take the largest feasible step and drop the
      // coordinates that hit zero until the trial solution is non-negative.
      for (int inner = 0; inner < inner_budget_total; ++inner) {
        const int np = sm_n_active;

        // Complete the solve L L^T s = c_P using the incrementally-maintained
        // forward-solve state y = L^-1 c_P: only the back substitution is needed.
        block_chol_backsolve<BlockSize>(L, n, np, S.y, S.s);

        block_min<T, BlockSize>(S.s, np, S.red_val);
        T min_s = S.red_val[0];
        if (min_s > T(0)) {
          for (int j = tid; j < n; j += BlockSize)
            S.x[j] = T(0);
          __syncthreads();
          for (int j = tid; j < np; j += BlockSize)
            S.x[S.idx[j]] = S.s[j];
          __syncthreads();
          break;
        }

        block_min_alpha<T, BlockSize>(S.x, S.s, S.idx, np, S.red_val, S.red_idx);
        T alpha       = S.red_val[0];
        int n_binding = S.red_idx[0];
        if (n_binding == 0) break;

        for (int jj = tid; jj < np; jj += BlockSize) {
          int j_idx  = S.idx[jj];
          T xi       = S.x[j_idx];
          T si       = S.s[jj];
          S.x[j_idx] = xi + alpha * (si - xi);
        }
        __syncthreads();

        // Compact the active set, recording the removed local positions (ascending)
        // in the free scratch backed by S.w (unused inside the inner loop).
        int* rem = reinterpret_cast<int*>(S.w);
        if (tid == 0) {
          const T zero_eps = lawson_zero_eps<T>();
          int new_n        = 0;
          int n_rem        = 0;
          for (int jj = 0; jj < np; ++jj) {
            int j_idx = S.idx[jj];
            if (S.x[j_idx] > zero_eps) {
              S.idx[new_n++] = j_idx;
            } else {
              S.act[j_idx] = 0;
              S.x[j_idx]   = T(0);
              rem[n_rem++] = jj;
            }
          }
          sm_new_n     = new_n;
          sm_n_removed = n_rem;
        }
        __syncthreads();

        // Downdate L for each removed position, deleting in descending order so
        // earlier (lower-index) deletions stay valid as np shrinks.
        int cur_np = np;
        for (int r = sm_n_removed - 1; r >= 0; --r) {
          block_chol_delete_one<BlockSize>(L, n, cur_np, rem[r], S.y, S.s);
          --cur_np;
        }
        if (tid == 0) sm_n_active = sm_new_n;
        __syncthreads();

        if (sm_n_active == 0) break;
      }
    }

    for (int j = tid; j < n; j += BlockSize)
      X(j, p) = S.x[j];
    // Barrier before the next problem reuses shared memory, so a fast thread's
    // Phase-1 re-init of S.x cannot clobber a slow thread's writeback read.
    __syncthreads();
  }
}

/**
 * Opt a kernel in to more dynamic shared memory than the device's default
 * per-block budget, but only when the requirement actually exceeds it.  Returns
 * true if the kernel is allowed to use `smem_bytes` of dynamic shared memory
 * (either because it already fits the default budget, or because the opt-in
 * succeeded); returns false otherwise, after resetting the pending CUDA error
 * so a later query is not misattributed.  The caller decides how to react to a
 * false result: the Lawson selector defers to the occupancy query
 * (`blocks_per_sm <= 0`).
 *
 * Both thresholds come from the device (`sharedMemPerBlock` /
 * `sharedMemPerBlockOptin`) rather than hardcoded 48 KB / 96 KB constants, so
 * the limits track the actual architecture.  The shmem/L1 carveout split is
 * never touched: no `cudaFuncAttributePreferredSharedMemoryCarveout` is issued,
 * and the opt-in is skipped entirely when it is not needed.
 */
template <typename Kernel>
inline bool nnls_set_smem_attr(raft::resources const& handle, Kernel kernel, std::size_t smem_bytes)
{
  const cudaDeviceProp& dev_props = raft::resource::get_device_properties(handle);
  if (smem_bytes <= static_cast<std::size_t>(dev_props.sharedMemPerBlock)) return true;
  cudaError_t err = cudaFuncSetAttribute(
    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes));
  if (err == cudaSuccess) return true;
  // Reset the sticky error so a subsequent RAFT_CUDA_TRY doesn't observe it.
  (void)cudaGetLastError();
  return false;
}

// Desired number of co-resident blocks per problem before we keep a larger
// (lower-occupancy) block size instead of shrinking it.
constexpr int kResidenceMultiple = 2;

/**
 * One block-size candidate for a batched Lawson solve: the kernel to launch,
 * its block size, its dynamic-shared-memory requirement, and its resident-block
 * count.  A chain of these forms a `lawson_plan<T>` (see below).  The kernel
 * signature does not depend on the block size, so a single function pointer can
 * hold any instantiation.
 */
template <typename T>
struct lawson_selected {
  using GView    = raft::device_matrix_view<const T, int, raft::col_major>;
  using XView    = raft::device_matrix_view<T, int, raft::col_major>;
  using MView    = raft::device_matrix_view<const std::uint8_t, int, raft::col_major>;
  using kernel_t = void (*)(GView, GView, MView, XView, T*, int, T);

  kernel_t kernel  = nullptr;
  int block_size   = 0;
  std::size_t smem = 0;
  // Number of co-resident blocks of this instantiation across the whole GPU
  // (max active blocks per SM * number of SMs).  It is the sole channel through
  // which `n_problems` influences the choice of block size: a launch saturates
  // the device once `n_problems` exceeds `resident * kResidenceMultiple`, i.e.
  // it takes more than `kResidenceMultiple` waves to drain the batch.
  long long resident = 0;
};

/**
 * Per-`n` plan for the batched Lawson kernel: the fixed chain of block-size
 * candidates (largest first), each tagged with its resident-block count.  The
 * chain depends only on `(T, n)` -- the shared-memory footprint and the per-SM
 * occupancy do not depend on `n_problems` -- so it is built once (a handful of
 * CUDA API calls) and cached per handle keyed by `n` alone.  `n_problems` then
 * selects a step by cheap arithmetic (`pick`), which means every batch size for
 * a given `n` shares one cache entry -- the widest possible equivalence class.
 */
template <typename T>
struct lawson_plan {
  // Block sizes 1024 -> 512 -> ... -> 32 (raft::WarpSize): at most 6 steps.
  static constexpr int kMaxSteps = 6;
  lawson_selected<T> steps[kMaxSteps];
  int count = 0;

  /**
   * Pick the block size for `n_problems`.  The chain is ordered from the
   * largest block (fewest resident blocks) to the smallest, and the occupancy
   * gate that admits each smaller step is `n_problems`-independent, so we simply
   * walk to a smaller block while the current one cannot already saturate the
   * device (`resident * kResidenceMultiple < n_problems`).  No CUDA API calls
   * are made here; the chain was built once at plan-construction time.
   */
  const lawson_selected<T>& pick(int n_problems) const
  {
    int i = 0;
    while (i + 1 < count &&
           steps[i].resident * static_cast<long long>(kResidenceMultiple) < n_problems) {
      ++i;
    }
    return steps[i];
  }
};

/** Per-handle custom resource holding the LRU of per-`n` Lawson plans. */
template <typename T>
struct lawson_kernel_cache {
  static constexpr std::size_t kDefaultSize = 32;
  raft::cache::lru<int, std::hash<int>, std::equal_to<>, lawson_plan<T>> value{kDefaultSize};
};

/**
 * Occupancy-driven block-size planner for the batched Lawson kernel.
 *
 * For each block size (largest first) we ask the driver how many blocks of this
 * kernel instantiation can be co-resident across the whole GPU
 * (R = max active blocks per SM * number of SMs) and record it as a plan step.
 * We then consider halving the block size -- which usually raises the block
 * count -- and continue the chain.
 *
 * The halving is gated by a condition that does *not* depend on `n_problems`:
 * we only extend the chain if a smaller block keeps at least the same per-SM
 * occupancy, measured as resident threads (blocks_per_sm * BlockSize), or still
 * has enough threads for the work (>= 2n).  Near a hardware blocks-per-SM cap,
 * or when the kernel is shared-memory bound, a smaller block can fail to raise
 * the block count enough to compensate for the fewer threads each block carries,
 * which would trade device saturation for lower utilisation -- so in that case
 * we stop the chain.  It bottoms out at a single warp (raft::WarpSize).
 *
 * The result is a `lawson_plan<T>` (never launched), so the caller can cache it
 * per `n` and pick a step for any `n_problems` without repeating CUDA API calls.
 */
template <typename T, int BlockSize = 16 * raft::WarpSize>
struct LawsonBlockDispatch {
  // Append this block size to the plan, then -- if the (n_problems-independent)
  // occupancy gate allows -- the smaller ones.  `blocks_per_sm` is this level's
  // occupancy, already measured by the caller, so each level performs at most
  // one new occupancy query (the one for the half-sized candidate).
  static void build(raft::resources const& handle, int n, int blocks_per_sm, lawson_plan<T>& plan)
  {
    const int n_sm           = raft::resource::get_device_properties(handle).multiProcessorCount;
    const std::size_t smem   = lawson_smem_bytes<T, BlockSize>(n);
    plan.steps[plan.count++] = lawson_selected<T>{&nnls_lawson_batched_kernel<T, BlockSize>,
                                                  BlockSize,
                                                  smem,
                                                  static_cast<long long>(blocks_per_sm) * n_sm};

    if constexpr (BlockSize > raft::WarpSize) {
      constexpr int half_block    = BlockSize / 2;
      const std::size_t smem_half = lawson_smem_bytes<T, half_block>(n);
      // Raise the smem cap before querying; otherwise the driver reports 0
      // active blocks for a kernel whose smem exceeds the device default.
      nnls_set_smem_attr(handle, nnls_lawson_batched_kernel<T, half_block>, smem_half);
      int blocks_per_sm_half = 0;
      RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm_half, nnls_lawson_batched_kernel<T, half_block>, half_block, smem_half));
      const int occ_full = blocks_per_sm * BlockSize;
      const int occ_half = blocks_per_sm_half * half_block;
      // Reuse this query as the next level's occupancy.
      if ((occ_half >= occ_full) || (half_block >= n)) {
        LawsonBlockDispatch<T, half_block>::build(handle, n, blocks_per_sm_half, plan);
      }
    }
  }

  /** Entry point: query the largest block's occupancy once, then build the chain. */
  static lawson_plan<T> start(raft::resources const& handle, int n)
  {
    const std::size_t smem = lawson_smem_bytes<T, BlockSize>(n);
    // The occupancy query returns 0 for a kernel whose smem exceeds the device
    // default unless its max-dynamic-smem attribute is raised first.
    nnls_set_smem_attr(handle, nnls_lawson_batched_kernel<T, BlockSize>, smem);
    int blocks_per_sm = 0;
    RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, nnls_lawson_batched_kernel<T, BlockSize>, BlockSize, smem));

    lawson_plan<T> plan;
    build(handle, n, blocks_per_sm, plan);

    // The occupancy query is the definitive "can even one block be placed" test.
    // Because our smem is independent of the block size, a zero top-level
    // occupancy means no block size can run, so it is a hard failure.
    RAFT_EXPECTS(
      plan.count > 0 && plan.steps[0].resident > 0,
      "ML::Solver::nnlsBatched: no block of the Lawson kernel can be placed on an SM for "
      "n=%d (dynamic shared memory %zu B exceeds the device per-block opt-in limit %zu B); "
      "reduce n_cols or pick a different solver.",
      n,
      smem,
      static_cast<std::size_t>(
        raft::resource::get_device_properties(handle).sharedMemPerBlockOptin));
    return plan;
  }
};

/** Dispatch a batched Lawson solve, choosing the block size from the kernel's
 *  occupancy and the batch size (see LawsonBlockDispatch).  The per-`n` plan is
 *  cached per-handle keyed by `n` alone, so the CUDA API calls behind it run
 *  once per distinct `n`; the batch size `n_problems` then selects a plan step
 *  by cheap arithmetic on every dispatch. */
template <typename T>
inline void nnls_lawson_batched_dispatch(
  raft::resources const& handle,
  raft::device_matrix_view<const T, int, raft::col_major> G,
  raft::device_matrix_view<const T, int, raft::col_major> C,
  std::optional<raft::device_matrix_view<const std::uint8_t, int, raft::col_major>> masks,
  raft::device_matrix_view<T, int, raft::col_major> X,
  int max_iter,
  T tol)
{
  using MView = raft::device_matrix_view<const std::uint8_t, int, raft::col_major>;

  const int n          = X.extent(0);
  const int n_problems = X.extent(1);

  auto& cache = raft::resource::get_custom_resource<lawson_kernel_cache<T>>(handle)->value;
  lawson_plan<T> plan;
  if (!cache.get(n, &plan)) {
    // Cache miss: build the (CUDA-API-heavy) plan once for this `n` and memoise
    // it.  On a hit no cudaFuncSetAttribute / occupancy query runs -- the smem
    // attribute set while first building this entry persists at CUDA-context
    // scope.
    plan = LawsonBlockDispatch<T>::start(handle, n);
    cache.set(n, plan);
  }
  const lawson_selected<T>& sel = plan.pick(n_problems);

  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  // Persistent grid: launch min(n_problems, resident) co-resident blocks that
  // stride over the problems, so the per-block global scratch for the Cholesky
  // factor L is bounded by hardware occupancy (resident * n*n) rather than by
  // the batch size.  Each block owns the slab at blockIdx.x.
  int grid = static_cast<int>(std::min<long long>(n_problems, sel.resident));
  grid     = std::max(grid, 1);
  rmm::device_uvector<T> L_scratch(static_cast<std::size_t>(grid) * n * n, stream);

  // Empty view => "all columns eligible" inside the kernel.
  MView mv = masks.has_value() ? *masks : MView{};
  sel.kernel<<<grid, sel.block_size, sel.smem, stream>>>(
    G, C, mv, X, L_scratch.data(), max_iter, tol);
}

}  // namespace detail
}  // namespace Solver
}  // namespace ML
