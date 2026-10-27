/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/device_loads_stores.cuh>

#include <cub/block/block_reduce.cuh>

// Anonymous namespace for internal auxiliary functions
namespace {

template <bool trans,
          int BlockSize,
          int VecLen,
          int LdRows,
          int LdCols,
          int LdCount,
          int TileRows,
          int TileCols,
          typename T>
DI void _load_tile(const T* global_matrix, T* shared_tile, int i0, int j0, int m, int n)
{
  int th_i = threadIdx.x % LdRows;
  int th_j = threadIdx.x / LdRows;

  for (int ld_idx = 0; ld_idx < LdCount; ld_idx++) {
    T ldgData[VecLen];

    /* First, load from global mem to registers */
    int i = i0 + VecLen * th_i;
    int j = j0 + th_j + ld_idx * LdCols;
    if (i < m && j < n) {
      raft::ldg(ldgData, global_matrix + j * m + i);
    } else {
#pragma unroll
      for (int h = 0; h < VecLen; h++) {
        ldgData[h] = (T)0;
      }
    }

    /* Then, write to shared memory */
    if (trans) {
#pragma unroll
      for (int h = 0; h < VecLen; h++) {
        shared_tile[(VecLen * th_i + h) * TileCols + th_j + ld_idx * LdCols] = ldgData[h];
      }
    } else {
      raft::sts(shared_tile + (th_j + ld_idx * LdCols) * TileRows + VecLen * th_i, ldgData);
    }
  }
}

}  // namespace

namespace MLCommon {
namespace LinAlg {

/**
 * Generic block policy, that can be inherited by more specific policies.
 * Describes the shape of a tile worked by a thread block.
 *
 * @tparam _rpt    Rows worked per thread
 * @tparam _cpt    Columns worked per thread
 * @tparam _tr     Number of thread rows
 * @tparam _tc     Number of thread columns
 */
template <int _rpt, int _cpt, int _tr, int _tc>
struct BlockPolicy {
  /** Rows worked per thread */
  static constexpr int RowsPerTh = _rpt;
  /** Columns worked per thread */
  static constexpr int ColsPerTh = _cpt;
  /** Total elements worked per thread */
  static constexpr int WorkPerTh = RowsPerTh * ColsPerTh;
  /** Number of thread rows */
  static constexpr int ThRows = _tr;
  /** Number of thread columns */
  static constexpr int ThCols = _tc;
  /** Tile dimension m */
  static constexpr int Mblk = RowsPerTh * ThRows;
  /** Tile dimension n */
  static constexpr int Nblk = ColsPerTh * ThCols;
  /** Total size of a tile */
  static constexpr int TileSize = Mblk * Nblk;
  /** Number of threads per block */
  static constexpr int BlockSize = ThRows * ThCols;
};

/**
 * Execution policy for a block-local GEMM
 *
 * @tparam _veclen Length for vectorized loads (1 or 2 for fp64 + 4 for fp32)
 * @tparam _kblk   Tile dimension k
 * @tparam _rpt    Rows worked per thread
 * @tparam _cpt    Columns worked per thread
 * @tparam _tr     Number of thread rows
 * @tparam _tc     Number of thread columns
 */
template <int _veclen, int _kblk, int _rpt, int _cpt, int _tr, int _tc>
struct BlockGemmPolicy : BlockPolicy<_rpt, _cpt, _tr, _tc> {
  using Base = BlockPolicy<_rpt, _cpt, _tr, _tc>;

  /** Length for vectorized loads */
  static constexpr int VecLen = _veclen;
  /** Tile dimension k */
  static constexpr int Kblk = _kblk;

  /** Number of threads required to load a single column of the A tile */
  static constexpr int AN_LdRows = Base::Mblk / VecLen;
  /** Number of threads required to load a single row of the A' tile */
  static constexpr int AT_LdRows = Kblk / VecLen;
  /** Number of threads required to load a single column of the B tile */
  static constexpr int BN_LdRows = Kblk / VecLen;
  /** Number of threads required to load a single row of the B' tile */
  static constexpr int BT_LdRows = Base::Nblk / VecLen;

  /* Check that the block size is a multiple of LdRows, i.e one load
   * with the whole block corresponds to a number of full columns */
  static_assert(Base::BlockSize % AN_LdRows == 0);
  static_assert(Base::BlockSize % AT_LdRows == 0);
  static_assert(Base::BlockSize % BN_LdRows == 0);
  static_assert(Base::BlockSize % BT_LdRows == 0);

  /** Number of columns of the A tile in one load with the whole block */
  static constexpr int AN_LdCols = Base::BlockSize / AN_LdRows;
  /** Number of rows of the A' tile in one load with the whole block */
  static constexpr int AT_LdCols = Base::BlockSize / AT_LdRows;
  /** Number of columns of the B tile in one load with the whole block */
  static constexpr int BN_LdCols = Base::BlockSize / BN_LdRows;
  /** Number of rows of the B' tile in one load with the whole block */
  static constexpr int BT_LdCols = Base::BlockSize / BT_LdRows;

  /* Number of loads per thread necessary to load the A tile */
  static constexpr int AN_LdCount = Kblk / AN_LdCols;
  /* Number of loads per thread necessary to load the A' tile */
  static constexpr int AT_LdCount = Base::Mblk / AT_LdCols;
  /* Number of loads per thread necessary to load the B tile */
  static constexpr int BN_LdCount = Base::Nblk / BN_LdCols;
  /* Number of loads per thread necessary to load the B' tile */
  static constexpr int BT_LdCount = Kblk / BT_LdCols;
};

/**
 * Execution policy for a block-local GEMV
 *
 * @tparam _tr Number of thread rows
 * @tparam _tc Number of thread columns
 */
template <int _tr, int _tc>
struct BlockGemvPolicy {
  /** Number of thread rows */
  static constexpr int ThRows = _tr;
  /** Number of thread columns */
  static constexpr int ThCols = _tc;
  /** Number of threads per block */
  static constexpr int BlockSize = ThRows * ThCols;
};

/**
 * Structure to hold the shared memory used by a block-local GEMM
 */
template <typename GemmPolicy, typename T>
struct GemmStorage {
  /** Tile of A or A' */
  T a_tile[GemmPolicy::Mblk * GemmPolicy::Kblk];
  /** Tile of B or B' */
  T b_tile[GemmPolicy::Nblk * GemmPolicy::Kblk];
};

/**
 * Structure to hold the shared memory used by a block-local GEMV
 */
template <typename GemvPolicy, typename T>
struct GemvStorage {
  /** Accumulators to be reduced per row */
  T acc[GemvPolicy::BlockSize];
};

/**
 * Structure to hold the shared memory used by covariance numerical stability operation
 */
template <typename CovStabilityPolicy, typename T>
struct CovStabilityStorage {
  /** Transposed tile */
  T tile[CovStabilityPolicy::TileSize];
};

/**
 * Structure to hold the shared memory used by a block reduction
 */
template <int BlockSize, typename T>
struct ReductionStorage {
  using BlockReduce = cub::BlockReduce<T, BlockSize, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;

  /** Temp storage for a cub::BlockReduce */
  typename BlockReduce::TempStorage temp;
  /** Holds the value to be broadcasted to all threads */
  T broadcast;
};

/**
 * Block-local GEMM primitive C = alpha * A * B
 *
 * @note: This implementation assumes beta == 0
 *
 * @tparam     GemmPolicy   Execution policy
 * @tparam     T            Floating-point type
 * @tparam     StorageT     Temporary storage type
 * @param[in]  transa       Transpose A
 * @param[in]  transa       Transpose B
 * @param[in]  m            Number of rows of A or A' and C
 * @param[in]  n            Number of columns of B or B' and C
 * @param[in]  k            Number of columns of A or A', rows of B or B'
 * @param[in]  alpha        Coefficient alpha
 * @param[in]  a            Column-major matrix A (m x k)
 * @param[in]  b            Column-major matrix B (k x n)
 * @param[out] c            Column-major matrix C (m x n)
 * @param[in]  gemm_storage Shared temporary storage
 */
template <typename GemmPolicy, typename T, typename StorageT>
DI void _block_gemm(bool transa,
                    bool transb,
                    int m,
                    int n,
                    int k,
                    T alpha,
                    const T* a,
                    const T* b,
                    T* c,
                    StorageT& gemm_storage)
{
  const int th_off_i = threadIdx.x % GemmPolicy::ThRows;
  const int th_off_j = threadIdx.x / GemmPolicy::ThRows;

  T* shared_a_tile = gemm_storage.a_tile;
  T* shared_b_tile = gemm_storage.b_tile;
  T reg_acc[GemmPolicy::WorkPerTh];

  /* Loop over blocks of C */
  for (int blk_j = 0; blk_j < raft::ceildiv<int>(n, GemmPolicy::Nblk); blk_j++) {
    for (int blk_i = 0; blk_i < raft::ceildiv<int>(m, GemmPolicy::Mblk); blk_i++) {
      /* Initialize accumulation registers */
      for (int i = 0; i < GemmPolicy::WorkPerTh; i++) {
        reg_acc[i] = (T)0;
      }

      /* Loop over tiles in A and B corresponding to that block in C */
      for (int tile_k = 0; tile_k < raft::ceildiv<int>(k, GemmPolicy::Kblk); tile_k++) {
        /* Load a tile from A */
        if (transa)
          _load_tile<true,
                     GemmPolicy::BlockSize,
                     GemmPolicy::VecLen,
                     GemmPolicy::AT_LdRows,
                     GemmPolicy::AT_LdCols,
                     GemmPolicy::AT_LdCount,
                     GemmPolicy::Kblk,
                     GemmPolicy::Mblk>(
            a, shared_a_tile, tile_k * GemmPolicy::Kblk, blk_i * GemmPolicy::Mblk, k, m);
        else
          _load_tile<false,
                     GemmPolicy::BlockSize,
                     GemmPolicy::VecLen,
                     GemmPolicy::AN_LdRows,
                     GemmPolicy::AN_LdCols,
                     GemmPolicy::AN_LdCount,
                     GemmPolicy::Mblk,
                     GemmPolicy::Kblk>(
            a, shared_a_tile, blk_i * GemmPolicy::Mblk, tile_k * GemmPolicy::Kblk, m, k);

        /* Load a tile from B */
        if (transb)
          _load_tile<true,
                     GemmPolicy::BlockSize,
                     GemmPolicy::VecLen,
                     GemmPolicy::BT_LdRows,
                     GemmPolicy::BT_LdCols,
                     GemmPolicy::BT_LdCount,
                     GemmPolicy::Nblk,
                     GemmPolicy::Kblk>(
            b, shared_b_tile, blk_j * GemmPolicy::Nblk, tile_k * GemmPolicy::Kblk, n, k);
        else
          _load_tile<false,
                     GemmPolicy::BlockSize,
                     GemmPolicy::VecLen,
                     GemmPolicy::BN_LdRows,
                     GemmPolicy::BN_LdCols,
                     GemmPolicy::BN_LdCount,
                     GemmPolicy::Kblk,
                     GemmPolicy::Nblk>(
            b, shared_b_tile, tile_k * GemmPolicy::Kblk, blk_j * GemmPolicy::Nblk, k, n);

        __syncthreads();

        /* Loop over accumulators owned by this thread */
#pragma unroll
        for (int th_j = 0; th_j < GemmPolicy::ColsPerTh; th_j++) {
#pragma unroll
          for (int th_i = 0; th_i < GemmPolicy::RowsPerTh; th_i++) {
            int i = th_off_i + th_i * GemmPolicy::ThRows;
            int j = th_off_j + th_j * GemmPolicy::ThCols;
            /* Loop over corresponding items in the tile and accumulate */
#pragma unroll
            for (int th_k = 0; th_k < GemmPolicy::Kblk; th_k++) {
              reg_acc[GemmPolicy::RowsPerTh * th_j + th_i] +=
                shared_a_tile[GemmPolicy::Mblk * th_k + i] *
                shared_b_tile[GemmPolicy::Kblk * j + th_k];
            }
          }
        }

        __syncthreads();
      }

      /* Write accumulators in C */
#pragma unroll
      for (int th_j = 0; th_j < GemmPolicy::ColsPerTh; th_j++) {
#pragma unroll
        for (int th_i = 0; th_i < GemmPolicy::RowsPerTh; th_i++) {
          int i = blk_i * GemmPolicy::Mblk + th_off_i + th_i * GemmPolicy::ThRows;
          int j = blk_j * GemmPolicy::Nblk + th_off_j + th_j * GemmPolicy::ThCols;
          if (i < m and j < n) c[j * m + i] = alpha * reg_acc[th_j * GemmPolicy::RowsPerTh + th_i];
        }
      }
    }
  }
}

/**
 * Block-local GEMV primitive y = alpha * A * x
 *
 * @note: This implementation assumes beta == 0
 *
 * @tparam     GemvPolicy   Execution policy
 * @tparam     PreloadX     Whether to preload x to shared memory
 * @tparam     T            Floating-point type
 * @tparam     StorageT     Temporary storage type
 * @param[in]  m            Number of rows of A, length of y
 * @param[in]  n            Number of columns of A, length of x
 * @param[in]  alpha        Coefficient alpha
 * @param[in]  a            Column-major matrix A (m x n)
 * @param[in]  x            Vector x              (n)
 * @param[out] y            Vector y              (m)
 * @param[in]  gemv_storage Shared temporary storage
 * @param[out] shared_vec   (optional) Temporary storage for preloaded x
 */
template <typename GemvPolicy, bool PreloadX, typename T, typename StorageT>
DI void _block_gemv(int m,
                    int n,
                    T alpha,
                    const T* a,
                    const T* x,
                    T* y,
                    StorageT& gemv_storage,
                    T* shared_vec = nullptr)
{
  if (PreloadX) {
    /* Load x into shared vector */
    for (int i = threadIdx.x; i < n; i += GemvPolicy::BlockSize) {
      shared_vec[i] = x[i];
    }
    __syncthreads();
  }

  const T* x_ = PreloadX ? shared_vec : x;

  const int th_off_i = threadIdx.x % GemvPolicy::ThRows;
  const int th_off_j = threadIdx.x / GemvPolicy::ThRows;

  /* Loop on tiles */
  for (int tile_i = 0; tile_i < raft::ceildiv<int>(m, GemvPolicy::ThRows); tile_i++) {
    int i = tile_i * GemvPolicy::ThRows + th_off_i;
    if (i < m) {
      /* Accumulate values owned by this thread and write to shared mem */
      T acc = 0;
      for (int j = th_off_j; j < n; j += GemvPolicy::ThCols) {
        acc += a[j * m + i] * x_[j];
      }

      gemv_storage.acc[threadIdx.x] = acc;
    }
    __syncthreads();
    if (i < m) {
      /* First thread in each row does sequential reduction of other's results */
      T acc = 0;
      if (th_off_j == 0) {
        for (int tj = 0; tj < GemvPolicy::ThCols; tj++) {
          acc += gemv_storage.acc[tj * GemvPolicy::ThRows + th_off_i];
        }
        y[i] = alpha * acc;
      }
    }
    // Ensure that thread 0 has finished reading from shared memory
    // before other threads move onto the next iteration in the loop
    // and overwrite the shared memory with new values
    __syncthreads();
  }
}

/**
 * Block-local operation y = alpha * x
 *
 * @param[in]  n     Length of x and y
 * @param[in]  alpha Coefficient alpha
 * @param[in]  x     Vector x
 * @param[out] y     Vector y
 */
template <typename T>
DI void _block_ax(int n, T alpha, const T* x, T* y)
{
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    y[i] = alpha * x[i];
  }
}

/**
 * Wrapper around CUB::BlockReduce
 *
 * @tparam    BlockSize         Number of threads per block
 * @tparam    Broadcast         Whether to broadcast the result to all threads
 * @tparam    T                 Floating-point type
 * @tparam    StorageT          Temporary storage type
 * @param[in] val               Value to reduce
 * @param[in] reduction_storage Shared temporary storage
 *
 * @return The reduction result
 */
template <int BlockSize, bool Broadcast, typename T, typename StorageT>
DI T _block_reduce(T& val, StorageT& reduction_storage)
{
  using BlockReduce = cub::BlockReduce<T, BlockSize, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;

  T res = BlockReduce(reduction_storage.temp).Sum(val);

  if (Broadcast) {
    if (threadIdx.x == 0) reduction_storage.broadcast = res;
    __syncthreads();
    res = reduction_storage.broadcast;
  }

  return res;
}

/**
 * Block local dot product
 *
 * @tparam     BlockSize         Number of threads per block
 * @tparam     Broadcast         Whether to broadcast the result to all threads
 * @tparam     T                 Floating-point type
 * @tparam     StorageT          Temporary storage type
 * @param[in]  n                 Length of x and y
 * @param[in]  x                 Vector x
 * @param[out] y                 Vector y
 * @param[in]  reduction_storage Shared temporary storage
 *
 * @return Dot product of x and y
 */
template <int BlockSize, bool Broadcast, typename T, typename StorageT>
DI T _block_dot(int n, const T* x, const T* y, StorageT& reduction_storage)
{
  /* Compute dot product terms and sequential reduction per thread */
  T acc = (T)0;
  for (int i = threadIdx.x; i < n; i += BlockSize) {
    acc += x[i] * y[i];
  }

  /* Complete reduction and return dot product */
  return _block_reduce<BlockSize, Broadcast>(acc, reduction_storage);
}

/**
 * Block local operation x * A * x' (if x is a row vector)
 *
 * @tparam     BlockSize         Number of threads per block
 * @tparam     Broadcast         Whether to broadcast the result to all threads
 * @tparam     PreloadX          Whether to preload x to shared memory
 * @tparam     T                 Floating-point type
 * @tparam     StorageT          Temporary storage type
 * @param[in]  n                 Length of x
 * @param[in]  x                 Vector x
 * @param[out] A                 Column-major matrix A (n x n)
 * @param[in]  reduction_storage Shared temporary storage
 * @param[out] shared_vec        (optional) Temporary storage for preloaded x
 *
 * @return Result of x * A * x'
 */
template <int BlockSize, bool Broadcast, bool PreloadX, typename T, typename StorageT>
DI T _block_xAxt(
  int n, const T* x, const T* A, StorageT& reduction_storage, T* shared_vec = nullptr)
{
  if (PreloadX) {
    /* Load x into shared vector */
    for (int i = threadIdx.x; i < n; i += BlockSize) {
      shared_vec[i] = x[i];
    }
    __syncthreads();
  }

  const T* x_ = PreloadX ? shared_vec : x;

  /* Compute terms and sequential reduction per thread */
  T acc = (T)0;
  for (int idx = threadIdx.x; idx < n * n; idx += BlockSize) {
    int i = idx % n;
    int j = idx / n;

    acc += x_[i] * A[idx] * x_[j];
  }

  /* Complete reduction and return dot product */
  return _block_reduce<BlockSize, Broadcast>(acc, reduction_storage);
}

/**
 * @brief Improves numerical accuracy by making sure that the covariance matrix
 *        is symmetric and only has positive elements along the diagonal.
 *
 * @todo: solve bank conflicts
 *
 * @tparam     CovPolicy   Execution policy
 * @tparam     T           Floating-point type
 * @tparam     StorageT    Shared memory storage structure type
 * @param[in]  n           Matrix size
 * @param[in]  in          Input covariance matrix
 * @param[out] out         Output covariance matrix
 * @param[in]  cov_storage Temporary shared memory storage
 */
template <typename CovPolicy, typename T, typename StorageT>
DI void _block_covariance_stability(int n, const T* in, T* out, StorageT& cov_storage)
{
  int th_off_i = threadIdx.x % CovPolicy::ThRows;
  int th_off_j = threadIdx.x / CovPolicy::ThRows;

  /* Loop over tiles */
  for (int blk_j = 0; blk_j < raft::ceildiv<int>(n, CovPolicy::Nblk); blk_j++) {
    for (int blk_i = 0; blk_i < raft::ceildiv<int>(n, CovPolicy::Mblk); blk_i++) {
      // Load the tile of the transpose matrix into a N x M shared memory tile
      _load_tile<false,
                 CovPolicy::BlockSize,
                 1,
                 CovPolicy::Nblk,
                 CovPolicy::BlockSize / CovPolicy::Nblk,
                 CovPolicy::RowsPerTh * CovPolicy::ColsPerTh,
                 CovPolicy::Nblk,
                 CovPolicy::Mblk>(
        in, cov_storage.tile, blk_j * CovPolicy::Nblk, blk_i * CovPolicy::Mblk, n, n);
      __syncthreads();

      // Read from matrix and transposed tile, write to output matrix
#pragma unroll
      for (int th_j = 0; th_j < CovPolicy::ColsPerTh; th_j++) {
#pragma unroll
        for (int th_i = 0; th_i < CovPolicy::RowsPerTh; th_i++) {
          int i  = th_off_i + th_i * CovPolicy::ThRows;
          int j  = th_off_j + th_j * CovPolicy::ThCols;
          int gi = blk_i * CovPolicy::Mblk + i;
          int gj = blk_j * CovPolicy::Nblk + j;

          if (gi < n && gj < n) {
            T in0            = in[gj * n + gi];
            T in1            = cov_storage.tile[i * CovPolicy::Nblk + j];
            out[gj * n + gi] = gi == gj ? abs(in0) : 0.5 * (in0 + in1);
          }
        }
      }

      __syncthreads();
    }
  }
}

}  // namespace LinAlg
}  // namespace MLCommon
