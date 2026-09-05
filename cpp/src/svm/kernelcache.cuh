/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "sparse_util.cuh"

#include <cuml/common/logger.hpp>
#include <cuml/svm/svm_parameter.h>

#include <raft/core/handle.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/init.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/util/cache.cuh>
#include <raft/util/cache_util.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reverse.h>
#include <thrust/transform.h>

#include <cuvs/distance/distance.hpp>
#include <cuvs/distance/grammian.hpp>

#include <algorithm>
#include <cstddef>

namespace ML {
namespace SVM {

namespace {  // unnamed namespace to avoid multiple definition error

/**
 * @brief Extract columns from a matrix for precomputed kernels
 *
 * Given a matrix src of shape (n_rows_src, n_cols_src), extract columns
 * specified by col_indices and store in dst of shape (n_rows_src, n_cols_dst).
 *
 * @param [out] dst destination matrix, size [n_rows_src x n_cols_dst]
 * @param [in] src source matrix, size [n_rows_src x n_cols_src]
 * @param [in] n_rows_src number of rows in source matrix
 * @param [in] col_indices column indices to extract, size [n_cols_dst]
 * @param [in] n_cols_dst number of columns to extract
 */
template <typename math_t>
CUML_KERNEL void extractColumnsKernel(
  math_t* dst, const math_t* src, int n_rows_src, const int* col_indices, int n_cols_dst)
{
  int64_t tid   = static_cast<int64_t>(threadIdx.x) + static_cast<int64_t>(blockIdx.x) * blockDim.x;
  int64_t total = static_cast<int64_t>(n_rows_src) * n_cols_dst;
  if (tid < total) {
    int64_t row = tid % n_rows_src;
    int64_t col = tid / n_rows_src;
    int src_col = col_indices[col];
    // Both source and destination are column-major:
    // src[row, col] = src[row + col * n_rows_src]
    // dst[row, col] = dst[row + col * n_rows_src] = dst[tid]
    dst[tid] = src[row + static_cast<int64_t>(src_col) * n_rows_src];
  }
}

template <typename math_t>
void extractColumnsForPrecomputed(math_t* dst,
                                  const math_t* src,
                                  int n_rows_src,
                                  const int* col_indices,
                                  int n_cols_dst,
                                  cudaStream_t stream)
{
  int total    = n_rows_src * n_cols_dst;
  int TPB      = 256;
  int n_blocks = raft::ceildiv(total, TPB);
  extractColumnsKernel<math_t>
    <<<n_blocks, TPB, 0, stream>>>(dst, src, n_rows_src, col_indices, n_cols_dst);
}

/**
 * @brief Re-raise working set indexes to SVR scope [0..2*n_rows)
 *
 * On exit, out is the permutation of n_ws such that out[k]%n_rows == n_ws_perm[k]
 * In case n_ws_perm contains duplicates they are considered to
 * represent the subspace [0..n_rows) first and [n_rows..2*n_rows) second
 *
 * @param [in] ws array with working set indices, size [n_ws]
 * @param [in] n_ws number of elements in the working set
 * @param [in] n_rows number of rows in the original problem
 * @param [in] n_ws_perm array with indices of vectors in the working set, size [n_ws]
 * @param [out] out array with workspace idx to column idx mapping, size [n_ws]
 */
CUML_KERNEL void mapColumnIndicesToSVRSpace(
  const int* ws, int n_ws, int n_rows, const int* n_ws_perm, int* out)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n_ws) {
    int wsx       = ws[tid];
    int idx       = wsx % n_rows;
    bool is_upper = idx < wsx;
    int k         = -1;
    // we have only max 1024 elements, we do a linear search
    for (int i = 0; i < n_ws; i++) {
      if (n_ws_perm[i] == idx && (is_upper || k < 0)) k = i;
      // since the array is derived from ws, the search will always return
      //  a) the first occurrence k within [0..n_rows)
      //  b) the last occurrence k within [n_rows..2*n_rows)
    }
    out[k] = wsx;
  }
}

template <typename math_t>
struct select_at_index {
  const math_t* dot_;
  select_at_index(const math_t* dot) : dot_(dot) {}

  __device__ math_t operator()(const int& i) const { return dot_[i]; }
};

/**
 * @brief Helper class to to allow batch-wise interaction with Cache
 *
 * Allows for partial row updates with an underlying cache.
 * It is assumed that 'AssignAndStoreVecs' operations are always performed
 * for all batches starting at batch 0.
 *
 */
template <typename math_t>
class BatchCache : public raft::cache::Cache<math_t> {
 public:
  /**
   * @brief BatchCache Constructor
   *
   * @param [in] n_rows number of elements in a single vector that is stored in a
   *   cache entry
   * @param [in] cache_size in MiB
   * @param [out] tmp_buffer temporary buffer of size [2*n_ws]
   * @param [in] stream cuda stream
   */
  BatchCache(int n_rows, float cache_size, cudaStream_t stream)
    : raft::cache::Cache<math_t>(stream, n_rows, cache_size), d_temp_storage(0, stream)
  {
  }
  ~BatchCache() {}

  /**
   * @brief Initialize BatchCache
   *
   * This will initialize internal tmp structures for cub
   *
   * @param [in] batch_size_base maximum number of rows in a batch
   * @param [in] n_ws array with indices of vectors in the working set, size [n_ws]
   * @param [out] tmp_buffer temporary buffer of size [2*n_ws]
   * @param [in] stream cuda stream
   */
  void Initialize(int batch_size_base, int n_ws, int* tmp_buffer, cudaStream_t stream)
  {
    this->batch_size_base = batch_size_base;
    RAFT_CUDA_TRY(cudaMemsetAsync(tmp_buffer, 0, n_ws * 2 * sizeof(int), stream));

    // Init cub buffers
    cub::DeviceRadixSort::SortPairs(NULL,
                                    d_temp_storage_size,
                                    tmp_buffer,
                                    tmp_buffer,
                                    tmp_buffer,
                                    tmp_buffer,
                                    n_ws,
                                    0,
                                    sizeof(int) * 8,
                                    stream);
    d_temp_storage.resize(d_temp_storage_size, stream);
  }

  /**
   * @brief Prepare sort order of indices
   *
   * This will reorder the keys w.r.t. the state of the cache in a way that
   *   - the keys are partitioned in cached, uncached
   *   - the uncached elements are sorted based on the value returned
   *     in cache_idx (which refers to the target cache set for uncached)
   * This will ensure that neither cache retrieval nor cache updates will
   * require additional reordering of keys.
   *
   * @param [inout] keys key indices to be reordered
   * @param [in] n number of keys
   * @param [out] cache_idx tmp buffer for cache indices [n]
   * @param [out] reorder_buffer tmp buffer for reordering of size [2*n]
   * @param [in] stream cuda stream
   */
  void PreparePartitionedIdxOrder(
    int* keys, int n, int* cache_idx, int* reorder_buffer, cudaStream_t stream)
  {
    int n_cached = 0;
    raft::cache::Cache<math_t>::GetCacheIdxPartitioned(keys, n, cache_idx, &n_cached, stream);

    int n_uncached = n - n_cached;
    if (n_uncached > 1) {
      // we also need to make sure that the next cache assignment
      // does not need to rearrange. This way the resulting ws_idx_mod
      // keys won't change during the cache update
      cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                      d_temp_storage_size,
                                      cache_idx + n_cached,
                                      reorder_buffer,
                                      keys + n_cached,
                                      reorder_buffer + n,
                                      n_uncached,
                                      0,
                                      sizeof(int) * 8,
                                      stream);
      // We can skip cache_idx as we are only interested in keys here
      raft::copy(keys + n_cached, reorder_buffer + n, n_uncached, stream);
    }
  }

  /**
   * @brief Get cache indices for keys
   *
   * This will return the cache indices for cached keys as well as the
   * cache set for uncached keys.
   * When this is called with keys coming from 'PreparePartitionedIdxOrder'
   * the keys should be unchanged upon return.
   *
   * @param [in] keys key indices
   * @param [in] n number of keys
   * @param [out] cache_idx buffer for cache indices [n]
   * @param [out] n_cached number of cached keys
   * @param [in] stream cuda stream
   */
  void GetCacheIdxPartitionedStable(
    int* keys, int n, int* cache_idx, int* n_cached, cudaStream_t stream)
  {
    raft::cache::Cache<math_t>::GetCacheIdxPartitioned(keys, n, cache_idx, n_cached, stream);

    int n_uncached = n - *n_cached;
    if (n_uncached > 1) {
      // reverse the uncached values (due to cub::DevicePartition:Flagged)
      thrust::device_ptr<int> keys_v(keys + *n_cached);
      thrust::reverse(thrust::cuda::par.on(stream), keys_v, keys_v + n_uncached);
      thrust::device_ptr<int> cache_idx_v(cache_idx + *n_cached);
      thrust::reverse(thrust::cuda::par.on(stream), cache_idx_v, cache_idx_v + n_uncached);
    }
  }

  /**
   * @brief Retrieve cached rows
   *
   * This will retrieve cached rows at the given positions for a given batch.
   *
   * @param [in] batch_idx batch id
   * @param [in] batch_size batch size
   * @param [in] idx indices to be retrieved
   * @param [in] n number of indices
   * @param [out] out buffer for cache rows, should be at least [n*batch_size]
   * @param [in] stream cuda stream
   */
  void GetVecs(
    int batch_idx, int batch_size, const int* idx, int n, math_t* out, cudaStream_t stream)
  {
    if (n > 0) {
      size_t offset = raft::cache::Cache<math_t>::GetSize() * batch_size_base * batch_idx;
      rmm::device_uvector<math_t>& cache = raft::cache::Cache<math_t>::cache;
      raft::cache::get_vecs<<<raft::ceildiv(n * batch_size, TPB), TPB, 0, stream>>>(
        cache.data() + offset, batch_size, idx, n, out);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }
  }

  /**
   * @brief Store rows to cache
   *
   * This will store new rows for a given batch.
   *
   * @param [in] batch_idx batch id
   * @param [in] batch_size batch size
   * @param [in] keys keys for rows to store
   * @param [in] n number of keys
   * @param [inout] cache_idx cache set ids
   * @param [in] tile rows store, should be at least [n*batch_size]
   * @param [in] stream cuda stream
   */
  void AssignAndStoreVecs(int batch_idx,
                          int batch_size,
                          int* keys,
                          int n,
                          int* cache_idx,
                          const math_t* tile,
                          cudaStream_t stream)
  {
    // here we assume that the input keys are already ordered by cache_idx
    // this will prevent AssignCacheIdx to modify it further
    if (n > 0) {
      if (batch_idx == 0) {
        // we only need to do this for the initial batch
        raft::cache::Cache<math_t>::AssignCacheIdx(keys, n, cache_idx, stream);
      }
      size_t offset = raft::cache::Cache<math_t>::GetSize() * batch_size_base * batch_idx;
      rmm::device_uvector<math_t>& cache = raft::cache::Cache<math_t>::cache;
      raft::cache::store_vecs<<<raft::ceildiv(n * batch_size, TPB), TPB, 0, stream>>>(
        tile,
        n,
        batch_size,
        nullptr,
        n,
        cache_idx,
        cache.data() + offset,
        raft::cache::Cache<math_t>::GetSize());
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }
  }

 private:
  int batch_size_base;  //!< maximum number of rows per batch

  // tmp storage for cub sort
  rmm::device_uvector<char> d_temp_storage;
  size_t d_temp_storage_size = 0;

  const int TPB = 256;  //!< threads per block for kernels launched
};

}  // end unnamed namespace

/**
 * @brief KernelCache to provide kernel tiles
 *
 * We calculate the kernel matrix elements for the vectors in the working set.
 *
 * Two tiles can be calculated:
 *  - SquareTile[i,j] = K(x_i, x_j) where i,j are vector indices from the working set
 *  - FullTile[i,j] = K(x_i, x_j) where i=0.._rows-1, and j is a vector index from the working set
 * The smaller square tile is calculated without caching. The larger tile can load already cached
 * columns from the cache. The large tile can be also computed batch wise to limit memory usage.
 *
 * This cache supports large matrix dimensions as well as sparse data.
 *  - For large n_rows the FullTile will be processed batch-wise.
 *  - For large n_cols the intermediate storages are kept sparse.
 *
 */
template <typename math_t, typename MatrixViewType>
class KernelCache {
 public:
  /**
   * Construct an object to manage kernel cache
   *
   * @param handle reference to raft::handle_t implementation
   * @param matrix device matrix of training vectors [n_rows x n_cols]
   * @param n_rows number of training vectors
   * @param n_cols number of features
   * @param n_ws size of working set
   * @param kernel pointer to kernel
   * @param kernel_type kernel type
   * @param cache_size (default 200 MiB)
   * @param svmType is this SVR or SVC (default)
   * @param kernel_tile_byte_limit maximum kernel size (default 1GB)
   *        Larger kernels will result in batching.
   * @param dense_extract_byte_limit sparse rows will be extracted as dense
   *        up to this limit to speed up kernel computation. Only valid
   *        for sparse input. (default 1GB)
   * @param is_precomputed if true, the matrix is a precomputed kernel matrix
   *        and no kernel computation is performed
   */
  KernelCache(const raft::handle_t& handle,
              MatrixViewType matrix,
              int n_rows,
              int n_cols,
              int n_ws,
              cuvs::distance::kernels::GramMatrixBase<math_t>* kernel,
              cuvs::distance::kernels::KernelType kernel_type,
              float cache_size                = 200,
              SvmType svmType                 = C_SVC,
              size_t kernel_tile_byte_limit   = 1 << 30,
              size_t dense_extract_byte_limit = 1 << 30,
              bool is_precomputed             = false)
    : batch_cache(n_rows, cache_size, handle.get_stream().get()),
      handle(handle),
      kernel(kernel),
      kernel_type(kernel_type),
      matrix(matrix),
      n_rows(n_rows),
      n_cols(n_cols),
      n_ws(n_ws),
      svmType(svmType),
      is_precomputed(is_precomputed),
      kernel_tile(0, handle.get_stream().get()),
      matrix_l2(0, handle.get_stream().get()),
      matrix_l2_ws(0, handle.get_stream().get()),
      ws_idx_mod(n_ws, handle.get_stream().get()),
      ws_idx_mod_svr(svmType == EPSILON_SVR ? n_ws : 0, handle.get_stream().get()),
      x_ws_csr(nullptr),
      x_ws_dense(0, handle.get_stream()),
      indptr_batched(0, handle.get_stream()),
      ws_cache_idx(n_ws * 2, handle.get_stream())
  {
    ASSERT(kernel != nullptr || is_precomputed, "Kernel pointer required for KernelCache!");
    stream = handle.get_stream().get();

    batching_enabled = false;
    is_csr           = !isDenseType<MatrixViewType>();
    sparse_extract   = false;
    batch_size_base  = n_rows;

    // enable batching for kernel > 1 GB (default)
    if ((size_t)n_rows * n_ws * sizeof(math_t) > kernel_tile_byte_limit) {
      batching_enabled = true;
      // only select based on desired big-kernel size
      batch_size_base = std::max(1ul, kernel_tile_byte_limit / n_ws / sizeof(math_t));
    }

    batch_cache.Initialize(batch_size_base, n_ws, ws_cache_idx.data(), stream);
    kernel_tile.reserve(n_ws * std::max<size_t>(batch_size_base, n_ws), stream);

    // enable sparse row extraction for sparse input where n_ws * n_cols > 1 GB
    // Warning: kernel computation will be much slower!
    if (is_csr && ((size_t)n_cols * n_ws * sizeof(math_t) > dense_extract_byte_limit)) {
      sparse_extract = true;
    }

    if (sparse_extract) {
      x_ws_csr =
        std::make_unique<raft::device_csr_matrix<math_t, int, int, int>>(handle, n_ws, n_cols);
      // we need to make an initial sparsity init before we can retrieve the structure_view
      x_ws_csr->initialize_sparsity(10);
    } else {
      x_ws_dense.resize(n_ws * static_cast<size_t>(n_cols), stream);
    }

    // store matrix l2 norm for RBF kernels (not needed for precomputed)
    if (!is_precomputed && kernel_type == cuvs::distance::kernels::KernelType::RBF) {
      matrix_l2.resize(n_rows, stream);
      matrix_l2_ws.resize(n_ws, stream);
      ML::SVM::matrixRowNorm(handle, matrix, matrix_l2.data(), raft::linalg::NormType::L2Norm);
    }

    // additional row pointer information needed for batched CSR access
    // copy matrix row pointer to host to compute partial nnz on the fly
    if (is_csr && batching_enabled) {
      host_indptr.resize(n_rows + 1);
      indptr_batched.resize(batch_size_base + 1, stream);
      copyIndptrToHost(matrix, host_indptr.data(), stream);
    }
  }
  ~KernelCache() {};

  /**
   * Helper object to pass batch information of cache while iterating batches
   */
  struct BatchDescriptor {
    int batch_id;
    int offset;
    int batch_size;
    math_t* kernel_data;
    int* nz_da_idx;
    int nnz_da;
    int n_cached;
  };

  // debugging
  enum CacheState {
    READY                = 0,
    WS_INITIALIZED       = 1,
    BATCHING_INITIALIZED = 2,
  };

  /**
   * @brief Initialize cache for new working set
   *
   * Will initialize the cache for a new working set.
   * In particular the indices will be re-ordered to allow for cache retrieval and update.
   * The re-ordered indices will stored and are accessible via 'getKernelIndices'.
   *
   * @param [in] ws_idx indices of size [n_ws]
   */
  void InitWorkingSet(const int* ws_idx)
  {
    ASSERT(cache_state != CacheState::WS_INITIALIZED, "Working set has already been initialized!");
    ASSERT(cache_state != CacheState::BATCHING_INITIALIZED, "Previous batching step incomplete!");
    this->ws_idx = ws_idx;
    if (svmType == EPSILON_SVR) {
      raft::copy(ws_idx_mod_svr.data(), ws_idx, n_ws, stream);
      GetVecIndices(ws_idx, n_ws, ws_idx_mod.data());
    } else {
      raft::copy(ws_idx_mod.data(), ws_idx, n_ws, stream);
    }

    if (batch_cache.GetSize() > 0) {
      // perform reordering of indices to partition into cached/uncached
      // batch_id 0 should behave the same as all other batches
      // provide currently unused 'kernel_tile' as temporary storage
      batch_cache.PreparePartitionedIdxOrder(
        ws_idx_mod.data(), n_ws, ws_cache_idx.data(), (int*)kernel_tile.data(), stream);

      // re-compute original indices that got flattened by GetVecIndices
      if (svmType == EPSILON_SVR) {
        mapColumnIndicesToSVRSpace<<<raft::ceildiv(n_ws, TPB), TPB, 0, stream>>>(
          ws_idx, n_ws, n_rows, ws_idx_mod.data(), ws_idx_mod_svr.data());
        RAFT_CUDA_TRY(cudaPeekAtLastError());
      }
    }

    cache_state = CacheState::WS_INITIALIZED;
  }

  /**
   * @brief Retrieve kernel indices
   *
   * Returns the reordered (!) workspace indices corresponding
   * to the order used in the provided kernel matrices.
   *
   * If allow_svr is true, input indices >= n_rows (only valid for SVR)
   * will be returned as such. Otherwise they will be projected to [0,nrows).
   *
   * This should only be called after 'InitWorkingSet'.
   *
   * @param [in] allow_svr allows indices >= n_rows (only SVR)
   * @return pointer to indices corresponding to kernel
   */
  int* getKernelIndices(int allow_svr)
  {
    ASSERT(cache_state != CacheState::READY, "Working set not initialized!");
    if (allow_svr && svmType == EPSILON_SVR) {
      return ws_idx_mod_svr.data();
    } else {
      return ws_idx_mod.data();
    }
  }

  /**
   * @brief Retrieve kernel matrix for n_ws*n_ws square
   *
   * Computes and returns the square kernel tile corresponding to the indices
   * provided by 'InitWorkingSet'.
   *
   * TODO: utilize cache read without update
   *
   * @return pointer to kernel matrix
   */
  math_t* getSquareTileWithoutCaching()
  {
    ASSERT(cache_state != CacheState::READY, "Working set not initialized!");
    ASSERT(cache_state != CacheState::BATCHING_INITIALIZED, "Previous batching step incomplete!");

    if (sparse_extract) {
      ML::SVM::extractRows<math_t>(matrix, *x_ws_csr, ws_idx_mod.data(), n_ws, handle);
    } else {
      ML::SVM::extractRows<math_t>(matrix, x_ws_dense.data(), ws_idx_mod.data(), n_ws, handle);
    }

    if (is_precomputed) {
      // For precomputed kernels, x_ws_dense contains K[ws, :] (shape n_ws x n_cols)
      // We need to extract columns ws to get K[ws, ws]
      // Since n_cols == n_rows for precomputed, we extract columns using ws_idx_mod
      extractColumnsForPrecomputed(
        kernel_tile.data(), x_ws_dense.data(), n_ws, ws_idx_mod.data(), n_ws, stream);
    } else {
      // extract dot array for RBF
      if (kernel_type == cuvs::distance::kernels::KernelType::RBF) {
        selectValueSubset(matrix_l2_ws.data(), matrix_l2.data(), ws_idx_mod.data(), n_ws);
      }

      // compute kernel
      {
        if (sparse_extract) {
          auto ws_view = getViewWithFixedDimension(*x_ws_csr, n_ws, n_cols);
          KernelOp(handle,
                   kernel,
                   ws_view,
                   ws_view,
                   kernel_tile.data(),
                   matrix_l2_ws.data(),
                   matrix_l2_ws.data());
        } else {
          KernelOp(handle,
                   kernel,
                   x_ws_dense.data(),
                   n_ws,
                   n_cols,
                   x_ws_dense.data(),
                   n_ws,
                   kernel_tile.data(),
                   matrix_l2_ws.data(),
                   matrix_l2_ws.data());
        }
      }
    }
    return kernel_tile.data();
  }

  /**
   * @brief Initialize the (batched) kernel retrieval for full rows
   *
   * Note: Values of nz_da_idx should be a subset of kernel indices with unmodified ordering!
   *
   * @param [in] nz_da_idx sub-set of working set indices to be requested
   * @param [in] nnz_da size of nz_da_idx
   * @return initialized batch descriptor object for iterating batches
   */
  BatchDescriptor InitFullTileBatching(int* nz_da_idx, int nnz_da)
  {
    ASSERT(cache_state != CacheState::READY, "Working set not initialized!");
    ASSERT(cache_state != CacheState::BATCHING_INITIALIZED, "Previous batching step incomplete!");
    int n_cached = 0;
    if (batch_cache.GetSize() > 0) {
      // we only do this once per workingset for
      batch_cache.GetCacheIdxPartitionedStable(
        nz_da_idx, nnz_da, ws_cache_idx.data() + n_ws, &n_cached, stream);
      // the second instance will be permuted during the assign step
      raft::copy(ws_cache_idx.data(), ws_cache_idx.data() + n_ws, nnz_da, stream);
    }

    int n_uncached = nnz_da - n_cached;
    if (n_uncached > 0) {
      if (sparse_extract) {
        ML::SVM::extractRows<math_t>(matrix, *x_ws_csr, nz_da_idx + n_cached, n_uncached, handle);
      } else {
        ML::SVM::extractRows<math_t>(
          matrix, x_ws_dense.data(), nz_da_idx + n_cached, n_uncached, handle);
      }
      // extract dot array for RBF
      if (kernel_type == cuvs::distance::kernels::KernelType::RBF) {
        selectValueSubset(matrix_l2_ws.data(), matrix_l2.data(), nz_da_idx + n_cached, n_uncached);
      }
    }

    cache_state = CacheState::BATCHING_INITIALIZED;

    return {.batch_id    = -1,
            .offset      = 0,
            .batch_size  = 0,
            .kernel_data = nullptr,
            .nz_da_idx   = nz_da_idx,
            .nnz_da      = nnz_da,
            .n_cached    = n_cached};
  }

  // workaround to create a view based on an owning csr_matrix that fixes
  // the initial dimensions
  // TODO: remove once not needed anymore
  raft::device_csr_matrix_view<math_t, int, int, int> getViewWithFixedDimension(
    raft::device_csr_matrix<math_t, int, int, int>& tmp_matrix, int n_rows, int n_cols)
  {
    auto csr_struct_in = tmp_matrix.structure_view();
    auto csr_struct_out =
      raft::make_device_compressed_structure_view<int, int, int>(csr_struct_in.get_indptr().data(),
                                                                 csr_struct_in.get_indices().data(),
                                                                 n_rows,
                                                                 n_cols,
                                                                 csr_struct_in.get_nnz());
    return raft::make_device_csr_matrix_view(tmp_matrix.get_elements().data(), csr_struct_out);
  }

  /**
   * @brief Iterate batches of full kernel tile [n_rows, nnz_da]
   *
   * In order to keep the cache consistent the function should always be called
   * until it returns false and all batches have been processed.
   *
   * @param [inout] batch_descriptor batching state information
   * @return true if there is still a batch to be processed
   */
  bool getNextBatchKernel(BatchDescriptor& batch_descriptor)
  {
    ASSERT(cache_state == CacheState::BATCHING_INITIALIZED, "Batching step not initialized!");

    int offset = batch_descriptor.offset + batch_descriptor.batch_size;
    if (offset >= n_rows) {
      cache_state = CacheState::READY;
      return false;
    }

    int batch_size = std::min(batch_size_base, n_rows - offset);
    int batch_id   = offset / batch_size_base;

    ASSERT(offset % batch_size_base == 0, "Inconsistent offset!");
    ASSERT(batch_id == batch_descriptor.batch_id + 1, "Inconsistent batch_id!");

    int nnz_da     = batch_descriptor.nnz_da;
    int n_cached   = batch_descriptor.n_cached;
    int n_uncached = nnz_da - n_cached;

    // fill in n_cached ids from cache
    if (n_cached > 0) {
      batch_cache.GetVecs(
        batch_id, batch_size, ws_cache_idx.data(), n_cached, kernel_tile.data(), stream);
    }

    if (n_uncached > 0) {
      int* ws_idx_new  = batch_descriptor.nz_da_idx + n_cached;
      math_t* tile_new = kernel_tile.data() + (size_t)n_cached * batch_size;

      if (is_precomputed) {
        // For precomputed kernels, extract K[offset:offset+batch_size, ws_idx_new]
        // Input matrix is column-major: K[row, col] = K[row + col * n_rows]
        // Output tile_new is column-major: tile_new[i, j] = tile_new[i + j * batch_size]
        if constexpr (isDenseType<MatrixViewType>()) {
          const math_t* matrix_data = getDenseData(matrix);
          thrust::counting_iterator<int> iter(0);
          int n_elems     = batch_size * n_uncached;
          int matrix_rows = n_rows;  // Copy member to local for lambda capture
          thrust::transform(
            thrust::cuda::par.on(stream),
            iter,
            iter + n_elems,
            tile_new,
            [matrix_data, ws_idx_new, matrix_rows, offset, batch_size] __device__(int tid) {
              // Column-major output: tile_new[row, col] = tile_new[row + col * batch_size]
              int row     = tid % batch_size;
              int col     = tid / batch_size;
              int src_row = offset + row;
              int src_col = ws_idx_new[col];
              // Column-major input: K[row, col] = K[row + col * matrix_rows]
              return matrix_data[src_row + src_col * matrix_rows];
            });
        }
      } else {
        auto batch_matrix = getMatrixBatch(
          matrix, batch_size, offset, host_indptr.data(), indptr_batched.data(), stream);

        // compute kernel
        math_t* norm_with_offset =
          matrix_l2.data() != nullptr ? matrix_l2.data() + offset : nullptr;
        if (sparse_extract) {
          auto ws_view = getViewWithFixedDimension(*x_ws_csr, n_uncached, n_cols);
          KernelOp(
            handle, kernel, batch_matrix, ws_view, tile_new, norm_with_offset, matrix_l2_ws.data());
        } else {
          KernelOp(handle,
                   kernel,
                   batch_matrix,
                   x_ws_dense.data(),
                   n_uncached,
                   tile_new,
                   norm_with_offset,
                   matrix_l2_ws.data());
        }
      }

      RAFT_CUDA_TRY(cudaPeekAtLastError());

      if (batch_cache.GetSize() > 0 && n_uncached > 0) {
        // AssignCacheIdx should not permute ws_idx_new anymore as we have sorted
        // it already during InitWorkingSet
        batch_cache.AssignAndStoreVecs(batch_id,
                                       batch_size,
                                       ws_idx_new,
                                       n_uncached,
                                       ws_cache_idx.data() + n_ws + n_cached,
                                       tile_new,
                                       stream);
      }
    }

    batch_descriptor.batch_id    = batch_id;
    batch_descriptor.offset      = offset;
    batch_descriptor.batch_size  = batch_size;
    batch_descriptor.kernel_data = kernel_tile.data();

    return true;
  }

  /** @brief Select a subset of values
   *
   * Select a subset of values
   *
   * @param [out] target array of values selected, size at least [num_indices]
   * @param [in] source source array
   * @param [in] indices indices within range [0,source.size)
   * @param [in] num_indices number of indices
   */
  void selectValueSubset(math_t* target, const math_t* source, const int* indices, int num_indices)
  {
    thrust::device_ptr<const int> indices_ptr(indices);
    thrust::device_ptr<math_t> target_ptr(target);
    thrust::transform(thrust::cuda::par.on(stream),
                      indices_ptr,
                      indices_ptr + num_indices,
                      target_ptr,
                      select_at_index(source));
  }

  /** @brief Get the original training vector idx.
   *
   * Only used for SVR (for SVC this is identity operation).
   *
   * For SVR we have duplicate set of training vectors, we return the original
   * idx, which is simply ws_idx % n_rows.
   *
   * @param [in] ws_idx array of working set indices, size [n_ws]
   * @param [in] n_ws number of elements in the working set
   * @param [out] vec_idx original training vector indices, size [n_ws]
   */
  void GetVecIndices(const int* ws_idx, int n_ws, int* vec_idx)
  {
    int n = n_rows;
    raft::linalg::unaryOp(
      vec_idx, ws_idx, n_ws, [n] __device__(math_t y) { return y < n ? y : y - n; }, stream);
  }

 private:
  MatrixViewType matrix;

  const int* ws_idx;  //!< ptr to the original working set

  bool batching_enabled;
  bool is_csr;
  bool sparse_extract;
  int batch_size_base;

  // cache state
  CacheState cache_state = CacheState::READY;

  rmm::device_uvector<math_t> kernel_tile;

  // permutation of working set indices to partition cached/uncached
  rmm::device_uvector<int> ws_idx_mod;
  rmm::device_uvector<int> ws_idx_mod_svr;

  // tmp storage for row extractions
  // needs to ne a ptr atm as there is no way to resize rows
  std::unique_ptr<raft::device_csr_matrix<math_t, int, int, int>> x_ws_csr;
  rmm::device_uvector<math_t> x_ws_dense;

  // matrix l2 norm for RBF kernels
  rmm::device_uvector<math_t> matrix_l2;
  rmm::device_uvector<math_t> matrix_l2_ws;

  // additional row pointer information needed for batched CSR access
  // copy matrix row pointer to host to compute partial nnz on the fly
  std::vector<int> host_indptr;
  rmm::device_uvector<int> indptr_batched;

  cuvs::distance::kernels::GramMatrixBase<math_t>* kernel;
  cuvs::distance::kernels::KernelType kernel_type;
  bool is_precomputed;  //!< if true, matrix is precomputed kernel

  int n_rows;  //!< number of rows in x
  int n_cols;  //!< number of columns in x
  int n_ws;    //!< number of elements in the working set

  // cache position of a workspace vectors
  // will fit n_ws twice in order to backup values
  rmm::device_uvector<int> ws_cache_idx;

  const raft::handle_t handle;

  BatchCache<math_t> batch_cache;

  cudaStream_t stream;
  SvmType svmType;

  const int TPB = 256;  //!< threads per block for kernels launched
};

};  // end namespace SVM
};  // end namespace ML
