/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "base.hpp"

#include <raft/core/handle.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/ternary_op.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream>

#include <iostream>
#include <vector>

namespace ML {

/**
 * Sparse matrix in CSR format.
 *
 * Note, we use cuSPARSE to manimulate matrices, and it guarantees:
 *
 *  1. row_ids[m] == nnz
 *  2. cols are sorted within rows.
 *
 * However, when the data comes from the outside, we cannot guarantee that.
 */
template <typename T, typename I = int>
struct SimpleSparseMat : SimpleMat<T> {
  typedef SimpleMat<T> Super;
  T* values;
  I* cols;
  I* row_ids;
  I nnz;

  SimpleSparseMat() : Super(0, 0), values(nullptr), cols(nullptr), row_ids(nullptr), nnz(0) {}

  SimpleSparseMat(T* values, I* cols, I* row_ids, I nnz, int m, int n)
    : Super(m, n), values(values), cols(cols), row_ids(row_ids), nnz(nnz)
  {
    check_csr(*this, 0);
  }

  void print(std::ostream& oss) const override { oss << (*this) << std::endl; }

  void operator=(const SimpleSparseMat<T, I>& other) = delete;

  inline void gemmb(const raft::handle_t& handle,
                    const T alpha,
                    const SimpleDenseMat<T>& A,
                    const bool transA,
                    const bool transB,
                    const T beta,
                    SimpleDenseMat<T>& C,
                    cudaStream_t stream) const override
  {
    const SimpleSparseMat<T, I>& B = *this;
    int kA                         = A.n;
    int kB                         = B.m;

    if (transA) {
      ASSERT(A.n == C.m, "GEMM invalid dims: m");
      kA = A.m;
    } else {
      ASSERT(A.m == C.m, "GEMM invalid dims: m");
    }

    if (transB) {
      ASSERT(B.m == C.n, "GEMM invalid dims: n");
      kB = B.n;
    } else {
      ASSERT(B.n == C.n, "GEMM invalid dims: n");
    }
    ASSERT(kA == kB, "GEMM invalid dims: k");

    // matrix C must change the order and be transposed, because we need
    // to swap arguments A and B in cusparseSpMM.
    cusparseDnMatDescr_t descrC;
    auto order = C.ord == COL_MAJOR ? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL;
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatednmat(
      &descrC, C.n, C.m, order == CUSPARSE_ORDER_COL ? C.n : C.m, C.data, order));

    /*
      The matrix A must have the same order as the matrix C in the input
      of function cusparseSpMM (i.e. swapped order w.r.t. original C).
      To account this requirement, I may need to flip transA (whether to transpose A).

         C   C' rowsC' colsC' ldC'   A  A' rowsA' colsA' ldA'  flipTransA
         c   r    n      m     m     c  r    n      m     m       x
         c   r    n      m     m     r  r    m      n     n       o
         r   c    n      m     n     c  c    m      n     m       o
         r   c    n      m     n     r  c    n      m     n       x

      where:
        c/r    - column/row major order
        A,C    - input to gemmb
        A', C' - input to cusparseSpMM
        ldX'   - leading dimension - m or n, depending on order and transX
     */
    cusparseDnMatDescr_t descrA;
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatednmat(&descrA,
                                                                C.ord == A.ord ? A.n : A.m,
                                                                C.ord == A.ord ? A.m : A.n,
                                                                A.ord == COL_MAJOR ? A.m : A.n,
                                                                A.data,
                                                                order));
    auto opA =
      transA ^ (C.ord == A.ord) ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;

    cusparseSpMatDescr_t descrB;
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatecsr(
      &descrB, B.m, B.n, B.nnz, B.row_ids, B.cols, B.values));
    auto opB = transB ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;

    auto alg = order == CUSPARSE_ORDER_COL ? CUSPARSE_SPMM_CSR_ALG1 : CUSPARSE_SPMM_CSR_ALG2;

    size_t bufferSize;
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsespmm_bufferSize(handle.get_cusparse_handle(),
                                                                    opB,
                                                                    opA,
                                                                    &alpha,
                                                                    descrB,
                                                                    descrA,
                                                                    &beta,
                                                                    descrC,
                                                                    alg,
                                                                    &bufferSize,
                                                                    stream));

    raft::interruptible::synchronize(stream);
    rmm::device_uvector<T> tmp(bufferSize, stream);

    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsespmm(handle.get_cusparse_handle(),
                                                         opB,
                                                         opA,
                                                         &alpha,
                                                         descrB,
                                                         descrA,
                                                         &beta,
                                                         descrC,
                                                         alg,
                                                         tmp.data(),
                                                         stream));

    RAFT_CUSPARSE_TRY(cusparseDestroyDnMat(descrA));
    RAFT_CUSPARSE_TRY(cusparseDestroySpMat(descrB));
    RAFT_CUSPARSE_TRY(cusparseDestroyDnMat(descrC));
  }
};

template <typename T, typename I = int>
inline void check_csr(const SimpleSparseMat<T, I>& mat, cudaStream_t stream)
{
  I row_ids_nnz;
  raft::update_host(&row_ids_nnz, &mat.row_ids[mat.m], 1, stream);
  raft::interruptible::synchronize(stream);
  ASSERT(row_ids_nnz == mat.nnz,
         "SimpleSparseMat: the size of CSR row_ids array must be `m + 1`, and "
         "the last element must be equal nnz.");
}

template <typename T, typename I = int>
std::ostream& operator<<(std::ostream& os, const SimpleSparseMat<T, I>& mat)
{
  check_csr(mat, 0);
  os << "SimpleSparseMat (CSR)"
     << "\n";
  std::vector<T> values(mat.nnz);
  std::vector<I> cols(mat.nnz);
  std::vector<I> row_ids(mat.m + 1);
  raft::update_host(
    &values[0], mat.values, mat.nnz, cuda::stream_ref{cudaStream_t{cudaStreamDefault}});
  raft::update_host(&cols[0], mat.cols, mat.nnz, cuda::stream_ref{cudaStream_t{cudaStreamDefault}});
  raft::update_host(
    &row_ids[0], mat.row_ids, mat.m + 1, cuda::stream_ref{cudaStream_t{cudaStreamDefault}});
  raft::interruptible::synchronize(cuda::stream_ref());

  int i, row_end = 0;
  for (int row = 0; row < mat.m; row++) {
    i       = row_end;
    row_end = row_ids[row + 1];
    for (int col = 0; col < mat.n; col++) {
      if (i >= row_end || col < cols[i]) {
        os << "0";
      } else {
        os << values[i];
        i++;
      }
      if (col < mat.n - 1) os << ",";
    }

    os << std::endl;
  }

  return os;
}

};  // namespace ML
