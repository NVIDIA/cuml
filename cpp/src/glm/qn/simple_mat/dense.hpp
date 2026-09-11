/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "base.hpp"

#include <raft/core/handle.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/ternary_op.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cuda/stream>

#include <iostream>
#include <vector>
// #TODO: Replace with public header when ready
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/unary_op.cuh>

#include <rmm/device_uvector.hpp>

namespace ML {

enum STORAGE_ORDER { COL_MAJOR = 0, ROW_MAJOR = 1 };

template <typename T>
struct SimpleDenseMat : SimpleMat<T> {
  typedef SimpleMat<T> Super;
  int len;
  T* data;

  STORAGE_ORDER ord;  // storage order: runtime param for compile time sake

  SimpleDenseMat(STORAGE_ORDER order = COL_MAJOR) : Super(0, 0), data(nullptr), len(0), ord(order)
  {
  }

  SimpleDenseMat(T* data, int m, int n, STORAGE_ORDER order = COL_MAJOR)
    : Super(m, n), data(data), len(m * n), ord(order)
  {
  }

  void reset(T* data_, int m_, int n_)
  {
    this->m = m_;
    this->n = n_;
    data    = data_;
    len     = m_ * n_;
  }

  // Implemented GEMM as a static method here to improve readability
  inline static void gemm(const raft::handle_t& handle,
                          const T alpha,
                          const SimpleDenseMat<T>& A,
                          const bool transA,
                          const SimpleDenseMat<T>& B,
                          const bool transB,
                          const T beta,
                          SimpleDenseMat<T>& C,
                          cudaStream_t stream)
  {
    int kA = A.n;
    int kB = B.m;

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

    if (A.ord == COL_MAJOR && B.ord == COL_MAJOR && C.ord == COL_MAJOR) {
      // #TODO: Call from public API when ready
      raft::linalg::detail::cublasgemm(handle.get_cublas_handle(),          // handle
                                       transA ? CUBLAS_OP_T : CUBLAS_OP_N,  // transA
                                       transB ? CUBLAS_OP_T : CUBLAS_OP_N,  // transB
                                       C.m,
                                       C.n,
                                       kA,  // dimensions m,n,k
                                       &alpha,
                                       A.data,
                                       A.m,  // lda
                                       B.data,
                                       B.m,  // ldb
                                       &beta,
                                       C.data,
                                       C.m,  // ldc,
                                       stream);
      return;
    }
    if (A.ord == ROW_MAJOR) {
      const SimpleDenseMat<T> Acm(A.data, A.n, A.m, COL_MAJOR);
      gemm(handle, alpha, Acm, !transA, B, transB, beta, C, stream);
      return;
    }
    if (B.ord == ROW_MAJOR) {
      const SimpleDenseMat<T> Bcm(B.data, B.n, B.m, COL_MAJOR);
      gemm(handle, alpha, A, transA, Bcm, !transB, beta, C, stream);
      return;
    }
    if (C.ord == ROW_MAJOR) {
      SimpleDenseMat<T> Ccm(C.data, C.n, C.m, COL_MAJOR);
      gemm(handle, alpha, B, !transB, A, !transA, beta, Ccm, stream);
      return;
    }
  }

  inline void gemmb(const raft::handle_t& handle,
                    const T alpha,
                    const SimpleDenseMat<T>& A,
                    const bool transA,
                    const bool transB,
                    const T beta,
                    SimpleDenseMat<T>& C,
                    cudaStream_t stream) const override
  {
    SimpleDenseMat<T>::gemm(handle, alpha, A, transA, *this, transB, beta, C, stream);
  }

  /**
   * GEMM assigning to C where `this` refers to C.
   *
   * ```
   * *this <- alpha * A^transA * B^transB + beta * (*this)
   * ```
   */
  inline void assign_gemm(const raft::handle_t& handle,
                          const T alpha,
                          const SimpleDenseMat<T>& A,
                          const bool transA,
                          const SimpleMat<T>& B,
                          const bool transB,
                          const T beta,
                          cudaStream_t stream)
  {
    B.gemmb(handle, alpha, A, transA, transB, beta, *this, stream);
  }

  // this = a*x
  inline void ax(const T a, const SimpleDenseMat<T>& x, cudaStream_t stream)
  {
    ASSERT(ord == x.ord, "SimpleDenseMat::ax: Storage orders must match");

    auto scale = [a] __device__(const T x) { return a * x; };
    raft::linalg::unaryOp(data, x.data, len, scale, stream);
  }

  // this = a*x + y
  inline void axpy(const T a,
                   const SimpleDenseMat<T>& x,
                   const SimpleDenseMat<T>& y,
                   cudaStream_t stream)
  {
    ASSERT(ord == x.ord, "SimpleDenseMat::axpy: Storage orders must match");
    ASSERT(ord == y.ord, "SimpleDenseMat::axpy: Storage orders must match");

    auto axpy = [a] __device__(const T x, const T y) { return a * x + y; };
    raft::linalg::binaryOp(data, x.data, y.data, len, axpy, stream);
  }

  template <typename Lambda>
  inline void assign_unary(const SimpleDenseMat<T>& other, Lambda f, cudaStream_t stream)
  {
    ASSERT(ord == other.ord, "SimpleDenseMat::assign_unary: Storage orders must match");

    raft::linalg::unaryOp(data, other.data, len, f, stream);
  }

  template <typename Lambda>
  inline void assign_binary(const SimpleDenseMat<T>& other1,
                            const SimpleDenseMat<T>& other2,
                            Lambda& f,
                            cudaStream_t stream)
  {
    ASSERT(ord == other1.ord, "SimpleDenseMat::assign_binary: Storage orders must match");
    ASSERT(ord == other2.ord, "SimpleDenseMat::assign_binary: Storage orders must match");

    raft::linalg::binaryOp(data, other1.data, other2.data, len, f, stream);
  }

  template <typename Lambda>
  inline void assign_ternary(const SimpleDenseMat<T>& other1,
                             const SimpleDenseMat<T>& other2,
                             const SimpleDenseMat<T>& other3,
                             Lambda& f,
                             cudaStream_t stream)
  {
    ASSERT(ord == other1.ord, "SimpleDenseMat::assign_ternary: Storage orders must match");
    ASSERT(ord == other2.ord, "SimpleDenseMat::assign_ternary: Storage orders must match");
    ASSERT(ord == other3.ord, "SimpleDenseMat::assign_ternary: Storage orders must match");

    raft::linalg::ternaryOp(data, other1.data, other2.data, other3.data, len, f, stream);
  }

  inline void fill(const T val, cudaStream_t stream)
  {
    // TODO this reads data unnecessary, though it's mostly used for testing
    auto f = [val] __device__(const T x) { return val; };
    raft::linalg::unaryOp(data, data, len, f, stream);
  }

  inline void copy_async(const SimpleDenseMat<T>& other, cudaStream_t stream)
  {
    ASSERT((ord == other.ord) && (this->m == other.m) && (this->n == other.n),
           "SimpleDenseMat::copy: matrices not compatible");

    RAFT_CUDA_TRY(
      cudaMemcpyAsync(data, other.data, len * sizeof(T), cudaMemcpyDeviceToDevice, stream));
  }

  void print(std::ostream& oss) const override { oss << (*this) << std::endl; }

  void operator=(const SimpleDenseMat<T>& other) = delete;
};

template <typename T>
struct SimpleVec : SimpleDenseMat<T> {
  typedef SimpleDenseMat<T> Super;

  SimpleVec(T* data, const int n) : Super(data, n, 1, COL_MAJOR) {}
  // this = alpha * A * x + beta * this
  void assign_gemv(const raft::handle_t& handle,
                   const T alpha,
                   const SimpleDenseMat<T>& A,
                   bool transA,
                   const SimpleVec<T>& x,
                   const T beta,
                   cudaStream_t stream)
  {
    Super::assign_gemm(handle, alpha, A, transA, x, false, beta, stream);
  }

  SimpleVec() : Super(COL_MAJOR) {}

  inline void reset(T* new_data, int n) { Super::reset(new_data, n, 1); }
};

template <typename T>
inline void col_ref(const SimpleDenseMat<T>& mat, SimpleVec<T>& mask_vec, int c)
{
  ASSERT(mat.ord == COL_MAJOR, "col_ref only available for column major mats");
  T* tmp = &mat.data[mat.m * c];
  mask_vec.reset(tmp, mat.m);
}

template <typename T>
inline void col_slice(const SimpleDenseMat<T>& mat,
                      SimpleDenseMat<T>& mask_mat,
                      int c_from,
                      int c_to)
{
  ASSERT(c_from >= 0 && c_from < mat.n, "col_slice: invalid from");
  ASSERT(c_to >= 0 && c_to <= mat.n, "col_slice: invalid to");

  ASSERT(mat.ord == COL_MAJOR, "col_ref only available for column major mats");
  ASSERT(mask_mat.ord == COL_MAJOR, "col_ref only available for column major mask");
  T* tmp = &mat.data[mat.m * c_from];
  mask_mat.reset(tmp, mat.m, c_to - c_from);
}

// Reductions such as dot or norm require an additional location in dev mem
// to hold the result. We don't want to deal with this in the SimpleVec class
// as it  impedes thread safety and constness

template <typename T>
inline T dot(const SimpleVec<T>& u, const SimpleVec<T>& v, T* tmp_dev, cudaStream_t stream)
{
  auto f = [] __device__(const T x, const T y) { return x * y; };
  raft::linalg::mapThenSumReduce(tmp_dev, u.len, f, stream, u.data, v.data);
  T tmp_host;
  raft::update_host(&tmp_host, tmp_dev, 1, stream);

  raft::interruptible::synchronize(stream);
  return tmp_host;
}

template <typename T>
inline T squaredNorm(const SimpleVec<T>& u, T* tmp_dev, cudaStream_t stream)
{
  return dot(u, u, tmp_dev, stream);
}

template <typename T>
inline T nrmMax(const SimpleVec<T>& u, T* tmp_dev, cudaStream_t stream)
{
  auto f = [] __device__(const T x) { return raft::abs<T>(x); };
  auto r = [] __device__(const T x, const T y) { return raft::max<T>(x, y); };
  raft::linalg::mapThenReduce(tmp_dev, u.len, T(0), f, r, stream, u.data);
  T tmp_host;
  raft::update_host(&tmp_host, tmp_dev, 1, stream);
  raft::interruptible::synchronize(stream);
  return tmp_host;
}

template <typename T>
inline T nrm2(const SimpleVec<T>& u, T* tmp_dev, cudaStream_t stream)
{
  return raft::mySqrt<T>(squaredNorm(u, tmp_dev, stream));
}

template <typename T>
inline T nrm1(const SimpleVec<T>& u, T* tmp_dev, cudaStream_t stream)
{
  raft::linalg::rowNorm<raft::linalg::NormType::L1Norm, true>(
    tmp_dev, u.data, u.len, 1, stream, raft::Nop<T>());
  T tmp_host;
  raft::update_host(&tmp_host, tmp_dev, 1, stream);
  raft::interruptible::synchronize(stream);
  return tmp_host;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const SimpleVec<T>& v)
{
  std::vector<T> out(v.len);
  raft::update_host(&out[0], v.data, v.len, 0);
  raft::interruptible::synchronize(cuda::stream_ref());
  int it = 0;
  for (; it < v.len - 1;) {
    os << out[it] << " ";
    it++;
  }
  os << out[it];
  return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const SimpleDenseMat<T>& mat)
{
  os << "ord=" << (mat.ord == COL_MAJOR ? "CM" : "RM") << "\n";
  std::vector<T> out(mat.len);
  raft::update_host(&out[0], mat.data, mat.len, cuda::stream_ref{cudaStream_t{cudaStreamDefault}});
  raft::interruptible::synchronize(cuda::stream_ref());
  if (mat.ord == COL_MAJOR) {
    for (int r = 0; r < mat.m; r++) {
      int idx = r;
      for (int c = 0; c < mat.n - 1; c++) {
        os << out[idx] << ",";
        idx += mat.m;
      }
      os << out[idx] << std::endl;
    }
  } else {
    for (int c = 0; c < mat.m; c++) {
      int idx = c * mat.n;
      for (int r = 0; r < mat.n - 1; r++) {
        os << out[idx] << ",";
        idx += 1;
      }
      os << out[idx] << std::endl;
    }
  }

  return os;
}

template <typename T>
struct SimpleVecOwning : SimpleVec<T> {
  typedef SimpleVec<T> Super;
  typedef rmm::device_uvector<T> Buffer;
  Buffer buf;

  SimpleVecOwning() = delete;

  SimpleVecOwning(int n, cudaStream_t stream) : Super(), buf(n, stream)
  {
    Super::reset(buf.data(), n);
  }

  void operator=(const SimpleVec<T>& other) = delete;
};

template <typename T>
struct SimpleMatOwning : SimpleDenseMat<T> {
  typedef SimpleDenseMat<T> Super;
  typedef rmm::device_uvector<T> Buffer;
  Buffer buf;
  using Super::m;
  using Super::n;
  using Super::ord;

  SimpleMatOwning() = delete;

  SimpleMatOwning(int m, int n, cudaStream_t stream, STORAGE_ORDER order = COL_MAJOR)
    : Super(order), buf(m * n, stream)
  {
    Super::reset(buf.data(), m, n);
  }

  void operator=(const SimpleVec<T>& other) = delete;
};

};  // namespace ML
