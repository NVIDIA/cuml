/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/math.hpp>
#include <raft/linalg/add.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>
#include <linalg/batched/matrix.cuh>
#include <linalg_naive.h>
#include <test_utils.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

namespace MLCommon {
namespace LinAlg {
namespace Batched {

enum MatrixOperation {
  AB_op,          // Matrix-matrix product (with GEMM)
  AZT_op,         // Matrix-vector product (with GEMM)
  ZA_op,          // Vector-matrix product (with GEMM)
  ApB_op,         // Addition
  AmB_op,         // Subtraction
  AkB_op,         // Kronecker product
  AsolveZ_op,     // Linear equation solver Ax=b
  LaggedZ_op,     // Lag matrix
  CopyA2D_op,     // 2D copy
  DiffA_op,       // Vector first difference
  Hessenberg_op,  // Hessenberg decomposition A=UHU'
  Schur_op,       // Schur decomposition A=USU'
  Lyapunov_op,    // Lyapunov equation solver AXA'-X+B=0
};

template <typename T>
struct MatrixInputs {
  MatrixOperation operation;
  int batch_size;
  int m;  // Usually the dimensions of A and/or Z
  int n;
  int p;  // Usually the dimensions of B or other parameters
  int q;
  int s;  // Additional parameters for operations that need more than 4
  int t;
  T tolerance;
};

template <typename T>
class MatrixTest : public ::testing::TestWithParam<MatrixInputs<T>> {
 protected:
  void SetUp() override
  {
    using std::vector;
    params = ::testing::TestWithParam<MatrixInputs<T>>::GetParam();

    // Find out whether A, B and Z will be used (depending on the operation)
    bool use_A = (params.operation != LaggedZ_op);
    bool use_B = (params.operation == AB_op) || (params.operation == ApB_op) ||
                 (params.operation == AmB_op) || (params.operation == AkB_op) ||
                 (params.operation == Lyapunov_op);
    bool use_Z = (params.operation == AZT_op) || (params.operation == ZA_op) ||
                 (params.operation == AsolveZ_op) || (params.operation == LaggedZ_op);
    bool Z_col = (params.operation == AsolveZ_op);
    int r      = params.operation == AZT_op ? params.n : params.m;

    // Check if the dimensions are valid and compute the output dimensions
    int m_r{};
    int n_r{};

    switch (params.operation) {
      case AB_op:
        ASSERT_TRUE(params.n == params.p);
        m_r = params.m;
        n_r = params.q;
        break;
      case ApB_op:
      case AmB_op:
        ASSERT_TRUE(params.m == params.p && params.n == params.q);
        m_r = params.m;
        n_r = params.n;
        break;
      case AkB_op:
        m_r = params.m * params.p;
        n_r = params.n * params.q;
        break;
      case AZT_op:
        m_r = params.m;
        n_r = 1;
        break;
      case ZA_op:
        m_r = 1;
        n_r = params.n;
        break;
      case AsolveZ_op:
        ASSERT_TRUE(params.n == params.m);
        // For this test we multiply A by the solution and check against Z
        m_r = params.m;
        n_r = 1;
        break;
      case LaggedZ_op:
        // For this operation params.n holds the number of lags
        m_r = params.m - params.n;
        n_r = params.n;
        break;
      case CopyA2D_op:
        // For this operation p and q are the dimensions of the copy window
        m_r = params.p;
        n_r = params.q;
        break;
      case DiffA_op:
        // Note: A can represent either a row or column vector
        ASSERT_TRUE(params.m == 1 || params.n == 1);
        m_r = std::max(1, params.m - 1);
        n_r = std::max(1, params.n - 1);
        break;
      case Hessenberg_op:
      case Schur_op:
      case Lyapunov_op:
        ASSERT_TRUE(params.m == params.n && params.m == params.p && params.m == params.q);
        m_r = params.m;
        n_r = params.m;
        break;
    }

    // Create test matrices and vector
    std::vector<T> A;
    std::vector<T> B;
    std::vector<T> Z;
    if (use_A) A.resize(params.batch_size * params.m * params.n);
    if (use_B) B.resize(params.batch_size * params.p * params.q);
    if (use_Z) Z.resize(params.batch_size * r);

    // Generate random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> udis(-1.0, 3.0);
    for (std::size_t i = 0; i < A.size(); i++)
      A[i] = udis(gen);
    for (std::size_t i = 0; i < B.size(); i++)
      B[i] = udis(gen);
    for (std::size_t i = 0; i < Z.size(); i++)
      Z[i] = udis(gen);

    // Create handles, stream
    RAFT_CUBLAS_TRY(cublasCreate(&handle));
    RAFT_CUDA_TRY(cudaStreamCreate(&stream));

    // Created batched matrices
    Matrix<T> AbM(params.m, params.n, params.batch_size, handle, stream);
    Matrix<T> BbM(params.p, params.q, params.batch_size, handle, stream);
    Matrix<T> ZbM(Z_col ? r : 1, Z_col ? 1 : r, params.batch_size, handle, stream);

    // Copy the data to the device
    if (use_A) raft::update_device(AbM.raw_data(), A.data(), A.size(), stream);
    if (use_B) raft::update_device(BbM.raw_data(), B.data(), B.size(), stream);
    if (use_Z) raft::update_device(ZbM.raw_data(), Z.data(), Z.size(), stream);

    // Create fake batched matrices to be overwritten by results
    res_bM = new Matrix<T>(1, 1, 1, handle, stream);

    // Compute the tested results
    switch (params.operation) {
      case AB_op: *res_bM = AbM * BbM; break;
      case ApB_op: *res_bM = AbM + BbM; break;
      case AmB_op: *res_bM = AbM - BbM; break;
      case AkB_op: *res_bM = b_kron(AbM, BbM); break;
      case AZT_op: *res_bM = b_gemm(AbM, ZbM, false, true); break;
      case ZA_op: *res_bM = ZbM * AbM; break;
      case AsolveZ_op:
        // A * A\Z -> should be Z
        *res_bM = AbM * b_solve(AbM, ZbM);
        break;
      case LaggedZ_op: *res_bM = b_lagged_mat(ZbM, params.n); break;
      case CopyA2D_op: *res_bM = b_2dcopy(AbM, params.s, params.t, params.p, params.q); break;
      case DiffA_op: *res_bM = AbM.difference(); break;
      case Hessenberg_op: {
        constexpr T zero_tolerance = std::is_same<T, double>::value ? 1e-7 : 1e-3f;

        int n = params.m;
        Matrix<T> HbM(n, n, params.batch_size, handle, stream);
        Matrix<T> UbM(n, n, params.batch_size, handle, stream);
        b_hessenberg(AbM, UbM, HbM);

        // Check that H is in Hessenberg form
        std::vector<T> H = std::vector<T>(n * n * params.batch_size);
        raft::update_host(H.data(), HbM.raw_data(), H.size(), stream);
        raft::interruptible::synchronize(stream);
        for (int ib = 0; ib < params.batch_size; ib++) {
          for (int j = 0; j < n - 2; j++) {
            for (int i = j + 2; i < n; i++) {
              ASSERT_TRUE(raft::abs(H[n * n * ib + n * j + i]) < zero_tolerance);
            }
          }
        }

        // Check that U is unitary (UU'=I)
        std::vector<T> UUt = std::vector<T>(n * n * params.batch_size);
        raft::update_host(UUt.data(), b_gemm(UbM, UbM, false, true).raw_data(), UUt.size(), stream);
        raft::interruptible::synchronize(stream);
        for (int ib = 0; ib < params.batch_size; ib++) {
          for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
              ASSERT_TRUE(raft::abs(UUt[n * n * ib + n * j + i] - (i == j ? (T)1 : (T)0)) <
                          zero_tolerance);
            }
          }
        }

        // Write UHU' in the result (will be compared against A)
        *res_bM = UbM * b_gemm(HbM, UbM, false, true);
        break;
      }
      case Schur_op: {
        constexpr T zero_tolerance = std::is_same<T, double>::value ? 1e-7 : 1e-3f;

        int n = params.m;
        Matrix<T> SbM(n, n, params.batch_size, handle, stream);
        Matrix<T> UbM(n, n, params.batch_size, handle, stream);
        b_schur(AbM, UbM, SbM);

        // Check that S is in Schur form
        std::vector<T> S = std::vector<T>(n * n * params.batch_size);
        raft::update_host(S.data(), SbM.raw_data(), S.size(), stream);
        raft::interruptible::synchronize(stream);
        for (int ib = 0; ib < params.batch_size; ib++) {
          for (int j = 0; j < n - 2; j++) {
            for (int i = j + 2; i < n; i++) {
              ASSERT_TRUE(raft::abs(S[n * n * ib + n * j + i]) < zero_tolerance);
            }
          }
        }
        for (int ib = 0; ib < params.batch_size; ib++) {
          for (int k = 0; k < n - 3; k++) {
            ASSERT_FALSE(raft::abs(S[n * n * ib + n * k + k + 1]) > zero_tolerance &&
                         raft::abs(S[n * n * ib + n * (k + 1) + k + 2]) > zero_tolerance &&
                         raft::abs(S[n * n * ib + n * (k + 2) + k + 3]) > zero_tolerance);
          }
        }

        // Check that U is unitary (UU'=I)
        std::vector<T> UUt = std::vector<T>(n * n * params.batch_size);
        raft::update_host(UUt.data(), b_gemm(UbM, UbM, false, true).raw_data(), UUt.size(), stream);
        raft::interruptible::synchronize(stream);
        for (int ib = 0; ib < params.batch_size; ib++) {
          for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
              ASSERT_TRUE(raft::abs(UUt[n * n * ib + n * j + i] - (i == j ? (T)1 : (T)0)) <
                          zero_tolerance);
            }
          }
        }

        // Write USU' in the result (will be compared against A)
        *res_bM = UbM * b_gemm(SbM, UbM, false, true);
        break;
      }
      case Lyapunov_op: {
        Matrix<T> XbM = b_lyapunov(AbM, BbM);

        // Write AXA'-X in the result (will be compared against -B)
        *res_bM = AbM * b_gemm(XbM, AbM, false, true) - XbM;
        break;
      }
    }

    // Compute the expected results
    res_h.resize(params.batch_size * m_r * n_r);
    switch (params.operation) {
      case AB_op:
        for (int bid = 0; bid < params.batch_size; bid++) {
          Naive::matMul(res_h.data() + bid * m_r * n_r,
                        A.data() + bid * params.m * params.n,
                        B.data() + bid * params.p * params.q,
                        params.m,
                        params.n,
                        params.q);
        }
        break;
      case ApB_op: Naive::add(res_h.data(), A.data(), B.data(), A.size()); break;
      case AmB_op: Naive::add(res_h.data(), A.data(), B.data(), A.size(), T(-1.0)); break;
      case AkB_op:
        for (int bid = 0; bid < params.batch_size; bid++) {
          Naive::kronecker(res_h.data() + bid * m_r * n_r,
                           A.data() + bid * params.m * params.n,
                           B.data() + bid * params.p * params.q,
                           params.m,
                           params.n,
                           params.p,
                           params.q);
        }
        break;
      case AZT_op:
        for (int bid = 0; bid < params.batch_size; bid++) {
          Naive::matMul(res_h.data() + bid * m_r * n_r,
                        A.data() + bid * params.m * params.n,
                        Z.data() + bid * r,
                        params.m,
                        params.n,
                        1);
        }
        break;
      case ZA_op:
        for (int bid = 0; bid < params.batch_size; bid++) {
          Naive::matMul(res_h.data() + bid * m_r * n_r,
                        Z.data() + bid * r,
                        A.data() + bid * params.m * params.n,
                        1,
                        params.m,
                        params.n);
        }
        break;
      case AsolveZ_op:
        // Simply copy Z in the result
        memcpy(res_h.data(), Z.data(), r * params.batch_size * sizeof(T));
        break;
      case LaggedZ_op:
        for (int bid = 0; bid < params.batch_size; bid++) {
          Naive::laggedMat(
            res_h.data() + bid * m_r * n_r, Z.data() + bid * params.m, params.m, params.n);
        }
        break;
      case CopyA2D_op:
        for (int bid = 0; bid < params.batch_size; bid++) {
          Naive::copy2D(res_h.data() + bid * m_r * n_r,
                        A.data() + bid * params.m * params.n,
                        params.s,
                        params.t,
                        params.m,
                        m_r,
                        n_r);
        }
        break;
      case DiffA_op: {
        int len = params.m * params.n;
        for (int bid = 0; bid < params.batch_size; bid++) {
          Naive::diff(res_h.data() + bid * (len - 1), A.data() + bid * len, len);
        }
        break;
      }
      case Hessenberg_op:
      case Schur_op:
        // Simply copy A (will be compared against UHU')
        memcpy(res_h.data(), A.data(), params.m * params.m * params.batch_size * sizeof(T));
        break;
      case Lyapunov_op:
        // Simply copy -B (will be compared against AXA'-X)
        for (int i = 0; i < params.m * params.m * params.batch_size; i++) {
          res_h[i] = -B[i];
        }
        break;
    }

    raft::interruptible::synchronize(stream);
  }

  void TearDown() override
  {
    delete res_bM;
    RAFT_CUBLAS_TRY(cublasDestroy(handle));
    RAFT_CUDA_TRY(cudaStreamDestroy(stream));
  }

 protected:
  MatrixInputs<T> params;
  Matrix<T>* res_bM;
  std::vector<T> res_h;
  cublasHandle_t handle;
  cudaStream_t stream = 0;
};

// Test parameters (op, batch_size, m, n, p, q, s, t, tolerance)
const std::vector<MatrixInputs<double>> inputsd = {
  {AB_op, 7, 15, 37, 37, 11, 0, 0, 1e-6},
  {AZT_op, 5, 33, 65, 1, 1, 0, 0, 1e-6},
  {ZA_op, 8, 12, 41, 1, 1, 0, 0, 1e-6},
  {ApB_op, 4, 16, 48, 16, 48, 0, 0, 1e-6},
  {AmB_op, 17, 9, 3, 9, 3, 0, 0, 1e-6},
  {AkB_op, 5, 3, 13, 31, 8, 0, 0, 1e-6},
  {AkB_op, 3, 7, 12, 31, 15, 0, 0, 1e-6},
  {AkB_op, 2, 11, 2, 8, 46, 0, 0, 1e-6},
  {AsolveZ_op, 6, 17, 17, 1, 1, 0, 0, 1e-6},
  {LaggedZ_op, 5, 31, 9, 1, 1, 0, 0, 1e-6},
  {LaggedZ_op, 7, 129, 3, 1, 1, 0, 0, 1e-6},
  {CopyA2D_op, 11, 31, 63, 17, 14, 5, 9, 1e-6},
  {CopyA2D_op, 4, 33, 7, 30, 4, 3, 0, 1e-6},
  {DiffA_op, 5, 11, 1, 1, 1, 0, 0, 1e-6},
  {DiffA_op, 15, 1, 37, 1, 1, 0, 0, 1e-6},
  {Hessenberg_op, 10, 15, 15, 15, 15, 0, 0, 1e-6},
  {Hessenberg_op, 30, 61, 61, 61, 61, 0, 0, 1e-6},
  // {Schur_op, 7, 12, 12, 12, 12, 0, 0, 1e-3},
  // {Schur_op, 17, 77, 77, 77, 77, 0, 0, 1e-3},
  // {Lyapunov_op, 5, 14, 14, 14, 14, 0, 0, 1e-2},
  // {Lyapunov_op, 13, 100, 100, 100, 100, 0, 0, 1e-2}
};

// Note: Schur and Lyapunov tests have had stability issues on CI so
// they are disabled temporarily. See issue:
// https://github.com/NVIDIA/cuml/issues/1949

// Test parameters (op, batch_size, m, n, p, q, s, t, tolerance)
const std::vector<MatrixInputs<float>> inputsf = {
  {AB_op, 7, 15, 37, 37, 11, 0, 0, 1e-2},
  {AZT_op, 5, 33, 65, 1, 1, 0, 0, 1e-2},
  {ZA_op, 8, 12, 41, 1, 1, 0, 0, 1e-2},
  {ApB_op, 4, 16, 48, 16, 48, 0, 0, 1e-2},
  {AmB_op, 17, 9, 3, 9, 3, 0, 0, 1e-2},
  {AkB_op, 5, 3, 13, 31, 8, 0, 0, 1e-2},
  {AkB_op, 3, 7, 12, 31, 15, 0, 0, 1e-2},
  {AkB_op, 2, 11, 2, 8, 46, 0, 0, 1e-2},
  {AsolveZ_op, 6, 17, 17, 1, 1, 0, 0, 1e-2},
  {LaggedZ_op, 5, 31, 9, 1, 1, 0, 0, 1e-5},
  {LaggedZ_op, 7, 129, 3, 1, 1, 0, 0, 1e-5},
  {CopyA2D_op, 11, 31, 63, 17, 14, 5, 9, 1e-5},
  {CopyA2D_op, 4, 33, 7, 30, 4, 3, 0, 1e-5},
  {DiffA_op, 5, 11, 1, 1, 1, 0, 0, 1e-2},
  {DiffA_op, 15, 1, 37, 1, 1, 0, 0, 1e-2},
  {Hessenberg_op, 10, 15, 15, 15, 15, 0, 0, 1e-2},
  {Hessenberg_op, 30, 61, 61, 61, 61, 0, 0, 1e-2},
  // {Schur_op, 7, 12, 12, 12, 12, 0, 0, 1e-2},
  // {Schur_op, 17, 77, 77, 77, 77, 0, 0, 1e-2},
  // {Lyapunov_op, 5, 14, 14, 14, 14, 0, 0, 1e-2},
  // {Lyapunov_op, 13, 100, 100, 100, 100, 0, 0, 1e-2}
};

// Note: Schur and Lyapunov operations don't give good precision for
// single-precision floating-point numbers yet...

using BatchedMatrixTestD = MatrixTest<double>;
using BatchedMatrixTestF = MatrixTest<float>;
TEST_P(BatchedMatrixTestD, Result)
{
  ASSERT_TRUE(MLCommon::devArrMatchHost(res_h.data(),
                                        res_bM->raw_data(),
                                        res_h.size(),
                                        MLCommon::CompareApprox<double>(params.tolerance),
                                        stream));
}
TEST_P(BatchedMatrixTestF, Result)
{
  ASSERT_TRUE(MLCommon::devArrMatchHost(res_h.data(),
                                        res_bM->raw_data(),
                                        res_h.size(),
                                        MLCommon::CompareApprox<float>(params.tolerance),
                                        stream));
}

INSTANTIATE_TEST_CASE_P(BatchedMatrixTests, BatchedMatrixTestD, ::testing::ValuesIn(inputsd));
INSTANTIATE_TEST_CASE_P(BatchedMatrixTests, BatchedMatrixTestF, ::testing::ValuesIn(inputsf));

}  // namespace Batched
}  // namespace LinAlg
}  // namespace MLCommon
