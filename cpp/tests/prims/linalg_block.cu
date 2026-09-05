/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_utils.h"

#include <cuml/common/logger.hpp>
#include <cuml/common/utils.hpp>

#include <raft/core/handle.hpp>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>
#include <linalg/block.cuh>

#include <random>
#include <vector>

namespace MLCommon {
namespace LinAlg {

using namespace std;

/* GEMM */

template <typename T>
struct BlockGemmInputs {
  int m, k, n;
  bool transa, transb;
  int batch_size;
  int vec_len;
  T eps;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const BlockGemmInputs<T>& dims)
{
  return os;
}

template <typename Policy, typename T>
CUML_KERNEL void block_gemm_test_kernel(
  bool transa, bool transb, int m, int n, int k, T alpha, const T* a, const T* b, T* c)
{
  __shared__ MLCommon::LinAlg::GemmStorage<Policy, T> gemm_storage;

  _block_gemm<Policy>(transa,
                      transb,
                      m,
                      n,
                      k,
                      alpha,
                      a + m * k * blockIdx.x,
                      b + k * n * blockIdx.x,
                      c + m * n * blockIdx.x,
                      gemm_storage);
}

template <typename Policy, typename T>
class BlockGemmTest : public ::testing::TestWithParam<BlockGemmInputs<T>> {
 protected:
  void basicTest()
  {
    raft::handle_t handle;

    params = ::testing::TestWithParam<BlockGemmInputs<T>>::GetParam();

    rmm::device_uvector<T> a(params.m * params.k * params.batch_size, handle.get_stream());
    rmm::device_uvector<T> b(params.k * params.n * params.batch_size, handle.get_stream());
    rmm::device_uvector<T> c(params.m * params.n * params.batch_size, handle.get_stream().get());

    std::vector<T> h_a(params.m * params.k * params.batch_size);
    std::vector<T> h_b(params.k * params.n * params.batch_size);
    std::vector<T> h_c_ref(params.m * params.n * params.batch_size);

    /* Generate random data on device */
    raft::random::Rng r(params.seed);
    r.uniform(a.data(), params.m * params.k * params.batch_size, (T)-2, (T)2, handle.get_stream().get());
    r.uniform(b.data(), params.k * params.n * params.batch_size, (T)-2, (T)2, handle.get_stream().get());

    /* Generate random alpha */
    std::default_random_engine generator(params.seed);
    std::uniform_real_distribution<T> distribution(-2.0, 2.0);
    T alpha = distribution(generator);

    /* Copy to host */
    raft::update_host(
      h_a.data(), a.data(), params.m * params.k * params.batch_size, handle.get_stream().get());
    raft::update_host(
      h_b.data(), b.data(), params.k * params.n * params.batch_size, handle.get_stream().get());
    handle.sync_stream(handle.get_stream().get());

    /* Compute using tested prims */
    block_gemm_test_kernel<Policy>
      <<<params.batch_size, Policy::BlockSize, 0, handle.get_stream().get()>>>(params.transa,
                                                                         params.transb,
                                                                         params.m,
                                                                         params.n,
                                                                         params.k,
                                                                         alpha,
                                                                         a.data(),
                                                                         b.data(),
                                                                         c.data());

    /* Compute reference results */
    for (int bid = 0; bid < params.batch_size; bid++) {
      for (int i = 0; i < params.m; i++) {
        for (int j = 0; j < params.n; j++) {
          T acc = (T)0;
          for (int h = 0; h < params.k; h++) {
            T _a = params.transa ? h_a[bid * params.m * params.k + i * params.k + h]
                                 : h_a[bid * params.m * params.k + h * params.m + i];
            T _b = params.transb ? h_b[bid * params.k * params.n + h * params.n + j]
                                 : h_b[bid * params.k * params.n + j * params.k + h];
            acc += _a * _b;
          }

          h_c_ref[bid * params.m * params.n + j * params.m + i] = alpha * acc;
        }
      }
    }

    /* Check results */
    match = devArrMatchHost(h_c_ref.data(),
                            c.data(),
                            params.m * params.n * params.batch_size,
                            MLCommon::CompareApprox<T>(params.eps),
                            handle.get_stream().get());
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  BlockGemmInputs<T> params;

  testing::AssertionResult match = testing::AssertionFailure();
};

const std::vector<BlockGemmInputs<float>> gemm_inputsf = {
  {42, 42, 42, false, false, 20, 1, 1e-4, 12345U},
  {65, 10, 20, false, true, 50, 1, 1e-4, 12345U},
  {5, 80, 31, true, false, 80, 1, 1e-4, 12345U},
  {11, 50, 41, true, true, 100, 1, 1e-4, 12345U},
};

const std::vector<BlockGemmInputs<double>> gemm_inputsd = {
  {42, 42, 42, false, false, 20, 1, 1e-4, 12345U},
  {65, 10, 20, false, true, 50, 1, 1e-4, 12345U},
  {5, 80, 31, true, false, 80, 1, 1e-4, 12345U},
  {11, 50, 41, true, true, 100, 1, 1e-4, 12345U},
};

const std::vector<BlockGemmInputs<float>> gemm_inputsf_vec2 = {
  {30, 34, 16, false, false, 20, 2, 1e-4, 12345U},
  {10, 42, 20, false, true, 20, 2, 1e-4, 12345U},
  {14, 8, 22, true, false, 20, 2, 1e-4, 12345U},
  {56, 72, 28, true, true, 20, 2, 1e-4, 12345U},
};

const std::vector<BlockGemmInputs<double>> gemm_inputsd_vec2 = {
  {30, 34, 16, false, false, 20, 2, 1e-4, 12345U},
  {10, 42, 20, false, true, 20, 2, 1e-4, 12345U},
  {14, 8, 22, true, false, 20, 2, 1e-4, 12345U},
  {56, 72, 28, true, true, 20, 2, 1e-4, 12345U},
};

typedef BlockGemmTest<BlockGemmPolicy<1, 16, 1, 4, 16, 4>, float> BlockGemmTestF_1_16_1_4_16_4;
TEST_P(BlockGemmTestF_1_16_1_4_16_4, Result) { EXPECT_TRUE(match); }

typedef BlockGemmTest<BlockGemmPolicy<1, 16, 1, 4, 16, 4>, double> BlockGemmTestD_1_16_1_4_16_4;
TEST_P(BlockGemmTestD_1_16_1_4_16_4, Result) { EXPECT_TRUE(match); }

typedef BlockGemmTest<BlockGemmPolicy<1, 32, 1, 4, 32, 8>, float> BlockGemmTestF_1_32_1_4_32_8;
TEST_P(BlockGemmTestF_1_32_1_4_32_8, Result) { EXPECT_TRUE(match); }

typedef BlockGemmTest<BlockGemmPolicy<1, 32, 1, 4, 32, 8>, double> BlockGemmTestD_1_32_1_4_32_8;
TEST_P(BlockGemmTestD_1_32_1_4_32_8, Result) { EXPECT_TRUE(match); }

typedef BlockGemmTest<BlockGemmPolicy<1, 32, 1, 16, 64, 4>, float> BlockGemmTestF_1_32_1_16_64_4;
TEST_P(BlockGemmTestF_1_32_1_16_64_4, Result) { EXPECT_TRUE(match); }

typedef BlockGemmTest<BlockGemmPolicy<1, 32, 1, 16, 64, 4>, double> BlockGemmTestD_1_32_1_16_64_4;
TEST_P(BlockGemmTestD_1_32_1_16_64_4, Result) { EXPECT_TRUE(match); }

typedef BlockGemmTest<BlockGemmPolicy<1, 16, 1, 16, 128, 2>, float> BlockGemmTestF_1_16_1_16_128_2;
TEST_P(BlockGemmTestF_1_16_1_16_128_2, Result) { EXPECT_TRUE(match); }

typedef BlockGemmTest<BlockGemmPolicy<1, 16, 1, 16, 128, 2>, double> BlockGemmTestD_1_16_1_16_128_2;
TEST_P(BlockGemmTestD_1_16_1_16_128_2, Result) { EXPECT_TRUE(match); }

typedef BlockGemmTest<BlockGemmPolicy<2, 32, 2, 2, 16, 16>, float> BlockGemmTestF_2_32_2_2_16_16;
TEST_P(BlockGemmTestF_2_32_2_2_16_16, Result) { EXPECT_TRUE(match); }

typedef BlockGemmTest<BlockGemmPolicy<2, 32, 2, 2, 16, 16>, double> BlockGemmTestD_2_32_2_2_16_16;
TEST_P(BlockGemmTestD_2_32_2_2_16_16, Result) { EXPECT_TRUE(match); }

INSTANTIATE_TEST_CASE_P(BlockGemmTests,
                        BlockGemmTestF_1_16_1_4_16_4,
                        ::testing::ValuesIn(gemm_inputsf));

INSTANTIATE_TEST_CASE_P(BlockGemmTests,
                        BlockGemmTestD_1_16_1_4_16_4,
                        ::testing::ValuesIn(gemm_inputsd));

INSTANTIATE_TEST_CASE_P(BlockGemmTests,
                        BlockGemmTestF_1_32_1_4_32_8,
                        ::testing::ValuesIn(gemm_inputsf));

INSTANTIATE_TEST_CASE_P(BlockGemmTests,
                        BlockGemmTestD_1_32_1_4_32_8,
                        ::testing::ValuesIn(gemm_inputsd));

INSTANTIATE_TEST_CASE_P(BlockGemmTests,
                        BlockGemmTestF_1_32_1_16_64_4,
                        ::testing::ValuesIn(gemm_inputsf));

INSTANTIATE_TEST_CASE_P(BlockGemmTests,
                        BlockGemmTestD_1_32_1_16_64_4,
                        ::testing::ValuesIn(gemm_inputsd));

INSTANTIATE_TEST_CASE_P(BlockGemmTests,
                        BlockGemmTestF_1_16_1_16_128_2,
                        ::testing::ValuesIn(gemm_inputsf));

INSTANTIATE_TEST_CASE_P(BlockGemmTests,
                        BlockGemmTestD_1_16_1_16_128_2,
                        ::testing::ValuesIn(gemm_inputsd));

INSTANTIATE_TEST_CASE_P(BlockGemmTests,
                        BlockGemmTestF_2_32_2_2_16_16,
                        ::testing::ValuesIn(gemm_inputsf_vec2));

INSTANTIATE_TEST_CASE_P(BlockGemmTests,
                        BlockGemmTestD_2_32_2_2_16_16,
                        ::testing::ValuesIn(gemm_inputsd_vec2));

/* GEMV */

template <typename T>
struct BlockGemvInputs {
  bool preload;
  int m, n;
  int batch_size;
  T eps;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const BlockGemvInputs<T>& dims)
{
  return os;
}

template <typename Policy, typename T>
CUML_KERNEL void block_gemv_test_kernel(
  int m, int n, T alpha, const T* a, const T* x, T* y, bool preload)
{
  __shared__ MLCommon::LinAlg::GemvStorage<Policy, T> gemv_storage;

  extern __shared__ char dyna_shared_mem[];
  T* shared_vec = (T*)dyna_shared_mem;

  if (preload) {
    _block_gemv<Policy, true>(m,
                              n,
                              alpha,
                              a + m * n * blockIdx.x,
                              x + n * blockIdx.x,
                              y + m * blockIdx.x,
                              gemv_storage,
                              shared_vec);
  } else {
    for (int i = threadIdx.x; i < n; i += Policy::BlockSize) {
      shared_vec[i] = x[n * blockIdx.x + i];
    }
    __syncthreads();

    _block_gemv<Policy, false>(
      m, n, alpha, a + m * n * blockIdx.x, shared_vec, y + m * blockIdx.x, gemv_storage);
  }
}

template <typename Policy, typename T>
class BlockGemvTest : public ::testing::TestWithParam<BlockGemvInputs<T>> {
 protected:
  void basicTest()
  {
    raft::handle_t handle;

    params = ::testing::TestWithParam<BlockGemvInputs<T>>::GetParam();

    rmm::device_uvector<T> a(params.m * params.n * params.batch_size, handle.get_stream());
    rmm::device_uvector<T> x(params.n * params.batch_size, handle.get_stream());
    rmm::device_uvector<T> y(params.m * params.batch_size, handle.get_stream().get());

    std::vector<T> h_a(params.m * params.n * params.batch_size);
    std::vector<T> h_x(params.n * params.batch_size);
    std::vector<T> h_y_ref(params.m * params.batch_size);

    /* Generate random data on device */
    raft::random::Rng r(params.seed);
    r.uniform(a.data(), params.m * params.n * params.batch_size, (T)-2, (T)2, handle.get_stream().get());
    r.uniform(x.data(), params.n * params.batch_size, (T)-2, (T)2, handle.get_stream().get());

    /* Generate random alpha */
    std::default_random_engine generator(params.seed);
    std::uniform_real_distribution<T> distribution(-2.0, 2.0);
    T alpha = distribution(generator);

    /* Copy to host */
    raft::update_host(
      h_a.data(), a.data(), params.m * params.n * params.batch_size, handle.get_stream().get());
    raft::update_host(h_x.data(), x.data(), params.n * params.batch_size, handle.get_stream().get());
    handle.sync_stream(handle.get_stream().get());

    /* Compute using tested prims */
    int shared_mem_size = params.n * sizeof(T);
    block_gemv_test_kernel<Policy>
      <<<params.batch_size, Policy::BlockSize, shared_mem_size, handle.get_stream().get()>>>(
        params.m, params.n, alpha, a.data(), x.data(), y.data(), params.preload);

    /* Compute reference results */
    for (int bid = 0; bid < params.batch_size; bid++) {
      for (int i = 0; i < params.m; i++) {
        T acc = (T)0;
        for (int j = 0; j < params.n; j++) {
          acc += h_a[bid * params.m * params.n + j * params.m + i] * h_x[bid * params.n + j];
        }
        h_y_ref[bid * params.m + i] = alpha * acc;
      }
    }

    /* Check results */
    match = devArrMatchHost(h_y_ref.data(),
                            y.data(),
                            params.m * params.batch_size,
                            MLCommon::CompareApprox<T>(params.eps),
                            handle.get_stream().get());
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  BlockGemvInputs<T> params;

  testing::AssertionResult match = testing::AssertionFailure();
};

const std::vector<BlockGemvInputs<float>> gemv_inputsf = {{true, 42, 42, 20, 1e-4, 12345U},
                                                          {true, 65, 10, 50, 1e-4, 12345U},
                                                          {false, 5, 80, 100, 1e-4, 12345U}};

const std::vector<BlockGemvInputs<double>> gemv_inputsd = {{true, 42, 42, 20, 1e-4, 12345U},
                                                           {true, 65, 10, 50, 1e-4, 12345U},
                                                           {false, 5, 80, 100, 1e-4, 12345U}};

typedef BlockGemvTest<BlockGemvPolicy<16, 4>, float> BlockGemvTestF_16_4;
TEST_P(BlockGemvTestF_16_4, Result) { EXPECT_TRUE(match); }

typedef BlockGemvTest<BlockGemvPolicy<16, 4>, double> BlockGemvTestD_16_4;
TEST_P(BlockGemvTestD_16_4, Result) { EXPECT_TRUE(match); }

typedef BlockGemvTest<BlockGemvPolicy<32, 8>, float> BlockGemvTestF_32_8;
TEST_P(BlockGemvTestF_32_8, Result) { EXPECT_TRUE(match); }

typedef BlockGemvTest<BlockGemvPolicy<32, 8>, double> BlockGemvTestD_32_8;
TEST_P(BlockGemvTestD_32_8, Result) { EXPECT_TRUE(match); }

typedef BlockGemvTest<BlockGemvPolicy<128, 2>, float> BlockGemvTestF_128_2;
TEST_P(BlockGemvTestF_128_2, Result) { EXPECT_TRUE(match); }

typedef BlockGemvTest<BlockGemvPolicy<128, 2>, double> BlockGemvTestD_128_2;
TEST_P(BlockGemvTestD_128_2, Result) { EXPECT_TRUE(match); }

INSTANTIATE_TEST_CASE_P(BlockGemvTests, BlockGemvTestF_16_4, ::testing::ValuesIn(gemv_inputsf));

INSTANTIATE_TEST_CASE_P(BlockGemvTests, BlockGemvTestD_16_4, ::testing::ValuesIn(gemv_inputsd));

INSTANTIATE_TEST_CASE_P(BlockGemvTests, BlockGemvTestF_32_8, ::testing::ValuesIn(gemv_inputsf));

INSTANTIATE_TEST_CASE_P(BlockGemvTests, BlockGemvTestD_32_8, ::testing::ValuesIn(gemv_inputsd));

INSTANTIATE_TEST_CASE_P(BlockGemvTests, BlockGemvTestF_128_2, ::testing::ValuesIn(gemv_inputsf));

INSTANTIATE_TEST_CASE_P(BlockGemvTests, BlockGemvTestD_128_2, ::testing::ValuesIn(gemv_inputsd));

/* DOT */

template <typename T>
struct BlockDotInputs {
  bool broadcast;
  int n;
  int batch_size;
  T eps;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const BlockDotInputs<T>& dims)
{
  return os;
}

template <int BlockSize, bool Broadcast, typename T>
CUML_KERNEL void block_dot_test_kernel(int n, const T* x, const T* y, T* d_dot)
{
  __shared__ ReductionStorage<BlockSize, T> reduction_storage;

  T dot_ =
    _block_dot<BlockSize, Broadcast>(n, x + n * blockIdx.x, y + n * blockIdx.x, reduction_storage);

  if (!Broadcast && threadIdx.x == 0)
    d_dot[blockIdx.x] = dot_;
  else if (Broadcast && threadIdx.x == BlockSize - 1)
    d_dot[blockIdx.x] = dot_;
}

template <typename T>
class BlockDotTest : public ::testing::TestWithParam<BlockDotInputs<T>> {
 protected:
  void basicTest()
  {
    raft::handle_t handle;

    params = ::testing::TestWithParam<BlockDotInputs<T>>::GetParam();

    rmm::device_uvector<T> x(params.n * params.batch_size, handle.get_stream());
    rmm::device_uvector<T> y(params.n * params.batch_size, handle.get_stream());
    rmm::device_uvector<T> dot_dev(params.batch_size, handle.get_stream().get());

    std::vector<T> h_x(params.n * params.batch_size);
    std::vector<T> h_y(params.n * params.batch_size);
    std::vector<T> h_dot_ref(params.batch_size, (T)0);

    /* Generate random data on device */
    raft::random::Rng r(params.seed);
    r.uniform(x.data(), params.n * params.batch_size, (T)-2, (T)2, handle.get_stream().get());
    r.uniform(y.data(), params.n * params.batch_size, (T)-2, (T)2, handle.get_stream().get());

    /* Copy to host */
    raft::update_host(h_x.data(), x.data(), params.n * params.batch_size, handle.get_stream().get());
    raft::update_host(h_y.data(), y.data(), params.n * params.batch_size, handle.get_stream().get());
    handle.sync_stream(handle.get_stream().get());

    /* Compute using tested prims */
    constexpr int BlockSize = 64;
    if (params.broadcast)
      block_dot_test_kernel<BlockSize, true>
        <<<params.batch_size, BlockSize, 0, handle.get_stream().get()>>>(
          params.n, x.data(), y.data(), dot_dev.data());
    else
      block_dot_test_kernel<BlockSize, false>
        <<<params.batch_size, BlockSize, 0, handle.get_stream().get()>>>(
          params.n, x.data(), y.data(), dot_dev.data());

    /* Compute reference results */
    for (int bid = 0; bid < params.batch_size; bid++) {
      for (int i = 0; i < params.n; i++) {
        h_dot_ref[bid] += h_x[bid * params.n + i] * h_y[bid * params.n + i];
      }
    }

    /* Check results */
    match = devArrMatchHost(h_dot_ref.data(),
                            dot_dev.data(),
                            params.batch_size,
                            MLCommon::CompareApprox<T>(params.eps),
                            handle.get_stream().get());
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  BlockDotInputs<T> params;

  testing::AssertionResult match = testing::AssertionFailure();
};

const std::vector<BlockDotInputs<float>> dot_inputsf = {{true, 9, 20, 1e-4, 12345U},
                                                        {true, 65, 50, 1e-4, 12345U},
                                                        {true, 200, 100, 1e-4, 12345U},
                                                        {false, 200, 100, 1e-4, 12345U}};

const std::vector<BlockDotInputs<double>> dot_inputsd = {{true, 9, 20, 1e-4, 12345U},
                                                         {true, 65, 50, 1e-4, 12345U},
                                                         {true, 200, 100, 1e-4, 12345U},
                                                         {false, 200, 100, 1e-4, 12345U}};

typedef BlockDotTest<float> BlockDotTestF;
TEST_P(BlockDotTestF, Result) { EXPECT_TRUE(match); }

typedef BlockDotTest<double> BlockDotTestD;
TEST_P(BlockDotTestD, Result) { EXPECT_TRUE(match); }

INSTANTIATE_TEST_CASE_P(BlockDotTests, BlockDotTestF, ::testing::ValuesIn(dot_inputsf));

INSTANTIATE_TEST_CASE_P(BlockDotTests, BlockDotTestD, ::testing::ValuesIn(dot_inputsd));

/* x*A*x' */

template <typename T>
struct BlockXaxtInputs {
  bool broadcast;
  bool preload;
  int n;
  int batch_size;
  T eps;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const BlockXaxtInputs<T>& dims)
{
  return os;
}

template <int BlockSize, bool Broadcast, typename T>
CUML_KERNEL void block_xAxt_test_kernel(int n, const T* x, const T* A, T* d_res, bool preload)
{
  extern __shared__ char dyna_shared_mem[];
  T* shared_vec = (T*)dyna_shared_mem;
  __shared__ ReductionStorage<BlockSize, T> reduction_storage;

  T res_;
  if (preload) {
    res_ = _block_xAxt<BlockSize, Broadcast, true>(
      n, x + n * blockIdx.x, A + n * n * blockIdx.x, reduction_storage, shared_vec);
  } else {
    for (int i = threadIdx.x; i < n; i += BlockSize) {
      shared_vec[i] = x[n * blockIdx.x + i];
    }
    __syncthreads();

    res_ = _block_xAxt<BlockSize, Broadcast, false>(
      n, shared_vec, A + n * n * blockIdx.x, reduction_storage);
  }

  if (!Broadcast && threadIdx.x == 0)
    d_res[blockIdx.x] = res_;
  else if (Broadcast && threadIdx.x == BlockSize - 1)
    d_res[blockIdx.x] = res_;
}

template <typename T>
class BlockXaxtTest : public ::testing::TestWithParam<BlockXaxtInputs<T>> {
 protected:
  void basicTest()
  {
    raft::handle_t handle;

    params = ::testing::TestWithParam<BlockXaxtInputs<T>>::GetParam();

    rmm::device_uvector<T> x(params.n * params.batch_size, handle.get_stream());
    rmm::device_uvector<T> A(params.n * params.n * params.batch_size, handle.get_stream());
    rmm::device_uvector<T> res_dev(params.batch_size, handle.get_stream().get());

    std::vector<T> h_x(params.n * params.batch_size);
    std::vector<T> h_A(params.n * params.n * params.batch_size);
    std::vector<T> h_res_ref(params.batch_size, (T)0);

    /* Generate random data on device */
    raft::random::Rng r(params.seed);
    r.uniform(x.data(), params.n * params.batch_size, (T)-2, (T)2, handle.get_stream().get());
    r.uniform(A.data(), params.n * params.n * params.batch_size, (T)-2, (T)2, handle.get_stream().get());

    /* Copy to host */
    raft::update_host(h_x.data(), x.data(), params.n * params.batch_size, handle.get_stream().get());
    raft::update_host(
      h_A.data(), A.data(), params.n * params.n * params.batch_size, handle.get_stream().get());
    handle.sync_stream(handle.get_stream().get());

    /* Compute using tested prims */
    constexpr int BlockSize = 64;
    int shared_mem_size     = params.n * sizeof(T);
    if (params.broadcast)
      block_xAxt_test_kernel<BlockSize, true>
        <<<params.batch_size, BlockSize, shared_mem_size, handle.get_stream().get()>>>(
          params.n, x.data(), A.data(), res_dev.data(), params.preload);
    else
      block_xAxt_test_kernel<BlockSize, false>
        <<<params.batch_size, BlockSize, shared_mem_size, handle.get_stream().get()>>>(
          params.n, x.data(), A.data(), res_dev.data(), params.preload);

    /* Compute reference results */
    for (int bid = 0; bid < params.batch_size; bid++) {
      for (int i = 0; i < params.n; i++) {
        T acc = 0;
        for (int j = 0; j < params.n; j++) {
          acc += h_A[bid * params.n * params.n + j * params.n + i] * h_x[bid * params.n + j];
        }
        h_res_ref[bid] += acc * h_x[bid * params.n + i];
      }
    }

    /* Check results */
    match = devArrMatchHost(h_res_ref.data(),
                            res_dev.data(),
                            params.batch_size,
                            MLCommon::CompareApprox<T>(params.eps),
                            handle.get_stream().get());
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  BlockXaxtInputs<T> params;

  testing::AssertionResult match = testing::AssertionFailure();
};

const std::vector<BlockXaxtInputs<float>> xAxt_inputsf = {{true, true, 9, 20, 1e-2, 12345U},
                                                          {true, true, 65, 50, 1e-2, 12345U},
                                                          {true, true, 200, 100, 1e-2, 12345U},
                                                          {false, true, 200, 100, 1e-2, 12345U},
                                                          {true, false, 200, 100, 1e-2, 12345U}};

const std::vector<BlockXaxtInputs<double>> xAxt_inputsd = {{true, true, 9, 20, 1e-4, 12345U},
                                                           {true, true, 65, 50, 1e-4, 12345U},
                                                           {true, true, 200, 100, 1e-4, 12345U},
                                                           {false, true, 200, 100, 1e-4, 12345U},
                                                           {true, false, 200, 100, 1e-2, 12345U}};

typedef BlockXaxtTest<float> BlockXaxtTestF;
TEST_P(BlockXaxtTestF, Result) { EXPECT_TRUE(match); }

typedef BlockXaxtTest<double> BlockXaxtTestD;
TEST_P(BlockXaxtTestD, Result) { EXPECT_TRUE(match); }

INSTANTIATE_TEST_CASE_P(BlockXaxtTests, BlockXaxtTestF, ::testing::ValuesIn(xAxt_inputsf));

INSTANTIATE_TEST_CASE_P(BlockXaxtTests, BlockXaxtTestD, ::testing::ValuesIn(xAxt_inputsd));

/* y=alpha*x */

template <typename T>
struct BlockAxInputs {
  int n;
  int batch_size;
  T eps;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const BlockAxInputs<T>& dims)
{
  return os;
}

template <typename T>
CUML_KERNEL void block_ax_test_kernel(int n, T alpha, const T* x, T* y)
{
  _block_ax(n, alpha, x + n * blockIdx.x, y + n * blockIdx.x);
}

template <typename T>
class BlockAxTest : public ::testing::TestWithParam<BlockAxInputs<T>> {
 protected:
  void basicTest()
  {
    raft::handle_t handle;

    params = ::testing::TestWithParam<BlockAxInputs<T>>::GetParam();

    rmm::device_uvector<T> x(params.n * params.batch_size, handle.get_stream().get());
    rmm::device_uvector<T> y(params.n * params.batch_size, handle.get_stream().get());

    std::vector<T> h_x(params.n * params.batch_size);
    std::vector<T> h_y_ref(params.n * params.batch_size, (T)0);

    /* Generate random data on device */
    raft::random::Rng r(params.seed);
    r.uniform(x.data(), params.n * params.batch_size, (T)-2, (T)2, handle.get_stream().get());

    /* Generate random alpha */
    std::default_random_engine generator(params.seed);
    std::uniform_real_distribution<T> distribution(-2.0, 2.0);
    T alpha = distribution(generator);

    /* Copy to host */
    raft::update_host(h_x.data(), x.data(), params.n * params.batch_size, handle.get_stream().get());
    handle.sync_stream(handle.get_stream().get());

    /* Compute using tested prims */
    constexpr int BlockSize = 64;
    block_ax_test_kernel<<<params.batch_size, BlockSize, 0, handle.get_stream().get()>>>(
      params.n, alpha, x.data(), y.data());

    /* Compute reference results */
    for (int bid = 0; bid < params.batch_size; bid++) {
      for (int i = 0; i < params.n; i++) {
        h_y_ref[bid * params.n + i] = alpha * h_x[bid * params.n + i];
      }
    }

    /* Check results */
    match = devArrMatchHost(h_y_ref.data(),
                            y.data(),
                            params.n * params.batch_size,
                            MLCommon::CompareApprox<T>(params.eps),
                            handle.get_stream().get());
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  BlockAxInputs<T> params;

  testing::AssertionResult match = testing::AssertionFailure();
};

const std::vector<BlockAxInputs<float>> ax_inputsf = {
  {9, 20, 1e-4, 12345U}, {65, 50, 1e-4, 12345U}, {200, 100, 1e-4, 12345U}};

const std::vector<BlockAxInputs<double>> ax_inputsd = {
  {9, 20, 1e-4, 12345U}, {65, 50, 1e-4, 12345U}, {200, 100, 1e-4, 12345U}};

typedef BlockAxTest<float> BlockAxTestF;
TEST_P(BlockAxTestF, Result) { EXPECT_TRUE(match); }

typedef BlockAxTest<double> BlockAxTestD;
TEST_P(BlockAxTestD, Result) { EXPECT_TRUE(match); }

INSTANTIATE_TEST_CASE_P(BlockAxTests, BlockAxTestF, ::testing::ValuesIn(ax_inputsf));

INSTANTIATE_TEST_CASE_P(BlockAxTests, BlockAxTestD, ::testing::ValuesIn(ax_inputsd));

/* Covariance stability */

template <typename T>
struct BlockCovStabilityInputs {
  int n;
  int batch_size;
  T eps;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const BlockCovStabilityInputs<T>& dims)
{
  return os;
}

template <typename CovPolicy, typename T>
CUML_KERNEL void block_cov_stability_test_kernel(int n, const T* in, T* out)
{
  __shared__ CovStabilityStorage<CovPolicy, T> cov_stability_storage;
  _block_covariance_stability<CovPolicy>(
    n, in + n * n * blockIdx.x, out + n * n * blockIdx.x, cov_stability_storage);
}

template <typename CovPolicy, typename T>
class BlockCovStabilityTest : public ::testing::TestWithParam<BlockCovStabilityInputs<T>> {
 protected:
  void basicTest()
  {
    raft::handle_t handle;

    params = ::testing::TestWithParam<BlockCovStabilityInputs<T>>::GetParam();

    rmm::device_uvector<T> d_in(params.n * params.n * params.batch_size, handle.get_stream().get());
    rmm::device_uvector<T> d_out(params.n * params.n * params.batch_size, handle.get_stream().get());

    std::vector<T> h_in(params.n * params.n * params.batch_size);
    std::vector<T> h_out(params.n * params.n * params.batch_size);

    /* Generate random data on device */
    raft::random::Rng r(params.seed);
    r.uniform(
      d_in.data(), params.n * params.n * params.batch_size, (T)-2, (T)2, handle.get_stream().get());

    /* Copy to host */
    raft::update_host(
      h_in.data(), d_in.data(), params.n * params.n * params.batch_size, handle.get_stream().get());
    handle.sync_stream(handle.get_stream().get());

    /* Compute using tested prims */
    block_cov_stability_test_kernel<CovPolicy>
      <<<params.batch_size, CovPolicy::BlockSize, 0, handle.get_stream().get()>>>(
        params.n, d_in.data(), d_out.data());

    /* Compute reference results */
    for (int bid = 0; bid < params.batch_size; bid++) {
      for (int i = 0; i < params.n - 1; i++) {
        for (int j = i + 1; j < params.n; j++) {
          T val = 0.5 * (h_in[bid * params.n * params.n + j * params.n + i] +
                         h_in[bid * params.n * params.n + i * params.n + j]);
          h_out[bid * params.n * params.n + j * params.n + i] = val;
          h_out[bid * params.n * params.n + i * params.n + j] = val;
        }
      }
      for (int i = 0; i < params.n; i++) {
        h_out[bid * params.n * params.n + i * params.n + i] =
          abs(h_in[bid * params.n * params.n + i * params.n + i]);
      }
    }

    /* Check results */
    match = devArrMatchHost(h_out.data(),
                            d_out.data(),
                            params.n * params.n * params.batch_size,
                            MLCommon::CompareApprox<T>(params.eps),
                            handle.get_stream().get());
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  BlockCovStabilityInputs<T> params;

  testing::AssertionResult match = testing::AssertionFailure();
};

const std::vector<BlockCovStabilityInputs<float>> cs_inputsf = {
  {15, 4, 1e-4, 12345U},
  {33, 10, 1e-4, 12345U},
  {220, 130, 1e-4, 12345U},
};

const std::vector<BlockCovStabilityInputs<double>> cs_inputsd = {
  {15, 4, 1e-4, 12345U},
  {33, 10, 1e-4, 12345U},
  {220, 130, 1e-4, 12345U},
};

typedef BlockCovStabilityTest<BlockPolicy<1, 1, 8, 4>, float> BlockCovStabilityTestF_1_1_8_4;
TEST_P(BlockCovStabilityTestF_1_1_8_4, Result) { EXPECT_TRUE(match); }

typedef BlockCovStabilityTest<BlockPolicy<1, 1, 8, 4>, double> BlockCovStabilityTestD_1_1_8_4;
TEST_P(BlockCovStabilityTestD_1_1_8_4, Result) { EXPECT_TRUE(match); }

typedef BlockCovStabilityTest<BlockPolicy<1, 4, 32, 8>, float> BlockCovStabilityTestF_1_4_32_8;
TEST_P(BlockCovStabilityTestF_1_4_32_8, Result) { EXPECT_TRUE(match); }

typedef BlockCovStabilityTest<BlockPolicy<1, 4, 32, 8>, double> BlockCovStabilityTestD_1_4_32_8;
TEST_P(BlockCovStabilityTestD_1_4_32_8, Result) { EXPECT_TRUE(match); }

INSTANTIATE_TEST_CASE_P(BlockCovStabilityTests,
                        BlockCovStabilityTestF_1_1_8_4,
                        ::testing::ValuesIn(cs_inputsf));

INSTANTIATE_TEST_CASE_P(BlockCovStabilityTests,
                        BlockCovStabilityTestD_1_1_8_4,
                        ::testing::ValuesIn(cs_inputsd));

INSTANTIATE_TEST_CASE_P(BlockCovStabilityTests,
                        BlockCovStabilityTestF_1_4_32_8,
                        ::testing::ValuesIn(cs_inputsf));

INSTANTIATE_TEST_CASE_P(BlockCovStabilityTests,
                        BlockCovStabilityTestD_1_4_32_8,
                        ::testing::ValuesIn(cs_inputsd));

}  // namespace LinAlg
}  // namespace MLCommon
