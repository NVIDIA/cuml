/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/common/logger.hpp>
#include <cuml/datasets/make_blobs.hpp>
#include <cuml/matrix/kernel_params.hpp>
#include <cuml/svm/svc.hpp>
#include <cuml/svm/svm_model.h>
#include <cuml/svm/svm_parameter.h>
#include <cuml/svm/svr.hpp>

#include <raft/core/math.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/linalg/transpose.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cub/cub.cuh>
#include <cuda/std/tuple>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <cuvs/distance/distance.hpp>
#include <cuvs/distance/grammian.hpp>
#include <gtest/gtest.h>
#include <matrix/kernel_params.hpp>
#include <svm/smoblocksolve.cuh>
#include <svm/smosolver.cuh>
#include <svm/workingset.cuh>
#include <test_utils.h>

#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

namespace ML {
namespace SVM {
using cuvs::distance::kernels::GramMatrixBase;
using cuvs::distance::kernels::KernelFactory;
using ML::matrix::KernelParams;
using ML::matrix::KernelType;

// Initialize device vector C_vec with scalar C
template <typename math_t>
void init_C(math_t C, math_t* C_vec, int n, cudaStream_t stream)
{
  thrust::device_ptr<math_t> c_ptr(C_vec);
  thrust::fill(thrust::cuda::par.on(stream), c_ptr, c_ptr + n, C);
}

template <typename math_t>
class WorkingSetTest : public ::testing::Test {
 public:
  WorkingSetTest()
    : stream(handle.get_stream()),
      f_dev(10, stream),
      y_dev(10, stream),
      C_dev(10, stream),
      alpha_dev(10, stream)
  {
    init_C(C, C_dev.data(), 10, stream);
    raft::update_device(f_dev.data(), f_host, 10, stream);
    raft::update_device(y_dev.data(), y_host, 10, stream);
    raft::update_device(alpha_dev.data(), alpha_host, 10, stream);
  }

 protected:
  raft::handle_t handle;
  cudaStream_t stream = 0;

  WorkingSet<math_t>* ws;

  math_t f_host[10] = {1, 3, 10, 4, 2, 8, 6, 5, 9, 7};
  rmm::device_uvector<math_t> f_dev;

  math_t y_host[10] = {-1, -1, -1, -1, -1, 1, 1, 1, 1, 1};
  rmm::device_uvector<math_t> y_dev;

  rmm::device_uvector<math_t> C_dev;
  math_t C = 1.5;

  math_t alpha_host[10] = {0, 0, 0.1, 0.2, 1.5, 0, 0.2, 0.4, 1.5, 1.5};
  rmm::device_uvector<math_t> alpha_dev;  //   l  l  l/u  l/u    u  u  l/u  l/u  l    l

  int expected_idx[4]  = {4, 3, 8, 2};
  int expected_idx2[4] = {8, 2, 4, 9};
};

typedef ::testing::Types<float, double> FloatTypes;

TYPED_TEST_CASE(WorkingSetTest, FloatTypes);

TYPED_TEST(WorkingSetTest, Init)
{
  auto stream = this->handle.get_stream();
  this->ws    = new WorkingSet<TypeParam>(this->handle, stream, 10);
  EXPECT_EQ(this->ws->GetSize(), 10);
  delete this->ws;

  this->ws = new WorkingSet<TypeParam>(this->handle, stream, 100000);
  EXPECT_EQ(this->ws->GetSize(), 1024);
  delete this->ws;
}

TYPED_TEST(WorkingSetTest, Select)
{
  auto stream = this->handle.get_stream();
  this->ws    = new WorkingSet<TypeParam>(this->handle, stream, 10, 4);
  EXPECT_EQ(this->ws->GetSize(), 4);
  this->ws->SimpleSelect(
    this->f_dev.data(), this->alpha_dev.data(), this->y_dev.data(), this->C_dev.data());
  ASSERT_TRUE(devArrMatchHost(this->expected_idx,
                              this->ws->GetIndices(),
                              this->ws->GetSize(),
                              MLCommon::Compare<int>(),
                              stream));

  this->ws->Select(
    this->f_dev.data(), this->alpha_dev.data(), this->y_dev.data(), this->C_dev.data());
  ASSERT_TRUE(devArrMatchHost(this->expected_idx,
                              this->ws->GetIndices(),
                              this->ws->GetSize(),
                              MLCommon::Compare<int>(),
                              stream));
  this->ws->Select(
    this->f_dev.data(), this->alpha_dev.data(), this->y_dev.data(), this->C_dev.data());

  ASSERT_TRUE(devArrMatchHost(this->expected_idx2,
                              this->ws->GetIndices(),
                              this->ws->GetSize(),
                              MLCommon::Compare<int>(),
                              stream));
  delete this->ws;
}

// TYPED_TEST(WorkingSetTest, Priority) {
// See Issue #946
//}

struct KernelCacheTestInput {
  bool sparse;
  bool batching;
  bool sparse_compute;
};

std::ostream& operator<<(std::ostream& os, const KernelCacheTestInput& b)
{
  os << "sparse=" << b.sparse << ", batching=" << b.batching
     << ", sparse_compute=" << b.sparse_compute;
  return os;
}

template <typename math_t>
class KernelCacheTest : public ::testing::Test {
 public:
  KernelCacheTest()
    : stream(handle.get_stream()),
      n_rows(4),
      n_cols(2),
      n_ws(3),
      x_dev(n_rows * n_cols, stream),
      x_indptr_dev(n_rows + 1, stream),
      x_indices_dev(n_rows * n_cols, stream),
      x_data_dev(n_rows * n_cols, stream),
      ws_idx_dev(2 * n_ws, stream)
  {
    raft::update_device(x_dev.data(), x_host, n_rows * n_cols, stream);
    raft::update_device(x_indptr_dev.data(), x_host_indptr, n_rows + 1, stream);
    raft::update_device(x_indices_dev.data(), x_host_indices, n_rows * n_cols, stream);
    raft::update_device(x_data_dev.data(), x_host_data, n_rows * n_cols, stream);
    raft::update_device(ws_idx_dev.data(), ws_idx_host, n_ws, stream);
  }

 protected:
  // Naive host side kernel implementation used for comparison
  void ApplyNonlin(KernelParams params)
  {
    switch (params.kernel) {
      case KernelType::LINEAR: break;
      case KernelType::PRECOMPUTED: break;  // No transformation needed
      case KernelType::POLYNOMIAL:
        for (int z = 0; z < n_rows * n_ws; z++) {
          math_t val            = params.gamma * tile_host_expected[z] + params.coef0;
          tile_host_expected[z] = pow(val, params.degree);
        }
        break;
      case KernelType::TANH:
        for (int z = 0; z < n_rows * n_ws; z++) {
          math_t val            = params.gamma * tile_host_expected[z] + params.coef0;
          tile_host_expected[z] = tanh(val);
        }
        break;
      case KernelType::RBF:
        for (int i = 0; i < n_ws; i++) {
          for (int j = 0; j < n_rows; j++) {
            math_t d = 0;
            for (int k = 0; k < n_cols; k++) {
              int idx_i   = ws_idx_host[i];
              math_t diff = x_host[idx_i + k * n_rows] - x_host[j + k * n_rows];
              d += diff * diff;
            }
            tile_host_expected[i * n_rows + j] = exp(-params.gamma * d);
          }
        }
        break;
    }
  }

  void check(math_t* kernel_data, int* nz_da_idx, int nnz_da, int batch_size, int offset)
  {
    auto stream = this->handle.get_stream();
    std::vector<int> ws_idx_h(nnz_da);
    raft::update_host(ws_idx_h.data(), nz_da_idx, nnz_da, stream);
    handle.sync_stream(stream);
    // Note: kernel cache can permute the working set, so we have to look
    // up which rows we compare
    for (int i = 0; i < nnz_da; i++) {
      SCOPED_TRACE(i);
      const math_t* cache_row = kernel_data + i * batch_size;
      const math_t* row_exp   = tile_host_all + ws_idx_h[i] * this->n_rows + offset;
      EXPECT_TRUE(devArrMatchHost(
        row_exp, cache_row, batch_size, MLCommon::CompareApprox<math_t>(1e-6f), stream));
    }
  }

  raft::handle_t handle;
  cudaStream_t stream = 0;

  int n_rows;  // =4
  int n_cols;  // =2
  int n_ws;

  rmm::device_uvector<math_t> x_dev;
  rmm::device_uvector<int> ws_idx_dev;

  math_t x_host[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  // csr representation
  int x_host_indptr[5]  = {0, 2, 4, 6, 8};
  int x_host_indices[8] = {0, 1, 0, 1, 0, 1, 0, 1};
  math_t x_host_data[8] = {1, 5, 2, 6, 3, 7, 4, 8};
  rmm::device_uvector<int> x_indptr_dev;
  rmm::device_uvector<int> x_indices_dev;
  rmm::device_uvector<math_t> x_data_dev;

  int ws_idx_host[4]            = {0, 1, 3};
  math_t tile_host_expected[12] = {26, 32, 38, 44, 32, 40, 48, 56, 44, 56, 68, 80};
  math_t tile_host_all[16]      = {26, 32, 38, 44, 32, 40, 48, 56, 38, 48, 58, 68, 44, 56, 68, 80};
};

TYPED_TEST_CASE_P(KernelCacheTest);

TYPED_TEST_P(KernelCacheTest, EvalTest)
{
  auto stream = this->handle.get_stream();
  std::vector<KernelParams> param_vec{{KernelType::LINEAR, 3, 1, 0},
                                      {KernelType::POLYNOMIAL, 2, 1.3, 1},
                                      {KernelType::TANH, 2, 0.5, 2.4},
                                      {KernelType::RBF, 2, 0.5, 0}};
  float cache_size = 0;

  auto dense_view =
    raft::make_device_strided_matrix_view<TypeParam, int, raft::layout_f_contiguous>(
      this->x_dev.data(), this->n_rows, this->n_cols, 0);

  for (auto params : param_vec) {
    GramMatrixBase<TypeParam>* kernel =
      KernelFactory<TypeParam>::create(ML::matrix::to_cuvs(params));
    KernelCache<TypeParam, raft::device_matrix_view<TypeParam, int, raft::layout_stride>> cache(
      this->handle,
      dense_view,
      this->n_rows,
      this->n_cols,
      this->n_ws,
      kernel,
      static_cast<cuvs::distance::kernels::KernelType>(params.kernel),
      cache_size,
      C_SVC);

    cache.InitWorkingSet(this->ws_idx_dev.data());
    auto batch_descriptor = cache.InitFullTileBatching(cache.getKernelIndices(false), this->n_ws);

    // there should be only one batch for this test that contains the full n_rows x n_ws tile
    ASSERT_TRUE(cache.getNextBatchKernel(batch_descriptor));

    // apply nonlinearity on tile_host_expected
    this->ApplyNonlin(params);
    ASSERT_TRUE(devArrMatchHost(this->tile_host_expected,
                                batch_descriptor.kernel_data,
                                this->n_rows * this->n_ws,
                                MLCommon::CompareApprox<TypeParam>(1e-6f),
                                stream));

    ASSERT_FALSE(cache.getNextBatchKernel(batch_descriptor));
    delete kernel;
  }
}

TYPED_TEST_P(KernelCacheTest, SvcCacheEvalTest)
{
  KernelParams param{KernelType::LINEAR, 3, 1, 0};
  float cache_size = sizeof(TypeParam) * this->n_rows * 32 / (1024.0 * 1024);

  std::vector<KernelCacheTestInput> data{{KernelCacheTestInput{false, false, false}},
                                         {KernelCacheTestInput{false, true, false}},
                                         {KernelCacheTestInput{true, false, false}},
                                         {KernelCacheTestInput{true, true, false}},
                                         {KernelCacheTestInput{true, false, true}},
                                         {KernelCacheTestInput{true, true, true}}};

  for (auto input : data) {
    SCOPED_TRACE(input);

    size_t tile_byte_limit   = input.batching ? (2 * this->n_ws * sizeof(TypeParam)) : (1 << 30);
    size_t sparse_byte_limit = input.sparse_compute ? 1 : (1 << 30);
    GramMatrixBase<TypeParam>* kernel =
      KernelFactory<TypeParam>::create(ML::matrix::to_cuvs(param));
    if (input.sparse) {
      auto csr_structure =
        raft::make_device_compressed_structure_view<int, int, int>(this->x_indptr_dev.data(),
                                                                   this->x_indices_dev.data(),
                                                                   this->n_rows,
                                                                   this->n_cols,
                                                                   this->n_rows * this->n_cols);
      auto csr_view = raft::make_device_csr_matrix_view(this->x_data_dev.data(), csr_structure);
      KernelCache<TypeParam, raft::device_csr_matrix_view<TypeParam, int, int, int>> cache(
        this->handle,
        csr_view,
        this->n_rows,
        this->n_cols,
        this->n_ws,
        kernel,
        static_cast<cuvs::distance::kernels::KernelType>(param.kernel),
        cache_size,
        C_SVC,
        tile_byte_limit,
        sparse_byte_limit);

      for (int i = 0; i < 2; i++) {
        // We calculate cache tile multiple times to see if cache lookup works
        cache.InitWorkingSet(this->ws_idx_dev.data());
        auto batch_descriptor =
          cache.InitFullTileBatching(cache.getKernelIndices(false), this->n_ws);
        while (cache.getNextBatchKernel(batch_descriptor)) {
          this->check(batch_descriptor.kernel_data,
                      batch_descriptor.nz_da_idx,
                      batch_descriptor.nnz_da,
                      batch_descriptor.batch_size,
                      batch_descriptor.offset);
        }
      }
    } else {
      auto dense_view =
        raft::make_device_strided_matrix_view<TypeParam, int, raft::layout_f_contiguous>(
          this->x_dev.data(), this->n_rows, this->n_cols, 0);
      KernelCache<TypeParam, raft::device_matrix_view<TypeParam, int, raft::layout_stride>> cache(
        this->handle,
        dense_view,
        this->n_rows,
        this->n_cols,
        this->n_ws,
        kernel,
        static_cast<cuvs::distance::kernels::KernelType>(param.kernel),
        cache_size,
        C_SVC,
        tile_byte_limit,
        sparse_byte_limit);

      for (int i = 0; i < 2; i++) {
        // We calculate cache tile multiple times to see if cache lookup works
        cache.InitWorkingSet(this->ws_idx_dev.data());
        auto batch_descriptor =
          cache.InitFullTileBatching(cache.getKernelIndices(false), this->n_ws);
        while (cache.getNextBatchKernel(batch_descriptor)) {
          this->check(batch_descriptor.kernel_data,
                      batch_descriptor.nz_da_idx,
                      batch_descriptor.nnz_da,
                      batch_descriptor.batch_size,
                      batch_descriptor.offset);
        }
      }
    }

    delete kernel;
  }
}

TYPED_TEST_P(KernelCacheTest, SvrCacheEvalTest)
{
  KernelParams param{KernelType::LINEAR, 3, 1, 0};
  float cache_size = sizeof(TypeParam) * this->n_rows * 32 / (1024.0 * 1024);

  this->n_ws        = 6;
  int ws_idx_svr[6] = {0, 5, 1, 4, 3, 7};
  raft::update_device(this->ws_idx_dev.data(), ws_idx_svr, 6, this->stream);

  std::vector<KernelCacheTestInput> data{{KernelCacheTestInput{false, false, false}},
                                         {KernelCacheTestInput{false, true, false}},
                                         {KernelCacheTestInput{true, false, false}},
                                         {KernelCacheTestInput{true, true, false}},
                                         {KernelCacheTestInput{true, false, true}},
                                         {KernelCacheTestInput{true, true, true}}};

  for (auto input : data) {
    SCOPED_TRACE(input);

    size_t tile_byte_limit   = input.batching ? (2 * this->n_ws * sizeof(TypeParam)) : (1 << 30);
    size_t sparse_byte_limit = input.sparse_compute ? 1 : (1 << 30);
    GramMatrixBase<TypeParam>* kernel =
      KernelFactory<TypeParam>::create(ML::matrix::to_cuvs(param));
    if (input.sparse) {
      auto csr_structure =
        raft::make_device_compressed_structure_view<int, int, int>(this->x_indptr_dev.data(),
                                                                   this->x_indices_dev.data(),
                                                                   this->n_rows,
                                                                   this->n_cols,
                                                                   this->n_rows * this->n_cols);
      auto csr_view = raft::make_device_csr_matrix_view(this->x_data_dev.data(), csr_structure);
      KernelCache<TypeParam, raft::device_csr_matrix_view<TypeParam, int, int, int>> cache(
        this->handle,
        csr_view,
        this->n_rows,
        this->n_cols,
        this->n_ws,
        kernel,
        static_cast<cuvs::distance::kernels::KernelType>(param.kernel),
        cache_size,
        EPSILON_SVR,
        tile_byte_limit,
        sparse_byte_limit);
      for (int i = 0; i < 2; i++) {
        // We calculate cache tile multiple times to see if cache lookup works
        cache.InitWorkingSet(this->ws_idx_dev.data());
        auto batch_descriptor =
          cache.InitFullTileBatching(cache.getKernelIndices(false), this->n_ws);
        while (cache.getNextBatchKernel(batch_descriptor)) {
          this->check(batch_descriptor.kernel_data,
                      batch_descriptor.nz_da_idx,
                      batch_descriptor.nnz_da,
                      batch_descriptor.batch_size,
                      batch_descriptor.offset);
        }
      }

    } else {
      auto dense_view =
        raft::make_device_strided_matrix_view<TypeParam, int, raft::layout_f_contiguous>(
          this->x_dev.data(), this->n_rows, this->n_cols, 0);
      KernelCache<TypeParam, raft::device_matrix_view<TypeParam, int, raft::layout_stride>> cache(
        this->handle,
        dense_view,
        this->n_rows,
        this->n_cols,
        this->n_ws,
        kernel,
        static_cast<cuvs::distance::kernels::KernelType>(param.kernel),
        cache_size,
        EPSILON_SVR,
        tile_byte_limit,
        sparse_byte_limit);
      for (int i = 0; i < 2; i++) {
        // We calculate cache tile multiple times to see if cache lookup works
        cache.InitWorkingSet(this->ws_idx_dev.data());
        auto batch_descriptor =
          cache.InitFullTileBatching(cache.getKernelIndices(false), this->n_ws);
        while (cache.getNextBatchKernel(batch_descriptor)) {
          this->check(batch_descriptor.kernel_data,
                      batch_descriptor.nz_da_idx,
                      batch_descriptor.nnz_da,
                      batch_descriptor.batch_size,
                      batch_descriptor.offset);
        }
      }
    }

    delete kernel;
  }
}

REGISTER_TYPED_TEST_CASE_P(KernelCacheTest, EvalTest, SvcCacheEvalTest, SvrCacheEvalTest);
INSTANTIATE_TYPED_TEST_CASE_P(My, KernelCacheTest, FloatTypes);

template <typename math_t>
class GetResultsTest : public ::testing::Test {
 public:
  GetResultsTest() : stream(handle.get_stream()) {}

 protected:
  void FreeDenseSupport()
  {
    rmm::device_async_resource_ref rmm_alloc = rmm::mr::get_current_device_resource_ref();
    auto stream                              = this->handle.get_stream();
    rmm_alloc.deallocate(stream, support_matrix.data, n_coefs * n_cols * sizeof(math_t));
    support_matrix.data = nullptr;
  }

  void TestResults()
  {
    auto stream = this->handle.get_stream();
    rmm::device_uvector<math_t> x_dev(n_rows * n_cols, stream);
    raft::update_device(x_dev.data(), x_host, n_rows * n_cols, stream);
    rmm::device_uvector<math_t> f_dev(n_rows, stream);
    raft::update_device(f_dev.data(), f_host, n_rows, stream);
    rmm::device_uvector<math_t> y_dev(n_rows, stream);
    raft::update_device(y_dev.data(), y_host, n_rows, stream);
    rmm::device_uvector<math_t> alpha_dev(n_rows, stream);
    raft::update_device(alpha_dev.data(), alpha_host, n_rows, stream);
    rmm::device_uvector<math_t> C_dev(n_rows, stream);
    init_C(C, C_dev.data(), n_rows, stream);

    auto dense_view = raft::make_device_strided_matrix_view<math_t, int, raft::layout_f_contiguous>(
      x_dev.data(), n_rows, n_cols, 0);
    Results<math_t, raft::device_matrix_view<math_t, int, raft::layout_stride>> res(
      handle, dense_view, n_rows, n_cols, y_dev.data(), C_dev.data(), C_SVC);
    res.Get(alpha_dev.data(), f_dev.data(), &dual_coefs, &n_coefs, &idx, &support_matrix, &b);

    ASSERT_EQ(n_coefs, 7);

    math_t dual_coefs_exp[] = {-0.1, -0.2, -1.5, 0.2, 0.4, 1.5, 1.5};
    EXPECT_TRUE(devArrMatchHost(
      dual_coefs_exp, dual_coefs, n_coefs, MLCommon::CompareApprox<math_t>(1e-6f), stream));

    int idx_exp[] = {2, 3, 4, 6, 7, 8, 9};
    EXPECT_TRUE(devArrMatchHost(idx_exp, idx, n_coefs, MLCommon::Compare<int>(), stream));

    math_t x_support_exp[] = {3, 4, 5, 7, 8, 9, 10, 13, 14, 15, 17, 18, 19, 20};

    EXPECT_TRUE(devArrMatchHost(x_support_exp,
                                support_matrix.data,
                                n_coefs * n_cols,
                                MLCommon::CompareApprox<math_t>(1e-6f),
                                stream));

    EXPECT_FLOAT_EQ(b, -6.25f);

    // Modify the test by setting all SVs bound, then b is calculated differently
    math_t alpha_host2[10] = {0, 0, 1.5, 1.5, 1.5, 0, 1.5, 1.5, 1.5, 1.5};
    raft::update_device(alpha_dev.data(), alpha_host2, n_rows, stream);
    FreeDenseSupport();
    res.Get(alpha_dev.data(), f_dev.data(), &dual_coefs, &n_coefs, &idx, &support_matrix, &b);
    FreeDenseSupport();
    EXPECT_FLOAT_EQ(b, -5.5f);
  }

  raft::handle_t handle;
  cudaStream_t stream = 0;

  int n_rows            = 10;
  int n_cols            = 2;
  math_t x_host[20]     = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
  math_t f_host[10]     = {1, 3, 10, 4, 2, 8, 6, 5, 9, 7};
  math_t y_host[10]     = {-1, -1, -1, -1, -1, 1, 1, 1, 1, 1};
  math_t alpha_host[10] = {0, 0, 0.1, 0.2, 1.5, 0, 0.2, 0.4, 1.5, 1.5};
  //                      l  l  l/u  l/u    u  u  l/u  l/u  l    l
  math_t C = 1.5;

  math_t* dual_coefs;
  int n_coefs;
  int* idx;
  SupportStorage<math_t> support_matrix;
  math_t b;
};

TYPED_TEST_CASE(GetResultsTest, FloatTypes);

TYPED_TEST(GetResultsTest, Results) { this->TestResults(); }

SvmParameter getDefaultSvmParameter()
{
  SvmParameter param;
  param.C              = 1;
  param.tol            = 0.001;
  param.cache_size     = 200;
  param.max_iter       = -1;
  param.max_outer_iter = -1;
  param.nochange_steps = 1000;
  param.verbosity      = rapids_logger::level_enum::info;
  param.epsilon        = 0.1;
  param.svmType        = C_SVC;
  return param;
}

template <typename math_t>
class SmoUpdateTest : public ::testing::Test {
 public:
  SmoUpdateTest()
    : stream(handle.get_stream()),
      n_rows(6),
      n_ws(2),
      f_dev(n_rows, stream),
      kernel_dev(n_rows * n_ws, stream),
      delta_alpha_dev(n_ws, stream)
  {
    RAFT_CUDA_TRY(cudaMemsetAsync(f_dev.data(), 0, f_dev.size() * sizeof(float), stream));
    raft::update_device(kernel_dev.data(), kernel_host, n_ws * n_rows, stream);
    raft::update_device(delta_alpha_dev.data(), delta_alpha_host, n_ws, stream);
  }

 protected:
  void RunTest()
  {
    SvmParameter param = getDefaultSvmParameter();
    SmoSolver<float> smo(handle, param, cuvs::distance::kernels::LINEAR, nullptr);
    smo.UpdateF(f_dev.data(), n_rows, delta_alpha_dev.data(), n_ws, kernel_dev.data());

    float f_host_expected[] = {0.1f, 7.4505806e-9f, 0.3f, 0.2f, 0.5f, 0.4f};
    devArrMatchHost(f_host_expected, f_dev.data(), n_rows, MLCommon::CompareApprox<math_t>(1e-6));
  }

  raft::handle_t handle;
  cudaStream_t stream = 0;

  int n_rows;
  int n_ws;
  rmm::device_uvector<float> kernel_dev;
  rmm::device_uvector<float> f_dev;
  rmm::device_uvector<float> delta_alpha_dev;
  float kernel_host[12]     = {3, 5, 4, 6, 5, 7, 4, 5, 7, 8, 10, 11};
  float delta_alpha_host[2] = {-0.1f, 0.1f};
};

TYPED_TEST_CASE(SmoUpdateTest, FloatTypes);
TYPED_TEST(SmoUpdateTest, Update) { this->RunTest(); }

template <typename math_t>
class SmoBlockSolverTest : public ::testing::Test {
 public:
  SmoBlockSolverTest()
    : stream(handle.get_stream()),
      n_rows(4),
      n_cols(2),
      n_ws(4),
      ws_idx_dev(n_ws, stream),
      y_dev(n_rows, stream),
      C_dev(n_rows, stream),
      f_dev(n_rows, stream),
      alpha_dev(n_rows, stream),
      delta_alpha_dev(n_ws, stream),
      kernel_dev(n_ws * n_rows, stream),
      return_buff_dev(2, stream)
  {
    RAFT_CUDA_TRY(cudaMemsetAsync(alpha_dev.data(), 0, alpha_dev.size() * sizeof(math_t), stream));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(delta_alpha_dev.data(), 0, delta_alpha_dev.size() * sizeof(math_t), stream));
    init_C(C, C_dev.data(), n_rows, stream);
    raft::update_device(ws_idx_dev.data(), ws_idx_host, n_ws, stream);
    raft::update_device(y_dev.data(), y_host, n_rows, stream);
    raft::update_device(f_dev.data(), f_host, n_rows, stream);
    raft::update_device(kernel_dev.data(), kernel_host, n_ws * n_rows, stream);
  }

 public:  // because of the device lambda
  void testBlockSolve()
  {
    SmoBlockSolve<math_t, 1024><<<1, n_ws, 0, stream>>>(y_dev.data(),
                                                        n_rows,
                                                        alpha_dev.data(),
                                                        n_ws,
                                                        delta_alpha_dev.data(),
                                                        f_dev.data(),
                                                        kernel_dev.data(),
                                                        ws_idx_dev.data(),
                                                        C_dev.data(),
                                                        1e-3f,
                                                        return_buff_dev.data(),
                                                        1);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    math_t return_buff_exp[2] = {0.2, 1};
    devArrMatchHost(
      return_buff_exp, return_buff_dev.data(), 2, MLCommon::CompareApprox<math_t>(1e-6), stream);

    rmm::device_uvector<math_t> delta_alpha_calc(n_rows, stream);
    raft::linalg::binaryOp(
      delta_alpha_calc.data(),
      y_dev.data(),
      alpha_dev.data(),
      n_rows,
      [] __device__(math_t a, math_t b) { return a * b; },
      stream);
    MLCommon::devArrMatch(delta_alpha_dev.data(),
                          delta_alpha_calc.data(),
                          n_rows,
                          MLCommon::CompareApprox<math_t>(1e-6),
                          stream);
    math_t alpha_expected[] = {0, 0.1f, 0.1f, 0};
    MLCommon::devArrMatch(
      alpha_expected, alpha_dev.data(), n_rows, MLCommon::CompareApprox<math_t>(1e-6), stream);
  }

 protected:
  raft::handle_t handle;
  cudaStream_t stream = 0;

  int n_rows;
  int n_cols;
  int n_ws;

  rmm::device_uvector<int> ws_idx_dev;
  rmm::device_uvector<math_t> y_dev;
  rmm::device_uvector<math_t> f_dev;
  rmm::device_uvector<math_t> C_dev;
  rmm::device_uvector<math_t> alpha_dev;
  rmm::device_uvector<math_t> delta_alpha_dev;
  rmm::device_uvector<math_t> kernel_dev;
  rmm::device_uvector<math_t> return_buff_dev;

  int ws_idx_host[4]     = {0, 1, 2, 3};
  math_t y_host[4]       = {1, 1, -1, -1};
  math_t C               = 1.5;
  math_t f_host[4]       = {0.4, 0.3, 0.5, 0.1};
  math_t kernel_host[16] = {26, 32, 38, 44, 32, 40, 48, 56, 38, 48, 58, 68, 44, 56, 68, 80};
};

TYPED_TEST_CASE(SmoBlockSolverTest, FloatTypes);

// test a single iteration of the block solver
TYPED_TEST(SmoBlockSolverTest, SolveSingleTest) { this->testBlockSolve(); }

template <typename math_t>
struct smoInput {
  math_t C;
  math_t tol;
  KernelParams kernel_params;
  int max_outer_iter;
  int max_inner_iter;
};

template <typename math_t>
struct svcInput {
  math_t C;
  math_t tol;
  KernelParams kernel_params;
  int n_rows;
  int n_cols;
  math_t* x_dev;
  math_t* y_dev;
  bool predict;
};

template <typename math_t>
struct smoOutput {
  int n_support;
  std::vector<math_t> dual_coefs;
  math_t b;
  std::vector<math_t> w;
  std::vector<math_t> x_support;
  std::vector<int> idx;
};

// If we want to compare decision function values too
template <typename math_t>
struct smoOutput2 {  //: smoOutput<math_t> {
  int n_support;
  std::vector<math_t> dual_coefs;
  math_t b;
  std::vector<math_t> w;
  std::vector<math_t> x_support;
  std::vector<int> idx;
  std::vector<math_t> decision_function;
};

template <typename math_t>
smoOutput<math_t> toSmoOutput(smoOutput2<math_t> x)
{
  smoOutput<math_t> y{x.n_support, x.dual_coefs, x.b, x.w, x.x_support, x.idx};
  return y;
}

template <typename math_t>
struct svmTol {
  math_t b;
  math_t cs;
  int n_sv;
};

template <typename math_t>
void checkResults(SvmModel<math_t> model,
                  smoOutput<math_t> expected,
                  cudaStream_t stream,
                  svmTol<math_t> tol = svmTol<math_t>{0.001, 0.99999, -1})
{
  math_t* dcoef_exp     = expected.dual_coefs.size() > 0 ? expected.dual_coefs.data() : nullptr;
  math_t* w_exp         = expected.w.size() > 0 ? expected.w.data() : nullptr;
  math_t* x_support_exp = expected.x_support.size() > 0 ? expected.x_support.data() : nullptr;
  int* idx_exp          = expected.idx.size() > 0 ? expected.idx.data() : nullptr;

  math_t ay_tol = 1e-5;

  if (tol.n_sv == -1) {
    tol.n_sv = expected.n_support * 0.01;
    if (expected.n_support > 10 && tol.n_sv < 3) tol.n_sv = 3;
  }
  EXPECT_LE(abs(model.n_support - expected.n_support), tol.n_sv);
  if (dcoef_exp) {
    EXPECT_TRUE(devArrMatchHost(
      dcoef_exp, model.dual_coefs, model.n_support, MLCommon::CompareApprox<math_t>(1e-3f)));
  }
  math_t* dual_coefs_host = new math_t[model.n_support];
  raft::update_host(dual_coefs_host, model.dual_coefs, model.n_support, stream);
  raft::interruptible::synchronize(stream);
  math_t ay = 0;
  for (int i = 0; i < model.n_support; i++) {
    ay += dual_coefs_host[i];
  }
  // Test if \sum \alpha_i y_i = 0
  EXPECT_LT(raft::abs(ay), ay_tol);

  if (x_support_exp) {
    EXPECT_TRUE(model.support_matrix.data != nullptr && model.support_matrix.nnz == -1);
    EXPECT_TRUE(devArrMatchHost(x_support_exp,
                                model.support_matrix.data,
                                model.n_support * model.n_cols,
                                MLCommon::CompareApprox<math_t>(1e-6f),
                                stream));
  }

  if (idx_exp) {
    EXPECT_TRUE(devArrMatchHost(
      idx_exp, model.support_idx, model.n_support, MLCommon::Compare<int>(), stream));
  }

  math_t* x_support_host = new math_t[model.n_support * model.n_cols];
  if (model.n_support * model.n_cols > 0) {
    EXPECT_TRUE(model.support_matrix.data != nullptr && model.support_matrix.nnz == -1);
    raft::update_host(
      x_support_host, model.support_matrix.data, model.n_support * model.n_cols, stream);
  }
  raft::interruptible::synchronize(stream);

  if (w_exp) {
    std::vector<math_t> w(model.n_cols, 0);
    for (int i = 0; i < model.n_support; i++) {
      for (int j = 0; j < model.n_cols; j++)
        w[j] += x_support_host[i + model.n_support * j] * dual_coefs_host[i];
    }
    // Calculate the cosine similarity between w and w_exp
    math_t abs_w     = 0;
    math_t abs_w_exp = 0;
    math_t cs        = 0;
    for (int i = 0; i < model.n_cols; i++) {
      abs_w += w[i] * w[i];
      abs_w_exp += w_exp[i] * w_exp[i];
      cs += w[i] * w_exp[i];
    }
    cs /= sqrt(abs_w * abs_w_exp);
    EXPECT_GT(cs, tol.cs);
  }

  EXPECT_LT(raft::abs(model.b - expected.b), tol.b);

  delete[] dual_coefs_host;
  delete[] x_support_host;
}

template <typename math_t>
class SmoSolverTest : public ::testing::Test {
 public:
  SmoSolverTest()
    : stream(handle.get_stream()),
      x_dev(n_rows * n_cols, stream),
      x_dev_indptr(n_rows + 1, stream),
      x_dev_indices(n_nnz, stream),
      x_dev_data(n_nnz, stream),
      ws_idx_dev(n_ws, stream),
      y_dev(n_rows, stream),
      C_dev(n_rows, stream),
      y_pred(n_rows, stream),
      f_dev(n_rows, stream),
      alpha_dev(n_rows, stream),
      delta_alpha_dev(n_ws, stream),
      kernel_dev(n_ws * n_rows, stream),
      return_buff_dev(2, stream),
      sample_weights_dev(n_rows, stream)
  {
    RAFT_CUDA_TRY(cudaMemsetAsync(alpha_dev.data(), 0, alpha_dev.size() * sizeof(math_t), stream));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(delta_alpha_dev.data(), 0, delta_alpha_dev.size() * sizeof(math_t), stream));
  }

 protected:
  void SetUp() override
  {
    raft::linalg::range(sample_weights_dev.data(), 1, n_rows + 1, stream);

    raft::update_device(x_dev.data(), x_host, n_rows * n_cols, stream);
    raft::update_device(x_dev_indptr.data(), x_host_indptr, n_rows + 1, stream);
    raft::update_device(x_dev_indices.data(), x_host_indices, n_nnz, stream);
    raft::update_device(x_dev_data.data(), x_host_data, n_nnz, stream);
    raft::update_device(ws_idx_dev.data(), ws_idx_host, n_ws, stream);
    raft::update_device(y_dev.data(), y_host, n_rows, stream);
    init_C(C, C_dev.data(), n_rows, stream);
    raft::update_device(f_dev.data(), f_host, n_rows, stream);
    raft::update_device(kernel_dev.data(), kernel_host, n_ws * n_rows, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(delta_alpha_dev.data(), 0, n_ws * sizeof(math_t), stream));

    kernel = std::make_unique<GramMatrixBase<math_t>>();
  }

 public:
  void blockSolveTest()
  {
    SmoBlockSolve<math_t, 1024><<<1, n_ws, 0, stream>>>(y_dev.data(),
                                                        n_rows,
                                                        alpha_dev.data(),
                                                        n_ws,
                                                        delta_alpha_dev.data(),
                                                        f_dev.data(),
                                                        kernel_dev.data(),
                                                        ws_idx_dev.data(),
                                                        C_dev.data(),
                                                        1e-3,
                                                        return_buff_dev.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    math_t return_buff[2];
    raft::update_host(return_buff, return_buff_dev.data(), 2, stream);
    handle.sync_stream(stream);
    EXPECT_FLOAT_EQ(return_buff[0], 2.0f) << return_buff[0];
    EXPECT_LT(return_buff[1], 100) << return_buff[1];

    // check results won't work, because it expects that GetResults was called
    rmm::device_uvector<math_t> delta_alpha_calc(n_rows, stream);
    raft::linalg::binaryOp(
      delta_alpha_calc.data(),
      y_dev.data(),
      alpha_dev.data(),
      n_rows,
      [] __device__(math_t a, math_t b) { return a * b; },
      stream);
    MLCommon::devArrMatch(delta_alpha_dev.data(),
                          delta_alpha_calc.data(),
                          n_rows,
                          MLCommon::CompareApprox<math_t>(1e-6),
                          stream);

    math_t alpha_expected[] = {0.6f, 0, 1, 1, 0, 0.6f};
    // for C=10: {0.25f, 0, 2.25f, 3.75f, 0, 1.75f};
    MLCommon::devArrMatch(
      alpha_expected, alpha_dev.data(), n_rows, MLCommon::CompareApprox<math_t>(1e-6), stream);

    math_t host_alpha[6];
    raft::update_host(host_alpha, alpha_dev.data(), n_rows, stream);

    math_t w[] = {0, 0};
    math_t ay  = 0;
    for (int i = 0; i < n_rows; i++) {
      EXPECT_FLOAT_EQ(host_alpha[i], alpha_expected[i]) << "alpha " << i;
      w[0] += x_host[i] * host_alpha[i] * y_host[i];
      w[1] += x_host[i + n_rows] * host_alpha[i] * y_host[i];
      ay += host_alpha[i] * y_host[i];
    }
    EXPECT_FLOAT_EQ(ay, 0.0);
    EXPECT_FLOAT_EQ(w[0], -0.4);
    EXPECT_FLOAT_EQ(w[1], 1.2);
    // for C=10
    // EXPECT_FLOAT_EQ(w[0], -2.0);
    // EXPECT_FLOAT_EQ(w[1],  2.0);
  }

  void svrBlockSolveTest()
  {
    auto stream = this->handle.get_stream();
    int n_ws    = 4;
    int n_rows  = 2;
    // int n_cols = 1;
    // math_t x[2] = {1, 2};
    // yr = {2, 3}
    math_t f[4]      = {-1.9, -2.9, -2.1 - 3.1};
    math_t kernel[4] = {1, 2, 2, 4};
    // ws_idx is defined as {0, 1, 2, 3}
    int kColIdx[4] = {0, 1, 0, 1};
    rmm::device_uvector<int> kColIdx_dev(4, stream);
    raft::update_device(f_dev.data(), f, 4, stream);
    raft::update_device(kernel_dev.data(), kernel, 4, stream);
    raft::update_device(kColIdx_dev.data(), kColIdx, 4, stream);
    SmoBlockSolve<math_t, 1024><<<1, n_ws, 0, stream>>>(y_dev.data(),
                                                        2 * n_rows,
                                                        alpha_dev.data(),
                                                        n_ws,
                                                        delta_alpha_dev.data(),
                                                        f_dev.data(),
                                                        kernel_dev.data(),
                                                        ws_idx_dev.data(),
                                                        C_dev.data(),
                                                        1e-3,
                                                        return_buff_dev.data(),
                                                        10,
                                                        EPSILON_SVR);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    math_t return_buff[2];
    raft::update_host(return_buff, return_buff_dev.data(), 2, stream);
    handle.sync_stream(stream);
    EXPECT_LT(return_buff[1], 10) << return_buff[1];

    math_t alpha_exp[] = {0, 0.8, 0.8, 0};
    MLCommon::devArrMatch(
      alpha_exp, alpha_dev.data(), 4, MLCommon::CompareApprox<math_t>(1e-6), stream);

    math_t dalpha_exp[] = {-0.8, 0.8};
    MLCommon::devArrMatch(
      dalpha_exp, delta_alpha_dev.data(), 2, MLCommon::CompareApprox<math_t>(1e-6), stream);
  }

 protected:
  raft::handle_t handle;
  cudaStream_t stream = 0;

  std::unique_ptr<GramMatrixBase<math_t>> kernel;
  int n_rows       = 6;
  const int n_cols = 2;
  int n_nnz        = 12;
  int n_ws         = 6;

  rmm::device_uvector<math_t> x_dev;
  rmm::device_uvector<int> x_dev_indptr;
  rmm::device_uvector<int> x_dev_indices;
  rmm::device_uvector<math_t> x_dev_data;
  rmm::device_uvector<int> ws_idx_dev;
  rmm::device_uvector<math_t> y_dev;
  rmm::device_uvector<math_t> C_dev;
  rmm::device_uvector<math_t> y_pred;
  rmm::device_uvector<math_t> f_dev;
  rmm::device_uvector<math_t> alpha_dev;
  rmm::device_uvector<math_t> delta_alpha_dev;
  rmm::device_uvector<math_t> kernel_dev;
  rmm::device_uvector<math_t> return_buff_dev;
  rmm::device_uvector<math_t> sample_weights_dev;

  math_t x_host[12] = {1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 3, 3};

  // csr representation
  int x_host_indptr[7]   = {0, 2, 4, 6, 8, 10, 12};
  int x_host_indices[12] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  math_t x_host_data[12] = {1, 1, 2, 1, 1, 2, 2, 2, 1, 3, 2, 3};

  int ws_idx_host[6] = {0, 1, 2, 3, 4, 5};
  math_t y_host[6]   = {-1, -1, 1, -1, 1, 1};
  math_t C           = 1;
  math_t f_host[6]   = {1, 1, -1, 1, -1, -1};

  math_t kernel_host[36] = {2, 3, 3, 4, 4, 5,  3, 5, 4, 6, 5,  7,  3, 4, 5, 6,  7,  8,
                            4, 6, 6, 8, 8, 10, 4, 5, 7, 8, 10, 11, 5, 7, 8, 10, 11, 13};

  int n_coefs;
  math_t b;
};

TYPED_TEST_CASE(SmoSolverTest, FloatTypes);

TYPED_TEST(SmoSolverTest, BlockSolveTest) { this->blockSolveTest(); }
TYPED_TEST(SmoSolverTest, SvrBlockSolveTest) { this->svrBlockSolveTest(); }

std::string kernelName(KernelParams k)
{
  std::vector<std::string> names{"linear", "poly", "rbf", "tanh"};
  return names[static_cast<int>(k.kernel)];
}

template <typename math_t>
std::ostream& operator<<(std::ostream& os, const smoInput<math_t>& b)
{
  os << kernelName(b.kernel_params) << ", C=" << b.C << ", tol=" << b.tol;
  return os;
}

TYPED_TEST(SmoSolverTest, SmoSolveTest)
{
  auto stream = this->handle.get_stream();
  std::vector<std::pair<smoInput<TypeParam>, smoOutput<TypeParam>>> data{
    {smoInput<TypeParam>{1, 0.001, KernelParams{KernelType::LINEAR, 3, 1, 0}, 100, 1},
     smoOutput<TypeParam>{4,                         // n_sv
                          {-0.6, 1, -1, 0.6},        // dual_coefs
                          -1.8,                      // b
                          {-0.4, 1.2},               // w
                          {1, 1, 2, 2, 1, 2, 2, 3},  // x_support
                          {0, 2, 3, 5}}},            // support idx
    /*
    {smoInput<TypeParam>{10, 0.001, KernelParams{KernelType::LINEAR, 3, 1, 0}, 100, 1},
     smoOutput<TypeParam>{3, {2, -4, 2, 0, 0}, -1.0, {-2, 2}, {}, {}}},
    {smoInput<TypeParam>{1, 1e-6, KernelParams{KernelType::POLYNOMIAL, 3, 1, 1}, 100, 1},
     smoOutput<TypeParam>{
       3, {-0.0255614, 0.0397971, -0.0142357}, -1.07739149, {}, {1, 1, 2, 1, 2, 2}, {0, 2, 3}}}
    */
  };

  for (auto d : data) {
    auto p   = d.first;
    auto exp = d.second;
    SCOPED_TRACE(p);
    SvmParameter param = getDefaultSvmParameter();
    param.C            = p.C;
    param.tol          = p.tol;
    // param.max_outer_iter = p.max_outer_iter;
    GramMatrixBase<TypeParam>* kernel =
      KernelFactory<TypeParam>::create(ML::matrix::to_cuvs(p.kernel_params));
    SmoSolver<TypeParam> smo(
      this->handle,
      param,
      static_cast<cuvs::distance::kernels::KernelType>(p.kernel_params.kernel),
      kernel);
    {
      SvmModel<TypeParam> model1{0, this->n_cols, 0, nullptr, {}, nullptr, 0, nullptr};
      auto dense_view =
        raft::make_device_strided_matrix_view<TypeParam, int, raft::layout_f_contiguous>(
          this->x_dev.data(), this->n_rows, this->n_cols, 0);
      smo.Solve(dense_view,
                this->n_rows,
                this->n_cols,
                this->y_dev.data(),
                nullptr,
                &model1.dual_coefs,
                &model1.n_support,
                &model1.support_matrix,
                &model1.support_idx,
                &model1.b,
                -1,
                p.max_outer_iter,
                p.max_inner_iter);
      checkResults(model1, exp, stream);
      svmFreeBuffers(this->handle, model1);
    }

    // also check sparse input
    {
      SvmModel<TypeParam> model2{0, this->n_cols, 0, nullptr, {}, nullptr, 0, nullptr};
      auto csr_structure =
        raft::make_device_compressed_structure_view<int, int, int>(this->x_dev_indptr.data(),
                                                                   this->x_dev_indices.data(),
                                                                   this->n_rows,
                                                                   this->n_cols,
                                                                   this->n_nnz);
      auto csr_view = raft::make_device_csr_matrix_view(this->x_dev_data.data(), csr_structure);
      smo.Solve(csr_view,
                this->n_rows,
                this->n_cols,
                this->y_dev.data(),
                nullptr,
                &model2.dual_coefs,
                &model2.n_support,
                &model2.support_matrix,
                &model2.support_idx,
                &model2.b,
                -1,
                p.max_outer_iter,
                p.max_inner_iter);
      checkResults(model2, exp, stream);
      svmFreeBuffers(this->handle, model2);
    }
  }
}

TYPED_TEST(SmoSolverTest, SvcTest)
{
  auto stream = this->handle.get_stream();
  std::vector<std::pair<svcInput<TypeParam>, smoOutput2<TypeParam>>> data{
    {svcInput<TypeParam>{1,
                         0.001,
                         KernelParams{KernelType::LINEAR, 3, 1, 0},
                         this->n_rows,
                         this->n_cols,
                         this->x_dev.data(),
                         this->y_dev.data(),
                         true},
     smoOutput2<TypeParam>{4,
                           {-0.6, 1, -1, 0.6},
                           -1.8f,
                           {-0.4, 1.2},
                           {1, 1, 2, 2, 1, 2, 2, 3},
                           {0, 2, 3, 5},
                           {-1.0, -1.4, 0.2, -0.2, 1.4, 1.0}}},
    /*
    {// C == 0 marks a special test case with sample weights
     svcInput<TypeParam>{0,
                         0.001,
                         KernelParams{KernelType::LINEAR, 3, 1, 0},
                         this->n_rows,
                         this->n_cols,
                         this->x_dev.data(),
                         this->y_dev.data(),
                         true},
     smoOutput2<TypeParam>{
       3, {}, -1.0f, {-2, 2}, {1, 2, 2, 2, 2, 3}, {2, 3, 5}, {-1.0, -3.0, 1.0, -1.0, 3.0, 1.0}}},
    {svcInput<TypeParam>{1,
                         1e-6,
                         KernelParams{KernelType::POLYNOMIAL, 3, 1, 0},
                         this->n_rows,
                         this->n_cols,
                         this->x_dev.data(),
                         this->y_dev.data(),
                         true},
     smoOutput2<TypeParam>{
       3,
       {-0.0390089, 0.0590406, -0.0200316},
       -0.99999959,
       {},
       {1, 1, 2, 1, 2, 2},
       {0, 2, 3},
       {-0.9996812, -2.60106647, 0.9998406, -1.0001594, 6.49681105, 4.31951232}}},
    */
    {svcInput<TypeParam>{10,
                         1e-6,
                         KernelParams{KernelType::TANH, 3, 0.3, 1.0},
                         this->n_rows,
                         this->n_cols,
                         this->x_dev.data(),
                         this->y_dev.data(),
                         false},
     smoOutput2<TypeParam>{
       6,
       {-10., -10., 10., -10., 10., 10.},
       -0.3927505,
       {},
       {1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 3, 3},
       {0, 1, 2, 3, 4, 5},
       {0.25670694, -0.16451539, 0.16451427, -0.1568888, -0.04496891, -0.2387212}}},
    {svcInput<TypeParam>{1,
                         1.0e-6,
                         KernelParams{KernelType::RBF, 0, 0.15, 0},
                         this->n_rows,
                         this->n_cols,
                         this->x_dev.data(),
                         this->y_dev.data(),
                         true},
     smoOutput2<TypeParam>{
       6,
       {-1., -1, 1., -1., 1, 1.},
       0,
       {},
       {1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 3, 3},
       {0, 1, 2, 3, 4, 5},
       {-0.71964003, -0.95941954, 0.13929202, -0.13929202, 0.95941954, 0.71964003}}}};

  for (auto d : data) {
    auto p   = d.first;
    auto exp = d.second;
    SCOPED_TRACE(kernelName(p.kernel_params));
    TypeParam* sample_weights = nullptr;
    if (p.C == 0) {
      p.C            = 1;
      sample_weights = this->sample_weights_dev.data();
    }
    SVC<TypeParam> svc(this->handle, p.C, p.tol, p.kernel_params);
    svc.fit(p.x_dev, p.n_rows, p.n_cols, p.y_dev, sample_weights);
    checkResults(svc.model, toSmoOutput(exp), stream);
    rmm::device_uvector<TypeParam> y_pred(p.n_rows, stream);
    if (p.predict) {
      svc.predict(p.x_dev, p.n_rows, p.n_cols, y_pred.data());
      EXPECT_TRUE(MLCommon::devArrMatch(this->y_dev.data(),
                                        y_pred.data(),
                                        p.n_rows,
                                        MLCommon::CompareApprox<TypeParam>(1e-6f),
                                        stream));
    }
    if (exp.decision_function.size() > 0) {
      svc.decisionFunction(p.x_dev, p.n_rows, p.n_cols, y_pred.data());
      EXPECT_TRUE(devArrMatchHost(exp.decision_function.data(),
                                  y_pred.data(),
                                  p.n_rows,
                                  MLCommon::CompareApprox<TypeParam>(1e-3f),
                                  stream));
    }
  }
}

struct blobInput {
  double C;
  double tol;
  KernelParams kernel_params;
  int n_rows;
  int n_cols;
};

std::ostream& operator<<(std::ostream& os, const blobInput& b)
{
  os << kernelName(b.kernel_params) << " " << b.n_rows << "x" << b.n_cols;
  return os;
}

// until there is progress with Issue #935
template <typename inType, typename outType>
CUML_KERNEL void cast(outType* out, int n, inType* in)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) out[tid] = in[tid];
}

// To have the same input data for both single and double precision,
// we generate the blobs in single precision only, and cast to dp if needed.
template <typename math_t>
void make_blobs(const raft::handle_t& handle,
                math_t* x,
                math_t* y,
                int n_rows,
                int n_cols,
                int n_cluster,
                float* centers = nullptr)
{
  size_t free1, total;
  RAFT_CUDA_TRY(cudaMemGetInfo(&free1, &total));
  {
    auto stream = handle.get_stream();
    rmm::device_uvector<float> x_float(n_rows * n_cols, stream);
    rmm::device_uvector<int> y_int(n_rows, stream);

    Datasets::make_blobs(handle,
                         x_float.data(),
                         y_int.data(),
                         n_rows,
                         n_cols,
                         n_cluster,
                         true,
                         centers,
                         (float*)nullptr,
                         1.0f,
                         true,
                         -2.0f,
                         2.0f,
                         0);
    int TPB = 256;
    if (std::is_same<float, math_t>::value) {
      raft::linalg::transpose(handle, x_float.data(), (float*)x, n_cols, n_rows, stream);
    } else {
      rmm::device_uvector<math_t> x2(n_rows * n_cols, stream);
      cast<<<raft::ceildiv(n_rows * n_cols, TPB), TPB, 0, stream>>>(
        x2.data(), n_rows * n_cols, x_float.data());
      {
        raft::linalg::transpose(handle, x2.data(), x, n_cols, n_rows, stream);
      }
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }
    cast<<<raft::ceildiv(n_rows, TPB), TPB, 0, stream>>>(y, n_rows, y_int.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
}

struct is_same_functor {
  template <typename Tuple>
  __host__ __device__ int operator()(Tuple t)
  {
    return cuda::std::get<0>(t) == cuda::std::get<1>(t);
  }
};

TYPED_TEST(SmoSolverTest, BlobPredict)
{
  auto stream = this->handle.get_stream();
  // Pair.second is the expected accuracy. It might change if the Rng changes.
  std::vector<std::pair<blobInput, TypeParam>> data{
    {blobInput{1, 0.001, KernelParams{KernelType::LINEAR, 3, 1, 0}, 200, 10}, 98},
    {blobInput{1, 0.001, KernelParams{KernelType::POLYNOMIAL, 3, 1, 0}, 200, 10}, 98},
    {blobInput{1, 0.001, KernelParams{KernelType::RBF, 3, 1, 0}, 200, 2}, 98},
    {blobInput{1, 0.009, KernelParams{KernelType::TANH, 3, 0.1, 0}, 200, 10}, 98}};

  // This should be larger then N_PRED_BATCH in svcPredict
  const int n_pred = 5000;

  for (auto d : data) {
    auto p = d.first;
    SCOPED_TRACE(p);
    // explicit centers for the blobs
    rmm::device_uvector<float> centers(2 * p.n_cols, stream);
    thrust::device_ptr<float> thrust_ptr(centers.data());
    thrust::fill(thrust::cuda::par.on(stream), thrust_ptr, thrust_ptr + p.n_cols, -5.0f);
    thrust::fill(
      thrust::cuda::par.on(stream), thrust_ptr + p.n_cols, thrust_ptr + 2 * p.n_cols, +5.0f);

    rmm::device_uvector<TypeParam> x(p.n_rows * p.n_cols, stream);
    rmm::device_uvector<TypeParam> y(p.n_rows, stream);
    rmm::device_uvector<TypeParam> x_pred(n_pred * p.n_cols, stream);
    rmm::device_uvector<TypeParam> y_pred(n_pred, stream);

    make_blobs(this->handle, x.data(), y.data(), p.n_rows, p.n_cols, 2, centers.data());
    SVC<TypeParam> svc(
      this->handle, p.C, p.tol, p.kernel_params, 0, -1, 50, rapids_logger::level_enum::info);
    svc.fit(x.data(), p.n_rows, p.n_cols, y.data());

    // Create a different dataset for prediction
    make_blobs(this->handle, x_pred.data(), y_pred.data(), n_pred, p.n_cols, 2, centers.data());
    rmm::device_uvector<TypeParam> y_pred2(n_pred, stream);
    svc.predict(x_pred.data(), n_pred, p.n_cols, y_pred2.data());

    // Count the number of correct predictions
    rmm::device_uvector<int> is_correct(n_pred, stream);
    thrust::device_ptr<TypeParam> ptr1(y_pred.data());
    thrust::device_ptr<TypeParam> ptr2(y_pred2.data());
    thrust::device_ptr<int> ptr3(is_correct.data());
    auto first = thrust::make_zip_iterator(cuda::std::make_tuple(ptr1, ptr2));
    auto last  = thrust::make_zip_iterator(cuda::std::make_tuple(ptr1 + n_pred, ptr2 + n_pred));
    thrust::transform(thrust::cuda::par.on(stream), first, last, ptr3, is_same_functor());
    int n_correct = thrust::reduce(thrust::cuda::par.on(stream), ptr3, ptr3 + n_pred);

    TypeParam accuracy     = 100 * n_correct / n_pred;
    TypeParam accuracy_exp = d.second;
    EXPECT_GE(accuracy, accuracy_exp);
  }
}

TYPED_TEST(SmoSolverTest, MemoryLeak)
{
  GTEST_SKIP();  // Skip the tests in CI for release 24.02
                 // https://github.com/NVIDIA/cuml/issues/5763
  auto stream = this->handle.get_stream();
  // We measure that we have the same amount of free memory available on the GPU
  // before and after we call SVM. This can help catch memory leaks, but it is
  // not 100% sure. Small allocations might be pooled together by cudaMalloc,
  // and some of those would be missed by this method.
  enum class ThrowException { Yes, No };
  std::vector<std::pair<blobInput, ThrowException>> data{
    {blobInput{1, 0.001, KernelParams{KernelType::LINEAR, 3, 0.01, 0}, 1000, 1000},
     ThrowException::No},
    {blobInput{1, 0.001, KernelParams{KernelType::POLYNOMIAL, 400, 5, 10}, 1000, 1000},
     ThrowException::Yes}};
  // For the second set of input parameters  training will fail, some kernel
  // function values would be 1e400 or larger, which does not fit fp64.
  // This will lead to NaN diff in SmoSolver, which will throw an exception
  // to stop fitting.

  {
    // the first time transpose is called it leaks 8MB -- probably for cuBlas init
    rmm::device_uvector<TypeParam> in(100, stream);
    rmm::device_uvector<TypeParam> out(100, stream);
    raft::linalg::transpose(this->handle, in.data(), out.data(), 10, 10, stream);
  }

  size_t free1, total, free2;
  RAFT_CUDA_TRY(cudaMemGetInfo(&free1, &total));

  for (auto d : data) {
    auto p = d.first;
    SCOPED_TRACE(p);

    {
      rmm::device_uvector<TypeParam> x(p.n_rows * p.n_cols, stream);
      rmm::device_uvector<TypeParam> y(p.n_rows, stream);
      make_blobs(this->handle, x.data(), y.data(), p.n_rows, p.n_cols, 2);
      SVC<TypeParam> svc(this->handle, p.C, p.tol, p.kernel_params);
      if (d.second == ThrowException::Yes) {
        // We want to check whether we leak any memory while we unwind the stack
        EXPECT_THROW(svc.fit(x.data(), p.n_rows, p.n_cols, y.data()), raft::exception);
      } else {
        svc.fit(x.data(), p.n_rows, p.n_cols, y.data());
        rmm::device_uvector<TypeParam> y_pred(p.n_rows, stream);
        raft::interruptible::synchronize(stream);
        RAFT_CUDA_TRY(cudaMemGetInfo(&free2, &total));
        float delta = (free1 - free2);
        // Just to make sure that we measure any mem consumption at all:
        // we check if we see the memory consumption of x[n_rows*n_cols].
        // If this error is triggered, increasing the test size might help to fix
        // it (one could additionally control the exec time by the max_iter arg to
        // SVC).
        EXPECT_GT(delta, p.n_rows * p.n_cols * 4);
        svc.predict(x.data(), p.n_rows, p.n_cols, y_pred.data());
      }
    }
  }
  raft::interruptible::synchronize(stream);
  RAFT_CUDA_TRY(cudaMemGetInfo(&free2, &total));
  float delta = (free1 - free2);
  EXPECT_EQ(delta, 0);
}

TYPED_TEST(SmoSolverTest, DISABLED_MillionRows)
{
  auto stream = this->handle.get_stream();
  if (sizeof(TypeParam) == 8) {
    GTEST_SKIP();  // Skip the test for double input
  } else {
    // Stress test the kernel matrix calculation by calculating a kernel tile
    // with more the 2.8B elements. This would fail with int32 addressing. The test
    // is currently disabled because the memory usage might be prohibitive on CI
    // The test will be enabled once https://github.com/NVIDIA/cuml/pull/2449
    // is merged, that PR would reduce the kernel tile memory size.
    std::vector<std::pair<blobInput, TypeParam>> data{
      {blobInput{1, 0.001, KernelParams{KernelType::RBF, 3, 1, 0}, 2800000, 4}, 98},
      {blobInput{1, 0.001, KernelParams{KernelType::LINEAR, 3, 1, 0}, 2800000, 4}, 98},
      {blobInput{1, 0.001, KernelParams{KernelType::POLYNOMIAL, 3, 1, 0}, 2800000, 4}, 98},
      {blobInput{1, 0.001, KernelParams{KernelType::TANH, 3, 1, 0}, 2800000, 4}, 98}};

    for (auto d : data) {
      auto p = d.first;
      SCOPED_TRACE(p);
      // explicit centers for the blobs
      rmm::device_uvector<float> centers(2 * p.n_cols, stream);
      thrust::device_ptr<float> thrust_ptr(centers.data());
      thrust::fill(thrust::cuda::par.on(stream), thrust_ptr, thrust_ptr + p.n_cols, -5.0f);
      thrust::fill(
        thrust::cuda::par.on(stream), thrust_ptr + p.n_cols, thrust_ptr + 2 * p.n_cols, +5.0f);

      rmm::device_uvector<TypeParam> x(p.n_rows * p.n_cols, stream);
      rmm::device_uvector<TypeParam> y(p.n_rows, stream);
      rmm::device_uvector<TypeParam> y_pred(p.n_rows, stream);
      make_blobs(this->handle, x.data(), y.data(), p.n_rows, p.n_cols, 2, centers.data());
      const int max_outer_iter = 2;
      SVC<TypeParam> svc(this->handle,
                         p.C,
                         p.tol,
                         p.kernel_params,
                         0,
                         max_outer_iter,
                         50,
                         rapids_logger::level_enum::debug);
      svc.fit(x.data(), p.n_rows, p.n_cols, y.data());
      // predict on the same dataset
      svc.predict(x.data(), p.n_rows, p.n_cols, y_pred.data());
    }
  }
}

template <typename math_t>
void initializeTestMatrix(
  const raft::handle_t& handle, math_t* dense_matrix, int n_rows, int n_cols, math_t* y)
{
  auto stream = handle.get_stream();
  assert(n_cols % n_rows * n_rows % n_cols == 0);

  /*
  1 0 0 1 0 0
  0 1 0 0 1 0
  0 0 1 0 0 1

  1 0 0
  0 1 0
  0 0 1
  1 0 0
  0 1 0
  0 0 1
  */

  // fill col-major
  thrust::device_ptr<math_t> data_ptr(dense_matrix);
  auto one_or_zero = [n_rows, n_cols] __device__(const int& a) {
    int cycle  = min(n_rows, n_cols);
    int row_id = a % n_rows;
    int col_id = a / n_rows;
    return (row_id % cycle == col_id % cycle) ? (math_t)1 : (math_t)0;
  };
  thrust::transform(thrust::cuda::par.on(stream),
                    thrust::make_counting_iterator<int>(0),
                    thrust::make_counting_iterator<int>(n_rows * n_cols),
                    data_ptr,
                    one_or_zero);

  // init y label to 1 for all that contain the first half of the features
  {
    thrust::device_ptr<math_t> label_ptr(y);
    auto lable_hit = [n_rows, n_cols] __device__(const int& row) {
      int cycle     = min(n_rows, n_cols);
      int first_col = row % cycle;
      return (first_col < cycle / 2) ? (math_t)1 : (math_t)0;
    };
    thrust::transform(thrust::cuda::par.on(stream),
                      thrust::make_counting_iterator<int>(0),
                      thrust::make_counting_iterator<int>(n_rows),
                      label_ptr,
                      lable_hit);
  }

  handle.sync_stream(stream);
}

template <typename math_t>
void initializeTestMatrix(const raft::handle_t& handle,
                          raft::device_csr_matrix<math_t, int, int, int>& csr_matrix,
                          int n_rows,
                          int n_cols,
                          math_t* y)
{
  auto stream = handle.get_stream();
  assert(n_cols % n_rows * n_rows % n_cols == 0);

  /*
  1 0 0 1 0 0
  0 1 0 0 1 0
  0 0 1 0 0 1

  1 0 0
  0 1 0
  0 0 1
  1 0 0
  0 1 0
  0 0 1

  */
  int nnz_per_row = std::max(n_cols / n_rows, 1);
  int nnz         = n_rows * nnz_per_row;
  csr_matrix.initialize_sparsity(nnz);
  auto csr_structure = csr_matrix.structure_view();

  {
    // init indptr with nnz_per_row
    thrust::device_ptr<int> indptr_ptr(csr_structure.get_indptr().data());
    auto mul_x = [] __device__(const int& a, const int& b) { return a * b; };
    thrust::transform(thrust::cuda::par.on(stream),
                      thrust::make_counting_iterator<int>(0),
                      thrust::make_counting_iterator<int>(n_rows + 1),
                      thrust::make_constant_iterator<int>(nnz_per_row),
                      indptr_ptr,
                      mul_x);
  }

  // init indices/ data round-robin
  {
    thrust::device_ptr<int> indices_ptr(csr_structure.get_indices().data());
    auto one_or_zero = [n_rows, n_cols, nnz_per_row] __device__(const int& a) {
      int cycle  = min(n_rows, n_cols);
      int row_id = a / nnz_per_row;
      int ith1   = a % nnz_per_row;
      int col_id = ith1 * cycle + row_id % cycle;
      return col_id;
    };
    thrust::transform(thrust::cuda::par.on(stream),
                      thrust::make_counting_iterator<int>(0),
                      thrust::make_counting_iterator<int>(nnz),
                      indices_ptr,
                      one_or_zero);
  }

  // init data to 1
  {
    thrust::device_ptr<math_t> data_ptr(csr_matrix.get_elements().data());
    thrust::fill(thrust::cuda::par.on(stream), data_ptr, data_ptr + nnz, (math_t)1);
  }

  // init y label to 1 for all that contain the first half of the features
  {
    thrust::device_ptr<math_t> label_ptr(y);
    auto lable_hit = [n_rows, n_cols] __device__(const int& row) {
      int cycle     = min(n_rows, n_cols);
      int first_col = row % cycle;
      return (first_col < cycle / 2) ? (math_t)1 : (math_t)0;
    };
    thrust::transform(thrust::cuda::par.on(stream),
                      thrust::make_counting_iterator<int>(0),
                      thrust::make_counting_iterator<int>(n_rows),
                      label_ptr,
                      lable_hit);
  }

  handle.sync_stream(stream);
}

TYPED_TEST(SmoSolverTest, DenseBatching)
{
  auto stream = this->handle.get_stream();
  if (sizeof(TypeParam) == 8) {
    GTEST_SKIP();  // Skip the test for double input
  } else {
    std::vector<blobInput> data{
      {blobInput{1, 0.001, KernelParams{KernelType::RBF, 3, 1, 0}, 1000000, 4}},
      {blobInput{1, 0.001, KernelParams{KernelType::LINEAR, 3, 1, 0}, 1000000, 4}},
      {blobInput{1, 0.001, KernelParams{KernelType::POLYNOMIAL, 3, 1, 0}, 1000000, 4}},
      {blobInput{1, 0.001, KernelParams{KernelType::TANH, 3, 1, 0}, 1000000, 4}}};

    for (auto input : data) {
      SCOPED_TRACE(input);
      // this will result in a big kernel tile of ~4GB which will result in batching
      rmm::device_uvector<TypeParam> y(input.n_rows, stream);
      rmm::device_uvector<TypeParam> dense_input(input.n_rows * input.n_cols, stream);
      initializeTestMatrix(this->handle, dense_input.data(), input.n_rows, input.n_cols, y.data());

      SvmParameter param   = getDefaultSvmParameter();
      param.max_outer_iter = 2;

      SvmModel<TypeParam> model;
      TypeParam* sample_weights = nullptr;
      svcFit(this->handle,
             dense_input.data(),
             input.n_rows,
             input.n_cols,
             y.data(),
             param,
             input.kernel_params,
             model,
             sample_weights);

      // TODO predict with subset csr & dense
      rmm::device_uvector<TypeParam> y_pred(input.n_rows, stream);
      svcPredict(this->handle,
                 dense_input.data(),
                 input.n_rows,
                 input.n_cols,
                 input.kernel_params,
                 model,
                 y_pred.data(),
                 (TypeParam)200.0,
                 false);

      svmFreeBuffers(this->handle, model);
    }
  }
}

TYPED_TEST(SmoSolverTest, SparseBatching)
{
  auto stream = this->handle.get_stream();
  if (sizeof(TypeParam) == 8) {
    GTEST_SKIP();  // Skip the test for double input
  } else {
    std::vector<blobInput> data{
      // sparse input with batching
      {blobInput{1, 0.001, KernelParams{KernelType::RBF, 3, 1, 0}, 1000000, 4}},
      {blobInput{1, 0.001, KernelParams{KernelType::LINEAR, 3, 1, 0}, 1000000, 4}},
      {blobInput{1, 0.001, KernelParams{KernelType::POLYNOMIAL, 3, 1, 0}, 1000000, 4}},
      {blobInput{1, 0.001, KernelParams{KernelType::TANH, 3, 1, 0}, 1000000, 4}},
      // sparse input with sparse row extraction (also sparse support)
      {blobInput{1, 0.001, KernelParams{KernelType::RBF, 3, 1, 0}, 1000, 300000}},
      {blobInput{1, 0.001, KernelParams{KernelType::LINEAR, 3, 1, 0}, 1000, 300000}},
      {blobInput{1, 0.001, KernelParams{KernelType::POLYNOMIAL, 3, 1, 0}, 1000, 300000}},
      {blobInput{1, 0.001, KernelParams{KernelType::TANH, 3, 1, 0}, 1000, 300000}},
      // sparse input with batching AND sparse row extraction (also sparse support)
      {blobInput{1, 0.001, KernelParams{KernelType::RBF, 3, 1, 0}, 290000, 290000}},
      {blobInput{1, 0.001, KernelParams{KernelType::LINEAR, 3, 1, 0}, 290000, 290000}},
      {blobInput{1, 0.001, KernelParams{KernelType::POLYNOMIAL, 3, 1, 0}, 290000, 290000}},
      {blobInput{1, 0.001, KernelParams{KernelType::TANH, 3, 1, 0}, 290000, 290000}},
      // sparse input with sparse support
      {blobInput{1, 0.001, KernelParams{KernelType::RBF, 3, 1, 0}, 100000, 10000}},
      {blobInput{1, 0.001, KernelParams{KernelType::LINEAR, 3, 1, 0}, 100000, 10000}},
      {blobInput{1, 0.001, KernelParams{KernelType::POLYNOMIAL, 3, 1, 0}, 100000, 10000}},
      {blobInput{1, 0.001, KernelParams{KernelType::TANH, 3, 1, 0}, 100000, 10000}}};

    for (auto input : data) {
      SCOPED_TRACE(input);

      // this will result in a big kernel tile of ~4GB which will result in batching
      auto csr_input = raft::make_device_csr_matrix<TypeParam, int, int, int>(
        this->handle, input.n_rows, input.n_cols);
      rmm::device_uvector<TypeParam> y(input.n_rows, stream);
      initializeTestMatrix(this->handle, csr_input, input.n_rows, input.n_cols, y.data());

      auto csr_structure = csr_input.structure_view();

      SvmParameter param   = getDefaultSvmParameter();
      param.max_outer_iter = 2;

      SvmModel<TypeParam> model;
      TypeParam* sample_weights = nullptr;
      svcFitSparse(this->handle,
                   csr_structure.get_indptr().data(),
                   csr_structure.get_indices().data(),
                   csr_input.get_elements().data(),
                   csr_structure.get_n_rows(),
                   csr_structure.get_n_cols(),
                   csr_structure.get_nnz(),
                   y.data(),
                   param,
                   input.kernel_params,
                   model,
                   sample_weights);

      // predict with full input
      rmm::device_uvector<TypeParam> y_pred(input.n_rows, stream);
      svcPredictSparse(this->handle,
                       csr_structure.get_indptr().data(),
                       csr_structure.get_indices().data(),
                       csr_input.get_elements().data(),
                       csr_structure.get_n_rows(),
                       csr_structure.get_n_cols(),
                       csr_structure.get_nnz(),
                       input.kernel_params,
                       model,
                       y_pred.data(),
                       (TypeParam)200.0,
                       false);
      MLCommon::devArrMatch(
        y.data(), y_pred.data(), input.n_rows, MLCommon::CompareApprox<TypeParam>(1e-6), stream);

      // predict with subset csr & dense for all edge cases
      if (model.support_matrix.nnz >= 0) {
        int n_extract = 100;
        rmm::device_uvector<int> sequence(n_extract, stream);
        auto csr_subset = raft::make_device_csr_matrix<TypeParam, int, int, int>(
          this->handle, n_extract, input.n_cols);
        csr_subset.initialize_sparsity(10);  //! otherwise structure_view() call will fail
        rmm::device_uvector<TypeParam> dense_subset(n_extract * input.n_cols, stream);
        {
          thrust::device_ptr<int> sequence_ptr(sequence.data());
          thrust::sequence(
            thrust::cuda::par.on(stream), sequence_ptr, sequence_ptr + n_extract, (int)0);
          ML::SVM::extractRows(
            csr_input.view(), csr_subset, sequence.data(), n_extract, this->handle);
          ML::SVM::extractRows(
            csr_input.view(), dense_subset.data(), sequence.data(), n_extract, this->handle);
        }
        rmm::device_uvector<TypeParam> y_pred_csr(n_extract, stream);
        rmm::device_uvector<TypeParam> y_pred_dense(n_extract, stream);
        // also reduce buffer memory to ensure batching
        svcPredictSparse(this->handle,
                         csr_subset.structure_view().get_indptr().data(),
                         csr_subset.structure_view().get_indices().data(),
                         csr_subset.get_elements().data(),
                         csr_subset.structure_view().get_n_rows(),
                         csr_subset.structure_view().get_n_cols(),
                         csr_subset.structure_view().get_nnz(),
                         input.kernel_params,
                         model,
                         y_pred_csr.data(),
                         (TypeParam)50.0,
                         false);
        svcPredict(this->handle,
                   dense_subset.data(),
                   n_extract,
                   input.n_cols,
                   input.kernel_params,
                   model,
                   y_pred_dense.data(),
                   (TypeParam)50.0,
                   false);
        MLCommon::devArrMatch(y_pred_csr.data(),
                              y_pred_dense.data(),
                              n_extract,
                              MLCommon::CompareApprox<TypeParam>(1e-6),
                              stream);
      }

      svmFreeBuffers(this->handle, model);
    }
  }
}

template <typename math_t>
struct SvrInput {
  SvmParameter param;
  KernelParams kernel;
  int n_rows;
  int n_cols;
  std::vector<math_t> x;
  std::vector<math_t> y;
  std::vector<math_t> sample_weighs;
};

template <typename math_t>
std::ostream& operator<<(std::ostream& os, const SvrInput<math_t>& b)
{
  os << kernelName(b.kernel) << " " << b.n_rows << "x" << b.n_cols << ", C=" << b.param.C
     << ", tol=" << b.param.tol;
  return os;
}

template <typename math_t>
class SvrTest : public ::testing::Test {
 public:
  SvrTest()
    : stream(handle.get_stream()),
      x_dev(n_rows * n_cols, stream),
      y_dev(n_rows, stream),
      C_dev(2 * n_rows, stream),
      yc(n_train, stream),
      f(n_train, stream),
      alpha(n_train, stream)
  {
  }

 protected:
  void SetUp() override
  {
    raft::update_device(x_dev.data(), x_host, n_rows * n_cols, stream);
    raft::update_device(y_dev.data(), y_host, n_rows, stream);

    model.n_support      = 0;
    model.dual_coefs     = nullptr;
    model.support_matrix = {};
    model.support_idx    = nullptr;
    model.n_classes      = 0;
    model.unique_labels  = nullptr;
  }

  void TearDown() override { svmFreeBuffers(handle, model); }

 public:
  void TestSvrInit()
  {
    auto stream        = this->handle.get_stream();
    SvmParameter param = getDefaultSvmParameter();
    param.svmType      = EPSILON_SVR;
    SmoSolver<math_t> smo(handle, param, cuvs::distance::kernels::KernelType::LINEAR, nullptr);
    smo.SvrInit(y_dev.data(), n_rows, yc.data(), f.data());

    EXPECT_TRUE(
      devArrMatchHost(yc_exp, yc.data(), n_train, MLCommon::CompareApprox<math_t>(1.0e-9), stream));
    EXPECT_TRUE(devArrMatchHost(f_exp, f.data(), n_train, MLCommon::Compare<math_t>(), stream));
  }

  void TestSvrWorkingSet()
  {
    init_C((math_t)1.0, C_dev.data(), 2 * n_rows, stream);
    WorkingSet<math_t>* ws;
    ws = new WorkingSet<math_t>(handle, stream, n_rows, 20, EPSILON_SVR);
    EXPECT_EQ(ws->GetSize(), 2 * n_rows);

    raft::update_device(alpha.data(), alpha_host, n_train, stream);
    raft::update_device(f.data(), f_exp, n_train, stream);
    raft::update_device(yc.data(), yc_exp, n_train, stream);

    ws->Select(f.data(), alpha.data(), yc.data(), C_dev.data());
    int exp_idx[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
    ASSERT_TRUE(
      devArrMatchHost(exp_idx, ws->GetIndices(), ws->GetSize(), MLCommon::Compare<int>(), stream));

    delete ws;

    ws = new WorkingSet<math_t>(handle, stream, n_rows, 10, EPSILON_SVR);
    EXPECT_EQ(ws->GetSize(), 10);
    ws->Select(f.data(), alpha.data(), yc.data(), C_dev.data());
    int exp_idx2[] = {6, 12, 5, 11, 3, 9, 8, 1, 7, 0};
    ASSERT_TRUE(
      devArrMatchHost(exp_idx2, ws->GetIndices(), ws->GetSize(), MLCommon::Compare<int>(), stream));
    delete ws;
  }

  void TestSvrResults()
  {
    raft::update_device(yc.data(), yc_exp, n_train, stream);
    init_C((math_t)0.001, C_dev.data(), n_rows * 2, stream);
    auto dense_view = raft::make_device_strided_matrix_view<math_t, int, raft::layout_f_contiguous>(
      x_dev.data(), n_rows, n_cols, 0);
    Results<math_t, raft::device_matrix_view<math_t, int, raft::layout_stride>> res(
      handle, dense_view, n_rows, n_cols, yc.data(), C_dev.data(), EPSILON_SVR);
    model.n_cols = n_cols;
    raft::update_device(alpha.data(), alpha_host, n_train, stream);
    raft::update_device(f.data(), f_exp, n_train, stream);

    res.Get(alpha.data(),
            f.data(),
            &model.dual_coefs,
            &model.n_support,
            &model.support_idx,
            &model.support_matrix,
            &model.b);
    ASSERT_EQ(model.n_support, 5);
    math_t dc_exp[] = {0.1, 0.3, -0.4, 0.9, -0.9};
    EXPECT_TRUE(devArrMatchHost(
      dc_exp, model.dual_coefs, model.n_support, MLCommon::CompareApprox<math_t>(1.0e-6), stream));

    EXPECT_TRUE(model.support_matrix.nnz == -1);

    math_t x_exp[] = {1, 2, 3, 5, 6};
    EXPECT_TRUE(devArrMatchHost(x_exp,
                                model.support_matrix.data,
                                model.n_support * n_cols,
                                MLCommon::CompareApprox<math_t>(1.0e-6),
                                stream));

    int idx_exp[] = {0, 1, 2, 4, 5};
    EXPECT_TRUE(devArrMatchHost(idx_exp,
                                model.support_idx,
                                model.n_support,
                                MLCommon::CompareApprox<math_t>(1.0e-6),
                                stream));
  }

  void TestSvrFitPredict()
  {
    auto stream = this->handle.get_stream();
    std::vector<std::pair<SvrInput<math_t>, smoOutput2<math_t>>> data{
      {SvrInput<math_t>{
         SvmParameter{1, 0, 1, -1, 10, 1e-3, rapids_logger::level_enum::info, 0.1, EPSILON_SVR},
         KernelParams{KernelType::LINEAR, 3, 1, 0},
         2,       // n_rows
         1,       // n_cols
         {0, 1},  // x
         {2, 3}   // y
       },
       smoOutput2<math_t>{2, {-0.8, 0.8}, 2.1, {0.8}, {0, 1}, {0, 1}, {2.1, 2.9}}},

      {SvrInput<math_t>{
         SvmParameter{1, 10, 1, -1, 1, 1e-3, rapids_logger::level_enum::info, 0.1, EPSILON_SVR},
         KernelParams{KernelType::LINEAR, 3, 1, 0},
         2,       // n_rows
         1,       // n_cols
         {1, 2},  // x
         {2, 3}   // y
       },
       smoOutput2<math_t>{2, {-0.8, 0.8}, 1.3, {0.8}, {1, 2}, {0, 1}, {2.1, 2.9}}},

      {SvrInput<math_t>{
         SvmParameter{1, 0, 1, -1, 1, 1e-3, rapids_logger::level_enum::info, 0.1, EPSILON_SVR},
         KernelParams{KernelType::LINEAR, 3, 1, 0},
         2,             // n_rows
         2,             // n_cols
         {1, 2, 5, 5},  // x
         {2, 3}         // y
       },
       smoOutput2<math_t>{2, {-0.8, 0.8}, 1.3, {0.8, 0.0}, {1, 2, 5, 5}, {0, 1}, {2.1, 2.9}}},

      {SvrInput<math_t>{
         SvmParameter{1, 0, 100, -1, 10, 1e-6, rapids_logger::level_enum::info, 0.1, EPSILON_SVR},
         KernelParams{KernelType::LINEAR, 3, 1, 0},
         7,                      // n_rows
         1,                      // n_cols
         {1, 2, 3, 4, 5, 6, 7},  // x
         {0, 2, 3, 4, 5, 6, 8}   // y
       },
       smoOutput2<math_t>{6,
                          {-1, 1, 0.45, -0.45, -1, 1},
                          -0.4,
                          {1.1},
                          {1.0, 2.0, 3.0, 5.0, 6.0, 7.0},
                          {0, 1, 2, 4, 5, 6},
                          {0.7, 1.8, 2.9, 4, 5.1, 6.2, 7.3}}},
      // Almost same as above, but with sample weights
      {SvrInput<math_t>{
         SvmParameter{1, 0, 100, -1, 10, 1e-3, rapids_logger::level_enum::info, 0.1, EPSILON_SVR},
         KernelParams{KernelType::LINEAR, 3, 1, 0},
         7,                       // n_rows
         1,                       // n_cols
         {1, 2, 3, 4, 5, 6, 7},   // x
         {0, 2, 3, 0, 4, 8, 12},  // y
         {1, 1, 1, 10, 2, 10, 1}  // sample weights
       },
       smoOutput2<math_t>{
         6, {}, -15.5, {3.9}, {1.0, 2.0, 3.0, 4.0, 6.0, 7.0}, {0, 1, 2, 3, 5, 6}, {}}},
      {SvrInput<math_t>{
         SvmParameter{1, 0, 100, -1, 10, 1e-6, rapids_logger::level_enum::info, 0.1, EPSILON_SVR},
         KernelParams{KernelType::LINEAR, 3, 1, 0},
         7,                      // n_rows
         1,                      // n_cols
         {1, 2, 3, 4, 5, 6, 7},  // x
         {2, 2, 2, 2, 2, 2, 2}   // y
       },
       smoOutput2<math_t>{0, {}, 2, {}, {}, {}, {}}}};
    for (auto d : data) {
      auto p   = d.first;
      auto exp = d.second;
      SCOPED_TRACE(p);
      raft::update_device(x_dev.data(), p.x.data(), p.n_rows * p.n_cols, stream);
      raft::update_device(y_dev.data(), p.y.data(), p.n_rows, stream);
      rmm::device_uvector<math_t> sample_weights_dev(0, stream);
      math_t* sample_weights = nullptr;
      if (!p.sample_weighs.empty()) {
        sample_weights_dev.resize(p.n_rows, stream);
        sample_weights = sample_weights_dev.data();
        raft::update_device(sample_weights_dev.data(), p.sample_weighs.data(), p.n_rows, stream);
      }
      svrFit(handle,
             x_dev.data(),
             p.n_rows,
             p.n_cols,
             y_dev.data(),
             p.param,
             p.kernel,
             model,
             sample_weights);
      checkResults(model, toSmoOutput(exp), stream);
      rmm::device_uvector<math_t> preds(p.n_rows, stream);
      svcPredict(handle,
                 x_dev.data(),
                 p.n_rows,
                 p.n_cols,
                 p.kernel,
                 model,
                 preds.data(),
                 (math_t)200.0,
                 false);
      if (!exp.decision_function.empty()) {
        EXPECT_TRUE(devArrMatchHost(exp.decision_function.data(),
                                    preds.data(),
                                    p.n_rows,
                                    MLCommon::CompareApprox<math_t>(1.0e-5),
                                    stream));
      }
    }
  }

 protected:
  raft::handle_t handle;
  cudaStream_t stream = 0;
  int n_rows          = 7;
  int n_train         = 2 * n_rows;
  const int n_cols    = 1;

  SvmModel<math_t> model;
  rmm::device_uvector<math_t> x_dev;
  rmm::device_uvector<math_t> y_dev;
  rmm::device_uvector<math_t> C_dev;
  rmm::device_uvector<math_t> yc;
  rmm::device_uvector<math_t> f;
  rmm::device_uvector<math_t> alpha;

  math_t x_host[7]  = {1, 2, 3, 4, 5, 6, 7};
  math_t y_host[7]  = {0, 2, 3, 4, 5, 6, 8};
  math_t yc_exp[14] = {1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1};
  math_t f_exp[14]  = {
    0.1, -1.9, -2.9, -3.9, -4.9, -5.9, -7.9, -0.1, -2.1, -3.1, -4.1, -5.1, -6.1, -8.1};
  math_t alpha_host[14] = {0.2, 0.3, 0, 0, 1, 0.1, 0, 0.1, 0, 0.4, 0, 0.1, 1, 0};
};  // namespace SVM

typedef ::testing::Types<float> OnlyFp32;
TYPED_TEST_CASE(SvrTest, FloatTypes);

TYPED_TEST(SvrTest, Init) { this->TestSvrInit(); }
TYPED_TEST(SvrTest, WorkingSet) { this->TestSvrWorkingSet(); }
TYPED_TEST(SvrTest, Results) { this->TestSvrResults(); }
TYPED_TEST(SvrTest, FitPredict) { this->TestSvrFitPredict(); }
};  // namespace SVM
};  // namespace ML
