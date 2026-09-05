/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/decomposition/params.hpp>
#include <cuml/decomposition/tsvd.hpp>

#include <raft/core/handle.hpp>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>
#include <test_utils.h>

#include <vector>

namespace ML {

template <typename T>
struct TsvdInputs {
  T tolerance;
  int n_row;
  int n_col;
  int n_row2;
  int n_col2;
  float redundancy;
  unsigned long long int seed;
  int algo;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const TsvdInputs<T>& dims)
{
  return os;
}

template <typename T>
class TsvdTest : public ::testing::TestWithParam<TsvdInputs<T>> {
 public:
  TsvdTest()
    : params(::testing::TestWithParam<TsvdInputs<T>>::GetParam()),
      stream(handle.get_stream().get()),
      components(0, stream),
      components_ref(0, stream),
      data2(0, stream),
      data2_back(0, stream)
  {
    basicTest();
    advancedTest();
  }

 protected:
  void basicTest()
  {
    raft::random::Rng r(params.seed, raft::random::GenPC);
    int len = params.n_row * params.n_col;

    rmm::device_uvector<T> data(len, stream);

    std::vector<T> data_h = {1.0, 2.0, 4.0, 2.0, 4.0, 5.0, 5.0, 4.0, 2.0, 1.0, 6.0, 4.0};
    data_h.resize(len);
    raft::update_device(data.data(), data_h.data(), len, stream);

    int len_comp = params.n_col * params.n_col;
    components.resize(len_comp, stream);
    rmm::device_uvector<T> singular_vals(params.n_col, stream);

    std::vector<T> components_ref_h = {
      0.3951, 0.1532, 0.9058, 0.7111, -0.6752, -0.1959, 0.5816, 0.7215, -0.3757};
    components_ref_h.resize(len_comp);

    components_ref.resize(len_comp, stream);
    raft::update_device(components_ref.data(), components_ref_h.data(), len_comp, stream);

    paramsTSVD prms;
    prms.n_cols       = params.n_col;
    prms.n_rows       = params.n_row;
    prms.n_components = params.n_col;
    if (params.algo == 0)
      prms.algorithm = solver::COV_EIG_DQ;
    else
      prms.algorithm = solver::COV_EIG_JACOBI;

    tsvdFit(handle, data.data(), components.data(), singular_vals.data(), prms);
  }

  void advancedTest()
  {
    raft::random::Rng r(params.seed, raft::random::GenPC);
    int len = params.n_row2 * params.n_col2;

    paramsTSVD prms;
    prms.n_cols       = params.n_col2;
    prms.n_rows       = params.n_row2;
    prms.n_components = params.n_col2;
    if (params.algo == 0)
      prms.algorithm = solver::COV_EIG_DQ;
    else if (params.algo == 1)
      prms.algorithm = solver::COV_EIG_JACOBI;
    else
      prms.n_components = params.n_col2 - 15;

    data2.resize(len, stream);
    int redundant_cols = int(params.redundancy * params.n_col2);
    int redundant_len  = params.n_row2 * redundant_cols;

    int informative_cols = params.n_col2 - redundant_cols;
    int informative_len  = params.n_row2 * informative_cols;

    r.uniform(data2.data(), informative_len, T(-1.0), T(1.0), stream);
    RAFT_CUDA_TRY(cudaMemcpyAsync(data2.data() + informative_len,
                                  data2.data(),
                                  redundant_len * sizeof(T),
                                  cudaMemcpyDeviceToDevice,
                                  stream));
    rmm::device_uvector<T> data2_trans(prms.n_rows * prms.n_components, stream);

    int len_comp = params.n_col2 * prms.n_components;
    rmm::device_uvector<T> components2(len_comp, stream);
    rmm::device_uvector<T> explained_vars2(prms.n_components, stream);
    rmm::device_uvector<T> explained_var_ratio2(prms.n_components, stream);
    rmm::device_uvector<T> singular_vals2(prms.n_components, stream);

    tsvdFitTransform(handle,
                     data2.data(),
                     data2_trans.data(),
                     components2.data(),
                     explained_vars2.data(),
                     explained_var_ratio2.data(),
                     singular_vals2.data(),
                     prms);

    data2_back.resize(len, stream);
    tsvdInverseTransform(handle, data2_trans.data(), components2.data(), data2_back.data(), prms);
  }

 protected:
  raft::handle_t handle;
  cudaStream_t stream = 0;

  TsvdInputs<T> params;
  rmm::device_uvector<T> components, components_ref, data2, data2_back;
};

const std::vector<TsvdInputs<float>> inputsf2 = {{0.01f, 4, 3, 1024, 128, 0.25f, 1234ULL, 0},
                                                 {0.01f, 4, 3, 1024, 128, 0.25f, 1234ULL, 1},
                                                 {0.04f, 4, 3, 512, 64, 0.25f, 1234ULL, 2},
                                                 {0.04f, 4, 3, 512, 64, 0.25f, 1234ULL, 2}};

const std::vector<TsvdInputs<double>> inputsd2 = {{0.01, 4, 3, 1024, 128, 0.25f, 1234ULL, 0},
                                                  {0.01, 4, 3, 1024, 128, 0.25f, 1234ULL, 1},
                                                  {0.05, 4, 3, 512, 64, 0.25f, 1234ULL, 2},
                                                  {0.05, 4, 3, 512, 64, 0.25f, 1234ULL, 2}};

typedef TsvdTest<float> TsvdTestLeftVecF;
TEST_P(TsvdTestLeftVecF, Result)
{
  ASSERT_TRUE(MLCommon::devArrMatch(components.data(),
                                    components_ref.data(),
                                    (params.n_col * params.n_col),
                                    MLCommon::CompareApprox<float>(params.tolerance),
                                    handle.get_stream().get()));
}

typedef TsvdTest<double> TsvdTestLeftVecD;
TEST_P(TsvdTestLeftVecD, Result)
{
  ASSERT_TRUE(MLCommon::devArrMatch(components.data(),
                                    components_ref.data(),
                                    (params.n_col * params.n_col),
                                    MLCommon::CompareApprox<double>(params.tolerance),
                                    handle.get_stream().get()));
}

typedef TsvdTest<float> TsvdTestDataVecF;
TEST_P(TsvdTestDataVecF, Result)
{
  ASSERT_TRUE(MLCommon::devArrMatch(data2.data(),
                                    data2_back.data(),
                                    (params.n_col2 * params.n_col2),
                                    MLCommon::CompareApprox<float>(params.tolerance),
                                    handle.get_stream().get()));
}

typedef TsvdTest<double> TsvdTestDataVecD;
TEST_P(TsvdTestDataVecD, Result)
{
  ASSERT_TRUE(MLCommon::devArrMatch(data2.data(),
                                    data2_back.data(),
                                    (params.n_col2 * params.n_col2),
                                    MLCommon::CompareApprox<double>(params.tolerance),
                                    handle.get_stream().get()));
}

INSTANTIATE_TEST_CASE_P(TsvdTests, TsvdTestLeftVecF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(TsvdTests, TsvdTestLeftVecD, ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_CASE_P(TsvdTests, TsvdTestDataVecF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(TsvdTests, TsvdTestDataVecD, ::testing::ValuesIn(inputsd2));

}  // end namespace ML
