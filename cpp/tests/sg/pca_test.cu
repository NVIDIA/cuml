/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/decomposition/params.hpp>
#include <cuml/decomposition/pca.hpp>

#include <raft/core/handle.hpp>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>
#include <test_utils.h>

#include <vector>

namespace ML {

template <typename T>
struct PcaInputs {
  T tolerance;
  int len;
  int n_row;
  int n_col;
  int len2;
  int n_row2;
  int n_col2;
  unsigned long long int seed;
  int algo;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const PcaInputs<T>& dims)
{
  return os;
}

template <typename T>
class PcaTest : public ::testing::TestWithParam<PcaInputs<T>> {
 public:
  PcaTest()
    : params(::testing::TestWithParam<PcaInputs<T>>::GetParam()),
      stream(handle.get_stream()),
      explained_vars(params.n_col, stream),
      explained_vars_ref(params.n_col, stream),
      components(params.n_col * params.n_col, stream),
      components_ref(params.n_col * params.n_col, stream),
      trans_data(params.len, stream),
      trans_data_ref(params.len, stream),
      data(params.len, stream),
      data_back(params.len, stream),
      data2(params.len2, stream),
      data2_back(params.len2, stream)
  {
    basicTest();
    advancedTest();
  }

 protected:
  void basicTest()
  {
    raft::random::Rng r(params.seed, raft::random::GenPC);
    int len = params.len;

    std::vector<T> data_h = {1.0, 2.0, 5.0, 4.0, 2.0, 1.0};
    data_h.resize(len);
    raft::update_device(data.data(), data_h.data(), len, stream);

    std::vector<T> trans_data_ref_h = {-2.3231, -0.3517, 2.6748, 0.3979, -0.6571, 0.2592};
    trans_data_ref_h.resize(len);
    raft::update_device(trans_data_ref.data(), trans_data_ref_h.data(), len, stream);

    int len_comp = params.n_col * params.n_col;
    rmm::device_uvector<T> explained_var_ratio(params.n_col, stream);
    rmm::device_uvector<T> singular_vals(params.n_col, stream);
    rmm::device_uvector<T> mean(params.n_col, stream);
    rmm::device_uvector<T> noise_vars(1, stream);

    std::vector<T> components_ref_h = {0.8163, 0.5776, -0.5776, 0.8163};
    components_ref_h.resize(len_comp);
    std::vector<T> explained_vars_ref_h = {6.338, 0.3287};
    explained_vars_ref_h.resize(params.n_col);

    raft::update_device(components_ref.data(), components_ref_h.data(), len_comp, stream);
    raft::update_device(
      explained_vars_ref.data(), explained_vars_ref_h.data(), params.n_col, stream);

    paramsPCA prms;
    prms.n_cols       = params.n_col;
    prms.n_rows       = params.n_row;
    prms.n_components = params.n_col;
    prms.whiten       = false;
    if (params.algo == 0)
      prms.algorithm = solver::COV_EIG_DQ;
    else
      prms.algorithm = solver::COV_EIG_JACOBI;

    pcaFit(handle,
           data.data(),
           components.data(),
           explained_vars.data(),
           explained_var_ratio.data(),
           singular_vals.data(),
           mean.data(),
           noise_vars.data(),
           prms);
    pcaTransform(handle,
                 data.data(),
                 components.data(),
                 trans_data.data(),
                 singular_vals.data(),
                 mean.data(),
                 prms);
    pcaInverseTransform(handle,
                        trans_data.data(),
                        components.data(),
                        singular_vals.data(),
                        mean.data(),
                        data_back.data(),
                        prms);
  }

  void advancedTest()
  {
    raft::random::Rng r(params.seed, raft::random::GenPC);
    int len = params.len2;

    paramsPCA prms;
    prms.n_cols       = params.n_col2;
    prms.n_rows       = params.n_row2;
    prms.n_components = params.n_col2;
    prms.whiten       = false;
    if (params.algo == 0)
      prms.algorithm = solver::COV_EIG_DQ;
    else if (params.algo == 1)
      prms.algorithm = solver::COV_EIG_JACOBI;

    r.uniform(data2.data(), len, T(-1.0), T(1.0), stream);
    rmm::device_uvector<T> data2_trans(prms.n_rows * prms.n_components, stream);

    int len_comp = params.n_col2 * prms.n_components;
    rmm::device_uvector<T> components2(len_comp, stream);
    rmm::device_uvector<T> explained_vars2(prms.n_components, stream);
    rmm::device_uvector<T> explained_var_ratio2(prms.n_components, stream);
    rmm::device_uvector<T> singular_vals2(prms.n_components, stream);
    rmm::device_uvector<T> mean2(prms.n_cols, stream);
    rmm::device_uvector<T> noise_vars2(1, stream);

    pcaFitTransform(handle,
                    data2.data(),
                    data2_trans.data(),
                    components2.data(),
                    explained_vars2.data(),
                    explained_var_ratio2.data(),
                    singular_vals2.data(),
                    mean2.data(),
                    noise_vars2.data(),
                    prms);

    pcaInverseTransform(handle,
                        data2_trans.data(),
                        components2.data(),
                        singular_vals2.data(),
                        mean2.data(),
                        data2_back.data(),
                        prms);
  }

 protected:
  raft::handle_t handle;
  cudaStream_t stream = 0;

  PcaInputs<T> params;

  rmm::device_uvector<T> explained_vars, explained_vars_ref, components, components_ref, trans_data,
    trans_data_ref, data, data_back, data2, data2_back;
};

const std::vector<PcaInputs<float>> inputsf2 = {
  {0.01f, 3 * 2, 3, 2, 1024 * 128, 1024, 128, 1234ULL, 0},
  {0.01f, 3 * 2, 3, 2, 256 * 32, 256, 32, 1234ULL, 1}};

const std::vector<PcaInputs<double>> inputsd2 = {
  {0.01, 3 * 2, 3, 2, 1024 * 128, 1024, 128, 1234ULL, 0},
  {0.01, 3 * 2, 3, 2, 256 * 32, 256, 32, 1234ULL, 1}};

typedef PcaTest<float> PcaTestValF;
TEST_P(PcaTestValF, Result)
{
  ASSERT_TRUE(devArrMatch(explained_vars.data(),
                          explained_vars_ref.data(),
                          params.n_col,
                          MLCommon::CompareApprox<float>(params.tolerance),
                          handle.get_stream()));
}

typedef PcaTest<double> PcaTestValD;
TEST_P(PcaTestValD, Result)
{
  ASSERT_TRUE(devArrMatch(explained_vars.data(),
                          explained_vars_ref.data(),
                          params.n_col,
                          MLCommon::CompareApprox<double>(params.tolerance),
                          handle.get_stream()));
}

typedef PcaTest<float> PcaTestLeftVecF;
TEST_P(PcaTestLeftVecF, Result)
{
  ASSERT_TRUE(devArrMatch(components.data(),
                          components_ref.data(),
                          (params.n_col * params.n_col),
                          MLCommon::CompareApprox<float>(params.tolerance),
                          handle.get_stream()));
}

typedef PcaTest<double> PcaTestLeftVecD;
TEST_P(PcaTestLeftVecD, Result)
{
  ASSERT_TRUE(devArrMatch(components.data(),
                          components_ref.data(),
                          (params.n_col * params.n_col),
                          MLCommon::CompareApprox<double>(params.tolerance),
                          handle.get_stream()));
}

typedef PcaTest<float> PcaTestTransDataF;
TEST_P(PcaTestTransDataF, Result)
{
  ASSERT_TRUE(devArrMatch(trans_data.data(),
                          trans_data_ref.data(),
                          (params.n_row * params.n_col),
                          MLCommon::CompareApprox<float>(params.tolerance),
                          handle.get_stream()));
}

typedef PcaTest<double> PcaTestTransDataD;
TEST_P(PcaTestTransDataD, Result)
{
  ASSERT_TRUE(devArrMatch(trans_data.data(),
                          trans_data_ref.data(),
                          (params.n_row * params.n_col),
                          MLCommon::CompareApprox<double>(params.tolerance),
                          handle.get_stream()));
}

typedef PcaTest<float> PcaTestDataVecSmallF;
TEST_P(PcaTestDataVecSmallF, Result)
{
  ASSERT_TRUE(devArrMatch(data.data(),
                          data_back.data(),
                          (params.n_col * params.n_col),
                          MLCommon::CompareApprox<float>(params.tolerance),
                          handle.get_stream()));
}

typedef PcaTest<double> PcaTestDataVecSmallD;
TEST_P(PcaTestDataVecSmallD, Result)
{
  ASSERT_TRUE(devArrMatch(data.data(),
                          data_back.data(),
                          (params.n_col * params.n_col),
                          MLCommon::CompareApprox<double>(params.tolerance),
                          handle.get_stream()));
}

// FIXME: These tests are disabled due to driver 418+ making them fail:
// https://github.com/NVIDIA/cuml/issues/379
typedef PcaTest<float> PcaTestDataVecF;
TEST_P(PcaTestDataVecF, Result)
{
  ASSERT_TRUE(devArrMatch(data2.data(),
                          data2_back.data(),
                          (params.n_col2 * params.n_col2),
                          MLCommon::CompareApprox<float>(params.tolerance),
                          handle.get_stream()));
}

typedef PcaTest<double> PcaTestDataVecD;
TEST_P(PcaTestDataVecD, Result)
{
  ASSERT_TRUE(MLCommon::devArrMatch(data2.data(),
                                    data2_back.data(),
                                    (params.n_col2 * params.n_col2),
                                    MLCommon::CompareApprox<double>(params.tolerance),
                                    handle.get_stream()));
}

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestValF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestValD, ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestLeftVecF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestLeftVecD, ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestDataVecSmallF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestDataVecSmallD, ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestTransDataF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestTransDataD, ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestDataVecF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestDataVecD, ::testing::ValuesIn(inputsd2));

}  // end namespace ML
