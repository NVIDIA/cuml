/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION. *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "test_utils.h"

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>
#include <timeSeries/jones_transform.cuh>

#include <algorithm>
#include <iostream>
#include <random>

namespace MLCommon {
namespace TimeSeries {

// parameter structure definition
struct JonesTransParam {
  int batchSize;
  int pValue;
  double tolerance;
};

// test fixture class
template

  <typename DataT>
  class JonesTransTest : public ::testing::TestWithParam<JonesTransParam> {
 public:
  JonesTransTest()
    : params(::testing::TestWithParam<JonesTransParam>::GetParam()),
      stream(handle.get_stream().get()),
      nElements(params.batchSize * params.pValue),
      d_golden_ar_trans(0, stream),
      d_computed_ar_trans(0, stream),
      d_params(0, stream),
      d_golden_ma_trans(0, stream),
      d_computed_ma_trans(0, stream),
      d_computed_ar_invtrans(0, stream),
      d_computed_ma_invtrans(0, stream)
  {
  }

 protected:
  // the constructor
  void SetUp() override
  {
    // generating random value test input that is stored in row major
    std::vector<double> arr1(nElements, 0);
    std::random_device rd;
    std::default_random_engine dre(rd());
    std::uniform_real_distribution<double> realGenerator(0, 1);

    std::generate(arr1.begin(), arr1.end(), [&]() { return realGenerator(dre); });

    //>>>>>>>>> AR transform golden output generation<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    double* newParams = (double*)malloc(nElements * sizeof(double*));
    double* tmp       = (double*)malloc(params.pValue * sizeof(double*));

    // for every model in the batch
    for (int i = 0; i < params.batchSize; ++i) {
      // storing the partial autocorrelation of each ar coefficient of a given batch in newParams
      // and the same in another temporary copy
      for (int j = 0; j < params.pValue; ++j) {
        newParams[i * params.pValue + j] = ((1 - exp(-1 * arr1[i * params.pValue + j])) /
                                            (1 + exp(-1 * arr1[i * params.pValue + j])));
        tmp[j]                           = newParams[i * params.pValue + j];
      }

      // calculating according to jone's recursive formula: phi(j,k) = phi(j-1,k) -
      // a(j)*phi(j-1,j-k)
      for (int j = 1; j < params.pValue; ++j) {
        // a is partial autocorrelation for jth coefficient
        DataT a = newParams[i * params.pValue + j];

        /*the recursive implementation of the transformation with:
        - lhs tmp[k] => phi(j,k)
        - rhs tmp[k] => phi(j-1,k)
        - a => a(j)
        - newParam[i*params.pValue + j-k-1] => phi(j-1, j-k)
        */
        for (int k = 0; k < j; ++k) {
          tmp[k] -= a * newParams[i * params.pValue + (j - k - 1)];
        }

        // copying it back for the next iteration
        for (int iter = 0; iter < j; ++iter) {
          newParams[i * params.pValue + iter] = tmp[iter];
        }
      }
    }

    // allocating and initializing device memory
    d_golden_ar_trans.resize(nElements, stream);
    d_computed_ar_trans.resize(nElements, stream);
    d_params.resize(nElements, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(
      d_golden_ar_trans.data(), 0, d_golden_ar_trans.size() * sizeof(DataT), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(
      d_computed_ar_trans.data(), 0, d_computed_ar_trans.size() * sizeof(DataT), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(d_params.data(), 0, d_params.size() * sizeof(DataT), stream));

    raft::update_device(d_params.data(), &arr1[0], (size_t)nElements, stream);
    raft::update_device(d_golden_ar_trans.data(), newParams, (size_t)nElements, stream);

    // calling the ar_trans_param CUDA implementation
    MLCommon::TimeSeries::jones_transform(d_params.data(),
                                          params.batchSize,
                                          params.pValue,
                                          d_computed_ar_trans.data(),
                                          true,
                                          false,
                                          stream,
                                          false);

    //>>>>>>>>> MA transform golden output generation<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    // for every model in the batch
    for (int i = 0; i < params.batchSize; ++i) {
      // storing the partial autocorrelation of each ma coefficient of a given batch in newParams
      // and the same in another temporary copy
      for (int j = 0; j < params.pValue; ++j) {
        newParams[i * params.pValue + j] = ((1 - exp(-1 * arr1[i * params.pValue + j])) /
                                            (1 + exp(-1 * arr1[i * params.pValue + j])));
        tmp[j]                           = newParams[i * params.pValue + j];
      }

      // calculating according to jone's recursive formula: phi(j,k) = phi(j-1,k) -
      // a(j)*phi(j-1,j-k)
      for (int j = 1; j < params.pValue; ++j) {
        // a is partial autocorrelation for jth coefficient
        DataT a = newParams[i * params.pValue + j];

        /*the recursive implementation of the transformation with:
        - lhs tmp[k] => phi(j,k)
        - rhs tmp[k] => phi(j-1,k)
        - a => a(j)
        - newParam[i*params.pValue + j-k-1] => phi(j-1, j-k)
        */
        for (int k = 0; k < j; ++k) {
          tmp[k] += a * newParams[i * params.pValue + (j - k - 1)];
        }

        // copying it back for the next iteration
        for (int iter = 0; iter < j; ++iter) {
          newParams[i * params.pValue + iter] = tmp[iter];
        }
      }
    }

    d_golden_ma_trans.resize(nElements, stream);
    d_computed_ma_trans.resize(nElements, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(
      d_golden_ma_trans.data(), 0, d_golden_ma_trans.size() * sizeof(DataT), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(
      d_computed_ma_trans.data(), 0, d_computed_ma_trans.size() * sizeof(DataT), stream));

    raft::update_device(d_golden_ma_trans.data(), newParams, (size_t)nElements, stream);

    // calling the ma_param_transform CUDA implementation
    MLCommon::TimeSeries::jones_transform(d_params.data(),
                                          params.batchSize,
                                          params.pValue,
                                          d_computed_ma_trans.data(),
                                          false,
                                          false,
                                          stream,
                                          false);

    //>>>>>>>>> AR inverse transform <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    d_computed_ar_invtrans.resize(nElements, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(
      d_computed_ar_invtrans.data(), 0, d_computed_ar_invtrans.size() * sizeof(DataT), stream));

    // calling the ar_param_inverse_transform CUDA implementation
    MLCommon::TimeSeries::jones_transform(d_computed_ar_trans.data(),
                                          params.batchSize,
                                          params.pValue,
                                          d_computed_ar_invtrans.data(),
                                          true,
                                          true,
                                          stream);

    //>>>>>>>>> MA inverse transform <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    d_computed_ma_invtrans.resize(nElements, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(
      d_computed_ma_invtrans.data(), 0, d_computed_ma_invtrans.size() * sizeof(DataT), stream));

    // calling the ma_param_inverse_transform CUDA implementation
    MLCommon::TimeSeries::jones_transform(d_computed_ma_trans.data(),
                                          params.batchSize,
                                          params.pValue,
                                          d_computed_ma_invtrans.data(),
                                          false,
                                          true,
                                          stream);
  }

  raft::handle_t handle;
  cudaStream_t stream = 0;

  // declaring the data values
  JonesTransParam params;
  rmm::device_uvector<DataT> d_golden_ar_trans, d_golden_ma_trans, d_computed_ar_trans,
    d_computed_ma_trans, d_computed_ar_invtrans, d_computed_ma_invtrans, d_params;
  int nElements = -1;
};

// setting test parameter values
const std::vector<JonesTransParam> inputs = {{500, 4, 0.001},
                                             {500, 3, 0.001},
                                             {500, 2, 0.001},
                                             {500, 1, 0.001},
                                             {5000, 4, 0.001},
                                             {5000, 3, 0.001},
                                             {5000, 2, 0.001},
                                             {5000, 1, 0.001},
                                             {4, 4, 0.001},
                                             {4, 3, 0.001},
                                             {4, 2, 0.001},
                                             {4, 1, 0.001},
                                             {500000, 4, 0.0001},
                                             {500000, 3, 0.0001},
                                             {500000, 2, 0.0001},
                                             {500000, 1, 0.0001}};

// writing the test suite
typedef JonesTransTest<double> JonesTransTestClass;
TEST_P(JonesTransTestClass, Result)
{
  ASSERT_TRUE(MLCommon::devArrMatch(d_golden_ar_trans.data(),
                                    d_computed_ar_trans.data(),
                                    nElements,
                                    MLCommon::CompareApprox<double>(params.tolerance)));
  ASSERT_TRUE(MLCommon::devArrMatch(d_golden_ma_trans.data(),
                                    d_computed_ma_trans.data(),
                                    nElements,
                                    MLCommon::CompareApprox<double>(params.tolerance)));
  /*
  Test verifying the inversion property:
  initially generated random coefficients -> ar_param_transform() / ma_param_transform() ->
  transformed coefficients -> ar_param_inverse_transform()/ma_param_inverse_transform() ->
  initially generated random coefficients
  */
  ASSERT_TRUE(MLCommon::devArrMatch(d_computed_ma_invtrans.data(),
                                    d_params.data(),
                                    nElements,
                                    MLCommon::CompareApprox<double>(params.tolerance)));
  ASSERT_TRUE(MLCommon::devArrMatch(d_computed_ar_invtrans.data(),
                                    d_params.data(),
                                    nElements,
                                    MLCommon::CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(JonesTrans, JonesTransTestClass, ::testing::ValuesIn(inputs));

}  // end namespace TimeSeries
}  // end namespace MLCommon
