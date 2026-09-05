/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_utils.h"

#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <functions/penalty.cuh>
#include <gtest/gtest.h>

namespace MLCommon {
namespace Functions {

template <typename T>
struct PenaltyInputs {
  T tolerance;
  int len;
};

template <typename T>
class PenaltyTest : public ::testing::TestWithParam<PenaltyInputs<T>> {
 public:
  PenaltyTest()
    : params(::testing::TestWithParam<PenaltyInputs<T>>::GetParam()),
      stream(handle.get_stream().get()),
      in(params.len, stream),
      out_lasso(1, stream),
      out_ridge(1, stream),
      out_elasticnet(1, stream),
      out_lasso_grad(params.len, stream),
      out_ridge_grad(params.len, stream),
      out_elasticnet_grad(params.len, stream),
      out_lasso_ref(1, stream),
      out_ridge_ref(1, stream),
      out_elasticnet_ref(1, stream),
      out_lasso_grad_ref(params.len, stream),
      out_ridge_grad_ref(params.len, stream),
      out_elasticnet_grad_ref(params.len, stream)
  {
  }

 protected:
  void SetUp() override
  {
    int len = params.len;

    T h_in[len] = {0.1, 0.35, -0.9, -1.4};
    raft::update_device(in.data(), h_in, len, stream);

    T h_out_lasso_ref[1] = {1.65};
    raft::update_device(out_lasso_ref.data(), h_out_lasso_ref, 1, stream);

    T h_out_ridge_ref[1] = {1.741499};
    raft::update_device(out_ridge_ref.data(), h_out_ridge_ref, 1, stream);

    T h_out_elasticnet_ref[1] = {1.695749};
    raft::update_device(out_elasticnet_ref.data(), h_out_elasticnet_ref, 1, stream);

    T h_out_lasso_grad_ref[len] = {0.6, 0.6, -0.6, -0.6};
    raft::update_device(out_lasso_grad_ref.data(), h_out_lasso_grad_ref, len, stream);

    T h_out_ridge_grad_ref[len] = {0.12, 0.42, -1.08, -1.68};
    raft::update_device(out_ridge_grad_ref.data(), h_out_ridge_grad_ref, len, stream);

    T h_out_elasticnet_grad_ref[len] = {0.36, 0.51, -0.84, -1.14};
    raft::update_device(out_elasticnet_grad_ref.data(), h_out_elasticnet_grad_ref, len, stream);

    T alpha    = 0.6;
    T l1_ratio = 0.5;

    lasso(out_lasso.data(), in.data(), len, alpha, stream);
    ridge(out_ridge.data(), in.data(), len, alpha, stream);
    elasticnet(out_elasticnet.data(), in.data(), len, alpha, l1_ratio, stream);
    lassoGrad(out_lasso_grad.data(), in.data(), len, alpha, stream);
    ridgeGrad(out_ridge_grad.data(), in.data(), len, alpha, stream);
    elasticnetGrad(out_elasticnet_grad.data(), in.data(), len, alpha, l1_ratio, stream);
  }

 protected:
  PenaltyInputs<T> params;
  raft::handle_t handle;
  cudaStream_t stream;

  rmm::device_uvector<T> in, out_lasso, out_ridge, out_elasticnet;
  rmm::device_uvector<T> out_lasso_ref, out_ridge_ref, out_elasticnet_ref;
  rmm::device_uvector<T> out_lasso_grad, out_ridge_grad, out_elasticnet_grad;
  rmm::device_uvector<T> out_lasso_grad_ref, out_ridge_grad_ref, out_elasticnet_grad_ref;
};

const std::vector<PenaltyInputs<float>> inputsf = {{0.01f, 4}};

const std::vector<PenaltyInputs<double>> inputsd = {{0.01, 4}};

typedef PenaltyTest<float> PenaltyTestF;
TEST_P(PenaltyTestF, Result)
{
  ASSERT_TRUE(devArrMatch(
    out_lasso_ref.data(), out_lasso.data(), 1, MLCommon::CompareApprox<float>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_lasso_grad_ref.data(),
                          out_lasso_grad.data(),
                          params.len,
                          MLCommon::CompareApprox<float>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(
    out_ridge_ref.data(), out_ridge.data(), 1, MLCommon::CompareApprox<float>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_ridge_grad_ref.data(),
                          out_ridge_grad.data(),
                          params.len,
                          MLCommon::CompareApprox<float>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_elasticnet_ref.data(),
                          out_elasticnet.data(),
                          1,
                          MLCommon::CompareApprox<float>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_elasticnet_grad_ref.data(),
                          out_elasticnet_grad.data(),
                          params.len,
                          MLCommon::CompareApprox<float>(params.tolerance)));
}

typedef PenaltyTest<double> PenaltyTestD;
TEST_P(PenaltyTestD, Result)
{
  ASSERT_TRUE(devArrMatch(
    out_lasso_ref.data(), out_lasso.data(), 1, MLCommon::CompareApprox<double>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_lasso_grad_ref.data(),
                          out_lasso_grad.data(),
                          params.len,
                          MLCommon::CompareApprox<double>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(
    out_ridge_ref.data(), out_ridge.data(), 1, MLCommon::CompareApprox<double>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_ridge_grad_ref.data(),
                          out_ridge_grad.data(),
                          params.len,
                          MLCommon::CompareApprox<double>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_elasticnet_ref.data(),
                          out_elasticnet.data(),
                          1,
                          MLCommon::CompareApprox<double>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_elasticnet_grad_ref.data(),
                          out_elasticnet_grad.data(),
                          params.len,
                          MLCommon::CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(PenaltyTests, PenaltyTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(PenaltyTests, PenaltyTestD, ::testing::ValuesIn(inputsd));

}  // end namespace Functions
}  // end namespace MLCommon
