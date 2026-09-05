/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_utils.h"

#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <functions/linearReg.cuh>
#include <gtest/gtest.h>

namespace MLCommon {
namespace Functions {

template <typename T>
struct LinRegLossInputs {
  T tolerance;
  T n_rows;
  T n_cols;
  int len;
};

template <typename T>
class LinRegLossTest : public ::testing::TestWithParam<LinRegLossInputs<T>> {
 public:
  LinRegLossTest()
    : params(::testing::TestWithParam<LinRegLossInputs<T>>::GetParam()),
      stream(handle.get_stream().get()),
      in(params.len, stream),
      out(1, stream),
      out_lasso(1, stream),
      out_ridge(1, stream),
      out_elasticnet(1, stream),
      out_grad(params.n_cols, stream),
      out_lasso_grad(params.n_cols, stream),
      out_ridge_grad(params.n_cols, stream),
      out_elasticnet_grad(params.n_cols, stream),
      out_ref(1, stream),
      out_lasso_ref(1, stream),
      out_ridge_ref(1, stream),
      out_elasticnet_ref(1, stream),
      out_grad_ref(params.n_cols, stream),
      out_lasso_grad_ref(params.n_cols, stream),
      out_ridge_grad_ref(params.n_cols, stream),
      out_elasticnet_grad_ref(params.n_cols, stream)
  {
  }

 protected:
  void SetUp() override
  {
    int len    = params.len;
    int n_rows = params.n_rows;
    int n_cols = params.n_cols;

    rmm::device_uvector<T> labels(params.n_rows, stream);
    rmm::device_uvector<T> coef(params.n_cols, stream);

    T h_in[len] = {0.1, 0.35, -0.9, -1.4, 2.0, 3.1};
    raft::update_device(in.data(), h_in, len, stream);

    T h_labels[n_rows] = {0.3, 2.0, -1.1};
    raft::update_device(labels.data(), h_labels, n_rows, stream);

    T h_coef[n_cols] = {0.35, -0.24};
    raft::update_device(coef.data(), h_coef, n_cols, stream);

    T h_out_ref[1] = {1.854842};
    raft::update_device(out_ref.data(), h_out_ref, 1, stream);

    T h_out_lasso_ref[1] = {2.2088};
    raft::update_device(out_lasso_ref.data(), h_out_lasso_ref, 1, stream);

    T h_out_ridge_ref[1] = {1.9629};
    raft::update_device(out_ridge_ref.data(), h_out_ridge_ref, 1, stream);

    T h_out_elasticnet_ref[1] = {2.0858};
    raft::update_device(out_elasticnet_ref.data(), h_out_elasticnet_ref, 1, stream);

    T h_out_grad_ref[n_cols] = {-0.56995, -3.12486};
    raft::update_device(out_grad_ref.data(), h_out_grad_ref, n_cols, stream);

    T h_out_lasso_grad_ref[n_cols] = {0.03005, -3.724866};
    raft::update_device(out_lasso_grad_ref.data(), h_out_lasso_grad_ref, n_cols, stream);

    T h_out_ridge_grad_ref[n_cols] = {-0.14995, -3.412866};
    raft::update_device(out_ridge_grad_ref.data(), h_out_ridge_grad_ref, n_cols, stream);

    T h_out_elasticnet_grad_ref[n_cols] = {-0.05995, -3.568866};
    raft::update_device(out_elasticnet_grad_ref.data(), h_out_elasticnet_grad_ref, n_cols, stream);

    T alpha    = 0.6;
    T l1_ratio = 0.5;

    linearRegLoss(handle,
                  in.data(),
                  params.n_rows,
                  params.n_cols,
                  labels.data(),
                  coef.data(),
                  out.data(),
                  penalty::NONE,
                  alpha,
                  l1_ratio,
                  stream);

    raft::update_device(in.data(), h_in, len, stream);

    linearRegLossGrads(handle,
                       in.data(),
                       params.n_rows,
                       params.n_cols,
                       labels.data(),
                       coef.data(),
                       out_grad.data(),
                       penalty::NONE,
                       alpha,
                       l1_ratio,
                       stream);

    raft::update_device(in.data(), h_in, len, stream);

    linearRegLoss(handle,
                  in.data(),
                  params.n_rows,
                  params.n_cols,
                  labels.data(),
                  coef.data(),
                  out_lasso.data(),
                  penalty::L1,
                  alpha,
                  l1_ratio,
                  stream);

    raft::update_device(in.data(), h_in, len, stream);

    linearRegLossGrads(handle,
                       in.data(),
                       params.n_rows,
                       params.n_cols,
                       labels.data(),
                       coef.data(),
                       out_lasso_grad.data(),
                       penalty::L1,
                       alpha,
                       l1_ratio,
                       stream);

    raft::update_device(in.data(), h_in, len, stream);

    linearRegLoss(handle,
                  in.data(),
                  params.n_rows,
                  params.n_cols,
                  labels.data(),
                  coef.data(),
                  out_ridge.data(),
                  penalty::L2,
                  alpha,
                  l1_ratio,
                  stream);

    linearRegLossGrads(handle,
                       in.data(),
                       params.n_rows,
                       params.n_cols,
                       labels.data(),
                       coef.data(),
                       out_ridge_grad.data(),
                       penalty::L2,
                       alpha,
                       l1_ratio,
                       stream);

    raft::update_device(in.data(), h_in, len, stream);

    linearRegLoss(handle,
                  in.data(),
                  params.n_rows,
                  params.n_cols,
                  labels.data(),
                  coef.data(),
                  out_elasticnet.data(),
                  penalty::ELASTICNET,
                  alpha,
                  l1_ratio,
                  stream);

    linearRegLossGrads(handle,
                       in.data(),
                       params.n_rows,
                       params.n_cols,
                       labels.data(),
                       coef.data(),
                       out_elasticnet_grad.data(),
                       penalty::ELASTICNET,
                       alpha,
                       l1_ratio,
                       stream);

    raft::update_device(in.data(), h_in, len, stream);
  }

 protected:
  LinRegLossInputs<T> params;
  raft::handle_t handle;
  cudaStream_t stream;

  rmm::device_uvector<T> in, out, out_lasso, out_ridge, out_elasticnet;
  rmm::device_uvector<T> out_ref, out_lasso_ref, out_ridge_ref, out_elasticnet_ref;
  rmm::device_uvector<T> out_grad, out_lasso_grad, out_ridge_grad, out_elasticnet_grad;
  rmm::device_uvector<T> out_grad_ref, out_lasso_grad_ref, out_ridge_grad_ref,
    out_elasticnet_grad_ref;
};

const std::vector<LinRegLossInputs<float>> inputsf = {{0.01f, 3, 2, 6}};

const std::vector<LinRegLossInputs<double>> inputsd = {{0.01, 3, 2, 6}};

typedef LinRegLossTest<float> LinRegLossTestF;
TEST_P(LinRegLossTestF, Result)
{
  ASSERT_TRUE(
    devArrMatch(out_ref.data(), out.data(), 1, MLCommon::CompareApprox<float>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(
    out_lasso_ref.data(), out_lasso.data(), 1, MLCommon::CompareApprox<float>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(
    out_ridge_ref.data(), out_ridge.data(), 1, MLCommon::CompareApprox<float>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_elasticnet_ref.data(),
                          out_elasticnet.data(),
                          1,
                          MLCommon::CompareApprox<float>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_grad_ref.data(),
                          out_grad.data(),
                          params.n_cols,
                          MLCommon::CompareApprox<float>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_lasso_grad_ref.data(),
                          out_lasso_grad.data(),
                          params.n_cols,
                          MLCommon::CompareApprox<float>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_ridge_grad_ref.data(),
                          out_ridge_grad.data(),
                          params.n_cols,
                          MLCommon::CompareApprox<float>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_elasticnet_grad_ref.data(),
                          out_elasticnet_grad.data(),
                          params.n_cols,
                          MLCommon::CompareApprox<float>(params.tolerance)));
}

typedef LinRegLossTest<double> LinRegLossTestD;
TEST_P(LinRegLossTestD, Result)
{
  ASSERT_TRUE(
    devArrMatch(out_ref.data(), out.data(), 1, MLCommon::CompareApprox<double>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(
    out_lasso_ref.data(), out_lasso.data(), 1, MLCommon::CompareApprox<double>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(
    out_ridge_ref.data(), out_ridge.data(), 1, MLCommon::CompareApprox<double>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_elasticnet_ref.data(),
                          out_elasticnet.data(),
                          1,
                          MLCommon::CompareApprox<double>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_grad_ref.data(),
                          out_grad.data(),
                          params.n_cols,
                          MLCommon::CompareApprox<double>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_lasso_grad_ref.data(),
                          out_lasso_grad.data(),
                          params.n_cols,
                          MLCommon::CompareApprox<double>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_ridge_grad_ref.data(),
                          out_ridge_grad.data(),
                          params.n_cols,
                          MLCommon::CompareApprox<double>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_elasticnet_grad_ref.data(),
                          out_elasticnet_grad.data(),
                          params.n_cols,
                          MLCommon::CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(LinRegLossTests, LinRegLossTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(LinRegLossTests, LinRegLossTestD, ::testing::ValuesIn(inputsd));

}  // end namespace Functions
}  // end namespace MLCommon
