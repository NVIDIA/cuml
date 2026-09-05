/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/linear_model/glm.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>
#include <test_utils.h>

namespace ML {
namespace GLM {

template <typename T>
struct RidgeInputs {
  T tol;
  size_t n_row;
  size_t n_col;
  size_t n_row_2;
  int algo;
  T alpha;
};

template <typename T>
class RidgeTest : public ::testing::TestWithParam<RidgeInputs<T>> {
 public:
  RidgeTest()
    : params(::testing::TestWithParam<RidgeInputs<T>>::GetParam()),
      stream(handle.get_stream().get()),
      coef(params.n_col, stream),
      coef2(params.n_col, stream),
      coef_ref(params.n_col, stream),
      coef2_ref(params.n_col, stream),
      pred(params.n_row_2, stream),
      pred_ref(params.n_row_2, stream),
      pred2(params.n_row_2, stream),
      pred2_ref(params.n_row_2, stream),
      coef_sc(1, stream),
      coef_sc_ref(1, stream),
      coef_sw(1, stream),
      coef_sw_ref(1, stream)
  {
    basicTest();
    basicTest2();
    testSampleWeight();
  }

 protected:
  void basicTest()
  {
    int len  = params.n_row * params.n_col;
    int len2 = params.n_row_2 * params.n_col;

    rmm::device_uvector<T> data(len, stream);
    rmm::device_uvector<T> pred_data(len2, stream);
    rmm::device_uvector<T> labels(params.n_row, stream);
    T alpha = params.alpha;

    /* How to reproduce the coefficients for this test:

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    scaler = StandardScaler(with_mean=True, with_std=True)
    x_norm = scaler.fit_transform(x_train)

    m = Ridge(
        fit_intercept=False, normalize=False, alpha=0.5)
    m.fit(x_train, y)
    print(m.coef_, m.predict(x_test))

    m = Ridge(
        fit_intercept=True, normalize=False, alpha=0.5)
    m.fit(x_train, y)
    print(m.coef_, m.predict(x_test))

    m = Ridge(
        fit_intercept=True, normalize=False, alpha=0.5)
    m.fit(x_norm, y)
    print(m.coef_ / scaler.scale_, m.predict(scaler.transform(x_test)))

     */

    T data_h[len] = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    raft::update_device(data.data(), data_h, len, stream);

    T labels_h[params.n_row] = {0.0, 0.1, 1.0};
    raft::update_device(labels.data(), labels_h, params.n_row, stream);

    T coef_ref_h[params.n_col] = {0.4, 0.4};
    raft::update_device(coef_ref.data(), coef_ref_h, params.n_col, stream);

    T coef2_ref_h[params.n_col] = {0.3454546, 0.34545454};
    raft::update_device(coef2_ref.data(), coef2_ref_h, params.n_col, stream);

    T pred_data_h[len2] = {0.5, 2.0, 0.2, 1.0};
    raft::update_device(pred_data.data(), pred_data_h, len2, stream);

    T pred_ref_h[params.n_row_2] = {0.28, 1.2};
    raft::update_device(pred_ref.data(), pred_ref_h, params.n_row_2, stream);

    T pred2_ref_h[params.n_row_2] = {0.37818182, 1.17272727};
    raft::update_device(pred2_ref.data(), pred2_ref_h, params.n_row_2, stream);

    intercept = T(0);

    ridgeFit(handle,
             data.data(),
             params.n_row,
             params.n_col,
             labels.data(),
             &alpha,
             1,
             coef.data(),
             &intercept,
             false,
             params.algo);

    gemmPredict(
      handle, pred_data.data(), params.n_row_2, params.n_col, coef.data(), intercept, pred.data());

    raft::update_device(data.data(), data_h, len, stream);
    raft::update_device(labels.data(), labels_h, params.n_row, stream);

    intercept2 = T(0);
    ridgeFit(handle,
             data.data(),
             params.n_row,
             params.n_col,
             labels.data(),
             &alpha,
             1,
             coef2.data(),
             &intercept2,
             true,
             params.algo);

    gemmPredict(handle,
                pred_data.data(),
                params.n_row_2,
                params.n_col,
                coef2.data(),
                intercept2,
                pred2.data());
  }

  void basicTest2()
  {
    int len = params.n_row * params.n_col;

    rmm::device_uvector<T> data_sc(len, stream);
    rmm::device_uvector<T> labels_sc(len, stream);

    std::vector<T> data_h = {1.0, 1.0, 2.0, 2.0, 1.0, 2.0};
    data_h.resize(len);
    raft::update_device(data_sc.data(), data_h.data(), len, stream);

    std::vector<T> labels_h = {6.0, 8.0, 9.0, 11.0, -1.0, 2.0};
    labels_h.resize(len);
    raft::update_device(labels_sc.data(), labels_h.data(), len, stream);

    std::vector<T> coef_sc_ref_h = {1.8};
    coef_sc_ref_h.resize(1);
    raft::update_device(coef_sc_ref.data(), coef_sc_ref_h.data(), 1, stream);

    T intercept_sc = T(0);
    T alpha_sc     = T(1.0);

    ridgeFit(handle,
             data_sc.data(),
             len,
             1,
             labels_sc.data(),
             &alpha_sc,
             1,
             coef_sc.data(),
             &intercept_sc,
             true,
             params.algo);
  }

  void testSampleWeight()
  {
    int len = params.n_row * params.n_col;

    rmm::device_uvector<T> data_sw(len, stream);
    rmm::device_uvector<T> labels_sw(len, stream);
    rmm::device_uvector<T> sample_weight(len, stream);

    std::vector<T> data_h = {1.0, 1.0, 2.0, 2.0, 1.0, 2.0};
    data_h.resize(len);
    raft::update_device(data_sw.data(), data_h.data(), len, stream);

    std::vector<T> labels_h = {6.0, 8.0, 9.0, 11.0, -1.0, 2.0};
    labels_h.resize(len);
    raft::update_device(labels_sw.data(), labels_h.data(), len, stream);

    std::vector<T> coef_sw_ref_h = {0.26052};
    coef_sw_ref_h.resize(1);
    raft::update_device(coef_sw_ref.data(), coef_sw_ref_h.data(), 1, stream);

    std::vector<T> sample_weight_h = {0.2, 0.3, 0.09, 0.15, 0.11, 0.15};
    sample_weight_h.resize(len);
    raft::update_device(sample_weight.data(), sample_weight_h.data(), len, stream);

    T intercept_sw = T(0);
    T alpha_sw     = T(1.0);

    ridgeFit(handle,
             data_sw.data(),
             len,
             1,
             labels_sw.data(),
             &alpha_sw,
             1,
             coef_sw.data(),
             &intercept_sw,
             true,
             params.algo,
             sample_weight.data());
  }

 protected:
  raft::handle_t handle;
  cudaStream_t stream = 0;

  RidgeInputs<T> params;
  rmm::device_uvector<T> coef, coef_ref, pred, pred_ref;
  rmm::device_uvector<T> coef2, coef2_ref, pred2, pred2_ref;
  rmm::device_uvector<T> coef_sc, coef_sc_ref;
  rmm::device_uvector<T> coef_sw, coef_sw_ref;
  T intercept, intercept2, intercept3;
};

const std::vector<RidgeInputs<float>> inputsf2 = {{0.001f, 3, 2, 2, 0, 0.5f},
                                                  {0.001f, 3, 2, 2, 1, 0.5f}};

const std::vector<RidgeInputs<double>> inputsd2 = {{0.001, 3, 2, 2, 0, 0.5},
                                                   {0.001, 3, 2, 2, 1, 0.5}};

typedef RidgeTest<float> RidgeTestF;
TEST_P(RidgeTestF, Fit)
{
  ASSERT_TRUE(MLCommon::devArrMatch(
    coef_ref.data(), coef.data(), params.n_col, MLCommon::CompareApproxAbs<float>(params.tol)));

  ASSERT_TRUE(MLCommon::devArrMatch(
    coef2_ref.data(), coef2.data(), params.n_col, MLCommon::CompareApproxAbs<float>(params.tol)));

  ASSERT_TRUE(MLCommon::devArrMatch(
    pred_ref.data(), pred.data(), params.n_row_2, MLCommon::CompareApproxAbs<float>(params.tol)));

  ASSERT_TRUE(MLCommon::devArrMatch(
    pred2_ref.data(), pred2.data(), params.n_row_2, MLCommon::CompareApproxAbs<float>(params.tol)));

  ASSERT_TRUE(MLCommon::devArrMatch(
    coef_sc_ref.data(), coef_sc.data(), 1, MLCommon::CompareApproxAbs<float>(params.tol)));

  ASSERT_TRUE(MLCommon::devArrMatch(
    coef_sw_ref.data(), coef_sw.data(), 1, MLCommon::CompareApproxAbs<float>(params.tol)));
}

typedef RidgeTest<double> RidgeTestD;
TEST_P(RidgeTestD, Fit)
{
  ASSERT_TRUE(MLCommon::devArrMatch(
    coef_ref.data(), coef.data(), params.n_col, MLCommon::CompareApproxAbs<double>(params.tol)));

  ASSERT_TRUE(MLCommon::devArrMatch(
    coef2_ref.data(), coef2.data(), params.n_col, MLCommon::CompareApproxAbs<double>(params.tol)));

  ASSERT_TRUE(MLCommon::devArrMatch(
    pred_ref.data(), pred.data(), params.n_row_2, MLCommon::CompareApproxAbs<double>(params.tol)));

  ASSERT_TRUE(MLCommon::devArrMatch(pred2_ref.data(),
                                    pred2.data(),
                                    params.n_row_2,
                                    MLCommon::CompareApproxAbs<double>(params.tol)));

  ASSERT_TRUE(MLCommon::devArrMatch(
    coef_sc_ref.data(), coef_sc.data(), 1, MLCommon::CompareApproxAbs<double>(params.tol)));

  ASSERT_TRUE(MLCommon::devArrMatch(
    coef_sw_ref.data(), coef_sw.data(), 1, MLCommon::CompareApproxAbs<double>(params.tol)));
}

INSTANTIATE_TEST_CASE_P(RidgeTests, RidgeTestF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(RidgeTests, RidgeTestD, ::testing::ValuesIn(inputsd2));

}  // namespace GLM
}  // end namespace ML
