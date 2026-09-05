/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>
#include <solver/sgd.cuh>
#include <test_utils.h>

namespace ML {
namespace Solver {

using namespace MLCommon;

template <typename T>
struct SgdInputs {
  T tol;
  int n_row;
  int n_col;
  int n_row2;
  int n_col2;
  int batch_size;
};

template <typename T>
class SgdTest : public ::testing::TestWithParam<SgdInputs<T>> {
 public:
  SgdTest()
    : params(::testing::TestWithParam<SgdInputs<T>>::GetParam()),
      stream(handle.get_stream().get()),
      coef(params.n_col, stream),
      coef_ref(params.n_col, stream),
      coef2(params.n_col, stream),
      coef2_ref(params.n_col, stream),
      pred_log(0, stream),
      pred_log_ref(0, stream),
      pred_svm(0, stream),
      pred_svm_ref(0, stream)
  {
    RAFT_CUDA_TRY(cudaMemsetAsync(coef.data(), 0, coef.size() * sizeof(T), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(coef2.data(), 0, coef2.size() * sizeof(T), stream));
    linearRegressionTest();
    logisticRegressionTest();
    svmTest();
  }

 protected:
  void linearRegressionTest()
  {
    int len = params.n_row * params.n_col;
    rmm::device_uvector<T> data(len, stream);
    rmm::device_uvector<T> labels(params.n_row, stream);

    T data_h[len] = {1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 3.0};
    raft::update_device(data.data(), data_h, len, stream);

    T labels_h[params.n_row] = {6.0, 8.0, 9.0, 11.0};
    raft::update_device(labels.data(), labels_h, params.n_row, stream);

    T coef_ref_h[params.n_col] = {2.087, 2.5454557};
    raft::update_device(coef_ref.data(), coef_ref_h, params.n_col, stream);

    T coef2_ref_h[params.n_col] = {1.000001, 1.9999998};
    raft::update_device(coef2_ref.data(), coef2_ref_h, params.n_col, stream);

    bool fit_intercept               = false;
    intercept                        = T(0);
    int epochs                       = 2000;
    T lr                             = T(0.01);
    ML::lr_type lr_type              = ML::lr_type::ADAPTIVE;
    T power_t                        = T(0.5);
    T alpha                          = T(0.0001);
    T l1_ratio                       = T(0.15);
    bool shuffle                     = true;
    T tol                            = T(1e-10);
    ML::loss_funct loss              = ML::loss_funct::SQRD_LOSS;
    MLCommon::Functions::penalty pen = MLCommon::Functions::penalty::NONE;
    int n_iter_no_change             = 10;

    sgdFit(handle,
           data.data(),
           params.n_row,
           params.n_col,
           labels.data(),
           coef.data(),
           &intercept,
           fit_intercept,
           params.batch_size,
           epochs,
           lr_type,
           lr,
           power_t,
           loss,
           pen,
           alpha,
           l1_ratio,
           shuffle,
           tol,
           n_iter_no_change,
           stream);

    fit_intercept = true;
    intercept2    = T(0);
    sgdFit(handle,
           data.data(),
           params.n_row,
           params.n_col,
           labels.data(),
           coef2.data(),
           &intercept2,
           fit_intercept,
           params.batch_size,
           epochs,
           ML::lr_type::CONSTANT,
           lr,
           power_t,
           loss,
           pen,
           alpha,
           l1_ratio,
           shuffle,
           tol,
           n_iter_no_change,
           stream);
  }

  void logisticRegressionTest()
  {
    int len = params.n_row2 * params.n_col2;
    rmm::device_uvector<T> data_logreg(len, stream);
    rmm::device_uvector<T> data_logreg_test(len, stream);
    rmm::device_uvector<T> labels_logreg(params.n_row2, stream);
    rmm::device_uvector<T> coef_class(params.n_row2, stream);
    pred_log.resize(params.n_row2, stream);
    pred_log_ref.resize(params.n_row2, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(coef_class.data(), 0, coef_class.size() * sizeof(T), stream));

    T data_h[len] = {0.1, -2.1, 5.4, 5.4, -1.5, -2.15, 2.65, 2.65, 3.25, -0.15, -7.35, -7.35};
    raft::update_device(data_logreg.data(), data_h, len, stream);

    T data_test_h[len] = {0.3, 1.1, 2.1, -10.1, 0.5, 2.5, -3.55, -20.5, -1.3, 3.0, -5.0, 15.0};
    raft::update_device(data_logreg_test.data(), data_test_h, len, stream);

    T labels_logreg_h[params.n_row2] = {0.0, 1.0, 1.0, 0.0};
    raft::update_device(labels_logreg.data(), labels_logreg_h, params.n_row2, stream);

    T pred_log_ref_h[params.n_row2] = {1.0, 0.0, 1.0, 1.0};
    raft::update_device(pred_log_ref.data(), pred_log_ref_h, params.n_row2, stream);

    bool fit_intercept               = true;
    T intercept_class                = T(0);
    int epochs                       = 1000;
    T lr                             = T(0.05);
    ML::lr_type lr_type              = ML::lr_type::CONSTANT;
    T power_t                        = T(0.5);
    T alpha                          = T(0.0);
    T l1_ratio                       = T(0.0);
    bool shuffle                     = false;
    T tol                            = T(0.0);
    ML::loss_funct loss              = ML::loss_funct::LOG;
    MLCommon::Functions::penalty pen = MLCommon::Functions::penalty::NONE;
    int n_iter_no_change             = 10;

    sgdFit(handle,
           data_logreg.data(),
           params.n_row2,
           params.n_col2,
           labels_logreg.data(),
           coef_class.data(),
           &intercept_class,
           fit_intercept,
           params.batch_size,
           epochs,
           lr_type,
           lr,
           power_t,
           loss,
           pen,
           alpha,
           l1_ratio,
           shuffle,
           tol,
           n_iter_no_change,
           stream);

    sgdPredictBinaryClass(handle,
                          data_logreg_test.data(),
                          params.n_row2,
                          params.n_col2,
                          coef_class.data(),
                          intercept_class,
                          pred_log.data(),
                          loss,
                          stream);
  }

  void svmTest()
  {
    int len = params.n_row2 * params.n_col2;

    rmm::device_uvector<T> data_svmreg(len, stream);
    rmm::device_uvector<T> data_svmreg_test(len, stream);
    rmm::device_uvector<T> labels_svmreg(params.n_row2, stream);
    rmm::device_uvector<T> coef_class(params.n_row2, stream);
    pred_svm.resize(params.n_row2, stream);
    pred_svm_ref.resize(params.n_row2, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(coef_class.data(), 0, coef_class.size() * sizeof(T), stream));

    T data_h[len] = {0.1, -2.1, 5.4, 5.4, -1.5, -2.15, 2.65, 2.65, 3.25, -0.15, -7.35, -7.35};
    raft::update_device(data_svmreg.data(), data_h, len, stream);

    T data_test_h[len] = {0.3, 1.1, 2.1, -10.1, 0.5, 2.5, -3.55, -20.5, -1.3, 3.0, -5.0, 15.0};
    raft::update_device(data_svmreg_test.data(), data_test_h, len, stream);

    T labels_svmreg_h[params.n_row2] = {0.0, 1.0, 1.0, 0.0};
    raft::update_device(labels_svmreg.data(), labels_svmreg_h, params.n_row2, stream);

    T pred_svm_ref_h[params.n_row2] = {1.0, 0.0, 1.0, 1.0};
    raft::update_device(pred_svm_ref.data(), pred_svm_ref_h, params.n_row2, stream);

    bool fit_intercept               = true;
    T intercept_class                = T(0);
    int epochs                       = 1000;
    T lr                             = T(0.05);
    ML::lr_type lr_type              = ML::lr_type::CONSTANT;
    T power_t                        = T(0.5);
    T alpha                          = T(1) / T(epochs);
    T l1_ratio                       = T(0.0);
    bool shuffle                     = false;
    T tol                            = T(0.0);
    ML::loss_funct loss              = ML::loss_funct::HINGE;
    MLCommon::Functions::penalty pen = MLCommon::Functions::penalty::L2;
    int n_iter_no_change             = 10;

    sgdFit(handle,
           data_svmreg.data(),
           params.n_row2,
           params.n_col2,
           labels_svmreg.data(),
           coef_class.data(),
           &intercept_class,
           fit_intercept,
           params.batch_size,
           epochs,
           lr_type,
           lr,
           power_t,
           loss,
           pen,
           alpha,
           l1_ratio,
           shuffle,
           tol,
           n_iter_no_change,
           stream);

    sgdPredictBinaryClass(handle,
                          data_svmreg_test.data(),
                          params.n_row2,
                          params.n_col2,
                          coef_class.data(),
                          intercept_class,
                          pred_svm.data(),
                          loss,
                          stream);
  }

 protected:
  raft::handle_t handle;
  cudaStream_t stream = 0;

  SgdInputs<T> params;
  rmm::device_uvector<T> coef, coef_ref;
  rmm::device_uvector<T> coef2, coef2_ref;
  rmm::device_uvector<T> pred_log, pred_log_ref;
  rmm::device_uvector<T> pred_svm, pred_svm_ref;
  T intercept, intercept2;
};

const std::vector<SgdInputs<float>> inputsf2 = {{0.01f, 4, 2, 4, 3, 2}};

const std::vector<SgdInputs<double>> inputsd2 = {{0.01, 4, 2, 4, 3, 2}};

typedef SgdTest<float> SgdTestF;
TEST_P(SgdTestF, Fit)
{
  ASSERT_TRUE(MLCommon::devArrMatch(
    coef_ref.data(), coef.data(), params.n_col, MLCommon::CompareApproxAbs<float>(params.tol)));

  ASSERT_TRUE(MLCommon::devArrMatch(
    coef2_ref.data(), coef2.data(), params.n_col, MLCommon::CompareApproxAbs<float>(params.tol)));

  ASSERT_TRUE(MLCommon::devArrMatch(pred_log_ref.data(),
                                    pred_log.data(),
                                    params.n_row,
                                    MLCommon::CompareApproxAbs<float>(params.tol)));

  ASSERT_TRUE(MLCommon::devArrMatch(pred_svm_ref.data(),
                                    pred_svm.data(),
                                    params.n_row,
                                    MLCommon::CompareApproxAbs<float>(params.tol)));
}

typedef SgdTest<double> SgdTestD;
TEST_P(SgdTestD, Fit)
{
  ASSERT_TRUE(MLCommon::devArrMatch(
    coef_ref.data(), coef.data(), params.n_col, MLCommon::CompareApproxAbs<double>(params.tol)));

  ASSERT_TRUE(MLCommon::devArrMatch(
    coef2_ref.data(), coef2.data(), params.n_col, MLCommon::CompareApproxAbs<double>(params.tol)));

  ASSERT_TRUE(MLCommon::devArrMatch(pred_log_ref.data(),
                                    pred_log.data(),
                                    params.n_row,
                                    MLCommon::CompareApproxAbs<double>(params.tol)));

  ASSERT_TRUE(MLCommon::devArrMatch(pred_svm_ref.data(),
                                    pred_svm.data(),
                                    params.n_row,
                                    MLCommon::CompareApproxAbs<double>(params.tol)));
}

INSTANTIATE_TEST_CASE_P(SgdTests, SgdTestF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(SgdTests, SgdTestD, ::testing::ValuesIn(inputsd2));

}  // namespace Solver
}  // end namespace ML
