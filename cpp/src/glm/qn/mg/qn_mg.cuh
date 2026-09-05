/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "glm_base_mg.cuh"
#include "standardization.cuh"

#include <cuml/linear_model/qn.h>

#include <rmm/device_uvector.hpp>

#include <glm/qn/glm_logistic.cuh>
#include <glm/qn/glm_regularizer.cuh>
#include <glm/qn/glm_softmax.cuh>
#include <glm/qn/glm_svm.cuh>
#include <glm/qn/qn_solvers.cuh>
#include <glm/qn/qn_util.cuh>

namespace ML {
namespace GLM {
namespace opg {
using namespace ML::GLM::detail;

template <typename T, typename LossFunction>
int qn_fit_mg(const raft::handle_t& handle,
              const qn_params& pams,
              LossFunction& loss,
              const SimpleMat<T>& X,
              const SimpleVec<T>& y,
              SimpleDenseMat<T>& Z,
              T* w0_data,  // initial value and result
              T* fx,
              int* num_iters,
              size_t n_samples,
              int rank,
              int n_ranks,
              const Standardizer<T>* stder_p = NULL)
{
  cudaStream_t stream = handle.get_stream().get();
  LBFGSParam<T> opt_param(pams);
  SimpleVec<T> w0(w0_data, loss.n_param);

  // Scale the regularization strength with the number of samples.
  T l1 = pams.penalty_l1;
  T l2 = pams.penalty_l2;
  if (pams.penalty_normalized) {
    l1 /= n_samples;
    l2 /= n_samples;
  }

  ML::GLM::detail::Tikhonov<T> reg(l2);
  ML::GLM::detail::RegularizedGLM<T, LossFunction, decltype(reg)> regularizer_obj(&loss, &reg);

  auto obj_function =
    GLMWithDataMG(handle, rank, n_ranks, n_samples, &regularizer_obj, X, y, Z, stder_p);
  return ML::GLM::detail::qn_minimize(handle,
                                      w0,
                                      fx,
                                      num_iters,
                                      obj_function,
                                      l1,
                                      opt_param,
                                      static_cast<rapids_logger::level_enum>(pams.verbose));
}

template <typename T>
inline void qn_fit_x_mg(const raft::handle_t& handle,
                        const qn_params& pams,
                        SimpleMat<T>& X,
                        T* y_data,
                        int C,
                        T* w0_data,
                        T* f,
                        int* num_iters,
                        int64_t n_samples,
                        int rank,
                        int n_ranks,
                        const Standardizer<T>* stder_p = NULL,
                        T* sample_weight               = nullptr,
                        T svr_eps                      = 0)
{
  /*
   NB:
    N - number of data rows
    D - number of data columns (features)
    C - number of output classes

    X in R^[N, D]
    w in R^[D, C]
    y in {0, 1}^[N, C] or {cat}^N

    Dimensionality of w0 depends on loss, so we initialize it later.
   */
  cudaStream_t stream = handle.get_stream().get();
  int N               = X.m;
  int D               = X.n;
  int n_targets       = ML::GLM::detail::qn_is_classification(pams.loss) && C == 2 ? 1 : C;
  rmm::device_uvector<T> tmp(n_targets * N, stream);
  SimpleDenseMat<T> Z(tmp.data(), n_targets, N);
  SimpleVec<T> y(y_data, N);

  switch (pams.loss) {
    case QN_LOSS_LOGISTIC: {
      ASSERT(C > 0, "qn_mg.cuh: logistic loss invalid C");
      ML::GLM::detail::LogisticLoss<T> loss(handle, D, pams.fit_intercept);
      ML::GLM::opg::qn_fit_mg<T, decltype(loss)>(
        handle, pams, loss, X, y, Z, w0_data, f, num_iters, n_samples, rank, n_ranks, stder_p);
    } break;
    case QN_LOSS_SOFTMAX: {
      ASSERT(C > 2, "qn_mg.cuh: softmax invalid C");
      ML::GLM::detail::Softmax<T> loss(handle, D, C, pams.fit_intercept);
      ML::GLM::opg::qn_fit_mg<T, decltype(loss)>(
        handle, pams, loss, X, y, Z, w0_data, f, num_iters, n_samples, rank, n_ranks, stder_p);
    } break;
    default: {
      ASSERT(false, "qn_mg.cuh: unknown loss function type (id = %d).", pams.loss);
    }
  }
}

};  // namespace opg
};  // namespace GLM
};  // namespace ML
