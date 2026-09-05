/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cd.cuh"
#include "sgd.cuh"

#include <cuml/solvers/params.hpp>
#include <cuml/solvers/solver.hpp>

#include <raft/core/handle.hpp>

namespace ML {
namespace Solver {

using namespace ML;

void sgdFit(raft::handle_t& handle,
            float* input,
            int n_rows,
            int n_cols,
            float* labels,
            float* coef,
            float* intercept,
            bool fit_intercept,
            int batch_size,
            int epochs,
            int lr_type,
            float eta0,
            float power_t,
            int loss,
            int penalty,
            float alpha,
            float l1_ratio,
            bool shuffle,
            float tol,
            int n_iter_no_change)
{
  ML::loss_funct loss_funct = ML::loss_funct::SQRD_LOSS;
  if (loss == 0) {
    loss_funct = ML::loss_funct::SQRD_LOSS;
  } else if (loss == 1) {
    loss_funct = ML::loss_funct::LOG;
  } else if (loss == 2) {
    loss_funct = ML::loss_funct::HINGE;
  } else {
    ASSERT(false, "glm.cu: other functions are not supported yet.");
  }

  MLCommon::Functions::penalty pen;

  if (penalty == 0) {
    pen = MLCommon::Functions::penalty::NONE;
  } else if (penalty == 1) {
    pen = MLCommon::Functions::penalty::L1;
  } else if (penalty == 2) {
    pen = MLCommon::Functions::penalty::L2;
  } else if (penalty == 3) {
    pen = MLCommon::Functions::penalty::ELASTICNET;
  } else {
    ASSERT(false, "glm.cu: penalty is not supported yet.");
  }

  ML::lr_type learning_rate_type;
  if (lr_type == 0) {
    learning_rate_type = ML::lr_type::OPTIMAL;
  } else if (lr_type == 1) {
    learning_rate_type = ML::lr_type::CONSTANT;
  } else if (lr_type == 2) {
    learning_rate_type = ML::lr_type::INVSCALING;
  } else if (lr_type == 3) {
    learning_rate_type = ML::lr_type::ADAPTIVE;
  } else {
    ASSERT(false, "glm.cu: this learning rate type is not supported.");
  }

  sgdFit(handle,
         input,
         n_rows,
         n_cols,
         labels,
         coef,
         intercept,
         fit_intercept,
         batch_size,
         epochs,
         learning_rate_type,
         eta0,
         power_t,
         loss_funct,
         pen,
         alpha,
         l1_ratio,
         shuffle,
         tol,
         n_iter_no_change,
         handle.get_stream().get());
}

void sgdFit(raft::handle_t& handle,
            double* input,
            int n_rows,
            int n_cols,
            double* labels,
            double* coef,
            double* intercept,
            bool fit_intercept,
            int batch_size,
            int epochs,
            int lr_type,
            double eta0,
            double power_t,
            int loss,
            int penalty,
            double alpha,
            double l1_ratio,
            bool shuffle,
            double tol,
            int n_iter_no_change)
{
  ML::loss_funct loss_funct = ML::loss_funct::SQRD_LOSS;
  if (loss == 0) {
    loss_funct = ML::loss_funct::SQRD_LOSS;
  } else if (loss == 1) {
    loss_funct = ML::loss_funct::LOG;
  } else if (loss == 2) {
    loss_funct = ML::loss_funct::HINGE;
  } else {
    ASSERT(false, "glm.cu: other functions are not supported yet.");
  }

  MLCommon::Functions::penalty pen;
  if (penalty == 0) {
    pen = MLCommon::Functions::penalty::NONE;
  } else if (penalty == 1) {
    pen = MLCommon::Functions::penalty::L1;
  } else if (penalty == 2) {
    pen = MLCommon::Functions::penalty::L2;
  } else if (penalty == 3) {
    pen = MLCommon::Functions::penalty::ELASTICNET;
  } else {
    ASSERT(false, "glm.cu: penalty is not supported yet.");
  }

  ML::lr_type learning_rate_type;
  if (lr_type == 0) {
    learning_rate_type = ML::lr_type::OPTIMAL;
  } else if (lr_type == 1) {
    learning_rate_type = ML::lr_type::CONSTANT;
  } else if (lr_type == 2) {
    learning_rate_type = ML::lr_type::INVSCALING;
  } else if (lr_type == 3) {
    learning_rate_type = ML::lr_type::ADAPTIVE;
  } else {
    ASSERT(false, "glm.cu: this learning rate type is not supported.");
  }

  sgdFit(handle,
         input,
         n_rows,
         n_cols,
         labels,
         coef,
         intercept,
         fit_intercept,
         batch_size,
         epochs,
         learning_rate_type,
         eta0,
         power_t,
         loss_funct,
         pen,
         alpha,
         l1_ratio,
         shuffle,
         tol,
         n_iter_no_change,
         handle.get_stream().get());
}

void sgdPredict(raft::handle_t& handle,
                const float* input,
                int n_rows,
                int n_cols,
                const float* coef,
                float intercept,
                float* preds,
                int loss)
{
  ML::loss_funct loss_funct = ML::loss_funct::SQRD_LOSS;
  if (loss == 0) {
    loss_funct = ML::loss_funct::SQRD_LOSS;
  } else if (loss == 1) {
    loss_funct = ML::loss_funct::LOG;
  } else if (loss == 2) {
    loss_funct = ML::loss_funct::HINGE;
  } else {
    ASSERT(false, "glm.cu: other functions are not supported yet.");
  }

  sgdPredict(
    handle, input, n_rows, n_cols, coef, intercept, preds, loss_funct, handle.get_stream().get());
}

void sgdPredict(raft::handle_t& handle,
                const double* input,
                int n_rows,
                int n_cols,
                const double* coef,
                double intercept,
                double* preds,
                int loss)
{
  ML::loss_funct loss_funct = ML::loss_funct::SQRD_LOSS;
  if (loss == 0) {
    loss_funct = ML::loss_funct::SQRD_LOSS;
  } else if (loss == 1) {
    loss_funct = ML::loss_funct::LOG;
  } else if (loss == 2) {
    loss_funct = ML::loss_funct::HINGE;
  } else {
    ASSERT(false, "glm.cu: other functions are not supported yet.");
  }

  sgdPredict(
    handle, input, n_rows, n_cols, coef, intercept, preds, loss_funct, handle.get_stream().get());
}

void sgdPredictBinaryClass(raft::handle_t& handle,
                           const float* input,
                           int n_rows,
                           int n_cols,
                           const float* coef,
                           float intercept,
                           float* preds,
                           int loss)
{
  ML::loss_funct loss_funct = ML::loss_funct::SQRD_LOSS;
  if (loss == 0) {
    loss_funct = ML::loss_funct::SQRD_LOSS;
  } else if (loss == 1) {
    loss_funct = ML::loss_funct::LOG;
  } else if (loss == 2) {
    loss_funct = ML::loss_funct::HINGE;
  } else {
    ASSERT(false, "glm.cu: other functions are not supported yet.");
  }

  sgdPredictBinaryClass(
    handle, input, n_rows, n_cols, coef, intercept, preds, loss_funct, handle.get_stream().get());
}

void sgdPredictBinaryClass(raft::handle_t& handle,
                           const double* input,
                           int n_rows,
                           int n_cols,
                           const double* coef,
                           double intercept,
                           double* preds,
                           int loss)
{
  ML::loss_funct loss_funct = ML::loss_funct::SQRD_LOSS;
  if (loss == 0) {
    loss_funct = ML::loss_funct::SQRD_LOSS;
  } else if (loss == 1) {
    loss_funct = ML::loss_funct::LOG;
  } else if (loss == 2) {
    loss_funct = ML::loss_funct::HINGE;
  } else {
    ASSERT(false, "glm.cu: other functions are not supported yet.");
  }

  sgdPredictBinaryClass(
    handle, input, n_rows, n_cols, coef, intercept, preds, loss_funct, handle.get_stream().get());
}

int cdFit(raft::handle_t& handle,
          float* input,
          int n_rows,
          int n_cols,
          float* labels,
          float* coef,
          float* intercept,
          bool fit_intercept,
          int epochs,
          int loss,
          float alpha,
          float l1_ratio,
          bool shuffle,
          float tol,
          float* sample_weight)
{
  ASSERT(loss == 0, "Parameter loss: Only SQRT_LOSS function is supported for now");

  ML::loss_funct loss_funct = ML::loss_funct::SQRD_LOSS;

  return cdFit<float>(handle,
                      input,
                      n_rows,
                      n_cols,
                      labels,
                      coef,
                      intercept,
                      fit_intercept,
                      epochs,
                      loss_funct,
                      alpha,
                      l1_ratio,
                      shuffle,
                      tol,
                      sample_weight);
}

int cdFit(raft::handle_t& handle,
          double* input,
          int n_rows,
          int n_cols,
          double* labels,
          double* coef,
          double* intercept,
          bool fit_intercept,
          int epochs,
          int loss,
          double alpha,
          double l1_ratio,
          bool shuffle,
          double tol,
          double* sample_weight)
{
  ASSERT(loss == 0, "Parameter loss: Only SQRT_LOSS function is supported for now");

  ML::loss_funct loss_funct = ML::loss_funct::SQRD_LOSS;

  return cdFit<double>(handle,
                       input,
                       n_rows,
                       n_cols,
                       labels,
                       coef,
                       intercept,
                       fit_intercept,
                       epochs,
                       loss_funct,
                       alpha,
                       l1_ratio,
                       shuffle,
                       tol,
                       sample_weight);
}

void cdPredict(raft::handle_t& handle,
               const float* input,
               int n_rows,
               int n_cols,
               const float* coef,
               float intercept,
               float* preds,
               int loss)
{
  ML::loss_funct loss_funct = ML::loss_funct::SQRD_LOSS;
  if (loss == 0) {
    loss_funct = ML::loss_funct::SQRD_LOSS;
  } else {
    ASSERT(false, "glm.cu: other functions are not supported yet.");
  }

  cdPredict<float>(handle, input, n_rows, n_cols, coef, intercept, preds, loss_funct);
}

void cdPredict(raft::handle_t& handle,
               const double* input,
               int n_rows,
               int n_cols,
               const double* coef,
               double intercept,
               double* preds,
               int loss)
{
  ML::loss_funct loss_funct = ML::loss_funct::SQRD_LOSS;
  if (loss == 0) {
    loss_funct = ML::loss_funct::SQRD_LOSS;
  } else {
    ASSERT(false, "glm.cu: other functions are not supported yet.");
  }

  cdPredict<double>(handle, input, n_rows, n_cols, coef, intercept, preds, loss_funct);
}

}  // namespace Solver
}  // namespace ML
