/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "hw_eval.cuh"
#include "hw_utils.cuh"

#include <cuml/common/checked_arithmetic.hpp>

#include <raft/util/cudart_utils.hpp>

template <typename Dtype>
__device__ Dtype golden_step(Dtype a, Dtype b, Dtype c)
{
  Dtype mid = (a + c) * 0.5;
  if (b > mid)
    return (a - b) * GOLD;
  else
    return (c - b) * GOLD;
}

template <typename Dtype>
__device__ Dtype fix_step(Dtype a, Dtype b, Dtype c, Dtype step, Dtype e)
{
  Dtype min_step = abs_device(e * b) + PG_EPS;
  if (abs_device(step) < min_step) return step > 0 ? min_step : -min_step;
  if (abs_device(b + step - a) <= e || abs_device(b + step - c) <= e)
    return 0.0;  // steps are too close to each others
  return step;
}

template <typename Dtype>
__device__ Dtype calculate_step(
  Dtype a, Dtype b, Dtype c, Dtype loss_a, Dtype loss_b, Dtype loss_c, Dtype pstep, Dtype e)
{
  // parabola step
  Dtype p    = (b - a) * (loss_b - loss_c);
  Dtype q    = (b - c) * (loss_b - loss_a);
  Dtype x    = q * (b - c) - p * (b - a);
  Dtype y    = (p - q) * 2.;
  Dtype step = abs_device(y) < PG_EPS ? golden_step(a, b, c) : x / y;
  step       = fix_step(a, b, c, step, e);  // ensure point is new

  if (abs_device(step) > abs_device(pstep / 2) || step == 0.0) step = golden_step(a, b, c);
  return step;
}

template <typename Dtype>
__device__ void parabolic_interpolation_golden_optim(int tid,
                                                     const Dtype* ts,
                                                     int n,
                                                     int batch_size,
                                                     int frequency,
                                                     int shift,
                                                     Dtype plevel,
                                                     Dtype ptrend,
                                                     Dtype* pseason,
                                                     int pseason_width,
                                                     const Dtype* start_season,
                                                     const Dtype* beta,
                                                     const Dtype* gamma,
                                                     bool optim_alpha,
                                                     Dtype* alpha_,
                                                     bool optim_beta,
                                                     Dtype* beta_,
                                                     bool optim_gamma,
                                                     Dtype* gamma_,
                                                     Dtype eps,
                                                     bool ADDITIVE_KERNEL)
{
  Dtype a = (Dtype).0;
  Dtype b = (Dtype)GOLD;
  Dtype c = (Dtype)1.;

  Dtype loss_a = holtwinters_eval_device<Dtype>(tid,
                                                ts,
                                                n,
                                                batch_size,
                                                frequency,
                                                shift,
                                                plevel,
                                                ptrend,
                                                pseason,
                                                pseason_width,
                                                start_season,
                                                beta,
                                                gamma,
                                                optim_alpha ? a : *alpha_,
                                                optim_beta ? a : *beta_,
                                                optim_gamma ? a : *gamma_,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                ADDITIVE_KERNEL);
  Dtype loss_b = holtwinters_eval_device<Dtype>(tid,
                                                ts,
                                                n,
                                                batch_size,
                                                frequency,
                                                shift,
                                                plevel,
                                                ptrend,
                                                pseason,
                                                pseason_width,
                                                start_season,
                                                beta,
                                                gamma,
                                                optim_alpha ? b : *alpha_,
                                                optim_beta ? b : *beta_,
                                                optim_gamma ? b : *gamma_,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                ADDITIVE_KERNEL);
  Dtype loss_c = holtwinters_eval_device<Dtype>(tid,
                                                ts,
                                                n,
                                                batch_size,
                                                frequency,
                                                shift,
                                                plevel,
                                                ptrend,
                                                pseason,
                                                pseason_width,
                                                start_season,
                                                beta,
                                                gamma,
                                                optim_alpha ? c : *alpha_,
                                                optim_beta ? c : *beta_,
                                                optim_gamma ? c : *gamma_,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                ADDITIVE_KERNEL);

  Dtype pstep = (c - a) / 2;
  Dtype cstep = pstep;

  while (abs_device(c - a) > abs_device(b * eps) + PG_EPS) {
    Dtype step      = calculate_step(a, b, c, loss_a, loss_b, loss_c, cstep, eps);
    Dtype optim_val = b + step;
    Dtype loss_val  = holtwinters_eval_device<Dtype>(tid,
                                                    ts,
                                                    n,
                                                    batch_size,
                                                    frequency,
                                                    shift,
                                                    plevel,
                                                    ptrend,
                                                    pseason,
                                                    pseason_width,
                                                    start_season,
                                                    beta,
                                                    gamma,
                                                    optim_alpha ? optim_val : *alpha_,
                                                    optim_beta ? optim_val : *beta_,
                                                    optim_gamma ? optim_val : *gamma_,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    ADDITIVE_KERNEL);
    if (loss_val < loss_b) {
      if (optim_val > b) {
        SUBSTITUTE(a, b, loss_a, loss_b);
      } else {
        SUBSTITUTE(c, b, loss_c, loss_b);
      }
      SUBSTITUTE(b, optim_val, loss_b, loss_val);
    } else {
      if (optim_val > b) {
        SUBSTITUTE(c, optim_val, loss_c, loss_val);
      } else {
        SUBSTITUTE(a, optim_val, loss_a, loss_val);
      }
    }
    SUBSTITUTE(cstep, pstep, pstep, step);
  }
  if (optim_alpha) *alpha_ = b;
  if (optim_beta) *beta_ = b;
  if (optim_gamma) *gamma_ = b;
}

template <typename Dtype>
__device__ void holtwinters_finite_gradient_device(int tid,
                                                   const Dtype* ts,
                                                   int n,
                                                   int batch_size,
                                                   int frequency,
                                                   int shift,
                                                   Dtype plevel,
                                                   Dtype ptrend,
                                                   Dtype* pseason,
                                                   int pseason_width,
                                                   const Dtype* start_season,
                                                   const Dtype* beta,
                                                   const Dtype* gamma,
                                                   Dtype alpha_,
                                                   Dtype beta_,
                                                   Dtype gamma_,
                                                   Dtype* g_alpha,
                                                   Dtype* g_beta,
                                                   Dtype* g_gamma,
                                                   Dtype eps,
                                                   bool ADDITIVE_KERNEL)
{
  Dtype left_error, right_error;
  if (g_alpha) {  // alpha gradient
    left_error  = holtwinters_eval_device<Dtype>(tid,
                                                ts,
                                                n,
                                                batch_size,
                                                frequency,
                                                shift,
                                                plevel,
                                                ptrend,
                                                pseason,
                                                pseason_width,
                                                start_season,
                                                beta,
                                                gamma,
                                                alpha_ - eps,
                                                beta_,
                                                gamma_,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                ADDITIVE_KERNEL);
    right_error = holtwinters_eval_device<Dtype>(tid,
                                                 ts,
                                                 n,
                                                 batch_size,
                                                 frequency,
                                                 shift,
                                                 plevel,
                                                 ptrend,
                                                 pseason,
                                                 pseason_width,
                                                 start_season,
                                                 beta,
                                                 gamma,
                                                 alpha_ + eps,
                                                 beta_,
                                                 gamma_,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 ADDITIVE_KERNEL);
    *g_alpha    = (right_error - left_error) / (eps * 2.);
  }
  if (g_beta) {  // beta gradient
    left_error  = holtwinters_eval_device<Dtype>(tid,
                                                ts,
                                                n,
                                                batch_size,
                                                frequency,
                                                shift,
                                                plevel,
                                                ptrend,
                                                pseason,
                                                pseason_width,
                                                start_season,
                                                beta,
                                                gamma,
                                                alpha_,
                                                beta_ - eps,
                                                gamma_,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                ADDITIVE_KERNEL);
    right_error = holtwinters_eval_device<Dtype>(tid,
                                                 ts,
                                                 n,
                                                 batch_size,
                                                 frequency,
                                                 shift,
                                                 plevel,
                                                 ptrend,
                                                 pseason,
                                                 pseason_width,
                                                 start_season,
                                                 beta,
                                                 gamma,
                                                 alpha_,
                                                 beta_ + eps,
                                                 gamma_,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 ADDITIVE_KERNEL);
    *g_beta     = (right_error - left_error) / (eps * 2.);
  }
  if (g_gamma) {  // gamma gradient
    left_error  = holtwinters_eval_device<Dtype>(tid,
                                                ts,
                                                n,
                                                batch_size,
                                                frequency,
                                                shift,
                                                plevel,
                                                ptrend,
                                                pseason,
                                                pseason_width,
                                                start_season,
                                                beta,
                                                gamma,
                                                alpha_,
                                                beta_,
                                                gamma_ - eps,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                ADDITIVE_KERNEL);
    right_error = holtwinters_eval_device<Dtype>(tid,
                                                 ts,
                                                 n,
                                                 batch_size,
                                                 frequency,
                                                 shift,
                                                 plevel,
                                                 ptrend,
                                                 pseason,
                                                 pseason_width,
                                                 start_season,
                                                 beta,
                                                 gamma,
                                                 alpha_,
                                                 beta_,
                                                 gamma_ + eps,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 ADDITIVE_KERNEL);
    *g_gamma    = (right_error - left_error) / (eps * 2.);
  }
}

// There's a bug here, where the wrong values are updated if line search
// iter limit is reached. Last values are of nx are put in x, whereas it
// should be the nx values which minimised loss. For summary, check
// https://github.com/NVIDIA/cuml/issues/888
template <typename Dtype>
__device__ ML::OptimCriterion holtwinters_bfgs_optim_device(
  int tid,
  const Dtype* ts,
  int n,
  int batch_size,
  int frequency,
  int shift,
  Dtype plevel,
  Dtype ptrend,
  Dtype* pseason,
  int pseason_width,
  const Dtype* start_season,
  const Dtype* beta,
  const Dtype* gamma,
  bool optim_alpha,
  Dtype* x1,
  bool optim_beta,
  Dtype* x2,
  bool optim_gamma,
  Dtype* x3,
  const ML::OptimParams<Dtype> optim_params,
  bool ADDITIVE_KERNEL)
{
  Dtype H11 = 1., H12 = .0, H13 = .0, H22 = 1., H23 = .0,
        H33 = 1.;                   // Hessian approximiation (Hessian is symmetric)
  Dtype g1 = .0, g2 = .0, g3 = .0;  // gradients

  // initial gradient
  holtwinters_finite_gradient_device<Dtype>(tid,
                                            ts,
                                            n,
                                            batch_size,
                                            frequency,
                                            shift,
                                            plevel,
                                            ptrend,
                                            pseason,
                                            pseason_width,
                                            start_season,
                                            beta,
                                            gamma,
                                            *x1,
                                            *x2,
                                            *x3,
                                            optim_alpha ? &g1 : nullptr,
                                            optim_beta ? &g2 : nullptr,
                                            optim_gamma ? &g3 : nullptr,
                                            optim_params.eps,
                                            ADDITIVE_KERNEL);

  for (int iter = 0; iter < optim_params.bfgs_iter_limit; ++iter) {
    // Step direction
    Dtype p1 = -H11 * g1 - H12 * g2 - H13 * g3;
    Dtype p2 = -H12 * g1 - H22 * g2 - H23 * g3;
    Dtype p3 = -H13 * g1 - H23 * g2 - H33 * g3;

    const Dtype phi = p1 * g1 + p2 * g2 + p3 * g3;
    if (phi > 0) {
      H11 = 1.;
      H12 = 0.;
      H13 = 0.;
      H22 = 1.;
      H23 = 0.;
      H33 = 1.;
      p1  = -g1;
      p2  = -g2;
      p3  = -g3;
    }

    // {next_params} = {params}+step_size*p;
    // start of line search

    // starting step size, we assume the largest distance between x and nx is going to be sqrt(3)/2.
    // where sqrt(3) is the largest allowed step in a 1x1x1 cube.
    Dtype step_size;
    if (optim_params.linesearch_step_size <= 0)
      step_size = (Dtype)0.866 / sqrt(p1 * p1 + p2 * p2 + p3 * p3);
    else
      step_size = optim_params.linesearch_step_size;
    Dtype nx1 = *x1 + step_size * p1;
    Dtype nx2 = *x2 + step_size * p2;
    Dtype nx3 = *x3 + step_size * p3;

    // line search params
    const Dtype cauchy   = optim_params.linesearch_c * (g1 * p1 + g2 * p2 + g3 * p3);
    const Dtype loss_ref = holtwinters_eval_device<Dtype>(tid,
                                                          ts,
                                                          n,
                                                          batch_size,
                                                          frequency,
                                                          shift,
                                                          plevel,
                                                          ptrend,
                                                          pseason,
                                                          pseason_width,
                                                          start_season,
                                                          beta,
                                                          gamma,
                                                          *x1,
                                                          *x2,
                                                          *x3,
                                                          nullptr,
                                                          nullptr,
                                                          nullptr,
                                                          nullptr,
                                                          ADDITIVE_KERNEL);
    Dtype loss           = holtwinters_eval_device<Dtype>(tid,
                                                ts,
                                                n,
                                                batch_size,
                                                frequency,
                                                shift,
                                                plevel,
                                                ptrend,
                                                pseason,
                                                pseason_width,
                                                start_season,
                                                beta,
                                                gamma,
                                                nx1,
                                                nx2,
                                                nx3,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                ADDITIVE_KERNEL);

    for (int i = 0;
         i < optim_params.linesearch_iter_limit && (loss > loss_ref + step_size * cauchy);
         ++i) {
      step_size *= optim_params.linesearch_tau;
      nx1  = *x1 + step_size * p1;
      nx2  = *x2 + step_size * p2;
      nx3  = *x3 + step_size * p3;
      loss = holtwinters_eval_device<Dtype>(tid,
                                            ts,
                                            n,
                                            batch_size,
                                            frequency,
                                            shift,
                                            plevel,
                                            ptrend,
                                            pseason,
                                            pseason_width,
                                            start_season,
                                            beta,
                                            gamma,
                                            nx1,
                                            nx2,
                                            nx3,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            ADDITIVE_KERNEL);
    }
    // end of line search

    // see if new {params} meet stop condition
    const Dtype dx1 = abs_device(*x1 - nx1);
    const Dtype dx2 = abs_device(*x2 - nx2);
    const Dtype dx3 = abs_device(*x3 - nx3);
    Dtype max       = max3(dx1, dx2, dx3);
    // update {params}
    *x1 = nx1;
    *x2 = nx2;
    *x3 = nx3;
    if (optim_params.min_param_diff > max) return ML::OptimCriterion::OPTIM_MIN_PARAM_DIFF;
    if (optim_params.min_error_diff > abs_device(loss - loss_ref))
      return ML::OptimCriterion::OPTIM_MIN_ERROR_DIFF;

    Dtype ng1 = .0, ng2 = .0, ng3 = .0;  // next gradient
    holtwinters_finite_gradient_device<Dtype>(tid,
                                              ts,
                                              n,
                                              batch_size,
                                              frequency,
                                              shift,
                                              plevel,
                                              ptrend,
                                              pseason,
                                              pseason_width,
                                              start_season,
                                              beta,
                                              gamma,
                                              nx1,
                                              nx2,
                                              nx3,
                                              optim_alpha ? &ng1 : nullptr,
                                              optim_beta ? &ng2 : nullptr,
                                              optim_gamma ? &ng3 : nullptr,
                                              optim_params.eps,
                                              ADDITIVE_KERNEL);
    // see if new gradients meet stop condition
    max = max3(abs_device(ng1), abs_device(ng2), abs_device(ng3));
    if (optim_params.min_grad_norm > max) return ML::OptimCriterion::OPTIM_MIN_GRAD_NORM;

    // s = step_size*p;
    const Dtype s1 = step_size * p1;
    const Dtype s2 = step_size * p2;
    const Dtype s3 = step_size * p3;

    // y = next_grad-grad
    const Dtype y1 = ng1 - g1;
    const Dtype y2 = ng2 - g2;
    const Dtype y3 = ng3 - g3;

    // rho_ = y(*)s; rho = 1/rho_
    const Dtype rho_ = y1 * s1 + y2 * s2 + y3 * s3;
    const Dtype rho  = 1.0 / rho_;

    const Dtype Hy1 = H11 * y1 + H12 * y2 + H13 * y3;
    const Dtype Hy2 = H12 * y1 + H22 * y2 + H23 * y3;
    const Dtype Hy3 = H13 * y1 + H23 * y2 + H33 * y3;
    const Dtype k   = rho * rho * (y1 * Hy1 + y2 * Hy2 + y3 * Hy3 + rho_);

    H11 += k * s1 * s1 - 2. * rho * s1 * Hy1;
    H12 += k * s1 * s2 - rho * (s2 * Hy1 + s1 * Hy2);
    H13 += k * s1 * s3 - rho * (s3 * Hy1 + s1 * Hy3);
    H22 += k * s2 * s2 - 2 * rho * s2 * Hy2;
    H23 += k * s2 * s3 - rho * (s3 * Hy2 + s2 * Hy3);
    H33 += k * s3 * s3 - 2. * rho * s3 * Hy3;

    g1 = ng1;
    g2 = ng2;
    g3 = ng3;
  }

  return ML::OptimCriterion::OPTIM_BFGS_ITER_LIMIT;
}

template <typename Dtype>
CUML_KERNEL void holtwinters_optim_gpu_shared_kernel(const Dtype* ts,
                                                     int n,
                                                     int batch_size,
                                                     int frequency,
                                                     const Dtype* start_level,
                                                     const Dtype* start_trend,
                                                     const Dtype* start_season,
                                                     Dtype* alpha,
                                                     bool optim_alpha,
                                                     Dtype* beta,
                                                     bool optim_beta,
                                                     Dtype* gamma,
                                                     bool optim_gamma,
                                                     Dtype* level,
                                                     Dtype* trend,
                                                     Dtype* season,
                                                     Dtype* xhat,
                                                     Dtype* error,
                                                     ML::OptimCriterion* optim_result,
                                                     const ML::OptimParams<Dtype> optim_params,
                                                     bool ADDITIVE_KERNEL,
                                                     bool single_param)
{
  int tid = GET_TID;
  extern __shared__ unsigned char pseason_[];
  Dtype* pseason = reinterpret_cast<Dtype*>(pseason_);

  if (tid < batch_size) {
    int shift = 1;
    ML::OptimCriterion optim;
    Dtype plevel = start_level[tid], ptrend = .0;
    Dtype alpha_ = alpha[tid];
    Dtype beta_  = beta ? beta[tid] : .0;
    Dtype gamma_ = gamma ? gamma[tid] : .0;

    if (gamma) {
      shift  = frequency;
      ptrend = beta ? start_trend[tid] : .0;
    } else if (beta) {
      shift  = 2;
      ptrend = start_trend[tid];
    }

    // Optimization
    if (single_param)
      parabolic_interpolation_golden_optim<Dtype>(tid,
                                                  ts,
                                                  n,
                                                  batch_size,
                                                  frequency,
                                                  shift,
                                                  plevel,
                                                  ptrend,
                                                  pseason + threadIdx.x,
                                                  blockDim.x,
                                                  start_season,
                                                  beta,
                                                  gamma,
                                                  optim_alpha,
                                                  &alpha_,
                                                  optim_beta,
                                                  &beta_,
                                                  optim_gamma,
                                                  &gamma_,
                                                  optim_params.eps,
                                                  ADDITIVE_KERNEL);
    else
      optim = holtwinters_bfgs_optim_device<Dtype>(tid,
                                                   ts,
                                                   n,
                                                   batch_size,
                                                   frequency,
                                                   shift,
                                                   plevel,
                                                   ptrend,
                                                   pseason + threadIdx.x,
                                                   blockDim.x,
                                                   start_season,
                                                   beta,
                                                   gamma,
                                                   optim_alpha,
                                                   &alpha_,
                                                   optim_beta,
                                                   &beta_,
                                                   optim_gamma,
                                                   &gamma_,
                                                   optim_params,
                                                   ADDITIVE_KERNEL);

    if (optim_alpha) alpha[tid] = bound_device(alpha_);
    if (optim_beta) beta[tid] = bound_device(beta_);
    if (optim_gamma) gamma[tid] = bound_device(gamma_);
    if (!single_param && optim_result) optim_result[tid] = optim;

    if (error || level || trend || season || xhat) {
      // Final fit
      Dtype error_ = holtwinters_eval_device<Dtype>(tid,
                                                    ts,
                                                    n,
                                                    batch_size,
                                                    frequency,
                                                    shift,
                                                    plevel,
                                                    ptrend,
                                                    pseason + threadIdx.x,
                                                    blockDim.x,
                                                    start_season,
                                                    beta,
                                                    gamma,
                                                    alpha_,
                                                    beta_,
                                                    gamma_,
                                                    level,
                                                    trend,
                                                    season,
                                                    xhat,
                                                    ADDITIVE_KERNEL);
      if (error) error[tid] = error_;
    }
  }
}

template <typename Dtype>
CUML_KERNEL void holtwinters_optim_gpu_global_kernel(const Dtype* ts,
                                                     int n,
                                                     int batch_size,
                                                     int frequency,
                                                     const Dtype* start_level,
                                                     const Dtype* start_trend,
                                                     const Dtype* start_season,
                                                     Dtype* pseason,
                                                     Dtype* alpha,
                                                     bool optim_alpha,
                                                     Dtype* beta,
                                                     bool optim_beta,
                                                     Dtype* gamma,
                                                     bool optim_gamma,
                                                     Dtype* level,
                                                     Dtype* trend,
                                                     Dtype* season,
                                                     Dtype* xhat,
                                                     Dtype* error,
                                                     ML::OptimCriterion* optim_result,
                                                     const ML::OptimParams<Dtype> optim_params,
                                                     bool ADDITIVE_KERNEL,
                                                     bool single_param)
{
  int tid = GET_TID;
  if (tid < batch_size) {
    int shift = 1;
    ML::OptimCriterion optim;
    Dtype plevel = start_level[tid], ptrend = .0;
    Dtype alpha_ = alpha[tid];
    Dtype beta_  = beta ? beta[tid] : .0;
    Dtype gamma_ = gamma ? gamma[tid] : .0;

    if (gamma) {
      shift  = frequency;
      ptrend = beta ? start_trend[tid] : .0;
    } else if (beta) {
      shift  = 2;
      ptrend = start_trend[tid];
    }

    // Optimization
    if (single_param)
      parabolic_interpolation_golden_optim<Dtype>(tid,
                                                  ts,
                                                  n,
                                                  batch_size,
                                                  frequency,
                                                  shift,
                                                  plevel,
                                                  ptrend,
                                                  pseason + tid,
                                                  batch_size,
                                                  start_season,
                                                  beta,
                                                  gamma,
                                                  optim_alpha,
                                                  &alpha_,
                                                  optim_beta,
                                                  &beta_,
                                                  optim_gamma,
                                                  &gamma_,
                                                  optim_params.eps,
                                                  ADDITIVE_KERNEL);
    else
      optim = holtwinters_bfgs_optim_device<Dtype>(tid,
                                                   ts,
                                                   n,
                                                   batch_size,
                                                   frequency,
                                                   shift,
                                                   plevel,
                                                   ptrend,
                                                   pseason + tid,
                                                   batch_size,
                                                   start_season,
                                                   beta,
                                                   gamma,
                                                   optim_alpha,
                                                   &alpha_,
                                                   optim_beta,
                                                   &beta_,
                                                   optim_gamma,
                                                   &gamma_,
                                                   optim_params,
                                                   ADDITIVE_KERNEL);

    if (optim_alpha) alpha[tid] = bound_device(alpha_);
    if (optim_beta) beta[tid] = bound_device(beta_);
    if (optim_gamma) gamma[tid] = bound_device(gamma_);
    if (!single_param && optim_result) optim_result[tid] = optim;

    if (error || level || trend || season || xhat) {
      // Final fit
      Dtype error_ = holtwinters_eval_device<Dtype>(tid,
                                                    ts,
                                                    n,
                                                    batch_size,
                                                    frequency,
                                                    shift,
                                                    plevel,
                                                    ptrend,
                                                    pseason + tid,
                                                    batch_size,
                                                    start_season,
                                                    beta,
                                                    gamma,
                                                    alpha_,
                                                    beta_,
                                                    gamma_,
                                                    level,
                                                    trend,
                                                    season,
                                                    xhat,
                                                    ADDITIVE_KERNEL);
      if (error) error[tid] = error_;
    }
  }
}

// Test Global and Shared kernels
// https://github.com/NVIDIA/cuml/issues/890
template <typename Dtype>
void holtwinters_optim_gpu(const raft::handle_t& handle,
                           const Dtype* ts,
                           int n,
                           int batch_size,
                           int frequency,
                           const Dtype* start_level,
                           const Dtype* start_trend,
                           const Dtype* start_season,
                           Dtype* alpha,
                           bool optim_alpha,
                           Dtype* beta,
                           bool optim_beta,
                           Dtype* gamma,
                           bool optim_gamma,
                           Dtype* level,
                           Dtype* trend,
                           Dtype* season,
                           Dtype* xhat,
                           Dtype* error,
                           ML::OptimCriterion* optim_result,
                           ML::SeasonalType seasonal,
                           const ML::OptimParams<Dtype> optim_params)
{
  cudaStream_t stream = handle.get_stream();

  // int total_blocks = GET_NUM_BLOCKS(batch_size);
  // int threads_per_block = GET_THREADS_PER_BLOCK(batch_size);
  int total_blocks      = (batch_size - 1) / 128 + 1;
  int threads_per_block = 128;

  // How much sm needed for shared kernel
  int sm_needed     = sizeof(Dtype) * threads_per_block * frequency;
  bool is_additive  = seasonal == ML::SeasonalType::ADDITIVE;
  bool single_param = (optim_alpha + optim_beta + optim_gamma > 1) ? false : true;

  if (sm_needed > raft::getSharedMemPerBlock()) {  // Global memory //
    rmm::device_uvector<Dtype> pseason(ML::checked_mul<std::size_t>(batch_size, frequency), stream);
    holtwinters_optim_gpu_global_kernel<Dtype>
      <<<total_blocks, threads_per_block, 0, stream>>>(ts,
                                                       n,
                                                       batch_size,
                                                       frequency,
                                                       start_level,
                                                       start_trend,
                                                       start_season,
                                                       pseason.data(),
                                                       alpha,
                                                       optim_alpha,
                                                       beta,
                                                       optim_beta,
                                                       gamma,
                                                       optim_gamma,
                                                       level,
                                                       trend,
                                                       season,
                                                       xhat,
                                                       error,
                                                       optim_result,
                                                       optim_params,
                                                       is_additive,
                                                       single_param);
  } else {  // Shared memory
    holtwinters_optim_gpu_shared_kernel<Dtype>
      <<<total_blocks, threads_per_block, sm_needed, stream>>>(ts,
                                                               n,
                                                               batch_size,
                                                               frequency,
                                                               start_level,
                                                               start_trend,
                                                               start_season,
                                                               alpha,
                                                               optim_alpha,
                                                               beta,
                                                               optim_beta,
                                                               gamma,
                                                               optim_gamma,
                                                               level,
                                                               trend,
                                                               season,
                                                               xhat,
                                                               error,
                                                               optim_result,
                                                               optim_params,
                                                               is_additive,
                                                               single_param);
  }
}
