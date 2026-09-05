/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/*
 * This file contains implementations of two popular Quasi-Newton methods:
 * - Limited-memory Broyden Fletcher Goldfarb Shanno (L-BFGS) [Nocedal, Wright -
 * Numerical Optimization (1999)]
 * - Orthant-wise limited-memory quasi-newton (OWL-QN) [Andrew, Gao - ICML 2007]
 *   https://www.microsoft.com/en-us/research/publication/scalable-training-of-l1-regularized-log-linear-models/
 *
 * L-BFGS is a classical method to solve unconstrained optimization problems of
 * differentiable multi-variate functions f: R^D \mapsto R, i.e. it solves
 *
 * \min_{x \in R^D} f(x)
 *
 * iteratively by building up a m-dimensional (inverse) Hessian approximation.
 *
 * OWL-QN is an extension of L-BFGS that is specifically designed to optimize
 * functions of the form
 *
 * f(x) + \lambda * \sum_i |x_i|,
 *
 * i.e. functions with an l1 penalty, by leveraging that |z| is differentiable
 * when restricted to an orthant.
 *
 */

#include "qn_linesearch.cuh"
#include "qn_util.cuh"
#include "simple_mat.cuh"

#include <cuml/common/logger.hpp>

#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>

namespace ML {
namespace GLM {
namespace detail {

// TODO better way to deal with alignment? Smaller align possible?
constexpr size_t qn_align = 256;

template <typename T>
inline size_t lbfgs_workspace_size(const LBFGSParam<T>& param, const int n)
{
  size_t mat_size = raft::alignTo<size_t>(sizeof(T) * param.m * n, qn_align);
  size_t vec_size = raft::alignTo<size_t>(sizeof(T) * n, qn_align);
  return 2 * mat_size + 4 * vec_size + qn_align;
}

template <typename T>
inline size_t owlqn_workspace_size(const LBFGSParam<T>& param, const int n)
{
  size_t vec_size = raft::alignTo<size_t>(sizeof(T) * n, qn_align);
  return lbfgs_workspace_size(param, n) + vec_size;
}

template <typename T>
inline bool update_and_check(const char* solver,
                             const LBFGSParam<T>& param,
                             int iter,
                             LINE_SEARCH_RETCODE lsret,
                             T& fx,
                             T& fxp,
                             const T& gnorm,
                             ML::SimpleVec<T>& x,
                             ML::SimpleVec<T>& xp,
                             ML::SimpleVec<T>& grad,
                             ML::SimpleVec<T>& gradp,
                             std::vector<T>& fx_hist,
                             T* dev_scalar,
                             OPT_RETCODE& outcode,
                             cudaStream_t stream)
{
  bool stop      = false;
  bool converged = false;
  bool isLsValid = !isnan(fx) && !isinf(fx);
  // Linesearch may fail to converge, but still come closer to the solution;
  // if that is not the case, let `check_convergence` ("insufficient change")
  // below terminate the loop.
  bool isLsNonCritical = lsret == LS_INVALID_STEP_MIN || lsret == LS_MAX_ITERS_REACHED;
  // If the error is not critical, check that the target function does not grow.
  // This shouldn't really happen, but weird things can happen if the convergence
  // thresholds are too small.
  bool isLsInDoubt = isLsValid && fx <= fxp + param.ftol && isLsNonCritical;
  bool isLsSuccess = lsret == LS_SUCCESS || isLsInDoubt;

  CUML_LOG_TRACE("%s iteration %d, fx=%f", solver, iter, fx);

  // if the target is at least finite, we can check the convergence
  if (isLsValid) converged = check_convergence(param, iter, fx, gnorm, fx_hist);

  if (!isLsSuccess && !converged) {
    CUML_LOG_WARN(
      "%s line search failed (code %d); stopping at the last valid step", solver, lsret);
    outcode = OPT_LS_FAILED;
    stop    = true;
  } else if (!isLsValid) {
    CUML_LOG_ERROR(
      "%s error fx=%f at iteration %d; stopping at the last valid step", solver, fx, iter);
    outcode = OPT_NUMERIC_ERROR;
    stop    = true;
  } else if (converged) {
    CUML_LOG_DEBUG("%s converged", solver);
    outcode = OPT_SUCCESS;
    stop    = true;
  } else if (isLsInDoubt && fx + param.ftol >= fxp) {
    // If a non-critical error has happened during the line search, check if the target
    // is improved at least a bit. Otherwise, stop to avoid spinning till the iteration limit.
    CUML_LOG_WARN(
      "%s stopped, because the line search failed to advance (step delta = %f)", solver, fx - fxp);
    outcode = OPT_LS_FAILED;
    stop    = true;
  }

  // if linesearch wasn't successful, undo the update.
  if (!isLsSuccess || !isLsValid) {
    fx = fxp;
    x.copy_async(xp, stream);
    grad.copy_async(gradp, stream);
  }

  return stop;
}

template <typename T, typename Function>
inline OPT_RETCODE min_lbfgs(const LBFGSParam<T>& param,
                             Function& f,              // function to minimize
                             SimpleVec<T>& x,          // initial point, holds result
                             T& fx,                    // output function value
                             int* k,                   // output iterations
                             SimpleVec<T>& workspace,  // scratch space
                             cudaStream_t stream,
                             rapids_logger::level_enum verbosity = 0)
{
  int n                    = x.len;
  const int workspace_size = lbfgs_workspace_size(param, n);
  ASSERT(workspace.len >= workspace_size, "LBFGS: workspace insufficient");

  // SETUP WORKSPACE
  size_t mat_size = raft::alignTo<size_t>(sizeof(T) * param.m * n, qn_align);
  size_t vec_size = raft::alignTo<size_t>(sizeof(T) * n, qn_align);
  T* p_ws         = workspace.data;
  SimpleDenseMat<T> S(p_ws, n, param.m);
  p_ws += mat_size;
  SimpleDenseMat<T> Y(p_ws, n, param.m);
  p_ws += mat_size;
  SimpleVec<T> xp(p_ws, n);
  p_ws += vec_size;
  SimpleVec<T> grad(p_ws, n);
  p_ws += vec_size;
  SimpleVec<T> gradp(p_ws, n);
  p_ws += vec_size;
  SimpleVec<T> drt(p_ws, n);
  p_ws += vec_size;
  T* dev_scalar = p_ws;

  SimpleVec<T> svec, yvec;  // mask vectors

  std::vector<T> ys(param.m);
  std::vector<T> alpha(param.m);
  std::vector<T> fx_hist(param.past > 0 ? param.past : 0);

  *k = 0;
  ML::default_logger().set_level(verbosity);
  CUML_LOG_DEBUG("Running L-BFGS");

  // Evaluate function and compute gradient
  fx      = f(x, grad, dev_scalar, stream);
  T gnorm = f.gradNorm(grad, dev_scalar, stream);

  if (param.past > 0) fx_hist[0] = fx;

  // Early exit if the initial x is already a minimizer
  if (check_convergence(param, *k, fx, gnorm, fx_hist)) {
    CUML_LOG_DEBUG("Initial solution fulfills optimality condition.");
    return OPT_SUCCESS;
  }

  // Initial direction
  drt.ax(-1.0, grad, stream);

  // Initial step
  T step = T(1.0) / nrm2(drt, dev_scalar, stream);
  T fxp  = fx;

  *k        = 1;
  int end   = 0;
  int n_vec = 0;  // number of vector updates made in lbfgs_search_dir
  OPT_RETCODE retcode;
  LINE_SEARCH_RETCODE lsret;
  for (; *k <= param.max_iterations; (*k)++) {
    // Save the current x and gradient
    xp.copy_async(x, stream);
    gradp.copy_async(grad, stream);
    fxp = fx;

    // Line search to update x, fx and gradient
    lsret = ls_backtrack(param, f, fx, x, grad, step, drt, xp, dev_scalar, stream);
    gnorm = f.gradNorm(grad, dev_scalar, stream);

    if (update_and_check("L-BFGS",
                         param,
                         *k,
                         lsret,
                         fx,
                         fxp,
                         gnorm,
                         x,
                         xp,
                         grad,
                         gradp,
                         fx_hist,
                         dev_scalar,
                         retcode,
                         stream))
      return retcode;

    // Update s and y
    // s_{k+1} = x_{k+1} - x_k
    // y_{k+1} = g_{k+1} - g_k
    col_ref(S, svec, end);
    col_ref(Y, yvec, end);
    svec.axpy(-1.0, xp, x, stream);
    yvec.axpy(-1.0, gradp, grad, stream);
    // drt <- -H * g
    end = lbfgs_search_dir(
      param, &n_vec, end, S, Y, grad, svec, yvec, drt, ys, alpha, dev_scalar, stream);

    // step = 1.0 as initial guess
    step = T(1.0);
  }
  CUML_LOG_WARN("L-BFGS: max iterations reached");
  *k = param.max_iterations;
  return OPT_MAX_ITERS_REACHED;
}

template <typename T>
inline void update_pseudo(const SimpleVec<T>& x,
                          const SimpleVec<T>& grad,
                          const op_pseudo_grad<T>& pseudo_grad,
                          const int pg_limit,
                          SimpleVec<T>& pseudo,
                          cudaStream_t stream)
{
  if (grad.len > pg_limit) {
    pseudo.copy_async(grad, stream);
    SimpleVec<T> mask(pseudo.data, pg_limit);
    mask.assign_binary(x, grad, pseudo_grad, stream);
  } else {
    pseudo.assign_binary(x, grad, pseudo_grad, stream);
  }
}

template <typename T, typename Function>
inline OPT_RETCODE min_owlqn(const LBFGSParam<T>& param,
                             Function& f,
                             const T l1_penalty,
                             const int pg_limit,
                             SimpleVec<T>& x,
                             T& fx,
                             int* k,
                             SimpleVec<T>& workspace,  // scratch space
                             cudaStream_t stream,
                             const rapids_logger::level_enum verbosity = 0)
{
  int n                    = x.len;
  const int workspace_size = owlqn_workspace_size(param, n);
  ASSERT(workspace.len >= workspace_size, "LBFGS: workspace insufficient");
  ASSERT(pg_limit <= n && pg_limit > 0, "OWL-QN: Invalid pseudo grad limit parameter");

  // SETUP WORKSPACE
  size_t mat_size = raft::alignTo<size_t>(sizeof(T) * param.m * n, qn_align);
  size_t vec_size = raft::alignTo<size_t>(sizeof(T) * n, qn_align);
  T* p_ws         = workspace.data;
  SimpleDenseMat<T> S(p_ws, n, param.m);
  p_ws += mat_size;
  SimpleDenseMat<T> Y(p_ws, n, param.m);
  p_ws += mat_size;
  SimpleVec<T> xp(p_ws, n);
  p_ws += vec_size;
  SimpleVec<T> grad(p_ws, n);
  p_ws += vec_size;
  SimpleVec<T> gradp(p_ws, n);
  p_ws += vec_size;
  SimpleVec<T> drt(p_ws, n);
  p_ws += vec_size;
  SimpleVec<T> pseudo(p_ws, n);
  p_ws += vec_size;
  T* dev_scalar = p_ws;

  ML::default_logger().set_level(verbosity);

  SimpleVec<T> svec, yvec;  // mask vectors

  std::vector<T> ys(param.m);
  std::vector<T> alpha(param.m);
  std::vector<T> fx_hist(param.past > 0 ? param.past : 0);

  op_project<T> project_neg(T(-1.0));

  auto f_wrap = [&f, &l1_penalty, &pg_limit](
                  SimpleVec<T>& x, SimpleVec<T>& grad, T* dev_scalar, cudaStream_t stream) {
    T tmp = f(x, grad, dev_scalar, stream);
    SimpleVec<T> mask(x.data, pg_limit);
    return tmp + l1_penalty * nrm1(mask, dev_scalar, stream);
  };

  *k = 0;
  CUML_LOG_DEBUG("Running OWL-QN with lambda=%f", l1_penalty);

  // op to compute the pseudo gradients
  op_pseudo_grad<T> pseudo_grad(l1_penalty);

  fx      = f_wrap(x, grad, dev_scalar,
              stream);  // fx is loss+regularizer, grad is grad of loss only
  T gnorm = f.gradNorm(grad, dev_scalar, stream);

  // compute pseudo grad, but don't overwrite grad: used to build H
  // pseudo.assign_binary(x, grad, pseudo_grad);
  update_pseudo(x, grad, pseudo_grad, pg_limit, pseudo, stream);

  if (param.past > 0) fx_hist[0] = fx;

  // Early exit if the initial x is already a minimizer
  if (check_convergence(param, *k, fx, gnorm, fx_hist)) {
    CUML_LOG_DEBUG("Initial solution fulfills optimality condition.");
    return OPT_SUCCESS;
  }

  // Initial direction
  drt.ax(-1.0, pseudo, stream);  // using Pseudo gradient here
  // below should be done for consistency but seems unnecessary
  // drt.assign_k_ary(project, pseudo, x);

  // Initial step
  T step = T(1.0) / std::max(T(1), nrm2(drt, dev_scalar, stream));
  T fxp  = fx;

  int end   = 0;
  int n_vec = 0;  // number of vector updates made in lbfgs_search_dir
  OPT_RETCODE retcode;
  LINE_SEARCH_RETCODE lsret;
  for ((*k) = 1; (*k) <= param.max_iterations; (*k)++) {
    // Save the current x and gradient
    xp.copy_async(x, stream);
    gradp.copy_async(grad, stream);
    fxp = fx;

    // Projected line search to update x, fx and gradient
    lsret = ls_backtrack_projected(
      param, f_wrap, fx, x, grad, pseudo, step, drt, xp, l1_penalty, dev_scalar, stream);
    gnorm = f.gradNorm(grad, dev_scalar, stream);

    if (update_and_check("QWL-QN",
                         param,
                         *k,
                         lsret,
                         fx,
                         fxp,
                         gnorm,
                         x,
                         xp,
                         grad,
                         gradp,
                         fx_hist,
                         dev_scalar,
                         retcode,
                         stream))
      return retcode;

    // recompute pseudo
    //  pseudo.assign_binary(x, grad, pseudo_grad);
    update_pseudo(x, grad, pseudo_grad, pg_limit, pseudo, stream);

    // Update s and y - We should only do this if there is no skipping condition

    col_ref(S, svec, end);
    col_ref(Y, yvec, end);
    svec.axpy(-1.0, xp, x, stream);
    yvec.axpy(-1.0, gradp, grad, stream);
    // drt <- -H * -> pseudo grad <-
    end = lbfgs_search_dir(
      param, &n_vec, end, S, Y, pseudo, svec, yvec, drt, ys, alpha, dev_scalar, stream);

    // Project drt onto orthant of -pseudog
    drt.assign_binary(drt, pseudo, project_neg, stream);

    // step = 1.0 as initial guess
    step = T(1.0);
  }
  CUML_LOG_WARN("QWL-QN: max iterations reached");
  *k = param.max_iterations;
  return OPT_MAX_ITERS_REACHED;
}
/*
 * Chooses the right algorithm, depending on presence of l1 term
 */
template <typename T, typename LossFunction>
inline int qn_minimize(const raft::handle_t& handle,
                       SimpleVec<T>& x,
                       T* fx,
                       int* num_iters,
                       LossFunction& loss,
                       const T l1,
                       const LBFGSParam<T>& opt_param,
                       const rapids_logger::level_enum verbosity = 0)
{
  // TODO should the worksapce allocation happen outside?
  cudaStream_t stream = handle.get_stream().get();
  OPT_RETCODE ret;
  if (l1 == 0.0) {
    rmm::device_uvector<T> tmp(lbfgs_workspace_size(opt_param, x.len), stream);
    SimpleVec<T> workspace(tmp.data(), tmp.size());

    ret = min_lbfgs(opt_param,
                    loss,       // function to minimize
                    x,          // initial point, holds result
                    *fx,        // output function value
                    num_iters,  // output iterations
                    workspace,  // scratch space
                    stream,
                    verbosity);

    CUML_LOG_DEBUG("L-BFGS Done");
  } else {
    // There might not be a better way to deal with dispatching
    // for the l1 case:
    // The algorithm explicitly expects a differentiable
    // function f(x). It takes care of adding and
    // handling the term l1norm(x) * l1_pen explicitly, i.e.
    // it needs to evaluate f(x) and its gradient separately

    rmm::device_uvector<T> tmp(owlqn_workspace_size(opt_param, x.len), stream);
    SimpleVec<T> workspace(tmp.data(), tmp.size());

    ret = min_owlqn(opt_param,
                    loss,  // function to minimize
                    l1,
                    loss.D * loss.C,
                    x,          // initial point, holds result
                    *fx,        // output function value
                    num_iters,  // output iterations
                    workspace,  // scratch space
                    stream,
                    verbosity);

    CUML_LOG_DEBUG("OWL-QN Done");
  }
  if (ret == OPT_MAX_ITERS_REACHED) {
    CUML_LOG_WARN(
      "Maximum iterations reached before solver is converged. To increase "
      "model accuracy you can increase the number of iterations (max_iter) or "
      "improve the scaling of the input data.");
  }
  return ret;
}
};  // namespace detail
};  // namespace GLM
};  // namespace ML
