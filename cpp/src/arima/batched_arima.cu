/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <common/nvtx.hpp>

#include <cuml/common/checked_arithmetic.hpp>
#include <cuml/tsa/batched_arima.hpp>
#include <cuml/tsa/batched_kalman.hpp>

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/stats/information_criterion.cuh>
#include <raft/stats/stats_types.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>

#include <linalg/batched/matrix.cuh>
#include <timeSeries/arima_helpers.cuh>
#include <timeSeries/fillna.cuh>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace ML {

void pack(raft::handle_t& handle,
          const ARIMAParams<double>& params,
          const ARIMAOrder& order,
          int batch_size,
          double* param_vec)
{
  const auto stream = handle.get_stream().get();
  params.pack(order, batch_size, param_vec, stream);
}

void unpack(raft::handle_t& handle,
            ARIMAParams<double>& params,
            const ARIMAOrder& order,
            int batch_size,
            const double* param_vec)
{
  const auto stream = handle.get_stream().get();
  params.unpack(order, batch_size, param_vec, stream);
}

void batched_diff(raft::handle_t& handle,
                  double* d_y_diff,
                  const double* d_y,
                  int batch_size,
                  int n_obs,
                  const ARIMAOrder& order)
{
  const auto stream = handle.get_stream().get();
  MLCommon::TimeSeries::prepare_data(
    d_y_diff, d_y, batch_size, n_obs, order.d, order.D, order.s, stream);
}

template <typename T>
struct is_missing {
  typedef T argument_type;
  typedef T result_type;

  __device__ const T operator()(const T& x) const { return isnan(x); }
};  // end is_missing

bool detect_missing(raft::handle_t& handle, const double* d_y, int n_elem)
{
  return thrust::any_of(
    thrust::cuda::par.on(handle.get_stream().get()), d_y, d_y + n_elem, is_missing<double>());
}

void predict(raft::handle_t& handle,
             const ARIMAMemory<double>& arima_mem,
             const double* d_y,
             const double* d_exog,
             const double* d_exog_fut,
             int batch_size,
             int n_obs,
             int start,
             int end,
             const ARIMAOrder& order,
             const ARIMAParams<double>& params,
             double* d_y_p,
             bool pre_diff,
             double level,
             double* d_lower,
             double* d_upper)
{
  raft::common::nvtx::range fun_scope(__func__);
  const auto stream = handle.get_stream().get();

  bool diff     = order.need_diff() && pre_diff && level == 0;
  int num_steps = std::max(end - n_obs, 0);

  // Prepare data
  int n_obs_kf;
  const double* d_y_kf;
  const double* d_exog_kf;
  const double* d_exog_fut_kf = d_exog_fut;
  ARIMAOrder order_after_prep = order;
  rmm::device_uvector<double> exog_fut_buffer(0, stream);
  if (diff) {
    n_obs_kf = n_obs - order.n_diff();
    MLCommon::TimeSeries::prepare_data(
      arima_mem.y_diff, d_y, batch_size, n_obs, order.d, order.D, order.s, stream);
    if (order.n_exog > 0) {
      MLCommon::TimeSeries::prepare_data(arima_mem.exog_diff,
                                         d_exog,
                                         checked_mul<int>(order.n_exog, batch_size),
                                         n_obs,
                                         order.d,
                                         order.D,
                                         order.s,
                                         stream);

      if (num_steps > 0) {
        exog_fut_buffer.resize(checked_mul<std::size_t>(num_steps, order.n_exog, batch_size),
                               stream);

        MLCommon::TimeSeries::prepare_future_data(exog_fut_buffer.data(),
                                                  d_exog,
                                                  d_exog_fut,
                                                  checked_mul<int>(order.n_exog, batch_size),
                                                  n_obs,
                                                  num_steps,
                                                  order.d,
                                                  order.D,
                                                  order.s,
                                                  stream);

        d_exog_fut_kf = exog_fut_buffer.data();
      }
    }
    order_after_prep.d = 0;
    order_after_prep.D = 0;

    d_y_kf    = arima_mem.y_diff;
    d_exog_kf = arima_mem.exog_diff;
  } else {
    n_obs_kf  = n_obs;
    d_y_kf    = d_y;
    d_exog_kf = d_exog;
  }

  double* d_pred = arima_mem.pred;

  // Create temporary array for the forecasts
  rmm::device_uvector<double> fc_buffer(checked_mul<std::size_t>(num_steps, batch_size), stream);
  double* d_y_fc = fc_buffer.data();

  // Compute the residual and forecast
  std::vector<double> loglike = std::vector<double>(batch_size);
  /// TODO: use device loglike to avoid useless copy ; part of #2233
  batched_loglike(handle,
                  arima_mem,
                  d_y_kf,
                  d_exog_kf,
                  batch_size,
                  n_obs_kf,
                  order_after_prep,
                  params,
                  loglike.data(),
                  false,
                  true,
                  MLE,
                  0,
                  num_steps,
                  d_y_fc,
                  d_exog_fut_kf,
                  level,
                  d_lower,
                  d_upper);

  auto counting  = thrust::make_counting_iterator(0);
  int predict_ld = end - start;

  //
  // In-sample prediction
  //

  // The prediction loop starts by filling undefined predictions with NaN,
  // then computes the predictions from the observations and residuals
  if (start < n_obs) {
    int res_offset = diff ? checked_add<int>(order.d, checked_mul<int>(order.s, order.D)) : 0;
    int p_start    = std::max(start, res_offset);
    int p_end      = std::min(n_obs, end);
    int dD         = diff ? order.d + order.D : 0;
    int period1    = order.d ? 1 : order.s;
    int period2    = order.d == 2 ? 1 : order.s;

    thrust::for_each(
      thrust::cuda::par.on(stream), counting, counting + batch_size, [=] __device__(int bid) {
        d_y_p[0] = 0.0;
        for (int i = 0; i < res_offset - start; i++) {
          d_y_p[bid * predict_ld + i] = nan("");
        }
        for (int i = p_start; i < p_end; i++) {
          if (dD == 0) {
            d_y_p[bid * predict_ld + i - start] = d_pred[bid * n_obs + i];
          } else if (dD == 1) {
            d_y_p[bid * predict_ld + i - start] =
              d_y[bid * n_obs + i - period1] + d_pred[bid * n_obs_kf + i - res_offset];
          } else {
            d_y_p[bid * predict_ld + i - start] =
              d_y[bid * n_obs + i - period1] + d_y[bid * n_obs + i - period2] -
              d_y[bid * n_obs + i - period1 - period2] + d_pred[bid * n_obs_kf + i - res_offset];
          }
        }
      });
  }

  //
  // Finalize out-of-sample forecast and copy in-sample predictions
  //

  if (num_steps) {
    if (diff) {
      MLCommon::TimeSeries::finalize_forecast(
        d_y_fc, d_y, num_steps, batch_size, n_obs, n_obs, order.d, order.D, order.s, stream);
    }

    // Copy forecast in d_y_p
    thrust::for_each(
      thrust::cuda::par.on(stream), counting, counting + batch_size, [=] __device__(int bid) {
        for (int i = 0; i < num_steps; i++) {
          d_y_p[bid * predict_ld + n_obs - start + i] = d_y_fc[num_steps * bid + i];
        }
      });
    /// TODO: 2D copy kernel?
  }
}

/**
 * Kernel to compute the sum-of-squares log-likelihood estimation
 *
 * @param[in]  d_y        Series to fit
 * @param[in]  d_mu       mu parameters
 * @param[in]  d_ar       AR parameters
 * @param[in]  d_ma       MA parameters
 * @param[in]  d_sar      Seasonal AR parameters
 * @param[in]  d_sma      Seasonal MA parameters
 * @param[out] d_loglike  Evaluated log-likelihood
 * @param[in]  n_obs      Number of observations in a time series
 * @param[in]  n_phi      Number of phi coefficients (combined AR-SAR)
 * @param[in]  n_theta    Number of theta coefficients (combined MA-SMA)
 * @param[in]  p          Number of AR parameters
 * @param[in]  q          Number of MA parameters
 * @param[in]  P          Number of seasonal AR parameters
 * @param[in]  Q          Number of seasonal MA parameters
 * @param[in]  s          Seasonal period or 0
 * @param[in]  k          Whether to use an intercept
 * @param[in]  start_sum  At which index to start the sum
 * @param[in]  start_y    First used y index (observation)
 * @param[in]  start_v    First used v index (residual)
 */
template <typename DataT>
CUML_KERNEL void sum_of_squares_kernel(const DataT* d_y,
                                       const DataT* d_mu,
                                       const DataT* d_ar,
                                       const DataT* d_ma,
                                       const DataT* d_sar,
                                       const DataT* d_sma,
                                       DataT* d_loglike,
                                       int n_obs,
                                       int n_phi,
                                       int n_theta,
                                       int p,
                                       int q,
                                       int P,
                                       int Q,
                                       int s,
                                       int k,
                                       int start_sum,
                                       int start_y,
                                       int start_v)
{
  // Load phi, theta and mu to registers
  DataT phi, theta;
  if (threadIdx.x < n_phi) {
    phi = MLCommon::TimeSeries::reduced_polynomial<true>(
      blockIdx.x, d_ar, p, d_sar, P, s, threadIdx.x + 1);
  }
  if (threadIdx.x < n_theta) {
    theta = MLCommon::TimeSeries::reduced_polynomial<false>(
      blockIdx.x, d_ma, q, d_sma, Q, s, threadIdx.x + 1);
  }
  DataT mu = k ? d_mu[blockIdx.x] : (DataT)0;

  // Shared memory: load y and initialize the residuals
  extern __shared__ DataT shared_mem[];
  DataT* b_y  = shared_mem;
  DataT* b_vs = shared_mem + n_obs - start_y;
  for (int i = threadIdx.x; i < n_obs - start_y; i += blockDim.x) {
    b_y[i] = d_y[n_obs * blockIdx.x + i + start_y];
  }
  for (int i = threadIdx.x; i < start_sum - start_v; i += blockDim.x) {
    b_vs[i] = (DataT)0;
  }

  // Main loop
  char* temp_smem = (char*)(shared_mem + 2 * n_obs - start_y - start_v);
  DataT res, ssq = 0;
  for (int i = start_sum; i < n_obs; i++) {
    __syncthreads();
    res = (DataT)0;
    res -= threadIdx.x < n_phi ? phi * b_y[i - threadIdx.x - 1 - start_y] : (DataT)0;
    res -= threadIdx.x < n_theta ? theta * b_vs[i - threadIdx.x - 1 - start_v] : (DataT)0;
    res = raft::blockReduce(res, temp_smem);
    if (threadIdx.x == 0) {
      res += b_y[i - start_y] - mu;
      b_vs[i - start_v] = res;
      ssq += res * res;
    }
  }

  // Compute log-likelihood and write it to global memory
  if (threadIdx.x == 0) {
    d_loglike[blockIdx.x] =
      -0.5 * static_cast<DataT>(n_obs) * raft::log(ssq / static_cast<DataT>(n_obs - start_sum));
  }
}

/**
 * Sum-of-squares estimation method
 *
 * @param[in]  handle     cuML handle
 * @param[in]  d_y        Series to fit: shape = (n_obs, batch_size)
 * @param[in]  batch_size Number of time series
 * @param[in]  n_obs      Number of observations in a time series
 * @param[in]  order      ARIMA hyper-parameters
 * @param[in]  Tparams    Transformed parameters
 * @param[out] d_loglike  Evaluated log-likelihood (device)
 * @param[in]  truncate   Number of observations to skip in the sum
 */
void conditional_sum_of_squares(raft::handle_t& handle,
                                const double* d_y,
                                int batch_size,
                                int n_obs,
                                const ARIMAOrder& order,
                                const ARIMAParams<double>& Tparams,
                                double* d_loglike,
                                int truncate)
{
  raft::common::nvtx::range fun_scope(__func__);
  auto stream = handle.get_stream().get();

  int n_phi     = order.n_phi();
  int n_theta   = order.n_theta();
  int max_lags  = std::max(n_phi, n_theta);
  int start_sum = std::max(max_lags, truncate);
  int start_y   = start_sum - n_phi;
  int start_v   = start_sum - n_theta;

  // Compute the sum-of-squares and the log-likelihood
  int n_warps            = std::max(raft::ceildiv<int>(max_lags, 32), 1);
  size_t shared_mem_size = (2 * n_obs - start_y - start_v + n_warps) * sizeof(double);
  sum_of_squares_kernel<<<batch_size, 32 * n_warps, shared_mem_size, stream>>>(d_y,
                                                                               Tparams.mu,
                                                                               Tparams.ar,
                                                                               Tparams.ma,
                                                                               Tparams.sar,
                                                                               Tparams.sma,
                                                                               d_loglike,
                                                                               n_obs,
                                                                               n_phi,
                                                                               n_theta,
                                                                               order.p,
                                                                               order.q,
                                                                               order.P,
                                                                               order.Q,
                                                                               order.s,
                                                                               order.k,
                                                                               start_sum,
                                                                               start_y,
                                                                               start_v);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

void batched_loglike(raft::handle_t& handle,
                     const ARIMAMemory<double>& arima_mem,
                     const double* d_y,
                     const double* d_exog,
                     int batch_size,
                     int n_obs,
                     const ARIMAOrder& order,
                     const ARIMAParams<double>& params,
                     double* loglike,
                     bool trans,
                     bool host_loglike,
                     LoglikeMethod method,
                     int truncate,
                     int fc_steps,
                     double* d_fc,
                     const double* d_exog_fut,
                     double level,
                     double* d_lower,
                     double* d_upper)
{
  raft::common::nvtx::range fun_scope(__func__);

  auto stream = handle.get_stream().get();

  double* d_pred = arima_mem.pred;

  ARIMAParams<double> Tparams = {params.mu,
                                 params.beta,
                                 arima_mem.Tparams_ar,
                                 arima_mem.Tparams_ma,
                                 arima_mem.Tparams_sar,
                                 arima_mem.Tparams_sma,
                                 arima_mem.Tparams_sigma2};

  ASSERT(method == MLE || fc_steps == 0, "Only MLE method is valid for forecasting");

  /* Create log-likelihood device array if host pointer is provided */
  double* d_loglike = host_loglike ? arima_mem.loglike : loglike;

  if (trans) {
    MLCommon::TimeSeries::batched_jones_transform(
      order, batch_size, false, params, Tparams, stream);
  } else {
    // non-transformed case: just use original parameters
    Tparams.ar     = params.ar;
    Tparams.ma     = params.ma;
    Tparams.sar    = params.sar;
    Tparams.sma    = params.sma;
    Tparams.sigma2 = params.sigma2;
  }

  if (method == CSS) {
    conditional_sum_of_squares(handle, d_y, batch_size, n_obs, order, Tparams, d_loglike, truncate);
  } else {
    batched_kalman_filter(handle,
                          arima_mem,
                          d_y,
                          d_exog,
                          n_obs,
                          Tparams,
                          order,
                          batch_size,
                          d_loglike,
                          d_pred,
                          fc_steps,
                          d_fc,
                          d_exog_fut,
                          level,
                          d_lower,
                          d_upper);
  }

  if (host_loglike) {
    /* Transfer log-likelihood device -> host */
    raft::update_host(loglike, d_loglike, batch_size, stream);
  }
}

void batched_loglike(raft::handle_t& handle,
                     const ARIMAMemory<double>& arima_mem,
                     const double* d_y,
                     const double* d_exog,
                     int batch_size,
                     int n_obs,
                     const ARIMAOrder& order,
                     const double* d_params,
                     double* loglike,
                     bool trans,
                     bool host_loglike,
                     LoglikeMethod method,
                     int truncate)
{
  raft::common::nvtx::range fun_scope(__func__);

  // unpack parameters
  auto stream = handle.get_stream().get();

  ARIMAParams<double> params = {arima_mem.params_mu,
                                arima_mem.params_beta,
                                arima_mem.params_ar,
                                arima_mem.params_ma,
                                arima_mem.params_sar,
                                arima_mem.params_sma,
                                arima_mem.params_sigma2};

  params.unpack(order, batch_size, d_params, stream);

  batched_loglike(handle,
                  arima_mem,
                  d_y,
                  d_exog,
                  batch_size,
                  n_obs,
                  order,
                  params,
                  loglike,
                  trans,
                  host_loglike,
                  method,
                  truncate);
}

void batched_loglike_grad(raft::handle_t& handle,
                          const ARIMAMemory<double>& arima_mem,
                          const double* d_y,
                          const double* d_exog,
                          int batch_size,
                          int n_obs,
                          const ARIMAOrder& order,
                          const double* d_x,
                          double* d_grad,
                          double h,
                          bool trans,
                          LoglikeMethod method,
                          int truncate)
{
  raft::common::nvtx::range fun_scope(__func__);
  auto stream   = handle.get_stream().get();
  auto counting = thrust::make_counting_iterator(0);
  int N         = order.complexity();

  // Initialize the perturbed x vector
  double* d_x_pert = arima_mem.x_pert;
  raft::copy(d_x_pert, d_x, N * batch_size, stream);

  double* d_ll_base = arima_mem.loglike_base;
  double* d_ll_pert = arima_mem.loglike_pert;

  // Evaluate the log-likelihood with the given parameter vector
  batched_loglike(handle,
                  arima_mem,
                  d_y,
                  d_exog,
                  batch_size,
                  n_obs,
                  order,
                  d_x,
                  d_ll_base,
                  trans,
                  false,
                  method,
                  truncate);

  for (int i = 0; i < N; i++) {
    // Add the perturbation to the i-th parameter
    thrust::for_each(
      thrust::cuda::par.on(stream), counting, counting + batch_size, [=] __device__(int bid) {
        d_x_pert[N * bid + i] = d_x[N * bid + i] + h;
      });

    // Evaluate the log-likelihood with the positive perturbation
    batched_loglike(handle,
                    arima_mem,
                    d_y,
                    d_exog,
                    batch_size,
                    n_obs,
                    order,
                    d_x_pert,
                    d_ll_pert,
                    trans,
                    false,
                    method,
                    truncate);

    // First derivative with a first-order accuracy
    thrust::for_each(
      thrust::cuda::par.on(stream), counting, counting + batch_size, [=] __device__(int bid) {
        d_grad[N * bid + i] = (d_ll_pert[bid] - d_ll_base[bid]) / h;
      });

    // Reset the i-th parameter
    thrust::for_each(
      thrust::cuda::par.on(stream), counting, counting + batch_size, [=] __device__(int bid) {
        d_x_pert[N * bid + i] = d_x[N * bid + i];
      });
  }
}

void information_criterion(raft::handle_t& handle,
                           const ARIMAMemory<double>& arima_mem,
                           const double* d_y,
                           const double* d_exog,
                           int batch_size,
                           int n_obs,
                           const ARIMAOrder& order,
                           const ARIMAParams<double>& params,
                           double* d_ic,
                           int ic_type)
{
  raft::common::nvtx::range fun_scope(__func__);
  auto stream = handle.get_stream().get();

  /* Compute log-likelihood in d_ic */
  batched_loglike(
    handle, arima_mem, d_y, d_exog, batch_size, n_obs, order, params, d_ic, false, false, MLE);

  /* Compute information criterion from log-likelihood and base term */
  raft::stats::information_criterion_batched(d_ic,
                                             d_ic,
                                             static_cast<raft::stats::IC_Type>(ic_type),
                                             order.complexity(),
                                             batch_size,
                                             n_obs - order.n_diff(),
                                             stream);
}

/**
 * Test that the parameters are valid for the inverse transform
 *
 * @tparam isAr        Are these (S)AR or (S)MA parameters?
 * @param[in]  params  Parameters
 * @param[in]  pq      p for AR, q for MA, P for SAR, Q for SMA
 */
template <bool isAr>
DI bool test_invparams(const double* params, int pq)
{
  double new_params[8];
  double tmp[8];

  constexpr double coef = isAr ? 1 : -1;

  for (int i = 0; i < pq; i++) {
    tmp[i]        = params[i];
    new_params[i] = tmp[i];
  }

  // Perform inverse transform and stop before atanh step
  for (int j = pq - 1; j > 0; --j) {
    double a = new_params[j];
    for (int k = 0; k < j; ++k) {
      tmp[k] = (new_params[k] + coef * a * new_params[j - k - 1]) / (1 - (a * a));
    }
    for (int iter = 0; iter < j; ++iter) {
      new_params[iter] = tmp[iter];
    }
  }

  // Verify that the values are between -1 and 1
  bool result = true;
  for (int i = 0; i < pq; i++) {
    result = result && !(new_params[i] <= -1 || new_params[i] >= 1);
  }
  return result;
}

/**
 * Auxiliary function of _start_params: least square approximation of an
 * ARMA model (with or without seasonality)
 * @note: in this function the non-seasonal case has s=1, not s=0!
 */
void _arma_least_squares(raft::handle_t& handle,
                         double* d_ar,
                         double* d_ma,
                         double* d_sigma2,
                         const MLCommon::LinAlg::Batched::Matrix<double>& bm_y,
                         int p,
                         int q,
                         int s,
                         bool estimate_sigma2,
                         int k        = 0,
                         double* d_mu = nullptr)
{
  const auto& handle_impl = handle;
  auto stream             = handle_impl.get_stream().get();
  auto cublas_handle      = handle_impl.get_cublas_handle();
  auto counting           = thrust::make_counting_iterator(0);

  int batch_size = narrow_cast<int>(bm_y.batches());
  int n_obs      = narrow_cast<int>(bm_y.shape().first);

  int ps = p * s, qs = q * s;
  int p_ar = std::max(ps, 2 * qs);
  int r    = std::max(p_ar + qs, ps);

  if ((q && p_ar >= n_obs - p_ar) || p + q + k >= n_obs - r) {
    // Too few observations for the estimate, fill with 0 (1 for sigma2)
    if (k) RAFT_CUDA_TRY(cudaMemsetAsync(d_mu, 0, sizeof(double) * batch_size, stream));
    if (p) RAFT_CUDA_TRY(cudaMemsetAsync(d_ar, 0, sizeof(double) * p * batch_size, stream));
    if (q) RAFT_CUDA_TRY(cudaMemsetAsync(d_ma, 0, sizeof(double) * q * batch_size, stream));
    if (estimate_sigma2) {
      thrust::device_ptr<double> sigma2_thrust = thrust::device_pointer_cast(d_sigma2);
      thrust::fill(thrust::cuda::par.on(stream), sigma2_thrust, sigma2_thrust + batch_size, 1.0);
    }
    return;
  }

  /* Matrix formed by lag matrices of y and the residuals respectively,
   * side by side. The left side will be used to estimate AR, the right
   * side to estimate MA */
  MLCommon::LinAlg::Batched::Matrix<double> bm_ls_ar_res(
    n_obs - r, p + q + k, batch_size, cublas_handle, stream, false);
  int ar_offset  = r - ps;
  int res_offset = r - p_ar - qs;

  // Get residuals from an AR(p_ar) model to estimate the MA parameters
  if (q) {
    // Create lagged y
    int ls_height = n_obs - p_ar;
    MLCommon::LinAlg::Batched::Matrix<double> bm_ls =
      MLCommon::LinAlg::Batched::b_lagged_mat(bm_y, p_ar);

    /* Matrix for the initial AR fit, initialized by copy of y
     * (note: this is because gels works in-place ; the matrix has larger
     *  dimensions than the actual AR fit) */
    MLCommon::LinAlg::Batched::Matrix<double> bm_ar_fit =
      MLCommon::LinAlg::Batched::b_2dcopy(bm_y, p_ar, 0, ls_height, 1);

    // Residual, initialized as offset y to avoid one kernel call
    MLCommon::LinAlg::Batched::Matrix<double> bm_residual(bm_ar_fit);

    // Initial AR fit
    MLCommon::LinAlg::Batched::b_gels(bm_ls, bm_ar_fit);

    // Compute residual (technically a gemv)
    MLCommon::LinAlg::Batched::b_gemm(
      false, false, ls_height, 1, p_ar, -1.0, bm_ls, bm_ar_fit, 1.0, bm_residual);

    // Lags of the residual
    MLCommon::LinAlg::Batched::b_lagged_mat(
      bm_residual, bm_ls_ar_res, q, n_obs - r, res_offset, (n_obs - r) * (k + p), s);
  }

  // Fill the first column of the matrix with 1 if we fit an intercept
  if (k) {
    double* d_ls_ar_res = bm_ls_ar_res.raw_data();
    thrust::for_each(
      thrust::cuda::par.on(stream), counting, counting + batch_size, [=] __device__(int bid) {
        double* b_ls_ar_res = d_ls_ar_res + bid * (n_obs - r) * (p + q + k);
        for (int i = 0; i < n_obs - r; i++) {
          b_ls_ar_res[i] = 1.0;
        }
      });
  }

  // Lags of y
  MLCommon::LinAlg::Batched::b_lagged_mat(
    bm_y, bm_ls_ar_res, p, n_obs - r, ar_offset, (n_obs - r) * k, s);

  /* Initializing the vector for the ARMA fit
   * (note: also in-place as described for AR fit) */
  MLCommon::LinAlg::Batched::Matrix<double> bm_arma_fit =
    MLCommon::LinAlg::Batched::b_2dcopy(bm_y, r, 0, n_obs - r, 1);

  // The residuals will be computed only if sigma2 is requested
  if (n_obs <= r) { RAFT_FAIL("ARIMA fit: n_obs (%d) must be greater than r (%d)", n_obs, r); }
  std::size_t const residual_rows = checked_sub<std::size_t>(n_obs, r);
  MLCommon::LinAlg::Batched::Matrix<double> bm_final_residual(
    residual_rows, 1, batch_size, cublas_handle, stream, false);
  if (estimate_sigma2) {
    raft::copy(bm_final_residual.raw_data(),
               bm_arma_fit.raw_data(),
               checked_mul<std::size_t>(residual_rows, batch_size),
               stream);
  }

  // ARMA fit
  MLCommon::LinAlg::Batched::b_gels(bm_ls_ar_res, bm_arma_fit);

  // Copy the results in the parameter vectors
  const double* d_arma_fit = bm_arma_fit.raw_data();
  thrust::for_each(
    thrust::cuda::par.on(stream), counting, counting + batch_size, [=] __device__(int bid) {
      const double* b_arma_fit = d_arma_fit + bid * (n_obs - r);
      if (k) { d_mu[bid] = b_arma_fit[0]; }
      if (p) {
        double* b_ar = d_ar + bid * p;
        for (int i = 0; i < p; i++) {
          b_ar[i] = b_arma_fit[i + k];
        }
      }
      if (q) {
        double* b_ma = d_ma + bid * q;
        for (int i = 0; i < q; i++) {
          b_ma[i] = b_arma_fit[i + p + k];
        }
      }
    });

  if (estimate_sigma2) {
    // Compute final residual (technically a gemv)
    MLCommon::LinAlg::Batched::b_gemm(false,
                                      false,
                                      n_obs - r,
                                      1,
                                      p + q + k,
                                      -1.0,
                                      bm_ls_ar_res,
                                      bm_arma_fit,
                                      1.0,
                                      bm_final_residual);

    // Compute variance
    double* d_residual = bm_final_residual.raw_data();
    thrust::for_each(
      thrust::cuda::par.on(stream), counting, counting + batch_size, [=] __device__(int bid) {
        double acc               = 0.0;
        const double* b_residual = d_residual + (n_obs - r) * bid;
        for (int i = q; i < n_obs - r; i++) {
          double res = b_residual[i];
          acc += res * res;
        }
        d_sigma2[bid] = acc / static_cast<double>(n_obs - r - q);
      });
  }

  // If (S)AR or (S)MA are not valid for the inverse transform, set them to zero
  thrust::for_each(
    thrust::cuda::par.on(stream), counting, counting + batch_size, [=] __device__(int bid) {
      if (p) {
        double* b_ar = d_ar + bid * p;
        bool valid   = test_invparams<true>(b_ar, p);
        if (!valid) {
          for (int ip = 0; ip < p; ip++)
            b_ar[ip] = 0;
        }
      }
      if (q) {
        double* b_ma = d_ma + bid * q;
        bool valid   = test_invparams<false>(b_ma, q);
        if (!valid) {
          for (int iq = 0; iq < q; iq++)
            b_ma[iq] = 0;
        }
      }
    });
}

/**
 * Auxiliary function of estimate_x0: compute the starting parameters for
 * the series pre-processed by estimate_x0
 */
void _start_params(raft::handle_t& handle,
                   ARIMAParams<double>& params,
                   MLCommon::LinAlg::Batched::Matrix<double>& bm_y,
                   const MLCommon::LinAlg::Batched::Matrix<double>& bm_exog,
                   const ARIMAOrder& order)
{
  int batch_size      = narrow_cast<int>(bm_exog.batches());
  cudaStream_t stream = bm_exog.stream();

  // Estimate exog coefficients and subtract component to endog.
  // Exog coefficients are estimated by fitting a linear regression with X=exog, y=endog
  if (order.n_exog > 0) {
    // In most cases, the system will be overdetermined and we can use gels
    if (bm_exog.shape().first > static_cast<unsigned int>(order.n_exog)) {
      // Make a copy of the exogenous series for in-place gels
      MLCommon::LinAlg::Batched::Matrix<double> bm_exog_copy(bm_exog);
      // Make a copy of the endogenous series for in-place gels
      MLCommon::LinAlg::Batched::Matrix<double> bm_y_copy(bm_y);

      // Least-squares solution of overdetermined system
      rmm::device_uvector<int> info(batch_size, stream);
      b_gels(bm_exog_copy, bm_y_copy, info.data());

      // Make a batched matrix around the exogenous coefficients
      rmm::device_uvector<double*> beta_pointers(batch_size, stream);
      MLCommon::LinAlg::Batched::Matrix<double> bm_exog_coef(order.n_exog,
                                                             1,
                                                             batch_size,
                                                             bm_exog.cublasHandle(),
                                                             beta_pointers.data(),
                                                             params.beta,
                                                             stream,
                                                             false);

      // Copy the solution of the system to the parameters array
      b_2dcopy(bm_y_copy, bm_exog_coef, 0, 0, order.n_exog, 1);

      // Set parameters to zero when solving was not successful
      auto counting       = thrust::make_counting_iterator(0);
      int* devInfoArray   = info.data();
      double* d_exog_coef = bm_exog_coef.raw_data();
      const int& n_exog   = order.n_exog;
      thrust::for_each(
        thrust::cuda::par.on(stream), counting, counting + batch_size, [=] __device__(int bid) {
          if (devInfoArray[bid] > 0) {
            for (int i = 0; i < n_exog; i++) {
              d_exog_coef[bid * n_exog + i] = 0.0;
            }
          }
        });

      // Compute exogenous component and store the result in bm_y_copy
      b_gemm(false,
             false,
             bm_exog.shape().first,
             1,
             bm_exog.shape().second,
             1.0,
             bm_exog,
             bm_exog_coef,
             0.0,
             bm_y_copy);

      // Subtract exogenous component to endogenous variable
      b_aA_op_B(bm_y, bm_y_copy, bm_y, [] __device__(double a, double b) { return a - b; });
    }
    // In other cases, we initialize to zero
    else {
      RAFT_CUDA_TRY(
        cudaMemsetAsync(params.beta,
                        0,
                        checked_mul<std::size_t>(order.n_exog, batch_size, sizeof(double)),
                        stream));
    }
  }

  // Estimate an ARMA fit without seasonality
  if (order.p + order.q + order.k)
    _arma_least_squares(handle,
                        params.ar,
                        params.ma,
                        params.sigma2,
                        bm_y,
                        order.p,
                        order.q,
                        1,
                        true,
                        order.k,
                        params.mu);

  // Estimate a seasonal ARMA fit independently
  if (order.P + order.Q)
    _arma_least_squares(handle,
                        params.sar,
                        params.sma,
                        params.sigma2,
                        bm_y,
                        order.P,
                        order.Q,
                        order.s,
                        order.p + order.q + order.k == 0);
}

void estimate_x0(raft::handle_t& handle,
                 ARIMAParams<double>& params,
                 const double* d_y,
                 const double* d_exog,
                 int batch_size,
                 int n_obs,
                 const ARIMAOrder& order,
                 bool missing)
{
  raft::common::nvtx::range fun_scope(__func__);
  const auto& handle_impl = handle;
  auto stream             = handle_impl.get_stream().get();
  auto cublas_handle      = handle_impl.get_cublas_handle();

  /// TODO: solve exogenous coefficients with only valid rows instead of interpolation?
  // Pros: better coefficients
  // Cons: harder to test, a bit more complicated

  // Least squares can't deal with missing values: create copy with naive
  // replacements for missing values
  const double* d_y_no_missing;
  rmm::device_uvector<double> y_no_missing(0, stream);
  if (missing) {
    std::size_t const yn = checked_mul<std::size_t>(n_obs, batch_size);
    y_no_missing.resize(yn, stream);
    d_y_no_missing = y_no_missing.data();

    raft::copy(y_no_missing.data(), d_y, yn, stream);
    MLCommon::TimeSeries::fillna(y_no_missing.data(), batch_size, n_obs, stream);
  } else {
    d_y_no_missing = d_y;
  }

  // Difference if necessary, copy otherwise
  int const d_sD = checked_add<int>(order.d, checked_mul<int>(order.s, order.D));
  if (n_obs <= d_sD) {
    RAFT_FAIL("ARIMA: n_obs (%d) must be greater than d + s*D (%d) for differencing", n_obs, d_sD);
  }
  std::size_t const diff_rows = checked_sub<std::size_t>(n_obs, d_sD);
  MLCommon::LinAlg::Batched::Matrix<double> bm_yd(
    diff_rows, 1, batch_size, cublas_handle, stream, false);
  MLCommon::TimeSeries::prepare_data(
    bm_yd.raw_data(), d_y_no_missing, batch_size, n_obs, order.d, order.D, order.s, stream);

  // Difference or copy exog
  MLCommon::LinAlg::Batched::Matrix<double> bm_exog_diff(
    diff_rows, order.n_exog, batch_size, cublas_handle, stream, false);
  if (order.n_exog > 0) {
    MLCommon::TimeSeries::prepare_data(bm_exog_diff.raw_data(),
                                       d_exog,
                                       checked_mul<int>(order.n_exog, batch_size),
                                       n_obs,
                                       order.d,
                                       order.D,
                                       order.s,
                                       stream);
  }

  // Do the computation of the initial parameters
  _start_params(handle, params, bm_yd, bm_exog_diff, order);
}

}  // namespace ML
