/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/common/checked_arithmetic.hpp>
#include <cuml/common/export.hpp>

#include <raft/util/cudart_utils.hpp>

#include <rmm/aligned.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime.h>

#include <algorithm>

namespace CUML_EXPORT ML {

/**
 * Structure to hold the ARIMA order (makes it easier to pass as an argument)
 */
struct ARIMAOrder {
  int p;  // Basic order
  int d;
  int q;
  int P;  // Seasonal order
  int D;
  int Q;
  int s;       // Seasonal period
  int k;       // Fit intercept?
  int n_exog;  // Number of exogenous regressors

  inline int n_diff() const { return checked_add<int>(d, checked_mul<int>(s, D)); }
  inline int n_phi() const { return checked_add<int>(p, checked_mul<int>(s, P)); }
  inline int n_theta() const { return checked_add<int>(q, checked_mul<int>(s, Q)); }
  inline int r() const { return std::max(n_phi(), checked_add<int>(n_theta(), 1)); }
  inline int rd() const { return checked_add<int>(n_diff(), r()); }
  inline int complexity() const { return checked_add<int>(p, P, q, Q, k, n_exog, 1); }
  inline bool need_diff() const { return d != 0 || D != 0; }
};

/**
 * Structure to hold the parameters (makes it easier to pass as an argument)
 * @note: the qualifier const applied to this structure will only guarantee
 *        that the pointers are not changed, but the user can still modify the
 *        arrays when using the pointers directly!
 */
template <typename DataT>
struct ARIMAParams {
  DataT* mu     = nullptr;
  DataT* beta   = nullptr;
  DataT* ar     = nullptr;
  DataT* ma     = nullptr;
  DataT* sar    = nullptr;
  DataT* sma    = nullptr;
  DataT* sigma2 = nullptr;

  /**
   * Allocate all the parameter device arrays
   *
   * @tparam      AllocatorT Type of allocator used
   * @param[in]   order      ARIMA order
   * @param[in]   batch_size Batch size
   * @param[in]   stream     CUDA stream
   * @param[in]   tr         Whether these are the transformed parameters
   */
  void allocate(const ARIMAOrder& order, int batch_size, cudaStream_t stream, bool tr = false)
  {
    rmm::device_async_resource_ref rmm_alloc = rmm::mr::get_current_device_resource_ref();
    std::size_t const bs                     = static_cast<std::size_t>(batch_size);
    if (order.k && !tr)
      mu = (DataT*)rmm_alloc.allocate(stream, checked_mul<std::size_t>(bs, sizeof(DataT)));
    if (order.n_exog && !tr)
      beta = (DataT*)rmm_alloc.allocate(stream,
                                        checked_mul<std::size_t>(order.n_exog, bs, sizeof(DataT)));
    if (order.p)
      ar = (DataT*)rmm_alloc.allocate(stream, checked_mul<std::size_t>(order.p, bs, sizeof(DataT)));
    if (order.q)
      ma = (DataT*)rmm_alloc.allocate(stream, checked_mul<std::size_t>(order.q, bs, sizeof(DataT)));
    if (order.P)
      sar =
        (DataT*)rmm_alloc.allocate(stream, checked_mul<std::size_t>(order.P, bs, sizeof(DataT)));
    if (order.Q)
      sma =
        (DataT*)rmm_alloc.allocate(stream, checked_mul<std::size_t>(order.Q, bs, sizeof(DataT)));
    sigma2 = (DataT*)rmm_alloc.allocate(stream, checked_mul<std::size_t>(bs, sizeof(DataT)));
  }

  /**
   * Deallocate all the parameter device arrays
   *
   * @tparam      AllocatorT Type of allocator used
   * @param[in]   order      ARIMA order
   * @param[in]   batch_size Batch size
   * @param[in]   stream     CUDA stream
   * @param[in]   tr         Whether these are the transformed parameters
   */
  void deallocate(const ARIMAOrder& order, int batch_size, cudaStream_t stream, bool tr = false)
  {
    rmm::device_async_resource_ref rmm_alloc = rmm::mr::get_current_device_resource_ref();
    std::size_t const bs                     = static_cast<std::size_t>(batch_size);
    if (order.k && !tr)
      rmm_alloc.deallocate(stream, mu, checked_mul<std::size_t>(bs, sizeof(DataT)));
    if (order.n_exog && !tr)
      rmm_alloc.deallocate(stream, beta, checked_mul<std::size_t>(order.n_exog, bs, sizeof(DataT)));
    if (order.p)
      rmm_alloc.deallocate(stream, ar, checked_mul<std::size_t>(order.p, bs, sizeof(DataT)));
    if (order.q)
      rmm_alloc.deallocate(stream, ma, checked_mul<std::size_t>(order.q, bs, sizeof(DataT)));
    if (order.P)
      rmm_alloc.deallocate(stream, sar, checked_mul<std::size_t>(order.P, bs, sizeof(DataT)));
    if (order.Q)
      rmm_alloc.deallocate(stream, sma, checked_mul<std::size_t>(order.Q, bs, sizeof(DataT)));
    rmm_alloc.deallocate(stream, sigma2, checked_mul<std::size_t>(bs, sizeof(DataT)));
  }

  /**
   * Pack the separate parameter arrays into a unique parameter vector
   *
   * @param[in]   order      ARIMA order
   * @param[in]   batch_size Batch size
   * @param[out]  param_vec  Linear array of all parameters grouped by batch
   *                         [mu, ar, ma, sar, sma, sigma2] (device)
   * @param[in]  stream      CUDA stream
   */
  void pack(const ARIMAOrder& order, int batch_size, DataT* param_vec, cudaStream_t stream) const;

  /**
   * Unpack a parameter vector into separate arrays of parameters.
   *
   * @param[in]  order      ARIMA order
   * @param[in]  batch_size Batch size
   * @param[in]  param_vec  Linear array of all parameters grouped by batch
   *                        [mu, ar, ma, sar, sma, sigma2] (device)
   * @param[in]  stream     CUDA stream
   */
  void unpack(const ARIMAOrder& order, int batch_size, const DataT* param_vec, cudaStream_t stream);
};

/**
 * Structure to manage ARIMA temporary memory allocations
 * @note The user is expected to give a preallocated buffer to the constructor,
 *       and ownership is not transferred to this struct! The buffer must be allocated
 *       as long as the object lives, and deallocated afterwards.
 */
template <typename T, int ALIGN = 256>
struct ARIMAMemory {
  T *params_mu, *params_beta, *params_ar, *params_ma, *params_sar, *params_sma, *params_sigma2,
    *Tparams_ar, *Tparams_ma, *Tparams_sar, *Tparams_sma, *Tparams_sigma2, *d_params, *d_Tparams,
    *Z_dense, *R_dense, *T_dense, *RQR_dense, *RQ_dense, *P_dense, *alpha_dense, *ImT_dense,
    *ImT_inv_dense, *v_tmp_dense, *m_tmp_dense, *K_dense, *TP_dense, *pred, *y_diff, *exog_diff,
    *loglike, *loglike_base, *loglike_pert, *x_pert, *I_m_AxA_dense, *I_m_AxA_inv_dense, *Ts_dense,
    *RQRs_dense, *Ps_dense;
  T **Z_batches, **R_batches, **T_batches, **RQR_batches, **RQ_batches, **P_batches,
    **alpha_batches, **ImT_batches, **ImT_inv_batches, **v_tmp_batches, **m_tmp_batches,
    **K_batches, **TP_batches, **I_m_AxA_batches, **I_m_AxA_inv_batches, **Ts_batches,
    **RQRs_batches, **Ps_batches;
  int *ImT_inv_P, *ImT_inv_info, *I_m_AxA_P, *I_m_AxA_info;

  size_t size;

 protected:
  char* buf;

  template <bool assign, typename ValType, checked_source... Factors>
  inline void append_buffer(ValType*& ptr, Factors... factors)
  {
    static_assert(ALIGN > 0, "ARIMA buffer alignment must be positive");

    constexpr auto alignment = static_cast<std::size_t>(ALIGN);
    auto const n_elem        = checked_mul<std::size_t>(std::size_t{1}, factors...);
    auto const n_bytes       = checked_mul<std::size_t>(n_elem, sizeof(ValType));
    auto const remainder     = n_bytes % alignment;
    auto const aligned_bytes =
      remainder == 0 ? n_bytes : checked_add<std::size_t>(n_bytes, alignment - remainder);

    if (assign) { ptr = reinterpret_cast<ValType*>(buf + size); }
    size = checked_add<std::size_t>(size, aligned_bytes);
  }

  template <bool assign>
  inline void buf_offsets(const ARIMAOrder& order,
                          int batch_size,
                          int n_obs,
                          char* in_buf = nullptr)
  {
    buf  = in_buf;
    size = 0;

    RAFT_EXPECTS(order.p >= 0 && order.d >= 0 && order.q >= 0,
                 "ARIMA orders p, d, and q must be non-negative");
    RAFT_EXPECTS(order.P >= 0 && order.D >= 0 && order.Q >= 0,
                 "Seasonal ARIMA orders P, D, and Q must be non-negative");
    RAFT_EXPECTS(order.s >= 0, "Seasonal period must be non-negative");
    RAFT_EXPECTS(order.k >= 0, "Intercept count must be non-negative");
    RAFT_EXPECTS(order.n_exog >= 0, "Number of exogenous regressors must be non-negative");
    RAFT_EXPECTS(batch_size >= 0, "batch_size must be non-negative");
    RAFT_EXPECTS(n_obs >= 0, "n_obs must be non-negative");

    auto const n_diff = order.n_diff();
    auto const r      = order.r();
    auto const rd     = order.rd();
    auto const N      = order.complexity();

    append_buffer<assign>(params_mu, order.k, batch_size);
    append_buffer<assign>(params_beta, order.n_exog, batch_size);
    append_buffer<assign>(params_ar, order.p, batch_size);
    append_buffer<assign>(params_ma, order.q, batch_size);
    append_buffer<assign>(params_sar, order.P, batch_size);
    append_buffer<assign>(params_sma, order.Q, batch_size);
    append_buffer<assign>(params_sigma2, batch_size);

    append_buffer<assign>(Tparams_ar, order.p, batch_size);
    append_buffer<assign>(Tparams_ma, order.q, batch_size);
    append_buffer<assign>(Tparams_sar, order.P, batch_size);
    append_buffer<assign>(Tparams_sma, order.Q, batch_size);
    append_buffer<assign>(Tparams_sigma2, batch_size);

    append_buffer<assign>(d_params, N, batch_size);
    append_buffer<assign>(d_Tparams, N, batch_size);
    append_buffer<assign>(Z_dense, rd, batch_size);
    append_buffer<assign>(Z_batches, batch_size);
    append_buffer<assign>(R_dense, rd, batch_size);
    append_buffer<assign>(R_batches, batch_size);
    append_buffer<assign>(T_dense, rd, rd, batch_size);
    append_buffer<assign>(T_batches, batch_size);
    append_buffer<assign>(RQ_dense, rd, batch_size);
    append_buffer<assign>(RQ_batches, batch_size);
    append_buffer<assign>(RQR_dense, rd, rd, batch_size);
    append_buffer<assign>(RQR_batches, batch_size);
    append_buffer<assign>(P_dense, rd, rd, batch_size);
    append_buffer<assign>(P_batches, batch_size);
    append_buffer<assign>(alpha_dense, rd, batch_size);
    append_buffer<assign>(alpha_batches, batch_size);
    append_buffer<assign>(ImT_dense, r, r, batch_size);
    append_buffer<assign>(ImT_batches, batch_size);
    append_buffer<assign>(ImT_inv_dense, r, r, batch_size);
    append_buffer<assign>(ImT_inv_batches, batch_size);
    append_buffer<assign>(ImT_inv_P, r, batch_size);
    append_buffer<assign>(ImT_inv_info, batch_size);
    append_buffer<assign>(v_tmp_dense, rd, batch_size);
    append_buffer<assign>(v_tmp_batches, batch_size);
    append_buffer<assign>(m_tmp_dense, rd, rd, batch_size);
    append_buffer<assign>(m_tmp_batches, batch_size);
    append_buffer<assign>(K_dense, rd, batch_size);
    append_buffer<assign>(K_batches, batch_size);
    append_buffer<assign>(TP_dense, rd, rd, batch_size);
    append_buffer<assign>(TP_batches, batch_size);

    append_buffer<assign>(pred, n_obs, batch_size);
    append_buffer<assign>(y_diff, n_obs, batch_size);
    append_buffer<assign>(exog_diff, n_obs, order.n_exog, batch_size);
    append_buffer<assign>(loglike, batch_size);
    append_buffer<assign>(loglike_base, batch_size);
    append_buffer<assign>(loglike_pert, batch_size);
    append_buffer<assign>(x_pert, N, batch_size);

    if (n_diff > 0) {
      append_buffer<assign>(Ts_dense, r, r, batch_size);
      append_buffer<assign>(Ts_batches, batch_size);
      append_buffer<assign>(RQRs_dense, r, r, batch_size);
      append_buffer<assign>(RQRs_batches, batch_size);
      append_buffer<assign>(Ps_dense, r, r, batch_size);
      append_buffer<assign>(Ps_batches, batch_size);
    }

    if (r <= 5) {
      // Note: temp mem for the direct Lyapunov solver grows very quickly!
      // This solver is used iff the condition above is satisfied
      append_buffer<assign>(I_m_AxA_dense, r, r, r, r, batch_size);
      append_buffer<assign>(I_m_AxA_batches, batch_size);
      append_buffer<assign>(I_m_AxA_inv_dense, r, r, r, r, batch_size);
      append_buffer<assign>(I_m_AxA_inv_batches, batch_size);
      append_buffer<assign>(I_m_AxA_P, r, r, batch_size);
      append_buffer<assign>(I_m_AxA_info, batch_size);
    }
  }

  /** Protected constructor to estimate max size */
  ARIMAMemory(const ARIMAOrder& order, int batch_size, int n_obs)
  {
    buf_offsets<false>(order, batch_size, n_obs);
  }

 public:
  /** Constructor to create pointers from buffer
   * @param[in] order      ARIMA order
   * @param[in] batch_size Number of series in the batch
   * @param[in] n_obs      Length of the series
   * @param[in] in_buf     Pointer to the temporary memory buffer.
   *                       Ownership is retained by the caller
   */
  ARIMAMemory(const ARIMAOrder& order, int batch_size, int n_obs, char* in_buf)
  {
    buf_offsets<true>(order, batch_size, n_obs, in_buf);
  }

  /** Static method to get the size of the required buffer allocation
   * @param[in] order      ARIMA order
   * @param[in] batch_size Number of series in the batch
   * @param[in] n_obs      Length of the series
   * @return Buffer size in bytes
   */
  static size_t compute_size(const ARIMAOrder& order, int batch_size, int n_obs)
  {
    ARIMAMemory temp(order, batch_size, n_obs);
    return temp.size;
  }
};

}  // namespace CUML_EXPORT ML
