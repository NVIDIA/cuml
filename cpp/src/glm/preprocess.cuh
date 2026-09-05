/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/handle.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/subtract.cuh>
#include <raft/stats/mean.cuh>
#include <raft/stats/mean_center.cuh>
#include <raft/stats/meanvar.cuh>
#include <raft/stats/stddev.cuh>
#include <raft/stats/weighted_mean.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

namespace ML {
namespace GLM {

/**
 * @brief Center and scale the data, depending on fit_intercept
 *
 * @tparam math_t the element type
 * @param [inout] input the column-major data of size [n_rows, n_cols]
 * @param [in] n_rows
 * @param [in] n_cols
 * @param [inout] labels vector of size [n_rows]
 * @param [out] intercept
 * @param [out] mu_input the column-wise means of the input of size [n_cols]
 * @param [out] mu_labels the scalar mean of the target (labels vector)
 * @param [in] fit_intercept whether to center the data / to fit the intercept
 * @param [in] stream
 */
template <typename math_t>
void preProcessData(const raft::handle_t& handle,
                    math_t* input,
                    size_t n_rows,
                    size_t n_cols,
                    math_t* labels,
                    math_t* intercept,
                    math_t* mu_input,
                    math_t* mu_labels,
                    bool fit_intercept,
                    math_t* sample_weight = nullptr)
{
  cudaStream_t stream = handle.get_stream().get();
  raft::common::nvtx::range fun_scope("ML::GLM::preProcessData-%d-%d", n_rows, n_cols);
  ASSERT(n_cols > 0, "Parameter n_cols: number of columns cannot be less than one");
  ASSERT(n_rows > 1, "Parameter n_rows: number of rows cannot be less than two");

  if (fit_intercept) {
    if (sample_weight != nullptr) {
      raft::stats::weightedMean<false, false>(
        mu_input, input, sample_weight, n_cols, n_rows, stream);
    } else {
      raft::stats::mean<false>(mu_input, input, n_cols, n_rows, false, stream);
    }
    raft::stats::meanCenter<false, true>(input, input, mu_input, n_cols, n_rows, stream);
    if (sample_weight != nullptr) {
      raft::stats::weightedMean<true, false>(
        mu_labels, labels, sample_weight, (size_t)1, n_rows, stream);
    } else {
      raft::stats::mean<false>(mu_labels, labels, (size_t)1, n_rows, false, stream);
    }
    raft::stats::meanCenter<false, true>(labels, labels, mu_labels, (size_t)1, n_rows, stream);
  }
}

template <typename math_t>
void postProcessData(const raft::handle_t& handle,
                     math_t* input,
                     size_t n_rows,
                     size_t n_cols,
                     math_t* labels,
                     math_t* coef,
                     math_t* intercept,
                     math_t* mu_input,
                     math_t* mu_labels,
                     bool fit_intercept)
{
  cudaStream_t stream = handle.get_stream().get();
  raft::common::nvtx::range fun_scope("ML::GLM::postProcessData-%d-%d", n_rows, n_cols);
  ASSERT(n_cols > 0, "Parameter n_cols: number of columns cannot be less than one");
  ASSERT(n_rows > 1, "Parameter n_rows: number of rows cannot be less than two");

  cublasHandle_t cublas_handle = handle.get_cublas_handle();
  rmm::device_scalar<math_t> d_intercept(stream);

  raft::linalg::gemm(handle,
                     mu_input,
                     (size_t)1,
                     n_cols,
                     coef,
                     d_intercept.data(),
                     1,
                     1,
                     CUBLAS_OP_N,
                     CUBLAS_OP_N,
                     stream);

  raft::linalg::subtract(d_intercept.data(), mu_labels, d_intercept.data(), 1, stream);
  *intercept = d_intercept.value(stream);

  raft::stats::meanAdd<false, true>(input, input, mu_input, n_cols, n_rows, stream);
  raft::stats::meanAdd<false, true>(labels, labels, mu_labels, (size_t)1, n_rows, stream);
}

};  // namespace GLM
};  // namespace ML
// end namespace ML
