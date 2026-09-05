/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/datasets/make_arima.hpp>

#include <raft/core/handle.hpp>

#include <random/make_arima.cuh>

namespace ML {
namespace Datasets {

template <typename DataT, typename IdxT>
inline void make_arima_helper(const raft::handle_t& handle,
                              DataT* out,
                              IdxT batch_size,
                              IdxT n_obs,
                              ARIMAOrder order,
                              DataT scale,
                              DataT noise_scale,
                              DataT intercept_scale,
                              uint64_t seed)
{
  auto stream = handle.get_stream().get();

  MLCommon::Random::make_arima(
    out, batch_size, n_obs, order, stream, scale, noise_scale, intercept_scale, seed);
}

void make_arima(const raft::handle_t& handle,
                float* out,
                int batch_size,
                int n_obs,
                ARIMAOrder order,
                float scale,
                float noise_scale,
                float intercept_scale,
                uint64_t seed)
{
  make_arima_helper(
    handle, out, batch_size, n_obs, order, scale, noise_scale, intercept_scale, seed);
}

void make_arima(const raft::handle_t& handle,
                double* out,
                int batch_size,
                int n_obs,
                ARIMAOrder order,
                double scale,
                double noise_scale,
                double intercept_scale,
                uint64_t seed)
{
  make_arima_helper(
    handle, out, batch_size, n_obs, order, scale, noise_scale, intercept_scale, seed);
}

}  // namespace Datasets
}  // namespace ML
