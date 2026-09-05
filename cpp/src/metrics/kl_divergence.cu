
/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/metrics/metrics.hpp>

#include <raft/core/handle.hpp>
#include <raft/stats/kl_divergence.cuh>

namespace ML {

namespace Metrics {

double kl_divergence(const raft::handle_t& handle, const double* y, const double* y_hat, int n)
{
  return raft::stats::kl_divergence(y, y_hat, n, handle.get_stream().get());
}

float kl_divergence(const raft::handle_t& handle, const float* y, const float* y_hat, int n)
{
  return raft::stats::kl_divergence(y, y_hat, n, handle.get_stream().get());
}
}  // namespace Metrics
}  // namespace ML
