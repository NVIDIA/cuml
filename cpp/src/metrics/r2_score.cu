/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/metrics/metrics.hpp>

#include <raft/core/handle.hpp>
#include <raft/stats/r2_score.cuh>

namespace ML {

namespace Metrics {

float r2_score_py(const raft::handle_t& handle, float* y, float* y_hat, int n)
{
  return raft::stats::r2_score(y, y_hat, n, handle.get_stream().get());
}

double r2_score_py(const raft::handle_t& handle, double* y, double* y_hat, int n)
{
  return raft::stats::r2_score(y, y_hat, n, handle.get_stream().get());
}

}  // namespace Metrics
}  // namespace ML
