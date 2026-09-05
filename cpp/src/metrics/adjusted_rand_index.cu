
/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/metrics/metrics.hpp>

#include <raft/core/handle.hpp>
#include <raft/stats/adjusted_rand_index.cuh>

namespace ML {

namespace Metrics {
double adjusted_rand_index(const raft::handle_t& handle,
                           const int64_t* y,
                           const int64_t* y_hat,
                           const int64_t n)
{
  return raft::stats::adjusted_rand_index<int64_t, unsigned long long>(
    y, y_hat, n, handle.get_stream().get());
}

double adjusted_rand_index(const raft::handle_t& handle,
                           const int* y,
                           const int* y_hat,
                           const int n)
{
  return raft::stats::adjusted_rand_index<int, unsigned long long>(
    y, y_hat, n, handle.get_stream().get());
}
}  // namespace Metrics
}  // namespace ML
