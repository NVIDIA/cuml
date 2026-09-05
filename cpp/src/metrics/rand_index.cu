
/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/metrics/metrics.hpp>

#include <raft/core/handle.hpp>
#include <raft/stats/rand_index.cuh>

namespace ML {

namespace Metrics {

double rand_index(const raft::handle_t& handle, const double* y, const double* y_hat, int n)
{
  return raft::stats::rand_index(y, y_hat, (uint64_t)n, handle.get_stream().get());
}
}  // namespace Metrics
}  // namespace ML
