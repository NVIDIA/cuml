
/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/metrics/metrics.hpp>

#include <raft/core/handle.hpp>
#include <raft/stats/entropy.cuh>

namespace ML {

namespace Metrics {
double entropy(const raft::handle_t& handle,
               const int* y,
               const int n,
               const int lower_class_range,
               const int upper_class_range)
{
  return raft::stats::entropy(y, n, lower_class_range, upper_class_range, handle.get_stream().get());
}
}  // namespace Metrics
}  // namespace ML
