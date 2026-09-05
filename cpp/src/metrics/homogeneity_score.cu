
/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/metrics/metrics.hpp>

#include <raft/core/handle.hpp>
#include <raft/stats/homogeneity_score.cuh>

namespace ML {

namespace Metrics {

double homogeneity_score(const raft::handle_t& handle,
                         const int* y,
                         const int* y_hat,
                         const int n,
                         const int lower_class_range,
                         const int upper_class_range)
{
  return raft::stats::homogeneity_score(
    y, y_hat, n, lower_class_range, upper_class_range, handle.get_stream().get());
}
}  // namespace Metrics
}  // namespace ML
