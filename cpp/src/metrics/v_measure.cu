
/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/metrics/metrics.hpp>

#include <raft/core/handle.hpp>
#include <raft/stats/v_measure.cuh>

namespace ML {

namespace Metrics {

double v_measure(const raft::handle_t& handle,
                 const int* y,
                 const int* y_hat,
                 const int n,
                 const int lower_class_range,
                 const int upper_class_range,
                 double beta)
{
  return raft::stats::v_measure(
    y, y_hat, n, lower_class_range, upper_class_range, handle.get_stream().get(), beta);
}
}  // namespace Metrics
}  // namespace ML
