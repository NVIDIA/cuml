
/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/metrics/metrics.hpp>

#include <raft/core/handle.hpp>
#include <raft/stats/accuracy.cuh>

namespace ML {

namespace Metrics {

float accuracy_score_py(const raft::handle_t& handle,
                        const int* predictions,
                        const int* ref_predictions,
                        int n)
{
  return raft::stats::accuracy(predictions, ref_predictions, n, handle.get_stream().get());
}
}  // namespace Metrics
}  // namespace ML
