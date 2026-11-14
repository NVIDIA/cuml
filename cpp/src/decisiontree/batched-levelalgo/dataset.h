/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/util/cuda_utils.cuh>

#include <cstddef>
#include <cstdint>

namespace ML {
namespace DT {

template <typename DataT, typename LabelT>
struct Dataset {
  /** input dataset */
  const DataT* data;
  /** input labels */
  const LabelT* labels;
  /** optional input sample weights */
  const double* sample_weight;
  /** optional bootstrap multiplicities, aligned with row_ids */
  std::uint32_t* bootstrap_count;
  /** total rows in dataset */
  std::int64_t n_rows;
  /** total cols in dataset */
  std::int64_t n_cols;
  /** row stride in input data elements */
  std::int64_t row_stride;
  /** column stride in input data elements */
  std::int64_t col_stride;
  /** total sampled rows in dataset */
  std::int64_t n_sampled_rows;
  /** total sampled cols in dataset */
  std::int64_t n_sampled_cols;
  /** indices of sampled rows */
  std::int64_t* row_ids;
  /** Number of classes or regression outputs*/
  int num_outputs;

  HDI DataT value(std::int64_t row, std::int64_t col) const
  {
    return data[row * row_stride + col * col_stride];
  }

  HDI double weight(std::size_t row_position, std::int64_t row) const
  {
    if (bootstrap_count != nullptr) { return static_cast<double>(bootstrap_count[row_position]); }
    return sample_weight == nullptr ? 1.0 : sample_weight[row];
  }
};

}  // namespace DT
}  // namespace ML
