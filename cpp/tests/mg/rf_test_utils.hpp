/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <numeric>
#include <vector>

namespace ML {
namespace Test {
namespace opg {

enum class PartitionKind { Contiguous, Strided, Imbalanced, EmptyNonRootRanks };

inline std::vector<int> local_rows_for_rank(int n_rows, int rank, int size, PartitionKind kind)
{
  std::vector<int> rows;
  if (kind == PartitionKind::Strided) {
    for (int row = rank; row < n_rows; row += size) {
      rows.push_back(row);
    }
    return rows;
  }

  std::vector<int> counts(size, n_rows / size);
  for (int i = 0; i < n_rows % size; ++i) {
    counts[i]++;
  }
  if (kind == PartitionKind::Imbalanced && size > 1) {
    counts.assign(size, 0);
    counts[0]     = std::max(1, (n_rows * 3) / 4);
    int remaining = n_rows - counts[0];
    for (int i = 1; i < size; ++i) {
      counts[i] = remaining / (size - 1);
    }
    for (int i = 1; i <= remaining % (size - 1); ++i) {
      counts[i]++;
    }
  } else if (kind == PartitionKind::EmptyNonRootRanks && size > 1) {
    counts.assign(size, 0);
    counts[0] = n_rows;
  }

  int begin = std::accumulate(counts.begin(), counts.begin() + rank, 0);
  rows.resize(counts[rank]);
  std::iota(rows.begin(), rows.end(), begin);
  rows.erase(std::remove_if(rows.begin(), rows.end(), [=](int row) { return row >= n_rows; }),
             rows.end());
  return rows;
}

}  // namespace opg
}  // namespace Test
}  // namespace ML
