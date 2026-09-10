/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/tsa/arima_common.h>

#include <raft/core/error.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <limits>
#include <vector>

namespace ML {
namespace {

TEST(ArimaMemory, ComputesAlignedOffsetsConsistently)
{
  ARIMAOrder const order{1, 1, 1, 0, 0, 0, 0, 1, 2};
  constexpr int batch_size = 3;
  constexpr int n_obs      = 20;
  constexpr std::size_t alignment{256};

  auto const size = ARIMAMemory<double>::compute_size(order, batch_size, n_obs);
  std::vector<char> buffer(size);
  ARIMAMemory<double> memory(order, batch_size, n_obs, buffer.data());

  EXPECT_EQ(memory.size, size);
  for (auto const* ptr : {reinterpret_cast<char*>(memory.params_mu),
                          reinterpret_cast<char*>(memory.T_dense),
                          reinterpret_cast<char*>(memory.T_batches),
                          reinterpret_cast<char*>(memory.exog_diff),
                          reinterpret_cast<char*>(memory.I_m_AxA_info)}) {
    auto const offset = static_cast<std::size_t>(ptr - buffer.data());
    EXPECT_LT(offset, size);
    EXPECT_EQ(offset % alignment, 0);
  }
}

TEST(ArimaMemory, ComputesElementCountsBeforeWidening)
{
  ARIMAOrder const order{};
  constexpr int batch_size = std::numeric_limits<int>::max();
  auto const size_one_obs  = ARIMAMemory<double>::compute_size(order, batch_size, 1);
  auto const size_two_obs  = ARIMAMemory<double>::compute_size(order, batch_size, 2);
  auto aligned_bytes       = [](int n_obs) {
    constexpr std::size_t alignment{256};
    auto const n_bytes   = checked_mul<std::size_t>(n_obs, batch_size, sizeof(double));
    auto const remainder = n_bytes % alignment;
    return remainder == 0 ? n_bytes : checked_add<std::size_t>(n_bytes, alignment - remainder);
  };
  auto const expected_growth = checked_mul<std::size_t>(2, aligned_bytes(2) - aligned_bytes(1));

  EXPECT_EQ(size_two_obs - size_one_obs, expected_growth);
}

TEST(ArimaMemory, RejectsDerivedDimensionOverflow)
{
  ARIMAOrder const order{0, 0, 0, std::numeric_limits<int>::max(), 0, 0, 2, 0, 0};

  EXPECT_THROW(order.n_phi(), raft::exception);
  EXPECT_THROW(ARIMAMemory<double>::compute_size(order, 1, 1), raft::exception);
}

TEST(ArimaMemory, RejectsSizeOverflow)
{
  ARIMAOrder const order{std::numeric_limits<int>::max(), 0, 0, 0, 0, 0, 0, 0, 0};

  EXPECT_THROW(ARIMAMemory<double>::compute_size(order, std::numeric_limits<int>::max(), 1),
               raft::exception);
}

TEST(ArimaMemory, RejectsNegativeDimensions)
{
  ARIMAOrder const order{};
  ARIMAOrder const negative_order{-1, 0, 0, 0, 0, 0, 0, 0, 0};

  EXPECT_THROW(ARIMAMemory<double>::compute_size(order, -1, 1), raft::exception);
  EXPECT_THROW(ARIMAMemory<double>::compute_size(order, 1, -1), raft::exception);
  EXPECT_THROW(ARIMAMemory<double>::compute_size(negative_order, 1, 1), raft::exception);
}

}  // namespace
}  // namespace ML
