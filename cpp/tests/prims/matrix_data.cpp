/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/prims/opg/matrix/data.hpp>

#include <raft/core/error.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <limits>

namespace MLCommon {
namespace Matrix {

TEST(MatrixData, ComputesBytesAndElements)
{
  float value = 1.0f;
  Data<float> data(&value, size_t(4));

  EXPECT_EQ(data.numElements(), 4u);
  EXPECT_EQ(data.totalSize, data.numElements() * sizeof(float));

  data.setNumElements(2);
  EXPECT_EQ(data.numElements(), 2u);
  EXPECT_EQ(data.totalSize, data.numElements() * sizeof(float));
}

TEST(MatrixData, ThrowsOnElementCountOverflowForByteSize)
{
  float* ptr        = nullptr;
  auto max_elements = std::numeric_limits<size_t>::max() / sizeof(float);

  EXPECT_THROW(Data<float> data(ptr, max_elements + 1), raft::exception);
  Data<float> data(ptr, size_t(0));
  EXPECT_THROW(data.setNumElements(max_elements + 1), raft::exception);
}

}  // namespace Matrix
}  // namespace MLCommon
