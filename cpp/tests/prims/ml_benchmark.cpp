/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../bench/common/ml_benchmark.hpp"

#include <raft/core/error.hpp>

#include <gtest/gtest.h>

#include <benchmark/benchmark.h>

#include <cstddef>
#include <limits>

namespace MLCommon {
namespace Bench {

class TestFixture : public Fixture {
 public:
  TestFixture() : Fixture("MLBenchmarkFixtureTest") {}

  void runBenchmark(::benchmark::State&) override {}

  template <typename T>
  void testAlloc(T*& ptr, size_t len, bool init = false)
  {
    alloc(ptr, len, init);
  }

  template <typename T>
  void testDealloc(T* ptr, size_t len)
  {
    dealloc(ptr, len);
  }
};

TEST(MlBenchmarkFixtureAllocator, ThrowOnHugeAllocationOrDeallocationLength)
{
  TestFixture fixture;

  int* ptr      = nullptr;
  auto const len = std::numeric_limits<size_t>::max() / sizeof(int) + 1;

  EXPECT_THROW(fixture.testAlloc(ptr, len), raft::exception);
  EXPECT_THROW(fixture.testDealloc(ptr, len), raft::exception);
}

}  // namespace Bench
}  // namespace MLCommon
