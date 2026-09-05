/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../common/ml_benchmark.hpp"
#include "dataset.cuh"
#include "dataset_ts.cuh"

#include <cuml/common/logger.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/cuda_stream_pool.hpp>

#include <cuda_runtime.h>

#include <benchmark/benchmark.h>

namespace ML {
namespace Bench {

/** Main fixture to be inherited and used by all algos in cuML benchmark */
class Fixture : public MLCommon::Bench::Fixture {
 public:
  Fixture(const std::string& name) : MLCommon::Bench::Fixture(name) {}
  Fixture() = delete;

  void SetUp(const ::benchmark::State& state) override
  {
    if (stream == 0) { RAFT_CUDA_TRY(cudaStreamCreate(&stream)); }
    auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(numStreams());
    handle.reset(new raft::handle_t{rmm::cuda_stream_view{stream}, stream_pool});
    MLCommon::Bench::Fixture::SetUp(state);
  }

  void TearDown(const ::benchmark::State& state) override
  {
    handle->sync_stream_pool();
    handle->sync_stream();
    MLCommon::Bench::Fixture::TearDown(state);
    handle.reset();
  }

  // to keep compiler happy
  void SetUp(::benchmark::State& st) override { SetUp(const_cast<const ::benchmark::State&>(st)); }

  // to keep compiler happy
  void TearDown(::benchmark::State& st) override
  {
    TearDown(const_cast<const ::benchmark::State&>(st));
  }

 protected:
  // every benchmark should be overriding this
  virtual void runBenchmark(::benchmark::State& state) = 0;
  virtual void generateMetrics(::benchmark::State& state) {}
  virtual void allocateData(const ::benchmark::State& state) {}
  virtual void deallocateData(const ::benchmark::State& state) {}
  virtual void allocateTempBuffers(const ::benchmark::State& state) {}
  virtual void deallocateTempBuffers(const ::benchmark::State& state) {}

  void allocateBuffers(const ::benchmark::State& state) override
  {
    allocateData(state);
    allocateTempBuffers(state);
  }

  void deallocateBuffers(const ::benchmark::State& state) override
  {
    deallocateTempBuffers(state);
    deallocateData(state);
  }

  void BenchmarkCase(::benchmark::State& state)
  {
    runBenchmark(state);
    generateMetrics(state);
  }

  std::unique_ptr<raft::handle_t> handle;

  ///@todo: ideally, this should be determined at runtime based on the inputs
  ///       passed to the fixture. That will require a whole lot of plumbing of
  ///       interfaces. Thus, as a quick workaround, defining this static var.
  constexpr static std::int32_t numStreams() { return 16; }
};  // end class Fixture

/**
 * Fixture to be used for benchmarking classification algorithms when the input
 * suffices to be generated via `make_blobs`.
 */
template <typename D, typename L = int>
class BlobsFixture : public Fixture {
 public:
  BlobsFixture(const std::string& name, const DatasetParams p, const BlobsParams b)
    : Fixture(name), params(p), bParams(b)
  {
  }
  BlobsFixture() = delete;

 protected:
  void allocateData(const ::benchmark::State& state) override
  {
    data.allocate(*handle, params);
    data.blobs(*handle, params, bParams);
  }

  void deallocateData(const ::benchmark::State& state) override
  {
    data.deallocate(*handle, params);
  }

  DatasetParams params;
  /** parameters passed to `make_blobs` */
  BlobsParams bParams;
  Dataset<D, L> data;
};  // end class BlobFixture

/**
 * Fixture to be used for benchmarking regression algorithms when the input
 * suffices to be generated via `make_regression`.
 */
template <typename D>
class RegressionFixture : public Fixture {
 public:
  RegressionFixture(const std::string& name, const DatasetParams p, const RegressionParams r)
    : Fixture(name), params(p), rParams(r)
  {
  }
  RegressionFixture() = delete;

 protected:
  void allocateData(const ::benchmark::State& state) override
  {
    data.allocate(*handle, params);
    data.regression(*handle, params, rParams);
  }

  void deallocateData(const ::benchmark::State& state) override
  {
    data.deallocate(*handle, params);
  }

  DatasetParams params;
  /** parameters passed to `make_regression` */
  RegressionParams rParams;
  Dataset<D, D> data;
};  // end class RegressionFixture

/**
 * Fixture to be used for benchmarking time series algorithms when
 * the input suffices to be generated with a normal distribution.
 */
template <typename D>
class TsFixtureRandom : public Fixture {
 public:
  TsFixtureRandom(const std::string& name, const TimeSeriesParams p) : Fixture(name), params(p) {}
  TsFixtureRandom() = delete;

 protected:
  void allocateData(const ::benchmark::State& state) override
  {
    data.allocate(*handle, params);
    data.random(*handle, params);
  }

  TimeSeriesParams params;
  TimeSeriesDataset<D> data;
};  // end class TsFixtureRandom

}  // end namespace Bench
}  // end namespace ML
