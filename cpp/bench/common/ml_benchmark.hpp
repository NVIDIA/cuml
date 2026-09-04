/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/common/logger.hpp>
#include <cuml/common/utils.hpp>
#include <cuml/common/checked_arithmetic.hpp>

#include <raft/util/cudart_utils.hpp>

#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime.h>

#include <benchmark/benchmark.h>

#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace MLCommon {
namespace Bench {

/**
 * RAII way of timing cuda calls. This has been shamelessly copied from the
 * cudf codebase. So, credits for this class goes to cudf developers.
 */
struct CudaEventTimer {
 public:
  /**
   * @brief This ctor clears the L2 cache by cudaMemset'ing a buffer of the size
   *        of L2 and then starts the timer.
   * @param st the benchmark::State whose timer we are going to update.
   * @param ptr         flush the L2 cache by writing to this buffer before
   *                    every iteration. It is the responsibility of the caller
   *                    to manage this buffer. Pass a `nullptr` if L2 flush is
   *                    not needed.
   * @param l2CacheSize L2 Cache size (in B). Passing this as 0 also disables
   *                    the L2 cache flush.
   * @param s           CUDA stream we are measuring time on.
   */
  CudaEventTimer(::benchmark::State& st, char* ptr, int l2CacheSize, cudaStream_t s)
    : state(&st), stream(s)
  {
    RAFT_CUDA_TRY(cudaEventCreate(&start));
    RAFT_CUDA_TRY(cudaEventCreate(&stop));
    // flush L2?
    if (ptr != nullptr && l2CacheSize > 0) {
      RAFT_CUDA_TRY(cudaMemsetAsync(ptr, 0, sizeof(char) * l2CacheSize, s));
      RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    }
    RAFT_CUDA_TRY(cudaEventRecord(start, stream));
  }
  CudaEventTimer() = delete;

  /**
   * @brief The dtor stops the timer and performs a synchroniazation. Time of
   *       the benchmark::State object provided to the ctor will be set to the
   *       value given by `cudaEventElapsedTime()`.
   */
  ~CudaEventTimer()
  {
    RAFT_CUDA_TRY_NO_THROW(cudaEventRecord(stop, stream));
    RAFT_CUDA_TRY_NO_THROW(cudaEventSynchronize(stop));
    float milliseconds = 0.0f;
    RAFT_CUDA_TRY_NO_THROW(cudaEventElapsedTime(&milliseconds, start, stop));
    state->SetIterationTime(milliseconds / 1000.f);
    RAFT_CUDA_TRY_NO_THROW(cudaEventDestroy(start));
    RAFT_CUDA_TRY_NO_THROW(cudaEventDestroy(stop));
  }

 private:
  ::benchmark::State* state;
  cudaStream_t stream = 0;
  cudaEvent_t start;
  cudaEvent_t stop;
};  // end struct CudaEventTimer

/** Main fixture to be inherited and used by all other c++ benchmarks in cuml */
class Fixture : public ::benchmark::Fixture {
 public:
  Fixture(const std::string& name) : ::benchmark::Fixture() { SetName(name.c_str()); }
  Fixture() = delete;

  void SetUp(const ::benchmark::State& state) override
  {
    if (stream == 0) { RAFT_CUDA_TRY(cudaStreamCreate(&stream)); }
    allocateBuffers(state);
    int devId = 0;
    RAFT_CUDA_TRY(cudaGetDevice(&devId));
    l2CacheSize = 0;
    RAFT_CUDA_TRY(cudaDeviceGetAttribute(&l2CacheSize, cudaDevAttrL2CacheSize, devId));
    if (l2CacheSize > 0) {
      alloc(scratchBuffer, l2CacheSize, false);
    } else {
      scratchBuffer = nullptr;
    }
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

  void TearDown(const ::benchmark::State& state) override
  {
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    if (l2CacheSize > 0) {
      dealloc(scratchBuffer, l2CacheSize);
      scratchBuffer = nullptr;
    }
    deallocateBuffers(state);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    RAFT_CUDA_TRY(cudaStreamDestroy(stream));
    stream = 0;
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
  virtual void allocateBuffers(const ::benchmark::State& state) {}
  virtual void deallocateBuffers(const ::benchmark::State& state) {}

  void BenchmarkCase(::benchmark::State& state)
  {
    runBenchmark(state);
    generateMetrics(state);
  }

  template <typename Lambda>
  void loopOnState(::benchmark::State& state, Lambda benchmarkFunc, bool flushL2 = true)
  {
    char* buff;
    int size;
    if (flushL2) {
      buff = scratchBuffer;
      size = l2CacheSize;
    } else {
      buff = nullptr;
      size = 0;
    }
    for (auto _ : state) {
      CudaEventTimer timer(state, buff, size, stream);
      benchmarkFunc();
    }
  }

  template <typename T>
  void alloc(T*& ptr, size_t len, bool init = false)
  {
    auto nBytes = ML::checked_mul<size_t>(len, sizeof(T));
    auto d_alloc = rmm::mr::get_current_device_resource_ref();
    ptr          = (T*)d_alloc.allocate(stream, nBytes);
    if (init) { RAFT_CUDA_TRY(cudaMemsetAsync(ptr, 0, nBytes, stream)); }
  }

  template <typename T>
  void dealloc(T* ptr, size_t len)
  {
    auto d_alloc = rmm::mr::get_current_device_resource_ref();
    auto nBytes = ML::checked_mul<size_t>(len, sizeof(T));
    d_alloc.deallocate(stream, ptr, nBytes);
  }

  cudaStream_t stream = 0;
  int l2CacheSize;
  char* scratchBuffer;
};  // class Fixture

namespace internal {
template <typename Params, typename Class>
struct Registrar {
  Registrar(const std::vector<Params>& paramsList,
            const std::string& testClass,
            const std::string& testName)
  {
    int counter = 0;
    for (const auto& param : paramsList) {
      std::stringstream oss;
      oss << testClass;
      if (!testName.empty()) oss << "/" << testName;
      oss << "/" << counter;
      auto testFullName = oss.str();
      auto* b           = ::benchmark::internal::RegisterBenchmarkInternal(
        std::make_unique<Class>(testFullName, param));
      ///@todo: expose a currying-like interface to the final macro
      b->UseManualTime();
      b->Unit(benchmark::kMillisecond);
      ++counter;
    }
  }
};  // end struct Registrar
};  // end namespace internal

/**
 * This is the entry point macro for all ml benchmarks. This needs to be called
 * for the set of benchmarks to be registered so that the main harness inside
 * google bench can find these benchmarks and run them.
 *
 * @param ParamsClass a struct which contains all the parameters needed to
 *                    generate inputs and run the underlying prim on it.
 *                    Ideally, one such struct is needed for every ml-prim.
 * @param TestClass   child class of `MLCommon::Bench::Fixture` which contains
 *                    the logic to generate the dataset and run training on it
 *                    for a given algo. Ideally, once such struct is needed for
 *                    every algo to be benchmarked
 * @param TestName    a unique string to identify these tests at the end of run
 *                    This is optional and if choose not to use this, pass an
 *                    empty string
 * @param params      list of params upon which to benchmark the prim. It can be
 *                    a statically populated vector or from the result of
 *                    calling a function
 */
#define ML_BENCH_REGISTER(ParamsClass, TestClass, TestName, params)                           \
  static MLCommon::Bench::internal::Registrar<ParamsClass, TestClass> BENCHMARK_PRIVATE_NAME( \
    registrar)(params, #TestClass, TestName)

}  // end namespace Bench
}  // end namespace MLCommon
