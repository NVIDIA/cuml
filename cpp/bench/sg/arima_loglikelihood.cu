/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "benchmark.cuh"

#include <cuml/tsa/arima_common.h>
#include <cuml/tsa/batched_arima.hpp>

#include <raft/core/handle.hpp>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace ML {
namespace Bench {
namespace Arima {

struct ArimaParams {
  TimeSeriesParams data;
  ARIMAOrder order;
};

template <typename DataT>
class ArimaLoglikelihood : public TsFixtureRandom<DataT> {
 public:
  ArimaLoglikelihood(const std::string& name, const ArimaParams& p)
    : TsFixtureRandom<DataT>(name, p.data),
      order(p.order),
      param(0, rmm::cuda_stream_default),
      loglike(0, rmm::cuda_stream_default),
      temp_mem(0, rmm::cuda_stream_default)
  {
  }

  // Note: public function because of the __device__ lambda
  void runBenchmark(::benchmark::State& state) override
  {
    using MLCommon::Bench::CudaEventTimer;

    auto& handle  = *this->handle;
    auto stream   = handle.get_stream().get();
    auto counting = thrust::make_counting_iterator(0);

    // Generate random parameters
    int N = order.complexity();
    raft::random::Rng gpu_gen(this->params.seed, raft::random::GenPhilox);
    gpu_gen.uniform(param.data(), N * this->params.batch_size, -1.0, 1.0, stream);
    // Set sigma2 parameters to 1.0
    DataT* x = param.data();  // copy the object attribute for thrust
    thrust::for_each(thrust::cuda::par.on(stream),
                     counting,
                     counting + this->params.batch_size,
                     [=] __device__(int bid) { x[(bid + 1) * N - 1] = 1.0; });

    handle.sync_stream(stream);

    // Benchmark loop
    this->loopOnState(state, [this]() {
      ARIMAMemory<double> arima_mem(
        order, this->params.batch_size, this->params.n_obs, temp_mem.data());

      // Evaluate log-likelihood
      batched_loglike(*this->handle,
                      arima_mem,
                      this->data.X.data(),
                      nullptr,
                      this->params.batch_size,
                      this->params.n_obs,
                      order,
                      param.data(),
                      loglike.data(),
                      true,
                      false);
    });
  }

  void allocateBuffers(const ::benchmark::State& state)
  {
    Fixture::allocateBuffers(state);

    auto& handle = *this->handle;
    auto stream  = handle.get_stream();

    // Buffer for the model parameters
    param.resize(order.complexity() * this->params.batch_size, stream);

    // Buffers for the log-likelihood
    loglike.resize(this->params.batch_size, stream);

    // Temporary memory
    size_t temp_buf_size =
      ARIMAMemory<double>::compute_size(order, this->params.batch_size, this->params.n_obs);
    temp_mem.resize(temp_buf_size, stream);
  }

  void deallocateBuffers(const ::benchmark::State& state) { Fixture::deallocateBuffers(state); }

 protected:
  ARIMAOrder order;
  rmm::device_uvector<DataT> param;
  rmm::device_uvector<DataT> loglike;
  rmm::device_uvector<char> temp_mem;
};

std::vector<ArimaParams> getInputs()
{
  struct std::vector<ArimaParams> out;
  ArimaParams p;
  p.data.seed                        = 12345ULL;
  std::vector<ARIMAOrder> list_order = {{1, 1, 1, 0, 0, 0, 0, 0, 0},
                                        {1, 1, 1, 1, 1, 1, 4, 0, 0},
                                        {1, 1, 1, 1, 1, 1, 12, 0, 0},
                                        {1, 1, 1, 1, 1, 1, 24, 0, 0},
                                        {1, 1, 1, 1, 1, 1, 52, 0, 0}};
  std::vector<int> list_batch_size   = {10, 100, 1000, 10000};
  std::vector<int> list_n_obs        = {200, 500, 1000};
  for (auto& order : list_order) {
    for (auto& batch_size : list_batch_size) {
      for (auto& n_obs : list_n_obs) {
        p.order           = order;
        p.data.batch_size = batch_size;
        p.data.n_obs      = n_obs;
        out.push_back(p);
      }
    }
  }
  return out;
}

ML_BENCH_REGISTER(ArimaParams, ArimaLoglikelihood<double>, "arima", getInputs());

}  // namespace Arima
}  // namespace Bench
}  // namespace ML
