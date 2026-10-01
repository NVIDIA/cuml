/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rf_test_utils.hpp"
#include "test_opg_utils.h"

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cuda_utils.cuh>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/stream>

#include <gtest/gtest.h>
#include <mpi.h>
#include <randomforest/randomforest.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace ML {
namespace Test {
namespace opg {

enum class WeightKind { ThreeRows, Uniform, WithZeros, Skewed };

struct RowSamplerTestParams {
  PartitionKind partition_kind;
  WeightKind weight_kind;
};

void initialize_mpi_once()
{
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) { MPI_Init(nullptr, nullptr); }
}

void get_mpi_local_rank_size(int& local_rank, int& local_size)
{
  MPI_Comm local_comm{};
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
  MPI_Comm_rank(local_comm, &local_rank);
  MPI_Comm_size(local_comm, &local_size);
  MPI_Comm_free(&local_comm);
}

std::vector<double> make_weights(WeightKind kind)
{
  switch (kind) {
    case WeightKind::ThreeRows: return {0.1, 0.4, 0.5};
    case WeightKind::Uniform: return std::vector<double>(23, 1.0);
    case WeightKind::WithZeros:
      return {0.0, 1.0, 0.0, 2.0, 4.0, 0.0, 3.0, 0.0, 5.0, 1.0, 0.0, 4.0, 0.0};
    case WeightKind::Skewed:
      return {0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.245, 0.5};
  }
  return {};
}

class RfMgRowSamplerTest : public ::testing::TestWithParam<RowSamplerTestParams> {};

TEST_P(RfMgRowSamplerTest, SamplesGlobalWeightDistribution)
{
  initialize_mpi_once();
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int local_rank = 0;
  int local_size = 1;
  get_mpi_local_rank_size(local_rank, local_size);
  int n_gpus = 0;
  RAFT_CUDA_TRY(cudaGetDeviceCount(&n_gpus));
  ASSERT_GE(n_gpus, local_size);
  RAFT_CUDA_TRY(cudaSetDevice(local_rank));

  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
  raft::handle_t handle(cuda::stream_ref{cudaStreamPerThread}, stream_pool);
  raft::comms::initialize_mpi_comms(&handle, MPI_COMM_WORLD);

  auto global_weights = make_weights(GetParam().weight_kind);
  auto local_rows     = local_rows_for_rank(
    static_cast<int>(global_weights.size()), rank, size, GetParam().partition_kind);
  std::vector<double> local_weights(local_rows.size());
  std::transform(local_rows.begin(), local_rows.end(), local_weights.begin(), [&](int row) {
    return global_weights[row];
  });

  auto weight_buffer_size = std::max(std::size_t{1}, local_weights.size());
  rmm::device_uvector<double> d_weights(weight_buffer_size, handle.get_stream());
  raft::update_device(
    d_weights.data(), local_weights.data(), local_weights.size(), handle.get_stream());

  RF_params rf_params{};
  rf_params.bootstrap = true;
  rf_params.seed      = 123456789ULL;
  rf_params.n_streams = 1;

  constexpr std::int64_t sample_count = 200000;
  detail::RowSampler sampler(handle,
                             rf_params,
                             static_cast<std::int64_t>(local_rows.size()),
                             sample_count,
                             1,
                             nullptr,
                             d_weights.data());

  auto stream   = handle.get_stream_from_stream_pool(0);
  auto& row_ids = sampler.sample(0, 0, stream.get());
  std::vector<std::int64_t> h_row_ids(row_ids.size());
  raft::update_host(h_row_ids.data(), row_ids.data(), row_ids.size(), stream);
  handle.sync_stream(stream);

  std::vector<std::uint64_t> local_counts(global_weights.size(), 0);
  std::uint64_t local_invalid_row_count = 0;
  for (auto local_row : h_row_ids) {
    if (local_row < 0 || local_row >= static_cast<std::int64_t>(local_rows.size())) {
      local_invalid_row_count++;
      continue;
    }
    local_counts[local_rows[local_row]]++;
  }

  std::uint64_t global_invalid_row_count = 0;
  MPI_Allreduce(
    &local_invalid_row_count, &global_invalid_row_count, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
  std::vector<std::uint64_t> global_counts(global_weights.size(), 0);
  MPI_Allreduce(local_counts.data(),
                global_counts.data(),
                static_cast<int>(global_counts.size()),
                MPI_UINT64_T,
                MPI_SUM,
                MPI_COMM_WORLD);

  ASSERT_EQ(global_invalid_row_count, 0);
  auto observed_total =
    std::accumulate(global_counts.begin(), global_counts.end(), std::uint64_t{0});
  ASSERT_EQ(observed_total, static_cast<std::uint64_t>(sample_count));

  double weight_sum = std::accumulate(global_weights.begin(), global_weights.end(), 0.0);
  for (std::size_t row = 0; row < global_weights.size(); ++row) {
    double expected_probability = global_weights[row] / weight_sum;
    double observed_probability = static_cast<double>(global_counts[row]) / sample_count;
    if (expected_probability == 0.0) {
      EXPECT_EQ(global_counts[row], 0) << "global row " << row;
      continue;
    }
    double standard_error =
      std::sqrt(expected_probability * (1.0 - expected_probability) / sample_count);
    double tolerance = std::max(0.002, 6.0 * standard_error);
    EXPECT_NEAR(observed_probability, expected_probability, tolerance) << "global row " << row;
  }
}

std::string row_sampler_test_name(::testing::TestParamInfo<RowSamplerTestParams> const& test_info)
{
  char const* partition_name = nullptr;
  switch (test_info.param.partition_kind) {
    case PartitionKind::Contiguous: partition_name = "Contiguous"; break;
    case PartitionKind::Strided: partition_name = "Strided"; break;
    case PartitionKind::Imbalanced: partition_name = "Imbalanced"; break;
    case PartitionKind::EmptyNonRootRanks: partition_name = "EmptyNonRootRanks"; break;
  }
  char const* weight_name = nullptr;
  switch (test_info.param.weight_kind) {
    case WeightKind::ThreeRows: weight_name = "ThreeRows"; break;
    case WeightKind::Uniform: weight_name = "Uniform"; break;
    case WeightKind::WithZeros: weight_name = "WithZeros"; break;
    case WeightKind::Skewed: weight_name = "Skewed"; break;
  }
  return std::string{partition_name} + weight_name;
}

std::vector<RowSamplerTestParams> make_row_sampler_inputs()
{
  std::vector<RowSamplerTestParams> inputs;
  for (auto partition : {PartitionKind::Contiguous,
                         PartitionKind::Strided,
                         PartitionKind::Imbalanced,
                         PartitionKind::EmptyNonRootRanks}) {
    for (auto weights :
         {WeightKind::ThreeRows, WeightKind::Uniform, WeightKind::WithZeros, WeightKind::Skewed}) {
      inputs.push_back({partition, weights});
    }
  }
  return inputs;
}

INSTANTIATE_TEST_SUITE_P(RfRowSamplerTests,
                         RfMgRowSamplerTest,
                         ::testing::ValuesIn(make_row_sampler_inputs()),
                         row_sampler_test_name);

}  // namespace opg
}  // namespace Test
}  // namespace ML

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MLCommon::Test::opg::MPIEnvironment());
  return RUN_ALL_TESTS();
}
