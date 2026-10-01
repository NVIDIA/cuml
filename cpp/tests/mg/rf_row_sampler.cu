/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../prims/test_utils.h"
#include "rf_test_utils.hpp"
#include "test_opg_utils.h"

#include <cuml/ensemble/randomforest.hpp>

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

struct RowSamplerTestParams {
  std::string name;
  PartitionKind partition_kind;
  std::vector<double> sample_weights;
  int n_trees;
  float max_samples;
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

class RfRowSamplerTest : public ::testing::TestWithParam<RowSamplerTestParams> {};

TEST_P(RfRowSamplerTest, SamplesFollowGlobalWeights)
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
  int insufficient_local_gpus = n_gpus < local_size;
  int any_insufficient_gpus   = 0;
  MPI_Allreduce(
    &insufficient_local_gpus, &any_insufficient_gpus, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (any_insufficient_gpus) { GTEST_SKIP() << "This test requires one GPU per local MPI rank"; }
  RAFT_CUDA_TRY(cudaSetDevice(local_rank));

  auto const& params = GetParam();
  auto n_rows        = static_cast<int>(params.sample_weights.size());
  auto local_rows    = local_rows_for_rank(n_rows, rank, size, params.partition_kind);

  std::vector<double> local_weights(local_rows.size());
  for (std::size_t i = 0; i < local_rows.size(); ++i) {
    local_weights[i] = params.sample_weights[local_rows[i]];
  }

  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
  raft::handle_t handle(cuda::stream_ref{cudaStreamPerThread}, stream_pool);
  raft::comms::initialize_mpi_comms(&handle, MPI_COMM_WORLD);

  // Empty ranks still pass a non-null pointer so all ranks enter the weighted bootstrap path.
  auto weight_buffer_size = std::max(std::size_t{1}, local_weights.size());
  rmm::device_uvector<double> device_weights(weight_buffer_size, handle.get_stream());
  if (!local_weights.empty()) {
    raft::update_device(
      device_weights.data(), local_weights.data(), local_weights.size(), handle.get_stream());
  }

  RF_params rf_params{};
  rf_params.bootstrap   = true;
  rf_params.max_samples = params.max_samples;
  rf_params.seed        = 123456789ULL;
  rf_params.n_streams   = 1;
  auto global_sample_count =
    static_cast<std::int64_t>(std::round(params.max_samples * params.sample_weights.size()));
  detail::RowSampler sampler(handle,
                             rf_params,
                             static_cast<std::int64_t>(local_rows.size()),
                             global_sample_count,
                             1,
                             nullptr,
                             device_weights.data());

  std::vector<std::int64_t> local_counts(params.sample_weights.size(), 0);
  auto sample_stream = handle.get_stream_from_stream_pool(0);
  for (int tree_id = 0; tree_id < params.n_trees; ++tree_id) {
    auto& selected_rows = sampler.sample(tree_id, 0, sample_stream.get());
    std::vector<std::int64_t> selected_rows_host(selected_rows.size());
    if (!selected_rows_host.empty()) {
      raft::update_host(selected_rows_host.data(),
                        selected_rows.data(),
                        selected_rows_host.size(),
                        sample_stream.get());
      handle.sync_stream(sample_stream.get());
    }

    for (auto local_row : selected_rows_host) {
      if (local_row < 0 || local_row >= static_cast<std::int64_t>(local_rows.size())) {
        ADD_FAILURE() << "RowSampler returned invalid rank-local row " << local_row;
        continue;
      }
      ++local_counts[local_rows[local_row]];
    }
  }

  std::vector<std::int64_t> global_counts(params.sample_weights.size(), 0);
  MPI_Allreduce(local_counts.data(),
                global_counts.data(),
                static_cast<int>(global_counts.size()),
                MPI_INT64_T,
                MPI_SUM,
                MPI_COMM_WORLD);

  if (rank != 0) { return; }

  auto draws_per_tree =
    static_cast<std::int64_t>(std::round(params.max_samples * params.sample_weights.size()));
  auto expected_total = draws_per_tree * params.n_trees;
  auto observed_total =
    std::accumulate(global_counts.begin(), global_counts.end(), std::int64_t{0});
  ASSERT_EQ(observed_total, expected_total);

  auto weight_sum =
    std::accumulate(params.sample_weights.begin(), params.sample_weights.end(), 0.0);
  ASSERT_GT(weight_sum, 0.0);
  for (std::size_t row = 0; row < params.sample_weights.size(); ++row) {
    auto probability = params.sample_weights[row] / weight_sum;
    if (probability == 0.0) {
      EXPECT_EQ(global_counts[row], 0) << "global row " << row;
      continue;
    }

    auto expected_count     = expected_total * probability;
    auto standard_deviation = std::sqrt(expected_total * probability * (1.0 - probability));
    auto tolerance          = std::max(3.0, 7.0 * standard_deviation);
    EXPECT_NEAR(static_cast<double>(global_counts[row]), expected_count, tolerance)
      << "global row " << row << " has weight " << params.sample_weights[row];
  }
}

std::vector<double> imbalanced_zero_root_weights()
{
  std::vector<double> weights(40, 0.0);
  for (std::size_t i = 30; i < weights.size(); ++i) {
    weights[i] = static_cast<double>(i - 29);
  }
  return weights;
}

std::vector<RowSamplerTestParams> row_sampler_inputs = {
  {"ThreeRowsContiguous", PartitionKind::Contiguous, {0.1, 0.4, 0.5}, 1024, 1.0f},
  {"UnequalStrided",
   PartitionKind::Strided,
   {0.0, 1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 89.0},
   256,
   0.75f},
  {"ZeroWeightRoot", PartitionKind::Imbalanced, imbalanced_zero_root_weights(), 128, 1.0f},
  {"EmptyNonRootRanks", PartitionKind::EmptyNonRootRanks, {0.1, 0.4, 0.5}, 512, 1.0f}};

INSTANTIATE_TEST_SUITE_P(RfRowSamplerTests,
                         RfRowSamplerTest,
                         ::testing::ValuesIn(row_sampler_inputs),
                         [](auto const& info) { return info.param.name; });

}  // namespace opg
}  // namespace Test
}  // namespace ML

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MLCommon::Test::opg::MPIEnvironment());
  return RUN_ALL_TESTS();
}
