/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_opg_utils.h"

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cuda_utils.cuh>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/stream>

#include <decisiontree/batched-levelalgo/quantiles.cuh>
#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

namespace ML {
namespace Test {
namespace opg {

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

template <typename T>
MPI_Datatype mpi_datatype()
{
  if constexpr (std::is_same_v<T, float>) {
    return MPI_FLOAT;
  } else {
    static_assert(std::is_same_v<T, double>, "Unsupported RF quantile test type");
    return MPI_DOUBLE;
  }
}

template <typename T>
std::vector<T> gather_global_column_major_data(
  std::vector<T> const& h_data, int n_rows, int n_cols, int rank, int size, int& global_rows)
{
  std::vector<int> rank_rows(size);
  MPI_Allgather(&n_rows, 1, MPI_INT, rank_rows.data(), 1, MPI_INT, MPI_COMM_WORLD);

  std::vector<int> rank_row_offsets(size + 1, 0);
  std::vector<int> value_counts(size);
  std::vector<int> value_offsets(size);
  for (int i = 0; i < size; ++i) {
    rank_row_offsets[i + 1] = rank_row_offsets[i] + rank_rows[i];
    value_counts[i]         = rank_rows[i] * n_cols;
    value_offsets[i]        = i == 0 ? 0 : value_offsets[i - 1] + value_counts[i - 1];
  }
  global_rows = rank_row_offsets[size];

  std::vector<T> gathered_by_rank(rank == 0 ? static_cast<std::size_t>(n_cols) * global_rows : 0);
  MPI_Gatherv(h_data.data(),
              static_cast<int>(h_data.size()),
              mpi_datatype<T>(),
              rank == 0 ? gathered_by_rank.data() : nullptr,
              value_counts.data(),
              value_offsets.data(),
              mpi_datatype<T>(),
              0,
              MPI_COMM_WORLD);

  if (rank != 0) { return {}; }

  std::vector<T> global_data(static_cast<std::size_t>(n_cols) * global_rows);
  for (int rank_idx = 0; rank_idx < size; ++rank_idx) {
    for (int col = 0; col < n_cols; ++col) {
      std::copy_n(gathered_by_rank.data() + value_offsets[rank_idx] +
                    static_cast<std::size_t>(col) * rank_rows[rank_idx],
                  rank_rows[rank_idx],
                  global_data.data() + static_cast<std::size_t>(col) * global_rows +
                    rank_row_offsets[rank_idx]);
    }
  }
  return global_data;
}

template <typename T>
class RfMgQuantileTest : public ::testing::Test {
 public:
  void SetUp() override
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

    constexpr int n_cols         = 3;
    constexpr int max_n_bins     = 16;
    constexpr int oversample     = 4;
    constexpr std::uint64_t seed = 123456789ULL;
    int n_rows                   = 257 + 17 * rank;

    std::vector<T> h_data(static_cast<std::size_t>(n_cols) * n_rows);
    for (int col = 0; col < n_cols; ++col) {
      for (int row = 0; row < n_rows; ++row) {
        h_data[static_cast<std::size_t>(col) * n_rows + row] =
          static_cast<T>(col * 100000 + rank * 1000 + row + 1);
      }
    }

    rmm::device_uvector<T> data(h_data.size(), handle.get_stream());
    raft::update_device(data.data(), h_data.data(), h_data.size(), handle.get_stream());

    auto quantile_result =
      DT::computeQuantiles(handle, data.data(), max_n_bins, n_rows, n_cols, oversample, seed);

    std::vector<int> h_n_bins(n_cols);
    std::vector<T> h_quantiles(static_cast<std::size_t>(n_cols) * max_n_bins);
    raft::update_host(
      h_n_bins.data(), quantile_result.n_bins_array.data(), n_cols, handle.get_stream());
    raft::update_host(h_quantiles.data(),
                      quantile_result.quantiles_array.data(),
                      h_quantiles.size(),
                      handle.get_stream());
    handle.sync_stream();

    int global_rows = 0;
    auto h_global_data =
      gather_global_column_major_data(h_data, n_rows, n_cols, rank, size, global_rows);
    std::vector<int> h_reference_n_bins(n_cols);
    std::vector<T> h_reference_quantiles(static_cast<std::size_t>(n_cols) * max_n_bins);

    if (rank == 0) {
      raft::handle_t reference_handle(cuda::stream_ref{cudaStreamPerThread}, stream_pool);
      rmm::device_uvector<T> reference_data(h_global_data.size(), reference_handle.get_stream());
      raft::update_device(reference_data.data(),
                          h_global_data.data(),
                          h_global_data.size(),
                          reference_handle.get_stream());
      auto reference_result = DT::computeQuantiles(
        reference_handle, reference_data.data(), max_n_bins, global_rows, n_cols, oversample, seed);
      raft::update_host(h_reference_n_bins.data(),
                        reference_result.n_bins_array.data(),
                        n_cols,
                        reference_handle.get_stream());
      raft::update_host(h_reference_quantiles.data(),
                        reference_result.quantiles_array.data(),
                        h_reference_quantiles.size(),
                        reference_handle.get_stream());
      reference_handle.sync_stream();
    }

    MPI_Bcast(h_reference_n_bins.data(), n_cols, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_reference_quantiles.data(),
              static_cast<int>(h_reference_quantiles.size()),
              mpi_datatype<T>(),
              0,
              MPI_COMM_WORLD);

    // Compare against a single-GPU reference so the test checks correctness, not just rank
    // agreement.
    for (int col = 0; col < n_cols; ++col) {
      ASSERT_GT(h_n_bins[col], 1);
      ASSERT_LE(h_n_bins[col], max_n_bins);
      ASSERT_EQ(h_n_bins[col], h_reference_n_bins[col]);
      for (int bin = 0; bin < h_n_bins[col]; ++bin) {
        EXPECT_EQ(h_quantiles[static_cast<std::size_t>(col) * max_n_bins + bin],
                  h_reference_quantiles[static_cast<std::size_t>(col) * max_n_bins + bin]);
      }
    }
  }
};

using RfMgQuantileTestF = RfMgQuantileTest<float>;
TEST_F(RfMgQuantileTestF, SharedAcrossRanks) {}

using RfMgQuantileTestD = RfMgQuantileTest<double>;
TEST_F(RfMgQuantileTestD, SharedAcrossRanks) {}

}  // namespace opg
}  // namespace Test
}  // namespace ML
