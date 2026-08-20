/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/comms.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/resource/comms.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cstddef>
#include <vector>

namespace ML::detail {

inline void cuml_rf_allreduce_validation_status(const raft::handle_t& handle,
                                                const int* local_status,
                                                int* global_status)
{
  auto const& comm    = raft::resource::get_comms(handle);
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  comm.allreduce(local_status, global_status, 1, raft::comms::op_t::MAX, stream);
  RAFT_EXPECTS(comm.sync_stream(stream) == raft::comms::status_t::SUCCESS,
               "Input validation status all-reduce failed");
}

inline void cuml_rf_allreduce_oob_stats(const raft::handle_t& handle,
                                        const double* local_stats,
                                        double* global_stats,
                                        std::size_t count)
{
  auto const& comm    = raft::resource::get_comms(handle);
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  comm.allreduce(local_stats, global_stats, count, raft::comms::op_t::SUM, stream);
  RAFT_EXPECTS(comm.sync_stream(stream) == raft::comms::status_t::SUCCESS,
               "OOB statistics all-reduce failed");
}

inline void cuml_rf_allgather_oob_predictions(const raft::handle_t& handle,
                                              const double* local_predictions,
                                              double* global_predictions,
                                              std::size_t local_num_rows,
                                              std::size_t num_outputs,
                                              std::size_t global_num_rows)
{
  auto const& comm    = raft::resource::get_comms(handle);
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto local_rows     = raft::make_device_scalar<std::size_t>(handle, local_num_rows);
  auto row_counts     = raft::make_device_vector<std::size_t>(handle, comm.get_size());

  comm.allgather(local_rows.data_handle(), row_counts.data_handle(), 1, stream);
  RAFT_EXPECTS(comm.sync_stream(stream) == raft::comms::status_t::SUCCESS,
               "OOB row count all-gather failed");

  std::vector<std::size_t> row_counts_host(comm.get_size());
  raft::copy(row_counts_host.data(), row_counts.data_handle(), comm.get_size(), stream);
  raft::resource::sync_stream(handle);

  std::vector<std::size_t> recv_counts(comm.get_size());
  std::vector<std::size_t> displacements(comm.get_size());
  std::size_t total_rows = 0;
  for (int rank = 0; rank < comm.get_size(); ++rank) {
    recv_counts[rank]   = row_counts_host[rank] * num_outputs;
    displacements[rank] = total_rows * num_outputs;
    total_rows += row_counts_host[rank];
  }
  RAFT_EXPECTS(total_rows == global_num_rows,
               "Global OOB prediction size does not match the distributed row count");

  comm.allgatherv(
    local_predictions, global_predictions, recv_counts.data(), displacements.data(), stream);
  RAFT_EXPECTS(comm.sync_stream(stream) == raft::comms::status_t::SUCCESS,
               "OOB predictions all-gather failed");
}

}  // namespace ML::detail
