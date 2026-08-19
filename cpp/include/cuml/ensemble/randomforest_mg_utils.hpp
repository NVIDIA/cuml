/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/resource/comms.hpp>
#include <raft/core/resource/cuda_stream.hpp>

namespace ML {

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

}  // namespace ML
