/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/util/cuda_utils.cuh>

#include <gtest/gtest.h>
#include <mpi.h>

namespace MLCommon {
namespace Test {
namespace opg {

/**
 * @brief This process's rank among the MPI ranks on its node, and how many of them there are.
 *
 * Ranks that share a node also share their device visibility, so the device index has to be
 * node-local: with one rank per one-GPU node, every rank has global rank != 0 but each process
 * still sees a single device.
 */
inline void get_node_local_rank_size(int& local_rank, int& local_size)
{
  MPI_Comm node_comm{};
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
  MPI_Comm_rank(node_comm, &local_rank);
  MPI_Comm_size(node_comm, &local_size);
  MPI_Comm_free(&node_comm);
}

/**
 * @brief Initialise MPI if this process has not already done so.
 */
inline void initialize_mpi_once()
{
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) { MPI_Init(nullptr, nullptr); }
}

/**
 *
 * @brief Testing environment to handle googletest runs
 * @note Inspired from:
 * http://www.parresianz.com/mpi/c++/mpi-unit-testing-googletests-cmake/
 */
class MPIEnvironment : public ::testing::Environment {
 public:
  void SetUp()
  {
    MPI_Init(NULL, NULL);

    int local_rank, local_size;
    get_node_local_rank_size(local_rank, local_size);

    int nGpus;
    RAFT_CUDA_TRY(cudaGetDeviceCount(&nGpus));

    // A rank that leaves SetUp while the others enter a collective leaves the whole job
    // hanging, so every rank has to agree on whether the topology can be used.
    int insufficient_local_gpus = nGpus < local_size ? 1 : 0;
    int insufficient_gpus       = 0;
    MPI_Allreduce(
      &insufficient_local_gpus, &insufficient_gpus, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    ASSERT(!insufficient_gpus,
           "Number of GPUs are lesser than local MPI ranks! ngpus=%d, local_ranks=%d",
           nGpus,
           local_size);

    RAFT_CUDA_TRY(cudaSetDevice(local_rank));
  }

  void TearDown() { MPI_Finalize(); }
};

};  // end namespace opg
};  // end namespace Test
};  // end namespace MLCommon
