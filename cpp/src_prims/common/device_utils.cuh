/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/util/cuda_utils.cuh>

#include <utility>  // pair

namespace MLCommon {

// TODO move to raft https://github.com/NVIDIA/raft/issues/90
/** helper method to get the compute capability version numbers */
inline std::pair<int, int> getDeviceCapability()
{
  int devId;
  RAFT_CUDA_TRY(cudaGetDevice(&devId));
  int major, minor;
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devId));
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devId));
  return std::make_pair(major, minor);
}

/**
 * @brief Batched warp-level sum reduction
 *
 * @tparam T        data type
 * @tparam NThreads Number of threads in the warp doing independent reductions
 *
 * @param[in] val input value
 * @return        for the first "group" of threads, the reduced value. All
 *                others will contain unusable values!
 *
 * @note Why not cub? Because cub doesn't seem to allow working with arbitrary
 *       number of warps in a block and also doesn't support this kind of
 *       batched reduction operation
 * @note All threads in the warp must enter this function together
 *
 * @todo Expand this to support arbitrary reduction ops
 */
template <typename T, int NThreads>
DI T batchedWarpReduce(T val)
{
#pragma unroll
  for (int i = NThreads; i < raft::WarpSize; i <<= 1) {
    val += raft::shfl(val, raft::laneId() + i);
  }
  return val;
}

/**
 * @brief 1-D block-level batched sum reduction
 *
 * @tparam T        data type
 * @tparam NThreads Number of threads in the warp doing independent reductions
 *
 * @param val  input value
 * @param smem shared memory region needed for storing intermediate results. It
 *             must at least be of size: `sizeof(T) * nWarps * NThreads`
 * @return     for the first "group" of threads in the block, the reduced value.
 *             All others will contain unusable values!
 *
 * @note Why not cub? Because cub doesn't seem to allow working with arbitrary
 *       number of warps in a block and also doesn't support this kind of
 *       batched reduction operation
 * @note All threads in the block must enter this function together
 *
 * @todo Expand this to support arbitrary reduction ops
 */
template <typename T, int NThreads>
DI T batchedBlockReduce(T val, char* smem)
{
  auto* sTemp                  = reinterpret_cast<T*>(smem);
  constexpr int nGroupsPerWarp = raft::WarpSize / NThreads;
  static_assert(raft::isPo2(nGroupsPerWarp), "nGroupsPerWarp must be a PO2!");
  const int nGroups = (blockDim.x + NThreads - 1) / NThreads;
  const int lid     = raft::laneId();
  const int lgid    = lid % NThreads;
  const int gid     = threadIdx.x / NThreads;
  const auto wrIdx  = (gid / nGroupsPerWarp) * NThreads + lgid;
  const auto rdIdx  = gid * NThreads + lgid;
  for (int i = nGroups; i > 0;) {
    auto iAligned = ((i + nGroupsPerWarp - 1) / nGroupsPerWarp) * nGroupsPerWarp;
    if (gid < iAligned) {
      val = batchedWarpReduce<T, NThreads>(val);
      if (lid < NThreads) sTemp[wrIdx] = val;
    }
    __syncthreads();
    i /= nGroupsPerWarp;
    if (i > 0) { val = gid < i ? sTemp[rdIdx] : T(0); }
    __syncthreads();
  }
  return val;
}

}  // namespace MLCommon
