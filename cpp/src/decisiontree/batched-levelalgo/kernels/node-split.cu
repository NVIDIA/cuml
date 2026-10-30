/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../dataset.h"  // for Dataset
#include "builder_kernels_impl.cuh"

namespace ML {
namespace DT {

// Explicit instantiations are split across separate .cu files to increase compilation parallelism.
template void launchNodeSplitKernel<float, int, TPB_DEFAULT>(const Dataset<float, int>& dataset,
                                                             const NodeWorkItem* work_items,
                                                             Split<float>* splits,
                                                             const WorkloadInfo* workload_info,
                                                             size_t n_blocks_dimx,
                                                             size_t n_work_items,
                                                             std::int64_t* partition_row_ids,
                                                             cudaStream_t builder_stream);

// Explicit instantiations are split across separate .cu files to increase compilation parallelism.
template void launchNodeSplitKernel<double, int, TPB_DEFAULT>(const Dataset<double, int>& dataset,
                                                              const NodeWorkItem* work_items,
                                                              Split<double>* splits,
                                                              const WorkloadInfo* workload_info,
                                                              size_t n_blocks_dimx,
                                                              size_t n_work_items,
                                                              std::int64_t* partition_row_ids,
                                                              cudaStream_t builder_stream);

// Explicit instantiations are split across separate .cu files to increase compilation parallelism.
template void launchNodeSplitKernel<float, float, TPB_DEFAULT>(const Dataset<float, float>& dataset,
                                                               const NodeWorkItem* work_items,
                                                               Split<float>* splits,
                                                               const WorkloadInfo* workload_info,
                                                               size_t n_blocks_dimx,
                                                               size_t n_work_items,
                                                               std::int64_t* partition_row_ids,
                                                               cudaStream_t builder_stream);

// Explicit instantiations are split across separate .cu files to increase compilation parallelism.
template void launchNodeSplitKernel<double, double, TPB_DEFAULT>(
  const Dataset<double, double>& dataset,
  const NodeWorkItem* work_items,
  Split<double>* splits,
  const WorkloadInfo* workload_info,
  size_t n_blocks_dimx,
  size_t n_work_items,
  std::int64_t* partition_row_ids,
  cudaStream_t builder_stream);

}  // namespace DT
}  // namespace ML
