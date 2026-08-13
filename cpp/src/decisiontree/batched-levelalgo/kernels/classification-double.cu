/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "builder_kernels_impl.cuh"

#include <cuml/tree/flatnode.h>

namespace ML {
namespace DT {
using DataT      = double;
using LabelT     = int;
using ObjectiveT = ClassificationObjectiveFunction<DataT, LabelT>;
using BinT       = typename ObjectiveT::BinT;
using DatasetT   = Dataset<DataT, LabelT>;
using NodeT      = SparseTreeNode<DataT, LabelT>;

// Explicit instantiations are split across separate .cu files to increase compilation parallelism.
template void launchBuildLeafHistogramsKernel<DatasetT, NodeT, ObjectiveT>(
  ObjectiveT objective,
  DatasetT& dataset,
  const NodeT* tree,
  const InstanceRange* instance_ranges,
  BinT* leaf_histograms,
  int batch_size,
  size_t smem_size,
  cudaStream_t builder_stream);

template void launchFinalizeLeafKernel<ObjectiveT, DataT>(const BinT* leaf_histograms,
                                                          DataT* leaves,
                                                          int num_outputs,
                                                          int batch_size,
                                                          cudaStream_t builder_stream);

// Explicit instantiations are split across separate .cu files to increase compilation parallelism.
template void launchBuildHistogramsKernel<DataT, LabelT, TPB_DEFAULT, ObjectiveT>(
  BinT* histograms,
  std::int64_t n_bins,
  const DatasetT& dataset,
  const Quantiles<DataT>& quantiles,
  const NodeWorkItem* work_items,
  std::int64_t colStart,
  const std::int64_t* column_samples,
  ObjectiveT& objective,
  const WorkloadInfo* workload_info,
  dim3 histogram_grid,
  const SharedMemoryConfig& split_smem_config,
  cudaStream_t builder_stream);

// Explicit instantiations are split across separate .cu files to increase compilation parallelism.
template void launchFindBestSplitsKernel<DataT, LabelT, TPB_DEFAULT, ObjectiveT>(
  BinT* histograms,
  std::int64_t n_bins,
  const DatasetT& dataset,
  const Quantiles<DataT>& quantiles,
  std::int64_t colStart,
  const std::int64_t* column_samples,
  int* mutex,
  volatile Split<DataT>* splits,
  ObjectiveT& objective,
  dim3 split_grid,
  cudaStream_t builder_stream);

}  // namespace DT
}  // namespace ML
