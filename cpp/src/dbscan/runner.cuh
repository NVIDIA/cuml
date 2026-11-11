/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "adjgraph/runner.cuh"
#include "corepoints/compute.cuh"
#include "corepoints/exchange.cuh"
#include "mergelabels/runner.cuh"
#include "mergelabels/tree_reduction.cuh"
#include "vertexdeg/runner.cuh"

#include <common/nvtx.hpp>

#include <cuml/cluster/dbscan.hpp>
#include <cuml/common/distance_type.hpp>
#include <cuml/common/logger.hpp>
#include <cuml/common/utils.hpp>

#include <raft/core/nvtx.hpp>
#include <raft/label/classlabels.cuh>
#include <raft/sparse/csr.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuda/functional>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>

#include <cuvs/neighbors/ball_cover.hpp>

#include <cstddef>
#include <memory>

namespace ML {
namespace Dbscan {

static const int TPB = 256;

/**
 * Adjust labels from weak_cc primitive to match sklearn:
 * 1. Turn any labels matching MAX_LABEL into -1
 * 2. Subtract 1 from all other labels.
 */
template <typename Index_ = int>
CUML_KERNEL void relabelForSkl(Index_* labels, Index_ N, Index_ MAX_LABEL)
{
  Index_ tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < N) {
    if (labels[tid] == MAX_LABEL) {
      labels[tid] = -1;
    } else {
      --labels[tid];
    }
  }
}

/**
 * Turn the non-monotonic labels from weak_cc primitive into
 * an array of labels drawn from a monotonically increasing set.
 */
template <typename Index_ = int>
void final_relabel(Index_* db_cluster, Index_ N, cudaStream_t stream)
{
  Index_ MAX_LABEL = std::numeric_limits<Index_>::max();
  raft::label::make_monotonic(
    db_cluster, db_cluster, N, stream, [MAX_LABEL] __device__(Index_ val) {
      return val == MAX_LABEL;
    });
}

/**
 * Run the DBSCAN algorithm (common code for single-GPU and multi-GPU)
 * @tparam opg Whether we are running in a multi-node multi-GPU context
 * @param[in]  handle        raft handle
 * @param[in]  x             Input data (N*D row-major device array, or N*N for precomputed)
 * @param[in]  N             Number of points
 * @param[in]  D             Dimensionality of the points
 * @param[in]  start_row     Index of the offset for this node
 * @param[in]  n_owned_rows  Number of rows (points) owned by this node
 * @param[in]  eps           Epsilon neighborhood criterion
 * @param[in]  min_pts       Core points criterion
 * @param[out] labels        Output labels (device array of length N)
 * @param[out] core_indices  If not nullptr, the indices of core points are written in this array
 * @param[in]  algo_vd       Algorithm used for the vertex degrees
 * @param[in]  algo_adj      Algorithm used for the adjacency graph
 * @param[in]  algo_ccl      Algorithm used for the final relabel
 * @param[in]  workspace     Temporary global memory buffer used to store intermediate computations
 *                           If nullptr, then this function will return the workspace size needed.
 *                           It is the responsibility of the user to allocate and free this buffer!
 * @param[in]  batch_size    Batch size
 * @param[in]  eps_nn_method Determines method for computing epsilon neighborhood.
 * @param[in]  stream        The CUDA stream where to launch the kernels
 * @return In case the workspace pointer is null, this returns the size needed.
 */
template <typename Type_f, typename Index_ = int, bool opg = false>
std::size_t run(const raft::handle_t& handle,
                const Type_f* x,
                Index_ N,
                Index_ D,
                Index_ start_row,
                Index_ n_owned_rows,
                Type_f eps,
                Index_ min_pts,
                Index_* labels,
                Index_* core_indices,
                const Type_f* sample_weight,
                int algo_vd,
                int algo_adj,
                int algo_ccl,
                void* workspace,
                std::size_t batch_size,
                EpsNnMethod eps_nn_method,
                cudaStream_t stream,
                ML::distance::DistanceType metric)
{
  const std::size_t align = 256;
  Index_ n_batches        = raft::ceildiv((std::size_t)n_owned_rows, batch_size);

  int my_rank;
  if (opg) {
    const auto& comm = handle.get_comms();
    my_rank          = comm.get_rank();
  } else
    my_rank = 0;

  // switch compute mode based on feature dimension
  bool sparse_rbc_mode = eps_nn_method == EpsNnMethod::RBC;

  if constexpr (std::is_same_v<Type_f, double> || std::is_same_v<Index_, int32_t>) {
    if (sparse_rbc_mode) {
      sparse_rbc_mode = false;
      CUML_LOG_WARN(
        "RBC does not support double precision or int32 labels. Falling back to BRUTE_FORCE "
        "strategy.");
    }
  }

  if (sparse_rbc_mode && metric != ML::distance::DistanceType::L2SqrtExpanded &&
      metric != ML::distance::DistanceType::L2SqrtUnexpanded) {
    CUML_LOG_WARN("Metric not supported by RBC yet. Falling back to BRUTE_FORCE strategy.");
    sparse_rbc_mode = false;
  }

  /**
   * Note on coupling between data types:
   * - adjacency graph has a worst case size of N * batch_size elements. Thus,
   * if N is very close to being greater than the maximum 32-bit IdxType type used, a
   * 64-bit IdxType should probably be used instead.
   * - exclusive scan is the CSR row index for the adjacency graph and its values have a
   * risk of overflowing when N * batch_size becomes larger what can be stored in IdxType
   * - the vertex degree array has a worst case of each element having all other
   * elements in their neighborhood, so any IdxType can be safely used, so long as N doesn't
   * overflow.
   */
  std::size_t adj_size      = raft::alignTo<std::size_t>(sizeof(bool) * N * batch_size, align);
  std::size_t core_pts_size = raft::alignTo<std::size_t>(sizeof(bool) * N, align);
  std::size_t m_size        = raft::alignTo<std::size_t>(sizeof(bool), align);
  std::size_t vd_size       = raft::alignTo<std::size_t>(sizeof(Index_) * (batch_size + 1), align);
  std::size_t ex_scan_size  = raft::alignTo<std::size_t>(sizeof(Index_) * (batch_size + 1), align);
  std::size_t row_cnt_size  = raft::alignTo<std::size_t>(sizeof(Index_) * batch_size, align);
  std::size_t labels_size   = raft::alignTo<std::size_t>(sizeof(Index_) * N, align);
  std::size_t wght_sum_size =
    sample_weight != nullptr ? raft::alignTo<std::size_t>(sizeof(Type_f) * batch_size, align) : 0;

  Index_ MAX_LABEL = std::numeric_limits<Index_>::max();

  ASSERT(N * batch_size < static_cast<std::size_t>(MAX_LABEL),
         "An overflow occurred with the current choice of precision "
         "and the number of samples. (Max allowed batch size is %ld, but was %ld). "
         "Consider using double precision for the output labels.",
         (unsigned long)(MAX_LABEL / N),
         (unsigned long)batch_size);

  if (workspace == NULL) {
    auto size = adj_size + core_pts_size + m_size + vd_size + ex_scan_size + row_cnt_size +
                2 * labels_size + wght_sum_size;
    return size;
  }

  if (sparse_rbc_mode && (std::size_t)D > static_cast<std::size_t>(MAX_LABEL / N)) {
    CUML_LOG_WARN(
      "You are using an index type of size (%d bytes) which is not sufficient "
      "to represent the input dimensions in the RBC index. "
      "Consider using a larger index type. Falling back to BRUTE_FORCE strategy.",
      (int)sizeof(Index_));
    sparse_rbc_mode = false;
  }

  // partition the temporary workspace needed for different stages of dbscan.

  std::vector<Index_> batchadjlen(n_batches);
  std::vector<Index_> maxklen(n_batches);
  char* temp = (char*)workspace;
  bool* adj  = (bool*)temp;
  temp += adj_size;
  bool* core_pts = (bool*)temp;
  temp += core_pts_size;
  bool* m = (bool*)temp;
  temp += m_size;
  Index_* vd = (Index_*)temp;
  temp += vd_size;
  Index_* ex_scan = (Index_*)temp;
  temp += ex_scan_size;
  Index_* row_counters = (Index_*)temp;
  temp += row_cnt_size;
  Index_* labels_temp = (Index_*)temp;
  temp += labels_size;
  Index_* work_buffer = (Index_*)temp;
  temp += labels_size;
  Type_f* wght_sum = nullptr;
  if (sample_weight != nullptr) {
    wght_sum = (Type_f*)temp;
    temp += wght_sum_size;
  }

  rmm::device_uvector<Index_> adj_graph(0, stream);

  // build index for rbc
  std::unique_ptr<cuvs::neighbors::ball_cover::index<int64_t, float>> rbc_index;
  void* rbc_index_ptr = nullptr;
  if (sparse_rbc_mode) {
    if constexpr (std::is_same_v<Type_f, float> && std::is_same_v<Index_, int64_t>) {
      auto x_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(x, N, D);
      rbc_index   = std::make_unique<cuvs::neighbors::ball_cover::index<int64_t, float>>(
        handle, x_view, static_cast<cuvs::distance::DistanceType>(metric));
      cuvs::neighbors::ball_cover::build(handle, *rbc_index);
      rbc_index_ptr = rbc_index.get();
    }
  }

  // Compute the mask
  // 1. Compute the part owned by this worker (reversed order of batches to
  // keep the batch 0 in memory)
  for (int i = n_batches - 1; i >= 0; i--) {
    Index_ start_vertex_id = start_row + i * batch_size;
    Index_ n_points        = min(n_owned_rows - i * batch_size, batch_size);

    CUML_LOG_DEBUG(
      "- Batch %d / %ld (%ld samples)", i + 1, (unsigned long)n_batches, (unsigned long)n_points);

    CUML_LOG_DEBUG("--> Computing vertex degrees");
    raft::common::nvtx::push_range("Trace::Dbscan::VertexDeg");

    bool need_ja_compute = sparse_rbc_mode && ((i == 0) || (sample_weight != nullptr));
    VertexDeg::run<Type_f, Index_>(handle,
                                   rbc_index_ptr,
                                   ex_scan,
                                   need_ja_compute ? &adj_graph : nullptr,
                                   0,
                                   adj,
                                   vd,
                                   wght_sum,
                                   x,
                                   sample_weight,
                                   eps,
                                   N,
                                   D,
                                   algo_vd,
                                   start_vertex_id,
                                   n_points,
                                   stream,
                                   metric);

    if (sparse_rbc_mode && adj_graph.size() > 0) {
      batchadjlen.at(i) = adj_graph.size();
    } else {
      Index_ curradjlen = 0;
      raft::update_host(&curradjlen, vd + n_points, 1, stream);
      handle.sync_stream(stream);
      batchadjlen.at(i) = curradjlen;
    }

    // check for maximum row-length (number of connections) in batch
    // if sufficiently small we can compute neighbors in one pass later
    if (sparse_rbc_mode) {
      Index_ max_k =
        thrust::reduce(handle.get_thrust_policy(), vd, vd + n_points, (Index_)0, cuda::maximum{});
      CUML_LOG_DEBUG(
        "Adjacency matrix (batch %d) maximum row length %ld.", i, (unsigned long)max_k);
      maxklen.at(i) = max_k;
    }

    raft::common::nvtx::pop_range();

    CUML_LOG_DEBUG("--> Computing core point mask");
    raft::common::nvtx::push_range("Trace::Dbscan::CorePoints");
    if (wght_sum != nullptr) {
      CorePoints::compute<Type_f, Index_>(
        handle, wght_sum, core_pts, min_pts, start_vertex_id, n_points, stream);
    } else {
      CorePoints::compute<Index_, Index_>(
        handle, vd, core_pts, min_pts, start_vertex_id, n_points, stream);
    }
    raft::common::nvtx::pop_range();
  }
  // 2. Exchange with the other workers
  if (opg) CorePoints::exchange(handle, core_pts, N, start_row, stream);

  // Compute the labelling for the owned part of the graph
  raft::sparse::WeakCCState state(m);

  Index_ maxadjlen = *std::max_element(batchadjlen.begin(), batchadjlen.end());
  CUML_LOG_DEBUG("Allocating for %ld elements", (unsigned long)maxadjlen);
  adj_graph.resize(maxadjlen, stream);

  for (int i = 0; i < n_batches; i++) {
    Index_ start_vertex_id = start_row + i * batch_size;
    Index_ n_points        = min(n_owned_rows - i * batch_size, batch_size);
    if (n_points <= 0) break;

    CUML_LOG_DEBUG(
      "- Batch %d / %ld (%ld samples)", i + 1, (unsigned long)n_batches, (unsigned long)n_points);

    // i==0 -> adj and vd for batch 0 already in memory
    if (i > 0) {
      CUML_LOG_DEBUG("--> Computing vertex degrees");
      raft::common::nvtx::push_range("Trace::Dbscan::VertexDeg");
      VertexDeg::run<Type_f, Index_>(handle,
                                     rbc_index_ptr,
                                     ex_scan,
                                     &adj_graph,
                                     maxklen.at(i),
                                     adj,
                                     sparse_rbc_mode ? nullptr : vd,
                                     nullptr,
                                     x,
                                     nullptr,
                                     eps,
                                     N,
                                     D,
                                     algo_vd,
                                     start_vertex_id,
                                     n_points,
                                     stream,
                                     metric);
      raft::common::nvtx::pop_range();
    }

    if (!sparse_rbc_mode) {
      CUML_LOG_DEBUG("--> Computing adjacency graph with %ld nnz.",
                     (unsigned long)batchadjlen.at(i));
      raft::common::nvtx::push_range("Trace::Dbscan::AdjGraph");
      handle.sync_stream(stream);
      AdjGraph::run<Index_>(handle,
                            adj,
                            vd,
                            adj_graph.data(),
                            batchadjlen.at(i),
                            ex_scan,
                            N,
                            algo_adj,
                            n_points,
                            row_counters,
                            stream);

      raft::common::nvtx::pop_range();
    }

    CUML_LOG_DEBUG("--> Computing connected components");
    raft::common::nvtx::push_range("Trace::Dbscan::WeakCC");
    raft::sparse::weak_cc_batched<Index_>(
      i == 0 ? labels : labels_temp,
      ex_scan,
      adj_graph.data(),
      batchadjlen.at(i),
      N,
      start_vertex_id,
      n_points,
      &state,
      stream,
      [core_pts, N] __device__(Index_ global_id) {
        return global_id < N ? __ldg((char*)core_pts + global_id) : 0;
      });
    raft::common::nvtx::pop_range();

    if (i > 0) {
      // The labels_temp array contains the labelling for the neighborhood
      // graph of the current batch. This needs to be merged with the labelling
      // created by the previous batches.
      // Using the labelling from the previous batches as initial value for
      // weak_cc_batched and skipping the merge step would lead to incorrect
      // results as described in #3094.
      CUML_LOG_DEBUG("--> Accumulating labels");
      raft::common::nvtx::push_range("Trace::Dbscan::MergeLabels");
      MergeLabels::run<Index_>(handle, labels, labels_temp, core_pts, work_buffer, m, N, stream);
      raft::common::nvtx::pop_range();
    }
  }

  // Combine the results in the multi-node multi-GPU case
  if (opg)
    MergeLabels::tree_reduction(handle, labels, labels_temp, core_pts, work_buffer, m, N, stream);

  /// TODO: optional minimalization step for border points

  // Final relabel
  if (my_rank == 0) {
    raft::common::nvtx::push_range("Trace::Dbscan::FinalRelabel");
    if (algo_ccl == 2) final_relabel(labels, N, stream);
    std::size_t nblks = raft::ceildiv<std::size_t>(N, TPB);
    relabelForSkl<Index_><<<nblks, TPB, 0, stream>>>(labels, N, MAX_LABEL);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    raft::common::nvtx::pop_range();

    // Calculate the core_indices only if an array was passed in
    if (core_indices != nullptr) {
      raft::common::nvtx::range fun_scope("Trace::Dbscan::CoreSampleIndices");

      // Create the execution policy
      auto thrust_exec_policy = handle.get_thrust_policy();

      // Get wrappers for the device ptrs
      thrust::device_ptr<bool> dev_core_pts       = thrust::device_pointer_cast(core_pts);
      thrust::device_ptr<Index_> dev_core_indices = thrust::device_pointer_cast(core_indices);

      // First fill the core_indices with -1 which will be used if core_point_count < N
      thrust::fill_n(thrust_exec_policy, dev_core_indices, N, (Index_)-1);

      auto index_iterator = thrust::counting_iterator<Index_>(0);

      // Perform stream reduction on the core points. The core_pts acts as the stencil and we use
      // thrust::counting_iterator to return the index
      thrust::copy_if(thrust_exec_policy,
                      index_iterator,
                      index_iterator + N,
                      dev_core_pts,
                      dev_core_indices,
                      [=] __device__(const bool is_core_point) { return is_core_point; });
    }
  }

  CUML_LOG_DEBUG("Done.");
  return (std::size_t)0;
}
}  // namespace Dbscan
}  // namespace ML
