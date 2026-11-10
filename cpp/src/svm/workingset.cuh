/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "smo_sets.cuh"
#include "workingset.h"
#include "ws_util.cuh"

#include <cuml/common/logger.hpp>
#include <cuml/svm/svm_parameter.h>

#include <raft/core/handle.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/init.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/permutation_iterator.h>

#include <algorithm>
#include <cstddef>
#include <limits>

namespace ML {
namespace SVM {

namespace {
//  placeholder function passed to configuration call to Cub::DeviceSelect
__device__ bool always_true(int) { return true; }
}  // end unnamed namespace

/**
 * Working set selection for the SMO algorithm.
 *
 * The working set is a subset of the training vectors, by default it has 1024 elements.
 * At every outer iteration in SmoSolver::Solve, we select a different working set, and
 * optimize the dual coefficients for the working set.
 *
 * The vectors are selected based on the f values, which is the difference between the
 * target label and the decision function value.
 */
template <typename math_t>
inline void WorkingSet<math_t>::SimpleSelect(
  math_t* f, math_t* alpha, math_t* y, const math_t* C, int n_already_selected)
{
  // We are not using the topK kernel, because of the additional lower/upper
  // constraint
  int n_needed = n_ws - n_already_selected;

  // Zero the priority of the elements that will be newly selected
  RAFT_CUDA_TRY(
    cudaMemsetAsync(ws_priority.data() + n_already_selected, 0, n_needed * sizeof(int), stream));

  cub::DeviceRadixSort::SortPairs((void*)cub_storage.data(),
                                  cub_bytes,
                                  f,
                                  f_sorted.data(),
                                  f_idx.data(),
                                  f_idx_sorted.data(),
                                  n_train,
                                  0,
                                  (int)8 * sizeof(math_t),
                                  stream);

  if (ML::default_logger().should_log(rapids_logger::level_enum::debug) && n_train < 20) {
    std::stringstream ss;
    raft::print_device_vector("idx_sorted", f_idx_sorted.data(), n_train, ss);
    CUML_LOG_DEBUG(ss.str().c_str());
  }
  // Select n_ws/2 elements from the upper set with the smallest f value
  bool* available = this->available.data();
  set_upper<<<raft::ceildiv(n_train, TPB), TPB, 0, stream>>>(available, n_train, alpha, y, C);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  n_already_selected += GatherAvailable(n_already_selected, n_needed / 2, true);

  // Select n_ws/2 elements from the lower set with the highest f values
  set_lower<<<raft::ceildiv(n_train, TPB), TPB, 0, stream>>>(available, n_train, alpha, y, C);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  n_already_selected += GatherAvailable(n_already_selected, n_ws - n_already_selected, false);

  // In case we could not find enough elements, then we just fill using the
  // still available elements.
  if (n_already_selected < n_ws) {
    CUML_LOG_DEBUG(
      "Could not fill working set: found only %d elements, "
      "filling remainder with additional elements",
      n_already_selected);
    RAFT_CUDA_TRY(cudaMemset(available, 1, sizeof(bool) * n_train));
    n_already_selected += GatherAvailable(n_already_selected, n_ws - n_already_selected, true);
  }
}

/**
 * @brief Select elements from the previous working set based on their priority.
 *
 * We sort the old working set based on their priority in ascending order,
 * and then select nc elements from free, and then lower/upper bound vectors.
 * For details see [2].
 *
 * See Issue #946.
 *
 * References:
 * [2] T Serafini, L Zanni: On the Working Set selection in grad. projection
 *     based decomposition techniques for Support Vector Machines
 *     DOI: 10.1080/10556780500140714
 *
 * @param [in] alpha device vector of dual coefficients, size [n_train]
 * @param [in] C_vec penalty parameter
 * @param [in] nc number of elements to select
 */
template <typename math_t>
int WorkingSet<math_t>::PrioritySelect(math_t* alpha, const math_t* C, int nc)
{
  int n_selected = 0;

  cub::DeviceRadixSort::SortPairs((void*)cub_storage.data(),
                                  cub_bytes,
                                  ws_priority.data(),
                                  ws_priority_sorted.data(),
                                  idx.data(),
                                  ws_idx_sorted.data(),
                                  n_ws,
                                  0,
                                  sizeof(int) * 8,
                                  stream);

  // Select first from free vectors (0<alpha<C)
  n_selected += SelectPrevWs(
    2 * nc, n_selected, [alpha, C] HD(int idx) { return 0 < alpha[idx] && alpha[idx] < C[idx]; });

  // then from lower bound (alpha=0)
  n_selected += SelectPrevWs(2 * nc, n_selected, [alpha] HD(int idx) { return alpha[idx] <= 0; });
  // and in the end from upper bound vectors (alpha=c)
  n_selected +=
    SelectPrevWs(2 * nc, n_selected, [alpha, C] HD(int idx) { return alpha[idx] >= C[idx]; });
  // we have now idx[0:n_selected] indices from the old working set
  // we need to update their priority.
  update_priority<<<raft::ceildiv(n_selected, TPB), TPB, 0, stream>>>(ws_priority.data(),
                                                                      n_selected,
                                                                      idx.data(),
                                                                      n_ws,
                                                                      ws_idx_sorted.data(),
                                                                      ws_priority_sorted.data());
  return n_selected;
}

template <typename math_t>
inline void WorkingSet<math_t>::AllocateBuffers()
{
  if (n_ws > 0) {
    f_idx.resize(n_train, stream);
    f_idx_sorted.resize(n_train, stream);
    idx_tmp.resize(n_train, stream);
    f_sorted.resize(n_train, stream);
    available.resize(n_train, stream);
    available_sorted.resize(n_train, stream);

    idx.resize(n_ws, stream);  // allocate(idx, n_ws, stream);
    ws_idx_sorted.resize(n_ws, stream);
    ws_idx_save.resize(n_ws, stream);
    ws_idx_selected.resize(n_ws, stream);
    ws_priority.resize(n_ws, stream);
    ws_priority_sorted.resize(n_ws, stream);

    // Determine temporary device storage requirements for cub
    std::size_t cub_bytes2 = 0;
    cub::DeviceRadixSort::SortPairs(NULL,
                                    cub_bytes,
                                    f_sorted.data(),
                                    f_sorted.data(),
                                    f_idx.data(),
                                    f_idx_sorted.data(),
                                    n_train,
                                    0,
                                    8 * sizeof(math_t),
                                    stream);
    cub::DeviceSelect::If(NULL,
                          cub_bytes2,
                          f_idx.data(),
                          f_idx.data(),
                          d_num_selected.data(),
                          n_train,
                          always_true,
                          stream);
    cub_bytes = std::max(cub_bytes, cub_bytes2);
    cub_storage.resize(cub_bytes, stream);
    Initialize();
  }
}

/**
 * @brief Gather available elements from the working set.
 *
 * We select the first (last) n_needed element from the front (end) of
 * f_idx_sorted. We ignore the elements that are already selected, and those
 * where this->available is false.
 *
 * @param n_already_selected number of element already selected (their indices
 *   are stored in idx[0:n_already_selected])
 * @param n_needed number of elements to be selected
 * @param copy_front if true, then copy the elements from the front of the
 *        selected list, otherwise copy from the end of the list
 * @return the number of elements copied (which might be less than n_needed)
 */
template <typename math_t>
inline int WorkingSet<math_t>::GatherAvailable(int n_already_selected,
                                               int n_needed,
                                               bool copy_front)
{
  // First we update the mask to ignores already selected elements
  bool* available = this->available.data();
  if (n_already_selected > 0) {
    set_unavailable<<<raft::ceildiv(n_train, TPB), TPB, 0, stream>>>(
      available, n_train, idx.data(), n_already_selected);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
  if (ML::default_logger().should_log(rapids_logger::level_enum::debug) && n_train < 20) {
    std::stringstream ss;
    raft::print_device_vector("avail", available, n_train, ss);
    CUML_LOG_DEBUG(ss.str().c_str());
  }

  // Map the mask to the sorted indices
  thrust::device_ptr<bool> av_ptr(available);
  thrust::device_ptr<bool> av_sorted_ptr(available_sorted.data());
  thrust::device_ptr<int> idx_ptr(f_idx_sorted.data());
  thrust::copy(thrust::cuda::par.on(stream),
               thrust::make_permutation_iterator(av_ptr, idx_ptr),
               thrust::make_permutation_iterator(av_ptr, idx_ptr + n_train),
               av_sorted_ptr);
  if (ML::default_logger().should_log(rapids_logger::level_enum::debug) && n_train < 20) {
    std::stringstream ss;
    raft::print_device_vector("avail_sorted", available_sorted.data(), n_train, ss);
    CUML_LOG_DEBUG(ss.str().c_str());
  }

  // Select the available elements
  cub::DeviceSelect::Flagged((void*)cub_storage.data(),
                             cub_bytes,
                             f_idx_sorted.data(),
                             available_sorted.data(),
                             idx_tmp.data(),
                             d_num_selected.data(),
                             n_train,
                             stream);
  int n_selected = d_num_selected.value(stream);
  handle.sync_stream(stream);

  // Copy to output
  int n_copy = n_selected > n_needed ? n_needed : n_selected;
  if (copy_front) {
    raft::copy(idx.data() + n_already_selected, idx_tmp.data(), n_copy, stream);
  } else {
    raft::copy(
      idx.data() + n_already_selected, idx_tmp.data() + n_selected - n_copy, n_copy, stream);
  }
  if (ML::default_logger().should_log(rapids_logger::level_enum::debug) && n_train < 20) {
    std::stringstream ss;
    raft::print_device_vector("selected", idx.data(), n_already_selected + n_copy, ss);
    CUML_LOG_DEBUG(ss.str().c_str());
  }
  return n_copy;
}
template <typename math_t>
inline void WorkingSet<math_t>::Initialize()
{
  raft::linalg::range(f_idx.data(), n_train, stream);
  raft::linalg::range(idx.data(), n_ws, stream);
}

/**
 * @brief Select the first n_needed elements from ws_idx_sorted where op is true.
 *
 * The selected elements are appended to this->idx.
 *
 * @param n_needed number of elements that should be selected
 * @param n_already_selected number of already selected elements
 * @param op selection condition
 * @return the number of elements selected
 */
template <typename math_t>
template <typename select_op>
inline int WorkingSet<math_t>::SelectPrevWs(int n_needed, int n_already_selected, select_op op)
{
  n_needed -= n_already_selected;
  if (n_needed <= 0) { return 0; }
  cub::DeviceSelect::If(cub_storage.data(),
                        cub_bytes,
                        ws_idx_sorted.data(),
                        ws_idx_selected.data(),
                        d_num_selected.data(),
                        n_ws,
                        op,
                        stream);
  int n_selected = d_num_selected.value(stream);
  handle.sync_stream(stream);
  int n_copy = n_selected < n_needed ? n_selected : n_needed;
  raft::copy(idx.data() + n_already_selected, ws_idx_selected.data(), n_copy, stream);
  return n_copy;
}

}  // end namespace SVM
}  // end namespace ML
