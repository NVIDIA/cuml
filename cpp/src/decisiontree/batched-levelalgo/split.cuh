/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/linalg/unary_op.cuh>
#include <raft/util/cuda_utils.cuh>

#include <cstdint>
#include <limits>  // for numeric_limits

namespace ML {
namespace DT {
namespace detail {

template <typename BinT>
DI std::int64_t CountLeft(BinT const* hist,
                          std::int64_t i,
                          std::int64_t n_bins,
                          std::int64_t n_outputs)
{
  auto nLeft = hist[i].Count();
  for (std::int64_t j = 1; j < n_outputs; ++j) {
    nLeft += hist[n_bins * j + i].Count();
  }
  return static_cast<std::int64_t>(nLeft);
}

}  // namespace detail

/**
 * @brief All info pertaining to splitting a node
 *
 * @tparam DataT input data type
 */
template <typename DataT>
struct Split {
  typedef Split<DataT> SplitT;
  /** start with this as the initial gain */
  static constexpr DataT Min = -std::numeric_limits<DataT>::max();

  /** threshold to compare in this node */
  DataT quesval;
  /** feature index */
  std::int64_t colid;
  /** best info gain on this node */
  DataT best_metric_val;
  /** global number of samples in the left child */
  std::int64_t global_nLeft;
  /** rank-local number of samples in the left child */
  std::int64_t local_nLeft;
  /** first quantile index in an inclusive range of training-equivalent splits */
  std::int64_t split_start;
  /** last quantile index in an inclusive range of training-equivalent splits */
  std::int64_t split_end;

  HDI Split()
  {
    quesval = best_metric_val = Min;
    colid                     = -1;
    global_nLeft              = 0;
    local_nLeft               = 0;
    split_start               = -1;
    split_end                 = -1;
  }

  HDI SplitT& operator=(const SplitT& other)
  {
    quesval         = other.quesval;
    colid           = other.colid;
    best_metric_val = other.best_metric_val;
    global_nLeft    = other.global_nLeft;
    local_nLeft     = other.local_nLeft;
    split_start     = other.split_start;
    split_end       = other.split_end;
    return *this;
  }

  HDI bool IsValid() const { return colid != static_cast<std::int64_t>(-1); }

  DI bool has_valid_split_range() const
  {
    return split_start >= std::int64_t{0} && split_end >= split_start;
  }

  DI bool can_merge_equivalent_split_range(std::int64_t other_global_nLeft,
                                           std::int64_t other_split_start,
                                           std::int64_t other_split_end) const
  {
    return global_nLeft == other_global_nLeft && has_valid_split_range() &&
           other_split_start >= std::int64_t{0} && other_split_end >= other_split_start;
  }

  // Extend the candidate's inclusive range of training-equivalent split
  // thresholds. `quesval` tracks the upper edge only to preserve the existing
  // threshold tie-break against candidates outside this equivalent range.
  DI void merge_equivalent_split_range(DataT other_quesval,
                                       std::int64_t other_split_start,
                                       std::int64_t other_split_end)
  {
    split_start = other_split_start < split_start ? other_split_start : split_start;
    split_end   = other_split_end > split_end ? other_split_end : split_end;
    if (other_quesval > quesval) { quesval = other_quesval; }
  }

  DI bool replace_with(DataT other_quesval,
                       std::int64_t other_colid,
                       DataT other_best_metric_val,
                       std::int64_t other_global_nLeft,
                       std::int64_t other_split_start,
                       std::int64_t other_split_end)
  {
    quesval         = other_quesval;
    colid           = other_colid;
    best_metric_val = other_best_metric_val;
    global_nLeft    = other_global_nLeft;
    local_nLeft     = 0;
    split_start     = other_split_start;
    split_end       = other_split_end;
    return true;
  }

  // Several thresholds can be equally good for the training data while still
  // routing future inference values differently. Select the middle split in
  // that equivalent range so deterministic tie-breaking does not pick an edge.
  DI void select_split_range_midpoint(DataT const* quantiles, std::int64_t n_bins)
  {
    if (has_valid_split_range() && split_end < n_bins) {
      auto bin    = split_start + (split_end - split_start + 1) / 2;
      quesval     = quantiles[bin];
      split_start = bin;
      split_end   = bin;
    }
  }

  /**
   * @brief updates the current split if the input gain is better
   *
   * local_nLeft is intentionally not accepted here: split selection is based
   * on global counts, and local counts are filled just before partitioning.
   */
  DI bool update(DataT other_quesval,
                 std::int64_t other_colid,
                 DataT other_best_metric_val,
                 std::int64_t other_global_nLeft,
                 std::int64_t other_split_start,
                 std::int64_t other_split_end)
  {
    // Primary ordering: higher gain wins; lower or unordered gain loses.
    if (other_best_metric_val > best_metric_val) {
      return replace_with(other_quesval,
                          other_colid,
                          other_best_metric_val,
                          other_global_nLeft,
                          other_split_start,
                          other_split_end);
    }
    if (other_best_metric_val != best_metric_val) { return false; }

    // Equal gain: preserve the existing deterministic feature tie-break.
    if (other_colid > colid) {
      return replace_with(other_quesval,
                          other_colid,
                          other_best_metric_val,
                          other_global_nLeft,
                          other_split_start,
                          other_split_end);
    }
    if (other_colid != colid) { return false; }

    // Equal gain and feature: multiple thresholds can send the same training
    // rows left and right. Merge that range and choose its representative after
    // reduction.
    if (can_merge_equivalent_split_range(other_global_nLeft, other_split_start, other_split_end)) {
      merge_equivalent_split_range(other_quesval, other_split_start, other_split_end);
      return true;
    }

    // Equal gain and feature, but a different training partition: keep the
    // existing deterministic threshold tie-break.
    if (other_quesval > quesval) {
      return replace_with(other_quesval,
                          other_colid,
                          other_best_metric_val,
                          other_global_nLeft,
                          other_split_start,
                          other_split_end);
    }

    return false;
  }

  DI bool update(DataT other_quesval,
                 std::int64_t other_colid,
                 DataT other_best_metric_val,
                 std::int64_t other_global_nLeft,
                 std::int64_t other_bin)
  {
    return update(
      other_quesval, other_colid, other_best_metric_val, other_global_nLeft, other_bin, other_bin);
  }

  /**
   * @brief reduce the split info in the warp. Best split will be with 0th lane
   */
  DI void warpReduce()
  {
    auto lane = raft::laneId();
#pragma unroll
    for (int i = raft::WarpSize / 2; i >= 1; i /= 2) {
      auto id  = lane + i;
      auto qu  = raft::shfl(quesval, id);
      auto co  = raft::shfl(colid, id);
      auto be  = raft::shfl(best_metric_val, id);
      auto gnl = raft::shfl(global_nLeft, id);
      auto bs  = raft::shfl(split_start, id);
      auto bn  = raft::shfl(split_end, id);
      update(qu, co, be, gnl, bs, bn);
    }
  }

  /**
   * @brief Computes the best split across the threadblocks
   *
   * @param[inout] split_scratch shared scratch with at least one entry per warp
   * @param[inout] split         current split to be updated
   * @param[inout] mutex         location which provides exclusive access to node update
   *
   * @note all threads in the block must enter this function together. At the
   *       end thread0 will contain the best split.
   */
  DI void evalBestSplit(SplitT* split_scratch,
                        volatile SplitT* split,
                        int* mutex,
                        DataT const* quantiles,
                        std::int64_t n_bins)
  {
    warpReduce();
    auto warp   = threadIdx.x / raft::WarpSize;
    auto nWarps = blockDim.x / raft::WarpSize;
    auto lane   = raft::laneId();
    if (lane == 0) split_scratch[warp] = *this;
    __syncthreads();
    if (warp == 0) {
      if (lane < nWarps)
        *this = split_scratch[lane];
      else
        *this = SplitT();
      warpReduce();
      // only the first thread will go ahead and update the best split info
      // for current node
      if (threadIdx.x == 0 && this->IsValid()) {
        select_split_range_midpoint(quantiles, n_bins);
        while (atomicCAS(mutex, 0, 1))
          ;
        SplitT split_reg;
        split_reg.quesval         = split->quesval;
        split_reg.colid           = split->colid;
        split_reg.best_metric_val = split->best_metric_val;
        split_reg.global_nLeft    = split->global_nLeft;
        split_reg.split_start     = split->split_start;
        split_reg.split_end       = split->split_end;
        bool update_result =
          split_reg.update(quesval, colid, best_metric_val, global_nLeft, split_start, split_end);
        if (update_result) {
          split->quesval         = split_reg.quesval;
          split->colid           = split_reg.colid;
          split->best_metric_val = split_reg.best_metric_val;
          split->global_nLeft    = split_reg.global_nLeft;
          split->split_start     = split_reg.split_start;
          split->split_end       = split_reg.split_end;
        }
        __threadfence();
        atomicExch(mutex, 0);
      }
    }
  }
};  // struct Split

/**
 * @brief Initialize the split array
 *
 * @param[out] splits the array to be initialized
 * @param[in]  len    length of this array
 * @param[in]  s      cuda stream where to schedule work
 */
template <typename DataT, int TPB = 256>
void initSplit(Split<DataT>* splits, std::int64_t len, cudaStream_t s)
{
  auto op = [] __device__(Split<DataT> * ptr, std::int64_t idx) { *ptr = Split<DataT>(); };
  raft::linalg::writeOnlyUnaryOp<Split<DataT>, decltype(op), std::int64_t, TPB>(splits, len, op, s);
}

template <typename DataT, int TPB = 256>
void printSplits(Split<DataT>* splits, std::int64_t len, cudaStream_t s)
{
  auto op = [] __device__(Split<DataT> * ptr, std::int64_t idx) {
    printf(
      "quesval = %e, colid = %lld, best_metric_val = %e, global_nLeft = %lld, "
      "local_nLeft = %lld, split_range = [%lld, %lld]\n",
      ptr->quesval,
      static_cast<long long>(ptr->colid),
      ptr->best_metric_val,
      static_cast<long long>(ptr->global_nLeft),
      static_cast<long long>(ptr->local_nLeft),
      static_cast<long long>(ptr->split_start),
      static_cast<long long>(ptr->split_end));
  };
  raft::linalg::writeOnlyUnaryOp<Split<DataT>, decltype(op), std::int64_t, TPB>(splits, len, op, s);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
}

}  // namespace DT
}  // namespace ML
