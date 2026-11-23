/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

// We want to define some functions as usable on device
// But need to guard against this file being compiled by a host compiler
#ifdef __CUDACC__
#define FLATNODE_HD __host__ __device__
#else
#define FLATNODE_HD
#endif

/**
 * A node in Decision Tree.
 * @tparam T data type
 * @tparam L label type
 */
template <typename DataT, typename LabelT>
struct SparseTreeNode {
 private:
  std::int64_t colid                = 0;
  DataT quesval                     = DataT(0);
  double weighted_impurity_decrease = 0.0;  // Local gain times node weight / root weight.
  std::int64_t left_child_id        = -1;
  std::int64_t instance_count       = 0;
  FLATNODE_HD SparseTreeNode(std::int64_t colid,
                             DataT quesval,
                             double weighted_impurity_decrease,
                             std::int64_t left_child_id,
                             std::int64_t instance_count)
    : colid(colid),
      quesval(quesval),
      weighted_impurity_decrease(weighted_impurity_decrease),
      left_child_id(left_child_id),
      instance_count(instance_count)
  {
  }

 public:
  FLATNODE_HD std::int64_t ColumnId() const { return colid; }
  FLATNODE_HD DataT QueryValue() const { return quesval; }
  FLATNODE_HD double WeightedImpurityDecrease() const { return weighted_impurity_decrease; }
  FLATNODE_HD std::int64_t LeftChildId() const { return left_child_id; }
  FLATNODE_HD std::int64_t RightChildId() const { return left_child_id + 1; }
  FLATNODE_HD std::int64_t InstanceCount() const { return instance_count; }

  FLATNODE_HD static SparseTreeNode CreateSplitNode(std::int64_t colid,
                                                    DataT quesval,
                                                    double weighted_impurity_decrease,
                                                    std::int64_t left_child_id,
                                                    std::int64_t instance_count)
  {
    return SparseTreeNode{
      colid, quesval, weighted_impurity_decrease, left_child_id, instance_count};
  }
  FLATNODE_HD static SparseTreeNode CreateLeafNode(std::int64_t instance_count)
  {
    return SparseTreeNode{0, 0, 0.0, -1, instance_count};
  }
  FLATNODE_HD bool IsLeaf() const { return left_child_id == -1; }
  bool operator==(const SparseTreeNode& other) const
  {
    return (this->colid == other.colid) && (this->quesval == other.quesval) &&
           (this->weighted_impurity_decrease == other.weighted_impurity_decrease) &&
           (this->left_child_id == other.left_child_id) &&
           (this->instance_count == other.instance_count);
  }
};
