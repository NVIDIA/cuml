/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../prims/test_utils.h"
#include "test_opg_utils.h"

#include <cuml/ensemble/randomforest.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cuda_utils.cuh>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>

namespace ML {
namespace Test {
namespace opg {

enum class PartitionKind { Contiguous, Strided, Imbalanced, EmptyNonRootRanks };

struct RfMgTestParams {
  int n_rows;
  int n_cols;
  int n_trees;
  float max_features;
  int max_depth;
  int max_leaves;
  int max_n_bins;
  int min_samples_leaf;
  int min_samples_split;
  float min_impurity_decrease;
  int n_streams;
  int handle_n_streams;
  CRITERION split_criterion;
  int seed;
  int n_labels;
  bool double_precision;
  bool use_sample_weights;
  PartitionKind partition_kind;
};

template <typename T>
void expect_floating_values_equal(T distributed_value, T single_node_value, char const* field)
{
  static_assert(std::is_floating_point_v<T>);
  constexpr T absolute_tolerance = std::is_same_v<T, float> ? T{1e-5} : T{1e-12};
  constexpr T relative_tolerance = std::is_same_v<T, float> ? T{1e-5} : T{1e-10};
  auto scale     = std::max(std::abs(distributed_value), std::abs(single_node_value));
  auto tolerance = absolute_tolerance + relative_tolerance * scale;
  EXPECT_NEAR(distributed_value, single_node_value, tolerance)
    << "Mismatched distributed RF " << field;
}

template <typename T, typename L>
void expect_forests_equal(RandomForestMetaData<T, L> const& distributed_forest,
                          RandomForestMetaData<T, L> const& single_node_forest)
{
  EXPECT_EQ(distributed_forest.n_features, single_node_forest.n_features);

  auto const& distributed_params = distributed_forest.rf_params;
  auto const& single_node_params = single_node_forest.rf_params;
  EXPECT_EQ(distributed_params.n_trees, single_node_params.n_trees);
  EXPECT_EQ(distributed_params.bootstrap, single_node_params.bootstrap);
  expect_floating_values_equal(
    distributed_params.max_samples, single_node_params.max_samples, "max_samples");
  EXPECT_EQ(distributed_params.seed, single_node_params.seed);

  auto const& distributed_tree_params = distributed_params.tree_params;
  auto const& single_node_tree_params = single_node_params.tree_params;
  EXPECT_EQ(distributed_tree_params.max_depth, single_node_tree_params.max_depth);
  EXPECT_EQ(distributed_tree_params.max_leaves, single_node_tree_params.max_leaves);
  expect_floating_values_equal(
    distributed_tree_params.max_features, single_node_tree_params.max_features, "max_features");
  EXPECT_EQ(distributed_tree_params.max_n_bins, single_node_tree_params.max_n_bins);
  EXPECT_EQ(distributed_tree_params.min_samples_leaf, single_node_tree_params.min_samples_leaf);
  EXPECT_EQ(distributed_tree_params.min_samples_split, single_node_tree_params.min_samples_split);
  expect_floating_values_equal(distributed_tree_params.min_impurity_decrease,
                               single_node_tree_params.min_impurity_decrease,
                               "min_impurity_decrease");
  EXPECT_EQ(distributed_tree_params.split_criterion, single_node_tree_params.split_criterion);
  EXPECT_EQ(distributed_tree_params.max_batch_size, single_node_tree_params.max_batch_size);

  ASSERT_EQ(distributed_forest.trees.size(), single_node_forest.trees.size());
  for (size_t tree_idx = 0; tree_idx < distributed_forest.trees.size(); ++tree_idx) {
    SCOPED_TRACE(::testing::Message() << "tree_idx=" << tree_idx);
    auto const& distributed_tree = distributed_forest.trees[tree_idx];
    auto const& single_node_tree = single_node_forest.trees[tree_idx];
    ASSERT_NE(distributed_tree, nullptr);
    ASSERT_NE(single_node_tree, nullptr);
    EXPECT_EQ(distributed_tree->treeid, single_node_tree->treeid);
    EXPECT_EQ(distributed_tree->depth_counter, single_node_tree->depth_counter);
    EXPECT_EQ(distributed_tree->leaf_counter, single_node_tree->leaf_counter);
    EXPECT_EQ(distributed_tree->num_outputs, single_node_tree->num_outputs);

    ASSERT_EQ(distributed_tree->sparsetree.size(), single_node_tree->sparsetree.size());
    for (size_t node_idx = 0; node_idx < distributed_tree->sparsetree.size(); ++node_idx) {
      SCOPED_TRACE(::testing::Message() << "node_idx=" << node_idx);
      auto const& distributed_node = distributed_tree->sparsetree[node_idx];
      auto const& single_node_node = single_node_tree->sparsetree[node_idx];
      EXPECT_EQ(distributed_node.ColumnId(), single_node_node.ColumnId());
      expect_floating_values_equal(
        distributed_node.QueryValue(), single_node_node.QueryValue(), "split threshold");
      expect_floating_values_equal(
        distributed_node.BestMetric(), single_node_node.BestMetric(), "split metric");
      EXPECT_EQ(distributed_node.LeftChildId(), single_node_node.LeftChildId());
      EXPECT_EQ(distributed_node.InstanceCount(), single_node_node.InstanceCount());
      EXPECT_EQ(distributed_node.IsLeaf(), single_node_node.IsLeaf());
    }

    ASSERT_EQ(distributed_tree->vector_leaf.size(), single_node_tree->vector_leaf.size());
    for (size_t value_idx = 0; value_idx < distributed_tree->vector_leaf.size(); ++value_idx) {
      SCOPED_TRACE(::testing::Message() << "leaf_value_idx=" << value_idx);
      expect_floating_values_equal(distributed_tree->vector_leaf[value_idx],
                                   single_node_tree->vector_leaf[value_idx],
                                   "leaf value");
    }
  }
}

std::vector<int> local_rows_for_rank(int n_rows, int rank, int size, PartitionKind kind)
{
  std::vector<int> rows;
  if (kind == PartitionKind::Strided) {
    for (int row = rank; row < n_rows; row += size) {
      rows.push_back(row);
    }
    return rows;
  }

  std::vector<int> counts(size, n_rows / size);
  for (int i = 0; i < n_rows % size; ++i) {
    counts[i]++;
  }
  if (kind == PartitionKind::Imbalanced && size > 1) {
    counts.assign(size, 0);
    counts[0]     = std::max(1, (n_rows * 3) / 4);
    int remaining = n_rows - counts[0];
    for (int i = 1; i < size; ++i) {
      counts[i] = remaining / (size - 1);
    }
    for (int i = 1; i <= remaining % (size - 1); ++i) {
      counts[i]++;
    }
  } else if (kind == PartitionKind::EmptyNonRootRanks && size > 1) {
    counts.assign(size, 0);
    counts[0] = n_rows;
  }

  int begin = std::accumulate(counts.begin(), counts.begin() + rank, 0);
  rows.resize(counts[rank]);
  std::iota(rows.begin(), rows.end(), begin);
  rows.erase(std::remove_if(rows.begin(), rows.end(), [=](int row) { return row >= n_rows; }),
             rows.end());
  return rows;
}

template <typename DataT, typename LabelT>
void make_local_dataset(RfMgTestParams const& params,
                        std::vector<int> const& rows,
                        std::vector<DataT>& X,
                        std::vector<LabelT>& y,
                        std::vector<double>& sample_weights)
{
  X.resize(rows.size() * params.n_cols);
  y.resize(rows.size());
  sample_weights.resize(params.use_sample_weights ? rows.size() : 0);
  for (size_t i = 0; i < rows.size(); ++i) {
    int global_row = rows[i];
    DataT signal   = static_cast<DataT>((global_row % 97) - 48);
    for (int col = 0; col < params.n_cols; ++col) {
      DataT feature = signal * static_cast<DataT>(col + 1);
      feature += static_cast<DataT>(((global_row + 13 * col + params.seed) % 11) - 5) /
                 static_cast<DataT>(10);
      X[static_cast<size_t>(col) * rows.size() + i] = feature;
    }
    if constexpr (std::is_integral_v<LabelT>) {
      y[i] = (signal >= DataT(0)) ? 1 : 0;
      if (params.n_labels > 2 && global_row % 17 == 0) { y[i] = 2; }
    } else {
      y[i] = signal * DataT(0.5) + static_cast<DataT>((global_row % 7) - 3);
    }
    if (params.use_sample_weights) { sample_weights[i] = global_row % 2 == 0 ? 0.8 : 0.6; }
  }
}

std::vector<int> global_rows_in_rank_order(RfMgTestParams const& params, int size)
{
  std::vector<int> rows;
  rows.reserve(params.n_rows);
  for (int rank = 0; rank < size; ++rank) {
    auto rank_rows = local_rows_for_rank(params.n_rows, rank, size, params.partition_kind);
    rows.insert(rows.end(), rank_rows.begin(), rank_rows.end());
  }
  return rows;
}

template <typename T, typename L>
void expect_global_tree_counts(RandomForestMetaData<T, L> const& forest, int n_rows)
{
  for (auto const& tree : forest.trees) {
    ASSERT_FALSE(tree->sparsetree.empty());
    EXPECT_EQ(tree->sparsetree.front().InstanceCount(), n_rows);
    for (auto const& node : tree->sparsetree) {
      if (!node.IsLeaf()) {
        ASSERT_GE(node.LeftChildId(), 0);
        ASSERT_GE(node.RightChildId(), 0);
        ASSERT_LT(static_cast<std::size_t>(node.LeftChildId()), tree->sparsetree.size());
        ASSERT_LT(static_cast<std::size_t>(node.RightChildId()), tree->sparsetree.size());
        auto left_count  = tree->sparsetree[node.LeftChildId()].InstanceCount();
        auto right_count = tree->sparsetree[node.RightChildId()].InstanceCount();
        EXPECT_EQ(left_count + right_count, node.InstanceCount());
      }
    }
  }
}

template <typename T, typename L>
void expect_tree_limits(RandomForestMetaData<T, L> const& forest, RfMgTestParams const& params)
{
  EXPECT_EQ(forest.trees.size(), params.n_trees);
  for (auto const& tree : forest.trees) {
    EXPECT_LE(tree->depth_counter, params.max_depth);
    if (params.max_leaves > 0) { EXPECT_LE(tree->leaf_counter, params.max_leaves); }
    for (auto const& node : tree->sparsetree) {
      if (!node.IsLeaf()) { EXPECT_GT(node.BestMetric(), params.min_impurity_decrease); }
    }
  }
}

void initialize_mpi_once()
{
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) { MPI_Init(nullptr, nullptr); }
}

void get_mpi_local_rank_size(int& local_rank, int& local_size)
{
  MPI_Comm local_comm{};
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
  MPI_Comm_rank(local_comm, &local_rank);
  MPI_Comm_size(local_comm, &local_size);
  MPI_Comm_free(&local_comm);
}

template <typename DataT, typename LabelT>
class RfMgPropertyTestImpl {
 public:
  explicit RfMgPropertyTestImpl(RfMgTestParams const& params) : params(params)
  {
    initialize_mpi_once();
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_rank = 0;
    int local_size = 1;
    get_mpi_local_rank_size(local_rank, local_size);

    int n_gpus = 0;
    RAFT_CUDA_TRY(cudaGetDeviceCount(&n_gpus));
    int insufficient_local_gpus = n_gpus < local_size;
    int any_insufficient_gpus   = 0;
    MPI_Allreduce(
      &insufficient_local_gpus, &any_insufficient_gpus, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (any_insufficient_gpus) {
      if (insufficient_local_gpus) {
        ADD_FAILURE() << "Number of GPUs is smaller than local MPI ranks: ngpus=" << n_gpus
                      << ", local_ranks=" << local_size;
      }
      return;
    }
    RAFT_CUDA_TRY(cudaSetDevice(local_rank));

    auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(params.handle_n_streams);
    raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
    raft::comms::initialize_mpi_comms(&handle, MPI_COMM_WORLD);

    auto local_rows = local_rows_for_rank(params.n_rows, rank, size, params.partition_kind);
    std::vector<DataT> h_X;
    std::vector<LabelT> h_y;
    std::vector<double> h_sample_weights;
    make_local_dataset<DataT, LabelT>(params, local_rows, h_X, h_y, h_sample_weights);

    rmm::device_uvector<DataT> X(h_X.size(), handle.get_stream());
    rmm::device_uvector<LabelT> y(h_y.size(), handle.get_stream());
    // A non-null pointer keeps empty ranks on the weighted objective path.
    auto sample_weight_buffer_size =
      params.use_sample_weights ? std::max(size_t{1}, h_sample_weights.size()) : size_t{0};
    rmm::device_uvector<double> sample_weights(sample_weight_buffer_size, handle.get_stream());
    raft::update_device(X.data(), h_X.data(), h_X.size(), handle.get_stream());
    raft::update_device(y.data(), h_y.data(), h_y.size(), handle.get_stream());
    raft::update_device(
      sample_weights.data(), h_sample_weights.data(), h_sample_weights.size(), handle.get_stream());
    auto sample_weight_ptr = params.use_sample_weights ? sample_weights.data() : nullptr;

    auto rf_params = set_rf_params(params.max_depth,
                                   params.max_leaves,
                                   params.max_features,
                                   params.max_n_bins,
                                   params.min_samples_leaf,
                                   params.min_samples_split,
                                   params.min_impurity_decrease,
                                   false,
                                   params.n_trees,
                                   1.0f,
                                   params.seed,
                                   params.split_criterion,
                                   params.n_streams,
                                   128);

    RandomForestMetaData<DataT, LabelT> distributed_forest;
    if constexpr (std::is_integral_v<LabelT>) {
      fit(handle,
          &distributed_forest,
          X.data(),
          static_cast<int>(local_rows.size()),
          params.n_cols,
          y.data(),
          params.n_labels,
          rf_params,
          rapids_logger::level_enum::info,
          nullptr,
          sample_weight_ptr);
    } else {
      fit(handle,
          &distributed_forest,
          X.data(),
          static_cast<int>(local_rows.size()),
          params.n_cols,
          y.data(),
          rf_params,
          rapids_logger::level_enum::info,
          nullptr,
          sample_weight_ptr);
    }

    expect_global_tree_counts(distributed_forest, params.n_rows);
    expect_tree_limits(distributed_forest, params);

    // The distributed quantile sampler assigns global row ids in rank-major order. Reconstruct the
    // single-node input in the same order so both algorithms receive the identical training data.
    auto global_rows = global_rows_in_rank_order(params, size);
    if (global_rows.size() != static_cast<size_t>(params.n_rows)) {
      ADD_FAILURE() << "Reconstructed " << global_rows.size() << " global rows, expected "
                    << params.n_rows;
      return;
    }
    std::vector<DataT> h_global_X;
    std::vector<LabelT> h_global_y;
    std::vector<double> h_global_sample_weights;
    make_local_dataset<DataT, LabelT>(
      params, global_rows, h_global_X, h_global_y, h_global_sample_weights);

    auto single_node_stream_pool = std::make_shared<rmm::cuda_stream_pool>(params.handle_n_streams);
    raft::handle_t single_node_handle(rmm::cuda_stream_per_thread, single_node_stream_pool);
    rmm::device_uvector<DataT> global_X(h_global_X.size(), single_node_handle.get_stream());
    rmm::device_uvector<LabelT> global_y(h_global_y.size(), single_node_handle.get_stream());
    rmm::device_uvector<double> global_sample_weights(h_global_sample_weights.size(),
                                                      single_node_handle.get_stream());
    raft::update_device(
      global_X.data(), h_global_X.data(), h_global_X.size(), single_node_handle.get_stream());
    raft::update_device(
      global_y.data(), h_global_y.data(), h_global_y.size(), single_node_handle.get_stream());
    raft::update_device(global_sample_weights.data(),
                        h_global_sample_weights.data(),
                        h_global_sample_weights.size(),
                        single_node_handle.get_stream());
    auto global_sample_weight_ptr =
      params.use_sample_weights ? global_sample_weights.data() : nullptr;

    auto single_node_rf_params      = rf_params;
    single_node_rf_params.n_streams = 1;
    RandomForestMetaData<DataT, LabelT> single_node_forest;
    if constexpr (std::is_integral_v<LabelT>) {
      fit(single_node_handle,
          &single_node_forest,
          global_X.data(),
          params.n_rows,
          params.n_cols,
          global_y.data(),
          params.n_labels,
          single_node_rf_params,
          rapids_logger::level_enum::info,
          nullptr,
          global_sample_weight_ptr);
    } else {
      fit(single_node_handle,
          &single_node_forest,
          global_X.data(),
          params.n_rows,
          params.n_cols,
          global_y.data(),
          single_node_rf_params,
          rapids_logger::level_enum::info,
          nullptr,
          global_sample_weight_ptr);
    }

    expect_forests_equal(distributed_forest, single_node_forest);
  }

 private:
  RfMgTestParams params;
};

class RfMgPropertyTest : public ::testing::TestWithParam<RfMgTestParams> {
 public:
  void SetUp() override
  {
    auto params        = GetParam();
    bool is_regression = params.split_criterion != GINI && params.split_criterion != ENTROPY;
    if (params.double_precision) {
      if (is_regression) {
        RfMgPropertyTestImpl<double, double> test(params);
      } else {
        RfMgPropertyTestImpl<double, int> test(params);
      }
    } else {
      if (is_regression) {
        RfMgPropertyTestImpl<float, float> test(params);
      } else {
        RfMgPropertyTestImpl<float, int> test(params);
      }
    }
  }
};

TEST_P(RfMgPropertyTest, DistributedProperties) {}

constexpr auto UNLIMITED_DEPTH = std::numeric_limits<std::int32_t>::max();

std::vector<RfMgTestParams> inputs = {
  {128,
   4,
   1,
   1.0f,
   3,
   -1,
   16,
   1,
   2,
   0.0f,
   1,
   1,
   GINI,
   7,
   2,
   false,
   false,
   PartitionKind::Contiguous},
  {128,
   4,
   3,
   0.5f,
   4,
   16,
   32,
   1,
   2,
   0.0f,
   4,
   4,
   ENTROPY,
   11,
   2,
   false,
   false,
   PartitionKind::Strided},
  {192,
   6,
   1,
   1.0f,
   5,
   -1,
   32,
   2,
   4,
   0.0f,
   1,
   1,
   MSE,
   13,
   2,
   false,
   false,
   PartitionKind::Imbalanced},
  {96, 3, 2, 1.0f, 4, 8, 8, 1, 2, 0.0f, 1, 1, GINI, 17, 2, true, false, PartitionKind::Imbalanced},
  {144, 5, 2, 0.8f, 4, -1, 16, 1, 2, 0.0f, 1, 1, GINI, 31, 3, false, false, PartitionKind::Strided},
  {160,
   5,
   1,
   0.8f,
   4,
   -1,
   16,
   1,
   2,
   0.0f,
   1,
   1,
   MSE,
   19,
   2,
   true,
   false,
   PartitionKind::Contiguous},
  {256,
   4,
   5,
   1.0f,
   5,
   -1,
   16,
   1,
   2,
   0.0f,
   1,
   1,
   POISSON,
   7,
   1,
   true,
   false,
   PartitionKind::Strided},
  {256,
   4,
   2,
   0.8f,
   5,
   -1,
   16,
   1,
   2,
   0.0f,
   1,
   1,
   GAMMA,
   7,
   1,
   true,
   false,
   PartitionKind::Contiguous},
  {256,
   4,
   2,
   0.8f,
   5,
   -1,
   16,
   1,
   2,
   0.0f,
   1,
   1,
   INVERSE_GAUSSIAN,
   7,
   1,
   true,
   false,
   PartitionKind::Contiguous},
  {256,
   4,
   5,
   1.0f,
   UNLIMITED_DEPTH,
   -1,
   16,
   1,
   2,
   0.0f,
   1,
   1,
   GINI,
   7,
   2,
   false,
   false,
   PartitionKind::Imbalanced},
  {160,
   4,
   5,
   1.0f,
   UNLIMITED_DEPTH,
   -1,
   16,
   1,
   2,
   0.0f,
   1,
   1,
   ENTROPY,
   7,
   2,
   false,
   false,
   PartitionKind::Imbalanced},
  {64,
   4,
   2,
   1.0f,
   4,
   -1,
   16,
   1,
   2,
   0.0f,
   3,
   3,
   GINI,
   23,
   2,
   false,
   false,
   PartitionKind::EmptyNonRootRanks},
  {80,
   5,
   2,
   0.8f,
   4,
   -1,
   16,
   1,
   2,
   0.0f,
   3,
   3,
   MSE,
   29,
   2,
   false,
   false,
   PartitionKind::EmptyNonRootRanks},
  {80,
   5,
   2,
   0.8f,
   4,
   -1,
   16,
   1,
   2,
   0.0f,
   3,
   3,
   MSE,
   29,
   2,
   false,
   true,
   PartitionKind::EmptyNonRootRanks}};

INSTANTIATE_TEST_SUITE_P(RfTests, RfMgPropertyTest, ::testing::ValuesIn(inputs));

}  // namespace opg
}  // namespace Test
}  // namespace ML

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MLCommon::Test::opg::MPIEnvironment());

  return RUN_ALL_TESTS();
}
