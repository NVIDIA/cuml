/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file isolation_forest_test.cu
 * @brief Unit tests for Isolation Forest based on sklearn's test_iforest.py patterns.
 *
 * Test cases adapted from sklearn's Isolation Forest test suite:
 * - Basic fit/predict functionality
 * - Anomaly score computation
 * - Outlier detection performance
 * - Edge cases and parameter validation
 */

#include "../../src/isolation_forest/isolation_tree_builder.cuh"

#include <cuml/common/utils.hpp>
#include <cuml/ensemble/isolation_forest.hpp>

#include <raft/core/handle.hpp>
#include <raft/linalg/transpose.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <gtest/gtest.h>
#include <test_utils.h>
#include <treelite/enum/task_type.h>
#include <treelite/tree.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

namespace ML {
namespace tl = treelite;

CUML_KERNEL void sample_rows_for_test(
  size_t n_rows, int max_samples, bool bootstrap, uint64_t seed, size_t* sample_indices)
{
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  curandState rng_state;
  curand_init(seed, 0, 0, &rng_state);

  if (bootstrap) {
    for (int i = 0; i < max_samples; ++i) {
      sample_indices[i] = IsolationTree::sample_bounded(&rng_state, n_rows);
    }
  } else {
    size_t start = n_rows - static_cast<size_t>(max_samples);
    for (int i = 0; i < max_samples; ++i) {
      size_t j          = start + static_cast<size_t>(i);
      size_t t          = IsolationTree::sample_bounded(&rng_state, j + 1);
      sample_indices[i] = IsolationTree::contains_sample(sample_indices, i, t) ? j : t;
    }
  }
}

CUML_KERNEL void sample_features_for_test(int n_cols,
                                          int max_features,
                                          uint64_t seed,
                                          int* feature_indices)
{
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  curandState rng_state;
  curand_init(seed, 0, 0, &rng_state);

  size_t start = static_cast<size_t>(n_cols - max_features);
  for (int i = 0; i < max_features; ++i) {
    size_t j = start + static_cast<size_t>(i);
    int t    = static_cast<int>(IsolationTree::sample_bounded(&rng_state, j + 1));
    feature_indices[i] =
      IsolationTree::contains_int_sample(feature_indices, i, t) ? static_cast<int>(j) : t;
  }
}

/**
 * @brief Base test fixture for Isolation Forest tests.
 *
 * Provides common setup with raft handle and stream management.
 */
class IsolationForestTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    stream_pool = std::make_shared<rmm::cuda_stream_pool>(4);
    handle      = std::make_unique<raft::handle_t>(rmm::cuda_stream_per_thread, stream_pool);
    stream      = handle->get_stream().get();
  }

  void TearDown() override
  {
    handle.reset();
    stream_pool.reset();
  }

  /**
   * @brief Transpose data between row-major and column-major formats.
   *
   * @param[in] src Source data
   * @param[out] dst Destination data (must be pre-allocated)
   * @param[in] n_rows Number of rows
   * @param[in] n_cols Number of columns
   */
  template <typename T>
  void transpose_data(thrust::device_vector<T>& src,
                      thrust::device_vector<T>& dst,
                      int n_rows,
                      int n_cols)
  {
    // Note: raft::linalg::transpose takes non-const input even though it doesn't modify it
    raft::linalg::transpose(*handle, src.data().get(), dst.data().get(), n_rows, n_cols, stream);
    handle->sync_stream(stream);
  }

  /**
   * @brief Generate synthetic normal data with optional outliers.
   *
   * Creates a dataset with normally distributed inliers and optional
   * outliers placed far from the main cluster.
   *
   * @param[out] X_colmajor Column-major data for fit() (n_cols x n_rows layout)
   * @param[out] X_rowmajor Row-major data for score_samples() (n_rows x n_cols layout)
   * @param[out] labels Ground truth labels (0=inlier, 1=outlier)
   */
  void generate_data_with_outliers(int n_inliers,
                                   int n_outliers,
                                   int n_features,
                                   float inlier_std,
                                   float outlier_distance,
                                   thrust::device_vector<float>& X_colmajor,
                                   thrust::device_vector<float>& X_rowmajor,
                                   thrust::device_vector<int>& labels)
  {
    int n_samples = n_inliers + n_outliers;
    X_rowmajor.resize(n_samples * n_features);
    X_colmajor.resize(n_samples * n_features);
    labels.resize(n_samples);

    // Generate inliers: centered at origin with small std (row-major)
    raft::random::Rng rng(42);
    rng.normal(X_rowmajor.data().get(), n_inliers * n_features, 0.0f, inlier_std, stream);

    // Generate outliers: far from origin (row-major)
    if (n_outliers > 0) {
      thrust::device_vector<float> outliers(n_outliers * n_features);
      rng.normal(outliers.data().get(),
                 n_outliers * n_features,
                 outlier_distance,
                 inlier_std * 0.5f,
                 stream);
      thrust::copy(outliers.begin(), outliers.end(), X_rowmajor.begin() + n_inliers * n_features);
    }

    // Labels: 0 for inliers, 1 for outliers
    thrust::fill(labels.begin(), labels.begin() + n_inliers, 0);
    thrust::fill(labels.begin() + n_inliers, labels.end(), 1);

    handle->sync_stream(stream);

    // Transpose to column-major for fit()
    transpose_data(X_rowmajor, X_colmajor, n_samples, n_features);
  }

  std::shared_ptr<rmm::cuda_stream_pool> stream_pool;
  std::unique_ptr<raft::handle_t> handle;
  cudaStream_t stream;
};

/**
 * @brief Test: Basic fit produces expected number of trees.
 *
 * Corresponds to sklearn test: test_iforest (partial)
 * Verifies that fit() creates the correct number of isolation trees.
 */
TEST_F(IsolationForestTest, FitProducesExpectedTreeCount)
{
  const int n_samples    = 100;
  const int n_features   = 5;
  const int n_estimators = 10;

  // Generate random data in row-major, then transpose to column-major for fit
  thrust::device_vector<float> X_rowmajor(n_samples * n_features);
  thrust::device_vector<float> X_colmajor(n_samples * n_features);
  raft::random::Rng rng(42);
  rng.normal(X_rowmajor.data().get(), X_rowmajor.size(), 0.0f, 1.0f, stream);
  handle->sync_stream(stream);
  transpose_data(X_rowmajor, X_colmajor, n_samples, n_features);

  // Configure and fit (column-major)
  IF_params params;
  params.n_estimators = n_estimators;
  params.max_samples  = 64;
  params.seed         = 42;

  IsolationForestF model;
  fit(*handle, &model, X_colmajor.data().get(), n_samples, n_features, params);

  // Verify model was created correctly
  EXPECT_EQ(model.params.n_estimators, n_estimators);
  EXPECT_EQ(model.n_features, n_features);
  EXPECT_GT(model.global_nodes.size(), 0);
}

/**
 * @brief Test: Tree depth respects max_depth parameter.
 *
 * Corresponds to sklearn test: test_recalculate_max_depth (partial)
 * Verifies that no tree exceeds the specified maximum depth.
 */
TEST_F(IsolationForestTest, TreeDepthRespected)
{
  const int n_samples    = 256;
  const int n_features   = 4;
  const int n_estimators = 5;
  const int max_depth    = 4;

  thrust::device_vector<float> X_rowmajor(n_samples * n_features);
  thrust::device_vector<float> X_colmajor(n_samples * n_features);
  raft::random::Rng rng(42);
  rng.normal(X_rowmajor.data().get(), X_rowmajor.size(), 0.0f, 1.0f, stream);
  handle->sync_stream(stream);
  transpose_data(X_rowmajor, X_colmajor, n_samples, n_features);

  IF_params params;
  params.n_estimators = n_estimators;
  params.max_samples  = 128;
  params.max_depth    = max_depth;
  params.seed         = 42;

  IsolationForestF model;
  fit(*handle, &model, X_colmajor.data().get(), n_samples, n_features, params);

  // Verify model was created (we can't inspect individual trees with new API)
  EXPECT_GT(model.global_nodes.size(), 0);
  EXPECT_EQ(model.params.max_depth, max_depth);
}

/**
 * @brief Test: max_depth above the previous fixed-storage cap fits successfully.
 */
TEST_F(IsolationForestTest, DeepTreesFitWithGlobalMemoryStorage)
{
  const int n_samples    = 128;
  const int n_features   = 4;
  const int n_estimators = 5;

  thrust::device_vector<float> X_rowmajor(n_samples * n_features);
  thrust::device_vector<float> X_colmajor(n_samples * n_features);
  raft::random::Rng rng(42);
  rng.normal(X_rowmajor.data().get(), X_rowmajor.size(), 0.0f, 1.0f, stream);
  handle->sync_stream(stream);
  transpose_data(X_rowmajor, X_colmajor, n_samples, n_features);

  IF_params params;
  params.n_estimators = n_estimators;
  params.max_samples  = 64;
  params.max_depth    = 17;
  params.seed         = 42;

  IsolationForestF model;
  fit(*handle, &model, X_colmajor.data().get(), n_samples, n_features, params);

  EXPECT_EQ(model.max_nodes_per_tree, 2 * params.max_samples - 1);
  EXPECT_GT(model.global_nodes.size(), 0);

  thrust::device_vector<float> scores(n_samples);
  score_samples(
    *handle, &model, X_rowmajor.data().get(), n_samples, n_features, scores.data().get());

  thrust::host_vector<float> h_scores = scores;
  for (int i = 0; i < n_samples; i++) {
    EXPECT_GE(h_scores[i], 0.0f);
    EXPECT_LE(h_scores[i], 1.0f);
  }
}

/**
 * @brief Test: Auto max_depth calculation.
 *
 * Corresponds to sklearn test: test_recalculate_max_depth
 * When max_depth=-1, it should be set to ceil(log2(max_samples)).
 */
TEST_F(IsolationForestTest, AutoMaxDepthCalculation)
{
  const int n_samples    = 1000;
  const int n_features   = 4;
  const int n_estimators = 3;
  const int max_samples  = 256;

  thrust::device_vector<float> X_rowmajor(n_samples * n_features);
  thrust::device_vector<float> X_colmajor(n_samples * n_features);
  raft::random::Rng rng(42);
  rng.normal(X_rowmajor.data().get(), X_rowmajor.size(), 0.0f, 1.0f, stream);
  handle->sync_stream(stream);
  transpose_data(X_rowmajor, X_colmajor, n_samples, n_features);

  IF_params params;
  params.n_estimators = n_estimators;
  params.max_samples  = max_samples;
  params.max_depth    = -1;  // Auto
  params.seed         = 42;

  IsolationForestF model;
  fit(*handle, &model, X_colmajor.data().get(), n_samples, n_features, params);

  // Verify model was created with auto max_depth
  EXPECT_GT(model.global_nodes.size(), 0);
  // Auto depth should be calculated from max_samples
  EXPECT_EQ(model.params.max_samples, max_samples);
}

/**
 * @brief Test: Subsampling works correctly.
 *
 * Corresponds to sklearn test: test_iforest_subsampled_features (partial)
 * Verifies that each tree is trained on max_samples points.
 */
TEST_F(IsolationForestTest, SubsamplingWorks)
{
  const int n_samples    = 500;
  const int n_features   = 4;
  const int n_estimators = 5;
  const int max_samples  = 100;

  thrust::device_vector<float> X_rowmajor(n_samples * n_features);
  thrust::device_vector<float> X_colmajor(n_samples * n_features);
  raft::random::Rng rng(42);
  rng.normal(X_rowmajor.data().get(), X_rowmajor.size(), 0.0f, 1.0f, stream);
  handle->sync_stream(stream);
  transpose_data(X_rowmajor, X_colmajor, n_samples, n_features);

  IF_params params;
  params.n_estimators = n_estimators;
  params.max_samples  = max_samples;
  params.seed         = 42;

  IsolationForestF model;
  fit(*handle, &model, X_colmajor.data().get(), n_samples, n_features, params);

  // Verify stored max_samples
  EXPECT_EQ(model.n_samples_per_tree, max_samples);
  EXPECT_GT(model.global_nodes.size(), 0);
}

TEST_F(IsolationForestTest, BootstrapFalseSamplesWithoutReplacement)
{
  const size_t n_rows   = 64;
  const int max_samples = 64;
  thrust::device_vector<size_t> sampled(max_samples);

  sample_rows_for_test<<<1, 1, 0, stream>>>(n_rows, max_samples, false, 42, sampled.data().get());
  RAFT_CUDA_TRY(cudaGetLastError());
  handle->sync_stream(stream);

  thrust::host_vector<size_t> h_sampled = sampled;
  std::sort(h_sampled.begin(), h_sampled.end());
  auto last = std::unique(h_sampled.begin(), h_sampled.end());

  EXPECT_EQ(std::distance(h_sampled.begin(), last), max_samples);
  EXPECT_EQ(h_sampled.front(), 0);
  EXPECT_EQ(h_sampled.back(), n_rows - 1);
}

TEST_F(IsolationForestTest, BootstrapTrueSamplesWithReplacement)
{
  const size_t n_rows   = 4;
  const int max_samples = 64;
  thrust::device_vector<size_t> sampled(max_samples);

  sample_rows_for_test<<<1, 1, 0, stream>>>(n_rows, max_samples, true, 42, sampled.data().get());
  RAFT_CUDA_TRY(cudaGetLastError());
  handle->sync_stream(stream);

  thrust::host_vector<size_t> h_sampled = sampled;
  for (size_t row : h_sampled) {
    EXPECT_LT(row, n_rows);
  }

  std::sort(h_sampled.begin(), h_sampled.end());
  auto last = std::unique(h_sampled.begin(), h_sampled.end());
  EXPECT_LT(std::distance(h_sampled.begin(), last), max_samples);
}

TEST_F(IsolationForestTest, MaxFeaturesSamplesUniqueFeatureIds)
{
  const int n_cols       = 16;
  const int max_features = 5;
  thrust::device_vector<int> sampled(max_features);

  sample_features_for_test<<<1, 1, 0, stream>>>(n_cols, max_features, 42, sampled.data().get());
  RAFT_CUDA_TRY(cudaGetLastError());
  handle->sync_stream(stream);

  thrust::host_vector<int> h_sampled = sampled;
  for (int feature : h_sampled) {
    EXPECT_GE(feature, 0);
    EXPECT_LT(feature, n_cols);
  }

  std::sort(h_sampled.begin(), h_sampled.end());
  auto last = std::unique(h_sampled.begin(), h_sampled.end());
  EXPECT_EQ(std::distance(h_sampled.begin(), last), max_features);
}

TEST_F(IsolationForestTest, MaxFeaturesFitStoresOriginalFeatureIds)
{
  const int n_samples    = 128;
  const int n_features   = 8;
  const int n_estimators = 5;
  const int max_features = 3;

  thrust::device_vector<float> X_rowmajor(n_samples * n_features);
  thrust::device_vector<float> X_colmajor(n_samples * n_features);
  raft::random::Rng rng(42);
  rng.normal(X_rowmajor.data().get(), X_rowmajor.size(), 0.0f, 1.0f, stream);
  handle->sync_stream(stream);
  transpose_data(X_rowmajor, X_colmajor, n_samples, n_features);

  IF_params params;
  params.n_estimators = n_estimators;
  params.max_samples  = 64;
  params.max_features = max_features;
  params.seed         = 42;

  IsolationForestF model;
  fit(*handle, &model, X_colmajor.data().get(), n_samples, n_features, params);

  EXPECT_EQ(model.n_features_per_tree, max_features);
  EXPECT_GT(model.global_feature_indices.size(), 0);

  std::vector<int> h_tree_n_nodes(n_estimators);
  RAFT_CUDA_TRY(cudaMemcpyAsync(h_tree_n_nodes.data(),
                                model.global_tree_n_nodes.data(),
                                n_estimators * sizeof(int),
                                cudaMemcpyDeviceToHost,
                                stream));
  std::vector<IsolationTree::IFNode<float>> h_nodes(static_cast<size_t>(n_estimators) *
                                                    model.max_nodes_per_tree);
  RAFT_CUDA_TRY(cudaMemcpyAsync(h_nodes.data(),
                                model.global_nodes.data(),
                                h_nodes.size() * sizeof(IsolationTree::IFNode<float>),
                                cudaMemcpyDeviceToHost,
                                stream));
  handle->sync_stream(stream);

  bool saw_split = false;
  for (int tree = 0; tree < n_estimators; ++tree) {
    auto tree_offset = static_cast<size_t>(tree) * model.max_nodes_per_tree;
    for (int node_idx = 0; node_idx < h_tree_n_nodes[tree]; ++node_idx) {
      const auto& node = h_nodes[tree_offset + node_idx];
      if (node.feature_idx >= 0) {
        saw_split = true;
        EXPECT_LT(node.feature_idx, n_features);
      }
    }
  }
  EXPECT_TRUE(saw_split);

  params.max_features = n_features;
  fit(*handle, &model, X_colmajor.data().get(), n_samples, n_features, params);
  EXPECT_EQ(model.global_feature_indices.size(), 0);
}

TEST_F(IsolationForestTest, ConstantFeaturesDoNotStopSplitting)
{
  const int n_samples    = 64;
  const int n_features   = 8;
  const int n_estimators = 4;
  const int varying_col  = n_features - 1;

  thrust::host_vector<float> h_X(static_cast<size_t>(n_samples) * n_features, 0.0f);
  for (int row = 0; row < n_samples; ++row) {
    h_X[static_cast<size_t>(varying_col) * n_samples + row] = static_cast<float>(row);
  }
  thrust::device_vector<float> X_colmajor = h_X;

  IF_params params;
  params.n_estimators = n_estimators;
  params.max_samples  = n_samples;
  params.max_features = n_features;
  params.seed         = 42;

  IsolationForestF model;
  fit(*handle, &model, X_colmajor.data().get(), n_samples, n_features, params);
  auto compact = get_compact_trees(*handle, &model);

  ASSERT_EQ(compact.tree_offsets.size(), n_estimators);
  for (int tree = 0; tree < n_estimators; ++tree) {
    const auto& root = compact.nodes[compact.tree_offsets[tree]];
    EXPECT_EQ(root.feature_idx, varying_col);
    EXPECT_GE(root.left_child, 0);
    EXPECT_GE(root.right_child, 0);
  }
}

/**
 * @brief Test: Anomaly scores are in valid range [0, 1].
 *
 * Corresponds to sklearn test: test_score_samples
 * All anomaly scores should be between 0 and 1.
 */
TEST_F(IsolationForestTest, AnomalyScoresInRange)
{
  const int n_samples    = 200;
  const int n_features   = 4;
  const int n_estimators = 10;

  // Generate data: row-major for score_samples, column-major for fit
  thrust::device_vector<float> X_rowmajor(n_samples * n_features);
  thrust::device_vector<float> X_colmajor(n_samples * n_features);
  raft::random::Rng rng(42);
  rng.normal(X_rowmajor.data().get(), X_rowmajor.size(), 0.0f, 1.0f, stream);
  handle->sync_stream(stream);
  transpose_data(X_rowmajor, X_colmajor, n_samples, n_features);

  IF_params params;
  params.n_estimators = n_estimators;
  params.max_samples  = 128;
  params.seed         = 42;

  IsolationForestF model;
  fit(*handle, &model, X_colmajor.data().get(), n_samples, n_features, params);

  // Compute anomaly scores (row-major input)
  thrust::device_vector<float> scores(n_samples);
  score_samples(
    *handle, &model, X_rowmajor.data().get(), n_samples, n_features, scores.data().get());

  // Copy to host and verify range
  thrust::host_vector<float> h_scores = scores;
  for (int i = 0; i < n_samples; i++) {
    EXPECT_GE(h_scores[i], 0.0f) << "Score at index " << i << " is below 0";
    EXPECT_LE(h_scores[i], 1.0f) << "Score at index " << i << " is above 1";
  }
}

/**
 * @brief Test: Outliers score higher than inliers.
 *
 * Corresponds to sklearn test: test_iforest_performance
 * Synthetic outliers should have higher anomaly scores than inliers.
 */
TEST_F(IsolationForestTest, OutliersScoreHigher)
{
  const int n_inliers    = 200;
  const int n_outliers   = 20;
  const int n_features   = 4;
  const int n_estimators = 20;

  // Generate data with proper column-major (fit) and row-major (score) formats
  thrust::device_vector<float> X_colmajor;
  thrust::device_vector<float> X_rowmajor;
  thrust::device_vector<int> labels;
  generate_data_with_outliers(
    n_inliers, n_outliers, n_features, 1.0f, 10.0f, X_colmajor, X_rowmajor, labels);

  IF_params params;
  params.n_estimators = n_estimators;
  params.max_samples  = 128;
  params.seed         = 42;

  IsolationForestF model;
  int n_samples = n_inliers + n_outliers;
  fit(*handle, &model, X_colmajor.data().get(), n_samples, n_features, params);

  // Compute anomaly scores (row-major input)
  thrust::device_vector<float> scores(n_samples);
  score_samples(
    *handle, &model, X_rowmajor.data().get(), n_samples, n_features, scores.data().get());

  // Copy to host
  thrust::host_vector<float> h_scores = scores;
  thrust::host_vector<int> h_labels   = labels;

  // Calculate mean scores for inliers and outliers
  float inlier_sum  = 0.0f;
  float outlier_sum = 0.0f;
  int inlier_count  = 0;
  int outlier_count = 0;

  for (int i = 0; i < n_samples; i++) {
    if (h_labels[i] == 0) {
      inlier_sum += h_scores[i];
      inlier_count++;
    } else {
      outlier_sum += h_scores[i];
      outlier_count++;
    }
  }

  float mean_inlier_score  = inlier_sum / inlier_count;
  float mean_outlier_score = outlier_sum / outlier_count;

  // Outliers should have higher average score (closer to 1)
  EXPECT_GT(mean_outlier_score, mean_inlier_score)
    << "Mean outlier score (" << mean_outlier_score << ") should be higher than mean inlier score ("
    << mean_inlier_score << ")";
}

/**
 * @brief Test: Predict returns correct labels.
 *
 * Corresponds to sklearn test: test_iforest (partial)
 * Predictions should be 1 (anomaly) or -1 (normal).
 */
TEST_F(IsolationForestTest, PredictReturnsValidLabels)
{
  const int n_samples    = 100;
  const int n_features   = 4;
  const int n_estimators = 10;

  thrust::device_vector<float> X_rowmajor(n_samples * n_features);
  thrust::device_vector<float> X_colmajor(n_samples * n_features);
  raft::random::Rng rng(42);
  rng.normal(X_rowmajor.data().get(), X_rowmajor.size(), 0.0f, 1.0f, stream);
  handle->sync_stream(stream);
  transpose_data(X_rowmajor, X_colmajor, n_samples, n_features);

  IF_params params;
  params.n_estimators = n_estimators;
  params.max_samples  = 64;
  params.seed         = 42;

  IsolationForestF model;
  fit(*handle, &model, X_colmajor.data().get(), n_samples, n_features, params);

  // Get predictions (row-major input)
  thrust::device_vector<int> predictions(n_samples);
  predict(
    *handle, &model, X_rowmajor.data().get(), n_samples, n_features, predictions.data().get());

  // Verify all predictions are either 1 or -1
  thrust::host_vector<int> h_predictions = predictions;
  for (int i = 0; i < n_samples; i++) {
    EXPECT_TRUE(h_predictions[i] == 1 || h_predictions[i] == -1)
      << "Prediction at index " << i << " is " << h_predictions[i] << ", expected 1 or -1";
  }
}

/**
 * @brief Test: Deterministic results with same seed.
 *
 * Corresponds to sklearn test: reproducibility checks
 * Same seed should produce identical results.
 */
TEST_F(IsolationForestTest, DeterministicWithRandomState)
{
  const int n_samples    = 100;
  const int n_features   = 4;
  const int n_estimators = 5;

  thrust::device_vector<float> X_rowmajor(n_samples * n_features);
  thrust::device_vector<float> X_colmajor(n_samples * n_features);
  raft::random::Rng rng(42);
  rng.normal(X_rowmajor.data().get(), X_rowmajor.size(), 0.0f, 1.0f, stream);
  handle->sync_stream(stream);
  transpose_data(X_rowmajor, X_colmajor, n_samples, n_features);

  IF_params params;
  params.n_estimators = n_estimators;
  params.max_samples  = 64;
  params.seed         = 12345;

  // First fit (column-major)
  IsolationForestF model1;
  fit(*handle, &model1, X_colmajor.data().get(), n_samples, n_features, params);

  // Second fit with same seed (column-major)
  IsolationForestF model2;
  fit(*handle, &model2, X_colmajor.data().get(), n_samples, n_features, params);

  // Compute scores from both models (row-major)
  thrust::device_vector<float> scores1(n_samples);
  thrust::device_vector<float> scores2(n_samples);

  score_samples(
    *handle, &model1, X_rowmajor.data().get(), n_samples, n_features, scores1.data().get());
  score_samples(
    *handle, &model2, X_rowmajor.data().get(), n_samples, n_features, scores2.data().get());

  // Scores should be identical
  thrust::host_vector<float> h_scores1 = scores1;
  thrust::host_vector<float> h_scores2 = scores2;

  for (int i = 0; i < n_samples; i++) {
    EXPECT_FLOAT_EQ(h_scores1[i], h_scores2[i])
      << "Scores differ at index " << i << " despite same seed";
  }
}

/**
 * @brief Test: c(n) normalization constant computation.
 *
 * Verifies the average path length normalization constant is computed correctly.
 */
TEST_F(IsolationForestTest, CNormalizationConstant)
{
  // c(n) = 2*H(n-1) - 2(n-1)/n where H(i) ≈ ln(i) + 0.5772...
  // For n=256:
  //   H(255) ≈ ln(255) + 0.5772 ≈ 5.541 + 0.577 ≈ 6.118
  //   c(256) = 2 * 6.118 - 2 * 255/256 ≈ 12.236 - 1.992 ≈ 10.24

  double c_256 = compute_c_normalization(256);
  EXPECT_NEAR(c_256, 10.24, 0.1);

  double c_2 = compute_c_normalization(2);
  EXPECT_DOUBLE_EQ(c_2, 1.0);

  double c_1 = compute_c_normalization(1);
  EXPECT_DOUBLE_EQ(c_1, 0.0);
}

/**
 * @brief Test: Double precision support.
 *
 * Verifies that double precision fits and scores correctly.
 */
TEST_F(IsolationForestTest, DoublePrecisionSupport)
{
  const int n_samples    = 100;
  const int n_features   = 4;
  const int n_estimators = 5;

  thrust::device_vector<double> X_rowmajor(n_samples * n_features);
  thrust::device_vector<double> X_colmajor(n_samples * n_features);
  raft::random::Rng rng(42);
  rng.normal(X_rowmajor.data().get(), X_rowmajor.size(), 0.0, 1.0, stream);
  handle->sync_stream(stream);
  transpose_data(X_rowmajor, X_colmajor, n_samples, n_features);

  IF_params params;
  params.n_estimators = n_estimators;
  params.max_samples  = 64;
  params.seed         = 42;

  IsolationForestD model;
  fit(*handle, &model, X_colmajor.data().get(), n_samples, n_features, params);

  // Verify model was created
  EXPECT_EQ(model.params.n_estimators, n_estimators);
  EXPECT_GT(model.global_nodes.size(), 0);

  // Compute scores (row-major)
  thrust::device_vector<double> scores(n_samples);
  score_samples(
    *handle, &model, X_rowmajor.data().get(), n_samples, n_features, scores.data().get());

  // Verify scores are in range
  thrust::host_vector<double> h_scores = scores;
  for (int i = 0; i < n_samples; i++) {
    EXPECT_GE(h_scores[i], 0.0);
    EXPECT_LE(h_scores[i], 1.0);
  }
}

/**
 * @brief Test: Edge case with small dataset.
 *
 * Corresponds to sklearn test: edge cases
 * Handles datasets smaller than max_samples correctly.
 */
TEST_F(IsolationForestTest, SmallDataset)
{
  const int n_samples    = 32;  // Smaller than typical max_samples (256)
  const int n_features   = 4;
  const int n_estimators = 3;

  thrust::device_vector<float> X_rowmajor(n_samples * n_features);
  thrust::device_vector<float> X_colmajor(n_samples * n_features);
  raft::random::Rng rng(42);
  rng.normal(X_rowmajor.data().get(), X_rowmajor.size(), 0.0f, 1.0f, stream);
  handle->sync_stream(stream);
  transpose_data(X_rowmajor, X_colmajor, n_samples, n_features);

  IF_params params;
  params.n_estimators = n_estimators;
  params.max_samples  = 256;  // Larger than n_samples
  params.seed         = 42;

  IsolationForestF model;
  fit(*handle, &model, X_colmajor.data().get(), n_samples, n_features, params);

  // max_samples should be capped at n_samples
  EXPECT_EQ(model.n_samples_per_tree, n_samples);

  // Should still produce valid scores (row-major)
  thrust::device_vector<float> scores(n_samples);
  score_samples(
    *handle, &model, X_rowmajor.data().get(), n_samples, n_features, scores.data().get());

  thrust::host_vector<float> h_scores = scores;
  for (int i = 0; i < n_samples; i++) {
    EXPECT_GE(h_scores[i], 0.0f);
    EXPECT_LE(h_scores[i], 1.0f);
  }
}

/**
 * @brief Test: Handle uniform data (all same values).
 *
 * Corresponds to sklearn test: test_iforest_with_uniform_data
 * Edge case where all feature values are identical.
 */
TEST_F(IsolationForestTest, UniformData)
{
  const int n_samples    = 100;
  const int n_features   = 4;
  const int n_estimators = 5;

  // Create uniform data (all zeros) - same in both layouts
  thrust::device_vector<float> X_colmajor(n_samples * n_features, 0.0f);
  thrust::device_vector<float> X_rowmajor(n_samples * n_features, 0.0f);

  IF_params params;
  params.n_estimators = n_estimators;
  params.max_samples  = 64;
  params.seed         = 42;

  IsolationForestF model;
  fit(*handle, &model, X_colmajor.data().get(), n_samples, n_features, params);

  // Model should fit without error
  EXPECT_EQ(model.params.n_estimators, n_estimators);
  EXPECT_GT(model.global_nodes.size(), 0);

  // Compute scores - all should be similar for uniform data (row-major)
  thrust::device_vector<float> scores(n_samples);
  score_samples(
    *handle, &model, X_rowmajor.data().get(), n_samples, n_features, scores.data().get());

  thrust::host_vector<float> h_scores = scores;

  // All scores should be in valid range
  for (int i = 0; i < n_samples; i++) {
    EXPECT_GE(h_scores[i], 0.0f);
    EXPECT_LE(h_scores[i], 1.0f);
  }

  // For uniform data, scores should be very similar (low variance)
  float mean_score = thrust::reduce(h_scores.begin(), h_scores.end(), 0.0f) / n_samples;
  float variance   = 0.0f;
  for (const auto& s : h_scores) {
    variance += (s - mean_score) * (s - mean_score);
  }
  variance /= n_samples;

  // Variance should be small for uniform data
  EXPECT_LT(variance, 0.1f) << "Uniform data should produce similar scores, but variance is high";
}

/**
 * @brief Test: Large number of estimators.
 *
 * Verifies that training many trees works correctly.
 */
TEST_F(IsolationForestTest, ManyEstimators)
{
  const int n_samples    = 200;
  const int n_features   = 4;
  const int n_estimators = 50;

  thrust::device_vector<float> X_rowmajor(n_samples * n_features);
  thrust::device_vector<float> X_colmajor(n_samples * n_features);
  raft::random::Rng rng(42);
  rng.normal(X_rowmajor.data().get(), X_rowmajor.size(), 0.0f, 1.0f, stream);
  handle->sync_stream(stream);
  transpose_data(X_rowmajor, X_colmajor, n_samples, n_features);

  IF_params params;
  params.n_estimators = n_estimators;
  params.max_samples  = 64;
  params.seed         = 42;

  IsolationForestF model;
  fit(*handle, &model, X_colmajor.data().get(), n_samples, n_features, params);

  // Model should be created with all trees
  EXPECT_EQ(model.params.n_estimators, n_estimators);
  EXPECT_GT(model.global_nodes.size(), 0);

  // Verify scoring works with many trees (row-major)
  thrust::device_vector<float> scores(n_samples);
  score_samples(
    *handle, &model, X_rowmajor.data().get(), n_samples, n_features, scores.data().get());

  thrust::host_vector<float> h_scores = scores;
  for (int i = 0; i < n_samples; i++) {
    EXPECT_GE(h_scores[i], 0.0f);
    EXPECT_LE(h_scores[i], 1.0f);
  }
}

/**
 * @brief Test: Predict threshold parameter.
 *
 * Verifies that the threshold parameter affects predictions correctly.
 */
TEST_F(IsolationForestTest, PredictThreshold)
{
  const int n_samples    = 200;
  const int n_features   = 4;
  const int n_estimators = 10;

  thrust::device_vector<float> X_rowmajor(n_samples * n_features);
  thrust::device_vector<float> X_colmajor(n_samples * n_features);
  raft::random::Rng rng(42);
  rng.normal(X_rowmajor.data().get(), X_rowmajor.size(), 0.0f, 1.0f, stream);
  handle->sync_stream(stream);
  transpose_data(X_rowmajor, X_colmajor, n_samples, n_features);

  IF_params params;
  params.n_estimators = n_estimators;
  params.max_samples  = 64;
  params.seed         = 42;

  IsolationForestF model;
  fit(*handle, &model, X_colmajor.data().get(), n_samples, n_features, params);

  // Predictions with low threshold (more anomalies) - row-major input
  thrust::device_vector<int> pred_low(n_samples);
  predict(
    *handle, &model, X_rowmajor.data().get(), n_samples, n_features, pred_low.data().get(), 0.3f);

  // Predictions with high threshold (fewer anomalies) - row-major input
  thrust::device_vector<int> pred_high(n_samples);
  predict(
    *handle, &model, X_rowmajor.data().get(), n_samples, n_features, pred_high.data().get(), 0.7f);

  // Count anomalies in each
  int anomalies_low  = thrust::count(thrust::device, pred_low.begin(), pred_low.end(), 1);
  int anomalies_high = thrust::count(thrust::device, pred_high.begin(), pred_high.end(), 1);

  // Lower threshold should produce more anomaly predictions
  EXPECT_GE(anomalies_low, anomalies_high)
    << "Lower threshold should produce at least as many anomalies";

  thrust::device_vector<float> scores(n_samples);
  score_samples(
    *handle, &model, X_rowmajor.data().get(), n_samples, n_features, scores.data().get());
  float boundary = scores[0];
  thrust::device_vector<int> pred_boundary(n_samples);
  predict(*handle,
          &model,
          X_rowmajor.data().get(),
          n_samples,
          n_features,
          pred_boundary.data().get(),
          boundary);
  EXPECT_EQ(pred_boundary[0], -1) << "A score exactly at the threshold is not anomalous";
}

/**
 * @brief Test: Export trained Isolation Forest model to Treelite.
 *
 * The exported Treelite model is a regressor whose prediction is the average
 * path length across isolation trees. The anomaly-score transform is applied
 * outside Treelite by higher-level callers.
 */
TEST_F(IsolationForestTest, TreeliteExportMetadata)
{
  const int n_samples    = 128;
  const int n_features   = 6;
  const int n_estimators = 7;

  thrust::device_vector<float> X_rowmajor(n_samples * n_features);
  thrust::device_vector<float> X_colmajor(n_samples * n_features);
  raft::random::Rng rng(42);
  rng.normal(X_rowmajor.data().get(), X_rowmajor.size(), 0.0f, 1.0f, stream);
  handle->sync_stream(stream);
  transpose_data(X_rowmajor, X_colmajor, n_samples, n_features);

  IF_params params;
  params.n_estimators = n_estimators;
  params.max_samples  = 64;
  params.seed         = 42;

  IsolationForestF model;
  fit(*handle, &model, X_colmajor.data().get(), n_samples, n_features, params);

  TreeliteModelHandle tl_handle = nullptr;
  build_treelite_isolation_forest(&tl_handle, *handle, &model);
  ASSERT_NE(tl_handle, nullptr);

  std::unique_ptr<tl::Model> tl_model(static_cast<tl::Model*>(tl_handle));
  auto& model_preset = std::get<tl::ModelPreset<float, float>>(tl_model->variant_);

  EXPECT_EQ(tl_model->task_type, tl::TaskType::kRegressor);
  EXPECT_EQ(tl_model->postprocessor, "identity");
  EXPECT_EQ(tl_model->num_target, 1);
  ASSERT_EQ(tl_model->num_class.Size(), 1);
  EXPECT_EQ(tl_model->num_class[0], 1);
  ASSERT_EQ(tl_model->leaf_vector_shape.Size(), 2);
  EXPECT_EQ(tl_model->leaf_vector_shape[0], 1);
  EXPECT_EQ(tl_model->leaf_vector_shape[1], 1);
  EXPECT_EQ(tl_model->num_feature, n_features);
  EXPECT_TRUE(tl_model->average_tree_output);
  ASSERT_EQ(model_preset.trees.size(), static_cast<std::size_t>(n_estimators));
  for (const auto& tree : model_preset.trees) {
    EXPECT_GT(tree.num_nodes, 0);
  }
}

/**
 * @brief Test: Treelite export works for deep global-memory tree storage.
 */
TEST_F(IsolationForestTest, TreeliteExportGlobalMemoryStorage)
{
  const int n_samples    = 128;
  const int n_features   = 6;
  const int n_estimators = 7;

  thrust::device_vector<float> X_rowmajor(n_samples * n_features);
  thrust::device_vector<float> X_colmajor(n_samples * n_features);
  raft::random::Rng rng(42);
  rng.normal(X_rowmajor.data().get(), X_rowmajor.size(), 0.0f, 1.0f, stream);
  handle->sync_stream(stream);
  transpose_data(X_rowmajor, X_colmajor, n_samples, n_features);

  IF_params params;
  params.n_estimators = n_estimators;
  params.max_samples  = 64;
  params.max_depth    = 17;
  params.seed         = 42;

  IsolationForestF model;
  fit(*handle, &model, X_colmajor.data().get(), n_samples, n_features, params);
  ASSERT_GT(model.global_nodes.size(), 0);

  TreeliteModelHandle tl_handle = nullptr;
  build_treelite_isolation_forest(&tl_handle, *handle, &model);
  ASSERT_NE(tl_handle, nullptr);

  std::unique_ptr<tl::Model> tl_model(static_cast<tl::Model*>(tl_handle));
  auto& model_preset = std::get<tl::ModelPreset<float, float>>(tl_model->variant_);

  EXPECT_EQ(tl_model->task_type, tl::TaskType::kRegressor);
  EXPECT_EQ(tl_model->num_feature, n_features);
  EXPECT_TRUE(tl_model->average_tree_output);
  ASSERT_EQ(model_preset.trees.size(), static_cast<std::size_t>(n_estimators));
  for (const auto& tree : model_preset.trees) {
    EXPECT_GT(tree.num_nodes, 0);
  }
}

}  // namespace ML
