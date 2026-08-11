/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuml/common/checked_arithmetic.hpp>
#include <cuml/common/logger.hpp>
#include <cuml/datasets/make_blobs.hpp>
#include <cuml/ensemble/randomforest.hpp>
#include <cuml/tree/algo_helper.h>

#include <raft/core/handle.hpp>
#include <raft/linalg/transpose.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cub/device/device_segmented_reduce.cuh>
#include <cuda/iterator>
#include <cuda/std/functional>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/transform.h>

#include <decisiontree/batched-levelalgo/kernels/builder_kernels.cuh>
#include <decisiontree/batched-levelalgo/objectives.cuh>
#include <decisiontree/batched-levelalgo/quantiles.cuh>
#include <gtest/gtest.h>
#include <nvforest/device_type.hpp>
#include <nvforest/infer_kind.hpp>
#include <nvforest/tree_layout.hpp>
#include <nvforest/treelite_importer.hpp>
#include <test_utils.h>
#include <treelite/tree.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>

namespace ML {

// Utils for changing tuple into struct
namespace detail {
template <typename result_type, typename... types, std::size_t... indices>
result_type make_struct(std::tuple<types...> t,
                        std::index_sequence<indices...>)  // &, &&, const && etc.
{
  return {std::get<indices>(t)...};
}

}  // namespace detail

template <typename result_type, typename... types>
result_type make_struct(std::tuple<types...> t)  // &, &&, const && etc.
{
  return detail::make_struct<result_type, types...>(
    t, std::index_sequence_for<types...>{});  // if there is repeated types, then the change for
                                              // using std::index_sequence_for is trivial
}

template <int I, typename RandomGenT, typename ParamT, typename T>
void SampleWithoutReplacemment(RandomGenT& gen, std::vector<ParamT>& sample, std::vector<T> x)
{
  std::vector<T> parameter_sample(sample.size());
  std::shuffle(x.begin(), x.end(), gen);
  for (size_t i = 0; i < sample.size(); i++) {
    parameter_sample[i] = x[i % x.size()];
  }
  std::shuffle(parameter_sample.begin(), parameter_sample.end(), gen);
  for (size_t i = 0; i < sample.size(); i++) {
    std::get<I>(sample[i]) = parameter_sample[i];
  }
}

template <int I, typename RandomGenT, typename ParamT, typename T, typename... Args>
void AddParameters(RandomGenT& gen, std::vector<ParamT>& sample, std::vector<T> x, Args... args)
{
  SampleWithoutReplacemment<I>(gen, sample, x);
  if constexpr (sizeof...(args) > 0) { AddParameters<I + 1>(gen, sample, args...); }
}

template <typename ParamT, typename... Args>
std::vector<ParamT> SampleParameters(int num_samples, size_t seed, Args... args)
{
  std::vector<typename ParamT::types> tuple_sample(num_samples);
  std::default_random_engine gen(seed);
  AddParameters<0>(gen, tuple_sample, args...);
  std::vector<ParamT> sample(num_samples);
  for (int i = 0; i < num_samples; i++) {
    sample[i] = make_struct<ParamT>(tuple_sample[i]);
  }
  return sample;
}

struct RfTestParams {
  std::size_t n_rows;
  std::size_t n_cols;
  int n_trees;
  float max_features;
  float max_samples;
  int max_depth;
  int max_leaves;
  bool bootstrap;
  int max_n_bins;
  int min_samples_leaf;
  int min_samples_split;
  float min_impurity_decrease;
  int n_streams;
  CRITERION split_criterion;
  int seed;
  int n_labels;
  bool sample_weight;
  bool double_precision;
  bool input_row_major;
  // c++ has no reflection, so we enumerate the types here
  // This must be updated if new fields are added
  using types = std::tuple<std::size_t,
                           std::size_t,
                           int,
                           float,
                           float,
                           int,
                           int,
                           bool,
                           int,
                           int,
                           int,
                           float,
                           int,
                           CRITERION,
                           int,
                           int,
                           bool,
                           bool,
                           bool>;
};

std::ostream& operator<<(std::ostream& os, const RfTestParams& ps)
{
  os << "n_rows = " << ps.n_rows << ", n_cols = " << ps.n_cols;
  os << ", n_trees = " << ps.n_trees << ", max_features = " << ps.max_features;
  os << ", max_samples = " << ps.max_samples << ", max_depth = " << ps.max_depth;
  os << ", max_leaves = " << ps.max_leaves << ", bootstrap = " << ps.bootstrap;
  os << ", max_n_bins = " << ps.max_n_bins << ", min_samples_leaf = " << ps.min_samples_leaf;
  os << ", min_samples_split = " << ps.min_samples_split;
  os << ", min_impurity_decrease = " << ps.min_impurity_decrease
     << ", n_streams = " << ps.n_streams;
  os << ", split_criterion = " << ps.split_criterion << ", seed = " << ps.seed;
  os << ", n_labels = " << ps.n_labels << ", sample_weight = " << ps.sample_weight
     << ", double_precision = " << ps.double_precision
     << ", input_row_major = " << ps.input_row_major;
  return os;
}

namespace {

template <typename BinT>
void expectBinEqual(const BinT& actual, const BinT& expected)
{
  EXPECT_EQ(actual.Count(), expected.Count());
  EXPECT_DOUBLE_EQ(actual.Weight(), expected.Weight());
  if constexpr (std::is_same_v<BinT, DT::RegressionBin> ||
                std::is_same_v<BinT, DT::WeightedRegressionBin>) {
    EXPECT_DOUBLE_EQ(actual.LabelSum(), expected.LabelSum());
  }
}

template <typename BinT>
void testBinReductionRoundTrip(std::vector<BinT> const& input)
{
  if (input.empty()) { return; }

  raft::handle_t handle;
  auto stream = handle.get_stream();
  rmm::device_uvector<BinT> d_input(input.size(), stream);
  rmm::device_uvector<double> d_packed(DT::reduction_buffer_size_v<BinT> * input.size(), stream);
  rmm::device_uvector<BinT> d_output(input.size(), stream);

  raft::update_device(d_input.data(), input.data(), input.size(), stream);
  DT::packHistograms(d_input.data(), d_packed.data(), input.size(), stream);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  DT::unpackHistograms(d_packed.data(), d_output.data(), input.size(), stream);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  std::vector<BinT> output(input.size());
  raft::update_host(output.data(), d_output.data(), output.size(), stream);
  handle.sync_stream();

  for (std::size_t i = 0; i < input.size(); ++i) {
    expectBinEqual(output[i], input[i]);
  }
}

}  // namespace

TEST(RfTests, BinReductionRoundTrip)
{
  testBinReductionRoundTrip<DT::ClassificationBin>({});
  testBinReductionRoundTrip<DT::ClassificationBin>({{7}});
  testBinReductionRoundTrip<DT::WeightedClassificationBin>({});
  testBinReductionRoundTrip<DT::WeightedClassificationBin>({{4, 2.5}});
  testBinReductionRoundTrip<DT::RegressionBin>({});
  testBinReductionRoundTrip<DT::RegressionBin>({{3.5, 4}});
  testBinReductionRoundTrip<DT::WeightedRegressionBin>({});
  testBinReductionRoundTrip<DT::WeightedRegressionBin>({{6.75, 3, 2.25}});
  testBinReductionRoundTrip<DT::ClassificationBin>({{0}, {7}, {13}});
  testBinReductionRoundTrip<DT::WeightedClassificationBin>({{0, 0.0}, {4, 2.5}, {11, 8.25}});
  testBinReductionRoundTrip<DT::RegressionBin>({{0.0, 0}, {3.5, 4}, {-2.25, 9}});
  testBinReductionRoundTrip<DT::WeightedRegressionBin>(
    {{0.0, 0, 0.0}, {6.75, 3, 2.25}, {-1.5, 8, 5.5}});
}

template <typename DataT, typename LabelT>
std::shared_ptr<thrust::device_vector<LabelT>> nvForestPredict(
  const raft::handle_t& handle,
  RfTestParams params,
  DataT* X_transpose,
  RandomForestMetaData<DataT, LabelT>* forest)
{
  auto pred      = std::shared_ptr<thrust::device_vector<LabelT>>();
  auto workspace = std::shared_ptr<thrust::device_vector<DataT>>();  // Scratch space
  if constexpr (std::is_integral_v<LabelT>) {
    // For classifiers, allocate extra scratch space to store probabilities from nvForest
    // We will perform argmax to convert probabilities into class outputs.
    pred      = std::make_shared<thrust::device_vector<LabelT>>(params.n_rows);
    workspace = std::make_shared<thrust::device_vector<DataT>>(params.n_rows * params.n_labels);
  } else {
    // For regressors, no need to post-process predictions from nvForest
    static_assert(std::is_same_v<LabelT, DataT>,
                  "LabelT and DataT must be identical for regression task");
    pred      = std::make_shared<thrust::device_vector<LabelT>>(params.n_rows);
    workspace = pred;
  }
  TreeliteModelHandle model;
  build_treelite_forest(&model, forest, params.n_cols);

  auto nvforest_model = nvforest::import_from_treelite_handle(model,
                                                              nvforest::tree_layout::breadth_first,
                                                              128,
                                                              std::is_same_v<DataT, double>,
                                                              nvforest::device_type::gpu,
                                                              handle.get_device(),
                                                              handle.get_next_usable_stream());
  handle.sync_stream();
  handle.sync_stream_pool();
  delete static_cast<treelite::Model*>(model);

  nvforest_model.predict(handle,
                         workspace->data().get(),
                         X_transpose,
                         params.n_rows,
                         nvforest::device_type::gpu,
                         nvforest::device_type::gpu,
                         nvforest::infer_kind::default_kind,
                         1);
  handle.sync_stream();
  handle.sync_stream_pool();

  if constexpr (std::is_integral_v<LabelT>) {
    // Perform argmax to convert probabilities into class outputs
    auto offsets_it =
      cuda::make_transform_iterator(thrust::make_counting_iterator(0),
                                    [=] __device__(int x) -> int { return x * params.n_labels; });
    using kv_type = cub::KeyValuePair<int, DataT>;

    // Compute size of workspace for the segmented reduce operation
    std::size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedReduce::ArgMax(nullptr,
                                       temp_storage_bytes,
                                       workspace->begin(),
                                       thrust::device_pointer_cast<kv_type>(nullptr),
                                       params.n_rows,
                                       offsets_it,
                                       offsets_it + 1);

    // Allocate workspace and perform segmented reduce
    thrust::device_vector<kv_type> workspace2(params.n_rows + temp_storage_bytes / sizeof(kv_type) +
                                              1);
    cub::DeviceSegmentedReduce::ArgMax(thrust::raw_pointer_cast(workspace2.data() + params.n_rows),
                                       temp_storage_bytes,
                                       workspace->begin(),
                                       workspace2.begin(),
                                       params.n_rows,
                                       offsets_it,
                                       offsets_it + 1);
    thrust::transform(workspace2.begin(),
                      workspace2.begin() + params.n_rows,
                      pred->begin(),
                      [] __device__(kv_type x) -> int { return x.key; });
  }

  return pred;
}

template <typename DataT, typename LabelT>
auto nvForestPredictProba(const raft::handle_t& handle,
                          RfTestParams params,
                          DataT* X_transpose,
                          RandomForestMetaData<DataT, LabelT>* forest)
{
  static_assert(std::is_integral_v<LabelT>, "Must be classification");

  std::size_t num_outputs = params.n_labels;
  auto pred = std::make_shared<thrust::device_vector<float>>(params.n_rows * num_outputs);
  TreeliteModelHandle model;
  build_treelite_forest(&model, forest, params.n_cols);

  auto nvforest_model = nvforest::import_from_treelite_handle(model,
                                                              nvforest::tree_layout::breadth_first,
                                                              128,
                                                              std::is_same_v<DataT, double>,
                                                              nvforest::device_type::gpu,
                                                              handle.get_device(),
                                                              handle.get_next_usable_stream());
  handle.sync_stream();
  handle.sync_stream_pool();
  delete static_cast<treelite::Model*>(model);

  nvforest_model.predict(handle,
                         pred->data().get(),
                         X_transpose,
                         params.n_rows,
                         nvforest::device_type::gpu,
                         nvforest::device_type::gpu,
                         nvforest::infer_kind::default_kind,
                         1);
  handle.sync_stream();
  handle.sync_stream_pool();

  return pred;
}
template <typename LabelT>
RF_metrics Score(const raft::handle_t& handle,
                 RfTestParams params,
                 const LabelT* y,
                 const LabelT* pred,
                 const double* sample_weight)
{
  thrust::host_vector<LabelT> h_y(params.n_rows);
  thrust::host_vector<LabelT> h_pred(params.n_rows);
  raft::update_host(h_y.data(), y, params.n_rows, handle.get_stream());
  raft::update_host(h_pred.data(), pred, params.n_rows, handle.get_stream());
  thrust::host_vector<double> h_sample_weight;
  if (sample_weight != nullptr) {
    h_sample_weight.resize(params.n_rows);
    raft::update_host(h_sample_weight.data(), sample_weight, params.n_rows, handle.get_stream());
  }
  handle.sync_stream();

  double weight_sum = 0.0;
  if constexpr (std::is_integral_v<LabelT>) {
    double correct = 0.0;
    for (std::size_t i = 0; i < params.n_rows; ++i) {
      double weight = sample_weight == nullptr ? 1.0 : double(h_sample_weight[i]);
      weight_sum += weight;
      if (h_y[i] == h_pred[i]) { correct += weight; }
    }
    return set_rf_metrics_classification(float(correct / weight_sum));
  } else {
    double mean_abs_error     = 0.0;
    double mean_squared_error = 0.0;
    for (std::size_t i = 0; i < params.n_rows; ++i) {
      double weight = sample_weight == nullptr ? 1.0 : double(h_sample_weight[i]);
      double diff   = double(h_y[i]) - double(h_pred[i]);
      weight_sum += weight;
      mean_abs_error += weight * std::abs(diff);
      mean_squared_error += weight * diff * diff;
    }
    mean_abs_error /= weight_sum;
    mean_squared_error /= weight_sum;
    return set_rf_metrics_regression(mean_abs_error, mean_squared_error, -1.0);
  }
}

template <typename DataT, typename LabelT>
auto TrainScore(const raft::handle_t& handle,
                RfTestParams params,
                DataT* X,
                DataT* X_transpose,
                LabelT* y,
                const double* sample_weight)
{
  RF_params rf_params = set_rf_params(params.max_depth,
                                      params.max_leaves,
                                      params.max_features,
                                      params.max_n_bins,
                                      params.min_samples_leaf,
                                      params.min_samples_split,
                                      params.min_impurity_decrease,
                                      params.bootstrap,
                                      params.n_trees,
                                      params.max_samples,
                                      0,
                                      params.split_criterion,
                                      params.n_streams,
                                      128);

  auto forest     = std::make_shared<RandomForestMetaData<DataT, LabelT>>();
  auto forest_ptr = forest.get();
  if constexpr (std::is_integral_v<LabelT>) {
    fit(handle,
        forest_ptr,
        X,
        params.n_rows,
        params.n_cols,
        y,
        params.n_labels,
        rf_params,
        rapids_logger::level_enum::info,
        nullptr,
        sample_weight,
        params.input_row_major);
  } else {
    fit(handle,
        forest_ptr,
        X,
        params.n_rows,
        params.n_cols,
        y,
        rf_params,
        rapids_logger::level_enum::info,
        nullptr,
        sample_weight,
        params.input_row_major);
  }

  auto pred = std::make_shared<thrust::device_vector<LabelT>>(params.n_rows);
  predict(handle, forest_ptr, X_transpose, params.n_rows, params.n_cols, pred->data().get());

  // Predict and compare against known labels
  RF_metrics metrics = Score(handle, params, y, pred->data().get(), sample_weight);
  return std::make_tuple(forest, pred, metrics);
}

template <typename DataT, typename LabelT>
class RfSpecialisedTest {
 public:
  RfSpecialisedTest(RfTestParams params) : params(params)
  {
    auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(params.n_streams);
    raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
    X.resize(params.n_rows * params.n_cols);
    X_transpose.resize(params.n_rows * params.n_cols);
    y.resize(params.n_rows);
    // Make data
    if constexpr (std::is_integral<LabelT>::value) {
      Datasets::make_blobs(handle,
                           X.data().get(),
                           y.data().get(),
                           params.n_rows,
                           params.n_cols,
                           params.n_labels,
                           false,
                           nullptr,
                           nullptr,
                           5.0,
                           false,
                           -10.0f,
                           10.0f,
                           params.seed);
    } else {
      thrust::device_vector<int> y_temp(params.n_rows);
      Datasets::make_blobs(handle,
                           X.data().get(),
                           y_temp.data().get(),
                           params.n_rows,
                           params.n_cols,
                           params.n_labels,
                           false,
                           nullptr,
                           nullptr,
                           5.0,
                           false,
                           -10.0f,
                           10.0f,
                           params.seed);
      // if regression, make the labels normally distributed
      raft::random::Rng r(4);
      thrust::device_vector<double> normal(params.n_rows);
      r.normal(normal.data().get(), normal.size(), 0.0, 2.0, nullptr);
      thrust::transform(
        normal.begin(), normal.end(), y_temp.begin(), y.begin(), cuda::std::plus<LabelT>());
    }
    raft::linalg::transpose(
      handle, X.data().get(), X_transpose.data().get(), params.n_rows, params.n_cols, nullptr);
    if (params.sample_weight) {
      thrust::host_vector<double> h_sample_weight(params.n_rows);
      for (std::size_t i = 0; i < params.n_rows; ++i) {
        int bucket         = (int(i) * 37 + params.seed * 13) % 17;
        h_sample_weight[i] = 0.25 + double(bucket) * 0.125;
      }
      sample_weight = h_sample_weight;
    }
    forest.reset(new typename ML::RandomForestMetaData<DataT, LabelT>);
    std::tie(forest, predictions, training_metrics) = TrainScore(handle,
                                                                 params,
                                                                 TrainingInputPtr(),
                                                                 X_transpose.data().get(),
                                                                 y.data().get(),
                                                                 SampleWeightPtr());

    Test();
  }
  // Current model should be at least as accurate as a model with depth - 1
  void TestAccuracyImprovement()
  {
    if (params.max_depth <= 1) { return; }
    // avereraging between models can introduce variance
    if (params.n_trees > 1) { return; }
    // accuracy is not guaranteed to improve with bootstrapping
    if (params.bootstrap) { return; }
    auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(params.n_streams);
    raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
    RfTestParams alt_params = params;
    alt_params.max_depth--;
    auto [alt_forest, alt_predictions, alt_metrics] = TrainScore(handle,
                                                                 alt_params,
                                                                 TrainingInputPtr(),
                                                                 X_transpose.data().get(),
                                                                 y.data().get(),
                                                                 SampleWeightPtr());
    double eps                                      = 1e-8;
    if (params.split_criterion == MSE) {
      EXPECT_LE(training_metrics.mean_squared_error, alt_metrics.mean_squared_error + eps);
    } else if (params.split_criterion == MAE) {
      EXPECT_LE(training_metrics.mean_abs_error, alt_metrics.mean_abs_error + eps);
    } else {
      EXPECT_GE(training_metrics.accuracy, alt_metrics.accuracy);
    }
  }
  // Regularisation parameters are working correctly
  void TestTreeSize()
  {
    for (int i = 0u; i < forest->rf_params.n_trees; i++) {
      // Check we have actually built something, otherwise these tests can all pass when the tree
      // algorithm produces only stumps
      size_t effective_rows = params.n_rows * params.max_samples;
      if (params.max_depth > 0 && params.min_impurity_decrease == 0 && effective_rows >= 100) {
        EXPECT_GT(forest->trees[i]->leaf_counter, 1);
      }

      // Check number of leaves is accurate
      int num_leaves = 0;
      for (auto n : forest->trees[i]->sparsetree) {
        num_leaves += n.IsLeaf();
      }
      EXPECT_EQ(num_leaves, forest->trees[i]->leaf_counter);
      if (params.max_leaves > 0) { EXPECT_LE(forest->trees[i]->leaf_counter, params.max_leaves); }

      EXPECT_LE(forest->trees[i]->depth_counter, params.max_depth);
      EXPECT_LE(forest->trees[i]->leaf_counter,
                raft::ceildiv(int(params.n_rows), params.min_samples_leaf));
    }
  }

  void TestMinImpurity()
  {
    for (int i = 0u; i < forest->rf_params.n_trees; i++) {
      for (auto n : forest->trees[i]->sparsetree) {
        if (!n.IsLeaf()) { EXPECT_GT(n.BestMetric(), params.min_impurity_decrease); }
      }
    }
  }

  void TestDeterminism()
  {
    // Regression models use floating point atomics, so are not bitwise reproducible
    bool is_regression = params.split_criterion != GINI and params.split_criterion != ENTROPY;
    if (is_regression) return;
    // Weighted classification uses floating-point histogram accumulation.
    if (params.sample_weight) return;

    // Repeat training
    auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(params.n_streams);
    raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
    auto [alt_forest, alt_predictions, alt_metrics] = TrainScore(handle,
                                                                 params,
                                                                 TrainingInputPtr(),
                                                                 X_transpose.data().get(),
                                                                 y.data().get(),
                                                                 SampleWeightPtr());

    for (int i = 0u; i < forest->rf_params.n_trees; i++) {
      EXPECT_EQ(forest->trees[i]->sparsetree, alt_forest->trees[i]->sparsetree);
    }
  }
  // Instance counts in children sums up to parent.
  void TestInstanceCounts()
  {
    for (int i = 0u; i < forest->rf_params.n_trees; i++) {
      const auto& tree = forest->trees[i]->sparsetree;
      for (auto n : tree) {
        if (!n.IsLeaf()) {
          auto sum = tree[n.LeftChildId()].InstanceCount() + tree[n.RightChildId()].InstanceCount();
          EXPECT_EQ(sum, n.InstanceCount());
        }
      }
    }

    // Bootstrap row samples are not exposed after fit, so reconstruct only full-data trees.
    if (params.bootstrap) { return; }

    thrust::host_vector<DataT> h_X = X;
    std::vector<int> rows(params.n_rows);
    for (std::size_t row = 0; row < params.n_rows; ++row) {
      rows[row] = row;
    }

    for (int i = 0u; i < forest->rf_params.n_trees; i++) {
      const auto& tree = forest->trees[i]->sparsetree;
      ASSERT_FALSE(tree.empty());
      ASSERT_EQ(static_cast<std::size_t>(tree.front().InstanceCount()), params.n_rows);
      ExpectNodeCountsMatchTrainingData(tree, 0, rows, h_X);
    }
  }

  void ExpectNodeCountsMatchTrainingData(const std::vector<SparseTreeNode<DataT, LabelT>>& tree,
                                         std::size_t node_id,
                                         const std::vector<int>& rows,
                                         const thrust::host_vector<DataT>& h_X)
  {
    ASSERT_LT(node_id, tree.size());
    const auto& node = tree[node_id];
    EXPECT_EQ(rows.size(), static_cast<std::size_t>(node.InstanceCount()));
    if (node.IsLeaf()) { return; }

    ASSERT_GE(node.LeftChildId(), 0);
    ASSERT_GE(node.RightChildId(), 0);
    const auto left_id  = static_cast<std::size_t>(node.LeftChildId());
    const auto right_id = static_cast<std::size_t>(node.RightChildId());
    ASSERT_LT(left_id, tree.size());
    ASSERT_LT(right_id, tree.size());

    std::vector<int> left_rows;
    std::vector<int> right_rows;
    left_rows.reserve(rows.size());
    right_rows.reserve(rows.size());
    for (auto row : rows) {
      const auto col_idx = static_cast<std::size_t>(node.ColumnId()) * params.n_rows + row;
      (h_X[col_idx] <= node.QueryValue() ? left_rows : right_rows).push_back(row);
    }

    EXPECT_EQ(left_rows.size(), static_cast<std::size_t>(tree[left_id].InstanceCount()));
    EXPECT_EQ(right_rows.size(), static_cast<std::size_t>(tree[right_id].InstanceCount()));
    ExpectNodeCountsMatchTrainingData(tree, left_id, left_rows, h_X);
    ExpectNodeCountsMatchTrainingData(tree, right_id, right_rows, h_X);
  }

  // Difference between the largest element and second largest
  DataT MinDifference(DataT* begin, std::size_t len)
  {
    std::size_t max_element_index = 0;
    DataT max_element             = 0.0;
    for (std::size_t i = 0; i < len; i++) {
      if (begin[i] > max_element) {
        max_element_index = i;
        max_element       = begin[i];
      }
    }
    DataT second_max_element = 0.0;
    for (std::size_t i = 0; i < len; i++) {
      if (begin[i] > second_max_element && i != max_element_index) {
        second_max_element = begin[i];
      }
    }

    return std::abs(max_element - second_max_element);
  }

  // Compare nvForest against native rf predictions
  // Only for single precision models
  void TestNvForestPredict()
  {
    if constexpr (std::is_same_v<DataT, double>) {
      return;
    } else {
      auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(params.n_streams);
      raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
      auto nvforest_pred = nvForestPredict(handle, params, X_transpose.data().get(), forest.get());

      thrust::host_vector<float> h_nvforest_pred(*nvforest_pred);
      thrust::host_vector<float> h_pred(*predictions);

      thrust::host_vector<float> h_nvforest_pred_prob;
      if constexpr (std::is_integral_v<LabelT>) {
        h_nvforest_pred_prob =
          *nvForestPredictProba(handle, params, X_transpose.data().get(), forest.get());
      }

      float tol = 1e-2;
      for (std::size_t i = 0; i < h_nvforest_pred.size(); i++) {
        // If the output probabilities are very similar for different classes
        // nvForest may output a different class due to numerical differences
        // Skip these cases
        if constexpr (std::is_integral_v<LabelT>) {
          int num_outputs = forest->trees[0]->num_outputs;
          auto min_diff   = MinDifference(&h_nvforest_pred_prob[i * num_outputs], num_outputs);
          if (min_diff < tol) continue;
        }

        EXPECT_LE(abs(h_nvforest_pred[i] - h_pred[i]), tol);
      }
    }
  }

  void TestFeatureImportances()
  {
    // Test feature importances for both regression and classification
    std::vector<DataT> importances(params.n_cols);
    ML::compute_feature_importances(forest.get(), importances.data());

    // Basic checks for feature importances
    EXPECT_EQ(importances.size(), static_cast<size_t>(params.n_cols));

    bool has_splits = false;
    for (int i = 0; i < forest->rf_params.n_trees; i++) {
      if (forest->trees[i]->leaf_counter > 1) {
        has_splits = true;
        break;
      }
    }

    if (!has_splits) {
      for (auto v : importances) {
        EXPECT_EQ(v, 0.0);
      }
      return;
    }

    double sum = 0.0;
    for (auto v : importances) {
      EXPECT_GE(v, 0.0);
      sum += v;
    }

    EXPECT_NEAR(sum, 1.0, 1e-6);
  }

  const double* SampleWeightPtr() const
  {
    return params.sample_weight ? sample_weight.data().get() : nullptr;
  }

  DataT* TrainingInputPtr()
  {
    return params.input_row_major ? X_transpose.data().get() : X.data().get();
  }

  void Test()
  {
    TestAccuracyImprovement();
    TestDeterminism();
    TestMinImpurity();
    TestTreeSize();
    TestInstanceCounts();
    TestNvForestPredict();
    TestFeatureImportances();
  }

  RF_metrics training_metrics;
  thrust::device_vector<DataT> X;
  thrust::device_vector<DataT> X_transpose;
  thrust::device_vector<LabelT> y;
  thrust::device_vector<double> sample_weight;
  RfTestParams params;
  std::shared_ptr<RandomForestMetaData<DataT, LabelT>> forest;
  std::shared_ptr<thrust::device_vector<LabelT>> predictions;
};

// Dispatch tests based on any template parameters
class RfTest : public ::testing::TestWithParam<RfTestParams> {
 public:
  void SetUp() override
  {
    RfTestParams params = ::testing::TestWithParam<RfTestParams>::GetParam();
    bool is_regression  = params.split_criterion != GINI and params.split_criterion != ENTROPY;
    if (params.double_precision) {
      if (is_regression) {
        RfSpecialisedTest<double, double> test(params);
      } else {
        RfSpecialisedTest<double, int> test(params);
      }
    } else {
      if (is_regression) {
        RfSpecialisedTest<float, float> test(params);
      } else {
        RfSpecialisedTest<float, int> test(params);
      }
    }
  }
};

TEST_P(RfTest, PropertyBasedTest) {}

// Parameter ranges to test
std::vector<int> n_rows                  = {10, 100, 1452};
std::vector<int> n_cols                  = {1, 5, 152, 1014};
std::vector<int> n_trees                 = {1, 5, 17};
std::vector<float> max_features          = {0.1f, 0.5f, 1.0f};
std::vector<float> max_samples           = {0.1f, 0.5f, 1.0f};
std::vector<int> max_depth               = {1, 10, 30};
std::vector<int> max_leaves              = {-1, 16, 50};
std::vector<bool> bootstrap              = {false, true};
std::vector<int> max_n_bins              = {2, 57, 128, 256};
std::vector<int> min_samples_leaf        = {1, 10, 30};
std::vector<int> min_samples_split       = {2, 10};
std::vector<float> min_impurity_decrease = {0.0f, 1.0f, 10.0f};
std::vector<int> n_streams               = {1, 2, 10};
std::vector<CRITERION> split_criterion   = {CRITERION::INVERSE_GAUSSIAN,
                                            CRITERION::GAMMA,
                                            CRITERION::POISSON,
                                            CRITERION::MSE,
                                            CRITERION::GINI,
                                            CRITERION::ENTROPY};
std::vector<int> seed                    = {0, 17};
std::vector<int> n_labels                = {2, 10, 20};
std::vector<bool> sample_weight          = {false, true};
std::vector<bool> double_precision       = {false, true};
std::vector<bool> input_row_major        = {false, true};

int n_tests = 100;

INSTANTIATE_TEST_CASE_P(RfTests,
                        RfTest,
                        ::testing::ValuesIn(SampleParameters<RfTestParams>(n_tests,
                                                                           0,
                                                                           n_rows,
                                                                           n_cols,
                                                                           n_trees,
                                                                           max_features,
                                                                           max_samples,
                                                                           max_depth,
                                                                           max_leaves,
                                                                           bootstrap,
                                                                           max_n_bins,
                                                                           min_samples_leaf,
                                                                           min_samples_split,
                                                                           min_impurity_decrease,
                                                                           n_streams,
                                                                           split_criterion,
                                                                           seed,
                                                                           n_labels,
                                                                           sample_weight,
                                                                           double_precision,
                                                                           input_row_major)));

TEST(RfTests, FeatureImportancesHandleInfiniteMetrics)
{
  // This direct metadata fixture covers the old inf / inf normalization path,
  // which returned NaN feature importances even though the splits were valid.
  auto forest        = std::make_shared<RandomForestMetaData<float, float>>();
  forest->n_features = 2;

  auto tree           = std::make_shared<DT::TreeMetaDataNode<float, float>>();
  tree->treeid        = 0;
  tree->depth_counter = 2;
  tree->leaf_counter  = 3;
  tree->train_time    = 0.0;
  tree->num_outputs   = 1;
  tree->sparsetree    = {SparseTreeNode<float, float>::CreateSplitNode(
                        0, 0.5f, std::numeric_limits<float>::infinity(), 1, 8),
                         SparseTreeNode<float, float>::CreateSplitNode(
                        1, 1.5f, std::numeric_limits<float>::infinity(), 3, 4),
                         SparseTreeNode<float, float>::CreateLeafNode(4),
                         SparseTreeNode<float, float>::CreateLeafNode(2),
                         SparseTreeNode<float, float>::CreateLeafNode(2)};
  forest->trees.push_back(tree);

  std::vector<float> importances(forest->n_features);
  ML::compute_feature_importances(forest.get(), importances.data());
  EXPECT_TRUE(std::isfinite(importances[0]));
  EXPECT_TRUE(std::isfinite(importances[1]));
  EXPECT_FLOAT_EQ(importances[0], 0.5f);
  EXPECT_FLOAT_EQ(importances[1], 0.5f);
}

TEST(RfTests, InvalidNStreams)
{
  for (auto n_streams : {0, -1}) {
    EXPECT_THROW(
      set_rf_params(3, 100, 1.0, 256, 1, 2, 0.0, false, 1, 1.0, 0, CRITERION::MSE, n_streams, 128),
      raft::exception);
  }
}

TEST(RfTests, IntegerOverflow)
{
  std::size_t m = 1000000;
  std::size_t n = 2150;
  EXPECT_GE(m * n, 1ull << 31);
  thrust::device_vector<float> X(m * n);
  thrust::device_vector<float> y(m);
  raft::random::Rng r(4);
  r.normal(X.data().get(), X.size(), 0.0f, 2.0f, nullptr);
  r.normal(y.data().get(), y.size(), 0.0f, 2.0f, nullptr);
  auto forest      = std::make_shared<RandomForestMetaData<float, float>>();
  auto forest_ptr  = forest.get();
  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(4);
  raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
  RF_params rf_params =
    set_rf_params(3, 100, 1.0, 256, 1, 2, 0.0, false, 1, 1.0, 0, CRITERION::MSE, 4, 128);
  fit(handle, forest_ptr, X.data().get(), m, n, y.data().get(), rf_params);

  // Check we have actually learned something
  EXPECT_GT(forest->trees[0]->leaf_counter, 1);

  // See if nvForest overflows
  thrust::device_vector<float> pred(m);
  TreeliteModelHandle model;
  build_treelite_forest(&model, forest_ptr, n);

  auto nvforest_model = nvforest::import_from_treelite_handle(model,
                                                              nvforest::tree_layout::breadth_first,
                                                              128,
                                                              false,
                                                              nvforest::device_type::gpu,
                                                              handle.get_device(),
                                                              handle.get_next_usable_stream());
  handle.sync_stream();
  handle.sync_stream_pool();
  delete static_cast<treelite::Model*>(model);

  nvforest_model.predict(handle,
                         pred.data().get(),
                         X.data().get(),
                         m,
                         nvforest::device_type::gpu,
                         nvforest::device_type::gpu,
                         nvforest::infer_kind::default_kind,
                         1);
  handle.sync_stream();
  handle.sync_stream_pool();
}

TEST(RfTests, EmptyGlobalRowsRejected)
{
  thrust::device_vector<float> X(1);
  thrust::device_vector<float> y(1);
  auto forest      = std::make_shared<RandomForestMetaData<float, float>>();
  auto forest_ptr  = forest.get();
  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
  raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
  RF_params rf_params =
    set_rf_params(3, 100, 1.0, 16, 1, 2, 0.0, false, 1, 1.0, 0, CRITERION::MSE, 1, 128);

  EXPECT_THROW(fit(handle, forest_ptr, X.data().get(), 0, 1, y.data().get(), rf_params),
               raft::exception);
}

TEST(RfTests, HighClassCountSplitHistogramFallsBackToGlobalMemory)
{
  constexpr std::size_t n_rows = 640;
  constexpr std::size_t n_cols = 4;
  constexpr int n_classes      = 80;
  constexpr int max_n_bins     = 256;

  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
  raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
  thrust::device_vector<float> X(n_rows * n_cols);
  thrust::device_vector<int> y(n_rows);

  Datasets::make_blobs(handle,
                       X.data().get(),
                       y.data().get(),
                       n_rows,
                       n_cols,
                       n_classes,
                       false,
                       nullptr,
                       nullptr,
                       5.0,
                       false,
                       -10.0f,
                       10.0f,
                       1234);

  auto forest = std::make_shared<RandomForestMetaData<float, int>>();
  auto rf_params =
    set_rf_params(2, -1, 1.0f, max_n_bins, 1, 2, 0.0f, false, 1, 1.0f, 1234, CRITERION::GINI, 1, 4);
  auto forest_ptr = forest.get();

  ASSERT_NO_THROW(
    fit(handle, forest_ptr, X.data().get(), n_rows, n_cols, y.data().get(), n_classes, rf_params));
  ASSERT_EQ(forest->trees.size(), 1);
  EXPECT_GT(forest->trees.front()->depth_counter, 0);
}

TEST(RfTests, InvalidSampleWeightThrows)
{
  constexpr std::size_t n_rows = 16;
  constexpr std::size_t n_cols = 2;

  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
  raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
  thrust::device_vector<float> X(n_rows * n_cols);
  thrust::device_vector<int> y(n_rows);
  thrust::device_vector<double> sample_weight(n_rows, 1.0);
  raft::random::Rng r(8);
  r.normal(X.data().get(), X.size(), 0.0f, 1.0f, handle.get_stream());
  thrust::host_vector<int> h_y(n_rows);
  for (std::size_t i = 0; i < n_rows; ++i) {
    h_y[i] = i % 2;
  }
  y = h_y;

  RF_params rf_params =
    set_rf_params(3, 100, 1.0, 8, 1, 2, 0.0, false, 1, 1.0, 0, CRITERION::GINI, 1, 128);

  auto expect_invalid_weight_throws = [&](double invalid_weight) {
    thrust::fill(
      thrust::cuda::par.on(handle.get_stream()), sample_weight.begin(), sample_weight.end(), 1.0);
    sample_weight[0] = invalid_weight;
    auto forest      = std::make_shared<RandomForestMetaData<float, int>>();
    auto forest_ptr  = forest.get();
    EXPECT_THROW(fit(handle,
                     forest_ptr,
                     X.data().get(),
                     n_rows,
                     n_cols,
                     y.data().get(),
                     2,
                     rf_params,
                     rapids_logger::level_enum::info,
                     nullptr,
                     sample_weight.data().get()),
                 raft::exception);
  };

  expect_invalid_weight_throws(-1.0);
  expect_invalid_weight_throws(std::numeric_limits<double>::quiet_NaN());

  thrust::fill(
    thrust::cuda::par.on(handle.get_stream()), sample_weight.begin(), sample_weight.end(), 0.0);
  auto forest     = std::make_shared<RandomForestMetaData<float, int>>();
  auto forest_ptr = forest.get();
  EXPECT_THROW(fit(handle,
                   forest_ptr,
                   X.data().get(),
                   n_rows,
                   n_cols,
                   y.data().get(),
                   2,
                   rf_params,
                   rapids_logger::level_enum::info,
                   nullptr,
                   sample_weight.data().get()),
               raft::exception);
}

TEST(RfTests, WeightedBootstrapSamplesOnlyPositiveWeightRows)
{
  constexpr int n_rows             = 32;
  constexpr int n_cols             = 2;
  constexpr int n_trees            = 8;
  constexpr int n_zero_weight_rows = 16;

  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(2);
  raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
  thrust::device_vector<float> X(n_rows * n_cols);
  thrust::device_vector<int> y(n_rows);
  thrust::device_vector<double> sample_weight(n_rows);

  raft::random::Rng r(8);
  r.normal(X.data().get(), X.size(), 0.0f, 1.0f, handle.get_stream());

  thrust::host_vector<int> h_y(n_rows);
  thrust::host_vector<double> h_sample_weight(n_rows);
  for (int i = 0; i < n_rows; ++i) {
    h_y[i]             = i < n_zero_weight_rows ? 0 : 1;
    h_sample_weight[i] = i < n_zero_weight_rows ? 0.0 : 1.0;
  }
  y             = h_y;
  sample_weight = h_sample_weight;

  RF_params rf_params =
    set_rf_params(0, 100, 1.0, 8, 1, 2, 0.0, true, n_trees, 1.0, 0, CRITERION::GINI, 2, 128);

  auto forest     = std::make_shared<RandomForestMetaData<float, int>>();
  auto forest_ptr = forest.get();
  rmm::device_uvector<bool> bootstrap_masks(std::size_t(n_trees) * n_rows, handle.get_stream());
  fit(handle,
      forest_ptr,
      X.data().get(),
      n_rows,
      n_cols,
      y.data().get(),
      2,
      rf_params,
      rapids_logger::level_enum::info,
      bootstrap_masks.data(),
      sample_weight.data().get());
  handle.sync_stream();

  thrust::host_vector<bool> h_bootstrap_masks(bootstrap_masks.size());
  raft::update_host(
    h_bootstrap_masks.data(), bootstrap_masks.data(), bootstrap_masks.size(), handle.get_stream());
  handle.sync_stream();

  for (int tree_id = 0; tree_id < n_trees; ++tree_id) {
    for (int row_id = 0; row_id < n_zero_weight_rows; ++row_id) {
      EXPECT_FALSE(h_bootstrap_masks[tree_id * n_rows + row_id]);
    }
  }

  for (auto const& tree_ptr : forest->trees) {
    const auto& tree = *tree_ptr;
    ASSERT_EQ(tree.sparsetree.size(), 1);
    EXPECT_TRUE(tree.sparsetree[0].IsLeaf());
    EXPECT_EQ(tree.sparsetree[0].InstanceCount(), n_rows);
    ASSERT_EQ(tree.vector_leaf.size(), 2);
    EXPECT_NEAR(tree.vector_leaf[0], 0.0f, 1e-6f);
    EXPECT_NEAR(tree.vector_leaf[1], 1.0f, 1e-6f);
  }
}

//-------------------------------------------------------------------------------------------------------------------------------------
struct QuantileTestParameters {
  int n_rows;
  int max_n_bins;
  uint64_t seed;
};

template <typename T>
class RFQuantileTest : public ::testing::TestWithParam<QuantileTestParameters> {
 public:
  void SetUp() override
  {
    auto params = ::testing::TestWithParam<QuantileTestParameters>::GetParam();

    thrust::device_vector<T> data(params.n_rows);

    raft::random::Rng r(8);
    r.normal(data.data().get(), data.size(), T(0.0), T(2.0), nullptr);
    auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
    raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);

    // computing the quantiles
    auto quantile_result =
      DT::computeQuantiles(handle, data.data().get(), params.max_n_bins, params.n_rows, 1);
    auto quantiles = quantile_result.view();

    int n_unique_bins;
    raft::copy(&n_unique_bins, quantiles.n_bins_array, 1, handle.get_stream());
    if (n_unique_bins < params.max_n_bins) { ASSERT_GT(n_unique_bins, 1); }
    ASSERT_LE(n_unique_bins, params.max_n_bins);

    thrust::host_vector<T> h_quantiles(params.max_n_bins);
    raft::update_host(
      h_quantiles.data(), quantiles.quantiles_array, params.max_n_bins, handle.get_stream());
    handle.sync_stream();
    for (int b = 1; b < n_unique_bins; b++) {
      ASSERT_LT(h_quantiles[b - 1], h_quantiles[b]);
    }
  }
};

// test to make sure that the quantiles and offsets calculated implement
// variable binning properly for categorical data, with unique values less than the `max_n_bins`
template <typename T>
class RFQuantileVariableBinsTest : public ::testing::TestWithParam<QuantileTestParameters> {
 public:
  void SetUp() override
  {
    auto params = ::testing::TestWithParam<QuantileTestParameters>::GetParam();
    srand(params.seed);

    auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
    raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
    thrust::device_vector<T> data(params.n_rows);

    // n_uniques guaranteed to be non-zero and smaller than `max_n_bins`
    int n_uniques;
    while ((n_uniques = rand() % params.max_n_bins) == 0) {}

    // populating random elements in data in [0, n_uniques)
    thrust::counting_iterator<float> first(0);
    thrust::copy(first, first + data.size(), data.begin());
    thrust::transform(data.begin(), data.end(), data.begin(), [=] __device__(auto& x) {
      x = T(int(x) % n_uniques);
      return x;
    });
    thrust::shuffle(data.begin(), data.end(), thrust::default_random_engine(n_uniques));

    // Use full-sample mode to verify duplicate compaction exactly.
    auto quantile_result = DT::computeQuantiles(
      handle, data.data().get(), params.max_n_bins, params.n_rows, 1, params.n_rows, params.seed);
    auto quantiles = quantile_result.view();
    int n_uniques_obtained;
    raft::copy(&n_uniques_obtained, quantile_result.n_bins_array.data(), 1, handle.get_stream());

    ASSERT_EQ(n_uniques_obtained, n_uniques) << "No. of unique bins is supposed to be " << n_uniques
                                             << ", but got " << n_uniques_obtained << std::endl;

    thrust::device_vector<int> histogram(n_uniques);
    thrust::host_vector<int> h_histogram(n_uniques);
    auto d_quantiles = quantiles.quantiles_array;
    auto d_histogram = histogram.data().get();

    // creating a cumulative histogram from data based on the quantiles
    // where histogram[i] has number of elements that are less-than-equal quantiles[i]
    thrust::for_each(data.begin(), data.end(), [=] __device__(T x) {
      for (int j = 0; j < n_uniques; j++) {
        if (x <= d_quantiles[j]) {
          atomicAdd(&d_histogram[j], 1);
          break;
        }
      }
    });

    // since the elements are randomly and equally distributed, we verify the calculated histogram
    h_histogram           = histogram;
    int max_items_per_bin = raft::ceildiv(params.n_rows, n_uniques);
    int min_items_per_bin = max_items_per_bin - 1;
    int total_items       = 0;
    for (int b = 0; b < n_uniques; b++) {
      ASSERT_TRUE(h_histogram[b] == max_items_per_bin or h_histogram[b] == min_items_per_bin)
        << "No. samples in bin[" << b << "] = " << h_histogram[b] << " Expected "
        << max_items_per_bin << " or " << min_items_per_bin << std::endl;
      total_items += h_histogram[b];
    }

    // recalculate the items for checking proper counting
    ASSERT_EQ(params.n_rows, total_items)
      << "Some samples from dataset are either missed of double counted in quantile bins"
      << std::endl;
  }
};

template <typename T>
class RFSampledQuantileExactFallbackTest : public ::testing::TestWithParam<QuantileTestParameters> {
 public:
  void SetUp() override
  {
    auto params = ::testing::TestWithParam<QuantileTestParameters>::GetParam();

    auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
    raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
    thrust::device_vector<T> data(params.n_rows);
    thrust::sequence(data.begin(), data.end(), T(0));

    auto sampled_quantile_result = DT::computeQuantiles(
      handle, data.data().get(), params.max_n_bins, params.n_rows, 1, params.n_rows, params.seed);
    auto sampled_quantiles = sampled_quantile_result.view();

    int sampled_n_bins;
    raft::copy(
      &sampled_n_bins, sampled_quantile_result.n_bins_array.data(), 1, handle.get_stream());
    handle.sync_stream();

    ASSERT_EQ(sampled_n_bins, params.max_n_bins);

    thrust::host_vector<T> h_sampled(params.max_n_bins);
    raft::update_host(
      h_sampled.data(), sampled_quantiles.quantiles_array, params.max_n_bins, handle.get_stream());
    handle.sync_stream();

    double bin_width = static_cast<double>(params.n_rows) / params.max_n_bins;
    for (int bin = 0; bin < sampled_n_bins; ++bin) {
      int idx = int(round((bin + 1) * bin_width)) - 1;
      idx     = std::min(std::max(0, idx), params.n_rows - 1);
      ASSERT_EQ(h_sampled[bin], T(idx));
    }
  }
};

template <typename T>
class RFSampledQuantileDeterminismTest : public ::testing::TestWithParam<QuantileTestParameters> {
 public:
  void SetUp() override
  {
    auto params = ::testing::TestWithParam<QuantileTestParameters>::GetParam();

    auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
    raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
    thrust::device_vector<T> data(params.n_rows);
    raft::random::Rng r(params.seed);
    r.normal(data.data().get(), data.size(), T(0.0), T(2.0), nullptr);

    auto quantile_result_a = DT::computeQuantiles(
      handle, data.data().get(), params.max_n_bins, params.n_rows, 1, 4, params.seed);
    auto quantile_result_b = DT::computeQuantiles(
      handle, data.data().get(), params.max_n_bins, params.n_rows, 1, 4, params.seed);
    auto quantiles_a = quantile_result_a.view();
    auto quantiles_b = quantile_result_b.view();

    int n_bins_a;
    int n_bins_b;
    raft::copy(&n_bins_a, quantile_result_a.n_bins_array.data(), 1, handle.get_stream());
    raft::copy(&n_bins_b, quantile_result_b.n_bins_array.data(), 1, handle.get_stream());
    handle.sync_stream();

    ASSERT_EQ(n_bins_a, n_bins_b);
    ASSERT_GT(n_bins_a, 1);
    ASSERT_LE(n_bins_a, params.max_n_bins);

    thrust::host_vector<T> h_quantiles_a(params.max_n_bins);
    thrust::host_vector<T> h_quantiles_b(params.max_n_bins);
    raft::update_host(
      h_quantiles_a.data(), quantiles_a.quantiles_array, params.max_n_bins, handle.get_stream());
    raft::update_host(
      h_quantiles_b.data(), quantiles_b.quantiles_array, params.max_n_bins, handle.get_stream());
    handle.sync_stream();

    for (int i = 0; i < n_bins_a; ++i) {
      ASSERT_EQ(h_quantiles_a[i], h_quantiles_b[i]);
      if (i > 0) { ASSERT_LT(h_quantiles_a[i - 1], h_quantiles_a[i]); }
    }
  }
};

template <typename ObjectiveT, typename BinT, typename DataT>
__global__ void objectiveGainKernel(BinT const* hist,
                                    DataT const* quantiles,
                                    DT::Split<DataT>* out,
                                    int* mutex,
                                    ObjectiveT objective,
                                    std::int64_t col,
                                    std::int64_t len,
                                    std::int64_t n_bins)
{
  __shared__ __align__(
    alignof(DT::Split<DataT>)) unsigned char split_scratch_storage[sizeof(DT::Split<DataT>)];
  auto* split_scratch = reinterpret_cast<DT::Split<DataT>*>(split_scratch_storage);
  if (threadIdx.x == 0) {
    *out   = DT::Split<DataT>();
    *mutex = 0;
  }
  __syncthreads();

  auto split = objective.Gain(hist, quantiles, col, len, n_bins);
  __syncthreads();
  split.evalBestSplit(split_scratch, out, mutex, quantiles, n_bins);
}

TEST(RFEquivalentSplitRangeTest, ClassificationChoosesUpperMiddleBin)
{
  using DataT                   = float;
  constexpr std::int64_t len    = 8;
  constexpr std::int64_t n_bins = 6;

  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
  raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);

  std::vector<DT::ClassificationBin> h_hist = {
    {2},
    {4},
    {4},
    {4},
    {4},
    {4},
    {0},
    {0},
    {0},
    {0},
    {0},
    {4},
  };
  std::vector<DataT> h_quantiles = {0, 1, 2, 3, 4, 5};

  thrust::device_vector<DT::ClassificationBin> hist(h_hist.begin(), h_hist.end());
  thrust::device_vector<DataT> quantiles(h_quantiles.begin(), h_quantiles.end());
  thrust::device_vector<DT::Split<DataT>> split(1);
  thrust::device_vector<int> mutex(1);

  DT::ClassificationObjectiveFunction<DataT, int> objective(2, 1, CRITERION::GINI);
  objectiveGainKernel<<<1, 32, 0, handle.get_stream()>>>(hist.data().get(),
                                                         quantiles.data().get(),
                                                         split.data().get(),
                                                         mutex.data().get(),
                                                         objective,
                                                         std::int64_t{0},
                                                         len,
                                                         n_bins);
  RAFT_CUDA_TRY(cudaGetLastError());

  DT::Split<DataT> h_split;
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    &h_split, split.data().get(), sizeof(h_split), cudaMemcpyDeviceToHost, handle.get_stream()));
  handle.sync_stream();

  EXPECT_EQ(h_split.global_nLeft, 4);
  EXPECT_EQ(h_split.local_nLeft, 0);
  EXPECT_EQ(h_split.quesval, DataT{3});
  EXPECT_EQ(h_split.split_start, 3);
  EXPECT_EQ(h_split.split_end, 3);
}

TEST(RFEquivalentSplitRangeTest, RegressionChoosesUpperMiddleBin)
{
  using DataT                   = float;
  constexpr std::int64_t len    = 8;
  constexpr std::int64_t n_bins = 6;

  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
  raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);

  std::vector<DT::RegressionBin> h_hist = {
    {0.0, 2},
    {0.0, 4},
    {0.0, 4},
    {0.0, 4},
    {0.0, 4},
    {8.0, 8},
  };
  std::vector<DataT> h_quantiles = {0, 1, 2, 3, 4, 5};

  thrust::device_vector<DT::RegressionBin> hist(h_hist.begin(), h_hist.end());
  thrust::device_vector<DataT> quantiles(h_quantiles.begin(), h_quantiles.end());
  thrust::device_vector<DT::Split<DataT>> split(1);
  thrust::device_vector<int> mutex(1);

  DT::RegressionObjectiveFunction<DataT, DataT> objective(1, 1, CRITERION::MSE);
  objectiveGainKernel<<<1, 32, 0, handle.get_stream()>>>(hist.data().get(),
                                                         quantiles.data().get(),
                                                         split.data().get(),
                                                         mutex.data().get(),
                                                         objective,
                                                         std::int64_t{0},
                                                         len,
                                                         n_bins);
  RAFT_CUDA_TRY(cudaGetLastError());

  DT::Split<DataT> h_split;
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    &h_split, split.data().get(), sizeof(h_split), cudaMemcpyDeviceToHost, handle.get_stream()));
  handle.sync_stream();

  EXPECT_EQ(h_split.global_nLeft, 4);
  EXPECT_EQ(h_split.local_nLeft, 0);
  EXPECT_EQ(h_split.quesval, DataT{3});
  EXPECT_EQ(h_split.split_start, 3);
  EXPECT_EQ(h_split.split_end, 3);
}

template <typename T>
class RFSampledQuantileRankErrorTest : public ::testing::TestWithParam<QuantileTestParameters> {
 public:
  void SetUp() override
  {
    auto params = ::testing::TestWithParam<QuantileTestParameters>::GetParam();

    auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
    raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
    thrust::device_vector<T> data(params.n_rows);
    thrust::sequence(data.begin(), data.end(), T(0));

    auto quantile_result = DT::computeQuantiles(
      handle, data.data().get(), params.max_n_bins, params.n_rows, 1, 4, params.seed);
    auto quantiles = quantile_result.view();

    int n_bins;
    raft::copy(&n_bins, quantile_result.n_bins_array.data(), 1, handle.get_stream());
    handle.sync_stream();

    ASSERT_EQ(n_bins, params.max_n_bins);

    thrust::host_vector<T> h_quantiles(params.max_n_bins);
    raft::update_host(
      h_quantiles.data(), quantiles.quantiles_array, params.max_n_bins, handle.get_stream());
    handle.sync_stream();

    double total_abs_rank_error = 0.0;
    double max_abs_rank_error   = 0.0;
    for (int bin = 0; bin < n_bins; ++bin) {
      double expected_rank = static_cast<double>(bin + 1) / params.max_n_bins;
      double actual_rank   = (static_cast<double>(h_quantiles[bin]) + 1.0) / params.n_rows;
      double rank_error    = std::abs(actual_rank - expected_rank);
      total_abs_rank_error += rank_error;
      max_abs_rank_error = std::max(max_abs_rank_error, rank_error);
    }

    double mean_abs_rank_error = total_abs_rank_error / n_bins;
    double sample_count        = static_cast<double>(params.max_n_bins * 4);
    double rank_error_scale    = 1.0 / std::sqrt(sample_count);
    EXPECT_LT(mean_abs_rank_error, rank_error_scale);
    EXPECT_LT(max_abs_rank_error, 3.0 * rank_error_scale);
  }
};

const std::vector<QuantileTestParameters> inputs = {{1000, 16, 6078587519764079670LLU},
                                                    {1130, 32, 4884670006177930266LLU},
                                                    {1752, 67, 9175325892580481371LLU},
                                                    {2307, 99, 9507819643927052255LLU},
                                                    {5000, 128, 9507819643927052255LLU}};

const std::vector<QuantileTestParameters> rank_error_inputs = {
  {10000, 128, 9507819643927052255LLU}};

// Check that all possible global samples can be selected, including the
// one-row partition when drawing a single sample from {1, 9} rows.
TEST(RFSampledQuantileGlobalSamplingTest, SmallPartitionCanReceiveSingleSample)
{
  std::vector<std::uint64_t> rank_row_offsets{0, 1, 10};
  constexpr std::uint64_t global_rows = 10;

  bool sampled_small_rank = false;
  for (std::uint64_t seed = 0; seed < 4096 && !sampled_small_rank; ++seed) {
    raft::random::UniformIntDistParams<std::uint64_t, std::uint64_t> uniform_int_dist_params;
    uniform_int_dist_params.start = 0;
    uniform_int_dist_params.end   = global_rows;
    uniform_int_dist_params.diff  = global_rows;
    raft::random::PCGenerator gen(seed, uint64_t{0}, uint64_t{0});
    std::uint64_t global_row;
    raft::random::custom_next(
      gen, &global_row, uniform_int_dist_params, std::uint64_t{0}, std::uint64_t{0});

    auto sample_rank =
      std::lower_bound(rank_row_offsets.begin() + 1, rank_row_offsets.end(), global_row + 1) -
      (rank_row_offsets.begin() + 1);
    ASSERT_LT(sample_rank + 1, rank_row_offsets.size());
    sampled_small_rank = sample_rank == 0;
  }

  ASSERT_TRUE(sampled_small_rank)
    << "A one-row partition should be reachable when drawing one global sample.";
}

// float type quantile test
typedef RFQuantileTest<float> RFQuantileTestF;
TEST_P(RFQuantileTestF, test) {}
INSTANTIATE_TEST_CASE_P(RfTests, RFQuantileTestF, ::testing::ValuesIn(inputs));

// double type quantile test
typedef RFQuantileTest<double> RFQuantileTestD;
TEST_P(RFQuantileTestD, test) {}
INSTANTIATE_TEST_CASE_P(RfTests, RFQuantileTestD, ::testing::ValuesIn(inputs));

// float type quantile variable binning test
typedef RFQuantileVariableBinsTest<float> RFQuantileVariableBinsTestF;
TEST_P(RFQuantileVariableBinsTestF, test) {}
INSTANTIATE_TEST_CASE_P(RfTests, RFQuantileVariableBinsTestF, ::testing::ValuesIn(inputs));

// double type quantile variable binning test
typedef RFQuantileVariableBinsTest<double> RFQuantileVariableBinsTestD;
TEST_P(RFQuantileVariableBinsTestD, test) {}
INSTANTIATE_TEST_CASE_P(RfTests, RFQuantileVariableBinsTestD, ::testing::ValuesIn(inputs));

typedef RFSampledQuantileExactFallbackTest<float> RFSampledQuantileExactFallbackTestF;
TEST_P(RFSampledQuantileExactFallbackTestF, test) {}
INSTANTIATE_TEST_CASE_P(RfTests, RFSampledQuantileExactFallbackTestF, ::testing::ValuesIn(inputs));

typedef RFSampledQuantileExactFallbackTest<double> RFSampledQuantileExactFallbackTestD;
TEST_P(RFSampledQuantileExactFallbackTestD, test) {}
INSTANTIATE_TEST_CASE_P(RfTests, RFSampledQuantileExactFallbackTestD, ::testing::ValuesIn(inputs));

typedef RFSampledQuantileDeterminismTest<float> RFSampledQuantileDeterminismTestF;
TEST_P(RFSampledQuantileDeterminismTestF, test) {}
INSTANTIATE_TEST_CASE_P(RfTests, RFSampledQuantileDeterminismTestF, ::testing::ValuesIn(inputs));

typedef RFSampledQuantileDeterminismTest<double> RFSampledQuantileDeterminismTestD;
TEST_P(RFSampledQuantileDeterminismTestD, test) {}
INSTANTIATE_TEST_CASE_P(RfTests, RFSampledQuantileDeterminismTestD, ::testing::ValuesIn(inputs));

typedef RFSampledQuantileRankErrorTest<float> RFSampledQuantileRankErrorTestF;
TEST_P(RFSampledQuantileRankErrorTestF, test) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        RFSampledQuantileRankErrorTestF,
                        ::testing::ValuesIn(rank_error_inputs));

typedef RFSampledQuantileRankErrorTest<double> RFSampledQuantileRankErrorTestD;
TEST_P(RFSampledQuantileRankErrorTestD, test) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        RFSampledQuantileRankErrorTestD,
                        ::testing::ValuesIn(rank_error_inputs));

//------------------------------------------------------------------------------------------------------

TEST(RfTest, TextDump)
{
  RF_params rf_params = set_rf_params(2, 2, 1.0, 2, 1, 2, 0.0, false, 1, 1.0, 0, GINI, 1, 128);
  auto forest         = std::make_shared<RandomForestMetaData<float, int>>();

  std::vector<float> X_host      = {1, 2, 3, 6, 7, 8};
  thrust::device_vector<float> X = X_host;
  std::vector<int> y_host        = {0, 0, 1, 1, 1, 0};
  thrust::device_vector<int> y   = y_host;

  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
  raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
  auto forest_ptr = forest.get();
  fit(handle, forest_ptr, X.data().get(), y.size(), 1, y.data().get(), 2, rf_params);

  std::string expected_start_text = R"(Forest has 1 trees, max_depth 2, and max_leaves 2
Tree #0
 Decision Tree depth --> 1 and n_leaves --> 2
 Tree Fitting - Overall time -->)";

  std::string expected_end_text = R"(└(colid: 0, quesval: 3, best_metric_val: 0.0555556)
    ├(leaf, prediction: [0.666667, 0.333333], best_metric_val: 0)
    └(leaf, prediction: [0.333333, 0.666667], best_metric_val: 0))";

  EXPECT_TRUE(get_rf_detailed_text(forest_ptr).find(expected_start_text) != std::string::npos);
  EXPECT_TRUE(get_rf_detailed_text(forest_ptr).find(expected_end_text) != std::string::npos);
  std::string expected_json = R"([
{"nodeid": 0, "split_feature": 0, "split_threshold": 3, "gain": 0.055555582, "instance_count": 6, "yes": 1, "no": 2, "children": [
  {"nodeid": 1, "leaf_value": [0.666666687, 0.333333343], "instance_count": 3},
  {"nodeid": 2, "leaf_value": [0.333333343, 0.666666687], "instance_count": 3}
]}
])";

  EXPECT_EQ(get_rf_json(forest_ptr), expected_json);
}

TEST(RfTest, EquivalentSplitRangePersistsThroughBuilder)
{
  RF_params rf_params = set_rf_params(2, 4, 1.0, 6, 1, 2, 0.0, false, 1, 1.0, 0, GINI, 1, 128);
  auto forest         = std::make_shared<RandomForestMetaData<float, int>>();

  // Column-major: feature 1 has global quantiles {0, 1, 2, 3, 4, 5},
  // but the left child only sees values {0, 5}. The child split therefore has
  // an equivalent threshold range that should persist as the centered value.
  std::vector<float> X_host = {0, 0, 0, 0, 10, 10, 10, 10, 10, 10, 0, 0, 5, 5, 1, 2, 2, 3, 3, 4};
  thrust::device_vector<float> X = X_host;
  std::vector<int> y_host        = {0, 1, 1, 1, 0, 0, 0, 0, 0, 0};
  thrust::device_vector<int> y   = y_host;

  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
  raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);
  auto forest_ptr = forest.get();
  fit(handle, forest_ptr, X.data().get(), y.size(), 2, y.data().get(), 2, rf_params);

  const auto& tree = forest->trees[0]->sparsetree;
  ASSERT_GE(tree.size(), std::size_t{5});

  const auto& root = tree[0];
  ASSERT_FALSE(root.IsLeaf());
  EXPECT_EQ(root.ColumnId(), 0);
  EXPECT_EQ(root.QueryValue(), 0.0f);
  EXPECT_EQ(root.InstanceCount(), 10);

  const auto& left_child = tree[root.LeftChildId()];
  ASSERT_FALSE(left_child.IsLeaf());
  EXPECT_EQ(left_child.ColumnId(), 1);
  EXPECT_EQ(left_child.QueryValue(), 2.0f);
  EXPECT_EQ(left_child.InstanceCount(), 4);

  EXPECT_EQ(tree[left_child.LeftChildId()].InstanceCount(), 2);
  EXPECT_EQ(tree[left_child.RightChildId()].InstanceCount(), 2);
  EXPECT_TRUE(tree[root.RightChildId()].IsLeaf());
  EXPECT_EQ(tree[root.RightChildId()].InstanceCount(), 6);
}

//-------------------------------------------------------------------------------------------------------------------------------------

TEST(RfWeightedTest, ClassificationRootLeafUsesWeights)
{
  RF_params rf_params = set_rf_params(0, -1, 1.0, 4, 1, 2, 0.0, false, 1, 1.0, 0, GINI, 1, 128);
  auto forest         = std::make_shared<RandomForestMetaData<float, int>>();

  std::vector<float> X_host             = {0.0f, 1.0f, 2.0f};
  thrust::device_vector<float> X        = X_host;
  std::vector<int> y_host               = {0, 1, 1};
  thrust::device_vector<int> y          = y_host;
  std::vector<double> weight_host       = {100.0f, 1.0f, 1.0f};
  thrust::device_vector<double> weights = weight_host;
  auto stream_pool                      = std::make_shared<rmm::cuda_stream_pool>(1);
  raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);

  fit(handle,
      forest.get(),
      X.data().get(),
      y.size(),
      1,
      y.data().get(),
      2,
      rf_params,
      rapids_logger::level_enum::info,
      nullptr,
      weights.data().get());

  ASSERT_EQ(forest->trees.size(), 1);
  const auto& tree = *forest->trees[0];
  ASSERT_EQ(tree.sparsetree.size(), 1);
  EXPECT_TRUE(tree.sparsetree[0].IsLeaf());
  EXPECT_EQ(tree.sparsetree[0].InstanceCount(), 3);
  ASSERT_EQ(tree.vector_leaf.size(), 2);
  EXPECT_NEAR(tree.vector_leaf[0], 100.0f / 102.0f, 1e-6f);
  EXPECT_NEAR(tree.vector_leaf[1], 2.0f / 102.0f, 1e-6f);
}

TEST(RfWeightedTest, RegressionRootLeafUsesWeights)
{
  RF_params rf_params = set_rf_params(0, -1, 1.0, 4, 1, 2, 0.0, false, 1, 1.0, 0, MSE, 1, 128);
  auto forest         = std::make_shared<RandomForestMetaData<float, float>>();

  std::vector<float> X_host             = {0.0f, 1.0f, 2.0f};
  thrust::device_vector<float> X        = X_host;
  std::vector<float> y_host             = {0.0f, 10.0f, 10.0f};
  thrust::device_vector<float> y        = y_host;
  std::vector<double> weight_host       = {1.0f, 0.0f, 3.0f};
  thrust::device_vector<double> weights = weight_host;
  auto stream_pool                      = std::make_shared<rmm::cuda_stream_pool>(1);
  raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);

  fit(handle,
      forest.get(),
      X.data().get(),
      y.size(),
      1,
      y.data().get(),
      rf_params,
      rapids_logger::level_enum::info,
      nullptr,
      weights.data().get());

  ASSERT_EQ(forest->trees.size(), 1);
  const auto& tree = *forest->trees[0];
  ASSERT_EQ(tree.sparsetree.size(), 1);
  EXPECT_TRUE(tree.sparsetree[0].IsLeaf());
  EXPECT_EQ(tree.sparsetree[0].InstanceCount(), 2);
  ASSERT_EQ(tree.vector_leaf.size(), 1);
  EXPECT_NEAR(tree.vector_leaf[0], 7.5f, 1e-6f);
}

TEST(RfWeightedTest, MinSamplesLeafUsesCountsNotWeights)
{
  RF_params rf_params = set_rf_params(1, -1, 1.0, 4, 2, 2, 0.0, false, 1, 1.0, 0, GINI, 1, 128);
  auto forest         = std::make_shared<RandomForestMetaData<float, int>>();

  std::vector<float> X_host             = {0.0f, 1.0f, 2.0f, 3.0f};
  thrust::device_vector<float> X        = X_host;
  std::vector<int> y_host               = {0, 0, 1, 1};
  thrust::device_vector<int> y          = y_host;
  std::vector<double> weight_host       = {0.1f, 0.1f, 100.0f, 100.0f};
  thrust::device_vector<double> weights = weight_host;
  auto stream_pool                      = std::make_shared<rmm::cuda_stream_pool>(1);
  raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);

  fit(handle,
      forest.get(),
      X.data().get(),
      y.size(),
      1,
      y.data().get(),
      2,
      rf_params,
      rapids_logger::level_enum::info,
      nullptr,
      weights.data().get());

  ASSERT_EQ(forest->trees.size(), 1);
  const auto& tree = *forest->trees[0];
  ASSERT_EQ(tree.sparsetree.size(), 3);
  EXPECT_FALSE(tree.sparsetree[0].IsLeaf());
  EXPECT_EQ(tree.sparsetree[0].InstanceCount(), 4);
  EXPECT_EQ(tree.sparsetree[1].InstanceCount(), 2);
  EXPECT_EQ(tree.sparsetree[2].InstanceCount(), 2);
  ASSERT_EQ(tree.vector_leaf.size(), 6);
  EXPECT_NEAR(tree.vector_leaf[2], 1.0f, 1e-6f);
  EXPECT_NEAR(tree.vector_leaf[3], 0.0f, 1e-6f);
  EXPECT_NEAR(tree.vector_leaf[4], 0.0f, 1e-6f);
  EXPECT_NEAR(tree.vector_leaf[5], 1.0f, 1e-6f);
}

TEST(RfWeightedTest, ZeroWeightSamplesDoNotCreatePositiveWeightSplit)
{
  RF_params rf_params = set_rf_params(1, -1, 1.0, 4, 1, 2, 0.0, false, 1, 1.0, 0, GINI, 1, 128);
  auto forest         = std::make_shared<RandomForestMetaData<float, int>>();

  std::vector<float> X_host             = {0.0f, 1.0f, 2.0f, 3.0f};
  thrust::device_vector<float> X        = X_host;
  std::vector<int> y_host               = {0, 0, 1, 1};
  thrust::device_vector<int> y          = y_host;
  std::vector<double> weight_host       = {0.0f, 0.0f, 1.0f, 1.0f};
  thrust::device_vector<double> weights = weight_host;
  auto stream_pool                      = std::make_shared<rmm::cuda_stream_pool>(1);
  raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);

  fit(handle,
      forest.get(),
      X.data().get(),
      y.size(),
      1,
      y.data().get(),
      2,
      rf_params,
      rapids_logger::level_enum::info,
      nullptr,
      weights.data().get());

  ASSERT_EQ(forest->trees.size(), 1);
  const auto& tree = *forest->trees[0];
  ASSERT_EQ(tree.sparsetree.size(), 1);
  EXPECT_TRUE(tree.sparsetree[0].IsLeaf());
  EXPECT_EQ(tree.sparsetree[0].InstanceCount(), 2);
  ASSERT_EQ(tree.vector_leaf.size(), 2);
  EXPECT_NEAR(tree.vector_leaf[0], 0.0f, 1e-6f);
  EXPECT_NEAR(tree.vector_leaf[1], 1.0f, 1e-6f);
}

TEST(RfWeightedTest, BootstrapDuplicatesContributePerOccurrence)
{
  std::vector<float> X_host             = {0.0f, 1.0f, 2.0f};
  thrust::device_vector<float> X        = X_host;
  std::vector<float> y_host             = {0.0f, 10.0f, 100.0f};
  thrust::device_vector<float> y        = y_host;
  std::vector<double> weight_host       = {1.0f, 2.0f, 5.0f};
  thrust::device_vector<double> weights = weight_host;
  auto stream_pool                      = std::make_shared<rmm::cuda_stream_pool>(1);
  raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);

  constexpr int n_rows = 3;
  bool found_duplicate = false;
  for (uint64_t seed = 0; seed < 64 && !found_duplicate; ++seed) {
    RF_params rf_params = set_rf_params(0, -1, 1.0, 3, 1, 2, 0.0, true, 1, 1.0, seed, MSE, 1, 128);
    auto forest         = std::make_shared<RandomForestMetaData<float, float>>();
    rmm::device_uvector<bool> bootstrap_masks(n_rows, handle.get_stream());

    fit(handle,
        forest.get(),
        X.data().get(),
        n_rows,
        1,
        y.data().get(),
        rf_params,
        rapids_logger::level_enum::info,
        bootstrap_masks.data(),
        weights.data().get());
    handle.sync_stream();

    std::array<bool, n_rows> mask{};
    raft::update_host(mask.data(), bootstrap_masks.data(), mask.size(), handle.get_stream());
    handle.sync_stream();

    std::array<int, 2> included{};
    int included_count = 0;
    for (int i = 0; i < n_rows; ++i) {
      if (mask[i]) {
        if (included_count < 2) { included[included_count] = i; }
        ++included_count;
      }
    }
    if (included_count != 2) { continue; }

    found_duplicate  = true;
    const auto& tree = *forest->trees[0];
    ASSERT_EQ(tree.sparsetree.size(), 1);
    EXPECT_TRUE(tree.sparsetree[0].IsLeaf());
    EXPECT_EQ(tree.sparsetree[0].InstanceCount(), n_rows);
    ASSERT_EQ(tree.vector_leaf.size(), 1);

    auto mean_with_counts = [&](int count_a, int count_b) {
      auto a         = included[0];
      auto b         = included[1];
      auto label_sum = y_host[a] * count_a + y_host[b] * count_b;
      auto count_sum = count_a + count_b;
      return label_sum / count_sum;
    };

    auto unique_mean      = mean_with_counts(1, 1);
    auto duplicate_a_mean = mean_with_counts(2, 1);
    auto duplicate_b_mean = mean_with_counts(1, 2);
    auto observed         = tree.vector_leaf[0];

    EXPECT_GT(std::abs(observed - unique_mean), 1e-5f);
    EXPECT_TRUE(std::abs(observed - duplicate_a_mean) < 1e-5f ||
                std::abs(observed - duplicate_b_mean) < 1e-5f);
  }

  EXPECT_TRUE(found_duplicate);
}

//-------------------------------------------------------------------------------------------------------------------------------------
namespace DT {

struct ObjectiveTestParameters {
  uint64_t seed;
  int n_rows;
  int max_n_bins;
  int n_classes;
  int min_samples_leaf;
  double tolerance;
};

template <typename ObjectiveT_, CRITERION Criterion_>
struct ObjectiveTestConfig {
  using ObjectiveT                         = ObjectiveT_;
  static constexpr CRITERION splitCriteria = Criterion_;
};

template <typename ObjectiveConfig>
class ObjectiveTest : public ::testing::TestWithParam<ObjectiveTestParameters> {
  using ObjectiveT = typename ObjectiveConfig::ObjectiveT;
  typedef typename ObjectiveT::DataT DataT;
  typedef typename ObjectiveT::LabelT LabelT;
  typedef typename ObjectiveT::BinT BinT;

  static constexpr auto eps_              = 10 * std::numeric_limits<DataT>::epsilon();
  static constexpr bool is_classification = std::is_same<BinT, ClassificationBin>::value ||
                                            std::is_same<BinT, WeightedClassificationBin>::value;
  static constexpr bool is_weighted = ObjectiveT::weighted;

  ObjectiveTestParameters params;
  std::mt19937_64 rng;

 public:
  auto RandUnder(int const end = 10000)
  {
    std::uniform_int_distribution<int> dist(0, end - 1);
    return dist(rng);
  }

  auto GenRandomData()
  {
    std::vector<DataT> data(params.n_rows);
    if constexpr (is_classification)  // classification case
    {
      for (auto& d : data) {
        d = RandUnder(params.n_classes);
      }
    } else {
      std::normal_distribution<DataT> normal(1.0, 2.0);
      for (auto& d : data) {
        auto rand_element{DataT(0)};
        while (1) {
          rand_element = normal(rng);
          if (rand_element > 0) break;  // only positive random numbers
        }
        d = rand_element;
      }
    }
    return data;
  }

  auto GenSampleWeights()
  {
    std::vector<double> sample_weights(params.n_rows, 1.0);
    if constexpr (is_weighted) {
      std::uniform_real_distribution<double> weight_dist(0.2, 3.0);
      for (auto& w : sample_weights) {
        w = weight_dist(rng);
      }
    }
    return sample_weights;
  }

  auto GenHist(std::vector<DataT> const& data, std::vector<double> const& sample_weights)
  {
    std::vector<BinT> cdf_hist, pdf_hist;
    auto bin_width = static_cast<std::size_t>(raft::ceildiv(params.n_rows, params.max_n_bins));

    for (auto c = 0; c < params.n_classes; ++c) {
      for (auto b = 0; b < params.max_n_bins; ++b) {
        auto bin_begin =
          std::min<std::size_t>(static_cast<std::size_t>(b) * bin_width, data.size());
        auto split_end = std::min<std::size_t>(bin_begin + bin_width, data.size());
        auto bin_count = static_cast<BinCountT>(split_end - bin_begin);
        if constexpr (is_classification) {
          auto count{BinCountT(0)};
          auto weight{DataT(0)};
          for (auto i = bin_begin; i < split_end; ++i) {
            if (data[i] == DataT(c)) {
              ++count;
              weight += sample_weights[i];
            }
          }
          if constexpr (is_weighted) {
            pdf_hist.emplace_back(count, weight);
          } else {
            pdf_hist.emplace_back(count);
          }
        } else {  // regression case
          auto label_sum{DataT(0)};
          auto weight{DataT(0)};
          for (auto i = bin_begin; i < split_end; ++i) {
            label_sum += data[i] * sample_weights[i];
            weight += sample_weights[i];
          }
          if constexpr (is_weighted) {
            pdf_hist.emplace_back(label_sum, bin_count, weight);
          } else {
            pdf_hist.emplace_back(label_sum, bin_count);
          }
        }

        auto cumulative = b > 0 ? cdf_hist.back() : BinT();
        cdf_hist.emplace_back(pdf_hist.empty() ? BinT() : pdf_hist.back());
        cdf_hist.back() += cumulative;
      }
    }

    return std::make_pair(cdf_hist, pdf_hist);
  }

  auto SplitOffset(std::size_t const split_bin_index)
  {
    auto bin_width = static_cast<std::size_t>(raft::ceildiv(params.n_rows, params.max_n_bins));
    return std::min<std::size_t>((split_bin_index + 1) * bin_width,
                                 static_cast<std::size_t>(params.n_rows));
  }

  auto SplitBinIndexUpperBound()
  {
    auto bin_width        = raft::ceildiv(params.n_rows, params.max_n_bins);
    auto non_empty_n_bins = raft::ceildiv(params.n_rows, bin_width);
    return std::max(1, non_empty_n_bins - 1);
  }

  auto MSE(std::vector<DataT> const& data,
           std::vector<double> const&
             sample_weights)  //  1/w * 1/2 * sum(w_i * (y - y_pred) * (y - y_pred))
  {
    DataT weight_sum = std::accumulate(sample_weights.begin(), sample_weights.end(), DataT(0));
    DataT sum{0};
    for (std::size_t i = 0; i < data.size(); ++i) {
      sum += data[i] * sample_weights[i];
    }
    DataT const mean = sum / weight_sum;
    auto mse{DataT(0.0)};  // mse: mean squared error

    for (std::size_t i = 0; i < data.size(); ++i) {
      auto d = data[i];
      mse += sample_weights[i] * (d - mean) * (d - mean);  // unit deviance
    }

    mse /= 2 * weight_sum;
    return std::make_tuple(mse, sum, DataT(data.size()), weight_sum);
  }

  auto MSEGroundTruthGain(std::vector<DataT> const& data,
                          std::vector<double> const& sample_weights,
                          std::size_t split_bin_index)
  {
    auto split_offset = SplitOffset(split_bin_index);
    std::vector<DataT> left_data(data.begin(), data.begin() + split_offset);
    std::vector<DataT> right_data(data.begin() + split_offset, data.end());
    std::vector<double> left_sample_weights(sample_weights.begin(),
                                            sample_weights.begin() + split_offset);
    std::vector<double> right_sample_weights(sample_weights.begin() + split_offset,
                                             sample_weights.end());

    auto [parent_mse, label_sum, n, weight_sum]              = MSE(data, sample_weights);
    auto [left_mse, label_sum_left, n_left, weight_sum_left] = MSE(left_data, left_sample_weights);
    auto [right_mse, label_sum_right, n_right, weight_sum_right] =
      MSE(right_data, right_sample_weights);

    auto gain = parent_mse -
                ((weight_sum_left / weight_sum) * left_mse +    // the minimizing objective function
                                                                // is half deviance
                 (weight_sum_right / weight_sum) * right_mse);  // gain in long form without proxy

    // edge cases
    if (n_left < params.min_samples_leaf or n_right < params.min_samples_leaf)
      return -std::numeric_limits<DataT>::max();
    else
      return gain;
  }

  auto InverseGaussianHalfDeviance(
    std::vector<DataT> const& data,
    std::vector<double> const& sample_weights)  //  1/w * 2 * sum(w_i * (y - y_pred) * (y -
                                                //  y_pred)/(y * (y_pred) * (y_pred)))
  {
    DataT weight_sum = std::accumulate(sample_weights.begin(), sample_weights.end(), DataT(0));
    DataT sum{0};
    for (std::size_t i = 0; i < data.size(); ++i) {
      sum += data[i] * sample_weights[i];
    }
    DataT const mean = sum / weight_sum;
    auto ighd{DataT(0.0)};  // ighd: inverse gaussian half deviance

    for (std::size_t i = 0; i < data.size(); ++i) {
      auto d = data[i];
      ighd += sample_weights[i] * (d - mean) * (d - mean) / (d * mean * mean);  // unit deviance
    }

    ighd /= 2 * weight_sum;
    return std::make_tuple(ighd, sum, DataT(data.size()), weight_sum);
  }

  auto InverseGaussianGroundTruthGain(std::vector<DataT> const& data,
                                      std::vector<double> const& sample_weights,
                                      std::size_t split_bin_index)
  {
    auto split_offset = SplitOffset(split_bin_index);
    std::vector<DataT> left_data(data.begin(), data.begin() + split_offset);
    std::vector<DataT> right_data(data.begin() + split_offset, data.end());
    std::vector<double> left_sample_weights(sample_weights.begin(),
                                            sample_weights.begin() + split_offset);
    std::vector<double> right_sample_weights(sample_weights.begin() + split_offset,
                                             sample_weights.end());

    auto [parent_ighd, label_sum, n, weight_sum] =
      InverseGaussianHalfDeviance(data, sample_weights);
    auto [left_ighd, label_sum_left, n_left, weight_sum_left] =
      InverseGaussianHalfDeviance(left_data, left_sample_weights);
    auto [right_ighd, label_sum_right, n_right, weight_sum_right] =
      InverseGaussianHalfDeviance(right_data, right_sample_weights);

    auto gain = parent_ighd -
                ((weight_sum_left / weight_sum) *
                   left_ighd +  // the minimizing objective function is half deviance
                 (weight_sum_right / weight_sum) * right_ighd);  // gain in long form without proxy

    // edge cases
    if (n_left < params.min_samples_leaf or n_right < params.min_samples_leaf or label_sum < eps_ or
        label_sum_right < eps_ or label_sum_left < eps_)
      return -std::numeric_limits<DataT>::max();
    else
      return gain;
  }

  auto GammaHalfDeviance(std::vector<DataT> const& data, std::vector<double> const& sample_weights)
  //  1/w * 2 * sum(w_i * (log(y_pred/y_true) + y_true/y_pred - 1))
  {
    DataT weight_sum = std::accumulate(sample_weights.begin(), sample_weights.end(), DataT(0));
    DataT sum(0);
    for (std::size_t i = 0; i < data.size(); ++i) {
      sum += data[i] * sample_weights[i];
    }
    DataT const mean = sum / weight_sum;
    DataT ghd(0);  // gamma half deviance

    for (std::size_t i = 0; i < data.size(); ++i) {
      auto& element = data[i];
      auto log_y    = raft::log(element ? element : DataT(1.0));
      ghd += sample_weights[i] * (raft::log(mean) - log_y + element / mean - 1);
    }

    ghd /= weight_sum;
    return std::make_tuple(ghd, sum, DataT(data.size()), weight_sum);
  }

  auto GammaGroundTruthGain(std::vector<DataT> const& data,
                            std::vector<double> const& sample_weights,
                            std::size_t split_bin_index)
  {
    auto split_offset = SplitOffset(split_bin_index);
    std::vector<DataT> left_data(data.begin(), data.begin() + split_offset);
    std::vector<DataT> right_data(data.begin() + split_offset, data.end());
    std::vector<double> left_sample_weights(sample_weights.begin(),
                                            sample_weights.begin() + split_offset);
    std::vector<double> right_sample_weights(sample_weights.begin() + split_offset,
                                             sample_weights.end());

    auto [parent_ghd, label_sum, n, weight_sum] = GammaHalfDeviance(data, sample_weights);
    auto [left_ghd, label_sum_left, n_left, weight_sum_left] =
      GammaHalfDeviance(left_data, left_sample_weights);
    auto [right_ghd, label_sum_right, n_right, weight_sum_right] =
      GammaHalfDeviance(right_data, right_sample_weights);

    auto gain = parent_ghd -
                ((weight_sum_left / weight_sum) * left_ghd +    // the minimizing objective function
                                                                // is half deviance
                 (weight_sum_right / weight_sum) * right_ghd);  // gain in long form without proxy

    // edge cases
    if (n_left < params.min_samples_leaf or n_right < params.min_samples_leaf or label_sum < eps_ or
        label_sum_right < eps_ or label_sum_left < eps_)
      return -std::numeric_limits<DataT>::max();
    else
      return gain;
  }

  auto PoissonHalfDeviance(
    std::vector<DataT> const& data,
    std::vector<double> const&
      sample_weights)  //  1/w * sum(w_i * (y_true * log(y_true/y_pred) + y_pred - y_true))
  {
    DataT weight_sum = std::accumulate(sample_weights.begin(), sample_weights.end(), DataT(0));
    DataT sum{0};
    for (std::size_t i = 0; i < data.size(); ++i) {
      sum += data[i] * sample_weights[i];
    }
    auto const mean = sum / weight_sum;
    auto poisson_half_deviance{DataT(0.0)};

    for (std::size_t i = 0; i < data.size(); ++i) {
      auto d     = data[i];
      auto log_y = raft::log(d ? d : DataT(1.0));  // we don't want nans
      poisson_half_deviance += sample_weights[i] * (d * (log_y - raft::log(mean)) + mean - d);
    }

    poisson_half_deviance /= weight_sum;
    return std::make_tuple(poisson_half_deviance, sum, DataT(data.size()), weight_sum);
  }

  auto PoissonGroundTruthGain(std::vector<DataT> const& data,
                              std::vector<double> const& sample_weights,
                              std::size_t split_bin_index)
  {
    auto split_offset = SplitOffset(split_bin_index);
    std::vector<DataT> left_data(data.begin(), data.begin() + split_offset);
    std::vector<DataT> right_data(data.begin() + split_offset, data.end());
    std::vector<double> left_sample_weights(sample_weights.begin(),
                                            sample_weights.begin() + split_offset);
    std::vector<double> right_sample_weights(sample_weights.begin() + split_offset,
                                             sample_weights.end());

    auto [parent_phd, label_sum, n, weight_sum] = PoissonHalfDeviance(data, sample_weights);
    auto [left_phd, label_sum_left, n_left, weight_sum_left] =
      PoissonHalfDeviance(left_data, left_sample_weights);
    auto [right_phd, label_sum_right, n_right, weight_sum_right] =
      PoissonHalfDeviance(right_data, right_sample_weights);

    auto gain = parent_phd -
                ((weight_sum_left / weight_sum) * left_phd +
                 (weight_sum_right / weight_sum) * right_phd);  // gain in long form without proxy

    // edge cases
    if (n_left < params.min_samples_leaf or n_right < params.min_samples_leaf or label_sum < eps_ or
        label_sum_right < eps_ or label_sum_left < eps_)
      return -std::numeric_limits<DataT>::max();
    else
      return gain;
  }

  auto Entropy(std::vector<DataT> const& data, std::vector<double> const& sample_weights)
  {  // sum((n_c/n_total)*(log(n_c/n_total)))
    DataT weight_sum = std::accumulate(sample_weights.begin(), sample_weights.end(), DataT(0));
    DataT entropy(0);
    for (auto c = 0; c < params.n_classes; ++c) {
      DataT sum(0);
      for (std::size_t i = 0; i < data.size(); ++i) {
        if (data[i] == DataT(c)) { sum += sample_weights[i]; }
      }
      DataT class_proba = sum / weight_sum;
      entropy += -class_proba * raft::log(class_proba ? class_proba : DataT(1)) /
                 raft::log(DataT(2));  // adding gain
    }
    return entropy;
  }

  auto EntropyGroundTruthGain(std::vector<DataT> const& data,
                              std::vector<double> const& sample_weights,
                              std::size_t const split_bin_index)
  {
    auto split_offset = SplitOffset(split_bin_index);
    std::vector<DataT> left_data(data.begin(), data.begin() + split_offset);
    std::vector<DataT> right_data(data.begin() + split_offset, data.end());
    std::vector<double> left_sample_weights(sample_weights.begin(),
                                            sample_weights.begin() + split_offset);
    std::vector<double> right_sample_weights(sample_weights.begin() + split_offset,
                                             sample_weights.end());

    auto parent_entropy = Entropy(data, sample_weights);
    auto left_entropy   = Entropy(left_data, left_sample_weights);
    auto right_entropy  = Entropy(right_data, right_sample_weights);
    DataT n             = std::accumulate(sample_weights.begin(), sample_weights.end(), DataT(0));
    DataT left_n =
      std::accumulate(left_sample_weights.begin(), left_sample_weights.end(), DataT(0));
    DataT right_n =
      std::accumulate(right_sample_weights.begin(), right_sample_weights.end(), DataT(0));

    auto gain = parent_entropy - ((left_n / n) * left_entropy + (right_n / n) * right_entropy);

    // edge cases
    if (left_data.size() < std::size_t(params.min_samples_leaf) or
        right_data.size() < std::size_t(params.min_samples_leaf)) {
      return -std::numeric_limits<DataT>::max();
    } else {
      return gain;
    }
  }

  auto GiniImpurity(std::vector<DataT> const& data, std::vector<double> const& sample_weights)
  {  // sum((n_c/n_total)(1-(n_c/n_total)))
    DataT weight_sum = std::accumulate(sample_weights.begin(), sample_weights.end(), DataT(0));
    DataT gini(0);
    for (auto c = 0; c < params.n_classes; ++c) {
      DataT sum(0);
      for (std::size_t i = 0; i < data.size(); ++i) {
        if (data[i] == DataT(c)) { sum += sample_weights[i]; }
      }
      DataT class_proba = sum / weight_sum;
      gini += class_proba * (1 - class_proba);  // adding gain
    }
    return gini;
  }

  auto GiniGroundTruthGain(std::vector<DataT> const& data,
                           std::vector<double> const& sample_weights,
                           std::size_t const split_bin_index)
  {
    auto split_offset = SplitOffset(split_bin_index);
    std::vector<DataT> left_data(data.begin(), data.begin() + split_offset);
    std::vector<DataT> right_data(data.begin() + split_offset, data.end());
    std::vector<double> left_sample_weights(sample_weights.begin(),
                                            sample_weights.begin() + split_offset);
    std::vector<double> right_sample_weights(sample_weights.begin() + split_offset,
                                             sample_weights.end());

    auto parent_gini = GiniImpurity(data, sample_weights);
    auto left_gini   = GiniImpurity(left_data, left_sample_weights);
    auto right_gini  = GiniImpurity(right_data, right_sample_weights);
    DataT n          = std::accumulate(sample_weights.begin(), sample_weights.end(), DataT(0));
    DataT left_n =
      std::accumulate(left_sample_weights.begin(), left_sample_weights.end(), DataT(0));
    DataT right_n =
      std::accumulate(right_sample_weights.begin(), right_sample_weights.end(), DataT(0));

    auto gain = parent_gini - ((left_n / n) * left_gini + (right_n / n) * right_gini);

    // edge cases
    if (left_data.size() < std::size_t(params.min_samples_leaf) or
        right_data.size() < std::size_t(params.min_samples_leaf)) {
      return -std::numeric_limits<DataT>::max();
    } else {
      return gain;
    }
  }

  auto GroundTruthGain(std::vector<DataT> const& data,
                       std::vector<double> const& sample_weights,
                       std::size_t const split_bin_index)
  {
    if constexpr (ObjectiveConfig::splitCriteria == CRITERION::MSE) {
      return MSEGroundTruthGain(data, sample_weights, split_bin_index);
    } else if constexpr (ObjectiveConfig::splitCriteria == CRITERION::POISSON) {
      return PoissonGroundTruthGain(data, sample_weights, split_bin_index);
    } else if constexpr (ObjectiveConfig::splitCriteria == CRITERION::GAMMA) {
      return GammaGroundTruthGain(data, sample_weights, split_bin_index);
    } else if constexpr (ObjectiveConfig::splitCriteria == CRITERION::INVERSE_GAUSSIAN) {
      return InverseGaussianGroundTruthGain(data, sample_weights, split_bin_index);
    } else if constexpr (ObjectiveConfig::splitCriteria == CRITERION::ENTROPY) {
      return EntropyGroundTruthGain(data, sample_weights, split_bin_index);
    } else if constexpr (ObjectiveConfig::splitCriteria == CRITERION::GINI) {
      return GiniGroundTruthGain(data, sample_weights, split_bin_index);
    }
    return DataT(0.0);
  }

  auto NumLeftOfBin(std::vector<BinT> const& cdf_hist, std::int64_t idx)
  {
    auto count{std::int64_t(0)};
    for (auto c = 0; c < params.n_classes; ++c) {
      count += static_cast<std::int64_t>(cdf_hist[params.max_n_bins * c + idx].Count());
    }
    return count;
  }

  void SetUp() override
  {
    params = ::testing::TestWithParam<ObjectiveTestParameters>::GetParam();
    rng.seed(params.seed);
    ObjectiveT objective(params.n_classes, params.min_samples_leaf, ObjectiveConfig::splitCriteria);

    auto data                 = GenRandomData();
    auto sample_weights       = GenSampleWeights();
    auto [cdf_hist, pdf_hist] = GenHist(data, sample_weights);
    auto split_bin_index      = RandUnder(SplitBinIndexUpperBound());
    auto ground_truth_gain    = GroundTruthGain(data, sample_weights, split_bin_index);
    auto len                  = NumLeftOfBin(cdf_hist, params.max_n_bins - 1);
    auto nLeft                = NumLeftOfBin(cdf_hist, split_bin_index);

    auto hypothesis_gain = objective.GainPerSplit(
      &cdf_hist[0], split_bin_index, params.max_n_bins, len, nLeft, len - nLeft);

    // The gain may actually be NaN. If so, a comparison between the result and
    // ground truth would yield false, even if they are both (correctly) NaNs.
    if (!std::isnan(ground_truth_gain) || !std::isnan(hypothesis_gain)) {
      ASSERT_NEAR(ground_truth_gain, hypothesis_gain, params.tolerance);
    }
  }
};

TEST(WeightedObjectiveEdgeCases, ClassificationRejectsZeroWeightChild)
{
  using ObjectiveT = ClassificationObjectiveFunction<double, int, true>;
  WeightedClassificationBin hist[]{{1, 0.0}, {1, 0.0}, {0, 0.0}, {1, 1.0}};
  CRITERION criteria[] = {CRITERION::GINI, CRITERION::ENTROPY};

  for (auto criterion : criteria) {
    ObjectiveT objective(2, 1, criterion);
    auto gain = objective.GainPerSplit(hist, 0, 2, 2, 1, 1);
    EXPECT_EQ(gain, -std::numeric_limits<double>::max());
  }
}

TEST(WeightedObjectiveEdgeCases, RegressionRejectsZeroWeightChild)
{
  using ObjectiveT = RegressionObjectiveFunction<double, double, true>;
  WeightedRegressionBin hist[]{{0.0, 1, 0.0}, {2.0, 2, 1.0}};
  CRITERION criteria[] = {
    CRITERION::MSE, CRITERION::POISSON, CRITERION::GAMMA, CRITERION::INVERSE_GAUSSIAN};

  for (auto criterion : criteria) {
    ObjectiveT objective(1, 1, criterion);
    auto gain = objective.GainPerSplit(hist, 0, 2, 2, 1, 1);
    EXPECT_EQ(gain, -std::numeric_limits<double>::max());
  }
}

const std::vector<ObjectiveTestParameters> mse_objective_test_parameters = {
  {9507819643927052255LLU, 2048, 64, 1, 0, 0.00001},
  {9507819643927052259LLU, 2048, 128, 1, 1, 0.00001},
  {9507819643927052251LLU, 2048, 256, 1, 1, 0.00001},
  {9507819643927052258LLU, 2048, 512, 1, 5, 0.00001},
  {9507819643927052260LLU, 2050, 128, 1, 1, 0.00001},
};

const std::vector<ObjectiveTestParameters> poisson_objective_test_parameters = {
  {9507819643927052255LLU, 2048, 64, 1, 0, 0.00001},
  {9507819643927052259LLU, 2048, 128, 1, 1, 0.00001},
  {9507819643927052251LLU, 2048, 256, 1, 1, 0.00001},
  {9507819643927052258LLU, 2048, 512, 1, 5, 0.00001},
  {9507819643927052260LLU, 2050, 128, 1, 1, 0.00001},
};

const std::vector<ObjectiveTestParameters> gamma_objective_test_parameters = {
  {9507819643927052255LLU, 2048, 64, 1, 0, 0.00001},
  {9507819643927052259LLU, 2048, 128, 1, 1, 0.00001},
  {9507819643927052251LLU, 2048, 256, 1, 1, 0.00001},
  {9507819643927052258LLU, 2048, 512, 1, 5, 0.00001},
  {9507819643927052260LLU, 2050, 128, 1, 1, 0.00001},
};

const std::vector<ObjectiveTestParameters> invgauss_objective_test_parameters = {
  {9507819643927052255LLU, 2048, 64, 1, 0, 0.00001},
  {9507819643927052259LLU, 2048, 128, 1, 1, 0.00001},
  {9507819643927052251LLU, 2048, 256, 1, 1, 0.00001},
  {9507819643927052258LLU, 2048, 512, 1, 5, 0.00001},
  {9507819643927052260LLU, 2050, 128, 1, 1, 0.00001},
};

const std::vector<ObjectiveTestParameters> entropy_objective_test_parameters = {
  {9507819643927052255LLU, 2048, 64, 2, 0, 0.00001},
  {9507819643927052256LLU, 2048, 128, 10, 1, 0.00001},
  {9507819643927052257LLU, 2048, 256, 100, 1, 0.00001},
  {9507819643927052258LLU, 2048, 512, 100, 5, 0.00001},
  {9507819643927052260LLU, 2050, 128, 10, 1, 0.00001},
};

const std::vector<ObjectiveTestParameters> gini_objective_test_parameters = {
  {9507819643927052255LLU, 2048, 64, 2, 0, 0.00001},
  {9507819643927052256LLU, 2048, 128, 10, 1, 0.00001},
  {9507819643927052257LLU, 2048, 256, 100, 1, 0.00001},
  {9507819643927052258LLU, 2048, 512, 100, 5, 0.00001},
  {9507819643927052260LLU, 2050, 128, 10, 1, 0.00001},
};

// mse objective test
typedef ObjectiveTest<
  ObjectiveTestConfig<RegressionObjectiveFunction<double, double>, CRITERION::MSE>>
  MSEObjectiveTestD;
TEST_P(MSEObjectiveTestD, MSEObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        MSEObjectiveTestD,
                        ::testing::ValuesIn(mse_objective_test_parameters));
typedef ObjectiveTest<
  ObjectiveTestConfig<RegressionObjectiveFunction<float, float>, CRITERION::MSE>>
  MSEObjectiveTestF;
TEST_P(MSEObjectiveTestF, MSEObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        MSEObjectiveTestF,
                        ::testing::ValuesIn(mse_objective_test_parameters));
typedef ObjectiveTest<
  ObjectiveTestConfig<RegressionObjectiveFunction<double, double, true>, CRITERION::MSE>>
  WeightedMSEObjectiveTestD;
TEST_P(WeightedMSEObjectiveTestD, MSEObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        WeightedMSEObjectiveTestD,
                        ::testing::ValuesIn(mse_objective_test_parameters));
typedef ObjectiveTest<
  ObjectiveTestConfig<RegressionObjectiveFunction<float, float, true>, CRITERION::MSE>>
  WeightedMSEObjectiveTestF;
TEST_P(WeightedMSEObjectiveTestF, MSEObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        WeightedMSEObjectiveTestF,
                        ::testing::ValuesIn(mse_objective_test_parameters));

// poisson objective test
typedef ObjectiveTest<
  ObjectiveTestConfig<RegressionObjectiveFunction<double, double>, CRITERION::POISSON>>
  PoissonObjectiveTestD;
TEST_P(PoissonObjectiveTestD, poissonObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        PoissonObjectiveTestD,
                        ::testing::ValuesIn(poisson_objective_test_parameters));
typedef ObjectiveTest<
  ObjectiveTestConfig<RegressionObjectiveFunction<float, float>, CRITERION::POISSON>>
  PoissonObjectiveTestF;
TEST_P(PoissonObjectiveTestF, poissonObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        PoissonObjectiveTestF,
                        ::testing::ValuesIn(poisson_objective_test_parameters));
typedef ObjectiveTest<
  ObjectiveTestConfig<RegressionObjectiveFunction<double, double, true>, CRITERION::POISSON>>
  WeightedPoissonObjectiveTestD;
TEST_P(WeightedPoissonObjectiveTestD, poissonObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        WeightedPoissonObjectiveTestD,
                        ::testing::ValuesIn(poisson_objective_test_parameters));
typedef ObjectiveTest<
  ObjectiveTestConfig<RegressionObjectiveFunction<float, float, true>, CRITERION::POISSON>>
  WeightedPoissonObjectiveTestF;
TEST_P(WeightedPoissonObjectiveTestF, poissonObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        WeightedPoissonObjectiveTestF,
                        ::testing::ValuesIn(poisson_objective_test_parameters));

// gamma objective test
typedef ObjectiveTest<
  ObjectiveTestConfig<RegressionObjectiveFunction<double, double>, CRITERION::GAMMA>>
  GammaObjectiveTestD;
TEST_P(GammaObjectiveTestD, GammaObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        GammaObjectiveTestD,
                        ::testing::ValuesIn(gamma_objective_test_parameters));
typedef ObjectiveTest<
  ObjectiveTestConfig<RegressionObjectiveFunction<float, float>, CRITERION::GAMMA>>
  GammaObjectiveTestF;
TEST_P(GammaObjectiveTestF, GammaObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        GammaObjectiveTestF,
                        ::testing::ValuesIn(gamma_objective_test_parameters));
typedef ObjectiveTest<
  ObjectiveTestConfig<RegressionObjectiveFunction<double, double, true>, CRITERION::GAMMA>>
  WeightedGammaObjectiveTestD;
TEST_P(WeightedGammaObjectiveTestD, GammaObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        WeightedGammaObjectiveTestD,
                        ::testing::ValuesIn(gamma_objective_test_parameters));
typedef ObjectiveTest<
  ObjectiveTestConfig<RegressionObjectiveFunction<float, float, true>, CRITERION::GAMMA>>
  WeightedGammaObjectiveTestF;
TEST_P(WeightedGammaObjectiveTestF, GammaObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        WeightedGammaObjectiveTestF,
                        ::testing::ValuesIn(gamma_objective_test_parameters));

// InvGauss objective test
typedef ObjectiveTest<
  ObjectiveTestConfig<RegressionObjectiveFunction<double, double>, CRITERION::INVERSE_GAUSSIAN>>
  InverseGaussianObjectiveTestD;
TEST_P(InverseGaussianObjectiveTestD, InverseGaussianObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        InverseGaussianObjectiveTestD,
                        ::testing::ValuesIn(invgauss_objective_test_parameters));
typedef ObjectiveTest<
  ObjectiveTestConfig<RegressionObjectiveFunction<float, float>, CRITERION::INVERSE_GAUSSIAN>>
  InverseGaussianObjectiveTestF;
TEST_P(InverseGaussianObjectiveTestF, InverseGaussianObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        InverseGaussianObjectiveTestF,
                        ::testing::ValuesIn(invgauss_objective_test_parameters));
typedef ObjectiveTest<ObjectiveTestConfig<RegressionObjectiveFunction<double, double, true>,
                                          CRITERION::INVERSE_GAUSSIAN>>
  WeightedInverseGaussianObjectiveTestD;
TEST_P(WeightedInverseGaussianObjectiveTestD, InverseGaussianObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        WeightedInverseGaussianObjectiveTestD,
                        ::testing::ValuesIn(invgauss_objective_test_parameters));
typedef ObjectiveTest<
  ObjectiveTestConfig<RegressionObjectiveFunction<float, float, true>, CRITERION::INVERSE_GAUSSIAN>>
  WeightedInverseGaussianObjectiveTestF;
TEST_P(WeightedInverseGaussianObjectiveTestF, InverseGaussianObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        WeightedInverseGaussianObjectiveTestF,
                        ::testing::ValuesIn(invgauss_objective_test_parameters));

// entropy objective test
typedef ObjectiveTest<
  ObjectiveTestConfig<ClassificationObjectiveFunction<double, int>, CRITERION::ENTROPY>>
  EntropyObjectiveTestD;
TEST_P(EntropyObjectiveTestD, entropyObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        EntropyObjectiveTestD,
                        ::testing::ValuesIn(entropy_objective_test_parameters));
typedef ObjectiveTest<
  ObjectiveTestConfig<ClassificationObjectiveFunction<float, int>, CRITERION::ENTROPY>>
  EntropyObjectiveTestF;
TEST_P(EntropyObjectiveTestF, entropyObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        EntropyObjectiveTestF,
                        ::testing::ValuesIn(entropy_objective_test_parameters));
typedef ObjectiveTest<
  ObjectiveTestConfig<ClassificationObjectiveFunction<double, int, true>, CRITERION::ENTROPY>>
  WeightedEntropyObjectiveTestD;
TEST_P(WeightedEntropyObjectiveTestD, entropyObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        WeightedEntropyObjectiveTestD,
                        ::testing::ValuesIn(entropy_objective_test_parameters));
typedef ObjectiveTest<
  ObjectiveTestConfig<ClassificationObjectiveFunction<float, int, true>, CRITERION::ENTROPY>>
  WeightedEntropyObjectiveTestF;
TEST_P(WeightedEntropyObjectiveTestF, entropyObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        WeightedEntropyObjectiveTestF,
                        ::testing::ValuesIn(entropy_objective_test_parameters));

// gini objective test
typedef ObjectiveTest<
  ObjectiveTestConfig<ClassificationObjectiveFunction<double, int>, CRITERION::GINI>>
  GiniObjectiveTestD;
TEST_P(GiniObjectiveTestD, giniObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        GiniObjectiveTestD,
                        ::testing::ValuesIn(gini_objective_test_parameters));
typedef ObjectiveTest<
  ObjectiveTestConfig<ClassificationObjectiveFunction<float, int>, CRITERION::GINI>>
  GiniObjectiveTestF;
TEST_P(GiniObjectiveTestF, giniObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        GiniObjectiveTestF,
                        ::testing::ValuesIn(gini_objective_test_parameters));
typedef ObjectiveTest<
  ObjectiveTestConfig<ClassificationObjectiveFunction<double, int, true>, CRITERION::GINI>>
  WeightedGiniObjectiveTestD;
TEST_P(WeightedGiniObjectiveTestD, giniObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        WeightedGiniObjectiveTestD,
                        ::testing::ValuesIn(gini_objective_test_parameters));
typedef ObjectiveTest<
  ObjectiveTestConfig<ClassificationObjectiveFunction<float, int, true>, CRITERION::GINI>>
  WeightedGiniObjectiveTestF;
TEST_P(WeightedGiniObjectiveTestF, giniObjectiveTest) {}
INSTANTIATE_TEST_CASE_P(RfTests,
                        WeightedGiniObjectiveTestF,
                        ::testing::ValuesIn(gini_objective_test_parameters));

#ifndef NDEBUG
// Feature sampling bias test
struct FeatureSamplingBiasTestParams {
  int n_features;
  int n_nodes;
  int k;                   // features sampled per node
  double tolerance_ratio;  // acceptable deviation from 1.0
};

class FeatureSamplingBiasTest : public ::testing::TestWithParam<FeatureSamplingBiasTestParams> {
 protected:
  void SetUp() override
  {
    params      = ::testing::TestWithParam<FeatureSamplingBiasTestParams>::GetParam();
    stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
    handle.reset(new raft::handle_t(rmm::cuda_stream_per_thread, stream_pool));
  }

  void TearDown() override
  {
    handle.reset();
    stream_pool.reset();
  }

  void TestFeatureSamplingBias()
  {
    auto stream = handle->get_stream();

    const auto node_count        = ML::narrow_cast<std::size_t>(params.n_nodes);
    const auto feature_count     = ML::narrow_cast<std::size_t>(params.n_features);
    const auto sampled_col_count = ML::checked_mul<std::size_t>(params.n_nodes, params.k);

    // Allocate device memory
    rmm::device_uvector<std::int64_t> d_colids(sampled_col_count, stream);
    rmm::device_uvector<NodeWorkItem> d_work_items(node_count, stream);

    // Initialize work items on host
    std::vector<NodeWorkItem> h_work_items(node_count);
    for (std::size_t i = 0; i < node_count; ++i) {
      h_work_items[i].idx             = i;
      h_work_items[i].depth           = 0;
      h_work_items[i].instances.begin = 0;
      h_work_items[i].instances.count = 100;  // arbitrary value
    }

    // Copy to device
    raft::update_device(d_work_items.data(), h_work_items.data(), node_count, stream);

    sample_features(d_colids.data(),
                    d_work_items.data(),
                    node_count,
                    std::int64_t{0},  // treeid
                    std::uint64_t{42},
                    std::int64_t{0},  // sample_offset
                    params.n_features,
                    params.k,
                    stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

    // Copy colids back to host for duplicate checking and bias counting
    std::vector<std::int64_t> h_colids(sampled_col_count);
    raft::update_host(h_colids.data(), d_colids.data(), sampled_col_count, stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

    std::vector<unsigned long long> h_counts(feature_count, 0);

    // Verify that each node's sampled features are unique and valid
    for (int node = 0; node < params.n_nodes; ++node) {
      std::vector<bool> feature_seen(feature_count, false);
      int unique_count = 0;

      for (int j = 0; j < params.k; ++j) {
        const auto sample_idx =
          ML::checked_add<std::size_t>(ML::checked_mul<std::size_t>(node, params.k), j);
        auto feature_idx = h_colids[sample_idx];

        // Check feature index is within valid range
        EXPECT_GE(feature_idx, 0) << "Node " << node << " has invalid feature index " << feature_idx
                                  << " (< 0)";
        EXPECT_LT(feature_idx, params.n_features)
          << "Node " << node << " has invalid feature index " << feature_idx
          << " (>= n_features=" << params.n_features << ")";

        // Check for duplicates
        if (feature_idx >= 0 && feature_idx < params.n_features) {
          const auto feature_pos = ML::narrow_cast<std::size_t>(feature_idx);
          h_counts[feature_pos]++;
          EXPECT_FALSE(feature_seen[feature_pos]) << "Node " << node << " has duplicate feature "
                                                  << feature_idx << " at positions in sampled set";
          if (!feature_seen[feature_pos]) {
            feature_seen[feature_pos] = true;
            unique_count++;
          }
        }
      }

      EXPECT_EQ(unique_count, params.k) << "Node " << node << " should have exactly " << params.k
                                        << " unique features, but got " << unique_count;
    }

    // Verify uniform sampling (no bias)
    auto total_samples          = ML::checked_mul<unsigned long long>(params.n_nodes, params.k);
    double expected_per_feature = double(total_samples) / params.n_features;

    // Check for feature 0 under-sampling
    double feature_0_ratio = h_counts[0] / expected_per_feature;
    EXPECT_GT(feature_0_ratio, 1.0 - params.tolerance_ratio)
      << "Feature 0 is under-sampled! Ratio: " << feature_0_ratio << " (expected ~1.0)";

    // Check for feature n-1 over-sampling
    double feature_n1_ratio = h_counts[feature_count - 1] / expected_per_feature;
    EXPECT_LT(feature_n1_ratio, 1.0 + params.tolerance_ratio)
      << "Feature n-1 is over-sampled! Ratio: " << feature_n1_ratio << " (expected ~1.0)";

    // Check all features are reasonably sampled
    for (std::size_t i = 0; i < feature_count; ++i) {
      double ratio = h_counts[i] / expected_per_feature;
      EXPECT_GT(ratio, 1.0 - params.tolerance_ratio)
        << "Feature " << i << " under-sampled. Ratio: " << ratio;
      EXPECT_LT(ratio, 1.0 + params.tolerance_ratio)
        << "Feature " << i << " over-sampled. Ratio: " << ratio;
    }
  }

  std::shared_ptr<rmm::cuda_stream_pool> stream_pool;
  std::shared_ptr<raft::handle_t> handle;
  FeatureSamplingBiasTestParams params;
};

TEST_P(FeatureSamplingBiasTest, UniformSampling) { TestFeatureSamplingBias(); }

const std::vector<FeatureSamplingBiasTestParams> feature_sampling_bias_test_parameters = {
  {10, 1000, 5, 0.15},   // 10 features, 1000 nodes, sample 5, allow 15% deviation
  {20, 1000, 10, 0.15},  // 20 features, 1000 nodes, sample 10, allow 15% deviation
  {50, 2000, 25, 0.12},  // 50 features, 2000 nodes, sample 25, allow 12% deviation
};

INSTANTIATE_TEST_CASE_P(RfTests,
                        FeatureSamplingBiasTest,
                        ::testing::ValuesIn(feature_sampling_bias_test_parameters));
#endif  // NDEBUG

}  // end namespace DT
}  // end namespace ML
