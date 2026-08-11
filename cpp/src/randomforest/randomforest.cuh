/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/common/checked_arithmetic.hpp>
#include <cuml/ensemble/randomforest.hpp>

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/resource/comms.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/permute.cuh>
#include <raft/random/rng.cuh>
#include <raft/stats/accuracy.cuh>
#include <raft/stats/regression_metrics.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>

#include <decisiontree/batched-levelalgo/quantiles.cuh>
#include <decisiontree/decisiontree.cuh>
#include <decisiontree/treelite_util.h>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num()  0
#define omp_get_max_threads() 1
#endif

#include <cstdint>
#include <deque>
#include <map>

namespace ML {

namespace detail {
template <typename T>
struct InvalidSampleWeight {
  __device__ bool operator()(T weight) const { return weight < T(0) || !isfinite(weight); }
};

template <typename T>
struct NonzeroSampleWeight {
  __device__ bool operator()(T weight) const { return weight != T(0); }
};

// Matches estimator behavior: when bootstrapping is enabled and sample weights exist,
// those weights are materialized by drawing bootstrap rows according to them.
class RowSampler {
 public:
  RowSampler(const raft::handle_t& handle,
             const RF_params& rf_params,
             std::int64_t n_rows,
             std::int64_t n_sampled_rows,
             int n_streams,
             bool* bootstrap_masks,
             const double* sample_weight)
    : bootstrap_(rf_params.bootstrap),
      seed_(rf_params.seed),
      n_rows_(n_rows),
      n_sampled_rows_(n_sampled_rows),
      bootstrap_masks_(bootstrap_masks),
      sample_weight_(sample_weight),
      sample_weight_sum_(0.0),
      sample_weight_cdf_(0, handle.get_stream())
  {
    ASSERT(bootstrap_masks_ == nullptr || DT::is_dev_ptr(bootstrap_masks_),
           "bootstrap_masks must be a GPU pointer");
    validate_sample_weight(handle, sample_weight_, n_rows_);
    if (use_weighted_bootstrap()) {
      sample_weight_cdf_.resize(ML::narrow_cast<std::size_t>(n_rows_), handle.get_stream());
      thrust::inclusive_scan(rmm::exec_policy(handle.get_stream()),
                             sample_weight_,
                             sample_weight_ + n_rows_,
                             sample_weight_cdf_.begin());
    }

    // Empty distributed partitions still pass a non-null pointer so every rank selects the same
    // weighted objective type, but they have no local weight sum to validate.
    if (sample_weight_ != nullptr && n_rows_ > 0) {
      sample_weight_sum_ = compute_sample_weight_sum(handle);
      ASSERT(sample_weight_sum_ > 0.0,
             "sample_weight values must contain at least one positive value");
    }
    // Use a deque instead of vector because device_uvector has a deleted copy constructor.
    auto const n_sampled_rows_size = ML::narrow_cast<std::size_t>(n_sampled_rows_);
    for (int i = 0; i < n_streams; i++) {
      auto stream = handle.get_stream_from_stream_pool(i);
      selected_rows_.emplace_back(n_sampled_rows_size, stream);
      if (use_weighted_bootstrap()) {
        weighted_draw_scratch_.emplace_back(n_sampled_rows_size, stream);
      }
    }
  }

  RowSampler(const RowSampler&)            = delete;
  RowSampler& operator=(const RowSampler&) = delete;

  rmm::device_uvector<std::int64_t>& sample(int tree_id, int stream_id, cudaStream_t stream)
  {
    raft::common::nvtx::range fun_scope("bootstrapping row IDs @randomforest.cuh");

    auto& selected_rows = selected_rows_[stream_id];
    if (n_rows_ == 0) { return selected_rows; }

    raft::resources stream_resources;
    raft::resource::set_cuda_stream(stream_resources, stream);

    // Hash these together so per-tree row samples are uncorrelated.
    auto rs = DT::fnv1a32_basis;
    rs      = DT::fnv1a32(rs, seed_);
    rs      = DT::fnv1a32(rs, tree_id);
    raft::random::RngState rng_state(rs, raft::random::GenPhilox);

    if (use_weighted_bootstrap()) {
      // Draw bootstrap rows according to sample weights.
      auto& weighted_draw_scratch = weighted_draw_scratch_[stream_id];
      raft::random::uniform<double>(stream_resources,
                                    rng_state,
                                    weighted_draw_scratch.data(),
                                    weighted_draw_scratch.size(),
                                    0.0,
                                    sample_weight_sum_);
      thrust::upper_bound(rmm::exec_policy(stream),
                          sample_weight_cdf_.data(),
                          sample_weight_cdf_.data() + n_rows_,
                          weighted_draw_scratch.begin(),
                          weighted_draw_scratch.end(),
                          selected_rows.begin());
    } else if (bootstrap_) {
      // Draw bootstrap rows uniformly when there are no sample weights.
      raft::random::uniformInt<std::int64_t>(
        stream_resources, rng_state, selected_rows.data(), selected_rows.size(), 0, n_rows_);
    } else if (sample_weight_ != nullptr) {
      // Remove zero-weight rows from the non-bootstrap row set.
      selected_rows.resize(ML::narrow_cast<std::size_t>(n_sampled_rows_), stream);
      auto rows_begin        = thrust::make_counting_iterator<std::int64_t>(0);
      auto selected_rows_end = thrust::copy_if(rmm::exec_policy(stream),
                                               rows_begin,
                                               rows_begin + n_rows_,
                                               sample_weight_,
                                               selected_rows.begin(),
                                               NonzeroSampleWeight<double>{});
      auto n_selected        = selected_rows_end - selected_rows.begin();
      ASSERT(n_selected > 0, "sample_weight values must contain at least one positive value");
      selected_rows.resize(n_selected, stream);
    } else {
      selected_rows.resize(ML::narrow_cast<std::size_t>(n_sampled_rows_), stream);
      thrust::sequence(rmm::exec_policy(stream), selected_rows.begin(), selected_rows.end());
    }

    store_bootstrap_mask(tree_id, selected_rows, stream);
    return selected_rows;
  }

  // Use sample weights in impurity / objective calculation only when bootstrapping is not enabled.
  const double* tree_sample_weight() const { return bootstrap_ ? nullptr : sample_weight_; }

 private:
  void store_bootstrap_mask(int tree_id,
                            rmm::device_uvector<std::int64_t>& selected_rows,
                            cudaStream_t stream)
  {
    if (bootstrap_masks_ == nullptr) { return; }

    bool* tree_mask = bootstrap_masks_ + (ML::checked_mul<std::size_t>(tree_id, n_rows_));
    thrust::fill(rmm::exec_policy(stream), tree_mask, tree_mask + n_rows_, false);
    thrust::scatter(rmm::exec_policy(stream),
                    thrust::make_constant_iterator(true),
                    thrust::make_constant_iterator(true) + selected_rows.size(),
                    selected_rows.data(),
                    tree_mask);
  }

  double compute_sample_weight_sum(const raft::handle_t& handle) const
  {
    if (use_weighted_bootstrap()) {
      double weight_sum = 0.0;
      raft::update_host(
        &weight_sum, sample_weight_cdf_.data() + n_rows_ - 1, 1, handle.get_stream());
      handle.sync_stream();
      return weight_sum;
    }

    return thrust::reduce(
      rmm::exec_policy(handle.get_stream()), sample_weight_, sample_weight_ + n_rows_, 0.0);
  }

  static void validate_sample_weight(const raft::handle_t& handle,
                                     const double* sample_weight,
                                     std::int64_t n_rows)
  {
    ASSERT(sample_weight == nullptr || DT::is_dev_ptr(sample_weight),
           "sample_weight must be a GPU pointer");
    if (sample_weight == nullptr) { return; }

    bool has_invalid = thrust::any_of(rmm::exec_policy(handle.get_stream()),
                                      sample_weight,
                                      sample_weight + n_rows,
                                      InvalidSampleWeight<double>{});
    ASSERT(!has_invalid, "sample_weight values must be finite and non-negative");
  }

  bool use_weighted_bootstrap() const { return bootstrap_ && sample_weight_ != nullptr; }

  bool bootstrap_;
  uint64_t seed_;
  std::int64_t n_rows_;
  std::int64_t n_sampled_rows_;
  bool* bootstrap_masks_;
  const double* sample_weight_;
  double sample_weight_sum_;
  rmm::device_uvector<double> sample_weight_cdf_;
  std::deque<rmm::device_uvector<std::int64_t>> selected_rows_;
  std::deque<rmm::device_uvector<double>> weighted_draw_scratch_;
};
}  // namespace detail

template <class T, class L>
class RandomForest {
 protected:
  RF_params rf_params;  // structure containing RF hyperparameters
  int rf_type;          // 0 for classification 1 for regression

  void error_checking(const T* input,
                      L* predictions,
                      int n_rows,
                      int n_cols,
                      bool predict,
                      bool allow_empty_local_rows = false) const
  {
    if (predict) {
      ASSERT(predictions != nullptr, "Error! User has not allocated memory for predictions.");
    }
    ASSERT(allow_empty_local_rows ? (n_rows >= 0) : (n_rows > 0), "Invalid n_rows %d", n_rows);
    ASSERT((n_cols > 0), "Invalid n_cols %d", n_cols);

    if (n_rows == 0) { return; }

    bool input_is_dev_ptr = DT::is_dev_ptr(input);
    bool preds_is_dev_ptr = DT::is_dev_ptr(predictions);

    if (!input_is_dev_ptr || (input_is_dev_ptr != preds_is_dev_ptr)) {
      ASSERT(false,
             "RF Error: Expected both input and labels/predictions to be GPU "
             "pointers");
    }
  }

 public:
  /**
   * @brief Construct RandomForest object.
   * @param[in] cfg_rf_params: Random forest hyper-parameter struct.
   * @param[in] cfg_rf_type: Task type: 0 for classification, 1 for regression
   */
  RandomForest(RF_params cfg_rf_params, int cfg_rf_type = RF_type::CLASSIFICATION)
    : rf_params(cfg_rf_params), rf_type(cfg_rf_type) {};

  /**
   * @brief Build (i.e., fit, train) random forest for input data.
   * @param[in] user_handle: raft::handle_t
   * @param[in] input: train data (n_rows samples, n_cols features), excluding labels.
   *   Column-major by default, or row-major when `input_row_major` is true. Device pointer.
   * @param[in] n_rows: number of training data samples.
   * @param[in] n_cols: number of features (i.e., columns) excluding target feature.
   * @param[in] labels: 1D array of target predictions/labels. Device Pointer.
            For classification task, only labels of type int are supported.
              Assumption: labels were preprocessed to map to ascending numbers from 0;
              needed for current gini impl in decision tree
            For regression task, the labels (predictions) can be float or double data type.
  * @param[in] n_unique_labels: (meaningful only for classification) #unique label values (known
  during preprocessing)
  * @param[in] forest: CPU point to RandomForestMetaData struct.
  * @param[out] bootstrap_masks: optional device pointer to store bootstrap masks
  *   (n_trees * n_rows), only populated if a non-null pointer is provided.
  * @param[in] sample_weight: optional device pointer to per-row sample weights. With bootstrap
  *   enabled, rows are sampled with probability proportional to these weights and the sampled
  *   counts drive tree training. Without bootstrap, zero-weight rows are removed from the tree
  *   row set and remaining weights are used for impurity/objective math.
  * @param[in] input_row_major: whether train data is row-major instead of the default
  *   column-major layout.
  */
  void fit(const raft::handle_t& user_handle,
           const T* input,
           int n_rows,
           int n_cols,
           L* labels,
           int n_unique_labels,
           RandomForestMetaData<T, L>* forest,
           bool* bootstrap_masks       = nullptr,
           const double* sample_weight = nullptr,
           bool input_row_major        = false)
  {
    raft::common::nvtx::range fun_scope("RandomForest::fit @randomforest.cuh");
    const raft::handle_t& handle = user_handle;
    bool distributed =
      raft::resource::comms_initialized(handle) && handle.get_comms().get_size() > 1;
    this->error_checking(input, labels, n_rows, n_cols, false, distributed);
    std::int64_t const n_rows_i64 = n_rows;
    std::int64_t n_sampled_rows   = 0;
    if (this->rf_params.bootstrap) {
      n_sampled_rows =
        static_cast<std::int64_t>(std::round(this->rf_params.max_samples * n_rows_i64));
    } else {
      if (this->rf_params.max_samples != 1.0) {
        CUML_LOG_WARN(
          "If bootstrap sampling is disabled, max_samples value is ignored and "
          "whole dataset is used for building each tree");
        this->rf_params.max_samples = 1.0;
      }
      n_sampled_rows = n_rows_i64;
    }
    int n_streams = this->rf_params.n_streams;
    // Distributed tree builders issue collectives independently, so train them serially until
    // the forest-level scheduler can impose a global collective order across concurrent trees.
    if (distributed) { n_streams = 1; }
    ASSERT(static_cast<std::size_t>(n_streams) <= handle.get_stream_pool_size(),
           "effective RF n_streams (=%d) should be <= raft::handle_t.n_streams (=%lu)",
           n_streams,
           handle.get_stream_pool_size());

    auto quantile_result = DT::computeQuantiles(handle,
                                                input,
                                                this->rf_params.tree_params.max_n_bins,
                                                n_rows,
                                                n_cols,
                                                4,
                                                rf_params.seed,
                                                input_row_major);
    auto quantiles       = quantile_result.view();

    // n_streams should not be less than n_trees
    if (this->rf_params.n_trees < n_streams) n_streams = this->rf_params.n_trees;

    detail::RowSampler row_sampler(handle,
                                   this->rf_params,
                                   n_rows_i64,
                                   n_sampled_rows,
                                   n_streams,
                                   bootstrap_masks,
                                   sample_weight);

    forest->n_features = n_cols;

#pragma omp parallel for num_threads(n_streams)
    for (int i = 0; i < this->rf_params.n_trees; i++) {
      int stream_id = omp_get_thread_num();
      auto s        = handle.get_stream_from_stream_pool(stream_id);

      auto& selected_rows = row_sampler.sample(i, stream_id, s);

      /* Build individual tree in the forest.
        - input is a pointer to orig data that have n_cols features and n_rows rows.
        - n_sampled_rows: # rows sampled or retained for this tree.
        - sorted_selected_rows: points to a list of row #s (w/ n_sampled_rows elements)
          used to build the bootstrapped sample.
          Expectation: Each tree node will contain (a) # n_sampled_rows and
          (b) a pointer to a list of row numbers w.r.t original data.
      */

      forest->trees[i] = DT::DecisionTree::fit(handle,
                                               s,
                                               input,
                                               n_cols,
                                               n_rows,
                                               labels,
                                               &selected_rows,
                                               n_unique_labels,
                                               this->rf_params.tree_params,
                                               this->rf_params.seed,
                                               quantiles,
                                               i,
                                               row_sampler.tree_sample_weight(),
                                               input_row_major);
    }
    // Cleanup
    handle.sync_stream_pool();
    handle.sync_stream();
  }

  /**
   * @brief Predict target feature for input data
   * @param[in] user_handle: raft::handle_t.
   * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU
   * pointer.
   * @param[in] n_rows: number of  data samples.
   * @param[in] n_cols: number of features (excluding target feature).
   * @param[in, out] predictions: n_rows predicted labels. GPU pointer, user allocated.
   * @param[in] verbosity: verbosity level for logging messages during execution
   */
  void predict(const raft::handle_t& user_handle,
               const T* input,
               int n_rows,
               int n_cols,
               L* predictions,
               const RandomForestMetaData<T, L>* forest,
               rapids_logger::level_enum verbosity) const
  {
    ML::default_logger().set_level(verbosity);
    this->error_checking(input, predictions, n_rows, n_cols, true);
    std::vector<L> h_predictions(n_rows);
    cudaStream_t stream = user_handle.get_stream();

    std::vector<T> h_input(std::size_t(n_rows) * n_cols);
    raft::update_host(h_input.data(), input, std::size_t(n_rows) * n_cols, stream);
    user_handle.sync_stream(stream);

    int row_size = n_cols;

    default_logger().set_pattern("%v");
    for (int row_id = 0; row_id < n_rows; row_id++) {
      std::vector<T> row_prediction(forest->trees[0]->num_outputs);
      for (int i = 0; i < this->rf_params.n_trees; i++) {
        DT::DecisionTree::predict(user_handle,
                                  *forest->trees[i],
                                  &h_input[row_id * row_size],
                                  1,
                                  n_cols,
                                  row_prediction.data(),
                                  forest->trees[i]->num_outputs,
                                  verbosity);
      }
      for (int k = 0; k < forest->trees[0]->num_outputs; k++) {
        row_prediction[k] /= this->rf_params.n_trees;
      }
      if (rf_type == RF_type::CLASSIFICATION) {  // classification task: use 'majority' prediction
        L best_class = 0;
        T best_prob  = 0.0;
        for (int k = 0; k < forest->trees[0]->num_outputs; k++) {
          if (row_prediction[k] > best_prob) {
            best_class = k;
            best_prob  = row_prediction[k];
          }
        }

        h_predictions[row_id] = best_class;
      } else {
        h_predictions[row_id] = row_prediction[0];
      }
    }

    raft::update_device(predictions, h_predictions.data(), n_rows, stream);
    user_handle.sync_stream(stream);
    default_logger().set_pattern(default_pattern());
  }

  /**
   * @brief Predict target feature for input data and score against ref_labels.
   * @param[in] user_handle: raft::handle_t.
   * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU
   * pointer.
   * @param[in] ref_labels: label values for cross validation (n_rows elements); GPU pointer.
   * @param[in] n_rows: number of  data samples.
   * @param[in] n_cols: number of features (excluding target feature).
   * @param[in] predictions: n_rows predicted labels. GPU pointer, user allocated.
   * @param[in] verbosity: verbosity level for logging messages during execution
   * @param[in] rf_type: task type: 0 for classification, 1 for regression
   */
  static RF_metrics score(const raft::handle_t& user_handle,
                          const L* ref_labels,
                          int n_rows,
                          const L* predictions,
                          rapids_logger::level_enum verbosity,
                          int rf_type = RF_type::CLASSIFICATION)
  {
    ML::default_logger().set_level(verbosity);
    cudaStream_t stream = user_handle.get_stream();
    RF_metrics stats;
    if (rf_type == RF_type::CLASSIFICATION) {  // task classifiation: get classification metrics
      float accuracy = raft::stats::accuracy(predictions, ref_labels, n_rows, stream);
      stats          = set_rf_metrics_classification(accuracy);
      if (ML::default_logger().should_log(rapids_logger::level_enum::debug)) print(stats);

      /* TODO: Potentially augment RF_metrics w/ more metrics (e.g., precision, F1, etc.).
        For non binary classification problems (i.e., one target and  > 2 labels), need avg.
        for each of these metrics */
    } else {  // regression task: get regression metrics
      double mean_abs_error, mean_squared_error, median_abs_error;
      raft::stats::regression_metrics(predictions,
                                      ref_labels,
                                      n_rows,
                                      stream,
                                      mean_abs_error,
                                      mean_squared_error,
                                      median_abs_error);
      stats = set_rf_metrics_regression(mean_abs_error, mean_squared_error, median_abs_error);
      if (ML::default_logger().should_log(rapids_logger::level_enum::debug)) print(stats);
    }

    return stats;
  }
};

// class specializations
template class RandomForest<float, int>;
template class RandomForest<float, float>;
template class RandomForest<double, int>;
template class RandomForest<double, double>;

}  // End namespace ML
