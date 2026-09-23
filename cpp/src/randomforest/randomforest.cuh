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
#include <thrust/iterator/transform_iterator.h>
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
#include <vector>

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

template <typename T>
struct SubtractOffset {
  T offset;

  __device__ T operator()(T value) const { return value - offset; }
};

template <typename T>
struct IsPositiveAndLessThan {
  T upper;

  __device__ bool operator()(T value) const { return value >= T{0} && value < upper; }
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
      distributed_(raft::resource::comms_initialized(handle) && handle.get_comms().get_size() > 1),
      rank_(distributed_ ? handle.get_comms().get_rank() : 0),
      comm_size_(distributed_ ? handle.get_comms().get_size() : 1),
      global_n_rows_(n_rows),
      rank_row_offset_(0),
      local_sample_weight_sum_(0.0),
      sample_weight_sum_(0.0),
      rank_weight_offset_(0.0),
      sample_weight_cdf_(0, handle.get_stream())
  {
    ASSERT(bootstrap_masks_ == nullptr || DT::is_dev_ptr(bootstrap_masks_),
           "bootstrap_masks must be a GPU pointer");
    validate_distributed_inputs(handle);
    compute_global_row_counts(handle);
    validate_sample_weight(handle, sample_weight_, n_rows_, distributed_);
    if (use_weighted_bootstrap()) {
      sample_weight_cdf_.resize(ML::narrow_cast<std::size_t>(n_rows_), handle.get_stream());
      thrust::inclusive_scan(rmm::exec_policy(handle.get_stream()),
                             sample_weight_,
                             sample_weight_ + n_rows_,
                             sample_weight_cdf_.begin());
    }

    // Empty distributed partitions still pass a non-null pointer so every rank selects the same
    // weighted objective type, but they have no local weight sum to validate.
    if (sample_weight_ != nullptr) {
      compute_global_sample_weights(handle);
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
        if (distributed_) {
          rank_local_weighted_draw_scratch_.emplace_back(n_sampled_rows_size, stream);
        }
      } else if (distributed_ && bootstrap_) {
        uniform_draw_scratch_.emplace_back(n_sampled_rows_size, stream);
      }
    }
  }

  RowSampler(const RowSampler&)            = delete;
  RowSampler& operator=(const RowSampler&) = delete;

  rmm::device_uvector<std::int64_t>& sample(int tree_id, int stream_id, cudaStream_t stream)
  {
    raft::common::nvtx::range fun_scope("bootstrapping row IDs @randomforest.cuh");

    auto& selected_rows = selected_rows_[stream_id];
    if (n_rows_ == 0) {
      selected_rows.resize(0, stream);
      return selected_rows;
    }

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
      if (distributed_) {
        // Each rank filters weighted_draw_scratch and only keeps the element in the range
        // [rank_weight_offset_, rank_weight_offset_ + local_sample_weight_sum_).
        // This ensures that the rank selects only the samples that are local to the rank.
        selected_rows.resize(ML::narrow_cast<std::size_t>(n_sampled_rows_), stream);
        auto local_draws_begin = thrust::make_transform_iterator(
          weighted_draw_scratch.begin(), SubtractOffset<double>{rank_weight_offset_});
        auto& rank_local_draws = rank_local_weighted_draw_scratch_[stream_id];
        auto rank_local_draw_end =
          thrust::copy_if(rmm::exec_policy(stream),
                          local_draws_begin,
                          local_draws_begin + weighted_draw_scratch.size(),
                          rank_local_draws.begin(),
                          IsPositiveAndLessThan<double>{local_sample_weight_sum_});
        auto n_rank_local_draws = rank_local_draw_end - rank_local_draws.begin();
        selected_rows.resize(n_rank_local_draws, stream);
        thrust::upper_bound(rmm::exec_policy(stream),
                            sample_weight_cdf_.data(),
                            sample_weight_cdf_.data() + n_rows_,
                            rank_local_draws.begin(),
                            rank_local_draw_end,
                            selected_rows.begin());
      } else {
        thrust::upper_bound(rmm::exec_policy(stream),
                            sample_weight_cdf_.data(),
                            sample_weight_cdf_.data() + n_rows_,
                            weighted_draw_scratch.begin(),
                            weighted_draw_scratch.end(),
                            selected_rows.begin());
      }
    } else if (bootstrap_) {
      // Draw bootstrap rows uniformly when there are no sample weights.
      if (distributed_) {
        // Each rank filters uniform_draw_scratch and only keeps the element in the range
        // [rank_row_offset_, rank_row_offset_ + n_rows_).
        // This ensures that the rank selects only the samples that are local to the rank.
        auto& uniform_draw_scratch = uniform_draw_scratch_[stream_id];
        raft::random::uniformInt<std::int64_t>(stream_resources,
                                               rng_state,
                                               uniform_draw_scratch.data(),
                                               uniform_draw_scratch.size(),
                                               0,
                                               global_n_rows_);
        selected_rows.resize(ML::narrow_cast<std::size_t>(n_sampled_rows_), stream);
        auto local_rows_begin = thrust::make_transform_iterator(
          uniform_draw_scratch.begin(), SubtractOffset<std::int64_t>{rank_row_offset_});
        auto selected_rows_end = thrust::copy_if(rmm::exec_policy(stream),
                                                 local_rows_begin,
                                                 local_rows_begin + uniform_draw_scratch.size(),
                                                 selected_rows.begin(),
                                                 IsPositiveAndLessThan<std::int64_t>{n_rows_});
        selected_rows.resize(selected_rows_end - selected_rows.begin(), stream);
      } else {
        raft::random::uniformInt<std::int64_t>(
          stream_resources, rng_state, selected_rows.data(), selected_rows.size(), 0, n_rows_);
      }
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

  void validate_distributed_inputs(const raft::handle_t& handle) const
  {
    ASSERT(n_rows_ >= 0, "n_rows must be non-negative");
    ASSERT(n_sampled_rows_ >= 0, "n_sampled_rows must be non-negative");
    if (!distributed_) { return; }

    auto stream = handle.get_stream().get();
    rmm::device_uvector<std::int64_t> local_values(2, stream);
    rmm::device_uvector<std::int64_t> gathered_values(ML::checked_mul<std::size_t>(2, comm_size_),
                                                      stream);
    std::int64_t h_local_values[2] = {sample_weight_ == nullptr ? 0 : 1,
                                      bootstrap_ ? n_sampled_rows_ : 0};
    raft::update_device(local_values.data(), h_local_values, 2, stream);
    handle.get_comms().allgather(local_values.data(), gathered_values.data(), 2, stream);
    ASSERT(handle.get_comms().sync_stream(stream) == raft::comms::status_t::SUCCESS,
           "An error occurred while validating distributed RF row-sampler inputs.");

    std::vector<std::int64_t> h_gathered_values(gathered_values.size());
    raft::update_host(
      h_gathered_values.data(), gathered_values.data(), gathered_values.size(), stream);
    handle.sync_stream(stream);
    for (int i = 0; i < comm_size_; ++i) {
      ASSERT(h_gathered_values[2 * i] == h_gathered_values[0],
             "sample_weight must be supplied consistently on every rank");
      if (bootstrap_) {
        ASSERT(h_gathered_values[2 * i + 1] == h_gathered_values[1],
               "n_sampled_rows must be identical on every rank when bootstrapping");
      }
    }
  }

  void compute_global_row_counts(const raft::handle_t& handle)
  {
    if (!distributed_) { return; }

    auto stream = handle.get_stream().get();
    rmm::device_uvector<std::int64_t> local_row_count(1, stream);
    rmm::device_uvector<std::int64_t> rank_row_counts(comm_size_, stream);
    raft::update_device(local_row_count.data(), &n_rows_, 1, stream);
    handle.get_comms().allgather(local_row_count.data(), rank_row_counts.data(), 1, stream);
    ASSERT(handle.get_comms().sync_stream(stream) == raft::comms::status_t::SUCCESS,
           "An error occurred in the distributed RF row-count all-gather.");

    // Compute the total row count over all ranks
    global_n_rows_ = thrust::reduce(
      rmm::exec_policy(stream), rank_row_counts.begin(), rank_row_counts.end(), std::int64_t{0});

    // Compute the sum of row counts in ranks 0, 1, ..., (rank_ - 1).
    rank_row_offset_ = thrust::reduce(rmm::exec_policy(stream),
                                      rank_row_counts.begin(),
                                      rank_row_counts.begin() + rank_,
                                      std::int64_t{0});
    ASSERT(global_n_rows_ > 0, "global row count must be positive");
  }

  void compute_global_sample_weights(const raft::handle_t& handle)
  {
    if (n_rows_ > 0) {
      if (use_weighted_bootstrap()) {
        raft::update_host(&local_sample_weight_sum_,
                          sample_weight_cdf_.data() + n_rows_ - 1,
                          1,
                          handle.get_stream());
        handle.sync_stream();
      } else {
        local_sample_weight_sum_ = thrust::reduce(
          rmm::exec_policy(handle.get_stream()), sample_weight_, sample_weight_ + n_rows_, 0.0);
      }
    }

    if (!distributed_) {
      sample_weight_sum_ = local_sample_weight_sum_;
      return;
    }

    auto stream = handle.get_stream().get();
    rmm::device_uvector<double> local_weight_sum(1, stream);
    rmm::device_uvector<double> rank_weight_sums(comm_size_, stream);
    raft::update_device(local_weight_sum.data(), &local_sample_weight_sum_, 1, stream);
    handle.get_comms().allgather(local_weight_sum.data(), rank_weight_sums.data(), 1, stream);
    ASSERT(handle.get_comms().sync_stream(stream) == raft::comms::status_t::SUCCESS,
           "An error occurred in the distributed RF weight-sum all-gather.");

    // Compute the sum of sample weights in all ranks
    sample_weight_sum_ = thrust::reduce(
      rmm::exec_policy(stream), rank_weight_sums.begin(), rank_weight_sums.end(), 0.0);
    // Compute the sum of sample weights in ranks 0, 1, ..., (rank_ - 1).
    rank_weight_offset_ = thrust::reduce(
      rmm::exec_policy(stream), rank_weight_sums.begin(), rank_weight_sums.begin() + rank_, 0.0);
  }

  static void validate_sample_weight(const raft::handle_t& handle,
                                     const double* sample_weight,
                                     std::int64_t n_rows,
                                     bool distributed)
  {
    ASSERT(sample_weight == nullptr || DT::is_dev_ptr(sample_weight),
           "sample_weight must be a GPU pointer");
    if (sample_weight == nullptr) { return; }

    bool has_invalid = thrust::any_of(rmm::exec_policy(handle.get_stream()),
                                      sample_weight,
                                      sample_weight + n_rows,
                                      InvalidSampleWeight<double>{});
    if (distributed) {
      int invalid_status = has_invalid ? 1 : 0;
      rmm::device_uvector<int> d_invalid_status(1, handle.get_stream());
      raft::update_device(d_invalid_status.data(), &invalid_status, 1, handle.get_stream());
      handle.get_comms().allreduce(d_invalid_status.data(),
                                   d_invalid_status.data(),
                                   1,
                                   raft::comms::op_t::MAX,
                                   handle.get_stream().get());
      ASSERT(
        handle.get_comms().sync_stream(handle.get_stream().get()) == raft::comms::status_t::SUCCESS,
        "An error occurred while validating distributed RF sample weights.");
      raft::update_host(&invalid_status, d_invalid_status.data(), 1, handle.get_stream());
      handle.sync_stream();
      has_invalid = invalid_status != 0;
    }
    ASSERT(!has_invalid, "sample_weight values must be finite and non-negative");
  }

  bool use_weighted_bootstrap() const { return bootstrap_ && sample_weight_ != nullptr; }

  bool bootstrap_;
  uint64_t seed_;
  std::int64_t n_rows_;
  std::int64_t n_sampled_rows_;
  bool* bootstrap_masks_;
  const double* sample_weight_;
  bool distributed_;
  int rank_;
  int comm_size_;
  std::int64_t global_n_rows_;
  std::int64_t rank_row_offset_;
  double local_sample_weight_sum_;
  double sample_weight_sum_;
  double rank_weight_offset_;
  rmm::device_uvector<double> sample_weight_cdf_;
  std::deque<rmm::device_uvector<std::int64_t>> selected_rows_;
  std::deque<rmm::device_uvector<double>> weighted_draw_scratch_;
  std::deque<rmm::device_uvector<double>> rank_local_weighted_draw_scratch_;
  std::deque<rmm::device_uvector<std::int64_t>> uniform_draw_scratch_;
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
    std::int64_t global_n_rows    = n_rows_i64;
    if (distributed) {
      rmm::device_uvector<std::int64_t> d_global_n_rows(1, handle.get_stream());
      raft::update_device(d_global_n_rows.data(), &global_n_rows, 1, handle.get_stream());
      handle.get_comms().allreduce(d_global_n_rows.data(),
                                   d_global_n_rows.data(),
                                   1,
                                   raft::comms::op_t::SUM,
                                   handle.get_stream().get());
      ASSERT(
        handle.get_comms().sync_stream(handle.get_stream().get()) == raft::comms::status_t::SUCCESS,
        "An error occurred in the distributed RF global row-count all-reduce.");
      raft::update_host(&global_n_rows, d_global_n_rows.data(), 1, handle.get_stream());
      handle.sync_stream();
    }
    std::int64_t n_sampled_rows = 0;
    if (this->rf_params.bootstrap) {
      n_sampled_rows =
        static_cast<std::int64_t>(std::round(this->rf_params.max_samples * global_n_rows));
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
    auto stream_pool_size = handle.get_stream_pool_size();
    if (static_cast<std::size_t>(n_streams) > stream_pool_size) {
      CUML_LOG_WARN("Resizing n_streams to fit the available stream pool size (%lu)",
                    stream_pool_size);
      n_streams = ML::narrow_cast<int>(stream_pool_size);
    }

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

      auto& selected_rows = row_sampler.sample(i, stream_id, s.get());

      /* Build individual tree in the forest.
        - input is a pointer to orig data that have n_cols features and n_rows rows.
        - n_sampled_rows: # rows sampled or retained for this tree.
        - sorted_selected_rows: points to a list of row #s (w/ n_sampled_rows elements)
          used to build the bootstrapped sample.
          Expectation: Each tree node will contain (a) # n_sampled_rows and
          (b) a pointer to a list of row numbers w.r.t original data.
      */

      forest->trees[i] = DT::DecisionTree::fit(handle,
                                               s.get(),
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
    cudaStream_t stream = user_handle.get_stream().get();

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
    cudaStream_t stream = user_handle.get_stream().get();
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
