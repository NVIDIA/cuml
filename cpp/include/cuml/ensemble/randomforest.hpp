/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/common/export.hpp>
#include <cuml/common/logger.hpp>
#include <cuml/ensemble/treelite_defs.hpp>
#include <cuml/tree/decisiontree.hpp>

#include <memory>

namespace raft {
class handle_t;  // forward decl
}

namespace CUML_EXPORT ML {

struct RF_params {
  /**
   * Number of decision trees in the random forest.
   */
  int n_trees;
  /**
   * Control bootstrapping.
   * If bootstrapping is set to true, bootstrapped samples are used for building
   * each tree. Bootstrapped sampling is done by randomly drawing
   * round(max_samples * n_samples) number of samples with replacement. More on
   * bootstrapping:
   *     https://en.wikipedia.org/wiki/Bootstrap_aggregating
   * If bootstrapping is set to false, whole dataset is used to build each
   * tree.
   */
  bool bootstrap;
  /**
   * Ratio of dataset rows used while fitting each tree.
   */
  float max_samples;
  /**
   * Decision tree training hyper parameter struct.
   */
  /**
   * random seed
   */
  uint64_t seed;
  /**
   * Number of concurrent GPU streams for parallel tree building.
   * Each stream is independently managed by CPU thread.
   * N streams need N times RF workspace.
   */
  int n_streams;
  DT::DecisionTreeParams tree_params;
};

template <class T, class L>
struct RandomForestMetaData {
  std::vector<std::shared_ptr<DT::TreeMetaDataNode<T, L>>> trees;
  RF_params rf_params;

  /**
   * Number of features in the training data.
   */
  int n_features = 0;
};

template <class T, class L>
std::string get_rf_detailed_text(const RandomForestMetaData<T, L>* forest);

template <class T, class L>
std::string get_rf_json(const RandomForestMetaData<T, L>* forest);

template <class T, class L>
void build_treelite_forest(TreeliteModelHandle* model,
                           const RandomForestMetaData<T, L>* forest,
                           int num_features);

/**
 * @brief Compute the feature importances of the trained RandomForest model.
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 * @param[in] forest: CPU pointer to RandomForestMetaData
 * @param[out] importances: output feature importance scores
 */
template <class T, class L>
void compute_feature_importances(const RandomForestMetaData<T, L>* forest, T* importances);

// ----------------------------- Classification ----------------------------------- //

typedef RandomForestMetaData<float, int> RandomForestClassifierF;
typedef RandomForestMetaData<double, int> RandomForestClassifierD;

void fit(const raft::handle_t& user_handle,
         RandomForestClassifierF* forest,
         float* input,
         int n_rows,
         int n_cols,
         int* labels,
         int n_unique_labels,
         RF_params rf_params,
         rapids_logger::level_enum verbosity = rapids_logger::level_enum::info,
         bool* bootstrap_masks               = nullptr,
         const double* sample_weight         = nullptr,
         bool input_row_major                = false);
void fit(const raft::handle_t& user_handle,
         RandomForestClassifierD* forest,
         double* input,
         int n_rows,
         int n_cols,
         int* labels,
         int n_unique_labels,
         RF_params rf_params,
         rapids_logger::level_enum verbosity = rapids_logger::level_enum::info,
         bool* bootstrap_masks               = nullptr,
         const double* sample_weight         = nullptr,
         bool input_row_major                = false);

/**
 * @brief Train a random forest classifier and export it as a Treelite model.
 *
 * @param[in] user_handle RAFT handle for stream and allocator resources.
 * @param[out] model Treelite model handle populated by training.
 * @param[in] input Training data, column-major by default or row-major when
 *   input_row_major is true.
 * @param[in] n_rows Number of rows in input.
 * @param[in] n_cols Number of columns in input.
 * @param[in] labels Training labels.
 * @param[in] n_unique_labels Number of unique classes in labels.
 * @param[in] rf_params Random forest training parameters.
 * @param[in] bootstrap_masks Optional bootstrap masks.
 * @param[out] feature_importances Output feature importances.
 * @param[in] verbosity Logging verbosity.
 * @param[in] sample_weight Optional per-row sample weights.
 * @param[in] input_row_major Whether input is row-major instead of column-major.
 */
template <typename T, typename L>
void fit_treelite(const raft::handle_t& user_handle,
                  TreeliteModelHandle* model,
                  T* input,
                  int n_rows,
                  int n_cols,
                  L* labels,
                  int n_unique_labels,
                  RF_params rf_params,
                  bool* bootstrap_masks,
                  T* feature_importances,
                  rapids_logger::level_enum verbosity,
                  const double* sample_weight = nullptr,
                  bool input_row_major        = false);

RF_params set_rf_params(int max_depth,
                        int max_leaves,
                        float max_features,
                        int max_n_bins,
                        int min_samples_leaf,
                        int min_samples_split,
                        float min_impurity_decrease,
                        bool bootstrap,
                        int n_trees,
                        float max_samples,
                        uint64_t seed,
                        CRITERION split_criterion,
                        int cfg_n_streams,
                        int max_batch_size);

// ----------------------------- Regression ----------------------------------- //

typedef RandomForestMetaData<float, float> RandomForestRegressorF;
typedef RandomForestMetaData<double, double> RandomForestRegressorD;

void fit(const raft::handle_t& user_handle,
         RandomForestRegressorF* forest,
         float* input,
         int n_rows,
         int n_cols,
         float* labels,
         RF_params rf_params,
         rapids_logger::level_enum verbosity = rapids_logger::level_enum::info,
         bool* bootstrap_masks               = nullptr,
         const double* sample_weight         = nullptr,
         bool input_row_major                = false);
void fit(const raft::handle_t& user_handle,
         RandomForestRegressorD* forest,
         double* input,
         int n_rows,
         int n_cols,
         double* labels,
         RF_params rf_params,
         rapids_logger::level_enum verbosity = rapids_logger::level_enum::info,
         bool* bootstrap_masks               = nullptr,
         const double* sample_weight         = nullptr,
         bool input_row_major                = false);

/**
 * @brief Train a random forest regressor and export it as a Treelite model.
 *
 * @param[in] user_handle RAFT handle for stream and allocator resources.
 * @param[out] model Treelite model handle populated by training.
 * @param[in] input Training data, column-major by default or row-major when
 *   input_row_major is true.
 * @param[in] n_rows Number of rows in input.
 * @param[in] n_cols Number of columns in input.
 * @param[in] labels Training labels.
 * @param[in] rf_params Random forest training parameters.
 * @param[in] bootstrap_masks Optional bootstrap masks.
 * @param[out] feature_importances Output feature importances.
 * @param[in] verbosity Logging verbosity.
 * @param[in] sample_weight Optional per-row sample weights.
 * @param[in] input_row_major Whether input is row-major instead of column-major.
 */
template <typename T, typename L>
void fit_treelite(const raft::handle_t& user_handle,
                  TreeliteModelHandle* model,
                  T* input,
                  int n_rows,
                  int n_cols,
                  L* labels,
                  RF_params rf_params,
                  bool* bootstrap_masks,
                  T* feature_importances,
                  rapids_logger::level_enum verbosity,
                  const double* sample_weight = nullptr,
                  bool input_row_major        = false);
};  // namespace CUML_EXPORT ML
