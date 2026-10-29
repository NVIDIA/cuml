/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "randomforest.cuh"

#include <cuml/common/logger.hpp>
#include <cuml/ensemble/randomforest.hpp>

#include <raft/core/error.hpp>
#include <raft/core/handle.hpp>

#include <treelite/c_api.h>
#include <treelite/enum/task_type.h>
#include <treelite/tree.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace ML {

using namespace std;
namespace tl = treelite;

template <class T, class L>
std::string _get_rf_text(const RandomForestMetaData<T, L>* forest)
{
  if (!forest) {
    return "Empty forest";
  } else {
    default_logger().set_pattern("%v");
    std::ostringstream oss;
    oss << "Forest has " << forest->rf_params.n_trees << " trees, "
        << "max_depth " << forest->rf_params.tree_params.max_depth << ", and max_leaves "
        << forest->rf_params.tree_params.max_leaves << "\n";
    for (int i = 0; i < forest->rf_params.n_trees; i++) {
      oss << "Tree #" << i << "\n";
      oss << DT::get_tree_text<T, L>(forest->trees[i].get()) << "\n";
    }
    default_logger().set_pattern(default_pattern());
    return oss.str();
  }
}

template <class T, class L>
std::string _get_rf_json(const RandomForestMetaData<T, L>* forest)
{
  if (!forest) { return "[]"; }
  std::ostringstream oss;
  oss << "[\n";
  for (int i = 0; i < forest->rf_params.n_trees; i++) {
    oss << DT::get_tree_json<T, L>(forest->trees[i].get());
    if (i < forest->rf_params.n_trees - 1) { oss << ",\n"; }
  }
  oss << "\n]";
  return oss.str();
}

/**
 * @brief Print detailed view of all trees in the random forest.
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 * @param[in] forest: CPU pointer to RandomForestMetaData struct.
 */
template <class T, class L>
std::string get_rf_detailed_text(const RandomForestMetaData<T, L>* forest)
{
  return _get_rf_text(forest);
}

template <class T, class L>
std::string get_rf_json(const RandomForestMetaData<T, L>* forest)
{
  return _get_rf_json(forest);
}

template <class T, class L>
void build_treelite_forest(TreeliteModelHandle* model_handle,
                           const RandomForestMetaData<T, L>* forest,
                           int num_features)
{
  auto model                          = tl::Model::Create<T, T>();
  tl::ModelPreset<T, T>& model_preset = std::get<tl::ModelPreset<T, T>>(model->variant_);

  // Determine number of outputs
  ASSERT(forest->trees.size() == static_cast<std::size_t>(forest->rf_params.n_trees),
         "Inconsistent number of trees.");
  ASSERT(forest->trees.size() > 0, "Empty forest.");
  int num_outputs = forest->trees.front()->num_outputs;
  ASSERT(num_outputs > 0, "Invalid forest");
  for (const auto& tree : forest->trees) {
    ASSERT(num_outputs == tree->num_outputs, "Invalid forest");
  }

  if constexpr (std::is_integral_v<L>) {
    num_outputs = std::max(num_outputs, 2);
    // Ensure that num_outputs is at least 2
    model->task_type     = tl::TaskType::kMultiClf;
    model->postprocessor = "identity_multiclass";
  } else {
    ASSERT(num_outputs == 1, "Only one variable expected for regression problem.");
    model->task_type     = tl::TaskType::kRegressor;
    model->postprocessor = "identity";
  }

  model->num_target        = 1;
  model->num_class         = std::vector<std::int32_t>{static_cast<std::int32_t>(num_outputs)};
  model->leaf_vector_shape = std::vector<std::int32_t>{1, static_cast<std::int32_t>(num_outputs)};
  model->target_id         = std::vector<std::int32_t>(forest->rf_params.n_trees, 0);
  model->class_id =
    std::vector<std::int32_t>(forest->rf_params.n_trees, (std::is_integral_v<L> ? -1 : 0));
  model->num_feature         = num_features;
  model->average_tree_output = true;
  model->base_scores         = std::vector<double>(num_outputs, 0.0);
  model->SetTreeLimit(forest->rf_params.n_trees);

#pragma omp parallel for
  for (int i = 0; i < forest->rf_params.n_trees; i++) {
    auto rf_tree = forest->trees[i];

    if (rf_tree->sparsetree.size() != 0) {
      model_preset.trees[i] = DT::build_treelite_tree<T, L>(*rf_tree, num_outputs);
    }
  }

  *model_handle = static_cast<TreeliteModelHandle>(model.release());
}

/**
 * @defgroup RandomForestClassificationFit Random Forest Classification - Fit function
 * @brief Build (i.e., fit, train) random forest classifier for input data.
 * @param[in] user_handle: raft::handle_t
 * @param[in,out] forest: CPU pointer to RandomForestMetaData object. User allocated.
 * @param[in] input: train data (n_rows samples, n_cols features), excluding labels. Column-major
 *   by default, or row-major when `input_row_major` is true. Device pointer.
 * @param[in] n_rows: number of training data samples.
 * @param[in] n_cols: number of features (i.e., columns) excluding target feature.
 * @param[in] labels: 1D array of target features (int only), with one label per
 *   training sample. Device pointer.
 *   Assumption: labels were preprocessed to map to ascending numbers from 0;
 *   needed for current gini impl. in decision tree
 * @param[in] n_unique_labels: #unique label values (known during preprocessing)
 * @param[in] rf_params: Random Forest training hyper parameter struct.
 * @param[in] verbosity: verbosity level for logging messages during execution
 * @param[in] input_row_major: whether input training data is row-major instead of the default
 *   column-major layout.
 * @{
 */
void fit(const raft::handle_t& user_handle,
         RandomForestClassifierF* forest,
         float* input,
         int n_rows,
         int n_cols,
         int* labels,
         int n_unique_labels,
         RF_params rf_params,
         rapids_logger::level_enum verbosity,
         bool* bootstrap_masks,
         const double* sample_weight,
         bool input_row_major)
{
  raft::common::nvtx::range fun_scope("RF::fit @randomforest.cu");
  ML::default_logger().set_level(verbosity);
  ASSERT(forest->trees.empty(), "Cannot fit an existing forest.");
  forest->trees.resize(rf_params.n_trees);
  forest->rf_params = rf_params;

  std::shared_ptr<RandomForest<float, int>> rf_classifier =
    std::make_shared<RandomForest<float, int>>(rf_params);
  rf_classifier->fit(user_handle,
                     input,
                     n_rows,
                     n_cols,
                     labels,
                     n_unique_labels,
                     forest,
                     bootstrap_masks,
                     sample_weight,
                     input_row_major);
}

void fit(const raft::handle_t& user_handle,
         RandomForestClassifierD* forest,
         double* input,
         int n_rows,
         int n_cols,
         int* labels,
         int n_unique_labels,
         RF_params rf_params,
         rapids_logger::level_enum verbosity,
         bool* bootstrap_masks,
         const double* sample_weight,
         bool input_row_major)
{
  raft::common::nvtx::range fun_scope("RF::fit @randomforest.cu");
  ML::default_logger().set_level(verbosity);
  ASSERT(forest->trees.empty(), "Cannot fit an existing forest.");
  forest->trees.resize(rf_params.n_trees);
  forest->rf_params = rf_params;

  std::shared_ptr<RandomForest<double, int>> rf_classifier =
    std::make_shared<RandomForest<double, int>>(rf_params);
  rf_classifier->fit(user_handle,
                     input,
                     n_rows,
                     n_cols,
                     labels,
                     n_unique_labels,
                     forest,
                     bootstrap_masks,
                     sample_weight,
                     input_row_major);
}

template <typename value_t, typename label_t>
void fit_treelite(const raft::handle_t& user_handle,
                  TreeliteModelHandle* model,
                  value_t* input,
                  int n_rows,
                  int n_cols,
                  label_t* labels,
                  int n_unique_labels,
                  RF_params rf_params,
                  bool* bootstrap_masks,
                  value_t* feature_importances,
                  rapids_logger::level_enum verbosity,
                  const double* sample_weight,
                  bool input_row_major)
{
  RandomForestMetaData<value_t, label_t> metadata;
  fit(user_handle,
      &metadata,
      input,
      n_rows,
      n_cols,
      labels,
      n_unique_labels,
      rf_params,
      verbosity,
      bootstrap_masks,
      sample_weight,
      input_row_major);

  // Compute feature importances if requested
  if (feature_importances != nullptr) {
    compute_feature_importances(&metadata, feature_importances);
  }

  build_treelite_forest(model, &metadata, n_cols);
}

/** @} */

/**
 * @brief Check validity of all random forest hyper-parameters.
 * @param[in] rf_params: random forest hyper-parameters
 */
void validity_check(const RF_params rf_params)
{
  ASSERT((rf_params.n_trees > 0), "Invalid n_trees %d", rf_params.n_trees);
  ASSERT((rf_params.max_samples > 0) && (rf_params.max_samples <= 1.0),
         "max_samples value %f outside permitted (0, 1] range",
         rf_params.max_samples);
  ASSERT((rf_params.n_streams > 0), "Invalid n_streams %d", rf_params.n_streams);
}

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
                        int max_batch_size)
{
  DT::DecisionTreeParams tree_params;
  DT::set_tree_params(tree_params,
                      max_depth,
                      max_leaves,
                      max_features,
                      max_n_bins,
                      min_samples_leaf,
                      min_samples_split,
                      min_impurity_decrease,
                      split_criterion,
                      max_batch_size);
  RF_params rf_params;
  rf_params.n_trees     = n_trees;
  rf_params.bootstrap   = bootstrap;
  rf_params.max_samples = max_samples;
  rf_params.seed        = seed;
  rf_params.n_streams   = min(cfg_n_streams, omp_get_max_threads());
  if (n_trees < rf_params.n_streams) rf_params.n_streams = n_trees;
  rf_params.tree_params = tree_params;
  validity_check(rf_params);
  return rf_params;
}

/**
 * @defgroup RandomForestRegressorFit Random Forest Regression - Fit function
 * @brief Build (i.e., fit, train) random forest regressor for input data.
 * @param[in] user_handle: raft::handle_t
 * @param[in,out] forest: CPU pointer to RandomForestMetaData object. User allocated.
 * @param[in] input: train data (n_rows samples, n_cols features), excluding labels. Column-major
 *   by default, or row-major when `input_row_major` is true. Device pointer.
 * @param[in] n_rows: number of training data samples.
 * @param[in] n_cols: number of features (i.e., columns) excluding target feature.
 * @param[in] labels: 1D array of target features (float or double), with one label per
 *   training sample. Device pointer.
 * @param[in] rf_params: Random Forest training hyper parameter struct.
 * @param[in] verbosity: verbosity level for logging messages during execution
 * @param[in] input_row_major: whether input training data is row-major instead of the default
 *   column-major layout.
 * @{
 */
void fit(const raft::handle_t& user_handle,
         RandomForestRegressorF* forest,
         float* input,
         int n_rows,
         int n_cols,
         float* labels,
         RF_params rf_params,
         rapids_logger::level_enum verbosity,
         bool* bootstrap_masks,
         const double* sample_weight,
         bool input_row_major)
{
  raft::common::nvtx::range fun_scope("RF::fit @randomforest.cu");
  ML::default_logger().set_level(verbosity);
  ASSERT(forest->trees.empty(), "Cannot fit an existing forest.");
  forest->trees.resize(rf_params.n_trees);
  forest->rf_params = rf_params;

  std::shared_ptr<RandomForest<float, float>> rf_regressor =
    std::make_shared<RandomForest<float, float>>(rf_params);
  rf_regressor->fit(user_handle,
                    input,
                    n_rows,
                    n_cols,
                    labels,
                    1,
                    forest,
                    bootstrap_masks,
                    sample_weight,
                    input_row_major);
}

void fit(const raft::handle_t& user_handle,
         RandomForestRegressorD* forest,
         double* input,
         int n_rows,
         int n_cols,
         double* labels,
         RF_params rf_params,
         rapids_logger::level_enum verbosity,
         bool* bootstrap_masks,
         const double* sample_weight,
         bool input_row_major)
{
  raft::common::nvtx::range fun_scope("RF::fit @randomforest.cu");
  ML::default_logger().set_level(verbosity);
  ASSERT(forest->trees.empty(), "Cannot fit an existing forest.");
  forest->trees.resize(rf_params.n_trees);
  forest->rf_params = rf_params;

  std::shared_ptr<RandomForest<double, double>> rf_regressor =
    std::make_shared<RandomForest<double, double>>(rf_params);
  rf_regressor->fit(user_handle,
                    input,
                    n_rows,
                    n_cols,
                    labels,
                    1,
                    forest,
                    bootstrap_masks,
                    sample_weight,
                    input_row_major);
}

template <typename value_t, typename label_t>
void fit_treelite(const raft::handle_t& user_handle,
                  TreeliteModelHandle* model,
                  value_t* input,
                  int n_rows,
                  int n_cols,
                  label_t* labels,
                  RF_params rf_params,
                  bool* bootstrap_masks,
                  value_t* feature_importances,
                  rapids_logger::level_enum verbosity,
                  const double* sample_weight,
                  bool input_row_major)
{
  RandomForestMetaData<value_t, label_t> metadata;
  fit(user_handle,
      &metadata,
      input,
      n_rows,
      n_cols,
      labels,
      rf_params,
      verbosity,
      bootstrap_masks,
      sample_weight,
      input_row_major);

  // Compute feature importances if requested
  if (feature_importances != nullptr) {
    compute_feature_importances(&metadata, feature_importances);
  }

  build_treelite_forest(model, &metadata, n_cols);
}

/** @} */

// Functions' specializations

/**
 * @brief Compute the feature importances of the trained RandomForest model.
 */
template <class T, class L>
void compute_feature_importances(const RandomForestMetaData<T, L>* forest, T* importances)
{
  if (forest->n_features == 0) { return; }

  int n_cols = forest->n_features;
  std::vector<double> accumulated_importances(n_cols, 0.0);

  for (const auto& tree : forest->trees) {
    std::vector<double> finite_importances(n_cols, 0.0);
    std::vector<double> infinite_importances(n_cols, 0.0);
    bool has_infinite_importance = false;

    if (tree->sparsetree.empty()) continue;
    int root_sample_count = tree->sparsetree[0].InstanceCount();

    if (root_sample_count <= 0) continue;

    // Some criteria can produce +inf split metrics on extreme inputs. Raw
    // normalization would then divide inf by inf and return NaN importances;
    // when that happens, finite contributions are negligible and the useful
    // signal is which features received positive infinite contributions.
    for (const auto& node : tree->sparsetree) {
      if (!node.IsLeaf()) {
        int feature_id = node.ColumnId();
        double contribution =
          static_cast<double>(node.BestMetric()) * static_cast<double>(node.InstanceCount());
        if (std::isfinite(contribution)) {
          if (contribution > 0.0) { finite_importances[feature_id] += contribution; }
        } else if (std::isinf(contribution) && contribution > 0.0) {
          infinite_importances[feature_id] += 1.0;
          has_infinite_importance = true;
        }
      }
    }
    const auto& tree_importances =
      has_infinite_importance ? infinite_importances : finite_importances;
    double sum = 0.0;
    for (int i = 0; i < n_cols; i++) {
      sum += tree_importances[i];
    }

    if (sum > 0) {
      for (int i = 0; i < n_cols; i++) {
        accumulated_importances[i] += tree_importances[i] / sum;
      }
    }
  }

  double sum = 0.0;
  for (auto i = 0; i < n_cols; i++) {
    sum += accumulated_importances[i];
  }
  if (sum > 0) {
    for (auto i = 0; i < n_cols; i++) {
      importances[i] = T(accumulated_importances[i] / sum);
    }
  } else {
    for (auto i = 0; i < n_cols; i++) {
      importances[i] = T(0);
    }
  }
}

template CUML_EXPORT std::string get_rf_detailed_text<float, int>(
  const RandomForestClassifierF* forest);
template CUML_EXPORT std::string get_rf_detailed_text<double, int>(
  const RandomForestClassifierD* forest);
template CUML_EXPORT std::string get_rf_detailed_text<float, float>(
  const RandomForestRegressorF* forest);
template CUML_EXPORT std::string get_rf_detailed_text<double, double>(
  const RandomForestRegressorD* forest);

template CUML_EXPORT std::string get_rf_json<float, int>(const RandomForestClassifierF* forest);
template CUML_EXPORT std::string get_rf_json<double, int>(const RandomForestClassifierD* forest);
template CUML_EXPORT std::string get_rf_json<float, float>(const RandomForestRegressorF* forest);
template CUML_EXPORT std::string get_rf_json<double, double>(const RandomForestRegressorD* forest);

template CUML_EXPORT void build_treelite_forest<float, int>(
  TreeliteModelHandle* model, const RandomForestMetaData<float, int>* forest, int num_features);
template CUML_EXPORT void build_treelite_forest<double, int>(
  TreeliteModelHandle* model, const RandomForestMetaData<double, int>* forest, int num_features);
template CUML_EXPORT void build_treelite_forest<float, float>(
  TreeliteModelHandle* model, const RandomForestMetaData<float, float>* forest, int num_features);
template CUML_EXPORT void build_treelite_forest<double, double>(
  TreeliteModelHandle* model, const RandomForestMetaData<double, double>* forest, int num_features);

// Template instantiations for get functions
template CUML_EXPORT void compute_feature_importances<float, int>(
  const RandomForestMetaData<float, int>* forest, float* importances);
template CUML_EXPORT void compute_feature_importances<double, int>(
  const RandomForestMetaData<double, int>* forest, double* importances);
template CUML_EXPORT void compute_feature_importances<float, float>(
  const RandomForestMetaData<float, float>* forest, float* importances);
template CUML_EXPORT void compute_feature_importances<double, double>(
  const RandomForestMetaData<double, double>* forest, double* importances);

// Template instantiations for fit_treelite
template CUML_EXPORT void fit_treelite<float, int>(const raft::handle_t& user_handle,
                                                   TreeliteModelHandle* model,
                                                   float* input,
                                                   int n_rows,
                                                   int n_cols,
                                                   int* labels,
                                                   int n_unique_labels,
                                                   RF_params rf_params,
                                                   bool* bootstrap_masks,
                                                   float* feature_importances,
                                                   rapids_logger::level_enum verbosity,
                                                   const double* sample_weight,
                                                   bool input_row_major);
template CUML_EXPORT void fit_treelite<double, int>(const raft::handle_t& user_handle,
                                                    TreeliteModelHandle* model,
                                                    double* input,
                                                    int n_rows,
                                                    int n_cols,
                                                    int* labels,
                                                    int n_unique_labels,
                                                    RF_params rf_params,
                                                    bool* bootstrap_masks,
                                                    double* feature_importances,
                                                    rapids_logger::level_enum verbosity,
                                                    const double* sample_weight,
                                                    bool input_row_major);
template CUML_EXPORT void fit_treelite<float, float>(const raft::handle_t& user_handle,
                                                     TreeliteModelHandle* model,
                                                     float* input,
                                                     int n_rows,
                                                     int n_cols,
                                                     float* labels,
                                                     RF_params rf_params,
                                                     bool* bootstrap_masks,
                                                     float* feature_importances,
                                                     rapids_logger::level_enum verbosity,
                                                     const double* sample_weight,
                                                     bool input_row_major);
template CUML_EXPORT void fit_treelite<double, double>(const raft::handle_t& user_handle,
                                                       TreeliteModelHandle* model,
                                                       double* input,
                                                       int n_rows,
                                                       int n_cols,
                                                       double* labels,
                                                       RF_params rf_params,
                                                       bool* bootstrap_masks,
                                                       double* feature_importances,
                                                       rapids_logger::level_enum verbosity,
                                                       const double* sample_weight,
                                                       bool input_row_major);

// Explicit instantiations for the implementation declared in randomforest.cuh.
template class RandomForest<float, int>;
template class RandomForest<float, float>;
template class RandomForest<double, int>;
template class RandomForest<double, double>;

}  // End namespace ML
