/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "benchmark.cuh"

#include <cuml/ensemble/randomforest.hpp>

#include <cmath>
#include <utility>

namespace ML {
namespace Bench {
namespace rf {

struct RegParams {
  DatasetParams data;
  RegressionParams regression;
  RF_params rf;
};

template <typename D>
struct RFRegressorModel {};

template <>
struct RFRegressorModel<float> {
  ML::RandomForestRegressorF model;
};

template <>
struct RFRegressorModel<double> {
  ML::RandomForestRegressorD model;
};

template <typename D>
class RFRegressor : public RegressionFixture<D> {
 public:
  RFRegressor(const std::string& name, const RegParams& p)
    : RegressionFixture<D>(name, p.data, p.regression), rfParams(p.rf)
  {
  }

 protected:
  void runBenchmark(::benchmark::State& state) override
  {
    using MLCommon::Bench::CudaEventTimer;
    this->loopOnState(state, [this]() {
      auto* mPtr = &model.model;
      mPtr->trees.clear();
      fit(*this->handle,
          mPtr,
          this->data.X.data(),
          this->params.nrows,
          this->params.ncols,
          this->data.y.data(),
          rfParams,
          rapids_logger::level_enum::info,
          nullptr,
          nullptr,
          this->params.rowMajor);
      this->handle->sync_stream(this->stream);
    });
  }

 private:
  RFRegressorModel<D> model;
  RF_params rfParams;
};

template <typename D>
std::vector<RegParams> getInputs(int max_leaves)
{
  struct DimInfo {
    int nrows, ncols, n_informative, n_trees;
    float max_features;
  };
  std::vector<RegParams> out;
  RegParams p;
  p.data.rowMajor = false;
  p.regression    = {.n_informative  = 0,
                     .effective_rank = -1,  // dataset generation will be faster
                     .bias           = 4.5,
                     .tail_strength  = 0.5,  // unused when effective_rank = -1
                     .noise          = 1.0,
                     .shuffle        = true,  // Better to shuffle when n_informative < ncols
                     .seed           = 12345ULL};

  p.rf                          = set_rf_params(10,                 /*max_depth */
                       max_leaves,         /* max_leaves */
                       0.3,                /* max_features */
                       32,                 /* max_n_bins */
                       3,                  /* min_samples_leaf */
                       3,                  /* min_samples_split */
                       0.0f,               /* min_impurity_decrease */
                       true,               /* bootstrap */
                       500,                /* n_trees */
                       1.f,                /* max_samples */
                       1234ULL,            /* seed */
                       ML::CRITERION::MSE, /* split_criterion */
                       8,                  /* n_streams */
                       128                 /* max_batch_size */
  );
  std::vector<DimInfo> dim_info = {
    {160000, 64, 48, 500, 0.5f},
    {320000, 128, 96, 300, 0.35f},
    {500000, 500, 400, 100, 0.3f},
  };
  for (auto& di : dim_info) {
    p.data.nrows                  = di.nrows;
    p.data.ncols                  = di.ncols;
    p.regression.n_informative    = di.n_informative;
    p.rf.n_trees                  = di.n_trees;
    p.rf.tree_params.max_features = di.max_features;
    for (auto max_depth : std::vector<int>({7, 11})) {
      p.rf.tree_params.max_depth = max_depth;
      p.data.rowMajor            = false;
      out.push_back(p);
      p.data.rowMajor = true;
      out.push_back(p);
    }
  }
  return out;
}

// Keep the greedy leaf limit nonbinding at the tested depths for a fair policy comparison.
ML_BENCH_REGISTER(RegParams, RFRegressor<float>, "regression/depthwise", getInputs<float>(-1));
ML_BENCH_REGISTER(RegParams, RFRegressor<double>, "regression/depthwise", getInputs<double>(-1));
ML_BENCH_REGISTER(RegParams, RFRegressor<float>, "regression/greedy", getInputs<float>(1 << 20));
ML_BENCH_REGISTER(RegParams, RFRegressor<double>, "regression/greedy", getInputs<double>(1 << 20));

}  // namespace rf
}  // namespace Bench
}  // namespace ML
