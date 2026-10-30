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

struct Params {
  DatasetParams data;
  BlobsParams blobs;
  RF_params rf;
};

template <typename D>
struct RFClassifierModel {};

template <>
struct RFClassifierModel<float> {
  ML::RandomForestClassifierF model;
};

template <>
struct RFClassifierModel<double> {
  ML::RandomForestClassifierD model;
};

template <typename D>
class RFClassifier : public BlobsFixture<D> {
 public:
  RFClassifier(const std::string& name, const Params& p)
    : BlobsFixture<D>(name, p.data, p.blobs), rfParams(p.rf)
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
          this->params.nclasses,
          rfParams,
          rapids_logger::level_enum::info,
          nullptr,
          nullptr,
          this->params.rowMajor);
      this->handle->sync_stream(this->stream);
    });
  }

 private:
  RFClassifierModel<D> model;
  RF_params rfParams;
};

template <typename D>
std::vector<Params> getInputs(int max_leaves)
{
  std::vector<Params> out;
  Params p;
  p.data.rowMajor = false;
  p.blobs         = {10.0,         // cluster_std
                     false,        // shuffle
                     -10.0,        // center_box_min
                     10.0,         // center_box_max
                     2152953ULL};  // seed

  p.rf = set_rf_params(10,                  /*max_depth */
                       max_leaves,          /* max_leaves */
                       0.3,                 /* max_features */
                       32,                  /* max_n_bins */
                       3,                   /* min_samples_leaf */
                       3,                   /* min_samples_split */
                       0.0f,                /* min_impurity_decrease */
                       true,                /* bootstrap */
                       500,                 /* n_trees */
                       1.f,                 /* max_samples */
                       1234ULL,             /* seed */
                       ML::CRITERION::GINI, /* split_criterion */
                       8,                   /* n_streams */
                       128                  /* max_batch_size */
  );

  struct Config {
    int nrows, ncols, nclasses, max_n_bins;
    ML::CRITERION criterion;
  };
  std::vector<Config> configs = {
    {160000, 64, 2, 32, ML::CRITERION::GINI},
    {640000, 64, 8, 32, ML::CRITERION::GINI},
    {1184000, 968, 2, 32, ML::CRITERION::GINI},  // Mimicking Bosch dataset
    // High class count (44) + large bins (128) stress the shared-memory histogram path.
    {160000, 64, 44, 128, ML::CRITERION::GINI},
    // Entropy variant exercises the entropy-{float,double}.cu instantiations.
    {640000, 64, 8, 32, ML::CRITERION::ENTROPY},
  };
  for (auto& cfg : configs) {
    // Let's run Bosch only for float type
    if (!std::is_same<D, float>::value && cfg.ncols == 968) continue;
    p.data.nrows                     = cfg.nrows;
    p.data.ncols                     = cfg.ncols;
    p.data.nclasses                  = cfg.nclasses;
    p.rf.tree_params.max_n_bins      = cfg.max_n_bins;
    p.rf.tree_params.split_criterion = cfg.criterion;
    p.rf.tree_params.max_features    = 1.f / std::sqrt(float(cfg.ncols));
    for (auto max_depth : std::vector<int>({7, 9})) {
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
ML_BENCH_REGISTER(Params, RFClassifier<float>, "blobs/depthwise", getInputs<float>(-1));
ML_BENCH_REGISTER(Params, RFClassifier<double>, "blobs/depthwise", getInputs<double>(-1));
ML_BENCH_REGISTER(Params, RFClassifier<float>, "blobs/greedy", getInputs<float>(1 << 20));
ML_BENCH_REGISTER(Params, RFClassifier<double>, "blobs/greedy", getInputs<double>(1 << 20));

}  // end namespace rf
}  // end namespace Bench
}  // end namespace ML
