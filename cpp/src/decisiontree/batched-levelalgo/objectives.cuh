/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "bins.cuh"
#include "split.cuh"

#include <cuml/tree/algo_helper.h>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace ML {
namespace DT {
template <typename DataT_, typename LabelT_, bool weighted_ = false>
class ClassificationObjectiveFunction {
 public:
  using DataT  = DataT_;
  using LabelT = LabelT_;
  using BinT   = std::conditional_t<weighted_, WeightedClassificationBin, ClassificationBin>;
  static constexpr bool weighted = weighted_;

 private:
  std::int64_t nclasses;
  std::int64_t min_samples_leaf;
  CRITERION criterion;
  DataT min_impurity_decrease;
  double root_weight;

  HDI double WeightAt(BinT const* hist, std::int64_t i, std::int64_t n_bins) const
  {
    double weight = 0.0;
    for (std::int64_t j = 0; j < nclasses; ++j) {
      weight += hist[n_bins * j + i].Weight();
    }
    return weight;
  }

  HDI DataT GiniGain(BinT const* hist,
                     std::int64_t i,
                     std::int64_t n_bins,
                     std::int64_t,
                     std::int64_t,
                     std::int64_t) const
  {
    constexpr DataT One = DataT(1.0);
    auto total_weight   = WeightAt(hist, n_bins - 1, n_bins);
    auto left_weight    = WeightAt(hist, i, n_bins);
    auto right_weight   = total_weight - left_weight;

    if (total_weight <= 0.0 || left_weight <= 0.0 || right_weight <= 0.0)
      return -std::numeric_limits<DataT>::max();

    auto invLen   = One / DataT(total_weight);
    auto invLeft  = One / DataT(left_weight);
    auto invRight = One / DataT(right_weight);
    auto gain     = DataT(0.0);

    for (std::int64_t j = 0; j < nclasses; ++j) {
      double val_i = 0.0;
      auto lval_i  = hist[n_bins * j + i].Weight();
      auto lval    = DataT(lval_i);
      gain += lval * invLeft * lval * invLen;

      val_i += lval_i;
      auto total_sum = hist[n_bins * j + n_bins - 1].Weight();
      auto rval_i    = total_sum - lval_i;
      auto rval      = DataT(rval_i);
      gain += rval * invRight * rval * invLen;

      val_i += rval_i;
      auto val = DataT(val_i) * invLen;
      gain -= val * val;
    }

    return gain;
  }

  HDI DataT EntropyGain(BinT const* hist,
                        std::int64_t i,
                        std::int64_t n_bins,
                        std::int64_t,
                        std::int64_t,
                        std::int64_t) const
  {
    auto total_weight = WeightAt(hist, n_bins - 1, n_bins);
    auto left_weight  = WeightAt(hist, i, n_bins);
    auto right_weight = total_weight - left_weight;

    if (total_weight <= 0.0 || left_weight <= 0.0 || right_weight <= 0.0)
      return -std::numeric_limits<DataT>::max();

    auto gain{DataT(0.0)};
    auto invLeft{DataT(1.0) / DataT(left_weight)};
    auto invRight{DataT(1.0) / DataT(right_weight)};
    auto invLen{DataT(1.0) / DataT(total_weight)};
    for (std::int64_t c = 0; c < nclasses; ++c) {
      double val_i = 0.0;
      auto lval_i  = hist[n_bins * c + i].Weight();
      if (lval_i != 0) {
        auto lval = DataT(lval_i);
        gain += raft::log(lval * invLeft) / raft::log(DataT(2)) * lval * invLen;
      }

      val_i += lval_i;
      auto total_sum = hist[n_bins * c + n_bins - 1].Weight();
      auto rval_i    = total_sum - lval_i;
      if (rval_i != 0) {
        auto rval = DataT(rval_i);
        gain += raft::log(rval * invRight) / raft::log(DataT(2)) * rval * invLen;
      }

      val_i += rval_i;
      if (val_i != 0) {
        auto val = DataT(val_i) * invLen;
        gain -= val * raft::log(val) / raft::log(DataT(2));
      }
    }

    return gain;
  }

 public:
  HDI DataT GainPerSplit(BinT const* hist,
                         std::int64_t i,
                         std::int64_t n_bins,
                         std::int64_t len,
                         std::int64_t nLeft,
                         std::int64_t nRight) const
  {
    if (nLeft < static_cast<std::int64_t>(min_samples_leaf) ||
        nRight < static_cast<std::int64_t>(min_samples_leaf))
      return -std::numeric_limits<DataT>::max();

    switch (criterion) {
      case CRITERION::GINI: return GiniGain(hist, i, n_bins, len, nLeft, nRight);
      case CRITERION::ENTROPY: return EntropyGain(hist, i, n_bins, len, nLeft, nRight);
      default: return -std::numeric_limits<DataT>::max();
    }
  }

  HDI ClassificationObjectiveFunction(std::int64_t nclasses,
                                      std::int64_t min_samples_leaf,
                                      CRITERION criterion,
                                      double root_weight,
                                      DataT min_impurity_decrease = DataT{0})
    : nclasses(nclasses),
      min_samples_leaf(min_samples_leaf),
      criterion(criterion),
      min_impurity_decrease(min_impurity_decrease),
      root_weight(root_weight)
  {
  }

  DI std::int64_t NumClasses() const { return nclasses; }

  template <typename DatasetT>
  DI void IncrementHistogram(BinT* histogram,
                             int n_bins,
                             int bin,
                             LabelT label,
                             const DatasetT& dataset,
                             std::int64_t row) const
  {
    double weight = 1.0;
    if constexpr (weighted) {
      weight = dataset.sample_weight == nullptr ? 1.0 : double(dataset.sample_weight[row]);
    }
    BinT::IncrementHistogram(histogram, n_bins, bin, label, weight);
  }

  DI Split<DataT> Gain(BinT const* shist,
                       DataT const* squantiles,
                       std::int64_t col,
                       std::int64_t len,
                       std::int64_t n_bins) const
  {
    Split<DataT> sp;
    auto weight_fraction = WeightAt(shist, n_bins - 1, n_bins) / root_weight;
    for (std::int64_t i = threadIdx.x; i < n_bins; i += blockDim.x) {
      auto nLeft  = detail::CountLeft(shist, i, n_bins, nclasses);
      auto nRight = len - nLeft;
      if (nLeft >= static_cast<std::int64_t>(min_samples_leaf) &&
          nRight >= static_cast<std::int64_t>(min_samples_leaf)) {
        auto gain = GainPerSplit(shist, i, n_bins, len, nLeft, nRight);
        // Use weighted gain for the threshold; store local gain.
        if (gain * weight_fraction > min_impurity_decrease) {
          sp.update(squantiles[i], col, gain, nLeft, i);
        }
      }
    }
    return sp;
  }

  static DI void SetLeafVector(BinT const* shist, int nclasses, DataT* out)
  {
    // Output probability
    double total = 0.0;
    for (int i = 0; i < nclasses; i++) {
      total += shist[i].Weight();
    }
    if (total <= 0.0) {
      for (int i = 0; i < nclasses; i++) {
        out[i] = DataT(0);
      }
      return;
    }
    for (int i = 0; i < nclasses; i++) {
      out[i] = DataT(shist[i].Weight()) / total;
    }
  }
};

template <typename DataT_, typename LabelT_, bool weighted_ = false>
class RegressionObjectiveFunction {
 public:
  using DataT  = DataT_;
  using LabelT = LabelT_;
  using BinT   = std::conditional_t<weighted_, WeightedRegressionBin, RegressionBin>;
  static constexpr bool weighted = weighted_;

 private:
  std::int64_t min_samples_leaf;
  CRITERION criterion;
  DataT min_impurity_decrease;
  double root_weight;
  static constexpr auto eps_ = 10 * std::numeric_limits<DataT>::epsilon();

  HDI DataT MSEGain(BinT const* hist,
                    std::int64_t i,
                    std::int64_t n_bins,
                    std::int64_t,
                    std::int64_t,
                    std::int64_t) const
  {
    auto parent_weight = hist[n_bins - 1].Weight();
    auto left_weight   = hist[i].Weight();
    auto right_weight  = parent_weight - left_weight;

    if (parent_weight <= 0.0 || left_weight <= 0.0 || right_weight <= 0.0)
      return -std::numeric_limits<DataT>::max();

    auto invLen           = DataT(1.0) / DataT(parent_weight);
    auto label_sum        = DataT(hist[n_bins - 1].LabelSum());
    auto left_label_sum   = DataT(hist[i].LabelSum());
    DataT parent_obj      = -label_sum * label_sum * invLen;
    DataT left_obj        = -(left_label_sum * left_label_sum) / DataT(left_weight);
    DataT right_label_sum = label_sum - left_label_sum;
    DataT right_obj       = -(right_label_sum * right_label_sum) / DataT(right_weight);
    DataT gain            = parent_obj - (left_obj + right_obj);
    gain *= invLen;

    return gain;
  }

  HDI DataT PoissonGain(BinT const* hist,
                        std::int64_t i,
                        std::int64_t n_bins,
                        std::int64_t,
                        std::int64_t,
                        std::int64_t) const
  {
    auto parent_weight = hist[n_bins - 1].Weight();
    auto left_weight   = hist[i].Weight();
    auto right_weight  = parent_weight - left_weight;

    if (parent_weight <= 0.0 || left_weight <= 0.0 || right_weight <= 0.0)
      return -std::numeric_limits<DataT>::max();

    auto invLen          = DataT(1) / DataT(parent_weight);
    auto label_sum       = DataT(hist[n_bins - 1].LabelSum());
    auto left_label_sum  = DataT(hist[i].LabelSum());
    auto right_label_sum = DataT(hist[n_bins - 1].LabelSum() - hist[i].LabelSum());

    // label sum cannot be non-positive
    if (label_sum <= eps_ || left_label_sum <= eps_ || right_label_sum <= eps_)
      return -std::numeric_limits<DataT>::max();

    DataT parent_obj = -label_sum * raft::log(label_sum * invLen);
    DataT left_obj   = -left_label_sum * raft::log(left_label_sum / DataT(left_weight));
    DataT right_obj  = -right_label_sum * raft::log(right_label_sum / DataT(right_weight));
    DataT gain       = parent_obj - (left_obj + right_obj);
    gain             = gain * invLen;

    return gain;
  }

  HDI DataT GammaGain(BinT const* hist,
                      std::int64_t i,
                      std::int64_t n_bins,
                      std::int64_t,
                      std::int64_t,
                      std::int64_t) const
  {
    auto parent_weight = hist[n_bins - 1].Weight();
    auto left_weight   = hist[i].Weight();
    auto right_weight  = parent_weight - left_weight;

    if (parent_weight <= 0.0 || left_weight <= 0.0 || right_weight <= 0.0)
      return -std::numeric_limits<DataT>::max();

    auto invLen          = DataT(1) / DataT(parent_weight);
    auto label_sum       = DataT(hist[n_bins - 1].LabelSum());
    auto left_label_sum  = DataT(hist[i].LabelSum());
    auto right_label_sum = DataT(hist[n_bins - 1].LabelSum() - hist[i].LabelSum());

    // label sum cannot be non-positive
    if (label_sum <= eps_ || left_label_sum <= eps_ || right_label_sum <= eps_)
      return -std::numeric_limits<DataT>::max();

    DataT parent_obj = DataT(parent_weight) * raft::log(label_sum * invLen);
    DataT left_obj   = DataT(left_weight) * raft::log(left_label_sum / DataT(left_weight));
    DataT right_obj  = DataT(right_weight) * raft::log(right_label_sum / DataT(right_weight));
    DataT gain       = parent_obj - (left_obj + right_obj);
    gain             = gain * invLen;

    return gain;
  }

  HDI DataT InverseGaussianGain(BinT const* hist,
                                std::int64_t i,
                                std::int64_t n_bins,
                                std::int64_t,
                                std::int64_t,
                                std::int64_t) const
  {
    auto parent_weight = hist[n_bins - 1].Weight();
    auto left_weight   = hist[i].Weight();
    auto right_weight  = parent_weight - left_weight;

    if (parent_weight <= 0.0 || left_weight <= 0.0 || right_weight <= 0.0)
      return -std::numeric_limits<DataT>::max();

    auto label_sum       = DataT(hist[n_bins - 1].LabelSum());
    auto left_label_sum  = DataT(hist[i].LabelSum());
    auto right_label_sum = DataT(hist[n_bins - 1].LabelSum() - hist[i].LabelSum());

    // label sum cannot be non-positive
    if (label_sum <= eps_ || left_label_sum <= eps_ || right_label_sum <= eps_)
      return -std::numeric_limits<DataT>::max();

    DataT parent_obj = -DataT(parent_weight) * DataT(parent_weight) / label_sum;
    DataT left_obj   = -DataT(left_weight) * DataT(left_weight) / left_label_sum;
    DataT right_obj  = -DataT(right_weight) * DataT(right_weight) / right_label_sum;
    DataT gain       = parent_obj - (left_obj + right_obj);
    gain             = gain / (2 * DataT(parent_weight));

    return gain;
  }

 public:
  HDI DataT GainPerSplit(BinT const* hist,
                         std::int64_t i,
                         std::int64_t n_bins,
                         std::int64_t len,
                         std::int64_t nLeft,
                         std::int64_t nRight) const
  {
    if (nLeft < static_cast<std::int64_t>(min_samples_leaf) ||
        nRight < static_cast<std::int64_t>(min_samples_leaf))
      return -std::numeric_limits<DataT>::max();

    switch (criterion) {
      case CRITERION::MSE: return MSEGain(hist, i, n_bins, len, nLeft, nRight);
      case CRITERION::POISSON: return PoissonGain(hist, i, n_bins, len, nLeft, nRight);
      case CRITERION::GAMMA: return GammaGain(hist, i, n_bins, len, nLeft, nRight);
      case CRITERION::INVERSE_GAUSSIAN:
        return InverseGaussianGain(hist, i, n_bins, len, nLeft, nRight);
      default: return -std::numeric_limits<DataT>::max();
    }
  }

  HDI RegressionObjectiveFunction(std::int64_t,
                                  std::int64_t min_samples_leaf,
                                  CRITERION criterion,
                                  double root_weight,
                                  DataT min_impurity_decrease = DataT{0})
    : min_samples_leaf(min_samples_leaf),
      criterion(criterion),
      min_impurity_decrease(min_impurity_decrease),
      root_weight(root_weight)
  {
  }

  DI std::int64_t NumClasses() const { return 1; }

  template <typename DatasetT>
  DI void IncrementHistogram(BinT* histogram,
                             int n_bins,
                             int bin,
                             LabelT label,
                             const DatasetT& dataset,
                             std::int64_t row) const
  {
    double weight = 1.0;
    if constexpr (weighted) {
      weight = dataset.sample_weight == nullptr ? 1.0 : double(dataset.sample_weight[row]);
    }
    BinT::IncrementHistogram(histogram, n_bins, bin, label, weight);
  }

  DI Split<DataT> Gain(BinT const* shist,
                       DataT const* squantiles,
                       std::int64_t col,
                       std::int64_t len,
                       std::int64_t n_bins) const
  {
    Split<DataT> sp;
    auto weight_fraction = shist[n_bins - 1].Weight() / root_weight;
    for (std::int64_t i = threadIdx.x; i < n_bins; i += blockDim.x) {
      auto nLeft  = detail::CountLeft(shist, i, n_bins, std::int64_t{1});
      auto nRight = len - nLeft;
      if (nLeft >= static_cast<std::int64_t>(min_samples_leaf) &&
          nRight >= static_cast<std::int64_t>(min_samples_leaf)) {
        auto gain = GainPerSplit(shist, i, n_bins, len, nLeft, nRight);
        // Use weighted gain for the threshold; store local gain.
        if (gain * weight_fraction > min_impurity_decrease) {
          sp.update(squantiles[i], col, gain, nLeft, i);
        }
      }
    }
    return sp;
  }

  static DI void SetLeafVector(BinT const* shist, int nclasses, DataT* out)
  {
    for (int i = 0; i < nclasses; i++) {
      auto weight = shist[i].Weight();
      out[i]      = weight > 0.0 ? shist[i].LabelSum() / weight : DataT(0);
    }
  }
};
}  // end namespace DT
}  // end namespace ML
