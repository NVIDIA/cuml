/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cuml/common/distance_type.hpp>
#include <cuml/common/export.hpp>

#include <raft/random/rng_state.hpp>

#include <rapids_logger/logger.hpp>

namespace CUML_EXPORT ML {
namespace kmeans {

struct KMeansParams {
  enum class InitMethod { KMeansPlusPlus, Random, Array };
  ML::distance::DistanceType metric   = ML::distance::DistanceType::L2Expanded;
  int n_clusters                      = 8;
  InitMethod init                     = InitMethod::KMeansPlusPlus;
  int max_iter                        = 300;
  double tol                          = 1e-4;
  rapids_logger::level_enum verbosity = rapids_logger::level_enum::info;
  raft::random::RngState rng_state{0, raft::random::GeneratorType::GenPhilox};
  int n_init                         = 1;
  double oversampling_factor         = 2.0;
  int batch_samples                  = 1 << 15;
  int batch_centroids                = 0;
  std::int64_t init_size             = 0;
  std::int64_t device_buffer_samples = 0;
};

}  // end namespace kmeans
}  // end namespace CUML_EXPORT ML
