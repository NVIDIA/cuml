/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/cluster/kmeans_params.hpp>

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/distance/distance.hpp>

namespace ML::kmeans {

inline cuvs::cluster::kmeans::params to_cuvs(KMeansParams const& config)
{
  cuvs::cluster::kmeans::params params;

  params.metric              = static_cast<cuvs::distance::DistanceType>(config.metric);
  params.n_clusters          = config.n_clusters;
  params.init                = static_cast<cuvs::cluster::kmeans::params::InitMethod>(config.init);
  params.max_iter            = config.max_iter;
  params.tol                 = config.tol;
  params.verbosity           = config.verbosity;
  params.rng_state           = config.rng_state;
  params.n_init              = config.n_init;
  params.oversampling_factor = config.oversampling_factor;
  params.batch_samples       = config.batch_samples;
  params.batch_centroids     = config.batch_centroids;
  params.init_size           = config.init_size;
  params.device_buffer_samples = config.device_buffer_samples;

  return params;
}

}  // namespace ML::kmeans
