/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "dbscan.cuh"

#include <cuml/cluster/dbscan.hpp>
#include <cuml/common/distance_type.hpp>

#include <raft/util/cudart_utils.hpp>

namespace ML {
namespace Dbscan {

void fit(const raft::handle_t& handle,
         float* input,
         int n_rows,
         int n_cols,
         float eps,
         int min_pts,
         ML::distance::DistanceType metric,
         int* labels,
         int* core_sample_indices,
         float* sample_weight,
         size_t max_bytes_per_batch,
         EpsNnMethod eps_nn_method,
         rapids_logger::level_enum verbosity,
         bool opg)
{
  if (opg)
    dbscanFitImpl<float, int, true>(handle,
                                    input,
                                    n_rows,
                                    n_cols,
                                    eps,
                                    min_pts,
                                    metric,
                                    labels,
                                    core_sample_indices,
                                    sample_weight,
                                    max_bytes_per_batch,
                                    eps_nn_method,
                                    handle.get_stream().get(),
                                    verbosity);
  else
    dbscanFitImpl<float, int, false>(handle,
                                     input,
                                     n_rows,
                                     n_cols,
                                     eps,
                                     min_pts,
                                     metric,
                                     labels,
                                     core_sample_indices,
                                     sample_weight,
                                     max_bytes_per_batch,
                                     eps_nn_method,
                                     handle.get_stream().get(),
                                     verbosity);
}

void fit(const raft::handle_t& handle,
         double* input,
         int n_rows,
         int n_cols,
         double eps,
         int min_pts,
         ML::distance::DistanceType metric,
         int* labels,
         int* core_sample_indices,
         double* sample_weight,
         size_t max_bytes_per_batch,
         EpsNnMethod eps_nn_method,
         rapids_logger::level_enum verbosity,
         bool opg)
{
  if (opg)
    dbscanFitImpl<double, int, true>(handle,
                                     input,
                                     n_rows,
                                     n_cols,
                                     eps,
                                     min_pts,
                                     metric,
                                     labels,
                                     core_sample_indices,
                                     sample_weight,
                                     max_bytes_per_batch,
                                     eps_nn_method,
                                     handle.get_stream().get(),
                                     verbosity);
  else
    dbscanFitImpl<double, int, false>(handle,
                                      input,
                                      n_rows,
                                      n_cols,
                                      eps,
                                      min_pts,
                                      metric,
                                      labels,
                                      core_sample_indices,
                                      sample_weight,
                                      max_bytes_per_batch,
                                      eps_nn_method,
                                      handle.get_stream().get(),
                                      verbosity);
}

void fit(const raft::handle_t& handle,
         float* input,
         int64_t n_rows,
         int64_t n_cols,
         float eps,
         int min_pts,
         ML::distance::DistanceType metric,
         int64_t* labels,
         int64_t* core_sample_indices,
         float* sample_weight,
         size_t max_bytes_per_batch,
         EpsNnMethod eps_nn_method,
         rapids_logger::level_enum verbosity,
         bool opg)
{
  if (opg)
    dbscanFitImpl<float, int64_t, true>(handle,
                                        input,
                                        n_rows,
                                        n_cols,
                                        eps,
                                        min_pts,
                                        metric,
                                        labels,
                                        core_sample_indices,
                                        sample_weight,
                                        max_bytes_per_batch,
                                        eps_nn_method,
                                        handle.get_stream().get(),
                                        verbosity);
  else
    dbscanFitImpl<float, int64_t, false>(handle,
                                         input,
                                         n_rows,
                                         n_cols,
                                         eps,
                                         min_pts,
                                         metric,
                                         labels,
                                         core_sample_indices,
                                         sample_weight,
                                         max_bytes_per_batch,
                                         eps_nn_method,
                                         handle.get_stream().get(),
                                         verbosity);
}

void fit(const raft::handle_t& handle,
         double* input,
         int64_t n_rows,
         int64_t n_cols,
         double eps,
         int min_pts,
         ML::distance::DistanceType metric,
         int64_t* labels,
         int64_t* core_sample_indices,
         double* sample_weight,
         size_t max_bytes_per_batch,
         EpsNnMethod eps_nn_method,
         rapids_logger::level_enum verbosity,
         bool opg)
{
  if (opg)
    dbscanFitImpl<double, int64_t, true>(handle,
                                         input,
                                         n_rows,
                                         n_cols,
                                         eps,
                                         min_pts,
                                         metric,
                                         labels,
                                         core_sample_indices,
                                         sample_weight,
                                         max_bytes_per_batch,
                                         eps_nn_method,
                                         handle.get_stream().get(),
                                         verbosity);
  else
    dbscanFitImpl<double, int64_t, false>(handle,
                                          input,
                                          n_rows,
                                          n_cols,
                                          eps,
                                          min_pts,
                                          metric,
                                          labels,
                                          core_sample_indices,
                                          sample_weight,
                                          max_bytes_per_batch,
                                          eps_nn_method,
                                          handle.get_stream().get(),
                                          verbosity);
}

}  // namespace Dbscan
}  // namespace ML
