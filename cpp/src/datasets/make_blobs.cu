/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/datasets/make_blobs.hpp>

#include <raft/core/handle.hpp>
#include <raft/random/make_blobs.cuh>

namespace ML {
namespace Datasets {

void make_blobs(const raft::handle_t& handle,
                float* out,
                int64_t* labels,
                int64_t n_rows,
                int64_t n_cols,
                int64_t n_clusters,
                bool row_major,
                const float* centers,
                const float* cluster_std,
                const float cluster_std_scalar,
                bool shuffle,
                float center_box_min,
                float center_box_max,
                uint64_t seed)
{
  raft::random::make_blobs(out,
                           labels,
                           n_rows,
                           n_cols,
                           n_clusters,
                           handle.get_stream().get(),
                           row_major,
                           centers,
                           cluster_std,
                           cluster_std_scalar,
                           shuffle,
                           center_box_min,
                           center_box_max,
                           seed);
}

void make_blobs(const raft::handle_t& handle,
                double* out,
                int64_t* labels,
                int64_t n_rows,
                int64_t n_cols,
                int64_t n_clusters,
                bool row_major,
                const double* centers,
                const double* cluster_std,
                const double cluster_std_scalar,
                bool shuffle,
                double center_box_min,
                double center_box_max,
                uint64_t seed)
{
  raft::random::make_blobs(out,
                           labels,
                           n_rows,
                           n_cols,
                           n_clusters,
                           handle.get_stream().get(),
                           row_major,
                           centers,
                           cluster_std,
                           cluster_std_scalar,
                           shuffle,
                           center_box_min,
                           center_box_max,
                           seed);
}

void make_blobs(const raft::handle_t& handle,
                float* out,
                int* labels,
                int n_rows,
                int n_cols,
                int n_clusters,
                bool row_major,
                const float* centers,
                const float* cluster_std,
                const float cluster_std_scalar,
                bool shuffle,
                float center_box_min,
                float center_box_max,
                uint64_t seed)
{
  raft::random::make_blobs(out,
                           labels,
                           n_rows,
                           n_cols,
                           n_clusters,
                           handle.get_stream().get(),
                           row_major,
                           centers,
                           cluster_std,
                           cluster_std_scalar,
                           shuffle,
                           center_box_min,
                           center_box_max,
                           seed);
}

void make_blobs(const raft::handle_t& handle,
                double* out,
                int* labels,
                int n_rows,
                int n_cols,
                int n_clusters,
                bool row_major,
                const double* centers,
                const double* cluster_std,
                const double cluster_std_scalar,
                bool shuffle,
                double center_box_min,
                double center_box_max,
                uint64_t seed)
{
  raft::random::make_blobs(out,
                           labels,
                           n_rows,
                           n_cols,
                           n_clusters,
                           handle.get_stream().get(),
                           row_major,
                           centers,
                           cluster_std,
                           cluster_std_scalar,
                           shuffle,
                           center_box_min,
                           center_box_max,
                           seed);
}
}  // namespace Datasets
}  // namespace ML
