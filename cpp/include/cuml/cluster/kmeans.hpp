/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/cluster/kmeans_params.hpp>
#include <cuml/common/export.hpp>

namespace raft {
class handle_t;
}

namespace CUML_EXPORT ML {

namespace kmeans {

/**
 * @brief Compute k-means clustering for each sample in the input.
 *
 * @param[in]     handle        The handle to the cuML library context that
 manages the CUDA resources.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster, in row-major
 * format. May or may not be device accessible.
 * @param[in]     n_samples     Number of samples in the input X.
 * @param[in]     n_features    Number of features or the dimensions of each
 * sample.
 * @param[in]     sample_weight The weights for each observation in X. If non-null, sample_weight
 must have the same memory residency as X.
 * @param[inout]  centroids     [in] When init is InitMethod::Array, use
 centroids  as the initial cluster centers
 *                              [out] Otherwise, generated centroids from the
 kmeans algorithm is stored at the address pointed by 'centroids'. `centroids`
 * must always be device accessible.
 * @param[out]    inertia       Sum of squared distances of samples to their
 closest cluster center.
 * @param[out]    n_iter        Number of iterations run.
 */
void fit(const raft::handle_t& handle,
         const KMeansParams& params,
         const float* X,
         int n_samples,
         int n_features,
         const float* sample_weight,
         float* centroids,
         float& inertia,
         int& n_iter);

void fit(const raft::handle_t& handle,
         const KMeansParams& params,
         const double* X,
         int n_samples,
         int n_features,
         const double* sample_weight,
         double* centroids,
         double& inertia,
         int& n_iter);

void fit(const raft::handle_t& handle,
         const KMeansParams& params,
         const float* X,
         int64_t n_samples,
         int64_t n_features,
         const float* sample_weight,
         float* centroids,
         float& inertia,
         int64_t& n_iter);

void fit(const raft::handle_t& handle,
         const KMeansParams& params,
         const double* X,
         int64_t n_samples,
         int64_t n_features,
         const double* sample_weight,
         double* centroids,
         double& inertia,
         int64_t& n_iter);

/**
 * @brief Multi-GPU / out-of-core k-means fit over multiple local data partitions.
 *
 * Each rank (e.g. each Dask worker) supplies its local training data as an
 * array of `n_parts` partitions.
 * All partitions on a given rank must share the same residency. The
 * distributed reduction across ranks is performed via the NCCL communicator
 * that must be initialized on `handle`.
 *
 * @param[in]     handle              cuML handle with NCCL comms initialized.
 * @param[in]     params              Parameters for the KMeans model. For
 *                                    host-resident partitions the
 *                                    host-to-device batch size is read from
 *                                    `params.device_buffer_samples`.
 * @param[in]     X_parts             Array of `n_parts` pointers to the local
 *                                    row-major partitions (all host- or all
 *                                    device-resident). Partition `i` has shape
 *                                    [`n_samples_parts[i]`, `n_features`].
 * @param[in]     n_samples_parts     Array of `n_parts` per-partition row counts.
 * @param[in]     n_parts             Number of local partitions on this rank.
 * @param[in]     n_features          Number of features (shared by all partitions).
 * @param[in]     sample_weight_parts Optional array of `n_parts` pointers to the
 *                                    per-partition weight vectors (matching the
 *                                    residency of `X_parts`), or `nullptr` for
 *                                    uniform weights.
 * @param[inout]  centroids           Device matrix [n_clusters x n_features].
 * @param[out]    inertia             Sum of squared distances to closest center.
 * @param[out]    n_iter              Number of iterations run.
 */
void fit(const raft::handle_t& handle,
         const KMeansParams& params,
         const float* const* X_parts,
         const int64_t* n_samples_parts,
         int64_t n_parts,
         int64_t n_features,
         const float* const* sample_weight_parts,
         float* centroids,
         float& inertia,
         int64_t& n_iter);

void fit(const raft::handle_t& handle,
         const KMeansParams& params,
         const double* const* X_parts,
         const int64_t* n_samples_parts,
         int64_t n_parts,
         int64_t n_features,
         const double* const* sample_weight_parts,
         double* centroids,
         double& inertia,
         int64_t& n_iter);

/**
 * @brief Predict the closest cluster each sample in X belongs to.
 *
 * @param[in]     handle            The handle to the cuML library context
 * that manages the CUDA resources.
 * @param[in]     params            Parameters for KMeans model.
 * @param[in]     centroids         Cluster centroids. It must be noted that
 * the data must be in row-major format and stored in device accessible
 * location.
 * @param[in]     X                 New data to predict.
 * @param[in]     n_samples         Number of samples in the input X.
 * @param[in]     n_features        Number of features or the dimensions of
 * each sample in 'X' (value should be same as the dimension for each cluster
 * centers in 'centroids').
 * @param[in]     sample_weight     The weights for each observation in X.
 * @param[in]     normalize_weights True if the weights should be normalized
 * @param[out]    labels            Index of the cluster each sample in X
 * belongs to.
 * @param[out]    inertia           Sum of squared distances of samples to
 * their closest cluster center.
 */

void predict(const raft::handle_t& handle,
             const KMeansParams& params,
             const float* centroids,
             const float* X,
             int n_samples,
             int n_features,
             const float* sample_weight,
             bool normalize_weights,
             int* labels,
             float& inertia);

void predict(const raft::handle_t& handle,
             const KMeansParams& params,
             const double* centroids,
             const double* X,
             int n_samples,
             int n_features,
             const double* sample_weight,
             bool normalize_weights,
             int* labels,
             double& inertia);
void predict(const raft::handle_t& handle,
             const KMeansParams& params,
             const float* centroids,
             const float* X,
             int64_t n_samples,
             int64_t n_features,
             const float* sample_weight,
             bool normalize_weights,
             int64_t* labels,
             float& inertia);

void predict(const raft::handle_t& handle,
             const KMeansParams& params,
             const double* centroids,
             const double* X,
             int64_t n_samples,
             int64_t n_features,
             const double* sample_weight,
             bool normalize_weights,
             int64_t* labels,
             double& inertia);
/**
 * @brief Transform X to a cluster-distance space.
 *
 * @param[in]     handle        The handle to the cuML library context that
 * manages the CUDA resources.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     centroids     Cluster centroids. It must be noted that the
 * data must be in row-major format and stored in device accessible location.
 * @param[in]     X             Training instances to cluster. It must be noted
 * that the data must be in row-major format and stored in device accessible
 * location.
 * @param[in]     n_samples     Number of samples in the input X.
 * @param[in]     n_features    Number of features or the dimensions of each
 * sample in 'X' (it should be same as the dimension for each cluster centers in
 * 'centroids').
 * @param[out]    X_new         X transformed in the new space..
 */
void transform(const raft::handle_t& handle,
               const KMeansParams& params,
               const float* centroids,
               const float* X,
               int n_samples,
               int n_features,
               float* X_new);

void transform(const raft::handle_t& handle,
               const KMeansParams& params,
               const double* centroids,
               const double* X,
               int n_samples,
               int n_features,
               double* X_new);
void transform(const raft::handle_t& handle,
               const KMeansParams& params,
               const float* centroids,
               const float* X,
               int64_t n_samples,
               int64_t n_features,
               float* X_new);

void transform(const raft::handle_t& handle,
               const KMeansParams& params,
               const double* centroids,
               const double* X,
               int64_t n_samples,
               int64_t n_features,
               double* X_new);
};  // end namespace kmeans
};  // end namespace CUML_EXPORT ML
