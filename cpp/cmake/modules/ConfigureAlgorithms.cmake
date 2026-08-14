# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

if(CUML_ALGORITHMS STREQUAL "ALL")
  set(CUML_USE_RAFT_NN ON)
  set(LINK_TREELITE ON)
  set(LINK_CUFFT ON)
  set(LINK_CUVS ON)
  set(LINK_NVFOREST ON)
  set(all_algo ON)
  # setting treeshap to ON to get the gputreeshap include in the cuml target
  set(treeshap_algo ON)
else()

  # Initial configurable version only supports single GPU and no example compilation
  set(SINGLEGPU ON)
  set(BUILD_CUML_BENCH OFF)
  set(BUILD_CUML_EXAMPLES OFF)
  set(CUML_USE_RAFT_NN OFF)

  foreach(algo ${CUML_ALGORITHMS})
    string(TOLOWER ${algo} lower_algo)
    set(${lower_algo}_algo ON)
  endforeach()

  # Set groups of algorithms based on include/scikit-learn #######

  if(cluster_algo)
    set(dbscan_algo ON)
    set(hdbscan_algo ON)
    set(kmeans_algo ON)
    set(hierarchicalclustering_algo ON)
    set(spectralclustering_algo ON)
  endif()

  if(decomposition_algo)
    set(pca_algo ON)
    set(tsvd_algo ON)
  endif()

  if(ensemble_algo)
    set(randomforest_algo ON)
    set(isolationforest_algo ON)
  endif()

  # todo: organize linear model headers better
  if(linear_model_algo)
    set(linearregression_algo ON)
    set(ridge_algo ON)
    set(lasso_algo ON)
    set(logisticregression_algo ON)

    # we need solvers for ridge, lasso, logistic
    set(solvers_algo ON)
  endif()

  if(manifold_algo)
    set(tsne_algo ON)
    set(umap_algo ON)
  endif()

  if(solvers_algo)
    set(lars_algo ON)
    set(cd_algo ON)
    set(sgd_algo ON)
    set(qn_algo ON)
  endif()

  if(tsa_algo)
    set(arima_algo ON)
    set(autoarima_algo ON)
    set(holtwinters_algo ON)
  endif()

  # Set linking options and algorithms that require other algorithms #######

  if(treeshap_algo)
    set(LINK_TREELITE ON)
  endif()

  if(hdbscan_algo)
    set(hierarchicalclustering_algo ON)
  endif()

  if(hdbscan_algo
     OR tsne_algo
     OR umap_algo
  )
    set(knn_algo ON)
  endif()

  if(tsne_algo)
    set(LINK_CUFFT ON)
  endif()

  if(knn_algo)
    set(CUML_USE_RAFT_NN ON)
  endif()

  if(randomforest_algo)
    set(decisiontree_algo ON)
    set(LINK_TREELITE ON)
    set(LINK_NVFOREST ON)
  endif()

  if(isolationforest_algo)
    set(decisiontree_algo ON)
    set(LINK_TREELITE ON)
  endif()

  if(hierarchicalclustering_algo OR kmeans_algo)
    set(metrics_algo ON)
  endif()

  if(dbscan_algo
     OR hdbscan_algo
     OR kmeans_algo
     OR knn_algo
     OR metrics_algo
     OR tsne_algo
     OR umap_algo
  )
    set(LINK_CUVS ON)
  endif()
endif()
