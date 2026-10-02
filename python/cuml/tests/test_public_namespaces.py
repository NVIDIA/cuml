#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import sklearn.model_selection
import sklearn.pipeline

import cuml.impute
import cuml.manifold
import cuml.metrics
import cuml.metrics.cluster.silhouette_score
import cuml.metrics.pairwise
import cuml.metrics.trustworthiness
import cuml.model_selection
import cuml.pipeline
import cuml.preprocessing


def test_impute_namespace_exports():
    assert cuml.impute.SimpleImputer is cuml.preprocessing.SimpleImputer
    assert cuml.impute.MissingIndicator is cuml.preprocessing.MissingIndicator
    assert set(cuml.impute.__all__) == {"SimpleImputer", "MissingIndicator"}


def test_metrics_pairwise_namespace_exports():
    assert (
        cuml.metrics.pairwise.pairwise_distances
        is cuml.metrics.pairwise_distances
    )
    assert (
        cuml.metrics.pairwise.pairwise_kernels is cuml.metrics.pairwise_kernels
    )
    assert (
        cuml.metrics.pairwise.nan_euclidean_distances
        is cuml.metrics.nan_euclidean_distances
    )
    assert (
        cuml.metrics.pairwise.PAIRWISE_DISTANCE_METRICS
        is cuml.metrics.PAIRWISE_DISTANCE_METRICS
    )
    assert (
        cuml.metrics.pairwise.PAIRWISE_KERNEL_FUNCTIONS
        is cuml.metrics.PAIRWISE_KERNEL_FUNCTIONS
    )
    assert cuml.metrics.pairwise is cuml.metrics.pairwise


def test_metrics_silhouette_exports():
    assert (
        cuml.metrics.silhouette_score
        is cuml.metrics.cluster.silhouette_score.cython_silhouette_score
    )
    assert (
        cuml.metrics.silhouette_samples
        is cuml.metrics.cluster.silhouette_score.cython_silhouette_samples
    )
    assert "silhouette_score" in cuml.metrics.__all__
    assert "silhouette_samples" in cuml.metrics.__all__
    assert "pairwise" in cuml.metrics.__all__


def test_manifold_trustworthiness_export():
    assert (
        cuml.manifold.trustworthiness
        is cuml.metrics.trustworthiness.trustworthiness
    )
    assert "trustworthiness" in cuml.manifold.__all__


def test_pipeline_wrappers():
    assert cuml.pipeline.Pipeline is sklearn.pipeline.Pipeline
    assert cuml.pipeline.make_pipeline is sklearn.pipeline.make_pipeline
    assert cuml.pipeline.FeatureUnion is sklearn.pipeline.FeatureUnion
    assert cuml.pipeline.make_union is sklearn.pipeline.make_union
    assert "FeatureUnion" in cuml.pipeline.__all__
    assert "make_union" in cuml.pipeline.__all__

    with pytest.raises(AttributeError):
        _ = cuml.pipeline.non_existent_attribute


def test_model_selection_wrappers():
    assert (
        cuml.model_selection.GridSearchCV
        is sklearn.model_selection.GridSearchCV
    )
    assert (
        cuml.model_selection.RandomizedSearchCV
        is sklearn.model_selection.RandomizedSearchCV
    )
    assert "RandomizedSearchCV" in cuml.model_selection.__all__

    with pytest.raises(AttributeError):
        _ = cuml.model_selection.non_existent_attribute
