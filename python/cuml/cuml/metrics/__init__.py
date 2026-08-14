#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from cuml.metrics._classification import accuracy_score, log_loss
from cuml.metrics._ranking import precision_recall_curve, roc_auc_score
from cuml.metrics.cluster.adjusted_rand_index import adjusted_rand_score
from cuml.metrics.cluster.completeness_score import (
    cython_completeness_score as completeness_score,
)
from cuml.metrics.cluster.entropy import cython_entropy as entropy
from cuml.metrics.cluster.homogeneity_score import (
    cython_homogeneity_score as homogeneity_score,
)
from cuml.metrics.cluster.mutual_info_score import (
    cython_mutual_info_score as mutual_info_score,
)
from cuml.metrics.cluster.v_measure import cython_v_measure as v_measure_score
from cuml.metrics.confusion_matrix import confusion_matrix
from cuml.metrics.hinge_loss import hinge_loss
from cuml.metrics.kl_divergence import kl_divergence
from cuml.metrics.pairwise_distances import (
    PAIRWISE_DISTANCE_METRICS,
    PAIRWISE_DISTANCE_SPARSE_METRICS,
    nan_euclidean_distances,
    pairwise_distances,
)
from cuml.metrics.pairwise_kernels import (
    PAIRWISE_KERNEL_FUNCTIONS,
    pairwise_kernels,
)
from cuml.metrics.regression import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    r2_score,
)
from cuml.metrics.trustworthiness import trustworthiness

__all__ = [
    "trustworthiness",
    "r2_score",
    "mean_squared_error",
    "mean_squared_log_error",
    "mean_absolute_error",
    "median_absolute_error",
    "accuracy_score",
    "adjusted_rand_score",
    "roc_auc_score",
    "precision_recall_curve",
    "log_loss",
    "homogeneity_score",
    "completeness_score",
    "mutual_info_score",
    "confusion_matrix",
    "entropy",
    "nan_euclidean_distances",
    "pairwise_distances",
    "pairwise_kernels",
    "hinge_loss",
    "kl_divergence",
    "v_measure_score",
]
