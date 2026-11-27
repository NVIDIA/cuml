#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import math

import cupy as cp
import cupyx.scipy.sparse as cp_sp
import numpy as np

from cuml.metrics.cluster._expected_mutual_information import (
    expected_mutual_information,
)
from cuml.metrics.cluster.entropy import cython_entropy as entropy
from cuml.metrics.cluster.mutual_info_score import (
    cython_mutual_info_score as mutual_info_score,
)
from cuml.metrics.cluster.utils import prepare_cluster_metric_inputs

_AVERAGE_METHODS = ("min", "geometric", "arithmetic", "max")


def _generalized_average(u, v, average_method) -> float:
    """Return a particular mean of two numbers."""
    if average_method == "min":
        return min(u, v)
    if average_method == "geometric":
        return math.sqrt(u * v)
    if average_method == "arithmetic":
        return (u + v) / 2
    return max(u, v)


def _clip_away_from_zero(value) -> float:
    """Push ``value`` at least machine epsilon away from zero, keeping its sign."""
    eps = float(np.finfo("float64").eps)
    return min(value, -eps) if value < 0 else max(value, eps)


def adjusted_mutual_info_score(
    labels_true, labels_pred, *, average_method="arithmetic"
) -> float:
    """
    Adjusted Mutual Information between two clusterings.

    Adjusted Mutual Information (AMI) is an adjustment of the Mutual
    Information (MI) score to account for chance. It accounts for the fact that
    the MI is generally higher for two clusterings with a larger number of
    clusters, regardless of whether there is actually more information shared.
    For two clusterings :math:`U` and :math:`V`, the AMI is given as::

        AMI(U, V) = [MI(U, V) - E(MI(U, V))] / [avg(H(U), H(V)) - E(MI(U, V))]

    This metric is independent of the absolute values of the labels: a
    permutation of the class or cluster label values won't change the score
    value in any way.

    This metric is furthermore symmetric: switching :math:`U` (``label_true``)
    with :math:`V` (``labels_pred``) will return the same score value. This can
    be useful to measure the agreement of two independent label assignment
    strategies on the same dataset when the real ground truth is not known.

    Parameters
    ----------
    labels_true : array-like (device or host) shape = (n_samples,)
        A clustering of the data (ints) into disjoint subsets.
        Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
        ndarray, cuda array interface compliant array like CuPy
    labels_pred : array-like (device or host) shape = (n_samples,)
        A clustering of the data (ints) into disjoint subsets.
        Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
        ndarray, cuda array interface compliant array like CuPy
    average_method : {'min', 'geometric', 'arithmetic', 'max'}, \
            default='arithmetic'
        How to compute the normalizer in the denominator.

    Returns
    -------
    float
        The AMI returns a value of 1 when the two partitions are identical
        (i.e. perfectly matched). Random partitions (independent labellings)
        have an expected AMI around 0 on average hence can be negative. The
        value is in adjusted nats (based on the natural logarithm). The AMI is
        undefined, and ``nan`` is returned, if ``average_method='min'`` and
        exactly one of the two partitions has every sample in a cluster of its
        own, unless the other one consists of a single cluster.

    See Also
    --------
    adjusted_rand_score : Adjusted Rand Index.
    mutual_info_score : Mutual Information (not adjusted for chance).

    Examples
    --------
    >>> import cupy as cp
    >>> from cuml.metrics.cluster import adjusted_mutual_info_score
    >>> labels_true = cp.asarray([0, 0, 1, 1])
    >>> labels_pred = cp.asarray([1, 1, 0, 0])
    >>> adjusted_mutual_info_score(labels_true, labels_pred)
    1.0
    >>> labels_pred = cp.asarray([0, 0, 1, 2])
    >>> score = adjusted_mutual_info_score(labels_true, labels_pred)
    >>> print(f"{score:.4f}")
    0.5714
    """
    if average_method not in _AVERAGE_METHODS:
        raise ValueError(
            f"'average_method' must be one of {_AVERAGE_METHODS}, got "
            f"{average_method!r}"
        )

    y_true, y_pred, n_samples, _, _ = prepare_cluster_metric_inputs(
        labels_true, labels_pred
    )
    classes, class_idx = cp.unique(y_true, return_inverse=True)
    clusters, cluster_idx = cp.unique(y_pred, return_inverse=True)
    n_classes, n_clusters = classes.shape[0], clusters.shape[0]

    # Special limit cases: no clustering since the data is not split.
    # It corresponds to both labellings having zero entropy.
    # This is a perfect match hence return 1.0.
    if n_classes == n_clusters == 1 or n_classes == n_clusters == 0:
        return 1.0
    # If there is only one class or one cluster return 0.0
    if n_classes == 1 or n_clusters == 1:
        return 0.0
    # If exactly one labelling has every sample in a cluster of its own, MI and
    # EMI both equal the entropy of the other labelling, hence AMI is 0.0. For
    # the "min" average, that entropy is also the normalizer and AMI is 0 / 0.
    # Return NaN then, unlike scikit-learn, where rounding decides the result.
    if (n_classes == n_samples) != (n_clusters == n_samples):
        return math.nan if average_method == "min" else 0.0

    # cupyx.scipy.sparse doesn't support integral dtypes
    contingency = cp_sp.coo_matrix(
        (
            cp.ones(n_samples, dtype=cp.float64),
            (class_idx, cluster_idx),
        ),
        shape=(n_classes, n_clusters),
    )
    contingency.sum_duplicates()

    # A single nonzero count per class and cluster is a perfect match, up to a
    # permutation of the labels. Return 1.0 exactly, avoiding rounding errors.
    if contingency.nnz == n_classes == n_clusters:
        return 1.0

    mi = mutual_info_score(y_true, y_pred)
    emi = expected_mutual_information(contingency, n_samples)
    h_true, h_pred = entropy(y_true), entropy(y_pred)
    normalizer = _generalized_average(h_true, h_pred, average_method)

    # Avoid 0.0 / 0.0 when the expectation equals the maximum. The normalizer
    # can also end up slightly below emi due to rounding, so preserve the sign.
    denominator = _clip_away_from_zero(normalizer - emi)
    numerator = _clip_away_from_zero(mi - emi)
    return float(numerator / denominator)
