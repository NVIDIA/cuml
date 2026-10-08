#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import math
from math import log

import cupy as cp
import numpy as np

from cuml.metrics.cluster.utils import prepare_cluster_metric_inputs


def _generalized_average(U, V, average_method):
    if average_method == "min":
        return min(U, V)
    elif average_method == "geometric":
        return np.sqrt(U * V)
    elif average_method == "arithmetic":
        return np.mean([U, V])
    elif average_method == "max":
        return max(U, V)
    else:
        raise ValueError(
            "'average_method' must be 'min', 'geometric', 'arithmetic', or 'max'"
        )


def _entropy_labels(labels):
    counts = np.unique(labels, return_counts=True)[1].astype(np.float64)
    if counts.size == 0:
        return 1.0
    pi_sum = counts.sum()
    if counts.size == 1:
        return 0.0
    return float(-np.sum((counts / pi_sum) * (np.log(counts) - log(pi_sum))))


def _mutual_info_from_contingency(contingency):
    contingency_sum = contingency.sum()
    if contingency_sum == 0:
        return 0.0
    pi = contingency.sum(axis=1)
    pj = contingency.sum(axis=0)
    if pi.size == 1 or pj.size == 1:
        return 0.0
    nzx, nzy = np.nonzero(contingency)
    nz_val = contingency[nzx, nzy].astype(np.float64)
    log_contingency_nm = np.log(nz_val)
    contingency_nm = nz_val / contingency_sum
    outer = pi.take(nzx).astype(np.float64) * pj.take(nzy).astype(np.float64)
    log_outer = -np.log(outer) + log(pi.sum()) + log(pj.sum())
    mi = contingency_nm * (log_contingency_nm - log(contingency_sum)) + contingency_nm * log_outer
    mi = np.where(np.abs(mi) < np.finfo(mi.dtype).eps, 0.0, mi)
    return float(np.clip(mi.sum(), 0.0, None))


def _lchoose(n, k):
    if k < 0 or k > n:
        return float("-inf")
    return (
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    )


def _expected_mutual_information(contingency, n_samples):
    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)
    emi = 0.0
    n = float(n_samples)
    for i in range(contingency.shape[0]):
        a_i = row_sums[i]
        if a_i == 0:
            continue
        for j in range(contingency.shape[1]):
            b_j = col_sums[j]
            if b_j == 0:
                continue
            lo = max(1, a_i + b_j - n_samples)
            hi = min(a_i, b_j)
            for k in range(lo, hi + 1):
                log_p = (
                    _lchoose(a_i, k)
                    + _lchoose(n_samples - a_i, b_j - k)
                    - _lchoose(n_samples, b_j)
                )
                mi_cell = (k / n) * log((k * n) / (a_i * b_j))
                emi += math.exp(log_p) * mi_cell
    return emi


def _ami_impl(a, b, average_method, n_rows, upper_class_range) -> float:
    classes = np.unique(a)
    clusters = np.unique(b)

    # Special limit cases: no clustering since the data is not split.
    # It corresponds to both labellings having zero entropy.
    # This is a perfect match hence return 1.0.
    if (classes.shape[0] == clusters.shape[0] == 1
            or classes.shape[0] == clusters.shape[0] == 0):
        return 1.0
    # if there is only one class or one cluster return 0.0.
    elif classes.shape[0] == 1 or clusters.shape[0] == 1:
        return 0.0

    k = upper_class_range + 1
    flat = a.astype(np.int64) * k + b.astype(np.int64)
    contingency = np.bincount(flat, minlength=k * k).reshape(k, k)

    mi = _mutual_info_from_contingency(contingency)
    emi = _expected_mutual_information(contingency, n_rows)
    h_true = _entropy_labels(a)
    h_pred = _entropy_labels(b)
    normalizer = _generalized_average(h_true, h_pred, average_method)
    denominator = normalizer - emi
    if denominator < 0:
        denominator = min(denominator, -np.finfo("float64").eps)
    else:
        denominator = max(denominator, np.finfo("float64").eps)
    numerator = mi - emi
    if numerator < 0:
        numerator = min(numerator, -np.finfo("float64").eps)
    else:
        numerator = max(numerator, np.finfo("float64").eps)
    return float(numerator / denominator)


def cython_adjusted_mutual_info_score(labels_true, labels_pred, *, average_method="arithmetic") -> float:
    """Adjusted Mutual Information between two clusterings.

    Adjusted Mutual Information (AMI) is an adjustment of the Mutual
    Information (MI) score to account for chance. For two clusterings
    :math:`U` and :math:`V`, the AMI is given as::

        AMI(U, V) = [MI(U, V) - E(MI(U, V))] / [avg(H(U), H(V)) - E(MI(U, V))]

    Parameters
    ----------
    labels_true : array-like (device or host) shape = (n_samples,)
        Ground truth class labels.
    labels_pred : array-like (device or host) shape = (n_samples,)
        Predicted cluster labels.
    average_method : {'min', 'geometric', 'arithmetic', 'max'}, default='arithmetic'
        How to compute the normalizer in the denominator.

    Returns
    -------
    ami : float
       The Adjusted Mutual Information, with 1.0 for a perfect match and
       a value close to 0.0 for random labelings.
    """
    (y_true, y_pred, n_rows,
     lower_class_range, upper_class_range) = prepare_cluster_metric_inputs(
        labels_true,
        labels_pred
    )

    a = cp.asnumpy(y_true)
    b = cp.asnumpy(y_pred)
    return _ami_impl(a, b, average_method, n_rows, upper_class_range)
