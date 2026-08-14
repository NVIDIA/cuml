#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import cupyx
import numpy as np

from cuml.internals.validation import (
    check_array,
    check_consistent_length,
    check_sample_weight,
)


def confusion_matrix(
    y_true,
    y_pred,
    labels=None,
    sample_weight=None,
    normalize=None,
) -> cp.ndarray:
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Parameters
    ----------
    y_true : array-like (device or host) shape = (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like (device or host) shape = (n_samples,)
        Estimated target values.
    labels : array-like (device or host) shape = (n_classes,), optional
        List of labels to index the matrix. This may be used to reorder or
        select a subset of labels. If None is given, those that appear at least
        once in y_true or y_pred are used in sorted order.
    sample_weight : array-like (device or host) shape = (n_samples,), optional
        Sample weights.
    normalize : string in ['true', 'pred', 'all'] or None (default=None)
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, confusion matrix will not be
        normalized.

    Returns
    -------
    C : cupy.ndarray of shape (n_classes, n_classes)
        Confusion matrix on device.
    """
    y_true = check_array(
        y_true,
        ensure_2d=False,
        dtype=("int32", "int64"),
        input_name="y_true",
    )
    y_pred = check_array(
        y_pred,
        ensure_2d=False,
        dtype=("int32", "int64"),
        input_name="y_pred",
    )
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(
            f"y_true and y_pred must be 1D arrays, got shapes "
            f"{y_true.shape} and {y_pred.shape}"
        )
    sample_weight = check_sample_weight(
        sample_weight,
        dtype=("float32", "float64", "int32", "int64"),
    )
    check_consistent_length(y_true, y_pred, sample_weight)

    if labels is not None:
        labels = check_array(
            labels,
            ensure_2d=False,
            dtype=("int32", "int64"),
            input_name="labels",
            ensure_min_samples=0,
        )
        if labels.ndim != 1:
            raise ValueError(
                f"labels must be a 1D array, got shape {labels.shape}"
            )
        if len(labels) == 0:
            raise ValueError("'labels' should contain at least one label")
    else:
        labels = cp.unique(
            cp.concatenate([cp.unique(y_true), cp.unique(y_pred)])
        )

    if sample_weight is None:
        sample_weight = cp.ones(y_true.shape[0], dtype=y_true.dtype)

    # Sort provided labels for binary search, but keep track of
    # original indices to maintain provided order.
    sort_indices = cp.argsort(labels)
    sorted_labels = labels[sort_indices]

    valid = cp.in1d(y_true, sorted_labels) & cp.in1d(y_pred, sorted_labels)
    if not valid.all():
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        sample_weight = sample_weight[valid]

    # Map each label to its index in sorted labels using binary search
    true_indices = cp.searchsorted(sorted_labels, y_true)
    pred_indices = cp.searchsorted(sorted_labels, y_pred)

    # Map valid labels to their indices in original labels
    y_true = sort_indices[true_indices]
    y_pred = sort_indices[pred_indices]

    cm = cupyx.scipy.sparse.coo_matrix(
        (sample_weight.astype("float64", copy=False), (y_true, y_pred)),
        shape=(labels.shape[0], labels.shape[0]),
    ).toarray()

    # Choose the accumulator dtype to always have high precision
    if sample_weight.dtype.kind in {"i", "u", "b"}:
        cm = cm.astype(np.int64)

    with np.errstate(all="ignore"):
        if normalize == "true":
            cm = cp.divide(cm, cm.sum(axis=1, keepdims=True))
        elif normalize == "pred":
            cm = cp.divide(cm, cm.sum(axis=0, keepdims=True))
        elif normalize == "all":
            cm = cp.divide(cm, cm.sum())
        elif normalize is not None:
            raise ValueError(
                "normalize must be one of "
                f"{{'true', 'pred', 'all', None}}, got {normalize}."
            )
        cm = cp.nan_to_num(cm)

    return cm
