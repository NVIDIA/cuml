#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cudf
import cupy as cp
import numpy as np

from cuml.internals.validation import (
    check_array,
    check_consistent_length,
    check_sample_weight,
)


def _input_to_cupy_or_cudf_series(x):
    """Coerce the input to a 1D cupy array or cudf Series.

    For classification problems we need to support the full range
    of supported input dtypes. cupy cannot support string labels,
    and cudf cannot support float16. To handle this, we prefer cudf
    if the input is cudf, otherwise try to coerce to cupy, falling
    back to cudf if the dtype isn't supported.
    """
    if isinstance(x, cudf.Series):
        # Drop the index so comparisons don't try to align on index
        out = x.reset_index(drop=True)
    else:
        try:
            out = check_array(
                x,
                ensure_2d=False,
                ensure_all_finite=False,
                mem_type="device",
                order=None,
            )
        except (ValueError, TypeError):
            # Unsupported dtype (e.g. strings), use cudf instead
            # Drop the index so comparisons don't try to align on index
            out = cudf.Series(x, nan_as_null=False, copy=False).reset_index(
                drop=True
            )
        else:
            if out.ndim > 1:
                if out.shape[1] > 1:
                    raise ValueError(
                        f"Expected 1 column but got {out.shape[1]} columns."
                    )
                out = out.squeeze()  # ensure 1D

    return out


def accuracy_score(y_true, y_pred, *, sample_weight=None, normalize=True):
    """
    Accuracy classification score.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    sample_weight : array-like of shape (n_samples,)
        Sample weights.
    normalize : bool
        If ``False``, return the number of correctly classified samples.
        Otherwise, return the fraction of correctly classified samples.

    Returns
    -------
    score : float
        The fraction of correctly classified samples, or the number of correctly
        classified samples if ``normalize == False``.
    """

    y_true = _input_to_cupy_or_cudf_series(y_true)
    y_pred = _input_to_cupy_or_cudf_series(y_pred)

    check_consistent_length(y_true, y_pred)

    # Categorical dtypes in cudf currently don't coerce nicely on equality,
    # we need to manually cast to cudf.Series and align dtypes.
    # This whole code block can be removed once
    # https://github.com/NVIDIA/cudf/issues/18196 is resolved.
    if y_true.dtype == "category":
        if y_pred.dtype != y_true.dtype:
            y_pred = cudf.Series(y_pred, copy=False, nan_as_null=False).astype(
                y_true.dtype
            )
    elif y_pred.dtype == "category":
        y_true = cudf.Series(y_true, copy=False, nan_as_null=False).astype(
            y_pred.dtype
        )

    if (
        sample_weight := check_sample_weight(sample_weight, dtype=np.float64)
    ) is not None:
        check_consistent_length(y_true, sample_weight)

    correct = y_true == y_pred

    if normalize:
        return float(cp.average(correct, weights=sample_weight))
    elif sample_weight is not None:
        return float(cp.dot(correct, sample_weight))
    else:
        return float(cp.count_nonzero(correct))


def log_loss(
    y_true, y_pred, eps=1e-15, normalize=True, sample_weight=None
) -> float:
    """Log loss, aka logistic loss or cross-entropy loss.
    This is the loss function used in (multinomial) logistic regression
    and extensions of it such as neural networks, defined as the negative
    log-likelihood of a logistic model that returns ``y_pred`` probabilities
    for its training data ``y_true``.
    The log loss is only defined for two or more labels.

    Parameters
    ----------
    y_true : array-like, shape = (n_samples,)
    y_pred : array-like of float,
        shape = (n_samples, n_classes) or (n_samples,)
    eps : float (default=1e-15)
        Log loss is undefined for p=0 or p=1, so probabilities are
        clipped to max(eps, min(1 - eps, p)).
    normalize : bool, optional (default=True)
        If true, return the mean loss per sample.
        Otherwise, return the sum of the per-sample losses.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    loss : float

    Examples
    --------
    .. code-block:: python

        >>> from cuml.metrics import log_loss
        >>> import cupy as cp
        >>> log_loss(cp.array([1, 0, 0, 1]),
        ...          cp.array([[.1, .9], [.9, .1], [.8, .2], [.35, .65]]))
        0.21616...

    References
    ----------
    C.M. Bishop (2006). Pattern Recognition and Machine Learning. Springer,
    p. 209.

    Notes
    -----
    The logarithm used is the natural logarithm (base-e).

    """
    y_true = check_array(
        y_true,
        ensure_2d=False,
        dtype=(np.int32, np.int64, np.float32, np.float64),
        ensure_non_negative=True,
        input_name="y_true",
    )

    if y_true.dtype.kind == "f" and np.any(y_true != y_true.astype(int)):
        raise ValueError("'y_true' can only have integer values")

    y_pred = check_array(
        y_pred,
        ensure_2d=False,
        dtype=(np.float32, np.float64),
        input_name="y_pred",
    )

    check_consistent_length(y_true, y_pred)

    if (
        sample_weight := check_sample_weight(sample_weight, dtype=np.float64)
    ) is not None:
        check_consistent_length(y_true, sample_weight)

    y_true_max = y_true.max()
    if (y_pred.ndim == 1 and y_true_max > 1) or (
        y_pred.ndim > 1 and y_pred.shape[1] <= y_true_max
    ):
        raise ValueError(
            "The shape of y_pred doesn't match the number of classes"
        )

    y_true = y_true.astype("int32")
    y_pred = cp.clip(y_pred, eps, 1 - eps)
    if y_pred.ndim == 1:
        y_pred = cp.expand_dims(y_pred, axis=1)
    if y_pred.shape[1] == 1:
        y_pred = cp.hstack([1 - y_pred, y_pred])

    y_pred /= cp.sum(y_pred, axis=1, keepdims=True)
    loss = -cp.log(y_pred)[cp.arange(y_pred.shape[0]), y_true]
    return _weighted_sum(loss, sample_weight, normalize).item()


def _weighted_sum(sample_score, sample_weight, normalize):
    if normalize:
        return cp.average(sample_score, weights=sample_weight)
    elif sample_weight is not None:
        return cp.dot(sample_score, sample_weight)
    else:
        return sample_score.sum()
