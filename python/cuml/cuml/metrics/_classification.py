#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import warnings

import cudf
import cupy as cp
import numpy as np
from sklearn.exceptions import UndefinedMetricWarning

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
    # https://github.com/rapidsai/cudf/issues/18196 is resolved.
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


def precision_score(
    y_true,
    y_pred,
    *,
    labels=None,
    pos_label=1,
    average="binary",
    sample_weight=None,
    zero_division="warn",
):
    """
    Compute the precision.

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number
    of true positives and ``fp`` the number of false positives. The precision
    is intuitively the ability of the classifier not to label as positive a
    sample that is negative.

    The best value is 1 and the worst value is 0.

    Parameters
    ----------
    y_true : array-like (device or host) of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like (device or host) of shape (n_samples,)
        Estimated target values as returned by a classifier.
    labels : array-like (device or host), default=None
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, and labels not present in the data will receive the score
        given by ``zero_division``. Ignored when ``average == 'binary'``.
    pos_label : int, float, bool or str, default=1
        The class to report if ``average='binary'`` and the data is binary,
        otherwise this parameter is ignored.
    average : {'micro', 'macro', 'weighted', 'binary'} or None, \
            default='binary'
        This parameter is required for multiclass targets.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives
            and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted mean.
        ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label).
        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            Only applicable to binary targets.
        If ``None``, the scores for each label are returned individually.
    sample_weight : array-like (device or host) of shape (n_samples,), \
            default=None
        Sample weights.
    zero_division : {"warn", 0.0, 1.0}, default="warn"
        Sets the value to return when there is a zero division. If set to
        ``"warn"``, this acts like 0, but a warning is also raised.

    Returns
    -------
    score : float or numpy.ndarray of float
        Precision of the positive class in binary classification or the
        averaged precision of each class for the multiclass task. A NumPy
        array with one score per label, ordered following ``labels`` (or the
        sorted union of the observed labels when ``labels is None``), is
        returned when ``average is None``.

    See Also
    --------
    accuracy_score : Accuracy classification score.
    confusion_matrix : Compute confusion matrix to evaluate the accuracy of a
        classification.

    Notes
    -----
    Numeric labels (integer, float and bool dtypes) are counted on the GPU.
    String, object and categorical labels are supported through a device-side
    encoding against the sorted union of the observed labels, which matches
    scikit-learn's label ordering. Null values are not supported. The
    ``'samples'`` averaging strategy, multilabel indicator input and
    ``zero_division=np.nan`` accepted by scikit-learn are not supported.

    Examples
    --------
    .. code-block:: python

        >>> import cupy as cp
        >>> from cuml.metrics import precision_score
        >>> y_true = cp.array([0, 1, 2, 0, 1, 2])
        >>> y_pred = cp.array([0, 2, 1, 0, 0, 1])
        >>> precision_score(y_true, y_pred, average='macro')
        0.2222222222222222
        >>> precision_score(y_true, y_pred, average='micro')
        0.3333333333333333
        >>> precision_score(y_true, y_pred, average=None)
        array([0.66666667, 0.        , 0.        ])
    """

    average_options = (None, "micro", "macro", "weighted", "binary")
    if average not in average_options:
        raise ValueError(f"average has to be one of {average_options}")

    if isinstance(zero_division, str) and zero_division == "warn":
        zero_division_value = 0.0
    elif isinstance(zero_division, (int, float)) and zero_division in (0, 1):
        zero_division_value = float(zero_division)
    else:
        raise ValueError(
            'zero_division must be one of {"warn", 0, 1}, got '
            f"{zero_division!r}"
        )

    y_true = _input_to_cupy_or_cudf_series(y_true)
    y_pred = _input_to_cupy_or_cudf_series(y_pred)

    check_consistent_length(y_true, y_pred)

    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError(
            "Found empty input array (e.g., `y_true` or `y_pred`) while a "
            "minimum of 1 sample is required."
        )

    for name, y in (("y_true", y_true), ("y_pred", y_pred)):
        if isinstance(y, cudf.Series) and y.isna().any():
            raise ValueError(
                f"precision_score does not support null values in {name}"
            )

    if (
        sample_weight := check_sample_weight(sample_weight, dtype=np.float64)
    ) is not None:
        check_consistent_length(y_true, sample_weight)

    numeric = all(
        not isinstance(y, cudf.Series) or y.dtype.kind in "iufb"
        for y in (y_true, y_pred)
    )

    if numeric:
        y_true_t = (
            y_true.to_cupy() if isinstance(y_true, cudf.Series) else y_true
        )
        y_pred_t = (
            y_pred.to_cupy() if isinstance(y_pred, cudf.Series) else y_pred
        )
        present = cp.unique(
            cp.concatenate([cp.unique(y_true_t), cp.unique(y_pred_t)])
        )
        present_labels = cp.asnumpy(present).tolist()
        for name, arr in (("y_true", y_true_t), ("y_pred", y_pred_t)):
            if arr.dtype.kind == "f":
                if bool(cp.isnan(arr).any()):
                    raise ValueError(f"Input {name} contains NaN.")
                if bool(cp.isinf(arr).any()):
                    raise ValueError(
                        f"Input {name} contains infinity or a value too "
                        "large for dtype('float64')."
                    )
                if bool((arr != cp.floor(arr)).any()):
                    raise ValueError(f"'{name}' can only have integer values")
    else:
        y_true = (
            y_true if isinstance(y_true, cudf.Series) else cudf.Series(y_true)
        )
        y_pred = (
            y_pred if isinstance(y_pred, cudf.Series) else cudf.Series(y_pred)
        )
        try:
            present_labels = sorted(
                set().union(
                    *[
                        set(y.unique().dropna().to_pandas().tolist())
                        for y in (y_true, y_pred)
                    ]
                )
            )
        except TypeError:
            raise ValueError(
                "Mix of label input types (string and number)"
            ) from None

    if average == "binary":
        if len(present_labels) > 2:
            raise ValueError(
                "Target is multiclass but average='binary'. Please choose "
                "another average setting, one of [None, 'micro', 'macro', "
                "'weighted']."
            )
        if len(present_labels) >= 2 and pos_label not in present_labels:
            raise ValueError(
                f"pos_label={pos_label} is not a valid label. It should be "
                f"one of {present_labels}"
            )
        out_labels = [pos_label] if not numeric else cp.array([pos_label])
    else:
        if pos_label not in (None, 1):
            warnings.warn(
                "Note that pos_label (set to "
                f"{pos_label!r}) is ignored when average != 'binary' "
                f"(got {average!r}). You may use labels=[pos_label] to "
                "specify a single positive class.",
                UserWarning,
                stacklevel=2,
            )
        if labels is not None:
            out_labels = _labels_as_device_or_host(labels, numeric)
        else:
            out_labels = present if numeric else present_labels

    if numeric:
        table = cp.unique(cp.concatenate([present, out_labels]))
        pos = cp.searchsorted(table, out_labels)
        true_idx = cp.searchsorted(table, y_true_t)
        pred_idx = cp.searchsorted(table, y_pred_t)
        n_labels_total = table.shape[0]
    else:
        table = sorted(set(present_labels) | set(out_labels))
        cat_dtype = cudf.CategoricalDtype(categories=table)
        true_idx = (
            y_true.astype(cat_dtype).cat.codes.to_cupy().astype(np.int64)
        )
        pred_idx = (
            y_pred.astype(cat_dtype).cat.codes.to_cupy().astype(np.int64)
        )
        pos = cp.array(
            [table.index(label) for label in out_labels], dtype=np.int64
        )
        n_labels_total = len(table)

    weights = (
        cp.ones(y_true.shape[0], dtype=cp.float64)
        if sample_weight is None
        else sample_weight.astype(cp.float64, copy=False)
    )

    diag_mask = true_idx == pred_idx
    tp_sum = cp.bincount(
        true_idx[diag_mask],
        weights=weights[diag_mask],
        minlength=n_labels_total,
    )[pos]
    pred_sum = cp.bincount(
        pred_idx, weights=weights, minlength=n_labels_total
    )[pos]
    true_sum = cp.bincount(
        true_idx, weights=weights, minlength=n_labels_total
    )[pos]

    empty = pred_sum == 0
    per_class = cp.where(
        empty, zero_division_value, tp_sum / cp.where(empty, 1.0, pred_sum)
    )
    n_out = per_class.shape[0]

    if average is None:
        if zero_division == "warn" and empty.any():
            _warn_precision_undefined(n_out)
        return cp.asnumpy(per_class)

    if average == "binary":
        if zero_division == "warn" and pred_sum[0] == 0:
            _warn_precision_undefined(1)
        return float(per_class[0])

    if average == "micro":
        pred_total = pred_sum.sum()
        if pred_total == 0:
            if zero_division == "warn":
                _warn_precision_undefined(1)
            return zero_division_value
        return float(tp_sum.sum() / pred_total)

    if zero_division == "warn" and empty.any():
        _warn_precision_undefined(n_out)

    if average == "macro":
        return float(per_class.mean())

    per_class = cp.asnumpy(per_class)
    try:
        return float(np.average(per_class, weights=cp.asnumpy(true_sum)))
    except ZeroDivisionError:
        # all-zero support: scikit-learn ignores the weights entirely
        return float(np.average(per_class))


def _labels_as_device_or_host(labels, numeric):
    if numeric:
        labels = _input_to_cupy_or_cudf_series(labels)
        if isinstance(labels, cudf.Series):
            labels = labels.to_cupy()
        return cp.reshape(labels, (-1,))
    if isinstance(labels, cudf.Series):
        return labels.to_pandas().tolist()
    if hasattr(labels, "tolist"):
        return labels.tolist()
    return list(labels)


def _warn_precision_undefined(n_labels):
    due_to = "due to" if n_labels == 1 else "in labels with"
    warnings.warn(
        "Precision is ill-defined and being set to 0.0 "
        f"{due_to} no predicted samples. Use `zero_division` parameter "
        "to control this behavior.",
        UndefinedMetricWarning,
        stacklevel=3,
    )


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
