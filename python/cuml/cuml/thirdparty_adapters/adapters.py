#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import numpy as np
import pandas as pd


def _is_nan(value):
    """Return whether a scalar is a floating-point NaN."""
    return isinstance(value, (float, np.floating)) and np.isnan(value)


def _is_nan_sentinel(value):
    """Return whether a scalar selects floating-point NaN values."""
    return (isinstance(value, str) and value == "NaN") or _is_nan(value)


def _get_host_mask(X, predicate):
    return np.fromiter(
        (predicate(value) for value in X.flat), dtype=bool, count=X.size
    ).reshape(X.shape)


def _get_mask(X, value_to_mask):
    """Compute the boolean mask X == missing_values."""
    if value_to_mask is pd.NA:
        if isinstance(X, cp.ndarray):
            return cp.zeros(X.shape, dtype=bool)
        return _get_host_mask(X, lambda value: value is pd.NA)
    if value_to_mask is None:
        if isinstance(X, cp.ndarray):
            return cp.zeros(X.shape, dtype=bool)
        return _get_host_mask(X, lambda value: value is None)
    if _is_nan_sentinel(value_to_mask):
        if isinstance(X, cp.ndarray):
            return cp.isnan(X)
        return _get_host_mask(X, _is_nan)
    return X == value_to_mask


def _masked_column_median(arr, masked_value):
    """Compute the median of each column in the 2D array arr, ignoring any
    instances of masked_value"""
    mask = _get_mask(arr, masked_value)
    if arr.size == 0:
        return cp.full(arr.shape[1], cp.nan)
    if not _is_nan_sentinel(masked_value):
        arr_sorted = arr.copy()
        # If nan is not the missing value, any column with nans should
        # have a median of nan
        nan_cols = cp.any(cp.isnan(arr), axis=0)
        arr_sorted[mask] = cp.nan
        arr_sorted.sort(axis=0)
    else:
        nan_cols = cp.full(arr.shape[1], False)
        # nans are always sorted to end of array and the sort call
        # copies the data
        arr_sorted = cp.sort(arr, axis=0)

    count_missing_values = mask.sum(axis=0)
    # Ignore missing values in determining "halfway" index of sorted
    # array
    n_elems = arr.shape[0] - count_missing_values

    # If no elements remain after removing missing value, median for
    # that column is nan
    nan_cols = cp.logical_or(nan_cols, n_elems <= 0)

    col_index = cp.arange(arr_sorted.shape[1])
    median = (
        arr_sorted[cp.floor_divide(n_elems - 1, 2), col_index]
        + arr_sorted[cp.floor_divide(n_elems, 2), col_index]
    ) / 2

    median[nan_cols] = cp.nan
    return median


def _masked_column_mean(arr, masked_value):
    """Compute the mean of each column in the 2D array arr, ignoring any
    instances of masked_value"""
    mask = _get_mask(arr, masked_value)
    count_missing_values = mask.sum(axis=0)
    n_elems = arr.shape[0] - count_missing_values
    mean = cp.nansum(arr, axis=0)
    if not _is_nan_sentinel(masked_value):
        mean -= count_missing_values * masked_value
    mean /= n_elems
    return mean


def _masked_column_mode(arr, masked_value):
    """Determine the most frequently appearing element in each column in the 2D
    array arr, ignoring any instances of masked_value"""
    xp = cp.get_array_module(arr)
    mask = _get_mask(arr, masked_value)
    if arr.dtype.kind == "O":
        na_mask = pd.isna(arr)
        # Never pass NA-like values into object sorting. The strict sentinel
        # mask is still used later to select values for imputation.
        count_mask = xp.logical_or(mask, na_mask)
    else:
        count_mask = mask
    n_features = arr.shape[1]
    result_dtype = object if arr.dtype.kind == "O" else arr.dtype
    most_frequent = np.empty(n_features, dtype=result_dtype)
    for i in range(n_features):
        feature_mask_idxs = xp.where(~count_mask[:, i])[0]
        values, counts = xp.unique(
            arr[feature_mask_idxs, i], return_counts=True
        )
        if counts.size == 0:
            value = xp.nan
        else:
            count_max = counts.max()
            value = values[counts == count_max].min()
        most_frequent[i] = value
    return xp.array(most_frequent)
