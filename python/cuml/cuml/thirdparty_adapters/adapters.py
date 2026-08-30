#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import numpy as np


def _is_na_sentinel(value):
    """Return whether a scalar missing-value sentinel is NA-like."""
    if isinstance(value, str):
        return value == "NaN"
    return pd.isna(value)


def _get_mask(X, value_to_mask):
    """Compute the boolean mask X == missing_values."""
    if isinstance(value_to_mask, str) and value_to_mask == "NaN":
        if isinstance(X, cp.ndarray):
            return cp.isnan(X)
        return pd.isna(X)
    # NaN-like sentinels (np.nan, None, pd.NA, pd.NaT, ...) require an
    # NA-aware comparison: a plain `==` against e.g. pd.NA propagates
    # instead of returning a boolean mask, and `cp.isnan` doesn't accept
    # non-numeric scalars like pd.NA in the first place.
    if _is_na_sentinel(value_to_mask):
        if isinstance(X, cp.ndarray):
            return cp.isnan(X)
        # Host (e.g. object dtype) arrays can't use isnan - fall back to an
        # NA-aware elementwise check that also recognizes None/pd.NA.
        return pd.isna(X)
    else:
        return X == value_to_mask


def _masked_column_median(arr, masked_value):
    """Compute the median of each column in the 2D array arr, ignoring any
    instances of masked_value"""
    mask = _get_mask(arr, masked_value)
    if arr.size == 0:
        return cp.full(arr.shape[1], cp.nan)
    if not _is_na_sentinel(masked_value):
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
    if not _is_na_sentinel(masked_value):
        mean -= count_missing_values * masked_value
    mean /= n_elems
    return mean


def _masked_column_mode(arr, masked_value):
    """Determine the most frequently appearing element in each column in the 2D
    array arr, ignoring any instances of masked_value"""
    mask = _get_mask(arr, masked_value)
    n_features = arr.shape[1]
    most_frequent = np.empty(n_features, dtype=arr.dtype)
    for i in range(n_features):
        feature_mask_idxs = cp.where(~mask[:, i])[0]
        values, counts = cp.unique(
            arr[feature_mask_idxs, i], return_counts=True
        )
        count_max = counts.max()
        if count_max > 0:
            value = values[counts == count_max].min()
        else:
            value = cp.nan
        most_frequent[i] = value
    return cp.array(most_frequent)
