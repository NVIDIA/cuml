#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np

from cuml.internals import get_handle
from cuml.internals.validation import check_array, check_consistent_length

from libc.stdint cimport uintptr_t
from pylibraft.common.handle cimport handle_t


cdef extern from "cuml/metrics/metrics.hpp" namespace "ML::Metrics" nogil:

    double adjusted_rand_index(handle_t &handle,
                               int *y,
                               int *y_hat,
                               int n) except +


def adjusted_rand_score(labels_true, labels_pred) -> float:
    """
    Adjusted_rand_score is a clustering similarity metric based on the Rand
    index and is corrected for chance.

    Parameters
    ----------
    labels_true : Ground truth labels to be used as a reference

    labels_pred : Array of predicted labels used to evaluate the model

    Returns
    -------
        float
            The adjusted rand index value between -1.0 and 1.0
    """
    handle = get_handle()
    cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

    labels_true = check_array(
        labels_true,
        ensure_2d=False,
        ensure_min_samples=0,
        order='C',
        dtype=np.int32,
        input_name='labels_true',
    )
    labels_pred = check_array(
        labels_pred,
        ensure_2d=False,
        ensure_min_samples=0,
        order='C',
        dtype=np.int32,
        input_name='labels_pred',
    )
    if labels_true.ndim != 1 or labels_pred.ndim != 1:
        raise ValueError(
            "labels_true and labels_pred must be 1D arrays, got shapes "
            f"{labels_true.shape} and {labels_pred.shape}"
        )
    check_consistent_length(labels_true, labels_pred)
    cdef int n_rows = labels_true.shape[0]

    rand_score = adjusted_rand_index(handle_[0],
                                     <int*><uintptr_t> labels_true.data.ptr,
                                     <int*><uintptr_t> labels_pred.data.ptr,
                                     n_rows)

    return rand_score
