#
# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np

from cuml.internals import get_handle
from cuml.internals.validation import check_array, check_consistent_length

from libc.stdint cimport uintptr_t
from pylibraft.common.handle cimport handle_t

from cuml.metrics.distance_type cimport DistanceType


cdef extern from "cuml/common/distance_type.hpp" namespace "ML::distance" nogil:
    ctypedef DistanceType euclidean "(ML::distance::DistanceType)5"

cdef extern from "cuml/metrics/metrics.hpp" namespace "ML::Metrics" nogil:

    cdef double trustworthiness_score[T, DistanceType](const handle_t& h,
                                                       T* X,
                                                       T* X_embedded,
                                                       int n, int m,
                                                       int d,
                                                       int n_neighbors,
                                                       int batchSize) \
        except +


def _get_array_ptr(obj):
    """
    Get ctype pointer of a numba style device array
    """
    return obj.device_ctypes_pointer.value


def trustworthiness(
    X,
    X_embedded,
    n_neighbors=5,
    metric='euclidean',
    batch_size=512,
) -> float:
    """
    Expresses to what extent the local structure is retained in embedding.
    The score is defined in the range [0, 1].

    Parameters
    ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        X_embedded : array-like (device or host) shape= (n_samples, n_features)
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        n_neighbors : int, optional (default=5)
            Number of neighbors considered

        metric : str in ['euclidean'] (default='euclidean')
            Metric used to compute the trustworthiness. For the moment only
            'euclidean' is supported.

        batch_size : int (default=512)
            The number of samples to use for each batch.

    Returns
    -------
        trustworthiness score : double
            Trustworthiness of the low-dimensional embedding
    """

    cdef uintptr_t d_X_ptr
    cdef uintptr_t d_X_embedded_ptr

    if metric != 'euclidean':
        raise ValueError(
            "Unsupported metric {!r}. Supported metrics are: "
            "'euclidean'.".format(metric)
        )

    X_m = check_array(
        X,
        order='C',
        dtype=np.float32,
        input_name='X',
    )
    cdef int n_samples = X_m.shape[0]
    cdef int n_features = X_m.shape[1]
    d_X_ptr = X_m.data.ptr

    X_m2 = check_array(
        X_embedded,
        order='C',
        dtype=np.float32,
        input_name='X_embedded',
    )
    check_consistent_length(X_m, X_m2)
    cdef int n_components = X_m2.shape[1]
    d_X_embedded_ptr = X_m2.data.ptr

    if n_neighbors < 1 or 2 * n_neighbors >= n_samples:
        raise ValueError(
            "n_neighbors ({}) must be >= 1 and < n_samples / 2; "
            "n_samples is {}.".format(n_neighbors, n_samples)
        )

    handle = get_handle()
    cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

    ret = trustworthiness_score[float, euclidean](handle_[0],
                                                  <float*>d_X_ptr,
                                                  <float*>d_X_embedded_ptr,
                                                  n_samples,
                                                  n_features,
                                                  n_components,
                                                  n_neighbors,
                                                  batch_size)
    handle.sync()

    return ret
