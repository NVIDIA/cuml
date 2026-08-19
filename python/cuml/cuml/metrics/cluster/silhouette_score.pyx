#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import numpy as np

from cuml.internals import get_handle
from cuml.internals.validation import check_array, check_consistent_length
from cuml.metrics.pairwise_distances import _determine_metric

from libc.stdint cimport uintptr_t
from pylibraft.common.handle cimport handle_t

from cuml.metrics.distance_type cimport DistanceType


cdef extern from "cuml/metrics/metrics.hpp" namespace "ML::Metrics::Batched" nogil:
    float silhouette_score(
        const handle_t &handle,
        float *y,
        int n_rows,
        int n_cols,
        int *labels,
        int n_labels,
        float *sil_scores,
        int chunk,
        DistanceType metric) except +

    double silhouette_score(
        const handle_t &handle,
        double *y,
        int n_rows,
        int n_cols,
        int *labels,
        int n_labels,
        double *sil_scores,
        int chunk,
        DistanceType metric) except +


def _silhouette_coeff(X, labels, metric='euclidean', sil_scores=None, chunksize=None):
    """Function wrapped by silhouette_score and silhouette_samples to compute
    silhouette coefficients.

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_features)
        The feature vectors for all samples.
    labels : array-like, shape = (n_samples,)
        The assigned cluster labels for each sample.
    metric : string
        A string representation of the distance metric to use for evaluating
        the silhouette score. Available options are "cityblock", "cosine",
        "euclidean", "l1", "l2", "manhattan", and "sqeuclidean".
    sil_scores : array_like, shape = (1, n_samples), dtype='float64'
        An optional array in which to store the silhouette score for each
        sample.
    chunksize : integer (default = None)
        An integer, 1 <= chunksize <= n_samples to tile the pairwise distance
        matrix computations, so as to reduce the quadratic memory usage of
        having the entire pairwise distance matrix in GPU memory.
        If None, chunksize will automatically be set to 40000, which through
        experiments has proved to be a safe number for the computation
        to run on a GPU with 16 GB VRAM.
    """
    handle = get_handle()
    cdef handle_t *handle_ = <handle_t*> <size_t> handle.getHandle()

    if chunksize is None:
        chunksize = 40000

    data = check_array(
        X,
        order='C',
        dtype=[np.float32, np.float64],
        input_name='X',
    )
    cdef int n_rows = data.shape[0]
    cdef int n_cols = data.shape[1]
    dtype = data.dtype

    labels = check_array(
        labels,
        ensure_2d=False,
        order='C',
        dtype=np.int32,
        input_name='labels',
    )
    if labels.ndim != 1:
        raise ValueError(
            f"labels must be a 1D array, got shape {labels.shape}"
        )
    check_consistent_length(data, labels)

    # Use cp.unique with return_inverse to get monotonic labels efficiently.
    unique_labels, inverse = cp.unique(labels, return_inverse=True)
    cdef int n_labels = unique_labels.shape[0]
    mono_labels = cp.ascontiguousarray(inverse, dtype=np.int32)

    cdef uintptr_t scores_ptr
    if sil_scores is None:
        scores_ptr = <uintptr_t> NULL
    else:
        sil_scores = check_array(
            sil_scores,
            ensure_2d=False,
            order='C',
            dtype=[dtype],
            input_name='sil_scores',
            ensure_all_finite=False,  # output buffer may be uninitialized
        )
        scores_ptr = sil_scores.data.ptr

    metric = _determine_metric(metric)

    if dtype == np.float32:
        return silhouette_score(handle_[0],
                                <float*> <uintptr_t> data.data.ptr,
                                n_rows,
                                n_cols,
                                <int*> <uintptr_t> mono_labels.data.ptr,
                                n_labels,
                                <float*> scores_ptr,
                                <int> chunksize,
                                <DistanceType> metric)
    elif dtype == np.float64:
        return silhouette_score(handle_[0],
                                <double*> <uintptr_t> data.data.ptr,
                                n_rows,
                                n_cols,
                                <int*> <uintptr_t> mono_labels.data.ptr,
                                n_labels,
                                <double*> scores_ptr,
                                <int> chunksize,
                                <DistanceType> metric)


def cython_silhouette_score(
    X,
    labels,
    metric='euclidean',
    chunksize=None,
):
    """Calculate the mean silhouette coefficient for the provided data.

    Given a set of cluster labels for every sample in the provided data,
    compute the mean intra-cluster distance (a) and the mean nearest-cluster
    distance (b) for each sample. The silhouette coefficient for a sample is
    then (b - a) / max(a, b).

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_features)
        The feature vectors for all samples.
    labels : array-like, shape = (n_samples,)
        The assigned cluster labels for each sample.
    metric : string
        A string representation of the distance metric to use for evaluating
        the silhouette score. Available options are "cityblock", "cosine",
        "euclidean", "l1", "l2", "manhattan", and "sqeuclidean".
    chunksize : integer (default = None)
        An integer, 1 <= chunksize <= n_samples to tile the pairwise distance
        matrix computations, so as to reduce the quadratic memory usage of
        having the entire pairwise distance matrix in GPU memory.
        If None, chunksize will automatically be set to 40000, which through
        experiments has proved to be a safe number for the computation
        to run on a GPU with 16 GB VRAM.
    """

    return _silhouette_coeff(
        X, labels, chunksize=chunksize, metric=metric,
    )


def cython_silhouette_samples(
    X,
    labels,
    metric='euclidean',
    chunksize=None,
):
    """Calculate the silhouette coefficient for each sample in the provided data.

    Given a set of cluster labels for every sample in the provided data,
    compute the mean intra-cluster distance (a) and the mean nearest-cluster
    distance (b) for each sample. The silhouette coefficient for a sample is
    then (b - a) / max(a, b).

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_features)
        The feature vectors for all samples.
    labels : array-like, shape = (n_samples,)
        The assigned cluster labels for each sample.
    metric : string
        A string representation of the distance metric to use for evaluating
        the silhouette score. Available options are "cityblock", "cosine",
        "euclidean", "l1", "l2", "manhattan", and "sqeuclidean".
    chunksize : integer (default = None)
        An integer, 1 <= chunksize <= n_samples to tile the pairwise distance
        matrix computations, so as to reduce the quadratic memory usage of
        having the entire pairwise distance matrix in GPU memory.
        If None, chunksize will automatically be set to 40000, which through
        experiments has proved to be a safe number for the computation
        to run on a GPU with 16 GB VRAM.
    """

    sil_scores = cp.empty((X.shape[0],), dtype=X.dtype)

    _silhouette_coeff(
        X, labels, chunksize=chunksize, metric=metric, sil_scores=sil_scores,
    )

    return sil_scores
