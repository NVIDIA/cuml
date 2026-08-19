#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import warnings

import cupy as cp
import cupyx.scipy.sparse as cp_sp
from sklearn.exceptions import DataConversionWarning

from cuml.internals import get_handle, mlfunc
from cuml.internals.outputs import using_output_type
from cuml.internals.validation import check_array

from libc.stdint cimport uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t

from cuml.metrics.distance_type cimport DistanceType


cdef extern from "cuml/metrics/metrics.hpp" namespace "ML::Metrics" nogil:
    void pairwise_distance(const handle_t &handle, const double *x,
                           const double *y, double *dist, int m, int n, int k,
                           DistanceType metric, bool isRowMajor,
                           double metric_arg) except +
    void pairwise_distance(const handle_t &handle, const float *x,
                           const float *y, float *dist, int m, int n, int k,
                           DistanceType metric, bool isRowMajor,
                           float metric_arg) except +
    void pairwiseDistance_sparse(const handle_t &handle, float *x, float *y,
                                 float *dist, int x_nrows, int y_nrows,
                                 int n_cols, int x_nnz, int y_nnz,
                                 int* x_indptr, int* y_indptr,
                                 int* x_indices, int* y_indices,
                                 DistanceType metric,
                                 float metric_arg) except +
    void pairwiseDistance_sparse(const handle_t &handle, double *x, double *y,
                                 double *dist, int x_nrows, int y_nrows,
                                 int n_cols, int x_nnz, int y_nnz,
                                 int* x_indptr, int* y_indptr,
                                 int* x_indices, int* y_indices,
                                 DistanceType metric,
                                 float metric_arg) except +

# List of available distance metrics in `pairwise_distances`
PAIRWISE_DISTANCE_METRICS = {
    "cityblock": DistanceType.L1,
    "cosine": DistanceType.CosineExpanded,
    "euclidean": DistanceType.L2SqrtUnexpanded,
    "l1": DistanceType.L1,
    "l2": DistanceType.L2SqrtUnexpanded,
    "manhattan": DistanceType.L1,
    "sqeuclidean": DistanceType.L2Expanded,
    "canberra": DistanceType.Canberra,
    "chebyshev": DistanceType.Linf,
    "minkowski": DistanceType.LpUnexpanded,
    "hellinger": DistanceType.HellingerExpanded,
    "correlation": DistanceType.CorrelationExpanded,
    "jensenshannon": DistanceType.JensenShannon,
    "hamming": DistanceType.HammingUnexpanded,
    "kldivergence": DistanceType.KLDivergence,
    "russellrao": DistanceType.RusselRaoExpanded,
    "nan_euclidean": DistanceType.L2Expanded
}

PAIRWISE_DISTANCE_SPARSE_METRICS = {
    "cityblock": DistanceType.L1,
    "cosine": DistanceType.CosineExpanded,
    "euclidean": DistanceType.L2SqrtExpanded,
    "l1": DistanceType.L1,
    "l2": DistanceType.L2SqrtExpanded,
    "manhattan": DistanceType.L1,
    "sqeuclidean": DistanceType.L2Expanded,
    "canberra": DistanceType.Canberra,
    "inner_product": DistanceType.InnerProduct,
    "minkowski": DistanceType.LpUnexpanded,
    "jaccard": DistanceType.JaccardExpanded,
    "hellinger": DistanceType.HellingerExpanded,
    "chebyshev": DistanceType.Linf,
    "dice": DistanceType.DiceExpanded
}


def _determine_metric(metric, is_sparse=False):
    if is_sparse:
        metrics = PAIRWISE_DISTANCE_SPARSE_METRICS
        other = PAIRWISE_DISTANCE_METRICS
        kind = "sparse"
    else:
        metrics = PAIRWISE_DISTANCE_METRICS
        other = PAIRWISE_DISTANCE_SPARSE_METRICS
        kind = "dense"

    if metric not in metrics:
        if metric in other:
            raise ValueError(f"`{metric=!r}` is not supported on {kind} data")
        raise ValueError(f"`{metric=!r}` is not supported")
    return metrics[metric]


@mlfunc
def nan_euclidean_distances(
    X,
    Y=None,
    *,
    squared=False,
    missing_values=cp.nan,
    copy=True,
):
    """Calculate the euclidean distances in the presence of missing values.

    Compute the euclidean distance between each pair of samples in X and Y,
    where Y=X is assumed if Y=None. When calculating the distance between a
    pair of samples, this formulation ignores feature coordinates with a
    missing value in either sample and scales up the weight of the remaining
    coordinates:

        dist(x,y) = sqrt(weight * sq. distance from present coordinates)
        where,
        weight = Total # of coordinates / # of present coordinates

    For example, the distance between ``[3, na, na, 6]`` and ``[1, na, 4, 5]``
    is:

        .. math::
            \\sqrt{\\frac{4}{2}((3-1)^2 + (6-5)^2)}

    If all the coordinates are missing or if there are no common present
    coordinates then NaN is returned for that pair.

    Parameters
    ----------
    X : array-like (device or host) of shape (n_samples_X, n_features)
        Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
        ndarray, cuda array interface compliant array like CuPy.

    Y : array-like (device or host) of shape (n_samples_Y, n_features), \
        default=None
        A second feature array. If ``None``, ``Y`` is assumed to be ``X``.
        Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
        ndarray, cuda array interface compliant array like CuPy.

    squared : bool, default=False
        Return squared Euclidean distances.

    missing_values : np.nan or int, default=np.nan
        Representation of missing value.

    copy : bool, default=True,
        Whether to make a copy of X and Y when necessary. Setting to
        False can reduce memory usage, but may result in mutation
        of X and Y.

    Returns
    -------
    distances : array of shape (n_samples_X, n_samples_Y)
        Returns the distances between the row vectors of ``X``
        and the row vectors of ``Y``.
    """
    Y_is_X = Y is None or Y is X

    X = check_array(
        X,
        order="A",
        dtype=("float32", "float64"),
        ensure_all_finite="allow-nan",
        input_name="X",
        copy=copy,
    )

    if Y_is_X:
        Y = X
    else:
        Y = check_array(
            Y,
            # If X is both C and F contiguous, let Y decide contiguity
            order=(
                "C" if not X.flags.f_contiguous else
                "F" if not X.flags.c_contiguous else
                "A"
            ),
            dtype=X.dtype,
            ensure_all_finite="allow-nan",
            input_name="Y",
            copy=copy,
        )

    # Set missing values to zero
    missing_X = cp.isnan(X) if cp.isnan(missing_values) else (X == missing_values)
    X[missing_X] = 0
    if not Y_is_X:
        missing_Y = cp.isnan(Y) if cp.isnan(missing_values) else (Y == missing_values)
        Y[missing_Y] = 0

    with using_output_type("cupy"):
        distances = pairwise_distances(X, Y, metric="sqeuclidean")

    # Adjust distances for missing values
    if Y_is_X:
        XX = X * X
        distances -= cp.dot(XX, missing_X.T)
        distances -= cp.dot(missing_X, XX.T)
    else:
        XX = X * X
        YY = Y * Y
        distances -= cp.dot(XX, missing_Y.T)
        distances -= cp.dot(missing_X, YY.T)

    cp.clip(distances, 0, None, out=distances)

    if Y_is_X:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        cp.fill_diagonal(distances, 0.0)

    present_X = 1 - missing_X
    present_Y = present_X if Y_is_X else ~missing_Y
    present_count = cp.dot(present_X, present_Y.T)
    distances[present_count == 0] = cp.nan

    # avoid divide by zero
    cp.maximum(1, present_count, out=present_count)
    distances /= present_count
    distances *= X.shape[1]

    if not squared:
        cp.sqrt(distances, out=distances)

    return distances


_all_boolean = cp.ReductionKernel(
    "T x",
    "uint8 out",
    "x == 0 || x == 1",
    "a && b",
    "out = a",
    "1",
    "_all_boolean",
)


def _ensure_boolean(X, metric):
    """Ensure X is bool-like (all 0 or 1), warning if conversion performed."""
    if not _all_boolean(X):
        warnings.warn(
            f"Data was converted to boolean for metric {metric}",
            DataConversionWarning,
            stacklevel=2,
        )
        out = cp.zeros_like(X)
        out[X != 0] = 1
        return out
    return X


@mlfunc
def pairwise_distances(X, Y=None, metric="euclidean", **kwds):
    """Compute the distance matrix from a feature array X and optional Y.

    This function takes either one or two feature arrays, and returns
    a distance matrix.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape=(n_samples_X, n_features)
        A feature array.

    Y : {array-like, sparse matrix}, shape=(n_samples_y, n_features), default=None
        A second feature array. If None, Y=X will be used.

    metric : str, default="euclidean"
        The metric to use when calculating distance between instances in a
        feature array. Valid options are:

        - Supports both dense and sparse data: ['canberra', 'chebyshev',
          'cityblock', 'cosine', 'euclidean', 'hellinger', 'l1', 'l2',
          'manhattan', 'minkowski', 'sqeuclidean'].

        - Supports dense only: ['correlation', 'hamming', 'jensenshannon',
          'kldivergence', 'nan_euclidean', 'russellrao'].

        - Supports sparse only: ['dice', 'inner_product', 'jaccard'].

    **kwds : optional keyword parameters
        Additional metric-specific parameters:

        - ``minkowski``:

          - ``p`` : float, default=2.0
            The p-norm to apply for Minkowski distance. ``p=1`` gives
            Manhattan distance, ``p=2`` gives Euclidean distance.

        - ``nan_euclidean``:

          - ``squared`` : bool, default=False
            Return squared Euclidean distances.
          - ``missing_values`` : np.nan or int, default=np.nan
            Representation of missing values in the input data.
          - ``copy`` : bool, default=True
            Whether to make a copy of X and Y when necessary. Setting
            to ``False`` can reduce memory usage, but may result in
            mutation of X and Y.

        All other metrics do not accept additional keyword parameters.
        Passing unknown parameters raises a ``TypeError``.

    Returns
    -------
    D : array, shape=(n_samples_X, n_samples_X) or (n_samples_X, n_samples_Y)
        A distance matrix D such that D_{i, j} is the distance between the ith
        and jth vectors of the given matrix X, if Y is None. If Y is not None,
        then D_{i, j} is the distance between the ith array from X and the jth
        array from Y.

    Examples
    --------
    >>> import cupy as cp
    >>> from cuml.metrics import pairwise_distances

    >>> X = cp.array([[0., 0., 0.], [1., 1., 1.]])
    >>> Y = cp.array([[1., 0., 0.], [1., 1., 0.]])

    >>> pairwise_distances(X, metric="sqeuclidean")
    array([[0., 3.],
           [3., 0.]])

    >>> pairwise_distances(X, Y, metric="sqeuclidean")
    array([[1., 2.],
           [2., 1.]])
    """
    if metric == "nan_euclidean":
        return nan_euclidean_distances(X, Y, **kwds)

    cdef double p = 2
    if metric == "minkowski":
        p = kwds.pop("p", 2)
    if kwds:
        raise TypeError(f"Unknown parameters {sorted(kwds)}")

    Y_is_X = Y is None or Y is X

    X = check_array(
        X,
        order="A",
        dtype=("float32", "float64"),
        input_name="X",
        accept_sparse="csr",
    )
    cdef bool is_sparse = cp_sp.issparse(X)

    if Y_is_X:
        Y = X
    else:
        Y = check_array(
            Y,
            # If X is both C and F contiguous, let Y decide contiguity
            order=(
                "A" if is_sparse else
                "C" if not X.flags.f_contiguous else
                "F" if not X.flags.c_contiguous else
                "A"
            ),
            dtype=X.dtype,
            input_name="Y",
            accept_sparse="csr",
        )

    if is_sparse != cp_sp.issparse(Y):
        raise NotImplementedError(
            "Support for a mix of sparse and dense arrays is not implemented"
        )

    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Incompatible dimension for X and Y matrices: "
            f"X.shape[1] == {X.shape[1]} while Y.shape[1] == {Y.shape[1]}"
        )

    cdef DistanceType distance_type = _determine_metric(metric, is_sparse=is_sparse)

    # Decompose X and Y into components
    cdef int X_n_rows = X.shape[0]
    cdef int Y_n_rows = Y.shape[0]
    cdef int n_cols = X.shape[1]
    cdef int X_nnz, Y_nnz
    cdef uintptr_t X_indptr_ptr, X_indices_ptr, Y_indptr_ptr, Y_indices_ptr
    if is_sparse:
        X_data = X.data
        X_indptr_ptr = X.indptr.data.ptr
        X_indices_ptr = X.indices.data.ptr
        X_nnz = X.nnz

        Y_data = Y.data
        Y_indptr_ptr = Y.indptr.data.ptr
        Y_indices_ptr = Y.indices.data.ptr
        Y_nnz = Y.nnz
    else:
        X_data = X
        Y_data = Y

    # Maybe transform original data values before extracting value pointers
    if metric in ["jaccard", "dice", "russellrao"]:
        X_data = _ensure_boolean(X_data, metric)
        Y_data = X_data if Y_is_X else _ensure_boolean(Y_data, metric)

    cdef uintptr_t X_ptr = X_data.data.ptr
    cdef uintptr_t Y_ptr = Y_data.data.ptr
    cdef bool is_row_major = False if is_sparse else Y.flags.c_contiguous
    cdef bool is_float32 = X_data.dtype == "float32"

    # Create the output array
    out = cp.zeros(
        (X_n_rows, Y_n_rows),
        dtype=X.dtype,
        order="C" if is_row_major else "F"
    )
    cdef uintptr_t out_ptr = out.data.ptr

    handle = get_handle()
    cdef handle_t *handle_ = <handle_t*> <size_t> handle.getHandle()

    with nogil:
        if is_sparse:
            if is_float32:
                pairwiseDistance_sparse(
                    handle_[0],
                    <float*>X_ptr,
                    <float*>Y_ptr,
                    <float*>out_ptr,
                    X_n_rows,
                    Y_n_rows,
                    n_cols,
                    X_nnz,
                    Y_nnz,
                    <int*>X_indptr_ptr,
                    <int*>Y_indptr_ptr,
                    <int*>X_indices_ptr,
                    <int*>Y_indices_ptr,
                    distance_type,
                    p,
                )
            else:
                pairwiseDistance_sparse(
                    handle_[0],
                    <double*>X_ptr,
                    <double*>Y_ptr,
                    <double*>out_ptr,
                    X_n_rows,
                    Y_n_rows,
                    n_cols,
                    X_nnz,
                    Y_nnz,
                    <int*>X_indptr_ptr,
                    <int*>Y_indptr_ptr,
                    <int*>X_indices_ptr,
                    <int*>Y_indices_ptr,
                    distance_type,
                    p,
                )
        else:
            if is_float32:
                pairwise_distance(
                    handle_[0],
                    <float*>X_ptr,
                    <float*>Y_ptr,
                    <float*>out_ptr,
                    X_n_rows,
                    Y_n_rows,
                    n_cols,
                    distance_type,
                    is_row_major,
                    p,
                )
            else:
                pairwise_distance(
                    handle_[0],
                    <double*>X_ptr,
                    <double*>Y_ptr,
                    <double*>out_ptr,
                    X_n_rows,
                    Y_n_rows,
                    n_cols,
                    distance_type,
                    is_row_major,
                    p,
                )
    handle.sync()

    return out
