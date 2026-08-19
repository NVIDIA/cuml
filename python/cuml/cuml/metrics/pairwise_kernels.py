#
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import inspect

import cupy as cp
import numpy as np
from numba import cuda

import cuml.internals
from cuml.internals.validation import check_array
from cuml.metrics import pairwise_distances


def linear_kernel(X, Y):
    return cp.dot(X, Y.T)


def polynomial_kernel(X, Y, degree=3, gamma=None, coef0=1):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K = cp.dot(X, Y.T)
    K *= gamma
    K += coef0
    K **= degree
    return K


def sigmoid_kernel(X, Y, gamma=None, coef0=1):
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = cp.dot(X, Y.T)
    K *= gamma
    K += coef0
    cp.tanh(K, K)
    return K


def rbf_kernel(X, Y, gamma=None):
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = cp.asarray(pairwise_distances(X, Y, metric="sqeuclidean"))
    K *= -gamma
    cp.exp(K, K)
    return K


def laplacian_kernel(X, Y, gamma=None):
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = -gamma * cp.asarray(pairwise_distances(X, Y, metric="manhattan"))
    cp.exp(K, K)
    return K


def cosine_similarity(X, Y):
    K = 1.0 - cp.asarray(pairwise_distances(X, Y, metric="cosine"))
    return cp.nan_to_num(K, copy=False)


@cuda.jit(device=True)
def additive_chi2_kernel_element(x, y):
    res = 0.0
    for i in range(len(x)):
        denom = x[i] - y[i]
        nom = x[i] + y[i]
        if nom != 0.0:
            res += denom * denom / nom
    return -res


def additive_chi2_kernel(X, Y):
    return custom_kernel(X, Y, additive_chi2_kernel_element)


def chi2_kernel(X, Y, gamma=1.0):
    K = additive_chi2_kernel(X, Y)
    K *= gamma
    return cp.exp(K, K)


PAIRWISE_KERNEL_FUNCTIONS = {
    "linear": linear_kernel,
    "additive_chi2": additive_chi2_kernel,
    "chi2": chi2_kernel,
    "cosine": cosine_similarity,
    "laplacian": laplacian_kernel,
    "polynomial": polynomial_kernel,
    "poly": polynomial_kernel,
    "rbf": rbf_kernel,
    "sigmoid": sigmoid_kernel,
}


def _filter_params(func, filter_params, **kwds):
    # get all the possible extra function arguments, excluding x, y
    py_func = func.py_func if hasattr(func, "py_func") else func
    all_func_kwargs = list(inspect.signature(py_func).parameters.values())
    if len(all_func_kwargs) < 2:
        raise ValueError("Expected at least two arguments to kernel function.")

    extra_arg_names = set(arg.name for arg in all_func_kwargs[2:])
    if not filter_params:
        if not set(kwds.keys()) <= extra_arg_names:
            raise ValueError(
                "kwds contains arguments not used by kernel function"
            )
    return {k: v for k, v in kwds.items() if k in extra_arg_names}


def _kwds_to_tuple_args(func, **kwds):
    # Returns keyword arguments formed as a tuple
    # (numba kernels cannot deal with kwargs as a dict)
    if not hasattr(func, "py_func"):
        raise TypeError("Kernel function should be a numba device function.")

    # get all the possible extra function arguments, excluding x, y
    all_func_kwargs = list(inspect.signature(func.py_func).parameters.values())
    if len(all_func_kwargs) < 2:
        raise ValueError("Expected at least two arguments to kernel function.")

    all_func_kwargs = all_func_kwargs[2:]
    if any(p.default is inspect.Parameter.empty for p in all_func_kwargs):
        raise ValueError(
            "Extra kernel parameters must be passed as keyword arguments."
        )
    all_func_kwargs = [(k.name, k.default) for k in all_func_kwargs]

    kwds_tuple = tuple(
        kwds[k] if k in kwds.keys() else v for (k, v) in all_func_kwargs
    )
    return kwds_tuple


_kernel_cache = {}


def custom_kernel(X, Y, func, **kwds):
    kwds_tuple = _kwds_to_tuple_args(func, **kwds)

    def evaluate_pairwise_kernels(X, Y, K):
        idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        X_m = X.shape[0]
        Y_m = Y.shape[0]
        row = idx // Y_m
        col = idx % Y_m
        if idx < X_m * Y_m:
            if X is Y and row <= col:
                # matrix is symmetric, reuse half the evaluations
                k = func(X[row], Y[col], *kwds_tuple)
                K[row, col] = k
                K[col, row] = k
            else:
                k = func(X[row], Y[col], *kwds_tuple)
                K[row, col] = k

    if Y is None:
        Y = X
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y have different dimensions.")

    # Here we force K to use 64 bit, even if the input is 32 bit
    # 32 bit K results in serious numerical stability problems
    K = cp.zeros((X.shape[0], Y.shape[0]), dtype=np.float64)

    key = (func, kwds_tuple, X.dtype, Y.dtype)
    if key in _kernel_cache:
        compiled_kernel = _kernel_cache[key]
    else:
        compiled_kernel = cuda.jit(evaluate_pairwise_kernels)
        _kernel_cache[key] = compiled_kernel
    compiled_kernel.forall(X.shape[0] * Y.shape[0])(X, Y, K)
    return K


@cuml.internals.mlfunc
def pairwise_kernels(
    X,
    Y=None,
    metric="linear",
    *,
    filter_params=False,
    **kwds,
):
    """
    Compute the kernel between arrays X and optional array Y.
    This method takes either a vector array or a kernel matrix, and returns
    a kernel matrix. If the input is a vector array, the kernels are
    computed. If the input is a kernel matrix, it is returned instead.
    This method provides a safe way to take a kernel matrix as input, while
    preserving compatibility with many other algorithms that take a vector
    array.
    If Y is given (default is None), then the returned matrix is the pairwise
    kernel between the arrays from both X and Y.
    Valid values for metric are: ['additive_chi2', 'chi2', 'linear', 'poly',
    'polynomial', 'rbf', 'laplacian', 'sigmoid', 'cosine']

    Parameters
    ----------
    X : array-like (device or host) of shape (n_samples_X, n_samples_X) or \
            (n_samples_X, n_features)
        Array of pairwise kernels between samples, or a feature array.
        The shape of the array should be (n_samples_X, n_samples_X) if
        metric == "precomputed" and (n_samples_X, n_features) otherwise.
        Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
        ndarray, cuda array interface compliant array like CuPy.
    Y : array-like (device or host) of shape (n_samples_Y, n_features), \
        default=None
        A second feature array only if X has shape (n_samples_X, n_features).
        Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
        ndarray, cuda array interface compliant array like CuPy.
    metric : str or callable (numba device function), default="linear"
        The metric to use when calculating kernel between instances in a
        feature array.
        If metric is "precomputed", X is assumed to be a kernel matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two rows from X as input and return the corresponding
        kernel value as a single number.
    filter_params : bool, default=False
        Whether to filter invalid parameters or not.
    **kwds : optional keyword parameters
        Any further parameters are passed directly to the kernel function.

    Returns
    -------
    K : ndarray of shape (n_samples_X, n_samples_X) or \
            (n_samples_X, n_samples_Y)
        A kernel matrix K such that K_{i, j} is the kernel between the
        ith and jth vectors of the given matrix X, if Y is None.
        If Y is not None, then K_{i, j} is the kernel between the ith array
        from X and the jth array from Y.

    Notes
    -----
    If metric is 'precomputed', Y is ignored and X is returned.

    Examples
    --------

    .. code-block:: python

        >>> import cupy as cp
        >>> from cuml.metrics import pairwise_kernels
        >>> from numba import cuda
        >>> import math

        >>> X = cp.array([[2, 3], [3, 5], [5, 8]])
        >>> Y = cp.array([[1, 0], [2, 1]])

        >>> pairwise_kernels(X, Y, metric='linear')
        array([[ 2,  7],
            [ 3, 11],
            [ 5, 18]])
        >>> @cuda.jit(device=True)
        ... def custom_rbf_kernel(x, y, gamma=None):
        ...     if gamma is None:
        ...         gamma = 1.0 / len(x)
        ...     sum = 0.0
        ...     for i in range(len(x)):
        ...         sum += (x[i] - y[i]) ** 2
        ...     return math.exp(-gamma * sum)

        >>> pairwise_kernels(X, Y, metric=custom_rbf_kernel) # doctest: +SKIP
        array([[6.73794700e-03, 1.35335283e-01],
            [5.04347663e-07, 2.03468369e-04],
            [4.24835426e-18, 2.54366565e-13]])
    """
    ensure_non_negative = metric in ("additive_chi2", "chi2")
    X = check_array(X, input_name="X", ensure_non_negative=ensure_non_negative)
    if Y is None:
        Y = X
    else:
        Y = check_array(
            Y, input_name="Y", ensure_non_negative=ensure_non_negative
        )
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y have different dimensions.")

    if metric == "precomputed":
        return X

    if metric in PAIRWISE_KERNEL_FUNCTIONS:
        kwds = _filter_params(
            PAIRWISE_KERNEL_FUNCTIONS[metric], filter_params, **kwds
        )
        return PAIRWISE_KERNEL_FUNCTIONS[metric](X, Y, **kwds)
    elif isinstance(metric, str):
        raise ValueError("Unknown kernel %r" % metric)
    else:
        kwds = _filter_params(metric, filter_params, **kwds)

        return custom_kernel(X, Y, metric, **kwds)
