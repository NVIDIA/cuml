#
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import inspect
import math
import warnings

import cupy as cp
import numpy as np

import cuml.internals
from cuml.common.kernel_utils import cuda_kernel_factory
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


def additive_chi2_kernel(X, Y):
    assert X.flags.c_contiguous
    assert Y.flags.c_contiguous
    assert X.shape[1] == Y.shape[1]

    code = """
    ({0} *X, {1} *Y, double *K, int X_n_rows, int Y_n_rows, int n_cols, int X_is_y) {
        long long x_row = threadIdx.x + blockIdx.x * blockDim.x;
        long long y_row = threadIdx.y + blockIdx.y * blockDim.y;

        if (x_row >= X_n_rows) return;
        if (y_row >= Y_n_rows) return;
        if (X_is_y && x_row > y_row) return;

        long long x_start = x_row * n_cols;
        long long y_start = y_row * n_cols;

        double res = 0.0;
        for (int i = 0; i < n_cols; i++) {
            double x = X[x_start + i];
            double y = Y[y_start + i];
            if (x != y) {
                res += (x - y) * (x - y) / (x + y);
            }
        }
        res = -res;

        K[x_row * Y_n_rows + y_row] = res;
        if (X_is_y) {
            K[y_row * Y_n_rows + x_row] = res;
        }
    }
    """
    # Always use float64 for precision
    K = cp.empty((X.shape[0], Y.shape[0]), dtype="float64", order="C")
    kernel = cuda_kernel_factory(code, (X.dtype, Y.dtype), "additive_chi2")
    kernel(
        (math.ceil(X.shape[0] / 32), math.ceil(Y.shape[0] / 32)),
        (32, 32),
        (
            X,
            Y,
            K,
            np.int32(X.shape[0]),
            np.int32(Y.shape[0]),
            np.int32(X.shape[1]),
            np.int32(X is Y),
        ),
    )
    return K


def chi2_kernel(X, Y, gamma=1.0):
    K = additive_chi2_kernel(X, Y)
    K *= gamma
    return cp.exp(K, K)


PAIRWISE_KERNEL_FUNCTIONS = {
    "additive_chi2": additive_chi2_kernel,
    "chi2": chi2_kernel,
    "cosine": cosine_similarity,
    "laplacian": laplacian_kernel,
    "linear": linear_kernel,
    "poly": polynomial_kernel,
    "polynomial": polynomial_kernel,
    "rbf": rbf_kernel,
    "sigmoid": sigmoid_kernel,
}

_KERNEL_PARAMS = {
    "additive_chi2": (),
    "chi2": frozenset(["gamma"]),
    "cosine": (),
    "laplacian": frozenset(["gamma"]),
    "linear": (),
    "poly": frozenset(["gamma", "degree", "coef0"]),
    "polynomial": frozenset(["gamma", "degree", "coef0"]),
    "rbf": frozenset(["gamma"]),
    "sigmoid": frozenset(["gamma", "coef0"]),
}


def _custom_kernel(X, Y, metric, filter_params=False, **kwds):
    """Implement a custom pairwise_kernels metric.

    Custom metrics are compiled using numba. The compilation is cached per
    function to reduce overhead.
    """
    # Validate `metric` is a numba device function, and determine which
    # numba compiler backend to use
    cuda = None
    if hasattr(metric, "py_func"):
        mod = type(metric).__module__
        if mod.startswith("numba_cuda_mlir"):
            from numba_cuda_mlir import cuda
            from numba_cuda_mlir.numba_cuda.core.errors import (
                NumbaPerformanceWarning,
            )
        elif mod.startswith(("numba.cuda", "numba_cuda")):
            from numba import cuda
            from numba.cuda.core.errors import NumbaPerformanceWarning
    if cuda is None or not metric.targetoptions.get("device", False):
        raise TypeError("Kernel function should be a numba device function")

    # Inspect function parameters, ensuring at least X, Y present
    params = list(inspect.signature(metric.py_func).parameters.values())
    if len(params) < 2:
        raise ValueError("Expected at least two arguments to kernel function")

    # Validate any optional parameters, and convert to explicit arguments
    optional = {}
    for p in params[2:]:
        if p.default is inspect.Parameter.empty:
            raise ValueError(
                "Extra kernel parameters must be passed as keyword arguments"
            )
        optional[p.name] = p.default
    if filter_params:
        kwds = {k: v for k, v in kwds.items() if k in optional}
    elif unknown := set(kwds).difference(optional):
        raise TypeError(
            f"{metric.__qualname__}() got unexpected keyword arguments: "
            f"{list(unknown)}"
        )
    optional.update(kwds)

    if (kernel := getattr(metric, "__cuml_cached_kernel__", None)) is None:
        code = (
            "@cuda.jit\n"
            "def kernel(X, Y, K, symmetric, {args}):\n"
            "    x_row = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x\n"
            "    y_row = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y\n"
            "    if (x_row >= X.shape[0]) or (y_row >= Y.shape[0]):\n"
            "        return\n"
            "    if symmetric and x_row > y_row:\n"
            "        return\n"
            "    k = metric(X[x_row], Y[y_row], {args})\n"
            "    K[x_row, y_row] = k\n"
            "    if symmetric:\n"
            "        K[y_row, x_row] = k\n"
        ).format(args=", ".join(f"_{k}_" for k in optional))
        ns = {"cuda": cuda, "metric": metric}
        exec(code, ns)
        kernel = ns["kernel"]
        # Cache the kernel on the metric function. This cannot fail, since the
        # numba compilers will only jit python functions, and those always have
        # a mutable `__dict__`. We do this instead of using an external cache
        # so that the lifetime of the kernel is tied to the lifetime of the
        # metric function. Using a WeakKeyDictionary wouldn't work since
        # kernel.py_func==metric, which would keep the key alive.
        metric.__cuml_cached_kernel__ = kernel

    # float64 always used for the output to avoid numeric precision issues
    K = cp.zeros((X.shape[0], Y.shape[0]), dtype="float64")
    # Silence the NumbaPerformanceWarnings. These are raised for small inputs.
    # Since there's nothing the user can do (we chose numba here, we manage the
    # configuration) it doesn't make sense to surface these to the user.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NumbaPerformanceWarning)
        kernel[
            (math.ceil(X.shape[0] / 32), math.ceil(Y.shape[0] / 32)),
            (32, 32),
        ](X, Y, K, X is Y, *optional.values())
    return K


@cuml.internals.mlfunc
def pairwise_kernels(
    X, Y=None, metric="linear", *, filter_params=False, **kwds
):
    """
    Compute the kernel between array X and optional array Y.

    This function provides a safe way to take a kernel matrix as input, while
    preserving compatibility with many other algorithms that take a vector
    array.

    Parameters
    ----------
    X : array-like of shape (n_samples_X, n_samples_X) or (n_samples_X, n_features)
        Array of pairwise kernels between samples, or a feature array.
        The shape of the array should be (n_samples_X, n_samples_X) if
        metric == "precomputed" and (n_samples_X, n_features) otherwise.

    Y : array-like of shape (n_samples_Y, n_features), default=None
        A second feature array only if X has shape (n_samples_X, n_features).
        If ``Y=None``, ``Y=X`` is used instead.

    metric : str or callable, default="linear"
        The metric to use when calculating kernel between instances in a
        feature array.

        Valid metrics are: ['additive_chi2', 'chi2', 'cosine', 'laplacian',
        'linear', 'poly', 'polynomial', 'precomputed, 'rbf', 'sigmoid'].

        If metric is "precomputed", X is assumed to be a kernel matrix
        and is returned directly.

        If metric is a callable function, it is called on each pair of
        instances (rows) and the resulting value recorded. The callable should
        be a numba device function. It should take two rows from X as input and
        return the corresponding kernel value as a single number. Extra
        parameters are supported as keyword arguments.

    filter_params : bool, default=False
        Whether to filter invalid parameters or not.

    **kwds : optional keyword parameters
        Any further parameters are passed directly to the kernel function.

    Returns
    -------
    K : ndarray of shape (n_samples_X, n_samples_X) or (n_samples_X, n_samples_Y)
        A kernel matrix K such that K_{i, j} is the kernel between the
        ith and jth vectors of the given matrix X, if Y is None.
        If Y is not None, then K_{i, j} is the kernel between the ith array
        from X and the jth array from Y.

    Examples
    --------
    >>> import cupy as cp
    >>> from cuml.metrics import pairwise_kernels

    >>> X = cp.array([[2, 3], [3, 5], [5, 8]])
    >>> Y = cp.array([[1, 0], [2, 1]])
    >>> pairwise_kernels(X, Y, metric='linear')
    array([[ 2,  7],
           [ 3, 11],
           [ 5, 18]])

    Custom kernels may be provided as numba compiled device functions.

    >>> import math
    >>> from numba_cuda_mlir import cuda  # doctest: +SKIP

    >>> @cuda.jit(device=True)  # doctest: +SKIP
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
    if metric in ("additive_chi2", "chi2"):
        ensure_non_negative = True
        order = "C"
    else:
        ensure_non_negative = False
        order = "A"

    # Avoid duplicating host->device transfer if Y=X is explicitly provided
    if Y is X:
        Y = None

    X = check_array(
        X,
        input_name="X",
        ensure_non_negative=ensure_non_negative,
        order=order,
    )
    if Y is None:
        Y = X
    else:
        Y = check_array(
            Y,
            input_name="Y",
            ensure_non_negative=ensure_non_negative,
            order=order,
        )

    if isinstance(metric, str) and metric == "precomputed":
        if X.shape[1] != Y.shape[0]:
            raise ValueError(
                "Precomputed metric requires shape (n_queries, n_indexed). "
                f"Got {X.shape} for {Y.shape[0]} indexed."
            )
        return X

    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            "Incompatible dimension for X and Y matrices: "
            f"X.shape[1] == {X.shape[1]} while Y.shape[1] == {Y.shape[1]}"
        )

    if isinstance(metric, str):
        if (func := PAIRWISE_KERNEL_FUNCTIONS.get(metric)) is None:
            raise ValueError(f"metric={metric!r} is not supported")
        if filter_params:
            kwds = {k: kwds[k] for k in kwds if k in _KERNEL_PARAMS[metric]}
        return func(X, Y, **kwds)
    else:
        return _custom_kernel(X, Y, metric, filter_params, **kwds)
