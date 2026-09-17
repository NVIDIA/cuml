# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import inspect
import math

import cupy as cp
import numpy as np
import pytest
from cupy import linalg
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numba import cuda
from sklearn.kernel_ridge import KernelRidge as sklKernelRidge
from sklearn.metrics.pairwise import pairwise_kernels as skl_pairwise_kernels

import cuml
from cuml import KernelRidge as cuKernelRidge
from cuml.metrics import PAIRWISE_KERNEL_FUNCTIONS, pairwise_kernels
from cuml.testing.utils import as_cupy, as_numpy, as_type


def gradient_norm(model, X, y, K, sw=None):
    if sw is None:
        sw = cp.ones(X.shape[0])
    else:
        sw = cp.atleast_1d(cp.array(sw, dtype=np.float64))

    if y.ndim == 1:
        y = y[:, None]

    X = cp.array(X, dtype=np.float64)
    y = cp.array(y, dtype=np.float64)
    K = cp.array(K, dtype=np.float64)
    betas = as_cupy(model.dual_coef_).astype("f8").reshape(y.shape)
    alphas = cp.asarray(model.alpha)

    # initialise to NaN in case below loop has 0 iterations
    grads = cp.full_like(y, np.nan)
    for i, (beta, target, current_alpha) in enumerate(
        zip(betas.T, y.T, alphas)
    ):
        grads[:, i] = 0.0
        grads[:, i] = -cp.dot(K * sw, target)
        grads[:, i] += cp.dot(cp.dot(K * sw, K), beta)
        grads[:, i] += cp.dot(K * current_alpha, beta)
    return linalg.norm(grads)


def test_pairwise_kernels_basic():
    X = np.zeros((4, 4))
    # standard kernel with no argument
    pairwise_kernels(X, metric="chi2")
    pairwise_kernels(X, metric="linear")
    # standard kernel with correct kwd argument
    pairwise_kernels(X, metric="chi2", gamma=1.0)
    # standard kernel with incorrect kwd argument
    with pytest.raises(
        ValueError, match="kwds contains arguments not used by kernel function"
    ):
        pairwise_kernels(X, metric="linear", wrong_parameter_name=1.0)
    # standard kernel with filtered kwd argument
    pairwise_kernels(
        X, metric="rbf", filter_params=True, wrong_parameter_name=1.0
    )

    # incorrect function type
    def non_numba_kernel(x, y):
        return x.dot(y)

    with pytest.raises(
        TypeError, match="Kernel function should be a numba device function."
    ):
        pairwise_kernels(X, metric=non_numba_kernel)

    # correct function type
    @cuda.jit(device=True)
    def numba_kernel(x, y, special_argument=3.0):
        return 1 + 2

    pairwise_kernels(X, metric=numba_kernel)
    pairwise_kernels(X, metric=numba_kernel, special_argument=1.0)

    # malformed function
    @cuda.jit(device=True)
    def bad_numba_kernel(x):
        return 1 + 2

    with pytest.raises(
        ValueError, match="Expected at least two arguments to kernel function."
    ):
        pairwise_kernels(X, metric=bad_numba_kernel)

    # malformed function 2 - No default value
    @cuda.jit(device=True)
    def bad_numba_kernel2(x, y, z):
        return 1 + 2

    with pytest.raises(
        ValueError,
        match="Extra kernel parameters must be passed as keyword arguments.",
    ):
        pairwise_kernels(X, metric=bad_numba_kernel2)

    # Precomputed
    assert np.allclose(X, pairwise_kernels(X, metric="precomputed"))


@pytest.mark.parametrize("metric", ["additive_chi2", "chi2"])
def test_pairwise_kernels_rejects_negative_x_for_chi2(metric):
    X = np.array([[1.0, -1.0], [2.0, 3.0]])

    with pytest.raises(
        ValueError, match="Negative values in data passed to X"
    ):
        pairwise_kernels(X, metric=metric)


@pytest.mark.parametrize("metric", ["additive_chi2", "chi2"])
def test_pairwise_kernels_rejects_negative_y_for_chi2(metric):
    X = np.array([[1.0, 2.0]])
    Y = np.array([[3.0, -1.0]])

    with pytest.raises(
        ValueError, match="Negative values in data passed to Y"
    ):
        pairwise_kernels(X, Y, metric=metric)


@cuda.jit(device=True)
def custom_kernel(x, y, custom_arg=5.0):
    sum = 0.0
    for i in range(len(x)):
        sum += (x[i] - y[i]) ** 2
    return math.exp(-custom_arg * sum) + 0.1


test_kernels = sorted(PAIRWISE_KERNEL_FUNCTIONS.keys()) + [custom_kernel]


@st.composite
def kernel_arg_strategy(draw):
    kernel = draw(st.sampled_from(test_kernels))
    kernel_func = (
        PAIRWISE_KERNEL_FUNCTIONS[kernel]
        if isinstance(kernel, str)
        else kernel
    )
    # Inspect the function and generate some arguments
    py_func = (
        kernel_func.py_func if hasattr(kernel_func, "py_func") else kernel_func
    )
    all_func_kwargs = list(inspect.signature(py_func).parameters.values())[2:]
    param = {}
    for arg in all_func_kwargs:
        # 50% chance we generate this parameter or leave it as default
        if draw(st.booleans()):
            continue
        if isinstance(arg.default, float) or arg.default is None:
            param[arg.name] = draw(st.floats(0.0, 5.0))
        if isinstance(arg.default, int):
            param[arg.name] = draw(st.integers(1, 5))

    return (kernel, param)


@st.composite
def array_strategy(draw):
    X_m = draw(st.integers(1, 20))
    X_n = draw(st.integers(1, 10))
    dtype = draw(st.sampled_from([np.float64, np.float32]))
    X = draw(
        arrays(
            dtype=dtype,
            shape=(X_m, X_n),
            elements=st.floats(0, 5, width=32),
        )
    )
    if draw(st.booleans()):
        Y_m = draw(st.integers(1, 20))
        Y = draw(
            arrays(
                dtype=dtype,
                shape=(Y_m, X_n),
                elements=st.floats(0, 5, width=32),
            )
        )
    else:
        Y = None
    type = draw(st.sampled_from(["numpy", "cupy", "cudf", "pandas"]))

    if type == "cudf":
        assume(X_m > 1)
        if Y is not None:
            assume(Y_m > 1)
    return as_type(type, X, Y)


@example(
    kernel_arg=("linear", {}),
    XY=(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[1.5, 2.5]])),
)
@given(kernel_arg_strategy(), array_strategy())
@settings(deadline=None)
@pytest.mark.skip("https://github.com/NVIDIA/cuml/issues/5177")
def test_pairwise_kernels(kernel_arg, XY):
    X, Y = XY
    kernel, args = kernel_arg

    if kernel == "cosine":
        # this kernel is very unstable for both sklearn/cuml
        assume(as_numpy(X).min() > 0.1)
        if Y is not None:
            assume(as_numpy(Y).min() > 0.1)

    K = pairwise_kernels(X, Y, metric=kernel, **args)
    skl_kernel = kernel.py_func if hasattr(kernel, "py_func") else kernel
    K_sklearn = skl_pairwise_kernels(
        *as_numpy(X, Y), metric=skl_kernel, **args
    )
    assert np.allclose(as_numpy(K), K_sklearn, atol=0.01, rtol=0.01)


@st.composite
def estimator_array_strategy(draw):
    X_m = draw(st.integers(5, 20))
    X_n = draw(st.integers(2, 10))
    dtype = draw(st.sampled_from([np.float64, np.float32]))
    rs = np.random.RandomState(draw(st.integers(1, 10)))
    X = rs.rand(X_m, X_n).astype(dtype)
    X_test = rs.rand(draw(st.integers(5, 20)), X_n).astype(dtype)

    n_targets = draw(st.integers(1, 3))
    a = draw(
        arrays(
            dtype=dtype,
            shape=(X_n, n_targets),
            elements=st.floats(0, 5, width=32),
        )
    )
    y = X.dot(a)

    alpha = draw(
        arrays(
            dtype=dtype,
            shape=(n_targets),
            elements=st.floats(0.0010000000474974513, 5, width=32),
        )
    )

    sample_weight = draw(
        st.one_of(
            [
                st.just(None),
                st.floats(0.1, 1.5),
                arrays(
                    dtype=np.float64, shape=X_m, elements=st.floats(0.1, 5)
                ),
            ]
        )
    )
    type = draw(st.sampled_from(["numpy", "cupy", "cudf", "pandas"]))
    return (*as_type(type, X, y, X_test, alpha, sample_weight), dtype)


@example(
    kernel_arg=("linear", {}),
    arrays=(
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),  # X
        np.array([1.0, 2.0, 3.0]),  # y
        np.array([[2.0, 3.0], [4.0, 5.0]]),  # X_test
        np.array([0.1]),  # alpha
        None,  # sample_weight
        np.float64,  # dtype
    ),
    gamma=1.0,
    degree=1,
    coef0=0.0,
)
@example(
    kernel_arg=("linear", {}),
    arrays=(
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),  # X
        np.array([[1.0, 2.0, 3.0]]).T,  # y
        np.array([[2.0, 3.0], [4.0, 5.0]]),  # X_test
        np.array([0.1]),  # alpha
        0.5,  # sample_weight
        np.float64,  # dtype
    ),
    gamma=1.0,
    degree=1,
    coef0=0.0,
)
@given(
    kernel_arg_strategy(),
    estimator_array_strategy(),
    st.floats(1.0, 5.0),
    st.integers(1, 5),
    st.floats(1.0, 5.0),
)
@settings(deadline=None)
def test_estimator(kernel_arg, arrays, gamma, degree, coef0):
    kernel, args = kernel_arg
    X, y, X_test, alpha, sample_weight, dtype = arrays
    model = cuKernelRidge(
        kernel=kernel,
        alpha=alpha,
        gamma=gamma,
        degree=degree,
        coef0=coef0,
        kernel_params=args,
    )
    skl_kernel = kernel.py_func if hasattr(kernel, "py_func") else kernel
    skl_model = sklKernelRidge(
        kernel=skl_kernel,
        alpha=as_numpy(alpha),
        gamma=gamma,
        degree=degree,
        coef0=coef0,
        kernel_params=args,
    )
    if kernel == "chi2" or kernel == "additive_chi2":
        # X must be positive
        X = (X - as_numpy(X).min()) + 1.0

    model.fit(X, y, sample_weight)
    pred = model.predict(X_test)
    if dtype == np.float64:
        # For a convex optimisation problem we should arrive at gradient norm 0
        # If the solution has converged correctly
        K = model._get_kernel(X)
        grad_norm = gradient_norm(model, *as_cupy(X, y, K, sample_weight))
        assert grad_norm < 0.1
        try:
            skl_model.fit(*as_numpy(X, y, sample_weight))
        except np.linalg.LinAlgError:
            # sklearn can fail to fit multiclass models
            # with singular kernel matrices
            assume(False)

        skl_pred = skl_model.predict(as_numpy(X_test))
        assert np.allclose(
            as_numpy(pred).squeeze(),
            skl_pred.squeeze(),
            atol=1e-2,
            rtol=1e-2,
        )


def test_predict_output_type():
    rng = np.random.RandomState(42)

    X = 5 * rng.rand(10000, 1)
    y = np.sin(X).ravel()

    kr = cuKernelRidge(kernel="rbf", gamma=0.1)
    kr.fit(X, y)

    res = kr.predict(X)
    assert isinstance(res, np.ndarray)

    with cuml.using_output_type("cupy"):
        res = kr.predict(X)
    assert isinstance(res, cp.ndarray)


def test_precomputed():
    rs = np.random.RandomState(23)
    X = rs.normal(size=(10, 10))
    y = rs.normal(size=10)
    K = pairwise_kernels(X)
    precomputed_model = cuKernelRidge(kernel="precomputed")
    precomputed_model.fit(K, y)
    model = cuKernelRidge()
    model.fit(X, y)
    assert np.allclose(precomputed_model.dual_coef_, model.dual_coef_)
    assert np.allclose(
        precomputed_model.predict(K), model.predict(X), atol=1e-5, rtol=1e-5
    )
