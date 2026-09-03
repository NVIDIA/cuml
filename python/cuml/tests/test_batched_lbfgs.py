#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest
import scipy
from packaging.version import Version

from cuml.tsa.batched_lbfgs import batched_fmin_lbfgs_b


def rosenbrock(x, a=1, b=100):
    """Famous Rosenbrock example"""

    return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2


def g_rosenbrock(x, a=1, b=100):
    """Gradietn of rosenbrock example"""

    g = np.array(
        [
            -2 * a - 4 * b * x[0] * (-(x[0] ** 2) + x[1]) + 2 * x[0],
            b * (-2 * x[0] ** 2 + 2 * x[1]),
        ]
    )

    return g


def batched_rosenbrock(
    x: np.ndarray, num_batches: int, a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """A batched version of the rosenbrock example"""

    fall = np.zeros(num_batches)
    for i in range(num_batches):
        fall[i] = rosenbrock(x[i * 2 : (i + 1) * 2], a[i], b[i])

    return fall


def g_batched_rosenbrock(
    x: np.ndarray, num_batches: int, a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Gradient of the batched rosenbrock example."""
    gall = np.zeros(2 * num_batches)
    for i in range(num_batches):
        gall[i * 2 : (i + 1) * 2] = g_rosenbrock(
            x[i * 2 : (i + 1) * 2], a[i], b[i]
        )

    return gall


@pytest.mark.xfail(
    condition=Version(scipy.__version__) >= Version("1.15"),
    reason="https://github.com/NVIDIA/cuml/issues/6210",
)
def test_batched_lbfgs_rosenbrock():
    """Test batched rosenbrock using batched lbfgs implemtnation"""

    num_batches = 5
    np.random.seed(42)
    a = np.random.normal(1, scale=0.1, size=num_batches)
    b = np.random.normal(100, scale=10, size=num_batches)

    def f(x, n=None):
        nonlocal a
        nonlocal b
        nonlocal num_batches

        if n is not None:
            return rosenbrock(x, a[n], b[n])

        fb = batched_rosenbrock(x, num_batches, a, b)
        return fb

    def gf(x, n=None):
        nonlocal a
        nonlocal b
        nonlocal num_batches

        if n is not None:
            return g_rosenbrock(x, a[n], b[n])

        g = g_batched_rosenbrock(x, num_batches, a, b)
        return g

    x0 = np.zeros(2 * num_batches)
    x0[0] = 0.0
    x0[1] = 0.0

    # analytical minimum
    res_true = np.zeros(num_batches * 2)
    for i in range(num_batches):
        res_true[i * 2 : (i + 1) * 2] = np.array([a[i], a[i] ** 2])

    # our new batch-aware l-bfgs optimizer
    res_xk, _, _ = batched_fmin_lbfgs_b(
        f, x0, num_batches, gf, iprint=-1, factr=100
    )
    np.testing.assert_allclose(res_xk, res_true, rtol=1e-5)


if __name__ == "__main__":
    test_batched_lbfgs_rosenbrock()
