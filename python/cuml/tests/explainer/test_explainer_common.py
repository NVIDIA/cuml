#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
import numpy as np
import pytest

from cuml import PCA
from cuml import LinearRegression as reg
from cuml.datasets import make_regression
from cuml.explainer.common import (
    get_cai_ptr,
    get_link_fn_from_str_or_fn,
    link_dict,
    model_func_call,
)


def test_model_func_call_gpu():
    X, y = make_regression(
        n_samples=81,
        n_features=10,
        noise=0.1,
        random_state=42,
        dtype=np.float32,
    )

    model = reg().fit(X, y)

    z = model_func_call(X=X, model_func=model.predict, gpu_model=True)

    assert isinstance(z, cp.ndarray)

    z = model_func_call(
        X=cp.asnumpy(X), model_func=dummy_func, gpu_model=False
    )

    assert isinstance(z, cp.ndarray)

    with pytest.raises(TypeError):
        z = model_func_call(X=X, model_func=dummy_func, gpu_model=True)

    model = PCA(n_components=10).fit(X)

    z = model_func_call(X=X, model_func=model.transform, gpu_model=True)

    assert isinstance(z, cp.ndarray)


def test_get_cai_ptr():
    a = cp.ones(10)
    ptr = get_cai_ptr(a)

    assert ptr == a.__cuda_array_interface__["data"][0]

    b = np.ones(10)
    with pytest.raises(TypeError):
        ptr = get_cai_ptr(b)


@pytest.mark.parametrize("link_function", ["identity", "logit"])
def test_get_link_fn_from_str(link_function):
    fn = get_link_fn_from_str_or_fn(link_function)
    a = cp.ones(10)

    assert cp.all(fn(a) == link_dict[link_function](a))
    assert cp.all(fn.inverse(a) == link_dict[link_function].inverse(a))


def test_get_link_fn_from_wrong_str():
    with pytest.raises(ValueError):
        get_link_fn_from_str_or_fn("this_is_wrong")


def test_get_link_fn_from_fn():
    def dummylink(x):
        return 2 * x

    # check we raise error if link has no inverse
    with pytest.raises(TypeError):
        get_link_fn_from_str_or_fn(dummylink)

    def dummylink_inv(x):
        return x / 2

    dummylink.inverse = dummylink_inv

    fn = get_link_fn_from_str_or_fn(dummylink)

    assert fn(2) == 4
    assert fn.inverse(2) == 1


def dummy_func(x):
    if not isinstance(x, np.ndarray):
        raise TypeError("x must be a NumPy array")
    return np.mean(x, axis=1)
