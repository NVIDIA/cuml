# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Note: this isn't a strict test, the goal is to test the Cython interface
# and cover all the parameters.
# See the C++ test for an actual correction test

import pytest

import cuml

# Testing parameters

dtype = ["single", "double"]
n_samples = [100, 100000]
n_features = [10, 100]
n_informative = [7]
n_targets = [1, 3]
shuffle = [True, False]
coef = [True, False]
effective_rank = [None, 6]
random_state = [None, 1234]
bias = [-4.0]
noise = [3.5]


@pytest.mark.parametrize("dtype", dtype)
@pytest.mark.parametrize("n_samples", n_samples)
@pytest.mark.parametrize("n_features", n_features)
@pytest.mark.parametrize("n_informative", n_informative)
@pytest.mark.parametrize("n_targets", n_targets)
@pytest.mark.parametrize("shuffle", shuffle)
@pytest.mark.parametrize("coef", coef)
@pytest.mark.parametrize("effective_rank", effective_rank)
@pytest.mark.parametrize("random_state", random_state)
@pytest.mark.parametrize("bias", bias)
@pytest.mark.parametrize("noise", noise)
def test_make_regression(
    dtype,
    n_samples,
    n_features,
    n_informative,
    n_targets,
    shuffle,
    coef,
    effective_rank,
    random_state,
    bias,
    noise,
):
    result = cuml.make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_targets=n_targets,
        bias=bias,
        effective_rank=effective_rank,
        noise=noise,
        shuffle=shuffle,
        coef=coef,
        random_state=random_state,
        dtype=dtype,
    )

    if coef:
        X, y, coefs = result
    else:
        X, y = result

    assert X.shape == (n_samples, n_features)

    if n_targets == 1:
        assert y.shape == (n_samples,)
    else:
        assert y.shape == (n_samples, n_targets)

    if coef:
        if n_targets == 1:
            assert coefs.shape == (n_features,)
        else:
            assert coefs.shape == (n_features, n_targets)
