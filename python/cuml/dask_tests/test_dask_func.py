#
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import pytest
from dask import delayed

from cuml.dask.common.func import reduce, tree_reduce


@pytest.mark.parametrize("n_parts", [1, 2, 10, 15])
def test_tree_reduce_delayed(n_parts, client):
    func = delayed(sum)

    a = [delayed(i) for i in range(n_parts)]
    b = tree_reduce(a, func=func)
    c = client.compute(b, sync=True)

    assert sum(range(n_parts)) == c


# Using custom remote task for storing data on workers.
# `client.scatter` doesn't seem to work reliably
# Ref: https://github.com/dask/dask/issues/6027
def s(x):
    return x


@pytest.mark.parametrize("n_parts", [1, 2, 10, 15])
def test_tree_reduce_futures(n_parts, client):
    a = [client.submit(s, i) for i in range(n_parts)]
    b = tree_reduce(a)
    c = client.compute(b, sync=True)

    assert sum(range(n_parts)) == c


@pytest.mark.parametrize("n_parts", [1, 2, 10, 15])
def test_reduce_futures(n_parts, client):
    def s(x):
        return x

    a = [client.submit(s, i) for i in range(n_parts)]
    b = reduce(a, sum)
    c = client.compute(b, sync=True)

    # Testing this gets the correct result for now.
    assert sum(range(n_parts)) == c
