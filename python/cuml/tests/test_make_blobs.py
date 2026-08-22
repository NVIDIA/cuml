#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
import numpy as np
import pytest

import cuml

# Testing parameters for scalar parameter tests

dtype = ["single", "double"]

n_samples = [100, 1000]

n_features = [2, 10, 100]

centers = [
    None,
    2,
    5,
]

cluster_std = [0.01, 0.1]

center_box = [
    (-10.0, 10.0),
    [-20.0, 20.0],
]

shuffle = [True, False]


random_state = [None, 9]


@pytest.mark.parametrize("dtype", dtype)
@pytest.mark.parametrize("n_samples", n_samples)
@pytest.mark.parametrize("n_features", n_features)
@pytest.mark.parametrize("centers", centers)
@pytest.mark.parametrize("cluster_std", cluster_std)
@pytest.mark.parametrize("center_box", center_box)
@pytest.mark.parametrize("shuffle", shuffle)
@pytest.mark.parametrize("random_state", random_state)
@pytest.mark.parametrize("order", ["F", "C"])
def test_make_blobs_scalar_parameters(
    dtype,
    n_samples,
    n_features,
    centers,
    cluster_std,
    center_box,
    shuffle,
    random_state,
    order,
):
    out, labels = cuml.make_blobs(
        dtype=dtype,
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=0.001,
        center_box=center_box,
        shuffle=shuffle,
        random_state=random_state,
        order=order,
    )

    assert out.shape == (n_samples, n_features), "out shape mismatch"
    assert labels.shape == (n_samples,), "labels shape mismatch"

    if order == "F":
        assert out.flags["F_CONTIGUOUS"]
    elif order == "C":
        assert out.flags["C_CONTIGUOUS"]

    if centers is None:
        assert cp.unique(labels).shape == (3,), "unexpected number of clusters"
    elif centers <= n_samples:
        assert cp.unique(labels).shape == (centers,), (
            "unexpected number of clusters"
        )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("order", ["F", "C"])
def test_make_blobs_native_reproducible(dtype, order):
    kw = {
        "n_samples": 128,
        "n_features": 4,
        "centers": 4,
        "cluster_std": 0.3,
        "random_state": 1234,
        "order": order,
        "dtype": dtype,
    }

    x1, y1 = cuml.make_blobs(**kw)
    x2, y2 = cuml.make_blobs(**kw)

    cp.testing.assert_array_equal(x1, x2)
    cp.testing.assert_array_equal(y1, y2)

    assert x1.dtype == cp.dtype(dtype)
    assert y1.dtype == cp.dtype(dtype)
    assert x1.flags[f"{order}_CONTIGUOUS"]
    assert cp.unique(y1).shape == (4,)


def test_make_blobs_matches_native_bridge():
    from cuml.datasets._blobs import make_blobs as raft_blobs

    x1, y1 = cuml.make_blobs(
        n_samples=96,
        n_features=3,
        centers=3,
        cluster_std=0.25,
        center_box=(-2.0, 5.0),
        random_state=2026,
        order="C",
        dtype="float32",
    )

    x2, y2 = raft_blobs(
        n_samples=96,
        n_features=3,
        n_centers=3,
        centers=None,
        cluster_std=0.25,
        center_box_min=-2.0,
        center_box_max=5.0,
        shuffle=True,
        random_state=2026,
        order="C",
        dtype=cp.dtype("float32"),
    )

    cp.testing.assert_array_equal(x1, x2)
    cp.testing.assert_array_equal(y1, y2)


@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("xp", [cp, np])
def test_make_blobs_explicit_centers(order, xp):
    ctr = xp.asarray(
        [[-2.0, -2.0], [2.0, 2.0]],
        dtype=xp.float64,
        order="C",
    )

    out, labels, got = cuml.make_blobs(
        n_samples=96,
        n_features=2,
        centers=ctr,
        cluster_std=0.1,
        random_state=7,
        return_centers=True,
        order=order,
        dtype="float32",
    )

    assert got is ctr
    assert got.dtype == xp.float64
    assert got.flags["C_CONTIGUOUS"]
    assert out.flags[f"{order}_CONTIGUOUS"]
    assert cp.unique(labels).shape == (2,)


def test_make_blobs_empty_centers():
    ctr = cp.empty((0, 2), dtype=cp.float32)

    with pytest.raises(ValueError, match="at least one center"):
        cuml.make_blobs(
            n_samples=8,
            n_features=2,
            centers=ctr,
        )


@pytest.mark.parametrize("order", ["F", "C"])
def test_make_blobs_generated_return_centers(order):
    from cuml.datasets._blobs import make_blobs as raft_blobs

    x1, y1, ctr = cuml.make_blobs(
        n_samples=96,
        n_features=2,
        centers=2,
        cluster_std=0.1,
        center_box=(-4.0, 4.0),
        random_state=7,
        return_centers=True,
        order=order,
        dtype="float32",
    )

    x2, y2 = raft_blobs(
        n_samples=96,
        n_features=2,
        n_centers=2,
        centers=ctr,
        cluster_std=0.1,
        center_box_min=0.0,
        center_box_max=0.0,
        shuffle=True,
        random_state=7,
        order=order,
        dtype=cp.dtype("float32"),
    )

    assert ctr.shape == (2, 2)
    assert ctr.flags[f"{order}_CONTIGUOUS"]

    cp.testing.assert_array_equal(x1, x2)
    cp.testing.assert_array_equal(y1, y2)


def test_make_blobs_shuffle_false_keeps_block_labels():
    ctr = cp.asarray(
        [[-2.0], [2.0]],
        dtype=cp.float32,
    )

    out, labels = cuml.make_blobs(
        n_samples=6,
        n_features=1,
        centers=ctr,
        cluster_std=0.0,
        shuffle=False,
        random_state=7,
        order="C",
        dtype="float32",
    )

    exp_x = cp.asarray(
        [[-2.0], [-2.0], [-2.0], [2.0], [2.0], [2.0]],
        dtype=cp.float32,
    )
    exp_y = cp.asarray(
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        dtype=cp.float32,
    )

    cp.testing.assert_array_equal(out, exp_x)
    cp.testing.assert_array_equal(labels, exp_y)


def test_make_blobs_cluster_std_sequence_compatibility():
    ctr = cp.asarray(
        [[-2.0, -2.0], [2.0, 2.0]],
        dtype=cp.float32,
    )

    out, labels = cuml.make_blobs(
        n_samples=64,
        n_features=2,
        centers=ctr,
        cluster_std=[0.0, 0.0],
        shuffle=True,
        random_state=9,
        dtype="float32",
    )

    exp = ctr[labels.astype(cp.int64)]
    cp.testing.assert_array_equal(out, exp)


@pytest.mark.parametrize(
    ("n_samples", "n_features", "x_shape"),
    [
        (0, 2, (0, 2)),
        (8, 0, (8, 0)),
    ],
)
def test_make_blobs_zero_size_compatibility(n_samples, n_features, x_shape):
    x, y = cuml.make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        random_state=7,
    )

    assert x.shape == x_shape
    assert y.shape == (n_samples,)


def test_make_blobs_random_state_compatibility():
    kw = {
        "n_samples": 64,
        "n_features": 3,
        "centers": 3,
        "cluster_std": 0.2,
        "dtype": "float32",
    }

    x1, y1 = cuml.make_blobs(
        **kw,
        random_state=cp.random.RandomState(17),
    )
    x2, y2 = cuml.make_blobs(
        **kw,
        random_state=cp.random.RandomState(17),
    )

    cp.testing.assert_array_equal(x1, x2)
    cp.testing.assert_array_equal(y1, y2)
