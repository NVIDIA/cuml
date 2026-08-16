#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
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


# - Runs the existing CuPy path across its scalar parameter combinations.
# - Checks the output and label shapes.
# - Checks both C and F output layouts.
# - Checks the expected number of clusters.
# - Keeps the original make_blobs coverage in place.
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


# - Checks that leaving RAFT off still means the old CuPy path.
# - Uses a fixed seed so both calls should line up exactly.
# - Compares the samples instead of just checking their shapes.
# - Compares the labels too.
# - Keeps this as a small compatibility check.
def test_make_blobs_raft_disabled_matches_default():
    kw = {
        "n_samples": 64,
        "n_features": 3,
        "centers": 4,
        "cluster_std": 0.2,
        "random_state": 17,
    }

    out_a, labels_a = cuml.make_blobs(**kw)
    out_b, labels_b = cuml.make_blobs(**kw, use_raft=False)

    cp.testing.assert_array_equal(out_a, out_b)
    cp.testing.assert_array_equal(labels_a, labels_b)


# - Runs the RAFT path twice with the same seed.
# - Covers both supported floating dtypes.
# - Covers both C and F output layouts.
# - Checks the existing floating label dtype stays the same.
# - Makes sure the requested four clusters are present.
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("order", ["F", "C"])
def test_make_blobs_raft_reproducible(dtype, order):
    kw = {
        "n_samples": 128,
        "n_features": 4,
        "centers": 4,
        "cluster_std": 0.3,
        "random_state": 1234,
        "order": order,
        "dtype": dtype,
        "use_raft": True,
    }

    out_a, labels_a = cuml.make_blobs(**kw)
    out_b, labels_b = cuml.make_blobs(**kw)

    cp.testing.assert_array_equal(out_a, out_b)
    cp.testing.assert_array_equal(labels_a, labels_b)

    assert out_a.dtype == cp.dtype(dtype)
    assert labels_a.dtype == cp.dtype(dtype)
    assert out_a.flags[f"{order}_CONTIGUOUS"]
    assert cp.unique(labels_a).shape == (4,)


# - Calls the public RAFT option and the private native bridge.
# - Gives both calls the exact same inputs.
# - Checks that the Python wrapper forwards the seed correctly.
# - Checks that generated samples match exactly.
# - Checks that the labels match exactly too.
def test_make_blobs_raft_matches_native_bridge():
    from cuml.datasets._blobs import make_blobs as raft_blobs

    out_a, labels_a = cuml.make_blobs(
        n_samples=96,
        n_features=3,
        centers=3,
        cluster_std=0.25,
        center_box=(-2.0, 5.0),
        random_state=2026,
        order="C",
        dtype="float32",
        use_raft=True,
    )

    out_b, labels_b = raft_blobs(
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

    cp.testing.assert_array_equal(out_a, out_b)
    cp.testing.assert_array_equal(labels_a, labels_b)


# - Passes fixed centers instead of asking RAFT to make them.
# - Checks both output layouts with the same center values.
# - Asks for the centers back through the public API.
# - Makes sure those returned centers are unchanged.
# - Confirms both clusters show up in the labels.
@pytest.mark.parametrize("order", ["F", "C"])
def test_make_blobs_raft_explicit_centers(order):
    fixed = cp.asarray(
        [[-2.0, -2.0], [2.0, 2.0]],
        dtype=cp.float32,
        order="C",
    )

    out, labels, got = cuml.make_blobs(
        n_samples=96,
        n_features=2,
        centers=fixed,
        cluster_std=0.1,
        random_state=7,
        return_centers=True,
        order=order,
        dtype="float32",
        use_raft=True,
    )

    cp.testing.assert_array_equal(got, fixed)
    assert out.flags[f"{order}_CONTIGUOUS"]
    assert got.flags[f"{order}_CONTIGUOUS"]
    assert cp.unique(labels).shape == (2,)


# - Keeps the RAFT-only input limits covered in one small test.
# - Checks list sample counts are rejected.
# - Checks per-center standard deviations are rejected.
# - Checks unsupported order and dtype values are rejected.
# - Checks generated centers cannot be returned yet.
@pytest.mark.parametrize(
    ("extra", "msg"),
    [
        ({"n_samples": [10, 10]}, "n_samples"),
        ({"cluster_std": [0.1, 0.2]}, "cluster_std"),
        ({"order": "A"}, "order"),
        ({"dtype": "float16"}, "float32 and float64"),
        ({"return_centers": True}, "return_centers"),
    ],
)
def test_make_blobs_raft_rejects_unsupported_inputs(extra, msg):
    kw = {
        "n_samples": 32,
        "n_features": 2,
        "centers": 2,
        "random_state": 3,
        "use_raft": True,
    }
    kw.update(extra)

    with pytest.raises(ValueError, match=msg):
        cuml.make_blobs(**kw)
