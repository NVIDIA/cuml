#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import numbers
from collections.abc import Iterable
from random import getrandbits

import cupy as cp
import numpy as np

import cuml.internals
import cuml.internals.nvtx as nvtx
from cuml.datasets.utils import _create_rs_generator


# - Figures out how many centers this call needs.
# - Makes random centers when none were handed in.
# - Checks that sample counts and center shapes line up.
# - Leaves fixed centers alone when they already fit.
# - Hands back the centers plus their count.
def _get_centers(rs, centers, center_box, n_samples, n_features, dtype):
    if isinstance(n_samples, numbers.Integral):
        # Set n_centers by looking at centers arg
        if centers is None:
            centers = 3

        if isinstance(centers, numbers.Integral):
            n_centers = centers
            centers = rs.uniform(
                center_box[0],
                center_box[1],
                size=(n_centers, n_features),
                dtype=dtype,
            )

        else:
            if n_features != centers.shape[1]:
                raise ValueError(
                    "Expected `n_features` to be equal to"
                    " the length of axis 1 of centers array"
                )
            n_centers = centers.shape[0]

    else:
        # Set n_centers by looking at [n_samples] arg
        n_centers = len(n_samples)
        if centers is None:
            centers = rs.uniform(
                center_box[0],
                center_box[1],
                size=(n_centers, n_features),
                dtype=dtype,
            )
        try:
            assert len(centers) == n_centers
        except TypeError:
            raise ValueError(
                "Parameter `centers` must be array-like. "
                "Got {!r} instead".format(centers)
            )
        except AssertionError:
            raise ValueError(
                "Length of `n_samples` not consistent"
                " with number of centers. Got n_samples = {} "
                "and centers = {}".format(n_samples, centers)
            )
        else:
            if n_features != centers.shape[1]:
                raise ValueError(
                    "Expected `n_features` to be equal to"
                    " the length of axis 1 of centers array"
                )

    return centers, n_centers


# - Checks the smaller set of inputs RAFT can handle here.
# - Sorts out generated centers versus centers the caller passed in.
# - Keeps center memory order lined up with the output array.
# - Turns random_state into the uint64 seed RAFT expects.
# - Calls the Cython bridge and gives the same Python-style result back.
def _make_blobs_raft(
    n_samples,
    n_features,
    centers,
    cluster_std,
    center_box,
    shuffle,
    random_state,
    return_centers,
    order,
    dtype,
):
    if not isinstance(n_samples, numbers.Integral):
        raise ValueError(
            "`n_samples` must be an integer when `use_raft=True`."
        )
    if not isinstance(n_features, numbers.Integral):
        raise ValueError(
            "`n_features` must be an integer when `use_raft=True`."
        )

    n_samples, n_features = int(n_samples), int(n_features)
    if n_samples <= 0:
        raise ValueError("`n_samples` must be greater than 0.")
    if n_features <= 0:
        raise ValueError("`n_features` must be greater than 0.")

    if not isinstance(cluster_std, numbers.Real):
        raise ValueError(
            "`cluster_std` must be a scalar when `use_raft=True`."
        )
    if cluster_std < 0:
        raise ValueError("`cluster_std` must be non-negative.")

    dtype = cp.dtype(dtype)
    if dtype not in (cp.dtype("float32"), cp.dtype("float64")):
        raise ValueError(
            "RAFT make_blobs only supports float32 and float64 output."
        )
    if order not in ("C", "F"):
        raise ValueError("`order` must be either 'C' or 'F'.")

    made_ctr = False
    if centers is None:
        n_ctr, r_ctr, made_ctr = 3, None, True
    elif isinstance(centers, numbers.Integral):
        n_ctr, r_ctr, made_ctr = int(centers), None, True
        if n_ctr <= 0:
            raise ValueError("`centers` must be greater than 0.")
    else:
        r_ctr = cp.asarray(centers, dtype=dtype, order=order)
        if r_ctr.ndim != 2:
            raise ValueError(
                "`centers` must be a 2D array when `use_raft=True`."
            )
        if r_ctr.shape[1] != n_features:
            raise ValueError(
                "Expected `n_features` to be equal to"
                " the length of axis 1 of centers array"
            )

        n_ctr = int(r_ctr.shape[0])
        if n_ctr <= 0:
            raise ValueError("`centers` must contain at least one center.")

        # RAFT uses the same layout flag for X and centers, so keep them paired.
        # cp.asarray can keep an input layout in a few CUDA-array cases.
        if order == "C" and not r_ctr.flags["C_CONTIGUOUS"]:
            r_ctr = cp.ascontiguousarray(r_ctr)
        elif order == "F" and not r_ctr.flags["F_CONTIGUOUS"]:
            r_ctr = cp.asfortranarray(r_ctr)

    if made_ctr:
        try:
            box_lo, box_hi = center_box
        except (TypeError, ValueError):
            raise ValueError("`center_box` must contain exactly two values.")

        if not isinstance(box_lo, numbers.Real) or not isinstance(
            box_hi, numbers.Real
        ):
            raise ValueError("`center_box` values must be real numbers.")
        if box_lo > box_hi:
            raise ValueError(
                "`center_box` minimum must not exceed its maximum."
            )
    else:
        # The native call ignores the box once actual center locations are given.
        box_lo = box_hi = 0.0

    if return_centers and made_ctr:
        raise ValueError(
            "`return_centers=True` with generated centers is not supported "
            "when `use_raft=True`; pass explicit center locations instead."
        )

    if random_state is None:
        seed = getrandbits(64)
    elif isinstance(random_state, numbers.Integral):
        seed = int(random_state)
        if not 0 <= seed <= (1 << 64) - 1:
            raise ValueError(
                "`random_state` must be between 0 and 2**64 - 1 "
                "when `use_raft=True`."
            )
    else:
        raise ValueError(
            "`random_state` must be an integer or None when `use_raft=True`."
        )

    # Keep this import here so the normal CuPy path does not need the native module.
    from cuml.datasets._blobs import make_blobs as raft_blobs

    X, y = raft_blobs(
        n_samples=n_samples,
        n_features=n_features,
        n_centers=n_ctr,
        centers=r_ctr,
        cluster_std=float(cluster_std),
        center_box_min=float(box_lo),
        center_box_max=float(box_hi),
        shuffle=bool(shuffle),
        random_state=seed,
        order=order,
        dtype=dtype,
    )

    if return_centers:
        return X, y, r_ctr
    return X, y


# - Keeps the existing CuPy generator as the default path.
# - Sends the call to RAFT only when use_raft=True.
# - Uses the same public arguments either way.
# - Keeps the current return shape and label dtype behavior.
# - Falls straight back into the original generator code otherwise.
@nvtx.annotate(message="datasets.make_blobs", domain="cuml_python")
@cuml.internals.mlfunc(array_arg=None)
def make_blobs(
    n_samples=100,
    n_features=2,
    centers=None,
    cluster_std=1.0,
    center_box=(-10.0, 10.0),
    shuffle=True,
    random_state=None,
    return_centers=False,
    order="F",
    dtype="float32",
    use_raft=False,
):
    """Generate isotropic Gaussian blobs for clustering.

    Parameters
    ----------
    n_samples : int or array-like, optional (default=100)
        If int, it is the total number of points equally divided among
        clusters.
        If array-like, each element of the sequence indicates
        the number of samples per cluster.
    n_features : int, optional (default=2)
        The number of features for each sample.
    centers : int or array of shape [`n_centers`, `n_features`], optional
        (default=None)
        The number of centers to generate, or the fixed center locations.
        If `n_samples` is an int and centers is None, 3 centers are generated.
        If `n_samples` is array-like, centers must be
        either None or an array of length equal to the length of `n_samples`.
    cluster_std : float or sequence of floats, optional (default=1.0)
        The standard deviation of the clusters.
    center_box : pair of floats (min, max), optional (default=(-10.0, 10.0))
        The bounding box for each cluster center when centers are
        generated at random.
    shuffle : boolean, optional (default=True)
        Shuffle the samples.
    random_state : int, RandomState instance, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
    return_centers : bool, optional (default=False)
        If True, then return the centers of each cluster
    order: str, optional (default='F')
        The order of the generated samples
    dtype : str, optional (default='float32')
        Dtype of the generated samples
    use_raft : bool, optional (default=False)
        If True, send generation through RAFT's C++ ``make_blobs`` path.
        False keeps the existing CuPy path. RAFT currently expects integer
        ``n_samples``, scalar ``cluster_std``, and an integer or None
        ``random_state``; generated centers cannot be returned.

    Returns
    -------
    X : device array of shape [n_samples, n_features]
        The generated samples.
    y : device array of shape [n_samples]
        The integer labels for cluster membership of each sample.
    centers : device array, shape [n_centers, n_features]
        The centers of each cluster. Only returned if
        ``return_centers=True``.

    Examples
    --------

    .. code-block:: python

        >>> from sklearn.datasets import make_blobs
        >>> X, y = make_blobs(n_samples=10, centers=3, n_features=2,
        ...                   random_state=0)
        >>> print(X.shape)
        (10, 2)
        >>> y
        array([0, 0, 1, 0, 2, 2, 2, 1, 1, 0])
        >>> X, y = make_blobs(n_samples=[3, 3, 4], centers=None, n_features=2,
        ...                   random_state=0)
        >>> print(X.shape)
        (10, 2)
        >>> y
        array([0, 1, 2, 0, 2, 2, 2, 1, 1, 0])

    See also
    --------
    make_classification: a more intricate variant
    """
    if use_raft:
        return _make_blobs_raft(
            n_samples=n_samples,
            n_features=n_features,
            centers=centers,
            cluster_std=cluster_std,
            center_box=center_box,
            shuffle=shuffle,
            random_state=random_state,
            return_centers=return_centers,
            order=order,
            dtype=dtype,
        )

    generator = _create_rs_generator(random_state=random_state)

    centers, n_centers = _get_centers(
        generator, centers, center_box, n_samples, n_features, dtype
    )

    # stds: if cluster_std is given as list, it must be consistent
    # with the n_centers
    if hasattr(cluster_std, "__len__") and len(cluster_std) != n_centers:
        raise ValueError(
            "Length of `clusters_std` not consistent with "
            "number of centers. Got centers = {} "
            "and cluster_std = {}".format(centers, cluster_std)
        )

    if isinstance(cluster_std, numbers.Real):
        cluster_std = cp.full(len(centers), cluster_std)

    if isinstance(n_samples, Iterable):
        n_samples_per_center = n_samples
    else:
        n_samples_per_center = [int(n_samples // n_centers)] * n_centers

        for i in range(n_samples % n_centers):
            n_samples_per_center[i] += 1

    X = cp.zeros(n_samples * n_features, dtype=dtype)
    X = X.reshape((n_samples, n_features), order=order)
    y = cp.zeros(n_samples, dtype=dtype)

    if shuffle:
        proba_samples_per_center = np.array(n_samples_per_center) / np.sum(
            n_samples_per_center
        )
        shuffled_sample_indices = generator.choice(
            n_centers, n_samples, replace=True, p=proba_samples_per_center
        )
        for i, (n, std) in enumerate(zip(n_samples_per_center, cluster_std)):
            center_indices = cp.where(shuffled_sample_indices == i)

            y[center_indices[0]] = i

            X_k = generator.normal(
                scale=std,
                size=(len(center_indices[0]), n_features),
                dtype=dtype,
            )

            # NOTE: Adding the loc explicitly as cupy has a bug
            # when calling generator.normal with an array for loc.
            # cupy.random.normal, however, works with the same
            # arguments
            cp.add(X_k, centers[i], out=X_k)
            X[center_indices[0], :] = X_k
    else:
        stop = 0
        for i, (n, std) in enumerate(zip(n_samples_per_center, cluster_std)):
            start, stop = stop, stop + n_samples_per_center[i]

            y[start:stop] = i

            X_k = generator.normal(
                scale=std, size=(n, n_features), dtype=dtype
            )

            cp.add(X_k, centers[i], out=X_k)
            X[start:stop, :] = X_k

    if return_centers:
        return X, y, centers
    else:
        return X, y
