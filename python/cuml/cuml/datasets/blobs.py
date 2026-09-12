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
    n_samples = int(n_samples)
    n_features = int(n_features)
    dt = cp.dtype(dtype)

    if n_samples <= 0 or n_features <= 0:
        raise ValueError("`n_samples` and `n_features` must be positive.")

    if cluster_std < 0:
        raise ValueError("`cluster_std` must be non-negative.")

    gen_ctr = centers is None or isinstance(centers, numbers.Integral)

    if centers is None:
        n_ctr, ctr = 3, None
    elif isinstance(centers, numbers.Integral):
        n_ctr, ctr = int(centers), None
        if n_ctr <= 0:
            raise ValueError("`centers` must be greater than 0.")
    else:
        ctr = cp.asarray(centers, dtype=dt, order=order)

        if ctr.ndim != 2:
            raise ValueError("`centers` must be a 2D array.")
        if ctr.shape[1] != n_features:
            raise ValueError(
                "Expected `n_features` to be equal to"
                " the length of axis 1 of centers array"
            )

        n_ctr = ctr.shape[0]
        if n_ctr == 0:
            raise ValueError("`centers` must contain at least one center.")

    if gen_ctr:
        try:
            lo, hi = center_box
        except (TypeError, ValueError):
            raise ValueError(
                "`center_box` must contain exactly two values."
            ) from None

        if lo > hi:
            raise ValueError(
                "`center_box` minimum must not exceed its maximum."
            )
    else:
        lo = hi = 0.0

    if return_centers and gen_ctr:
        rs = _create_rs_generator(random_state=random_state)
        ctr, n_ctr = _get_centers(
            rs,
            centers,
            center_box,
            n_samples,
            n_features,
            dt,
        )

    if ctr is not None:
        ctr = cp.asarray(ctr, dtype=dt, order=order)

        if order == "C" and not ctr.flags["C_CONTIGUOUS"]:
            ctr = cp.ascontiguousarray(ctr)
        elif order == "F" and not ctr.flags["F_CONTIGUOUS"]:
            ctr = cp.asfortranarray(ctr)

    if random_state is None:
        seed = getrandbits(64)
    else:
        seed = int(random_state)

        if not 0 <= seed <= (1 << 64) - 1:
            raise ValueError("`random_state` must be between 0 and 2**64 - 1.")

    from cuml.datasets._blobs import make_blobs as cpp_blobs

    X, y = cpp_blobs(
        n_samples=n_samples,
        n_features=n_features,
        n_centers=n_ctr,
        centers=ctr,
        cluster_std=float(cluster_std),
        center_box_min=float(lo),
        center_box_max=float(hi),
        shuffle=bool(shuffle),
        random_state=seed,
        order=order,
        dtype=dt,
    )

    if return_centers:
        return X, y, ctr if gen_ctr else centers

    return X, y


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
    dt = cp.dtype(dtype)

    use_cpp = (
        isinstance(n_samples, numbers.Integral)
        and isinstance(n_features, numbers.Integral)
        and n_samples > 0
        and n_features > 0
        and isinstance(cluster_std, numbers.Real)
        and isinstance(random_state, (type(None), int))
        and shuffle is True
        and order in ("C", "F")
        and dt in (cp.dtype("float32"), cp.dtype("float64"))
    )

    if use_cpp:
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
            dtype=dt,
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
