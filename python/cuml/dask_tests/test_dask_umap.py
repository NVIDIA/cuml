# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import math

import cupy as cp
import numpy as np
import pytest
from sklearn.datasets import load_digits, load_iris

from cuml.internals import logger
from cuml.metrics import trustworthiness


def _load_dataset(dataset, n_rows):
    if dataset == "digits":
        local_X, local_y = load_digits(return_X_y=True)

    else:  # dataset == "iris"
        local_X, local_y = load_iris(return_X_y=True)

    local_X = cp.asarray(local_X)
    local_y = cp.asarray(local_y)

    local_X = local_X.repeat(math.ceil(n_rows / len(local_X)), axis=0)
    local_y = local_y.repeat(math.ceil(n_rows / len(local_y)), axis=0)

    # Add some gaussian noise
    local_X += cp.random.standard_normal(local_X.shape, dtype=cp.float32)

    return local_X, local_y


def _local_umap_trustworthiness(local_X, local_y, n_neighbors, supervised):
    """
    Train model on all data, report trustworthiness
    """
    from cuml.manifold import UMAP

    local_model = UMAP(n_neighbors=n_neighbors, random_state=42, init="random")
    y_train = None
    if supervised:
        y_train = local_y
    local_model.fit(local_X, y=y_train)
    embedding = local_model.transform(local_X)
    return trustworthiness(
        local_X, embedding, n_neighbors=n_neighbors, batch_size=5000
    )


def _umap_mnmg_trustworthiness(
    local_X, local_y, n_neighbors, supervised, n_parts, sampling_ratio
):
    """
    Train model on random sample of data, transform in
    parallel, report trustworthiness
    """
    import dask.array as da

    from cuml.dask.manifold import UMAP as MNMG_UMAP
    from cuml.manifold import UMAP

    local_model = UMAP(n_neighbors=n_neighbors, random_state=42, init="random")

    n_samples = local_X.shape[0]
    n_samples_per_part = math.ceil(n_samples / n_parts)

    selection = np.random.RandomState(42).choice(
        [True, False],
        n_samples,
        replace=True,
        p=[sampling_ratio, 1.0 - sampling_ratio],
    )
    X_train = local_X[selection]
    X_transform = local_X
    X_transform_d = da.from_array(X_transform, chunks=(n_samples_per_part, -1))

    y_train = None
    if supervised:
        y_train = local_y[selection]

    local_model.fit(X_train, y=y_train)

    distributed_model = MNMG_UMAP(model=local_model)
    embedding = distributed_model.transform(X_transform_d)

    embedding = embedding.compute()
    return trustworthiness(
        X_transform, embedding, n_neighbors=n_neighbors, batch_size=5000
    )


def _run_mnmg_test(
    n_parts, n_rows, sampling_ratio, supervised, dataset, n_neighbors, client
):
    local_X, local_y = _load_dataset(dataset, n_rows)

    dist_umap = _umap_mnmg_trustworthiness(
        local_X, local_y, n_neighbors, supervised, n_parts, sampling_ratio
    )

    loc_umap = _local_umap_trustworthiness(
        local_X, local_y, n_neighbors, supervised
    )

    logger.debug(
        "\nLocal UMAP trustworthiness score : {:.2f}".format(loc_umap)
    )
    logger.debug("UMAP MNMG trustworthiness score : {:.2f}".format(dist_umap))

    trust_diff = loc_umap - dist_umap

    return trust_diff <= 0.15


@pytest.mark.mg
@pytest.mark.parametrize("n_parts", [2, 9])
@pytest.mark.parametrize("n_rows", [100, 500])
@pytest.mark.parametrize("sampling_ratio", [0.55, 0.9])
@pytest.mark.parametrize("supervised", [True, False])
@pytest.mark.parametrize("dataset", ["digits", "iris"])
@pytest.mark.parametrize("n_neighbors", [10])
def test_umap_mnmg(
    n_parts, n_rows, sampling_ratio, supervised, dataset, n_neighbors, client
):
    result = _run_mnmg_test(
        n_parts,
        n_rows,
        sampling_ratio,
        supervised,
        dataset,
        n_neighbors,
        client,
    )

    if not result:
        result = _run_mnmg_test(
            n_parts,
            n_rows,
            sampling_ratio,
            supervised,
            dataset,
            n_neighbors,
            client,
        )

    assert result
