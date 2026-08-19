# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
import numpy as np
import pytest
from cupyx.scipy.sparse import coo_matrix as gpu_coo_matrix
from cupyx.scipy.sparse import csc_matrix as gpu_csc_matrix
from cupyx.scipy.sparse import csr_matrix as gpu_csr_matrix
from scipy.sparse import coo_matrix as cpu_coo_matrix
from scipy.sparse import csc_matrix as cpu_csc_matrix
from scipy.sparse import csr_matrix as cpu_csr_matrix

from cuml.datasets import make_blobs, make_classification
from cuml.internals.outputs import convert_arrays


def create_rand_clf(random_state):
    clf, _ = make_classification(
        n_samples=500,
        n_features=20,
        n_clusters_per_class=1,
        n_informative=12,
        n_classes=5,
        order="C",
        random_state=random_state,
    )
    return clf


def create_rand_blobs(random_state):
    blobs, _ = make_blobs(
        n_samples=500,
        n_features=20,
        centers=20,
        order="C",
        random_state=random_state,
    )
    return blobs


def create_rand_integers(random_state):
    cp.random.seed(random_state)
    randint = cp.random.randint(30, size=(500, 20)).astype(cp.float64)
    return randint


def create_positive_rand(random_state):
    cp.random.seed(random_state)
    rand = cp.random.rand(500, 20).astype(cp.float64)
    rand = cp.abs(rand) + 0.1
    return rand


def convert(dataset, output_type):
    converted_dataset = convert_arrays(dataset, output_type)
    if output_type == "cudf":
        renaming = {i: f"c{i}" for i in range(dataset.shape[1])}
        converted_dataset = converted_dataset.rename(columns=renaming)
        dataset = converted_dataset.to_pandas()
    else:
        dataset = cp.asnumpy(dataset)
    return dataset, converted_dataset


def sparsify_and_convert(dataset, conversion_format, sparsify_ratio=0.3):
    """Randomly set values to 0 and produce a sparse array.

    Parameters
    ----------
    dataset : array
        Input array to convert
    conversion_format : string
        Type of sparse array :
        - scipy-csr: SciPy CSR sparse array
        - scipy-csc: SciPy CSC sparse array
        - scipy-coo: SciPy COO sparse array
        - cupy-csr: CuPy CSR sparse array
        - cupy-csc: CuPy CSC sparse array
        - cupy-coo: CuPy COO sparse array
    sparsify_ratio: float [0-1]
        Ratio of zeros in the sparse array

    Returns
    -------
    SciPy CSR array and converted array
    """
    random_loc = cp.random.choice(
        dataset.size, int(dataset.size * sparsify_ratio), replace=False
    )
    dataset.ravel()[random_loc] = 0

    if conversion_format.startswith("scipy"):
        dataset = cp.asnumpy(dataset)

    if conversion_format == "scipy-csr":
        converted_dataset = cpu_csr_matrix(dataset)
    elif conversion_format == "scipy-csc":
        converted_dataset = cpu_csc_matrix(dataset)
    elif conversion_format == "scipy-coo":
        converted_dataset = cpu_coo_matrix(dataset)
    elif conversion_format == "cupy-csr":
        converted_dataset = gpu_csr_matrix(dataset)
    elif conversion_format == "cupy-csc":
        converted_dataset = gpu_csc_matrix(dataset)
    elif conversion_format == "cupy-coo":
        np_array = cp.asnumpy(dataset)
        np_coo_array = cpu_coo_matrix(np_array)
        converted_dataset = gpu_coo_matrix(np_coo_array)

    if conversion_format.startswith("cupy"):
        dataset = cp.asnumpy(dataset)

    return cpu_csr_matrix(dataset), converted_dataset


@pytest.fixture(scope="session", params=["numpy", "cupy", "cudf"])
def clf_dataset(request, random_seed):
    clf = create_rand_clf(random_seed)
    return convert(clf, request.param)


@pytest.fixture(scope="session", params=["numpy", "cupy", "cudf"])
def blobs_dataset(request, random_seed):
    blobs = create_rand_blobs(random_seed)
    return convert(blobs, request.param)


@pytest.fixture(scope="session", params=["numpy", "cupy", "cudf"])
def int_dataset(request, random_seed):
    randint = create_rand_integers(random_seed)
    cp.random.seed(random_seed)
    random_loc = cp.random.choice(
        randint.size, int(randint.size * 0.3), replace=False
    )

    zero_filled = randint.copy().ravel()
    zero_filled[random_loc] = 0
    zero_filled = zero_filled.reshape(randint.shape)
    zero_filled = convert(zero_filled, request.param)

    one_filled = randint.copy().ravel()
    one_filled[random_loc] = 1
    one_filled = one_filled.reshape(randint.shape)
    one_filled = convert(one_filled, request.param)

    nan_filled = randint.copy().ravel()
    nan_filled[random_loc] = cp.nan
    nan_filled = nan_filled.reshape(randint.shape)
    nan_filled = convert(nan_filled, request.param)

    return zero_filled, one_filled, nan_filled


@pytest.fixture(
    scope="session", params=["scipy-csr", "scipy-csc", "cupy-csr", "cupy-csc"]
)
def sparse_clf_dataset(request, random_seed):
    clf = create_rand_clf(random_seed)
    return sparsify_and_convert(clf, request.param)


@pytest.fixture(
    scope="session",
    params=[
        "scipy-csr",
        "scipy-csc",
        "scipy-coo",
        "cupy-csr",
        "cupy-csc",
        "cupy-coo",
    ],
)
def sparse_dataset_with_coo(request, random_seed):
    clf = create_rand_clf(random_seed)
    return sparsify_and_convert(clf, request.param)


@pytest.fixture(
    scope="session", params=["scipy-csr", "scipy-csc", "cupy-csr", "cupy-csc"]
)
def sparse_blobs_dataset(request, random_seed):
    blobs = create_rand_blobs(random_seed)
    return sparsify_and_convert(blobs, request.param)


@pytest.fixture(
    scope="session", params=["scipy-csr", "scipy-csc", "cupy-csr", "cupy-csc"]
)
def sparse_int_dataset(request, random_seed):
    randint = create_rand_integers(random_seed)
    return sparsify_and_convert(randint, request.param)


@pytest.fixture(
    scope="session",
    params=[
        ("scipy-csr", np.nan),
        ("scipy-csc", np.nan),
        ("cupy-csr", np.nan),
        ("cupy-csc", np.nan),
        ("scipy-csr", 1.0),
        ("scipy-csc", 1.0),
        ("cupy-csr", 1.0),
        ("cupy-csc", 1.0),
    ],
)
def sparse_imputer_dataset(request, random_seed):
    datatype, val = request.param
    randint = create_rand_integers(random_seed)
    random_loc = cp.random.choice(
        randint.size, int(randint.size * 0.3), replace=False
    )

    randint.ravel()[random_loc] = val
    X_sp, X = sparsify_and_convert(randint, datatype, sparsify_ratio=0.15)
    X_sp = X_sp.tocsc()
    return val, X_sp, X


@pytest.fixture(scope="session", params=["numpy", "cupy", "cudf"])
def nan_filled_positive(request, random_seed):
    rand = create_positive_rand(random_seed)
    cp.random.seed(random_seed)
    random_loc = cp.random.choice(
        rand.size, int(rand.size * 0.3), replace=False
    )
    rand.ravel()[random_loc] = cp.nan
    rand = convert(rand, request.param)
    return rand


@pytest.fixture(scope="session", params=["scipy-csc", "cupy-csc"])
def sparse_nan_filled_positive(request, random_seed):
    rand = create_positive_rand(random_seed)
    cp.random.seed(random_seed)
    random_loc = cp.random.choice(
        rand.size, int(rand.size * 0.3), replace=False
    )
    rand.ravel()[random_loc] = cp.nan
    return sparsify_and_convert(rand, request.param)


def assert_allclose(actual, desired, rtol=1e-05, atol=1e-05, ratio_tol=None):
    def as_numpy(x):
        if hasattr(x, "todense"):
            x = x.todense()
        elif hasattr(x, "to_numpy"):
            x = x.to_numpy()
        return cp.asnumpy(x)

    actual = as_numpy(actual)
    desired = as_numpy(desired)

    if ratio_tol:
        assert actual.shape == desired.shape
        diff_ratio = (actual != desired).sum() / actual.size
        assert diff_ratio <= ratio_tol
    else:
        return np.testing.assert_allclose(
            actual, desired, rtol=rtol, atol=atol
        )
