# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import cupy as cp
import pytest

from cuml.testing.datasets import make_text_classification_dataset

pytest_plugins = [
    "cuml.testing.plugins.gpu_test_config",
    "cuml.testing.plugins.restart_corrupted_xdist_worker",
]

enable_tcp_over_ucx = True
enable_nvlink = False
enable_infiniband = False


@pytest.fixture(scope="session")
def sparse_text_dataset():
    """Return a sparse text-like dataset and a CuPy target vector."""
    X, y = make_text_classification_dataset()
    return X, cp.array(y)


@pytest.fixture(scope="module")
def cluster():
    from dask_cuda import LocalCUDACluster
    from dask_cuda.utils_test import IncreasedCloseTimeoutNanny

    cluster = LocalCUDACluster(
        protocol="tcp",
        scheduler_port=0,
        worker_class=IncreasedCloseTimeoutNanny,
    )
    yield cluster
    cluster.close()


@pytest.fixture(scope="function")
def client(cluster):
    from dask.distributed import Client

    client = Client(cluster)
    yield client
    client.close()


@pytest.fixture(scope="module")
def ucx_cluster():
    from dask_cuda import LocalCUDACluster
    from dask_cuda.utils_test import IncreasedCloseTimeoutNanny

    cluster = LocalCUDACluster(
        protocol="ucx",
        worker_class=IncreasedCloseTimeoutNanny,
    )
    yield cluster
    cluster.close()


@pytest.fixture(scope="function")
def ucx_client(ucx_cluster):
    pytest.importorskip("distributed_ucxx")
    from dask.distributed import Client

    client = Client(ucx_cluster)
    yield client
    client.close()


def pytest_addoption(parser):
    group = parser.getgroup("Dask cuML Custom Options")

    group.addoption(
        "--run_ucx",
        action="store_true",
        default=False,
        help="run _only_ UCXX tests",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run_ucx"):
        skip_others = pytest.mark.skip(
            reason="only runs when --run_ucx is not specified"
        )
        for item in items:
            if "ucx" not in item.keywords:
                item.add_marker(skip_others)
    else:
        skip_ucx = pytest.mark.skip(reason="requires --run_ucx to run")
        for item in items:
            if "ucx" in item.keywords:
                item.add_marker(skip_ucx)
