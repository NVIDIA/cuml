# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import pytest
from dask.distributed import wait

from cuml.dask.common import raise_exception_from_futures


def _raise_exception():
    raise ValueError("intentional exception")


def test_dask_exceptions(client):
    fut = client.submit(_raise_exception)
    wait(fut)

    with pytest.raises(RuntimeError):
        raise_exception_from_futures([fut])
