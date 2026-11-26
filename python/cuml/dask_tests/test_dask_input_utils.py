#
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
import dask.array as da
import pytest

from cuml.dask.common.dask_arr_utils import to_dask_cudf
from cuml.dask.common.input_utils import DistributedDataHandler
from cuml.dask.datasets.blobs import make_blobs


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [1e4])
@pytest.mark.parametrize("ncols", [10])
@pytest.mark.parametrize("n_parts", [2, 23])
@pytest.mark.parametrize("input_type", ["dataframe", "array", "series"])
@pytest.mark.parametrize("colocated", [True, False])
def test_extract_partitions_worker_list(
    nrows, ncols, n_parts, input_type, colocated, client
):
    adj_input_type = "dataframe" if input_type == "series" else input_type

    X_arr, y_arr = make_blobs(
        n_samples=int(nrows), n_features=ncols, n_parts=n_parts
    )

    if adj_input_type == "dataframe" or input_type == "dataframe":
        X = to_dask_cudf(X_arr)
        y = to_dask_cudf(y_arr)
    elif input_type == "array":
        X, y = X_arr, y_arr

    if input_type == "series":
        X = X[X.columns[0]]

    if colocated:
        ddh = DistributedDataHandler.create((X, y), client)
    else:
        ddh = DistributedDataHandler.create(X, client)

    parts = list(map(lambda x: x[1], ddh.gpu_futures))
    assert len(parts) == n_parts


@pytest.mark.skip(reason="Segfault with pathological parameters (issue #7452)")
@pytest.mark.mg
@pytest.mark.parametrize("nrows", [24])
@pytest.mark.parametrize("ncols", [2])
@pytest.mark.parametrize("n_parts", [2, 23])
@pytest.mark.parametrize("input_type", ["dataframe", "array", "series"])
@pytest.mark.parametrize("colocated", [True, False])
def test_extract_partitions_shape(
    nrows, ncols, n_parts, input_type, colocated, client
):
    adj_input_type = "dataframe" if input_type == "series" else input_type

    X_arr, y_arr = make_blobs(
        n_samples=nrows, n_features=ncols, n_parts=n_parts
    )

    if adj_input_type == "dataframe" or input_type == "dataframe":
        X = to_dask_cudf(X_arr)
        y = to_dask_cudf(y_arr)
    elif input_type == "array":
        X, y = X_arr, y_arr

    if input_type == "series":
        X = X[X.columns[0]]

    if input_type == "dataframe" or input_type == "series":
        X_len_parts = X.map_partitions(len).compute()
        y_len_parts = y.map_partitions(len).compute()
    elif input_type == "array":
        X_len_parts = X.chunks[0]
        y_len_parts = y.chunks[0]

    if colocated:
        ddh = DistributedDataHandler.create((X, y), client)
        parts = [part.result() for worker, part in ddh.gpu_futures]
        for i in range(len(parts)):
            assert (parts[i][0].shape[0] == X_len_parts[i]) and (
                parts[i][1].shape[0] == y_len_parts[i]
            )
    else:
        ddh = DistributedDataHandler.create(X, client)
        parts = [part.result() for worker, part in ddh.gpu_futures]
        for i in range(len(parts)):
            assert parts[i].shape[0] == X_len_parts[i]


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [24])
@pytest.mark.parametrize("ncols", [2])
@pytest.mark.parametrize("n_parts", [2, 12])
@pytest.mark.parametrize("X_delayed", [True, False])
@pytest.mark.parametrize("y_delayed", [True, False])
@pytest.mark.parametrize("colocated", [True, False])
def test_extract_partitions_futures(
    nrows, ncols, n_parts, X_delayed, y_delayed, colocated, client
):
    X = cp.random.standard_normal((nrows, ncols))
    y = cp.random.standard_normal((nrows,))

    X = da.from_array(X, chunks=(nrows / n_parts, -1))
    y = da.from_array(y, chunks=(nrows / n_parts,))

    if not X_delayed:
        X = client.persist(X)
    if not y_delayed:
        y = client.persist(y)

    if colocated:
        ddh = DistributedDataHandler.create((X, y), client)
    else:
        ddh = DistributedDataHandler.create(X, client)

    parts = list(map(lambda x: x[1], ddh.gpu_futures))
    assert len(parts) == n_parts
