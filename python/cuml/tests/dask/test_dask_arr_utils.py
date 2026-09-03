# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import cudf
import cupy as cp
import cupyx
import dask
import dask_cudf
import pytest

from cuml.dask.common.dask_arr_utils import validate_dask_array
from cuml.dask.common.part_utils import _extract_partitions
from cuml.testing.utils import array_equal


@pytest.mark.parametrize(
    "input_type",
    [
        "dask_array",
        "dask_dataframe",
        "dataframe",
        "scipysparse",
        "cupysparse",
        "numpy",
        "cupy",
    ],
)
@pytest.mark.parametrize("nrows", [1000])
@pytest.mark.parametrize("ncols", [10])
def test_to_sparse_dask_array(input_type, nrows, ncols, client):
    from cuml.dask.common import to_sparse_dask_array

    c = client

    a = cupyx.scipy.sparse.random(nrows, ncols, format="csr", dtype=cp.float32)
    if input_type == "dask_dataframe":
        pytest.xfail(
            reason="Dask nightlies break task fusing for this, "
            "issue https://github.com/NVIDIA/cuml/issues/6169"
        )
        df = cudf.DataFrame(a.todense())
        inp = dask_cudf.from_cudf(df, npartitions=2)
    elif input_type == "dask_array":
        inp = dask.array.from_array(a.todense().get())
    elif input_type == "dataframe":
        inp = cudf.DataFrame(a.todense())
    elif input_type == "scipysparse":
        inp = a.get()
    elif input_type == "cupysparse":
        inp = a
    elif input_type == "numpy":
        inp = a.get().todense()
    elif input_type == "cupy":
        inp = a.todense()

    arr = to_sparse_dask_array(inp, c)
    arr.compute_chunk_sizes()

    assert arr.shape == (nrows, ncols)

    # We can't call compute directly on this array yet when it has
    # multiple partitions yet so we will manually concat any
    # potential pieces.
    parts = c.sync(_extract_partitions, arr)
    local_parts = cp.vstack(
        [part[1].result().todense() for part in parts]
    ).get()

    assert array_equal(a.todense().get(), local_parts)


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [24])
@pytest.mark.parametrize("ncols", [1, 4, 8])
@pytest.mark.parametrize("n_parts", [2, 12])
@pytest.mark.parametrize("col_chunking", [True, False])
@pytest.mark.parametrize("n_col_chunks", [2, 4])
def test_validate_dask_array(
    nrows, ncols, n_parts, col_chunking, n_col_chunks, client
):
    if ncols > 1:
        X = cp.random.standard_normal((nrows, ncols))
        X = dask.array.from_array(X, chunks=(nrows / n_parts, -1))
        if col_chunking:
            X = X.rechunk((nrows / n_parts, ncols / n_col_chunks))
    else:
        X = cp.random.standard_normal(nrows)
        X = dask.array.from_array(X, chunks=(nrows / n_parts))

    if col_chunking and ncols > 1:
        with pytest.raises(Exception):
            validate_dask_array(X, client)
    else:
        validate_dask_array(X, client)
        assert True
