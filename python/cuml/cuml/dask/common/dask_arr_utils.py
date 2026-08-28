# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#


import cudf
import cupy as cp
import cupyx
import dask
import dask.dataframe as dd
import numpy as np
import scipy.sparse
from dask.distributed import default_client


def validate_dask_array(darray, client=None):
    if len(darray.chunks) > 2:
        raise ValueError("Input array cannot have more than two dimensions")
    elif len(darray.chunks) == 2 and len(darray.chunks[1]) > 1:
        raise ValueError("Input array cannot be chunked along axis 1")


def _conv_df_to_sparse(x):
    cupy_ary = cp.asarray(x.to_cupy(), dtype=x.dtypes[0])

    return cupyx.scipy.sparse.csr_matrix(cupy_ary)


def _conv_array_to_sparse(arr):
    """
    Converts an array (or cudf.DataFrame) to a sparse array
    :param arr: scipy or cupy sparse matrix, cudf DataFrame,
                dense numpy or cupy array
    :return: cupy sparse CSR matrix
    """
    if scipy.sparse.isspmatrix(arr):
        ret = cupyx.scipy.sparse.csr_matrix(arr.tocsr())
    elif cupyx.scipy.sparse.isspmatrix(arr):
        ret = arr
    elif isinstance(arr, cudf.DataFrame):
        ret = _conv_df_to_sparse(arr)
    elif isinstance(arr, np.ndarray):
        cupy_ary = cp.asarray(arr, dtype=arr.dtype)
        ret = cupyx.scipy.sparse.csr_matrix(cupy_ary)

    elif isinstance(arr, cp.ndarray):
        ret = cupyx.scipy.sparse.csr_matrix(arr)
    else:
        raise ValueError("Unexpected input type %s" % type(arr))
    return ret


def to_sparse_dask_array(cudf_or_array, client=None):
    """
    Converts an array or cuDF to a sparse Dask array backed by sparse CuPy.
    Csr matrices.

    Parameters
    ----------
    cudf_or_array : cuDF Dataframe, array-like sparse / dense array, or
                    Dask DataFrame/Array
    client : dask.distributed.Client (optional) Dask client

    dtype : output dtype

    Returns
    -------
    dask_array : dask.Array backed by cupyx.scipy.sparse.csr_matrix
    """
    ret = cudf_or_array
    shape = cudf_or_array.shape
    meta = cupyx.scipy.sparse.csr_matrix(cp.zeros(1))

    if isinstance(ret, dask.dataframe.DataFrame):
        ret = ret.to_dask_array()

    if isinstance(cudf_or_array, dask.array.Array):
        return cudf_or_array.map_blocks(_conv_array_to_sparse, meta=meta)

    else:
        ret = _conv_array_to_sparse(ret)

        # Push to worker
        final_result = client.scatter(ret)

        return dask.array.from_delayed(final_result, shape=shape, meta=meta)


def _get_meta(df):
    ret = df.iloc[:0]
    return ret


def _to_cudf(arr):
    if arr.ndim == 2:
        return cudf.DataFrame(arr)
    return cudf.Series(arr)


def to_dask_cudf(dask_arr, client=None):
    client = default_client() if client is None else client

    func = dask.delayed(_to_cudf)
    elms = [func(dp) for dp in dask_arr.to_delayed().flatten()]
    dfs = client.compute(elms)

    meta = client.submit(_get_meta, dfs[0])
    meta_local = meta.result()

    return dd.from_delayed(dfs, meta=meta_local)
