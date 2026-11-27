# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
import dask
import dask.array as da
import numpy as np
import pytest
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from scipy.sparse import csr_matrix as scipy_csr_matrix
from sklearn.feature_extraction.text import (
    TfidfTransformer as SkTfidfTransformer,
)

from cuml.dask.feature_extraction.text import TfidfTransformer


# Testing Util Functions
def generate_dask_array(np_array, n_parts):
    """
    Creates a dask array from a numpy 2d array
    """
    n_samples = np_array.shape[0]
    n_samples_per_part = int(n_samples / n_parts)
    chunks = [n_samples_per_part] * n_parts
    samples_last_row = n_samples - ((n_parts - 1) * n_samples_per_part)
    chunks[-1] = samples_last_row
    chunks = tuple(chunks)
    return da.from_array(np_array, chunks=(chunks, -1))


def create_cp_sparse_ar_from_dense_np_ar(ar, dtype=np.float32):
    """
    Creates a gpu array from a dense cpu array
    """
    return cp_csr_matrix(scipy_csr_matrix(ar), dtype=dtype)


def create_cp_sparse_dask_array(np_ar, n_parts):
    """
    Creates a sparse gpu dask array from the given numpy array
    """
    ar = generate_dask_array(np_ar, n_parts)
    meta = dask.array.from_array(cp_csr_matrix(cp.zeros(1, dtype=cp.float32)))
    ar = ar.map_blocks(create_cp_sparse_ar_from_dense_np_ar, meta=meta)
    return ar


def create_scipy_sparse_array_from_dask_cp_sparse_array(ar):
    """
    Creates a cpu sparse array from the given numpy array
    Will not be needed probably once we have
    https://github.com/cupy/cupy/issues/3178
    """
    meta = dask.array.from_array(scipy_csr_matrix(np.zeros(1, dtype=ar.dtype)))
    ar = ar.map_blocks(lambda x: x.get(), meta=meta)
    ar = ar.compute()
    return ar


# data_ids correspond to data, order is important
data_ids = ["base_case", "diag", "empty_feature", "123", "empty_doc"]
data = [
    np.array(
        [
            [0, 1, 1, 1, 0, 0, 1, 0, 1],
            [0, 2, 0, 1, 0, 1, 1, 0, 1],
            [1, 0, 0, 1, 1, 0, 1, 1, 1],
            [0, 1, 1, 1, 0, 0, 1, 0, 1],
        ]
    ),
    np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]]),
    np.array([[1, 1, 0], [1, 1, 0], [1, 0, 0]]),
    np.array([[1], [2], [3]]),
    np.array([[1, 1, 1], [1, 1, 0], [0, 0, 0]]),
]


@pytest.mark.mg
@pytest.mark.parametrize("data", data, ids=data_ids)
@pytest.mark.parametrize("norm", ["l1", "l2", None])
@pytest.mark.parametrize("use_idf", [True, False])
@pytest.mark.parametrize("smooth_idf", [True, False])
@pytest.mark.parametrize("sublinear_tf", [True, False])
@pytest.mark.filterwarnings(
    "ignore:divide by zero(.*):RuntimeWarning:sklearn[.*]"
)
def test_tfidf_transformer(
    data, norm, use_idf, smooth_idf, sublinear_tf, client
):
    # Testing across multiple-n_parts
    for n_parts in range(1, data.shape[0]):
        dask_sp_array = create_cp_sparse_dask_array(data, n_parts)
        tfidf = TfidfTransformer(
            norm=norm,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf,
        )
        sk_tfidf = SkTfidfTransformer(
            norm=norm,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf,
        )

        res = tfidf.fit_transform(dask_sp_array)
        res = create_scipy_sparse_array_from_dask_cp_sparse_array(
            res
        ).todense()
        ref = sk_tfidf.fit_transform(data).todense()

        cp.testing.assert_array_almost_equal(res, ref)
