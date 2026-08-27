# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cudf
import cupy as cp
import dask.array as da
import dask_cudf
import numpy as np
import pandas as pd
import pytest
import sklearn.preprocessing

from cuml.dask.preprocessing import OneHotEncoder


@pytest.mark.mg
@pytest.mark.parametrize("array_input", [False, True])
@pytest.mark.parametrize("sparse_output", [False, True])
@pytest.mark.parametrize("drop", [None, "first"])
def test_onehot_encoder(client, array_input, sparse_output, drop):
    if array_input:
        data = cp.array([[10, 20, 20], [1, 3, 2]]).T
        X1 = da.from_array(data, chunks=(2, 2))
        X2 = data.get()
    else:
        data = cudf.DataFrame(
            {"gender": ["Male", "Female", "Female"], "int": [1, 3, 2]}
        )
        X1 = dask_cudf.from_cudf(data, npartitions=2)
        X2 = data.to_numpy()

    cu_enc = OneHotEncoder(sparse_output=sparse_output, drop=drop)
    sk_enc = sklearn.preprocessing.OneHotEncoder(drop=drop)

    res = cu_enc.fit_transform(X1).compute()
    sol = sk_enc.fit_transform(X2).toarray()

    if sparse_output:
        res = res.toarray().get()
    elif array_input:
        res = res.get()
    else:
        res = res.to_numpy()

    np.testing.assert_array_equal(res, sol)


@pytest.mark.mg
@pytest.mark.parametrize("drop", [None, "first", ["F", 3]])
def test_onehot_inverse_transform(client, drop):
    df = cudf.DataFrame({0: ["M", "F", "F"], 1: [1, 3, 2]})
    X = dask_cudf.from_cudf(df, npartitions=2)

    enc = OneHotEncoder(drop=drop)
    ohe = enc.fit_transform(X)
    res = (
        enc.inverse_transform(ohe).compute().to_pandas().reset_index(drop=True)
    )
    sol = X.compute().to_pandas().reset_index(drop=True)
    pd.testing.assert_frame_equal(res, sol, check_dtype=False)


@pytest.mark.mg
def test_onehot_categories(client):
    X = cudf.DataFrame({"chars": ["a", "b"], "int": [0, 2]})
    X = dask_cudf.from_cudf(X, npartitions=2)
    enc = OneHotEncoder(categories=[["a", "b", "c"], [0, 1, 2]])
    sol = np.array(
        [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]
    )
    res = enc.fit_transform(X).compute().toarray().get()
    np.testing.assert_array_equal(res, sol)


@pytest.mark.mg
def test_onehot_fit_handle_unknown(client):
    X = cudf.DataFrame({"chars": ["a", "b"], "int": [0, 2]})
    X = dask_cudf.from_cudf(X, npartitions=2)
    categories = [["c", "b"], [0, 2]]

    enc = OneHotEncoder(handle_unknown="error", categories=categories)
    with pytest.raises(
        ValueError,
        match="Found unknown categories \\['a'\\] in column 0 during fit",
    ):
        enc.fit(X)

    enc = OneHotEncoder(handle_unknown="ignore", categories=categories)
    enc.fit(X)


@pytest.mark.mg
def test_onehot_transform_handle_unknown(client):
    X1 = cudf.DataFrame({"chars": ["a", "b"], "int": [0, 2]})
    X1 = dask_cudf.from_cudf(X1, npartitions=2)
    X2 = cudf.DataFrame({"chars": ["c", "b"], "int": [0, 2]})
    X2 = dask_cudf.from_cudf(X2, npartitions=2)

    enc = OneHotEncoder(handle_unknown="error")
    enc = enc.fit(X1)
    with pytest.raises(
        ValueError,
        match="Found unknown categories \\['c'\\] in column 0 during transform",
    ):
        enc.transform(X2).compute()

    enc = OneHotEncoder(handle_unknown="ignore")
    enc = enc.fit(X1)
    res = enc.transform(X2).compute().toarray().get()
    sol = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    np.testing.assert_array_equal(res, sol)
