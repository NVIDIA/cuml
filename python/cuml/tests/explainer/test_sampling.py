# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cudf
import cupy as cp
import numpy as np
import pandas as pd
import pytest
from cudf.pandas import LOADED as cudf_pandas_active

from cuml.explainer.sampling import kmeans_sampling


@pytest.mark.parametrize(
    "input_type",
    ["cudf-df", "cudf-series", "pandas-df", "pandas-series", "cupy", "numpy"],
)
def test_kmeans_input(input_type):
    X = cp.array(
        [[0, 10], [1, 24], [0, 52], [0, 48.0], [0.2, 23], [1, 24], [1, 23]]
    )
    if input_type == "cudf-df":
        X = cudf.DataFrame(X)
    elif input_type == "cudf-series":
        X = cudf.Series(X[:, 1])
    elif input_type == "pandas-df":
        X = pd.DataFrame(cp.asnumpy(X))
    elif input_type == "pandas-series":
        X = pd.Series(cp.asnumpy(X[:, 1]))
    elif input_type == "numpy":
        X = cp.asnumpy(X)

    summary = kmeans_sampling(X, k=2, detailed=True)

    if input_type == "cudf-df":
        cp.testing.assert_array_equal(
            summary[0].values, [[1.0, 23.0], [0.0, 52.0]]
        )
        assert isinstance(summary[0], cudf.DataFrame)
    elif input_type == "pandas-df":
        cp.testing.assert_array_equal(
            summary[0].values, [[1.0, 23.0], [0.0, 52.0]]
        )
        assert isinstance(summary[0], pd.DataFrame)
    elif input_type == "numpy":
        cp.testing.assert_array_equal(summary[0], [[1.0, 23.0], [0.0, 52.0]])
        assert isinstance(summary[0], np.ndarray)
    elif input_type == "cudf-series":
        cp.testing.assert_array_equal(summary[0].to_numpy(), [[23.0], [52.0]])
        assert isinstance(summary[0], cudf.DataFrame)
    elif input_type == "pandas-series" and not cudf_pandas_active:
        cp.testing.assert_array_equal(summary[0].to_numpy(), [[23.0], [52.0]])
        assert isinstance(summary[0], pd.DataFrame)
    elif input_type == "cupy":
        cp.testing.assert_array_equal(
            summary[0].tolist(), [[1.0, 23.0], [0.0, 52.0]]
        )
        assert isinstance(summary[0], cp.ndarray)
