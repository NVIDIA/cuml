# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cudf
import dask_cudf
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import cuml

dask_sql = pytest.importorskip("dask_sql")


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("nrows", [100000])
@pytest.mark.parametrize("ncols", [20, 50])
@pytest.mark.parametrize("n_parts", [2, 20])
@pytest.mark.parametrize("wrap_predict", [True, False])
def test_dask_sql_sg_logistic_regression(
    datatype, nrows, ncols, n_parts, wrap_predict
):
    if wrap_predict:
        cuml.set_global_output_type("input")
    else:
        cuml.set_global_output_type("cudf")

    X, y = make_classification(
        n_samples=nrows, n_features=ncols, n_informative=5, random_state=0
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    train_df = cudf.DataFrame(
        X_train, dtype=datatype, columns=[chr(i) for i in range(ncols)]
    )
    train_df["target"] = y_train
    train_ddf = dask_cudf.from_cudf(train_df, npartitions=n_parts)

    c = dask_sql.Context()
    c.create_table("train_df", train_ddf)

    train_query = f"""
        CREATE MODEL model WITH (
            model_class = 'cuml.linear_model.LogisticRegression',
            wrap_predict = {wrap_predict},
            target_column = 'target'
        ) AS (
            SELECT * FROM train_df
        )
    """

    c.sql(train_query)

    skmodel = LogisticRegression().fit(X_train, y_train)

    test_df = cudf.DataFrame(
        X_test, dtype=datatype, columns=[chr(i) for i in range(ncols)]
    )
    test_ddf = dask_cudf.from_cudf(test_df, npartitions=n_parts)
    c.create_table("test_df", test_ddf)

    inference_query = """
        SELECT * FROM PREDICT(
            MODEL model,
            SELECT * FROM test_df
        )
    """

    preds = c.sql(inference_query).compute()
    score = cuml.metrics.accuracy_score(y_test, preds["target"].to_numpy())

    assert score >= skmodel.score(X_test, y_test) - 0.022
