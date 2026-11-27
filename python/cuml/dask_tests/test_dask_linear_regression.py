# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cudf
import dask_cudf
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

from cuml.dask.common import utils as dask_utils

pytestmark = pytest.mark.mg


def _prep_training_data(c, X_train, y_train, partitions_per_worker):
    workers = c.has_what().keys()
    n_partitions = partitions_per_worker * len(workers)
    X_cudf = cudf.DataFrame(pd.DataFrame(X_train))
    X_train_df = dask_cudf.from_cudf(X_cudf, npartitions=n_partitions)

    y_cudf = np.array(pd.DataFrame(y_train).values)
    y_cudf = y_cudf[:, 0]
    y_cudf = cudf.Series(y_cudf)
    y_train_df = dask_cudf.from_cudf(y_cudf, npartitions=n_partitions)

    X_train_df, y_train_df = dask_utils.persist_across_workers(
        c, [X_train_df, y_train_df], workers=workers
    )
    return X_train_df, y_train_df


def make_regression_dataset(datatype, nrows, ncols, n_info):
    X, y = make_regression(
        n_samples=nrows, n_features=ncols, n_informative=5, random_state=0
    )
    X = X.astype(datatype)
    y = y.astype(datatype)

    return X, y


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [1e5])
@pytest.mark.parametrize("ncols", [20])
@pytest.mark.parametrize("n_parts", [2, 23])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("delayed", [True, False])
def test_ols(nrows, ncols, n_parts, fit_intercept, datatype, delayed, client):
    def imp():
        import cuml.comm.serialize  # NOQA

    client.run(imp)

    from cuml.dask.linear_model import LinearRegression as cumlOLS_dask

    n_info = 5
    nrows = int(nrows)
    ncols = int(ncols)
    X, y = make_regression_dataset(datatype, nrows, ncols, n_info)

    X_df, y_df = _prep_training_data(client, X, y, n_parts)

    lr = cumlOLS_dask(fit_intercept=fit_intercept)

    lr.fit(X_df, y_df)

    ret = lr.predict(X_df, delayed=delayed)

    error_cuml = mean_squared_error(y, ret.compute().to_pandas().values)

    assert error_cuml < 1e-6
