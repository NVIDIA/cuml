# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import cudf
import dask_cudf
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

import cuml
from cuml.dask.common import utils as dask_utils
from cuml.internals.global_settings import GlobalSettings
from cuml.testing.utils import as_numpy

pytestmark = pytest.mark.mg


def _as_numpy_array(value):
    return np.asarray(as_numpy(value))


def _assert_ridge_fitted_attrs_match_model(
    model, internal_model, datatype, ncols, fit_intercept
):
    coef = model.coef_
    intercept = model.intercept_

    assert coef is not None
    assert intercept is not None
    assert isinstance(coef, cudf.Series)
    assert np.isscalar(intercept)

    coef_np = _as_numpy_array(coef)

    assert coef_np.dtype == datatype
    assert coef_np.shape == (ncols,)
    np.testing.assert_allclose(coef_np, _as_numpy_array(internal_model.coef_))
    assert intercept == internal_model.intercept_

    if not fit_intercept:
        assert intercept == 0.0

    previous_output_type = GlobalSettings().output_type
    with cuml.using_output_type("numpy"):
        assert isinstance(model.coef_, np.ndarray)
        assert np.isscalar(model.intercept_)
        np.testing.assert_allclose(model.coef_, internal_model.coef_)
        assert model.intercept_ == internal_model.intercept_
    assert GlobalSettings().output_type == previous_output_type


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
@pytest.mark.parametrize("nrows", [1e4])
@pytest.mark.parametrize("ncols", [10])
@pytest.mark.parametrize("n_parts", [2, 23])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("delayed", [True, False])
def test_ridge(
    nrows, ncols, n_parts, fit_intercept, datatype, delayed, client
):
    from cuml.dask.linear_model import Ridge as cumlRidge_dask

    n_info = 5
    nrows = int(nrows)
    ncols = int(ncols)
    X, y = make_regression_dataset(datatype, nrows, ncols, n_info)

    X_df, y_df = _prep_training_data(client, X, y, n_parts)

    lr = cumlRidge_dask(alpha=0.5, fit_intercept=fit_intercept)
    for attr in ("coef_", "intercept_"):
        with pytest.raises(AttributeError):
            getattr(lr, attr)

    lr.fit(X_df, y_df)

    with pytest.raises(AttributeError):
        lr.attribute_that_does_not_exist_

    internal_model = lr.get_combined_model()
    _assert_ridge_fitted_attrs_match_model(
        lr, internal_model, datatype, ncols, fit_intercept
    )

    ret = lr.predict(X_df, delayed=delayed)

    error_cuml = mean_squared_error(y, ret.compute().to_pandas().values)

    assert error_cuml < 1e-1


@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("datatype", [np.float32, np.float64])
def test_ridge_fitted_attributes_follow_refit(fit_intercept, datatype, client):
    from cuml.dask.linear_model import Ridge as cumlRidge_dask

    nrows = 500
    ncols = 10
    X, y = make_regression_dataset(datatype, nrows, ncols, n_info=5)
    X_df, y_df = _prep_training_data(client, X, y, partitions_per_worker=2)

    model = cumlRidge_dask(alpha=0.5, fit_intercept=fit_intercept)
    model.fit(X_df, y_df)
    _assert_ridge_fitted_attrs_match_model(
        model, model.get_combined_model(), datatype, ncols, fit_intercept
    )
    first_coef = _as_numpy_array(model.coef_).copy()

    refit_y_df = -y_df
    model.fit(X_df, refit_y_df)

    _assert_ridge_fitted_attrs_match_model(
        model, model.get_combined_model(), datatype, ncols, fit_intercept
    )
    assert not np.allclose(_as_numpy_array(model.coef_), first_coef)
