# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import numpy as np
import pytest

import cuml
from cuml.dask.datasets import make_regression
from cuml.dask.linear_model import ElasticNet, Lasso
from cuml.internals.global_settings import GlobalSettings
from cuml.metrics import r2_score
from cuml.testing.utils import (
    as_numpy,
    quality_param,
    stress_param,
    unit_param,
)


def _as_numpy_array(value):
    return np.asarray(as_numpy(value))


def _assert_cd_fitted_attrs_match_solver(model, dtype):
    coef = model.coef_
    intercept = model.intercept_
    n_iter = model.n_iter_
    solver_coef = model.solver.coef_
    solver_intercept = model.solver.intercept_

    assert coef is not None
    assert intercept is not None
    assert n_iter is not None
    assert isinstance(coef, cp.ndarray)
    assert np.isscalar(intercept)

    coef_np = _as_numpy_array(coef)
    intercept_np = _as_numpy_array(intercept)

    assert coef_np.dtype == dtype
    assert coef_np.shape == (20,)
    assert isinstance(n_iter, (int, np.integer))
    np.testing.assert_allclose(coef_np, _as_numpy_array(solver_coef))
    np.testing.assert_allclose(intercept_np, _as_numpy_array(solver_intercept))
    assert n_iter == model.solver.n_iter_

    previous_output_type = GlobalSettings().output_type
    with cuml.using_output_type("numpy"):
        assert isinstance(model.coef_, np.ndarray)
        assert np.isscalar(model.intercept_)
        np.testing.assert_allclose(model.coef_, model.solver.coef_)
        np.testing.assert_allclose(model.intercept_, model.solver.intercept_)
    assert GlobalSettings().output_type == previous_output_type


@pytest.mark.mg
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("alpha", [0.001])
@pytest.mark.parametrize("algorithm", ["cyclic", "random"])
@pytest.mark.parametrize(
    "nrows",
    [
        unit_param(50),
        quality_param(5000),
        stress_param(500000),
    ],
)
@pytest.mark.parametrize(
    "column_info",
    [
        unit_param([20, 10]),
        quality_param([100, 50]),
        stress_param([1000, 500]),
    ],
)
@pytest.mark.parametrize(
    "n_parts", [unit_param(4), quality_param(32), stress_param(64)]
)
@pytest.mark.parametrize("delayed", [True, False])
def test_lasso(
    dtype, alpha, algorithm, nrows, column_info, n_parts, delayed, client
):
    ncols, n_info = column_info

    X, y = make_regression(
        n_samples=nrows,
        n_features=ncols,
        n_informative=n_info,
        n_parts=n_parts,
        client=client,
        dtype=dtype,
    )

    lasso = Lasso(
        alpha=np.array([alpha]),
        fit_intercept=True,
        max_iter=1000,
        selection=algorithm,
        tol=1e-10,
        client=client,
    )

    lasso.fit(X, y)

    y_hat = lasso.predict(X, delayed=delayed)

    assert r2_score(y.compute(), y_hat.compute()) >= 0.99


@pytest.mark.mg
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "nrows", [unit_param(50), quality_param(5000), stress_param(500000)]
)
@pytest.mark.parametrize(
    "column_info",
    [
        unit_param([20, 10]),
        quality_param([100, 50]),
        stress_param([1000, 500]),
    ],
)
@pytest.mark.parametrize(
    "n_parts", [unit_param(16), quality_param(32), stress_param(64)]
)
def test_lasso_default(dtype, nrows, column_info, n_parts, client):
    ncols, n_info = column_info

    X, y = make_regression(
        n_samples=nrows,
        n_features=ncols,
        n_informative=n_info,
        client=client,
        dtype=dtype,
    )

    lasso = Lasso(client=client)

    lasso.fit(X, y)

    y_hat = lasso.predict(X)

    assert r2_score(y.compute(), y_hat.compute()) >= 0.99


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("alpha", [0.5])
@pytest.mark.parametrize("algorithm", ["cyclic", "random"])
@pytest.mark.parametrize(
    "nrows",
    [
        unit_param(500),
        quality_param(5000),
        stress_param(500000),
    ],
)
@pytest.mark.parametrize(
    "column_info",
    [
        unit_param([20, 10]),
        quality_param([100, 50]),
        stress_param([1000, 500]),
    ],
)
@pytest.mark.parametrize(
    "n_parts", [unit_param(16), quality_param(32), stress_param(64)]
)
@pytest.mark.parametrize("delayed", [True, False])
def test_elastic_net(
    dtype, alpha, algorithm, nrows, column_info, n_parts, client, delayed
):
    ncols, n_info = column_info

    X, y = make_regression(
        n_samples=nrows,
        n_features=ncols,
        n_informative=n_info,
        n_parts=n_parts,
        client=client,
        dtype=dtype,
    )

    elasticnet = ElasticNet(
        alpha=np.array([alpha]),
        fit_intercept=True,
        max_iter=1000,
        selection=algorithm,
        tol=1e-10,
        client=client,
    )

    elasticnet.fit(X, y)

    y_hat = elasticnet.predict(X, delayed=delayed)

    # based on differences with scikit-learn 0.22
    if alpha == 0.2:
        assert r2_score(y.compute(), y_hat.compute()) >= 0.96

    else:
        assert r2_score(y.compute(), y_hat.compute()) >= 0.80


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "nrows", [unit_param(500), quality_param(5000), stress_param(500000)]
)
@pytest.mark.parametrize(
    "column_info",
    [
        unit_param([20, 10]),
        quality_param([100, 50]),
        stress_param([1000, 500]),
    ],
)
@pytest.mark.parametrize(
    "n_parts", [unit_param(16), quality_param(32), stress_param(64)]
)
def test_elastic_net_default(dtype, nrows, column_info, n_parts, client):
    ncols, n_info = column_info

    X, y = make_regression(
        n_samples=nrows,
        n_features=ncols,
        n_informative=n_info,
        n_parts=n_parts,
        client=client,
        dtype=dtype,
    )

    elasticnet = ElasticNet(client=client)

    elasticnet.fit(X, y)

    y_hat = elasticnet.predict(X)

    assert r2_score(y.compute(), y_hat.compute()) >= 0.96


@pytest.mark.parametrize("cls", [ElasticNet, Lasso])
def test_max_iter_n_iter(cls, client):
    X, y = make_regression(
        n_samples=500,
        n_features=20,
        n_parts=10,
        n_informative=10,
        client=client,
        dtype=np.float32,
    )

    model = cls(max_iter=2, client=client).fit(X, y)
    assert model.solver.n_iter_ == 2
    assert model.n_iter_ == model.solver.n_iter_


@pytest.mark.parametrize("cls", [ElasticNet, Lasso])
@pytest.mark.parametrize("attr", ["coef_", "intercept_", "n_iter_"])
def test_cd_fitted_attributes_require_fit(cls, attr, client):
    model = cls(client=client)

    with pytest.raises(AttributeError, match="not fitted") as error:
        getattr(model, attr)

    assert error.value.__cause__ is not None


@pytest.mark.parametrize("cls", [ElasticNet, Lasso])
def test_cd_fitted_attributes_are_absent_before_fit(cls, client):
    assert not hasattr(cls(client=client), "coef_")


@pytest.mark.parametrize("cls", [ElasticNet, Lasso])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cd_fitted_attributes(cls, dtype, client):
    X, y = make_regression(
        n_samples=500,
        n_features=20,
        n_parts=10,
        n_informative=10,
        client=client,
        dtype=dtype,
    )

    model = cls(alpha=np.array([0.001]), max_iter=2, tol=0.0, client=client)
    for attr in ("coef_", "intercept_", "n_iter_"):
        with pytest.raises(AttributeError):
            getattr(model, attr)

    model.fit(X, y)

    with pytest.raises(AttributeError):
        model.attribute_that_does_not_exist_

    _assert_cd_fitted_attrs_match_solver(model, dtype)
    first_coef = _as_numpy_array(model.coef_).copy()
    first_n_iter = model.n_iter_
    assert first_n_iter == 2

    model.solver.kwargs["max_iter"] = 3
    model.fit(X, -y)

    _assert_cd_fitted_attrs_match_solver(model, dtype)
    assert not np.allclose(_as_numpy_array(model.coef_), first_coef)
    assert model.n_iter_ == 3
    assert model.n_iter_ != first_n_iter
