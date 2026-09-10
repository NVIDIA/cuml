#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
import numpy as np
import pytest
from sklearn.covariance import (
    EmpiricalCovariance as SklearnEmpiricalCovariance,
)

import cuml
from cuml.covariance import EmpiricalCovariance
from cuml.testing.utils import as_type


def _make_random_data(n_samples=100, n_features=10, dtype=np.float64):
    rng = np.random.RandomState(42)
    return rng.randn(n_samples, n_features).astype(dtype)


def _make_covariance_data():
    rng = np.random.RandomState(42)
    real_cov = np.array([[0.4, 0.2], [0.2, 0.8]])
    return rng.multivariate_normal(mean=[0.0, 0.0], cov=real_cov, size=200)


def test_fit_returns_self():
    X = _make_random_data()
    cov = EmpiricalCovariance()
    result = cov.fit(X)
    assert result is cov


def test_fitted_attributes_exist():
    X = _make_random_data()
    cov = EmpiricalCovariance().fit(X)

    assert hasattr(cov, "covariance_")
    assert hasattr(cov, "location_")
    assert hasattr(cov, "precision_")
    assert hasattr(cov, "n_features_in_")


def test_covariance_shape():
    X = _make_random_data()
    cov = EmpiricalCovariance().fit(X)

    covariance = np.asarray(cov.covariance_)
    assert covariance.shape == (X.shape[1], X.shape[1])


def test_location_shape():
    X = _make_random_data()
    cov = EmpiricalCovariance().fit(X)

    location = np.asarray(cov.location_)
    assert location.shape == (X.shape[1],)


def test_precision_shape():
    X = _make_random_data()
    cov = EmpiricalCovariance(store_precision=True).fit(X)

    precision = np.asarray(cov.precision_)
    assert precision.shape == (X.shape[1], X.shape[1])


def test_covariance_is_symmetric():
    X = _make_random_data()
    cov = EmpiricalCovariance().fit(X)
    covariance = np.asarray(cov.covariance_)
    np.testing.assert_allclose(covariance, covariance.T, rtol=1e-5)


def test_covariance_is_positive_semidefinite():
    X = _make_random_data()
    cov = EmpiricalCovariance().fit(X)
    covariance = np.asarray(cov.covariance_)
    eigenvalues = np.linalg.eigvalsh(covariance)
    assert np.all(eigenvalues > -1e-8)


def test_covariance_matches_sklearn():
    X = _make_random_data()
    cu_cov = EmpiricalCovariance().fit(X)
    sk_cov = SklearnEmpiricalCovariance().fit(X)

    np.testing.assert_allclose(
        np.asarray(cu_cov.covariance_),
        sk_cov.covariance_,
        rtol=1e-5,
        atol=1e-7,
    )


def test_location_matches_sklearn():
    X = _make_random_data()
    cu_cov = EmpiricalCovariance().fit(X)
    sk_cov = SklearnEmpiricalCovariance().fit(X)

    np.testing.assert_allclose(
        np.asarray(cu_cov.location_), sk_cov.location_, rtol=1e-6
    )


def test_precision_matches_sklearn():
    X = _make_random_data()
    cu_cov = EmpiricalCovariance(store_precision=True).fit(X)
    sk_cov = SklearnEmpiricalCovariance(store_precision=True).fit(X)

    np.testing.assert_allclose(
        np.asarray(cu_cov.precision_),
        sk_cov.precision_,
        rtol=1e-4,
        atol=1e-6,
    )


def test_score_matches_sklearn():
    X = _make_random_data()
    cu_cov = EmpiricalCovariance().fit(X)
    sk_cov = SklearnEmpiricalCovariance().fit(X)

    np.testing.assert_allclose(cu_cov.score(X), sk_cov.score(X), rtol=1e-5)


@pytest.mark.parametrize("assume_centered", [True, False])
def test_assume_centered_matches_sklearn(assume_centered):
    X = _make_covariance_data()
    cu_cov = EmpiricalCovariance(assume_centered=assume_centered).fit(X)
    sk_cov = SklearnEmpiricalCovariance(assume_centered=assume_centered).fit(X)

    np.testing.assert_allclose(
        np.asarray(cu_cov.covariance_),
        sk_cov.covariance_,
        rtol=1e-5,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(cu_cov.location_), sk_cov.location_, rtol=1e-6
    )


def test_store_precision_false():
    X = _make_random_data()
    cov = EmpiricalCovariance(store_precision=False).fit(X)

    assert cov.precision_ is None

    precision = cov.get_precision()
    assert precision is not None
    assert np.asarray(precision).shape == (X.shape[1], X.shape[1])


def test_score_with_store_precision_false_matches_sklearn():
    X = _make_random_data()
    cu_cov = EmpiricalCovariance(store_precision=False).fit(X)
    sk_cov = SklearnEmpiricalCovariance(store_precision=False).fit(X)

    np.testing.assert_allclose(cu_cov.score(X), sk_cov.score(X), rtol=1e-5)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dtypes(dtype):
    X = _make_random_data(dtype=dtype)
    cov = EmpiricalCovariance().fit(X)

    covariance = np.asarray(cov.covariance_)
    assert covariance.dtype == dtype


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dtype_matches_sklearn(dtype):
    X = _make_random_data(dtype=dtype)
    cu_cov = EmpiricalCovariance().fit(X)
    sk_cov = SklearnEmpiricalCovariance().fit(X)

    rtol = 1e-5 if dtype == np.float64 else 1e-4
    np.testing.assert_allclose(
        np.asarray(cu_cov.covariance_),
        sk_cov.covariance_,
        rtol=rtol,
        atol=1e-6,
    )


@pytest.mark.parametrize("input_type", ["numpy", "cupy", "cudf", "pandas"])
def test_input_types(input_type):
    X = as_type(input_type, _make_random_data())
    cov = EmpiricalCovariance().fit(X)

    assert cov.covariance_ is not None


def test_cupy_input():
    X = _make_random_data()
    X_cp = cp.asarray(X)
    cov = EmpiricalCovariance().fit(X_cp)

    covariance = cp.asnumpy(cp.asarray(cov.covariance_))
    assert covariance.shape == (X.shape[1], X.shape[1])


def test_get_precision():
    X = _make_random_data()
    cov = EmpiricalCovariance(store_precision=True).fit(X)
    precision = cov.get_precision()

    np.testing.assert_allclose(
        np.asarray(precision), np.asarray(cov.precision_)
    )


def test_get_precision_without_store():
    X = _make_random_data()
    cov = EmpiricalCovariance(store_precision=False).fit(X)
    precision = cov.get_precision()

    expected_precision = np.linalg.pinv(np.asarray(cov.covariance_))
    np.testing.assert_allclose(
        np.asarray(precision), expected_precision, rtol=1e-5
    )


def test_mahalanobis():
    X = _make_random_data()
    cov = EmpiricalCovariance().fit(X)
    mahal = cov.mahalanobis(X)

    mahal_np = np.asarray(mahal)
    assert mahal_np.shape == (X.shape[0],)
    assert np.all(mahal_np >= -1e-8)


def test_mahalanobis_matches_sklearn():
    X = _make_random_data()
    cu_cov = EmpiricalCovariance().fit(X)
    sk_cov = SklearnEmpiricalCovariance().fit(X)

    np.testing.assert_allclose(
        np.asarray(cu_cov.mahalanobis(X)), sk_cov.mahalanobis(X), rtol=1e-4
    )


@pytest.mark.parametrize("store_precision", [True, False])
def test_singular_covariance_matches_sklearn(store_precision):
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    cu_cov = EmpiricalCovariance(store_precision=store_precision).fit(X)
    sk_cov = SklearnEmpiricalCovariance(store_precision=store_precision).fit(X)

    np.testing.assert_allclose(
        np.asarray(cu_cov.get_precision()),
        sk_cov.get_precision(),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(cu_cov.mahalanobis(X[:2])),
        sk_cov.mahalanobis(X[:2]),
        rtol=1e-5,
        atol=1e-6,
    )


def test_error_norm_frobenius_matches_sklearn():
    X = _make_random_data()
    cu_cov = EmpiricalCovariance().fit(X)
    sk_cov = SklearnEmpiricalCovariance().fit(X)

    other_cov = np.asarray(cu_cov.covariance_) + 0.1 * np.eye(X.shape[1])
    np.testing.assert_allclose(
        cu_cov.error_norm(other_cov),
        sk_cov.error_norm(other_cov),
        rtol=1e-5,
    )


def test_error_norm_spectral_matches_sklearn():
    X = _make_random_data()
    cu_cov = EmpiricalCovariance().fit(X)
    sk_cov = SklearnEmpiricalCovariance().fit(X)

    other_cov = np.asarray(cu_cov.covariance_) + 0.1 * np.eye(X.shape[1])
    np.testing.assert_allclose(
        cu_cov.error_norm(other_cov, norm="spectral"),
        sk_cov.error_norm(other_cov, norm="spectral"),
        rtol=1e-5,
    )


def test_error_norm_not_squared():
    X = _make_random_data()
    cov = EmpiricalCovariance().fit(X)
    other_cov = np.asarray(cov.covariance_) + 0.1 * np.eye(X.shape[1])

    squared = cov.error_norm(other_cov, squared=True)
    not_squared = cov.error_norm(other_cov, squared=False)
    np.testing.assert_allclose(not_squared, np.sqrt(squared), rtol=1e-6)


def test_error_norm_invalid():
    X = _make_random_data()
    cov = EmpiricalCovariance().fit(X)

    with pytest.raises(NotImplementedError, match="Only spectral"):
        cov.error_norm(np.asarray(cov.covariance_), norm="invalid")


def test_single_feature_matches_sklearn():
    X = _make_random_data(n_samples=100, n_features=1)
    cu_cov = EmpiricalCovariance().fit(X)
    sk_cov = SklearnEmpiricalCovariance().fit(X)

    assert np.asarray(cu_cov.covariance_).shape == (1, 1)
    np.testing.assert_allclose(
        np.asarray(cu_cov.covariance_), sk_cov.covariance_, rtol=1e-5
    )


def test_single_sample_warns():
    X = _make_random_data(n_samples=1, n_features=3)

    with pytest.warns(UserWarning, match="Only one sample available"):
        cov = EmpiricalCovariance(store_precision=False).fit(X)

    np.testing.assert_allclose(np.asarray(cov.covariance_), np.zeros((3, 3)))


def test_more_features_than_samples_matches_sklearn():
    X = _make_random_data(n_samples=20, n_features=50)
    cu_cov = EmpiricalCovariance(store_precision=False).fit(X)
    sk_cov = SklearnEmpiricalCovariance(store_precision=False).fit(X)

    np.testing.assert_allclose(
        np.asarray(cu_cov.covariance_),
        sk_cov.covariance_,
        rtol=1e-5,
        atol=1e-7,
    )


def test_output_type_numpy():
    X = _make_random_data()
    cov = EmpiricalCovariance(output_type="numpy").fit(X)

    assert isinstance(cov.covariance_, np.ndarray)


def test_output_type_cupy():
    X = _make_random_data()
    cov = EmpiricalCovariance(output_type="cupy").fit(X)

    assert isinstance(cov.covariance_, cp.ndarray)


def test_using_output_type_context():
    X = _make_random_data()
    cov = EmpiricalCovariance().fit(X)

    with cuml.using_output_type("cupy"):
        covariance = cov.covariance_
    assert isinstance(covariance, cp.ndarray)


def test_from_sklearn():
    X = _make_random_data()
    sk_cov = SklearnEmpiricalCovariance().fit(X)
    cu_cov = EmpiricalCovariance.from_sklearn(sk_cov)

    np.testing.assert_allclose(
        np.asarray(cu_cov.covariance_), sk_cov.covariance_, rtol=1e-5
    )


def test_as_sklearn():
    X = _make_random_data()
    cu_cov = EmpiricalCovariance().fit(X)
    sk_cov = cu_cov.as_sklearn()

    np.testing.assert_allclose(
        np.asarray(cu_cov.covariance_), sk_cov.covariance_, rtol=1e-5
    )


def test_get_params():
    cov = EmpiricalCovariance(store_precision=False, assume_centered=True)
    params = cov.get_params()

    assert params["store_precision"] is False
    assert params["assume_centered"] is True


def test_set_params():
    cov = EmpiricalCovariance()
    cov.set_params(store_precision=False, assume_centered=True)

    assert cov.store_precision is False
    assert cov.assume_centered is True
