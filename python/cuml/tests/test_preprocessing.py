# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
import cupyx as cpx
import numpy as np
import pytest
import scipy
import sklearn
import sklearn.datasets
from packaging.version import Version
from sklearn.impute import MissingIndicator as skMissingIndicator
from sklearn.impute import SimpleImputer as skSimpleImputer
from sklearn.preprocessing import Binarizer as skBinarizer
from sklearn.preprocessing import FunctionTransformer as skFunctionTransformer
from sklearn.preprocessing import KBinsDiscretizer as skKBinsDiscretizer
from sklearn.preprocessing import KernelCenterer as skKernelCenterer
from sklearn.preprocessing import MaxAbsScaler as skMaxAbsScaler
from sklearn.preprocessing import MinMaxScaler as skMinMaxScaler
from sklearn.preprocessing import Normalizer as skNormalizer
from sklearn.preprocessing import PolynomialFeatures as skPolynomialFeatures
from sklearn.preprocessing import PowerTransformer as skPowerTransformer
from sklearn.preprocessing import QuantileTransformer as skQuantileTransformer
from sklearn.preprocessing import RobustScaler as skRobustScaler
from sklearn.preprocessing import StandardScaler as skStandardScaler
from sklearn.preprocessing import add_dummy_feature as sk_add_dummy_feature
from sklearn.preprocessing import binarize as sk_binarize
from sklearn.preprocessing import label_binarize as sk_label_binarize
from sklearn.preprocessing import maxabs_scale as sk_maxabs_scale
from sklearn.preprocessing import minmax_scale as sk_minmax_scale
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.preprocessing import power_transform as sk_power_transform
from sklearn.preprocessing import quantile_transform as sk_quantile_transform
from sklearn.preprocessing import robust_scale as sk_robust_scale
from sklearn.preprocessing import scale as sk_scale

from cuml.metrics import pairwise_kernels
from cuml.preprocessing import Binarizer as cuBinarizer
from cuml.preprocessing import FunctionTransformer as cuFunctionTransformer
from cuml.preprocessing import KBinsDiscretizer as cuKBinsDiscretizer
from cuml.preprocessing import KernelCenterer as cuKernelCenterer
from cuml.preprocessing import MaxAbsScaler as cuMaxAbsScaler
from cuml.preprocessing import MinMaxScaler as cuMinMaxScaler
from cuml.preprocessing import MissingIndicator as cuMissingIndicator
from cuml.preprocessing import Normalizer as cuNormalizer
from cuml.preprocessing import PolynomialFeatures as cuPolynomialFeatures
from cuml.preprocessing import PowerTransformer as cuPowerTransformer
from cuml.preprocessing import QuantileTransformer as cuQuantileTransformer
from cuml.preprocessing import RobustScaler as cuRobustScaler
from cuml.preprocessing import SimpleImputer as cuSimpleImputer
from cuml.preprocessing import StandardScaler as cuStandardScaler
from cuml.preprocessing import add_dummy_feature as cu_add_dummy_feature
from cuml.preprocessing import binarize as cu_binarize
from cuml.preprocessing import label_binarize as cu_label_binarize
from cuml.preprocessing import maxabs_scale as cu_maxabs_scale
from cuml.preprocessing import minmax_scale as cu_minmax_scale
from cuml.preprocessing import normalize as cu_normalize
from cuml.preprocessing import power_transform as cu_power_transform
from cuml.preprocessing import quantile_transform as cu_quantile_transform
from cuml.preprocessing import robust_scale as cu_robust_scale
from cuml.preprocessing import scale as cu_scale
from cuml.testing.test_preproc_utils import (  # noqa: F401
    assert_allclose,
    blobs_dataset,
    clf_dataset,
    int_dataset,
    nan_filled_positive,
    sparse_blobs_dataset,
    sparse_clf_dataset,
    sparse_dataset_with_coo,
    sparse_imputer_dataset,
    sparse_int_dataset,
    sparse_nan_filled_positive,
)


@pytest.mark.parametrize("feature_range", [(0, 1), (0.1, 0.8)])
def test_minmax_scaler(
    failure_logger,
    clf_dataset,
    feature_range,  # noqa: F811
):
    X_np, X = clf_dataset

    scaler = cuMinMaxScaler(feature_range=feature_range, copy=True)
    t_X = scaler.fit_transform(X)
    r_X = scaler.inverse_transform(t_X)
    assert type(t_X) is type(X)
    assert type(r_X) is type(t_X)

    scaler = skMinMaxScaler(feature_range=feature_range, copy=True)
    sk_t_X = scaler.fit_transform(X_np)
    sk_r_X = scaler.inverse_transform(sk_t_X)

    assert_allclose(t_X, sk_t_X)
    assert_allclose(r_X, sk_r_X)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("feature_range", [(0, 1), (0.1, 0.8)])
def test_minmax_scale(
    failure_logger,
    clf_dataset,
    axis,
    feature_range,  # noqa: F811
):
    X_np, X = clf_dataset

    t_X = cu_minmax_scale(X, feature_range=feature_range, axis=axis)
    assert type(t_X) is type(X)

    sk_t_X = sk_minmax_scale(X_np, feature_range=feature_range, axis=axis)

    assert_allclose(t_X, sk_t_X)


@pytest.mark.parametrize("with_mean", [True, False])
@pytest.mark.parametrize("with_std", [True, False])
def test_standard_scaler(
    failure_logger,
    clf_dataset,
    with_mean,
    with_std,  # noqa: F811
):
    X_np, X = clf_dataset

    scaler = cuStandardScaler(
        with_mean=with_mean, with_std=with_std, copy=True
    )
    t_X = scaler.fit_transform(X)
    r_X = scaler.inverse_transform(t_X)
    assert type(t_X) is type(X)
    assert type(r_X) is type(t_X)

    scaler = skStandardScaler(
        with_mean=with_mean, with_std=with_std, copy=True
    )
    sk_t_X = scaler.fit_transform(X_np)
    sk_r_X = scaler.inverse_transform(sk_t_X)

    assert_allclose(t_X, sk_t_X)
    assert_allclose(r_X, sk_r_X)


@pytest.mark.parametrize("with_std", [True, False])
def test_standard_scaler_sparse(
    failure_logger,
    sparse_clf_dataset,
    with_std,  # noqa: F811
):
    X_np, X = sparse_clf_dataset

    scaler = cuStandardScaler(with_mean=False, with_std=with_std, copy=True)
    t_X = scaler.fit_transform(X)
    r_X = scaler.inverse_transform(t_X)
    #  assert type(t_X) is type(X)
    #  assert type(r_X) is type(t_X)
    if cpx.scipy.sparse.issparse(X):
        assert cpx.scipy.sparse.issparse(t_X)
    if scipy.sparse.issparse(X):
        assert scipy.sparse.issparse(t_X)
    if cpx.scipy.sparse.issparse(t_X):
        assert cpx.scipy.sparse.issparse(r_X)
    if scipy.sparse.issparse(t_X):
        assert scipy.sparse.issparse(r_X)

    assert scaler.n_samples_seen_ == X.shape[0]

    scaler = skStandardScaler(copy=True, with_mean=False, with_std=with_std)
    sk_t_X = scaler.fit_transform(X_np)
    sk_r_X = scaler.inverse_transform(sk_t_X)

    assert_allclose(t_X, sk_t_X)
    assert_allclose(r_X, sk_r_X)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("with_mean", [True, False])
@pytest.mark.parametrize("with_std", [True, False])
# The numerical warning is triggered when centering or scaling
# cannot be done as single steps. Its display can be safely disabled.
# For more information see : https://github.com/rapidsai/cuml/issues/4203
@pytest.mark.filterwarnings("ignore:Numerical issues::")
def test_scale(
    failure_logger,
    clf_dataset,
    axis,
    with_mean,
    with_std,  # noqa: F811
):
    X_np, X = clf_dataset

    t_X = cu_scale(
        X, axis=axis, with_mean=with_mean, with_std=with_std, copy=True
    )
    assert type(t_X) is type(X)

    sk_t_X = sk_scale(
        X_np, axis=axis, with_mean=with_mean, with_std=with_std, copy=True
    )

    assert_allclose(t_X, sk_t_X)


@pytest.mark.parametrize("with_std", [True, False])
def test_scale_sparse(
    failure_logger,
    sparse_clf_dataset,
    with_std,  # noqa: F811
):
    X_np, X = sparse_clf_dataset

    t_X = cu_scale(X, with_mean=False, with_std=with_std, copy=True)
    #  assert type(t_X) is type(X)
    if cpx.scipy.sparse.issparse(X):
        assert cpx.scipy.sparse.issparse(t_X)
    if scipy.sparse.issparse(X):
        assert scipy.sparse.issparse(t_X)

    sk_t_X = sk_scale(X_np, with_mean=False, with_std=with_std, copy=True)

    assert_allclose(t_X, sk_t_X)


@pytest.mark.parametrize("axis", [0, 1])
def test_maxabs_scale(failure_logger, clf_dataset, axis):  # noqa: F811
    X_np, X = clf_dataset

    t_X = cu_maxabs_scale(X, axis=axis)
    assert type(t_X) is type(X)

    sk_t_X = sk_maxabs_scale(X_np, axis=axis)

    assert_allclose(t_X, sk_t_X)


def test_maxabs_scaler(failure_logger, clf_dataset):  # noqa: F811
    X_np, X = clf_dataset

    scaler = cuMaxAbsScaler(copy=True)
    t_X = scaler.fit_transform(X)
    r_X = scaler.inverse_transform(t_X)
    assert type(t_X) is type(X)
    assert type(r_X) is type(t_X)

    scaler = skMaxAbsScaler(copy=True)
    sk_t_X = scaler.fit_transform(X_np)
    sk_r_X = scaler.inverse_transform(sk_t_X)

    assert_allclose(t_X, sk_t_X)
    assert_allclose(r_X, sk_r_X)


def test_maxabs_scaler_sparse(failure_logger, sparse_clf_dataset):  # noqa: F811
    X_np, X = sparse_clf_dataset

    scaler = cuMaxAbsScaler(copy=True)
    t_X = scaler.fit_transform(X)
    r_X = scaler.inverse_transform(t_X)
    #  assert type(t_X) is type(X)
    #  assert type(r_X) is type(t_X)
    if cpx.scipy.sparse.issparse(X):
        assert cpx.scipy.sparse.issparse(t_X)
    if scipy.sparse.issparse(X):
        assert scipy.sparse.issparse(t_X)
    if cpx.scipy.sparse.issparse(t_X):
        assert cpx.scipy.sparse.issparse(r_X)
    if scipy.sparse.issparse(t_X):
        assert scipy.sparse.issparse(r_X)

    scaler = skMaxAbsScaler(copy=True)
    sk_t_X = scaler.fit_transform(X_np)
    sk_r_X = scaler.inverse_transform(sk_t_X)

    assert_allclose(t_X, sk_t_X)
    assert_allclose(r_X, sk_r_X)


@pytest.mark.parametrize("norm", ["l1", "l2", "max"])
def test_normalizer(failure_logger, clf_dataset, norm):  # noqa: F811
    X_np, X = clf_dataset

    normalizer = cuNormalizer(norm=norm, copy=True)
    t_X = normalizer.fit_transform(X)
    assert type(t_X) is type(X)

    normalizer = skNormalizer(norm=norm, copy=True)
    sk_t_X = normalizer.fit_transform(X_np)

    assert_allclose(t_X, sk_t_X)


@pytest.mark.parametrize("norm", ["l1", "l2", "max"])
def test_normalizer_sparse(
    failure_logger,
    sparse_clf_dataset,
    norm,  # noqa: F811
):
    X_np, X = sparse_clf_dataset

    if X.format == "csc":
        pytest.skip("Skipping CSC matrices")

    normalizer = cuNormalizer(norm=norm, copy=True)
    t_X = normalizer.fit_transform(X)
    #  assert type(t_X) is type(X)
    if cpx.scipy.sparse.issparse(X):
        assert cpx.scipy.sparse.issparse(t_X)
    if scipy.sparse.issparse(X):
        assert scipy.sparse.issparse(t_X)

    normalizer = skNormalizer(norm=norm, copy=True)
    sk_t_X = normalizer.fit_transform(X_np)

    assert_allclose(t_X, sk_t_X)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("norm", ["l1", "l2", "max"])
@pytest.mark.parametrize("return_norm", [True, False])
def test_normalize(
    failure_logger,
    clf_dataset,
    axis,
    norm,
    return_norm,  # noqa: F811
):
    X_np, X = clf_dataset

    if return_norm:
        t_X, t_norms = cu_normalize(
            X, axis=axis, norm=norm, return_norm=return_norm
        )
        sk_t_X, sk_t_norms = sk_normalize(
            X_np, axis=axis, norm=norm, return_norm=return_norm
        )
        assert_allclose(t_norms, sk_t_norms)
    else:
        t_X = cu_normalize(X, axis=axis, norm=norm, return_norm=return_norm)
        sk_t_X = sk_normalize(
            X_np, axis=axis, norm=norm, return_norm=return_norm
        )

    assert type(t_X) is type(X)
    assert_allclose(t_X, sk_t_X)


@pytest.mark.parametrize("norm", ["l1", "l2", "max"])
def test_normalize_sparse(
    failure_logger,
    sparse_clf_dataset,
    norm,  # noqa: F811
):
    X_np, X = sparse_clf_dataset

    axis = 0 if X.format == "csc" else 1

    t_X = cu_normalize(X, axis=axis, norm=norm)
    #  assert type(t_X) is type(X)
    if cpx.scipy.sparse.issparse(X):
        assert cpx.scipy.sparse.issparse(t_X)
    if scipy.sparse.issparse(X):
        assert scipy.sparse.issparse(t_X)

    sk_t_X = sk_normalize(X_np, axis=axis, norm=norm)

    assert_allclose(t_X, sk_t_X)


@pytest.mark.parametrize(
    "strategy", ["mean", "median", "most_frequent", "constant"]
)
@pytest.mark.parametrize("missing_values", [0, 1, np.nan])
@pytest.mark.parametrize("add_indicator", [False, True])
def test_imputer(
    failure_logger,
    random_seed,
    int_dataset,  # noqa: F811
    strategy,
    missing_values,
    add_indicator,
):
    zero_filled, one_filled, nan_filled = int_dataset
    if missing_values == 0:
        X_np, X = zero_filled
    elif missing_values == 1:
        X_np, X = one_filled
    else:
        X_np, X = nan_filled
    np.random.seed(random_seed)
    fill_value = np.random.randint(10, size=1)[0]

    imputer = cuSimpleImputer(
        copy=True,
        missing_values=missing_values,
        strategy=strategy,
        fill_value=fill_value,
        add_indicator=add_indicator,
    )
    t_X = imputer.fit_transform(X)
    assert type(t_X) is type(X)

    imputer = skSimpleImputer(
        copy=True,
        missing_values=missing_values,
        strategy=strategy,
        fill_value=fill_value,
        add_indicator=add_indicator,
    )
    sk_t_X = imputer.fit_transform(X_np)

    assert_allclose(t_X, sk_t_X)


@pytest.mark.parametrize(
    "strategy", ["mean", "median", "most_frequent", "constant"]
)
def test_imputer_sparse(sparse_imputer_dataset, strategy):  # noqa: F811
    missing_values, X_sp, X = sparse_imputer_dataset

    if X.format == "csr":
        pytest.skip("Skipping CSR matrices")

    fill_value = np.random.randint(10, size=1)[0]

    imputer = cuSimpleImputer(
        copy=True,
        missing_values=missing_values,
        strategy=strategy,
        fill_value=fill_value,
    )
    t_X = imputer.fit_transform(X)
    #  assert type(t_X) is type(X)
    if cpx.scipy.sparse.issparse(X):
        assert cpx.scipy.sparse.issparse(t_X)
    if scipy.sparse.issparse(X):
        assert scipy.sparse.issparse(t_X)

    imputer = skSimpleImputer(
        copy=True,
        missing_values=missing_values,
        strategy=strategy,
        fill_value=fill_value,
    )
    sk_t_X = imputer.fit_transform(X_sp)
    assert_allclose(t_X, sk_t_X)


@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("interaction_only", [True, False])
@pytest.mark.parametrize("include_bias", [True, False])
@pytest.mark.parametrize("order", ["C", "F"])
def test_poly_features(
    failure_logger,
    clf_dataset,
    degree,  # noqa: F811
    interaction_only,
    include_bias,
    order,
):
    X_np, X = clf_dataset

    polyfeatures = cuPolynomialFeatures(
        degree=degree,
        order=order,
        interaction_only=interaction_only,
        include_bias=include_bias,
    )
    t_X = polyfeatures.fit_transform(X)
    assert type(X) is type(t_X)
    cu_feature_names = polyfeatures.get_feature_names_out()

    if isinstance(t_X, np.ndarray):
        if order == "C":
            assert t_X.flags["C_CONTIGUOUS"]
        elif order == "F":
            assert t_X.flags["F_CONTIGUOUS"]

    polyfeatures = skPolynomialFeatures(
        degree=degree,
        order=order,
        interaction_only=interaction_only,
        include_bias=include_bias,
    )
    sk_t_X = polyfeatures.fit_transform(X_np)
    sk_feature_names = polyfeatures.get_feature_names_out()

    assert_allclose(t_X, sk_t_X, rtol=0.1, atol=0.1)
    np.testing.assert_array_equal(cu_feature_names, sk_feature_names)


def test_poly_features_get_feature_names_deprecated():
    X = np.array([[1.5, 2.5, 3.5], [1.6, 2.4, 3.7]])
    model = cuPolynomialFeatures().fit(X)
    with pytest.warns(FutureWarning, match="get_feature_names"):
        res = model.get_feature_names()

    np.testing.assert_array_equal(res, model.get_feature_names_out())


@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("interaction_only", [True, False])
@pytest.mark.parametrize("include_bias", [True, False])
def test_poly_features_sparse(
    failure_logger,
    sparse_clf_dataset,  # noqa: F811
    degree,
    interaction_only,
    include_bias,
):
    X_np, X = sparse_clf_dataset

    polyfeatures = cuPolynomialFeatures(
        degree=degree,
        interaction_only=interaction_only,
        include_bias=include_bias,
    )
    t_X = polyfeatures.fit_transform(X)
    #  assert type(t_X) is type(X)
    if cpx.scipy.sparse.issparse(X):
        assert cpx.scipy.sparse.issparse(t_X)
    if scipy.sparse.issparse(X):
        assert scipy.sparse.issparse(t_X)

    polyfeatures = skPolynomialFeatures(
        degree=degree,
        interaction_only=interaction_only,
        include_bias=include_bias,
    )
    sk_t_X = polyfeatures.fit_transform(X_np)

    assert_allclose(t_X, sk_t_X, rtol=0.1, atol=0.1)


@pytest.mark.parametrize("value", [1.0, 42])
def test_add_dummy_feature(failure_logger, clf_dataset, value):  # noqa: F811
    X_np, X = clf_dataset

    t_X = cu_add_dummy_feature(X, value=value)
    assert type(t_X) is type(X)

    sk_t_X = sk_add_dummy_feature(X_np, value=value)
    assert_allclose(t_X, sk_t_X)


@pytest.mark.parametrize("value", [1.0, 42])
def test_add_dummy_feature_sparse(
    failure_logger,
    sparse_dataset_with_coo,
    value,  # noqa: F811
):
    X_np, X = sparse_dataset_with_coo

    t_X = cu_add_dummy_feature(X, value=value)
    #  assert type(t_X) is type(X)
    if cpx.scipy.sparse.issparse(X):
        assert cpx.scipy.sparse.issparse(t_X)
    if scipy.sparse.issparse(X):
        assert scipy.sparse.issparse(t_X)

    sk_t_X = sk_add_dummy_feature(X_np, value=value)
    assert_allclose(t_X, sk_t_X)


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_binarize(failure_logger, clf_dataset, threshold):  # noqa: F811
    X_np, X = clf_dataset

    t_X = cu_binarize(X, threshold=threshold, copy=True)
    assert type(t_X) is type(X)

    sk_t_X = sk_binarize(X_np, threshold=threshold, copy=True)

    assert_allclose(t_X, sk_t_X)


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_binarize_sparse(
    failure_logger,
    sparse_clf_dataset,
    threshold,  # noqa: F811
):
    X_np, X = sparse_clf_dataset

    t_X = cu_binarize(X, threshold=threshold, copy=True)
    #  assert type(t_X) is type(X)
    if cpx.scipy.sparse.issparse(X):
        assert cpx.scipy.sparse.issparse(t_X)
    if scipy.sparse.issparse(X):
        assert scipy.sparse.issparse(t_X)

    sk_t_X = sk_binarize(X_np, threshold=threshold, copy=True)

    assert_allclose(t_X, sk_t_X)


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_binarizer(failure_logger, clf_dataset, threshold):  # noqa: F811
    X_np, X = clf_dataset

    binarizer = cuBinarizer(threshold=threshold, copy=True)
    t_X = binarizer.fit_transform(X)
    assert type(t_X) is type(X)

    binarizer = skBinarizer(threshold=threshold, copy=True)
    sk_t_X = binarizer.fit_transform(X_np)

    assert_allclose(t_X, sk_t_X)


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_binarizer_sparse(
    failure_logger,
    sparse_clf_dataset,
    threshold,  # noqa: F811
):
    X_np, X = sparse_clf_dataset

    binarizer = cuBinarizer(threshold=threshold, copy=True)
    t_X = binarizer.fit_transform(X)
    #  assert type(t_X) is type(X)
    if cpx.scipy.sparse.issparse(X):
        assert cpx.scipy.sparse.issparse(t_X)
    if scipy.sparse.issparse(X):
        assert scipy.sparse.issparse(t_X)

    binarizer = skBinarizer(threshold=threshold, copy=True)
    sk_t_X = binarizer.fit_transform(X_np)

    assert_allclose(t_X, sk_t_X)


@pytest.mark.parametrize("with_centering", [True, False])
@pytest.mark.parametrize("with_scaling", [True, False])
@pytest.mark.parametrize("quantile_range", [(25.0, 75.0), (10.0, 90.0)])
def test_robust_scaler(
    failure_logger,
    clf_dataset,  # noqa: F811
    with_centering,
    with_scaling,
    quantile_range,
):
    X_np, X = clf_dataset

    scaler = cuRobustScaler(
        with_centering=with_centering,
        with_scaling=with_scaling,
        quantile_range=quantile_range,
        copy=True,
    )
    t_X = scaler.fit_transform(X)
    r_X = scaler.inverse_transform(t_X)
    assert type(t_X) is type(X)
    assert type(r_X) is type(t_X)

    scaler = skRobustScaler(
        with_centering=with_centering,
        with_scaling=with_scaling,
        quantile_range=quantile_range,
        copy=True,
    )
    sk_t_X = scaler.fit_transform(X_np)
    sk_r_X = scaler.inverse_transform(sk_t_X)

    assert_allclose(t_X, sk_t_X)
    assert_allclose(r_X, sk_r_X)


@pytest.mark.parametrize("with_scaling", [True, False])
@pytest.mark.parametrize("quantile_range", [(25.0, 75.0), (10.0, 90.0)])
def test_robust_scaler_sparse(
    failure_logger,
    sparse_clf_dataset,  # noqa: F811
    with_scaling,
    quantile_range,
):
    X_np, X = sparse_clf_dataset

    if X.format != "csc":
        X = X.tocsc()

    scaler = cuRobustScaler(
        with_centering=False,
        with_scaling=with_scaling,
        quantile_range=quantile_range,
        copy=True,
    )
    t_X = scaler.fit_transform(X)
    r_X = scaler.inverse_transform(t_X)
    #  assert type(t_X) is type(X)
    #  assert type(r_X) is type(t_X)
    if cpx.scipy.sparse.issparse(X):
        assert cpx.scipy.sparse.issparse(t_X)
    if scipy.sparse.issparse(X):
        assert scipy.sparse.issparse(t_X)
    if cpx.scipy.sparse.issparse(t_X):
        assert cpx.scipy.sparse.issparse(r_X)
    if scipy.sparse.issparse(t_X):
        assert scipy.sparse.issparse(r_X)

    scaler = skRobustScaler(
        with_centering=False,
        with_scaling=with_scaling,
        quantile_range=quantile_range,
        copy=True,
    )
    sk_t_X = scaler.fit_transform(X_np)
    sk_r_X = scaler.inverse_transform(sk_t_X)

    assert_allclose(t_X, sk_t_X)
    assert_allclose(r_X, sk_r_X)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("with_centering", [True, False])
@pytest.mark.parametrize("with_scaling", [True, False])
@pytest.mark.parametrize("quantile_range", [(25.0, 75.0), (10.0, 90.0)])
def test_robust_scale(
    failure_logger,
    clf_dataset,  # noqa: F811
    with_centering,
    axis,
    with_scaling,
    quantile_range,
):
    X_np, X = clf_dataset

    t_X = cu_robust_scale(
        X,
        axis=axis,
        with_centering=with_centering,
        with_scaling=with_scaling,
        quantile_range=quantile_range,
        copy=True,
    )
    assert type(t_X) is type(X)

    sk_t_X = sk_robust_scale(
        X_np,
        axis=axis,
        with_centering=with_centering,
        with_scaling=with_scaling,
        quantile_range=quantile_range,
        copy=True,
    )

    assert_allclose(t_X, sk_t_X)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("with_scaling", [True, False])
@pytest.mark.parametrize("quantile_range", [(25.0, 75.0), (10.0, 90.0)])
def test_robust_scale_sparse(
    failure_logger,
    sparse_clf_dataset,  # noqa: F811
    axis,
    with_scaling,
    quantile_range,
):
    X_np, X = sparse_clf_dataset

    if X.format != "csc" and axis == 0:
        X = X.tocsc()
    elif X.format != "csr" and axis == 1:
        X = X.tocsr()

    t_X = cu_robust_scale(
        X,
        axis=axis,
        with_centering=False,
        with_scaling=with_scaling,
        quantile_range=quantile_range,
        copy=True,
    )
    #  assert type(t_X) is type(X)
    if cpx.scipy.sparse.issparse(X):
        assert cpx.scipy.sparse.issparse(t_X)
    if scipy.sparse.issparse(X):
        assert scipy.sparse.issparse(t_X)

    sk_t_X = sk_robust_scale(
        X_np,
        axis=axis,
        with_centering=False,
        with_scaling=with_scaling,
        quantile_range=quantile_range,
        copy=True,
    )

    assert_allclose(t_X, sk_t_X)


@pytest.mark.parametrize("n_bins", [5, 20])
@pytest.mark.parametrize("encode", ["ordinal", "onehot-dense", "onehot"])
@pytest.mark.parametrize(
    "strategy",
    [
        "uniform",
        "quantile",
        "kmeans",
    ],
)
def test_kbinsdiscretizer(
    failure_logger,
    blobs_dataset,
    n_bins,
    encode,
    strategy,  # noqa: F811
):
    X_np, X = blobs_dataset

    transformer = cuKBinsDiscretizer(
        n_bins=n_bins, encode=encode, strategy=strategy
    )
    t_X = transformer.fit_transform(X)
    r_X = transformer.inverse_transform(t_X)

    if encode != "onehot":
        assert type(t_X) is type(X)
        assert type(r_X) is type(t_X)

    sklearn_kwargs = {}
    if strategy == "quantile" and Version(sklearn.__version__) >= Version(
        "1.7"
    ):
        # cuML uses linear percentile interpolation. Scikit-learn exposed the
        # method in 1.7 and changed its default in 1.9.
        sklearn_kwargs["quantile_method"] = "linear"

    transformer = skKBinsDiscretizer(
        n_bins=n_bins,
        encode=encode,
        strategy=strategy,
        # Set subsample explicitly for parity with cuML's configured default.
        subsample=200_000
        if strategy in ("uniform", "quantile", "kmeans")
        else None,
        **sklearn_kwargs,
    )
    sk_t_X = transformer.fit_transform(X_np)
    sk_r_X = transformer.inverse_transform(sk_t_X)

    if strategy == "kmeans":
        assert_allclose(t_X, sk_t_X, ratio_tol=0.2)
    else:
        assert_allclose(t_X, sk_t_X)
        assert_allclose(r_X, sk_r_X)


@pytest.mark.parametrize(
    "X_fit,X_transform,n_bins",
    [
        pytest.param(
            [0.0, 1.0],
            [0.499999],
            2,
            id="value-below-bin-edge",
        ),
        # These float32 values put X_transform between edges produced when
        # CuPy receives a NumPy integer or a Python integer for ``num``.
        pytest.param(
            [-11.525976, 9.953962],
            [1.3619866],
            20,
            id="float32-linspace-num-promotion",
        ),
    ],
)
def test_kbinsdiscretizer_uniform_edge_parity(X_fit, X_transform, n_bins):
    X_fit = np.asarray(X_fit, dtype=np.float32)[:, None]
    X_transform = np.asarray(X_transform, dtype=np.float32)[:, None]

    cu_transformer = cuKBinsDiscretizer(
        n_bins=n_bins, encode="ordinal", strategy="uniform"
    ).fit(cp.asarray(X_fit))
    sk_transformer = skKBinsDiscretizer(
        n_bins=n_bins, encode="ordinal", strategy="uniform"
    ).fit(X_fit)

    assert_allclose(
        cu_transformer.transform(cp.asarray(X_transform)),
        sk_transformer.transform(X_transform),
    )


@pytest.mark.parametrize("missing_values", [0, 1, np.nan])
@pytest.mark.parametrize("features", ["missing-only", "all"])
def test_missing_indicator(
    failure_logger,
    int_dataset,
    missing_values,
    features,  # noqa: F811
):
    zero_filled, one_filled, nan_filled = int_dataset
    if missing_values == 0:
        X_np, X = zero_filled
    elif missing_values == 1:
        X_np, X = one_filled
    else:
        X_np, X = nan_filled

    indicator = cuMissingIndicator(
        missing_values=missing_values, features=features
    )
    ft_X = indicator.fit_transform(X)
    assert type(ft_X) is type(X)
    indicator.fit(X)
    t_X = indicator.transform(X)
    assert type(t_X) is type(X)

    indicator = skMissingIndicator(
        missing_values=missing_values, features=features
    )
    sk_ft_X = indicator.fit_transform(X_np)
    indicator.fit(X_np)
    sk_t_X = indicator.transform(X_np)

    assert_allclose(ft_X, sk_ft_X)
    assert_allclose(t_X, sk_t_X)


@pytest.mark.parametrize("features", ["missing-only", "all"])
def test_missing_indicator_sparse(
    failure_logger,
    sparse_int_dataset,
    features,  # noqa: F811
):
    X_np, X = sparse_int_dataset

    indicator = cuMissingIndicator(features=features, missing_values=1)
    ft_X = indicator.fit_transform(X)
    # assert type(ft_X) == type(X)
    assert cpx.scipy.sparse.issparse(ft_X) or scipy.sparse.issparse(ft_X)
    indicator.fit(X)
    t_X = indicator.transform(X)
    # assert type(t_X) is type(X)
    assert cpx.scipy.sparse.issparse(t_X) or scipy.sparse.issparse(t_X)

    indicator = skMissingIndicator(features=features, missing_values=1)
    sk_ft_X = indicator.fit_transform(X_np)
    indicator.fit(X_np)
    sk_t_X = indicator.transform(X_np)

    assert_allclose(ft_X, sk_ft_X)
    assert_allclose(t_X, sk_t_X)


@pytest.mark.filterwarnings(
    "ignore:X does not have valid feature names:UserWarning"
)
def test_function_transformer(clf_dataset):  # noqa: F811
    X_np, X = clf_dataset

    transformer = cuFunctionTransformer(
        func=cp.exp, inverse_func=cp.log, check_inverse=False
    )
    t_X = transformer.fit_transform(X)
    r_X = transformer.inverse_transform(t_X)
    assert type(t_X) is type(X)
    assert type(r_X) is type(t_X)

    transformer = skFunctionTransformer(
        func=np.exp, inverse_func=np.log, check_inverse=False
    )
    sk_t_X = transformer.fit_transform(X_np)
    sk_r_X = transformer.inverse_transform(sk_t_X)

    assert_allclose(t_X, sk_t_X)
    assert_allclose(r_X, sk_r_X)


def test_function_transformer_sparse(sparse_clf_dataset):  # noqa: F811
    X_np, X = sparse_clf_dataset

    assert not cuFunctionTransformer().__sklearn_tags__().input_tags.sparse

    transformer = cuFunctionTransformer(
        func=lambda x: x * 2, inverse_func=lambda x: x / 2, accept_sparse=True
    )
    assert transformer.__sklearn_tags__().input_tags.sparse

    t_X = transformer.fit_transform(X)
    r_X = transformer.inverse_transform(t_X)
    assert cpx.scipy.sparse.issparse(t_X) or scipy.sparse.issparse(t_X)
    assert cpx.scipy.sparse.issparse(r_X) or scipy.sparse.issparse(r_X)

    transformer = skFunctionTransformer(
        func=lambda x: x * 2, inverse_func=lambda x: x / 2, accept_sparse=True
    )
    sk_t_X = transformer.fit_transform(X_np)
    sk_r_X = transformer.inverse_transform(sk_t_X)

    assert_allclose(t_X, sk_t_X)
    assert_allclose(r_X, sk_r_X)


@pytest.mark.parametrize("Estimator", [cuNormalizer, cuBinarizer])
def test_stateless_transformer_tags(Estimator):
    tags = Estimator().__sklearn_tags__()
    assert not tags.requires_fit
    assert tags.input_tags.sparse


@pytest.mark.filterwarnings(
    "ignore:'ignore_implicit_zeros' takes effect only with sparse matrix.*:UserWarning"
)
@pytest.mark.filterwarnings(
    "ignore:X does not have valid feature names:UserWarning"
)
@pytest.mark.parametrize("n_quantiles", [30, 100])
@pytest.mark.parametrize("output_distribution", ["uniform", "normal"])
@pytest.mark.parametrize("ignore_implicit_zeros", [False, True])
@pytest.mark.parametrize("subsample", [100])
def test_quantile_transformer(
    failure_logger,
    nan_filled_positive,  # noqa: F811
    n_quantiles,
    output_distribution,
    ignore_implicit_zeros,
    subsample,
):
    X_np, X = nan_filled_positive

    transformer = cuQuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution=output_distribution,
        ignore_implicit_zeros=ignore_implicit_zeros,
        subsample=subsample,
        random_state=42,
        copy=True,
    )
    t_X = transformer.fit_transform(X)
    assert type(t_X) is type(X)
    r_X = transformer.inverse_transform(t_X)
    assert type(r_X) is type(t_X)

    quantiles_ = transformer.quantiles_
    references_ = transformer.references_

    transformer = skQuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution=output_distribution,
        ignore_implicit_zeros=ignore_implicit_zeros,
        subsample=subsample,
        random_state=42,
        copy=True,
    )
    sk_t_X = transformer.fit_transform(X_np)
    sk_r_X = transformer.inverse_transform(sk_t_X)

    sk_quantiles_ = transformer.quantiles_
    sk_references_ = transformer.references_

    assert_allclose(quantiles_, sk_quantiles_)
    assert_allclose(references_, sk_references_)

    assert_allclose(t_X, sk_t_X)
    assert_allclose(r_X, sk_r_X)


@pytest.mark.parametrize("n_quantiles", [30, 100])
@pytest.mark.parametrize("output_distribution", ["uniform", "normal"])
@pytest.mark.parametrize("ignore_implicit_zeros", [False, True])
@pytest.mark.parametrize("subsample", [100])
def test_quantile_transformer_sparse(
    failure_logger,
    sparse_nan_filled_positive,  # noqa: F811
    n_quantiles,
    output_distribution,
    ignore_implicit_zeros,
    subsample,
):
    X_np, X = sparse_nan_filled_positive
    X_np = X_np.tocsc()
    X = X.tocsr().tocsc()

    transformer = cuQuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution=output_distribution,
        ignore_implicit_zeros=ignore_implicit_zeros,
        subsample=subsample,
        random_state=42,
        copy=True,
    )
    t_X = transformer.fit_transform(X)
    t_X = t_X.tocsc()
    r_X = transformer.inverse_transform(t_X)

    if cpx.scipy.sparse.issparse(X):
        assert cpx.scipy.sparse.issparse(t_X)
    if scipy.sparse.issparse(X):
        assert scipy.sparse.issparse(t_X)

    quantiles_ = transformer.quantiles_
    references_ = transformer.references_

    transformer = skQuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution=output_distribution,
        ignore_implicit_zeros=ignore_implicit_zeros,
        subsample=subsample,
        random_state=42,
        copy=True,
    )
    sk_t_X = transformer.fit_transform(X_np)
    sk_r_X = transformer.inverse_transform(sk_t_X)

    sk_quantiles_ = transformer.quantiles_
    sk_references_ = transformer.references_

    assert_allclose(quantiles_, sk_quantiles_)
    assert_allclose(references_, sk_references_)

    assert_allclose(t_X, sk_t_X)
    assert_allclose(r_X, sk_r_X)


@pytest.mark.filterwarnings(
    "ignore:'ignore_implicit_zeros' takes effect only with sparse matrix.*:UserWarning"
)
@pytest.mark.filterwarnings(
    "ignore:n_quantiles .* is greater than the total number of samples.*:UserWarning"
)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("n_quantiles", [30, 100])
@pytest.mark.parametrize("output_distribution", ["uniform", "normal"])
@pytest.mark.parametrize("ignore_implicit_zeros", [False, True])
@pytest.mark.parametrize("subsample", [100])
def test_quantile_transform(
    failure_logger,
    nan_filled_positive,  # noqa: F811
    axis,
    n_quantiles,
    output_distribution,
    ignore_implicit_zeros,
    subsample,
):
    X_np, X = nan_filled_positive

    t_X = cu_quantile_transform(
        X,
        axis=axis,
        n_quantiles=n_quantiles,
        output_distribution=output_distribution,
        ignore_implicit_zeros=ignore_implicit_zeros,
        subsample=subsample,
        random_state=42,
        copy=True,
    )
    assert type(t_X) is type(X)

    sk_t_X = sk_quantile_transform(
        X_np,
        axis=axis,
        n_quantiles=n_quantiles,
        output_distribution=output_distribution,
        ignore_implicit_zeros=ignore_implicit_zeros,
        subsample=subsample,
        random_state=42,
        copy=True,
    )

    assert_allclose(t_X, sk_t_X)


@pytest.mark.parametrize("method", ["yeo-johnson", "box-cox"])
@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.filterwarnings(
    "ignore:X does not have valid feature names:UserWarning"
)
def test_power_transformer(
    failure_logger,
    nan_filled_positive,
    method,
    standardize,  # noqa: F811
):
    X_np, X = nan_filled_positive

    transformer = cuPowerTransformer(
        method=method, standardize=standardize, copy=True
    )
    ft_X = transformer.fit_transform(X)
    assert type(ft_X) is type(X)
    t_X = transformer.transform(X)
    assert type(t_X) is type(X)
    r_X = transformer.inverse_transform(t_X)
    assert type(r_X) is type(t_X)

    normalizer = skPowerTransformer(
        method=method, standardize=standardize, copy=True
    )
    sk_t_X = normalizer.fit_transform(X_np)
    sk_r_X = transformer.inverse_transform(sk_t_X)

    assert_allclose(ft_X, sk_t_X)
    assert_allclose(t_X, sk_t_X)
    assert_allclose(r_X, sk_r_X)


@pytest.mark.parametrize("method", ["yeo-johnson", "box-cox"])
@pytest.mark.parametrize("standardize", [False, True])
def test_power_transform(
    failure_logger,
    nan_filled_positive,
    method,
    standardize,  # noqa: F811
):
    X_np, X = nan_filled_positive

    t_X = cu_power_transform(X, method=method, standardize=standardize)
    assert type(t_X) is type(X)

    sk_t_X = sk_power_transform(X_np, method=method, standardize=standardize)

    assert_allclose(t_X, sk_t_X)


def test_kernel_centerer():
    X = np.array([[1.0, -2.0, 2.0], [-2.0, 1.0, 3.0], [4.0, 1.0, -2.0]])
    K = pairwise_kernels(X, metric="linear")

    model = cuKernelCenterer()
    model.fit(K)
    t_X = model.transform(K, copy=True)
    assert type(t_X) is type(X)

    model = skKernelCenterer()
    sk_t_X = model.fit_transform(K)

    assert_allclose(sk_t_X, t_X)


def test_label_binarize():
    cu_bin = cu_label_binarize(
        cp.array([1, 0, 1, 1]), classes=cp.array([0, 1])
    )
    sk_bin = sk_label_binarize([1, 0, 1, 1], classes=[0, 1])
    assert_allclose(cu_bin, sk_bin)

    cu_bin_sparse = cu_label_binarize(
        cp.array([1, 0, 1, 1]), classes=cp.array([0, 1]), sparse_output=True
    )
    sk_bin_sparse = sk_label_binarize(
        [1, 0, 1, 1], classes=[0, 1], sparse_output=True
    )
    assert_allclose(cu_bin_sparse, sk_bin_sparse)

    cu_multi = cu_label_binarize(
        cp.array([1, 6, 3]), classes=cp.array([1, 3, 4, 6])
    )
    sk_multi = sk_label_binarize([1, 6, 3], classes=[1, 3, 4, 6])
    assert_allclose(cu_multi, sk_multi)

    cu_multi_sparse = cu_label_binarize(
        cp.array([1, 6, 3]), classes=cp.array([1, 3, 4, 6]), sparse_output=True
    )
    sk_multi_sparse = sk_label_binarize(
        [1, 6, 3], classes=[1, 3, 4, 6], sparse_output=True
    )
    assert_allclose(cu_multi_sparse, sk_multi_sparse)


def test__repr__():
    assert cuBinarizer().__repr__() == "Binarizer()"
    assert cuFunctionTransformer().__repr__() == "FunctionTransformer()"
    assert cuKBinsDiscretizer().__repr__() == "KBinsDiscretizer()"
    assert cuKernelCenterer().__repr__() == "KernelCenterer()"
    assert cuMaxAbsScaler().__repr__() == "MaxAbsScaler()"
    assert cuMinMaxScaler().__repr__() == "MinMaxScaler()"
    assert cuMissingIndicator().__repr__() == "MissingIndicator()"
    assert cuNormalizer().__repr__() == "Normalizer()"
    assert cuPolynomialFeatures().__repr__() == "PolynomialFeatures()"
    assert cuQuantileTransformer().__repr__() == "QuantileTransformer()"
    assert cuRobustScaler().__repr__() == "RobustScaler()"
    assert cuSimpleImputer().__repr__() == "SimpleImputer()"
    assert cuStandardScaler().__repr__() == "StandardScaler()"


@pytest.mark.parametrize(
    "cu_cls, sk_cls, params",
    [
        (cuMinMaxScaler, skMinMaxScaler, {}),
        (cuMaxAbsScaler, skMaxAbsScaler, {}),
        (cuRobustScaler, skRobustScaler, {}),
        (cuStandardScaler, skStandardScaler, {}),
        (cuQuantileTransformer, skQuantileTransformer, {"n_quantiles": 10}),
        (cuPowerTransformer, skPowerTransformer, {}),
        (cuNormalizer, skNormalizer, {}),
        (cuBinarizer, skBinarizer, {}),
    ],
)
def test_get_feature_names_out(cu_cls, sk_cls, params):
    iris = sklearn.datasets.load_iris()
    cu_model = cu_cls(**params).fit(iris.data)
    sk_model = sk_cls(**params).fit(iris.data)

    res = cu_model.get_feature_names_out()
    sol = sk_model.get_feature_names_out()
    np.testing.assert_array_equal(res, sol)

    res = cu_model.get_feature_names_out(iris.feature_names)
    sol = sk_model.get_feature_names_out(iris.feature_names)
    np.testing.assert_array_equal(res, sol)


def test_kernel_centerer_get_feature_names_out():
    from sklearn.metrics.pairwise import linear_kernel

    rng = np.random.RandomState(0)
    X = rng.random_sample((6, 4))
    X_pairwise = linear_kernel(X)

    cu_model = cuKernelCenterer().fit(X_pairwise)
    sk_model = skKernelCenterer().fit(X_pairwise)
    res = cu_model.get_feature_names_out()
    sol = sk_model.get_feature_names_out()
    np.testing.assert_array_equal(res, sol)


@pytest.mark.parametrize(
    "encode",
    [
        "onehot",
        "onehot-dense",
        "ordinal",
    ],
)
def test_kbins_discretizer_get_feature_names_out(encode):
    X = np.array([[-2, 1, -4], [-1, 2, -3], [0, 3, -2], [1, 4, -1]])

    cu_model = cuKBinsDiscretizer(n_bins=4, encode=encode).fit(X)
    sk_model = skKBinsDiscretizer(n_bins=4, encode=encode).fit(X)
    res = cu_model.get_feature_names_out()
    sol = sk_model.get_feature_names_out()
    np.testing.assert_array_equal(res, sol)
