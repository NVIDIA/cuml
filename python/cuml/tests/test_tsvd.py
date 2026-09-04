# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.decomposition import TruncatedSVD as skTSVD
from sklearn.utils import check_random_state

from cuml import TruncatedSVD as cuTSVD
from cuml.testing.utils import (
    array_equal,
    quality_param,
    stress_param,
    unit_param,
)


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "name", [unit_param(None), quality_param("random"), stress_param("blobs")]
)
def test_tsvd_fit(datatype, name):
    if name == "blobs":
        X, y = make_blobs(n_samples=500000, n_features=1000, random_state=0)

    elif name == "random":
        pytest.skip(
            "fails when using random dataset used by sklearn for testing"
        )
        shape = 5000, 100
        rng = check_random_state(42)
        X = rng.randint(-100, 20, np.product(shape)).reshape(shape)

    else:
        n, p = 500, 5
        rng = np.random.RandomState(0)
        X = rng.randn(n, p) * 0.1 + np.array([3, 4, 2, 3, 5])

    if name != "blobs":
        sktsvd = skTSVD(n_components=1)
        sktsvd.fit(X)

    cutsvd = cuTSVD(n_components=1)

    cutsvd.fit(X)

    if name != "blobs":
        for attr in [
            "singular_values_",
            "components_",
            "explained_variance_ratio_",
        ]:
            assert array_equal(
                getattr(cutsvd, attr),
                getattr(sktsvd, attr),
                0.4,
                with_sign=True,
            )


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "name", [unit_param(None), quality_param("random"), stress_param("blobs")]
)
def test_tsvd_fit_transform(datatype, name):
    if name == "blobs":
        X, y = make_blobs(n_samples=500000, n_features=1000, random_state=0)

    elif name == "random":
        pytest.skip(
            "fails when using random dataset used by sklearn for testing"
        )
        shape = 5000, 100
        rng = check_random_state(42)
        X = rng.randint(-100, 20, np.product(shape)).reshape(shape)

    else:
        n, p = 500, 5
        rng = np.random.RandomState(0)
        X = rng.randn(n, p) * 0.1 + np.array([3, 4, 2, 3, 5])

    if name != "blobs":
        skpca = skTSVD(n_components=1)
        Xsktsvd = skpca.fit_transform(X)

    cutsvd = cuTSVD(n_components=1)

    Xcutsvd = cutsvd.fit_transform(X)

    if name != "blobs":
        assert array_equal(Xcutsvd, Xsktsvd, 1e-3, with_sign=True)


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "name", [unit_param(None), quality_param("random"), stress_param("blobs")]
)
def test_tsvd_inverse_transform(datatype, name):
    if name == "blobs":
        pytest.skip("fails when using blobs dataset")
        X, y = make_blobs(n_samples=500000, n_features=1000, random_state=0)

    elif name == "random":
        pytest.skip(
            "fails when using random dataset used by sklearn for testing"
        )
        shape = 5000, 100
        rng = check_random_state(42)
        X = rng.randint(-100, 20, np.product(shape)).reshape(shape)

    else:
        n, p = 500, 5
        rng = np.random.RandomState(0)
        X = rng.randn(n, p) * 0.1 + np.array([3, 4, 2, 3, 5])

    cutsvd = cuTSVD(n_components=1)
    Xcutsvd = cutsvd.fit_transform(X)
    input_gdf = cutsvd.inverse_transform(Xcutsvd)

    assert array_equal(input_gdf, X, 0.4, with_sign=True)


def test_get_feature_names_out():
    X, _ = make_blobs(n_features=5)
    cu_model = cuTSVD(n_components=2).fit(X)
    sk_model = skTSVD(n_components=2).fit(X)
    res = cu_model.get_feature_names_out()
    sol = sk_model.get_feature_names_out()
    np.testing.assert_array_equal(res, sol)
