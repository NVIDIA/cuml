# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
import numpy as np
import pytest

from cuml.dask.common.dask_arr_utils import to_dask_cudf
from cuml.testing.utils import array_equal, stress_param, unit_param


@pytest.mark.mg
@pytest.mark.parametrize(
    "data_info",
    [unit_param([1000, 20, 30]), stress_param([int(9e6), 5000, 30])],
)
@pytest.mark.parametrize("input_type", ["dataframe", "array"])
def test_tsvd_fit(data_info, input_type, client):
    # Assume at least 4GB memory
    max_gpu_memory = pytest.max_gpu_memory or 4

    nrows, ncols, n_parts = data_info
    if nrows == int(9e6) and max_gpu_memory < 48:
        if pytest.adapt_stress_test:
            nrows = nrows * max_gpu_memory // 256
            ncols = ncols * max_gpu_memory // 256
        else:
            pytest.skip(
                "Insufficient GPU memory for this test."
                "Re-run with 'CUML_ADAPT_STRESS_TESTS=True'"
            )

    from sklearn.decomposition import TruncatedSVD

    from cuml.dask.datasets import make_blobs
    from cuml.dask.decomposition import TruncatedSVD as daskTPCA

    X, _ = make_blobs(
        n_samples=nrows,
        n_features=ncols,
        centers=1,
        n_parts=n_parts,
        cluster_std=0.5,
        random_state=10,
        dtype=np.float32,
    )

    if input_type == "dataframe":
        X_train = to_dask_cudf(X)
        X_cpu = X_train.compute().to_pandas().values
    elif input_type == "array":
        X_train = X
        X_cpu = cp.asnumpy(X_train.compute())

    cutsvd = daskTPCA(n_components=5)
    cutsvd.fit(X_train)

    sktsvd = TruncatedSVD(n_components=5, algorithm="arpack")
    sktsvd.fit(X_cpu)

    all_attr = [
        "singular_values_",
        "components_",
        "explained_variance_",
        "explained_variance_ratio_",
    ]

    for attr in all_attr:
        cuml_res = getattr(cutsvd, attr)
        if type(cuml_res) is np.ndarray:
            cuml_res = cuml_res.to_numpy()
        skl_res = getattr(sktsvd, attr)
        if attr == "singular_values_":
            assert array_equal(cuml_res, skl_res, 1, with_sign=True)
        elif attr == "components_":
            assert array_equal(cuml_res, skl_res, 1e-1, with_sign=True)
        else:
            assert array_equal(cuml_res, skl_res, 1e-1, with_sign=True)


@pytest.mark.mg
@pytest.mark.parametrize(
    "data_info",
    [unit_param([1000, 20, 46]), stress_param([int(9e6), 5000, 46])],
)
def test_tsvd_fit_transform_fp32(data_info, client):
    nrows, ncols, n_parts = data_info
    from cuml.dask.datasets import make_blobs
    from cuml.dask.decomposition import TruncatedSVD as daskTPCA

    X_cudf, _ = make_blobs(
        n_samples=nrows,
        n_features=ncols,
        centers=1,
        n_parts=n_parts,
        cluster_std=1.5,
        random_state=10,
        dtype=np.float32,
    )

    cutsvd = daskTPCA(n_components=15)
    out = cutsvd.fit_transform(X_cudf)
    assert out.shape[1] == cutsvd.n_components_


@pytest.mark.mg
@pytest.mark.parametrize(
    "data_info",
    [unit_param([1000, 20, 33]), stress_param([int(9e6), 5000, 33])],
)
def test_tsvd_fit_transform_fp64(data_info, client):
    nrows, ncols, n_parts = data_info

    from cuml.dask.datasets import make_blobs
    from cuml.dask.decomposition import TruncatedSVD as daskTPCA

    X_cudf, _ = make_blobs(
        n_samples=nrows,
        n_features=ncols,
        centers=1,
        n_parts=n_parts,
        cluster_std=1.5,
        random_state=10,
        dtype=np.float64,
    )

    cutsvd = daskTPCA(n_components=15)
    out = cutsvd.fit_transform(X_cudf)
    assert out.shape[1] == cutsvd.n_components_


@pytest.mark.mg
def test_tsvd_n_components_exceeds_features(client):
    from cuml.dask.datasets import make_blobs
    from cuml.dask.decomposition import TruncatedSVD as daskTPCA

    # Create dataset with 20 features
    X, _ = make_blobs(
        n_samples=100,
        n_features=20,
        centers=1,
        n_parts=2,
        cluster_std=0.5,
        random_state=10,
        dtype=np.float32,
    )

    # Try to create TruncatedSVD with n_components > n_features (20)
    cutsvd = daskTPCA(n_components=25)

    with pytest.raises(
        RuntimeError, match=r"`n_components` \(25\) must be <= than"
    ):
        cutsvd.fit(X)
