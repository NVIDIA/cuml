# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import numpy as np
import pytest

from cuml.dask.common.dask_arr_utils import to_dask_cudf


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [1000])
@pytest.mark.parametrize("ncols", [20])
@pytest.mark.parametrize("n_parts", [67])
@pytest.mark.parametrize("input_type", ["dataframe", "array"])
def test_pca_fit(nrows, ncols, n_parts, input_type, client):
    from sklearn.decomposition import PCA

    from cuml.dask.datasets import make_blobs
    from cuml.dask.decomposition import PCA as daskPCA

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

    try:
        cupca = daskPCA(n_components=5, whiten=True)
        cupca.fit(X_train)
    except Exception as e:
        print(str(e))

    skpca = PCA(n_components=5, whiten=True, svd_solver="full")
    skpca.fit(X_cpu)

    from cuml.testing.utils import array_equal

    all_attr = [
        "singular_values_",
        "components_",
        "explained_variance_",
        "explained_variance_ratio_",
    ]

    for attr in all_attr:
        cuml_res = getattr(cupca, attr)
        if type(cuml_res) is np.ndarray:
            cuml_res = cuml_res.to_numpy()
        skl_res = getattr(skpca, attr)
        assert array_equal(cuml_res, skl_res, 1e-1, with_sign=True)


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [1000])
@pytest.mark.parametrize("ncols", [20])
@pytest.mark.parametrize("n_parts", [46])
def test_pca_fit_transform_fp32(nrows, ncols, n_parts, client):
    from cuml.dask.datasets import make_blobs
    from cuml.dask.decomposition import PCA as daskPCA

    X_cudf, _ = make_blobs(
        n_samples=nrows,
        n_features=ncols,
        centers=1,
        n_parts=n_parts,
        cluster_std=1.5,
        random_state=10,
        dtype=np.float32,
    )

    cupca = daskPCA(n_components=20, whiten=True)
    res = cupca.fit_transform(X_cudf)
    res = res.compute()
    assert res.shape[0] == nrows and res.shape[1] == 20


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [1000])
@pytest.mark.parametrize("ncols", [20])
@pytest.mark.parametrize("n_parts", [33])
def test_pca_fit_transform_fp64(nrows, ncols, n_parts, client):
    from cuml.dask.datasets import make_blobs
    from cuml.dask.decomposition import PCA as daskPCA

    X_cudf, _ = make_blobs(
        n_samples=nrows,
        n_features=ncols,
        centers=1,
        n_parts=n_parts,
        cluster_std=1.5,
        random_state=10,
        dtype=np.float64,
    )

    cupca = daskPCA(n_components=10, whiten=False)
    res = cupca.fit_transform(X_cudf)
    res = res.compute()
    assert res.shape[0] == nrows and res.shape[1] == 10


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [1000])
@pytest.mark.parametrize("ncols", [20])
@pytest.mark.parametrize("n_parts", [28])
def test_pca_fit_transform_fp32_noncomponents(nrows, ncols, n_parts, client):
    # Tests the case when n_components is not passed for MG scenarios
    from cuml.dask.datasets import make_blobs
    from cuml.dask.decomposition import PCA as daskPCA

    X_cudf, _ = make_blobs(
        n_samples=nrows,
        n_features=ncols,
        centers=1,
        n_parts=n_parts,
        cluster_std=1.5,
        random_state=10,
        dtype=np.float32,
    )

    cupca = daskPCA(whiten=False)
    res = cupca.fit_transform(X_cudf)
    res = res.compute()
    assert res.shape[0] == nrows and res.shape[1] == 20
