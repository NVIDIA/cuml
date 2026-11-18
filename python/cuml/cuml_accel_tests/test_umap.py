# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import version

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from packaging.version import Version
from scipy import sparse
from sklearn.base import clone
from umap import UMAP

from cuml.manifold import UMAP as GPUUMAP

UMAP_060 = Version(version("umap-learn")) >= Version("0.6.0")
pytestmark = pytest.mark.filterwarnings(
    "ignore:compatibility_layout was not set:FutureWarning"
)


@pytest.fixture
def data():
    return np.random.default_rng(42).normal(size=(50, 4)).astype("float32")


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "int32"])
@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_umap_gpu_dispatch_and_dtype(data, dtype, method):
    X = data.astype(dtype)
    model = UMAP(n_neighbors=5, n_epochs=10, random_state=42)
    result = getattr(model, method)(X)
    assert model._gpu is not None
    expected = (
        X.dtype if UMAP_060 and X.dtype.kind == "f" else np.dtype("float32")
    )
    if method == "fit_transform":
        assert result.dtype == expected
    assert model.embedding_.dtype == np.float32
    assert model.transform(X).dtype == expected
    assert model.transform(X[:10] + 1).dtype == expected
    assert clone(model).get_params() == model.get_params()


@pytest.mark.skipif(not UMAP_060, reason="Requires UMAP 0.6")
@pytest.mark.parametrize("layout", [None, True, np.bool_(True)])
@pytest.mark.parametrize("init", ["auto", "spectral", "random"])
def test_umap_compatibility_layout(data, layout, init):
    model = UMAP(
        compatibility_layout=layout, init=init, n_epochs=10, random_state=42
    ).fit(data)
    assert model._gpu is not None
    assert model._gpu.init == ("spectral" if init == "auto" else init)
    assert model.get_params()["init"] == init


@pytest.mark.skipif(not UMAP_060, reason="Requires UMAP 0.6")
@pytest.mark.parametrize(
    "params",
    [
        {"compatibility_layout": False},
        {"optimizer": "adam"},
        {"optimizer": "momentum"},
        {"init": "recursive"},
        {"negative_selection_range": 10},
        {"negative_sample_scale": 2.0},
        {"exclude_graph_neighbors": True},
        {"negative_sample_scale_adaptation_samples": 16},
        {"recursive_coarsening_ratio": 2},
        {"nearest_neighbor_backend": "pynndescent"},
        {"coarse_layout_weights": "fuzzy_union"},
        {"coarsen_on_layout_weights": False},
        {"init_rescale": "per_axis"},
        {"init_scale_factor": 2.0},
        {"recursive_init_kwds": {}},
    ],
)
def test_umap_new_options_fall_back(data, params):
    # An explicit supported init must not hide a modern optimizer request.
    params = {"init": "spectral", **params}
    model = UMAP(n_epochs=10, n_jobs=1, random_state=42, **params).fit(data)
    assert model._gpu is None
    for key, value in params.items():
        assert model.get_params()[key] == value


@pytest.mark.parametrize("sparse_input", [False, True])
def test_umap_cpu_gpu_hash_roundtrip(data, sparse_input):
    X = sparse.csr_matrix(data) if sparse_input else data
    params = {"compatibility_layout": True} if UMAP_060 else {}
    cpu = UMAP._cpu_class(
        n_epochs=10, n_jobs=1, random_state=42, **params
    ).fit(X)
    gpu = GPUUMAP.from_sklearn(cpu)
    assert_array_equal(gpu.transform(X), cpu.embedding_)
    restored = gpu.as_sklearn()
    assert_array_equal(restored.transform(X), cpu.embedding_)
    if UMAP_060:
        assert restored.compatibility_layout is True
        assert restored._input_hash_algorithm is None
    # Synchronizing into a previously CPU-fit model must clear its BLAKE2 marker.
    gpu._sync_attrs_to_cpu(cpu)
    assert_array_equal(cpu.transform(X), restored.embedding_)
