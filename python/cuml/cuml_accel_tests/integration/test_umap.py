#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import pickle
from importlib.metadata import version

import joblib
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from packaging.version import Version
from scipy import sparse
from sklearn.base import clone
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import trustworthiness
from umap import UMAP

from cuml.accel.profilers import profile
from cuml.manifold import UMAP as GPUUMAP

UMAP_060 = Version(version("umap-learn")) >= Version("0.6.0")


@pytest.fixture(scope="module")
def manifold_data():
    X, _ = make_swiss_roll(n_samples=100, noise=0.05, random_state=42)
    return X


@pytest.mark.parametrize("n_neighbors", [5])
def test_umap_n_neighbors(manifold_data, n_neighbors):
    X = manifold_data
    umap = UMAP(n_neighbors=n_neighbors, random_state=42)
    X_embedded = umap.fit_transform(X)
    trust = trustworthiness(X, X_embedded, n_neighbors=5)
    print(f"Trustworthiness with n_neighbors={n_neighbors}: {trust}")


@pytest.mark.parametrize("min_dist", [0.0, 0.5])
def test_umap_min_dist(manifold_data, min_dist):
    X = manifold_data
    umap = UMAP(min_dist=min_dist, random_state=42)
    X_embedded = umap.fit_transform(X)
    trust = trustworthiness(X, X_embedded, n_neighbors=5)
    print(f"Trustworthiness with min_dist={min_dist}: {trust}")


@pytest.mark.parametrize(
    "metric",
    [
        "euclidean",
        "manhattan",
        "chebyshev",
        "cosine",
        # These metrics are currently not supported in cuml,
        # we test them here to make sure no exception is raised
        "sokalsneath",
        "rogerstanimoto",
        "sokalmichener",
        "yule",
        "ll_dirichlet",
        "russellrao",
        "kulsinski",
        "dice",
        # Require 2D data for these metrics
        "wminkowski",
        "mahalanobis",
        "haversine",
    ],
)
@pytest.mark.filterwarnings(
    "ignore:gradient function is not yet implemented:UserWarning"
)
@pytest.mark.filterwarnings(
    "ignore:n_jobs value .* overridden to .* by setting random_state:UserWarning"
)
@pytest.mark.filterwarnings(
    "ignore:overflow encountered in cast:RuntimeWarning"
)
def test_umap_metric(manifold_data, metric):
    X = manifold_data
    # haversine only works for 2D data
    if metric in ["haversine", "wminkowski", "mahalanobis"]:
        X = X[:, :2]

    umap = UMAP(metric=metric, random_state=42)
    X_embedded = umap.fit_transform(X)
    trust = trustworthiness(X, X_embedded, n_neighbors=5)
    print(f"Trustworthiness with metric={metric}: {trust}")


@pytest.mark.parametrize("n_components", [2, 3])
def test_umap_n_components(manifold_data, n_components):
    X = manifold_data
    umap = UMAP(n_components=n_components, random_state=42)
    X_embedded = umap.fit_transform(X)
    trust = trustworthiness(X, X_embedded, n_neighbors=5)
    print(f"Trustworthiness with n_components={n_components}: {trust}")


@pytest.mark.parametrize("spread", [0.5, 1.5])
def test_umap_spread(manifold_data, spread):
    X = manifold_data
    umap = UMAP(spread=spread, random_state=42)
    X_embedded = umap.fit_transform(X)
    trust = trustworthiness(X, X_embedded, n_neighbors=5)
    print(f"Trustworthiness with spread={spread}: {trust}")


@pytest.mark.parametrize("negative_sample_rate", [5])
def test_umap_negative_sample_rate(manifold_data, negative_sample_rate):
    X = manifold_data
    umap = UMAP(negative_sample_rate=negative_sample_rate, random_state=42)
    X_embedded = umap.fit_transform(X)
    trust = trustworthiness(X, X_embedded, n_neighbors=5)
    print(
        f"Trustworthiness with negative_sample_rate={negative_sample_rate}: {trust}"
    )


@pytest.mark.parametrize("learning_rate", [0.1, 10.0])
def test_umap_learning_rate(manifold_data, learning_rate):
    X = manifold_data
    umap = UMAP(learning_rate=learning_rate, random_state=42)
    X_embedded = umap.fit_transform(X)
    trust = trustworthiness(X, X_embedded, n_neighbors=5)
    print(f"Trustworthiness with learning_rate={learning_rate}: {trust}")


@pytest.mark.parametrize("init", ["spectral", "random"])
def test_umap_init(manifold_data, init):
    X = manifold_data
    umap = UMAP(init=init, random_state=42)
    X_embedded = umap.fit_transform(X)
    trust = trustworthiness(X, X_embedded, n_neighbors=5)
    print(f"Trustworthiness with init={init}: {trust}")


@pytest.mark.parametrize("n_epochs", [100, 200, 500])
def test_umap_n_epochs(manifold_data, n_epochs):
    X = manifold_data
    umap = UMAP(n_epochs=n_epochs, random_state=42)
    X_embedded = umap.fit_transform(X)
    trust = trustworthiness(X, X_embedded, n_neighbors=5)
    print(f"Trustworthiness with n_epochs={n_epochs}: {trust}")


@pytest.mark.parametrize("local_connectivity", [1, 2, 5])
def test_umap_local_connectivity(manifold_data, local_connectivity):
    X = manifold_data
    umap = UMAP(local_connectivity=local_connectivity, random_state=42)
    X_embedded = umap.fit_transform(X)
    trust = trustworthiness(X, X_embedded, n_neighbors=5)
    print(
        f"Trustworthiness with local_connectivity={local_connectivity}: {trust}"
    )


@pytest.mark.parametrize("repulsion_strength", [0.5, 1.0, 2.0])
def test_umap_repulsion_strength(manifold_data, repulsion_strength):
    X = manifold_data
    umap = UMAP(repulsion_strength=repulsion_strength, random_state=42)
    X_embedded = umap.fit_transform(X)
    trust = trustworthiness(X, X_embedded, n_neighbors=5)
    print(
        f"Trustworthiness with repulsion_strength={repulsion_strength}: {trust}"
    )


@pytest.mark.parametrize("metric_kwds", [{"p": 1}, {"p": 2}, {"p": 3}])
def test_umap_metric_kwds(manifold_data, metric_kwds):
    X = manifold_data
    umap = UMAP(metric="minkowski", metric_kwds=metric_kwds, random_state=42)
    X_embedded = umap.fit_transform(X)
    trust = trustworthiness(X, X_embedded, n_neighbors=5)
    print(f"Trustworthiness with metric_kwds={metric_kwds}: {trust}")


@pytest.mark.parametrize("angular_rp_forest", [True, False])
def test_umap_angular_rp_forest(manifold_data, angular_rp_forest):
    X = manifold_data
    umap = UMAP(angular_rp_forest=angular_rp_forest, random_state=42)
    X_embedded = umap.fit_transform(X)
    trust = trustworthiness(X, X_embedded, n_neighbors=5)
    print(
        f"Trustworthiness with angular_rp_forest={angular_rp_forest}: {trust}"
    )


@pytest.mark.parametrize("densmap", [True, False])
@pytest.mark.filterwarnings(
    "ignore:n_jobs value .* overridden to .* by setting random_state:UserWarning"
)
def test_umap_densmap(manifold_data, densmap):
    X = manifold_data
    umap = UMAP(densmap=densmap, random_state=42)
    X_embedded = umap.fit_transform(X)
    trust = trustworthiness(X, X_embedded, n_neighbors=5)
    print(f"Trustworthiness with densmap={densmap}: {trust}")


@pytest.mark.parametrize("output_metric", ["euclidean", "manhattan"])
def test_umap_output_metric(manifold_data, output_metric):
    X = manifold_data
    umap = UMAP(output_metric=output_metric, random_state=42)
    X_embedded = umap.fit_transform(X)
    trust = trustworthiness(X, X_embedded, n_neighbors=5)
    print(f"Trustworthiness with output_metric={output_metric}: {trust}")


@pytest.fixture
def data():
    return np.random.default_rng(42).normal(size=(50, 4)).astype("float32")


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "int32"])
@pytest.mark.parametrize("method", ["fit", "fit_transform"])
@pytest.mark.filterwarnings(
    "ignore:compatibility_layout was not set:FutureWarning"
)
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


@pytest.mark.parametrize("sparse_input", [False, True])
@pytest.mark.filterwarnings(
    "ignore:compatibility_layout was not set:FutureWarning"
)
def test_umap_transform_cache(data, sparse_input):
    X = sparse.csr_matrix(data) if sparse_input else data.copy()
    model = UMAP(n_neighbors=5, n_epochs=20, random_state=42).fit(X)
    embedding = model.embedding_.copy()

    with profile(quiet=True) as results:
        # Device/host conversion need not preserve Python object identity.
        assert_array_equal(model.transform(X.copy()), embedding)
        X[0, 0] += 10.0
        transformed = model.transform(X)
        assert transformed.shape == embedding.shape
        assert np.isfinite(transformed).all()
        assert not np.array_equal(transformed, embedding)
        # A transform must not replace the stored training embedding or cache.
        assert_array_equal(model.embedding_, embedding)
        original = sparse.csr_matrix(data) if sparse_input else data
        assert_array_equal(model.transform(original.copy()), embedding)

    stats = results.method_calls["UMAP.transform"]
    assert stats.gpu_calls == 3
    assert stats.cpu_calls == 0


@pytest.mark.skipif(not UMAP_060, reason="Requires UMAP 0.6")
@pytest.mark.parametrize("layout", [None, True, np.bool_(True)])
@pytest.mark.parametrize("init", ["auto", "spectral", "random"])
@pytest.mark.filterwarnings(
    "ignore:compatibility_layout was not set:FutureWarning"
)
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
    ],
)
@pytest.mark.filterwarnings(
    "ignore:compatibility_layout was not set:FutureWarning"
)
def test_umap_new_options_fall_back(data, params):
    # An explicit supported init must not hide a modern optimizer request.
    params = {"init": "spectral", **params}
    model = UMAP(n_epochs=10, n_jobs=1, random_state=42, **params).fit(data)
    assert model._gpu is None
    for key, value in params.items():
        assert model.get_params()[key] == value


@pytest.mark.skipif(not UMAP_060, reason="Requires UMAP 0.6")
@pytest.mark.parametrize(
    "params",
    [
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
        # Invalid but inactive values are deliberately not validated during GPU fitting.
        {"recursive_coarsening_ratio": 5},
        {"init_scale_factor": -1},
        {"negative_sample_scale": -1},
        {"recursive_init_kwds": {"unknown": 1}},
    ],
)
@pytest.mark.filterwarnings(
    "ignore:compatibility_layout was not set:FutureWarning"
)
def test_umap_inactive_legacy_options(data, params):
    common = dict(compatibility_layout=True, n_epochs=10, random_state=42)
    baseline = UMAP(**common).fit_transform(data)
    model = UMAP(**common, **params)
    with profile(quiet=True) as results:
        embedding = model.fit_transform(data)
    assert results.method_calls["UMAP.fit_transform"].gpu_calls == 1
    assert results.method_calls["UMAP.fit_transform"].cpu_calls == 0
    assert_array_equal(embedding, baseline)
    for key, value in params.items():
        assert model.get_params()[key] == value


@pytest.mark.parametrize("n_jobs", [0, -2])
@pytest.mark.filterwarnings(
    "ignore:compatibility_layout was not set:FutureWarning"
)
def test_umap_invalid_n_jobs(data, n_jobs):
    model = UMAP(n_jobs=n_jobs)
    with pytest.raises(ValueError, match="n_jobs"):
        model.fit(data)
    assert model._gpu is None


@pytest.mark.skipif(not UMAP_060, reason="Requires UMAP 0.6")
@pytest.mark.filterwarnings(
    "ignore:compatibility_layout was not set:FutureWarning"
)
def test_umap_invalid_neighbor_backend(data):
    model = UMAP(nearest_neighbor_backend="unknown")
    with pytest.raises(ValueError, match="nearest_neighbor_backend"):
        model.fit(data)
    assert model._gpu is None


@pytest.mark.parametrize("sparse_input", [False, True])
@pytest.mark.filterwarnings(
    "ignore:compatibility_layout was not set:FutureWarning"
)
def test_umap_cpu_gpu_hash_roundtrip(data, sparse_input):
    X = sparse.csr_matrix(data) if sparse_input else data
    params = {"compatibility_layout": True} if UMAP_060 else {}
    cpu = UMAP._cpu_class(
        n_epochs=10, n_jobs=1, random_state=42, **params
    ).fit(X)
    gpu = GPUUMAP.from_sklearn(cpu)
    transformed = gpu.transform(X)
    if UMAP_060 or not sparse_input:
        assert_array_equal(transformed, cpu.embedding_)
    else:
        # Legacy joblib hashes include CSR metadata that GPU conversion can
        # change, so even the original training input may miss the cache.
        assert transformed.shape == cpu.embedding_.shape
        assert np.isfinite(transformed).all()
    restored = gpu.as_sklearn()
    # Only UMAP 0.6 exports use content hashes; older versions' joblib hashes
    # can differ for identical CSR copies because of cached matrix metadata.
    query = X.copy() if UMAP_060 else X
    assert_array_equal(restored.transform(query), cpu.embedding_)
    if UMAP_060:
        assert restored.compatibility_layout is True
        assert restored._input_hash_algorithm == cpu._input_hash_algorithm
    # Synchronizing must preserve the fit-time hash scheme.
    gpu._sync_attrs_to_cpu(cpu)
    assert_array_equal(cpu.transform(X), restored.embedding_)


@pytest.mark.parametrize("legacy_hash", [False, True])
@pytest.mark.skipif(not UMAP_060, reason="Requires UMAP 0.6")
def test_umap_import_preserves_fit_time_hash(data, legacy_hash):
    X = data.copy()
    cpu = UMAP._cpu_class(
        compatibility_layout=True, n_epochs=20, n_jobs=1, random_state=42
    ).fit(X)
    if legacy_hash:
        cpu._input_hash = joblib.hash(cpu._raw_data)
        cpu._input_hash_algorithm = None
    digest = cpu._input_hash
    original = X.copy()
    X[0, 0] += 10
    assert np.shares_memory(cpu._raw_data, X)
    gpu = GPUUMAP.from_sklearn(cpu)
    assert gpu._input_hash == digest
    assert_array_equal(gpu.transform(original), cpu.embedding_)
    assert not np.array_equal(gpu.transform(X), cpu.embedding_)
    restored = gpu.as_sklearn()
    assert restored._input_hash == digest
    assert restored._input_hash_algorithm == cpu._input_hash_algorithm
    # Refitting must reset the imported hashing policy.
    gpu.fit(original)
    assert gpu._input_hash_algorithm == "cuml-joblib"
    assert_array_equal(gpu.transform(original), gpu.embedding_)


@pytest.mark.skipif(not UMAP_060, reason="Requires UMAP 0.6")
@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_umap_two_step_layout_change(data, method):
    model = UMAP(compatibility_layout=True, n_epochs=10, n_jobs=1).fit(data)
    model.set_params(optimizer="adam")
    assert model.get_params()["optimizer"] == "adam"
    with pytest.raises(
        ValueError, match="requires compatibility_layout=False"
    ):
        getattr(model, method)(data)
    model.set_params(compatibility_layout=False)
    getattr(model, method)(data)
    assert model._gpu is None
    assert np.isfinite(model.embedding_).all()


@pytest.mark.skipif(not UMAP_060, reason="Requires UMAP 0.6 content hashing")
@pytest.mark.parametrize("hash_input", [False, True])
def test_umap_sparse_gpu_export_content_hash(data, hash_input):
    from umap.umap_ import _INPUT_HASH_ALGORITHM

    X = sparse.csr_matrix(data)
    gpu = GPUUMAP(n_epochs=20, random_state=42, hash_input=hash_input).fit(X)
    cpu = gpu.as_sklearn()
    assert cpu._input_hash_algorithm == _INPUT_HASH_ALGORITHM
    assert_array_equal(cpu.transform(X.copy()), gpu.embedding_)
    # Conversion of unchanged data can translate the digest without changing
    # which training values it identifies.
    restored = GPUUMAP.from_sklearn(cpu).as_sklearn()
    assert_array_equal(restored.transform(X.copy()), gpu.embedding_)


def _restore_simulated_legacy_model(model):
    """Restore 0.5-shaped state, not a pickle produced by an older release."""
    state = model.__getstate__().copy()
    for name in (
        "compatibility_layout",
        "optimizer",
        "negative_selection_range",
        "negative_sample_scale",
        "exclude_graph_neighbors",
        "negative_sample_scale_adaptation_samples",
        "recursive_coarsening_ratio",
        "nearest_neighbor_backend",
        "coarse_layout_weights",
        "coarsen_on_layout_weights",
        "init_rescale",
        "init_scale_factor",
        "recursive_init_kwds",
        "_compatibility_layout",
        "_optimizer",
        "_init",
        "_data_was_uniqued",
        "_input_hash_algorithm",
    ):
        state.pop(name, None)
    state["init"] = "spectral"
    if "_raw_data" in state:
        # A real 0.5 model used joblib, not the 0.6 BLAKE2 fingerprint.
        state["_input_hash"] = joblib.hash(state["_raw_data"])
    legacy = UMAP._cpu_class.__new__(UMAP._cpu_class)
    legacy.__setstate__(state)
    return legacy


@pytest.mark.skipif(not UMAP_060, reason="Requires UMAP 0.6 migration")
@pytest.mark.parametrize("roundtrip", [False, True])
@pytest.mark.filterwarnings(
    "ignore:compatibility_layout was not set:FutureWarning"
)
def test_umap_simulated_legacy_unfitted(data, roundtrip):
    cpu = UMAP._cpu_class(n_epochs=20, n_jobs=1, random_state=42)
    legacy = _restore_simulated_legacy_model(cpu)
    model = UMAP._reconstruct_from_cpu(legacy)
    if roundtrip:
        # Only deserialize data generated by this test.
        model = pickle.loads(pickle.dumps(model))
    assert model._gpu is None
    params = model.get_params()
    assert params["init"] == "spectral"
    assert params["compatibility_layout"] is None
    embedding = model.fit_transform(data)
    assert model._gpu is not None
    assert embedding.shape == (len(data), 2)
    assert np.isfinite(embedding).all()
    assert model.get_params() == params
    assert_array_equal(model.transform(data), embedding)


@pytest.mark.skipif(not UMAP_060, reason="Requires UMAP 0.6 migration")
@pytest.mark.parametrize("sparse_input", [False, True])
@pytest.mark.filterwarnings(
    "ignore:compatibility_layout was not set:FutureWarning"
)
def test_umap_simulated_legacy_fitted(data, sparse_input):
    X = sparse.csr_matrix(data) if sparse_input else data
    query = data[:10] + np.float32(0.1)
    if sparse_input:
        query = sparse.csr_matrix(query)
    cpu = UMAP._cpu_class(
        compatibility_layout=True, n_epochs=20, n_jobs=1, random_state=42
    ).fit(X)
    legacy = _restore_simulated_legacy_model(cpu)
    assert_array_equal(legacy.transform(X), cpu.embedding_)

    # Use the same reconstruction entry point as proxy deserialization.
    model = UMAP._reconstruct_from_cpu(legacy)
    assert model._gpu is not None
    assert model.get_params() == legacy.get_params()
    assert_array_equal(model.embedding_, cpu.embedding_)
    training_transformed = model.transform(X)
    if not sparse_input:
        assert_array_equal(training_transformed, cpu.embedding_)
    else:
        # Legacy sparse fingerprints need not survive input normalization.
        assert training_transformed.shape == cpu.embedding_.shape
        assert np.isfinite(training_transformed).all()
    transformed = model.transform(query)
    assert transformed.shape == (query.shape[0], 2)
    assert np.isfinite(transformed).all()

    # Only deserialize data generated by this test. Compare GPU to GPU:
    # CPU and GPU transforms need not produce bit-identical embeddings.
    restored = pickle.loads(pickle.dumps(model))
    assert restored._gpu is not None
    assert restored.get_params() == model.get_params()
    assert_array_equal(restored.embedding_, model.embedding_)
    assert_allclose(
        restored.transform(X), training_transformed, rtol=1e-5, atol=1e-5
    )
    assert_allclose(
        restored.transform(query), transformed, rtol=1e-5, atol=1e-5
    )


@pytest.mark.filterwarnings(
    "ignore:compatibility_layout was not set:FutureWarning"
)
def test_umap_inverse_transform_outside_hull(data):
    model = UMAP(n_epochs=30, n_jobs=1, random_state=42).fit(data)
    embedding = model.embedding_
    query = np.max(embedding, axis=0) + 10 * np.ptp(embedding, axis=0)
    with profile(quiet=True) as results:
        with pytest.warns(
            UserWarning, match="outside the embedding convex hull"
        ):
            inverse = model.inverse_transform(query.reshape(1, -1))
    assert inverse.shape == (1, data.shape[1])
    assert np.isfinite(inverse).all()
    stats = results.method_calls["UMAP.inverse_transform"]
    assert stats.gpu_calls == 1
    assert stats.cpu_calls == 0


@pytest.mark.skipif(not UMAP_060, reason="Requires UMAP 0.6")
@pytest.mark.filterwarnings(
    "ignore:compatibility_layout was not set:FutureWarning"
)
def test_umap_inverse_transform_wrong_dimension(data):
    model = UMAP(n_epochs=30, n_jobs=1, random_state=42).fit(data)
    assert model._gpu is not None
    with pytest.raises(ValueError, match="same number of dimensions"):
        model.inverse_transform(np.zeros((1, 3), dtype="float32"))


@pytest.mark.skipif(not UMAP_060, reason="Requires UMAP 0.6 migration")
@pytest.mark.filterwarnings(
    "ignore:compatibility_layout was not set:FutureWarning"
)
def test_umap_simulated_legacy_inverse_transform(data):
    cpu = UMAP._cpu_class(
        compatibility_layout=True, n_epochs=20, n_jobs=1, random_state=42
    ).fit(data)
    model = UMAP._reconstruct_from_cpu(_restore_simulated_legacy_model(cpu))
    query = cpu.embedding_[:2].copy()
    inverse = model.inverse_transform(query)
    assert model._gpu is not None
    assert inverse.shape == (len(query), data.shape[1])
    assert np.isfinite(inverse).all()

    # Only deserialize data generated by this test.
    restored = pickle.loads(pickle.dumps(model))
    assert restored._gpu is not None
    assert_allclose(
        restored.inverse_transform(query), inverse, rtol=1e-5, atol=1e-5
    )
