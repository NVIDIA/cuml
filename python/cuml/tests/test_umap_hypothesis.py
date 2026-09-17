#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""
Hypothesis-based property testing for UMAP components.

This test suite uses property-based testing to thoroughly validate:
1. Fuzzy simplicial set construction
2. Spectral initialization
3. Simplicial set embedding

All components are tested against reference UMAP implementations across a wide
range of parameters and synthetic dataset configurations.
"""

from importlib.metadata import version as package_version

import cupy as cp
import hypothesis
import numba
import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from packaging.version import Version
from scipy.sparse import csr_matrix
from sklearn.datasets import (
    make_circles,
    make_moons,
    make_s_curve,
    make_swiss_roll,
)

# Reference UMAP implementation
from umap.umap_ import find_ab_params
from umap.umap_ import fuzzy_simplicial_set as ref_fuzzy_simplicial_set
from umap.umap_ import simplicial_set_embedding as ref_simplicial_set_embedding
from umap.umap_ import spectral_layout

import cuml.datasets

# cuML implementation
from cuml.manifold import SpectralEmbedding
from cuml.manifold.umap import fuzzy_simplicial_set, simplicial_set_embedding

UMAP_VERSION = Version(package_version("umap-learn"))


# Custom dataset wrapper with concise repr to avoid printing giant arrays
class DatasetWrapper:
    """
    Wraps dataset arrays (X_np, X_cp) with metadata, providing dict-like access.
    Excludes arrays from __repr__ to prevent Hypothesis from printing large data
    in test failure messages.
    """

    def __init__(
        self, X_np, X_cp, metric, dataset_type, n_samples, n_features, **kwargs
    ):
        self.X_np = X_np
        self.X_cp = X_cp
        self.metric = metric
        self.dataset_type = dataset_type
        self.n_samples = n_samples
        self.n_features = n_features
        self.__dict__.update(kwargs)

    def __repr__(self):
        """Concise representation excluding X_np and X_cp arrays."""
        attrs = {
            k: v
            for k, v in self.__dict__.items()
            if k not in ["X_np", "X_cp"] and not k.startswith("_")
        }
        return f"DatasetWrapper({attrs})"

    def __getitem__(self, key):
        """Dict-like access to attributes."""
        return self.__dict__[key]

    def __contains__(self, key):
        """Check if attribute exists."""
        return key in self.__dict__

    def get(self, key, default=None):
        """Get attribute with optional default."""
        return self.__dict__.get(key, default)


# Hypothesis strategy for dataset generation
@st.composite
def diverse_dataset_strategy(draw):
    """Generate diverse synthetic datasets with various topologies."""
    dataset_type = draw(
        st.sampled_from(["blobs", "circles", "moons", "swiss_roll", "s_curve"])
    )
    n_samples = draw(st.integers(min_value=800, max_value=2000))

    # Cosine metric only makes sense for high-dimensional feature vectors (blobs).
    # For 2D/3D manifold datasets (circles, moons, swiss_roll, s_curve),
    # cosine distance is mathematically inappropriate and can cause NaN/Inf values
    # that hang the geodesic correlation computation.
    if dataset_type == "blobs":
        metric = draw(st.sampled_from(["euclidean", "cosine"]))
    else:
        metric = "euclidean"

    if dataset_type == "blobs":
        n_features = draw(st.integers(min_value=5, max_value=25))
        n_centers = draw(st.integers(min_value=2, max_value=6))
        cluster_std = draw(st.floats(min_value=0.5, max_value=3.0))
        random_state = draw(st.integers(min_value=0, max_value=10000))

        X_cp, y = cuml.datasets.make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=n_centers,
            cluster_std=cluster_std,
            random_state=random_state,
        )
        kwargs = {
            "n_features": n_features,
            "n_centers": n_centers,
            "cluster_std": cluster_std,
        }

    elif dataset_type == "circles":
        noise = draw(st.floats(min_value=0.03, max_value=0.15))
        factor = draw(st.floats(min_value=0.4, max_value=0.7))
        random_state = draw(st.integers(min_value=0, max_value=10000))

        X_np, y = make_circles(
            n_samples=n_samples,
            noise=noise,
            factor=factor,
            random_state=random_state,
        )
        X_cp = cp.asarray(X_np, dtype=cp.float32)
        kwargs = {"noise": noise, "factor": factor, "n_features": 2}

    elif dataset_type == "moons":
        noise = draw(st.floats(min_value=0.03, max_value=0.15))
        random_state = draw(st.integers(min_value=0, max_value=10000))

        X_np, y = make_moons(
            n_samples=n_samples, noise=noise, random_state=random_state
        )
        X_cp = cp.asarray(X_np, dtype=cp.float32)
        kwargs = {"noise": noise, "n_features": 2}

    elif dataset_type == "swiss_roll":
        noise = draw(st.floats(min_value=0.1, max_value=0.8))
        random_state = draw(st.integers(min_value=0, max_value=10000))

        X_np, y = make_swiss_roll(
            n_samples=n_samples, noise=noise, random_state=random_state
        )
        X_cp = cp.asarray(X_np, dtype=cp.float32)
        kwargs = {"noise": noise, "n_features": 3}

    else:  # s_curve
        noise = draw(st.floats(min_value=0.1, max_value=0.8))
        random_state = draw(st.integers(min_value=0, max_value=10000))

        X_np, y = make_s_curve(
            n_samples=n_samples, noise=noise, random_state=random_state
        )
        X_cp = cp.asarray(X_np, dtype=cp.float32)
        kwargs = {"noise": noise, "n_features": 3}

    # Ensure consistent numpy/cupy types
    X_np = (
        cp.asnumpy(X_cp)
        if isinstance(X_cp, cp.ndarray)
        else np.asarray(X_cp, dtype=np.float32)
    )
    X_cp = cp.asarray(X_np, dtype=cp.float32)

    return DatasetWrapper(
        X_np=X_np,
        X_cp=X_cp,
        metric=metric,
        dataset_type=dataset_type,
        n_samples=n_samples,
        **kwargs,
    )


@st.composite
def embedding_params_strategy(draw):
    """Generate valid UMAP embedding parameters (avoiding extreme combinations)."""
    # Learning rate: reasonable range avoiding extremes
    learning_rate = draw(st.floats(min_value=0.5, max_value=2.0))

    # min_dist: avoiding very small values
    min_dist = draw(st.floats(min_value=0.05, max_value=0.5))

    # spread: typically 0.5 to 2.5, must be >= min_dist
    # Cap min_value at 2.5 to ensure it doesn't exceed max_value
    spread = draw(
        st.floats(min_value=min(2.5, max(0.5, min_dist + 0.1)), max_value=2.5)
    )

    # n_epochs: reasonable range for good convergence
    n_epochs = draw(st.integers(min_value=200, max_value=500))

    # negative_sample_rate: moderate range
    negative_sample_rate = draw(st.integers(min_value=5, max_value=15))

    # gamma: avoiding very low values
    gamma = draw(st.floats(min_value=0.5, max_value=2.0))

    # init: prefer spectral for consistency
    init = draw(st.sampled_from(["spectral", "spectral", "random"]))

    # n_components: 2 or 3 for visualization
    n_components = draw(st.sampled_from([2, 3]))

    return {
        "learning_rate": learning_rate,
        "min_dist": min_dist,
        "spread": spread,
        "n_epochs": n_epochs,
        "negative_sample_rate": negative_sample_rate,
        "gamma": gamma,
        "init": init,
        "n_components": n_components,
    }


def evaluate_embedding_quality(
    X_cp,
    ref_emb,
    cu_emb,
    metric,
    compute_metrics_fn,
    procrustes_rmse_fn,
    k=15,
    mod_trust=0.10,
    mod_cont=0.15,
    mod_corr=0.25,
    mod_rel_kl=0.60,
    mod_rmse=0.20,
    sev_trust=0.20,
    sev_cont=0.25,
    sev_corr=0.50,
    sev_rel_kl=1.50,
    sev_rmse=0.40,
):
    """
    Evaluate embedding quality by comparing cuML against reference.

    Returns:
        tuple: (should_fail, fail_reason, metrics_dict)
    """
    # Evaluate metrics using helper functions
    ref_emb_cp = cp.asarray(ref_emb, dtype=cp.float32)
    cu_emb_cp = cp.asarray(cu_emb, dtype=cp.float32)

    metrics_ref = compute_metrics_fn(
        X_cp, ref_emb_cp, k=k, metric=metric, skip_topology_preservation=True
    )
    metrics_cu = compute_metrics_fn(
        X_cp, cu_emb_cp, k=k, metric=metric, skip_topology_preservation=True
    )

    # Extract key metrics
    trust_ref = metrics_ref["trustworthiness"]
    trust_cu = metrics_cu["trustworthiness"]
    cont_ref = metrics_ref["continuity"]
    cont_cu = metrics_cu["continuity"]
    sp_ref = metrics_ref["geodesic_spearman_correlation"]
    sp_cu = metrics_cu["geodesic_spearman_correlation"]
    pe_ref = metrics_ref["geodesic_pearson_correlation"]
    pe_cu = metrics_cu["geodesic_pearson_correlation"]
    xent_ref = metrics_ref["fuzzy_kl_divergence"]
    xent_cu = metrics_cu["fuzzy_kl_divergence"]
    kl_ref = metrics_ref["fuzzy_sym_kl_divergence"]
    kl_cu = metrics_cu["fuzzy_sym_kl_divergence"]

    rmse = procrustes_rmse_fn(ref_emb, cu_emb)

    # Compute deficits (positive means cuML is worse than reference)
    trust_def = max(0.0, trust_ref - trust_cu)
    cont_def = max(0.0, cont_ref - cont_cu)
    sp_def = max(0.0, sp_ref - sp_cu)
    pe_def = max(0.0, pe_ref - pe_cu)

    xent_abs_threshold = 0.001
    if abs(xent_ref) < xent_abs_threshold:
        # Use absolute difference for small reference values
        xent_rel_increase = max(0.0, xent_cu - xent_ref)
    else:
        # Use relative increase, capped at 5.0
        xent_rel_increase = min(
            5.0, max(0.0, (xent_cu - xent_ref) / abs(xent_ref))
        )

    kl_abs_threshold = 0.001
    if abs(kl_ref) < kl_abs_threshold:
        # Use absolute difference for small reference values
        kl_rel_increase = max(0.0, kl_cu - kl_ref)
    else:
        # Use relative increase, capped at 5.0
        kl_rel_increase = min(5.0, max(0.0, (kl_cu - kl_ref) / abs(kl_ref)))

    moderate_issues = []
    severe_issues = []

    # Trustworthiness
    if trust_def > sev_trust:
        severe_issues.append(
            f"trustworthiness deficit {trust_def:.3f} (cu={trust_cu:.4f}, ref={trust_ref:.4f})"
        )
    elif trust_def > mod_trust:
        moderate_issues.append(
            f"trustworthiness deficit {trust_def:.3f} (cu={trust_cu:.4f}, ref={trust_ref:.4f})"
        )

    # Continuity
    if cont_def > sev_cont:
        severe_issues.append(
            f"continuity deficit {cont_def:.3f} (cu={cont_cu:.4f}, ref={cont_ref:.4f})"
        )
    elif cont_def > mod_cont:
        moderate_issues.append(
            f"continuity deficit {cont_def:.3f} (cu={cont_cu:.4f}, ref={cont_ref:.4f})"
        )

    # Geodesic correlations
    if sp_def > sev_corr:
        severe_issues.append(
            f"spearman correlation deficit {sp_def:.3f} (cu={sp_cu:.4f}, ref={sp_ref:.4f})"
        )
    elif sp_def > mod_corr:
        moderate_issues.append(
            f"spearman correlation deficit {sp_def:.3f} (cu={sp_cu:.4f}, ref={sp_ref:.4f})"
        )

    if pe_def > sev_corr:
        severe_issues.append(
            f"pearson correlation deficit {pe_def:.3f} (cu={pe_cu:.4f}, ref={pe_ref:.4f})"
        )
    elif pe_def > mod_corr:
        moderate_issues.append(
            f"pearson correlation deficit {pe_def:.3f} (cu={pe_cu:.4f}, ref={pe_ref:.4f})"
        )

    # Fuzzy KL and symmetric KL (relative change)
    if xent_rel_increase > sev_rel_kl:
        severe_issues.append(
            f"fuzzy KL relative increase {xent_rel_increase:.3f} (cu={xent_cu:.4e}, ref={xent_ref:.4e})"
        )
    elif xent_rel_increase > mod_rel_kl:
        moderate_issues.append(
            f"fuzzy KL relative increase {xent_rel_increase:.3f} (cu={xent_cu:.4e}, ref={xent_ref:.4e})"
        )

    if kl_rel_increase > sev_rel_kl:
        severe_issues.append(
            f"symmetric KL relative increase {kl_rel_increase:.3f} (cu={kl_cu:.4e}, ref={kl_ref:.4e})"
        )
    elif kl_rel_increase > mod_rel_kl:
        moderate_issues.append(
            f"symmetric KL relative increase {kl_rel_increase:.3f} (cu={kl_cu:.4e}, ref={kl_ref:.4e})"
        )

    # Procrustes RMSE
    if rmse > sev_rmse:
        severe_issues.append(f"rmse {rmse:.3f} > {sev_rmse:.3f}")
    elif rmse > mod_rmse:
        moderate_issues.append(f"rmse {rmse:.3f} > {mod_rmse:.3f}")

    # Holistic decision rule:
    # - Fail on any severe issue
    # - Or if many moderate degradations accumulate
    should_fail = False
    fail_reason = ""

    if len(severe_issues) >= 1:
        should_fail = True
        fail_reason = f"Severe issues detected: {severe_issues}"
    elif len(moderate_issues) >= 5:
        should_fail = True
        fail_reason = f"Too many moderate issues: {moderate_issues}"

    metrics_dict = {
        "trust_ref": trust_ref,
        "trust_cu": trust_cu,
        "trust_def": trust_def,
        "cont_ref": cont_ref,
        "cont_cu": cont_cu,
        "cont_def": cont_def,
        "sp_ref": sp_ref,
        "sp_cu": sp_cu,
        "sp_def": sp_def,
        "pe_ref": pe_ref,
        "pe_cu": pe_cu,
        "pe_def": pe_def,
        "xent_ref": xent_ref,
        "xent_cu": xent_cu,
        "xent_rel_increase": xent_rel_increase,
        "kl_ref": kl_ref,
        "kl_cu": kl_cu,
        "kl_rel_increase": kl_rel_increase,
        "rmse": rmse,
        "moderate_issues": moderate_issues,
        "severe_issues": severe_issues,
    }

    return should_fail, fail_reason, metrics_dict


def _generate_baseline_dataset():
    """Generate a baseline dataset for the required example."""
    X_cp, y = cuml.datasets.make_blobs(
        n_samples=300,
        n_features=10,
        centers=3,
        cluster_std=1.0,
        random_state=42,
    )
    X_np = cp.asnumpy(X_cp) if isinstance(X_cp, cp.ndarray) else X_cp
    X_cp = cp.asarray(X_np, dtype=cp.float32)
    return DatasetWrapper(
        X_np=X_np,
        X_cp=X_cp,
        metric="euclidean",
        dataset_type="blobs",
        n_samples=300,
        n_features=10,
        n_centers=3,
        cluster_std=1.0,
    )


@pytest.mark.slow
@settings(
    max_examples=5,  # Random testing across all dataset types
    deadline=120000,  # 120 seconds per test case (embedding can be slow)
    phases=[
        hypothesis.Phase.explicit,
        hypothesis.Phase.reuse,
        hypothesis.Phase.generate,
        hypothesis.Phase.target,
    ],
    print_blob=False,  # Don't print large dataset arrays in failure messages
    verbosity=hypothesis.Verbosity.verbose,  # Display information about every run
)
@given(
    dataset=diverse_dataset_strategy(),
    params=embedding_params_strategy(),
)
@example(
    dataset=_generate_baseline_dataset(),
    params={
        "learning_rate": 1.0,
        "min_dist": 0.1,
        "spread": 1.0,
        "n_epochs": 200,
        "negative_sample_rate": 5,
        "gamma": 1.0,
        "init": "spectral",
        "n_components": 2,
    },
)
def test_simplicial_set_embedding_hypothesis(dataset, params):
    """
    Property-based test for UMAP simplicial set embedding using random parameters.

    This test validates the cuML implementation against the reference UMAP
    using Hypothesis to generate random combinations of datasets and parameters.

    Dataset Types (randomly selected):
    - Blobs: Traditional clustered data (5-25D, euclidean or cosine)
    - Circles: Concentric circles topology (2D, euclidean only)
    - Moons: Two interleaving half circles (2D, euclidean only)
    - Swiss Roll: Classic 3D manifold structure (euclidean only)
    - S-Curve: 3D curved manifold (euclidean only)

    Test Coverage:
    - 5 random parameter/dataset combinations
    - All parameters within reasonable ranges (no extreme values)
    - Both 2D and 3D embeddings
    - Euclidean metric for all; cosine metric only for high-dimensional blobs
    - Sample sizes: 800-2000
    - Epochs: 200-500

    Quality Validation:
    Compares cuML vs reference UMAP on multiple metrics including trustworthiness,
    continuity, geodesic correlations, fuzzy KL divergence, and Procrustes RMSE.
    """
    # Skip test if cuvs is not available
    pytest.importorskip("cuvs")

    # Import manifold_metrics (depends on cuvs)
    from cuml.testing.manifold_metrics import (
        _build_knn_with_cuvs,
        compute_simplicial_set_embedding_metrics,
        procrustes_rmse,
    )

    X_np = dataset["X_np"]
    X_cp = dataset["X_cp"]
    metric = dataset["metric"]
    dataset_type = dataset["dataset_type"]
    k = 15

    # Build KNN graph using cuVS
    knn_dists_np, knn_inds_np = _build_knn_with_cuvs(
        X_cp, k=k, metric=metric, backend="bruteforce"
    )

    # Build fuzzy simplicial set using cuML
    cu_graph_gpu = fuzzy_simplicial_set(
        X_cp,
        n_neighbors=k,
        random_state=42,
        metric=metric,
        knn_indices=cp.asarray(knn_inds_np),
        knn_dists=cp.asarray(knn_dists_np),
    )
    cu_graph_cpu = (
        cu_graph_gpu.get() if hasattr(cu_graph_gpu, "get") else cu_graph_gpu
    )
    cu_graph_cpu = csr_matrix(cu_graph_cpu)

    a, b = find_ab_params(spread=params["spread"], min_dist=params["min_dist"])

    # Reference embedding (CPU) using UMAP reference implementation
    ref_res = ref_simplicial_set_embedding(
        X_np,
        graph=cu_graph_cpu,
        n_components=params["n_components"],
        initial_alpha=params["learning_rate"],
        a=a,
        b=b,
        gamma=params["gamma"],
        negative_sample_rate=params["negative_sample_rate"],
        n_epochs=params["n_epochs"],
        init=params["init"],
        random_state=np.random.RandomState(42),
        metric=metric,
        metric_kwds={},
        densmap=False,
        densmap_kwds={},
        output_dens=False,
        output_metric="euclidean",
        output_metric_kwds={},
    )
    ref_emb = ref_res[0]

    # cuML embedding (GPU)
    cu_emb = simplicial_set_embedding(
        X_cp,
        graph=cu_graph_gpu,
        n_components=params["n_components"],
        initial_alpha=params["learning_rate"],
        a=a,
        b=b,
        gamma=params["gamma"],
        negative_sample_rate=params["negative_sample_rate"],
        n_epochs=params["n_epochs"],
        init=params["init"],
        random_state=42,
        metric=metric,
    )
    cu_emb = cp.asnumpy(cu_emb) if isinstance(cu_emb, cp.ndarray) else cu_emb

    # Evaluate embedding quality with relaxed thresholds for random init.
    # Random initialization causes cuML and reference to start from different
    # positions (different RNGs), which can lead to different local optima.
    # This is expected behavior, so we use more lenient thresholds.
    if params["init"] == "random":
        quality_kwargs = {
            "sev_corr": 0.65,  # Relaxed from 0.50 for random init variance
            "sev_rmse": 0.50,  # Relaxed from 0.40
            "mod_corr": 0.35,  # Relaxed from 0.25
        }
    else:
        quality_kwargs = {}

    should_fail, fail_reason, metrics = evaluate_embedding_quality(
        X_cp,
        ref_emb,
        cu_emb,
        metric,
        compute_metrics_fn=compute_simplicial_set_embedding_metrics,
        procrustes_rmse_fn=procrustes_rmse,
        k=k,
        **quality_kwargs,
    )

    if should_fail:
        # Format dataset-specific details for error message
        dataset_info = (
            f"dataset_type={dataset_type}, n_samples={dataset['n_samples']}"
        )
        if "n_features" in dataset:
            dataset_info += f", n_features={dataset['n_features']}"

        details = (
            f"Hypothesis test failed for {dataset_type} dataset "
            f"({dataset_info}, metric={metric}) "
            f"with params {params}: {fail_reason} | "
            f"RMSE={metrics['rmse']:.3f}, trust_def={metrics['trust_def']:.3f}, "
            f"cont_def={metrics['cont_def']:.3f}, sp_def={metrics['sp_def']:.3f}, "
            f"pe_def={metrics['pe_def']:.3f}, xent_rel={metrics['xent_rel_increase']:.3f}, "
            f"kl_rel={metrics['kl_rel_increase']:.3f}"
        )
        assert False, details


@st.composite
def spectral_params_strategy(draw):
    """Generate valid spectral initialization parameters."""
    # n_components: 2 or 3 for visualization
    n_components = draw(st.sampled_from([2, 3]))

    # n_neighbors: reasonable range for graph construction
    n_neighbors = draw(st.integers(min_value=10, max_value=30))

    return {
        "n_components": n_components,
        "n_neighbors": n_neighbors,
    }


def _generate_baseline_spectral_dataset():
    """Generate a baseline dataset for the spectral init required example."""
    X_cp, y = cuml.datasets.make_blobs(
        n_samples=400,
        n_features=10,
        centers=4,
        cluster_std=1.0,
        random_state=42,
    )
    X_np = cp.asnumpy(X_cp) if isinstance(X_cp, cp.ndarray) else X_cp
    X_cp = cp.asarray(X_np, dtype=cp.float32)
    return DatasetWrapper(
        X_np=X_np,
        X_cp=X_cp,
        metric="euclidean",
        dataset_type="blobs",
        n_samples=400,
        n_features=10,
        n_centers=4,
        cluster_std=1.0,
    )


def evaluate_spectral_quality(
    ref_emb,
    cu_emb,
    n_components,
    compare_spectral_fn,
    # RMSE threshold (Procrustes-aligned - the definitive metric)
    # Procrustes alignment finds optimal rotation/reflection, handling component
    # ordering and sign flip differences between implementations
    rmse_threshold=0.20,
):
    """
    Evaluate spectral embedding quality by comparing cuML against reference.

    For spectral embeddings, Procrustes RMSE is the definitive metric because:
    1. Eigenvectors are only defined up to sign (can be negated)
    2. Eigenvectors for equal/close eigenvalues can be in any order
    3. Different numerical methods may produce different but equivalent bases

    Procrustes alignment finds the optimal orthogonal transformation (rotation +
    reflection) to align the embeddings, making RMSE the only meaningful metric.
    Per-component correlations are computed for diagnostic purposes but are NOT
    used for pass/fail decisions since they don't account for component permutation.

    Returns:
        tuple: (should_fail, fail_reason, metrics_dict)
    """
    # Ensure numpy arrays
    ref_np = (
        cp.asnumpy(ref_emb)
        if isinstance(ref_emb, cp.ndarray)
        else np.asarray(ref_emb, dtype=np.float32)
    )
    cu_np = (
        cp.asnumpy(cu_emb)
        if isinstance(cu_emb, cp.ndarray)
        else np.asarray(cu_emb, dtype=np.float32)
    )

    # Compare embeddings
    result = compare_spectral_fn(ref_np, cu_np, n_components)
    rmse, correlations = result["rmse"], result["correlations"]

    # Compute average correlation (for diagnostics only)
    avg_corr = np.mean([abs(c) for c in correlations])

    # Decision based solely on Procrustes RMSE
    should_fail = rmse > rmse_threshold
    fail_reason = (
        f"RMSE {rmse:.3f} > threshold {rmse_threshold:.3f}"
        if should_fail
        else ""
    )

    metrics_dict = {
        "rmse": rmse,
        "correlations": correlations,
        "avg_corr": avg_corr,
    }

    return should_fail, fail_reason, metrics_dict


@pytest.mark.skip  # see https://github.com/NVIDIA/cuml/issues/7713
@pytest.mark.slow
@settings(
    max_examples=5,  # Random testing across dataset types
    deadline=120000,  # 120 seconds per test case
    phases=[
        hypothesis.Phase.explicit,
        hypothesis.Phase.reuse,
        hypothesis.Phase.generate,
        hypothesis.Phase.target,
    ],
    print_blob=False,  # Don't print large dataset arrays in failure messages
    verbosity=hypothesis.Verbosity.verbose,  # Display information about every run
)
@given(
    dataset=diverse_dataset_strategy(),
    params=spectral_params_strategy(),
)
@example(
    dataset=_generate_baseline_spectral_dataset(),
    params={
        "n_components": 2,
        "n_neighbors": 15,
    },
)
def test_spectral_init_hypothesis(dataset, params):
    """
    Property-based test for spectral initialization using random parameters.

    This test validates the cuML SpectralEmbedding against the reference UMAP
    spectral_layout using Hypothesis to generate random combinations of
    datasets and parameters.

    Dataset Types (randomly selected):
    - Blobs: Traditional clustered data (5-25D, euclidean or cosine)
    - Circles: Concentric circles topology (2D, euclidean only)
    - Moons: Two interleaving half circles (2D, euclidean only)
    - Swiss Roll: Classic 3D manifold structure (euclidean only)
    - S-Curve: 3D curved manifold (euclidean only)

    Test Coverage:
    - 5 random parameter/dataset combinations
    - Both 2D and 3D embeddings
    - n_neighbors: 10-30
    - Sample sizes: 800-2000

    Quality Validation:
    Compares cuML vs reference UMAP spectral layout on RMSE and per-component
    correlations after Procrustes alignment.
    """
    # Skip test if cuvs is not available
    pytest.importorskip("cuvs")

    from cuml.testing.manifold_metrics import (
        _build_knn_with_cuvs,
        compare_spectral_embeddings,
    )

    X_np = dataset["X_np"]
    X_cp = dataset["X_cp"]
    metric = dataset["metric"]
    dataset_type = dataset["dataset_type"]

    n_components = params["n_components"]
    n_neighbors = params["n_neighbors"]
    random_state = 42

    # Build KNN graph using cuVS
    knn_dists_np, knn_inds_np = _build_knn_with_cuvs(
        X_cp, k=n_neighbors, metric=metric, backend="bruteforce"
    )

    # Build fuzzy simplicial set using cuML (needed for spectral layout)
    cu_graph_gpu = fuzzy_simplicial_set(
        X_cp,
        n_neighbors=n_neighbors,
        random_state=random_state,
        metric=metric,
        knn_indices=cp.asarray(knn_inds_np),
        knn_dists=cp.asarray(knn_dists_np),
    )
    cu_graph_cpu = (
        cu_graph_gpu.get() if hasattr(cu_graph_gpu, "get") else cu_graph_gpu
    )
    cu_graph_cpu = csr_matrix(cu_graph_cpu)

    # Run reference UMAP spectral layout
    ref_emb = spectral_layout(
        data=None,
        graph=cu_graph_cpu,
        dim=n_components,
        random_state=np.random.RandomState(random_state),
    )
    ref_emb = (
        cp.asnumpy(ref_emb)
        if isinstance(ref_emb, cp.ndarray)
        else np.asarray(ref_emb, dtype=np.float32)
    )

    # Run cuML SpectralEmbedding
    cu_emb = SpectralEmbedding(
        affinity="precomputed",
        n_components=n_components,
        n_neighbors=n_neighbors,
        random_state=random_state,
    ).fit_transform(cu_graph_cpu)
    cu_emb = (
        cp.asnumpy(cu_emb)
        if isinstance(cu_emb, cp.ndarray)
        else np.asarray(cu_emb, dtype=np.float32)
    )

    # Validate embeddings (shape, NaN/inf, variance, value range)
    expected_shape = (X_np.shape[0], n_components)
    for name, emb in [("Reference", ref_emb), ("cuML", cu_emb)]:
        assert emb.shape == expected_shape, (
            f"{name} shape {emb.shape} != {expected_shape}"
        )
        assert not np.isnan(emb).any(), f"{name} contains NaN"
        assert not np.isinf(emb).any(), f"{name} contains infinite values"
        assert np.std(emb, axis=0).min() > 0.001, (
            f"{name} has insufficient variance"
        )
        abs_max = np.abs(emb).max()
        assert 0.001 < abs_max < 100.0, (
            f"{name} values out of range: {abs_max:.3e}"
        )

    # Evaluate spectral embedding quality
    should_fail, fail_reason, metrics = evaluate_spectral_quality(
        ref_emb,
        cu_emb,
        n_components,
        compare_spectral_fn=compare_spectral_embeddings,
    )

    if should_fail:
        # Format dataset-specific details for error message
        dataset_info = (
            f"dataset_type={dataset_type}, n_samples={dataset['n_samples']}"
        )
        if "n_features" in dataset:
            dataset_info += f", n_features={dataset['n_features']}"

        corr_str = ", ".join(f"{abs(c):.3f}" for c in metrics["correlations"])
        details = (
            f"Hypothesis spectral init test failed for {dataset_type} dataset "
            f"({dataset_info}, metric={metric}) "
            f"with params {params}: {fail_reason} | "
            f"RMSE={metrics['rmse']:.3f}, avg_corr={metrics['avg_corr']:.3f}, "
            f"corr=[{corr_str}]"
        )
        assert False, details


@st.composite
def fuzzy_params_strategy(draw):
    """Generate valid fuzzy simplicial set parameters."""
    # n_neighbors: reasonable range for graph construction
    n_neighbors = draw(st.integers(min_value=10, max_value=30))

    return {
        "n_neighbors": n_neighbors,
    }


def _generate_baseline_fuzzy_dataset():
    """Generate a baseline dataset for the fuzzy simplicial set required example."""
    X_cp, y = cuml.datasets.make_blobs(
        n_samples=500,
        n_features=10,
        centers=4,
        cluster_std=1.0,
        random_state=42,
    )
    X_np = cp.asnumpy(X_cp) if isinstance(X_cp, cp.ndarray) else X_cp
    X_cp = cp.asarray(X_np, dtype=cp.float32)
    return DatasetWrapper(
        X_np=X_np,
        X_cp=X_cp,
        metric="euclidean",
        dataset_type="blobs",
        n_samples=500,
        n_features=10,
        n_centers=4,
        cluster_std=1.0,
    )


def evaluate_fuzzy_quality(
    ref_graph,
    cu_graph,
    compute_metrics_fn,
    compute_js_fn,
    # Thresholds for synthetic data (slightly relaxed from benchmark datasets)
    kl_threshold=0.08,
    js_threshold=0.08,
    jaccard_threshold=0.85,
    row_l1_threshold=0.01,
):
    """
    Evaluate fuzzy simplicial set quality by comparing cuML against reference.

    Metrics:
    - KL divergence (symmetric): measures distributional difference
    - JS divergence: bounded version of KL, always finite
    - Jaccard similarity: edge overlap between graphs
    - Row-sum L1: difference in row normalization

    Returns:
        tuple: (should_fail, fail_reason, metrics_dict)
    """
    # Compute fuzzy simplicial set metrics
    kl_sym, jaccard, row_l1 = compute_metrics_fn(ref_graph, cu_graph)
    js_avg = compute_js_fn(ref_graph, cu_graph, average=True)

    # Check for finite values
    if not np.isfinite(kl_sym):
        return (
            True,
            "KL divergence is not finite",
            {
                "kl_sym": kl_sym,
                "js_avg": js_avg,
                "jaccard": jaccard,
                "row_l1": row_l1,
            },
        )
    if not np.isfinite(js_avg):
        return (
            True,
            "JS divergence is not finite",
            {
                "kl_sym": kl_sym,
                "js_avg": js_avg,
                "jaccard": jaccard,
                "row_l1": row_l1,
            },
        )

    # Collect issues
    issues = []

    if kl_sym > kl_threshold:
        issues.append(f"KL={kl_sym:.4f} > {kl_threshold}")
    if js_avg > js_threshold:
        issues.append(f"JS={js_avg:.4f} > {js_threshold}")
    if jaccard < jaccard_threshold:
        issues.append(f"Jaccard={jaccard:.3f} < {jaccard_threshold}")
    if row_l1 > row_l1_threshold:
        issues.append(f"row_L1={row_l1:.4f} > {row_l1_threshold}")

    # Fail if multiple issues or any severe issue
    should_fail = len(issues) >= 2
    fail_reason = f"Multiple issues: {issues}" if should_fail else ""

    metrics_dict = {
        "kl_sym": kl_sym,
        "js_avg": js_avg,
        "jaccard": jaccard,
        "row_l1": row_l1,
        "issues": issues,
    }

    return should_fail, fail_reason, metrics_dict


@pytest.mark.xfail(
    Version(numba.__version__) >= Version("0.62.0")
    and UMAP_VERSION < Version("0.5.12"),
    reason="Upstream regression in umap<0.5.12 with numba >= 0.62.0",
    strict=True,
)
@pytest.mark.slow
@settings(
    max_examples=5,  # Random testing across dataset types
    deadline=120000,  # 120 seconds per test case
    phases=[
        hypothesis.Phase.explicit,
        hypothesis.Phase.reuse,
        hypothesis.Phase.generate,
        hypothesis.Phase.target,
    ],
    print_blob=False,  # Don't print large dataset arrays in failure messages
    verbosity=hypothesis.Verbosity.verbose,  # Display information about every run
)
@given(
    dataset=diverse_dataset_strategy(),
    params=fuzzy_params_strategy(),
)
@example(
    dataset=_generate_baseline_fuzzy_dataset(),
    params={
        "n_neighbors": 15,
    },
)
def test_fuzzy_simplicial_set_hypothesis(dataset, params):
    """
    Property-based test for fuzzy simplicial set construction using random parameters.

    This test validates the cuML fuzzy_simplicial_set against the reference UMAP
    implementation using Hypothesis to generate random combinations of datasets
    and parameters.

    Dataset Types (randomly selected):
    - Blobs: Traditional clustered data (5-25D, euclidean or cosine)
    - Circles: Concentric circles topology (2D, euclidean only)
    - Moons: Two interleaving half circles (2D, euclidean only)
    - Swiss Roll: Classic 3D manifold structure (euclidean only)
    - S-Curve: 3D curved manifold (euclidean only)

    Test Coverage:
    - 5 random parameter/dataset combinations
    - n_neighbors: 10-30
    - Sample sizes: 800-2000

    Quality Validation:
    Compares cuML vs reference UMAP fuzzy graphs on:
    - Symmetric KL divergence
    - Jensen-Shannon divergence
    - Edge Jaccard similarity
    - Row-sum L1 norm difference
    """
    # Skip test if cuvs is not available
    pytest.importorskip("cuvs")

    # Import manifold_metrics (depends on cuvs)
    from cuml.testing.manifold_metrics import (
        _build_knn_with_cuvs,
        compute_fuzzy_js_divergence,
        compute_fuzzy_simplicial_set_metrics,
    )

    X_np = dataset["X_np"]
    X_cp = dataset["X_cp"]
    metric = dataset["metric"]
    dataset_type = dataset["dataset_type"]

    n_neighbors = params["n_neighbors"]
    random_state = 42

    # Build KNN graph using cuVS
    knn_dists_np, knn_inds_np = _build_knn_with_cuvs(
        X_cp, k=n_neighbors, metric=metric, backend="bruteforce"
    )

    # Reference fuzzy graph (CPU / SciPy sparse)
    ref_graph, _, _ = ref_fuzzy_simplicial_set(
        X_np,
        n_neighbors=n_neighbors,
        random_state=random_state,
        metric=metric,
        knn_indices=knn_inds_np,
        knn_dists=knn_dists_np,
    )
    ref_graph = csr_matrix(ref_graph)

    # cuML fuzzy graph (GPU)
    cu_graph_gpu = fuzzy_simplicial_set(
        X_cp,
        n_neighbors=n_neighbors,
        random_state=random_state,
        metric=metric,
        knn_indices=cp.asarray(knn_inds_np),
        knn_dists=cp.asarray(knn_dists_np),
    )
    cu_graph = (
        cu_graph_gpu.get() if hasattr(cu_graph_gpu, "get") else cu_graph_gpu
    )
    cu_graph = csr_matrix(cu_graph)

    # Evaluate fuzzy simplicial set quality
    should_fail, fail_reason, metrics = evaluate_fuzzy_quality(
        ref_graph,
        cu_graph,
        compute_metrics_fn=compute_fuzzy_simplicial_set_metrics,
        compute_js_fn=compute_fuzzy_js_divergence,
    )

    if should_fail:
        # Format dataset-specific details for error message
        dataset_info = (
            f"dataset_type={dataset_type}, n_samples={dataset['n_samples']}"
        )
        if "n_features" in dataset:
            dataset_info += f", n_features={dataset['n_features']}"

        details = (
            f"Hypothesis fuzzy simplicial set test failed for {dataset_type} dataset "
            f"({dataset_info}, metric={metric}) "
            f"with params {params}: {fail_reason} | "
            f"KL={metrics['kl_sym']:.4f}, JS={metrics['js_avg']:.4f}, "
            f"Jaccard={metrics['jaccard']:.3f}, row_L1={metrics['row_l1']:.4f}"
        )
        assert False, details
