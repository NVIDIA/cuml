# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Validate benchmark declarations for public cuML estimators.

This module only inspects Python classes and checked-in benchmark metadata. It
does not instantiate estimators, allocate arrays, or run benchmark workloads.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping
from pathlib import Path

# Public estimators that intentionally do not have a canonical benchmark pair.
# Every entry must remain tied to a currently discoverable estimator; otherwise
# the stale-exclusion check below fails and the map can be pruned.
BENCHMARK_EXCLUSIONS = {
    "ARIMA": "Time-series API has no canonical benchmark manifest.",
    "AutoARIMA": "Time-series API has no canonical benchmark manifest.",
    "CD": "Low-level solver API is outside the estimator benchmark suite.",
    "ExponentialSmoothing": "Time-series API has no canonical benchmark manifest.",
    "HDBSCAN": "Optional dependency has no canonical benchmark manifest.",
    "IsolationForest": "No benchmark registry pair is defined yet.",
    "Lars": "No benchmark registry pair is defined yet.",
    "MBSGDClassifier": "No canonical manifest entry is defined yet.",
    "QN": "Low-level solver API is outside the estimator benchmark suite.",
    "SGD": "Low-level solver API is outside the estimator benchmark suite.",
    "UMAP": "Optional dependency has no canonical benchmark manifest.",
}


def _public_estimators() -> dict[str, type]:
    """Return concrete, public estimators exported by the top-level API."""
    import cuml
    from cuml.internals.base import Base
    from cuml.testing.utils import ClassEnumerator

    models = ClassEnumerator(module=cuml).get_models()
    return {
        name: cls
        for name, cls in models.items()
        if not name.startswith("_") and cls is not Base and not inspect.isabstract(cls)
    }


def _registry_estimators() -> dict[str, set[str]]:
    """Map estimator class names to registered benchmark algorithm names."""
    from cuml.benchmark import algorithms

    registered: dict[str, set[str]] = {}
    for pair in algorithms.all_algorithms():
        estimator = pair.cuml_class
        if estimator is None or not inspect.isclass(estimator):
            continue
        registered.setdefault(estimator.__name__, set()).add(pair.name)
    return registered


def _manifest_algorithms() -> set[str]:
    """Read algorithm names from every checked-in benchmark manifest."""
    from cuml.benchmark.config import load_config_file

    configs_dir = Path(__file__).resolve().parents[1] / "cuml" / "benchmark" / "configs"
    manifests = sorted(configs_dir.glob("*.yaml"))
    assert manifests, f"No benchmark manifests found in {configs_dir}"

    names = set()
    for manifest in manifests:
        raw_config = load_config_file(str(manifest))
        names.update(
            entry["algorithm"]
            for entry in raw_config["benchmarks"]
            if entry.get("algorithm")
        )
    return names


def _coverage_errors(
    estimators: Mapping[str, type],
    registry: Mapping[str, Iterable[str]],
    manifest_names: set[str],
    exclusions: Mapping[str, str],
) -> list[str]:
    """Return actionable errors for missing, invalid, or stale declarations."""
    errors = []
    for name, reason in exclusions.items():
        if name not in estimators:
            errors.append(f"stale benchmark exclusion: {name}")
        elif not isinstance(reason, str) or not reason.strip():
            errors.append(f"empty benchmark exclusion reason: {name}")

    for name in sorted(set(estimators) - set(exclusions)):
        algorithm_names = set(registry.get(name, ()))
        if not algorithm_names:
            errors.append(f"{name}: missing benchmark registry entry")
        elif not algorithm_names & manifest_names:
            names = ", ".join(sorted(algorithm_names))
            errors.append(
                f"{name}: registry entries [{names}] are absent from manifests"
            )
    return errors


def test_public_estimators_have_benchmark_coverage():
    errors = _coverage_errors(
        _public_estimators(),
        _registry_estimators(),
        _manifest_algorithms(),
        BENCHMARK_EXCLUSIONS,
    )
    assert not errors, "\n".join(errors)


def test_benchmark_exclusion_reasons_are_required():
    errors = _coverage_errors(
        {"ExampleEstimator": object},
        {},
        set(),
        {"ExampleEstimator": ""},
    )
    assert errors == ["empty benchmark exclusion reason: ExampleEstimator"]


def test_uncovered_estimator_is_rejected():
    errors = _coverage_errors(
        {"ExampleEstimator": object},
        {},
        set(),
        {},
    )
    assert errors == ["ExampleEstimator: missing benchmark registry entry"]


def test_registry_entry_without_manifest_is_rejected():
    errors = _coverage_errors(
        {"ExampleEstimator": object},
        {"ExampleEstimator": {"ExampleEstimator"}},
        set(),
        {},
    )
    assert errors == [
        (
            "ExampleEstimator: registry entries [ExampleEstimator] are absent "
            "from manifests"
        )
    ]


def test_stale_benchmark_exclusion_is_rejected():
    errors = _coverage_errors(
        {},
        {},
        set(),
        {"RemovedEstimator": "No benchmark registry pair is defined yet."},
    )
    assert errors == ["stale benchmark exclusion: RemovedEstimator"]
