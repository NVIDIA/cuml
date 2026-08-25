#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import and render the generated cuml.accel Sphinx benchmark page.

``import`` validates raw mlbench CPU/GPU result pairs and writes a compact,
presentation-oriented data file. ``render`` converts that checked-in data and
the editable RST template into the published page and SVG heatmaps.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "docs/benchmarks/cuml-accel/benchmark-data.json"
DEFAULT_TEMPLATE = ROOT / "docs/source/cuml-accel/benchmarks.rst.in"
DEFAULT_PAGE = ROOT / "docs/source/cuml-accel/benchmarks.rst"
DEFAULT_STATIC = ROOT / "docs/source/_static/cuml-accel-benchmarks"

PCA_ULTRA_WIDE_CASES = {
    "pca.fit_transform.feature_wide_1024": {
        "rows": 50_000,
        "features": 1_024,
        "components": 512,
    },
    "pca.fit_transform.feature_wide_2048": {
        "rows": 5_000,
        "features": 2_048,
        "components": 1_024,
    },
}
PCA_ULTRA_WIDE_TOLERANCE = 0.01

WORKLOADS = (
    "small.balanced",
    "medium.thin",
    "medium.balanced",
    "medium.wide",
    "large.balanced",
)
TRAINING_OPERATIONS = {"fit", "fit_predict", "fit_transform"}
INFERENCE_HEATMAP_OPERATION_LIMIT = 10

DISPLAY_NAMES = {
    "dbscan": "DBSCAN",
    "hdbscan": "HDBSCAN",
    "k_neighbors_classifier": "KNeighborsClassifier",
    "kmeans": "KMeans",
    "linear_regression": "LinearRegression",
    "logistic_regression": "LogisticRegression",
    "nearest_neighbors": "NearestNeighbors",
    "pca": "PCA",
    "polynomial_features": "PolynomialFeatures",
    "random_forest_classifier": "RandomForestClassifier",
    "random_forest_regressor": "RandomForestRegressor",
    "ridge": "Ridge",
    "standard_scaler": "StandardScaler",
    "svc": "SVC",
    "target_encoder": "TargetEncoder",
    "truncated_svd": "TruncatedSVD",
    "umap": "UMAP",
}

FAMILIES = {
    "Linear models": ("linear_regression", "logistic_regression", "ridge"),
    "Clustering and manifold learning": (
        "dbscan",
        "hdbscan",
        "kmeans",
        "umap",
    ),
    "Neighbors": ("k_neighbors_classifier", "nearest_neighbors"),
    "Decomposition": ("pca", "truncated_svd"),
    "Ensembles": ("random_forest_classifier", "random_forest_regressor"),
    "Preprocessing": (
        "polynomial_features",
        "standard_scaler",
        "target_encoder",
    ),
    "Kernel methods": ("svc",),
}

ANCHORS = {
    "dbscan": "benchmark-dbscan",
    "hdbscan": "benchmark-hdbscan",
    "k_neighbors_classifier": "benchmark-kneighborsclassifier",
    "kmeans": "benchmark-kmeans",
    "linear_regression": "benchmark-linearregression",
    "logistic_regression": "benchmark-logisticregression",
    "nearest_neighbors": "benchmark-nearestneighbors",
    "pca": "benchmark-pca",
    "polynomial_features": "benchmark-polynomialfeatures",
    "random_forest_classifier": "benchmark-randomforestclassifier",
    "random_forest_regressor": "benchmark-randomforestregressor",
    "ridge": "benchmark-ridge",
    "standard_scaler": "benchmark-standardscaler",
    "svc": "benchmark-svc",
    "target_encoder": "benchmark-targetencoder",
    "truncated_svd": "benchmark-truncatedsvd",
    "umap": "benchmark-umap",
}


def _median_wall_time(result: dict[str, Any]) -> float | None:
    values = [
        timing["value"]
        for observation in result.get("observations", [])
        if observation.get("role") == "measurement"
        and observation.get("outcome", {}).get("status") == "success"
        for timing in observation.get("timings", [])
        if timing.get("name") == "wall_time"
        and isinstance(timing.get("value"), (int, float))
        and timing["value"] > 0
    ]
    return statistics.median(values) if values else None


def _is_timeout(result: dict[str, Any]) -> bool:
    outcome = result.get("outcome", {})
    return (
        outcome.get("status") == "failed"
        and outcome.get("error", {}).get("type") == "TimeoutExpired"
    )


def _dimension(result: dict[str, Any], name: str) -> int:
    return next(
        item["size"]
        for item in result["input"]["dimensions"]
        if item["name"] == name
    )


def _execution_profile(result: dict[str, Any]) -> str:
    profiles = {
        observation.get("extensions", {})
        .get("ai.rapids.cuml.accel", {})
        .get("classification")
        for observation in result.get("observations", [])
        if observation.get("role") == "measurement"
    }
    profiles.discard(None)
    return ", ".join(sorted(profiles)) if profiles else "unavailable"


def _package_versions(artifact: dict[str, Any]) -> dict[str, str]:
    return {
        item["name"]: item["version"]
        for item in artifact["run"]["software"].get("packages", [])
        if item["name"] != "cuml.accel"
    }


def _image(artifact: dict[str, Any]) -> dict[str, str]:
    return artifact["run"]["extensions"]["ai.rapids.mlbench"]["image"]


def _metric(result: dict[str, Any], name: str) -> float:
    values = [
        metric["value"]
        for observation in result.get("observations", [])
        if observation.get("role") == "measurement"
        and observation.get("outcome", {}).get("status") == "success"
        for metric in observation.get("metrics", [])
        if metric.get("name") == name
        and isinstance(metric.get("value"), (int, float))
    ]
    if not values:
        raise ValueError(f"expected at least one {name} value")
    return statistics.median(values)


def _timeout_metadata(artifact: dict[str, Any]) -> dict[str, Any]:
    return artifact["run"]["extensions"]["ai.rapids.mlbench"]["timeouts"]


def _validate_repetitions(result: dict[str, Any], case_label: str) -> None:
    """Require the published one-warmup/three-measurement timing policy."""
    if result.get("outcome", {}).get("status") != "success":
        return
    roles = [
        observation.get("role")
        for observation in result.get("observations", [])
    ]
    if roles.count("warmup") != 1 or roles.count("measurement") != 3:
        raise ValueError(
            f"unexpected warmup or repetition count for {case_label}"
        )


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    phases: dict[str, dict[str, Any]] = {}
    for phase in ("training", "inference"):
        phase_records = [
            record for record in records if record["phase"] == phase
        ]
        paired = [
            record["speedup"]
            for record in phase_records
            if record["speedup"] is not None
        ]
        phases[phase] = {
            "cells": len(phase_records),
            "paired": len(paired),
            "median_speedup": statistics.median(paired),
            "at_least_2x": sum(value >= 2 for value in paired),
            "slowdowns": sum(value < 1 for value in paired),
        }
    return {
        "estimators": len({record["estimator"] for record in records}),
        "operations": len(
            {(record["estimator"], record["operation"]) for record in records}
        ),
        "cases": len(records),
        "unavailable": sum(record["speedup"] is None for record in records),
        "timeouts": {
            side: sum(record["timeout_side"] == side for record in records)
            for side in ("cpu", "gpu", "both")
        },
        "phases": phases,
    }


def normalize(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    left = {item["case_label"]: item for item in reference["results"]}
    right = {item["case_label"]: item for item in candidate["results"]}
    if set(left) != set(right):
        missing_left = sorted(set(right) - set(left))
        missing_right = sorted(set(left) - set(right))
        raise ValueError(
            f"artifact case labels differ (reference missing {missing_left}; candidate missing {missing_right})"
        )
    reference_timeouts = _timeout_metadata(reference)
    candidate_timeouts = _timeout_metadata(candidate)
    reference_effective_timeouts = reference_timeouts[
        "effective_case_timeout_sec"
    ]
    candidate_effective_timeouts = candidate_timeouts[
        "effective_case_timeout_sec"
    ]
    if reference_effective_timeouts != candidate_effective_timeouts:
        raise ValueError("reference and candidate timeout policies differ")
    reference_workflow = reference["run"]["extensions"]["ai.rapids.mlbench"]
    candidate_workflow = candidate["run"]["extensions"]["ai.rapids.mlbench"]
    if reference_workflow["backend"] != {"name": "sklearn", "device": "cpu"}:
        raise ValueError("reference artifact has the wrong backend identity")
    if candidate_workflow["backend"] != {
        "name": "cuml_accel",
        "device": "gpu",
    }:
        raise ValueError("candidate artifact has the wrong backend identity")
    if reference_workflow["image"] != candidate_workflow["image"]:
        raise ValueError("reference and candidate image provenance differ")
    if _package_versions(reference) != _package_versions(candidate):
        raise ValueError("reference and candidate package versions differ")

    family_by_estimator = {
        estimator: family
        for family, estimators in FAMILIES.items()
        for estimator in estimators
    }
    records: list[dict[str, Any]] = []
    for case_label in sorted(left):
        cpu, gpu = left[case_label], right[case_label]
        if (
            cpu["algorithm"] != gpu["algorithm"]
            or cpu["operation"] != gpu["operation"]
        ):
            raise ValueError(f"artifact metadata differs for {case_label}")
        if (
            cpu["input"] != gpu["input"]
            or cpu["parameters"]["declared"] != gpu["parameters"]["declared"]
        ):
            raise ValueError(
                f"artifact input or parameters differ for {case_label}"
            )
        for result, backend in ((cpu, "CPU"), (gpu, "accelerated")):
            if result.get("outcome", {}).get(
                "status"
            ) != "success" and not _is_timeout(result):
                raise ValueError(
                    f"{backend} result is neither successful nor a timeout for {case_label}"
                )
            _validate_repetitions(result, case_label)
        if (
            gpu.get("outcome", {}).get("status") == "success"
            and _execution_profile(gpu) != "gpu_only"
        ):
            raise ValueError(
                f"accelerated result did not record GPU-only execution for {case_label}"
            )
        operation = cpu["operation"]["name"]
        phase = "training" if operation in TRAINING_OPERATIONS else "inference"
        cpu_time, gpu_time = _median_wall_time(cpu), _median_wall_time(gpu)
        cpu_timeout, gpu_timeout = _is_timeout(cpu), _is_timeout(gpu)
        if cpu_timeout and gpu_timeout:
            timeout_side = "both"
        elif cpu_timeout:
            timeout_side = "cpu"
        elif gpu_timeout:
            timeout_side = "gpu"
        else:
            timeout_side = None
        speedup = (
            None
            if timeout_side or cpu_time is None or gpu_time is None
            else cpu_time / gpu_time
        )
        label_prefix = f"{cpu['algorithm']}.{operation}."
        workload = case_label.removeprefix(label_prefix)
        records.append(
            {
                "estimator": cpu["algorithm"],
                "estimator_name": DISPLAY_NAMES[cpu["algorithm"]],
                "family": family_by_estimator[cpu["algorithm"]],
                "operation": operation,
                "phase": phase,
                "workload_label": workload,
                "rows": _dimension(cpu, "rows"),
                "features": _dimension(cpu, "features"),
                "input_bytes": cpu["input"]["attributes"][
                    "estimated_input_size_bytes"
                ],
                "cpu_median_wall_time_sec": cpu_time,
                "gpu_median_wall_time_sec": gpu_time,
                "speedup": speedup,
                "timeout_side": timeout_side,
                "timeout_limit_sec": reference_effective_timeouts[case_label],
                "execution_profile": _execution_profile(gpu),
                "correctness_status": "validated"
                if gpu.get("outcome", {}).get("status") == "success"
                else "not completed",
            }
        )

    return {
        "schema_version": 1,
        "definition": {
            "speedup": "CPU median wall time / accelerated median wall time",
            "training_operations": sorted(TRAINING_OPERATIONS),
            "workload_order": list(WORKLOADS),
        },
        "summary": _summarize_records(records),
        "system": candidate["run"]["system"],
        "methodology": candidate["run"]["methodology"],
        "timeout_policy": {
            "scope": "complete isolated case",
            "default_case_timeout_sec": reference_timeouts[
                "default_case_timeout_sec"
            ],
            "large_balanced_timeout_sec": next(
                iter(
                    {
                        value
                        for label, value in reference_effective_timeouts.items()
                        if label.endswith(".large.balanced")
                    }
                )
            ),
            "includes": [
                "process startup",
                "data preparation",
                "estimator setup",
                "one warmup repetition",
                "three measured repetitions",
                "correctness validation",
            ],
        },
        "packages": _package_versions(candidate),
        "validation": {
            "paired": True,
            "successful_accelerated_execution": "gpu_only",
            "successful_results": "correctness validated",
        },
        "records": records,
    }


def normalize_supplemental(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    core_reference: dict[str, Any],
    core_candidate: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate and normalize the publish-gated ultra-wide PCA pair."""
    left = {item["case_label"]: item for item in reference["results"]}
    right = {item["case_label"]: item for item in candidate["results"]}
    expected = set(PCA_ULTRA_WIDE_CASES)
    if set(left) != expected or set(right) != expected:
        raise ValueError(
            "supplemental artifacts must contain exactly the two ultra-wide PCA cases"
        )

    for artifact, backend, device in (
        (reference, "sklearn", "cpu"),
        (candidate, "cuml_accel", "gpu"),
    ):
        workflow = artifact["run"]["extensions"]["ai.rapids.mlbench"]
        if workflow["backend"] != {"name": backend, "device": device}:
            raise ValueError(
                f"supplemental {backend} artifact has the wrong backend identity"
            )
        if (
            workflow["suite"] != "accel_performance_pca_ultra_wide"
            or workflow["tier"] != "standard"
        ):
            raise ValueError(
                "supplemental artifact has the wrong suite or tier"
            )
        if set(
            workflow["timeouts"]["effective_case_timeout_sec"].values()
        ) != {180.0}:
            raise ValueError(
                "supplemental cases must use a 180-second complete-case timeout"
            )

    if _package_versions(reference) != _package_versions(core_reference):
        raise ValueError(
            "supplemental CPU packages differ from the fixed-scale CPU artifact"
        )
    if _package_versions(candidate) != _package_versions(core_candidate):
        raise ValueError(
            "supplemental accelerated packages differ from the fixed-scale accelerated artifact"
        )
    if _image(reference) != _image(core_reference) or _image(
        candidate
    ) != _image(core_candidate):
        raise ValueError(
            "supplemental image provenance differs from the fixed-scale artifacts"
        )
    if (
        candidate["run"]["system"]["components"]
        != core_candidate["run"]["system"]["components"]
    ):
        raise ValueError(
            "supplemental accelerated system differs from the fixed-scale artifact"
        )

    records = []
    excluded = []
    for case_label in sorted(expected):
        cpu, gpu = left[case_label], right[case_label]
        _validate_repetitions(cpu, case_label)
        _validate_repetitions(gpu, case_label)
        spec = PCA_ULTRA_WIDE_CASES[case_label]
        declared = {"components": spec["components"], "solver": "auto"}
        if (
            cpu["parameters"]["declared"] != declared
            or gpu["parameters"]["declared"] != declared
        ):
            raise ValueError(
                f"supplemental parameters differ for {case_label}"
            )
        if (
            _dimension(cpu, "rows") != spec["rows"]
            or _dimension(cpu, "features") != spec["features"]
            or cpu["input"]["data_type"] != "float32"
            or cpu["input"]["attributes"]["seed"] != 42
        ):
            raise ValueError(
                f"supplemental input metadata differs for {case_label}"
            )
        cpu_time, gpu_time = _median_wall_time(cpu), _median_wall_time(gpu)
        profile = _execution_profile(gpu)
        cpu_quality = _metric(cpu, "decomposition_quality")
        gpu_quality = _metric(gpu, "decomposition_quality")
        delta = abs(cpu_quality - gpu_quality)
        validated = delta <= PCA_ULTRA_WIDE_TOLERANCE
        completed = (
            cpu.get("outcome", {}).get("status") == "success"
            and gpu.get("outcome", {}).get("status") == "success"
            and cpu_time is not None
            and gpu_time is not None
        )
        exclusion_reasons = []
        if not completed:
            exclusion_reasons.append("did not complete on both backends")
        if profile != "gpu_only":
            exclusion_reasons.append("did not record GPU-only execution")
        if not validated:
            exclusion_reasons.append(
                f"decomposition-quality delta {delta:.6g} exceeds {PCA_ULTRA_WIDE_TOLERANCE:g}"
            )
        record = {
            "case_label": case_label,
            "estimator": "pca",
            "estimator_name": "PCA",
            "operation": "fit_transform",
            "phase": "training",
            "workload_label": case_label.rsplit(".", 1)[-1],
            "supplemental": True,
            "rows": spec["rows"],
            "features": spec["features"],
            "components": spec["components"],
            "parameters": declared,
            "input_bytes": cpu["input"]["attributes"][
                "estimated_input_size_bytes"
            ],
            "cpu_median_wall_time_sec": cpu_time,
            "gpu_median_wall_time_sec": gpu_time,
            "speedup": cpu_time / gpu_time if completed else None,
            "timeout_side": None,
            "timeout_limit_sec": 180.0,
            "execution_profile": profile,
            "correctness_status": "validated" if validated else "failed",
            "validation": {
                "kind": "decomposition_quality",
                "tolerance": PCA_ULTRA_WIDE_TOLERANCE,
                "cpu_value": cpu_quality,
                "gpu_value": gpu_quality,
                "absolute_delta": delta,
            },
        }
        if exclusion_reasons:
            excluded.append(
                {"case_label": case_label, "reasons": exclusion_reasons}
            )
        else:
            records.append(record)
    if not records:
        raise ValueError(
            "supplemental publication gate failed: no case passed correctness and GPU-only validation"
        )
    if not any(record["speedup"] >= 2 for record in records):
        raise ValueError(
            "supplemental publication gate failed: no aligned result reached 2×"
        )

    summary = {
        "cases": len(records),
        "measured_cases": len(expected),
        "excluded": excluded,
        "strongest_speedup": max(record["speedup"] for record in records),
        "timeout_policy": {
            "scope": "complete isolated case",
            "case_timeout_sec": 180.0,
        },
        "validation": {
            "paired": True,
            "successful_accelerated_execution": "gpu_only",
            "kind": "decomposition_quality",
            "tolerance": PCA_ULTRA_WIDE_TOLERANCE,
        },
    }
    return records, summary


def merge_pca_ultra_wide(
    data: dict[str, Any],
    supplemental_records: list[dict[str, Any]],
    supplemental_summary: dict[str, Any],
) -> None:
    selected_label = "pca.fit_transform.feature_wide_2048"
    selected = [
        record
        for record in supplemental_records
        if record["case_label"] == selected_label
    ]
    if len(selected) != 1:
        raise ValueError(
            f"supplemental publication gate failed: {selected_label} is not publishable"
        )
    target_indexes = [
        index
        for index, record in enumerate(data["records"])
        if record["estimator"] == "pca"
        and record["operation"] == "fit_transform"
        and record["workload_label"] == "large.balanced"
    ]
    if len(target_indexes) != 1:
        raise ValueError(
            "expected exactly one core PCA fit_transform large record"
        )
    replacement = {
        **selected[0],
        "family": "Decomposition",
        "workload_label": "large.balanced",
        "record_source": "supplemental",
        "replaces_workload": "pca.fit_transform.large.balanced",
    }
    replacement.pop("supplemental", None)
    data["records"][target_indexes[0]] = replacement
    data["records"].sort(
        key=lambda record: (
            record["estimator"],
            record["operation"],
            record["workload_label"],
        )
    )
    data["summary"] = _summarize_records(data["records"])
    data["supplemental_summary"] = {
        **supplemental_summary,
        "cases": 1,
        "selected_case_label": selected_label,
        "replaced_workload": "pca.fit_transform.large.balanced",
    }


def _fmt_speedup(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}×"
    if value >= 10:
        return f"{value:.1f}×"
    return f"{value:.2f}×"


def _fmt_time(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 0.001:
        return f"{value * 1000:.2f} ms"
    if value < 1:
        return f"{value * 1000:.1f} ms"
    return f"{value:.3g} s"


def _fmt_bytes(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3g} GB"
    return f"{value / 1_000_000:.3g} MB"


def _display_workload(workload: str) -> str:
    return "large" if workload == "large.balanced" else workload


def _status_text(record: dict[str, Any]) -> str:
    side = record["timeout_side"]
    if side == "cpu":
        return "CPU timeout"
    if side == "gpu":
        return "GPU timeout"
    if side == "both":
        return "CPU + GPU timeout"
    if record["speedup"] is None:
        return "Timing unavailable"
    return _fmt_speedup(record["speedup"])


def _detail_status_text(record: dict[str, Any]) -> str:
    status = _status_text(record)
    if record["timeout_side"] is None:
        return status
    return f"{status} ({record['timeout_limit_sec'] / 60:g} min)"


def _mix(
    a: tuple[int, int, int], b: tuple[int, int, int], amount: float
) -> str:
    rgb = tuple(round(x + (y - x) * amount) for x, y in zip(a, b))
    return "#" + "".join(f"{value:02x}" for value in rgb)


def _speedup_color(value: float) -> str:
    neutral = (239, 241, 243)
    # Saturate at 16x in either direction while retaining a true 1x midpoint.
    strength = min(1.0, abs(math.log2(value)) / 4)
    target = (118, 185, 0) if value >= 1 else (222, 111, 87)
    return _mix(neutral, target, strength)


def render_heatmap(records: list[dict[str, Any]], phase: str) -> str:
    subset = [record for record in records if record["phase"] == phase]
    operations = sorted(
        {(record["estimator"], record["operation"]) for record in subset},
        key=lambda key: (
            -statistics.median(
                record["speedup"]
                for record in subset
                if (record["estimator"], record["operation"]) == key
                and record["speedup"] is not None
            ),
            key,
        ),
    )
    if phase == "inference":
        operations = operations[:INFERENCE_HEATMAP_OPERATION_LIMIT]
    by_cell = {
        (
            record["estimator"],
            record["operation"],
            record["workload_label"],
        ): record
        for record in subset
    }
    left, top, cell_w, cell_h = 235, 76, 128, 42
    width, height = (
        left + cell_w * len(WORKLOADS) + 20,
        top + cell_h * len(operations) + 46,
    )
    title = (
        "Training and combined-operation speedups"
        if phase == "training"
        else "Inference and transform speedups"
    )
    desc = (
        f"Heatmap of {len(operations)} operations across five workloads, ranked by median completed speedup. "
        "Each completed cell is labeled with CPU wall time divided by accelerated wall time; patterned cells label timeouts."
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="title desc" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        f'<title id="title">{html.escape(title)}</title>',
        f'<desc id="desc">{html.escape(desc)}</desc>',
        '<defs><pattern id="timeout" width="8" height="8" patternUnits="userSpaceOnUse"><rect width="8" height="8" fill="#eceff1"/><path d="M-2 2L2-2M0 8L8 0M6 10L10 6" stroke="#a5abb0" stroke-width="2"/></pattern><pattern id="cpu-timeout" width="8" height="8" patternUnits="userSpaceOnUse"><rect width="8" height="8" fill="#e0f0c7"/><path d="M-2 2L2-2M0 8L8 0M6 10L10 6" stroke="#76b900" stroke-width="2"/></pattern></defs>',
        "<style>text{font-family:Arial,sans-serif;fill:#242629}.label{font-size:13px}.head{font-size:12px;font-weight:700}.cell{font-size:12px;font-weight:700}.timeout{font-size:10px}.operation-link{text-decoration:none}.operation-link .label{fill:#242629}</style>",
    ]
    for col, workload in enumerate(WORKLOADS):
        x = left + col * cell_w + cell_w / 2
        label = _display_workload(workload).replace(".", " · ") + (
            "*" if workload == "large.balanced" else ""
        )
        parts.append(
            f'<text class="head" x="{x:g}" y="55" text-anchor="middle">{html.escape(label)}</text>'
        )
    for row, (estimator, operation) in enumerate(operations):
        y = top + row * cell_h
        operation_label = f"{DISPLAY_NAMES[estimator]}.{operation}"
        parts.append(
            f'<a class="operation-link" href="../../cuml-accel/benchmarks/#{ANCHORS[estimator]}" target="_top" '
            f'aria-label="{html.escape(operation_label)}; open {html.escape(DISPLAY_NAMES[estimator])} estimator details">'
            f"<title>{html.escape(operation_label)}</title>"
            f'<text class="label" x="{left - 10}" y="{y + 26}" text-anchor="end">'
            f"{html.escape(DISPLAY_NAMES[estimator])}</text></a>"
        )
        for col, workload in enumerate(WORKLOADS):
            record = by_cell[(estimator, operation, workload)]
            x = left + col * cell_w
            if record["speedup"] is None:
                fill = (
                    "url(#cpu-timeout)"
                    if record["timeout_side"] == "cpu"
                    else "url(#timeout)"
                )
                label = {
                    "cpu": "CPU timeout",
                    "gpu": "GPU timeout",
                    "both": "both timeout",
                }.get(record["timeout_side"], "unavailable")
                text_class = "cell timeout"
            else:
                fill = _speedup_color(record["speedup"])
                label = _fmt_speedup(record["speedup"])
                text_class = "cell"
            aria = f"{operation_label}, {_display_workload(workload)}: {_status_text(record)}"
            parts.append(
                f'<g role="img" aria-label="{html.escape(aria)}"><rect x="{x}" y="{y}" width="{cell_w - 4}" height="{cell_h - 4}" rx="3" fill="{fill}"/><text class="{text_class}" x="{x + (cell_w - 4) / 2:g}" y="{y + 25}" text-anchor="middle">{html.escape(label)}</text></g>'
            )
    parts.append(
        f'<text class="label" x="{left}" y="{height - 12}">* large is operation-specific; open estimator details for its actual rows, features, and input size.</text>'
    )
    parts.append("</svg>\n")
    return "".join(parts)


def _rst_list_table(
    caption: str,
    headers: list[str],
    rows: list[list[str]],
    *,
    table_class: str,
) -> str:
    lines = [
        f".. list-table:: {caption}",
        "   :header-rows: 1",
        f"   :class: {table_class}",
        "",
        "   * - " + headers[0],
    ]
    lines.extend(f"     - {header}" for header in headers[1:])
    for row in rows:
        lines.append("   * - " + row[0])
        lines.extend(f"     - {cell}" for cell in row[1:])
    return "\n".join(lines)


def _workload_guide_rst(records: list[dict[str, Any]]) -> str:
    rows = []
    for workload in WORKLOADS:
        subset = [
            record
            for record in records
            if record["workload_label"] == workload
        ]
        row_values = sorted({record["rows"] for record in subset})
        feature_values = sorted({record["features"] for record in subset})
        byte_values = sorted({record["input_bytes"] for record in subset})
        variable = (
            len(row_values) > 1
            or len(feature_values) > 1
            or len(byte_values) > 1
        )

        def values(items: list[int], formatter: Any) -> str:
            selected = (items[0], items[-1]) if len(items) > 1 else (items[0],)
            return "–".join(formatter(value) for value in selected)

        label = _display_workload(workload) + (" *" if variable else "")
        rows.append(
            [
                f"``{label}``",
                values(row_values, lambda value: f"{value:,}"),
                values(feature_values, lambda value: f"{value:,}"),
                values(byte_values, _fmt_bytes),
            ]
        )
    return _rst_list_table(
        "Workload dimensions and decimal float32 X size",
        ["Label", "Rows", "Features", "Input"],
        rows,
        table_class="benchmark-workload-table",
    )


def _estimator_details_rst(
    estimator: str, records: list[dict[str, Any]]
) -> str:
    subset = [record for record in records if record["estimator"] == estimator]
    show_components = any(
        record.get("components") is not None for record in subset
    )
    workload_order = {label: index for index, label in enumerate(WORKLOADS)}
    headers = ["Operation", "Workload", "Rows", "Features", "Input"]
    if show_components:
        headers.append("Components")
    headers.extend(["CPU", "GPU", "Result"])
    rows = []
    for record in sorted(
        subset,
        key=lambda item: (
            item["operation"],
            workload_order.get(item["workload_label"], len(WORKLOADS)),
        ),
    ):
        row = [
            f"``{record['operation']}``",
            f"``{_display_workload(record['workload_label'])}``",
            f"{record['rows']:,}",
            f"{record['features']:,}",
            _fmt_bytes(record["input_bytes"]),
        ]
        if show_components:
            row.append(
                f"{record['components']:,}"
                if record.get("components") is not None
                else "—"
            )
        row.extend(
            [
                _fmt_time(record["cpu_median_wall_time_sec"]),
                _fmt_time(record["gpu_median_wall_time_sec"]),
                _detail_status_text(record),
            ]
        )
        rows.append(row)
    table = _rst_list_table(
        f"{DISPLAY_NAMES[estimator]} results for all measured operations and workloads",
        headers,
        rows,
        table_class="benchmark-result-table",
    )
    indented_table = "\n".join(
        f"   {line}" if line else "" for line in table.splitlines()
    )
    return (
        f".. dropdown:: {DISPLAY_NAMES[estimator]}\n"
        f"   :name: {ANCHORS[estimator]}\n\n"
        f"{indented_table}\n"
    )


def _estimator_sections_rst(records: list[dict[str, Any]]) -> str:
    sections = []
    for family, estimators in FAMILIES.items():
        sections.append(f".. rubric:: {family}\n")
        sections.extend(
            _estimator_details_rst(estimator, records)
            for estimator in estimators
        )
    return "\n".join(sections)


def _strongest(records: list[dict[str, Any]], estimator: str) -> float:
    return max(
        record["speedup"]
        for record in records
        if record["estimator"] == estimator and record["speedup"] is not None
    )


def render_rst(data: dict[str, Any], template: str) -> str:
    records = data["records"]
    summary = data["summary"]
    training = summary["phases"]["training"]
    inference = summary["phases"]["inference"]
    timeouts = data["timeout_policy"]
    components = {item["type"]: item for item in data["system"]["components"]}
    gpu = components["gpu"]
    cpu = components["cpu"]
    memory = components["memory"]
    packages = ", ".join(
        f"``{name} {version}``"
        for name, version in sorted(data["packages"].items())
    )
    replacements = {
        "TRAINING_MEDIAN": f"{training['median_speedup']:.1f}",
        "UMAP_SPEEDUP": f"{_strongest(records, 'umap'):.0f}×",
        "HDBSCAN_SPEEDUP": f"{_strongest(records, 'hdbscan'):.0f}×",
        "TRAINING_PAIRED": str(training["paired"]),
        "TRAINING_CELLS": str(training["cells"]),
        "TRAINING_2X": str(training["at_least_2x"]),
        "INFERENCE_MEDIAN": f"{inference['median_speedup']:.2f}",
        "INFERENCE_SLOWDOWNS": str(inference["slowdowns"]),
        "INFERENCE_PAIRED": str(inference["paired"]),
        "DEFAULT_TIMEOUT_MIN": f"{timeouts['default_case_timeout_sec'] / 60:g}",
        "LARGE_TIMEOUT_MIN": f"{timeouts['large_balanced_timeout_sec'] / 60:g}",
        "WORKLOAD_TABLE": _workload_guide_rst(records),
        "ESTIMATOR_SECTIONS": "\n".join(
            f"   {line}" if line else ""
            for line in _estimator_sections_rst(records).splitlines()
        ),
        "GPU_NAME": gpu["name"],
        "GPU_MEMORY_GB": f"{gpu['attributes']['total_memory_bytes'] / 1_000_000_000:.1f}",
        "CPU_NAME": cpu["name"],
        "SYSTEM_MEMORY_GB": f"{memory['attributes']['total_memory_bytes'] / 1_000_000_000:.1f}",
        "PACKAGES": packages,
        "CPU_TIMEOUTS": str(summary["timeouts"]["cpu"]),
        "GPU_TIMEOUTS": str(summary["timeouts"]["gpu"]),
        "BOTH_TIMEOUTS": str(summary["timeouts"]["both"]),
        "CASES": str(summary["cases"]),
        "UNAVAILABLE": str(summary["unavailable"]),
    }
    rendered = template
    for name, value in replacements.items():
        rendered = rendered.replace(f"@@{name}@@", value)
    unresolved = sorted(
        set(part.split("@@", 1)[0] for part in rendered.split("@@")[1::2])
    )
    if unresolved:
        raise ValueError(f"unresolved template placeholders: {unresolved}")
    return rendered.rstrip() + "\n"


def build_data(
    reference_path: Path,
    candidate_path: Path,
    supplemental_reference_path: Path,
    supplemental_candidate_path: Path,
) -> dict[str, Any]:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    data = normalize(reference, candidate)
    supplemental_reference = json.loads(
        supplemental_reference_path.read_text(encoding="utf-8")
    )
    supplemental_candidate = json.loads(
        supplemental_candidate_path.read_text(encoding="utf-8")
    )
    supplemental_records, supplemental_summary = normalize_supplemental(
        supplemental_reference,
        supplemental_candidate,
        core_reference=reference,
        core_candidate=candidate,
    )
    merge_pca_ultra_wide(data, supplemental_records, supplemental_summary)
    return data


def render_files(data: dict[str, Any], template: str) -> dict[Path, str]:
    return {
        DEFAULT_PAGE: render_rst(data, template),
        DEFAULT_STATIC / "training-heatmap.svg": render_heatmap(
            data["records"], "training"
        ),
        DEFAULT_STATIC / "inference-heatmap.svg": render_heatmap(
            data["records"], "inference"
        ),
    }


def _write_or_check(files: dict[Path, str], *, check: bool) -> bool:
    differences = []
    for path, content in sorted(files.items(), key=lambda item: str(item[0])):
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            differences.append(path)
            if not check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
    if check and differences:
        print(
            "generated benchmark documentation is out of date: "
            + ", ".join(map(str, differences)),
            file=sys.stderr,
        )
        return False
    return True


def sphinx_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    import_parser = subparsers.add_parser(
        "import", help="validate raw results and write curated data"
    )
    import_parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="core CPU result artifact",
    )
    import_parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="core accelerated result artifact",
    )
    import_parser.add_argument(
        "--supplemental-reference",
        type=Path,
        required=True,
        help="PCA CPU result artifact",
    )
    import_parser.add_argument(
        "--supplemental-candidate",
        type=Path,
        required=True,
        help="PCA accelerated result artifact",
    )
    import_parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA, help="curated output JSON"
    )
    import_parser.add_argument(
        "--check",
        action="store_true",
        help="fail instead of writing when curated data differs",
    )
    render_parser = subparsers.add_parser(
        "render", help="render RST and SVGs from curated data"
    )
    render_parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA, help="curated input JSON"
    )
    render_parser.add_argument(
        "--template", type=Path, default=DEFAULT_TEMPLATE, help="RST template"
    )
    render_parser.add_argument(
        "--check",
        action="store_true",
        help="fail instead of writing when generated files differ",
    )
    args = parser.parse_args(argv)
    if args.command == "import":
        data = build_data(
            args.reference.resolve(),
            args.candidate.resolve(),
            args.supplemental_reference.resolve(),
            args.supplemental_candidate.resolve(),
        )
        content = (
            json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)
            + "\n"
        )
        return (
            0
            if _write_or_check(
                {args.data.resolve(): content}, check=args.check
            )
            else 1
        )
    data = json.loads(args.data.read_text(encoding="utf-8"))
    template = args.template.read_text(encoding="utf-8")
    return (
        0
        if _write_or_check(render_files(data, template), check=args.check)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(sphinx_main())
