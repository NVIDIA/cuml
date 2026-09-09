#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synchronize and render the generated cuml.accel Sphinx benchmark page.

``sync`` validates portable publication data produced by cumlbench-dash,
copies it into the documentation tree, and renders the page and heatmaps.
``render`` uses the checked-in publication data for normal documentation
builds.
"""

from __future__ import annotations

import argparse
import copy
import html
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "docs/benchmarks/cuml-accel/benchmark-data.json"
DEFAULT_TEMPLATE = ROOT / "docs/source/cuml-accel/benchmarks.rst.in"
DEFAULT_PAGE = ROOT / "docs/source/cuml-accel/benchmarks.rst"
DEFAULT_STATIC = ROOT / "docs/source/_static/cuml-accel-benchmarks"
SOURCE_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")

# The portable performance summary deliberately contains no host metadata.
PUBLICATION_SYSTEM = {
    "components": [
        {
            "attributes": {"logical_cores": 64, "physical_cores": 32},
            "count": 1,
            "name": "AMD Ryzen Threadripper PRO 7975WX 32-Cores",
            "type": "cpu",
        },
        {
            "attributes": {"total_memory_bytes": 134137659392},
            "count": 1,
            "name": "System memory",
            "type": "memory",
        },
        {
            "attributes": {"total_memory_bytes": 101964644352},
            "count": 1,
            "name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
            "type": "gpu",
        },
    ]
}
# rapids-pre-commit-hooks: disable-next-line
PUBLICATION_PACKAGES = {"cuml": "26.10.0a69", "scikit-learn": "1.9.0"}

WORKLOADS = (
    "small.balanced",
    "medium.thin",
    "medium.balanced",
    "medium.wide",
    "large.balanced",
)
TRAINING_OPERATIONS = {"fit", "fit_predict", "fit_transform"}
INFERENCE_OPERATIONS = {
    "predict",
    "transform",
    "score_samples",
    "kneighbors",
}
INFERENCE_HEATMAP_MAX_OPERATIONS = 10

DISPLAY_NAMES = {
    "dbscan": "DBSCAN",
    "elastic_net": "ElasticNet",
    "hdbscan": "HDBSCAN",
    "k_neighbors_classifier": "KNeighborsClassifier",
    "k_neighbors_regressor": "KNeighborsRegressor",
    "kernel_density": "KernelDensity",
    "kmeans": "KMeans",
    "lasso": "Lasso",
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
    "tsne": "t-SNE",
    "umap": "UMAP",
}

FAMILIES = {
    "Linear models": (
        "elastic_net",
        "lasso",
        "linear_regression",
        "logistic_regression",
        "ridge",
    ),
    "Clustering and manifold learning": (
        "dbscan",
        "hdbscan",
        "kmeans",
        "tsne",
        "umap",
    ),
    "Neighbors and density estimation": (
        "k_neighbors_classifier",
        "k_neighbors_regressor",
        "kernel_density",
        "nearest_neighbors",
    ),
    "Decomposition": ("pca",),
    "Ensembles": ("random_forest_classifier", "random_forest_regressor"),
    "Preprocessing": ("standard_scaler", "target_encoder"),
    "Kernel methods": ("svc",),
}
FAMILY_ESTIMATORS = {
    estimator for estimators in FAMILIES.values() for estimator in estimators
}

ANCHORS = {
    "dbscan": "benchmark-dbscan",
    "elastic_net": "benchmark-elasticnet",
    "hdbscan": "benchmark-hdbscan",
    "k_neighbors_classifier": "benchmark-kneighborsclassifier",
    "k_neighbors_regressor": "benchmark-kneighborsregressor",
    "kernel_density": "benchmark-kerneldensity",
    "kmeans": "benchmark-kmeans",
    "lasso": "benchmark-lasso",
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
    "tsne": "benchmark-tsne",
    "umap": "benchmark-umap",
}


def _require_mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object")
    return value


def _require_keys(
    value: dict[str, Any], keys: set[str], location: str
) -> None:
    missing = sorted(keys - value.keys())
    if missing:
        raise ValueError(f"{location} is missing required fields: {missing}")


def _is_source_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and SOURCE_ID_PATTERN.fullmatch(value) is not None
    )


def _positive_number(value: Any, location: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{location} must be finite and positive")
    return value


def _parse_case_label(label: Any) -> tuple[str, str, str, str, int | None]:
    if not isinstance(label, str):
        raise ValueError("publication data case labels must be strings")
    parts = label.split(".")
    if len(parts) == 4:
        estimator, operation, size, shape = parts
        return estimator, operation, size, shape, None
    if len(parts) == 5 and parts[2].startswith("rank"):
        estimator, operation, rank_label, size, shape = parts
        match = re.fullmatch(r"rank([1-9]\d*)", rank_label)
        if match is None:
            raise ValueError(f"invalid rank-qualified case label: {label}")
        return estimator, operation, size, shape, int(match.group(1))
    raise ValueError(f"unsupported case label: {label}")


def validate_publication_data(data: Any) -> dict[str, Any]:
    """Validate the compact publication contract consumed by the renderer."""
    data = _require_mapping(data, "publication data")
    if set(data) != {"schema_version", "records"}:
        raise ValueError(
            "publication data must contain only schema_version and records"
        )
    if data.get("schema_version") != 1:
        raise ValueError(
            f"unsupported benchmark publication schema version "
            f"{data.get('schema_version')!r}; expected 1"
        )
    records = data["records"]
    if not isinstance(records, list) or len(records) != 168:
        raise ValueError(
            "publication data records must contain exactly 168 entries"
        )
    labels = set()
    required = {
        "case_label",
        "cpu_median_sec",
        "gpu_median_sec",
        "rows",
        "features",
    }
    optional = {"cpu_timeout_sec", "components"}
    for index, value in enumerate(records):
        record = _require_mapping(value, f"records[{index}]")
        keys = set(record)
        if not required.issubset(keys) or not keys.issubset(
            required | optional
        ):
            raise ValueError(
                f"records[{index}] has unsupported or missing fields"
            )
        label = record["case_label"]
        estimator, operation, size, shape, rank = _parse_case_label(label)
        if estimator not in FAMILY_ESTIMATORS:
            raise ValueError(
                f"records[{index}] has unsupported estimator {estimator!r}"
            )
        if operation not in TRAINING_OPERATIONS | INFERENCE_OPERATIONS:
            raise ValueError(
                f"records[{index}] has unsupported operation {operation!r}"
            )
        workload = f"{size}.{shape}"
        if workload not in WORKLOADS:
            raise ValueError(
                f"records[{index}] has unsupported workload {workload!r}"
            )
        if rank is not None and (
            estimator,
            operation,
            workload,
        ) != ("pca", "fit_transform", "medium.wide"):
            raise ValueError(
                f"records[{index}] has unsupported rank combination"
            )
        if label in labels:
            raise ValueError("publication data case labels must be unique")
        labels.add(label)
        for field in ("rows", "features"):
            if (
                not isinstance(record[field], int)
                or isinstance(record[field], bool)
                or record[field] <= 0
            ):
                raise ValueError(
                    f"records[{index}].{field} must be a positive integer"
                )
        _positive_number(
            record["gpu_median_sec"], f"records[{index}].gpu_median_sec"
        )
        cpu_time = record["cpu_median_sec"]
        timeout = record.get("cpu_timeout_sec")
        if cpu_time is None:
            _positive_number(timeout, f"records[{index}].cpu_timeout_sec")
        else:
            _positive_number(cpu_time, f"records[{index}].cpu_median_sec")
            if timeout is not None:
                raise ValueError(
                    f"records[{index}] has a timeout and CPU timing"
                )
        components = record.get("components")
        if estimator == "pca" and rank is None:
            if (
                not isinstance(components, int)
                or isinstance(components, bool)
                or components <= 0
            ):
                raise ValueError(f"records[{index}] lacks PCA components")
        elif components is not None:
            raise ValueError(f"records[{index}] has redundant components")
    return data


def load_publication_data(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid publication JSON at {path}: {error}"
        ) from error
    return validate_publication_data(data)


def _presentation_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return records


def _heatmap_records(
    records: list[dict[str, Any]], phase: str
) -> list[dict[str, Any]]:
    selected = [
        record
        for record in records
        if record["phase"] == phase
        and not record.get("is_rank_variant", False)
    ]
    return [
        {
            **record,
            "heatmap_detail": "medium-wide · 1,024 components",
        }
        if record["estimator"] == "pca"
        and record["operation"] == "fit_transform"
        and record["workload_label"] == "medium.wide"
        else record
        for record in selected
    ]


def _exact_speedups(records: list[dict[str, Any]]) -> list[float]:
    return [
        record["speedup"]
        for record in records
        if record["speedup"] is not None
        and not record.get("speedup_is_lower_bound", False)
    ]


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    phases: dict[str, dict[str, Any]] = {}
    for phase in ("training", "inference"):
        phase_records = [
            record for record in records if record["phase"] == phase
        ]
        paired = _exact_speedups(phase_records)
        phases[phase] = {
            "cells": len(phase_records),
            "paired": len(paired),
            "median_speedup": statistics.median(paired),
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


def _prepare_publication(data: dict[str, Any]) -> dict[str, Any]:
    records = []
    for source in data["records"]:
        estimator, operation, size, shape, rank = _parse_case_label(
            source["case_label"]
        )
        cpu_time = source["cpu_median_sec"]
        gpu_time = source["gpu_median_sec"]
        timeout = source.get("cpu_timeout_sec")
        components = rank if rank is not None else source.get("components")
        records.append(
            {
                "case_label": source["case_label"],
                "estimator": estimator,
                "operation": operation,
                "workload_label": f"{size}.{shape}",
                "parameters": {"components": components},
                "is_rank_variant": rank is not None,
                "execution_profile": "gpu_only",
                "unavailable_side": None,
                "unavailable_reason": None,
                "phase": (
                    "training"
                    if operation in TRAINING_OPERATIONS
                    else "inference"
                ),
                "family": next(
                    family
                    for family, members in FAMILIES.items()
                    if estimator in members
                ),
                "rows": source["rows"],
                "features": source["features"],
                "input_bytes": source["rows"] * source["features"] * 4,
                "cpu_median_wall_time_sec": cpu_time,
                "gpu_median_wall_time_sec": gpu_time,
                "speedup": (
                    cpu_time / gpu_time
                    if cpu_time is not None
                    else timeout / gpu_time
                ),
                "speedup_is_lower_bound": cpu_time is None,
                "timeout_side": "cpu" if cpu_time is None else None,
                "timeout_limit_sec": timeout,
            }
        )
    prepared = {
        "schema_version": 1,
        "records": records,
        "system": copy.deepcopy(PUBLICATION_SYSTEM),
        "packages": copy.deepcopy(PUBLICATION_PACKAGES),
        "methodology": {"id": "mlbench-accel-performance"},
        "validation": {"successful_accelerated_execution": "gpu_only"},
    }
    prepared["summary"] = _summarize(records)
    return prepared


def prepare_publication_data(data: Any) -> dict[str, Any]:
    """Normalize publication data for rendering."""
    return _prepare_publication(validate_publication_data(data))


def _fmt_speedup(value: float) -> str:
    """Format a measured value for tables and heatmaps."""
    if value >= 100:
        return f"{value:.0f}×"
    if value >= 10:
        return f"{value:.1f}×"
    return f"{value:.2f}×"


def _fmt_prose_speedup(value: float, *, lower_bound: bool = False) -> str:
    """Round a result for narrative prose without changing result displays."""
    if value >= 100:
        rounded = round(value / 100) * 100
        formatted = f"{rounded:.0f}×"
    elif value >= 10:
        rounded = round(value / 10) * 10
        formatted = f"{rounded:.0f}×"
    else:
        formatted = f"{value:.1f}×"
    if lower_bound:
        return f"an approximate lower bound of ≥{formatted}"
    return f"approximately {formatted}"


def _fmt_time(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 0.001:
        return f"{value * 1000:.2f} ms"
    if value < 1:
        return f"{value * 1000:.1f} ms"
    return f"{value:.3g} s"


def _fmt_throughput(rows: int, seconds: float | None) -> str:
    if seconds is None:
        return "—"
    throughput = rows / seconds
    if throughput >= 1_000_000:
        return f"{throughput / 1_000_000:.3g}M/s"
    if throughput >= 1_000:
        return f"{throughput / 1_000:.3g}k/s"
    return f"{throughput:.3g}/s"


def _fmt_backend_result(record: dict[str, Any], backend: str) -> str:
    seconds = record[f"{backend}_median_wall_time_sec"]
    wall_time = _fmt_time(seconds)
    if record["phase"] == "inference" and seconds is not None:
        throughput = _fmt_throughput(record["rows"], seconds)
        return (
            f":benchmark-throughput:`{throughput}` "
            f":benchmark-time:`{wall_time}`"
        )
    return wall_time


def _fmt_bytes(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3g} GB"
    megabytes = value / 1_000_000
    formatted = f"{megabytes:.3g}"
    if "e" in formatted.lower():
        formatted = f"{megabytes:,.0f}"
    return f"{formatted} MB"


def _display_workload(workload: str) -> str:
    return "large" if workload == "large.balanced" else workload


def _status_text(record: dict[str, Any]) -> str:
    if record.get("speedup_is_lower_bound"):
        return f"≥{_fmt_speedup(record['speedup'])} (CPU timeout)"
    if record["unavailable_side"] == "cpu":
        return "CPU unavailable"
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


def _fmt_timeout_limit(seconds: float) -> str:
    if seconds < 120:
        return f"{round(seconds / 10) * 10:g} s"
    return f"{round(seconds / 60):g} min"


def _detail_status_text(record: dict[str, Any]) -> str:
    status = _status_text(record)
    if record["unavailable_side"] is not None:
        return f"{status} ({record['unavailable_reason']})"
    if record.get("speedup_is_lower_bound"):
        return f"≥{_fmt_speedup(record['speedup'])}"
    if record["timeout_side"] is None:
        return status
    return "—"


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
    subset = _heatmap_records(records, phase)
    operation_keys = {
        (record["estimator"], record["operation"]) for record in subset
    }
    exact_by_operation = {
        key: _exact_speedups(
            [
                record
                for record in subset
                if (record["estimator"], record["operation"]) == key
            ]
        )
        for key in operation_keys
    }
    operations = sorted(
        (key for key in operation_keys if exact_by_operation[key]),
        key=lambda key: (-statistics.median(exact_by_operation[key]), key),
    )
    if phase == "inference":
        operations = operations[:INFERENCE_HEATMAP_MAX_OPERATIONS]
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
        top + cell_h * len(operations) + 20,
    )
    title = (
        "Training and combined-operation speedups"
        if phase == "training"
        else "Inference and transform speedups"
    )
    desc = (
        f"Heatmap of {len(operations)} operations across five workloads, ranked by median exact speedup. "
        "Each exact cell is labeled with CPU wall time divided by accelerated wall time; patterned cells mark CPU-timeout lower bounds and unavailable results."
    )
    lower_bound_colors = {
        _speedup_color(record["speedup"])
        for record in subset
        if record.get("speedup_is_lower_bound")
        and record["speedup"] is not None
    }
    lower_bound_patterns = "".join(
        f'<pattern id="lower-bound-{color[1:]}" width="10" height="10" patternUnits="userSpaceOnUse">'
        f'<rect width="10" height="10" fill="{color}"/>'
        '<path d="M-2 2L2-2M0 10L10 0M8 12L12 8" stroke="#4f6500" stroke-width="1" stroke-opacity="0.55"/></pattern>'
        for color in sorted(lower_bound_colors)
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="title desc" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        f'<title id="title">{html.escape(title)}</title>',
        f'<desc id="desc">{html.escape(desc)}</desc>',
        '<defs><pattern id="timeout" width="10" height="10" patternUnits="userSpaceOnUse"><rect width="10" height="10" fill="#eceff1"/><path d="M-2 2L2-2M0 10L10 0M8 12L12 8" stroke="#8f969c" stroke-width="1" stroke-opacity="0.6"/></pattern>'
        + lower_bound_patterns
        + "</defs>",
        "<style>text{font-family:Arial,sans-serif;fill:#242629}.label{font-size:13px}.head{font-size:12px;font-weight:700}.cell{font-size:12px;font-weight:700}.timeout{font-size:10px}.operation-link{text-decoration:none}.operation-link .label{fill:#242629}</style>",
    ]
    for col, workload in enumerate(WORKLOADS):
        x = left + col * cell_w + cell_w / 2
        label = _display_workload(workload).replace(".", " · ")
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
            record = by_cell.get((estimator, operation, workload))
            if record is None:
                continue
            x = left + col * cell_w
            if record["speedup"] is None:
                fill = (
                    "url(#cpu-timeout)"
                    if record["timeout_side"] == "cpu"
                    else "url(#timeout)"
                )
                label = (
                    "CPU unavailable"
                    if record["unavailable_side"] == "cpu"
                    else {
                        "cpu": "CPU timeout",
                        "gpu": "GPU timeout",
                        "both": "both timeout",
                    }.get(record["timeout_side"], "unavailable")
                )
                text_class = "cell timeout"
            else:
                color = _speedup_color(record["speedup"])
                fill = (
                    f"url(#lower-bound-{color[1:]})"
                    if record.get("speedup_is_lower_bound")
                    else color
                )
                label = (
                    "≥" if record.get("speedup_is_lower_bound") else ""
                ) + _fmt_speedup(record["speedup"])
                text_class = "cell"
            cell_detail = record.get("heatmap_detail")
            aria = (
                f"{operation_label}, {_display_workload(workload)}"
                f"{f' · {cell_detail}' if cell_detail else ''}: "
                f"{_status_text(record)}"
            )
            parts.append(
                f'<g role="img" aria-label="{html.escape(aria)}"><rect x="{x}" y="{y}" width="{cell_w - 4}" height="{cell_h - 4}" rx="3" fill="{fill}"/><text class="{text_class}" x="{x + (cell_w - 4) / 2:g}" y="{y + 25}" text-anchor="middle">{html.escape(label)}</text></g>'
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

        def values(items: list[int], formatter: Any) -> str:
            selected = (items[0], items[-1]) if len(items) > 1 else (items[0],)
            return "–".join(formatter(value) for value in selected)

        label = _display_workload(workload)
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
    workload_order = {label: index for index, label in enumerate(WORKLOADS)}
    headers = ["Operation", "Workload", "Rows", "Features", "Input"]
    headers.extend(["CPU", "GPU", "Result"])
    rows = []
    for record in sorted(
        subset,
        key=lambda item: (
            item["operation"],
            workload_order.get(item["workload_label"], len(WORKLOADS)),
            item["parameters"].get("components") or 0,
        ),
    ):
        workload_label = _display_workload(record["workload_label"])
        if estimator == "pca":
            components = record["parameters"]["components"]
            workload_label += f" · {components:,} components"
        row = [
            f"``{record['operation']}``",
            f"``{workload_label}``",
            f"{record['rows']:,}",
            f"{record['features']:,}",
            _fmt_bytes(record["input_bytes"]),
        ]
        cpu_time = _fmt_backend_result(record, "cpu")
        gpu_time = _fmt_backend_result(record, "gpu")
        if record["timeout_side"] in {"cpu", "both"}:
            cpu_time = (
                f"Timeout at {_fmt_timeout_limit(record['timeout_limit_sec'])}"
            )
        if record["timeout_side"] in {"gpu", "both"}:
            gpu_time = (
                f"Timeout at {_fmt_timeout_limit(record['timeout_limit_sec'])}"
            )
        row.extend([cpu_time, gpu_time, _detail_status_text(record)])
        rows.append(row)
    table = _rst_list_table(
        f"{DISPLAY_NAMES[estimator]} results for all measured operations and workloads",
        headers,
        rows,
        table_class="benchmark-result-table",
    )
    if estimator == "pca":
        table += (
            "\n\nPCA performance depends strongly on both input feature width "
            "and the number of retained components; results can vary "
            "substantially across these dimensions."
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
            if any(record["estimator"] == estimator for record in records)
        )
    return "\n".join(sections)


def _pca_rank_results_rst(records: list[dict[str, Any]]) -> str:
    rank_records = [
        record
        for record in records
        if record["estimator"] == "pca"
        and record["operation"] == "fit_transform"
        and record["workload_label"] == "medium.wide"
    ]
    rows = []
    for record in sorted(
        rank_records, key=lambda item: item["parameters"]["components"]
    ):
        result = (
            f"{_fmt_time(record['cpu_median_wall_time_sec'])} / "
            f"{_fmt_time(record['gpu_median_wall_time_sec'])} / "
            f"{_fmt_speedup(record['speedup'])}"
        )
        rows.append([f"{record['parameters']['components']:,}", result])
    return _rst_list_table(
        "Medium-wide PCA fit-transform by component rank",
        ["Components", "PCA CPU / GPU / result"],
        rows,
        table_class="benchmark-result-table",
    )


def _is_pca_large(record: dict[str, Any]) -> bool:
    return (
        record["estimator"] == "pca"
        and record["operation"] == "fit_transform"
        and record["workload_label"] == "large.balanced"
    )


def render_rst(data: dict[str, Any], template: str) -> str:
    data = prepare_publication_data(data)
    records = _presentation_records(data["records"])
    summary = _summarize(
        [
            record
            for record in records
            if not record.get("is_rank_variant", False)
        ]
    )
    training = summary["phases"]["training"]
    pca_large = next(record for record in records if _is_pca_large(record))
    components = {item["type"]: item for item in data["system"]["components"]}
    gpu = components["gpu"]
    cpu = components["cpu"]
    memory = components["memory"]
    packages = ", ".join(
        f"``{name} {version}``"
        for name, version in sorted(data["packages"].items())
    )
    replacements = {
        "TRAINING_PROSE_SPEEDUP": _fmt_prose_speedup(
            training["median_speedup"]
        ),
        "WORKLOAD_TABLE": _workload_guide_rst(records),
        "ESTIMATOR_SECTIONS": "\n".join(
            f"   {line}" if line else ""
            for line in _estimator_sections_rst(records).splitlines()
        ),
        "PCA_RANK_RESULTS": _pca_rank_results_rst(records),
        "GPU_NAME": gpu["name"],
        "GPU_MEMORY_GB": f"{gpu['attributes']['total_memory_bytes'] / 1_000_000_000:.1f}",
        "CPU_NAME": cpu["name"],
        "SYSTEM_MEMORY_GB": f"{memory['attributes']['total_memory_bytes'] / 1_000_000_000:.1f}",
        "PACKAGES": packages,
        "CPU_TIMEOUTS": str(summary["timeouts"]["cpu"]),
        "GPU_TIMEOUTS": str(summary["timeouts"]["gpu"]),
        "BOTH_TIMEOUTS": str(summary["timeouts"]["both"]),
        "PCA_LARGE_ROWS": f"{pca_large['rows']:,}",
        "PCA_LARGE_FEATURES": f"{pca_large['features']:,}",
    }
    rendered = template.replace(
        ".. This file is the editable template for benchmarks.rst. Run\n"
        ".. docs/benchmarks/generate_cuml_accel_benchmarks.py render after editing it.\n",
        ".. Generated from benchmarks.rst.in; do not edit this file directly.\n",
    )
    for name, value in replacements.items():
        rendered = rendered.replace(f"@@{name}@@", value)
    unresolved = sorted(
        set(part.split("@@", 1)[0] for part in rendered.split("@@")[1::2])
    )
    if unresolved:
        raise ValueError(f"unresolved template placeholders: {unresolved}")
    return rendered.rstrip() + "\n"


def render_files(data: dict[str, Any], template: str) -> dict[Path, str]:
    prepared = prepare_publication_data(data)
    records = _presentation_records(prepared["records"])
    return {
        DEFAULT_PAGE: render_rst(data, template),
        DEFAULT_STATIC / "training-heatmap.svg": render_heatmap(
            records, "training"
        ),
        DEFAULT_STATIC / "inference-heatmap.svg": render_heatmap(
            records, "inference"
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
            "cuml.accel benchmark files are out of date: "
            + ", ".join(map(str, differences)),
            file=sys.stderr,
        )
        return False
    return True


def sphinx_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    sync_parser = subparsers.add_parser(
        "sync", help="synchronize portable publication data and render outputs"
    )
    sync_parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="portable schema-v1 publication artifact from cumlbench-dash",
    )
    sync_parser.add_argument(
        "--template", type=Path, default=DEFAULT_TEMPLATE, help="RST template"
    )
    sync_parser.add_argument(
        "--check",
        action="store_true",
        help="verify supplied data, checked-in data, and rendered files",
    )
    render_parser = subparsers.add_parser(
        "render", help="render RST and SVGs from checked-in publication data"
    )
    render_parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="publication input JSON",
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
    if args.command == "sync":
        supplied_path = args.data.resolve()
        data = load_publication_data(supplied_path)
        content = supplied_path.read_text(encoding="utf-8")
        template = args.template.read_text(encoding="utf-8")
        if args.check:
            load_publication_data(DEFAULT_DATA)
        files = {
            DEFAULT_DATA: content,
            **render_files(data, template),
        }
        return 0 if _write_or_check(files, check=args.check) else 1
    data = load_publication_data(args.data)
    template = args.template.read_text(encoding="utf-8")
    return (
        0
        if _write_or_check(render_files(data, template), check=args.check)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(sphinx_main())
