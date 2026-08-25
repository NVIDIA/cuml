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
PUBLICATION_SCHEMA_VERSION = 1
SOURCE_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")

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


def validate_publication_data(data: Any) -> dict[str, Any]:
    """Validate the portable publication contract consumed by the renderer."""
    data = _require_mapping(data, "publication data")
    version = data.get("schema_version")
    if version != PUBLICATION_SCHEMA_VERSION:
        raise ValueError(
            "unsupported benchmark publication schema version "
            f"{version!r}; expected {PUBLICATION_SCHEMA_VERSION}"
        )
    _require_keys(
        data,
        {
            "definition",
            "summary",
            "system",
            "methodology",
            "packages",
            "validation",
            "sources",
            "records",
        },
        "publication data",
    )

    sources = data["sources"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("publication sources must be a non-empty array")
    allowed_source_fields = {
        "id",
        "run_id",
        "backend",
        "completed_at",
        "suite",
        "tier",
        "manifest_hash",
        "resolved_plan_hash",
    }
    sources_by_id = {}
    for index, source_value in enumerate(sources):
        source = _require_mapping(source_value, f"sources[{index}]")
        if set(source) != allowed_source_fields:
            raise ValueError(
                f"sources[{index}] has unsupported or missing provenance fields"
            )
        source_id = source["id"]
        if not _is_source_id(source_id):
            raise ValueError(f"sources[{index}].id is not a SHA-256 source ID")
        if source_id in sources_by_id:
            raise ValueError(f"duplicate publication source ID: {source_id}")
        if source["backend"] not in {"cpu", "gpu"}:
            raise ValueError(f"sources[{index}].backend is unsupported")
        for field in ("manifest_hash", "resolved_plan_hash"):
            if not _is_source_id(source[field]):
                raise ValueError(
                    f"sources[{index}].{field} is not a SHA-256 ID"
                )
        sources_by_id[source_id] = source

    records = data["records"]
    if not isinstance(records, list) or len(records) != 147:
        raise ValueError(
            "publication records must contain exactly 147 entries"
        )
    summary = _require_mapping(data["summary"], "summary")
    if summary.get("cases") != len(records):
        raise ValueError(
            "summary case count does not match publication records"
        )
    record_source_fields = {"cpu_source_id": "cpu", "gpu_source_id": "gpu"}
    case_labels = set()
    for index, record_value in enumerate(records):
        record = _require_mapping(record_value, f"records[{index}]")
        _require_keys(
            record,
            {
                "estimator",
                "operation",
                "workload_label",
                "case_label",
                "parameters",
                "execution_profile",
                "cpu_source_id",
                "gpu_source_id",
            },
            f"records[{index}]",
        )
        _require_mapping(record["parameters"], f"records[{index}].parameters")
        case_label = record["case_label"]
        if case_label in case_labels:
            raise ValueError(f"duplicate publication case label: {case_label}")
        case_labels.add(case_label)
        for field, backend in record_source_fields.items():
            source_id = record[field]
            if source_id not in sources_by_id:
                raise ValueError(
                    f"records[{index}].{field} references an unknown source"
                )
            if sources_by_id[source_id]["backend"] != backend:
                raise ValueError(
                    f"records[{index}].{field} references the wrong backend"
                )

    return data


def load_publication_data(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid publication JSON at {path}: {error}"
        ) from error
    return validate_publication_data(data)


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
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
        "unavailable": sum(
            record["speedup"] is None for record in records
        ),
        "timeouts": {
            side: sum(
                record["timeout_side"] == side for record in records
            )
            for side in ("cpu", "gpu", "both")
        },
        "phases": phases,
    }


def prepare_publication_data(data: Any) -> dict[str, Any]:
    """Select the widest measured PCA case for the documentation page."""
    transport = validate_publication_data(data)
    records = transport["records"]
    pca_records = [
        record
        for record in records
        if record["estimator"] == "pca"
        and record["operation"] == "fit_transform"
    ]
    if not pca_records:
        raise ValueError("publication data has no PCA fit_transform records")
    widest_pca = max(pca_records, key=lambda record: record["features"])
    presentation = [
        record
        for record in records
        if record["workload_label"] in WORKLOADS
    ]
    target_indexes = [
        index
        for index, record in enumerate(presentation)
        if _is_pca_large(record)
    ]
    if len(target_indexes) != 1:
        raise ValueError(
            "publication data must contain exactly one canonical PCA large case"
        )
    replacement = copy.deepcopy(widest_pca)
    replacement["workload_label"] = "large.balanced"
    presentation[target_indexes[0]] = replacement
    presentation.sort(
        key=lambda record: (
            record["estimator"],
            record["operation"],
            record["workload_label"],
        )
    )
    prepared = copy.deepcopy(transport)
    prepared["records"] = presentation
    prepared["summary"] = _summarize(presentation)
    return prepared


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
    workload_order = {label: index for index, label in enumerate(WORKLOADS)}
    headers = ["Operation", "Workload", "Rows", "Features", "Input"]
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


def _is_pca_large(record: dict[str, Any]) -> bool:
    return (
        record["estimator"] == "pca"
        and record["operation"] == "fit_transform"
        and record["workload_label"] == "large.balanced"
    )


def _timeout_limits(records: list[dict[str, Any]]) -> tuple[float, float]:
    pca_large = {
        record["timeout_limit_sec"]
        for record in records
        if _is_pca_large(record)
    }
    small_medium = {
        record["timeout_limit_sec"]
        for record in records
        if record["workload_label"] != "large.balanced"
    }
    default_limits = small_medium | pca_large
    if len(default_limits) != 1:
        raise ValueError(
            "default-timeout presentation cases use inconsistent limits"
        )
    other_large = {
        record["timeout_limit_sec"]
        for record in records
        if record["workload_label"] == "large.balanced"
        and not _is_pca_large(record)
    }
    if len(other_large) != 1:
        raise ValueError(
            "large-timeout presentation cases use inconsistent limits"
        )
    return next(iter(default_limits)), next(iter(other_large))


def render_rst(data: dict[str, Any], template: str) -> str:
    data = prepare_publication_data(data)
    records = data["records"]
    summary = data["summary"]
    training = summary["phases"]["training"]
    inference = summary["phases"]["inference"]
    default_timeout, large_timeout = _timeout_limits(records)
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
        "TRAINING_MEDIAN": f"{training['median_speedup']:.1f}",
        "UMAP_SPEEDUP": f"{_strongest(records, 'umap'):.0f}×",
        "HDBSCAN_SPEEDUP": f"{_strongest(records, 'hdbscan'):.0f}×",
        "TRAINING_PAIRED": str(training["paired"]),
        "TRAINING_CELLS": str(training["cells"]),
        "TRAINING_2X": str(training["at_least_2x"]),
        "INFERENCE_MEDIAN": f"{inference['median_speedup']:.2f}",
        "INFERENCE_SLOWDOWNS": str(inference["slowdowns"]),
        "INFERENCE_PAIRED": str(inference["paired"]),
        "DEFAULT_TIMEOUT_MIN": f"{default_timeout / 60:g}",
        "LARGE_TIMEOUT_MIN": f"{large_timeout / 60:g}",
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
        "PCA_LARGE_ROWS": f"{pca_large['rows']:,}",
        "PCA_LARGE_FEATURES": f"{pca_large['features']:,}",
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


def render_files(data: dict[str, Any], template: str) -> dict[Path, str]:
    prepared = prepare_publication_data(data)
    return {
        DEFAULT_PAGE: render_rst(data, template),
        DEFAULT_STATIC / "training-heatmap.svg": render_heatmap(
            prepared["records"], "training"
        ),
        DEFAULT_STATIC / "inference-heatmap.svg": render_heatmap(
            prepared["records"], "inference"
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
