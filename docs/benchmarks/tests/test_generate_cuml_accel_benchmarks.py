# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import importlib.util
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "docs/benchmarks/generate_cuml_accel_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("cuml_accel_benchmarks", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


@pytest.fixture(scope="module")
def publication_data() -> dict[str, Any]:
    return json.loads(generator.DEFAULT_DATA.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def template() -> str:
    return generator.DEFAULT_TEMPLATE.read_text(encoding="utf-8")


def test_publication_data_can_be_prepared(
    publication_data: dict[str, Any],
) -> None:
    prepared = generator.prepare_publication_data(publication_data)

    assert publication_data["schema_version"] == 1
    assert prepared["schema_version"] == 1
    assert len(prepared["records"]) == 168
    assert prepared["summary"]["cases"] == len(prepared["records"])
    assert any(record["estimator"] == "tsne" for record in prepared["records"])
    assert any(record["estimator"] == "kernel_density" for record in prepared["records"])
    assert any(record["speedup_is_lower_bound"] for record in prepared["records"])


def test_cpu_timeout_lower_bound_is_rendered(
    publication_data: dict[str, Any], template: str
) -> None:
    records = generator.prepare_publication_data(publication_data)["records"]
    record = next(record for record in records if record["speedup_is_lower_bound"])

    assert generator._status_text(record).startswith("≥")
    page = generator.render_rst(publication_data, template)
    assert "Timeout at" in page
    assert "(CPU timeout)" not in page


def test_pca_rank_variants_are_in_primary_publication(
    publication_data: dict[str, Any],
) -> None:
    records = generator.prepare_publication_data(publication_data)["records"]
    rank_records = [record for record in records if record["is_rank_variant"]]

    assert {record["parameters"]["components"] for record in rank_records} == {
        128, 256, 512
    }
    assert all(record["estimator"] == "pca" for record in rank_records)


def test_presentation_and_heatmap_record_sets_are_centralized(
    publication_data: dict[str, Any]
) -> None:
    records = generator.prepare_publication_data(publication_data)["records"]
    presented = generator._presentation_records(records)
    training = generator._heatmap_records(presented, "training")
    inference = generator._heatmap_records(presented, "inference")

    assert all(record["phase"] == "training" for record in training)
    pca_training = [
        record for record in training if record["estimator"] == "pca"
    ]
    assert len(pca_training) == len(generator.WORKLOADS)
    assert {record["workload_label"] for record in pca_training} == set(
        generator.WORKLOADS
    )
    pca_medium_wide = next(
        record
        for record in pca_training
        if record["workload_label"] == "medium.wide"
    )
    assert pca_medium_wide["case_label"] == "pca.fit_transform.medium.wide"
    assert pca_medium_wide["parameters"]["components"] == 1024
    assert all(record["phase"] == "inference" for record in inference)
    assert any(record["estimator"] == "pca" for record in inference)


def test_page_retains_pca_component_rank_results(
    publication_data: dict[str, Any], template: str
) -> None:
    page = generator.render_rst(publication_data, template)

    assert "8.14 s / 3.1 s / 2.62×" in page
    assert "PCA performance depends strongly" in page
    assert ":ref:`detailed PCA results <benchmark-pca>`" in page
    assert page.index("PCA performance depends strongly") < page.index(
        "Inference and transforms"
    )
    assert "1.12×" in page
    assert "2.62×" in page
    assert "Decomposition quality requires" not in page
    assert "All component-rank results satisfy both" not in page
    pca_details = page[page.index(".. dropdown:: PCA") :]
    next_dropdown = pca_details.index(".. dropdown::", len(".. dropdown:: PCA"))
    pca_details = pca_details[:next_dropdown]
    assert "Additional medium-wide PCA fit-transform results by component rank" not in pca_details
    assert pca_details.count(".. list-table::") == 1
    assert "``medium.wide · 16 components``" not in pca_details
    assert "``small.balanced · 256 components``" in pca_details
    assert "``medium.thin · 32 components``" in pca_details
    assert "``medium.balanced · 256 components``" in pca_details
    assert "``medium.wide · 128 components``" in pca_details
    assert "``medium.wide · 256 components``" in pca_details
    assert "``medium.wide · 512 components``" in pca_details
    assert "``medium.wide · 1,024 components``" in pca_details
    assert "0.48×" in pca_details
    assert "0.74×" in pca_details
    assert "1.12×" in pca_details
    assert "2.62×" in pca_details
    medium_wide_positions = [
        pca_details.index(f"``medium.wide · {rank:,} components``")
        for rank in (128, 256, 512, 1024)
    ]
    assert medium_wide_positions == sorted(medium_wide_positions)
    assert pca_details.index("``large · 256 components``") > medium_wide_positions[-1]
    comment = (
        "PCA performance depends strongly on both input feature width "
        "and the number of retained components; results can vary "
        "substantially across these dimensions."
    )
    assert comment in pca_details
    assert pca_details.index(comment) > pca_details.rindex("2.62×")


def test_byte_formatting_avoids_scientific_notation() -> None:
    assert generator._fmt_bytes(999_997_440) == "1,000 MB"
    assert generator._fmt_bytes(1_000_000_000) == "1 GB"
    assert generator._fmt_bytes(4_096_000_000) == "4.1 GB"


def test_page_can_be_rendered(
    publication_data: dict[str, Any], template: str
) -> None:
    page = generator.render_rst(publication_data, template)

    assert page.strip()
    assert "@@" not in page
    assert "Generated from benchmarks.rst.in; do not edit" in page
    assert "This file is the editable template" not in page


def test_training_heatmap_uses_medium_wide_rank_1024_pca_result(
    publication_data: dict[str, Any], template: str
) -> None:
    records = generator.prepare_publication_data(publication_data)["records"]
    svg = generator.render_heatmap(records, "training")
    page = generator.render_rst(publication_data, template)
    root = ET.fromstring(svg)
    namespaces = {"svg": "http://www.w3.org/2000/svg"}
    pca_links = [
        link
        for link in root.findall("svg:a", namespaces)
        if (link.attrib.get("aria-label") or "").startswith("PCA.fit_transform")
    ]
    pca_cells = [
        group
        for group in root.findall("svg:g", namespaces)
        if (group.attrib.get("aria-label") or "").startswith("PCA.fit_transform")
    ]

    assert len(pca_links) == 1
    assert len(pca_cells) == len(generator.WORKLOADS)
    assert pca_links[0].find("svg:text", namespaces).text == "PCA"
    medium_wide_cell = next(
        cell
        for cell in pca_cells
        if "medium.wide" in cell.attrib["aria-label"]
    )
    assert "medium-wide · 1,024 components" in medium_wide_cell.attrib["aria-label"]
    assert "2.62×" in "".join(medium_wide_cell.itertext())
    assert "PCA performance depends strongly" in page
    assert "1.12×" in page
    assert "2.62×" in page


def test_heatmaps_do_not_single_out_large_as_operation_specific(
    publication_data: dict[str, Any],
) -> None:
    records = generator.prepare_publication_data(publication_data)["records"]
    for phase in ("training", "inference"):
        svg = generator.render_heatmap(records, phase)
        assert "large*" not in svg
        assert "* large is operation-specific" not in svg


def test_heatmaps_are_valid_svg(
    publication_data: dict[str, Any],
) -> None:
    records = generator.prepare_publication_data(publication_data)["records"]

    for phase in ("training", "inference"):
        root = ET.fromstring(generator.render_heatmap(records, phase))
        assert root.tag == "{http://www.w3.org/2000/svg}svg"
        assert root.attrib["role"] == "img"


def _heatmap_operation(
    operation: str, *, cpu_time: float, gpu_time: float
) -> list[dict[str, Any]]:
    return [
        {
            "phase": "inference",
            "estimator": "pca",
            "operation": operation,
            "workload_label": workload,
            "speedup": 2.0,
            "timeout_side": None,
            "unavailable_side": None,
            "unavailable_reason": None,
            "cpu_median_wall_time_sec": cpu_time,
            "gpu_median_wall_time_sec": gpu_time,
        }
        for workload in generator.WORKLOADS
    ]


def test_inference_heatmap_ranking_ignores_lower_bounds() -> None:
    exact = _heatmap_operation("exact", cpu_time=0.001, gpu_time=0.001)
    lower = _heatmap_operation("lower", cpu_time=0.001, gpu_time=0.001)
    for record in exact:
        record["speedup"] = 2.0
    for record in lower:
        record["speedup"] = 1000.0
        record["speedup_is_lower_bound"] = True

    svg = generator.render_heatmap(exact + lower, "inference")

    assert "PCA.exact" in svg
    assert "PCA.lower" not in svg


def test_inference_heatmap_keeps_top_10_operations() -> None:
    records = []
    for index in range(12):
        operation_records = _heatmap_operation(
            f"operation_{index}", cpu_time=0.001, gpu_time=0.001
        )
        for record in operation_records:
            record["speedup"] = float(index + 1)
        records += operation_records

    svg = generator.render_heatmap(records, "inference")

    assert "Heatmap of 10 operations across five workloads" in svg
    assert "PCA.operation_11," in svg
    assert "PCA.operation_2," in svg
    assert "PCA.operation_1," not in svg
    assert "PCA.operation_0," not in svg


def test_timeout_lower_bound_is_hatched_and_exact_in_heatmap() -> None:
    records = _heatmap_operation("transform", cpu_time=0.05, gpu_time=0.01)
    records[0]["speedup"] = 528.967
    records[0]["speedup_is_lower_bound"] = True

    svg = generator.render_heatmap(records, "inference")

    assert 'fill="url(#lower-bound-' in svg
    assert 'width="10" height="10" patternUnits="userSpaceOnUse"' in svg
    assert 'stroke-width="1" stroke-opacity="0.55"' in svg
    assert "≥529×" in svg
    assert "CPU timeout" in svg


def test_heatmap_keeps_operation_with_only_unavailable_results() -> None:
    records = _heatmap_operation("transform", cpu_time=0.05, gpu_time=0.01)
    for record in records:
        record.update(
            speedup=None,
            unavailable_side="cpu",
            unavailable_reason="Reproduced worker failure.",
            cpu_median_wall_time_sec=None,
        )

    svg = generator.render_heatmap(records, "inference")

    assert "PCA.transform" not in svg


def test_prose_and_result_speedup_formatting_are_separate() -> None:
    assert generator._fmt_prose_speedup(528.967, lower_bound=True) == (
        "an approximate lower bound of ≥500×"
    )
    assert generator._fmt_speedup(528.967) == "529×"


def test_opening_prose_focuses_on_training_but_results_remain_exact(
    publication_data: dict[str, Any], template: str
) -> None:
    page = generator.render_rst(publication_data, template)
    assert "approximately 5.8× median speedup" in page
    assert "Training\nperformance varies with the estimator" in page
    assert "most measured workloads are faster on GPU" in page
    assert "largest gains appearing on wider or larger datasets" in page
    assert "The median exact speedup is" not in page
    assert "Timeout at 70 s" in page
    assert "Timeout at 100 s" in page
    assert "Timeout at 14 min" in page
    assert "≥529×" in page
    assert "≥529× (CPU timeout)" not in page
    assert "69.5 s" not in page
    assert "102 s" not in page
    assert "14.2 min" not in page


def test_publication_validator_rejects_redundant_fields(
    publication_data: dict[str, Any]
) -> None:
    data = copy.deepcopy(publication_data)
    data["records"][0]["exact_speedup"] = 2.0

    with pytest.raises(ValueError, match="unsupported or missing fields"):
        generator.validate_publication_data(data)


def test_publication_validator_requires_timeout_for_missing_cpu_time(
    publication_data: dict[str, Any]
) -> None:
    data = copy.deepcopy(publication_data)
    record = next(item for item in data["records"] if item["cpu_median_sec"] is None)
    record.pop("cpu_timeout_sec")

    with pytest.raises(ValueError, match="cpu_timeout_sec"):
        generator.validate_publication_data(data)


def test_generated_files_are_current(
    publication_data: dict[str, Any], template: str
) -> None:
    for path, content in generator.render_files(publication_data, template).items():
        assert path.read_text(encoding="utf-8") == content


def test_heatmap_css_aspect_ratios_match_generated_svgs(
    publication_data: dict[str, Any], template: str
) -> None:
    files = generator.render_files(publication_data, template)
    css = (generator.ROOT / "docs/source/_static/cuml-accel-benchmarks.css").read_text()
    for name in ("training", "inference"):
        svg = files[generator.DEFAULT_STATIC / f"{name}-heatmap.svg"]
        width = re.search(r'width="(\d+)"', svg).group(1)
        height = re.search(r'height="(\d+)"', svg).group(1)
        assert f"aspect-ratio: {width} / {height};" in css
