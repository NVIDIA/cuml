# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "docs/benchmarks/generate_cuml_accel_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("cuml_accel_benchmarks", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


def _inputs() -> tuple[dict, str]:
    data = json.loads(generator.DEFAULT_DATA.read_text(encoding="utf-8"))
    template = generator.DEFAULT_TEMPLATE.read_text(encoding="utf-8")
    return data, template


def test_benchmark_files_can_be_rendered() -> None:
    data, template = _inputs()
    prepared = generator.prepare_publication_data(data)
    files = generator.render_files(data, template)

    assert prepared["records"]
    assert prepared["summary"]["cases"] == len(prepared["records"])

    page = files[generator.DEFAULT_PAGE]
    assert page.strip()
    assert "@@" not in page

    for phase in ("training", "inference"):
        svg = files[generator.DEFAULT_STATIC / f"{phase}-heatmap.svg"]
        root = ET.fromstring(svg)
        assert root.tag == "{http://www.w3.org/2000/svg}svg"
        assert root.attrib["role"] == "img"


def test_generated_files_are_current() -> None:
    data, template = _inputs()

    for path, content in generator.render_files(data, template).items():
        assert path.read_text(encoding="utf-8") == content


def test_rendered_environment_comes_from_publication_data() -> None:
    data, template = _inputs()
    data["system"]["components"][2]["name"] = "Test GPU"
    data["packages"]["cuml"] = "test-version"

    page = generator.render_rst(data, template)

    assert "Test GPU" in page
    assert "``cuml test-version``" in page


def test_narrative_gpu_matches_publication_hardware() -> None:
    data, _ = _inputs()
    gpu = next(
        component
        for component in data["system"]["components"]
        if component["type"] == "gpu"
    )

    assert gpu["name"] == ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition")


@pytest.mark.parametrize(
    "case_prefix",
    ["pca.fit_transform.rank", "pca.fit_transform.medium.wide"],
)
def test_pca_rank_table_records_with_different_shapes_are_rejected(
    case_prefix: str,
) -> None:
    data, _ = _inputs()
    record = next(
        record
        for record in data["records"]
        if record["case_label"].startswith(case_prefix)
    )
    record["rows"] += 1

    with pytest.raises(ValueError, match="must use the same shape"):
        generator.validate_publication_data(data)


def test_pca_heatmap_detail_uses_record_components() -> None:
    data, _ = _inputs()
    record = next(
        record
        for record in data["records"]
        if record["case_label"] == "pca.fit_transform.medium.wide"
    )
    record["components"] = 2048

    prepared = generator.prepare_publication_data(data)
    heatmap = generator.render_heatmap(prepared["records"], "training")

    assert "medium-wide · 2,048 components" in heatmap


@pytest.mark.parametrize("field", ["system", "packages"])
def test_publication_environment_is_required(field: str) -> None:
    data, _ = _inputs()
    del data[field]

    with pytest.raises(ValueError):
        generator.validate_publication_data(data)


def test_duplicate_system_component_is_rejected() -> None:
    data, _ = _inputs()
    data["system"]["components"][1] = copy.deepcopy(
        data["system"]["components"][0]
    )

    with pytest.raises(ValueError, match="duplicate cpu component"):
        generator.validate_publication_data(data)


def test_required_package_version_is_rejected_when_missing() -> None:
    data, _ = _inputs()
    del data["packages"]["hdbscan"]

    with pytest.raises(ValueError, match="packages must contain"):
        generator.validate_publication_data(data)


@pytest.mark.parametrize(
    "case_label",
    [
        "unsupported.fit.small.balanced",
        "dbscan.unsupported.small.balanced",
        "dbscan.fit_predict.small.wide",
        "pca.fit_transform.rank0128.medium.wide",
        "pca.fit_transform.rank+128.medium.wide",
        "pca.transform.rank128.medium.wide",
        "pca.fit_transform.rank128.small.balanced",
    ],
)
def test_unsupported_case_label_combinations_are_rejected(
    case_label: str,
) -> None:
    data, _ = _inputs()
    data["records"][0]["case_label"] = case_label

    with pytest.raises(ValueError):
        generator.validate_publication_data(data)


@pytest.mark.parametrize(
    "case_label",
    [
        "dbscan.predict.large.balanced",
        "pca.fit_transform.rank64.medium.wide",
    ],
)
def test_supported_case_label_extensions_are_accepted(case_label: str) -> None:
    data, _ = _inputs()
    if ".rank" in case_label:
        rank_record = next(
            record
            for record in data["records"]
            if record["case_label"].startswith("pca.fit_transform.rank")
        )
        data["records"][0]["rows"] = rank_record["rows"]
        data["records"][0]["features"] = rank_record["features"]
    data["records"][0]["case_label"] = case_label

    generator.validate_publication_data(data)
