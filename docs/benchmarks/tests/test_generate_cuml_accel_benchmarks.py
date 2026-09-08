# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path


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
