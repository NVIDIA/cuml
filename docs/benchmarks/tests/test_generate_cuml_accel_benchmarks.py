# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import TestCase


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "docs/benchmarks/generate_cuml_accel_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("cuml_accel_benchmarks", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


class CumlAccelBenchmarkDocsTests(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = json.loads(
            generator.DEFAULT_DATA.read_text(encoding="utf-8")
        )
        cls.template = generator.DEFAULT_TEMPLATE.read_text(encoding="utf-8")

    def test_publication_data_can_be_prepared(self) -> None:
        prepared = generator.prepare_publication_data(self.data)

        self.assertTrue(prepared["records"])
        self.assertEqual(
            prepared["summary"]["cases"], len(prepared["records"])
        )

    def test_page_can_be_rendered(self) -> None:
        page = generator.render_rst(self.data, self.template)

        self.assertTrue(page.strip())
        self.assertNotIn("@@", page)

    def test_heatmaps_are_valid_svg(self) -> None:
        records = generator.prepare_publication_data(self.data)["records"]

        for phase in ("training", "inference"):
            root = ET.fromstring(generator.render_heatmap(records, phase))
            self.assertEqual(root.tag, "{http://www.w3.org/2000/svg}svg")
            self.assertEqual(root.attrib["role"], "img")

if __name__ == "__main__":
    import unittest

    unittest.main()
