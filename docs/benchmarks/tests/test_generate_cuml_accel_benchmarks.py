# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
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
        cls.page = generator.render_rst(cls.data, cls.template)
        cls.training_svg = generator.render_heatmap(
            cls.data["records"], "training"
        )
        cls.inference_svg = generator.render_heatmap(
            cls.data["records"], "inference"
        )

    def test_expected_coverage_and_statistics(self) -> None:
        summary = self.data["summary"]
        self.assertEqual(summary["estimators"], 17)
        self.assertEqual(summary["operations"], 29)
        self.assertEqual(summary["cases"], 145)
        self.assertEqual(summary["phases"]["training"]["cells"], 70)
        self.assertEqual(summary["phases"]["training"]["paired"], 54)
        self.assertEqual(summary["phases"]["training"]["at_least_2x"], 41)
        self.assertAlmostEqual(
            summary["phases"]["training"]["median_speedup"],
            4.290619761662686,
        )
        self.assertEqual(summary["phases"]["inference"]["cells"], 75)
        self.assertEqual(summary["phases"]["inference"]["paired"], 65)
        self.assertAlmostEqual(
            summary["phases"]["inference"]["median_speedup"],
            0.8073318324009143,
        )
        self.assertEqual(summary["timeouts"], {"both": 1, "cpu": 23, "gpu": 2})
        self.assertEqual(summary["unavailable"], 26)

    def test_curated_data_contains_validation_but_no_raw_provenance(
        self,
    ) -> None:
        forbidden_keys = {
            "source",
            "reference",
            "candidate",
            "reference_sha256",
            "candidate_sha256",
            "reference_run_id",
            "candidate_run_id",
            "observations",
        }

        def keys(value: object) -> set[str]:
            if isinstance(value, dict):
                return set(value) | set().union(
                    *(keys(item) for item in value.values())
                )
            if isinstance(value, list):
                return (
                    set().union(*(keys(item) for item in value))
                    if value
                    else set()
                )
            return set()

        self.assertFalse(keys(self.data) & forbidden_keys)
        self.assertEqual(
            self.data["validation"]["successful_accelerated_execution"],
            "gpu_only",
        )
        self.assertEqual(self.data["packages"]["cuml"], "26.10.0a69")
        self.assertNotIn("cuml.accel", self.data["packages"])
        self.assertTrue(
            all(
                record["execution_profile"] == "gpu_only"
                for record in self.data["records"]
                if record["gpu_median_wall_time_sec"] is not None
            )
        )

    def test_pca_large_is_the_validated_2048_feature_replacement(self) -> None:
        record = next(
            item
            for item in self.data["records"]
            if item["estimator"] == "pca"
            and item["operation"] == "fit_transform"
            and item["workload_label"] == "large.balanced"
        )
        self.assertEqual(
            record["case_label"], "pca.fit_transform.feature_wide_2048"
        )
        self.assertEqual(
            (record["rows"], record["features"], record["components"]),
            (5_000, 2_048, 1_024),
        )
        self.assertAlmostEqual(record["speedup"], 12.229351697712223)
        supplemental = self.data["supplemental_summary"]
        self.assertEqual(supplemental["measured_cases"], 2)
        self.assertEqual(
            supplemental["selected_case_label"], record["case_label"]
        )
        self.assertEqual(
            supplemental["excluded"],
            [
                {
                    "case_label": "pca.fit_transform.feature_wide_1024",
                    "reasons": [
                        "decomposition-quality delta 0.0226091 exceeds 0.01"
                    ],
                }
            ],
        )

    def test_page_contains_only_presented_results(self) -> None:
        self.assertIn("**4.3× median", self.page)
        self.assertIn("**92× for UMAP**", self.page)
        self.assertIn("**460× for HDBSCAN**", self.page)
        self.assertIn("5,000", self.page)
        self.assertIn("2,048", self.page)
        self.assertIn("1,024", self.page)
        self.assertIn("12.2×", self.page)
        self.assertEqual(self.page.count(".. dropdown:: "), 18)
        self.assertEqual(self.page.count("   :name: "), 17)
        self.assertNotIn("feature_wide_1024", self.page)
        self.assertNotIn("cuml.accel 26.10.0a69", self.page)
        self.assertNotIn('class="numeric slowdown"', self.page)
        for forbidden in (
            "artifact",
            "benchmark-data",
            "source hash",
            "run ID",
            ".json",
        ):
            self.assertNotIn(forbidden.lower(), self.page.lower())

    def test_heatmaps_preserve_current_selection_and_links(self) -> None:
        namespace = {"svg": "http://www.w3.org/2000/svg"}
        for svg, expected_cells, expected_operations in (
            (self.training_svg, 70, 14),
            (self.inference_svg, 50, 10),
        ):
            root = ET.fromstring(svg)
            self.assertEqual(root.attrib["role"], "img")
            self.assertEqual(
                len(re.findall(r'<g role="img" aria-label=', svg)),
                expected_cells,
            )
            links = root.findall(".//svg:a", namespace)
            self.assertEqual(len(links), expected_operations)
            self.assertTrue(
                all(
                    link.attrib["href"].startswith(
                        "../../cuml-accel/benchmarks/#"
                    )
                    for link in links
                )
            )
            self.assertTrue(
                all(link.attrib["target"] == "_top" for link in links)
            )
            for link in links:
                tooltip = link.find("svg:title", namespace)
                label = link.find("svg:text", namespace)
                assert tooltip is not None and label is not None
                self.assertRegex(tooltip.text or "", r"^[A-Za-z]+\.[a-z_]+$")
                self.assertNotIn(".", label.text or "")
        self.assertIn("PCA.fit_transform, large: 12.2×", self.training_svg)
        self.assertIn(
            "url(#cpu-timeout)", self.training_svg + self.inference_svg
        )
        self.assertNotIn("LogisticRegression.predict", self.inference_svg)
        self.assertNotIn("Ridge.predict", self.inference_svg)

    def test_generated_files_are_deterministic_and_current(self) -> None:
        files = generator.render_files(self.data, self.template)
        self.assertEqual(files[generator.DEFAULT_PAGE], self.page)
        self.assertEqual(
            files[generator.DEFAULT_STATIC / "training-heatmap.svg"],
            self.training_svg,
        )
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "render", "--check"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_repetition_gate_rejects_an_incomplete_success(self) -> None:
        result = {
            "outcome": {"status": "success"},
            "observations": [
                {"role": "warmup"},
                {"role": "measurement"},
                {"role": "measurement"},
            ],
        }
        with self.assertRaisesRegex(ValueError, "warmup or repetition count"):
            generator._validate_repetitions(
                result, "example.fit.small.balanced"
            )


if __name__ == "__main__":
    import unittest

    unittest.main()
