# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for read-only artifact comparison."""

import copy
import json

import pytest

from cuml.benchmark import cli, comparison, harness
from cuml.benchmark.suite import Suite, resolve_case


@pytest.fixture
def artifact(tmp_path):
    case = resolve_case(
        {
            "estimator": "LinearRegression",
            "operation": "fit",
            "parameters": {},
            "input_selection": ["X", "y"],
            "dataset": {
                "generator": "regression",
                "shape": {"rows": 64, "features": 4},
            },
            "scoring": {"name": "r2", "evaluation_rows": 32},
        },
        {"warmups": 1, "repetitions": 2, "size_scale": 1},
    )
    return harness.run_suite(
        Suite("test", "test", "scikit-learn", "standard", (case,)),
        tmp_path / "source.json",
    )


def write_pair(tmp_path, artifact, change=lambda a: None):
    left, right = copy.deepcopy(artifact), copy.deepcopy(artifact)
    measurements = [
        o
        for o in right["results"][0]["observations"]
        if o["role"] == "measurement"
    ]
    for observation in measurements:
        observation["metrics"][0]["value"] = -0.5
    change(right)
    paths = [tmp_path / "left.json", tmp_path / "right.json"]
    for path, value in zip(paths, (left, right), strict=True):
        path.write_text(json.dumps(value))
    return paths


@pytest.mark.parametrize(
    "direction", ["higher_is_better", "lower_is_better", "neutral"]
)
def test_compare_ranking_and_read_only(tmp_path, artifact, direction):
    for observation in artifact["results"][0]["observations"]:
        for metric in observation["metrics"]:
            metric["direction"] = direction
    paths = write_pair(tmp_path, artifact)
    original = [path.read_bytes() for path in paths]
    report = comparison.compare(paths)
    assert report["comparable_metrics"] == 1
    summaries = list(report["rows"][0]["artifacts"].values())
    assert len(report["inputs"]) == 2
    assert len({item["label"] for item in report["inputs"]}) == 2
    assert all(s["valid"] == s["expected"] == 2 for s in summaries)
    assert all(s["status"] == "complete" for s in summaries)
    if direction == "neutral":
        assert all(
            s["gap_to_best"] is None and not s["best"] for s in summaries
        )
    else:
        winner = 0 if direction == "higher_is_better" else 1
        assert summaries[winner]["best"]
        assert summaries[winner]["gap_to_best"] == 0
        assert summaries[1 - winner]["gap_to_best"] == pytest.approx(1.5)
    reversed_report = comparison.compare(paths[::-1])
    assert reversed_report["comparable_metrics"] == 1
    assert [path.read_bytes() for path in paths] == original


@pytest.mark.parametrize(
    "change", ["version", "parameters", "unit", "direction", "methodology"]
)
def test_compare_incompatible_scoring(tmp_path, artifact, change):
    def mutate(right):
        result = right["results"][0]
        if change in {"version", "parameters"}:
            descriptor = result["extensions"][harness.EXTENSION]["scoring"]
            descriptor[change] = (
                "different" if change == "version" else {"force_finite": True}
            )
            right["run"]["extensions"][harness.EXTENSION]["execution_plan"][0][
                "scoring"
            ] = copy.deepcopy(descriptor)
        elif change == "methodology":
            right["run"]["methodology"]["id"] = "different"
        else:
            for observation in result["observations"]:
                for metric in observation["metrics"]:
                    metric[change] = (
                        "different" if change == "unit" else "lower_is_better"
                    )

    report = comparison.compare(write_pair(tmp_path, artifact, mutate))
    assert report["comparable_metrics"] == 0
    assert all(
        s["status"] == "incompatible_scoring" and s["gap_to_best"] is None
        for s in report["rows"][0]["artifacts"].values()
    )


@pytest.mark.parametrize(
    "change,status",
    [
        ("missing", "missing_case"),
        ("partial", "partial"),
        ("failed", "partial"),
        ("unscored", "unscored"),
        ("no_plan", "partial"),
    ],
)
def test_compare_incomplete_results(tmp_path, artifact, change, status):
    def mutate(right):
        result = right["results"][0]
        if change == "missing":
            right["results"] = []
        elif change == "partial":
            result["observations"].pop()
        elif change == "failed":
            result["outcome"] = {"status": "failed", "last_phase": "scoring"}
            result["observations"][-1]["outcome"] = result["outcome"]
        elif change == "no_plan":
            right["run"]["extensions"][harness.EXTENSION].pop("execution_plan")
        else:
            result["extensions"][harness.EXTENSION].pop("scoring")
            for observation in result["observations"]:
                observation["metrics"] = []

    report = comparison.compare(write_pair(tmp_path, artifact, mutate))
    assert report["comparable_metrics"] == 0
    summaries = list(report["rows"][0]["artifacts"].values())
    assert summaries[1]["status"] == status
    assert all(s["gap_to_best"] is None for s in summaries)


def test_compare_excludes_warmups_and_marks_ties(tmp_path, artifact):
    artifact["results"][0]["observations"][0]["metrics"] = [
        {
            "name": "r2_score",
            "value": -100,
            "unit": "dimensionless",
            "direction": "higher_is_better",
        }
    ]

    def tie(right):
        right["results"] = copy.deepcopy(artifact["results"])

    report = comparison.compare(write_pair(tmp_path, artifact, tie))
    assert all(
        s["best"] and s["mean"] == pytest.approx(1)
        for s in report["rows"][0]["artifacts"].values()
    )


@pytest.mark.parametrize(
    "change",
    [
        "duplicate_case",
        "identity",
        "nonfinite",
        "duplicate_metric",
        "duplicate_observation",
    ],
)
def test_compare_rejects_invalid_artifacts(tmp_path, artifact, change):
    def mutate(right):
        result = right["results"][0]
        if change == "duplicate_case":
            right["results"].append(copy.deepcopy(result))
        elif change == "identity":
            result["id"] = "sha256:wrong"
        elif change == "nonfinite":
            result["observations"][-1]["metrics"][0]["value"] = float("nan")
        elif change == "duplicate_metric":
            result["observations"][-1]["metrics"] *= 2
        else:
            result["observations"].append(
                copy.deepcopy(result["observations"][-1])
            )

    with pytest.raises(ValueError):
        comparison.compare(write_pair(tmp_path, artifact, mutate))


def test_compare_workload_order(tmp_path, artifact):
    from cuml.benchmark.suite import Workload

    combinations = [
        ("Ridge", "fit", 64, 4),
        ("LinearRegression", "predict", 32, 4),
        ("LinearRegression", "fit", 128, 4),
        ("LinearRegression", "fit", 64, 8),
        ("LinearRegression", "fit", 64, 4),
    ]
    original = artifact["results"][0]
    artifact["results"] = []
    for algorithm, operation, rows, features in combinations:
        result = copy.deepcopy(original)
        result["algorithm"] = algorithm
        result["operation"]["name"] = operation
        result["input"]["dimensions"] = [
            {"name": "rows", "size": rows},
            {"name": "features", "size": features},
        ]
        result["id"] = Workload.from_artifact_fields(result).digest()
        artifact["results"].append(result)
    paths = write_pair(tmp_path, artifact)
    report = comparison.compare(paths)
    actual = [
        (
            row["algorithm"],
            row["operation"],
            *(dimension["size"] for dimension in row["dimensions"]),
        )
        for row in report["rows"]
    ]
    assert actual == sorted(combinations)


@pytest.mark.parametrize(
    "expression",
    [
        "linearregression",
        "LinearRegression.fit",
        "regression and not PCA",
        "(PCA or LinearRegression) and rows=64 and features=4 and dtype=float32",
    ],
)
def test_compare_keyword_selection(tmp_path, artifact, expression):
    paths = write_pair(tmp_path, artifact)
    report = comparison.compare(paths, keyword_expression=expression)
    assert report["keyword_expression"] == expression
    assert len(report["rows"]) == 1
    assert report["comparable_metrics"] == 1
    identity = artifact["results"][0]["id"]
    assert (
        comparison.compare(paths, keyword_expression=identity)["rows"][0][
            "workload_id"
        ]
        == identity
    )


@pytest.mark.parametrize("expression", ["PCA", "and", "(regression"])
def test_compare_keyword_errors(tmp_path, artifact, expression):
    with pytest.raises(ValueError, match="-k"):
        comparison.compare(
            write_pair(tmp_path, artifact), keyword_expression=expression
        )


def test_compare_omits_workloads_not_in_provider_plan(tmp_path, artifact):
    def mutate(right):
        right["run"]["extensions"][harness.EXTENSION]["provider"] = (
            "umap-learn"
        )
        right["run"]["extensions"][harness.EXTENSION]["execution_plan"] = []
        right["results"] = []

    report = comparison.compare(write_pair(tmp_path, artifact, mutate))
    assert set(report["rows"][0]["artifacts"]) == {"scikit-learn"}
    assert report["comparable_metrics"] == 0


@pytest.mark.parametrize("direction", ["higher_is_better", "lower_is_better"])
def test_compare_table_sorts_scores(tmp_path, artifact, direction):
    from io import StringIO

    from rich.console import Console

    for observation in artifact["results"][0]["observations"]:
        for metric in observation["metrics"]:
            metric["direction"] = direction
    paths = write_pair(tmp_path, artifact)
    if direction == "higher_is_better":
        paths.reverse()  # Put the worse result first in both directions.
    report = comparison.compare(paths)
    output = StringIO()
    Console(file=output, width=160).print(comparison.render_table(report))
    table = output.getvalue().split("LinearRegression.fit", 1)[1]
    assert table.index("scikit-learn [2]") < table.index("scikit-learn [1]")


@pytest.mark.parametrize("direction", ["higher_is_better", "lower_is_better"])
@pytest.mark.parametrize(
    "best,gap,color",
    [(1.0, 5e-7, "yellow"), (0.0, 5e-9, "yellow"), (1.0, 1e-4, "red")],
)
def test_compare_gap_numerical_tolerance(
    tmp_path, artifact, direction, best, gap, color
):
    from rich.table import Table

    for observation in artifact["results"][0]["observations"]:
        for metric in observation["metrics"]:
            metric["value"] = best
            metric["direction"] = direction

    def mutate(right):
        for observation in right["results"][0]["observations"]:
            for metric in observation["metrics"]:
                metric["value"] = (
                    best - gap
                    if direction == "higher_is_better"
                    else best + gap
                )

    report = comparison.compare(write_pair(tmp_path, artifact, mutate))
    summaries = list(report["rows"][0]["artifacts"].values())
    assert summaries[0]["best"]
    assert not summaries[1]["best"]
    assert summaries[1]["gap_to_best"] == pytest.approx(gap)
    table = next(
        item
        for item in comparison.render_table(report).renderables
        if isinstance(item, Table)
    )
    assert table.columns[5]._cells[0].style == "green"
    assert table.columns[5]._cells[1].style == color


@pytest.mark.parametrize("width", [80, 160])
def test_compare_tables_share_column_widths(tmp_path, artifact, width):
    from io import StringIO

    from rich.console import Console
    from rich.table import Table

    report = comparison.compare(write_pair(tmp_path, artifact))
    second = copy.deepcopy(report["rows"][0])
    second["artifacts"] = {
        "long-provider-label": next(iter(second["artifacts"].values()))
    }
    second["artifacts"]["long-provider-label"]["mean"] = -123456.789
    report["rows"].append(second)
    tables = [
        item
        for item in comparison.render_table(report).renderables
        if isinstance(item, Table)
    ]
    assert len(tables) == 2
    borders = []
    for table in tables:
        output = StringIO()
        Console(file=output, width=width).print(table)
        borders.append(
            next(
                line
                for line in output.getvalue().splitlines()
                if line.startswith("╭")
            )
        )
    assert borders[0] == borders[1]


def test_compare_grouped_colored_rendering(tmp_path, artifact):
    from io import StringIO

    from rich.console import Console

    report = comparison.compare(write_pair(tmp_path, artifact))
    output = StringIO()
    Console(
        file=output, force_terminal=True, color_system="standard", width=160
    ).print(comparison.render_table(report))
    text = output.getvalue()
    assert "Benchmark comparison" in text
    assert "LinearRegression.fit" in text
    from rich.table import Table

    table = next(
        item
        for item in comparison.render_table(report).renderables
        if isinstance(item, Table)
    )
    assert table.rows[0].style == "bold"
    assert "higher_is_better" in text
    assert "\x1b[32m" in text  # Best gap / complete status.
    assert "\x1b[31m" in text  # Nonzero gap.


def test_compare_cli_no_execution(tmp_path, artifact, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("comparison must not load suites or execute benchmarks")

    monkeypatch.setattr(cli, "_suite_aware_parser", forbidden)
    paths = write_pair(tmp_path, artifact)
    assert cli.main(["compare", *map(str, paths), "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["comparable_metrics"] == 1
    assert (
        cli.main(
            [
                "compare",
                *map(str, paths),
                "-k",
                "regression and rows=64",
                "--format",
                "json",
            ]
        )
        == 0
    )
    filtered = json.loads(capsys.readouterr().out)
    assert filtered["keyword_expression"] == "regression and rows=64"
    assert filtered["comparable_metrics"] == 1
    assert cli.main(["compare", *map(str, paths), "-k", "PCA"]) == 1
    assert "no benchmark cases match" in capsys.readouterr().err
    assert cli.main(["compare", *map(str, paths)]) == 0
    assert "Gap" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        cli.main(["compare", str(paths[0])])
    assert exc.value.code == 2
    paths[1].write_text("invalid")
    assert cli.main(["compare", *map(str, paths)]) == 1
