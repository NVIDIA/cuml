# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scoring, grouping, validation, and presentation regression tests."""

import copy
import importlib.util
from io import StringIO
import re
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "score_sizing", Path(__file__).parents[1] / "score_sizing.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def artifact():
    cases = []
    for tag, rows, features, times in [
        ("thin", 800, 16, [0.01, 0.01, 0.01]),
        ("balanced", 100, 128, [0.102, 0.12, 0.138]),
        ("wide", 25, 512, [2, 2, 2]),
    ]:
        cases.append(
            dict(
                id=tag,
                algorithm="PCA",
                operation={"name": "fit"},
                parameters={"declared": {}},
                dataset={"name": "matrix"},
                input={
                    "dimensions": [
                        {"name": "rows", "size": rows},
                        {"name": "features", "size": features},
                    ]
                },
                extensions={module.EXTENSION: {"tags": [tag]}},
                outcome={"status": "success"},
                observations=[observation(t) for t in times],
            )
        )
    return {"schema_version": 2, "results": cases}


def observation(t, role="measurement"):
    return dict(
        role=role,
        outcome={"status": "success"},
        timings=[dict(name="wall_time", unit="s", value=t)],
    )


def render(data, capsys, **kwargs):
    cases, units = module.assess(data)
    module.report(data, "artifact.json", cases, units, **kwargs)
    return capsys.readouterr().out


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0.001, 0.01),
        (0.05, 0.5),
        (0.1, 1),
        (1, 1),
        (5, 1 - module.math.log10(5)),
        (10, 0),
        (11, 0),
    ],
)
def test_timing_curve(seconds, expected):
    assert module.timing_score(seconds) == pytest.approx(expected)


@pytest.mark.parametrize(
    "spread,expected", [(0, 1), (0.1, 1), (0.3, 0.5), (0.5, 0.25)]
)
def test_variability_curve(spread, expected):
    assert module.variability_multiplier(spread) == pytest.approx(expected)


def test_group_anchor_and_worst_multiplier(capsys):
    data = artifact()
    cases, units = module.assess(data)
    assert len(units) == 1 and units[0].grouped
    assert units[0].timing == 1
    assert units[0].combined == pytest.approx(0.5)
    assert "FAST" not in cases[0].flags and "SLOW" not in cases[2].flags
    assert "combined coverage=1/1 groups/independent cases" in render(
        data, capsys
    )


def test_fixed_tolerance_and_group_counts(capsys):
    data = artifact()
    data["results"][0]["input"]["dimensions"][0]["size"] = 808
    assert not module.assess(data)[1][0].flags
    data["results"][0]["input"]["dimensions"][0]["size"] = 809
    assert module.assess(data)[1][0].flags == ["SIZE_MISMATCH"]
    assert "1 size-mismatched groups" in render(data, capsys)


def test_missing_dimensions_are_explicit():
    data = artifact()
    data["results"][0]["input"]["dimensions"] = []
    assert module.assess(data)[1][0].flags == ["MISSING_DIMENSIONS"]


def test_legacy_and_unknown_tags(capsys):
    data = artifact()
    for case in data["results"]:
        del case["extensions"]
    cases, units = module.assess(data)
    assert len(units) == 3 and not any(u.grouped for u in units)
    assert "FAST" in cases[0].flags and "SLOW" in cases[2].flags
    assert "Tag metadata is missing" in render(data, capsys)
    for case in data["results"]:
        case["extensions"] = {module.EXTENSION: {"tags": ["custom"]}}
    assert len(module.assess(data)[1]) == 3


def test_incomplete_ambiguous_and_duplicate_roles():
    data = artifact()
    data["results"].pop()
    assert all(
        "INCOMPLETE_SHAPE_GROUP" in c.flags for c in module.assess(data)[0]
    )
    data["results"][0]["extensions"][module.EXTENSION]["tags"] = [
        "thin",
        "wide",
    ]
    assert "AMBIGUOUS_SHAPE" in module.assess(data)[0][0].flags
    data = artifact()
    data["results"].append(copy.deepcopy(data["results"][0]))
    assert len(module.assess(data)[1]) == 4


@pytest.mark.parametrize("field", ["parameters", "provider", "reference"])
def test_group_isolation(field):
    data = artifact()
    other = copy.deepcopy(data["results"])
    for case in other:
        if field == "parameters":
            case["parameters"]["declared"] = {"n_components": 25}
        elif field == "provider":
            case["extensions"][module.EXTENSION]["provider"] = "other"
        else:
            case["dataset"]["parameters"] = {"train_rows": 2000}
    data["results"] += other
    assert len(module.assess(data)[1]) == 2


def test_inference_rows_not_reference_rows():
    data = artifact()
    for i, case in enumerate(data["results"]):
        case["dataset"]["parameters"] = {
            "train_rows": 1000,
            "inference_rows": i + 1,
        }
    assert len(module.assess(data)[1]) == 1
    data["results"][0]["dataset"]["parameters"]["train_rows"] = 2000
    assert len(module.assess(data)[1]) == 3


def test_failed_and_insufficient_repeats():
    data = artifact()
    data["results"][0]["outcome"]["status"] = "failed"
    assert module.assess(data)[1][0].combined == 0
    data["results"][0]["observations"] = [observation(0.1)]
    assert module.assess(data)[1][0].combined is None


@pytest.mark.parametrize(
    "invalid", [0, -1, float("nan"), float("inf"), "bad", True]
)
def test_invalid_observations(invalid):
    source = artifact()["results"][0]
    source["observations"].append(observation(invalid))
    case = module.assess_case(source, None)
    assert "INVALID" in case.flags and case.timing == 0
    assert case.maximum == 0.01


def test_duplicate_times_excluded_everywhere():
    source = artifact()["results"][0]
    obs = observation(100)
    obs["timings"].append(dict(obs["timings"][0]))
    source["observations"].append(obs)
    case = module.assess_case(source, None)
    assert "INVALID" in case.flags and "OVER_BUDGET" not in case.flags
    assert case.maximum == 0.01 and len(case.measured) == 3


def test_warmups_and_smoke():
    source = artifact()["results"][0]
    source["observations"].append(observation(11, "warmup"))
    case = module.assess_case(source, "smoke")
    assert case.mid == 0.01 and "FAST" not in case.flags
    assert case.maximum == 0.01 and "OVER_BUDGET" not in case.flags
    included = module.assess_case(source, "smoke", include_warmups=True)
    assert included.maximum == 11 and "OVER_BUDGET" in included.flags
    assert len(included.measured) == 4


def test_statistics_empty_singleton_and_population():
    assert module.statistics([]) == {}
    assert module.statistics([2])["stddev"] == 0
    stats = module.statistics([1, 2, 3, 4])
    assert stats["median"] == stats["mean"] == 2.5
    assert stats["stddev"] == pytest.approx(1.25**0.5)
    assert stats["q1"] == 1.75
    assert "variance" not in stats


def test_report_options_color_and_empty(capsys):
    data = artifact()
    plain = render(data, capsys, show_all=True, verbose=True)
    colored = render(data, capsys, show_all=True, verbose=True, color=True)
    assert "Case ID" in plain and "tags=balanced" in plain
    assert re.sub(r"\x1b\[[0-9;]*m", "", colored) == plain
    assert "No outliers." in render(
        {"schema_version": 2, "results": []}, capsys
    )


def test_priority_uses_resolved_group_score():
    cases, units = module.assess(artifact())
    unit = units[0]
    assert module.priority(cases[0], unit)[1] == pytest.approx(unit.combined)


def test_color_detection(monkeypatch):
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setenv("TERM", "xterm")
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert module.color_enabled("auto")
    monkeypatch.setenv("NO_COLOR", "")
    assert not module.color_enabled("auto")
    assert module.color_enabled("always") and not module.color_enabled("never")


@pytest.mark.parametrize("width", [60, 100, 180])
def test_rich_layout_preserves_fields(width):
    data = artifact()
    data["results"][0]["id"] = "sha256:" + "a" * 64
    data["results"][0]["parameters"]["declared"] = {
        "note": "[red]literal[/red]"
    }
    stream = StringIO()
    console = module.make_console("never", file=stream, width=width)
    cases, units = module.assess(data)
    module.report(
        data,
        "artifact.json",
        cases,
        units,
        show_all=True,
        verbose=True,
        console=console,
    )
    text = stream.getvalue()
    compact = "".join(text.split())
    assert "CaseID" in compact and "Max" in compact
    assert "[red]literal[/red]" in compact
    assert "sha256:" + "a" * 64 in compact
    assert "\x1b[" not in text
    assert max(map(len, text.splitlines())) <= width


def test_rich_forced_color_with_no_color_env(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    stream = StringIO()
    console = module.make_console("always", file=stream, width=120)
    console.print(module.Text("test", style="red"))
    assert "\x1b[31m" in stream.getvalue()


@pytest.mark.parametrize("include", [False, True])
def test_warmup_failure_always_counts(include):
    source = artifact()["results"][0]
    warmup = observation(11, "warmup")
    warmup["outcome"]["status"] = "failed"
    source["observations"].append(warmup)
    assert (
        "FAILED"
        in module.assess_case(source, None, include_warmups=include).flags
    )


def test_invalid_warmup_ignored_by_default():
    source = artifact()["results"][0]
    source["observations"].append(observation(float("nan"), "warmup"))
    assert "INVALID" not in module.assess_case(source, None).flags
    assert (
        "INVALID"
        in module.assess_case(source, None, include_warmups=True).flags
    )


def test_warmup_option_applies_to_scores_statistics_and_coverage():
    data = artifact()
    source = data["results"][0]
    del source["extensions"]
    source["observations"] = [observation(0.05), observation(0.15, "warmup")]
    data["results"] = [source]
    cases, units = module.assess(data)
    assert cases[0].mid == 0.05 and units[0].combined is None
    cases, units = module.assess(data, include_warmups=True)
    assert cases[0].mid == 0.1 and cases[0].maximum == 0.15
    assert units[0].combined is not None
    assert module.statistics(cases[0].measured)["count"] == 2
    stream = StringIO()
    module.report(
        data,
        "test",
        cases,
        units,
        include_warmups=True,
        console=module.make_console("never", file=stream, width=120),
    )
    assert "Warmups are included." in stream.getvalue()


@pytest.mark.parametrize("include", [False, True])
def test_footer_warmup_selection(include):
    data = artifact()
    cases, units = module.assess(data, include_warmups=include)
    stream = StringIO()
    module.report(
        data,
        "test",
        cases,
        units,
        include_warmups=include,
        console=module.make_console("never", file=stream, width=120),
    )
    output = " ".join(stream.getvalue().split())
    selected, other = (
        ("included", "excluded") if include else ("excluded", "included")
    )
    assert f"Warmups are {selected}." in output
    assert f"Warmups are {other}." not in output
    assert "Use these scores to identify workloads worth reviewing." in output
    assert output.endswith(
        f"Use these scores to identify workloads worth reviewing. Warmups are {selected}."
    )
