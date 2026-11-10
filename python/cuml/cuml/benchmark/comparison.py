# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only comparison of quality metrics in recorded benchmark artifacts."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from ._selection import compile_keyword_expression
from .suite import Workload

EXTENSION = "com.nvidia.cuml.benchmark"


def _load(path):
    """Read a schema-v2 artifact and reject ambiguous workload identities."""

    def reject_constant(value):
        raise ValueError(f"nonfinite JSON number: {value}")

    artifact = json.loads(
        Path(path).read_text(), parse_constant=reject_constant
    )
    if artifact["schema_version"] != 2:
        raise ValueError("unsupported artifact schema version")
    ids = set()
    for result in artifact["results"]:
        identity = result["id"]
        if (
            identity in ids
            or Workload.from_artifact_fields(result).digest() != identity
        ):
            raise ValueError("duplicate or inconsistent workload identity")
        ids.add(identity)
    return artifact


def _summary(result, artifact, name):
    """Summarize successful measurements; incomplete cases remain unranked."""
    summary = dict(
        status="missing_case",
        valid=0,
        expected=None,
        mean=None,
        min=None,
        max=None,
        gap_to_best=None,
        best=False,
    )
    if result is None:
        return summary
    descriptor = result["extensions"].get(EXTENSION, {}).get("scoring")
    plan = artifact["run"]["extensions"][EXTENSION].get("execution_plan", [])
    entries = [
        item for item in plan if item["case_label"] == result["case_label"]
    ]
    if len(entries) > 1:
        raise ValueError("duplicate case in execution plan")
    summary["expected"] = entries[0]["repetitions"] if entries else None
    if summary["expected"] is not None and (
        type(summary["expected"]) is not int or summary["expected"] < 1
    ):
        raise ValueError("invalid repetition count")
    if descriptor is not None:
        for key in ("name", "version", "population", "partition"):
            if not isinstance(descriptor[key], str) or not descriptor[key]:
                raise ValueError("invalid scoring descriptor")
        if (
            type(descriptor["evaluation_rows"]) is not int
            or descriptor["evaluation_rows"] < 1
            or not isinstance(descriptor["parameters"], dict)
        ):
            raise ValueError("invalid scoring descriptor")
    summary["scoring"] = descriptor
    summary["case_outcome"] = result["outcome"]
    values, definitions = [], []
    observation_ids = set()
    for observation in result["observations"]:
        if observation["id"] in observation_ids:
            raise ValueError("duplicate observation identity")
        observation_ids.add(observation["id"])
        if (
            observation["role"] != "measurement"
            or observation["outcome"]["status"] != "success"
        ):
            continue
        matches = [
            metric
            for metric in observation["metrics"]
            if metric["name"] == name
        ]
        if len(matches) > 1:
            raise ValueError("duplicate metric in observation")
        for metric in matches:
            value = metric["value"]
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError("invalid recorded metric value")
            if metric["direction"] not in {
                "higher_is_better",
                "lower_is_better",
                "neutral",
            }:
                raise ValueError("invalid metric direction")
            values.append(value)
            definitions.append(
                dict(
                    scoring=descriptor,
                    unit=metric["unit"],
                    direction=metric["direction"],
                )
            )
    if not values:
        summary["status"] = (
            "unscored" if descriptor is None else "no_valid_scores"
        )
        return summary
    summary.update(
        valid=len(values),
        mean=mean(values),
        min=min(values),
        max=max(values),
        definition=definitions[0],
    )
    if (
        descriptor is None
        or any(item != definitions[0] for item in definitions)
        or (entries and entries[0].get("scoring") != descriptor)
    ):
        summary["status"] = "incompatible_scoring"
    else:
        summary["status"] = (
            "complete"
            if len(values) == summary["expected"]
            and result["outcome"]["status"] == "success"
            else "partial"
        )
    return summary


def compare(
    paths: list[str | Path], *, keyword_expression: str = ""
) -> dict[str, Any]:
    """Compare matching workload/scorer metrics against the best complete mean.

    Each file is a separate experiment, including files from the same provider.
    Only successful, complete cases with matching scoring descriptors, metric
    units/directions, and run methodology participate in comparisons. Warmups
    are excluded; partial, missing, and incompatible results remain visible.
    """
    if len(paths) < 2:
        raise ValueError("compare requires at least two artifact files")
    artifacts = []
    for path in paths:
        try:
            artifacts.append(_load(path))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"unable to read artifact {path}: {exc}") from exc
    try:
        return _compare(paths, artifacts, keyword_expression)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid comparison data: {exc}") from exc


def _compare(paths, artifacts, keyword_expression):
    providers = [
        a["run"]["extensions"][EXTENSION]["provider"] for a in artifacts
    ]
    counts = Counter(providers)
    labels = [
        f"{provider} [{index + 1}]" if counts[provider] > 1 else provider
        for index, provider in enumerate(providers)
    ]
    inputs = [
        dict(
            label=label,
            path=str(path),
            provider=provider,
            run_id=a["run"]["id"],
            system=a["run"]["system"],
            software=a["run"]["software"],
            methodology=a["run"]["methodology"],
        )
        for label, path, provider, a in zip(
            labels, paths, providers, artifacts, strict=True
        )
    ]
    indexed = [{r["id"]: r for r in a["results"]} for a in artifacts]
    rows, comparable = [], 0
    workloads = {
        result["id"]: result
        for artifact in artifacts
        for result in artifact["results"]
    }

    def workload_order(identity):
        result = workloads[identity]
        dimensions = {
            dimension["name"]: dimension["size"]
            for dimension in result["input"]["dimensions"]
        }
        return (
            result["algorithm"],
            result["operation"]["name"],
            dimensions.get("rows", 0),
            dimensions.get("features", 0),
            identity,
        )

    matches = compile_keyword_expression(keyword_expression)
    selected = []
    for identity, result in workloads.items():
        keywords = (
            result["algorithm"],
            result["operation"]["name"],
            f"{result['algorithm']}.{result['operation']['name']}",
            result["dataset"]["name"],
            result["case_label"],
            identity,
            *(
                f"{dimension['name']}={dimension['size']}"
                for dimension in result["input"]["dimensions"]
            ),
            f"dtype={result['input']['data_type']}",
        )
        if matches(keywords):
            selected.append(identity)
    if keyword_expression and not selected:
        raise ValueError(f"no benchmark cases match -k {keyword_expression!r}")

    for identity in sorted(selected, key=workload_order):
        results = [index.get(identity) for index in indexed]
        exemplar = next(r for r in results if r is not None)
        names = sorted(
            {
                m["name"]
                for r in results
                if r is not None
                for o in r["observations"]
                if o["role"] == "measurement"
                for m in o["metrics"]
            }
        )
        for name in names or [None]:
            summaries = [
                _summary(r, a, name)
                for r, a in zip(results, artifacts, strict=True)
            ]
            valid = [
                (s, a["run"]["methodology"])
                for s, a in zip(summaries, artifacts, strict=True)
                if s["valid"]
            ]
            compatible = bool(valid) and all(
                s["status"] != "incompatible_scoring"
                and s["definition"] == valid[0][0]["definition"]
                and methodology == valid[0][1]
                for s, methodology in valid
            )
            if compatible:
                complete = [s for s, _ in valid if s["status"] == "complete"]
                comparable += len(complete) * (len(complete) - 1) // 2
                direction = valid[0][0]["definition"]["direction"]
                if len(complete) >= 2 and direction != "neutral":
                    best = (max if direction == "higher_is_better" else min)(
                        s["mean"] for s in complete
                    )
                    for s in complete:
                        s["best"] = s["mean"] == best
                        s["gap_to_best"] = (
                            best - s["mean"]
                            if direction == "higher_is_better"
                            else s["mean"] - best
                        )
            else:
                for s, _ in valid:
                    s["status"] = "incompatible_scoring"
            rows.append(
                dict(
                    workload_id=identity,
                    algorithm=exemplar["algorithm"],
                    operation=exemplar["operation"]["name"],
                    dimensions=exemplar["input"]["dimensions"],
                    metric=name,
                    artifacts={
                        label: summary
                        for label, summary, result, artifact in zip(
                            labels, summaries, results, artifacts, strict=True
                        )
                        if result is not None
                        or "execution_plan"
                        not in artifact["run"]["extensions"][EXTENSION]
                        or any(
                            item["case_label"] == exemplar["case_label"]
                            for item in artifact["run"]["extensions"][
                                EXTENSION
                            ]["execution_plan"]
                        )
                    },
                )
            )
    return dict(
        baseline="best_complete_mean",
        keyword_expression=keyword_expression,
        inputs=inputs,
        comparable_metrics=comparable,
        rows=rows,
    )


def render_table(report):
    """Group metrics by workload, with direction-aware colors and literal text."""
    from rich import box
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    renderables = []
    inputs = Table(box=box.SIMPLE, padding=(0, 1), expand=True)
    inputs.add_column("Artifact", style="cyan", no_wrap=True)
    inputs.add_column("File", overflow="fold")
    inputs.add_column("Run", style="dim", overflow="fold")
    for item in report["inputs"]:
        inputs.add_row(
            *(Text(item[key]) for key in ("label", "path", "run_id"))
        )
    renderables.append(
        Panel(inputs, title="Benchmark comparison", border_style="cyan")
    )
    if report["keyword_expression"]:
        renderables.append(
            Text(f"Selection: {report['keyword_expression']}", style="dim")
        )
    differing_fields = [
        field
        for field in ("system", "software", "methodology")
        if any(
            item[field] != report["inputs"][0][field]
            for item in report["inputs"][1:]
        )
    ]
    environment_fields = [
        field for field in differing_fields if field != "methodology"
    ]
    if environment_fields:
        renderables.append(
            Text(
                f"Note: {' and '.join(environment_fields)} metadata differs. "
                "These environment differences do not prevent score comparison; "
                "see JSON input metadata for details.",
                style="yellow",
            )
        )
    if "methodology" in differing_fields:
        renderables.append(
            Text(
                "Warning: run methodology differs. Scores from differing "
                "methodologies are incompatible and are not ranked together; "
                "see JSON input metadata for details.",
                style="red",
            )
        )
    workload_tables = []
    widths = [
        Text(header).cell_len
        for header in (
            "Artifact",
            "Scored/expected",
            "Mean",
            "Min",
            "Max",
            "Gap",
            "Status",
        )
    ]
    for row in report["rows"]:
        heading = Text(
            f"\n{row['algorithm']}.{row['operation']}", style="bold cyan"
        )
        shape = " × ".join(str(d["size"]) for d in row["dimensions"])
        heading.append(
            f"  {shape}  ·  {row['workload_id'][7:19]}", style="dim"
        )
        heading.append(f"  ·  {row['metric'] or 'no metrics'}", style="bold")
        renderables.append(heading)
        table = Table(
            box=box.ROUNDED,
            header_style="bold",
            border_style="dim",
            padding=(0, 1),
            expand=True,
        )
        table.add_column("Artifact", style="cyan", overflow="fold")
        for column in ("Scored/expected", "Mean", "Min", "Max", "Gap"):
            table.add_column(column, justify="right")
        table.add_column("Status", overflow="fold")
        ranked = sorted(
            (
                (label, summary)
                for label, summary in row["artifacts"].items()
                if summary["gap_to_best"] is not None
            ),
            key=lambda item: item[1]["gap_to_best"],
        )
        unranked = [
            (label, summary)
            for label, summary in row["artifacts"].items()
            if summary["gap_to_best"] is None
        ]
        for label, summary in [*ranked, *unranked]:
            values = [
                Text("—", style="dim")
                if summary[key] is None
                else Text(f"{summary[key]:.6g}")
                for key in ("mean", "min", "max")
            ]
            gap = summary["gap_to_best"]
            gap_text = Text("—", style="dim")
            if gap is not None:
                color = (
                    "green"
                    if summary["best"]
                    else "yellow"
                    if math.isclose(
                        summary["mean"],
                        ranked[0][1]["mean"],
                        rel_tol=1e-6,
                        abs_tol=1e-8,
                    )
                    else "red"
                )
                gap_text = Text(f"{gap:.6g}", style=color)
            status = summary["status"]
            status_style = (
                "green"
                if status == "complete"
                else "red"
                if status == "incompatible_scoring"
                else "yellow"
            )
            cells = (
                Text(label),
                Text(
                    f"{summary['valid']}/{summary['expected'] if summary['expected'] is not None else '?'}"
                ),
                *values,
                gap_text,
                Text(status, style=status_style),
            )
            widths = [
                max(width, cell.cell_len)
                for width, cell in zip(widths, cells, strict=True)
            ]
            table.add_row(*cells, style="bold" if summary["best"] else None)
        workload_tables.append(table)
        renderables.append(table)
        definition = next(
            (
                s["definition"]
                for s in row["artifacts"].values()
                if "definition" in s
            ),
            None,
        )
        if definition:
            renderables.append(
                Text(
                    f"{definition['unit']} · {definition['direction']} · Gap relative to best observed; zero is best",
                    style="dim",
                )
            )
    # Use the same preferred widths for every table, including provider labels and
    # scientific notation. Rich can shrink these uniformly for narrow consoles.
    for table in workload_tables:
        for column, width in zip(table.columns, widths, strict=True):
            column.width = width
            column.min_width = 1

    footer = Text(
        f"\nComparable metric pairs: {report['comparable_metrics']}",
        style="bold",
    )
    footer.append(
        "\nBold rows indicate best observed complete results, including ties. Partial and neutral results are unranked.\nYellow: numerically close (rtol=1e-6, atol=1e-8).",
        style="dim",
    )
    renderables.append(footer)
    return Group(*renderables)
