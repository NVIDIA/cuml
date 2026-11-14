#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate benchmark sizing using operation runtimes and variability."""

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median, pstdev, quantiles

from rich.console import Console
from rich.table import Table
from rich.text import Text

EXTENSION = "com.nvidia.cuml.benchmark"
ROLES = {"thin", "balanced", "wide"}
FAILURES = {"FAILED", "INVALID", "NO_MEASUREMENTS"}
GROUP_FLAGS = {
    "SIZE_MISMATCH",
    "INCOMPLETE_SHAPE_GROUP",
    "AMBIGUOUS_SHAPE",
    "MISSING_DIMENSIONS",
}


def timing_score(seconds):
    """Prefer 100 ms–1 s, declining linearly below and logarithmically above."""
    return (
        seconds / 0.1
        if seconds < 0.1
        else max(0, 1 - math.log10(max(1, seconds)))
    )


def variability_multiplier(relative_range):
    """Halve suitability per 20 percentage points above 10% relative range."""
    return 2 ** (-max(0, relative_range - 0.10) / 0.20)


@dataclass
class Case:
    source: dict
    measured: list = field(default_factory=list)
    maximum: float | None = None
    flags: list = field(default_factory=list)
    mid: float | None = None
    spread: float | None = None

    @property
    def metadata(self):
        return self.source.get("extensions", {}).get(EXTENSION, {})

    @property
    def timing(self):
        return (
            timing_score(self.mid)
            if self.mid is not None and not FAILURES.intersection(self.flags)
            else 0.0
        )

    @property
    def multiplier(self):
        return (
            variability_multiplier(self.spread)
            if len(self.measured) >= 2
            else None
        )

    @property
    def elements(self):
        dims = {
            d["name"]: d["size"] for d in self.source["input"]["dimensions"]
        }
        if any(
            not isinstance(dims.get(k), int)
            or isinstance(dims[k], bool)
            or dims[k] <= 0
            for k in ("rows", "features")
        ):
            return None
        return dims["rows"] * dims["features"]


def assess_case(source, profile, *, include_warmups=False):
    """Validate selected timings; execution failures always count."""
    case = Case(source)
    if source["outcome"]["status"] != "success":
        case.flags.append("FAILED")
    for obs in source["observations"]:
        if obs["outcome"]["status"] != "success":
            if "FAILED" not in case.flags:
                case.flags.append("FAILED")
            continue
        if obs["role"] != "measurement" and not (
            include_warmups and obs["role"] == "warmup"
        ):
            continue
        times = [
            t["value"]
            for t in obs["timings"]
            if t["name"] == "wall_time" and t["unit"] == "s"
        ]
        if (
            len(times) != 1
            or not isinstance(times[0], (float, int))
            or isinstance(times[0], bool)
            or not math.isfinite(times[0])
            or times[0] <= 0
        ):
            if "INVALID" not in case.flags:
                case.flags.append("INVALID")
            continue
        case.measured.extend(times)
    case.maximum = max(case.measured) if case.measured else None
    if case.measured:
        case.mid = median(case.measured)
        case.spread = (max(case.measured) - min(case.measured)) / case.mid
        if profile != "smoke" and case.mid < 0.1:
            case.flags.append("FAST")
        if case.mid > 1:
            case.flags.append("SLOW")
    else:
        case.flags.append("NO_MEASUREMENTS")
    if case.maximum is not None and case.maximum > 10:
        case.flags.append("OVER_BUDGET")
    if len(case.measured) < 2:
        case.flags.append("UNASSESSED_REPEATABILITY")
    elif case.spread > 0.1:
        case.flags.append("VARIABLE")
    return case


@dataclass
class Unit:
    members: list
    anchor: Case
    grouped: bool = False
    flags: list = field(default_factory=list)

    @property
    def timing(self):
        return (
            0.0
            if any(FAILURES.intersection(c.flags) for c in self.members)
            else self.anchor.timing
        )

    @property
    def multiplier(self):
        values = [c.multiplier for c in self.members]
        return None if None in values else min(values)

    @property
    def combined(self):
        multiplier = self.multiplier
        return None if multiplier is None else self.timing * multiplier


def group_key(case):
    source = case.source
    dataset = dict(source.get("dataset", {}))
    dataset["parameters"] = {
        k: v
        for k, v in dataset.get("parameters", {}).items()
        if k != "inference_rows"
    }
    inputs = {k: v for k, v in source["input"].items() if k != "dimensions"}
    return json.dumps(
        [
            case.metadata.get("provider"),
            source["algorithm"],
            source["operation"],
            dataset,
            inputs,
            source.get("parameters", {}).get("declared", {}),
        ],
        sort_keys=True,
    )


def assess(artifact, *, include_warmups=False):
    """Build cases and scoring units; unrecognized tags retain independent scoring."""
    if artifact["schema_version"] != 2:
        raise ValueError("expected a version-2 benchmark artifact")
    profile = (
        artifact.get("run", {})
        .get("extensions", {})
        .get(EXTENSION, {})
        .get("profile")
    )
    cases = [
        assess_case(c, profile, include_warmups=include_warmups)
        for c in artifact["results"]
    ]
    candidates, units = defaultdict(list), []
    for case in cases:
        roles = ROLES.intersection(case.metadata.get("tags", []))
        if len(roles) == 1:
            candidates[group_key(case)].append((case, next(iter(roles))))
        else:
            if roles:
                case.flags.append("AMBIGUOUS_SHAPE")
            units.append(Unit([case], case))
    for members in candidates.values():
        if len(members) != 3 or {role for _, role in members} != ROLES:
            for case, _ in members:
                case.flags.append("INCOMPLETE_SHAPE_GROUP")
                units.append(Unit([case], case))
            continue
        anchor = next(c for c, role in members if role == "balanced")
        unit = Unit([c for c, _ in members], anchor, grouped=True)
        elements = [c.elements for c in unit.members]
        if None in elements:
            unit.flags.append("MISSING_DIMENSIONS")
        elif any(
            abs(e / anchor.elements - 1) > 0.01 + 1e-12 for e in elements
        ):
            unit.flags.append("SIZE_MISMATCH")
        for case in unit.members:
            if case is not anchor:
                case.flags[:] = [
                    f for f in case.flags if f not in {"FAST", "SLOW"}
                ]
        units.append(unit)
    return cases, units


def statistics(values):
    """Population statistics and inclusive quartiles for a nonempty sequence."""
    if not values:
        return {}
    q1, _, q3 = (
        quantiles(values, method="inclusive")
        if len(values) > 1
        else values * 3
    )
    return dict(
        count=len(values),
        min=min(values),
        q1=q1,
        median=median(values),
        mean=mean(values),
        q3=q3,
        max=max(values),
        stddev=pstdev(values),
    )


def duration(seconds):
    return (
        "—"
        if seconds is None
        else (
            f"{seconds * 1000:.3g} ms" if seconds < 1 else f"{seconds:.3g} s"
        )
    )


def priority(case, unit):
    flags = case.flags + unit.flags
    if FAILURES.intersection(flags):
        return (0, unit.timing)
    if "OVER_BUDGET" in flags:
        return (1, unit.timing)
    if "VARIABLE" in flags:
        return (2, -case.spread)
    return (3, unit.combined if unit.combined is not None else unit.timing)


def details(case):
    source = case.source
    declared = source.get("parameters", {}).get("declared", {})
    data = source.get("dataset", {}).get("parameters", {})
    tags = case.metadata.get("tags")
    variants = [f"tags={','.join(tags)}"] if tags else []
    variants += [f"{k}={v}" for k, v in declared.items()]
    variants += [
        f"{k}={data[k]}" for k in ("n_classes", "n_categories") if k in data
    ]
    return [
        f"{source['algorithm']}.{source['operation']['name']}",
        " × ".join(f"{d['size']:,}" for d in source["input"]["dimensions"]),
        ", ".join(variants) or "—",
    ]


def color_enabled(mode):
    return mode == "always" or (
        mode == "auto"
        and "NO_COLOR" not in os.environ
        and os.environ.get("TERM") != "dumb"
        and sys.stdout.isatty()
    )


def make_console(color="auto", **kwargs):
    enabled = color_enabled(color)
    return Console(
        force_terminal=enabled,
        no_color=not enabled,
        color_system="standard" if enabled else None,
        markup=False,
        highlight=False,
        **kwargs,
    )


def print_table(entries, *, verbose, console):
    table = Table(
        box=None, padding=(0, 1), show_edge=False, header_style="bold"
    )
    for name in (
        "Combined",
        "Timing",
        "Var ×",
        "Median",
        "Range/med",
        "Max",
    ):
        table.add_column(name, justify="right", overflow="fold")
    for name in ("Flags", "Workload", "Shape", "Variant"):
        table.add_column(name, overflow="fold")
    if verbose:
        table.add_column("Case ID", overflow="fold")
    narrow = console.width < 120
    for case, unit in entries:
        flags = case.flags + unit.flags
        style = (
            "red"
            if (FAILURES | {"OVER_BUDGET"}).intersection(flags)
            else "yellow"
            if (
                GROUP_FLAGS | {"SLOW", "VARIABLE", "UNASSESSED_REPEATABILITY"}
            ).intersection(flags)
            else "cyan"
            if "FAST" in flags
            else "green"
        )
        cells = [
            Text(
                f"{unit.combined:.3f}" if unit.combined is not None else "—",
                style=style,
            ),
            f"{unit.timing:.3f}",
            f"{unit.multiplier:.3f}" if unit.multiplier is not None else "—",
            duration(case.mid),
            f"{case.spread:.1%}" if len(case.measured) >= 2 else "—",
            Text(
                duration(case.maximum),
                style="red" if "OVER_BUDGET" in flags else "",
            ),
            Text(",".join(flags) or "—", style=style),
            *details(case),
        ]
        if verbose:
            cells.append(case.source["id"])
        if narrow:
            console.print()
            record = Table.grid(padding=(0, 2))
            record.add_column(style="bold", no_wrap=True)
            record.add_column(overflow="fold")
            for name, value in zip((c.header for c in table.columns), cells):
                record.add_row(name, value)
            console.print(record)
        else:
            table.add_row(*cells)
    if narrow and entries:
        return
    if table.row_count:
        console.print()
        console.print(table)
    else:
        console.print("No outliers.")


def report(
    artifact,
    path,
    cases,
    units,
    *,
    show_all=False,
    verbose=False,
    color=False,
    console=None,
    include_warmups=False,
):
    console = console or make_console("always" if color else "never")
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold", no_wrap=True)
    summary.add_column(overflow="fold")
    meta = artifact.get("run", {}).get("extensions", {}).get(EXTENSION, {})
    console.print(
        f"{path} | provider={meta.get('provider', '?')} suite={meta.get('suite', '?')} profile={meta.get('profile', '?')}"
    )
    tagged = sum("tags" in c.metadata for c in cases)
    groups = sum(u.grouped for u in units)
    summary.add_row(
        "Tag metadata",
        f"{tagged}/{len(cases)} cases; {groups} balanced-anchor groups.",
    )
    if tagged < len(cases):
        summary.add_row(
            "",
            "Tag metadata is missing for some cases (legacy artifacts may omit it).",
        )
    stats = statistics([t for c in cases for t in c.measured])
    if stats:
        summary.add_row(
            "Runtime",
            f"n={stats['count']}  median={duration(stats['median'])}  mean={duration(stats['mean'])}  Q1–Q3={duration(stats['q1'])}–{duration(stats['q3'])}",
        )
        summary.add_row(
            "",
            f"min={duration(stats['min'])}  max={duration(stats['max'])}  stddev={duration(stats['stddev'])}",
        )
    else:
        summary.add_row("Runtime", "no valid measurements")
    variability = statistics(
        [
            c.spread
            for c in cases
            if c.multiplier is not None
            and not {"FAILED", "INVALID"}.intersection(c.flags)
        ]
    )
    if variability:
        summary.add_row(
            "Variability",
            f"n={variability['count']}  median={variability['median']:.1%}  mean={variability['mean']:.1%}  Q1–Q3={variability['q1']:.1%}–{variability['q3']:.1%}  max={variability['max']:.1%}",
        )
    else:
        summary.add_row(
            "Variability", "unavailable (need ≥2 valid measurements per case)"
        )
    combined = [u.combined for u in units if u.combined is not None]
    label = "groups/independent cases" if groups else "cases"
    if combined:
        average = mean(combined)
        value = Text(
            f"{average:.3f} / 1",
            style="bold green" if average >= 0.9 else "bold yellow",
        )
        value.append(
            f"  combined coverage={len(combined)}/{len(units)} {label}",
            style="default",
        )
        summary.add_row("Suitability", value)
    else:
        summary.add_row(
            "Suitability",
            f"unavailable (combined coverage=0/{len(units)} {label})",
        )
    if units:
        summary.add_row(
            "Timing only",
            f"{mean(u.timing for u in units):.3f} / 1  ({len(units)} {label})",
        )
    counts = Counter(f for c in cases for f in c.flags)
    summary.add_row(
        "Flags",
        f"{counts['FAST']} fast · {counts['VARIABLE']} variable · {counts['SLOW']} slow · {counts['OVER_BUDGET']} over budget · {counts['FAILED']} failed · {counts['INVALID']} invalid · {counts['NO_MEASUREMENTS']} unmeasured · {counts['UNASSESSED_REPEATABILITY']} insufficient repeats",
    )
    summary.add_row(
        "Group flags",
        f"{sum('SIZE_MISMATCH' in u.flags for u in units)} size-mismatched groups · {sum('MISSING_DIMENSIONS' in u.flags for u in units)} groups missing dimensions · {counts['INCOMPLETE_SHAPE_GROUP']} incomplete cases · {counts['AMBIGUOUS_SHAPE']} ambiguous cases",
    )
    console.print(summary)
    if meta.get("profile") == "smoke":
        console.print(
            "Smoke: suitability is informational; the 100 ms target does not apply."
        )
    entries = [
        (c, u)
        for u in units
        for c in u.members
        if show_all or c.flags or u.flags or u.combined != 1
    ]
    print_table(
        sorted(entries, key=lambda entry: priority(*entry)),
        verbose=verbose,
        console=console,
    )
    console.print(
        "\nUse these scores to identify workloads worth reviewing. "
        + (
            "Warmups are included."
            if include_warmups
            else "Warmups are excluded."
        )
    )


def main():
    """Read one artifact and print its summary and flagged workloads."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Reading the scores
  A score of 1 favors operations that take 100 ms to 1 s. Shorter
  operations score less because they may be too small to measure well.
  Longer operations score less because they make the suite expensive.

  Examples: 50 ms → 0.50, 100 ms–1 s → 1.00, 5 s → 0.30, 10 s → 0.

  Inconsistent timings reduce the score further. We use the range divided
  by the median: up to 10% has no penalty, 30% halves the score, and 50%
  quarters it. Compare runs with the same number of repetitions.

Flags
  FAST         Median below 100 ms. Not applied to smoke runs.
  SLOW         Median above 1 s. Outside the preferred scoring band,
               but still within the sizing policy's 10 s target unless
               OVER_BUDGET is also reported.
  OVER_BUDGET  At least one selected observation exceeds 10 s.

  Warmups are excluded from all timing calculations by default. Use
  --include-warmups to include them in statistics, scores, variability,
  and timing flags. Warmup execution failures always count as failures.
  VARIABLE     The measured range exceeds 10% of the median.

Shape tags
  Complete thin/balanced/wide groups use the balanced timing score and the
  worst variability multiplier across all three shapes. The group counts
  once in the suite average; its score is repeated on its detail rows.
  Thin/wide cases do not receive FAST or SLOW flags. OVER_BUDGET still
  applies to every observation. SIZE_MISMATCH means rows × features
  differs from balanced by more than 1% (query rows only for inference).
  MISSING_DIMENSIONS means that comparison lacks valid rows/features.

  Groups must match provider, estimator, operation, declared parameters,
  dataset settings, dtype, representation, and input selection. Feature
  widths and measured row counts may differ; reference sizes may not.
  Incomplete/ambiguous groups are flagged and scored independently.
  Other tags have no meaning to this scorer. Untagged and legacy artifacts
  keep independent scoring. Tags are not validated as policy by the loader.

  Failed or invalid cases score zero. Cases with fewer than two selected
  observations have no combined score and are left out of its average.
  The report shows how many cases contribute.

These scores help identify cases worth reviewing. A fast case may be
reasonable if larger inputs would use too much memory. A good score
doesn't tell you whether the workload is useful or the estimator converged.
""",
    )
    parser.add_argument("artifact", type=Path, help="provider JSON artifact")
    parser.add_argument(
        "--all", action="store_true", help="show every case, not only outliers"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="include full case IDs"
    )
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="ANSI colors (default: auto; respects NO_COLOR and TERM=dumb)",
    )
    parser.add_argument(
        "--include-warmups",
        action="store_true",
        help="include warmups in all timing calculations (default: exclude)",
    )
    args = parser.parse_args()
    try:
        artifact = json.loads(args.artifact.read_text())
        cases, units = assess(artifact, include_warmups=args.include_warmups)
        report(
            artifact,
            args.artifact,
            cases,
            units,
            show_all=args.all,
            verbose=args.verbose,
            console=make_console(args.color),
            include_warmups=args.include_warmups,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(f"invalid artifact: {exc}")


if __name__ == "__main__":
    main()
