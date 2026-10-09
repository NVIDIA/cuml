# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run benchmark suites and save measurement results."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import logging
import re
import sys
from pathlib import Path

from ._runner import (
    DEFAULT_PROVIDER,
    DEFAULT_SUITE,
    BenchmarkRunError,
    _run_plan,
)
from .suite import (
    BUILTIN_SUITES,
    SuiteError,
    load_suite_reference,
    suite_profile_names,
)

logger = logging.getLogger("cuml.benchmark")


def _entrypoint() -> list[str]:
    """Preserve standalone execution when launching provider workers."""
    if __package__ == "benchmark":
        return [
            sys.executable,
            str(Path(__file__).with_name("run_benchmarks.py")),
        ]
    return [sys.executable, "-m", "cuml.benchmark"]


def _parser(
    profile_names: tuple[str, ...] | None = None,
) -> argparse.ArgumentParser:
    """Build the benchmark argument parser with suite profile choices."""
    parser = argparse.ArgumentParser(
        prog=" ".join(_entrypoint())
        if __package__ == "benchmark"
        else "python -m cuml.benchmark"
    )
    parser.add_argument(
        "--suite",
        default=DEFAULT_SUITE,
        help=(
            "built-in suite name or strict YAML manifest path; built-ins: "
            + ", ".join(sorted(BUILTIN_SUITES))
            + f" (default: {DEFAULT_SUITE})"
        ),
    )
    parser.add_argument(
        "--output",
        help=(
            "output directory for one JSON results file per selected provider; "
            "defaults to a timestamped directory "
            "under .benchmarks/ in the current directory"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume --output directory, retaining successes and retrying failures",
    )
    parser.add_argument(
        "--provider",
        action="append",
        dest="providers",
        help=f"run only this suite provider (repeatable; default: {DEFAULT_PROVIDER})",
    )
    parser.add_argument(
        "--_worker", action="store_true", help=argparse.SUPPRESS
    )
    profile_metavar = (
        "{" + ",".join(profile_names) + "}" if profile_names else "PROFILE"
    )
    parser.add_argument(
        "--profile",
        metavar=profile_metavar,
        help="profile defined by the selected suite (default: standard)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="report every warmup and measurement repetition",
    )
    return parser


def _output_suite_name(suite) -> str:
    """Return a filename-safe suite name."""
    path = str(suite.path)
    name = (
        path.removeprefix("builtin:")
        if path.startswith("builtin:")
        else suite.name
    )
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-_").lower()
    return slug or "benchmark"


def default_output_path(suite, now: dt.datetime | None = None) -> Path:
    """Return an unused timestamped output directory under .benchmarks/.

    Parameters
    ----------
    suite : SuitePlan
        Suite used to name the output directory.
    now : datetime, optional
        Timestamp to use; defaults to the current UTC time.
    """
    timestamp = (now or dt.datetime.now(dt.timezone.utc)).astimezone(
        dt.timezone.utc
    )
    stem = (
        f"{_output_suite_name(suite)}-{suite.profile_name}-"
        f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}"
    )
    output_root = Path.cwd() / ".benchmarks"
    candidate = output_root / stem
    suffix = 2
    while candidate.exists():
        candidate = output_root / f"{stem}-{suffix}"
        suffix += 1
    return candidate


def _suite_aware_parser(argv: list[str]) -> argparse.ArgumentParser:
    """Build an argument parser using the selected suite's profiles."""
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--suite", default=DEFAULT_SUITE)
    probed, _ = probe.parse_known_args(argv)
    try:
        return _parser(suite_profile_names(probed.suite))
    except SuiteError:
        # Full parsing and suite loading below will report the actionable
        # manifest error. Generic help should remain available meanwhile.
        pass
    return _parser()


@contextlib.contextmanager
def _console_logging(verbose: bool):
    """Configure only benchmark output, restoring caller settings afterward."""
    target = logging.getLogger("cuml.benchmark")
    handlers, level, propagate = (
        target.handlers,
        target.level,
        target.propagate,
    )
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    target.handlers = [handler]
    target.setLevel(logging.DEBUG if verbose else logging.INFO)
    target.propagate = False
    try:
        yield
    finally:
        target.handlers = handlers
        target.setLevel(level)
        target.propagate = propagate
        handler.close()


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark command and return its exit status.

    Parameters
    ----------
    argv : list of str, optional
        Command arguments; defaults to sys.argv[1:].
    """
    argument_values = sys.argv[1:] if argv is None else argv
    parser = _suite_aware_parser(argument_values)
    args = parser.parse_args(argument_values)
    if args.resume and not args.output:
        parser.error("--resume requires --output")
    try:
        suite = load_suite_reference(
            args.suite,
            args.profile,
            args.providers or [DEFAULT_PROVIDER],
        )
        output = (
            Path(args.output).resolve()
            if args.output
            else default_output_path(suite)
        )
        with _console_logging(args.verbose):
            if args._worker:
                if len(suite.runs) != 1 or not args.output:
                    raise SuiteError(
                        "worker requires one provider and an output file"
                    )
                run = suite.runs[0]
                backend = run.provider_spec.backend
                backend.bootstrap_process()
                backend.prepare_process()
                # Import only after backend startup and prerequisite validation.
                from .harness import run_suite

                artifact = run_suite(
                    run,
                    output,
                    [*_entrypoint(), *argument_values],
                    resume=args.resume,
                )
                return int(
                    any(
                        r["outcome"]["status"] == "failed"
                        for r in artifact["results"]
                    )
                )
            try:
                _run_plan(
                    suite,
                    output,
                    resume=args.resume,
                    verbose=args.verbose,
                    entrypoint=_entrypoint(),
                )
            except BenchmarkRunError:
                return 1
            return 0
    except (SuiteError, OSError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
