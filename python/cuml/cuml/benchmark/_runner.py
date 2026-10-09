# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coordinate the CLI's isolated provider workers for Python callers too."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ._subprocess import run_command
from .suite import SuiteError, SuitePlan, load_suite_reference

logger = logging.getLogger("cuml.benchmark")

DEFAULT_SUITE = "estimators"
DEFAULT_PROVIDER = "cuml"
_EXTENSION = "com.nvidia.cuml.benchmark"


@dataclass(kw_only=True)
class BenchmarkResults:
    """Hold benchmark artifacts keyed by provider.

    Parameters
    ----------
    artifacts : dict[str, dict[str, Any]]
        Parsed JSON artifacts keyed by provider.
    """

    artifacts: dict[str, dict[str, Any]]


class BenchmarkRunError(RuntimeError):
    """Report provider execution failures after all selected workers finish.

    Attributes
    ----------
    results : BenchmarkResults
        Available results, including failed or partial checkpoints.
    failures : dict[str, str]
        Failed providers and diagnostic messages (including worker stderr).
    """

    def __init__(self, results: BenchmarkResults, failures: dict[str, str]):
        """Retain partial results and provider failure diagnostics."""
        self.results = results
        self.failures = failures
        super().__init__(
            "Benchmark providers failed: "
            + "; ".join(
                f"{name}: {message}" for name, message in failures.items()
            )
        )


def run(
    suite: str | Path = DEFAULT_SUITE,
    *,
    profile: str = "standard",
    providers: list[str] | None = None,
    output: str | Path | None = None,
    resume: bool = False,
) -> BenchmarkResults:
    """Run a built-in or YAML suite in sequential isolated interpreters.

    Parameters
    ----------
    suite : str or Path, default='estimators'
        Built-in suite name or YAML manifest path.
    profile : str, default='standard'
        Profile defined by the selected suite.
    providers : list of str, optional
        Selected suite providers; defaults to ['cuml']. Suite declaration order
        determines execution order.
    output : str or Path, optional
        Directory for results and checkpoints. If omitted, results are returned
        in memory and temporary files are removed.
    resume : bool, default=False
        Retain successful cases and retry failures in an explicit existing output
        directory. Each provider worker checks compatibility before executing
        its cases; earlier providers may finish before a later provider rejects
        resume.

    Returns
    -------
    BenchmarkResults
        Benchmark results containing JSON artifacts keyed by provider.

    Raises
    ------
    SuiteError
        Invalid suite, selection or output configuration, or a caller with
        cuml.accel enabled, before worker launch.
    BenchmarkRunError
        Execution or artifact failures, after attempting all selected providers.
        The exception retains available artifacts and failure diagnostics.
    KeyboardInterrupt
        Interruption, after stopping the current worker (and its process group
        on POSIX). Explicit output retains checkpoints already written.
    """
    if not isinstance(suite, (str, Path)):
        raise SuiteError("suite must be a built-in name or YAML path")
    if resume and output is None:
        raise SuiteError(
            "resume requires an explicit existing output directory"
        )
    plan = load_suite_reference(
        suite, profile, [DEFAULT_PROVIDER] if providers is None else providers
    )
    if output is not None:
        return _run_plan(plan, Path(output).resolve(), resume=resume)
    with TemporaryDirectory(prefix="cuml-benchmark-") as temporary:
        return _run_plan(plan, Path(temporary))


def _read_artifact(path: Path, provider: str) -> dict:
    """Check the artifact envelope and provider identity."""
    artifact = json.loads(path.read_text(encoding="utf-8"))
    try:
        valid = (
            isinstance(artifact, dict)
            and artifact["schema_version"] == 2
            and isinstance(artifact["run"], dict)
            and artifact["run"]["extensions"][_EXTENSION]["provider"]
            == provider
            and isinstance(artifact["results"], list)
            and all(
                isinstance(result, dict)
                and isinstance(result["id"], str)
                and result["outcome"]["status"] in ("success", "failed")
                and result["extensions"][_EXTENSION]["provider"] == provider
                for result in artifact["results"]
            )
        )
    except (KeyError, TypeError):
        valid = False
    if not valid:
        raise ValueError(f"invalid artifact envelope for {provider!r}: {path}")
    return artifact


def _run_plan(
    suite: SuitePlan,
    output: Path,
    *,
    resume: bool = False,
    verbose: bool = False,
    entrypoint: list[str] | None = None,
) -> BenchmarkResults:
    """Run isolated provider workers sequentially, retaining each checkpoint."""
    accel = sys.modules.get("cuml.accel")
    if os.environ.get("CUML_ACCEL_ENABLED", "").lower() in ("1", "true") or (
        accel is not None and accel.enabled()
    ):
        raise SuiteError(
            "Benchmarks must be launched from a process without cuml.accel "
            "enabled. Start a fresh Python process with CUML_ACCEL_ENABLED "
            "unset. To benchmark acceleration, select the cuml.accel provider."
        )
    if resume and not output.is_dir():
        raise SuiteError("--resume requires an existing output directory")
    for provider_run in suite.runs:
        if not re.fullmatch(
            r"[a-zA-Z0-9][a-zA-Z0-9._-]*", provider_run.provider
        ):
            raise SuiteError(
                f"unsafe provider filename: {provider_run.provider!r}"
            )
    if not resume:
        existing = [
            output / f"{provider_run.provider}.json"
            for provider_run in suite.runs
            if (output / f"{provider_run.provider}.json").exists()
            or (output / f"{provider_run.provider}.json").is_symlink()
        ]
        if existing:
            raise SuiteError(
                "results already exist: "
                + ", ".join(str(path) for path in existing)
                + "; use --resume or a new output directory"
            )
    try:
        output.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SuiteError(
            f"unable to create output directory {output}: {exc}"
        ) from exc
    logger.info("Writing benchmark artifacts to %s", output)
    artifacts, failures = {}, {}
    for provider_run in suite.runs:
        provider = provider_run.provider
        path = output / f"{provider}.json"
        command = [
            *(entrypoint or [sys.executable, "-m", "cuml.benchmark"]),
            "--_worker",
            "--suite",
            str(suite.path).removeprefix("builtin:"),
            "--profile",
            suite.profile_name,
            "--provider",
            provider,
            "--output",
            str(path),
        ]
        if resume and path.exists():
            command.append("--resume")
        if verbose:
            command.append("--verbose")
        logger.info(
            "Starting provider %s (%d cases)",
            provider,
            len(provider_run.cases),
        )
        try:
            run_command(command, os.environ.copy())
        except (OSError, RuntimeError) as exc:
            failures[provider] = str(exc)
        try:
            artifact = _read_artifact(path, provider)
            artifacts[provider] = artifact
            if any(
                r["outcome"]["status"] == "failed" for r in artifact["results"]
            ):
                failures.setdefault(provider, "artifact contains failed cases")
            if provider not in failures and (
                len(artifact["results"]) != len(provider_run.cases)
                or {r["id"] for r in artifact["results"]}
                != {case.workload_id() for case in provider_run.cases}
            ):
                failures[provider] = (
                    "artifact does not contain all expected cases"
                )
        except (OSError, ValueError) as exc:
            message = f"unable to read artifact: {exc}"
            failures[provider] = (
                f"{failures[provider]}; {message}"
                if provider in failures
                else message
            )
        if provider in failures:
            logger.error(
                "Provider %s failed: %s", provider, failures[provider]
            )
    logger.info(
        "Completed %d providers: %d passed, %d failed",
        len(suite.runs),
        len(suite.runs) - len(failures),
        len(failures),
    )
    results = BenchmarkResults(artifacts=artifacts)
    if failures:
        raise BenchmarkRunError(results, failures)
    return results
