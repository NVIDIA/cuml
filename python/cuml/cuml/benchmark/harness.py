# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute benchmark suites and record observation artifacts."""

from __future__ import annotations

import copy
import json
import logging
import os
import platform
import socket
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ._subprocess import SubprocessExited, SubprocessTimeout, run_in_subprocess
from ._utils import _failure, _gpu_components, _jsonable, _now, atomic_write
from .backends.base import Backend
from .datasets import generate_data
from .providers.base import EstimatorSpec
from .suite import ResolvedCase, Suite, SuiteError, Workload

logger = logging.getLogger("cuml.benchmark")

METHODOLOGY = "cuml-benchmark-observations-v4"
EXTENSION = "com.nvidia.cuml.benchmark"


def _run_record(suite: Suite, argv: list[str]) -> dict[str, Any]:
    """Collect suite, command, software, and system metadata."""
    # TODO: Record effective BLAS/OpenMP thread counts and CPU affinity, and
    # include them in resume compatibility checks. Otherwise a resumed run
    # can mix results from different CPU parallelism settings.
    cpu = platform.processor() or platform.machine() or "unknown processor"
    return {
        "id": f"urn:uuid:{uuid.uuid4()}",
        "created_at": _now(),
        "methodology": {"id": METHODOLOGY},
        "command": {"argv": argv, "cwd": os.getcwd()},
        "system": {
            "label": socket.gethostname(),
            "components": [
                {
                    "type": "processor",
                    "name": cpu,
                    "count": os.cpu_count() or 1,
                    "attributes": {"platform": platform.platform()},
                },
                *_gpu_components(),
            ],
        },
        "software": {
            "runtimes": [
                {"name": "python", "version": platform.python_version()},
                *suite.provider_spec.backend.software_runtimes(),
            ],
            "packages": suite.provider_spec.backend.software_packages(
                suite.provider_spec.estimator_spec(case.estimator)
                for case in suite.cases
            ),
        },
        "extensions": {
            EXTENSION: {
                "suite": suite.name,
                "suite_path": str(suite.path),
                "profile": suite.profile_name,
                "provider": suite.provider,
                "execution_plan": [
                    {
                        "case_label": c.label,
                        "warmups": c.warmups,
                        "repetitions": c.repetitions,
                        "timeout_sec": c.timeout_sec,
                    }
                    for c in suite.cases
                ],
            }
        },
    }


def _generate_data(case: ResolvedCase, backend: Backend) -> tuple[Any, Any]:
    """Generate case inputs and convert them for the backend."""
    X, y = generate_data(case)
    return backend.convert_data(case, X, y)


def _partition_data(
    case: ResolvedCase, backend: Backend
) -> tuple[Any, Any, Any, Any]:
    """Partition generated inputs into training and measurement data."""
    # TODO: Split host inputs before backend conversion, or provide a backend
    # row-slicing hook. Dask-cuDF does not support these positional slices,
    # so categorical inference fails even when only y is selected.
    X, y = _generate_data(case, backend)
    if case.lifecycle == "fit":
        return None, None, X, y
    split = case.training_rows
    return X[:split], y[:split], X[split:], y[split:]


def _inputs(
    selection: tuple[str, ...], inputs: dict[str, Any]
) -> tuple[Any, ...]:
    """Select ordered estimator arguments from named inputs."""
    return tuple(inputs[name] for name in selection)


def _base_result(
    suite: Suite,
    case: ResolvedCase,
    implementation_record: dict[str, Any],
) -> dict[str, Any]:
    """Build a case result with workload and implementation metadata."""
    workload = case.to_artifact_fields()
    result = {
        "case_label": case.label,
        **workload,
        "parameters": {**workload["parameters"], "effective": {}},
        # Keep this byte-for-byte equivalent to the corresponding run package
        # record. In particular, writing the artifact into a clean checkout
        # must not make result.source.dirty differ from run software metadata.
        "implementation": copy.deepcopy(implementation_record),
        "outcome": {"status": "failed", "last_phase": "preparation"},
        "observations": [],
        "extensions": {
            EXTENSION: {
                "provider": suite.provider,
                "profile": suite.profile_name,
                "timeout_sec": case.timeout_sec,
                "resolved_dataset_parameters": dict(case.dataset_parameters),
            }
        },
    }
    result["id"] = case.workload_id()
    return result


@dataclass(frozen=True)
class _PreparedCase:
    """Store the estimator class and inputs prepared for a case."""

    estimator_class: type[Any]
    operation_args: tuple[Any, ...]
    fit_args: tuple[Any, ...] | None


def _prepare_case(
    backend: Backend,
    suite: Suite,
    case: ResolvedCase,
    spec: EstimatorSpec,
    client: Any = None,
) -> _PreparedCase:
    """Load the estimator class and prepare partitioned case inputs."""
    estimator_class = backend.load_estimator(spec)
    X_train, y_train, X, y = _partition_data(case, backend)
    operation_args = _inputs(case.input_selection, {"X": X, "y": y})
    fit_args = (
        ()
        if case.lifecycle == "fit"
        else _inputs(case.fit_input_selection, {"X": X_train, "y": y_train})
    )
    # Keep materialized inputs alive in the prepared case across every warmup
    # and measurement, rather than rerunning a lazy conversion graph.
    prepared_inputs = backend.prepare_inputs(operation_args + fit_args, client)
    split = len(operation_args)
    return _PreparedCase(
        estimator_class=estimator_class,
        operation_args=prepared_inputs[:split],
        fit_args=None if case.lifecycle == "fit" else prepared_inputs[split:],
    )


def _construct_estimator(
    backend: Backend,
    prepared: _PreparedCase,
    case: ResolvedCase,
    client: Any,
    result: dict[str, Any],
) -> Any:
    """Construct a fresh estimator and record its effective parameters."""
    estimator = backend.construct_estimator(
        prepared.estimator_class, dict(case.parameters), client
    )
    if not result["parameters"]["effective"] and hasattr(
        estimator, "get_params"
    ):
        result["parameters"]["effective"] = _jsonable(
            backend.effective_parameters(estimator)
        )
    return estimator


def _run_observation(
    backend: Backend,
    suite: Suite,
    case: ResolvedCase,
    estimator: Any,
    prepared: _PreparedCase,
    role: str,
    repetition: int,
    sequence: int,
) -> dict[str, Any]:
    """Measure one synchronized operation and record verification results."""
    with backend.instrumentation(suite, role, repetition) as verification:
        backend.synchronize()
        start = time.perf_counter()
        output = getattr(estimator, case.operation)(*prepared.operation_args)
        backend.synchronize(output)
        elapsed = time.perf_counter() - start
    return {
        "id": f"{role}-{repetition}",
        "sequence": sequence,
        "role": role,
        "outcome": copy.deepcopy(verification.failure)
        if verification.failure is not None
        else {"status": "success"},
        "timings": [{"name": "wall_time", "value": elapsed, "unit": "s"}],
        "metrics": [],
        "extensions": dict(verification.extensions),
    }


def _record_case_failure(
    result: dict[str, Any], exc: Exception, phase: str
) -> None:
    """Append a failed observation and update the case outcome."""
    error = _failure(exc, phase)
    role = "warmup" if phase == "warmup" else "measurement"
    result["observations"].append(
        {
            "id": f"{role}-failed",
            "sequence": len(result["observations"]),
            "role": role,
            "outcome": error,
            "timings": [],
            "metrics": [],
            "extensions": {EXTENSION: {"traceback": traceback.format_exc()}},
        }
    )
    result["outcome"] = error


def _benchmark_case(
    suite: Suite,
    case: ResolvedCase,
    client: Any = None,
    *,
    implementation_record: dict[str, Any],
    report_phase: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run a case's warmups and measurements with failure recording."""
    backend = suite.provider_spec.backend
    spec = suite.provider_spec.estimator_spec(case.estimator)
    result = _base_result(suite, case, implementation_record)
    phase = "preparation"

    def set_phase(value: str) -> None:
        """Update and report the current execution phase."""
        nonlocal phase
        phase = value
        if report_phase is not None:
            report_phase(value)

    try:
        set_phase("preparation")
        prepared = _prepare_case(backend, suite, case, spec, client)
        is_training = case.lifecycle == "fit"
        if not is_training:
            set_phase("setup")
            estimator = _construct_estimator(
                backend, prepared, case, client, result
            )
            estimator.fit(*prepared.fit_args)
            backend.synchronize()
        observations = result["observations"]
        for role, count in (
            ("warmup", case.warmups),
            ("measurement", case.repetitions),
        ):
            for repetition in range(count):
                if is_training:
                    set_phase("setup")
                    estimator = _construct_estimator(
                        backend, prepared, case, client, result
                    )
                set_phase("warmup" if role == "warmup" else "timed_execution")
                observation = _run_observation(
                    backend,
                    suite,
                    case,
                    estimator,
                    prepared,
                    role,
                    repetition,
                    len(observations),
                )
                observations.append(observation)
                logger.debug(
                    "  %s %d/%d %s in %.3fs",
                    role,
                    repetition + 1,
                    count,
                    "completed"
                    if observation["outcome"]["status"] == "success"
                    else "failed",
                    observation["timings"][0]["value"],
                )
                if observation["outcome"]["status"] == "failed":
                    result["outcome"] = copy.deepcopy(observation["outcome"])
                    return result
        result["outcome"] = {"status": "success"}
        return result
    except Exception as exc:
        _record_case_failure(result, exc, phase)
        return result


def _run_case(
    suite: Suite,
    case: ResolvedCase,
    implementation_record: dict[str, Any],
    client: Any = None,
) -> dict[str, Any]:
    """Execute a case with optional subprocess timeout enforcement."""
    if case.timeout_sec is None:
        return _benchmark_case(
            suite, case, client, implementation_record=implementation_record
        )
    phase = "process_startup"

    def report_phase(value: str) -> None:
        """Track the worker's latest execution phase."""
        nonlocal phase
        phase = value

    try:
        return run_in_subprocess(
            _benchmark_case,
            args=(suite, case),
            kwargs={"implementation_record": implementation_record},
            timeout=case.timeout_sec,
            report_phase=report_phase,
        )
    except SubprocessTimeout:
        error = _failure(
            TimeoutError(f"Case exceeded {case.timeout_sec:g} seconds"), phase
        )
        error["error"]["type"] = "TimeoutExpired"
    except SubprocessExited as exc:
        error = _failure(
            RuntimeError(f"Case worker exited with code {exc.exitcode}"), phase
        )
    result = _base_result(suite, case, implementation_record)
    result["outcome"] = error
    return result


def _interrupted_result(
    suite: Suite,
    case: ResolvedCase,
    exc: KeyboardInterrupt,
    package_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build a failed case result for a keyboard interruption."""
    spec = suite.provider_spec.estimator_spec(case.estimator)
    result = _base_result(suite, case, package_records[spec.package])
    error = _failure(exc, "interrupted")
    result["outcome"] = error
    result["observations"] = [
        {
            "id": "measurement-interrupted",
            "sequence": 0,
            "role": "measurement",
            "outcome": error,
            "timings": [],
            "metrics": [],
            "extensions": {},
        }
    ]
    return result


def _validate_resume_artifact(
    previous: dict[str, Any],
    current: dict[str, Any],
    expected_case_ids: set[str],
) -> tuple[list[dict[str, Any]], set[str]]:
    """Validate resume compatibility and select successful prior results."""
    old_run = previous["run"]
    new_run = current["run"]
    if (
        previous["schema_version"] != 2
        or old_run["methodology"] != new_run["methodology"]
        or old_run["software"] != new_run["software"]
        or old_run["system"] != new_run["system"]
        or old_run["extensions"][EXTENSION] != new_run["extensions"][EXTENSION]
    ):
        raise SuiteError(
            "resume requires matching suite, software, and system metadata"
        )
    if any(
        result["id"] not in expected_case_ids
        or Workload.from_artifact_fields(result).digest() != result["id"]
        for result in previous["results"]
    ):
        raise SuiteError("resume artifact contains incompatible workload IDs")
    retained = [
        r for r in previous["results"] if r["outcome"]["status"] == "success"
    ]
    successful = {r["id"] for r in retained}
    if len(successful) != len(retained):
        raise SuiteError("resume artifact contains duplicate successful cases")
    return retained, successful


def _resume_artifact(
    output: str | Path,
    current: dict[str, Any],
    expected_case_ids: set[str],
) -> tuple[dict[str, Any], set[str]]:
    """Load a compatible artifact and retain its successful cases."""
    try:
        previous = json.loads(Path(output).read_text(encoding="utf-8"))
        retained, successful = _validate_resume_artifact(
            previous, current, expected_case_ids
        )
        previous["results"] = retained
        previous["run"].pop("completed_at", None)
        return previous, successful
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SuiteError(f"unable to resume {output}: {exc}") from exc


def _report_suite_completed(
    suite: Suite,
    output: str | Path,
    artifact: dict[str, Any],
    suite_started: float,
) -> None:
    """Log suite completion counts, elapsed time, and artifact location."""
    if logger.isEnabledFor(logging.INFO):
        passed = sum(
            result["outcome"]["status"] == "success"
            for result in artifact["results"]
        )
        failed = len(artifact["results"]) - passed
        logger.info(
            f"Completed suite {suite.name!r}: {passed} passed, "
            f"{failed} failed in {time.perf_counter() - suite_started:.3f}s; "
            f"artifact: {Path(output).resolve()}"
        )


def run_suite(
    suite: Suite,
    output: str | Path,
    argv: list[str] | None = None,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Execute a suite and persist its benchmark artifact.

    Parameters
    ----------
    suite : Suite
        Resolved suite to execute.
    output : str or Path
        Destination for the JSON artifact.
    argv : list of str, optional
        Command recorded in metadata; defaults to sys.argv.
    resume : bool, default=False
        Retain successful cases from a compatible existing artifact.
    """
    output = Path(output).absolute()
    if not resume and (output.exists() or output.is_symlink()):
        raise SuiteError(
            f"results already exist at {output}; use --resume or a new output path"
        )
    artifact = {
        "schema_version": 2,
        "run": _run_record(suite, list(argv or sys.argv)),
        "results": [],
    }
    package_records = {
        package["name"]: package
        for package in artifact["run"]["software"]["packages"]
    }
    case_ids = [case.workload_id() for case in suite.cases]
    successful = set()
    if resume:
        artifact, successful = _resume_artifact(
            output, artifact, set(case_ids)
        )
    try:
        atomic_write(output, artifact, overwrite=resume)
    except FileExistsError as exc:
        raise SuiteError(
            f"results already exist at {output}; use --resume or a new output path"
        ) from exc
    suite_started = time.perf_counter()
    total = len(suite.cases)
    logger.info(
        "Running suite %r, profile %r (%d cases)",
        suite.name,
        suite.profile_name,
        total,
    )
    with suite.provider_spec.backend.runtime(suite) as client:
        for index, (case, case_id) in enumerate(
            zip(suite.cases, case_ids), start=1
        ):
            if case_id in successful:
                logger.info(
                    "[%d/%d] %s.%s: already complete",
                    index,
                    total,
                    case.estimator,
                    case.operation,
                )
                continue
            label = (
                f"{case.estimator}.{case.operation} "
                f"({case.measured_rows}x{case.features})"
            )
            logger.info("[%d/%d] %s: starting", index, total, label)
            case_started = time.perf_counter()
            try:
                package = suite.provider_spec.estimator_spec(
                    case.estimator
                ).package
                result = _run_case(
                    suite,
                    case,
                    package_records[package],
                    client,
                )
            except KeyboardInterrupt as exc:
                result = _interrupted_result(suite, case, exc, package_records)
                artifact["results"].append(result)
                artifact["run"]["completed_at"] = _now()
                atomic_write(output, artifact)
                logger.error(
                    "[%d/%d] %s: interrupted after %.3fs",
                    index,
                    total,
                    label,
                    time.perf_counter() - case_started,
                )
                raise
            artifact["results"].append(result)
            atomic_write(output, artifact)
            elapsed = time.perf_counter() - case_started
            outcome = result["outcome"]
            if outcome["status"] == "success":
                logger.info(
                    "[%d/%d] %s: success in %.3fs",
                    index,
                    total,
                    label,
                    elapsed,
                )
            else:
                error = outcome.get("error", {})
                logger.error(
                    "[%d/%d] %s: failed in %.3fs during %s: %s: %s",
                    index,
                    total,
                    label,
                    elapsed,
                    outcome.get("last_phase", "unknown"),
                    error.get("type", "Failure"),
                    error.get("message", outcome.get("reason", "unknown")),
                )
    artifact["run"]["completed_at"] = _now()
    atomic_write(output, artifact)
    _report_suite_completed(suite, output, artifact, suite_started)
    return artifact
