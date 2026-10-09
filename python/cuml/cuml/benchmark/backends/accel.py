# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run accelerated estimators and verify warmup GPU dispatch."""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..providers.base import EstimatorSpec
    from ..suite import Suite
    from ...accel.profilers import ProfileResults

from .base import VerificationResult
from .cuml import CumlBackend

ACCEL_EXTENSION = "com.nvidia.cuml.accel"


def _dispatch_evidence(
    profile_result: ProfileResults, repetition: int
) -> dict[str, Any]:
    """Summarize GPU dispatch and CPU fallback evidence from a profile."""
    calls = []
    for name, stats in sorted(profile_result.method_calls.items()):
        calls.append(
            {
                "function": name,
                "gpu_calls": stats.gpu_calls,
                "gpu_time_s": stats.gpu_time,
                "cpu_calls": stats.cpu_calls,
                "cpu_time_s": stats.cpu_time,
                "fallback_reasons": sorted(stats.fallback_reasons),
            }
        )
    return {
        "scope": "warmup",
        "repetition": repetition,
        "calls": calls,
        "gpu_calls": sum(item["gpu_calls"] for item in calls),
        "cpu_calls": sum(item["cpu_calls"] for item in calls),
    }


class AccelBackend(CumlBackend):
    """Execute cuml.accel benchmarks with CPU fallback verification."""

    def bootstrap_process(self) -> None:
        """Activate cuml.accel at process startup, re-executing if needed."""
        import cuml.accel

        from ..suite import SuiteError

        if cuml.accel.enabled():
            return
        if os.environ.get("CUML_ACCEL_ENABLED") == "1":
            raise SuiteError(
                "cuml.accel startup bootstrap did not activate the accelerator"
            )
        # Re-exec once so the installed .pth hook enables acceleration before
        # cuML or sklearn imports. The environment flag prevents a second exec
        # if startup activation fails.
        from ..cli import _entrypoint

        environment = os.environ.copy()
        environment["CUML_ACCEL_ENABLED"] = "1"
        os.execvpe(
            sys.executable,
            [*_entrypoint(), *sys.argv[1:]],
            environment,
        )

    def validate_profile(self, profile_name: str, warmups: int) -> None:
        """Require a warmup for acceleration dispatch verification.

        Parameters
        ----------
        profile_name : str
            Selected profile name.
        warmups : int
            Number of warmup repetitions.
        """
        from ..suite import SuiteError

        if warmups < 1:
            raise SuiteError("cuml.accel requires at least one warmup")

    def load_estimator(self, spec: EstimatorSpec) -> type[Any]:
        """Import an estimator through the accelerator-aware lookup path.

        Parameters
        ----------
        spec : EstimatorSpec
            Estimator module and class name.
        """
        # Keep lookup in this module, which accel exempts from caller exclusion.
        return getattr(importlib.import_module(spec.module), spec.name)

    @contextlib.contextmanager
    def instrumentation(
        self, suite: Suite, role: str, repetition: int
    ) -> Iterator[VerificationResult]:
        """Verify warmup GPU dispatch without profiling measurements.

        Parameters
        ----------
        suite : Suite
            Suite being executed.
        role : str
            Observation role: warmup or measurement.
        repetition : int
            Zero-based repetition index within the role.
        """
        verification = VerificationResult()
        if role == "warmup":
            with importlib.import_module("cuml.accel").profile(
                quiet=True
            ) as profile:
                yield verification
            evidence = _dispatch_evidence(profile, repetition)
            verification.extensions[ACCEL_EXTENSION] = evidence
            if evidence["cpu_calls"]:
                verification.failure = {
                    "status": "failed",
                    "last_phase": "dispatch_verification",
                    "error": {
                        "type": "CpuFallback",
                        "message": "cuml.accel dispatch verification recorded CPU fallback",
                    },
                }
        else:
            # Measurements never import or initialize profiling.
            yield verification
