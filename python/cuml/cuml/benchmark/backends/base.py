# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Define backend hooks for benchmark execution and verification."""

from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .._utils import _version

if TYPE_CHECKING:
    from ..providers.base import EstimatorSpec
    from ..suite import ResolvedCase, Suite


@dataclass
class VerificationResult:
    """Store observation verification evidence and an optional failure."""

    extensions: dict[str, Any] = field(default_factory=dict)
    failure: dict[str, Any] | None = None


class Backend:
    """Provide estimator, runtime, and metadata hooks for benchmarks."""

    extra_packages: tuple[str, ...] = ()

    def bootstrap_process(self) -> None:
        """Bootstrap process startup for the backend."""
        pass

    def prepare_process(self) -> None:
        """Prepare process prerequisites before benchmark execution."""
        pass

    def validate_profile(self, profile_name: str, warmups: int) -> None:
        """Validate backend-specific execution profile requirements.

        Parameters
        ----------
        profile_name : str
            Selected profile name.
        warmups : int
            Number of warmup repetitions.
        """
        pass

    def load_estimator(self, spec: EstimatorSpec) -> type[Any]:
        """Import an estimator class from its specification.

        Parameters
        ----------
        spec : EstimatorSpec
            Estimator module and class name.
        """
        return getattr(importlib.import_module(spec.module), spec.name)

    def construct_estimator(
        self,
        estimator_class: type[Any],
        parameters: dict[str, Any],
        runtime: Any = None,
    ) -> Any:
        """Construct an estimator with the declared parameters.

        Parameters
        ----------
        estimator_class : type
            Estimator class to instantiate.
        parameters : dict
            Estimator constructor arguments.
        runtime : Any, optional
            Backend runtime context, unused by the default implementation.
        """
        return estimator_class(**parameters)

    def effective_parameters(self, estimator: Any) -> dict[str, Any]:
        """Return effective estimator parameters for result metadata.

        Parameters
        ----------
        estimator : Any
            Constructed estimator exposing get_params.
        """
        return estimator.get_params(deep=False)

    def convert_data(
        self, case: ResolvedCase, X: Any, y: Any
    ) -> tuple[Any, Any]:
        """Convert generated inputs to the backend's data representation.

        Parameters
        ----------
        case : ResolvedCase
            Case defining the input representation.
        X : Any
            Complete generated feature data.
        y : Any
            Complete generated target data.
        """
        return X, y

    def prepare_inputs(
        self, inputs: tuple[Any, ...], runtime: Any = None
    ) -> tuple[Any, ...]:
        """Materialize selected operation and fit inputs before execution.

        Parameters
        ----------
        inputs : tuple
            Inputs selected for the operation and any initial fit.
        runtime : Any, optional
            Backend runtime used to prepare inputs.
        """
        return inputs

    def runtime(self, suite: Suite) -> contextlib.AbstractContextManager[Any]:
        """Provide the runtime context for suite execution.

        Parameters
        ----------
        suite : Suite
            Suite whose runtime is being prepared.
        """
        return contextlib.nullcontext()

    def synchronize(self, value: Any = None) -> None:
        """Wait for backend work to complete before timing boundaries.

        Parameters
        ----------
        value : Any, optional
            Operation output to synchronize, if applicable.
        """
        pass

    def instrumentation(
        self, suite: Suite, role: str, repetition: int
    ) -> contextlib.AbstractContextManager[VerificationResult]:
        """Provide observation instrumentation and verification state.

        Parameters
        ----------
        suite : Suite
            Suite being executed.
        role : str
            Observation role: warmup or measurement.
        repetition : int
            Zero-based repetition index within the role.
        """
        return contextlib.nullcontext(VerificationResult())

    def package_source(self, package: str) -> dict[str, Any] | None:
        """Return available source provenance for a software package.

        Parameters
        ----------
        package : str
            Installed package name.
        """
        return None

    def software_runtimes(self) -> list[dict[str, str]]:
        """Return backend runtime versions used for resume compatibility."""
        return []

    def software_packages(
        self, specs: Iterable[EstimatorSpec]
    ) -> list[dict[str, Any]]:
        """Collect software package metadata used by the cases.

        Parameters
        ----------
        specs : iterable of EstimatorSpec
            Estimator specifications whose packages should be recorded.
        """
        # Snapshot the environment without importing optional GPU packages.
        # Estimator distributions alone omit generators, numerical libraries,
        # distributed execution, and CUDA libraries. A complete snapshot also
        # handles CUDA-suffixed wheels and future transitive dependencies.
        versions = {
            distribution.metadata["Name"]: distribution.version
            for distribution in importlib.metadata.distributions()
            if distribution.metadata["Name"]
        }
        names = {spec.package for spec in specs} | set(self.extra_packages)
        for name in names:
            versions.setdefault(name, _version(name))
        return [
            {
                "name": name,
                "version": version,
                "build": None,
                "source": self.package_source(name),
            }
            for name, version in sorted(versions.items())
        ]


class GPUBackend(Backend):
    """Provide device synchronization for GPU benchmark backends."""

    def software_runtimes(self) -> list[dict[str, str]]:
        """Record CUDA driver and runtime versions without suppressing errors."""
        runtime = importlib.import_module("cupy").cuda.runtime
        return [
            {
                "name": "cuda-driver",
                "version": str(runtime.driverGetVersion()),
            },
            {
                "name": "cuda-runtime",
                "version": str(runtime.runtimeGetVersion()),
            },
        ]

    def synchronize(self, value: Any = None) -> None:
        """Wait for work on the current CUDA device to complete.

        Parameters
        ----------
        value : Any, optional
            Operation output, unused by device synchronization.
        """
        # Asynchronous CUDA failures may first surface at this boundary. Let
        # them propagate so the harness cannot record a successful observation
        # for failed GPU work (or an unsynchronized timing).
        importlib.import_module("cupy").cuda.runtime.deviceSynchronize()
