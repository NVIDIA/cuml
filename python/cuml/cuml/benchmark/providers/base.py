# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Define provider bindings and lazy estimator import specifications."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..backends.base import Backend, InputRepresentation


@dataclass(frozen=True)
class EstimatorSpec:
    """Identify an estimator's import location and distribution package."""

    module: str
    name: str
    package: str


@dataclass(frozen=True)
class Provider:
    """Bind an estimator catalog to its execution backend."""

    backend: Backend
    catalog: Mapping[str, EstimatorSpec]

    def convert_data(
        self,
        value: Any,
        *,
        estimator: str,
        role: Literal["X", "y"],
        representation: InputRepresentation,
    ) -> Any:
        """Convert a selected input for the provider's estimator.

        Parameters
        ----------
        value : Any
            Host input data for the selected training or measurement partition.
        estimator : str
            Estimator name.
        role : {"X", "y"}
            Feature or target input.
        representation : {"array", "csr", "dataframe", "series"}
            Requested data representation.
        """
        return self.backend.convert_data(value, representation=representation)

    def estimator_spec(self, estimator: str) -> EstimatorSpec:
        """Resolve an estimator within this provider's catalog."""
        try:
            return self.catalog[estimator]
        except KeyError as exc:
            raise ValueError(
                f"estimator {estimator!r} is not supported by this provider"
            ) from exc
