# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Define provider bindings and lazy estimator import specifications."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ..backends.base import Backend


@dataclass(frozen=True)
class EstimatorSpec:
    """Identify an estimator's import location and distribution package."""

    module: str
    name: str
    package: str
    # Representation overrides and placement are provider input requirements,
    # resolved before conversion; conversion itself has no estimator semantics.
    target_representation: Literal["array", "csr", "dataframe", "series"] = (
        "array"
    )
    input_placement: Literal["default", "host"] = "default"


@dataclass(frozen=True)
class Provider:
    """Bind an estimator catalog to its execution backend."""

    backend: Backend
    catalog: Mapping[str, EstimatorSpec]

    def estimator_spec(self, estimator: str) -> EstimatorSpec:
        """Resolve an estimator within this provider's catalog."""
        try:
            return self.catalog[estimator]
        except KeyError as exc:
            raise ValueError(
                f"estimator {estimator!r} is not supported by this provider"
            ) from exc
