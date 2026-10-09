# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Define provider bindings and lazy estimator import specifications."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..backends.base import Backend


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

    def estimator_spec(self, estimator: str) -> EstimatorSpec:
        """Resolve an estimator within this provider's catalog."""
        try:
            return self.catalog[estimator]
        except KeyError as exc:
            raise ValueError(
                f"estimator {estimator!r} is not supported by this provider"
            ) from exc
