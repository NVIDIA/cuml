# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serialize resolved benchmark workloads into artifact descriptors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .suite import ResolvedCase


def case_artifact_fields(case: ResolvedCase) -> dict[str, Any]:
    """Build the workload fields for a case result artifact.

    Parameters
    ----------
    case : ResolvedCase
        Resolved workload to serialize.
    """
    workload = case.workload()
    return {
        "algorithm": workload.algorithm,
        "dataset": workload.dataset,
        "operation": workload.operation,
        "input": {
            "dimensions": list(workload.dimensions),
            "data_type": workload.data_type,
            "selection": list(workload.selection),
            # Representation is identity-bearing through dataset parameters,
            # not through these descriptive attributes.
            "attributes": {"format": "csr"}
            if case.input_format == "csr"
            else {},
        },
        "parameters": {"declared": workload.parameters},
    }
