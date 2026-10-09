# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Published schemas for cuML benchmark artifacts."""

from importlib.resources import files


def benchmark_result_schema():
    """Return the packaged benchmark result JSON Schema resource."""
    return files(__package__).joinpath("benchmark-result.schema.json")
