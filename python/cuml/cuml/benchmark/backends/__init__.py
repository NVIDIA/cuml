# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register execution adapters independently of estimator providers."""

from .accel import AccelBackend
from .base import Backend
from .cuml import CumlBackend
from .dask import DaskBackend

BACKENDS: dict[str, Backend] = {
    "cpu": Backend(),
    "cuml": CumlBackend(),
    "cuml.accel": AccelBackend(),
    "cuml.dask": DaskBackend(),
}


def get_backend(execution: str) -> Backend:
    """Return the registered execution adapter."""
    return BACKENDS[execution]
