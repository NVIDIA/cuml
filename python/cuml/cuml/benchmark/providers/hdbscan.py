# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Define the provider for standalone hdbscan CPU estimators."""

from ..backends import get_backend
from .base import EstimatorSpec, Provider

CATALOG = {
    "HDBSCAN": EstimatorSpec("hdbscan", "HDBSCAN", "hdbscan"),
}

PROVIDER = Provider(get_backend("cpu"), CATALOG)
