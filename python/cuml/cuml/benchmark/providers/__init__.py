# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register user-selectable benchmark providers without importing estimators."""

from .accel import PROVIDER as ACCEL_PROVIDER
from .base import Provider
from .cuml import PROVIDER as CUML_PROVIDER
from .dask import PROVIDER as DASK_PROVIDER
from .hdbscan import PROVIDER as HDBSCAN_PROVIDER
from .sklearn import PROVIDER as SKLEARN_PROVIDER
from .umap import PROVIDER as UMAP_PROVIDER

PROVIDERS: dict[str, Provider] = {
    "scikit-learn": SKLEARN_PROVIDER,
    "umap-learn": UMAP_PROVIDER,
    "hdbscan": HDBSCAN_PROVIDER,
    "cuml": CUML_PROVIDER,
    "cuml.accel": ACCEL_PROVIDER,
    "cuml.dask": DASK_PROVIDER,
}


def get_provider(name: str) -> Provider:
    """Return the benchmark provider registered under the given name."""
    return PROVIDERS[name]
