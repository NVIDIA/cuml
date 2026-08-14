#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import cuml.neighbors
from cuml.accel.estimator_proxy import ProxyBase
from cuml.common.sparse import is_sparse
from cuml.internals.interop import UnsupportedOnGPU

__all__ = (
    "NearestNeighbors",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    "KernelDensity",
)


class NearestNeighbors(ProxyBase):
    _gpu_class = cuml.neighbors.NearestNeighbors
    _other_attributes = frozenset(("_fit_method", "_tree", "_fit_X"))

    def _gpu_radius_neighbors_graph(
        self, X=None, radius=None, mode="connectivity", sort_results=False
    ):
        if mode != "connectivity":
            raise UnsupportedOnGPU(f"`mode={mode!r}` is not supported")
        if sort_results:
            raise UnsupportedOnGPU("`sort_results=True` is not supported")
        if is_sparse(X):
            raise UnsupportedOnGPU("Sparse inputs are not supported")
        if self.effective_metric_ not in ["l2", "euclidean"]:
            raise UnsupportedOnGPU(
                f"metric={self.effective_metric_!r} is not supported"
            )
        return self._gpu.radius_neighbors_graph(X=X, radius=radius)


class KNeighborsClassifier(ProxyBase):
    _gpu_class = cuml.neighbors.KNeighborsClassifier
    _other_attributes = frozenset(("_fit_method", "_tree", "_fit_X", "_y"))


class KNeighborsRegressor(ProxyBase):
    _gpu_class = cuml.neighbors.KNeighborsRegressor
    _other_attributes = frozenset(("_fit_method", "_tree", "_fit_X", "_y"))


class KernelDensity(ProxyBase):
    _gpu_class = cuml.neighbors.KernelDensity
