#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from cuml.neighbors.kernel_density import VALID_KERNELS, KernelDensity
from cuml.neighbors.kneighbors_classifier import KNeighborsClassifier
from cuml.neighbors.kneighbors_regressor import KNeighborsRegressor
from cuml.neighbors.nearest_neighbors import NearestNeighbors, kneighbors_graph

VALID_METRICS = {
    "brute": set(
        [
            "l2",
            "euclidean",
            "l1",
            "cityblock",
            "manhattan",
            "taxicab",
            # TODO: add "braycurtis" after https://github.com/NVIDIA/raft/issues/1285
            "canberra",
            "minkowski",
            "lp",
            "chebyshev",
            "linf",
            "jensenshannon",
            "cosine",
            "correlation",
            "inner_product",
            "sqeuclidean",
            "haversine",
        ]
    ),
    "rbc": set(["euclidean", "haversine", "l2"]),
    "ivfflat": set(
        [
            "l2",
            "euclidean",
            "sqeuclidean",
            "inner_product",
            "cosine",
            "correlation",
        ]
    ),
    "ivfpq": set(
        [
            "l2",
            "euclidean",
            "sqeuclidean",
            "inner_product",
            "cosine",
            "correlation",
        ]
    ),
}

VALID_METRICS_SPARSE = {
    "brute": set(
        [
            "euclidean",
            "l2",
            "inner_product",
            "l1",
            "cityblock",
            "manhattan",
            "taxicab",
            "canberra",
            "linf",
            "chebyshev",
            "jaccard",
            "minkowski",
            "lp",
            "cosine",
            "hellinger",
        ]
    )
}
