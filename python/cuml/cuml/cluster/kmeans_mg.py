#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from cuml.cluster import KMeans


class KMeansMG(KMeans):
    """
    A Multi-Node Multi-GPU implementation of KMeans
    """

    _multi_gpu = True

    def __init__(self, *, handle, **kwargs):
        self.handle = handle
        super().__init__(**kwargs)

    def fit(self, X, sample_weight=None):
        """Fit the multi-GPU KMeans model.

        When ``X`` is a sequence of local data partitions.
        """
        if isinstance(X, (list, tuple)):
            return self._fit_mg_parts(X, sample_weight_parts=sample_weight)
        return super().fit(X, sample_weight=sample_weight)

    def _validate_fit_params(self):
        super()._validate_fit_params()
        if isinstance(self.init, str):
            if self.init == "k-means++":
                raise ValueError(
                    "init='k-means++' is not supported for KMeansMG. "
                    "Use init='k-means||' or init='scalable-k-means++'."
                )
            if self.init not in {
                "scalable-k-means++",
                "k-means||",
                "random",
            }:
                raise ValueError(
                    f"init={self.init!r} is not supported for KMeansMG."
                )

        if self.oversampling_factor == 0:
            raise ValueError(
                "oversampling_factor=0 is not supported for KMeansMG."
            )

    def validate(self, X, rank, n_workers):
        """Validate rank-local data constraints before calling ``fit``.

        ``X`` may be the rank-local input or a sequence of local partitions.
        This optional preflight does not replace the validation performed by
        ``fit``.
        """
        if isinstance(X, (list, tuple)):
            n_rows = sum(len(part) for part in X)
        else:
            n_rows = len(X)

        if not isinstance(self.init, str) or self.init != "random":
            return

        n_sampling_workers = min(n_workers, self.n_clusters)
        if rank >= n_sampling_workers:
            return

        required_rows = self.n_clusters // n_sampling_workers
        if rank == 0:
            required_rows += self.n_clusters % n_sampling_workers

        if n_rows < required_rows:
            raise ValueError(
                f"init='random' requires rank {rank} to sample up to "
                f"{required_rows} initial centroid(s), but this rank only "
                f"has {n_rows} row(s). Repartition the data so each rank "
                f"has enough rows for initialization, reduce n_clusters, "
                f"or provide explicit initial centers."
            )
