#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import warnings

import cupy as cp
import numpy as np

import cuml.cluster
from cuml.accel.estimator_proxy import ProxyBase
from cuml.cluster.hdbscan.hdbscan import _HDBSCANState
from cuml.internals.interop import InteropMixin, UnsupportedOnGPU
from cuml.internals.outputs import ArrayIndexPair

__all__ = ("KMeans", "DBSCAN", "HDBSCAN", "SpectralClustering")


_SKLEARN_HIERARCHY_DTYPE = np.dtype(
    [
        ("left_node", np.intp),
        ("right_node", np.intp),
        ("value", np.float64),
        ("cluster_size", np.intp),
    ]
)


def _linkage_to_sklearn(linkage):
    """Convert a cuML dense linkage matrix to sklearn's structured array."""
    linkage = np.asarray(linkage)
    out = np.empty(linkage.shape[0], dtype=_SKLEARN_HIERARCHY_DTYPE)
    out["left_node"] = linkage[:, 0]
    out["right_node"] = linkage[:, 1]
    out["value"] = linkage[:, 2]
    out["cluster_size"] = linkage[:, 3]
    return out


def _linkage_from_sklearn(linkage):
    """Convert sklearn's structured linkage tree to a cuML dense matrix."""
    return np.column_stack(
        [
            linkage["left_node"],
            linkage["right_node"],
            linkage["value"],
            linkage["cluster_size"],
        ]
    ).astype(np.float64, copy=False)


def _check_hdbscan_input(model, X):
    """Check data-dependent conditions unsupported by cuML HDBSCAN."""
    try:
        if hasattr(X, "__cuda_array_interface__"):
            X = cp.asarray(X)
            is_finite = bool(cp.isfinite(X).all().item())
        else:
            X = np.asarray(X)
            if X.dtype.kind not in "buif":
                raise TypeError
            is_finite = bool(np.isfinite(X).all())
    except (TypeError, ValueError):
        # Defer input coercion and validation to sklearn.
        raise UnsupportedOnGPU("Input type is not supported") from None

    if X.ndim != 2:
        raise UnsupportedOnGPU("Input does not have a two-dimensional shape")

    min_samples = (
        model.min_cluster_size
        if model.min_samples is None
        else model.min_samples
    )
    if X.shape[0] < 2 or min_samples > X.shape[0]:
        # Let sklearn produce its data-dependent validation error.
        raise UnsupportedOnGPU("The number of samples is not supported")

    if not is_finite:
        raise UnsupportedOnGPU("Input contains NaN or infinity")


class _SklearnHDBSCAN(cuml.cluster.HDBSCAN):
    """cuML HDBSCAN adapter for sklearn.cluster.HDBSCAN."""

    _cpu_class_path = "sklearn.cluster.HDBSCAN"

    @classmethod
    def _params_from_cpu(cls, model):
        if model.metric not in ("euclidean", "l2"):
            raise UnsupportedOnGPU(
                f"`metric={model.metric!r}` is not supported"
            )
        if model.metric_params:
            raise UnsupportedOnGPU(
                f"`metric_params={model.metric_params!r}` is not supported"
            )
        if model.store_centers is not None:
            raise UnsupportedOnGPU(
                f"`store_centers={model.store_centers!r}` is not supported"
            )

        sklearn_min_samples = (
            model.min_cluster_size
            if model.min_samples is None
            else model.min_samples
        )
        # Native cuML follows the contrib HDBSCAN convention and adds one to
        # min_samples before constructing the KNN graph. sklearn's
        # min_samples=k therefore corresponds to native min_samples=k-1.
        if not 2 <= sklearn_min_samples <= 1024:
            raise UnsupportedOnGPU(
                f"`min_samples={model.min_samples!r}` is not supported"
            )
        min_samples = sklearn_min_samples - 1

        return {
            "min_cluster_size": model.min_cluster_size,
            "min_samples": min_samples,
            "cluster_selection_epsilon": model.cluster_selection_epsilon,
            "max_cluster_size": model.max_cluster_size or 0,
            "metric": model.metric,
            "alpha": model.alpha,
            "cluster_selection_method": model.cluster_selection_method,
            "allow_single_cluster": model.allow_single_cluster,
        }

    def _params_to_cpu(self):
        min_samples = (
            self.min_cluster_size
            if self.min_samples is None
            else self.min_samples
        )
        return {
            "min_cluster_size": self.min_cluster_size,
            "min_samples": min_samples + 1,
            "cluster_selection_epsilon": self.cluster_selection_epsilon,
            "max_cluster_size": self.max_cluster_size or None,
            "metric": self.metric,
            "metric_params": None,
            "alpha": self.alpha,
            "algorithm": "auto",
            "leaf_size": 40,
            "n_jobs": None,
            "cluster_selection_method": self.cluster_selection_method,
            "allow_single_cluster": self.allow_single_cluster,
            "store_centers": None,
            "copy": False,
        }

    def _attrs_from_cpu(self, model):
        raw_data_cpu = getattr(model, "_raw_data", None)
        if not isinstance(raw_data_cpu, np.ndarray):
            raise UnsupportedOnGPU("Sparse inputs are not supported")
        if not np.isfinite(raw_data_cpu).all():
            raise UnsupportedOnGPU("Input contains NaN or infinity")

        raw_data = cp.asarray(raw_data_cpu, order="C", dtype=np.float32)
        labels = cp.asarray(model.labels_, order="C", dtype=np.int64)
        try:
            linkage = _linkage_from_sklearn(model._single_linkage_tree_)
            state = _HDBSCANState.from_sklearn(self, raw_data, linkage)
        except (
            AttributeError,
            IndexError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            raise UnsupportedOnGPU(
                "Fitted model does not contain a supported single-linkage tree"
            ) from exc

        n_clusters = np.unique(model.labels_[model.labels_ >= 0]).size
        if state.n_clusters != n_clusters:
            raise UnsupportedOnGPU(
                "Fitted model cannot populate equivalent native cluster state"
            )

        return {
            "labels_": ArrayIndexPair(labels, None),
            "probabilities_": cp.asarray(
                model.probabilities_, dtype=np.float32
            ),
            "_raw_data": ArrayIndexPair(raw_data, None),
            "_raw_data_cpu": raw_data_cpu,
            "_single_linkage_tree": linkage,
            "_min_spanning_tree": None,
            "_prediction_data": None,
            "_state": state,
            "n_clusters_": state.n_clusters,
            **InteropMixin._attrs_from_cpu(self, model),
        }

    def _attrs_to_cpu(self, model):
        min_samples = (
            self.min_cluster_size
            if self.min_samples is None
            else self.min_samples
        )
        return {
            "labels_": self.labels_.array.get(order="A"),
            "probabilities_": self.probabilities_.get(order="A").astype(
                np.float64
            ),
            "_raw_data": np.asarray(
                self._get_raw_data_cpu(), dtype=np.float64
            ),
            "_single_linkage_tree_": _linkage_to_sklearn(
                self._single_linkage_tree
            ),
            "_min_samples": min_samples + 1,
            "_metric_params": {},
            **InteropMixin._attrs_to_cpu(self, model),
        }


class KMeans(ProxyBase):
    _gpu_class = cuml.cluster.KMeans

    def _gpu_fit_transform(self, X, y=None, sample_weight=None):
        # Fixes signature mismatch with cuml.KMeans. Can be removed after #6741.
        return self._gpu.fit_transform(X, y=y, sample_weight=sample_weight)

    def _init_centroids(self, *args, **kwargs):
        # Exposed for use by the sklearn test suite
        return self._cpu._init_centroids(*args, **kwargs)


class DBSCAN(ProxyBase):
    _gpu_class = cuml.cluster.DBSCAN

    def _gpu_fit(self, X, y=None, sample_weight=None):
        # Fixes signature mismatch with cuml.DBSCAN. Can be removed after #6741.
        return self._gpu.fit(X, y=y, sample_weight=sample_weight)

    def _gpu_fit_predict(self, X, y=None, sample_weight=None):
        # Fixes signature mismatch with cuml.DBSCAN. Can be removed after #6741.
        return self._gpu.fit_predict(X, y=y, sample_weight=sample_weight)


class HDBSCAN(ProxyBase):
    _gpu_class = _SklearnHDBSCAN
    # Used by sklearn's HDBSCAN helpers and test suite. The attribute is
    # populated on the CPU model when fitted state is synchronized.
    _other_attributes = frozenset(("_single_linkage_tree_",))

    def _warn_copy_default(self):
        # TODO(scikit-learn 1.10): Remove this compatibility warning when the
        # temporary "warn" default is removed from sklearn.
        if self._cpu.copy == "warn":
            warnings.warn(
                "The default value of `copy` will change from False to True "
                "in 1.10. Explicitly set a value for `copy` to silence this "
                "warning.",
                FutureWarning,
                stacklevel=3,
            )

    def _gpu_fit(self, X, y=None):
        _check_hdbscan_input(self._cpu, X)
        self._warn_copy_default()
        return self._gpu.fit(X, y=y)

    def _gpu_fit_predict(self, X, y=None):
        _check_hdbscan_input(self._cpu, X)
        self._warn_copy_default()
        return self._gpu.fit_predict(X, y=y)


class SpectralClustering(ProxyBase):
    _gpu_class = cuml.cluster.SpectralClustering
    _not_implemented_attributes = frozenset(("affinity_matrix_",))
