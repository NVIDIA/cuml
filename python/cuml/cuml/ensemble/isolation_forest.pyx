#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""
Isolation Forest implementation for GPU-accelerated anomaly detection.

This module provides a GPU-accelerated implementation of the Isolation Forest
algorithm, which is an unsupervised learning method for detecting anomalies.
"""

import builtins
import warnings
from numbers import Integral, Real

import cupy as cp
import numpy as np
import nvforest
import treelite

from cuml.internals.base import Base, get_handle
from cuml.internals.interop import InteropMixin, UnsupportedOnGPU
from cuml.internals.mixins import CMajorInputTagMixin
from cuml.internals.outputs import mlfunc
from cuml.internals.treelite import safe_treelite_call
from cuml.internals.validation import (
    check_inputs,
    check_is_fitted,
    check_random_seed,
)

from libc.stddef cimport size_t
from libc.stdint cimport uint64_t, uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t

from cuml.internals.logger cimport level_enum
from cuml.internals.treelite cimport (
    TreeliteFreeModel,
    TreeliteModelHandle,
    TreeliteSerializeModelToBytes,
)


# C++ declarations from isolation_forest.hpp
cdef extern from "cuml/ensemble/isolation_forest.hpp" namespace "ML" nogil:

    cdef struct IF_params:
        int n_estimators
        int max_samples
        int max_depth
        int max_features
        bool bootstrap
        uint64_t seed

    cdef void fit_treelite[T](
        const handle_t& handle,
        TreeliteModelHandle* model_handle,
        const T* input,
        size_t n_rows,
        int n_cols,
        const IF_params& params,
        double* c_normalization,
        level_enum verbosity
    ) except +


_SAMPLE_COUNT_ATOL = 1e-4


def _invert_average_path_length(value):
    """Recovers the integer sample count ``n`` with ``average_path_length(n)``
    equal to ``value``.

    The Treelite export of an isolation forest does not carry per-node sample
    counts, but every leaf value is ``depth + average_path_length(n_samples)``,
    so the count is recoverable because it is an integer. The average path
    length is strictly increasing in ``n``, with adjacent values separated by
    roughly ``2 / n``: recovery is exact for realistic ``max_samples`` and
    fails loudly once the separation approaches the tolerance rather than
    silently selecting a nearby count.
    """
    from sklearn.ensemble._iforest import _average_path_length

    def apl(n):
        return float(_average_path_length(np.asarray([n]))[0])

    if value < -_SAMPLE_COUNT_ATOL:
        raise ValueError(
            "Cannot recover a leaf sample count from negative average path "
            f"length {value!r}."
        )
    if value <= _SAMPLE_COUNT_ATOL:
        return 1
    # Bracket the value: apl is strictly increasing for n >= 2.
    hi = 2
    while apl(hi) < value:
        hi *= 2
    lo = hi // 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if apl(mid) < value:
            lo = mid
        else:
            hi = mid
    # The count is lo or hi; accept exactly one candidate within tolerance.
    matches = [n for n in (lo, hi) if abs(apl(n) - value) <= _SAMPLE_COUNT_ATOL]
    if len(matches) != 1:
        raise ValueError(
            f"Cannot recover a leaf sample count from average path length "
            f"{value!r}: {'no' if not matches else 'more than one'} integer "
            f"count matches within tolerance {_SAMPLE_COUNT_ATOL}. The "
            "Treelite export does not carry sample counts and the leaf values "
            "no longer identify them unambiguously."
        )
    return matches[0]


def _recover_node_sample_counts(tree, n_samples):
    """Recovers ``n_node_samples`` for every node of one exported isolation
    tree.

    Leaf counts come from inverting the leaf values; internal counts are
    bottom-up sums. The root count must equal ``n_samples`` (the per-tree
    sample count), which validates every inversion in the tree at once.
    """
    children_left = tree.children_left
    children_right = tree.children_right
    # Node depths are 1-based; leaf values encode the 0-based depth.
    depths = tree.compute_node_depths()
    values = tree.value.reshape(-1)
    counts = np.zeros(tree.node_count, dtype=np.int64)
    # Children are strictly deeper than their parent, so descending depth
    # order processes every child before its parent.
    for node in np.argsort(depths)[::-1]:
        if children_left[node] == -1:
            counts[node] = _invert_average_path_length(
                values[node] - (depths[node] - 1)
            )
        else:
            counts[node] = (
                counts[children_left[node]] + counts[children_right[node]]
            )
    if counts[0] != n_samples:
        raise ValueError(
            f"Recovered leaf sample counts sum to {counts[0]} at the root, "
            f"expected {n_samples}. The exported leaf values do not identify "
            "the per-node sample counts."
        )
    return counts


def _isolation_tree_to_sklearn(exported_tree, n_features, n_samples, max_depth):
    """Rebuilds one fitted sklearn ``ExtraTreeRegressor`` from one tree of the
    Treelite export, restoring the per-node sample counts that isolation
    forest scoring requires."""
    from sklearn.tree import ExtraTreeRegressor

    counts = _recover_node_sample_counts(exported_tree.tree_, n_samples)
    state = exported_tree.tree_.__getstate__()
    nodes = state["nodes"].copy()
    nodes["n_node_samples"] = counts
    nodes["weighted_n_node_samples"] = counts.astype(np.float64)
    rebuilt = ExtraTreeRegressor(max_features=1.0, max_depth=max_depth)
    rebuilt.n_features_in_ = n_features
    rebuilt.n_outputs_ = 1
    tree = type(exported_tree.tree_)(
        n_features, np.asarray([1], dtype=np.intp), 1
    )
    tree.__setstate__({**state, "nodes": nodes})
    rebuilt.tree_ = tree
    return rebuilt


class IsolationForest(InteropMixin, CMajorInputTagMixin, Base):
    """
    GPU-accelerated Isolation Forest for anomaly detection.

    Isolation Forest is an unsupervised learning algorithm for anomaly detection
    that works by isolating anomalies rather than profiling normal data points.
    It uses the concept that anomalies are few and different, so they are easier
    to isolate.

    The algorithm builds an ensemble of isolation trees where each tree is
    constructed by randomly selecting a feature and then randomly selecting a
    split value between the minimum and maximum values of the selected feature.
    Anomalies have shorter average path lengths in the trees because they are
    easier to isolate.

    Examples
    --------

    .. code-block:: python

        >>> import cupy as cp
        >>> from cuml.ensemble import IsolationForest

        >>> # Create synthetic data with some outliers
        >>> rng = cp.random.default_rng(42)
        >>> X_inliers = rng.standard_normal((100, 2), dtype=cp.float32)
        >>> X_outliers = rng.uniform(low=-4, high=4, size=(20, 2)).astype(cp.float32)
        >>> X = cp.vstack([X_inliers, X_outliers])

        >>> # Fit the model
        >>> clf = IsolationForest(n_estimators=100, random_state=42)
        >>> clf.fit(X)
        IsolationForest(random_state=42)

        >>> # Predict anomalies (-1 for anomaly, 1 for normal)
        >>> predictions = clf.predict(X)

        >>> # Get anomaly scores (lower = more anomalous)
        >>> scores = clf.score_samples(X)

    Parameters
    ----------
    n_estimators : int, default=100
        The number of isolation trees in the ensemble.
    max_samples : int, float or "auto", default="auto"
        The number of samples to draw from X to train each isolation tree.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * n_samples` samples.
        - If "auto", then `max_samples=min(256, n_samples)`.
    max_depth : int, default=None
        Maximum depth of each isolation tree. If None, depth is set to
        `ceil(log2(max_samples))`, which is the theoretical maximum depth
        needed to isolate any sample.
    max_features : float, default=1.0
        The number of features to draw from X to train each isolation tree.
        - If int, draw exactly ``max_features`` features.
        - If float, draw ``max_features * n_features`` features.
    bootstrap : bool, default=False
        If True, individual trees are fit on random subsets of the training
        data sampled with replacement. Otherwise, sampling is without
        replacement.
    random_state : int, RandomState instance or None, default=None
        Controls random row sampling and split selection. Pass an int for
        reproducible results across runs.
    contamination : float or "auto", default="auto"
        The proportion of outliers in the data set, used to define the offset
        for ``decision_function`` and ``predict``.
        - If ``"auto"``, the offset is set to -0.5.
        - If float, must be in the range (0, 0.5] and the offset is set to
          the corresponding training-score quantile.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {None, 'input', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    offset_ : float
        Offset used to compute `decision_function` from raw anomaly scores.
    max_samples_ : int
        The actual number of samples used to train each tree.

    Notes
    -----
    The implementation is based on the original Isolation Forest paper:
    Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest.
    In 2008 Eighth IEEE International Conference on Data Mining (pp. 413-422).

    **Scoring**

    The anomaly score is computed as: s(x) = 2^(-E[h(x)] / c(n))

    where:
    - h(x) is the path length of sample x in an isolation tree
    - E[h(x)] is the average path length over all trees
    - c(n) is the average path length in an unsuccessful search in a BST

    Higher values of s indicate more anomalous samples. ``score_samples()``
    returns the negative of s, so lower values indicate more anomalous samples.
    ``decision_function()`` subtracts ``offset_`` from these scores; negative
    decision-function values are predicted as anomalies.

    Fitted models can be exported to Treelite with ``as_treelite()`` and loaded
    into nvForest with ``as_nvforest()``. ``as_sklearn()`` converts a fitted
    model into an equivalent ``sklearn.ensemble.IsolationForest``;
    ``estimators_samples_`` is not available on the converted model because
    cuML does not record per-tree sample indices.
    """

    _cpu_class_path = "sklearn.ensemble.IsolationForest"

    def __init__(
        self,
        *,
        n_estimators=100,
        max_samples="auto",
        max_depth=None,
        max_features=1.0,
        bootstrap=False,
        random_state=None,
        contamination="auto",
        verbose=False,
        output_type=None,
    ):
        super().__init__(verbose=verbose, output_type=output_type)

        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.contamination = contamination

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "n_estimators",
            "max_samples",
            "max_depth",
            "max_features",
            "bootstrap",
            "random_state",
            "contamination",
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        """Convert sklearn model parameters to cuML parameters."""
        if model.warm_start:
            raise UnsupportedOnGPU("`warm_start=True` is not supported")

        return {
            "n_estimators": model.n_estimators,
            "max_samples": model.max_samples,
            "max_features": model.max_features,
            "bootstrap": model.bootstrap,
            "random_state": model.random_state,
            "contamination": model.contamination,
        }

    def _params_to_cpu(self):
        """Convert cuML parameters to sklearn parameters."""
        return {
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "random_state": self.random_state,
            "contamination": self.contamination,
        }

    def _attrs_from_cpu(self, model):
        raise UnsupportedOnGPU(
            "Conversion of a fitted sklearn IsolationForest is not supported"
        )

    def _attrs_to_cpu(self, model):
        """Converts fitted state to sklearn attributes.

        The tree structure comes from the Treelite export; the per-node sample
        counts that isolation forest scoring requires are recovered from the
        leaf values (see ``_invert_average_path_length``). ``_seeds`` is not
        transferable because cuML does not record per-tree sample indices, so
        ``estimators_samples_`` is unavailable on the converted model.
        """
        from sklearn.ensemble._iforest import _average_path_length
        from sklearn.tree import ExtraTreeRegressor

        tl_model = treelite.Model.deserialize_bytes(self._treelite_model_bytes)
        exported = treelite.sklearn.export_model(tl_model)
        n_features = self.n_features_in_
        n_samples = int(self.max_samples_)
        if self.max_depth is None:
            max_depth = int(np.ceil(np.log2(max(n_samples, 2))))
        else:
            max_depth = int(self.max_depth)
        estimators = [
            _isolation_tree_to_sklearn(tree, n_features, n_samples, max_depth)
            for tree in exported.estimators_
        ]
        return {
            "estimator_": ExtraTreeRegressor(max_features=1.0),
            "estimators_": estimators,
            "estimators_features_": [
                np.arange(n_features, dtype=np.int64) for _ in estimators
            ],
            "max_samples_": n_samples,
            "offset_": float(self.offset_),
            # The exported trees reference features globally, so scoring uses
            # the full feature set for every tree.
            "_max_features": n_features,
            "_max_samples": n_samples,
            "_sample_weight": None,
            "_average_path_length_per_tree": tuple(
                _average_path_length(est.tree_.n_node_samples)
                for est in estimators
            ),
            "_decision_path_lengths": tuple(
                est.tree_.compute_node_depths() for est in estimators
            ),
            **super()._attrs_to_cpu(model),
        }

    def __getstate__(self):
        """Pickle support - serialize state."""
        state = self.__dict__.copy()
        # nvForest model isn't currently pickleable. It's rebuilt on demand from
        # `_treelite_model_bytes`, which is the fitted model.
        state.pop("_nvforest_model", None)
        return state

    def __setstate__(self, state):
        """Pickle support - restore state."""
        self.__dict__.update(state)

    @mlfunc(set_input_type=True)
    def fit(self, X, y=None):
        """
        Fit the Isolation Forest model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to float32
            or float64.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : IsolationForest
            Fitted estimator.
        """
        # Convert input to a column-major device array for fit.
        X_m = check_inputs(
            self,
            X,
            dtype=(np.float32, np.float64),
            order="F",
            reset=True,
        )

        cdef size_t n_rows = X_m.shape[0]
        cdef int n_cols = X_m.shape[1]
        cdef uintptr_t X_ptr = X_m.data.ptr
        cdef double contamination_fraction = 0.0
        cdef bint use_contamination_quantile = False
        self.n_features_in_ = n_cols

        cdef int actual_max_features
        if isinstance(self.max_features, builtins.bool):
            raise ValueError(
                "max_features must be an int in [1, n_features] or a float "
                "in (0.0, 1.0]."
            )
        elif isinstance(self.max_features, Integral):
            if self.max_features < 1 or self.max_features > n_cols:
                raise ValueError(
                    "max_features must be an int in [1, n_features] or a "
                    "float in (0.0, 1.0]."
                )
            actual_max_features = int(self.max_features)
        elif isinstance(self.max_features, Real):
            if self.max_features <= 0.0 or self.max_features > 1.0:
                raise ValueError(
                    "max_features must be an int in [1, n_features] or a "
                    "float in (0.0, 1.0]."
                )
            actual_max_features = max(1, int(self.max_features * n_cols))
        else:
            raise ValueError(
                "max_features must be an int in [1, n_features] or a float "
                "in (0.0, 1.0]."
            )

        if isinstance(self.contamination, str):
            if self.contamination != "auto":
                raise ValueError(
                    "contamination must be 'auto' or a float in the range "
                    "(0, 0.5]."
                )
        elif isinstance(self.contamination, Real):
            contamination_fraction = float(self.contamination)
            if contamination_fraction <= 0.0 or contamination_fraction > 0.5:
                raise ValueError(
                    "contamination must be 'auto' or a float in the range "
                    "(0, 0.5]."
                )
            use_contamination_quantile = True
        else:
            raise ValueError(
                "contamination must be 'auto' or a float in the range "
                "(0, 0.5]."
            )

        # Compute max_samples
        cdef int actual_max_samples
        if isinstance(self.max_samples, str):
            if self.max_samples != "auto":
                raise ValueError(
                    "max_samples must be 'auto', a positive int, or a float "
                    "in (0.0, 1.0]."
                )
            actual_max_samples = min(256, n_rows)
        elif isinstance(self.max_samples, builtins.bool):
            raise ValueError(
                "max_samples must be 'auto', a positive int, or a float "
                "in (0.0, 1.0]."
            )
        elif isinstance(self.max_samples, Integral):
            if self.max_samples <= 0:
                raise ValueError("max_samples must be a positive integer.")
            if self.max_samples > n_rows:
                warnings.warn(
                    f"max_samples ({self.max_samples}) is greater than the "
                    f"total number of samples ({n_rows}). max_samples will "
                    "be set to n_samples for estimation.",
                    UserWarning,
                )
            actual_max_samples = min(self.max_samples, n_rows)
        elif isinstance(self.max_samples, Real):
            if self.max_samples <= 0.0 or self.max_samples > 1.0:
                raise ValueError("float max_samples must be in (0.0, 1.0].")
            actual_max_samples = int(self.max_samples * n_rows)
            if actual_max_samples < 1:
                raise ValueError(
                    "max_samples resolves to 0 samples; increase max_samples "
                    "or provide more training rows."
                )
        else:
            raise ValueError(
                "max_samples must be 'auto', a positive int, or a float "
                "in (0.0, 1.0]."
            )
        self.max_samples_ = actual_max_samples

        # Compute max_depth (-1 means auto in C++)
        cdef int actual_max_depth
        if self.max_depth is None:
            actual_max_depth = -1  # C++ will compute ceil(log2(max_samples))
        else:
            actual_max_depth = self.max_depth

        # Get random seed
        cdef uint64_t seed = check_random_seed(self.random_state)

        # Setup parameters
        cdef IF_params params
        params.n_estimators = self.n_estimators
        params.max_samples = actual_max_samples
        params.max_depth = actual_max_depth
        params.max_features = actual_max_features
        params.bootstrap = self.bootstrap
        params.seed = seed

        # Get handle and verbosity
        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><uintptr_t>handle.getHandle()
        cdef level_enum verbose = <level_enum>self._verbose_level

        cdef TreeliteModelHandle tl_handle = NULL
        cdef const char* tl_bytes = NULL
        cdef size_t tl_bytes_len
        cdef int tl_free_status
        cdef double c_normalization = 0.0
        cdef bint is_float32 = X_m.dtype == np.float32

        try:
            with nogil:
                if is_float32:
                    fit_treelite[float](
                        handle_[0],
                        &tl_handle,
                        <const float*>X_ptr,
                        n_rows,
                        n_cols,
                        params,
                        &c_normalization,
                        verbose,
                    )
                else:
                    fit_treelite[double](
                        handle_[0],
                        &tl_handle,
                        <const double*>X_ptr,
                        n_rows,
                        n_cols,
                        params,
                        &c_normalization,
                        verbose,
                    )

            # Serialize the Treelite handle immediately, following the
            # RandomForest ABI-safe pattern for Python wheels/conda environments.
            safe_treelite_call(
                TreeliteSerializeModelToBytes(
                    tl_handle, &tl_bytes, &tl_bytes_len
                ),
                "Failed to serialize Treelite model to bytes:"
            )
            tl_free_status = TreeliteFreeModel(tl_handle)
            tl_handle = NULL
            safe_treelite_call(
                tl_free_status, "Failed to free Treelite model:"
            )
        except Exception:
            if tl_handle != NULL:
                TreeliteFreeModel(tl_handle)
            raise

        self._treelite_model_bytes = <bytes>(tl_bytes[:tl_bytes_len])
        self._normalization_constant = c_normalization
        # Load the inference model here rather than on first use, so that
        # `predict` and friends don't mutate the estimator. The lazy path in
        # `_get_inference_nvforest_model` then only covers unpickled models.
        self._nvforest_model = self.as_nvforest()

        if use_contamination_quantile:
            training_scores = self.score_samples(X_m)
            self.offset_ = float(
                cp.percentile(
                    training_scores, 100.0 * contamination_fraction
                ).get()
            )
        else:
            self.offset_ = -0.5

        return self

    def as_treelite(self):
        """
        Converts this estimator to a Treelite model.

        The exported Treelite model predicts average path length across the
        isolation trees.

        Returns
        -------
        treelite.Model
        """
        check_is_fitted(self)

        return treelite.Model.deserialize_bytes(self._treelite_model_bytes)

    def as_nvforest(
        self, layout="depth_first", default_chunk_size=None, align_bytes=None,
    ):
        """
        Create a nvForest model from the Treelite-exported Isolation Forest.

        Returns
        -------
        nvforest_model : nvforest.ForestInference
            A forest inference model that predicts average path length.
        """
        check_is_fitted(self)

        return nvforest.load_from_treelite_model(
            tl_model=treelite.Model.deserialize_bytes(self._treelite_model_bytes),
            device="gpu",
            layout=layout,
            default_chunk_size=default_chunk_size,
            align_bytes=align_bytes,
            handle=get_handle(),
        )

    def _get_inference_nvforest_model(self):
        if (nvforest_model := getattr(self, "_nvforest_model", None)) is None:
            self._nvforest_model = nvforest_model = self.as_nvforest()
        return nvforest_model

    def _score_samples(self, X):
        """
        Compute anomaly scores through nvForest inference.

        Shared by ``score_samples``, ``decision_function`` and ``predict`` so
        that input validation runs exactly once per public call.
        """
        nvforest_model = self._get_inference_nvforest_model()
        dtype = nvforest_model.forest.get_dtype()

        # Convert input to a row-major device array for inference.
        X_m = check_inputs(
            self,
            X,
            dtype=dtype,
            order="C",
        )

        # Each exported leaf holds ``depth + c(n_node_samples)`` and the model
        # averages leaf values across trees, so this is E[h(x)].
        avg_path_lengths = cp.asarray(
            nvforest_model.predict(X_m), dtype=dtype
        ).reshape(-1)

        # Transform from original paper convention to sklearn convention:
        #
        # Original paper (Liu et al. 2008):
        #   s(x) = 2^(-E[h(x)] / c(n))
        #   - Anomalies: s ≈ 1 (short paths, easy to isolate)
        #   - Normal:    s ≈ 0.5 (average path length)
        #   - Very normal: s ≈ 0 (long paths, hard to isolate)
        #
        # sklearn convention:
        #   - score_samples returns the opposite of the paper score
        #   - decision_function = score_samples - offset_
        #
        # Transformation: sklearn_score = -paper_score
        #   - paper_score=1.0 (anomaly) → sklearn_score=-1.0
        #   - paper_score=0.5 (normal threshold) → sklearn_score=-0.5
        #   - paper_score=0.0 (v.normal) → sklearn_score=0.0
        #
        if self._normalization_constant <= 0:
            # c(n) is 0 for a single training sample per tree, leaving every
            # sample at the neutral score.
            return cp.full(avg_path_lengths.shape, -0.5, dtype=dtype)
        return -cp.exp2(-avg_path_lengths / self._normalization_constant)

    @mlfunc(preserve_index=True)
    def score_samples(self, X):
        """
        Compute the anomaly score of X.

        Lower scores indicate more anomalous samples. The returned scores are
        the negative of the anomaly scores defined in the original Isolation
        Forest paper.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The anomaly scores. Lower values indicate more anomalous samples.
            Typical range is approximately [-1.0, 0.0], where values below
            ``offset_`` are predicted as anomalies.
        """
        check_is_fitted(self)

        return self._score_samples(X)

    @mlfunc(preserve_index=True)
    def decision_function(self, X):
        """
        Compute the decision function of X.

        The decision function is ``score_samples(X) - offset_``.
        Negative values indicate anomalies.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The decision function. Negative values indicate anomalies.
        """
        check_is_fitted(self)

        return self._score_samples(X) - self.offset_

    @mlfunc(preserve_index=True)
    def predict(self, X):
        """
        Predict if samples are anomalies or not.

        Returns -1 for anomalies and 1 for normal samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            1 for inliers, -1 for outliers.
        """
        check_is_fitted(self)

        # ``decision_function(X) < 0`` rearranged to avoid materializing it.
        return cp.where(self._score_samples(X) < self.offset_, -1, 1)

    @mlfunc(preserve_index=True)
    def fit_predict(self, X, y=None):
        """
        Fit the model and predict on X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            1 for inliers, -1 for outliers.
        """
        return self.fit(X).predict(X)
