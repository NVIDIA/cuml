#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Local Outlier Factor for GPU-accelerated anomaly detection."""

import cupy as cp

from cuml.internals.base import Base
from cuml.internals.outputs import ReflectedAttr, mlfunc
from cuml.internals.validation import check_inputs, check_is_fitted
from cuml.neighbors.nearest_neighbors import NearestNeighbors

# Matches the constant scikit-learn adds to local reachability densities to
# avoid dividing by zero on duplicated training points.
_LRD_EPS = 1e-10


class LocalOutlierFactor(Base):
    """Unsupervised outlier detection using the Local Outlier Factor.

    The anomaly score of each sample measures how isolated it is from its
    neighborhood: the local reachability density of a sample is compared to
    the densities of its ``n_neighbors`` nearest neighbors, and samples with
    a substantially lower density are considered outliers.

    The nearest-neighbor search runs on GPU through
    :class:`cuml.neighbors.NearestNeighbors`; the factor computation is
    vectorized post-processing of the returned distances and indices.

    Parameters
    ----------
    n_neighbors : int, default=20
        Number of neighbors to use for the density estimate. Clamped to
        ``n_samples - 1`` when the training set is smaller.
    metric : str, default="euclidean"
        Distance metric, forwarded to
        :class:`cuml.neighbors.NearestNeighbors`. Only metrics supported by
        the underlying nearest-neighbor primitive are available.
    p : int, default=2
        Parameter of the Minkowski metric when ``metric="minkowski"``.
    contamination : "auto" or float, default="auto"
        The expected proportion of outliers, used to set ``offset_``.
        ``"auto"`` uses the original paper's threshold of -1.5; a float in
        (0, 0.5] sets the threshold at the matching quantile of the training
        scores.
    novelty : bool, default=False
        When False (outlier detection), only ``fit_predict`` and the fitted
        attributes are available. When True (novelty detection),
        ``predict``, ``decision_function`` and ``score_samples`` operate on
        new data and ``fit_predict`` is unavailable, matching scikit-learn.
    verbose : int or boolean, default=False
        Sets logging level.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return type of array outputs.

    Attributes
    ----------
    negative_outlier_factor_ : array of shape (n_samples,)
        The opposite of the local outlier factor of the training samples.
        The lower, the more abnormal.
    n_neighbors_ : int
        The effective number of neighbors used.
    offset_ : float
        Threshold on ``negative_outlier_factor_`` separating inliers from
        outliers.
    n_samples_fit_ : int
        Number of samples in the fitted data.
    effective_metric_ : str
        The metric used for the neighbor search.

    Notes
    -----
    When several training points are equidistant from a query, the selected
    neighbor set can differ from scikit-learn's, which may shift the factor
    of the affected points. This is inherent to nearest-neighbor ties and
    is bounded by the distance ties themselves.
    """

    negative_outlier_factor_ = ReflectedAttr()

    _cpu_estimator_import_path = "sklearn.neighbors.LocalOutlierFactor"

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + [
            "n_neighbors",
            "metric",
            "p",
            "contamination",
            "novelty",
        ]

    def __init__(
        self,
        *,
        n_neighbors=20,
        metric="euclidean",
        p=2,
        contamination="auto",
        novelty=False,
        verbose=False,
        output_type=None,
    ):
        super().__init__(verbose=verbose, output_type=output_type)
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.contamination = contamination
        self.novelty = novelty

    def _check_novelty(self, method, expected):
        if bool(self.novelty) != expected:
            state = "novelty=True" if expected else "novelty=False"
            raise AttributeError(
                f"{method} is only available when {state}. Set the novelty "
                "parameter accordingly before calling fit."
            )

    @mlfunc(set_input_type=True)
    def fit(self, X, y=None) -> "LocalOutlierFactor":
        """Fit the local outlier factor detector from the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : LocalOutlierFactor
            The fitted estimator.
        """
        if isinstance(self.contamination, str):
            if self.contamination != "auto":
                raise ValueError(
                    "contamination must be 'auto' or a float in (0, 0.5]."
                )
        elif not 0.0 < float(self.contamination) <= 0.5:
            raise ValueError(
                "contamination must be 'auto' or a float in (0, 0.5]."
            )
        if int(self.n_neighbors) < 1:
            raise ValueError("n_neighbors must be a positive integer.")

        X_m = check_inputs(
            self,
            X,
            dtype=("float32", "float64"),
            reset=True,
        )
        n_samples = X_m.shape[0]
        if n_samples < 2:
            raise ValueError(
                "LocalOutlierFactor requires at least 2 training samples."
            )
        self.n_samples_fit_ = n_samples
        self.n_neighbors_ = min(int(self.n_neighbors), n_samples - 1)
        self.effective_metric_ = self.metric

        nn = NearestNeighbors(
            n_neighbors=self.n_neighbors_ + 1,
            metric=self.metric,
            p=self.p,
            output_type="cupy",
        ).fit(X_m)
        dist, idx = nn.kneighbors(
            X_m, self.n_neighbors_ + 1, return_distance=True
        )

        # Drop each sample from its own neighborhood. The sample usually
        # comes back first at distance zero, but under exact duplicates it
        # can sit anywhere in the tied block, or be pushed out entirely.
        k = self.n_neighbors_
        rows = cp.arange(n_samples)
        self_pos = idx == rows[:, None]
        has_self = self_pos.any(axis=1)
        drop = cp.where(has_self, self_pos.argmax(axis=1), k)
        keep = cp.ones_like(self_pos, dtype=cp.bool_)
        keep[rows, drop] = False
        dist = dist[keep].reshape(n_samples, k)
        idx = idx[keep].reshape(n_samples, k)

        k_dist = dist[:, -1]
        reach = cp.maximum(k_dist[idx], dist)
        lrd = 1.0 / (reach.mean(axis=1) + _LRD_EPS)
        nof = -(lrd[idx].mean(axis=1) / lrd)

        self._nn = nn
        self._distances_fit_X_ = dist
        self._k_dist_fit_ = k_dist
        self._lrd = lrd
        self.negative_outlier_factor_ = nof

        if isinstance(self.contamination, str):
            self.offset_ = -1.5
        else:
            self.offset_ = float(
                cp.percentile(nof, 100.0 * float(self.contamination))
            )
        return self

    @mlfunc(set_input_type=True)
    def fit_predict(self, X, y=None):
        """Fit the detector and return training-sample labels.

        Only available when ``novelty=False``.

        Returns
        -------
        labels : array of shape (n_samples,)
            1 for inliers, -1 for outliers.
        """
        self._check_novelty("fit_predict", expected=False)
        self.fit(X)
        labels = cp.where(
            self.negative_outlier_factor_ < self.offset_, -1, 1
        ).astype(cp.int64)
        return labels

    def _score_samples(self, X):
        check_is_fitted(self)
        X_m = check_inputs(
            self,
            X,
            dtype=("float32", "float64"),
            reset=False,
        )
        dist, idx = self._nn.kneighbors(
            X_m, self.n_neighbors_, return_distance=True
        )
        reach = cp.maximum(self._k_dist_fit_[idx], dist)
        lrd_x = 1.0 / (reach.mean(axis=1) + _LRD_EPS)
        return -(self._lrd[idx].mean(axis=1) / lrd_x)

    @mlfunc
    def score_samples(self, X):
        """Opposite of the local outlier factor of X (novelty mode only).

        The lower, the more abnormal.
        """
        self._check_novelty("score_samples", expected=True)
        return self._score_samples(X)

    @mlfunc
    def decision_function(self, X):
        """Shifted opposite of the local outlier factor of X (novelty mode
        only). Negative values are outliers."""
        self._check_novelty("decision_function", expected=True)
        return self._score_samples(X) - self.offset_

    @mlfunc
    def predict(self, X):
        """Predict labels of X (novelty mode only).

        Returns
        -------
        labels : array of shape (n_samples,)
            1 for inliers, -1 for outliers.
        """
        self._check_novelty("predict", expected=True)
        scores = self._score_samples(X) - self.offset_
        return cp.where(scores < 0, -1, 1).astype(cp.int64)
