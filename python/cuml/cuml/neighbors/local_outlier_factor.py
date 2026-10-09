#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import warnings
from numbers import Real

import cupy as cp
from sklearn.utils.metaestimators import available_if

from cuml.internals.outputs import ReflectedAttr, mlfunc
from cuml.internals.validation import check_is_fitted
from cuml.neighbors.nearest_neighbors import NeighborsBase


def _novelty_enabled(model):
    if not model.novelty:
        raise AttributeError(
            "predict, decision_function and score_samples are only available "
            "when novelty=True. Use fit_predict to label the training data."
        )
    return True


def _novelty_disabled(model):
    if model.novelty:
        raise AttributeError(
            "fit_predict is only available when novelty=False. Use predict "
            "to label new data."
        )
    return True


class LocalOutlierFactor(NeighborsBase):
    """
    Unsupervised outlier detection using the Local Outlier Factor (LOF).

    The anomaly score of each sample is called the Local Outlier Factor. It
    measures the local deviation of the density of a given sample with
    respect to its neighbors. The local density is estimated from the
    distances to the ``n_neighbors`` nearest neighbors, and samples that have
    a substantially lower density than their neighbors are considered
    outliers.

    Parameters
    ----------
    n_neighbors : int (default=20)
        Number of neighbors to use. If it is larger than the number of
        samples provided, all other samples are used.
    algorithm : string (default='auto')
        The query algorithm to use, see
        :class:`~cuml.neighbors.NearestNeighbors` for the valid options.
    metric : string (default='euclidean')
        Distance metric to use, see :class:`~cuml.neighbors.NearestNeighbors`
        for the supported metrics of each algorithm.
    p : float (default=2)
        Parameter for the Minkowski metric. When p = 1, this is equivalent to
        manhattan distance (l1), and euclidean distance (l2) for p = 2. For
        arbitrary p, minkowski distance (lp) is used.
    algo_params : dict, optional (default=None)
        Used to configure the nearest neighbor algorithm to be used, see
        :class:`~cuml.neighbors.NearestNeighbors`.
    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.
    contamination : 'auto' or float (default='auto')
        The proportion of outliers in the data set, used to define the
        threshold on the scores of the samples.

        - if ``'auto'``, the threshold is determined as in the original
          paper (``offset_ = -1.5``).
        - if a float, it should be in the range (0, 0.5].
    novelty : bool (default=False)
        By default, LocalOutlierFactor is only meant to be used for outlier
        detection (``novelty=False``), and only ``fit_predict`` is available.
        Set to True to use it for novelty detection. In that case ``predict``,
        ``decision_function`` and ``score_samples`` should only be used on
        new unseen data and not on the training set.
    n_jobs : int (default = None)
        Ignored, here for scikit-learn API compatibility.
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
    negative_outlier_factor_ : array of shape (n_samples,)
        The opposite LOF of the training samples. The higher, the more
        normal. Inliers tend to have a score close to -1, while outliers tend
        to have a lower score.
    n_neighbors_ : int
        The actual number of neighbors used for the neighbor queries.
    offset_ : float
        Offset used to obtain binary labels from the raw scores. Samples with
        a ``negative_outlier_factor_`` below ``offset_`` are detected as
        outliers.
    effective_metric_ : str
        The effective metric used for the distance computation.
    n_samples_fit_ : int
        The number of samples in the fitted data.
    n_features_in_ : int
        Number of features seen during `fit`.

    Examples
    --------

    .. code-block:: python

        >>> import cupy as cp
        >>> from cuml.neighbors import LocalOutlierFactor

        >>> X = cp.array([[-1.1], [0.2], [101.1], [0.3]], dtype=cp.float32)
        >>> lof = LocalOutlierFactor(n_neighbors=2)
        >>> lof.fit_predict(X)
        array([ 1,  1, -1,  1])

    Notes
    -----
    The neighbor search runs in single precision, also for float64 inputs,
    and the scores are returned as float32. With the euclidean metric the
    distances come from the expanded form of the distance, which loses
    accuracy when the samples are far from the origin compared to the
    distances between them. On such data the neighbors, and so the scores,
    can differ from scikit-learn. Centering the features before fitting
    reduces this effect.

    When several training samples are at the same distance from a query,
    the selected neighbors can differ from scikit-learn, which changes the
    score of the samples around the tie.

    For additional docs, see `scikitlearn's LocalOutlierFactor
    <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html>`_.
    """

    negative_outlier_factor_ = ReflectedAttr()

    _cpu_class_path = "sklearn.neighbors.LocalOutlierFactor"

    @classmethod
    def _get_param_names(cls):
        return [*super()._get_param_names(), "contamination", "novelty"]

    @classmethod
    def _params_from_cpu(cls, model):
        return {
            "contamination": model.contamination,
            "novelty": model.novelty,
            **super()._params_from_cpu(model),
        }

    def _params_to_cpu(self):
        return {
            "contamination": self.contamination,
            "novelty": self.novelty,
            **super()._params_to_cpu(),
        }

    def _attrs_from_cpu(self, model):
        return {
            "n_neighbors_": model.n_neighbors_,
            "offset_": model.offset_,
            "negative_outlier_factor_": cp.asarray(
                model.negative_outlier_factor_, dtype="float32"
            ),
            "_distances_fit_X_": cp.asarray(
                model._distances_fit_X_, dtype="float32"
            ),
            "_lrd": cp.asarray(model._lrd, dtype="float32"),
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        return {
            "n_neighbors_": self.n_neighbors_,
            "offset_": self.offset_,
            "negative_outlier_factor_": cp.asnumpy(
                self.negative_outlier_factor_
            ),
            "_distances_fit_X_": cp.asnumpy(self._distances_fit_X_),
            "_lrd": cp.asnumpy(self._lrd),
            **super()._attrs_to_cpu(model),
        }

    def __init__(
        self,
        *,
        n_neighbors=20,
        algorithm="auto",
        metric="euclidean",
        p=2,
        algo_params=None,
        metric_params=None,
        contamination="auto",
        novelty=False,
        n_jobs=None,  # Ignored, here for sklearn API compatibility
        verbose=False,
        output_type=None,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            metric=metric,
            p=p,
            algo_params=algo_params,
            metric_params=metric_params,
            n_jobs=n_jobs,
            verbose=verbose,
            output_type=output_type,
        )
        self.contamination = contamination
        self.novelty = novelty

    @mlfunc(set_input_type=True)
    def fit(self, X, y=None) -> "LocalOutlierFactor":
        """
        Fit the local outlier factor detector from the training dataset.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : LocalOutlierFactor
            The fitted local outlier factor detector.
        """
        if self.contamination != "auto" and not (
            isinstance(self.contamination, Real)
            and 0.0 < self.contamination <= 0.5
        ):
            raise ValueError(
                "contamination must be 'auto' or a float in the range "
                "(0, 0.5]."
            )

        super().fit(X)

        n_samples = self.n_samples_fit_
        if n_samples < 2:
            raise ValueError(
                "LocalOutlierFactor needs at least 2 samples, got "
                f"n_samples = {n_samples}."
            )
        if self.n_neighbors > n_samples:
            warnings.warn(
                f"n_neighbors ({self.n_neighbors}) is greater than the "
                f"total number of samples ({n_samples}). n_neighbors will be "
                "set to (n_samples - 1) for estimation."
            )
        self.n_neighbors_ = min(self.n_neighbors, n_samples - 1)

        distances, indices = self.kneighbors(n_neighbors=self.n_neighbors_)
        self._distances_fit_X_ = distances
        self._lrd = self._local_reachability_density(distances, indices)
        self.negative_outlier_factor_ = -cp.mean(
            self._lrd[indices] / self._lrd[:, None], axis=1
        )

        if self.contamination == "auto":
            # Inliers score around -1, the higher the less abnormal
            self.offset_ = -1.5
        else:
            self.offset_ = float(
                cp.percentile(
                    self.negative_outlier_factor_, 100.0 * self.contamination
                )
            )

        if not self.novelty and self.negative_outlier_factor_.min() < -1e7:
            warnings.warn(
                "Duplicate values are leading to incorrect results. "
                "Increase the number of neighbors for more accurate results."
            )

        return self

    def _local_reachability_density(self, distances, indices):
        """The inverse of the mean reachability distance to the neighbors."""
        k_distances = self._distances_fit_X_[indices, self.n_neighbors_ - 1]
        reach_distances = cp.maximum(distances, k_distances)
        # 1e-10 avoids `nan` when there are more than n_neighbors_ duplicates
        return 1.0 / (cp.mean(reach_distances, axis=1) + 1e-10)

    @available_if(_novelty_disabled)
    @mlfunc(preserve_index=True)
    def fit_predict(self, X, y=None):
        """
        Fit the model to the training set X and return the labels.

        Only available when ``novelty=False``.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        labels : array of shape (n_samples,)
            1 for inliers, -1 for outliers.
        """
        self.fit(X)
        return cp.where(self.negative_outlier_factor_ < self.offset_, -1, 1)

    @available_if(_novelty_enabled)
    @mlfunc(preserve_index=True)
    def score_samples(self, X):
        """
        Opposite of the Local Outlier Factor of X.

        Only available when ``novelty=True``. The argument X is supposed to
        contain new data: the samples in X are not considered in the
        neighborhood of any point, so the scores of the training samples are
        available through ``negative_outlier_factor_`` instead.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            The query samples.

        Returns
        -------
        scores : array of shape (n_samples,)
            The opposite of the Local Outlier Factor of each sample. The
            lower, the more abnormal.
        """
        check_is_fitted(self)
        distances, indices = self.kneighbors(X, n_neighbors=self.n_neighbors_)
        lrd = self._local_reachability_density(distances, indices)
        return -cp.mean(self._lrd[indices] / lrd[:, None], axis=1)

    @available_if(_novelty_enabled)
    @mlfunc(preserve_index=True)
    def decision_function(self, X):
        """
        Shifted opposite of the Local Outlier Factor of X.

        Only available when ``novelty=True``. The shift offset allows a zero
        threshold for being an outlier.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            The query samples.

        Returns
        -------
        scores : array of shape (n_samples,)
            The shifted opposite of the Local Outlier Factor of each sample.
            Negative scores represent outliers, positive scores represent
            inliers.
        """
        return self.score_samples(X) - self.offset_

    @available_if(_novelty_enabled)
    @mlfunc(preserve_index=True)
    def predict(self, X):
        """
        Predict the labels (1 inlier, -1 outlier) of X according to LOF.

        Only available when ``novelty=True``.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            The query samples.

        Returns
        -------
        labels : array of shape (n_samples,)
            1 for inliers, -1 for outliers.
        """
        return cp.where(self.score_samples(X) < self.offset_, -1, 1)
