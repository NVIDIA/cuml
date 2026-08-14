#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import warnings

from cuml.dask.common.base import BaseEstimator, DelayedPredictionMixin
from cuml.dask.ensemble.base import BaseRandomForestModel
from cuml.ensemble import RandomForestRegressor as cuRFR


class RandomForestRegressor(
    BaseRandomForestModel, DelayedPredictionMixin, BaseEstimator
):
    """
    Multi-GPU Random Forest regressor model which fits multiple decision tree
    regressors in an ensemble. This uses Dask to partition data over multiple
    GPUs (possibly on different nodes).

    During fitting, all workers that hold training rows collectively build the
    same forest from the complete distributed dataset.

    Parameters
    ----------
    n_estimators : int (default = 100)
        total number of trees in the forest
    split_criterion : int or string (default = ``2`` (``'mse'``))
        The criterion used to split nodes.\n
         * ``0`` or ``'gini'`` for gini impurity
         * ``1`` or ``'entropy'`` for information gain (entropy)
         * ``2`` or ``'mse'`` for mean squared error
         * ``4`` or ``'poisson'`` for poisson half deviance
         * ``5`` or ``'gamma'`` for gamma half deviance
         * ``6`` or ``'inverse_gaussian'`` for inverse gaussian deviance

        ``0``, ``'gini'``, ``1``, ``'entropy'`` not valid for regression
    bootstrap : boolean (default = True)
        Control bootstrapping.\n
         * If ``True``, each tree in the forest is built on a bootstrapped
           sample with replacement.
         * If ``False``, the whole dataset is used to build each tree.
    max_samples : float (default = 1.0)
        Ratio of dataset rows used while fitting each tree.
    max_depth : int or None (default = None)
        Maximum tree depth. Use ``None`` for unlimited depth (trees grow
        until all leaves are pure). Must be a positive integer or ``None``.

        .. rapids-pre-commit-hooks: disable-next-line
        .. versionchanged:: 26.08
          The default of `max_depth` changed from `16` to `None`.
    max_leaves : int (default = -1)
        Maximum leaf nodes per tree. Soft constraint. Unlimited, If ``-1``.
    max_features : float (default = 'auto')
        Ratio of number of features (columns) to consider
        per node split.\n
         * If type ``int`` then ``max_features`` is the absolute count of
           features to be used.
         * If type ``float`` then ``max_features`` is a fraction.
         * If ``'auto'`` then ``max_features=n_features = 1.0``.
         * If ``'sqrt'`` then ``max_features=1/sqrt(n_features)``.
         * If ``'log2'`` then ``max_features=log2(n_features)/n_features``.
         * If ``None``, then ``max_features = 1.0``.
    n_bins : int (default = 128)
        Maximum number of bins used by the split algorithm per feature.
    min_samples_leaf : int or float (default = 1)
        The minimum number of samples (rows) in each leaf node.\n
         * If type ``int``, then ``min_samples_leaf`` represents the minimum
           number.
         * If ``float``, then ``min_samples_leaf`` represents a fraction and
           ``ceil(min_samples_leaf * n_rows)`` is the minimum number of
           samples for each leaf node.
    min_samples_split : int or float (default = 2)
        The minimum number of samples required to split an internal node.\n
         * If type ``int``, then ``min_samples_split`` represents the minimum
           number.
         * If type ``float``, then ``min_samples_split`` represents a fraction
           and ``ceil(min_samples_split * n_rows)`` is the minimum number of
           samples for each split.
    n_streams : int
        Deprecated. Distributed training currently builds trees serially to
        preserve collective order.
    workers : optional, list of strings
        Dask addresses of workers to use for computation.
        If None, all available Dask workers will be used.
    random_state : int (default = None)
        Seed for the random number generator. Unseeded by default.
    ignore_empty_partitions: optional, boolean
        Deprecated. This parameter no longer has any effect and
        will be removed in release 26.10.
    """

    def __init__(
        self,
        *,
        workers=None,
        client=None,
        verbose=False,
        n_estimators=100,
        random_state=None,
        ignore_empty_partitions=None,
        **kwargs,
    ):
        super().__init__(client=client, verbose=verbose, **kwargs)

        self._create_model(
            model_func=RandomForestRegressor._construct_rf,
            client=client,
            workers=workers,
            n_estimators=n_estimators,
            base_seed=random_state,
            ignore_empty_partitions=ignore_empty_partitions,
            **kwargs,
        )

    @staticmethod
    def _construct_rf(n_estimators, random_state, **kwargs):
        return cuRFR(
            n_estimators=n_estimators, random_state=random_state, **kwargs
        )

    def fit(self, X, y, broadcast_data=None):
        """
        Fit the input data with a Random Forest regression model

        Only workers holding one or more training rows participate in fitting.

        When persisting data, you can use
        `cuml.dask.common.utils.persist_across_workers` to simplify this:

        .. code-block:: python

            X_dask_cudf = dask_cudf.from_cudf(X_cudf, npartitions=n_workers)
            y_dask_cudf = dask_cudf.from_cudf(y_cudf, npartitions=n_workers)
            X_dask_cudf, y_dask_cudf = persist_across_workers(dask_client,
                                                              [X_dask_cudf,
                                                               y_dask_cudf])

        This is equivalent to calling `persist` with the data and workers):

        .. code-block:: python

            X_dask_cudf, y_dask_cudf = dask_client.persist([X_dask_cudf,
                                                            y_dask_cudf],
                                                           workers={
                                                           X_dask_cudf:workers,
                                                           y_dask_cudf:workers
                                                           })

        Parameters
        ----------
        X : Dask cuDF DataFrame or CuPy backed Dask Array (n_rows, n_features)
            Distributed dense matrix (floats or doubles) of shape
            (n_samples, n_features).
        y : Dask cuDF DataFrame or CuPy backed Dask Array (n_rows, 1)
            Labels of training examples.
            **y must be partitioned the same way as X**
        broadcast_data : bool, optional
            Deprecated. This parameter no longer has effect and will
            be removed in release 26.10.
        """
        if broadcast_data is not None:
            warnings.warn(
                (
                    "broadcast_data parameter is no longer valid "
                    "and will be removed in release 26.10."
                ),
                FutureWarning,
            )
        self.internal_model = None
        self._fit(
            model=self.rfs,
            dataset=(X, y),
        )
        return self

    def predict(
        self,
        X,
        layout="depth_first",
        default_chunk_size=None,
        align_bytes=None,
        delayed=True,
        broadcast_data=None,
    ):
        """
        Predicts the regressor outputs for X.

        Parameters
        ----------
        X : Dask cuDF dataframe or CuPy backed Dask Array (n_rows, n_features)
            Distributed dense matrix (floats or doubles) of shape
            (n_samples, n_features).
        layout : string (default = 'depth_first')
            Specifies the in-memory layout of nodes in nvForest models. Options:
            'depth_first', 'layered', 'breadth_first'.
        default_chunk_size : int, optional (default = None)
            Determines how batches are further subdivided for parallel processing.
            The optimal value depends on hardware, model, and batch size.
            If None, will be automatically determined.
        align_bytes : int, optional (default = None)
            If specified, trees will be padded such that their in-memory size is
            a multiple of this value. This can improve performance by guaranteeing
            that memory reads from trees begin on a cache line boundary.
            Typical values are 0 or 128.
        delayed : bool (default = True)
            Whether to do a lazy prediction (and return Delayed objects) or an
            eagerly executed one.
        broadcast_data : bool, optional
            Deprecated. This parameter no longer has effect and will
            be removed in release 26.10.

        Returns
        -------
        y : Dask cuDF dataframe or CuPy backed Dask Array (n_rows, 1)
        """
        if broadcast_data is not None:
            warnings.warn(
                (
                    "broadcast_data parameter is no longer valid "
                    "and will be removed in release 26.10."
                ),
                FutureWarning,
            )
        return self._predict_using_nvforest(
            X,
            layout=layout,
            default_chunk_size=default_chunk_size,
            align_bytes=align_bytes,
            delayed=delayed,
        )

    def get_params(self, deep=True):
        """
        Returns the value of all parameters
        required to configure this estimator as a dictionary.

        Parameters
        ----------
        deep : boolean (default = True)
        """
        return self._get_params(deep)

    def set_params(self, **params):
        """
        Sets the value of parameters required to
        configure this estimator, it functions similar to
        the sklearn set_params.

        Parameters
        ----------
        params : dict of new params.
        """
        return self._set_params(**params)
