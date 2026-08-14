# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from numbers import Integral

import cupy as cp
import dask.array as da
import dask.dataframe as dd
from dask.distributed import get_worker
from raft_dask.common.comms import Comms, get_raft_comm_state

from cuml.dask.common.base import (
    BaseEstimator,
    DelayedPredictionMixin,
    DelayedTransformMixin,
    mnmg_import,
)
from cuml.dask.common.input_utils import DistributedDataHandler
from cuml.dask.common.utils import wait_and_raise_from_futures
from cuml.internals.validation import check_random_seed


def _get_inertia_and_n_samples(estimator):
    return (estimator.inertia_, len(estimator.labels_))


def _validate_n_clusters(n_clusters):
    if not isinstance(n_clusters, Integral) or n_clusters <= 0:
        raise ValueError(
            f"n_clusters={n_clusters} should be a positive integer."
        )


class KMeans(BaseEstimator, DelayedPredictionMixin, DelayedTransformMixin):
    """
    Multi-Node Multi-GPU implementation of KMeans.

    This version minimizes data transfer by sharing only
    the centroids between workers in each iteration.

    Predictions are done embarrassingly parallel, using cuML's
    single-GPU version.

    For more information on this implementation, refer to the
    documentation for single-GPU K-Means.

    Parameters
    ----------
    n_clusters : int (default = 8)
        The number of centroids or clusters you want.
    max_iter : int (default = 300)
        The more iterations of EM, the more accurate, but slower.
    tol : float (default = 1e-4)
        Stopping criterion when centroid means do not change much.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    random_state : int or None (default = None)
        If you want results to be the same when you restart Python,
        select a state.
    init : {'scalable-kmeans++', 'k-means||' , 'random' or an ndarray} \
           (default = 'scalable-k-means++')
        'scalable-k-means++' or 'k-means||': Uses fast and stable scalable
        kmeans++ initialization.
        'random': Choose 'n_cluster' observations (rows) at random
        from data for the initial centroids. If an ndarray is passed,
        it should be of shape (n_clusters, n_features) and gives the
        initial centers.
    oversampling_factor : int (default = 2)
        The amount of points to sample in scalable k-means++ initialization for
        potential centroids. Increasing this value can lead to better initial
        centroids at the cost of memory. The total number of centroids sampled
        in scalable k-means++ is oversampling_factor * n_clusters * 8.
    max_samples_per_batch : int (default = 32768)
        The number of data samples to use for batches of the pairwise distance
        computation. This computation is done throughout both fit predict.
        The default should suit most cases. The total number of elements in the
        batched pairwise distance computation is max_samples_per_batch
        * n_clusters. It might become necessary to lower this number when
        n_clusters becomes prohibitively large.

    Attributes
    ----------
    cluster_centers_ : cuDF DataFrame or CuPy ndarray
        The coordinates of the final clusters. This represents of "mean" of
        each data cluster.

    """

    def __init__(self, *, client=None, verbose=False, n_clusters=8, **kwargs):
        super().__init__(
            client=client, verbose=verbose, n_clusters=n_clusters, **kwargs
        )

    @staticmethod
    @mnmg_import
    def _func_preflight_fit(sessionId, objs, datatype, has_weights, **kwargs):
        from cuml.cluster.kmeans_mg import KMeansMG

        comm_state = get_raft_comm_state(sessionId, get_worker())
        X = objs if not has_weights else [X for X, _ in objs]

        model = KMeansMG(
            handle=comm_state["handle"],
            output_type=datatype,
            **kwargs,
        )
        model._validate_fit_params()
        model.validate(X, comm_state["wid"], comm_state["nworkers"])

    @staticmethod
    @mnmg_import
    def _func_fit(sessionId, objs, datatype, has_weights, **kwargs):
        from cuml.cluster.kmeans_mg import KMeansMG

        handle = get_raft_comm_state(sessionId, get_worker())["handle"]

        if not has_weights:
            inp_data = list(objs)
            inp_weights = None
        else:
            inp_data = [X for X, weights in objs]
            inp_weights = [weights for X, weights in objs]

        return KMeansMG(handle=handle, output_type=datatype, **kwargs).fit(
            inp_data, sample_weight=inp_weights
        )

    @staticmethod
    def _score(model, data, sample_weight=None):
        ret = model.score(data, sample_weight=sample_weight)
        return ret

    @staticmethod
    def _check_normalize_sample_weight(sample_weight):
        if sample_weight is not None:
            n_samples = len(sample_weight)
            scale = n_samples / sample_weight.sum()
            sample_weight *= scale
        return sample_weight

    def fit(self, X, sample_weight=None):
        """
        Fit a multi-node multi-GPU KMeans model

        Parameters
        ----------
        X : Dask cuDF DataFrame or CuPy backed Dask Array
        Training data to cluster.

        sample_weight : Dask cuDF DataFrame or CuPy backed Dask Array \
                shape = (n_samples,), default=None # noqa

            The weights for each observation in X. If None, all observations
            are assigned equal weight.
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        """

        sample_weight = self._check_normalize_sample_weight(sample_weight)

        inputs = X if sample_weight is None else (X, sample_weight)

        data = DistributedDataHandler.create(inputs, client=self.client)
        self.datatype = data.datatype

        # Ensure a consistent `random_state` across all calls
        kwargs = self.kwargs.copy()
        kwargs["random_state"] = check_random_seed(kwargs.get("random_state"))

        # Validate predictable global failures before submitting distributed
        # work. MG-specific parameter and rank-local row checks are performed
        # in the all-worker preflight after the communicator is initialized.
        n_clusters = kwargs["n_clusters"]
        _validate_n_clusters(n_clusters)
        data._fetch_worker_sizes()
        total_rows = sum(total for (_, total) in data._worker_sizes.values())
        if total_rows < n_clusters:
            raise ValueError(
                f"n_samples={total_rows} should be >= n_clusters={n_clusters}. "
                f"There are fewer data points across all workers than the "
                f"number of requested clusters. Please reduce n_clusters or "
                f"increase the number of data points."
            )

        # This needs to happen on the scheduler
        comms = Comms(comms_p2p=False, client=self.client)
        comms.init(workers=data.workers)

        kmeans_fit = []
        try:
            preflight_futures = [
                self.client.submit(
                    KMeans._func_preflight_fit,
                    comms.sessionId,
                    parts,
                    self.datatype,
                    data.multiple,
                    **kwargs,
                    workers=[worker],
                    pure=False,
                )
                for worker, parts in data.worker_to_parts.items()
            ]
            self.client.gather(preflight_futures)

            kmeans_fit = [
                self.client.submit(
                    KMeans._func_fit,
                    comms.sessionId,
                    wf[1],
                    self.datatype,
                    data.multiple,
                    **kwargs,
                    workers=[wf[0]],
                    pure=False,
                )
                for idx, wf in enumerate(data.worker_to_parts.items())
            ]

            wait_and_raise_from_futures(kmeans_fit)
        finally:
            comms.destroy()

        # Collect the full model from only the first worker (for
        # cluster_centers_ etc). Since cluster centers are synchronized
        # via NCCL, all workers have identical copies — pulling more
        # than one would waste memory (N * n_clusters * n_features * 4B).
        first = kmeans_fit[0].result()
        self._set_internal_model(first)

        workers = list(data.worker_to_parts.keys())

        # Compute and store the total inertia_
        inertia_and_lengths = self.client.gather(
            [
                self.client.submit(_get_inertia_and_n_samples, f, workers=[w])
                for f, w in zip(kmeans_fit, workers)
            ]
        )
        self.inertia_ = sum(inertia for inertia, _ in inertia_and_lengths)

        # Store labels_ as a distributed dask array. This attribute scales with
        # n_samples, and shouldn't be pulled back to a local node.
        labels_meta = cp.zeros(0, dtype=first.labels_.dtype)
        labels = [
            self.client.submit(getattr, f, "labels_", workers=[w])
            for f, w in zip(kmeans_fit, workers)
        ]
        if self.datatype == "cudf":
            self.labels_ = dd.from_delayed(labels)
        else:
            self.labels_ = da.concatenate(
                [
                    da.from_delayed(f, shape=(length,), meta=labels_meta)
                    for f, (_, length) in zip(labels, inertia_and_lengths)
                ]
            )

        return self

    def fit_predict(self, X, sample_weight=None, delayed=True):
        """
        Compute cluster centers and predict cluster index for each sample.

        Parameters
        ----------
        X : Dask cuDF DataFrame or CuPy backed Dask Array
            Data to predict

        Returns
        -------
        result: Dask cuDF DataFrame or CuPy backed Dask Array
            Distributed object containing predictions

        """
        self.fit(X, sample_weight=sample_weight)
        return self.labels_ if delayed else self.labels_.persist()

    def predict(self, X, delayed=True):
        """
        Predict labels for the input

        Parameters
        ----------
        X : Dask cuDF DataFrame or CuPy backed Dask Array
            Data to predict

        delayed : bool (default = True)
            Whether to do a lazy prediction (and return Delayed objects) or an
            eagerly executed one.

        Returns
        -------
        result: Dask cuDF DataFrame or CuPy backed Dask Array
            Distributed object containing predictions
        """
        return self._predict(X, delayed=delayed)

    def fit_transform(self, X, sample_weight=None, delayed=True):
        """
        Calls fit followed by transform using a distributed KMeans model

        Parameters
        ----------
        X : Dask cuDF DataFrame or CuPy backed Dask Array
            Data to predict

        delayed : bool (default = True)
            Whether to execute as a delayed task or eager.

        Returns
        -------
        result: Dask cuDF DataFrame or CuPy backed Dask Array
            Distributed object containing the transformed data
        """
        return self.fit(X, sample_weight=sample_weight).transform(
            X, delayed=delayed
        )

    def transform(self, X, delayed=True):
        """
        Transforms the input into the learned centroid space

        Parameters
        ----------
        X : Dask cuDF DataFrame or CuPy backed Dask Array
            Data to predict

        delayed : bool (default = True)
            Whether to execute as a delayed task or eager.

        Returns
        -------
        result: Dask cuDF DataFrame or CuPy backed Dask Array
            Distributed object containing the transformed data
        """
        return self._transform(X, n_dims=2, delayed=delayed)

    def score(self, X, sample_weight=None):
        """
        Computes the inertia score for the trained KMeans centroids.

        Parameters
        ----------
        X : dask_cudf.Dataframe
            Dataframe to compute score

        Returns
        -------

        Inertial score
        """

        sample_weight = self._check_normalize_sample_weight(sample_weight)

        scores = self._run_parallel_func(
            KMeans._score,
            X,
            sample_weight=sample_weight,
            n_dims=1,
            delayed=False,
            output_futures=True,
        )
        return -1.0 * sum(
            -1.0 * score for score in self.client.compute(scores, sync=True)
        )

    def _get_param_names(self):
        return list(self.kwargs.keys())
