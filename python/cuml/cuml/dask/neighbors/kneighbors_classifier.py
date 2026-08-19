#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from uuid import uuid1

import cupy as cp
import dask.array as da
import numpy as np
from dask.dataframe import Series as DaskSeries
from dask.distributed import get_worker
from raft_dask.common.comms import get_raft_comm_state

from cuml.dask.common import flatten_grouped_results, parts_to_ranks
from cuml.dask.common.input_utils import DistributedDataHandler, to_output
from cuml.dask.common.utils import (
    raise_mg_import_exception,
    wait_and_raise_from_futures,
)
from cuml.dask.neighbors.nearest_neighbors import NearestNeighbors


class KNeighborsClassifier(NearestNeighbors):
    """
    Multi-node Multi-GPU K-Nearest Neighbors Classifier Model.

    K-Nearest Neighbors Classifier is an instance-based learning technique,
    that keeps training samples around for prediction, rather than trying
    to learn a generalizable set of model parameters.

    Parameters
    ----------
    n_neighbors : int (default=5)
        Default number of neighbors to query
    batch_size: int (optional, default 2000000)
        Maximum number of query rows processed at once. This parameter can
        greatly affect the throughput of the algorithm. The optimal setting
        of this value will vary for different layouts and index to query
        ratios, but it will require `batch_size * n_features * 4` bytes of
        additional memory on each worker hosting index partitions.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    """

    def __init__(
        self, *, client=None, streams_per_handle=0, verbose=False, **kwargs
    ):
        super().__init__(client=client, verbose=verbose, **kwargs)
        self.streams_per_handle = streams_per_handle

    def fit(self, X, y):
        """
        Fit a multi-node multi-GPU K-Nearest Neighbors Classifier index

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Index data.
            Acceptable formats: dask CuPy/NumPy/Numba Array

        y : array-like (device or host) shape = (n_samples, n_features)
            Index labels data.
            Acceptable formats: dask CuPy/NumPy/Numba Array

        Returns
        -------
        self : KNeighborsClassifier model
        """

        self.data_handler = DistributedDataHandler.create(
            data=[X, y], client=self.client
        )

        # uniq_labels: set of possible labels for each labels column
        # n_unique: number of possible labels for each labels column

        uniq_labels = []
        if self.data_handler.datatype == "cupy":
            if y.ndim == 1:
                uniq_labels.append(da.unique(y))
            else:
                n_targets = y.shape[1]
                for i in range(n_targets):
                    uniq_labels.append(da.unique(y[:, i]))
        else:
            if isinstance(y, DaskSeries):
                uniq_labels.append(y.unique())
            else:
                # Dask-expr does not support numerical column names
                # See: https://github.com/dask/dask-expr/issues/1015
                _y = y
                n_targets = len(_y.columns)
                for i in range(n_targets):
                    uniq_labels.append(_y.iloc[:, i].unique())

        uniq_labels = da.compute(uniq_labels)[0]
        if hasattr(uniq_labels[0], "to_numpy"):  # for pandas and cuDF Series
            uniq_labels = list(map(lambda x: x.to_numpy(), uniq_labels))
        elif isinstance(uniq_labels[0], cp.ndarray):  # for CuPy arrays
            uniq_labels = list(map(lambda x: cp.asnumpy(x), uniq_labels))
        self.uniq_labels = np.sort(np.array(uniq_labels))
        self.n_unique = list(map(lambda x: len(x), self.uniq_labels))

        return self

    @staticmethod
    def _func_create_model(sessionId, **kwargs):
        try:
            from cuml.neighbors.kneighbors_classifier_mg import (
                KNeighborsClassifierMG,
            )
        except ImportError:
            raise_mg_import_exception()

        handle = get_raft_comm_state(sessionId, get_worker())["handle"]
        return KNeighborsClassifierMG(handle=handle, **kwargs)

    @staticmethod
    def _func_predict(
        model,
        index,
        index_parts_to_ranks,
        index_nrows,
        query,
        query_parts_to_ranks,
        query_nrows,
        uniq_labels,
        n_unique,
        ncols,
        rank,
        probas_only,
    ):
        if probas_only:
            return model.predict_proba(
                index,
                index_parts_to_ranks,
                index_nrows,
                query,
                query_parts_to_ranks,
                query_nrows,
                uniq_labels,
                n_unique,
                ncols,
                rank,
            )
        else:
            return model.predict(
                index,
                index_parts_to_ranks,
                index_nrows,
                query,
                query_parts_to_ranks,
                query_nrows,
                uniq_labels,
                n_unique,
                ncols,
                rank,
            )

    def predict(self, X):
        """
        Predict labels for a query from previously stored index
        and index labels.
        The process is done in a multi-node multi-GPU fashion.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Query data.
            Acceptable formats: dask cuDF, dask CuPy/NumPy/Numba Array

        Returns
        -------
        predictions : Dask futures or Dask CuPy Arrays
        """
        query_handler = DistributedDataHandler.create(
            data=X, client=self.client
        )
        self.datatype = query_handler.datatype

        comms = KNeighborsClassifier._build_comms(
            self.data_handler, query_handler, self.streams_per_handle
        )

        worker_info = comms.worker_info(comms.worker_addresses)

        """
        Build inputs and outputs
        """
        self.data_handler.calculate_parts_to_sizes(comms=comms)
        query_handler.calculate_parts_to_sizes(comms=comms)

        data_parts_to_ranks, data_nrows = parts_to_ranks(
            self.client, worker_info, self.data_handler.gpu_futures
        )

        query_parts_to_ranks, query_nrows = parts_to_ranks(
            self.client, worker_info, query_handler.gpu_futures
        )

        """
        Each Dask worker creates a single model
        """
        key = uuid1()
        models = dict(
            [
                (
                    worker,
                    self.client.submit(
                        self._func_create_model,
                        comms.sessionId,
                        **self.kwargs,
                        workers=[worker],
                        key="%s-%s" % (key, idx),
                    ),
                )
                for idx, worker in enumerate(comms.worker_addresses)
            ]
        )

        """
        Invoke knn_classify on Dask workers to perform distributed query
        """
        key = uuid1()
        knn_clf_res = dict(
            [
                (
                    worker_info[worker]["rank"],
                    self.client.submit(
                        self._func_predict,
                        models[worker],
                        self.data_handler.worker_to_parts[worker]
                        if worker in self.data_handler.workers
                        else [],
                        data_parts_to_ranks,
                        data_nrows,
                        query_handler.worker_to_parts[worker]
                        if worker in query_handler.workers
                        else [],
                        query_parts_to_ranks,
                        query_nrows,
                        self.uniq_labels,
                        self.n_unique,
                        X.shape[1],
                        worker_info[worker]["rank"],
                        False,
                        key="%s-%s" % (key, idx),
                        workers=[worker],
                    ),
                )
                for idx, worker in enumerate(comms.worker_addresses)
            ]
        )

        wait_and_raise_from_futures(list(knn_clf_res.values()))

        """
        Gather resulting partitions and return result
        """
        out_futures = flatten_grouped_results(
            self.client, query_parts_to_ranks, knn_clf_res
        )
        comms.destroy()

        return to_output(out_futures, self.datatype).squeeze()

    def score(self, X, y):
        """
        Predict labels for a query from previously stored index
        and index labels.
        The process is done in a multi-node multi-GPU fashion.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Query test data.
            Acceptable formats: dask CuPy/NumPy/Numba Array

        y : array-like (device or host) shape = (n_samples, n_features)
            Labels test data.
            Acceptable formats: dask CuPy/NumPy/Numba Array

        Returns
        -------
        score
        """
        y_pred_plain = self.predict(X)
        if not isinstance(y_pred_plain, da.Array):
            y_pred = y_pred_plain.to_dask_array(lengths=True)
        else:
            y_pred = y_pred_plain

        if not isinstance(y, da.Array):
            y = y.to_dask_array(lengths=True)
        y_true = y.squeeze()
        matched = y_pred == y_true
        mean_match = matched.mean()
        return float(mean_match.compute())

    def predict_proba(self, X):
        """
        Provide score by comparing predictions and ground truth.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Query data.
            Acceptable formats: dask cuDF, dask CuPy/NumPy/Numba Array

        Returns
        -------
        probabilities : Dask futures or Dask CuPy Arrays
        """
        query_handler = DistributedDataHandler.create(
            data=X, client=self.client
        )
        self.datatype = query_handler.datatype

        comms = KNeighborsClassifier._build_comms(
            self.data_handler, query_handler, self.streams_per_handle
        )

        worker_info = comms.worker_info(comms.worker_addresses)

        """
        Build inputs and outputs
        """
        self.data_handler.calculate_parts_to_sizes(comms=comms)
        query_handler.calculate_parts_to_sizes(comms=comms)

        data_parts_to_ranks, data_nrows = parts_to_ranks(
            self.client, worker_info, self.data_handler.gpu_futures
        )

        query_parts_to_ranks, query_nrows = parts_to_ranks(
            self.client, worker_info, query_handler.gpu_futures
        )

        """
        Each Dask worker creates a single model
        """
        key = uuid1()
        models = dict(
            [
                (
                    worker,
                    self.client.submit(
                        self._func_create_model,
                        comms.sessionId,
                        **self.kwargs,
                        workers=[worker],
                        key="%s-%s" % (key, idx),
                    ),
                )
                for idx, worker in enumerate(comms.worker_addresses)
            ]
        )

        """
        Invoke knn_classify on Dask workers to perform distributed query
        """
        key = uuid1()
        knn_prob_res = dict(
            [
                (
                    worker_info[worker]["rank"],
                    self.client.submit(
                        self._func_predict,
                        models[worker],
                        self.data_handler.worker_to_parts[worker]
                        if worker in self.data_handler.workers
                        else [],
                        data_parts_to_ranks,
                        data_nrows,
                        query_handler.worker_to_parts[worker]
                        if worker in query_handler.workers
                        else [],
                        query_parts_to_ranks,
                        query_nrows,
                        self.uniq_labels,
                        self.n_unique,
                        X.shape[1],
                        worker_info[worker]["rank"],
                        True,
                        key="%s-%s" % (key, idx),
                        workers=[worker],
                    ),
                )
                for idx, worker in enumerate(comms.worker_addresses)
            ]
        )

        wait_and_raise_from_futures(list(knn_prob_res.values()))

        n_outputs = len(self.n_unique)

        def _custom_getter(o):
            def func_get(f, idx):
                return f[o][idx]

            return func_get

        """
        Gather resulting partitions and return result
        """
        outputs = []
        for o in range(n_outputs):
            futures = flatten_grouped_results(
                self.client,
                query_parts_to_ranks,
                knn_prob_res,
                getter_func=_custom_getter(o),
            )
            outputs.append(to_output(futures, self.datatype))

        comms.destroy()

        if n_outputs == 1:
            return da.concatenate(outputs, axis=0)
        return tuple(outputs)
