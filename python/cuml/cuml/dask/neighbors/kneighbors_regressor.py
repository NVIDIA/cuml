#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from uuid import uuid1

import dask.array as da
from dask.distributed import get_worker
from raft_dask.common.comms import get_raft_comm_state

from cuml.dask.common import flatten_grouped_results, parts_to_ranks
from cuml.dask.common.input_utils import DistributedDataHandler, to_output
from cuml.dask.common.utils import (
    raise_mg_import_exception,
    wait_and_raise_from_futures,
)
from cuml.dask.neighbors.nearest_neighbors import NearestNeighbors


class KNeighborsRegressor(NearestNeighbors):
    """
    Multi-node Multi-GPU K-Nearest Neighbors Regressor Model.

    K-Nearest Neighbors Regressor is an instance-based learning technique,
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
        Fit a multi-node multi-GPU K-Nearest Neighbors Regressor index

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Index data.
            Acceptable formats: dask CuPy/NumPy/Numba Array

        y : array-like (device or host) shape = (n_samples, n_features)
            Index output data.
            Acceptable formats: dask CuPy/NumPy/Numba Array

        Returns
        -------
        self : KNeighborsRegressor model
        """
        self.data_handler = DistributedDataHandler.create(
            data=[X, y], client=self.client
        )
        self.n_outputs = y.shape[1] if y.ndim != 1 else 1

        return self

    @staticmethod
    def _func_create_model(sessionId, **kwargs):
        try:
            from cuml.neighbors.kneighbors_regressor_mg import (
                KNeighborsRegressorMG,
            )
        except ImportError:
            raise_mg_import_exception()

        handle = get_raft_comm_state(sessionId, get_worker())["handle"]
        return KNeighborsRegressorMG(handle=handle, **kwargs)

    @staticmethod
    def _func_predict(
        model,
        index,
        index_parts_to_ranks,
        index_nrows,
        query,
        query_parts_to_ranks,
        query_nrows,
        ncols,
        rank,
        n_output,
    ):
        return model.predict(
            index,
            index_parts_to_ranks,
            index_nrows,
            query,
            query_parts_to_ranks,
            query_nrows,
            ncols,
            rank,
            n_output,
        )

    def predict(self, X):
        """
        Predict outputs for a query from previously stored index
        and outputs.
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

        comms = KNeighborsRegressor._build_comms(
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
        knn_reg_res = dict(
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
                        X.shape[1],
                        self.n_outputs,
                        worker_info[worker]["rank"],
                        key="%s-%s" % (key, idx),
                        workers=[worker],
                    ),
                )
                for idx, worker in enumerate(comms.worker_addresses)
            ]
        )

        wait_and_raise_from_futures(list(knn_reg_res.values()))

        """
        Gather resulting partitions and return result
        """
        out_futures = flatten_grouped_results(
            self.client, query_parts_to_ranks, knn_reg_res
        )

        comms.destroy()

        return to_output(out_futures, self.datatype).squeeze()

    def score(self, X, y):
        """
        Provide score by comparing predictions and ground truth.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Query test data.
            Acceptable formats: dask CuPy/NumPy/Numba Array

        y : array-like (device or host) shape = (n_samples, n_features)
            Outputs test data.
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
        y_mean = y_true.mean(axis=0)
        residual_sss = ((y_true - y_pred) ** 2).sum(axis=0, dtype="float64")
        total_sss = ((y_true - y_mean) ** 2).sum(axis=0, dtype="float64")
        r2_score = da.mean(1 - (residual_sss / total_sss))
        return r2_score.compute()
