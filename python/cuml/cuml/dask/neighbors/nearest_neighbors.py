# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from uuid import uuid1

from dask.distributed import get_worker
from raft_dask.common.comms import Comms, get_raft_comm_state

from cuml.dask.common import (
    flatten_grouped_results,
    parts_to_ranks,
    raise_mg_import_exception,
)
from cuml.dask.common.base import BaseEstimator
from cuml.dask.common.input_utils import DistributedDataHandler, to_output
from cuml.dask.common.utils import wait_and_raise_from_futures


class NearestNeighbors(BaseEstimator):
    """
    Multi-node Multi-GPU NearestNeighbors Model.

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

    def __init__(self, *, client=None, streams_per_handle=0, **kwargs):
        super().__init__(client=client, **kwargs)
        self.streams_per_handle = streams_per_handle

    def fit(self, X):
        """
        Fit a multi-node multi-GPU Nearest Neighbors index

        Parameters
        ----------
        X : dask_cudf.Dataframe

        Returns
        -------
        self: NearestNeighbors model
        """
        self.X_handler = DistributedDataHandler.create(
            data=X, client=self.client
        )
        self.datatype = self.X_handler.datatype
        self.n_cols = X.shape[1]

        # Brute force nearest neighbors does not set an internal model so
        # calls to get_combined_model() will just return None.
        # Approximate methods that build specialized indices, such as the
        # FAISS product quantized methods, will be combined into an internal
        # model.

        return self

    @staticmethod
    def _func_create_model(sessionId, **kwargs):
        try:
            from cuml.neighbors.nearest_neighbors_mg import NearestNeighborsMG
        except ImportError:
            raise_mg_import_exception()

        handle = get_raft_comm_state(sessionId, get_worker())["handle"]
        return NearestNeighborsMG(handle=handle, **kwargs)

    @staticmethod
    def _func_kneighbors(
        model,
        index,
        index_parts_to_ranks,
        index_nrows,
        query,
        query_parts_to_ranks,
        query_nrows,
        ncols,
        rank,
        n_neighbors,
    ):
        return model.kneighbors(
            index,
            index_parts_to_ranks,
            index_nrows,
            query,
            query_parts_to_ranks,
            query_nrows,
            ncols,
            rank,
            n_neighbors,
        )

    @staticmethod
    def _build_comms(index_handler, query_handler, streams_per_handle):
        workers = set(index_handler.workers)
        workers.update(query_handler.workers)

        comms = Comms(comms_p2p=True, streams_per_handle=streams_per_handle)
        comms.init(workers=workers)
        return comms

    def get_neighbors(self, n_neighbors):
        """
        Returns the default n_neighbors, initialized from the constructor,
        if n_neighbors is None.

        Parameters
        ----------
        n_neighbors : int
            Number of neighbors

        Returns
        -------
        n_neighbors: int
            Default n_neighbors if parameter n_neighbors is none
        """
        if n_neighbors is None:
            if (
                "n_neighbors" in self.kwargs
                and self.kwargs["n_neighbors"] is not None
            ):
                n_neighbors = self.kwargs["n_neighbors"]
            else:
                from cuml.neighbors import NearestNeighbors as cumlNN

                n_neighbors = cumlNN().n_neighbors

        return n_neighbors

    def _create_models(self, comms):
        """
        Each Dask worker creates a single model
        """
        key = uuid1()
        nn_models = dict(
            [
                (
                    worker,
                    self.client.submit(
                        NearestNeighbors._func_create_model,
                        comms.sessionId,
                        **self.kwargs,
                        workers=[worker],
                        key="%s-%s" % (key, idx),
                    ),
                )
                for idx, worker in enumerate(comms.worker_addresses)
            ]
        )

        return nn_models

    def _query_models(
        self, n_neighbors, comms, nn_models, index_handler, query_handler
    ):
        worker_info = comms.worker_info(comms.worker_addresses)

        """
        Build inputs and outputs
        """
        index_handler.calculate_parts_to_sizes(comms=comms)
        query_handler.calculate_parts_to_sizes(comms=comms)

        idx_parts_to_ranks, _ = parts_to_ranks(
            self.client, worker_info, index_handler.gpu_futures
        )

        query_parts_to_ranks, _ = parts_to_ranks(
            self.client, worker_info, query_handler.gpu_futures
        )

        """
        Invoke kneighbors on Dask workers to perform distributed query
        """
        key = uuid1()
        nn_fit = dict(
            [
                (
                    worker_info[worker]["rank"],
                    self.client.submit(
                        NearestNeighbors._func_kneighbors,
                        nn_models[worker],
                        index_handler.worker_to_parts[worker]
                        if worker in index_handler.workers
                        else [],
                        idx_parts_to_ranks,
                        index_handler.total_rows,
                        query_handler.worker_to_parts[worker]
                        if worker in query_handler.workers
                        else [],
                        query_parts_to_ranks,
                        query_handler.total_rows,
                        self.n_cols,
                        worker_info[worker]["rank"],
                        n_neighbors,
                        key="%s-%s" % (key, idx),
                        workers=[worker],
                    ),
                )
                for idx, worker in enumerate(comms.worker_addresses)
            ]
        )

        wait_and_raise_from_futures(list(nn_fit.values()))

        def _custom_getter(o):
            def func_get(f, idx):
                return f[o][idx]

            return func_get

        """
        Gather resulting partitions and return dask_cudfs
        """
        out_d_futures = flatten_grouped_results(
            self.client,
            query_parts_to_ranks,
            nn_fit,
            getter_func=_custom_getter(0),
        )

        out_i_futures = flatten_grouped_results(
            self.client,
            query_parts_to_ranks,
            nn_fit,
            getter_func=_custom_getter(1),
        )

        return nn_fit, out_d_futures, out_i_futures

    def kneighbors(
        self,
        X=None,
        n_neighbors=None,
        return_distance=True,
        _return_futures=False,
    ):
        """
        Query the distributed nearest neighbors index

        Parameters
        ----------
        X : dask_cudf.Dataframe
            Vectors to query. If not provided, neighbors of each indexed point
            are returned.
        n_neighbors : int
            Number of neighbors to query for each row in X. If not provided,
            the n_neighbors on the model are used.
        return_distance : boolean (default=True)
            If false, only indices are returned

        Returns
        -------
        ret : tuple (dask_cudf.DataFrame, dask_cudf.DataFrame)
            First dask-cuDF DataFrame contains distances, second contains the
            indices.
        """
        n_neighbors = self.get_neighbors(n_neighbors)

        query_handler = (
            self.X_handler
            if X is None
            else DistributedDataHandler.create(data=X, client=self.client)
        )

        if query_handler is None:
            raise ValueError(
                "Model needs to be trained using fit() "
                "before calling kneighbors()"
            )

        """
        Create communicator clique
        """
        comms = NearestNeighbors._build_comms(
            self.X_handler, query_handler, self.streams_per_handle
        )

        """
        Initialize models on workers
        """
        nn_models = self._create_models(comms)

        """
        Perform model query
        """
        nn_fit, out_d_futures, out_i_futures = self._query_models(
            n_neighbors, comms, nn_models, self.X_handler, query_handler
        )

        comms.destroy()

        if _return_futures:
            ret = (
                nn_fit,
                out_i_futures
                if not return_distance
                else (
                    nn_fit,
                    out_d_futures,
                    out_i_futures,
                ),
            )
        else:
            ret = (
                to_output(out_i_futures, self.datatype)
                if not return_distance
                else (
                    to_output(out_d_futures, self.datatype),
                    to_output(out_i_futures, self.datatype),
                )
            )

        return ret
