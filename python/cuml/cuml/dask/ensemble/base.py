# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import warnings

from dask.distributed import get_worker
from raft_dask.common.comms import Comms, get_raft_comm_state

from cuml.dask._compat import DASK_2025_4_0
from cuml.dask.common.base import mnmg_import
from cuml.dask.common.input_utils import DistributedDataHandler, concatenate
from cuml.dask.common.utils import get_client, wait_and_raise_from_futures


class BaseRandomForestModel(object):
    """
    BaseRandomForestModel defines functions used in both Random Forest
    Classifier and Regressor for Multi Node and Multi GPU models. The common
    functions are defined here and called from the main Random Forest Multi
    Node Multi GPU APIs. The functions defined here are not meant to be used
    as a part of the public API.
    """

    def _create_model(
        self,
        model_func,
        client,
        workers,
        n_estimators,
        base_seed,
        ignore_empty_partitions,
        **kwargs,
    ):
        self.client = get_client(client)
        if workers is None:
            # Default to all workers
            client_kwargs = {"n_workers": -1} if DASK_2025_4_0() else {}
            workers = list(
                self.client.scheduler_info(**client_kwargs)["workers"].keys()
            )
        self.workers = workers
        self._set_internal_model(None)
        self.n_estimators = n_estimators

        if "n_streams" in kwargs:
            warnings.warn(
                (
                    "n_streams has no effect on distributed training and "
                    "will be removed in release 26.12."
                ),
                FutureWarning,
                stacklevel=2,
            )
        if ignore_empty_partitions is not None:
            warnings.warn(
                (
                    "ignore_empty_partitions parameter is no longer valid "
                    "and will be removed in release 26.12."
                ),
                FutureWarning,
                stacklevel=2,
            )

        self.rfs = {
            worker: self.client.submit(
                model_func,
                n_estimators=self.n_estimators,
                random_state=base_seed,
                **kwargs,
                pure=False,
                workers=[worker],
            )
            for worker in self.workers
        }

        wait_and_raise_from_futures(list(self.rfs.values()))

    def _fit(self, model, dataset, classes=None):
        data = DistributedDataHandler.create(dataset, client=self.client)
        self.datatype = data.datatype

        unknown_workers = set(data.workers).difference(model)
        if unknown_workers:
            raise ValueError(
                "Training data was placed on workers that were not selected "
                f"for this estimator: {sorted(unknown_workers)}"
            )
        if not data.worker_to_parts:
            raise ValueError("No mapping found between workers and partitions")

        total_rows = sum(total for _, total in data._worker_sizes.values())
        comms = Comms(
            comms_p2p=False,
            client=self.client,
            streams_per_handle=1,
        )
        futures = []
        try:
            comms.init(workers=data.workers)
            for worker, worker_data in data.worker_to_parts.items():
                future = self.client.submit(
                    _func_fit,
                    comms.sessionId,
                    model[worker],
                    worker_data,
                    total_rows,
                    classes,
                    workers=[worker],
                    pure=False,
                )
                futures.append(future)
                self.rfs[worker] = future

            wait_and_raise_from_futures(futures)
        finally:
            comms.destroy()

        # Every distributed rank owns the same complete forest. Keep one
        # worker future as the canonical model for inference and serialization.
        self._set_internal_model(futures[0])
        return self

    def _predict_using_nvforest(self, X, delayed, **kwargs):
        data = DistributedDataHandler.create(X, client=self.client)
        return self._predict(
            X, delayed=delayed, output_collection_type=data.datatype, **kwargs
        )

    def _get_params(self, deep):
        model_params = list()
        for worker in self.workers:
            model_params.append(
                self.client.submit(
                    _func_get_params, self.rfs[worker], deep, workers=[worker]
                )
            )
        params_of_each_model = self.client.gather(model_params, errors="raise")
        return params_of_each_model

    def _set_params(self, **params):
        if "n_streams" in params:
            warnings.warn(
                (
                    "n_streams has no effect on distributed training and "
                    "will be removed in release 26.12."
                ),
                FutureWarning,
                stacklevel=2,
            )
        model_params = list()
        for worker in self.workers:
            model_params.append(
                self.client.submit(
                    _func_set_params,
                    self.rfs[worker],
                    **params,
                    workers=[worker],
                )
            )
        wait_and_raise_from_futures(model_params)
        return self


@mnmg_import
def _func_fit(session_id, model, input_data, total_rows, classes):
    handle = get_raft_comm_state(session_id, get_worker())["handle"]
    X = concatenate([item[0] for item in input_data])
    y = concatenate([item[1] for item in input_data])
    model._raft_handle = handle
    model._distributed_n_rows = total_rows
    if classes is not None:
        model._distributed_classes = classes
    try:
        return model.fit(X, y)
    finally:
        del model._raft_handle
        del model._distributed_n_rows
        if classes is not None:
            del model._distributed_classes


def _func_get_params(model, deep):
    return model.get_params(deep)


def _func_set_params(model, **params):
    return model.set_params(**params)
