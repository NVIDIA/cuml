# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run distributed cuML benchmarks on a local multi-GPU cluster."""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..suite import ResolvedCase, Suite

from .cuml import CumlBackend


class DaskBackend(CumlBackend):
    """Provide distributed inputs, estimators, and a multi-GPU runtime."""

    def convert_data(
        self, case: ResolvedCase, X: Any, y: Any
    ) -> tuple[Any, Any]:
        """Convert generated inputs to partitioned Dask collections.

        Parameters
        ----------
        case : ResolvedCase
            Case defining the dataset shape and representation.
        X : Any
            Complete generated feature data.
        y : Any
            Complete generated target data.
        """
        # Distributed DBSCAN broadcasts a complete local array itself; it
        # does not accept partitioned Dask collections.
        if case.estimator == "DBSCAN":
            return X, y
        da = importlib.import_module("dask.array")
        cp = importlib.import_module("cupy")
        # TODO: Split training and inference data before backend conversion,
        # then partition and distribute each using the runtime worker count.
        # Fixed two-way chunking leaves every built-in inference input in one
        # partition, so partition-parallel predictions use only one worker.
        # Training also underutilizes clusters with more than two workers.
        chunks = (max(1, case.generated_rows // 2), case.features)
        if case.dataset == "categorical":
            cudf = importlib.import_module("cudf")
            dask_cudf = importlib.import_module("dask_cudf")
            X = dask_cudf.from_cudf(cudf.from_pandas(X), npartitions=2)
        else:
            X = da.from_array(X, chunks=chunks).map_blocks(
                cp.asarray, meta=cp.empty((0, 0), dtype=X.dtype)
            )
        if case.estimator == "LabelEncoder":
            cudf = importlib.import_module("cudf")
            dask_cudf = importlib.import_module("dask_cudf")
            y = dask_cudf.from_cudf(cudf.Series(y), npartitions=2)
        else:
            y = da.from_array(y, chunks=(chunks[0],)).map_blocks(
                cp.asarray, meta=cp.empty((0,), dtype=y.dtype)
            )
        return X, y

    def prepare_inputs(
        self, inputs: tuple[Any, ...], runtime: Any = None
    ) -> tuple[Any, ...]:
        """Persist selected distributed inputs outside timed execution.

        Parameters
        ----------
        inputs : tuple
            Operation and fit inputs, already partitioned and selected.
        runtime : Any, optional
            Dask client owning the persisted input futures.
        """
        dask = importlib.import_module("dask")
        indices = [
            index
            for index, value in enumerate(inputs)
            if dask.is_dask_collection(value)
        ]
        if not indices:
            # Distributed DBSCAN accepts local arrays, not Dask collections.
            return inputs
        distributed = importlib.import_module("dask.distributed")
        client = runtime if runtime is not None else distributed.get_client()
        persisted = client.persist([inputs[index] for index in indices])
        # wait() alone does not raise failed-task exceptions. Resolve partition
        # futures (not their arrays) to surface conversion failures without
        # gathering the complete inputs onto the client.
        futures = distributed.futures_of(persisted)
        distributed.wait(futures)
        for future in futures:
            if future.status == "error":
                raise future.exception()
            if future.status == "cancelled":
                raise RuntimeError("Dask input preparation was cancelled")
        prepared = list(inputs)
        for index, value in zip(indices, persisted, strict=True):
            prepared[index] = value
        return tuple(prepared)

    def construct_estimator(
        self,
        estimator_class: type[Any],
        parameters: dict[str, Any],
        runtime: Any = None,
    ) -> Any:
        """Construct a distributed estimator using the runtime client.

        Parameters
        ----------
        estimator_class : type
            Distributed estimator class to instantiate.
        parameters : dict
            Estimator constructor arguments.
        runtime : Any, optional
            Dask client passed to the estimator.
        """
        return estimator_class(**{**parameters, "client": runtime})

    def effective_parameters(self, estimator: Any) -> dict[str, Any]:
        """Normalize distributed estimator parameters for result metadata.

        Parameters
        ----------
        estimator : Any
            Constructed distributed estimator exposing get_params.
        """
        parameters = super().effective_parameters(estimator)
        # Dask random forests return a list of per-worker parameter mappings,
        # unlike the dictionary expected by the estimator API and result schema.
        # The list is a legacy of training separate forests with worker-specific
        # tree counts and seeds. Workers now receive identical parameters, so
        # collapse the list only after checking that assumption.
        # TODO: Fix Dask RF get_params(deep=False) upstream to return a single
        # estimator-level dictionary, then remove this compatibility hack.
        if isinstance(parameters, list):
            assert parameters, "Dask estimator returned no worker parameters"
            assert all(isinstance(item, dict) for item in parameters), (
                "Dask worker parameters must be dictionaries"
            )
            assert all(item == parameters[0] for item in parameters[1:]), (
                "Dask estimator parameters differ between workers"
            )
            return parameters[0]
        return parameters

    def synchronize(self, value: Any = None) -> None:
        """Wait for distributed output and local GPU work to complete.

        Parameters
        ----------
        value : Any, optional
            Distributed output to wait for or compute.
        """
        if isinstance(value, (tuple, list)):
            for item in value:
                self.synchronize(item)
            return
        if value is not None:
            # wait() only waits for existing futures; it does not execute a
            # lazy collection's graph, and can return without doing any work.
            if hasattr(value, "compute"):
                value.compute()
            else:
                importlib.import_module("dask.distributed").wait(value)
        super().synchronize(value)

    @contextlib.contextmanager
    def runtime(self, suite: Suite) -> Iterator[Any]:
        """Create a local multi-GPU cluster and yield its Dask client.

        Parameters
        ----------
        suite : Suite
            Suite to validate for distributed execution.
        """
        from ..suite import SuiteError

        if any(case.input_format == "csr" for case in suite.cases):
            raise SuiteError("Dask CSR inputs are not supported yet")
        if any(case.timeout_sec is not None for case in suite.cases):
            raise SuiteError("Dask case timeouts are not supported yet")

        count = importlib.import_module("cupy").cuda.runtime.getDeviceCount()
        if count < 2:
            raise SuiteError(
                f"cuml.dask suite requires at least two visible GPUs; found {count}"
            )
        LocalCUDACluster = importlib.import_module(
            "dask_cuda"
        ).LocalCUDACluster
        Client = importlib.import_module("dask.distributed").Client
        with LocalCUDACluster(
            n_workers=count, threads_per_worker=1
        ) as cluster:
            with Client(cluster) as client:
                yield client
