# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run distributed cuML benchmarks on a local multi-GPU cluster."""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..suite import Suite

from .base import InputRepresentation
from .cuml import CumlBackend, NativeCumlBackend


def _wait_for_futures(distributed: Any, value: Any) -> None:
    """Await partition futures and propagate task failures."""
    futures = distributed.futures_of(value)
    distributed.wait(futures)
    for future in futures:
        if future.status == "error":
            raise future.exception()
        if future.status == "cancelled":
            raise RuntimeError("Dask execution was cancelled")


class DaskBackend(CumlBackend):
    """Provide distributed inputs, estimators, and a multi-GPU runtime."""

    def convert_data(
        self,
        value: Any,
        *,
        representation: InputRepresentation,
    ) -> Any:
        """Convert one structural representation to distributed device storage."""
        if representation == "csr":
            raise ValueError("Dask CSR inputs are not supported yet")
        if representation in {"dataframe", "series"}:
            local = NativeCumlBackend.convert_data(
                self, value, representation=representation
            )
            return importlib.import_module("dask_cudf").from_cudf(
                local, npartitions=2
            )
        if representation != "array":
            raise ValueError(
                f"Unsupported input representation {representation!r}"
            )
        da = importlib.import_module("dask.array")
        cp = importlib.import_module("cupy")
        # TODO: Use the runtime worker count instead of fixed two-way chunking.
        chunks = (max(1, value.shape[0] // 2), *value.shape[1:])
        return da.from_array(value, chunks=chunks).map_blocks(
            cp.asarray, meta=cp.empty((0,) * value.ndim, dtype=value.dtype)
        )

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
        # Inspect partition futures after completion to surface conversion
        # failures while retaining input data on the workers.
        _wait_for_futures(distributed, persisted)
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
            Distributed output to persist on workers and await completion.
        """
        if isinstance(value, (tuple, list)):
            for item in value:
                self.synchronize(item)
            return
        if value is not None:
            # Persist lazy collections to execute their graphs on the workers,
            # then await the resulting partition futures.
            dask = importlib.import_module("dask")
            distributed = importlib.import_module("dask.distributed")
            if dask.is_dask_collection(value):
                value = distributed.get_client().persist(value)
            _wait_for_futures(distributed, value)
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
