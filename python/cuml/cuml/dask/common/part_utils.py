# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from collections import OrderedDict
from collections.abc import Sequence
from functools import reduce

import numpy as np
from dask.array.core import Array as daskArray
from dask.dataframe import DataFrame as daskDataFrame
from dask.dataframe import Series as daskSeries
from dask.distributed import default_client, futures_of, wait
from dask_cudf import DataFrame as dcDataFrame
from dask_cudf import Series as dcSeries
from toolz import first
from tornado import gen

from cuml.dask.common.utils import parse_host_port


def hosts_to_parts(futures):
    """
    Builds an ordered dict mapping each host to their list
    of parts
    :param futures: list of (worker, part) tuples
    :return:
    """
    w_to_p_map = OrderedDict()
    for w, p in futures:
        host, port = parse_host_port(w)
        host_key = (host, port)
        if host_key not in w_to_p_map:
            w_to_p_map[host_key] = []
        w_to_p_map[host_key].append(p)
    return w_to_p_map


def workers_to_parts(futures):
    """
    Builds an ordered dict mapping each worker to their list
    of parts
    :param futures: list of (worker, part) tuples
    :return:
    """
    w_to_p_map = OrderedDict()
    for w, p in futures:
        if w not in w_to_p_map:
            w_to_p_map[w] = []
        w_to_p_map[w].append(p)
    return w_to_p_map


def _func_get_rows(df):
    if isinstance(df, tuple):
        return df[0].shape[0]
    else:
        return df.shape[0]


def parts_to_ranks(client, worker_info, part_futures):
    """
    Builds a list of (rank, size) tuples of partitions
    :param worker_info: dict of {worker, {"rank": rank }}. Note: \
        This usually comes from the underlying communicator
    :param part_futures: list of (worker, future) tuples
    :return: [(part, size)] in the same order of part_futures
    """
    futures = [
        (
            worker_info[wf[0]]["rank"],
            client.submit(_func_get_rows, wf[1], workers=[wf[0]], pure=False),
        )
        for idx, wf in enumerate(part_futures)
    ]

    sizes = client.compute(list(map(lambda x: x[1], futures)), sync=True)
    total = reduce(lambda a, b: a + b, sizes)

    return [(futures[idx][0], size) for idx, size in enumerate(sizes)], total


def _default_part_getter(f, idx):
    return f[idx]


def flatten_grouped_results(
    client, gpu_futures, worker_results_map, getter_func=_default_part_getter
):
    """
    This function is useful when a series of partitions have been grouped by
    the worker responsible for the data and the resulting partitions are
    stored on each worker as a list. This happens when a communications
    implementation is used which does not allow multiple ranks per device, so
    the partitions need to be grouped on the ranks to be processed concurrently
    using different streams.

    :param client: Dask client
    :param gpu_futures: [(future, part)] worker to part list of tuples
    :param worker_results_map: { rank: future } where future is a list
           of data partitions on a Dask worker
    :param getter_func: a function that takes a future and partition index
           as arguments and returns the data for a specific partitions
    :return: the ordered list of futures holding each partition on the workers
    """
    futures = []
    completed_part_map = {}
    for rank, part in gpu_futures:
        if rank not in completed_part_map:
            completed_part_map[rank] = 0

        f = worker_results_map[rank]

        futures.append(client.submit(getter_func, f, completed_part_map[rank]))

        completed_part_map[rank] += 1

    return futures


@gen.coroutine
def _extract_partitions(dask_obj, client=None):
    client = default_client() if client is None else client

    # dask.dataframe or dask.array
    if isinstance(
        dask_obj, (daskArray, daskSeries, daskDataFrame, dcSeries, dcDataFrame)
    ):
        persisted = client.persist(dask_obj)
        parts = futures_of(persisted)

    # iterable of dask collections (need to colocate them)
    elif isinstance(dask_obj, Sequence):
        # NOTE: We colocate (X, y) here by zipping delayed
        # n partitions of them as (X1, y1), (X2, y2)...
        # and asking client to compute a single future for
        # each tuple in the list
        dela = [np.asarray(d.to_delayed()) for d in dask_obj]

        # TODO: ravel() is causing strange behavior w/ delayed Arrays which are
        # not yet backed by futures. Need to investigate this behavior.
        # ref: https://github.com/NVIDIA/cuml/issues/2045
        raveled = [d.ravel() for d in dela]
        parts = client.compute([p for p in zip(*raveled)])

    yield wait(parts)

    key_to_part = [(part.key, part) for part in parts]
    who_has = yield client.who_has(parts)

    raise gen.Return(
        [(first(who_has[key]), part) for key, part in key_to_part]
    )
