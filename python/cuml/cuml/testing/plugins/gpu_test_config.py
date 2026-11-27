# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolate worker CUDA caches and configure GPU-memory and stress-test settings."""

import os
import tempfile
from math import ceil

import pynvml
import pytest


def _set_cuda_cache_path_per_xdist_worker(config):
    """Isolate CUDA JIT caches before worker CUDA runtime initialization."""
    workerinput = getattr(config, "workerinput", None)
    if workerinput is None:
        return
    workerid = workerinput["workerid"]
    safe = "".join(
        c if c.isalnum() or c in "-_" else "_" for c in str(workerid)
    )
    if existing := os.environ.get("CUDA_CACHE_PATH"):
        cache_dir = os.path.join(existing, safe)
    else:
        cache_dir = os.path.join(
            tempfile.gettempdir(), "cuml-pytest-cuda-cache", safe
        )
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["CUDA_CACHE_PATH"] = cache_dir


def pytest_configure(config):
    _set_cuda_cache_path_per_xdist_worker(config)
    pytest.max_gpu_memory = _get_gpu_memory()
    pytest.adapt_stress_test = "CUML_ADAPT_STRESS_TESTS" in os.environ


def _get_pynvml_device_handle(device_id=0):
    """Get a GPU handle by device index or UUID, using the parent for MIG."""
    pynvml.nvmlInit()
    try:
        if device_id and not str(device_id).isnumeric():
            handle = pynvml.nvmlDeviceGetHandleByUUID(str.encode(device_id))
            if pynvml.nvmlDeviceIsMigDeviceHandle(handle):
                handle = pynvml.nvmlDeviceGetDeviceHandleFromMigDeviceHandle(
                    handle
                )
        else:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        return handle
    except pynvml.NVMLError:
        raise ValueError(f"Invalid device index or UUID: {device_id}")


def _get_gpu_memory(device_index=0):
    """Return total GPU memory in GiB, or None when unsupported."""
    handle = _get_pynvml_device_handle(device_index)
    try:
        return ceil(pynvml.nvmlDeviceGetMemoryInfo(handle).total / 2**30)
    except pynvml.NVMLError_NotSupported:
        return None
