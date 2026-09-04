#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Keep a poisoned CUDA context from failing every later test in a worker.

Load this from any test suite that runs under pytest-xdist by adding
``pytest_plugins = "cuml.testing.plugins.restart_corrupted_xdist_worker"`` to
its ``conftest.py``. Suites in sibling directories do not share a
``conftest.py``, so each one has to register the plugin itself.
"""

import os

import cupy as cp
import pytest

IS_XDIST_WORKER = os.environ.get("PYTEST_XDIST_WORKER") is not None


@pytest.hookimpl(trylast=True)
def pytest_runtest_logreport(report):
    """Some CUDA errors are "sticky" and cause subsequent CUDA calls in a
    process to error.

    When running under pytest-xdist, this hook checks for sticky errors on
    failed tests, and if present will crash the worker after reporting the
    failure. pytest-xdist will then start a new worker with a clean CUDA
    context, avoiding sticky errors.

    This hook is a no-op when run outside of pytest-xdist.
    """
    if not IS_XDIST_WORKER:
        return

    # Only check on failed test runs, not other events
    if report.when == "call" and report.failed:
        try:
            # Try allocating a tiny array to invoke a CUDA API. If this fails,
            # then the context is definitely corrupted.
            cp.ones(1)
        except Exception:
            # Hardcrash the worker
            os._exit(1)
