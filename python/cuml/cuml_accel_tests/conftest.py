# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configure ``cuml.accel`` for its integration tests.

The upstream tests use the ``cuml.accel`` pytest plugin explicitly.
"""

import os

import pytest

from cuml.accel import install
from cuml.accel.core import logger

pytest_plugins = "cuml.testing.plugins.restart_corrupted_xdist_worker"

# Install the accelerator
install()


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_setupnodes(config, specs):
    """Suppress startup logs while xdist creates execnet gateways."""
    os.environ["CUML_ACCEL_LOG_LEVEL"] = "error"


@pytest.fixture(scope="session", autouse=True)
def restore_log_level():
    """Restore normal logging in xdist workers after startup."""
    if os.environ.get("PYTEST_XDIST_WORKER"):
        logger.set_level("warn")
        os.environ["CUML_ACCEL_LOG_LEVEL"] = "warn"


# Ignore the upstream directory, those tests need to be invoked separately
collect_ignore = ["upstream"]
