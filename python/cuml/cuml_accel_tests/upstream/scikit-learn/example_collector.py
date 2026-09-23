#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Pytest plugin that discovers scikit-learn example scripts and runs each
# one in an isolated subprocess under ``python -m cuml.accel``.
#
# Loaded via: pytest -p example_collector ...

import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest


class ExampleFailed(Exception):
    """Raised when a scikit-learn example script exits with non-zero status."""


class ExampleTimedOut(UserWarning):
    """Issued when a scikit-learn example exceeds the configured timeout."""


class ExampleNetworkError(UserWarning):
    """Issued when a scikit-learn example fails due to a network error."""


_NETWORK_ERROR_PATTERNS = (
    "urllib.error.HTTPError",
    "urllib.error.URLError",
    "http.client.RemoteDisconnected",
    "http.client.IncompleteRead",
    "http.client.BadStatusLine",
    "ConnectionError",
    "ConnectionAbortedError",
    "ConnectionRefusedError",
    "ConnectionResetError",
    "RemoteDisconnected",
    "TimeoutError",
    "requests.exceptions.ConnectionError",
    "requests.exceptions.Timeout",
    "sklearn.datasets._openml.OpenMLError",
    "md5 checksum of local file for https://openml.org/",
    "socket.gaierror",
    "socket.timeout",
    "ssl.SSLError",
)


def _network_error_pattern(output):
    for pattern in _NETWORK_ERROR_PATTERNS:
        if pattern in output:
            return pattern
    return None


class _FakeModule:
    """Minimal module-like object so the cuml.accel xfail plugin can read
    ``item.module.__name__`` without crashing on custom test items."""

    def __init__(self, name):
        self.__name__ = name


class ExampleFile(pytest.File):
    """Collector that wraps a single scikit-learn example script."""

    def collect(self):
        yield ExampleItem.from_parent(self, name=self.path.stem)


class ExampleItem(pytest.Item):
    """Test item that runs a scikit-learn example under ``cuml.accel``.

    Each example is executed in an isolated subprocess via
    ``python -m cuml.accel <script>``.  The test passes if the process
    exits with code 0.
    """

    def __init__(self, name, parent, **kwargs):
        super().__init__(name, parent, **kwargs)
        examples_dir = Path(self.config.getoption("examples_dir")).resolve()
        rel = self.path.relative_to(examples_dir)
        # Expose a module-like attribute so the cuml.accel xfail plugin
        # (which reads item.module.__name__) works for these items.
        # For "cluster/plot_hdbscan.py" -> module name "cluster",
        # giving xfail IDs like "cluster::plot_hdbscan".
        parent_str = str(rel.parent).replace(os.sep, ".")
        module_name = "" if parent_str in (".", "") else parent_str
        self.module = _FakeModule(module_name)

    def runtest(self):
        timeout = self.config.getoption("example_timeout")
        env = {
            **os.environ,
            "MPLBACKEND": "Agg",
            "PLOTLY_RENDERER": "json",
        }
        try:
            result = subprocess.run(
                [sys.executable, "-m", "cuml.accel", str(self.path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            warnings.warn(
                f"Example {self.path.name} timed out after {timeout}s",
                ExampleTimedOut,
            )
            pytest.xfail(reason=f"Timeout: example exceeded {timeout}s")
        if result.returncode != 0:
            stderr = result.stderr
            pattern = _network_error_pattern(result.stderr + result.stdout)
            if pattern:
                warnings.warn(
                    f"Example {self.path.name} failed due to network error"
                    f" ({pattern})",
                    ExampleNetworkError,
                )
                pytest.xfail(reason=f"Network error: {pattern}")
            if len(stderr) > 4000:
                stderr = "...\n" + stderr[-4000:]
            raise ExampleFailed(stderr)

    def repr_failure(self, excinfo):
        if isinstance(excinfo.value, ExampleFailed):
            return str(excinfo.value)
        return super().repr_failure(excinfo)

    def reportinfo(self):
        return self.path, None, self.nodeid


def pytest_addoption(parser):
    parser.addoption(
        "--examples-dir",
        action="store",
        default=None,
        help="Path to scikit-learn examples/ directory",
    )
    parser.addoption(
        "--example-timeout",
        action="store",
        default=300,
        type=int,
        help=("Timeout per example script in seconds (default: 300)"),
    )


def pytest_configure(config):
    """Download the 20newsgroups dataset before xdist workers spawn.

    Failures here are deliberately ignored. The examples download the
    dataset themselves if it is missing, and a genuine network failure
    is then classified by ``_network_error_pattern`` and xfailed as
    usual.
    """
    if hasattr(config, "workerinput"):
        # A worker, not the controller. Only the controller fetches.
        return

    from sklearn.datasets import fetch_20newsgroups_vectorized

    try:
        fetch_20newsgroups_vectorized(subset="all")
    except Exception as exc:
        print(f"Failed to pre-fetch the 20newsgroups dataset: {exc!r}")


def pytest_collect_file(parent, file_path):
    examples_dir = parent.config.getoption("examples_dir", default=None)
    if not examples_dir:
        return None
    examples_dir = Path(examples_dir).resolve()
    if file_path.suffix == ".py" and file_path.resolve().is_relative_to(
        examples_dir
    ):
        return ExampleFile.from_parent(parent, path=file_path)
    return None
