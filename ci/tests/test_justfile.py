# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU smoke checks; run with just on PATH and pytest -c /dev/null."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
JUST = shutil.which("just")
pytestmark = pytest.mark.skipif(JUST is None, reason="just is not installed")


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "checkout with spaces"
    root.mkdir()
    shutil.copyfile(ROOT / "justfile", root / "justfile")
    caller = root / "caller"
    caller.mkdir()
    return root, caller


def run(checkout, *args, env=None):
    return subprocess.run(
        [JUST, *args],
        cwd=checkout[1],
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )


def test_default_lists_recipes(checkout):
    result = run(checkout)
    assert result.returncode == 0, result.stderr
    for recipe in [
        "build",
        "test",
        "test-accel",
        "test-accel-sklearn",
        "test-dask",
        "test-cudf-pandas",
    ]:
        assert recipe in result.stdout


def test_pytest_runs_from_root(checkout):
    root, caller = checkout
    (root / "pytest.ini").write_text("[pytest]\n")
    (root / "test_sample.py").write_text("def test_pass(): pass\n")
    bindir = caller / "bin"
    bindir.mkdir()
    (bindir / "python").symlink_to(sys.executable)
    env = dict(
        os.environ,
        PATH=f"{bindir}:{os.environ['PATH']}",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
    result = run(
        checkout, "test", "--junitxml=relative report.xml", "-q", env=env
    )
    assert result.returncode == 0, result.stderr
    assert "1 passed" in result.stdout
    assert (root / "relative report.xml").is_file()
    assert not (caller / "relative report.xml").exists()
