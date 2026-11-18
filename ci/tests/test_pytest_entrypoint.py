# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only subprocess tests of the real pytest argument/collection lifecycle."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "ci").mkdir()
    shutil.copy(REPO / "ci/cuml_pytest.py", tmp_path / "ci")
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\naddopts = "-p cuml_pytest"\n'
        'pythonpath = ["ci"]\n'
    )
    for suite in ("tests", "tests/dask", "cuml_accel_tests"):
        directory = tmp_path / "python/cuml" / suite
        directory.mkdir(parents=True, exist_ok=True)
        name = suite.replace("/", "_")
        (directory / f"test_{name}.py").write_text(
            "def test_selected(): pass\ndef test_other(): pass\n"
        )
    # A suite plugin must be registered during initial conftest discovery,
    # not during collection after pytest's configure phase.
    (tmp_path / "ci/suite_plugin.py").write_text("# suite plugin\n")
    (tmp_path / "python/cuml/cuml_accel_tests/conftest.py").write_text(
        "pytest_plugins = ['suite_plugin']\n"
    )
    return tmp_path


def run(repo, *args, cwd=None, env=None):
    return subprocess.run(
        [sys.executable, "-m", "pytest", *args],
        cwd=cwd or repo,
        env={
            **os.environ,
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            **(env or {}),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


@pytest.mark.parametrize("args", [(), ("-k", "selected"), ("-m", "not slow")])
def test_default_suite(repo, args):
    result = run(repo, "--collect-only", "-q", *args)
    assert result.returncode == 0, result.stdout
    assert "test_tests.py::test_selected" in result.stdout
    assert "test_tests_dask" not in result.stdout
    assert "test_cuml_accel_tests" not in result.stdout
    if args[:1] == ("-k",):
        assert "::test_other" not in result.stdout


@pytest.mark.parametrize("subdir", [".", "python/cuml", "python/cuml/tests"])
def test_node_id_and_report_are_caller_relative(repo, subdir):
    cwd = repo / subdir
    target = os.path.relpath(repo / "python/cuml/tests/test_tests.py", cwd)
    result = run(
        repo, f"{target}::test_selected", "--junitxml=report.xml", cwd=cwd
    )
    assert result.returncode == 0, result.stdout
    assert "1 passed" in result.stdout
    assert (cwd / "report.xml").exists()


@pytest.mark.parametrize("suite", ["dask", "accel"])
def test_suite_defaults_and_overrides(repo, suite):
    result = run(repo, f"--cuml-suite={suite}", "-k", "selected")
    assert result.returncode == 0, result.stdout
    assert "1 passed" in result.stdout
    result = run(
        repo,
        f"--cuml-suite={suite}",
        "python/cuml/tests/test_tests.py::test_other",
    )
    assert result.returncode == 0, result.stdout
    assert "1 passed" in result.stdout


def test_explicit_dask_not_ignored(repo):
    result = run(repo, "python/cuml/tests/dask")
    assert result.returncode == 0, result.stdout
    assert "2 passed" in result.stdout


@pytest.mark.parametrize(
    "runner",
    [
        "ci/run_cuml_singlegpu_pytests.sh",
        "ci/run_cuml_dask_pytests.sh",
        "ci/run_cuml_singlegpu_accel_pytests.sh",
        "ci/run_cuml_integration_pytests.sh",
    ],
)
def test_runner_preserves_arguments_and_cwd(tmp_path, runner):
    # Capture the exact interpreter invocation; no GPU or upstream checkout.
    executable = tmp_path / "python"
    executable.write_text('#!/bin/bash\nprintf "%s\\n" "$PWD" "$@"\n')
    executable.chmod(0o755)
    result = subprocess.run(
        [
            str(REPO / runner),
            "some path/test_file.py::test_one",
            "-k",
            "a or b",
        ],
        cwd=tmp_path,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == str(tmp_path)
    assert lines[1:5] == ["-m", "pytest", "-c", str(REPO / "pyproject.toml")]
    assert lines[-3:] == ["some path/test_file.py::test_one", "-k", "a or b"]
    assert "." not in lines
