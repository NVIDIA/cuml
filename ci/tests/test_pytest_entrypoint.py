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


@pytest.fixture
def upstream_repo(repo):
    # Exercise the actual xfail plugin without importing GPU dependencies.
    accel = repo / "cuml/accel"
    accel.mkdir(parents=True)
    (repo / "cuml/__init__.py").touch()
    (accel / "core.py").write_text("def install(): pass\n")
    shutil.copy(REPO / "python/cuml/cuml/accel/pytest_plugin.py", accel)
    (accel / "__init__.py").write_text(
        "from .core import install\n"
        "from .pytest_plugin import (pytest_addoption, "
        "pytest_collection_modifyitems, pytest_load_initial_conftests)\n"
    )
    upstream = repo / "python/cuml/cuml_accel_tests/upstream"
    (upstream / "scikit-learn").mkdir(parents=True)
    shutil.copy(
        REPO / "python/cuml/cuml_accel_tests/upstream/pytest-upstream.ini",
        upstream,
    )
    (upstream / "scikit-learn/xfail-list.yaml").write_text(
        "- reason: known failure\n  marker: known\n  tests:\n"
        "    - sklearn.tests.test_sample::test_failure\n"
        "    - sklearn.tests.test_other::test_failure\n"
    )
    tests = repo / "sklearn/tests"
    tests.mkdir(parents=True)
    (tests.parent / "__init__.py").touch()
    (tests / "__init__.py").touch()
    (tests / "test_sample.py").write_text(
        "def test_failure(): assert False\ndef test_pass(): pass\n"
    )
    (tests / "test_other.py").write_text("def test_failure(): assert False\n")
    (repo / "python/cuml/tests/conftest.py").write_text(
        "raise AssertionError('standard suite must not load for upstream')\n"
    )
    metadata = repo / "umap_learn-0.0.1.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: umap-learn\nVersion: 0.0.1\n"
    )
    return repo


def git(checkout, *args):
    return subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def prepare_umap_git(checkout):
    git(checkout, "init")
    git(checkout, "add", ".")
    git(
        checkout,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-m",
        "Test sources",
    )
    git(checkout, "tag", "release-0.0.1")


@pytest.mark.parametrize(
    "selection", [(), ("-k", "test_pass"), ("-m", "known")]
)
def test_upstream_default_collection(upstream_repo, selection):
    result = run(
        upstream_repo,
        "--cuml-suite=accel/sklearn",
        "--cuml-fetch-upstream",
        *selection,
    )
    assert result.returncode == 0, result.stdout
    if selection[:1] == ("-k",):
        assert "1 passed" in result.stdout
    else:
        assert "2 xfailed" in result.stdout


@pytest.mark.parametrize(
    "target",
    [
        "sklearn.tests.test_sample",
        "sklearn/tests/test_sample.py::test_failure",
    ],
)
def test_upstream_focused_collection(upstream_repo, target):
    result = run(
        upstream_repo, "--cuml-suite=accel/sklearn", "--pyargs", target
    )
    assert result.returncode == 0, result.stdout
    assert "1 xfailed" in result.stdout
    result = run(
        upstream_repo,
        "--cuml-suite=accel/sklearn",
        "--pyargs",
        target,
        "--strict-xfail-list",
    )
    assert result.returncode != 0, result.stdout
    assert "UnmatchedXfailTests" in result.stdout


def test_upstream_full_collection_rejects_stale_xfail(upstream_repo):
    path = (
        upstream_repo
        / "python/cuml/cuml_accel_tests/upstream/scikit-learn/xfail-list.yaml"
    )
    path.write_text(
        path.read_text() + "    - sklearn.tests.test_gone::test_gone\n"
    )
    result = run(upstream_repo, "--cuml-suite=accel/sklearn")
    assert result.returncode != 0, result.stdout
    assert "UnmatchedXfailTests" in result.stdout


def test_upstream_keeps_legacy_long_parameter_ids(upstream_repo):
    text = "x" * 1000
    test = upstream_repo / "sklearn/tests/test_long.py"
    test.write_text(
        f"import pytest\n@pytest.mark.parametrize('value', [{text!r}])\n"
        "def test_long(value): pass\n"
    )
    result = run(
        upstream_repo,
        "--cuml-suite=accel/sklearn",
        "--pyargs",
        "sklearn.tests.test_long",
        "--collect-only",
        "-q",
    )
    assert result.returncode == 0, result.stdout
    assert f"test_long[{text}]" in result.stdout


def test_umap_sources_do_not_load_parent_conftest(upstream_repo):
    upstream = upstream_repo / "python/cuml/cuml_accel_tests/upstream"
    directory = upstream / "umap"
    tests = directory / "umap-upstream/umap/tests"
    tests.mkdir(parents=True)
    (tests.parent / "__init__.py").touch()
    (tests / "__init__.py").touch()
    (tests / "conftest.py").write_text("pytest_plugins = ['suite_plugin']\n")
    (tests / "test_sample.py").write_text("def test_sample(): pass\n")
    (directory / "xfail-list.yaml").write_text("[]\n")
    parent = upstream.parent / "conftest.py"
    parent.write_text("raise AssertionError('parent suite must not load')\n")
    prepare_umap_git(directory / "umap-upstream")
    for selection in [(), (str(tests / "test_sample.py"),)]:
        result = run(upstream_repo, "--cuml-suite=accel/umap", *selection)
        assert result.returncode == 0, result.stdout
        assert "1 passed" in result.stdout


@pytest.mark.parametrize("missing_tag", [False, True])
def test_umap_automatic_checkout(upstream_repo, tmp_path, missing_tag):
    remote = tmp_path / "remote"
    tests = remote / "umap/tests"
    tests.mkdir(parents=True)
    (tests.parent / "__init__.py").touch()
    (tests / "__init__.py").touch()
    (tests / "test_sample.py").write_text("def test_sample(): pass\n")
    prepare_umap_git(remote)
    if missing_tag:
        git(remote, "tag", "-d", "release-0.0.1")
    directory = upstream_repo / "python/cuml/cuml_accel_tests/upstream/umap"
    directory.mkdir(parents=True)
    (directory / "xfail-list.yaml").write_text("[]\n")
    config = tmp_path / "gitconfig"
    config.write_text(
        f'[url "{remote.as_uri()}"]\n'
        "    insteadOf = https://github.com/lmcinnes/umap.git\n"
    )
    env = {"GIT_CONFIG_GLOBAL": str(config), "GIT_CONFIG_NOSYSTEM": "1"}
    # Help must not trigger network access or source preparation.
    result = run(
        upstream_repo,
        "--cuml-suite=accel/umap",
        "--cuml-fetch-upstream",
        "--help",
        env=env,
    )
    assert result.returncode == 0, result.stdout
    assert not (directory / "umap-upstream").exists()
    result = run(upstream_repo, "--cuml-suite=accel/umap", env=env)
    assert result.returncode == 4, result.stdout
    assert "--cuml-fetch-upstream" in result.stdout
    assert "Obtaining UMAP" not in result.stdout
    assert not (directory / "umap-upstream").exists()
    result = run(
        upstream_repo,
        "--cuml-suite=accel/umap",
        "--cuml-fetch-upstream",
        env=env,
    )
    checkout = directory / "umap-upstream"
    if missing_tag:
        assert result.returncode == 4, result.stdout
        assert "Could not obtain UMAP sources" in result.stdout
        assert "prepare-tests.sh" in result.stdout
        assert "/path/to/umap-checkout/umap/tests" in result.stdout
        assert not checkout.exists()
    else:
        assert result.returncode == 0, result.stdout
        assert "1 passed" in result.stdout
        assert git(checkout, "rev-parse", "HEAD") == git(
            remote, "rev-parse", "HEAD"
        )
        # An existing valid clone is reusable without access to the remote.
        shutil.rmtree(remote)
        for opt_in in [(), ("--cuml-fetch-upstream",)]:
            result = run(
                upstream_repo, "--cuml-suite=accel/umap", *opt_in, env=env
            )
            assert result.returncode == 0, result.stdout
    assert not list(directory.glob(".umap-checkout-*"))


@pytest.mark.parametrize(
    "state", ["wrong-revision", "missing-tag", "dirty", "not-git"]
)
def test_umap_invalid_managed_sources_and_explicit_override(
    upstream_repo, state
):
    directory = upstream_repo / "python/cuml/cuml_accel_tests/upstream/umap"
    checkout = directory / "umap-upstream"
    tests = checkout / "umap/tests"
    tests.mkdir(parents=True)
    (tests.parent / "__init__.py").touch()
    (tests / "__init__.py").touch()
    test = tests / "test_sample.py"
    test.write_text("def test_sample(): pass\n")
    (directory / "xfail-list.yaml").write_text("[]\n")
    if state != "not-git":
        prepare_umap_git(checkout)
        if state == "missing-tag":
            git(checkout, "tag", "-d", "release-0.0.1")
        else:
            test.write_text("def test_sample(): assert True\n")
            if state == "wrong-revision":
                git(checkout, "add", ".")
                git(
                    checkout,
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.com",
                    "-c",
                    "commit.gpgsign=false",
                    "commit",
                    "-m",
                    "Modified sources",
                )
    result = run(upstream_repo, "--cuml-suite=accel/umap")
    assert result.returncode == 4, result.stdout
    assert "prepare-tests.sh" not in result.stdout
    if state != "dirty":
        assert "release-0.0.1" in result.stdout
    # Explicit paths are an intentional escape hatch, even for modified sources.
    result = run(
        upstream_repo,
        "--cuml-suite=accel/umap",
        "--cuml-fetch-upstream",
        str(test),
    )
    assert result.returncode == 0, result.stdout
    assert "1 passed" in result.stdout


def test_explicit_dask_not_ignored(repo):
    result = run(repo, "python/cuml/tests/dask")
    assert result.returncode == 0, result.stdout
    assert "2 passed" in result.stdout
