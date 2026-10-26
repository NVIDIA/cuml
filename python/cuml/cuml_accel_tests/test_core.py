# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib
import importlib.metadata
import multiprocessing
import os
import subprocess
import sys
from inspect import Parameter, signature

import pytest
import sklearn
from packaging.version import Version

import cuml.accel
from cuml.accel.core import _CONSTRAINTS, CheckConstraint
from cuml.accel.estimator_proxy import ProxyBase


def proxy_base_subclasses():
    """All defined subclasses of ProxyBase"""
    from cuml.accel.core import ACCELERATED_MODULES

    for mod in ACCELERATED_MODULES:
        importlib.import_module(mod)

    out = [
        cls for cls in ProxyBase.__subclasses__() if hasattr(cls, "_gpu_class")
    ]
    out.sort(key=lambda cls: cls.__name__)
    return out


def test_multiple_import_styles_work():
    from sklearn import linear_model
    from sklearn.linear_model import LogisticRegression

    assert linear_model.LogisticRegression is LogisticRegression
    assert cuml.accel.is_proxy(LogisticRegression)


def test_enabled():
    assert cuml.accel.enabled()


@pytest.mark.parametrize("method", ["spawn", "fork"])
def test_enabled_in_subprocesses(method):
    ctx = multiprocessing.get_context(method)
    with ctx.Pool(processes=1) as pool:
        enabled = pool.apply(cuml.accel.enabled)
    assert enabled


def test_enabled_in_loky_executor():
    try:
        # Recent versions of joblib vendor loky
        from joblib.externals import loky
    except ImportError:
        loky = pytest.importorskip("loky")

    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    X, y = make_classification()

    with loky.get_reusable_executor(max_workers=1) as executor:
        enabled = executor.submit(cuml.accel.enabled).result()
        assert enabled
        remote = executor.submit(LogisticRegression().fit, X, y).result()
        assert cuml.accel.is_proxy(remote)


_INSTALL_SCRIPT = """
import rmm
import cuml.accel.core as core

core._is_concurrent_managed_access_supported = lambda: True
if {fail_pool}:
    def fail(*args, **kwargs):
        raise MemoryError("no room")
    rmm.mr.PoolMemoryResource = fail
core.install(log_level="debug")
mr = rmm.mr.get_current_device_resource()
assert isinstance(mr, rmm.mr.PrefetchResourceAdaptor)
print("UPSTREAM", type(mr.upstream_mr).__name__)
"""


@pytest.mark.parametrize(
    "fail_pool, upstream, message",
    [
        (False, "PoolMemoryResource", "Enabled pooled managed memory."),
        (True, "ManagedMemoryResource", "Enabled managed memory."),
    ],
)
def test_install_managed_memory_resource(fail_pool, upstream, message):
    """install() pools managed memory, and falls back to (and reports)
    unpooled managed memory if the initial pool block cannot be allocated."""
    env = {
        k: v for k, v in os.environ.items() if not k.startswith("CUML_ACCEL")
    }
    proc = subprocess.run(
        [sys.executable, "-c", _INSTALL_SCRIPT.format(fail_pool=fail_pool)],
        env=env,
        capture_output=True,
        text=True,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output
    assert f"UPSTREAM {upstream}" in output
    assert message in output
    assert ("Enabled pooled managed memory." in output) is not fail_pool


def get_level():
    return cuml.accel.core.logger.level.name.lower()


def test_log_level_forwarded_to_subprocesses(monkeypatch):
    monkeypatch.setenv("CUML_ACCEL_LOG_LEVEL", "debug")
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=1) as pool:
        log_level = pool.apply(get_level)
    assert log_level == "debug"


def iter_proxy_class_methods():
    """Generate test cases of (cls, method_name) for all ProxyBase proxied methods"""
    classes = proxy_base_subclasses()

    for cls in classes:
        for name in dir(cls._cpu_class):
            if not name.startswith("_") and callable(
                getattr(cls._cpu_class, name)
            ):
                # XXX: xfail umap.UMAP.get_feature_names_out for umap-learn < 0.5.8
                if (
                    cls._cpu_class.__name__ == "UMAP"
                    and name == "get_feature_names_out"
                    and Version(importlib.metadata.version("umap-learn"))
                    < Version("0.5.8")
                ):
                    yield pytest.param(
                        cls,
                        name,
                        marks=[
                            pytest.mark.xfail(
                                reason="umap-learn < 0.5.8 doesn't implement `get_feature_names_out` properly",
                                strict=True,
                            )
                        ],
                    )
                else:
                    yield cls, name


@pytest.mark.parametrize("cls, name", iter_proxy_class_methods())
def test_proxied_methods_signature_compatibility(cls, name):
    """Check that the GPU proxy signatures are compatible with the CPU signatures.

    Any methods that fail this check should have an override `_gpu_{name}` method
    defined on the proxy class to fix the signature mismatch between the CPU method
    signature and cuml's version."""
    cpu_method = getattr(cls._cpu_class, name)

    # Get the GPU method definition the proxy will use
    gpu_method = getattr(cls, f"_gpu_{name}", None)
    if gpu_method is None:
        gpu_method = getattr(cls._gpu_class, name, None)
        if gpu_method is None:
            # CPU-only method, nothing to check
            return

    cpu_params = list(signature(cpu_method).parameters.values())
    gpu_params = list(signature(gpu_method).parameters.values())

    # Check that the GPU method signature is a superset of the CPU one
    assert len(gpu_params) >= len(cpu_params)
    for cpu_param, gpu_param in zip(cpu_params, gpu_params):
        assert cpu_param.name == gpu_param.name
        assert cpu_param.kind == gpu_param.kind

    # Check that any additional arguments are optional
    for gpu_param in gpu_params[len(cpu_params) :]:
        if gpu_param.kind == Parameter.POSITIONAL_OR_KEYWORD:
            assert gpu_param.default is not Parameter.empty
        else:
            assert gpu_param.kind in {
                Parameter.KEYWORD_ONLY,
                Parameter.VAR_KEYWORD,
            }


def test_check_constraints(capsys):
    # A constraint we know to be met
    constraint = CheckConstraint("scikit-learn>=1.0")
    assert constraint()
    stdout, _ = capsys.readouterr()
    logs = [
        line for line in stdout.split("\n") if line.startswith("[cuml.accel]")
    ]
    assert not logs

    # A constraint we know to not be met
    constraint = CheckConstraint("scikit-learn>=1.0,<=1.4")

    assert not constraint()
    stdout, _ = capsys.readouterr()
    logs = [
        line for line in stdout.split("\n") if line.startswith("[cuml.accel]")
    ]
    assert len(logs) == 1
    log = logs[0]
    assert str(constraint.requirement) in log
    assert sklearn.__version__ in log


def test_constraints_match_installed_versions(capsys):
    """This test checks all registered version constraints, and errors if
    they're not met in the test environment. Since we test oldest and latest
    versions in CI, this test will error if we ever update our dependencies
    without also updating the constraints."""

    missed_constraints = []
    for constraint in _CONSTRAINTS.values():
        version = importlib.metadata.version(constraint.requirement.name)
        # Skip checking constraints on prereleases so this test doesn't fail
        # when run with sklearn prereleases as part of nightly runs.
        if not Version(version).is_prerelease and not constraint():
            missed_constraints.append(constraint)

    if missed_constraints:
        constraint_lines = "\n".join(
            f"- {c.requirement!s}" for c in missed_constraints
        )
        stdout, _ = capsys.readouterr()
        log_lines = "\n".join(
            line
            for line in stdout.split("\n")
            if line.startswith("[cuml.accel]")
        )
        msg = (
            "Please update `cuml.accel.core._CONSTRAINTS` to match the new "
            "runtime constraints.\n\n"
            "Currently the following constraints failed:\n\n"
            f"{constraint_lines}\n\n"
            "leading to the following user-facing logs:\n\n"
            f"{log_lines}"
        )
        assert False, msg
