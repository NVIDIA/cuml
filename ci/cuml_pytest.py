# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository-local pytest defaults, shared by developers and CI adapters."""

import copy
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PYTHON = REPO / "python/cuml"


def pytest_addoption(parser):
    group = parser.getgroup("cuML options")
    group.addoption(
        "--cuml-suite",
        choices=(
            "standard",
            "dask",
            "accel",
            "cudf-pandas",
        ),
        default="standard",
        help=(
            "Test suite and execution configuration. "
            "standard (default) runs cuML's single-GPU Python tests, excluding "
            "Dask and accelerator suites. "
            "Explicit pytest targets replace the default collection target."
        ),
    )


@pytest.hookimpl(tryfirst=True)
def pytest_load_initial_conftests(early_config, parser, args):
    options = parser.parse_known_args(args)
    suite = options.cuml_suite
    targets = {
        "standard": PYTHON / "tests",
        "dask": PYTHON / "tests/dask",
        "accel": PYTHON / "cuml_accel_tests",
        "cudf-pandas": PYTHON / "tests",
    }
    # Let pytest parse its own arguments: -k/-m/-n and their values are not
    # collection targets. Do not append a default to explicit selections.
    if not options.file_or_dir:
        args.append(str(targets[suite]))
        if suite in ("standard", "cudf-pandas"):
            # Options must precede --, which makes all following args targets.
            index = args.index("--") if "--" in args else len(args)
            args.insert(index, f"--ignore={PYTHON / 'tests/dask'}")
    if suite == "cudf-pandas":
        early_config.pluginmanager.import_plugin("cudf.pandas")
        from cudf.pandas import install

        install()
    _refresh_initial_args(early_config, parser, args)


def _refresh_initial_args(config, parser, args):
    # Initial conftest discovery uses the pre-parsed namespace, not args.
    # Keep both in sync so newly selected suites load their conftests early.
    config.known_args_namespace = parser.parse_known_args(
        args, namespace=copy.copy(config.known_args_namespace)
    )
