# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository-local pytest defaults, shared by developers and CI adapters."""

import copy
from importlib.metadata import version
from pathlib import Path
import shlex
import subprocess
from tempfile import TemporaryDirectory

import pytest

REPO = Path(__file__).resolve().parents[1]
PYTHON = REPO / "python/cuml"


def pytest_addoption(parser):
    group = parser.getgroup("cuML options")
    group.addoption(
        "--cuml-fetch-upstream",
        action="store_true",
        help=(
            "Allow downloading missing upstream test sources (currently accel/umap). "
            "Existing checkouts are not updated."
        ),
    )
    group.addoption(
        "--cuml-suite",
        choices=(
            "standard",
            "dask",
            "accel",
            "cudf-pandas",
            "accel/sklearn",
            "accel/hdbscan",
            "accel/umap",
        ),
        default="standard",
        help=(
            "Test suite and execution configuration. "
            "standard (default) runs cuML's single-GPU Python tests, excluding "
            "Dask and accelerator suites. "
            "Explicit pytest targets replace the default collection target; "
            "accel/<library> uses upstream tests in combination with accepted xfails."
        ),
    )


@pytest.hookimpl(tryfirst=True)
def pytest_load_initial_conftests(early_config, parser, args):
    options = parser.parse_known_args(args)
    if options.cuml_suite.startswith("accel/"):
        if options.help:
            return
        _configure_upstream(early_config, parser, args, options)
        _refresh_initial_args(early_config, parser, args)
        return
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
            args.append(f"--ignore={PYTHON / 'tests/dask'}")
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


def _configure_upstream(config, parser, args, options):
    suite = options.cuml_suite.removeprefix("accel/")
    upstream = PYTHON / "cuml_accel_tests/upstream"
    directory = upstream / ("scikit-learn" if suite == "sklearn" else suite)
    # A plugin imported during this hook does not receive the ongoing hook
    # call. Install explicitly, before pytest imports upstream conftests.
    config.pluginmanager.import_plugin("cuml.accel")
    from cuml.accel import install

    install()
    options = parser.parse_known_args(args)
    if not options.xfail_list:
        args.append(f"--xfail-list={directory / 'xfail-list.yaml'}")
    # Use the shared upstream warning policy instead of cuML's test filters.
    # Preserve explicit command-line -W overrides.
    from configparser import ConfigParser

    policy = ConfigParser(interpolation=None)
    policy.read(upstream / "pytest-upstream.ini")
    config.getini("filterwarnings")[:] = [
        line
        for line in policy["pytest"]["filterwarnings"].splitlines()
        if line
    ]
    # This option is cached during pytest startup, before this hook. Updating
    # args alone would leave collection using pytest's cached default.
    config._inicache["parametrize_long_str_id_strategy"] = "legacy"
    args.extend(["-o", "parametrize_long_str_id_strategy=legacy"])
    # UMAP ships tests only in its source checkout. Match the legacy upstream
    # config (pytest's default), rather than cuML's append import mode.
    if suite == "umap":
        args.append("--import-mode=prepend")
    if suite == "umap" and not options.confcutdir:
        # Exclude the parent accelerator conftest, including for explicit
        # paths into the checkout; external source trees remain supported.
        args.append(f"--confcutdir={upstream}")
    if not options.file_or_dir:
        if suite == "umap":
            tests = _validated_umap_tests(
                directory, fetch=options.cuml_fetch_upstream
            )
            args.append(str(tests))
        else:
            args.extend(
                [
                    "--pyargs",
                    "sklearn" if suite == "sklearn" else "hdbscan.tests",
                ]
            )
        # Full default collection can validate the entire list, even with -k:
        # xfails are applied before pytest's keyword/marker deselection.
        args.append("--strict-xfail-list")


def _validated_umap_tests(directory, *, fetch=False):
    """Obtain missing managed sources, then validate without overwriting them."""
    installed_version = version("umap-learn")
    tag = f"release-{installed_version}"
    checkout = directory / "umap-upstream"
    tests = checkout / "umap/tests"
    prepare = shlex.quote(str(directory / "prepare-tests.sh"))
    config = shlex.quote(str(REPO / "pyproject.toml"))
    guidance = (
        "\nAllow pytest to obtain missing sources:\n"
        f"  pytest -c {config} --cuml-suite=accel/umap --cuml-fetch-upstream\n"
        f"Or prepare sources matching UMAP {installed_version} manually:\n"
        f"  {prepare}\n"
        "Then retry (from any working directory):\n"
        f"  pytest -c {config} --cuml-suite=accel/umap\n"
        "Or use an existing or modified checkout explicitly:\n"
        f"  pytest -c {config} --cuml-suite=accel/umap /path/to/umap-checkout/umap/tests\n"
        "Existing checkouts are never reset or overwritten automatically."
    )
    if not checkout.exists() and fetch:
        _checkout_umap(directory, checkout, tag, guidance)
    if not tests.is_dir():
        raise pytest.UsageError(
            f"UMAP test sources not found at {tests}.{guidance}"
        )

    def revision(ref):
        result = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "--verify", ref],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    try:
        head = revision("HEAD")
        expected = revision(f"refs/tags/{tag}^{{commit}}")
    except OSError as exc:
        raise pytest.UsageError(
            f"Cannot validate UMAP sources: {exc}."
        ) from exc
    if head is None or expected is None or head != expected:
        raise pytest.UsageError(
            f"UMAP sources at {checkout} do not match installed UMAP "
            f"{installed_version}: expected tag {tag} "
            f"({expected or 'not found'}), HEAD is {head or 'not found'}."
        )
    dirty = subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        capture_output=True,
        text=True,
    )
    if dirty.returncode != 0 or dirty.stdout.strip():
        raise pytest.UsageError(
            f"Managed UMAP sources at {checkout} have local changes or cannot "
            "be inspected. Use an explicit test path to test modified sources."
        )
    return tests


def _checkout_umap(directory, checkout, tag, guidance):
    """Publish only complete clones; failures leave no partial managed checkout."""
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with TemporaryDirectory(
            prefix=".umap-checkout-", dir=directory
        ) as tmp:
            source = Path(tmp) / "source"
            print(
                f"Obtaining UMAP test sources at {tag} from GitHub...",
                flush=True,
            )
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth=1",
                    "--branch",
                    tag,
                    "https://github.com/lmcinnes/umap.git",
                    str(source),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
            try:
                source.rename(checkout)
            except OSError:
                # Another pytest process (e.g. an xdist worker) may have
                # published a complete clone first. Validate that copy below.
                if not checkout.exists():
                    raise
    except (OSError, subprocess.SubprocessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise pytest.UsageError(
            f"Could not obtain UMAP sources for {tag}: {detail}{guidance}"
        ) from exc
