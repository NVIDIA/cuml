#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import warnings
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

from packaging.requirements import Requirement

from cuml.accel.core import install


class UnmatchedXfailTests(UserWarning):
    """Warning raised when xfail entries in the configuration file don't match any actual tests.

    This warning is raised during pytest collection when there are entries in the xfail
    list that don't correspond to any existing test functions. This typically indicates
    either:
    1. Tests have been renamed or removed but the xfail list wasn't updated
    2. There are typos in the test IDs in the xfail list
    3. The tests only exist for specific versions of dependencies (check the condition
       field in the xfail list)
    """

    ...


def pytest_load_initial_conftests(early_config, parser, args):
    # https://docs.pytest.org/en/7.1.x/reference/\
    # reference.html#pytest.hookspec.pytest_load_initial_conftests
    install()


def pytest_addoption(parser):
    """Add command line option for xfail list file."""
    parser.addoption(
        "--strict-xfail-list",
        action="store_true",
        help="Validate every active xfail entry (requires full suite collection)",
    )
    parser.addoption(
        "--xfail-list",
        action="store",
        help="Path to YAML file containing list of test IDs to mark as xfail",
    )


def _evaluate_single_condition(condition_str: str) -> bool:
    """Evaluate a single package version condition.

    Args:
        condition_str: String in format 'package[comparison]version',
                      e.g. 'scikit-learn>=1.6'

    Returns:
        bool: True if the condition is met, False otherwise
    """
    try:
        req = Requirement(condition_str.strip())
        installed_version = version(req.name)
        return req.specifier.contains(installed_version, prereleases=True)
    except Exception:
        return False


def create_version_condition(condition_str: str) -> bool:
    """Evaluate a version condition immediately.

    Supports a single package specifier or multiple specifiers joined by 'and',
    in which case all clauses must be satisfied.

    Args:
        condition_str: A version condition string. Examples:
                      - 'scikit-learn>=1.6'
                      - 'numpy<2.0.0'
                      - 'umap-learn<=0.5.8 and scikit-learn>=1.6'

    Returns:
        bool: True if all conditions are met, False otherwise
    """
    if not condition_str:
        return True

    return all(
        _evaluate_single_condition(clause)
        for clause in condition_str.split(" and ")
    )


def pytest_collection_modifyitems(config, items):
    """Apply xfail markers to tests listed in the xfail list file."""
    # Import pytest lazily to avoid requiring it for normal cuml usage.
    # pytest is only needed when running tests.
    import pytest

    strict_xfail_list = config.getoption("strict_xfail_list")
    xfail_list_path = config.getoption("xfail_list")
    if not xfail_list_path:
        return

    import yaml  # needed to parse the YAML-formatted xfail list

    xfail_list_path = Path(xfail_list_path)
    if not xfail_list_path.exists():
        raise ValueError(f"Xfail list file not found: {xfail_list_path}")

    xfail_list = yaml.safe_load(xfail_list_path.read_text())

    if not isinstance(xfail_list, list):
        raise ValueError("Xfail list must be a list of test groups")

    # Create markers for all unique markers in the xfail list
    markers = {
        marker: pytest.mark.__getattr__(marker)
        for group in xfail_list
        if (marker := group.get("marker"))
    }
    # Convert list of groups into dict mapping test IDs to lists of xfail
    # configs
    xfail_configs = defaultdict(list)
    for group in xfail_list:
        if not isinstance(group, dict):
            raise ValueError("Xfail list entry must be a dictionary")
        if "reason" not in group:
            raise ValueError("Xfail list entry must contain a 'reason' field")
        if "tests" not in group:
            raise ValueError("Xfail list entry must contain a 'tests' field")

        reason = group["reason"]
        strict = group.get("strict", True)
        run = group.get("run", True)
        tests = group["tests"]
        condition = True
        if "condition" in group:
            condition = create_version_condition(group["condition"])
        marker = markers.get(group.get("marker", None), None)

        config = {
            "reason": reason,
            "strict": strict,
            "run": run,
            "condition": condition,
            "extra_marker": marker,
        }

        for test_id in tests:
            xfail_configs[test_id].append(config)

    # Track which xfail test IDs were actually found
    found_xfail_tests = set()

    for item in items:
        test_id = f"{item.module.__name__}::{item.name}"
        if test_id in xfail_configs:
            found_xfail_tests.add(test_id)
            for config in xfail_configs[test_id]:
                # Add the xfail marker
                item.add_marker(
                    pytest.mark.xfail(
                        reason=config["reason"],
                        strict=config["strict"],
                        run=config["run"],
                        condition=config["condition"],
                    )
                )
                # If there's a marker, add it as a proper pytest marker
                if extra_marker := config["extra_marker"]:
                    item.add_marker(extra_marker)

    # Check for xfail entries that don't match any actual tests
    # Only include tests where at least one config has a met condition
    expected_tests = {
        test_id
        for test_id, configs in xfail_configs.items()
        if any(config["condition"] for config in configs)
    }
    missing_tests = expected_tests - found_xfail_tests
    if strict_xfail_list and missing_tests:
        missing_list = sorted(missing_tests)
        print("Did not find the following test ids:")
        print("\n".join(missing_list))

        warnings.warn(
            f"Found {len(missing_list)} xfail entries that don't match any present tests",
            category=UnmatchedXfailTests,
        )


# Equivalent to @pytest.hookimpl(tryfirst=True), without importing the optional
# pytest dependency when users import cuml.accel. Mark xfails and validate full
# collection before pytest applies -k/-m deselection (including xfail markers).
pytest_collection_modifyitems.pytest_impl = {"tryfirst": True}
