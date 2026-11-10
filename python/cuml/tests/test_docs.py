#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Test that every public estimator is listed in the API documentation.

The API reference is maintained by hand in two places, and a new estimator
needs an entry in both:

* ``docs/source/api/cuml.<module>.rst`` -- an ``autosummary`` entry, which is
  what generates the estimator's API page.
* ``docs/source/api/index.rst`` -- a row in one of the curated overview tables.
"""

import re
from pathlib import Path

from cuml.testing.utils import get_all_base_subclasses

API_DOCS = Path(__file__).resolve().parents[3] / "docs" / "source" / "api"


def _check_api_docs_present():
    assert API_DOCS.is_dir(), (
        f"API documentation not found at {API_DOCS}. These tests must be run "
        "from a full repository checkout."
    )


# ``cuml.dask`` mirrors the single-GPU API, counting it would
# let a dask-only entry hide a missing single-GPU one.
DASK_PAGE = "cuml.dask.rst"

# Estimators that are deliberately absent from the API reference, mapped to the
# reason why. Prefer documenting an estimator over adding it here.
EXCLUDED: dict = {}


def _public_estimators():
    """Names of the public estimator classes that should be documented."""
    return {
        cls.__name__
        for cls in get_all_base_subclasses().values()
        # ``Base`` itself and the intermediate ``*Base`` classes are not part
        # of the public API, and the ``*MG`` classes are multi-GPU internals.
        if not (cls.__name__.endswith("MG") or "Base" in cls.__name__)
        and not getattr(cls, "__abstractmethods__", None)
        # Restrict to cuml's own classes. Test modules define dummy ``Base``
        # subclasses that otherwise leak in once they have been imported.
        and cls.__module__.startswith("cuml.")
    }


def _autosummary_entries(text: str):
    """Names listed in the ``autosummary`` blocks of an rst document.

    Entries may be dotted to reach a submodule (``hdbscan.HDBSCAN``); only the
    final component is returned.
    """
    entries = set()
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        if lines[index].strip() != ".. autosummary::":
            index += 1
            continue

        # Consume the directive's indented body, which holds its options
        # (``:nosignatures:``, ...) followed by one object name per line.
        index += 1
        while index < len(lines):
            line = lines[index]
            if not line.strip():
                index += 1
                continue
            if not line.startswith(" "):
                break
            entry = line.strip()
            if not entry.startswith(":"):
                if not re.fullmatch(r"[A-Za-z_][\w.]*", entry):
                    break
                entries.add(entry.rpartition(".")[2])
            index += 1
    return entries


def _module_pages():
    """Map each per-module rst page to the names it documents."""
    return {
        path.name: _autosummary_entries(path.read_text())
        for path in sorted(API_DOCS.glob("cuml.*.rst"))
        if path.name != DASK_PAGE
    }


def _index_entries() -> set[str]:
    """Names referenced by the overview tables in ``index.rst``."""
    targets = re.findall(
        r":obj:`~(cuml\.[\w.]+)`", (API_DOCS / "index.rst").read_text()
    )
    return {
        target.rpartition(".")[2]
        for target in targets
        if not target.startswith("cuml.dask.")
    }


def _check_excluded_is_not_stale(estimators: set[str]) -> None:
    stale = set(EXCLUDED) - estimators
    assert not stale, (
        "EXCLUDED lists names that are no longer public estimators: "
        + ", ".join(sorted(stale))
        + ". Remove them from EXCLUDED."
    )


def test_estimators_documented_in_module_page():
    """Check that each estimator appears exactly once in a module documentation"""
    _check_api_docs_present()
    estimators = _public_estimators()
    _check_excluded_is_not_stale(estimators)

    pages = _module_pages()
    documented_in = {
        name: sorted(
            page for page, entries in pages.items() if name in entries
        )
        for name in estimators - set(EXCLUDED)
    }

    undocumented = sorted(
        name for name, found in documented_in.items() if not found
    )
    assert not undocumented, (
        "Estimators missing from the API documentation: "
        + ", ".join(undocumented)
        + ". Add each one to the autosummary block in the matching "
        "docs/source/api/cuml.<module>.rst, or add it to EXCLUDED with a "
        "reason."
    )

    duplicated = {
        name: found for name, found in documented_in.items() if len(found) > 1
    }
    assert not duplicated, (
        "Estimators documented on more than one page: "
        + "; ".join(
            f"{name} ({', '.join(found)})"
            for name, found in sorted(duplicated.items())
        )
        + ". Each estimator should be listed on exactly one page."
    )


def test_estimators_documented_in_api_index():
    _check_api_docs_present()
    estimators = _public_estimators()
    _check_excluded_is_not_stale(estimators)

    missing = sorted(estimators - set(EXCLUDED) - _index_entries())
    assert not missing, (
        "Estimators missing from the overview tables in "
        "docs/source/api/index.rst: "
        + ", ".join(missing)
        + ". Add a row for each under the appropriate section, or add it to "
        "EXCLUDED with a reason."
    )
