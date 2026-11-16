#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Test that pip and conda declare the same version constraints.

``docs/source/supported_versions.rst.template`` shows the pip constraints from
``python/cuml/pyproject.toml``. The conda requirements are maintained in other
files, so check that they agree for the packages on that page:

* required packages: ``[project].dependencies`` of pyproject.toml vs
  ``requirements.run`` of ``conda/recipes/cuml/recipe.yaml``, with variables
  from ``conda_build_config.yaml`` filled in. The recipe isn't generated from
  dependencies.yaml (rapidsai/dependency-file-generator#7).
* optional packages: the ``test`` extra of pyproject.toml vs
  ``conda/environments/all_cuda-*.yaml``.

Constraints must have the same specifiers (``SpecifierSet`` equality), so
``==4.*,>=4.8.1`` and ``>=4.8.1,<5.0.0`` differ although they allow the same
versions.
"""

import re
import tomllib
from functools import cache
from pathlib import Path

import pytest
import yaml
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

REPO = Path(__file__).resolve().parents[3]
TEMPLATE = REPO / "docs" / "source" / "supported_versions.rst.template"
PYPROJECT = REPO / "python" / "cuml" / "pyproject.toml"
RECIPE = REPO / "conda" / "recipes" / "cuml" / "recipe.yaml"
RECIPE_CONFIG = REPO / "conda" / "recipes" / "cuml" / "conda_build_config.yaml"
ENV_FILES = sorted((REPO / "conda" / "environments").glob("all_cuda-*.yaml"))

# pip name -> conda name
REQUIRED = {
    "numpy": "numpy",
    "scikit-learn": "scikit-learn",
    "scipy": "scipy",
    "treelite": "treelite",
    "cupy-cuda13x": "cupy",
}
OPTIONAL = [
    "xgboost",
    "hdbscan",
    "umap-learn",
    "pynndescent",
    "numba-cuda-mlir",
]

# Packages whose pip and conda constraints differ, with the reason. The docs
# show both constraints for these.
ALLOWED_MISMATCH = {
    "xgboost": (
        "conda's >=3.3.0 came from replacing 'rapids-xgboost' in CI (#8440), "
        "pip is >=2.1.0. Revisit once it's settled which is intended."
    ),
}

# Conda names of the checked packages
CHECKED = {canonicalize_name(n) for n in [*REQUIRED.values(), *OPTIONAL]}

# Leading package name of an entry, whether or not it parses
_NAME = re.compile(r"^\s*([A-Za-z0-9._-]+)")
# A recipe variable, e.g. '${{ treelite_version_run }}'
_RECIPE_VARIABLE = re.compile(r"\$\{\{\s*(\w+)\s*\}\}")


def _constraints(requirements):
    """Map package name to the specifiers it is listed with.

    Entries that aren't PEP 508 requirements (conda-only syntax such as
    ``rapids-logger =0.3`` or the recipe's ``${{ pin_compatible(...) }}``) are
    skipped. ``test_checked_packages_parse`` makes sure that doesn't hide
    entries for checked packages.
    """
    found = {}
    for text in requirements:
        try:
            requirement = Requirement(text)
        except InvalidRequirement:
            continue
        found.setdefault(requirement.name, []).append(requirement.specifier)
    return found


@cache
def _pyproject():
    with open(PYPROJECT, "rb") as f:
        return tomllib.load(f)["project"]


@cache
def _recipe_run():
    """Return the recipe's run requirements with variables filled in."""
    config = yaml.safe_load(RECIPE_CONFIG.read_text())

    def substitute(match):
        # Only single-valued variables; leave e.g. pin_compatible alone
        values = config.get(match[1])
        return str(values[0]) if values and len(values) == 1 else match[0]

    run = yaml.safe_load(RECIPE.read_text())["requirements"]["run"]
    # Skip 'if:' entries
    return [
        _RECIPE_VARIABLE.sub(substitute, r) for r in run if isinstance(r, str)
    ]


@cache
def _environment(path):
    dependencies = yaml.safe_load(path.read_text())["dependencies"]
    # Skip the 'pip:' sub-list
    return [d for d in dependencies if isinstance(d, str)]


def test_documented_packages_are_checked():
    template = TEMPLATE.read_text()
    pip = set(re.findall(r'\bversions\["([^"]+)"\]', template))
    conda = set(re.findall(r'\bconda_versions\["([^"]+)"\]', template))
    assert pip == set(REQUIRED) | set(OPTIONAL)
    assert conda == set(ALLOWED_MISMATCH)


@pytest.mark.parametrize("path", [RECIPE, *ENV_FILES], ids=lambda p: p.name)
def test_checked_packages_parse(path):
    # The other tests skip entries that don't parse. That must not hide an
    # entry for a checked package, e.g. conda-only 'treelite =4.8'.
    entries = _recipe_run() if path == RECIPE else _environment(path)
    invalid = []
    for text in entries:
        match = _NAME.match(text)
        if match and canonicalize_name(match[1]) in CHECKED:
            try:
                Requirement(text)
            except InvalidRequirement:
                invalid.append(text)
    assert not invalid, (
        f"{invalid} in {path.relative_to(REPO)} aren't PEP 508 "
        "requirements, write them as e.g. 'treelite>=4.8.1,<5.0.0'"
    )


@pytest.mark.parametrize("pip_name", REQUIRED)
def test_required(pip_name):
    pip = _constraints(_pyproject()["dependencies"])[pip_name]
    found = _constraints(_recipe_run())
    assert REQUIRED[pip_name] in found, (
        f"{REQUIRED[pip_name]} not found in {RECIPE.relative_to(REPO)}"
    )
    conda = found[REQUIRED[pip_name]]
    assert len(pip) == 1, f"{pip_name} listed more than once in {PYPROJECT}"
    assert conda == pip, (
        f"{pip_name}: {conda} in {RECIPE.relative_to(REPO)} differs from "
        f"{pip} in {PYPROJECT.relative_to(REPO)}"
    )


@pytest.mark.parametrize("env_file", ENV_FILES, ids=lambda p: p.name)
@pytest.mark.parametrize("name", OPTIONAL)
def test_optional(name, env_file):
    pip = _constraints(_pyproject()["optional-dependencies"]["test"])[name]
    found = _constraints(_environment(env_file))
    assert name in found, f"{name} not found in {env_file.relative_to(REPO)}"
    conda = found[name]
    assert len(pip) == 1, f"{name} listed more than once in {PYPROJECT}"
    if name in ALLOWED_MISMATCH:
        assert conda != pip, (
            f"{name} now matches, remove it from ALLOWED_MISMATCH"
        )
    else:
        assert conda == pip, (
            f"{name}: {conda} in {env_file.relative_to(REPO)} differs from "
            f"{pip} in {PYPROJECT.relative_to(REPO)}"
        )
