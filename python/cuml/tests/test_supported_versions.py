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

Constraints must be written the same way, ignoring whitespace and extras.
"""

import re
import tomllib
from functools import cache
from pathlib import Path

import pytest
import yaml

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

# Package name, optional extras, version constraint
_REQUIREMENT = re.compile(r"^([A-Za-z0-9._-]+)(?:\[[^\]]*\])?(.*)$")
# A recipe variable, e.g. '${{ treelite_version_run }}'
_RECIPE_VARIABLE = re.compile(r"\$\{\{\s*(\w+)\s*\}\}")


def _constraints(requirements):
    """Map package name to the constraints it is listed with.

    Entries that don't start with a package name (e.g. the recipe's
    ``${{ pin_compatible(...) }}``) are skipped.
    """
    found = {}
    for requirement in requirements:
        text = requirement.split(";")[0].replace(" ", "")
        if match := _REQUIREMENT.match(text):
            name, constraint = match.groups()
            found.setdefault(name, []).append(constraint)
    return found


@cache
def _pyproject():
    with open(PYPROJECT, "rb") as f:
        return tomllib.load(f)["project"]


@cache
def _recipe_run():
    config = yaml.safe_load(RECIPE_CONFIG.read_text())

    def substitute(match):
        # Only single-valued variables; leave e.g. pin_compatible alone
        values = config.get(match[1])
        return str(values[0]) if values and len(values) == 1 else match[0]

    run = yaml.safe_load(RECIPE.read_text())["requirements"]["run"]
    # Skip 'if:' entries
    return _constraints(
        _RECIPE_VARIABLE.sub(substitute, r) for r in run if isinstance(r, str)
    )


@cache
def _environment(path):
    dependencies = yaml.safe_load(path.read_text())["dependencies"]
    # Skip the 'pip:' sub-list
    return _constraints(d for d in dependencies if isinstance(d, str))


def test_documented_packages_are_checked():
    template = TEMPLATE.read_text()
    pip = set(re.findall(r'\bversions\["([^"]+)"\]', template))
    conda = set(re.findall(r'\bconda_versions\["([^"]+)"\]', template))
    assert pip == set(REQUIRED) | set(OPTIONAL)
    assert conda == set(ALLOWED_MISMATCH)


@pytest.mark.parametrize("pip_name", REQUIRED)
def test_required(pip_name):
    pip = _constraints(_pyproject()["dependencies"])[pip_name]
    conda = _recipe_run()[REQUIRED[pip_name]]
    assert len(pip) == 1, f"{pip_name} listed more than once in {PYPROJECT}"
    assert conda == pip, (
        f"{pip_name}: {conda} in {RECIPE.relative_to(REPO)} differs from "
        f"{pip} in {PYPROJECT.relative_to(REPO)}"
    )


@pytest.mark.parametrize("env_file", ENV_FILES, ids=lambda p: p.name)
@pytest.mark.parametrize("name", OPTIONAL)
def test_optional(name, env_file):
    pip = _constraints(_pyproject()["optional-dependencies"]["test"])[name]
    conda = _environment(env_file)[name]
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
