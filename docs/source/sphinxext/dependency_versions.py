# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render dependency version constraints on ``supported_versions.rst``.

``supported_versions.rst`` is a Jinja template. The curated list of packages
and their explanations live in that page; this extension only fills in the
version constraints.

The constraints are read from the files that define what the cuml packages
require:

* pip: ``python/cuml/pyproject.toml``. ``[project].dependencies`` for required
  packages and ``[project.optional-dependencies]`` for optional ones. This file
  is generated from ``dependencies.yaml`` and kept in sync by the
  ``rapids-dependency-file-generator`` pre-commit hook.
* conda, required: ``requirements.run`` of ``conda/recipes/cuml/recipe.yaml``,
  with ``${{ var }}`` filled in from ``conda_build_config.yaml``. The recipe is
  maintained by hand.
* conda, optional: ``conda/environments/all_cuda-*.yaml``, since optional
  packages are only declared for testing.

The build fails if a package is missing, listed more than once in
pyproject.toml or the recipe, or if its pip and conda constraints differ
(unless the entry sets ``allow_version_mismatch=True``).

The same check runs as a pre-commit hook, so mismatches are reported without
building the docs::

    python docs/source/sphinxext/dependency_versions.py
"""

import glob
import os
import re
import sys
import tomllib
from dataclasses import dataclass

import yaml
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

TEMPLATED_DOCNAME = "supported_versions"

PYPROJECT = os.path.join("python", "cuml", "pyproject.toml")
RECIPE = os.path.join("conda", "recipes", "cuml", "recipe.yaml")
RECIPE_CONFIG = os.path.join(
    "conda", "recipes", "cuml", "conda_build_config.yaml"
)
CONDA_ENVIRONMENTS = os.path.join("conda", "environments", "all_cuda-*.yaml")
# This file, for error messages
_SELF = os.path.join("docs", "source", "sphinxext", "dependency_versions.py")

_NAME = re.compile(r"^\s*([A-Za-z0-9._-]+)")
# Leading package name and optional extras of a requirement string
_NAME_AND_EXTRAS = re.compile(r"^\s*[A-Za-z0-9._-]+\s*(\[[^\]]*\])?\s*")
# A recipe variable reference, e.g. '${{ treelite_version_run }}'
_RECIPE_VARIABLE = re.compile(r"\$\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


@dataclass(frozen=True)
class Package:
    # Name in pyproject.toml. The committed pyproject.toml is the newest CUDA
    # variant, e.g. 'cupy-cuda13x'.
    pip_name: str
    # None: a required package ([project].dependencies and the conda recipe).
    # Otherwise the optional-dependencies key in pyproject.toml.
    extra: str | None = None
    # Name in the conda recipe or environment files, if different
    conda_name: str | None = None
    allow_version_mismatch: bool = False


# Template variable -> package
PACKAGES = {
    # Required runtime dependencies
    "numpy": Package("numpy"),
    "scikit_learn": Package("scikit-learn"),
    "scipy": Package("scipy"),
    "treelite": Package("treelite"),
    "cupy": Package("cupy-cuda13x", conda_name="cupy"),
    # Optional runtime dependencies. These are only declared as test
    # dependencies in dependencies.yaml.
    "xgboost": Package(
        "xgboost",
        extra="test",
        # conda's xgboost>=3.3.0 came from replacing 'rapids-xgboost' in CI
        # (#8440); pip is >=2.1.0. Revisit once it's settled which is intended.
        allow_version_mismatch=True,
    ),
    "hdbscan": Package("hdbscan", extra="test"),
    "umap_learn": Package("umap-learn", extra="test"),
    "pynndescent": Package("pynndescent", extra="test"),
    "numba_cuda_mlir": Package("numba-cuda-mlir", extra="test"),
}


@dataclass(frozen=True)
class Found:
    """A requirement found in a dependency file."""

    requirement: Requirement
    text: str  # version constraint as written, e.g. '>=2.0,<3.0'
    candidate: str  # the whole entry as written, e.g. 'numpy>=2.0,<3.0'


def _find(candidates, name, where):
    """Return every requirement in ``candidates`` named ``name``."""
    matches = []
    for candidate in candidates:
        # Skip conda 'pip:' sub-lists and recipe 'if:' entries
        if not isinstance(candidate, str):
            continue
        match = _NAME.match(candidate)
        if not match or canonicalize_name(match[1]) != canonicalize_name(name):
            continue
        try:
            requirement = Requirement(candidate)
        except InvalidRequirement as e:
            raise ValueError(f"can't parse {candidate!r} in {where}: {e}")
        # SpecifierSet doesn't keep the order of its specifiers, so take the
        # text as written instead.
        text = _NAME_AND_EXTRAS.sub("", candidate).split(";")[0].strip()
        matches.append(Found(requirement, text, candidate))
    return matches


def _find_one(candidates, name, where, key):
    matches = _find(candidates, name, where)
    if len(matches) != 1:
        raise ValueError(
            f"{key}: expected exactly one {name!r} in {where}, found "
            f"{[m.text for m in matches]}. Update PACKAGES in {_SELF}."
        )
    return matches[0]


def _pip_requirement(sources, key, package):
    project = sources["pyproject"]["project"]
    if package.extra is None:
        where = f"[project].dependencies of {PYPROJECT}"
        candidates = project["dependencies"]
    else:
        where = (
            f"[project.optional-dependencies].{package.extra} of {PYPROJECT}"
        )
        try:
            candidates = project["optional-dependencies"][package.extra]
        except KeyError:
            raise ValueError(f"{key}: {where} not found")
    return _find_one(candidates, package.pip_name, where, key)


def _conda_requirements(sources, key, package):
    """Return ``{path: [Found, ...]}`` with the conda constraints."""
    name = package.conda_name or package.pip_name
    if package.extra is None:
        where = f"requirements.run of {RECIPE}"
        found = _find_one(sources["recipe_run"], name, where, key)
        # Name conda_build_config.yaml too if the constraint came from there
        origin = sources["recipe_origins"].get(found.candidate, RECIPE)
        return {origin: [found]}

    found = {}
    for path, packages in sources["environments"].items():
        matches = _find(packages, name, path)
        if not matches:
            raise ValueError(
                f"{key}: {name!r} not found in {path}. Update PACKAGES in "
                f"{_SELF}."
            )
        found[path] = matches
    return found


def render_package(sources, key, package):
    """Render the version constraint text for one entry of ``PACKAGES``."""
    pip = _pip_requirement(sources, key, package)
    conda = _conda_requirements(sources, key, package)

    # The conda environments also contain build and test requirements, so a
    # package may be listed more than once (e.g. treelite's build pin). It's
    # enough for the pip constraint to be one of them.
    if all(
        any(m.requirement.specifier == pip.requirement.specifier for m in ms)
        for ms in conda.values()
    ):
        return pip.text

    conda_texts = {m.text for ms in conda.values() for m in ms}
    if not package.allow_version_mismatch:
        raise ValueError(
            f"{key}: pip constraint {pip.text!r} ({PYPROJECT}) differs from "
            f"conda {sorted(conda_texts)} ({', '.join(sorted(conda))}). Align "
            "them, or set allow_version_mismatch=True for "
            f"{key!r} in PACKAGES of {_SELF} to document both."
        )
    if len(conda_texts) != 1:
        raise ValueError(
            f"{key}: expected one conda constraint to document, found "
            f"{sorted(conda_texts)} in {', '.join(sorted(conda))}."
        )
    return f"pip: {pip.text}; conda: {conda_texts.pop()}"


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_recipe_run(root):
    """Return the recipe's run requirements with variables filled in.

    Also returns ``{filled-in entry: description}`` for entries that used a
    variable from conda_build_config.yaml, for error messages.
    """
    config = _load_yaml(os.path.join(root, RECIPE_CONFIG)) or {}

    def substitute(match):
        values = config.get(match[1])
        if not isinstance(values, list) or len(values) != 1:
            # Leave other expressions (e.g. pin_compatible) unresolved. If
            # one belongs to a documented package, it fails to parse.
            return match[0]
        used.append(match[1])
        return str(values[0])

    run, origins = [], {}
    for entry in _load_yaml(os.path.join(root, RECIPE))["requirements"]["run"]:
        if isinstance(entry, str):
            used = []
            entry = _RECIPE_VARIABLE.sub(substitute, entry)
            if used:
                origins[entry] = (
                    f"{RECIPE} via {', '.join(used)} in {RECIPE_CONFIG}"
                )
        run.append(entry)
    return run, origins


def load_conda_environments(root):
    paths = sorted(glob.glob(os.path.join(root, CONDA_ENVIRONMENTS)))
    if not paths:
        raise ValueError(f"no files match {CONDA_ENVIRONMENTS} in {root}")
    return {
        os.path.relpath(path, root): _load_yaml(path)["dependencies"]
        for path in paths
    }


def load_sources(root):
    with open(os.path.join(root, PYPROJECT), "rb") as f:
        pyproject = tomllib.load(f)
    recipe_run, recipe_origins = load_recipe_run(root)
    return {
        "pyproject": pyproject,
        "recipe_run": recipe_run,
        "recipe_origins": recipe_origins,
        "environments": load_conda_environments(root),
    }


def render_versions(root):
    """Return ``{template variable: rendered text}`` for every package."""
    sources = load_sources(root)
    return {
        key: render_package(sources, key, package)
        for key, package in PACKAGES.items()
    }


def _repo_root(app):
    return os.path.abspath(os.path.join(app.confdir, "..", ".."))


def _render_template(app, docname, source):
    if docname != TEMPLATED_DOCNAME:
        return
    import jinja2

    root = _repo_root(app)
    # Rebuild the page on incremental builds when the inputs change
    for path in [PYPROJECT, RECIPE, RECIPE_CONFIG]:
        app.env.note_dependency(os.path.join(root, path))
    for path in glob.glob(os.path.join(root, CONDA_ENVIRONMENTS)):
        app.env.note_dependency(path)
    template = jinja2.Environment(
        undefined=jinja2.StrictUndefined, keep_trailing_newline=True
    ).from_string(source[0])
    source[0] = template.render(deps=render_versions(root))


def setup(app):
    app.connect("source-read", _render_template)
    return {"parallel_read_safe": True, "parallel_write_safe": True}


def main():
    """Check every package in ``PACKAGES``, for the pre-commit hook."""
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), *[".."] * 3)
    )
    try:
        render_versions(root)
    except ValueError as e:
        print(f"supported_versions.rst: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
