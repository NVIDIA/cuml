# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render dependency version constraints from ``dependencies.yaml``.

``supported_versions.rst`` is a Jinja template. The curated list of packages
and their explanations live in that page; this extension only fills in the
version constraints so they cannot drift from ``dependencies.yaml``.

Each entry in ``PACKAGES`` names where a package's pip and conda requirements
live in ``dependencies.yaml``. When pip and conda only differ in package name
(e.g. ``cupy-cuda12x`` vs ``cupy``), both names are shown with a single
version constraint. When their version constraints differ, the build fails
unless the entry sets ``allow_version_mismatch=True``. This keeps CI-only pins
from silently becoming documented requirements.
"""

import os
import re
from dataclasses import dataclass

import jinja2
import yaml
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

TEMPLATED_DOCNAME = "supported_versions"

# Leading package name and optional extras of a requirement string
_NAME_AND_EXTRAS = re.compile(r"^\s*[A-Za-z0-9._-]+\s*(\[[^\]]*\])?\s*")


@dataclass(frozen=True)
class Lookup:
    """Location of one requirement in ``dependencies.yaml``.

    ``matrix`` must equal the selector of a ``specific`` entry exactly
    (``{}`` selects the fallback entry). Requirements from ``common`` entries
    match regardless of ``matrix``.
    """

    dependency_set: str
    name: str
    output_type: str
    matrix: tuple = ()


@dataclass(frozen=True)
class Package:
    # (label, lookup) pairs; label is shown next to the pip name, e.g. "CUDA 12"
    pip: tuple
    conda: Lookup
    allow_version_mismatch: bool = False


def _same_name(dependency_set, name):
    """A package with the same name and constraint for pip and conda."""
    return Package(
        pip=((None, Lookup(dependency_set, name, "requirements")),),
        conda=Lookup(dependency_set, name, "conda"),
    )


def _per_cuda(dependency_set, pip_names, conda_name, pip_matrices, **kwargs):
    """A package whose pip name depends on the CUDA major version."""
    return Package(
        pip=tuple(
            (
                f"CUDA {major}",
                Lookup(dependency_set, pip_name, "requirements", matrix),
            )
            for major, pip_name, matrix in zip(
                ("12", "13"), pip_names, pip_matrices
            )
        ),
        conda=Lookup(dependency_set, conda_name, "conda"),
        **kwargs,
    )


_CUDA_SUFFIXED = (
    (("cuda", "12.*"), ("cuda_suffixed", "true")),
    (("cuda", "13.*"), ("cuda_suffixed", "true")),
)

# Template variable -> where to find the package in dependencies.yaml.
PACKAGES = {
    # Required runtime dependencies
    "numpy": _same_name("py_run_cuml", "numpy"),
    "scikit_learn": _same_name("py_run_cuml", "scikit-learn"),
    "scipy": _same_name("py_run_cuml", "scipy"),
    "treelite": _same_name("py_run_cuml", "treelite"),
    "cupy": _per_cuda(
        "depends_on_cupy",
        ("cupy-cuda12x", "cupy-cuda13x"),
        "cupy",
        # CUDA 13 is the fallback entry (no selector)
        ((("cuda", "12.*"),), ()),
    ),
    # Optional runtime dependencies. These are only declared as test
    # dependencies in dependencies.yaml.
    "xgboost": _per_cuda(
        "test_python_xgboost",
        ("xgboost-cu12", "xgboost-cu13"),
        "xgboost",
        _CUDA_SUFFIXED,
        # conda's xgboost>=3.3.0 came from replacing 'rapids-xgboost' in CI
        # (#8440); pip is >=2.1.0. Revisit once it's settled which is intended.
        allow_version_mismatch=True,
    ),
    "hdbscan": _same_name("test_python", "hdbscan"),
    "umap_learn": _same_name("test_python", "umap-learn"),
    "pynndescent": _same_name("test_python", "pynndescent"),
    "numba_cuda_mlir": _per_cuda(
        "depends_on_numba_cuda_mlir",
        ("numba-cuda-mlir", "numba-cuda-mlir"),
        "numba-cuda-mlir",
        _CUDA_SUFFIXED,
    ),
}


def _as_list(value):
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def _candidate_requirements(dependency_set, lookup):
    """Yield requirement strings that apply to ``lookup``."""
    for entry in dependency_set.get("common", []):
        if lookup.output_type in _as_list(entry["output_types"]):
            yield from _as_list(entry["packages"])
    for entry in dependency_set.get("specific", []):
        if lookup.output_type not in _as_list(entry["output_types"]):
            continue
        for matrix_entry in entry["matrices"]:
            if (matrix_entry["matrix"] or {}) == dict(lookup.matrix):
                yield from _as_list(matrix_entry["packages"])
                break


@dataclass(frozen=True)
class Found:
    """A requirement found in ``dependencies.yaml``."""

    name: str  # package name including extras, e.g. 'numba-cuda-mlir[cu12]'
    specifier: SpecifierSet
    text: str  # version constraint as written in dependencies.yaml


def find_requirement(dependencies, lookup):
    """Return the single requirement in ``dependencies`` matching ``lookup``."""
    try:
        dependency_set = dependencies[lookup.dependency_set]
    except KeyError:
        raise ValueError(
            f"dependency set {lookup.dependency_set!r} not found in "
            "dependencies.yaml"
        ) from None

    matches = []
    for candidate in _candidate_requirements(dependency_set, lookup):
        # Skip conda 'pip:' sub-lists and pip options like '--extra-index-url'
        if not isinstance(candidate, str) or candidate.startswith("-"):
            continue
        try:
            requirement = Requirement(candidate)
        except InvalidRequirement:
            # conda match specs (e.g. 'doxygen=1.18.0') aren't PEP 508
            continue
        if canonicalize_name(requirement.name) != canonicalize_name(
            lookup.name
        ):
            continue
        extras = ",".join(sorted(requirement.extras))
        matches.append(
            Found(
                name=requirement.name + (f"[{extras}]" if extras else ""),
                specifier=requirement.specifier,
                # SpecifierSet doesn't keep the order of its specifiers, so
                # take the text as written in dependencies.yaml instead.
                text=_NAME_AND_EXTRAS.sub("", candidate).split(";")[0].strip(),
            )
        )

    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one requirement for {lookup}, found "
            f"{matches}. Update PACKAGES in {__file__} to match "
            "dependencies.yaml."
        )
    return matches[0]


def _labeled(name, label):
    return f"``{name}`` ({label})" if label else f"``{name}``"


def render_package(dependencies, key, package):
    """Render the version constraint text for one entry of ``PACKAGES``."""
    pip = [
        (label, find_requirement(dependencies, lookup))
        for label, lookup in package.pip
    ]
    conda = find_requirement(dependencies, package.conda)

    if len({found.specifier for _, found in pip}) != 1:
        raise ValueError(
            f"{key}: pip version constraints differ across variants: {pip}"
        )
    pip_text = pip[0][1].text
    pip_names = ", ".join(_labeled(found.name, label) for label, found in pip)

    if pip[0][1].specifier != conda.specifier:
        if not package.allow_version_mismatch:
            raise ValueError(
                f"{key}: pip ({pip_text}) and conda ({conda.text}) version "
                "constraints differ in dependencies.yaml. Align them, or set "
                f"allow_version_mismatch=True for {key!r} in {__file__} to "
                "document both."
            )
        return f"pip: {pip_names} {pip_text}; conda: ``{conda.name}`` {conda.text}"

    if all(found.name == conda.name for _, found in pip):
        return pip_text
    return f"{pip_text} (pip: {pip_names}; conda: ``{conda.name}``)".lstrip()


def load_dependencies(path):
    with open(path) as f:
        return yaml.safe_load(f)["dependencies"]


def render_versions(path):
    """Return ``{template variable: rendered text}`` for every package."""
    dependencies = load_dependencies(path)
    return {
        key: render_package(dependencies, key, package)
        for key, package in PACKAGES.items()
    }


def _dependencies_path(app):
    return os.path.abspath(
        os.path.join(app.confdir, "..", "..", "dependencies.yaml")
    )


def _render_template(app, docname, source):
    if docname != TEMPLATED_DOCNAME:
        return
    path = _dependencies_path(app)
    # Rebuild the page on incremental builds when dependencies.yaml changes
    app.env.note_dependency(path)
    template = jinja2.Environment(
        undefined=jinja2.StrictUndefined, keep_trailing_newline=True
    ).from_string(source[0])
    source[0] = template.render(deps=render_versions(path))


def setup(app):
    app.connect("source-read", _render_template)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
