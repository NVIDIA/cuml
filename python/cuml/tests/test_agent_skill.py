#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Test that the facts in the cuML agent skill (``skills/cuml``) are still true.

The skill tells coding agents about removed cuML APIs whose replacement cannot
be found by inspecting the installed cuML (``scripts/removed_apis.json``), and
where to find the ``cuml.accel`` compatibility documentation. When a PR changes
these, the tests fail until the skill is updated.
"""

import importlib
import importlib.util
import inspect
import json
import re
from pathlib import Path

import pytest

import cuml

REPO = Path(__file__).resolve().parents[3]
SKILL = REPO / "skills" / "cuml"
TABLE = json.loads((SKILL / "scripts" / "removed_apis.json").read_text())
ENTRIES = TABLE["entries"]


def _version(text):
    return tuple(int(x) for x in re.match(r"(\d+)\.(\d+)", text).groups())


INSTALLED = _version(cuml.__version__)


def _resolve(path):
    """Import the longest importable module prefix of ``path`` and getattr the rest."""
    parts = path.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for name in parts[i:]:
            obj = getattr(obj, name)
        return obj
    raise ImportError(path)


def _entry_id(entry):
    return f"{entry['kind']}:{entry['name']}"


@pytest.mark.parametrize("entry", ENTRIES, ids=_entry_id)
def test_removed_api_is_gone(entry):
    """Every API the skill calls removed is really gone in this cuML."""
    if INSTALLED < _version(entry["removed_in"]):
        pytest.skip(f"removed in {entry['removed_in']}")

    name = entry["name"]
    kind = entry["kind"]
    if kind == "module":
        with pytest.raises(ImportError):
            importlib.import_module(name)
    elif kind == "kwarg":
        for path in entry["check_callables"]:
            params = inspect.signature(_resolve(path)).parameters
            assert name not in params, f"{path} still accepts {name!r}"
    elif kind == "build_kwds_key":
        for source in entry["check_sources"]:
            text = (REPO / source).read_text()
            assert f'"{name}"' not in text, f"{source} still reads {name!r}"
    else:
        raise AssertionError(f"unknown kind {kind!r}")


@pytest.mark.parametrize(
    "target",
    sorted({t for e in ENTRIES for t in e["replacement_targets"]}),
)
def test_replacement_exists(target):
    """Every replacement the skill recommends exists."""
    _resolve(target)


def _load_checker():
    path = SKILL / "scripts" / "check_removed_apis.py"
    spec = importlib.util.spec_from_file_location("check_removed_apis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OLD_CODE = """\
import cuml
from cuml.fil import ForestInference
from cuml.svm import SVC
from cuml.manifold import UMAP
import sklearn.svm

sklearn.svm.SVC(probability=True)
SVC(probability=True)
UMAP(build_kwds={"nnd_n_clusters": 4})
"""


def test_checker_flags_removed_apis(tmp_path, capsys):
    """The skill's scanner reports the removed APIs in a snippet of old code."""
    checker = _load_checker()
    script = tmp_path / "old.py"
    script.write_text(OLD_CODE)

    assert checker.main(["--all", "--json", str(script)]) == 1
    findings = json.loads(capsys.readouterr().out)
    found = {(f["line"], f["api"]) for f in findings}
    assert found == {
        (2, "cuml.fil"),
        (8, "probability"),  # line 7 is sklearn's SVC, which keeps it
        (9, "nnd_n_clusters"),
    }


def test_checker_ignores_current_code(tmp_path, capsys):
    checker = _load_checker()
    script = tmp_path / "new.py"
    script.write_text(
        "import nvforest\n"
        "from cuml.manifold import UMAP\n"
        "UMAP(build_kwds={'knn_n_clusters': 4, 'knn_overlap_factor': 2})\n"
        "cuml.set_global_output_type('cupy')\n"
    )
    assert checker.main(["--all", "--json", str(script)]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_compatibility_doc_path():
    """The skill sends agents to this file on GitHub; a rename must update the skill."""
    path = "docs/source/cuml-accel/compatibility.rst"
    assert (REPO / path).is_file()
    assert path in (SKILL / "SKILL.md").read_text()


def test_evals_json_is_valid():
    """The eval dataset is valid SkillEvaluator (agentskills.io) input."""
    data = json.loads((SKILL / "evals" / "evals.json").read_text())
    assert data["skill_name"] == "cuml"
    ids = [case["id"] for case in data["evals"]]
    assert len(ids) == len(set(ids))
    for case in data["evals"]:
        assert case["prompt"].strip()
        assert case["expected_output"].strip()
        assert case["assertions"]
        assert all(isinstance(a, str) and a for a in case["assertions"])
