# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Critical contracts for benchmark workloads, measurements, and recovery."""

from __future__ import annotations

import ast
import contextlib
import copy
import datetime as dt
import importlib
import inspect
import json
import multiprocessing
import os
import subprocess
import sys
import time
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import jsonschema
import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import sparse

from cuml import benchmark
from cuml.benchmark import _runner as coordinator
from cuml.benchmark import _subprocess as runner
from cuml.benchmark import cli, harness, scoring
from cuml.benchmark._hashing import canonical_json
from cuml.benchmark._selection import compile_expression, select_plan
from cuml.benchmark.backends import base as base_backend
from cuml.benchmark.backends import cuml as cuml_backend
from cuml.benchmark.backends import get_backend
from cuml.benchmark.backends.accel import ACCEL_EXTENSION
from cuml.benchmark.backends.base import Backend, GPUBackend
from cuml.benchmark.datasets import generate_data, partition_data
from cuml.benchmark.providers import PROVIDERS, Provider, get_provider
from cuml.benchmark.providers.base import EstimatorSpec
from cuml.benchmark.schemas import benchmark_result_schema
from cuml.benchmark.suite import (
    BUILTIN_SUITES,
    Suite,
    SuiteError,
    Workload,
    load_suite,
    load_suite_reference,
    resolve_case,
)

SUITES = Path(__file__).resolve().parents[1] / "cuml" / "benchmark" / "suites"
PROFILE = {"warmups": 1, "repetitions": 2, "size_scale": 1}
PACKAGE = {
    "name": "scikit-learn",
    "version": "test",
    "build": None,
    "source": None,
}
ACCEL_BENCHMARK_EXCLUSIONS = {
    "IsolationForest": "No benchmark registry or suite entry is defined yet.",
}
BENCHMARK_EXCLUSIONS = {
    "CD": "Low-level solver API is outside the estimator benchmark suite.",
    "IsolationForest": "No benchmark registry entry is defined yet.",
    "Lars": "No benchmark registry entry is defined yet.",
    "MBSGDClassifier": "No benchmark registry or suite entry is defined yet.",
    "QN": "Low-level solver API is outside the estimator benchmark suite.",
    "SGD": "Low-level solver API is outside the estimator benchmark suite.",
}


def _request(**changes):
    return {
        "estimator": "PCA",
        "operation": "fit",
        "input_selection": ["X"],
        "parameters": {"n_components": 2},
        "dataset": {
            "generator": "matrix",
            "shape": {"rows": 64, "features": 8},
        },
        **changes,
    }


def _suite(*cases, provider="scikit-learn"):
    return Suite("test", "test", provider, "standard", tuple(cases))


def _benchmark(case, provider="scikit-learn", **kwargs):
    return harness._benchmark_case(
        _suite(case, provider=provider),
        case,
        implementation_record=PACKAGE,
        **kwargs,
    )


def _write_suite(tmp_path, document):
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def _validate_artifact(artifact):
    schema = json.loads(benchmark_result_schema().read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    ).validate(artifact)
    for result in artifact["results"]:
        assert result["id"] == Workload.from_artifact_fields(result).digest()
        assert (
            result["implementation"] in artifact["run"]["software"]["packages"]
        )


@pytest.fixture
def document():
    return {
        "version": 2,
        "name": "test",
        "providers": ["scikit-learn"],
        "profiles": {"standard": dict(PROFILE)},
        "cases": [_request()],
    }


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("", True),
        ("  ", True),
        ("pCa", True),
        ("pca.fit", True),
        ("matrix and rows=64 and features=8 and dtype=float32", True),
        ("PCA or UMAP and predict", True),
        ("(PCA or UMAP) and predict", False),
        ("not (UMAP or predict)", True),
        ("not not PCA", True),
        ("PCA and not fit", False),
        ("scikit-learn", False),
        ("n_components", False),
    ],
)
def test_keyword_expression(expression, expected):
    case = resolve_case(_request(), PROFILE)
    assert compile_expression(expression)(case) is expected
    assert compile_expression(case.label)(case)
    assert compile_expression(case.workload_id())(case)


@pytest.mark.parametrize(
    "expression",
    [
        "PCA UMAP",
        "PCA and",
        "or PCA",
        "()",
        "(PCA",
        "PCA)",
        "not",
        "PCA or or fit",
    ],
)
def test_invalid_keyword_expression(expression):
    with pytest.raises(SuiteError, match="invalid -k expression"):
        compile_expression(expression)


def test_keyword_selection_resolved_profiles(document, tmp_path):
    document["profiles"]["standard"]["size_scale"] = 0.5
    document["cases"][0]["dataset"]["shape"].update(rows=200, features=16)
    document["cases"].append(_request(estimator="TruncatedSVD"))
    plan = load_suite(_write_suite(tmp_path, document))
    selected = select_plan(plan, "rows=100 and features=16")
    assert selected.runs[0].cases == (plan.runs[0].cases[0],)
    assert selected.runs[0].cases[0] is plan.runs[0].cases[0]
    assert select_plan(plan, "").runs == plan.runs
    with pytest.raises(SuiteError, match="no benchmark cases match"):
        select_plan(plan, "rows=64")


def test_cli_collect_only(document, tmp_path, monkeypatch, capsys):
    document["providers"] = ["cuml", "scikit-learn"]
    document["cases"][0]["providers"] = ["cuml"]
    document["cases"].append(
        _request(estimator="TruncatedSVD", providers=["scikit-learn"])
    )
    path = _write_suite(tmp_path, document)
    output = tmp_path / "untouched"
    monkeypatch.chdir(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail(
            "collection must not prepare backends or execute benchmarks"
        )

    monkeypatch.setattr(cli, "_run_plan", forbidden)
    monkeypatch.setattr(cli, "default_output_path", forbidden)
    monkeypatch.setattr(harness, "run_suite", forbidden)
    monkeypatch.setattr(harness, "generate_data", forbidden)
    monkeypatch.setattr(
        cuml_backend.CumlBackend, "bootstrap_process", forbidden
    )
    monkeypatch.setattr(cuml_backend.CumlBackend, "prepare_process", forbidden)
    argv = [
        "--suite",
        str(path),
        "--provider",
        "cuml",
        "--provider",
        "scikit-learn",
        "-k",
        "pca",
        "--collect-only",
    ]
    assert cli.main(argv) == 0
    first = capsys.readouterr().out
    assert "cuml — 1 selected / 1 available" in first
    assert "scikit-learn — 0 selected / 1 available" in first
    assert (
        "PCA.fit  dataset=matrix  rows=64  features=8  dtype=float32" in first
    )
    assert "Collected 1 case across 1 provider." in first
    assert cli.main([*argv, "--output", str(output)]) == 0
    assert capsys.readouterr().out == first
    assert not output.exists()
    assert not (tmp_path / ".benchmarks").exists()
    with pytest.raises(SystemExit) as exc:
        cli.main([*argv, "--resume", "--output", str(output)])
    assert exc.value.code == 2


@pytest.mark.parametrize("expression", ["PCA", "-missing or PCA"])
def test_cli_keyword_workers_and_resume(
    document, tmp_path, monkeypatch, expression
):
    monkeypatch.delenv("CUML_ACCEL_ENABLED", raising=False)
    document["cases"].append(_request(estimator="TruncatedSVD"))
    path = _write_suite(tmp_path, document)
    output = tmp_path / "results"
    argv = [
        "--suite",
        str(path),
        "--provider",
        "scikit-learn",
        "--output",
        str(output),
    ]
    expected = select_plan(load_suite(path), "PCA").runs[0].cases
    assert cli.main([*argv, f"-k={expression}"]) == 0
    artifact_path = output / "scikit-learn.json"
    artifact = json.loads(artifact_path.read_text())
    assert [result["id"] for result in artifact["results"]] == [
        case.workload_id() for case in expected
    ]
    assert f"-k={expression}" in artifact["run"]["command"]["argv"]
    assert cli.main([*argv, "-k", "PCA and not UMAP", "--resume"]) == 0
    resumed = json.loads(artifact_path.read_text())
    assert resumed["results"] == artifact["results"]
    assert resumed["run"]["id"] == artifact["run"]["id"]
    assert cli.main([*argv, "--resume"]) == 1
    assert json.loads(artifact_path.read_text()) == resumed


@pytest.mark.parametrize("expression", ["PCA and", "unknown-estimator"])
def test_cli_keyword_errors_do_not_write(document, tmp_path, expression):
    path = _write_suite(tmp_path, document)
    output = tmp_path / "untouched"
    for flags in [[], ["--collect-only"]]:
        with pytest.raises(SystemExit) as exc:
            cli.main(
                [
                    "--suite",
                    str(path),
                    "--provider",
                    "scikit-learn",
                    "--output",
                    str(output),
                    "-k",
                    expression,
                    *flags,
                ]
            )
        assert exc.value.code == 2
        assert not output.exists()


def test_provider_qualified_estimator_workers(document, tmp_path, monkeypatch):
    """Two CPU providers share a workload, not an estimator binding."""
    # Covers cases such as HDBSCAN in scikit-learn and the hdbscan package,
    # without depending on either library's estimator implementation.
    executed = []

    def estimator(provider):
        class SharedEstimator:
            def get_params(self, deep=False):
                return {}

            def fit(self, X):
                executed.append(provider)
                return self

        return SharedEstimator

    providers = ["provider_a", "provider_b"]
    modules = [f"benchmark_test_{provider}" for provider in providers]
    for provider, name in zip(providers, modules, strict=True):
        module = ModuleType(name)
        module.SharedEstimator = estimator(provider)
        monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setitem(
            PROVIDERS,
            provider,
            Provider(
                get_backend("cpu"),
                {
                    "SharedEstimator": EstimatorSpec(
                        name, "SharedEstimator", provider
                    )
                },
            ),
        )

    # Synthetic registrations are process-local. Exercise the CLI worker path
    # in-process; test_cli_real_workers covers real interpreter isolation.
    def execute(command, environment):
        assert cli.main(command[command.index("--_worker") :]) == 0

    monkeypatch.setattr(coordinator, "run_command", execute)
    document["providers"] = providers
    document["cases"] = [_request(estimator="SharedEstimator", parameters={})]
    path = _write_suite(tmp_path, document)
    plan = load_suite(path)
    assert [run.provider for run in plan.runs] == document["providers"]
    assert (
        plan.runs[0].cases[0].workload().digest()
        == plan.runs[1].cases[0].workload().digest()
    )
    assert [
        run.provider_spec.estimator_spec("SharedEstimator").module
        for run in plan.runs
    ] == modules
    selected = load_suite(path, providers=list(reversed(providers)))
    assert selected == plan
    assert (
        cli.main(
            [
                "--suite",
                str(path),
                "--provider",
                providers[0],
                "--provider",
                providers[1],
                "--output",
                str(tmp_path / "results"),
            ]
        )
        == 0
    )
    assert executed == [
        provider
        for provider in providers
        for _ in range(PROFILE["warmups"] + PROFILE["repetitions"])
    ]
    for provider in document["providers"]:
        artifact = json.loads(
            (tmp_path / "results" / f"{provider}.json").read_text()
        )
        _validate_artifact(artifact)
        result = artifact["results"][0]
        assert result["outcome"]["status"] == "success"
        assert result["implementation"]["name"] == provider
        extension = result["extensions"][harness.EXTENSION]
        assert extension["provider"] == provider
        run_extension = artifact["run"]["extensions"][harness.EXTENSION]
        assert run_extension["provider"] == provider


def test_mixed_library_suite_provider_applicability(document, tmp_path):
    document["providers"] = ["scikit-learn", "umap-learn", "hdbscan", "cuml"]
    document["cases"] = [
        _request(
            estimator="UMAP", parameters={}, providers=["umap-learn", "cuml"]
        ),
        _request(
            estimator="HDBSCAN",
            parameters={},
            providers=["scikit-learn", "hdbscan", "cuml"],
        ),
    ]
    path = _write_suite(tmp_path, document)
    plan = load_suite(path)
    assert "UMAP" not in get_provider("scikit-learn").catalog
    assert (
        plan.runs[1].provider_spec.estimator_spec("UMAP").package
        == "umap-learn"
    )
    assert (
        get_provider("cuml.accel").estimator_spec("HDBSCAN").module
        == "hdbscan"
    )
    assert [[case.estimator for case in run.cases] for run in plan.runs] == [
        ["HDBSCAN"],
        ["UMAP"],
        ["HDBSCAN"],
        ["UMAP", "HDBSCAN"],
    ]
    selected = load_suite(path, providers=["hdbscan", "umap-learn"])
    assert selected.runs == (plan.runs[1], plan.runs[2])
    # Applicability is explicit, not an implicit catalog intersection.
    document["cases"][0].pop("providers")
    with pytest.raises(SuiteError, match="incompatible with 'scikit-learn'"):
        load_suite(_write_suite(tmp_path, document))


# Suite validation and workload definitions. No estimator execution here.


def test_estimator_benchmark_coverage():
    import cuml
    from cuml.internals.base import Base
    from cuml.testing.utils import get_all_base_subclasses

    public = {
        name
        for name, cls in get_all_base_subclasses().items()
        if getattr(cuml, name, None)
        is cls  # Cover estimators exported at the top level.
        and not name.startswith("_")
        and cls is not Base
        and not inspect.isabstract(cls)
    }
    assert BENCHMARK_EXCLUSIONS.keys() <= public
    assert all(reason.strip() for reason in BENCHMARK_EXCLUSIONS.values())
    manifest = {
        c.estimator
        for c in load_suite_reference("estimators", providers=["cuml"])
        .runs[0]
        .cases
    }
    covered = {
        spec.name
        for key, spec in get_provider("cuml").catalog.items()
        if key in manifest
    }
    missing = public - BENCHMARK_EXCLUSIONS.keys() - covered
    assert not missing, f"Missing benchmark declarations: {sorted(missing)}"

    overrides = SUITES.parents[1] / "accel" / "_overrides"
    exported = set()
    for path in sorted(overrides.rglob("*.py")):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets
                    if isinstance(node, ast.Assign)
                    else [node.target]
                )
                if any(
                    isinstance(t, ast.Name) and t.id == "__all__"
                    for t in targets
                ):
                    exported.update(ast.literal_eval(node.value))
    assert ACCEL_BENCHMARK_EXCLUSIONS.keys() <= exported
    assert all(
        reason.strip() for reason in ACCEL_BENCHMARK_EXCLUSIONS.values()
    )
    expected = exported - ACCEL_BENCHMARK_EXCLUSIONS.keys()
    assert set(get_provider("cuml.accel").catalog) == expected
    assert {
        c.estimator
        for c in load_suite_reference("estimators", providers=["cuml.accel"])
        .runs[0]
        .cases
    } == expected


@pytest.mark.parametrize(
    "tags",
    [[], ["balanced"], ["wide", "diagnostic"], ["thin", "wide"], ["", " "]],
)
def test_case_tags_are_metadata(tags):
    case = resolve_case(_request(tags=tags), PROFILE)
    assert case.tags == tuple(tags)
    assert (
        case.workload_id() == resolve_case(_request(), PROFILE).workload_id()
    )
    result = _benchmark(case)
    assert result["extensions"][harness.EXTENSION]["tags"] == tags


def test_tags_preserved_on_failure_and_across_profiles():
    request = _request(tags=["balanced"])
    case = resolve_case(request, PROFILE)
    failed = harness._base_result(_suite(case), case, PACKAGE)
    assert failed["outcome"]["status"] == "failed"
    assert failed["extensions"][harness.EXTENSION]["tags"] == ["balanced"]
    smoke = resolve_case(request, {**PROFILE, "size_scale": 0.01})
    assert smoke.tags == case.tags
    assert resolve_case(_request(), PROFILE).tags == ()


@pytest.mark.parametrize("tags", [["thin", "thin"], ["", ""], [" ", " "]])
def test_duplicate_case_tags(tags):
    with pytest.raises(SuiteError, match="duplicate tags"):
        resolve_case(_request(tags=tags), PROFILE)


def test_packaged_suites_are_valid_and_roundtrip():
    for path in sorted(SUITES.glob("*.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        for profile in document["profiles"]:
            plan = load_suite(path, profile)
            assert [run.provider for run in plan.runs] == document["providers"]
            for suite in plan.runs:
                assert suite.cases
                assert len(
                    {c.workload().digest() for c in suite.cases}
                ) == len(suite.cases)
                if path.stem in BUILTIN_SUITES:
                    builtin = load_suite_reference(
                        path.stem, profile, [suite.provider]
                    )
                    assert builtin.runs[0].cases == suite.cases
                    assert builtin.runs[0].path == f"builtin:{path.stem}"
                for case in suite.cases:
                    # A suite may contain multiple shapes/parameter settings
                    # per estimator, including provider-specific parameters.
                    identity = case.workload()
                    assert identity == Workload.from_artifact_fields(
                        case.to_artifact_fields()
                    )
    assert {p.stem for p in SUITES.glob("*.yaml")} >= BUILTIN_SUITES


@pytest.mark.parametrize(
    "timeout,expected", [("omitted", 10), (None, None), (3, 3)]
)
@pytest.mark.parametrize(
    "scale,partitions", [(0.01, (33, 1, 32)), (0.5, (76, 26, 50))]
)
def test_python_and_yaml_resolution_agree(
    document, tmp_path, timeout, expected, scale, partitions
):
    request = document["cases"][0]
    request.update(operation="transform", fit_input_selection=["X"])
    request["dataset"].update(dtype={"X": "float64", "y": "int64"})
    request["dataset"]["shape"].update(rows=101, train_rows=51)
    profile = document["profiles"]["standard"]
    profile.update(size_scale=scale, timeout_sec=10)
    if timeout != "omitted":
        request["timeout_sec"] = timeout
    before = copy.deepcopy(document)
    case = resolve_case(request, profile)
    assert load_suite(_write_suite(tmp_path, document)).runs[0].cases == (
        case,
    )
    assert (
        case.generated_rows,
        case.training_rows,
        case.measured_rows,
    ) == partitions
    assert case.features == 8
    assert case.dtypes == {"X": "float64", "y": "int64"}
    assert case.timeout_sec == expected
    assert document == before
    request["parameters"]["n_components"] = 3
    request["dataset"]["dtype"]["y"] = "float32"
    assert case.parameters == {"n_components": 2}
    assert case.dtypes["y"] == "int64"
    assert (
        resolve_case(request, profile).workload().digest()
        != case.workload().digest()
    )


@pytest.mark.parametrize(
    "change,match",
    [
        ("missing", "input_selection"),
        ("unknown", "unknown field"),
        ("boolean", "rows"),
        ("zero", "repetitions"),
        ("nonfinite", "timeout_sec"),
        ("estimator", "incompatible"),
        ("generator", "must not exceed features"),
        ("fit_inputs", "fit_input_selection is required"),
        ("train_rows", "train_rows is required"),
        ("duplicate", "duplicate case identity"),
        ("accel_warmup", "at least one warmup"),
    ],
)
def test_invalid_suite_requests_are_rejected(
    document, tmp_path, change, match
):
    case = document["cases"][0]
    profile = document["profiles"]["standard"]
    if change == "missing":
        del case["input_selection"]
    elif change == "unknown":
        case["dataset"]["shape"]["typo"] = 1
    elif change == "boolean":
        case["dataset"]["shape"]["rows"] = True
    elif change == "zero":
        profile["repetitions"] = 0
    elif change == "nonfinite":
        profile["timeout_sec"] = float("inf")
    elif change == "estimator":
        case["estimator"] = "Unknown"
    elif change == "generator":
        case["dataset"].update(
            generator="classification", parameters={"n_informative": 9}
        )
    elif change in {"fit_inputs", "train_rows"}:
        case["operation"] = "transform"
        if change == "train_rows":
            case["fit_input_selection"] = ["X"]
    elif change == "duplicate":
        duplicate = copy.deepcopy(case)
        duplicate["dataset"].update(
            dtype="float32", format="dense", parameters={}
        )
        document["cases"].append(duplicate)
    else:
        document["providers"] = ["cuml.accel"]
        profile["warmups"] = 0
    with pytest.raises(SuiteError, match=match):
        load_suite(_write_suite(tmp_path, document))
    if change not in {"estimator", "duplicate", "accel_warmup"}:
        with pytest.raises(SuiteError, match=match):
            resolve_case(case, profile)


@pytest.mark.parametrize("contents", ["cases: [", "[]", "version: 2\ntyop: 1"])
def test_suite_yaml_decode_errors(tmp_path, contents):
    path = tmp_path / "invalid.yaml"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(SuiteError, match="unable to load suite|invalid suite"):
        load_suite(path)


def test_suite_yaml_missing_file(tmp_path):
    with pytest.raises(SuiteError, match="unable to load suite"):
        load_suite(tmp_path / "missing.yaml")


def test_suite_provider_selection(document, tmp_path):
    document["providers"] = ["cuml", "scikit-learn", "cuml.accel"]
    restricted = _request(estimator="AgglomerativeClustering", parameters={})
    restricted["providers"] = ["cuml", "scikit-learn"]
    document["cases"].append(restricted)
    path = _write_suite(tmp_path, document)
    plan = load_suite(path)
    assert [len(run.cases) for run in plan.runs] == [2, 2, 1]
    assert len({run.cases[0].workload().digest() for run in plan.runs}) == 1
    selected = load_suite(path, providers=["cuml.accel", "cuml"])
    assert selected.runs == (plan.runs[0], plan.runs[2])
    document["profiles"]["standard"]["warmups"] = 0
    path = _write_suite(tmp_path, document)
    assert load_suite(path, providers=["scikit-learn"])
    with pytest.raises(SuiteError, match="at least one warmup"):
        load_suite(path)


@pytest.mark.parametrize(
    "change,match",
    [
        ("unknown", "invalid provider"),
        ("duplicate", "duplicate providers"),
        ("empty", "length >= 1"),
        ("scalar", "unknown field|missing required field"),
        ("case_unknown", "invalid provider"),
        ("case_duplicate", "duplicate providers"),
        ("case_empty", "length >= 1"),
        ("case_outside", "subset"),
        ("no_cases", "no applicable cases"),
    ],
)
def test_invalid_provider_declarations(document, tmp_path, change, match):
    if change == "scalar":
        document["provider"] = document.pop("providers")[0]
    elif change == "no_cases":
        document["providers"].append("cuml")
        document["cases"][0]["providers"] = ["scikit-learn"]
    else:
        values = {
            "unknown": ["unknown"],
            "duplicate": ["scikit-learn"] * 2,
            "empty": [],
            "outside": ["cuml"],
        }[change.removeprefix("case_")]
        target = (
            document["cases"][0] if change.startswith("case_") else document
        )
        target["providers"] = values
    with pytest.raises(SuiteError, match=match):
        load_suite(_write_suite(tmp_path, document))


@pytest.mark.parametrize(
    "selection", [[], ["cuml"], ["unknown"], ["scikit-learn"] * 2]
)
def test_invalid_provider_selection(document, tmp_path, selection):
    with pytest.raises(SuiteError):
        load_suite(_write_suite(tmp_path, document), providers=selection)


def test_duplicate_workloads_allowed_only_on_disjoint_providers(
    document, tmp_path
):
    document["providers"] = ["cuml", "scikit-learn"]
    document["cases"][0]["providers"] = ["cuml"]
    duplicate = copy.deepcopy(document["cases"][0])
    duplicate["providers"] = ["scikit-learn"]
    document["cases"].append(duplicate)
    assert len(load_suite(_write_suite(tmp_path, document)).runs) == 2
    duplicate["providers"].append("cuml")
    with pytest.raises(SuiteError, match="duplicate case identity"):
        load_suite(_write_suite(tmp_path, document))


def test_workload_identity_contract():
    # This vector is shared with the dashboard, not a snapshot of suite contents.
    golden = {
        "case_label": "display only",
        "algorithm": "kmeans/Δ",
        "dataset": {
            "name": "blobs-雪",
            "kind": "generated",
            "parameters": {
                "clusters": 8,
                "nested": {"scale": 1e-7, "zero": -0.0},
            },
            "generator": "org.example.gen-v1",
            "fingerprint": None,
            "random_seed": 42,
        },
        "operation": {"name": "fit_predict", "lifecycle": "fit"},
        "input": {
            "dimensions": [
                {"name": "rows", "size": 1000},
                {"name": "features", "size": 32},
            ],
            "data_type": "float32",
            "selection": ["X"],
            "attributes": {"ignored": True},
        },
        "parameters": {
            "declared": {
                "tolerance": 1e-6,
                "labels": ["é", "𝄞"],
                "max_iterations": 100,
            },
            "effective": {"ignored": 1},
        },
    }
    assert canonical_json(
        {"numbers": [333333333.33333329, 1e30, 4.50, 2e-3, 1e-27]}
    ) == ('{"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27]}')
    assert (
        Workload.from_artifact_fields(golden).digest()
        == "sha256:af3343fed9b578a2561c486766d45f38dc1a6c6e5bacc5a9477d97f8a8dc1e28"
    )
    request = _request(operation="transform", fit_input_selection=["X"])
    request["dataset"]["shape"]["train_rows"] = 128
    case = resolve_case(request, PROFILE)
    explicit = copy.deepcopy(request)
    explicit["dataset"].update(dtype="float32", format="dense", parameters={})
    assert (
        resolve_case(explicit, PROFILE).workload().digest()
        == case.workload().digest()
    )
    for changed in (
        replace(case, estimator="Other"),
        replace(case, parameters={"n_components": 3}),
        replace(case, measured_rows=65),
        replace(case, training_rows=129),
        replace(case, dtypes={"X": "float64", "y": "float32"}),
        replace(case, input_format="csr", dataset_parameters={"density": 0.1}),
        replace(case, input_selection=("y",)),
        replace(case, fit_input_selection=("X", "y")),
    ):
        assert changed.workload().digest() != case.workload().digest()
    result = case.to_artifact_fields()
    assert result["dataset"]["kind"] == "generated"
    assert result["dataset"]["name"] == "matrix"
    assert result["dataset"]["generator"] == (
        "com.nvidia.cuml.benchmark.generate-data-v1"
    )
    result.update(
        case_label="display",
        provider={"name": "other"},
        observations=[],
        extensions={},
    )
    result["parameters"]["effective"] = {"n_components": 99}
    assert (
        Workload.from_artifact_fields(result).digest()
        == case.workload().digest()
    )
    assert (
        replace(case, warmups=2, repetitions=5, timeout_sec=10).workload_id()
        == case.workload_id()
    )


def test_workload_is_frozen_and_roundtrips():
    case = resolve_case(
        _request(parameters={"nested": {"values": [1, 2]}}), PROFILE
    )
    identity = case.workload()
    with pytest.raises(FrozenInstanceError):
        identity.algorithm = "Other"

    artifact = json.loads(json.dumps(case.to_artifact_fields()))
    restored = Workload.from_artifact_fields(artifact)
    assert restored == identity
    assert restored.digest() == identity.digest() == case.workload_id()


@pytest.mark.parametrize(
    "kind,dtype,input_format,parameters",
    [
        ("matrix", "float32", "dense", {}),
        ("classification", "float64", "dense", {"n_classes": 3}),
        ("regression", "float32", "dense", {}),
        ("blobs", "float64", "dense", {"centers": 3, "cluster_std": 0.5}),
        ("positive", "float32", "dense", {}),
        ("categorical", {"X": "int32", "y": "int64"}, "dense", {}),
        ("classification", "float64", "csr", {"density": 0.25}),
    ],
)
def test_generated_inputs_match_workload(
    kind, dtype, input_format, parameters
):
    request = _request(
        dataset={
            "generator": kind,
            "shape": {"rows": 128, "features": 8},
            "dtype": dtype,
            "format": input_format,
            "parameters": parameters,
        }
    )
    case = resolve_case(request, PROFILE)
    X, y = generate_data(case)
    repeat_X, repeat_y = generate_data(case)
    assert X.shape == (128, 8) and y.shape == (128,)
    assert y.dtype == case.dtypes["y"]
    np.testing.assert_array_equal(y, repeat_y)
    if input_format == "csr":
        assert sparse.isspmatrix_csr(X)
        for name in ("data", "indices", "indptr"):
            np.testing.assert_array_equal(
                getattr(X, name), getattr(repeat_X, name)
            )
        dense_X, dense_y = generate_data(
            replace(
                case,
                input_format="dense",
                dataset_parameters={
                    k: v
                    for k, v in case.dataset_parameters.items()
                    if k != "density"
                },
            )
        )
        np.testing.assert_array_equal(y, dense_y)
        assert 0.15 < X.nnz / np.prod(X.shape) < 0.35
        values, repeat = X.toarray(), repeat_X.toarray()
        np.testing.assert_array_equal(
            values[values != 0], dense_X[values != 0]
        )
        assert sparse.isspmatrix_csr(X[:64])
    else:
        values, repeat = np.asarray(X), np.asarray(repeat_X)
    assert values.dtype == case.dtypes["X"]
    np.testing.assert_array_equal(values, repeat)
    if kind in {"classification", "blobs"}:
        assert len(np.unique(y)) == parameters.get(
            "n_classes", parameters.get("centers", 2)
        )
    elif kind == "positive":
        assert np.all(values >= 0)
    elif kind == "categorical":
        assert np.all((values >= 0) & (values < 8))
        assert set(y) <= {0, 1}


@pytest.mark.parametrize(
    "kind,estimator,other_estimator",
    [
        ("matrix", "PCA", "MultinomialNB"),
        ("categorical", "OneHotEncoder", "LabelEncoder"),
    ],
)
def test_generated_dataset_is_independent_of_estimator(
    kind, estimator, other_estimator
):
    case = resolve_case(
        _request(
            estimator=estimator,
            parameters={},
            dataset={
                "generator": kind,
                "shape": {"rows": 64, "features": 8},
            },
        ),
        PROFILE,
    )
    X, y = generate_data(case)
    other_X, other_y = generate_data(replace(case, estimator=other_estimator))
    expected_type = pd.DataFrame if kind == "categorical" else np.ndarray
    assert type(X) is type(other_X) is expected_type
    assert type(y) is type(other_y) is np.ndarray
    np.testing.assert_array_equal(other_y, y, strict=True)
    if kind == "categorical":
        pd.testing.assert_frame_equal(other_X, X)
    else:
        np.testing.assert_array_equal(other_X, X, strict=True)


# Measurement contracts use fake estimators, not estimator accuracy tests.


@pytest.mark.parametrize(
    "operation,selection,warmups",
    [
        ("fit", ("X", "y"), 1),
        ("fit_transform", ("X",), 1),
        ("predict", ("X",), 1),
        ("predict", ("X",), 0),
        ("transform", ("y",), 1),
    ],
)
def test_estimator_lifecycle_and_inputs(
    monkeypatch, operation, selection, warmups
):
    inference = not operation.startswith("fit")
    request = _request(operation=operation, input_selection=list(selection))
    if inference:
        request["fit_input_selection"] = (
            ["y"] if selection == ("y",) else ["X", "y"]
        )
        request["dataset"]["shape"]["train_rows"] = 32
    case = resolve_case(request, {**PROFILE, "warmups": warmups})
    X = np.arange(case.generated_rows * 8).reshape(-1, 8)
    y = np.arange(case.generated_rows)
    monkeypatch.setattr(harness, "generate_data", lambda *args: (X, y))
    instances, calls = [], []

    class Estimator:
        def __init__(self, **kwargs):
            instances.append(self)

        def fit(self, *args):
            calls.append((self, "fit", args))
            return self

        def fit_transform(self, *args):
            calls.append((self, "fit_transform", args))
            return args[0]

        def predict(self, *args):
            calls.append((self, operation, args))
            return args[0]

        transform = predict

    monkeypatch.setattr(
        get_backend("cpu"), "load_estimator", lambda spec: Estimator
    )
    result = _benchmark(case)
    assert result["outcome"] == {"status": "success"}
    assert len(instances) == (1 if inference else warmups + 2)
    assert [o["role"] for o in result["observations"]] == [
        "warmup"
    ] * warmups + ["measurement"] * 2
    inputs = {"X": X, "y": y}
    if inference:
        _, method, args = calls.pop(0)
        assert method == "fit"
        assert len(args) == len(case.fit_input_selection)
        for actual, name in zip(args, case.fit_input_selection, strict=True):
            np.testing.assert_array_equal(actual, inputs[name][:32])
    assert len(calls) == warmups + 2
    assert len({id(estimator) for estimator, _, _ in calls}) == len(instances)
    for estimator, method, args in calls:
        assert method == operation
        assert len(args) == len(selection)
        for actual, name in zip(args, selection, strict=True):
            np.testing.assert_array_equal(
                actual, inputs[name][32:] if inference else inputs[name]
            )


@pytest.mark.parametrize("operation", ["fit", "predict"])
@pytest.mark.parametrize("fit_returns_self", [False, True])
def test_only_synchronized_operation_is_timed(
    monkeypatch, operation, fit_returns_self
):
    events = []

    class Output:
        def __array__(self, *args, **kwargs):
            pytest.fail("must not convert outputs")

        def get(self):
            pytest.fail("must not copy outputs to host")

        def compute(self):
            pytest.fail("CPU backend must not compute outputs")

    output = Output()

    class Estimator:
        def __init__(self, **kwargs):
            events.append("construct")

        def fit(self, *args):
            events.append("fit")
            return self if fit_returns_self else output

        def predict(self, *args):
            events.append("predict")
            return output

        def score(self, *args):
            pytest.fail("must not score during benchmarking")

    def generate(case):
        events.append("generate")
        return np.ones((case.generated_rows, case.features)), np.zeros(
            case.generated_rows
        )

    backend = get_backend("cpu")
    monkeypatch.setattr(backend, "load_estimator", lambda spec: Estimator)
    monkeypatch.setattr(harness, "generate_data", generate)

    def synchronize(value=None):
        assert not isinstance(value, Estimator)
        events.append("sync-output" if value is output else "sync")

    monkeypatch.setattr(backend, "synchronize", synchronize)
    ticks = iter([10, 10.25, 20, 20.25, 30, 30.25])
    monkeypatch.setattr(
        harness.time,
        "perf_counter",
        lambda: events.append("clock") or next(ticks),
    )
    request = _request(operation=operation)
    if operation == "predict":
        request.update(fit_input_selection=["X", "y"])
        request["dataset"]["shape"]["train_rows"] = 32
    result = _benchmark(resolve_case(request, PROFILE))
    assert result["outcome"] == {"status": "success"}
    output_sync = (
        "sync" if operation == "fit" and fit_returns_self else "sync-output"
    )
    observation = ["sync", "clock", operation, output_sync, "clock"]
    assert events == (
        ["generate", "construct", "fit", "sync"] + observation * 3
        if operation == "predict"
        else ["generate"] + (["construct"] + observation) * 3
    )
    assert all(
        o["timings"] == [{"name": "wall_time", "value": 0.25, "unit": "s"}]
        and o["metrics"] == []
        for o in result["observations"]
    )


@pytest.mark.parametrize("failure_at", [0, 1])
def test_gpu_synchronization_failure_fails_observation(
    monkeypatch, failure_at
):
    backend = get_backend("cuml")
    calls = 0

    class Estimator:
        def __init__(self, **kwargs):
            pass

        def fit(self, X):
            return self

    def synchronize():
        nonlocal calls
        index = calls
        calls += 1
        if index == failure_at:
            raise RuntimeError("asynchronous CUDA failure")

    cupy = SimpleNamespace(
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(deviceSynchronize=synchronize)
        )
    )
    monkeypatch.setattr(backend, "load_estimator", lambda spec: Estimator)
    monkeypatch.setattr(
        harness,
        "generate_data",
        lambda case: (np.ones((64, 8)), None),
    )
    monkeypatch.setattr(
        backend, "prepare_inputs", lambda inputs, runtime: inputs
    )
    monkeypatch.setattr(backend, "convert_data", lambda value, **kwargs: value)
    monkeypatch.setattr(
        backend,
        "instrumentation",
        lambda *args: contextlib.nullcontext(
            base_backend.VerificationResult()
        ),
    )
    monkeypatch.setattr(importlib, "import_module", lambda name: cupy)
    result = _benchmark(resolve_case(_request(), PROFILE), provider="cuml")
    assert result["outcome"]["status"] == "failed"
    assert result["outcome"]["last_phase"] == "warmup"
    assert result["outcome"]["error"]["message"] == "asynchronous CUDA failure"
    assert len(result["observations"]) == 1
    assert result["observations"][0]["timings"] == []
    assert calls == failure_at + 1


def test_gpu_synchronization_requires_cupy(monkeypatch):
    def unavailable(name):
        raise ModuleNotFoundError("No module named 'cupy'", name="cupy")

    monkeypatch.setattr(importlib, "import_module", unavailable)
    with pytest.raises(ModuleNotFoundError, match="cupy"):
        GPUBackend().synchronize()


@pytest.mark.parametrize("fallback_at", [None, 0, 1])
def test_accel_warmup_dispatch_verification(monkeypatch, fallback_at):
    backend = get_backend("cuml.accel")
    profiled, executed = [], []
    active = False

    @contextlib.contextmanager
    def profile(quiet=False):
        nonlocal active
        assert quiet
        active = True
        repetition = len(profiled)
        profiled.append(repetition)
        try:
            yield SimpleNamespace(
                method_calls={
                    "PCA.fit": SimpleNamespace(
                        gpu_calls=1,
                        gpu_time=0.01,
                        cpu_calls=int(repetition == fallback_at),
                        cpu_time=0,
                        fallback_reasons={"unsupported"}
                        if repetition == fallback_at
                        else set(),
                    )
                }
            )
        finally:
            active = False

    class Estimator:
        def __init__(self, **kwargs):
            pass

        def fit(self, X):
            executed.append(active)
            return self

    original = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: SimpleNamespace(profile=profile)
        if name == "cuml.accel"
        else original(name),
    )
    monkeypatch.setattr(backend, "load_estimator", lambda spec: Estimator)
    monkeypatch.setattr(backend, "synchronize", lambda value=None: None)
    case = resolve_case(_request(), {**PROFILE, "warmups": 2})
    result = _benchmark(case, "cuml.accel")
    if fallback_at is None:
        assert result["outcome"] == {"status": "success"}
        assert executed == [True, True, False, False]
        assert [
            ACCEL_EXTENSION in o["extensions"] for o in result["observations"]
        ] == executed
    else:
        assert result["outcome"]["error"]["type"] == "CpuFallback"
        assert executed == [True] * (fallback_at + 1)
        assert all(o["role"] == "warmup" for o in result["observations"])
        evidence = result["observations"][-1]["extensions"][ACCEL_EXTENSION]
        assert evidence["cpu_calls"] == 1
        assert evidence["repetition"] == fallback_at
        assert evidence["calls"][0]["fallback_reasons"] == ["unsupported"]
    assert profiled == list(
        range(2 if fallback_at is None else fallback_at + 1)
    )


# One registered backend exercises the worker, artifacts, and resume end to end.


@pytest.fixture
def registered_suite(monkeypatch, tmp_path):
    events = []
    state = {"failure": None, "calls": 0}
    module = ModuleType("benchmark_test_estimator")

    class Estimator:
        def __init__(self, fail=False):
            self.fail = fail

        def get_params(self, deep=False):
            return {"fail": self.fail}

        def fit(self, X):
            events.append(("fit", len(X)))
            if self.fail:
                state["calls"] += 1
                if state["calls"] == 2 and state["failure"] is not None:
                    raise state["failure"]
            return self

    module.Custom = Estimator
    monkeypatch.setitem(sys.modules, module.__name__, module)

    class TestBackend(Backend):
        def bootstrap_process(self):
            events.append("bootstrap")

        def prepare_process(self):
            events.append("prepare")

        @contextlib.contextmanager
        def runtime(self, suite):
            assert events.index("bootstrap") < events.index("prepare")
            events.append("runtime-enter")
            try:
                yield "client"
            finally:
                events.append("runtime-exit")

        def construct_estimator(self, cls, parameters, runtime=None):
            assert runtime == "client"
            return super().construct_estimator(cls, parameters, runtime)

    monkeypatch.setitem(
        PROVIDERS,
        "test",
        Provider(
            TestBackend(),
            {"Custom": EstimatorSpec(module.__name__, "Custom", "custom")},
        ),
    )
    manifest = _write_suite(
        tmp_path,
        {
            "version": 2,
            "name": "test",
            "providers": ["test"],
            "profiles": {"standard": {**PROFILE, "repetitions": 1}},
            "cases": [
                _request(
                    estimator="Custom",
                    parameters={"fail": fail},
                    dataset={
                        "generator": "matrix",
                        "shape": {"rows": rows, "features": 2},
                    },
                )
                for rows, fail in [(32, True), (64, False)]
            ],
        },
    )
    # Keep provenance stable across checkpoint writes and resume attempts.
    monkeypatch.setattr(harness, "_gpu_components", lambda: [])
    return manifest, tmp_path / "artifact.json", events, state


def _run_cli(manifest, output, *, resume=False):
    return cli.main(
        [
            "--_worker",
            "--suite",
            str(manifest),
            "--output",
            str(output),
            "--provider",
            "test",
        ]
        + (["--resume"] if resume else [])
    )


def test_cli_artifact_checkpoint_and_resume(registered_suite, monkeypatch):
    manifest, output, events, state = registered_suite
    checkpoints = []
    write = harness.atomic_write

    def checkpoint(path, artifact, **kwargs):
        write(path, artifact, **kwargs)
        persisted = json.loads(Path(path).read_text())
        _validate_artifact(persisted)
        checkpoints.append(persisted)
        events.append(("checkpoint", len(persisted["results"])))

    monkeypatch.setattr(harness, "atomic_write", checkpoint)
    state["failure"] = RuntimeError("second observation failed")
    assert _run_cli(manifest, output) == 1
    original = json.loads(output.read_text())
    assert [len(a["results"]) for a in checkpoints] == [0, 1, 2, 2]
    assert [r["outcome"]["status"] for r in original["results"]] == [
        "failed",
        "success",
    ]
    failed, success = original["results"]
    assert failed["outcome"]["last_phase"] == "timed_execution"
    assert [o["outcome"]["status"] for o in failed["observations"]] == [
        "success",
        "failed",
    ]
    assert failed["observations"][-1]["timings"] == []
    assert all(
        o["metrics"] == []
        for r in original["results"]
        for o in r["observations"]
    )
    assert events.index(("checkpoint", 0)) < events.index("runtime-enter")
    assert events.index(("checkpoint", 1)) < events.index(("fit", 64))
    assert events[-2:] == ["runtime-exit", ("checkpoint", 2)]
    events.clear()
    state["failure"] = None
    assert _run_cli(manifest, output, resume=True) == 0
    resumed = json.loads(output.read_text())
    assert resumed["run"]["id"] == original["run"]["id"]
    assert resumed["results"][0] == success
    assert {r["id"] for r in resumed["results"]} == {
        r["id"] for r in original["results"]
    }
    assert all(
        r["outcome"] == {"status": "success"} for r in resumed["results"]
    )
    assert ("fit", 64) not in events
    assert events.count(("fit", 32)) == 2
    assert events[-2:] == ["runtime-exit", ("checkpoint", 2)]


@pytest.mark.parametrize(
    "change",
    [
        "schema",
        "methodology",
        "plan",
        "tags",
        "provider",
        "software",
        "system",
        "unknown_id",
        "workload",
        "failed_workload",
        "duplicate",
    ],
)
def test_resume_rejects_incompatible_artifact(registered_suite, change):
    manifest, output, events, _ = registered_suite
    assert _run_cli(manifest, output) == 0
    artifact = json.loads(output.read_text())
    if change == "schema":
        artifact["schema_version"] = 1
    elif change in {"methodology", "software", "system"}:
        artifact["run"][change] = {}
    elif change == "plan":
        artifact["run"]["extensions"][harness.EXTENSION]["execution_plan"][0][
            "repetitions"
        ] += 1
    elif change == "tags":
        artifact["run"]["extensions"][harness.EXTENSION]["execution_plan"][0][
            "tags"
        ] = ["wide"]
    elif change == "provider":
        artifact["run"]["extensions"][harness.EXTENSION][change] = "changed"
    elif change == "unknown_id":
        artifact["results"][0]["id"] = "sha256:unknown"
    elif change in {"workload", "failed_workload"}:
        artifact["results"][0]["parameters"]["declared"]["fail"] = False
        if change == "failed_workload":
            artifact["results"][0]["outcome"] = {"status": "failed"}
    else:
        artifact["results"].append(copy.deepcopy(artifact["results"][0]))
    output.write_text(json.dumps(artifact), encoding="utf-8")
    before = output.read_bytes()
    events.clear()
    with pytest.raises(SystemExit) as error:
        _run_cli(manifest, output, resume=True)
    assert error.value.code == 2
    assert output.read_bytes() == before
    assert "runtime-enter" not in events


def test_software_metadata_snapshots_dependencies(monkeypatch):
    distributions = [
        SimpleNamespace(metadata={"Name": name}, version=version)
        for name, version in [
            ("numpy", "2.0"),
            ("dask", "2026.7"),
            ("cuml", "test-version"),
        ]
    ]
    monkeypatch.setattr(
        base_backend.importlib.metadata, "distributions", lambda: distributions
    )
    backend = get_backend("cuml.dask")
    monkeypatch.setattr(backend, "distribution_name", lambda name: name)
    monkeypatch.setattr(base_backend, "_version", lambda name: "unknown")
    monkeypatch.setattr(backend, "package_source", lambda name: None)
    packages = backend.software_packages(
        [EstimatorSpec("cuml.decomposition", "PCA", "cuml")]
    )
    assert {p["name"]: p["version"] for p in packages} == {
        d.metadata["Name"]: d.version for d in distributions
    }
    assert [p["name"] for p in packages] == sorted(p["name"] for p in packages)


@pytest.mark.parametrize("distribution", ["cuml", "cuml-cu12", "cuml-cu13"])
def test_cuml_distribution_metadata(monkeypatch, distribution):
    source = {"repository": "test", "revision": "abc", "dirty": False}
    monkeypatch.setattr(cuml_backend, "_source", lambda: source)
    monkeypatch.setattr(
        base_backend.importlib.metadata,
        "distributions",
        lambda: [
            SimpleNamespace(
                metadata={"Name": distribution}, version="test-version"
            )
        ],
    )

    def version(name):
        if name != distribution:
            raise importlib.metadata.PackageNotFoundError(name)
        return "test-version"

    monkeypatch.setattr(base_backend.importlib.metadata, "version", version)
    monkeypatch.setattr(
        base_backend.importlib.metadata,
        "packages_distributions",
        lambda: {"cuml": [distribution]},
    )
    backend = get_backend("cuml")
    assert backend.distribution_name("cuml") == distribution
    assert backend.software_packages(
        [EstimatorSpec("cuml.decomposition", "PCA", "cuml")]
    ) == [
        {
            "name": distribution,
            "version": "test-version",
            "build": None,
            "source": source,
        }
    ]


@pytest.mark.parametrize("interrupted", [False, True])
def test_harness_resolves_implementation_distribution(
    registered_suite, monkeypatch, interrupted
):
    manifest, output, _, state = registered_suite
    backend = get_provider("test").backend
    monkeypatch.setattr(
        backend, "distribution_name", lambda name: "custom-dist"
    )
    monkeypatch.setattr(
        backend,
        "software_packages",
        lambda specs: [{**PACKAGE, "name": "custom-dist"}],
    )
    if interrupted:
        state["failure"] = KeyboardInterrupt()
        with pytest.raises(KeyboardInterrupt):
            _run_cli(manifest, output)
    else:
        assert _run_cli(manifest, output) == 0
    artifact = json.loads(output.read_text())
    _validate_artifact(artifact)
    assert all(
        r["implementation"]["name"] == "custom-dist"
        for r in artifact["results"]
    )


@pytest.mark.parametrize("distributions", [[], ["first", "second"]])
def test_distribution_resolution_retains_unmapped_or_ambiguous_names(
    monkeypatch, distributions
):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(base_backend.importlib.metadata, "version", missing)
    monkeypatch.setattr(
        base_backend.importlib.metadata,
        "packages_distributions",
        lambda: {"custom": distributions},
    )
    assert Backend().distribution_name("custom") == "custom"


def test_gpu_runtime_metadata(monkeypatch):
    runtime = SimpleNamespace(
        driverGetVersion=lambda: 13010, runtimeGetVersion=lambda: 13000
    )
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: SimpleNamespace(cuda=SimpleNamespace(runtime=runtime)),
    )
    assert GPUBackend().software_runtimes() == [
        {"name": "cuda-driver", "version": "13010"},
        {"name": "cuda-runtime", "version": "13000"},
    ]


@pytest.mark.parametrize(
    "dependency",
    ["numpy", "scikit-learn", "cupy-cuda13x", "dask", "distributed"],
)
def test_resume_rejects_changed_dependency(
    registered_suite, monkeypatch, dependency
):
    manifest, output, events, _ = registered_suite
    versions = {dependency: "original"}
    monkeypatch.setattr(
        base_backend.importlib.metadata, "packages_distributions", dict
    )
    monkeypatch.setattr(
        base_backend.importlib.metadata,
        "distributions",
        lambda: [
            SimpleNamespace(metadata={"Name": name}, version=version)
            for name, version in versions.items()
        ],
    )
    assert _run_cli(manifest, output) == 0
    before = output.read_bytes()
    versions[dependency] = "changed"
    events.clear()
    with pytest.raises(SystemExit) as error:
        _run_cli(manifest, output, resume=True)
    assert error.value.code == 2
    assert output.read_bytes() == before
    assert "runtime-enter" not in events


@pytest.mark.parametrize("runtime_name", ["cuda-driver", "cuda-runtime"])
def test_resume_rejects_changed_backend_runtime(
    registered_suite, monkeypatch, runtime_name
):
    manifest, output, events, _ = registered_suite
    versions = [{"name": runtime_name, "version": "original"}]
    monkeypatch.setattr(
        Backend, "software_runtimes", lambda self: copy.deepcopy(versions)
    )
    assert _run_cli(manifest, output) == 0
    before = output.read_bytes()
    versions[0]["version"] = "changed"
    events.clear()
    with pytest.raises(SystemExit) as error:
        _run_cli(manifest, output, resume=True)
    assert error.value.code == 2
    assert output.read_bytes() == before
    assert "runtime-enter" not in events


@pytest.fixture
def source_checkout(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    module = root / "python/cuml/cuml/benchmark/backends/cuml.py"
    module.parent.mkdir(parents=True)
    module.write_text("# tracked source\n", encoding="utf-8")
    ignore_file = Path(__file__).resolve().parents[3] / ".gitignore"
    (root / ".gitignore").write_bytes(ignore_file.read_bytes())

    def git(*args):
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Benchmark test")
    git("config", "user.email", "benchmark@example.invalid")
    git("config", "commit.gpgsign", "false")
    git("add", ".")
    git("commit", "-qm", "Initial source")
    monkeypatch.setattr(cuml_backend, "__file__", str(module))
    monkeypatch.chdir(root)
    return root, module, git


@pytest.mark.parametrize("distribution", ["cuml", "cuml-cu12", "cuml-cu13"])
def test_source_resume_with_ignored_default_output(
    source_checkout, monkeypatch, distribution
):
    _, module, git = source_checkout
    original_version = base_backend.importlib.metadata.version

    def version(name):
        if name == distribution:
            return "test-version"
        if name == "cuml":
            raise importlib.metadata.PackageNotFoundError(name)
        return original_version(name)

    monkeypatch.setattr(base_backend.importlib.metadata, "version", version)
    monkeypatch.setattr(
        base_backend.importlib.metadata,
        "packages_distributions",
        lambda: {"cuml": [distribution]},
    )
    backend = get_backend("cuml")
    monkeypatch.setattr(backend, "software_runtimes", lambda: [])
    suite = _suite(resolve_case(_request(), PROFILE), provider="cuml")
    output = cli.default_output_path(suite) / "cuml.json"
    original = harness.run_suite(suite, output)
    package = next(
        package
        for package in original["run"]["software"]["packages"]
        if package["name"] == backend.distribution_name("cuml")
    )
    assert package["name"] == distribution
    assert package["source"] == {
        "repository": None,
        "revision": git("rev-parse", "HEAD"),
        "dirty": False,
    }
    # Other provider outputs and logs in the default directory are ignored too.
    output.with_name("scikit-learn.json").write_text("{}", encoding="utf-8")
    output.with_name("console.log").write_text("benchmark log\n")
    resumed = harness.run_suite(suite, output, resume=True)
    assert resumed["run"]["id"] == original["run"]["id"]
    assert resumed["results"] == original["results"]
    assert git("status", "--porcelain") == ""

    module.write_text("# changed source\n", encoding="utf-8")
    before = output.read_bytes()
    with pytest.raises(
        SuiteError, match="matching suite, software, and system"
    ):
        harness.run_suite(suite, output, resume=True)
    assert output.read_bytes() == before


def test_interrupt_checkpoints_and_cleans_runtime(registered_suite):
    manifest, output, events, state = registered_suite
    state["failure"] = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        _run_cli(manifest, output)
    assert events[-1] == "runtime-exit"
    artifact = json.loads(output.read_text())
    _validate_artifact(artifact)
    assert "completed_at" in artifact["run"]
    assert len(artifact["results"]) == 1
    assert artifact["results"][0]["outcome"]["last_phase"] == "interrupted"


@pytest.mark.parametrize("kind", ["file", "directory", "dangling_symlink"])
def test_cli_refuses_existing_selected_artifact(
    document, tmp_path, monkeypatch, capsys, kind
):
    document["providers"] = ["cuml", "scikit-learn"]
    path = _write_suite(tmp_path, document)
    output = tmp_path / "results"
    output.mkdir()
    existing = output / "scikit-learn.json"
    if kind == "file":
        existing.write_bytes(b"existing results")
    elif kind == "directory":
        existing.mkdir()
    else:
        existing.symlink_to(tmp_path / "missing")
    monkeypatch.setattr(
        coordinator,
        "run_command",
        lambda *a, **k: pytest.fail("launched worker"),
    )
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "--suite",
                str(path),
                "--provider",
                "cuml",
                "--provider",
                "scikit-learn",
                "--output",
                str(output),
            ]
        )
    assert error.value.code == 2
    assert "use --resume or a new output directory" in capsys.readouterr().err
    assert not (output / "cuml.json").exists()
    if kind == "file":
        assert existing.read_bytes() == b"existing results"


def test_harness_refuses_existing_results(registered_suite):
    manifest, output, events, _ = registered_suite
    output.write_bytes(b"existing results")
    suite = load_suite(manifest).runs[0]
    with pytest.raises(SuiteError, match="results already exist"):
        harness.run_suite(suite, output)
    assert output.read_bytes() == b"existing results"
    assert not events


def test_harness_initial_write_handles_race(registered_suite, monkeypatch):
    manifest, output, events, _ = registered_suite
    suite = load_suite(manifest).runs[0]
    original = harness._run_record

    def record(*args):
        result = original(*args)
        output.write_bytes(b"concurrent results")
        return result

    monkeypatch.setattr(harness, "_run_record", record)
    with pytest.raises(SuiteError, match="results already exist"):
        harness.run_suite(suite, output)
    assert output.read_bytes() == b"concurrent results"
    assert "runtime-enter" not in events
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def _write_worker_artifact(command, *, failed=False):
    provider = command[command.index("--provider") + 1]
    plan = load_suite_reference(
        command[command.index("--suite") + 1],
        command[command.index("--profile") + 1],
        [provider],
    )
    extension = {harness.EXTENSION: {"provider": provider}}
    artifact = {
        "schema_version": 2,
        "run": {"extensions": extension},
        "results": [
            {
                "id": case.workload_id(),
                "outcome": {"status": "failed" if failed else "success"},
                "extensions": extension,
            }
            for case in plan.runs[0].cases
        ],
    }
    Path(command[command.index("--output") + 1]).write_text(
        json.dumps(artifact)
    )
    return artifact


def test_cli_allows_new_provider_in_existing_directory(
    document, tmp_path, monkeypatch
):
    path = _write_suite(tmp_path, document)
    output = tmp_path / "results"
    output.mkdir()
    unrelated = output / "cuml.json"
    unrelated.write_bytes(b"unselected results")
    calls = []

    def execute(command, environment):
        calls.append(command)
        _write_worker_artifact(command)

    monkeypatch.setattr(coordinator, "run_command", execute)
    assert (
        cli.main(
            [
                "--suite",
                str(path),
                "--provider",
                "scikit-learn",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert len(calls) == 1
    assert unrelated.read_bytes() == b"unselected results"


def test_cli_output_safety(monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["--resume"])
    assert error.value.code == 2
    assert "--resume requires --output" in capsys.readouterr().err
    monkeypatch.chdir(tmp_path)
    suite = load_suite_reference("estimators", "smoke", ["scikit-learn"])
    now = dt.datetime(2026, 8, 13, tzinfo=dt.timezone.utc)
    first = cli.default_output_path(suite, now)
    assert first == tmp_path / ".benchmarks/estimators-smoke-20260813T000000Z"
    assert not first.parent.exists()
    first.parent.mkdir()
    first.write_text("existing artifact", encoding="utf-8")
    second = cli.default_output_path(suite, now)
    assert second != first and not second.exists()
    assert first.read_text() == "existing artifact"


@pytest.mark.parametrize("entrypoint", ["python", "cli"])
@pytest.mark.parametrize(
    "inherited,enabled",
    [("1", False), ("true", False), ("TRUE", False), (None, True)],
)
def test_coordinator_rejects_accelerated_caller(
    document, tmp_path, monkeypatch, capsys, entrypoint, inherited, enabled
):
    if inherited is None:
        monkeypatch.delenv("CUML_ACCEL_ENABLED", raising=False)
    else:
        monkeypatch.setenv("CUML_ACCEL_ENABLED", inherited)
    monkeypatch.setitem(
        sys.modules, "cuml.accel", SimpleNamespace(enabled=lambda: enabled)
    )
    monkeypatch.setattr(
        coordinator,
        "run_command",
        lambda *a, **k: pytest.fail("launched worker"),
    )
    path = _write_suite(tmp_path, document)
    output = tmp_path / "results"
    message = "process without cuml.accel enabled"
    if entrypoint == "python":
        with pytest.raises(SuiteError, match=message):
            benchmark.run(path, providers=["scikit-learn"], output=output)
    else:
        with pytest.raises(SystemExit) as error:
            cli.main(
                [
                    "--suite",
                    str(path),
                    "--provider",
                    "scikit-learn",
                    "--output",
                    str(output),
                ]
            )
        assert error.value.code == 2
        assert message in capsys.readouterr().err
    assert not output.exists()
    assert os.environ.get("CUML_ACCEL_ENABLED") == inherited


@pytest.mark.parametrize("inherited", [None, "0", "false"])
def test_coordinator_preserves_worker_environment(
    document, tmp_path, monkeypatch, inherited
):
    if inherited is None:
        monkeypatch.delenv("CUML_ACCEL_ENABLED", raising=False)
    else:
        monkeypatch.setenv("CUML_ACCEL_ENABLED", inherited)
    monkeypatch.delitem(sys.modules, "cuml.accel", raising=False)
    monkeypatch.setenv("BENCHMARK_ENV_TEST", "preserved")
    original = dict(os.environ)
    document["providers"] = ["cuml.accel", "scikit-learn"]
    path = _write_suite(tmp_path, document)
    environments = []

    def execute(command, environment):
        assert environment["BENCHMARK_ENV_TEST"] == "preserved"
        assert environment.get("CUML_ACCEL_ENABLED") == inherited
        assert "cuml.accel" not in sys.modules
        environments.append(environment)
        environment["BENCHMARK_ENV_TEST"] = "changed"
        _write_worker_artifact(command)

    monkeypatch.setattr(coordinator, "run_command", execute)
    benchmark.run(
        path, providers=document["providers"], output=tmp_path / "results"
    )
    assert len(environments) == 2
    assert dict(os.environ) == original


def test_cli_coordinates_isolated_backends(document, tmp_path, monkeypatch):
    document["providers"] = ["cuml", "scikit-learn", "cuml.accel"]
    path = _write_suite(tmp_path, document)
    output = tmp_path / "results"
    calls = []
    monkeypatch.delenv("CUML_ACCEL_ENABLED", raising=False)
    original = dict(os.environ)

    def execute(command, environment):
        assert "--_worker" in command
        backend = command[command.index("--provider") + 1]
        assert environment.get("CUML_ACCEL_ENABLED") == original.get(
            "CUML_ACCEL_ENABLED"
        )
        assert "--verbose" in command
        artifact = Path(command[command.index("--output") + 1])
        assert artifact == output / f"{backend}.json"
        calls.append((backend, "--resume" in command))
        _write_worker_artifact(command, failed=backend == "scikit-learn")
        if backend == "scikit-learn":
            raise RuntimeError("worker exited with status 1")

    monkeypatch.setattr(coordinator, "run_command", execute)
    argv = ["--suite", str(path), "--output", str(output), "--verbose"]
    assert cli.main(argv) == 0
    assert calls == [("cuml", False)]
    calls.clear()
    selected = [
        arg for name in document["providers"] for arg in ("--provider", name)
    ]
    (output / "cuml.json").unlink()
    assert cli.main([*argv, *selected]) == 1
    assert calls == [(name, False) for name in document["providers"]]
    assert dict(os.environ) == original
    (output / "cuml.accel.json").unlink()
    calls.clear()
    assert cli.main([*argv, *selected, "--resume"]) == 1
    assert calls == [
        ("cuml", True),
        ("scikit-learn", True),
        ("cuml.accel", False),
    ]
    calls.clear()
    assert cli.main([*argv, "--provider", "cuml", "--resume"]) == 0
    assert calls == [("cuml", True)]
    calls.clear()
    assert cli.main([*argv, "--provider", "scikit-learn", "--resume"]) == 1
    assert calls == [("scikit-learn", True)]


def test_cli_default_requires_cuml_in_suite(
    document, tmp_path, monkeypatch, capsys
):
    path = _write_suite(tmp_path, document)
    monkeypatch.setattr(
        coordinator,
        "run_command",
        lambda *a, **k: pytest.fail("launched worker"),
    )
    with pytest.raises(SystemExit) as exc:
        cli.main(["--suite", str(path), "--output", str(tmp_path / "results")])
    assert exc.value.code == 2
    assert "subset of suite providers" in capsys.readouterr().err


def test_cli_coordinator_output_errors(document, tmp_path, monkeypatch):
    document["providers"] = ["cuml"]
    path = _write_suite(tmp_path, document)
    output = tmp_path / "missing"
    monkeypatch.setattr(
        coordinator,
        "run_command",
        lambda *a, **k: pytest.fail("launched worker"),
    )
    with pytest.raises(SystemExit):
        cli.main(["--suite", str(path), "--output", str(output), "--resume"])
    output.write_text("not a directory")
    with pytest.raises(SystemExit):
        cli.main(["--suite", str(path), "--output", str(output)])
    assert output.read_text() == "not a directory"


@pytest.mark.parametrize(
    "providers",
    [["scikit-learn"], ["cuml.accel", "scikit-learn", "cuml"]],
)
def test_cli_real_workers(document, tmp_path, monkeypatch, providers):
    # A CPU worker must remain unaccelerated even after an accel worker.
    monkeypatch.delenv("CUML_ACCEL_ENABLED", raising=False)
    document["providers"] = providers
    path = _write_suite(tmp_path, document)
    output = tmp_path / "results"
    argv = ["--suite", str(path), "--output", str(output)]
    argv.extend(arg for name in providers for arg in ("--provider", name))
    assert cli.main(argv) == 0
    artifacts = {}
    for provider in providers:
        artifact = json.loads((output / f"{provider}.json").read_text())
        artifacts[provider] = artifact
        _validate_artifact(artifact)
        assert len(artifact["results"]) == 1
        result = artifact["results"][0]
        assert result["outcome"]["status"] == "success"
        assert result["extensions"][harness.EXTENSION]["provider"] == provider
        if provider == "cuml.accel":
            evidence = result["observations"][0]["extensions"][ACCEL_EXTENSION]
            assert evidence["gpu_calls"] > 0 and evidence["cpu_calls"] == 0
    assert len({a["results"][0]["id"] for a in artifacts.values()}) == 1
    assert len({a["run"]["id"] for a in artifacts.values()}) == len(providers)
    assert cli.main([*argv, "--resume"]) == 0
    for provider, artifact in artifacts.items():
        resumed = json.loads((output / f"{provider}.json").read_text())
        assert resumed["run"]["id"] == artifact["run"]["id"]
        assert resumed["results"] == artifact["results"]


@pytest.mark.parametrize(
    "enabled,attempted", [(False, False), (False, True), (True, False)]
)
def test_accel_startup(monkeypatch, enabled, attempted):
    monkeypatch.setattr("cuml.accel.enabled", lambda: enabled)
    if attempted:
        monkeypatch.setenv("CUML_ACCEL_ENABLED", "1")
    else:
        monkeypatch.delenv("CUML_ACCEL_ENABLED", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--suite",
            "estimators",
            "--provider",
            "cuml.accel",
            "--_worker",
        ],
    )
    calls = []
    monkeypatch.setattr(os, "execvpe", lambda *args: calls.append(args))
    backend = get_backend("cuml.accel")
    if attempted:
        with pytest.raises(SuiteError, match="did not activate"):
            backend.bootstrap_process()
        assert not calls
    else:
        backend.bootstrap_process()
        if enabled:
            assert not calls
        else:
            assert len(calls) == 1
            executable, argv, env = calls[0]
            assert executable == sys.executable
            assert argv == [
                sys.executable,
                "-m",
                "cuml.benchmark",
                "--suite",
                "estimators",
                "--provider",
                "cuml.accel",
                "--_worker",
            ]
            assert env["CUML_ACCEL_ENABLED"] == "1"


# Process isolation, real runtimes, and distributed resource safety.


def test_spawned_case_completes(tmp_path, monkeypatch):
    monkeypatch.delenv("CUML_ACCEL_ENABLED", raising=False)
    request = _request(
        operation="transform", fit_input_selection=["X"], timeout_sec=60
    )
    request["dataset"]["shape"]["train_rows"] = 32
    case = resolve_case(request, PROFILE)
    output = tmp_path / "artifact.json"
    artifact = harness.run_suite(_suite(case), output)
    _validate_artifact(artifact)
    assert json.loads(output.read_text()) == artifact
    result = artifact["results"][0]
    assert result["outcome"] == {"status": "success"}
    assert result["id"] == case.workload_id()
    assert [o["role"] for o in result["observations"]] == [
        "warmup",
        "measurement",
        "measurement",
    ]


def _worker_failure(crash, *, report_phase):
    report_phase("ready")
    if crash:
        os._exit(7)
    time.sleep(120)


@pytest.mark.parametrize("failure", ["timeout", "exit", "interrupt"])
def test_subprocess_failure_cleans_up_worker(failure):
    before = {c.pid for c in multiprocessing.active_children()}
    phases = []

    def phase(value):
        phases.append(value)
        if failure == "interrupt":
            raise KeyboardInterrupt()

    expected = {
        "timeout": runner.SubprocessTimeout,
        "exit": runner.SubprocessExited,
        "interrupt": KeyboardInterrupt,
    }[failure]
    with pytest.raises(expected) as error:
        runner.run_in_subprocess(
            _worker_failure,
            args=(failure == "exit",),
            timeout=0.01 if failure == "timeout" else 60,
            report_phase=phase,
        )
    if failure == "exit":
        assert error.value.exitcode == 7
    if failure != "timeout":
        assert phases == ["ready"]
    assert {c.pid for c in multiprocessing.active_children()} <= before


@pytest.mark.parametrize(
    "failure,error_type",
    [
        (runner.SubprocessTimeout(), "TimeoutExpired"),
        (runner.SubprocessExited(7), "RuntimeError"),
    ],
)
def test_subprocess_failure_preserves_last_phase(
    monkeypatch, failure, error_type
):
    def fail(*args, report_phase, **kwargs):
        report_phase("warmup")
        raise failure

    monkeypatch.setattr(harness, "run_in_subprocess", fail)
    case = resolve_case(_request(timeout_sec=10), PROFILE)
    result = harness._run_case(_suite(case), case, PACKAGE)
    assert result["outcome"]["status"] == "failed"
    assert result["outcome"]["last_phase"] == "warmup"
    assert result["outcome"]["error"]["type"] == error_type
    assert result["outcome"]["error"]["message"]
    assert result["observations"] == []


def _real_backend_smoke(provider, workload, output, *, report_phase):
    import cuml.accel

    assert cuml.accel.enabled() == (provider == "cuml.accel")
    request = _request()
    if workload == "csr-inference":
        request.update(
            estimator="LogisticRegression",
            operation="predict",
            fit_input_selection=["X", "y"],
            parameters={"max_iter": 100},
        )
        request["dataset"].update(
            generator="classification",
            format="csr",
            dtype="float64",
            parameters={"density": 0.25},
        )
        request["dataset"]["shape"].update(rows=128, train_rows=128)
    case = resolve_case(request, {**PROFILE, "repetitions": 1})
    X, _ = generate_data(case)
    provider_spec = get_provider(provider)
    backend = provider_spec.backend
    cls = backend.load_estimator(provider_spec.estimator_spec(case.estimator))
    assert cuml.accel.is_proxy(cls) == (provider == "cuml.accel")
    invoke = getattr(cls, case.operation)
    calls = []

    def observe(estimator, *args):
        actual = args[0]
        expected = (
            X[case.training_rows :] if case.lifecycle == "inference" else X
        )
        assert actual.dtype == np.dtype(case.dtypes["X"])
        if provider == "cuml":
            import cupy as cp
            from cupyx.scipy import sparse as device_sparse

            assert device_sparse.isspmatrix_csr(actual) == (
                case.input_format == "csr"
            )
            assert isinstance(actual, cp.ndarray) or device_sparse.issparse(
                actual
            )
            host = actual.get()
        else:
            assert sparse.isspmatrix_csr(actual) == (
                case.input_format == "csr"
            )
            host = actual
        if case.input_format == "csr":
            for name in ("data", "indices", "indptr"):
                np.testing.assert_array_equal(
                    getattr(host, name), getattr(expected, name)
                )
        else:
            np.testing.assert_array_equal(host, expected)
        calls.append(actual.shape)
        output = invoke(estimator, *args)
        if provider == "cuml" and case.lifecycle == "inference":
            assert isinstance(output, cp.ndarray)
        return output

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cls, case.operation, observe)
        artifact = harness.run_suite(_suite(case, provider=provider), output)
    assert len(calls) == 2
    assert json.loads(output.read_text()) == artifact
    result = artifact["results"][0]
    assert result["outcome"] == {"status": "success"}, result["outcome"]
    assert result["id"] == case.workload_id()
    assert result["input"]["data_type"] == case.dtypes["X"]
    if provider == "cuml.accel":
        dispatch = result["observations"][0]["extensions"][ACCEL_EXTENSION]
        assert dispatch["gpu_calls"] >= 1 and dispatch["cpu_calls"] == 0
        assert ACCEL_EXTENSION not in result["observations"][1]["extensions"]
    return artifact


@pytest.mark.parametrize("provider", ["scikit-learn", "cuml", "cuml.accel"])
@pytest.mark.parametrize("workload", ["dense-fit", "csr-inference"])
def test_real_backend_smoke(monkeypatch, tmp_path, provider, workload):
    # Accelerator installation mutates upstream modules; never enable it in pytest.
    if provider == "cuml.accel":
        monkeypatch.setenv("CUML_ACCEL_ENABLED", "1")
    else:
        monkeypatch.delenv("CUML_ACCEL_ENABLED", raising=False)
    artifact = runner.run_in_subprocess(
        _real_backend_smoke,
        args=(provider, workload, tmp_path / "artifact.json"),
        timeout=90,
    )
    _validate_artifact(artifact)


@pytest.mark.parametrize("kind", ["dense", "csr", "categorical"])
def test_native_device_preparation_and_output_contract(kind):
    import cudf
    import cupy as cp
    from cupyx.scipy import sparse as device_sparse

    import cuml

    backend = get_backend("cuml")
    request = _request()
    if kind == "categorical":
        request["dataset"].update(generator="categorical")
    elif kind == "csr":
        request["dataset"]["format"] = "csr"
    case = resolve_case(request, PROFILE)
    X_host, y_host = generate_data(case)
    X = backend.convert_data(
        X_host,
        representation="csr"
        if kind == "csr"
        else "dataframe"
        if kind == "categorical"
        else "array",
    )
    y = backend.convert_data(y_host, representation="array")
    assert isinstance(y, cp.ndarray)
    if kind == "categorical":
        assert isinstance(X, cudf.DataFrame)
        np.testing.assert_array_equal(X.to_numpy(), X_host.to_numpy())
    elif kind == "csr":
        assert device_sparse.isspmatrix_csr(X)
        np.testing.assert_array_equal(X.get().toarray(), X_host.toarray())
    else:
        assert isinstance(X, cp.ndarray)
        np.testing.assert_array_equal(X.get(), X_host)
    inputs = (X, y)
    assert backend.prepare_inputs(inputs) is inputs
    assert cuml.global_settings.output_type is None
    with backend.instrumentation(None, "measurement", 0):
        assert cuml.global_settings.output_type is None


def test_native_instrumentation_rejects_nondefault_output_type():
    import cuml

    backend = get_backend("cuml")
    with cuml.using_output_type("numpy"):
        with pytest.raises(ValueError, match="default global output_type"):
            with backend.instrumentation(None, "measurement", 0):
                pytest.fail("nondefault output type must be rejected")
        assert cuml.global_settings.output_type == "numpy"


@pytest.mark.parametrize("output_type", ["numpy", "pandas"])
def test_native_estimator_rejects_host_output_default(output_type):
    class Estimator:
        def __init__(self):
            self.output_type = output_type

    with pytest.raises(ValueError, match="device-resident estimator outputs"):
        get_backend("cuml").construct_estimator(Estimator, {})


def test_native_preparation_precedes_initial_fit_and_timing(monkeypatch):
    backend = get_backend("cuml")
    events = []
    X_device, y_device = object(), object()

    class Estimator:
        def __init__(self, **kwargs):
            events.append("construct")

        def fit(self, X, y):
            assert X is X_device and y is y_device
            events.append("fit")
            return self

        def predict(self, X):
            assert X is X_device
            events.append("predict")
            return X

    def convert(value, *, representation):
        events.append("convert")
        return X_device if value.ndim == 2 else y_device

    monkeypatch.setattr(backend, "convert_data", convert)
    monkeypatch.setattr(backend, "load_estimator", lambda spec: Estimator)
    monkeypatch.setattr(
        backend, "synchronize", lambda value=None: events.append("sync")
    )
    monkeypatch.setattr(
        harness.time, "perf_counter", lambda: events.append("clock") or 1.0
    )
    request = _request(operation="predict", fit_input_selection=["X", "y"])
    request["dataset"]["shape"]["train_rows"] = 32
    result = _benchmark(resolve_case(request, PROFILE), provider="cuml")
    assert result["outcome"] == {"status": "success"}
    assert (
        events
        == [
            "convert",
            "convert",
            "convert",
            "sync",
            "construct",
            "fit",
            "sync",
        ]
        + ["sync", "clock", "predict", "sync", "clock"] * 3
    )


@pytest.mark.parametrize("inference", [False, True])
def test_dask_inputs_are_persisted_before_execution(monkeypatch, inference):
    events = []
    backend = get_backend("cuml.dask")
    client = object()
    live_inputs = []

    class Input:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, key):
            return Input((self.name, key.start, key.stop))

    class Estimator:
        def __init__(self, **kwargs):
            assert kwargs["client"] is client
            events.append("construct")

        def fit(self, *args):
            assert all(
                any(arg is live for live in live_inputs) for arg in args
            )
            events.append("fit")
            return self

        def predict(self, *args):
            assert all(
                any(arg is live for live in live_inputs) for arg in args
            )
            events.append("predict")
            return self

    class Client:
        def persist(self, values):
            events.append("persist")
            live_inputs.extend(Input(value.name) for value in values)
            return live_inputs

    runtime = Client()
    client = runtime
    futures = [SimpleNamespace(status="finished")]
    modules = {
        "dask": SimpleNamespace(
            is_dask_collection=lambda value: isinstance(value, Input)
        ),
        "dask.distributed": SimpleNamespace(
            futures_of=lambda values: futures,
            wait=lambda values: events.append("wait"),
        ),
    }
    monkeypatch.setattr(importlib, "import_module", modules.__getitem__)
    monkeypatch.setattr(backend, "load_estimator", lambda spec: Estimator)
    monkeypatch.setattr(backend, "synchronize", lambda value=None: None)
    monkeypatch.setattr(
        harness, "generate_data", lambda *args: (Input("X"), Input("y"))
    )
    monkeypatch.setattr(backend, "convert_data", lambda value, **kwargs: value)
    request = _request()
    if inference:
        request.update(operation="predict", fit_input_selection=["X", "y"])
        request["dataset"]["shape"]["train_rows"] = 32
    result = _benchmark(
        resolve_case(request, PROFILE), provider="cuml.dask", client=runtime
    )
    assert result["outcome"] == {"status": "success"}
    assert events[:2] == ["persist", "wait"]
    assert events.count("persist") == events.count("wait") == 1
    assert len(live_inputs) == (3 if inference else 1)
    assert events[2:] == (
        ["construct", "fit", "predict", "predict", "predict"]
        if inference
        else ["construct", "fit"] * 3
    )


@pytest.mark.parametrize("status", ["error", "cancelled"])
def test_dask_input_preparation_propagates_failure(monkeypatch, status):
    failure = RuntimeError("device conversion failed")
    future = SimpleNamespace(status=status, exception=lambda: failure)
    modules = {
        "dask": SimpleNamespace(is_dask_collection=lambda value: True),
        "dask.distributed": SimpleNamespace(
            futures_of=lambda values: [future], wait=lambda values: None
        ),
    }
    monkeypatch.setattr(importlib, "import_module", modules.__getitem__)
    client = SimpleNamespace(persist=lambda values: values)
    with pytest.raises(RuntimeError, match="conversion failed|cancelled"):
        get_backend("cuml.dask").prepare_inputs((object(),), client)


def test_dask_preparation_retains_real_persisted_inputs(tmp_path):
    da = pytest.importorskip("dask.array")
    distributed = pytest.importorskip("dask.distributed")
    from dask import delayed

    calls = tmp_path / "conversions.txt"

    @delayed(pure=False)
    def convert():
        with calls.open("a") as stream:
            stream.write("convert\n")
        return np.ones((32, 8))

    X = da.from_delayed(convert(), shape=(32, 8), dtype=np.float64)
    with (
        distributed.LocalCluster(
            n_workers=1,
            threads_per_worker=1,
            processes=False,
            dashboard_address=None,
        ) as cluster,
        distributed.Client(cluster) as client,
    ):
        (prepared,) = get_backend("cuml.dask").prepare_inputs((X,), client)
        assert calls.read_text() == "convert\n"
        for _ in range(3):
            np.testing.assert_array_equal(prepared.compute(), np.ones((32, 8)))
        assert calls.read_text() == "convert\n"


@pytest.mark.parametrize("collection", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_dask_synchronization(monkeypatch, collection, nested):
    events = []

    def persist(value):
        events.append("persist")
        return value

    future = SimpleNamespace(status="finished")
    modules = {
        "dask": SimpleNamespace(is_dask_collection=lambda value: collection),
        "dask.distributed": SimpleNamespace(
            wait=lambda values: events.append("wait"),
            futures_of=lambda value: [future],
            get_client=lambda: SimpleNamespace(persist=persist),
        ),
        "cupy": SimpleNamespace(
            cuda=SimpleNamespace(
                runtime=SimpleNamespace(
                    deviceSynchronize=lambda: events.append("gpu")
                )
            )
        ),
    }
    monkeypatch.setattr(importlib, "import_module", modules.__getitem__)
    value = SimpleNamespace(
        compute=lambda: pytest.fail(
            "Synchronization should persist outputs on workers"
        )
    )
    get_backend("cuml.dask").synchronize((value, [value]) if nested else value)
    expected = ["persist", "wait", "gpu"] if collection else ["wait", "gpu"]
    assert events == expected * (2 if nested else 1)


def test_dask_synchronization_executes_lazy_graph(monkeypatch, tmp_path):
    da = pytest.importorskip("dask.array")
    distributed = pytest.importorskip("dask.distributed")
    from dask import delayed

    calls = tmp_path / "outputs.txt"

    @delayed(pure=False)
    def execute():
        calls.write_text("executed", encoding="utf-8")
        return np.ones((4, 2))

    output = da.from_delayed(execute(), shape=(4, 2), dtype=np.float64)
    monkeypatch.setattr(
        type(output),
        "compute",
        lambda *args, **kwargs: pytest.fail("Must not gather output"),
    )
    with (
        distributed.LocalCluster(
            n_workers=1,
            threads_per_worker=1,
            processes=False,
            dashboard_address=None,
        ) as cluster,
        distributed.Client(cluster),
    ):
        get_backend("cuml.dask").synchronize(output)
        assert calls.read_text(encoding="utf-8") == "executed"


@pytest.mark.parametrize("collection", [False, True])
@pytest.mark.parametrize("status", ["error", "cancelled"])
def test_dask_synchronization_propagates_failure(
    monkeypatch, collection, status
):
    failure = RuntimeError("output execution failed")
    future = SimpleNamespace(status=status, exception=lambda: failure)
    modules = {
        "dask": SimpleNamespace(is_dask_collection=lambda value: collection),
        "dask.distributed": SimpleNamespace(
            futures_of=lambda value: [future],
            wait=lambda values: None,
            get_client=lambda: SimpleNamespace(persist=lambda value: value),
        ),
    }
    monkeypatch.setattr(importlib, "import_module", modules.__getitem__)
    with pytest.raises(RuntimeError, match="execution failed|cancelled"):
        get_backend("cuml.dask").synchronize(object())


@pytest.mark.parametrize("estimator", ["PCA", "LabelEncoder", "MultinomialNB"])
def test_dask_input_conversion(estimator):
    cp = pytest.importorskip("cupy")
    pytest.importorskip("dask_cudf")
    X = np.arange(512, dtype=np.float32).reshape(64, 8)
    y = np.arange(64, dtype=np.int64) % 3
    provider = get_provider("cuml.dask")
    converted_X = provider.convert_data(
        X, estimator=estimator, role="X", representation="array"
    )
    converted_y = provider.convert_data(
        y, estimator=estimator, role="y", representation="array"
    )
    assert isinstance(converted_X._meta, cp.ndarray)
    cp.testing.assert_array_equal(converted_X.compute(), cp.asarray(X))
    if estimator == "LabelEncoder":
        import cudf

        assert isinstance(converted_y._meta, cudf.Series)
        np.testing.assert_array_equal(converted_y.compute().to_numpy(), y)
    else:
        assert isinstance(converted_y._meta, cp.ndarray)
        cp.testing.assert_array_equal(converted_y.compute(), cp.asarray(y))


@pytest.mark.parametrize("distributed", [False, True])
def test_distributed_effective_parameters_are_an_object(distributed):
    parameters = (
        [{"n_estimators": 2}, {"n_estimators": 2}]
        if distributed
        else {"n_estimators": 2}
    )

    class Estimator:
        def __init__(self, **kwargs):
            pass

        def get_params(self, deep=False):
            assert deep is False
            return parameters

    prepared = harness._PreparedCase(Estimator, (), None)
    case = resolve_case(_request(), PROFILE)
    result = {"parameters": {"effective": {}}}
    harness._construct_estimator(
        get_backend("cuml.dask"), prepared, case, None, result
    )
    expected = parameters[0] if distributed else parameters
    assert result["parameters"]["effective"] == expected


def test_dask_dbscan_uses_local_input(monkeypatch):
    case = resolve_case(_request(estimator="DBSCAN"), PROFILE)
    X = np.zeros((64, 8), dtype=np.float32)
    y = np.zeros(64, dtype=np.int64)
    spec = get_provider("cuml.dask").estimator_spec(case.estimator)
    backend = get_backend("cuml.dask")
    monkeypatch.setattr(harness, "generate_data", lambda case: (X, y))
    monkeypatch.setattr(
        backend,
        "convert_data",
        lambda *args, **kwargs: pytest.fail(
            "host inputs must bypass conversion"
        ),
    )
    monkeypatch.setattr(
        backend, "prepare_inputs", lambda inputs, client: inputs
    )
    monkeypatch.setattr(backend, "load_estimator", lambda spec: object)
    prepared = harness._prepare_case(
        backend, _suite(case, provider="cuml.dask"), case, spec
    )
    assert prepared.operation_args[0] is X


@pytest.mark.parametrize("count", [0, 1])
def test_dask_requires_multiple_gpus(monkeypatch, count):
    def import_module(name):
        assert name == "cupy", (
            "must reject before importing cluster dependencies"
        )
        return SimpleNamespace(
            cuda=SimpleNamespace(
                runtime=SimpleNamespace(getDeviceCount=lambda: count)
            )
        )

    monkeypatch.setattr(importlib, "import_module", import_module)
    case = resolve_case(_request(), PROFILE)
    with pytest.raises(
        SuiteError, match=f"at least two visible GPUs; found {count}"
    ):
        with get_backend("cuml.dask").runtime(
            _suite(case, provider="cuml.dask")
        ):
            pytest.fail("must reject before resource creation")


@pytest.mark.parametrize(
    "failure", ["csr", "timeout", "execution", "client-startup"]
)
def test_dask_runtime_safety(monkeypatch, failure):
    case = resolve_case(_request(), PROFILE)
    backend = get_backend("cuml.dask")
    if failure in {"csr", "timeout"}:
        case = (
            replace(case, input_format="csr")
            if failure == "csr"
            else replace(case, timeout_sec=1)
        )
        monkeypatch.setattr(
            importlib,
            "import_module",
            lambda name: pytest.fail(f"runtime import: {name}"),
        )
        with pytest.raises(SuiteError, match="not supported"):
            with backend.runtime(_suite(case, provider="cuml.dask")):
                pytest.fail("must reject before resource creation")
        return
    events = []

    @contextlib.contextmanager
    def cluster(**kwargs):
        events.append("cluster-enter")
        try:
            yield "cluster"
        finally:
            events.append("cluster-exit")

    @contextlib.contextmanager
    def client(cluster):
        if failure == "client-startup":
            raise RuntimeError("startup failed")
        events.append("client-enter")
        try:
            yield "client"
        finally:
            events.append("client-exit")

    modules = {
        "cupy": SimpleNamespace(
            cuda=SimpleNamespace(
                runtime=SimpleNamespace(getDeviceCount=lambda: 2)
            )
        ),
        "dask_cuda": SimpleNamespace(LocalCUDACluster=cluster),
        "dask.distributed": SimpleNamespace(Client=client),
    }
    monkeypatch.setattr(importlib, "import_module", modules.__getitem__)
    with pytest.raises(RuntimeError, match="failed"):
        with backend.runtime(_suite(case, provider="cuml.dask")) as runtime:
            assert runtime == "client"
            raise RuntimeError("execution failed")
    assert events == (
        ["cluster-enter", "cluster-exit"]
        if failure == "client-startup"
        else ["cluster-enter", "client-enter", "client-exit", "cluster-exit"]
    )


# Public runner uses the same provider coordinator as the CLI.


def test_run_builtin_defaults_and_temporary_cleanup(monkeypatch):
    paths = []

    def execute(command, environment):
        assert command[command.index("--suite") + 1] == "estimators"
        assert command[command.index("--provider") + 1] == "cuml"
        paths.append(Path(command[command.index("--output") + 1]))
        _write_worker_artifact(command)

    monkeypatch.setattr(coordinator, "run_command", execute)
    results = benchmark.run(profile="smoke")
    assert isinstance(results, benchmark.BenchmarkResults)
    assert list(results.artifacts) == ["cuml"]
    assert results.artifacts["cuml"]["results"]
    assert not paths[0].parent.exists()


@pytest.mark.parametrize("reference_type", [str, Path])
def test_run_yaml_persistence_and_resume(
    document, tmp_path, monkeypatch, reference_type
):
    path = _write_suite(tmp_path, document)
    output = tmp_path / "results"
    calls = []

    def execute(command, environment):
        calls.append("--resume" in command)
        _write_worker_artifact(command)

    monkeypatch.setattr(coordinator, "run_command", execute)
    results = benchmark.run(
        reference_type(path), providers=["scikit-learn"], output=output
    )
    artifacts = results.artifacts
    assert (
        json.loads((output / "scikit-learn.json").read_text())
        == artifacts["scikit-learn"]
    )
    assert (
        benchmark.run(
            path, providers=["scikit-learn"], output=output, resume=True
        )
        == results
    )
    assert calls == [False, True]
    with pytest.raises(SuiteError, match="results already exist"):
        benchmark.run(path, providers=["scikit-learn"], output=output)
    assert calls == [False, True]


@pytest.mark.parametrize(
    "settings",
    [
        {"providers": []},
        {"providers": ["missing"]},
        {"profile": "missing"},
        {"resume": True},
        {"resume": True, "output": "nonexistent-output"},
        {"suite": object()},
    ],
)
def test_run_configuration_errors_before_launch(
    monkeypatch, tmp_path, settings
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        coordinator,
        "run_command",
        lambda *a: pytest.fail("launched worker"),
    )
    with pytest.raises(SuiteError):
        benchmark.run(**settings)


@pytest.mark.parametrize(
    "failure",
    [
        "startup",
        "exit",
        "case",
        "missing",
        "malformed",
    ],
)
def test_run_partial_failures_continue_and_clean_output(
    document, tmp_path, monkeypatch, failure
):
    document["providers"] = ["scikit-learn", "cuml"]
    path = _write_suite(tmp_path, document)
    paths = []

    def execute(command, environment):
        provider = command[command.index("--provider") + 1]
        artifact_path = Path(command[command.index("--output") + 1])
        paths.append(artifact_path)
        if provider == "cuml":
            _write_worker_artifact(command)
            return
        if failure == "startup":
            raise OSError("startup unavailable")
        if failure == "missing":
            return
        _write_worker_artifact(command, failed=failure == "case")
        if failure == "exit":
            raise RuntimeError(
                "worker exited with status 7: startup traceback"
            )
        if failure == "malformed":
            artifact_path.write_text("not json")

    monkeypatch.setattr(coordinator, "run_command", execute)
    with pytest.raises(benchmark.BenchmarkRunError) as error:
        benchmark.run(path, providers=document["providers"])
    assert [p.stem for p in paths] == document["providers"]
    assert not paths[0].parent.exists()
    assert list(error.value.failures) == ["scikit-learn"]
    assert (
        error.value.results.artifacts["cuml"]["results"][0]["outcome"][
            "status"
        ]
        == "success"
    )
    assert ("scikit-learn" in error.value.results.artifacts) == (
        failure in {"case", "exit"}
    )
    if failure == "startup":
        assert "startup unavailable" in str(error.value)
    elif failure == "exit":
        assert "status 7: startup traceback" in str(error.value)


@pytest.mark.parametrize("failure", ["provider", "incomplete", "duplicate"])
def test_run_rejects_invalid_worker_artifact(
    document, tmp_path, monkeypatch, failure
):
    path = _write_suite(tmp_path, document)

    def execute(command, environment):
        artifact = _write_worker_artifact(command)
        if failure == "provider":
            artifact["run"]["extensions"][harness.EXTENSION]["provider"] = (
                "wrong"
            )
        else:
            artifact["results"] = (
                [] if failure == "incomplete" else artifact["results"] * 2
            )
        Path(command[command.index("--output") + 1]).write_text(
            json.dumps(artifact)
        )

    monkeypatch.setattr(coordinator, "run_command", execute)
    with pytest.raises(benchmark.BenchmarkRunError) as error:
        benchmark.run(path, providers=["scikit-learn"])
    diagnostic = (
        "invalid artifact envelope"
        if failure == "provider"
        else "artifact does not contain all expected cases"
    )
    assert diagnostic in error.value.failures["scikit-learn"]


@pytest.mark.parametrize("persistent", [False, True])
def test_run_interrupt_output_cleanup(
    document, tmp_path, monkeypatch, persistent
):
    path = _write_suite(tmp_path, document)
    paths = []

    def execute(command, environment):
        _write_worker_artifact(command)
        paths.append(Path(command[command.index("--output") + 1]))
        raise KeyboardInterrupt

    monkeypatch.setattr(coordinator, "run_command", execute)
    with pytest.raises(KeyboardInterrupt):
        benchmark.run(
            path,
            providers=["scikit-learn"],
            output=tmp_path / "results" if persistent else None,
        )
    assert paths[0].exists() == persistent
    assert paths[0].parent.exists() == persistent


@pytest.mark.parametrize(
    "stderr", ["", "startup evidence\n", "x" * 10000 + "\nfailure evidence\n"]
)
def test_run_command_bounds_failure_diagnostics(stderr, capfd):
    script = f"import sys; sys.stderr.write({stderr!r}); sys.exit(7)"
    with pytest.raises(RuntimeError) as error:
        runner.run_command([sys.executable, "-c", script], dict(os.environ))
    tail = stderr.strip()
    if len(tail) > 8192:
        tail = "...\n" + tail[-8192:]
    assert str(error.value) == "worker exited with status 7" + (
        f":\n{tail}" if tail else ""
    )
    assert capfd.readouterr().err == stderr


def test_run_command_forwards_output(monkeypatch, capfd):
    monkeypatch.setattr(
        runner.locale, "getpreferredencoding", lambda _: "utf-8"
    )
    script = (
        "import os, sys, time; "
        "print(os.environ['BENCHMARK_COMMAND_TEST'], flush=True); "
        "sys.stderr.buffer.write(b'progress \\xe2'); sys.stderr.flush(); "
        "time.sleep(0.3); "
        "sys.stderr.buffer.write(b'\\x82\\xac\\n'); sys.stderr.flush()"
    )
    runner.run_command(
        [sys.executable, "-c", script],
        {**os.environ, "BENCHMARK_COMMAND_TEST": "inherited stdout"},
    )
    captured = capfd.readouterr()
    assert captured.out == "inherited stdout\n"
    assert captured.err == "progress €\n"


@pytest.mark.parametrize("initial_timeout", [False, True])
@pytest.mark.parametrize("returncode", [0, 7])
def test_run_command_missing_stderr_snapshot(
    monkeypatch, capsys, initial_timeout, returncode
):
    monkeypatch.setattr(
        runner.locale, "getpreferredencoding", lambda _: "utf-8"
    )
    snapshots = [b"progress \xe2", None, b"progress \xe2\x82\xac\n"]
    if initial_timeout:
        snapshots.insert(0, None)

    class Process:
        stderr = SimpleNamespace(close=lambda: None)

        def __init__(self, command, **kwargs):
            self.command = command
            self.returncode = returncode
            self.snapshots = iter(snapshots)

        def communicate(self, timeout):
            snapshot = next(self.snapshots)
            if snapshot == snapshots[-1]:
                return None, snapshot
            raise subprocess.TimeoutExpired(
                self.command, timeout, stderr=snapshot
            )

        def poll(self):
            return None

    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    if returncode:
        with pytest.raises(RuntimeError) as error:
            runner.run_command(["worker"], {})
        assert str(error.value) == ("worker exited with status 7:\nprogress €")
    else:
        runner.run_command(["worker"], {})
    assert capsys.readouterr().err == "progress €\n"


@pytest.mark.skipif(
    os.name != "posix", reason="process group cleanup is POSIX-specific"
)
def test_run_interrupt_stops_provider_and_descendant(tmp_path, monkeypatch):
    import psutil

    pid_file = tmp_path / "pids.json"
    processes = []
    original = subprocess.Popen

    class InterruptedProcess(original):
        def communicate(self, **kwargs):
            deadline = time.monotonic() + 30
            while not pid_file.exists():
                assert time.monotonic() < deadline
                time.sleep(0.01)
            processes.extend(
                psutil.Process(pid) for pid in json.loads(pid_file.read_text())
            )
            raise KeyboardInterrupt

    monkeypatch.setattr(runner.subprocess, "Popen", InterruptedProcess)
    tmp_file = tmp_path / "pids.json.tmp"
    # Write then rename so pid_file never exists with partial contents.
    script = (
        "import json, os, pathlib, subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
        f"pathlib.Path({str(tmp_file)!r}).write_text(json.dumps([os.getpid(), child.pid])); "
        f"os.replace({str(tmp_file)!r}, {str(pid_file)!r}); "
        "time.sleep(120)"
    )
    with pytest.raises(KeyboardInterrupt):
        runner.run_command([sys.executable, "-c", script], dict(os.environ))
    _, alive = psutil.wait_procs(processes, timeout=5)
    assert all(process.status() == psutil.STATUS_ZOMBIE for process in alive)


def test_run_unguarded_script_real_workers(document, tmp_path):
    document["providers"] = ["scikit-learn"]
    document["profiles"]["standard"]["timeout_sec"] = 60
    path = _write_suite(tmp_path, document)
    script = tmp_path / "run.py"
    script.write_text(
        "from cuml import benchmark\nimport json\n"
        f"artifacts = benchmark.run({str(path)!r}, providers={document['providers']!r}, output={str(tmp_path / 'results')!r})\n"
        "print(json.dumps(artifacts.artifacts))\n"
    )
    completed = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    artifacts = json.loads(completed.stdout)
    assert list(artifacts) == document["providers"]
    for provider, artifact in artifacts.items():
        _validate_artifact(artifact)
        assert artifact["results"][0]["outcome"] == {"status": "success"}
        assert (
            artifact["run"]["extensions"][harness.EXTENSION]["provider"]
            == provider
        )


@pytest.mark.parametrize(
    "failure", ["stderr", "descendant-stderr", "invalid-stderr"]
)
def test_run_worker_diagnostic_failures_continue(
    document, tmp_path, monkeypatch, capsys, failure
):
    if failure == "descendant-stderr" and os.name != "posix":
        pytest.skip("descendant cleanup is POSIX-specific")
    document["providers"] = ["scikit-learn", "cuml"]
    path = _write_suite(tmp_path, document)
    pid_file = tmp_path / "descendant.pid"
    if failure == "descendant-stderr":
        script = (
            "import subprocess, sys; "
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
            f"open({str(pid_file)!r}, 'w').write(str(child.pid)); "
            "print('provider crashed', file=sys.stderr); sys.exit(7)"
        )
        diagnostic = "provider crashed"
    elif failure == "stderr":
        script = "import sys; print('startup evidence', file=sys.stderr); sys.exit(7)"
        diagnostic = "startup evidence"
    else:
        script = (
            "import sys; sys.stderr.buffer.write(b'startup \\xff\\n'); "
            "sys.stderr.flush(); sys.exit(7)"
        )
        diagnostic = "startup \ufffd"
    original = subprocess.Popen

    class BoundedProcess(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.deadline = time.monotonic() + 30

        def communicate(self, **kwargs):
            # Bound draining without requiring fast process startup on busy CI.
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                pytest.fail("worker diagnostics did not finish within 30s")
            kwargs["timeout"] = min(
                kwargs.get("timeout", remaining), remaining
            )
            return super().communicate(**kwargs)

    monkeypatch.setattr(runner.subprocess, "Popen", BoundedProcess)
    execute_worker = runner.run_command
    calls = []

    def execute(command, environment):
        provider = command[command.index("--provider") + 1]
        calls.append(provider)
        _write_worker_artifact(command)
        execute_worker(
            [
                sys.executable,
                "-c",
                script if provider == "scikit-learn" else "pass",
            ],
            environment,
        )

    monkeypatch.setattr(coordinator, "run_command", execute)
    with pytest.raises(benchmark.BenchmarkRunError) as error:
        benchmark.run(path, providers=document["providers"])
    assert calls == document["providers"]
    assert isinstance(error.value.results, benchmark.BenchmarkResults)
    assert list(error.value.results.artifacts) == document["providers"]
    assert list(error.value.failures) == ["scikit-learn"]
    assert "status 7" in error.value.failures["scikit-learn"]
    assert diagnostic in error.value.failures["scikit-learn"]
    assert diagnostic in capsys.readouterr().err
    if failure == "descendant-stderr":
        import psutil

        try:
            descendant = psutil.Process(int(pid_file.read_text()))
        except psutil.NoSuchProcess:
            return
        _, alive = psutil.wait_procs([descendant], timeout=1)
        assert all(
            process.status() == psutil.STATUS_ZOMBIE for process in alive
        )


@pytest.mark.parametrize("inference", [False, True])
def test_harness_converts_selected_host_partitions(monkeypatch, inference):
    request = _request()
    if inference:
        request.update(operation="predict", fit_input_selection=["X", "y"])
        request["dataset"]["shape"]["train_rows"] = 32
    case = resolve_case(request, PROFILE)
    X = np.arange(case.generated_rows * 8).reshape(-1, 8)
    y = np.arange(case.generated_rows)
    calls = []

    class RecordingProvider(Provider):
        def convert_data(self, value, *, estimator, role, representation):
            assert estimator == case.estimator
            assert representation == "array"
            calls.append((role, value))
            return value + 1

    provider = get_provider("scikit-learn")
    preparation_calls = []
    client = object()

    def prepare_inputs(inputs, runtime):
        assert runtime is client
        preparation_calls.append(inputs)
        return tuple(value + 10 for value in inputs)

    monkeypatch.setattr(provider.backend, "prepare_inputs", prepare_inputs)
    monkeypatch.setitem(
        PROVIDERS,
        "scikit-learn",
        RecordingProvider(provider.backend, provider.catalog),
    )
    monkeypatch.setattr(harness, "generate_data", lambda case: (X, y))
    monkeypatch.setattr(
        provider.backend, "load_estimator", lambda spec: object
    )
    prepared = harness._prepare_case(
        provider.backend,
        _suite(case),
        case,
        provider.estimator_spec("PCA"),
        client,
    )
    assert prepared.estimator_class is object
    assert len(preparation_calls) == 1
    assert len(preparation_calls[0]) == (3 if inference else 1)
    for actual, (_, original) in zip(preparation_calls[0], calls, strict=True):
        np.testing.assert_array_equal(actual, original + 1)
    assert [role for role, _ in calls] == (
        ["X", "X", "y"] if inference else ["X"]
    )
    measured_X = X[32:] if inference else X
    np.testing.assert_array_equal(calls[0][1], measured_X)
    np.testing.assert_array_equal(prepared.operation_args[0], measured_X + 11)
    if inference:
        np.testing.assert_array_equal(calls[1][1], X[:32])
        np.testing.assert_array_equal(calls[2][1], y[:32])
        np.testing.assert_array_equal(prepared.fit_args[0], X[:32] + 11)
        np.testing.assert_array_equal(prepared.fit_args[1], y[:32] + 11)
    else:
        assert prepared.fit_args is None


@pytest.mark.parametrize("kind", ["dense", "csr", "categorical"])
@pytest.mark.parametrize("mode", ["training", "inference"])
def test_host_partitions(kind, mode):
    request = _request()
    if mode == "inference":
        request.update(operation="transform", fit_input_selection=["X"])
        request["dataset"]["shape"]["train_rows"] = 128
    if kind == "csr":
        request["dataset"]["format"] = "csr"
    elif kind == "categorical":
        request["dataset"]["generator"] = "categorical"
    case = resolve_case(request, PROFILE)
    X, y = generate_data(case)
    partitions = partition_data(case, X, y)
    assert partitions.training["X"].shape[0] == case.training_rows
    assert partitions.measured["X"].shape[0] == case.measured_rows
    assert (partitions.measured is partitions.training) == (
        mode != "inference"
    )
    if mode == "training":
        assert partitions.training["X"] is X
        assert partitions.training["y"] is y

    def assert_partition(actual, rows):
        expected_X = X[rows]
        assert type(actual["X"]) is type(X)
        if kind == "categorical":
            pd.testing.assert_frame_equal(actual["X"], expected_X)
        else:
            np.testing.assert_array_equal(
                actual["X"].toarray() if kind == "csr" else actual["X"],
                expected_X.toarray() if kind == "csr" else expected_X,
            )
        np.testing.assert_array_equal(actual["y"], y[rows])
        assert np.shares_memory(actual["y"], y)

    assert_partition(partitions.training, slice(None, case.training_rows))
    assert_partition(
        partitions.measured,
        slice(case.training_rows, None)
        if mode == "inference"
        else slice(None, case.training_rows),
    )


@pytest.mark.parametrize("shape", [(64, 8), (64,)])
def test_native_conversion_preserves_shape_and_values(shape):
    host = np.ones(shape)
    device = get_backend("cuml").convert_data(host, representation="array")
    np.testing.assert_array_equal(host, device.get())


@pytest.mark.parametrize("provider", ["scikit-learn", "cuml", "cuml.dask"])
@pytest.mark.parametrize(
    "representation", ["array", "csr", "dataframe", "series"]
)
def test_structural_conversion_contract(provider, representation):
    backend = get_provider(provider).backend
    value = np.arange(12, dtype=np.float32)
    if representation in {"csr", "dataframe"}:
        value = value.reshape(4, 3)
    if provider == "cuml.dask" and representation == "csr":
        with pytest.raises(ValueError, match="CSR"):
            backend.convert_data(value, representation=representation)
        return
    if provider == "cuml.dask":
        pytest.importorskip(
            "dask.array" if representation == "array" else "dask_cudf"
        )
    converted = backend.convert_data(value, representation=representation)
    if hasattr(converted, "compute"):
        converted = converted.compute()
    if hasattr(converted, "to_numpy"):
        converted = converted.to_numpy()
    elif hasattr(converted, "get"):
        converted = converted.get()
    if sparse.issparse(converted):
        converted = converted.toarray()
    np.testing.assert_array_equal(converted, value)
    assert converted.dtype == value.dtype


@pytest.mark.parametrize("provider", ["scikit-learn", "cuml", "cuml.dask"])
def test_conversion_rejects_unknown_requirements(provider):
    backend = get_provider(provider).backend
    with pytest.raises(ValueError, match="representation"):
        backend.convert_data(np.zeros(4), representation="unknown")


def _scored_request(name="r2", operation="fit"):
    estimator, dataset, parameters = {
        "r2": ("LinearRegression", "regression", {}),
        "accuracy": (
            "LogisticRegression",
            "classification",
            {"max_iter": 200},
        ),
        "adjusted_rand": (
            "KMeans",
            "blobs",
            {"n_clusters": 5, "random_state": 42},
        ),
    }[name]
    request = _request(
        estimator=estimator,
        dataset={"generator": dataset, "shape": {"rows": 128, "features": 8}},
        parameters=parameters,
        operation=operation,
        input_selection=["X"] if name == "adjusted_rand" else ["X", "y"],
        scoring={"name": name, "evaluation_rows": 64},
    )
    if operation == "predict":
        request["fit_input_selection"] = request["input_selection"]
        request["input_selection"] = ["X"]
        request["dataset"]["shape"]["train_rows"] = 128
        request["scoring"].pop("evaluation_rows")
    return request


@pytest.mark.parametrize("name", ["accuracy", "r2", "adjusted_rand"])
@pytest.mark.parametrize("operation", ["fit", "predict"])
@pytest.mark.parametrize("provider", ["scikit-learn", "cuml"])
def test_scoring_metrics_match_held_out_reference(name, operation, provider):
    case = resolve_case(_scored_request(name, operation), PROFILE)
    result = _benchmark(case, provider=provider)
    assert result["outcome"] == {"status": "success"}, result
    assert result["observations"][0]["metrics"] == []
    partitions = partition_data(case, *generate_data(case))
    spec = get_provider("scikit-learn").estimator_spec(case.estimator)
    estimator = Backend().load_estimator(spec)(**case.parameters)
    inputs = [
        partitions.training[role]
        for role in (case.fit_input_selection or case.input_selection)
    ]
    estimator.fit(*inputs)
    expected = scoring.evaluate(
        case.scoring,
        estimator=estimator,
        output=None,
        X=partitions.evaluation["X"],
        y=partitions.evaluation["y"],
        operation="fit",
    )[0]["value"]
    for observation in result["observations"][1:]:
        (metric,) = observation["metrics"]
        assert metric["value"] == pytest.approx(expected, abs=0.05)
        assert observation["timings"][0]["value"] >= 0
    assert (
        case.workload_id()
        == resolve_case(
            _scored_request(name, operation), PROFILE
        ).workload_id()
    )


def test_scoring_split_and_identity():
    request = _scored_request()
    case = resolve_case(request, PROFILE)
    X, y = generate_data(case)
    partitions = partition_data(case, X, y)
    assert case.generated_rows == 192
    assert partitions.training is partitions.measured
    np.testing.assert_array_equal(partitions.training["X"], X[:128])
    np.testing.assert_array_equal(partitions.evaluation["X"], X[128:])
    np.testing.assert_array_equal(partitions.evaluation["y"], y[128:])
    assert not np.shares_memory(
        partitions.training["X"], partitions.evaluation["X"]
    )
    unscored = resolve_case({**request, "scoring": None}, PROFILE)
    assert case.workload_id() != unscored.workload_id()
    changed = copy.deepcopy(request)
    changed["scoring"]["evaluation_rows"] += 1
    assert case.workload_id() != resolve_case(changed, PROFILE).workload_id()
    changed = copy.deepcopy(request)
    changed["scoring"]["parameters"] = {"force_finite": False}
    assert case.workload_id() == resolve_case(changed, PROFILE).workload_id()
    scaled = resolve_case(request, {**PROFILE, "size_scale": 0.5})
    assert scaled.scoring.evaluation_rows == 32
    assert scaled.training_rows == 64


@pytest.mark.parametrize(
    "change,match",
    [
        ({"name": "missing", "evaluation_rows": 64}, "unknown scorer"),
        (
            {"name": "accuracy", "evaluation_rows": 64},
            "requires.*classification",
        ),
        ({"name": "r2"}, "requires evaluation_rows"),
        ({"name": "r2", "evaluation_rows": 0}, "invalid"),
    ],
)
def test_invalid_scoring_configuration(change, match):
    with pytest.raises(SuiteError, match=match):
        resolve_case({**_scored_request(), "scoring": change}, PROFILE)


@pytest.mark.parametrize("failure", [False, True])
def test_scoring_outside_timing_and_verification(monkeypatch, failure):
    case = resolve_case(_scored_request(), PROFILE)
    backend = get_provider("scikit-learn").backend
    events = []

    @contextlib.contextmanager
    def instrumentation(*args):
        events.append("enter")
        yield base_backend.VerificationResult()
        events.append("exit")

    def clock():
        events.append("clock")
        return len(events)

    original = scoring.SCORERS["r2"]

    def score(**kwargs):
        assert events[-3:] == ["clock", "clock", "exit"]
        events.append("score")
        if failure:
            raise RuntimeError("evaluation failed")
        return original.evaluate(**kwargs)

    monkeypatch.setattr(backend, "instrumentation", instrumentation)
    monkeypatch.setattr(harness.time, "perf_counter", clock)
    monkeypatch.setitem(
        scoring.SCORERS, "r2", replace(original, evaluate=score)
    )
    result = _benchmark(case)
    assert events.count("score") == (1 if failure else 2)
    if failure:
        assert result["outcome"]["last_phase"] == "scoring"
        assert len(result["observations"]) == 2
        observation = result["observations"][-1]
        assert observation["outcome"]["status"] == "failed"
        assert observation["timings"][0]["value"] == 1
    else:
        assert result["outcome"]["status"] == "success"


@pytest.mark.parametrize("force_finite", [None, True, False])
def test_r2_constant_targets(force_finite):
    case = resolve_case(_scored_request(), PROFILE)
    parameters = {} if force_finite is None else {"force_finite": force_finite}
    spec = replace(case.scoring, parameters=parameters)
    kwargs = dict(
        estimator=None,
        output=np.zeros(32),
        X=None,
        y=np.ones(32),
        operation="predict",
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        if force_finite is True:
            assert scoring.evaluate(spec, **kwargs)[0]["value"] == 0
        else:
            with pytest.raises(ValueError, match="nonfinite"):
                scoring.evaluate(spec, **kwargs)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_scoring_results_fail_case(monkeypatch, value):
    original = scoring.SCORERS["r2"]

    def score(**kwargs):
        metrics = original.evaluate(**kwargs)
        metrics[0]["value"] = value
        return metrics

    monkeypatch.setitem(
        scoring.SCORERS, "r2", replace(original, evaluate=score)
    )
    result = _benchmark(resolve_case(_scored_request(), PROFILE))
    assert result["outcome"]["last_phase"] == "scoring"
    assert result["observations"][-1]["timings"]
    assert result["observations"][-1]["metrics"] == []


@pytest.mark.parametrize("change", ["version", "parameters", "disabled"])
def test_scoring_artifact_and_resume(tmp_path, monkeypatch, change):
    case = resolve_case(_scored_request(), PROFILE)
    suite = _suite(case)
    path = tmp_path / "scoring.json"
    artifact = harness.run_suite(suite, path)
    _validate_artifact(artifact)
    assert (
        artifact["results"][0]["extensions"][harness.EXTENSION]["scoring"]
        == case.scoring.descriptor()
    )
    assert (
        harness.run_suite(suite, path, resume=True)["results"]
        == artifact["results"]
    )
    if change == "version":
        original = scoring.SCORERS["r2"]
        monkeypatch.setitem(
            scoring.SCORERS,
            "r2",
            replace(original, version=original.version + "-changed"),
        )
    elif change == "parameters":
        suite = _suite(
            replace(
                case,
                scoring=replace(
                    case.scoring, parameters={"force_finite": False}
                ),
            )
        )
    else:
        suite = _suite(replace(case, scoring=None))
    with pytest.raises(SuiteError, match="matching suite"):
        harness.run_suite(suite, path, resume=True)


def test_scoring_failure_artifact_preserves_timing(tmp_path):
    request = _scored_request()
    request["scoring"]["parameters"] = {"unknown_parameter": True}
    artifact = harness.run_suite(
        _suite(resolve_case(request, PROFILE)), tmp_path / "failed.json"
    )
    _validate_artifact(artifact)
    (result,) = artifact["results"]
    assert result["outcome"]["last_phase"] == "scoring"
    assert len(result["observations"]) == 2
    observation = result["observations"][-1]
    assert observation["outcome"]["last_phase"] == "scoring"
    assert observation["timings"]
    assert observation["metrics"] == []


def test_scoring_inference_reuses_measured_predictions(monkeypatch):
    case = resolve_case(_scored_request(operation="predict"), PROFILE)
    from sklearn.linear_model import LinearRegression

    original = LinearRegression.predict
    calls = []

    def predict(self, X):
        calls.append(X)
        return original(self, X)

    monkeypatch.setattr(LinearRegression, "predict", predict)
    result = _benchmark(case)
    assert result["outcome"] == {"status": "success"}
    assert len(calls) == case.warmups + case.repetitions


@pytest.mark.parametrize(
    "estimator,scorer,parameters,dataset",
    [
        ("PCA", "reconstruction", {"n_components": 2}, "blobs"),
        ("TruncatedSVD", "reconstruction", {"n_components": 2}, "blobs"),
        (
            "GaussianRandomProjection",
            "distance_distortion",
            {"n_components": 2, "random_state": 42},
            "blobs",
        ),
        ("EmpiricalCovariance", "gaussian_log_likelihood", {}, "blobs"),
        ("LedoitWolf", "gaussian_log_likelihood", {}, "blobs"),
    ],
)
@pytest.mark.parametrize("provider", ["cuml", "scikit-learn"])
def test_scorer_provider_integration(
    estimator, scorer, parameters, dataset, provider
):
    case = resolve_case(
        _request(
            estimator=estimator,
            operation="fit",
            input_selection=["X"],
            dataset={
                "generator": dataset,
                "shape": {"rows": 512, "features": 8},
            },
            parameters=parameters,
            scoring={"name": scorer, "evaluation_rows": 64},
        ),
        PROFILE,
    )
    result = _benchmark(case, provider=provider)
    assert result["outcome"] == {"status": "success"}, result
    (metric,) = result["observations"][-1]["metrics"]
    assert np.isfinite(metric["value"])


@pytest.mark.parametrize("provider", ["scikit-learn", "cuml", "cuml.dask"])
def test_covariance_metric_matches_analytic_density(provider):
    from cuml.benchmark.metrics import covariance

    estimator = SimpleNamespace(covariance_=np.eye(2), location_=np.zeros(2))
    expected = -np.log(2 * np.pi) - 0.5
    actual = covariance(
        estimator=estimator,
        X=get_provider(provider).backend.convert_data(
            np.array([[1.0, 0.0], [0.0, 1.0]]), representation="array"
        ),
        parameters={},
    )
    assert actual[0]["value"] == pytest.approx(expected)


@pytest.mark.parametrize("provider", ["scikit-learn", "cuml", "cuml.dask"])
def test_projection_metric_preserves_distances(provider):
    from cuml.benchmark.metrics import projection

    X = np.random.default_rng(42).normal(size=(32, 4))
    actual = projection(
        estimator=SimpleNamespace(transform=lambda X: X),
        X=get_provider(provider).backend.convert_data(
            X, representation="array"
        ),
        parameters={"pairs": 100},
    )
    assert actual[0]["value"] == pytest.approx(0)
