#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic companion grader for cuML evals."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

TRAJECTORY = Path(
    os.environ.get("HARBOR_ATIF_PATH", "/logs/agent/trajectory.json")
)
ENTRY = Path(os.environ.get("HARBOR_ENTRY_JSON", "/tests/entry.json"))
REWARD_JSON = Path(
    os.environ.get("HARBOR_REWARD_JSON", "/logs/verifier/reward.json")
)
REWARD_TXT = Path(
    os.environ.get("HARBOR_REWARD_TXT", "/logs/verifier/reward.txt")
)
WORKSPACE = Path(os.environ.get("HARBOR_WORKSPACE", "/workspace"))


def find_fixture(filename):
    matches = list(WORKSPACE.rglob(filename))
    return matches[0].resolve() if matches else None


def trajectory_text():
    if not TRAJECTORY.exists():
        return ""
    data = json.loads(TRAJECTORY.read_text(encoding="utf-8"))
    return " ".join(
        json.dumps(step.get("message", ""))
        for step in data.get("steps", [])
        if step.get("source") == "agent"
    ).lower()


def main():
    entry = json.loads(ENTRY.read_text())
    case_id = entry["id"]
    text = trajectory_text()
    checks = {}

    if case_id == "cuml-ml-explicit-temporal-classifier":
        train = find_fixture("train.py")
        smoke = find_fixture("smoke_split.py")
        code = train.read_text().lower() if train else ""
        trusted = bool(
            smoke
            and hashlib.sha256(smoke.read_bytes()).hexdigest()
            == "c61fb2dce36756772301fc9edd21f0de03b24bd8489594748cb491f72db4797c"
        )
        smoke_ok = (
            subprocess.run(
                ["python3", str(smoke)],
                cwd=smoke.parent.parent,
                capture_output=True,
            ).returncode
            == 0
            if trusted
            else False
        )
        syntax_ok = (
            subprocess.run(
                ["python3", "-m", "py_compile", str(train)],
                capture_output=True,
            ).returncode
            == 0
            if train
            else False
        )
        checks = {
            "temporal_split_smoke": (0.5, smoke_ok),
            "syntax": (0.1, syntax_ok),
            "no_random_split": (0.15, "train_test_split" not in code),
            "keyed_output": (
                0.15,
                "prediction_origin" in code and "account_id" in code,
            ),
            "execution_evidence": (
                0.1,
                ("cpu" in text and ("gpu" in text or "cuml" in text))
                or (
                    "gpu" in text
                    and any(
                        term in text
                        for term in ("not run", "blocked", "unavailable")
                    )
                ),
            ),
        }
    elif case_id == "cuml-ml-implicit-accel-fallback":
        checks = {
            "dispatch_or_profile": (
                0.25,
                "profile" in text or "dispatch" in text or "verbose" in text,
            ),
            "known_fallback": (
                0.25,
                "positive=true" in text.replace(" ", "")
                and "fallback" in text,
            ),
            "preserves_constraint": (
                0.25,
                "constraint" in text
                and any(
                    term in text
                    for term in (
                        "do not",
                        "don't",
                        "not change",
                        "changes",
                        "if the",
                        "if this",
                    )
                ),
            ),
            "fair_timing": (
                0.25,
                any(
                    term in text for term in ("benchmark", "timing", "speedup")
                )
                and any(
                    term in text
                    for term in ("same", "identical", "controlled", "warm")
                ),
            ),
        }
    elif case_id == "cuml-ml-contextual-clustering":
        checks = {
            "arbitrary_labels": (0.2, "label" in text and "arbitrary" in text),
            "permutation_invariant": (
                0.2,
                any(
                    term in text
                    for term in ("permutation", "ari", "ami", "adjusted rand")
                ),
            ),
            "stability_and_setup": (
                0.3,
                "stability" in text
                and any(
                    term in text
                    for term in ("scal", "hyperparameter", "seed", "initial")
                )
                and any(
                    term in text
                    for term in ("useful", "task", "business", "downstream")
                ),
            ),
            "stable_device_key": (
                0.3,
                "device" in text and ("key" in text or "id" in text),
            ),
        }
    else:
        checks = {
            "routes_to_cudf": (
                0.7,
                "cudf" in text or "data preparation" in text,
            ),
            "no_invented_ml": (
                0.3,
                not any(
                    term in text
                    for term in (
                        "classifier",
                        "regressor",
                        "model.fit",
                        "kmeans",
                        "randomforest",
                    )
                ),
            ),
        }

    score = sum(weight for weight, passed in checks.values() if passed)
    details = {
        name: {
            "score": float(passed),
            "reason": "passed" if passed else "failed",
        }
        for name, (_, passed) in checks.items()
    }
    reward = {
        "overall": score,
        "custom_metrics": {"ml_contract_correctness": score},
        "details": details,
    }
    REWARD_JSON.parent.mkdir(parents=True, exist_ok=True)
    REWARD_JSON.write_text(json.dumps(reward, indent=2))
    REWARD_TXT.write_text(str(score))


if __name__ == "__main__":
    main()
