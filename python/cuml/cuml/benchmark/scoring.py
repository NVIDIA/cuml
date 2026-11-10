# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named, provider-independent quality scorers, executed outside timing."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any, Callable

from . import metrics


@dataclass(frozen=True)
class Scorer:
    """Versioned scorer returning artifact metric dictionaries.

    Increment ``version`` whenever evaluation semantics change. Callbacks must
    not refit or mutate the estimator, which inference repetitions may reuse.
    """

    version: str
    evaluate: Callable[..., list[dict[str, Any]]]
    dataset: str | tuple[str, ...]
    operations: tuple[str, ...] = (
        "fit",
        "fit_predict",
        "fit_transform",
        "predict",
    )
    input_selection: tuple[str, ...] = ("X",)


def _to_host(value: Any) -> Any:
    """Collect distributed outputs, then transfer device arrays to the host."""
    if hasattr(value, "compute"):
        value = value.compute()
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    if hasattr(value, "get"):
        value = value.get()
    return value


def _prediction_metric(
    *, estimator, output, X, y, operation, parameters, name, **unused
) -> list[dict[str, Any]]:
    """Evaluate a measured prediction or predict on held-out fitting inputs."""
    prediction = output if operation == "predict" else estimator.predict(X)
    metric = getattr(importlib.import_module("sklearn.metrics"), name)
    value = float(metric(y, _to_host(prediction), **parameters))
    return [
        {
            "name": name,
            "value": value,
            "unit": "dimensionless",
            "direction": "higher_is_better",
        }
    ]


def _accuracy(**kwargs):
    return _prediction_metric(name="accuracy_score", **kwargs)


def _r2(*, parameters, **kwargs):
    return _prediction_metric(
        name="r2_score",
        parameters={"force_finite": False, **parameters},
        **kwargs,
    )


def _adjusted_rand(**kwargs):
    return _prediction_metric(name="adjusted_rand_score", **kwargs)


SCORERS = {
    "accuracy": Scorer(
        "1", _accuracy, ("classification", "positive", "matrix", "categorical")
    ),
    "r2": Scorer("2", _r2, "regression"),
    "adjusted_rand": Scorer("1", _adjusted_rand, "blobs"),
    "reconstruction": Scorer("1", metrics.reconstruction, "blobs"),
    "distance_distortion": Scorer("1", metrics.projection, "blobs"),
    "trustworthiness": Scorer("1", metrics.trustworthiness, "blobs"),
    "gaussian_log_likelihood": Scorer("1", metrics.covariance, "blobs"),
    "log_density": Scorer("1", metrics.density, "blobs"),
}


@dataclass(frozen=True)
class ScoringSpec:
    """Resolved case scoring configuration, separate from timing identity."""

    name: str
    parameters: dict[str, Any]
    evaluation_rows: int

    def descriptor(self) -> dict[str, Any]:
        """Describe the evaluation contract for resume and score comparison."""
        return {
            "name": self.name,
            "version": SCORERS[self.name].version,
            "parameters": dict(self.parameters),
            "evaluation_rows": self.evaluation_rows,
            "population": "held-out",
            "partition": "disjoint-contiguous-v1",
        }


def evaluate(
    spec: ScoringSpec,
    *,
    estimator: Any,
    output: Any,
    X: Any,
    y: Any,
    operation: str,
) -> list[dict[str, Any]]:
    """Run an internal scorer and reject nonfinite metric values."""
    metrics = SCORERS[spec.name].evaluate(
        estimator=estimator,
        output=output,
        X=X,
        y=y,
        operation=operation,
        parameters=dict(spec.parameters),
    )
    for metric in metrics:
        if not math.isfinite(metric["value"]):
            raise ValueError("scorer returned a nonfinite metric")
    return metrics
