# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named, provider-independent quality scorers, executed outside timing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from . import metrics


@dataclass(frozen=True)
class EvaluationContext:
    """Provide a fitted estimator, held-out data, and the measured output.

    ``X`` retains the provider's representation; ``y`` remains on the host.
    Callbacks must preserve the estimator's fitted state.
    """

    estimator: Any
    X: Any
    y: Any
    output: Any
    operation: str

    def predictions(self) -> Any:
        """Reuse measured predictions or predict on held-out fitting inputs."""
        if self.operation == "predict":
            return self.output
        return self.estimator.predict(self.X)


def _to_host(value: Any) -> Any:
    """Collect distributed outputs, then transfer device arrays to the host."""
    if hasattr(value, "compute"):
        value = value.compute()
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    if hasattr(value, "get"):
        value = value.get()
    return value


@dataclass(frozen=True)
class Scorer:
    """Define a versioned metric callback and its supported workloads.

    Callbacks evaluate fitted estimators while preserving their state.
    Increment ``version`` when evaluation semantics change.
    """

    version: str
    callback: Callable[
        [EvaluationContext, dict[str, Any]], list[dict[str, Any]]
    ]
    operations: tuple[str, ...] = (
        "fit",
        "fit_predict",
        "fit_transform",
        "predict",
    )
    input_selection: tuple[str, ...] = ("X",)


def _accuracy(context: EvaluationContext, parameters):
    """Collect held-out predictions for host accuracy evaluation."""
    return metrics.accuracy(
        y=_to_host(context.y),
        prediction=_to_host(context.predictions()),
        parameters=parameters,
    )


def _r2(context: EvaluationContext, parameters):
    """Collect held-out predictions for host R² evaluation."""
    return metrics.r2(
        y=_to_host(context.y),
        prediction=_to_host(context.predictions()),
        parameters=parameters,
    )


def _adjusted_rand(context: EvaluationContext, parameters):
    """Collect held-out predictions for host clustering evaluation."""
    return metrics.adjusted_rand(
        y=_to_host(context.y),
        prediction=_to_host(context.predictions()),
        parameters=parameters,
    )


def _reconstruction(context: EvaluationContext, parameters):
    """Reconstruct held-out inputs using the provider, then collect to host."""
    reconstructed = context.estimator.inverse_transform(
        context.estimator.transform(context.X)
    )
    return metrics.reconstruction(
        X=_to_host(context.X),
        reconstructed=_to_host(reconstructed),
        parameters=parameters,
    )


def _projection(context: EvaluationContext, parameters):
    """Project held-out inputs using the provider, then collect to host."""
    Z = context.estimator.transform(context.X)
    return metrics.projection(
        X=_to_host(context.X), Z=_to_host(Z), parameters=parameters
    )


def _trustworthiness(context: EvaluationContext, parameters):
    """Embed held-out inputs using the provider, then collect to host."""
    Z = context.estimator.transform(context.X)
    return metrics.trustworthiness(
        X=_to_host(context.X), Z=_to_host(Z), parameters=parameters
    )


def _covariance(context: EvaluationContext, parameters):
    """Collect held-out inputs and fitted covariance parameters to host."""
    return metrics.covariance(
        X=_to_host(context.X),
        covariance=_to_host(context.estimator.covariance_),
        location=_to_host(context.estimator.location_),
        parameters=parameters,
    )


def _density(context: EvaluationContext, parameters):
    """Evaluate held-out log density using the provider, then collect to host."""
    return metrics.density(
        log_density=_to_host(context.estimator.score_samples(context.X)),
        parameters=parameters,
    )


SCORERS = {
    "accuracy": Scorer("1", _accuracy),
    "r2": Scorer("1", _r2),
    "adjusted_rand": Scorer("1", _adjusted_rand),
    "reconstruction": Scorer("1", _reconstruction),
    "distance_distortion": Scorer("1", _projection),
    "trustworthiness": Scorer("1", _trustworthiness),
    "gaussian_log_likelihood": Scorer("1", _covariance),
    "log_density": Scorer("1", _density),
}


@dataclass(frozen=True)
class ScoringSpec:
    """Store a case's scorer, parameters, and evaluation population size."""

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
    context: EvaluationContext,
) -> list[dict[str, Any]]:
    """Evaluate a registered scorer and validate that metric values are finite."""
    scorer = SCORERS[spec.name]
    metrics = scorer.callback(context, dict(spec.parameters))
    for metric in metrics:
        if not math.isfinite(metric["value"]):
            raise ValueError("scorer returned a nonfinite metric")
    return metrics
