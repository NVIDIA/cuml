# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host metrics for held-out benchmark scoring."""

from __future__ import annotations

import numpy as np


def metric(name, value, *, higher=False, unit="dimensionless"):
    """Build one scalar artifact metric."""
    return [
        {
            "name": name,
            "value": float(value),
            "unit": unit,
            "direction": "higher_is_better" if higher else "lower_is_better",
        }
    ]


def dense(value):
    """Collect a bounded scoring output as a host array."""
    from .scoring import _to_host

    value = _to_host(value)
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value)


def relative_error(actual, expected):
    """Return relative Frobenius error, with an absolute zero-reference fallback."""
    actual, expected = (
        dense(actual).astype(np.float64),
        dense(expected).astype(np.float64),
    )
    if actual.shape != expected.shape:
        raise ValueError(
            f"scoring shape mismatch: {actual.shape} != {expected.shape}"
        )
    return np.linalg.norm(actual - expected) / max(
        np.linalg.norm(expected), np.finfo(float).eps
    )


def reconstruction(*, estimator, X, parameters, **unused):
    """Measure held-out transform/inverse-transform reconstruction error."""
    if parameters:
        raise ValueError("reconstruction has no parameters")
    reconstructed = estimator.inverse_transform(estimator.transform(X))
    return metric(
        "relative_reconstruction_error",
        relative_error(reconstructed, X),
    )


def projection(*, estimator, X, parameters, **unused):
    """Measure relative distance distortion over deterministic held-out pairs."""
    pairs = parameters.get("pairs", 1024)
    if set(parameters) - {"pairs"} or not isinstance(pairs, int) or pairs < 1:
        raise ValueError("projection requires a positive pairs count")
    Z = dense(estimator.transform(X)).astype(np.float64)
    X = dense(X).astype(np.float64)
    rng = np.random.default_rng(42)
    left = rng.integers(len(X), size=pairs)
    right = (left + rng.integers(1, len(X), size=pairs)) % len(X)
    original = np.linalg.norm(X[left] - X[right], axis=1)
    projected = np.linalg.norm(Z[left] - Z[right], axis=1)
    nonzero = original > 0
    if not np.any(nonzero):
        raise ValueError("projection evaluation requires distinct points")
    return metric(
        "mean_relative_distance_distortion",
        np.mean(np.abs(projected[nonzero] / original[nonzero] - 1)),
    )


def trustworthiness(*, estimator, X, parameters, **unused):
    """Evaluate a bounded held-out embedding without comparing coordinates."""
    from sklearn.manifold import trustworthiness as score

    Z = dense(estimator.transform(X))
    return metric(
        "trustworthiness",
        score(dense(X), Z, **parameters),
        higher=True,
    )


def covariance(*, estimator, X, parameters, **unused):
    """Compute common Gaussian log likelihood from fitted mean and covariance."""
    if parameters:
        raise ValueError("gaussian_log_likelihood has no parameters")
    covariance = dense(estimator.covariance_).astype(np.float64)
    centered = dense(X).astype(np.float64) - dense(estimator.location_)
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("fitted covariance is not positive definite")
    quadratic = np.sum(
        centered * np.linalg.solve(covariance, centered.T).T, axis=1
    )
    value = -0.5 * np.mean(
        centered.shape[1] * np.log(2 * np.pi) + logdet + quadratic
    )
    return metric(
        "mean_gaussian_log_likelihood", value, higher=True, unit="nats/sample"
    )


def density(*, estimator, X, parameters, **unused):
    """Evaluate held-out log density with the fitted provider estimator."""
    if parameters:
        raise ValueError("log_density has no parameters")
    return metric(
        "mean_log_density",
        dense(estimator.score_samples(X)).mean(),
        higher=True,
        unit="nats/sample",
    )
