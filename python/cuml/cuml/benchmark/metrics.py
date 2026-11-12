# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quality metrics accepting only host data, never estimators or device values.

The scoring layer acquires estimator responses and transfers them to the host
before calling these functions. Sparse host inputs are densified as needed.
"""

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
    """Materialize a host value as a dense NumPy array."""
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value)


def relative_error(actual, expected):
    """Compute Frobenius error normalized by the epsilon-bounded reference norm."""
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


def accuracy(*, y, prediction, parameters):
    """Measure held-out classification accuracy."""
    from sklearn.metrics import accuracy_score

    return metric(
        "accuracy_score",
        accuracy_score(y, prediction, **parameters),
        higher=True,
    )


def r2(*, y, prediction, parameters):
    """Measure held-out R² without masking nonfinite scores by default."""
    from sklearn.metrics import r2_score

    return metric(
        "r2_score",
        r2_score(
            y,
            prediction,
            **{"force_finite": False, **parameters},
        ),
        higher=True,
    )


def adjusted_rand(*, y, prediction, parameters):
    """Measure held-out clustering agreement with reference labels."""
    from sklearn.metrics import adjusted_rand_score

    return metric(
        "adjusted_rand_score",
        adjusted_rand_score(y, prediction, **parameters),
        higher=True,
    )


def reconstruction(*, X, reconstructed, parameters):
    """Measure held-out reconstruction error from host inputs and responses."""
    if parameters:
        raise ValueError("reconstruction has no parameters")
    return metric(
        "relative_reconstruction_error",
        relative_error(reconstructed, X),
    )


def projection(*, X, Z, parameters):
    """Measure relative distance distortion over deterministic held-out pairs."""
    pairs = parameters.get("pairs", 1024)
    if set(parameters) - {"pairs"} or not isinstance(pairs, int) or pairs < 1:
        raise ValueError("projection requires a positive pairs count")
    Z = dense(Z).astype(np.float64)
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


def trustworthiness(*, X, Z, parameters):
    """Measure neighborhood preservation in a held-out embedding."""
    from sklearn.manifold import trustworthiness as score

    return metric(
        "trustworthiness",
        score(dense(X), dense(Z), **parameters),
        higher=True,
    )


def covariance(*, X, covariance, location, parameters):
    """Compute mean held-out Gaussian log likelihood from fitted parameters."""
    if parameters:
        raise ValueError("gaussian_log_likelihood has no parameters")
    covariance = dense(covariance).astype(np.float64)
    centered = dense(X).astype(np.float64) - dense(location)
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


def density(*, log_density, parameters):
    """Compute mean held-out log density from host per-sample values."""
    if parameters:
        raise ValueError("log_density has no parameters")
    return metric(
        "mean_log_density",
        dense(log_density).mean(),
        higher=True,
        unit="nats/sample",
    )
