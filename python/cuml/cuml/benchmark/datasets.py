# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate dataset settings and generate reproducible benchmark inputs."""

from __future__ import annotations

import importlib
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .suite import ResolvedCase

DATA_GENERATOR = "com.nvidia.cuml.benchmark.generate-data-v1"
DATA_SEED = 42
DATASETS = frozenset(
    {
        "blobs",
        "classification",
        "regression",
        "matrix",
        "positive",
        "categorical",
    }
)


def _integer(value: Any, name: str, minimum: int = 1) -> None:
    """Validate an integer setting against its lower bound."""
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _positive_finite(value: Any, name: str) -> None:
    """Validate a positive finite numeric setting."""
    valid = isinstance(value, (int, float)) and not isinstance(value, bool)
    try:
        valid = valid and value > 0 and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a positive finite scalar")


def resolve_dtypes(dtype: str | dict[str, str]) -> dict[str, str]:
    """Validate and normalize feature and target data types.

    Parameters
    ----------
    dtype : str or dict of str to str
        Shared type name or separate X and y type names.
    """
    if isinstance(dtype, str):
        requested = {"X": dtype, "y": dtype}
    elif isinstance(dtype, dict) and set(dtype) == {"X", "y"}:
        requested = dtype
    else:
        raise ValueError(
            "dtype must be a type name or a mapping specifying both X and y"
        )
    for name, value in requested.items():
        if not isinstance(value, str) or value not in {
            "float32",
            "float64",
            "int32",
            "int64",
        }:
            raise ValueError(
                f"dtype for {name} must be float32, float64, int32, or int64"
            )
    return dict(requested)


def resolve_dataset(
    dataset: str,
    features: int,
    dataset_parameters: dict[str, Any] | None = None,
    dtype: str | dict[str, str] = "float32",
    input_format: str = "dense",
) -> dict[str, Any]:
    """Validate dataset settings and resolve generator defaults.

    Parameters
    ----------
    dataset : str
        Dataset generator name.
    features : int
        Number of feature columns.
    dataset_parameters : dict, optional
        Generator settings overriding defaults.
    dtype : str or dict of str to str, default="float32"
        Shared or separate X and y data types.
    input_format : str, default="dense"
        Input representation: dense or csr.
    """
    if not isinstance(dataset, str) or dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}")
    _integer(features, "features")
    resolve_dtypes(dtype)
    if not isinstance(input_format, str) or input_format not in {
        "dense",
        "csr",
    }:
        raise ValueError("input_format must be 'dense' or 'csr'")
    if input_format == "csr" and dataset not in {"classification", "matrix"}:
        raise ValueError("CSR inputs support only classification and matrix")
    if dataset_parameters is None:
        dataset_parameters = {}
    if not isinstance(dataset_parameters, dict):
        raise ValueError("dataset_parameters must be a mapping")
    if any(not isinstance(key, str) for key in dataset_parameters):
        raise ValueError("dataset_parameters keys must be strings")

    parameters: dict[str, Any] = {}
    if dataset == "blobs":
        parameters = {"centers": 5, "cluster_std": 1.0}
    elif dataset == "classification":
        parameters = {
            "n_classes": 2,
            "n_informative": max(2, min(features, features // 2 + 1)),
            "n_redundant": 0,
            "n_clusters_per_class": 2,
        }
    if input_format == "csr":
        parameters["density"] = 0.1
    unknown = set(dataset_parameters) - set(parameters)
    if unknown:
        raise ValueError(
            f"unknown dataset parameter(s) for {dataset!r} ({input_format}): "
            f"{', '.join(sorted(unknown))}"
        )
    parameters.update(dataset_parameters)
    if dataset == "blobs":
        _integer(parameters["centers"], "centers")
        _positive_finite(parameters["cluster_std"], "cluster_std")
    elif dataset == "classification":
        for name in ("n_classes", "n_informative", "n_clusters_per_class"):
            _integer(parameters[name], name)
        _integer(parameters["n_redundant"], "n_redundant", minimum=0)
        if parameters["n_informative"] + parameters["n_redundant"] > features:
            raise ValueError(
                "n_informative + n_redundant must not exceed features"
            )
        clusters = parameters["n_classes"] * parameters["n_clusters_per_class"]
        if (clusters - 1).bit_length() > parameters["n_informative"]:
            raise ValueError(
                "n_classes * n_clusters_per_class must not exceed "
                "2 ** n_informative"
            )
    if input_format == "csr":
        _positive_finite(parameters["density"], "density")
        if parameters["density"] > 1:
            raise ValueError("density must be in (0, 1]")
    return parameters


def generate_data(case: ResolvedCase) -> tuple[Any, Any]:
    """Generate inputs from dataset settings, independently of the estimator.

    Categorical features are pandas DataFrames, other dense features are
    NumPy arrays, and CSR features are SciPy sparse matrices. Targets are
    always NumPy arrays. Backends adapt these host representations as needed.

    Parameters
    ----------
    case : ResolvedCase
        Resolved dataset shape, types, and generator settings.
    """
    input_format = case.input_format
    parameters = case.dataset_parameters
    rows = case.generated_rows
    np = importlib.import_module("numpy")
    if case.dataset in {"classification", "regression", "blobs"}:
        datasets = importlib.import_module("sklearn.datasets")
        generator_parameters = {
            key: value for key, value in parameters.items() if key != "density"
        }
        generator = getattr(datasets, f"make_{case.dataset}")
        X, y = generator(
            n_samples=rows,
            n_features=case.features,
            random_state=DATA_SEED,
            **generator_parameters,
        )
    else:
        rng = np.random.default_rng(DATA_SEED)
        X = rng.normal(size=(rows, case.features))
        y = rng.integers(0, 3, size=rows)
        if case.dataset == "positive":
            X = np.abs(X)
        elif case.dataset == "categorical":
            X = rng.integers(0, 8, size=(rows, case.features))
            y = rng.integers(0, 2, size=rows)
    dtypes = case.dtypes
    X = np.asarray(X, dtype=dtypes["X"])
    y = np.asarray(y, dtype=dtypes["y"])
    if case.dataset == "categorical":
        pandas = importlib.import_module("pandas")
        X = pandas.DataFrame(
            X, columns=[f"feature_{index}" for index in range(case.features)]
        )
    if input_format == "csr":
        sparse = importlib.import_module("scipy.sparse")
        mask = (
            np.random.default_rng(DATA_SEED).random(X.shape)
            < parameters["density"]
        )
        X = sparse.csr_matrix(np.where(mask, X, 0))
    return X, y
