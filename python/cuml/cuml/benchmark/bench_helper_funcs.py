#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import pickle as pickle
import sys

import numpy as np
import pandas as pd

# Supports both package and standalone execution
try:
    from cuml.benchmark import datagen
    from cuml.benchmark.gpu_check import is_cuml_available
except ImportError:
    if not any("cuml/benchmark" in p for p in sys.path):
        raise
    import datagen  # noqa: E402
    from gpu_check import is_cuml_available  # noqa: E402

# Conditional GPU imports
if is_cuml_available():
    import cudf
    import cupy as cp

    from cuml.manifold import UMAP
else:
    cudf = cp = UMAP = None


def call(m, func_name, X, y=None):
    def unwrap_and_get_args(func):
        if hasattr(func, "__wrapped__"):
            return unwrap_and_get_args(func.__wrapped__)
        else:
            return func.__code__.co_varnames

    if not hasattr(m, func_name):
        raise ValueError("Model does not have function " + func_name)
    func = getattr(m, func_name)
    argnames = unwrap_and_get_args(func)
    if y is not None and "y" in argnames:
        func(X, y=y)
    else:
        func(X)


def pass_func(m, x, y=None):
    pass


def fit(m, x, y=None):
    call(m, "fit", x, y)


def predict(m, x, y=None):
    call(m, "predict", x)


def transform(m, x, y=None):
    call(m, "transform", x)


def kneighbors(m, x, y=None):
    call(m, "kneighbors", x)


def score_samples(m, x, y=None):
    call(m, "score_samples", x[: min(x.shape[0], 1024)])


def fit_predict(m, x, y=None):
    if hasattr(m, "predict"):
        fit(m, x, y)
        predict(m, x)
    else:
        call(m, "fit_predict", x, y)


def fit_transform(m, x, y=None):
    if hasattr(m, "transform"):
        fit(m, x, y)
        transform(m, x)
    else:
        call(m, "fit_transform", x, y)


def fit_kneighbors(m, x, y=None):
    if hasattr(m, "kneighbors"):
        fit(m, x, y)
        kneighbors(m, x)
    else:
        call(m, "fit_kneighbors", x, y)


def fit_score_samples(m, x, y=None):
    fit(m, x, y)
    score_samples(m, x)


def _training_data_to_numpy(X, y):
    """Convert input training data into numpy format"""
    if isinstance(X, np.ndarray):
        X_np = X
        y_np = y
    elif is_cuml_available() and isinstance(X, cp.ndarray):
        X_np = cp.asnumpy(X)
        y_np = cp.asnumpy(y) if y is not None else None
    elif is_cuml_available() and isinstance(X, cudf.DataFrame):
        X_np = X.to_numpy()
        y_np = y.to_numpy() if y is not None else None
    elif isinstance(X, (pd.DataFrame, pd.Series)):
        X_np = datagen._convert_to_numpy(X)
        y_np = datagen._convert_to_numpy(y) if y is not None else None
    else:
        raise TypeError("Received unsupported input type: %s" % type(X))
    return X_np, y_np


def _build_mnmg_umap(m, data, args, tmpdir):
    """Build multi-node multi-GPU UMAP model.

    Note: This function requires GPU libraries (cuML) to be available.
    """
    if not is_cuml_available():
        raise RuntimeError(
            "MNMG UMAP requires GPU libraries (cuML). "
            "Not available in CPU-only mode."
        )
    client = args["client"]
    del args["client"]
    local_model = UMAP(**args)

    if isinstance(data, (tuple, list)):
        local_data = [x.compute() for x in data if x is not None]
    if len(local_data) == 2:
        X, y = local_data
        local_model.fit(X, y)
    else:
        X = local_data
        local_model.fit(X)

    return m(client=client, model=local_model, **args)
