# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from cuml.accel.core import ACCELERATED_MODULES
from cuml.accel.estimator_proxy import ProxyBase


def proxy_classes_with_set_output():
    for mod in ACCELERATED_MODULES:
        importlib.import_module(mod)

    return sorted(
        (
            cls
            for cls in ProxyBase.__subclasses__()
            if hasattr(cls, "set_output")
        ),
        key=lambda cls: cls.__name__,
    )


@pytest.mark.parametrize("cls", proxy_classes_with_set_output())
def test_set_output(cls):
    """Check that all proxy classes that define `set_output` handle it appropriately"""
    X, y = make_blobs(n_features=20, n_samples=100, random_state=42)

    model = cls().set_output(transform="pandas")

    # Only works with dense output
    if hasattr(model, "sparse_output"):
        model.sparse_output = False

    if hasattr(model, "transform"):
        out = model.fit(X, y).transform(X)
    else:
        out = model.fit_transform(X, y)

    # Output is a pandas dataframe with non-default column names
    assert isinstance(out, pd.DataFrame)
    assert all(isinstance(c, str) for c in out.columns)

    # Can call `get_feature_names_out` without error
    names = model.get_feature_names_out()
    assert isinstance(names, np.ndarray)
    assert names.dtype == object

    # No host transfer required (this isn't strictly necessary, but is currently
    # true for most proxied estimators). Can revisit this check if it proves tricky
    # when adding new estimators.
    # Some classes are excluded since sklearn's set_output accesses
    # n_features_in_ which triggers a sync.
    if cls.__name__ not in ("OneHotEncoder", "TargetEncoder"):
        assert not hasattr(model._cpu, "n_features_in_")
