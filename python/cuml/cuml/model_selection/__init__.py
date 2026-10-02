#
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""This code is developed and maintained by scikit-learn and imported
by cuML to maintain the familiar sklearn namespace structure.
cuML includes tests to ensure full compatibility of these wrappers
with CUDA-based data and cuML estimators, but all of the underlying code
is due to the scikit-learn developers."""

from cuml.model_selection._split import (
    KFold,
    StratifiedKFold,
    train_test_split,
)

__all__ = [
    "train_test_split",
    "KFold",
    "GridSearchCV",
    "RandomizedSearchCV",
    "StratifiedKFold",
]


def __getattr__(name):
    if name == "GridSearchCV":
        from sklearn.model_selection import GridSearchCV

        return GridSearchCV
    elif name == "RandomizedSearchCV":
        from sklearn.model_selection import RandomizedSearchCV

        return RandomizedSearchCV
    raise AttributeError(f"module {__name__} has no attribute {name}")
