#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""This code is developed and maintained by scikit-learn and imported
by cuML to maintain the familiar sklearn namespace structure.
cuML includes tests to ensure full compatibility of these wrappers
with CUDA-based data and cuML estimators, but all of the underlying code
is due to the scikit-learn developers."""

__all__ = ["Pipeline", "make_pipeline", "FeatureUnion", "make_union"]


def __getattr__(name):
    if name == "Pipeline":
        from sklearn.pipeline import Pipeline

        return Pipeline
    elif name == "make_pipeline":
        from sklearn.pipeline import make_pipeline

        return make_pipeline
    elif name == "FeatureUnion":
        from sklearn.pipeline import FeatureUnion

        return FeatureUnion
    elif name == "make_union":
        from sklearn.pipeline import make_union

        return make_union
    raise AttributeError(f"module {__name__} has no attribute {name}")
