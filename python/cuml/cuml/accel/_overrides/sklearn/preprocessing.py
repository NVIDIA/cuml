#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np

import cuml.preprocessing
from cuml.accel.estimator_proxy import (
    ArrayAPIProxyBase,
    ProxyBase,
    classproperty,
)
from cuml.internals.interop import UnsupportedOnGPU
from cuml.internals.validation import check_cudf

__all__ = (
    "StandardScaler",
    "MinMaxScaler",
    "MaxAbsScaler",
    "PolynomialFeatures",
    "OneHotEncoder",
    "TargetEncoder",
    "LabelEncoder",
    "LabelBinarizer",
)


class StandardScaler(ArrayAPIProxyBase):
    _cpu_class_path = "sklearn.preprocessing.StandardScaler"


class MinMaxScaler(ArrayAPIProxyBase):
    _cpu_class_path = "sklearn.preprocessing.MinMaxScaler"


class MaxAbsScaler(ArrayAPIProxyBase):
    _cpu_class_path = "sklearn.preprocessing.MaxAbsScaler"


class PolynomialFeatures(ArrayAPIProxyBase):
    _cpu_class_path = "sklearn.preprocessing.PolynomialFeatures"

    # These are staticmethods on the class that sklearn uses in the tests,
    # we can just re-export them here.
    @classproperty
    def _combinations(cls):
        return cls._cpu_class._combinations

    @classproperty
    def _num_combinations(cls):
        return cls._cpu_class._num_combinations

    @staticmethod
    def _params_from_cpu(model):
        if model.order == "F":
            raise UnsupportedOnGPU("order='F' is not supported")
        return model.get_params(deep=False)


def _check_onehotencoder_X(X):
    """Check if inputs are supported on GPU"""
    try:
        return check_cudf(X, input_name="X")
    except TypeError as exc:
        if str(exc).startswith("X with bytes dtype"):
            raise UnsupportedOnGPU(
                "X with bytes dtype is not supported"
            ) from None
        raise


class OneHotEncoder(ProxyBase):
    _gpu_class = cuml.preprocessing.OneHotEncoder

    def _gpu_fit(self, X, y=None):
        X = _check_onehotencoder_X(X)
        return self._gpu.fit(X, y=y)

    def _gpu_fit_transform(self, X, y=None, **fit_params):
        X = _check_onehotencoder_X(X)
        return self._gpu.fit_transform(X, y=y, **fit_params)


class LabelEncoder(ProxyBase):
    _gpu_class = cuml.preprocessing.LabelEncoder


class LabelBinarizer(ProxyBase):
    _gpu_class = cuml.preprocessing.LabelBinarizer

    def _gpu_inverse_transform(self, Y, threshold=None):
        return self._gpu.inverse_transform(Y, threshold=threshold)


def _check_targetencoder_y(y):
    """Check if inputs are supported on GPU.

    Raises UnsupportedOnGPU for unsupported cases to trigger CPU fallback.
    """
    from sklearn.utils.multiclass import type_of_target

    # Check for unsupported target types
    target_type = type_of_target(y)
    if target_type not in ("binary", "continuous"):
        raise UnsupportedOnGPU(
            f"y with target type {target_type!r} not supported on GPU"
        )


class TargetEncoder(ProxyBase):
    _gpu_class = cuml.preprocessing.TargetEncoder

    def _gpu_fit(self, X, y):
        _check_targetencoder_y(y)
        return self._gpu.fit(X, y)

    def _gpu_fit_transform(self, X, y, **params):
        _check_targetencoder_y(y)
        return self._gpu.fit_transform(X, y, **params)

    def _gpu_get_feature_names_out(self, input_features=None):
        """Return feature names for output features.

        sklearn's TargetEncoder returns input feature names (one output per input).
        For cuML in combination mode, we return a single column name.
        """
        n_features_out = getattr(self._gpu, "_n_features_out", 1)
        if input_features is not None:
            return np.asarray(input_features[:n_features_out], dtype=object)
        # Use feature_names_in_ if available (set during fit with DataFrame input)
        if hasattr(self._gpu, "feature_names_in_"):
            return np.asarray(
                self._gpu.feature_names_in_[:n_features_out], dtype=object
            )
        # Generate default names like sklearn does
        return np.array(
            [f"targetencoder{i}" for i in range(n_features_out)], dtype=object
        )
