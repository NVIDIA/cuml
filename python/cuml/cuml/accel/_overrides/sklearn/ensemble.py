#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import cuml.ensemble
from cuml.accel.estimator_proxy import ProxyBase
from cuml.internals.interop import UnsupportedOnGPU
from cuml.internals.validation import check_array

__all__ = (
    "RandomForestRegressor",
    "RandomForestClassifier",
    "IsolationForest",
)


class _RandomForestMixin:
    def _check_inputs(self, X, y=None, sample_weight=None):
        # Fallback to CPU if NaN in X
        try:
            check_array(
                X, mem_type=None, order=None, ensure_2d=False, input_name="X"
            )
        except ValueError as exc:
            if "NaN" in str(exc):
                raise UnsupportedOnGPU(
                    "Missing values are not supported"
                ) from None
            raise

        if y is not None:
            y = check_array(
                y,
                mem_type=None,
                order=None,
                ensure_2d=False,
                ensure_all_finite=False,
                input_name="y",
            )
            if len(y.shape) > 1 and y.shape[1] > 1:
                if isinstance(self, RandomForestClassifier) and self.oob_score:
                    raise ValueError(
                        "The type of target cannot be used to compute OOB estimates"
                    )
                raise UnsupportedOnGPU(
                    "Multi-output targets are not supported"
                )

    def _gpu_fit(self, X, y, sample_weight=None):
        self._check_inputs(X, y, sample_weight=sample_weight)
        return self._gpu.fit(X, y, sample_weight=sample_weight)

    def _gpu_predict(self, X):
        self._check_inputs(X)
        return self._gpu.predict(X)

    def _gpu_score(self, X, y, sample_weight=None):
        self._check_inputs(X, y, sample_weight=sample_weight)
        return self._gpu.score(X, y, sample_weight=sample_weight)


class RandomForestRegressor(ProxyBase, _RandomForestMixin):
    _gpu_class = cuml.ensemble.RandomForestRegressor

    def __len__(self):
        return self._call_method("__len__")

    def __iter__(self):
        return self._call_method("__iter__")

    def __getitem__(self, index):
        return self._call_method("__getitem__", index)


class RandomForestClassifier(ProxyBase, _RandomForestMixin):
    _gpu_class = cuml.ensemble.RandomForestClassifier

    def _gpu_predict_proba(self, X):
        self._check_inputs(X)
        return self._gpu.predict_proba(X)

    def _gpu_predict_log_proba(self, X):
        self._check_inputs(X)
        return self._gpu.predict_log_proba(X)

    def __len__(self):
        return self._call_method("__len__")

    def __iter__(self):
        return self._call_method("__iter__")

    def __getitem__(self, index):
        return self._call_method("__getitem__", index)


class IsolationForest(ProxyBase):
    _gpu_class = cuml.ensemble.IsolationForest

    @staticmethod
    def _validate_input(X):
        # cuML's IsolationForest requires dense, finite input and raises
        # ValueError (NaN/inf) or TypeError (sparse) otherwise. Convert
        # those into UnsupportedOnGPU so callers fall back to CPU instead
        # of crashing.
        try:
            check_array(
                X, mem_type=None, order=None, ensure_2d=False, input_name="X"
            )
        except (ValueError, TypeError) as exc:
            raise UnsupportedOnGPU(str(exc)) from None

    def _gpu_fit(self, X, y=None, sample_weight=None):
        self._validate_input(X)
        if sample_weight is not None:
            raise UnsupportedOnGPU("sample_weight is not supported")
        return self._gpu.fit(X, y=y)

    def _gpu_fit_predict(self, X, y=None, **kwargs):
        # IsolationForest.fit_predict() doesn't declare sample_weight itself;
        # it inherits OutlierMixin.fit_predict(self, X, y=None, **kwargs),
        # which forwards kwargs straight to fit(). Match that signature here
        # (rather than declaring sample_weight explicitly) so the proxy stays
        # signature-compatible with the CPU method.
        self._validate_input(X)
        if kwargs.get("sample_weight") is not None:
            raise UnsupportedOnGPU("sample_weight is not supported")
        return self._gpu.fit_predict(X, y=y)

    def _gpu_predict(self, X):
        self._validate_input(X)
        return self._gpu.predict(X)

    def _gpu_decision_function(self, X):
        self._validate_input(X)
        return self._gpu.decision_function(X)

    def _gpu_score_samples(self, X):
        self._validate_input(X)
        return self._gpu.score_samples(X)
