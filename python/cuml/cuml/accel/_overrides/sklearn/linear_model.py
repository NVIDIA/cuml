#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import sklearn
from packaging.version import Version

import cuml.linear_model
from cuml.accel.estimator_proxy import ProxyBase
from cuml.common.sparse import is_sparse
from cuml.internals.interop import UnsupportedOnGPU
from cuml.internals.outputs import using_output_type
from cuml.internals.validation import check_y

__all__ = (
    "LinearRegression",
    "LogisticRegression",
    "ElasticNet",
    "Ridge",
    "RidgeCV",
    "Lasso",
)


SKLEARN_16 = Version(sklearn.__version__) >= Version("1.6.0")


class LinearRegression(ProxyBase):
    _gpu_class = cuml.linear_model.LinearRegression
    _not_implemented_attributes = frozenset(("rank_", "singular_"))


class LogisticRegression(ProxyBase):
    _gpu_class = cuml.linear_model.LogisticRegression


class Ridge(ProxyBase):
    _gpu_class = cuml.linear_model.Ridge

    def _gpu_fit(self, X, y, sample_weight=None):
        if is_sparse(X) and (
            self.solver not in ("auto", "lsqr", "lbfgs", "sparse_cg")
        ):
            # cuml.Ridge's sparse solver is iterative. For maximum compatibility,
            # we only proxy through iterative sparse solvers.
            raise UnsupportedOnGPU(
                f"Sparse inputs are not supported with solver={self.solver!r}"
            )

        self._gpu.fit(X, y, sample_weight=sample_weight)

        # XXX: sklearn 1.6 changed the shape of `coef_` when fit with a 1
        # column 2D y. The sklearn 1.6+ behavior is what we implement in
        # cuml.Ridge, here we adjust the shape of `coef_` after the fit to
        # match the older behavior. This will also trickle down to change the
        # output shape of `predict` to match the older behavior transparently.
        if not SKLEARN_16 and (y_shape := getattr(y, "shape", ())):
            if len(y_shape) == 2 and y_shape[1] == 1:
                with using_output_type("cupy"):
                    # Reshape coef_ to be a 2D array
                    self._gpu.coef_ = self._gpu.coef_.reshape(1, -1)

        return self


class RidgeCV(ProxyBase):
    _gpu_class = cuml.linear_model.RidgeCV

    def _gpu_fit(self, X, y, sample_weight=None, **params):
        # `**params` is only used by sklearn for metadata routing, which we
        # don't support; it's absorbed here to match sklearn's signature.
        return self._gpu.fit(X, y, sample_weight=sample_weight)


class _ElasticNetMixin:
    def _gpu_fit(self, X, y, sample_weight=None, check_input=True):
        # check_input is ignored, only here to fix signature mismatch with sklearn
        y = check_y(y, mem_type=None, accept_multi_output=True)
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise UnsupportedOnGPU("Multi-output targets are not supported")
        return self._gpu.fit(X, y, sample_weight=sample_weight)


class ElasticNet(ProxyBase, _ElasticNetMixin):
    _gpu_class = cuml.linear_model.ElasticNet
    _not_implemented_attributes = frozenset(("dual_gap_",))


class Lasso(ProxyBase, _ElasticNetMixin):
    _gpu_class = cuml.linear_model.Lasso
    _not_implemented_attributes = frozenset(("dual_gap_",))
