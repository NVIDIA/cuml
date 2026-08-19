#
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import warnings

import cupy as cp
import cupyx
import numpy as np
from cupy import linalg
from cupyx import lapack

from cuml.common.doc_utils import generate_docstring
from cuml.internals.base import Base
from cuml.internals.interop import InteropMixin, UnsupportedOnGPU
from cuml.internals.mixins import RegressorMixin
from cuml.internals.outputs import ArrayIndexPair, ReflectedAttr, mlfunc
from cuml.internals.validation import (
    check_consistent_length,
    check_inputs,
    check_is_fitted,
    check_sample_weight,
)
from cuml.metrics import pairwise_kernels


# cholesky solve with fallback to least squares for singular problems
def _safe_solve(K, y):
    try:
        with cupyx.errstate(linalg="raise"):
            # we need to set the error mode of cupy to raise
            # otherwise we silently get an array of NaNs
            dual_coef = lapack.posv(K, y)
            # Perform following check as a workaround for cusolver issue to be
            # fixed in a future CUDA version
            if cp.all(cp.isnan(dual_coef)):
                raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        warnings.warn(
            "Singular matrix in solving dual problem. Using "
            "least-squares solution instead."
        )
        dual_coef = linalg.lstsq(K, y, rcond=None)[0]
    return dual_coef


def _solve_cholesky_kernel(K, y, alpha, sample_weight=None):
    # dual_coef = inv(X X^t + alpha*Id) y
    n_samples = K.shape[0]
    n_targets = y.shape[1]

    K = cp.asarray(K, dtype=np.float64)

    alpha = cp.atleast_1d(alpha)
    one_alpha = alpha.size == 1
    has_sw = sample_weight is not None

    if has_sw:
        sw = cp.sqrt(cp.atleast_1d(sample_weight))
        y = y * sw[:, cp.newaxis]
        K *= cp.outer(sw, sw)

    if one_alpha:
        # Only one penalty, we can solve multi-target problems in one time.
        K.flat[:: n_samples + 1] += alpha[0]

        dual_coef = _safe_solve(K, y)

        if has_sw:
            dual_coef *= sw[:, cp.newaxis]

        return dual_coef
    else:
        # One penalty per target. We need to solve each target separately.
        dual_coefs = cp.empty([n_targets, n_samples], K.dtype)

        for dual_coef, target, current_alpha in zip(dual_coefs, y.T, alpha):
            K.flat[:: n_samples + 1] += current_alpha

            dual_coef[:] = _safe_solve(K, target).ravel()

            K.flat[:: n_samples + 1] -= current_alpha

        if has_sw:
            dual_coefs *= sw[cp.newaxis, :]

        return dual_coefs.T


class KernelRidge(InteropMixin, RegressorMixin, Base):
    """
    Kernel ridge regression (KRR) performs l2 regularised ridge regression
    using the kernel trick. The kernel trick allows the estimator to learn a
    linear function in the space induced by the kernel. This may be a
    non-linear function in the original feature space (when a non-linear
    kernel is used).
    This estimator supports multi-output regression (when y is 2 dimensional).
    See the sklearn user guide for more information.

    Parameters
    ----------
    alpha : float or array-like of shape (n_targets,), default=1.0
        Regularization strength; must be a positive float. Regularization
        improves the conditioning of the problem and reduces the variance of
        the estimates. Larger values specify stronger regularization.
        If an array is passed, penalties are assumed to be specific
        to the targets.
    kernel : str or callable, default="linear"
        Kernel mapping used internally. This parameter is directly passed to
        :class:`~cuml.metrics.pairwise_kernel`.
        If `kernel` is a string, it must be one of the metrics
        in `cuml.metrics.PAIRWISE_KERNEL_FUNCTIONS` or "precomputed".
        If `kernel` is "precomputed", X is assumed to be a kernel matrix.
        `kernel` may be a callable numba device function. If so, is called on
        each pair of instances (rows) and the resulting value recorded.
    gamma : float, default=None
        Gamma parameter for the RBF, laplacian, polynomial, exponential chi2
        and sigmoid kernels. Interpretation of the default value is left to
        the kernel; see the documentation for sklearn.metrics.pairwise.
        Ignored by other kernels.
    degree : float, default=3
        Degree of the polynomial kernel. Ignored by other kernels.
    coef0 : float, default=1
        Zero coefficient for polynomial and sigmoid kernels.
        Ignored by other kernels.
    kernel_params : mapping of str to any, default=None
        Additional parameters (keyword arguments) for kernel function passed
        as callable object.
    output_type : {None, 'input', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.

    Attributes
    ----------
    dual_coef_ : ndarray of shape (n_samples,) or (n_samples, n_targets)
        Representation of weight vector(s) in kernel space
    X_fit_ : ndarray of shape (n_samples, n_features)
        Training data, which is also required for prediction. If
        kernel == "precomputed" this is instead the precomputed
        training matrix, of shape (n_samples, n_samples).

    Examples
    --------

    .. code-block:: python

        >>> import cupy as cp
        >>> from cuml.kernel_ridge import KernelRidge
        >>> from numba import cuda
        >>> import math

        >>> n_samples, n_features = 10, 5
        >>> rng = cp.random.RandomState(0)
        >>> y = rng.randn(n_samples)
        >>> X = rng.randn(n_samples, n_features)

        >>> model = KernelRidge(kernel="poly").fit(X, y)
        >>> pred = model.predict(X)


        >>> @cuda.jit(device=True)
        ... def custom_rbf_kernel(x, y, gamma=None):
        ...     if gamma is None:
        ...         gamma = 1.0 / len(x)
        ...     sum = 0.0
        ...     for i in range(len(x)):
        ...         sum += (x[i] - y[i]) ** 2
        ...     return math.exp(-gamma * sum)


        >>> model = KernelRidge(kernel=custom_rbf_kernel,
        ...                     kernel_params={"gamma": 2.0}).fit(X, y)
        >>> pred = model.predict(X)

    """

    dual_coef_ = ReflectedAttr()
    X_fit_ = ReflectedAttr()

    _cpu_class_path = "sklearn.kernel_ridge.KernelRidge"

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + [
            "alpha",
            "kernel",
            "gamma",
            "degree",
            "coef0",
            "kernel_params",
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        if not isinstance(model.kernel, str):
            raise UnsupportedOnGPU(
                "KernelRidge callable kernels are not supported."
            )
        return {
            "alpha": model.alpha,
            "kernel": model.kernel,
            "gamma": model.gamma,
            "degree": model.degree,
            "coef0": model.coef0,
            "kernel_params": model.kernel_params,
        }

    def _params_to_cpu(self):
        if cp.isscalar(self.alpha):
            alpha = self.alpha
        else:
            alpha = cp.asnumpy(self.alpha)
        return {
            "alpha": alpha,
            "kernel": self.kernel,
            "gamma": self.gamma,
            "degree": self.degree,
            "coef0": self.coef0,
            "kernel_params": self.kernel_params,
        }

    def _attrs_from_cpu(self, model):
        if not isinstance(model.X_fit_, np.ndarray):
            # We only support coercing dense X_fit_ values, but in sklearn
            # this may also be a sparse matrix
            raise UnsupportedOnGPU("Sparse inputs are not supported")

        return {
            "dual_coef_": cp.asarray(model.dual_coef_),
            "X_fit_": ArrayIndexPair(cp.asarray(model.X_fit_), None),
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        return {
            "dual_coef_": self.dual_coef_.get(order="A"),
            "X_fit_": self.X_fit_.array.get(order="A"),
            **super()._attrs_to_cpu(model),
        }

    def __init__(
        self,
        *,
        alpha=1,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        output_type=None,
        verbose=False,
    ):
        super().__init__(verbose=verbose, output_type=output_type)
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = True
        return tags

    @mlfunc(convert_output=False)
    def _get_kernel(self, X, Y=None):
        if isinstance(self.kernel, str):
            params = {
                "gamma": self.gamma,
                "degree": self.degree,
                "coef0": self.coef0,
            }
        else:
            params = self.kernel_params or {}
        return pairwise_kernels(
            X, Y, metric=self.kernel, filter_params=True, **params
        )

    @generate_docstring()
    @mlfunc(set_input_type=True)
    def fit(self, X, y, sample_weight=None) -> "KernelRidge":
        X, y, index = check_inputs(
            self,
            X,
            y,
            dtype=("float32", "float64"),
            accept_multi_output=True,
            return_index=True,
            reset=True,
        )
        if ravel := (y.ndim == 1):
            y = y.reshape(-1, 1)

        # Unlike other solvers, we need to special-case scalar sample weights,
        # because K might be a pre-computed kernel.
        if not (np.isscalar(sample_weight) and np.isfinite(sample_weight)):
            sample_weight = check_sample_weight(sample_weight, dtype=X.dtype)
            check_consistent_length(X, y, sample_weight)

        K = self._get_kernel(X)
        dual_coef = _solve_cholesky_kernel(
            K, y, cp.asarray(self.alpha), sample_weight
        ).astype(X.dtype, copy=False)
        if ravel:
            dual_coef = dual_coef.ravel()

        self.X_fit_ = ArrayIndexPair(X, index)
        self.dual_coef_ = dual_coef
        return self

    @mlfunc(preserve_index=True)
    def predict(self, X):
        """
        Predict using the kernel ridge model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples. If kernel == "precomputed" this is instead a
            precomputed kernel matrix, shape = [n_samples,
            n_samples_fitted], where n_samples_fitted is the number of
            samples used in the fitting for this estimator.

        Returns
        -------
        C : array of shape (n_samples,) or (n_samples, n_targets)
            Returns predicted values.
        """
        check_is_fitted(self)
        X = check_inputs(self, X, dtype=self.X_fit_.array.dtype)
        K = self._get_kernel(X, self.X_fit_.array).astype(X.dtype, copy=False)
        return cp.dot(K, self.dual_coef_)
