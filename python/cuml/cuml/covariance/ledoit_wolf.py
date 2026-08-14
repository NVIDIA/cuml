#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import warnings

import cupy as cp
import numpy as np

from cuml.internals.base import Base
from cuml.internals.interop import InteropMixin
from cuml.internals.outputs import ReflectedAttr, mlfunc
from cuml.internals.validation import check_inputs, check_is_fitted


def _ledoit_wolf_shrinkage(X, assume_centered=False, block_size=1000):
    """Estimate the shrunk Ledoit-Wolf covariance matrix.

    Parameters
    ----------
    X : cupy.ndarray of shape (n_samples, n_features)
        Data from which to compute the covariance estimate.
    assume_centered : bool, default=False
        If True, data will not be centered before computation.
    block_size : int, default=1000
        Size of blocks into which the covariance matrix will be split
        for memory efficiency.

    Returns
    -------
    shrinkage : float
        The optimal shrinkage coefficient.
    emp_cov : cupy.ndarray of shape (n_features, n_features)
        The empirical covariance matrix.
    mu : float
        The trace of the empirical covariance divided by n_features.
    """
    n_samples, n_features = X.shape

    # Handle single feature case
    if n_features == 1:
        if assume_centered:
            emp_cov = cp.dot(X.T, X) / n_samples
        else:
            emp_cov = cp.cov(X.T, ddof=0).reshape(1, 1)
        return 0.0, emp_cov, float(emp_cov[0, 0])

    if not assume_centered:
        X = X - cp.mean(X, axis=0, keepdims=True)

    emp_cov = cp.dot(X.T, X) / n_samples

    X2 = X**2
    emp_cov_trace = cp.sum(X2, axis=0) / n_samples
    mu = float(cp.sum(emp_cov_trace) / n_features)

    beta_ = 0.0
    delta_ = 0.0

    for i in range(0, n_features, block_size):
        i_end = min(i + block_size, n_features)
        for j in range(0, n_features, block_size):
            j_end = min(j + block_size, n_features)

            beta_ += float(cp.sum(cp.dot(X2[:, i:i_end].T, X2[:, j:j_end])))
            delta_ += float(
                cp.sum(cp.dot(X[:, i:i_end].T, X[:, j:j_end]) ** 2)
            )

    delta_ /= n_samples**2

    beta = (1.0 / (n_features * n_samples)) * (beta_ / n_samples - delta_)
    delta = (
        delta_ - 2.0 * mu * float(cp.sum(emp_cov_trace)) + n_features * mu**2
    )
    delta /= n_features

    beta = min(beta, delta)
    if beta == 0:
        shrinkage = 0.0
    else:
        shrinkage = beta / delta

    return shrinkage, emp_cov, mu


class LedoitWolf(InteropMixin, Base):
    """LedoitWolf Estimator for covariance matrix estimation.

    Computes the Ledoit-Wolf shrinkage estimator for the covariance matrix.
    This estimator regularizes the empirical covariance by shrinking it
    towards a scaled identity matrix, with the shrinkage coefficient
    determined by the Ledoit-Wolf formula.

    The regularized covariance is:
    ``(1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)``

    where ``mu = trace(cov) / n_features`` and ``shrinkage`` is computed
    to minimize the Mean Squared Error between the regularized estimate
    and the true covariance.

    Parameters
    ----------
    store_precision : bool, default=True
        Specifies if the estimated precision matrix is stored.
    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero. If False (default), data will be centered before computation.
    block_size : int, default=1000
        Size of blocks into which the covariance matrix will be split
        during its Ledoit-Wolf estimation. This is purely a memory
        optimization and does not affect results.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {None, 'input', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix.
    location_ : ndarray of shape (n_features,)
        Estimated location, i.e., the estimated mean.
    precision_ : ndarray of shape (n_features, n_features)
        Estimated pseudo inverse matrix. Only stored if ``store_precision``
        is True.
    shrinkage_ : float
        Coefficient in the convex combination used for the computation
        of the shrunk estimate. Range is [0, 1].
    n_features_in_ : int
        Number of features seen during :term:`fit`.

    Examples
    --------
    >>> import cupy as cp
    >>> from cuml.covariance import LedoitWolf
    >>> rng = cp.random.RandomState(42)
    >>> X = rng.randn(100, 5)
    >>> lw = LedoitWolf().fit(X)
    >>> lw.covariance_.shape
    (5, 5)
    >>> lw.shrinkage_  # doctest: +SKIP
    0.123...

    See Also
    --------
    sklearn.covariance.LedoitWolf : The scikit-learn CPU implementation.

    References
    ----------
    O. Ledoit and M. Wolf, "A Well-Conditioned Estimator for
    Large-Dimensional Covariance Matrices", Journal of Multivariate
    Analysis, Volume 88, Issue 2, February 2004, pages 365-411.
    """

    covariance_ = ReflectedAttr()
    location_ = ReflectedAttr()
    precision_ = ReflectedAttr()

    _cpu_class_path = "sklearn.covariance.LedoitWolf"

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + [
            "store_precision",
            "assume_centered",
            "block_size",
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        return {
            "store_precision": model.store_precision,
            "assume_centered": model.assume_centered,
            "block_size": model.block_size,
        }

    def _params_to_cpu(self):
        return {
            "store_precision": self.store_precision,
            "assume_centered": self.assume_centered,
            "block_size": self.block_size,
        }

    def _attrs_from_cpu(self, model):
        return {
            "covariance_": cp.asarray(model.covariance_),
            "location_": cp.asarray(model.location_),
            "precision_": (
                None
                if model.precision_ is None
                else cp.asarray(model.precision_)
            ),
            "shrinkage_": model.shrinkage_,
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        return {
            "covariance_": self.covariance_.get(order="A"),
            "location_": self.location_.get(order="A"),
            "precision_": (
                None
                if self.precision_ is None
                else self.precision_.get(order="A")
            ),
            "shrinkage_": self.shrinkage_,
            **super()._attrs_to_cpu(model),
        }

    def __init__(
        self,
        *,
        store_precision=True,
        assume_centered=False,
        block_size=1000,
        verbose=False,
        output_type=None,
    ):
        super().__init__(verbose=verbose, output_type=output_type)
        self.store_precision = store_precision
        self.assume_centered = assume_centered
        self.block_size = block_size

    @mlfunc(set_input_type=True)
    def fit(self, X, y=None) -> "LedoitWolf":
        """Fit the Ledoit-Wolf shrunk covariance model to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : LedoitWolf
            Returns the instance itself.
        """
        X = check_inputs(
            self,
            X,
            dtype=("float32", "float64"),
            reset=True,
        )
        if X.shape[0] == 1:
            warnings.warn(
                "Only one sample available. "
                "You may want to reshape your data array"
            )

        if self.assume_centered:
            location = cp.zeros(X.shape[1], dtype=X.dtype)
        else:
            location = cp.mean(X, axis=0)

        shrinkage, emp_cov, mu = _ledoit_wolf_shrinkage(
            X,
            assume_centered=self.assume_centered,
            block_size=self.block_size,
        )

        shrunk_cov = (1.0 - shrinkage) * emp_cov
        shrunk_cov.flat[:: X.shape[1] + 1] += shrinkage * mu

        self.shrinkage_ = shrinkage
        self.location_ = location
        self.covariance_ = shrunk_cov

        if self.store_precision:
            self.precision_ = cp.linalg.pinv(shrunk_cov)
        else:
            self.precision_ = None

        return self

    @mlfunc
    def get_precision(self):
        """Getter for the precision matrix.

        Returns
        -------
        precision_ : ndarray of shape (n_features, n_features)
            The precision matrix associated to the current covariance object.
        """
        check_is_fitted(self)

        if self.store_precision:
            return self.precision_
        return cp.linalg.pinv(self.covariance_)

    @mlfunc(convert_output=False)
    def score(self, X_test, y=None) -> float:
        """Compute the log-likelihood of X_test under the estimated model.

        The log-likelihood is computed using the Gaussian model.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data of which we compute the likelihood.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        log_likelihood : float
            Log-likelihood of the data under the fitted Gaussian model.
        """
        check_is_fitted(self)
        X_test = check_inputs(
            self,
            X_test,
            dtype=("float32", "float64"),
        )

        precision = self.get_precision()
        X_centered = X_test - self.location_
        log_det_precision = cp.linalg.slogdet(precision)[1]
        mahal = cp.sum(cp.dot(X_centered, precision) * X_centered, axis=1)

        log_likelihood = -0.5 * (
            X_test.shape[1] * np.log(2 * np.pi)
            - log_det_precision
            + cp.mean(mahal)
        )

        return float(log_likelihood)

    @mlfunc(convert_output=False)
    def error_norm(
        self, comp_cov, norm="frobenius", scaling=True, squared=True
    ):
        """Compute the Mean Squared Error between two covariance estimators.

        Parameters
        ----------
        comp_cov : array-like of shape (n_features, n_features)
            The covariance to compare with.
        norm : {"frobenius", "spectral"}, default="frobenius"
            The type of norm used to compute the error.
        scaling : bool, default=True
            If True, the squared error is scaled by n_features.
        squared : bool, default=True
            If True, return squared error. If False, return error.

        Returns
        -------
        error : float
            The Mean Squared Error (in the sense of the Frobenius norm)
            between `self` and `comp_cov`.
        """
        check_is_fitted(self)

        comp_cov = check_inputs(self, comp_cov, dtype=("float32", "float64"))

        diff = self.covariance_ - comp_cov

        if norm == "frobenius":
            error = cp.sum(diff**2)
        elif norm == "spectral":
            error = cp.linalg.norm(diff, ord=2) ** 2
        else:
            raise ValueError(
                f"Invalid norm '{norm}'. Must be 'frobenius' or 'spectral'."
            )

        if scaling:
            n_features = self.covariance_.shape[0]
            error = error / n_features

        if not squared:
            error = cp.sqrt(error)

        return float(error)

    @mlfunc
    def mahalanobis(self, X):
        """Compute the squared Mahalanobis distances of given observations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The observations, the Mahalanobis distances of which we compute.

        Returns
        -------
        mahalanobis_distances : ndarray of shape (n_samples,)
            Squared Mahalanobis distances of the observations.
        """
        check_is_fitted(self)
        X = check_inputs(self, X, dtype=("float32", "float64"))

        X_centered = X - self.location_
        mahal = cp.sum(
            cp.dot(X_centered, self.get_precision()) * X_centered, axis=1
        )

        return mahal
