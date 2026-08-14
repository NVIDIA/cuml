#
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import numpy as np
from cupyx.scipy.special import gammainc
from sklearn.exceptions import DataConversionWarning

from cuml.internals.base import Base, get_handle
from cuml.internals.interop import InteropMixin, UnsupportedOnGPU
from cuml.internals.outputs import mlfunc
from cuml.internals.validation import (
    check_inputs,
    check_is_fitted,
    check_non_negative,
    check_random_seed,
)
from cuml.metrics.pairwise_distances import (
    PAIRWISE_DISTANCE_METRICS as SUPPORTED_METRICS,
)
from cuml.metrics.pairwise_distances import _ensure_boolean

from libc.stdint cimport int64_t, uintptr_t
from libcpp cimport bool as cpp_bool
from pylibraft.common.handle cimport handle_t

from cuml.metrics.distance_type cimport DistanceType


cdef extern from "cuml/neighbors/kde.hpp" nogil:

    ctypedef enum class DensityKernelType "ML::KDE::DensityKernelType":
        Gaussian "ML::KDE::DensityKernelType::Gaussian"
        Tophat "ML::KDE::DensityKernelType::Tophat"
        Epanechnikov "ML::KDE::DensityKernelType::Epanechnikov"
        Exponential "ML::KDE::DensityKernelType::Exponential"
        Linear "ML::KDE::DensityKernelType::Linear"
        Cosine "ML::KDE::DensityKernelType::Cosine"

    void _cuml_kde_score_samples \
        "ML::KDE::score_samples"(const handle_t &handle,
                                 const float *query,
                                 const float *train,
                                 const float *weights,
                                 float *output,
                                 int64_t n_query,
                                 int64_t n_train,
                                 int64_t n_features,
                                 float bandwidth,
                                 float sum_weights,
                                 DensityKernelType kernel,
                                 DistanceType metric,
                                 float metric_arg) except +

    void _cuml_kde_score_samples \
        "ML::KDE::score_samples"(const handle_t &handle,
                                 const double *query,
                                 const double *train,
                                 const double *weights,
                                 double *output,
                                 int64_t n_query,
                                 int64_t n_train,
                                 int64_t n_features,
                                 double bandwidth,
                                 double sum_weights,
                                 DensityKernelType kernel,
                                 DistanceType metric,
                                 double metric_arg) except +


KDE_KERNEL_TYPES = {
    "gaussian": DensityKernelType.Gaussian,
    "tophat": DensityKernelType.Tophat,
    "epanechnikov": DensityKernelType.Epanechnikov,
    "exponential": DensityKernelType.Exponential,
    "linear": DensityKernelType.Linear,
    "cosine": DensityKernelType.Cosine,
}

VALID_KERNELS = list(KDE_KERNEL_TYPES.keys())


class KernelDensity(InteropMixin, Base):
    """
    Kernel Density Estimation. Computes a non-parametric density estimate
    from a finite data sample, smoothing the estimate according to a
    bandwidth parameter.

    Parameters
    ----------
    bandwidth : float or {"scott", "silverman"}, default=1.0
        The bandwidth of the kernel.
    kernel : {'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', \
            'cosine'}, default='gaussian'
        The kernel to use.
    metric : str, default='euclidean'
        The distance metric to use.  Note that not all metrics are
        valid with all algorithms. Note that the normalization of the density
        output is correct only for the Euclidean distance metric. Default
        is 'euclidean'.
    metric_params : dict, default=None
        Additional parameters to be passed to the tree for use with the
        metric.
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
    n_features_in_ : int
        Number of features seen during fit.
    bandwidth_ : float
        Value of the bandwidth used, either given directly via ``bandwidth`` or
        estimated with ``bandwidth="scott"`` or ``bandwidth="silverman"``.

    Examples
    --------
    >>> from cuml.neighbors import KernelDensity
    >>> import cupy as cp
    >>> rng = cp.random.RandomState(42)
    >>> X = rng.random_sample((100, 3))
    >>> kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)
    >>> log_density = kde.score_samples(X[:3])
    """

    _cpu_class_path = "sklearn.neighbors.KernelDensity"

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "bandwidth",
            "kernel",
            "metric",
            "metric_params",
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        if model.metric not in SUPPORTED_METRICS:
            raise UnsupportedOnGPU(
                f"`metric={model.metric!r}` is not supported"
            )
        return {
            "bandwidth": model.bandwidth,
            "kernel": model.kernel,
            "metric": model.metric,
            "metric_params": model.metric_params,
        }

    def _params_to_cpu(self):
        return {
            "bandwidth": self.bandwidth,
            "kernel": self.kernel,
            "metric": self.metric,
            "metric_params": self.metric_params,
        }

    def _attrs_from_cpu(self, model):
        X = cp.asarray(model.tree_.data, dtype=cp.float64)
        sample_weight = (
            None
            if model.tree_.sample_weight is None
            else cp.asarray(model.tree_.sample_weight, dtype=X.dtype)
        )
        return {
            "bandwidth_": model.bandwidth_,
            "_X": X,
            "_sample_weight": sample_weight,
            **super()._attrs_from_cpu(model),
        }

    def _sync_attrs_to_cpu(self, model):
        # There's no way to create a `sklearn.neighbors.KernelDensity` instance
        # from our stored state without fitting a new KernelDensity (since
        # sklearn always uses a pre-fit tree rather than a brute-force method).
        # As such, syncing to CPU is effectively just a refit. Fitting isn't
        # that expensive here, and unlike other estimators it's not clear that
        # a fit-on-gpu, predict-on-cpu model is useful given typical
        # KernelDensity use cases. What we have here should be fine for now.
        X = cp.asnumpy(self._X)
        sample_weight = (
            None
            if self._sample_weight is None
            else cp.asnumpy(self._sample_weight)
        )
        model.fit(X, sample_weight=sample_weight)
        if hasattr(self, "feature_names_in_"):
            model.feature_names_in_ = self.feature_names_in_

    def __init__(
        self,
        *,
        bandwidth=1.0,
        kernel="gaussian",
        metric="euclidean",
        metric_params=None,
        output_type=None,
        verbose=False,
    ):
        super().__init__(verbose=verbose, output_type=output_type)
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.metric = metric
        self.metric_params = metric_params

    @mlfunc(set_input_type=True)
    def fit(self, X, y=None, sample_weight=None) -> "KernelDensity":
        """Fit the Kernel Density model on the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        y : None
            Ignored.
        sample_weight : array-like of shape (n_samples,), default=None
            List of sample weights attached to the data X.

        Returns
        -------
        self
            Returns the instance itself.
        """
        if self.kernel not in VALID_KERNELS:
            raise ValueError(f"kernel={self.kernel!r} is not supported")

        if self.metric == "nan_euclidean":
            raise NotImplementedError(
                "metric='nan_euclidean' is not supported by cuML's "
                "KernelDensity; the fused kernel has no NaN-aware path."
            )

        if isinstance(self.bandwidth, str):
            if self.bandwidth not in ("scott", "silverman"):
                raise ValueError(
                    f"Expected bandwidth in ['scott', 'silverman'], got {self.bandwidth!r}"
                )
        elif self.bandwidth <= 0:
            raise ValueError(f"Expected bandwidth > 0, got {self.bandwidth}")

        self._X, self._sample_weight = check_inputs(
            self,
            X,
            sample_weight=sample_weight,
            dtype=("float32", "float64"),
            order="C",
            reset=True,
        )
        if self.metric == "russellrao":
            self._X = _ensure_boolean(self._X, metric=self.metric)
        if self._sample_weight is not None:
            check_non_negative(self._sample_weight, input_name="sample_weight")

        if isinstance(self.bandwidth, str):
            if self.bandwidth == "scott":
                self.bandwidth_ = self._X.shape[0] ** (
                    -1 / (self._X.shape[1] + 4)
                )
            else:  # silverman
                self.bandwidth_ = (
                    self._X.shape[0] * (self._X.shape[1] + 2) / 4
                ) ** (-1 / (self._X.shape[1] + 4))
        else:
            self.bandwidth_ = self.bandwidth

        return self

    @mlfunc(preserve_index=True)
    def score_samples(self, X):
        """Compute the log-likelihood of each sample under the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            An array of points to query.  Last dimension should match dimension
            of training data (n_features).

        Returns
        -------
        density : ndarray of shape (n_samples,)
            Log-likelihood of each sample in `X`. These are normalized to be
            probability densities, so values will be low for high-dimensional
            data.
        """
        check_is_fitted(self)
        X = check_inputs(
            self,
            X,
            dtype=[self._X.dtype],
            order="C",
        )
        if self.metric == "russellrao":
            X = _ensure_boolean(X, metric=self.metric)

        if self.metric_params:
            if len(self.metric_params) != 1:
                raise ValueError(
                    "Cuml only supports metrics with a single arg."
                )
            metric_arg = float(next(iter(self.metric_params.values())))
        else:
            metric_arg = 2.0

        if self.metric not in SUPPORTED_METRICS:
            raise ValueError(f"metric={self.metric!r} is not supported")

        sum_weights = (
            float(cp.sum(self._sample_weight))
            if self._sample_weight is not None
            else float(self._X.shape[0])
        )

        cdef DensityKernelType kernel_enum = KDE_KERNEL_TYPES[self.kernel]
        cdef DistanceType metric_enum = SUPPORTED_METRICS[self.metric]

        cdef cpp_bool is_float32 = X.dtype == np.float32
        cdef int64_t n_query = X.shape[0]
        cdef int64_t n_train = self._X.shape[0]
        cdef int64_t n_features = X.shape[1] if len(X.shape) > 1 else 1

        output = cp.empty(n_query, dtype=X.dtype)

        cdef uintptr_t query_ptr = X.data.ptr
        cdef uintptr_t train_ptr = self._X.data.ptr
        cdef uintptr_t weight_ptr = 0
        if self._sample_weight is not None:
            weight_ptr = self._sample_weight.data.ptr
        cdef uintptr_t output_ptr = output.data.ptr

        cdef const float* weights_f = (
            <const float*>weight_ptr if weight_ptr != 0
            else <const float*>NULL
        )
        cdef const double* weights_d = (
            <const double*>weight_ptr if weight_ptr != 0
            else <const double*>NULL
        )

        cdef double c_bandwidth = <double>self.bandwidth_
        cdef double c_sum_weights = <double>sum_weights
        cdef double c_metric_arg = <double>metric_arg

        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><uintptr_t>handle.getHandle()

        with nogil:
            if is_float32:
                _cuml_kde_score_samples(
                    handle_[0],
                    <const float*>query_ptr,
                    <const float*>train_ptr,
                    weights_f,
                    <float*>output_ptr,
                    n_query,
                    n_train,
                    n_features,
                    <float>c_bandwidth,
                    <float>c_sum_weights,
                    kernel_enum,
                    metric_enum,
                    <float>c_metric_arg,
                )
            else:
                _cuml_kde_score_samples(
                    handle_[0],
                    <const double*>query_ptr,
                    <const double*>train_ptr,
                    weights_d,
                    <double*>output_ptr,
                    n_query,
                    n_train,
                    n_features,
                    c_bandwidth,
                    c_sum_weights,
                    kernel_enum,
                    metric_enum,
                    c_metric_arg,
                )
        handle.sync()

        return output

    @mlfunc(convert_output=False)
    def score(self, X, y=None) -> float:
        """Compute the total log-likelihood under the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        y : None
            Ignored.

        Returns
        -------
        logprob : float
            Total log-likelihood of the data in X. This is normalized to be a
            probability density, so the value will be low for high-dimensional
            data.
        """
        return float(cp.sum(self.score_samples(X)))

    @mlfunc(array_arg=None)
    def sample(self, n_samples=1, random_state=None):
        """Generate random samples from the model.

        Currently, this is implemented only for gaussian and tophat kernels.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.
        random_state : int, RandomState instance or None, default=None
            Determines random number generation used to generate
            random samples.

        Returns
        -------
        X : cupy array of shape (n_samples, n_features)
            List of samples.
        """
        check_is_fitted(self)

        supported_kernels = ["gaussian", "tophat"]
        if self.kernel not in supported_kernels:
            raise NotImplementedError(
                f"Only {supported_kernels} kernels are supported."
            )

        rng = cp.random.RandomState(check_random_seed(random_state))

        u = rng.uniform(0, 1, size=n_samples)
        if self._sample_weight is None:
            i = (u * self._X.shape[0]).astype(np.int64)
        else:
            cumsum_weight = cp.cumsum(self._sample_weight)
            sum_weight = cumsum_weight[-1]
            i = cp.searchsorted(cumsum_weight, u * sum_weight)
        if self.kernel == "gaussian":
            return cp.atleast_2d(rng.normal(self._X[i], self.bandwidth_))

        elif self.kernel == "tophat":
            # we first draw points from a d-dimensional normal distribution,
            # then use an incomplete gamma function to map them to a uniform
            # d-dimensional tophat distribution.
            dim = self._X.shape[1]
            X = rng.normal(size=(n_samples, dim))
            s_sq = cp.einsum("ij,ij->i", X, X)

            correction = (
                gammainc(0.5 * dim, 0.5 * s_sq) ** (1.0 / dim)
                * self.bandwidth_
                / cp.sqrt(s_sq)
            )
            return self._X[i] + X * correction[:, np.newaxis]
