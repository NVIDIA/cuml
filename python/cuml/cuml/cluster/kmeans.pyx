#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numbers import Integral

import cupy as cp
import numpy as np

from cuml.common.doc_utils import generate_docstring
from cuml.internals.base import Base, get_handle
from cuml.internals.interop import InteropMixin, UnsupportedOnGPU
from cuml.internals.mixins import ClusterMixin, CMajorInputTagMixin
from cuml.internals.outputs import ReflectedAttr, mlfunc
from cuml.internals.validation import (
    check_array,
    check_inputs,
    check_is_fitted,
    check_random_seed,
)

from libc.limits cimport INT_MAX
from libc.stdint cimport int64_t, uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t

cimport cuml.cluster.cpp.kmeans as lib
from cuml.metrics.distance_type cimport DistanceType


cdef inline bool _kmeans_indices_i32(int64_t n_rows, int64_t n_cols) noexcept nogil:
    cdef int64_t int_max = INT_MAX

    if n_rows < 0 or n_cols < 0:
        return False
    if n_rows > int_max or n_cols > int_max:
        return False
    if n_rows == 0 or n_cols == 0:
        return True

    return n_rows <= ((int_max - 1) // n_cols)


cdef _kmeans_init_params(kmeans, lib.KMeansParams& params):
    """Initialize a passed KMeansParams instance from a KMeans instance."""
    cdef bool multi_gpu = kmeans._multi_gpu

    params.n_clusters = kmeans.n_clusters
    params.max_iter = kmeans.max_iter
    params.tol = kmeans.tol
    params.verbosity = kmeans._verbose_level
    params.metric = DistanceType.L2Expanded
    params.batch_samples = int(kmeans.max_samples_per_batch)
    params.init_size = int(kmeans.init_size)
    params.device_buffer_samples = int(kmeans.device_buffer_samples)
    params.oversampling_factor = kmeans.oversampling_factor

    # Ensure random_state is set when running on multi-gpu
    if multi_gpu and kmeans.random_state is None:
        raise ValueError(
            "KMeansMG requires `random_state != None`, please select a consistent "
            "non-None `random_state` to use across all partitions when calling "
            "KMeansMG"
        )
    params.rng_state.seed = check_random_seed(kmeans.random_state)

    if isinstance(kmeans.init, str):
        if kmeans.init == "k-means++":
            # K-means++ is the constrained case of k-means|| w/
            # oversampling factor = 0
            params.oversampling_factor = 0
            params.init = lib.InitMethod.KMeansPlusPlus
        elif kmeans.init in ('scalable-k-means++', 'k-means||'):
            params.init = lib.InitMethod.KMeansPlusPlus
        elif kmeans.init == "random":
            params.init = lib.InitMethod.Random
    else:
        params.init = lib.InitMethod.Array

    if multi_gpu and params.oversampling_factor == 0:
        raise ValueError(
            "init='k-means++' or oversampling_factor=0 not supported for KMeansMG"
        )

    if kmeans.n_init == "auto":
        if isinstance(kmeans.init, str) and params.init == lib.InitMethod.KMeansPlusPlus:
            params.n_init = 1
        else:
            params.n_init = 10
    else:
        params.n_init = kmeans.n_init


cdef _kmeans_fit(
    handle_t& handle,
    lib.KMeansParams& params,
    X,
    sample_weight,
    centers,
):
    """Fit the kmeans centers and return `n_iter`.

    `X` and `sample_weight` may live on either the device or the host.
    `centers` always lives on the device.
    """
    cdef int64_t n_rows = X.shape[0]
    cdef int64_t n_cols = X.shape[1]

    cdef bool values_f32 = X.dtype == cp.float32
    # Indices fitting in int32 is for device arrays
    cdef bool host_data = not isinstance(X, cp.ndarray)
    cdef bool indices_i32 = (not host_data) and _kmeans_indices_i32(n_rows, n_cols)

    cdef uintptr_t X_ptr = X.data.ptr if isinstance(X, cp.ndarray) else X.ctypes.data
    cdef uintptr_t centers_ptr = centers.data.ptr
    cdef uintptr_t sample_weight_ptr = 0
    if sample_weight is not None:
        sample_weight_ptr = (
            sample_weight.data.ptr if isinstance(sample_weight, cp.ndarray)
            else sample_weight.ctypes.data
        )

    cdef int n_iter_32 = 0
    cdef int64_t n_iter_64 = 0
    cdef float inertia_32 = 0
    cdef double inertia_64 = 0

    with nogil:
        if values_f32:
            if indices_i32:
                lib.fit(
                    handle,
                    params,
                    <float *>X_ptr,
                    <int>n_rows,
                    <int>n_cols,
                    <float *>sample_weight_ptr,
                    <float *>centers_ptr,
                    inertia_32,
                    n_iter_32,
                )
            else:
                lib.fit(
                    handle,
                    params,
                    <float *>X_ptr,
                    n_rows,
                    n_cols,
                    <float *>sample_weight_ptr,
                    <float *>centers_ptr,
                    inertia_32,
                    n_iter_64,
                )
        else:
            if indices_i32:
                lib.fit(
                    handle,
                    params,
                    <double *>X_ptr,
                    <int>n_rows,
                    <int>n_cols,
                    <double *>sample_weight_ptr,
                    <double *>centers_ptr,
                    inertia_64,
                    n_iter_32,
                )
            else:
                lib.fit(
                    handle,
                    params,
                    <double *>X_ptr,
                    n_rows,
                    n_cols,
                    <double *>sample_weight_ptr,
                    <double *>centers_ptr,
                    inertia_64,
                    n_iter_64,
                )
    return n_iter_32 if indices_i32 else n_iter_64


cdef _kmeans_fit_parts(
    handle_t& handle,
    lib.KMeansParams& params,
    parts,
    sample_weight_parts,
    centers,
):
    """Fit kmeans over multiple local partitions (multi-GPU / out-of-core).

    `parts` is a non-empty list of row-major partitions that share
    `n_features` and dtype. They may all be device-resident
    (`cupy.ndarray`) or all host-resident (`numpy.ndarray`); Host
    partitions are buffered to the device in batches of
    `params.device_buffer_samples`. `sample_weight_parts` is
    either None or a matching list of weight vectors. The distributed
    reduction across ranks is performed via the NCCL communicator
    initialized on `handle`.
    """
    cdef int64_t n_parts = len(parts)
    cdef int64_t n_cols = parts[0].shape[1]
    cdef bool values_f32 = parts[0].dtype == np.float32
    cdef bool has_weights = sample_weight_parts is not None

    X_ptrs = np.empty(n_parts, dtype=np.uintp)
    n_samples_parts = np.empty(n_parts, dtype=np.int64)
    sw_ptrs = np.empty(n_parts, dtype=np.uintp) if has_weights else None
    for i in range(n_parts):
        X_ptrs[i] = (
            parts[i].data.ptr if isinstance(parts[i], cp.ndarray)
            else parts[i].ctypes.data
        )
        n_samples_parts[i] = parts[i].shape[0]
        if has_weights:
            sw = sample_weight_parts[i]
            sw_ptrs[i] = (
                sw.data.ptr if isinstance(sw, cp.ndarray) else sw.ctypes.data
            )

    cdef uintptr_t X_ptrs_ptr = X_ptrs.ctypes.data
    cdef uintptr_t n_samples_ptr = n_samples_parts.ctypes.data
    cdef uintptr_t sw_ptrs_ptr = sw_ptrs.ctypes.data if has_weights else 0
    cdef uintptr_t centers_ptr = centers.data.ptr

    cdef float inertia_32 = 0
    cdef double inertia_64 = 0
    cdef int64_t n_iter_64 = 0

    with nogil:
        if values_f32:
            lib.fit(
                handle,
                params,
                <float **>X_ptrs_ptr,
                <const int64_t *>n_samples_ptr,
                n_parts,
                n_cols,
                <float **>sw_ptrs_ptr,
                <float *>centers_ptr,
                inertia_32,
                n_iter_64,
            )
        else:
            lib.fit(
                handle,
                params,
                <double **>X_ptrs_ptr,
                <const int64_t *>n_samples_ptr,
                n_parts,
                n_cols,
                <double **>sw_ptrs_ptr,
                <double *>centers_ptr,
                inertia_64,
                n_iter_64,
            )
    return n_iter_64


cdef _kmeans_predict(
    handle_t& handle,
    lib.KMeansParams &params,
    X,
    sample_weight,
    centers,
    bool normalize_weights=True,
):
    """Predict labels & inertia from a fit `KMeans`.

    Split out to be shared between `KMeans.fit` and `KMeans.predict`
    """
    cdef int64_t n_rows = X.shape[0]
    cdef int64_t n_cols = X.shape[1]
    cdef int64_t n_clusters = centers.shape[0]

    # Stop-gap: downstream predict code indexes both X and cluster_centers_
    # using the selected index type. Keep the int32 path only when both
    # matrices fit int32 indexing until large-shape support is fully covered.
    cdef bool indices_i32 = (
        _kmeans_indices_i32(n_rows, n_cols)
        and _kmeans_indices_i32(n_clusters, n_cols)
    )
    labels = cp.zeros(shape=n_rows, dtype=(cp.int32 if indices_i32 else cp.int64))

    cdef uintptr_t X_ptr = X.data.ptr
    cdef uintptr_t centers_ptr = centers.data.ptr
    cdef uintptr_t sample_weight_ptr = (
        sample_weight.data.ptr if sample_weight is not None else 0
    )
    cdef uintptr_t labels_ptr = labels.data.ptr

    cdef bool values_f32 = X.dtype == cp.float32

    cdef float inertia_f32 = 0
    cdef double inertia_f64 = 0

    with nogil:
        if values_f32:
            if indices_i32:
                lib.predict(
                    handle,
                    params,
                    <float*>centers_ptr,
                    <float*>X_ptr,
                    <int>n_rows,
                    <int>n_cols,
                    <float*>sample_weight_ptr,
                    normalize_weights,
                    <int*>labels_ptr,
                    inertia_f32,
                )
            else:
                lib.predict(
                    handle,
                    params,
                    <float*>centers_ptr,
                    <float*>X_ptr,
                    <int64_t>n_rows,
                    <int64_t>n_cols,
                    <float*>sample_weight_ptr,
                    normalize_weights,
                    <int64_t*>labels_ptr,
                    inertia_f32,
                )
        else:
            if indices_i32:
                lib.predict(
                    handle,
                    params,
                    <double*>centers_ptr,
                    <double*>X_ptr,
                    <int>n_rows,
                    <int>n_cols,
                    <double*>sample_weight_ptr,
                    normalize_weights,
                    <int*>labels_ptr,
                    inertia_f64,
                )
            else:
                lib.predict(
                    handle,
                    params,
                    <double*>centers_ptr,
                    <double*>X_ptr,
                    <int64_t>n_rows,
                    <int64_t>n_cols,
                    <double*>sample_weight_ptr,
                    normalize_weights,
                    <int64_t*>labels_ptr,
                    inertia_f64,
                )

    inertia = inertia_f32 if values_f32 else inertia_f64

    return labels, inertia


cdef _kmeans_predict_host_chunked(
    handle_t& handle,
    lib.KMeansParams& params,
    X,
    sample_weight,
    centers,
    int64_t batch_size,
    bool normalize_weights=True,
):
    """Predict labels & total inertia for host-resident `X` in chunks.

    Streams chunks of `batch_size` host rows into a single reusable device
    buffer, runs the existing device-data predict on each chunk, and stitches
    the per-chunk labels into a single host (`numpy.ndarray`) result.

    When `normalize_weights` is True the sample weights are rescaled to sum to
    `n_rows` (matching the device predict), so a single-GPU host fit reports the
    same inertia as the device fit. Pass False when the weights are already
    normalized by the caller (e.g. the multi-GPU Dask layer normalizes
    `sample_weight` globally before splitting it across workers); rescaling
    per-partition there would make host-resident partitions report a different
    inertia than device-resident ones.

    Returns (`numpy.ndarray` of labels, `float` total inertia).
    """
    cdef int64_t n_rows = X.shape[0]
    cdef int64_t n_cols = X.shape[1]
    cdef int64_t cap = batch_size if batch_size > 0 else n_rows
    if cap > n_rows:
        cap = n_rows

    cdef bool uniform = sample_weight is None
    cdef double scale = 1.0
    if uniform:
        sw_buf = cp.ones(cap, dtype=X.dtype)
    else:
        if normalize_weights:
            total_sw = float(sample_weight.sum())
            if total_sw > 0:
                scale = n_rows / total_sw
        sw_buf = cp.empty(shape=cap, dtype=X.dtype)

    X_buf = cp.empty(shape=(cap, n_cols), dtype=X.dtype, order="C")

    labels_dtype = (
        np.int32
        if _kmeans_indices_i32(n_rows, n_cols)
        and _kmeans_indices_i32(centers.shape[0], n_cols)
        else np.int64
    )
    labels_host = np.empty(n_rows, dtype=labels_dtype)

    cdef int64_t start = 0
    cdef int64_t end = 0
    cdef int64_t n = 0
    total_inertia = 0.0
    while start < n_rows:
        end = start + cap
        if end > n_rows:
            end = n_rows
        n = end - start

        # For non-uniform, copy the raw slice and (optionally) scale on-device.
        X_buf[:n].set(X[start:end])
        if not uniform:
            sw_buf[:n].set(sample_weight[start:end])
            if normalize_weights:
                sw_buf[:n] *= scale

        batch_labels, batch_inertia = _kmeans_predict(
            handle, params, X_buf[:n], sw_buf[:n], centers,
            normalize_weights=False,
        )
        labels_host[start:end] = cp.asnumpy(batch_labels)
        total_inertia += float(batch_inertia)

        start = end

    return labels_host, total_inertia


class KMeans(InteropMixin,
             ClusterMixin,
             CMajorInputTagMixin,
             Base):
    """
    KMeans is a basic but powerful clustering method which is optimized via
    Expectation Maximization. It randomly selects K data points in X, and
    computes which samples are close to these points.
    For every cluster of points, a mean is computed (hence the name), and this
    becomes the new centroid.

    cuML's KMeans expects an array-like object or cuDF DataFrame, and supports
    the scalable KMeans++ initialization method. This method is more stable
    than randomly selecting K points.

    Examples
    --------

    .. code-block:: python

        >>> # Both import methods supported
        >>> from cuml import KMeans
        >>> from cuml.cluster import KMeans
        >>> import cudf
        >>> import numpy as np
        >>> import pandas as pd
        >>>
        >>> a = np.asarray([[1.0, 1.0], [1.0, 2.0], [3.0, 2.0], [4.0, 3.0]],
        ...                dtype=np.float32)
        >>> b = cudf.DataFrame(a)
        >>> # Input:
        >>> b
            0    1
        0  1.0  1.0
        1  1.0  2.0
        2  3.0  2.0
        3  4.0  3.0
        >>>
        >>> # Calling fit
        >>> kmeans_float = KMeans(n_clusters=2, n_init="auto", random_state=1)
        >>> kmeans_float.fit(b)
        KMeans(n_clusters=2, random_state=1)
        >>>
        >>> # Labels:
        >>> kmeans_float.labels_
        0    0
        1    0
        2    1
        3    1
        dtype: int32
        >>> # cluster_centers:
        >>> kmeans_float.cluster_centers_
            0    1
        0  1.0  1.5
        1  3.5  2.5

    Parameters
    ----------
    n_clusters : int (default = 8)
        The number of centroids or clusters you want.
    max_iter : int (default = 300)
        The more iterations of EM, the more accurate, but slower.
    tol : float64 (default = 1e-4)
        Stopping criterion when centroid means do not change much.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    random_state : int or None (default = None)
        If you want results to be the same when you restart Python, select a
        state.
    init : {'scalable-k-means++', 'k-means||', 'k-means++', 'random'} or an \
            ndarray (default = 'scalable-k-means++')

         - ``'scalable-k-means++'`` or ``'k-means||'``: Uses fast and stable
           scalable kmeans++ initialization. k-means++ is the constrained case of k-means||
           with `oversampling_factor=0`
         - ``'random'``: Choose `n_cluster` observations (rows) at random
           from data for the initial centroids.
         - If an ndarray is passed, it should be of
           shape (`n_clusters`, `n_features`) and gives the initial centers.

    n_init: 'auto' or int (default = 'auto')
        Number of instances the k-means algorithm will be called with
        different seeds. The final results will be from the instance
        that produces lowest inertia out of n_init instances.

        When `n_init='auto'`, the number of runs depends on the value of
        `init`: 1 if using `init='"k-means||"` or `init="scalable-k-means++"`;
        10 otherwise.

        .. versionadded:: 25.02
           Added 'auto' option for `n_init`.

        .. versionchanged:: 25.04
            Default value for `n_init` will change from 1 to `'auto'` in version 25.04.

    oversampling_factor : float64 (default = 2.0)
        The amount of points to sample
        in scalable k-means++ initialization for potential centroids.
        Increasing this value can lead to better initial centroids at the
        cost of memory. The total number of centroids sampled in scalable
        k-means++ is oversampling_factor * n_clusters * 8.
    max_samples_per_batch : int (default = 32768)
        The number of data samples to use for batches of the pairwise distance
        computation. This computation is done throughout both fit predict. The
        default should suit most cases. The total number of elements in the
        batched pairwise distance computation is :py:`max_samples_per_batch *
        n_clusters`. It might become necessary to lower this number when
        `n_clusters` becomes prohibitively large.
    device_buffer_samples : int (default = 0)
        Number of host samples to buffer to the device per GPU batch when
        fitting with host-resident inputs. This selects the fit path for
        host-resident inputs: the out-of-core host fit path is used whenever
        ``device_buffer_samples > 0``. When ``device_buffer_samples == 0``
        (default), host inputs are copied to the device in full and the
        standard device fit is used. Ignored when the input is already
        device-resident (e.g. cupy.ndarray, cudf.DataFrame),
        which always uses the device fit.
    init_size : int (default = 0)
        Number of samples to randomly draw for the KMeansPlusPlus initialization
        step. A random subset of this size is used for centroid seeding. Only
        applies when the out-of-core host fit path is actually selected, i.e.
        when the input is host-resident and ``device_buffer_samples > 0``. When
        set to 0 (default) on that path, ``min(3 * n_clusters, n_samples)`` is
        used. It is ignored on the device fit path — device-resident inputs, as
        well as host inputs that are copied to the device in full
        (``device_buffer_samples == 0``) — where the full dataset is always used
        for seeding.
    output_type : {None, 'input', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Attributes
    ----------
    cluster_centers_ : array
        The coordinates of the final clusters. This represents of "mean" of
        each data cluster.
    labels_ : array
        Which cluster each datapoint belongs to.

    Notes
    -----
    KMeans requires `n_clusters` to be specified. This means one needs to
    approximately guess or know how many clusters a dataset has. If one is not
    sure, one can start with a small number of clusters, and visualize the
    resulting clusters with PCA, UMAP or T-SNE, and verify that they look
    appropriate.

    **Applications of KMeans**

        The biggest advantage of KMeans is its speed and simplicity. That is
        why KMeans is many practitioner's first choice of a clustering
        algorithm. KMeans has been extensively used when the number of clusters
        is approximately known, such as in big data clustering tasks,
        image segmentation and medical clustering.


    For additional docs, see `scikitlearn's Kmeans
    <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_.
    """
    _multi_gpu = False

    labels_ = ReflectedAttr()
    cluster_centers_ = ReflectedAttr()

    _cpu_class_path = "sklearn.cluster.KMeans"

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "n_init",
            "oversampling_factor",
            "max_samples_per_batch",
            "device_buffer_samples",
            "init_size",
            "init",
            "max_iter",
            "n_clusters",
            "random_state",
            "tol",
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        if callable(model.init):
            raise UnsupportedOnGPU(f"`init={model.init!r}` is not supported")
        elif isinstance(model.init, str):
            if model.init == "k-means++":
                init = "scalable-k-means++"
            elif model.init == "random":
                init = "random"
            else:
                # Should be unreachable, here in case sklearn adds more init values
                raise UnsupportedOnGPU(f"`init={model.init!r}` is not supported")
        else:
            init = model.init  # array-like

        return {
            "n_clusters": model.n_clusters,
            "init": init,
            "n_init": model.n_init,
            "max_iter": model.max_iter,
            "tol": model.tol,
            "random_state": model.random_state,
        }

    def _params_to_cpu(self):
        init = self.init
        if not isinstance(init, str):
            init = cp.asnumpy(init)  # array-like
        elif init == "scalable-k-means++":
            init = "k-means++"
        return {
            "n_clusters": self.n_clusters,
            "init": init,
            "n_init": self.n_init,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "random_state": self.random_state,
        }

    def _attrs_from_cpu(self, model):
        return {
            "cluster_centers_": cp.asarray(model.cluster_centers_, order="C"),
            "labels_": cp.asarray(model.labels_, order="C"),
            "inertia_": model.inertia_,
            "n_iter_": model.n_iter_,
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        try:
            from sklearn.utils._openmp_helpers import (
                _openmp_effective_n_threads,
            )
        except ImportError:
            n_threads = 1
        else:
            n_threads = _openmp_effective_n_threads()

        return {
            "cluster_centers_": self.cluster_centers_.get(order="A"),
            "labels_": cp.asnumpy(self.labels_),
            "inertia_": self.inertia_,
            "n_iter_": self.n_iter_,
            # sklearn's KMeans relies on a few private attributes to work
            "_n_features_out": self._n_features_out,
            "_n_threads": n_threads,
            **super()._attrs_to_cpu(model),
        }

    def __init__(
        self,
        *,
        n_clusters=8,
        max_iter=300,
        tol=1e-4,
        verbose=False,
        random_state=None,
        init='scalable-k-means++',
        n_init="auto",
        oversampling_factor=2.0,
        max_samples_per_batch=1<<15,
        device_buffer_samples=0,
        init_size=0,
        output_type=None,
    ):
        super().__init__(verbose=verbose, output_type=output_type)
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init = init
        self.n_init = n_init
        self.oversampling_factor = oversampling_factor
        self.max_samples_per_batch = max_samples_per_batch
        self.device_buffer_samples = device_buffer_samples
        self.init_size = init_size

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        # Exposed to support sklearn's `get_feature_names_out`
        return self.n_clusters

    @generate_docstring()
    @mlfunc(set_input_type=True)
    def fit(self, X, y=None, sample_weight=None) -> "KMeans":
        """
        Compute k-means clustering with X.

        """
        device_buffer_samples = int(self.device_buffer_samples)
        if device_buffer_samples > 0 and self._multi_gpu:
            raise ValueError(
                f"device_buffer_samples={device_buffer_samples} is not "
                f"supported for the multi-GPU KMeans fit path; set "
                f"device_buffer_samples=0."
            )

        self._validate_fit_params()

        # Process input arrays
        X, sample_weight = check_inputs(
            self,
            X,
            sample_weight=sample_weight,
            dtype=("float32", "float64"),
            order=None,
            mem_type=None,
            reset=True,
        )
        data_on_device = isinstance(X, cp.ndarray)

        # Take the host (out-of-core) fit path when the input is host-resident
        # and a positive batch size is set (``device_buffer_samples > 0``); if
        # the batch is ``>= n_samples`` cuVS processes all rows in a single
        # batch. When the batch is 0 (default), copy host inputs to the device
        # and use the device fit; device-resident inputs always use the device
        # fit.
        use_host_path = (not data_on_device) and device_buffer_samples > 0

        if use_host_path:
            X = cp.asnumpy(X, order="C")
            if sample_weight is not None:
                sample_weight = cp.asnumpy(sample_weight)
        else:
            X = cp.asarray(X, order="C")
            if sample_weight is not None:
                sample_weight = cp.asarray(sample_weight)

        n_rows, n_cols = X.shape

        self._validate_fit_row_constraints(n_rows)

        # Allocate output cluster_centers_
        if isinstance(self.init, str):
            centers = cp.zeros(
                shape=(self.n_clusters, n_cols), dtype=X.dtype, order="C",
            )
        else:
            centers = check_array(
                self.init,
                order="C",
                dtype=X.dtype,
            ).copy()
            if centers.shape[0] != self.n_clusters:
                raise ValueError(
                    f"The shape of the initial centers {centers.shape} does not "
                    f"match the number of clusters {self.n_clusters}."
                )
            if centers.shape[1] != X.shape[1]:
                raise ValueError(
                    f"The shape of the initial centers {centers.shape} does not "
                    f"match the number of features of the data {X.shape[1]}."
                )

        # Prepare for libcuml call
        # XXX: multi-gpu uses handle attribute to manage comms
        handle = self.handle if self._multi_gpu else get_handle()
        cdef handle_t* handle_ = <handle_t *><size_t>handle.getHandle()
        cdef lib.KMeansParams params
        _kmeans_init_params(self, params)
        n_iter = _kmeans_fit(handle_[0], params, X, sample_weight, centers)
        if use_host_path:
            labels, inertia = _kmeans_predict_host_chunked(
                handle_[0], params, X, sample_weight, centers,
                device_buffer_samples,
            )
        else:
            labels, inertia = _kmeans_predict(
                handle_[0], params, X, sample_weight, centers,
            )
        handle.sync()

        # Store fitted attributes and return
        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = inertia
        self.n_iter_ = n_iter

        return self

    def _fit_mg_parts(self, parts, sample_weight_parts=None):
        """Fit KMeans over a list of local partitions (multi-GPU / out-of-core).

        Each worker passes its local partitions as a vector of mdspans. The
        partitions may reside on host or device. The cross-rank reduction runs
        over the NCCL communicator on ``self.handle``.

        `parts` is a non-empty sequence of per-partition arrays.
        `sample_weight_parts`, if given, is a matching sequence of per-partition
        weight vectors.
        """
        if len(parts) == 0:
            raise ValueError(
                "KMeansMG requires at least one local data partition."
            )

        self._validate_fit_params()

        has_weights = sample_weight_parts is not None

        # All partitions on a rank (and their sample weights) must share a
        # single memory residency and value dtype. Each input is coerced
        # with ``mem_type=None`` to preserve its native residency; the first
        # partition sets the reference, and every other partition and weight is
        # validated against it.
        coerced_parts = []
        coerced_weights = [] if has_weights else None
        mem_type = None
        dtype = None
        for i, part in enumerate(parts):
            sw_in = sample_weight_parts[i] if has_weights else None
            part_c, sw_c = check_inputs(
                self,
                part,
                sample_weight=sw_in,
                dtype=("float32", "float64"),
                order="C",
                mem_type=None,
                ensure_min_samples=0,
                reset=(i == 0),
            )
            part_mem = "device" if isinstance(part_c, cp.ndarray) else "host"
            if i == 0:
                mem_type = part_mem
                dtype = part_c.dtype
            elif part_mem != mem_type:
                raise ValueError(
                    f"All KMeansMG partitions must share the same memory "
                    f"residency, but partition 0 is {mem_type!r} and partition "
                    f"{i} is {part_mem!r}."
                )
            elif part_c.dtype != dtype:
                raise ValueError(
                    f"All KMeansMG partitions must share the same dtype, but "
                    f"partition 0 is {dtype} and partition {i} is "
                    f"{part_c.dtype}."
                )
            if sw_c is not None:
                sw_mem = "device" if isinstance(sw_c, cp.ndarray) else "host"
                if sw_mem != mem_type:
                    raise ValueError(
                        f"KMeansMG sample_weight partition {i} has memory "
                        f"residency {sw_mem!r}, which must match the data "
                        f"residency {mem_type!r}."
                    )
                if sw_c.dtype != dtype:
                    raise ValueError(
                        f"KMeansMG sample_weight partition {i} has dtype "
                        f"{sw_c.dtype}, which must match the data dtype {dtype}."
                    )
            coerced_parts.append(part_c)
            if has_weights:
                coerced_weights.append(sw_c)

        parts = coerced_parts
        sample_weight_parts = coerced_weights
        n_cols = parts[0].shape[1]
        n_rows = sum(part.shape[0] for part in parts)
        self._validate_fit_row_constraints(n_rows)

        # Partitions on a rank share residency; a cupy first partition means
        # the data is device-accessible.
        on_device = isinstance(parts[0], cp.ndarray)

        # Allocate output cluster_centers_ on the device.
        if isinstance(self.init, str):
            centers = cp.zeros(
                shape=(self.n_clusters, n_cols), dtype=dtype, order="C",
            )
        else:
            centers = check_array(
                self.init, order="C", dtype=dtype,
            ).copy()
            if centers.shape[0] != self.n_clusters:
                raise ValueError(
                    f"The shape of the initial centers {centers.shape} does not "
                    f"match the number of clusters {self.n_clusters}."
                )
            if centers.shape[1] != n_cols:
                raise ValueError(
                    f"The shape of the initial centers {centers.shape} does not "
                    f"match the number of features of the data {n_cols}."
                )

        cdef int64_t batch = int(self.device_buffer_samples)

        handle = self.handle
        cdef handle_t* handle_ = <handle_t *><size_t>handle.getHandle()
        cdef lib.KMeansParams params
        _kmeans_init_params(self, params)

        n_iter = _kmeans_fit_parts(
            handle_[0], params, parts, sample_weight_parts, centers,
        )

        labels_parts = []
        inertia = 0.0
        for i, part in enumerate(parts):
            if part.shape[0] == 0:
                continue
            sw = sample_weight_parts[i] if has_weights else None
            if on_device:
                part_labels, part_inertia = _kmeans_predict(
                    handle_[0], params, part, sw, centers,
                    normalize_weights=False,
                )
            else:
                part_labels, part_inertia = _kmeans_predict_host_chunked(
                    handle_[0], params, part, sw, centers, batch,
                    normalize_weights=False,
                )
            labels_parts.append(part_labels)
            inertia += float(part_inertia)
        handle.sync()

        if not labels_parts:
            labels = (
                cp.empty(0, dtype=np.int32) if on_device
                else np.empty(0, dtype=np.int32)
            )
        elif len(labels_parts) == 1:
            labels = labels_parts[0]
        elif on_device:
            labels = cp.concatenate(labels_parts)
        else:
            labels = np.concatenate(labels_parts)
        labels = cp.asarray(labels)

        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = inertia
        self.n_iter_ = n_iter

        return self

    def _validate_fit_params(self):
        if not isinstance(self.n_clusters, Integral) or self.n_clusters <= 0:
            raise ValueError(
                f"n_clusters={self.n_clusters} should be a positive integer."
            )
        if int(self.device_buffer_samples) < 0:
            raise ValueError(
                f"device_buffer_samples must be >= 0, got "
                f"{int(self.device_buffer_samples)}."
            )

    def _validate_fit_row_constraints(self, n_rows):
        if not self._multi_gpu and n_rows < self.n_clusters:
            raise ValueError(
                f"n_samples={n_rows} should be >= n_clusters={self.n_clusters}."
            )

    @generate_docstring(return_values={'name': 'preds',
                                       'type': 'dense',
                                       'description': 'Cluster indexes',
                                       'shape': '(n_samples, 1)'})
    @mlfunc(preserve_index=True)
    def fit_predict(self, X, y=None, sample_weight=None):
        """
        Compute cluster centers and predict cluster index for each sample.

        """
        return self.fit(X, sample_weight=sample_weight).labels_

    def _predict_labels_inertia(self, X, sample_weight=None):
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        sample_weight : array-like (device or host) shape = (n_samples,), default=None # noqa
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels : array
        Which cluster each datapoint belongs to.

        inertia : float
        Sum of squared distances of samples to their closest cluster center.
        """
        check_is_fitted(self)
        X, sample_weight = check_inputs(
            self,
            X,
            sample_weight=sample_weight,
            dtype=self.cluster_centers_.dtype,
            order="C",
        )
        if sample_weight is None:
            sample_weight = cp.ones(shape=X.shape[0], dtype=X.dtype)

        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
        cdef lib.KMeansParams params
        _kmeans_init_params(self, params)

        labels, inertia = _kmeans_predict(
            handle_[0],
            params,
            X,
            sample_weight,
            self.cluster_centers_
        )
        handle.sync()
        return labels, inertia

    @generate_docstring(return_values={'name': 'preds',
                                       'type': 'dense',
                                       'description': 'Cluster indexes',
                                       'shape': '(n_samples, 1)'})
    @mlfunc(preserve_index=True)
    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.

        """
        labels, _ = self._predict_labels_inertia(X)
        return labels

    @generate_docstring(return_values={'name': 'X_new',
                                       'type': 'dense',
                                       'description': 'Transformed data',
                                       'shape': '(n_samples, n_clusters)'})
    @mlfunc(preserve_index=True)
    def transform(self, X):
        """
        Transform X to a cluster-distance space.

        """
        check_is_fitted(self)
        X = check_inputs(
            self,
            X,
            dtype=self.cluster_centers_.dtype,
            order="C",
        )

        cdef int64_t n_rows = X.shape[0]
        cdef int64_t n_cols = X.shape[1]
        cdef int64_t n_clusters = self.n_clusters

        # Stop-gap: the current C++/cuVS transform path does not preserve
        # int64 indexing end-to-end for the output matrix. Reject oversized
        # outputs here until transform supports int64 output indexing.

        if not _kmeans_indices_i32(n_rows, n_clusters):
            raise NotImplementedError(
                "KMeans.transform does not currently support output shapes "
                "that require int64 indexing. Got output shape "
                f"({n_rows}, {n_clusters})."
            )

        out = cp.zeros(
            shape=(n_rows, n_clusters), dtype=X.dtype, order="C",
        )

        cdef uintptr_t X_ptr = X.data.ptr
        cdef uintptr_t centers_ptr = self.cluster_centers_.data.ptr
        cdef uintptr_t out_ptr = out.data.ptr

        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
        cdef lib.KMeansParams params
        _kmeans_init_params(self, params)

        cdef bool values_f32 = X.dtype == cp.float32
        cdef bool indices_i32 = _kmeans_indices_i32(n_rows, n_cols)

        with nogil:
            if values_f32:
                if indices_i32:
                    lib.transform(
                        handle_[0],
                        params,
                        <float*>centers_ptr,
                        <float*>X_ptr,
                        <int>n_rows,
                        <int>n_cols,
                        <float*>out_ptr,
                    )
                else:
                    lib.transform(
                        handle_[0],
                        params,
                        <float*>centers_ptr,
                        <float*>X_ptr,
                        <int64_t>n_rows,
                        <int64_t>n_cols,
                        <float*>out_ptr,
                    )
            else:
                if indices_i32:
                    lib.transform(
                        handle_[0],
                        params,
                        <double*>centers_ptr,
                        <double*>X_ptr,
                        <int>n_rows,
                        <int>n_cols,
                        <double*>out_ptr,
                    )
                else:
                    lib.transform(
                        handle_[0],
                        params,
                        <double*>centers_ptr,
                        <double*>X_ptr,
                        <int64_t>n_rows,
                        <int64_t>n_cols,
                        <double*>out_ptr,
                    )
        handle.sync()
        return out

    @generate_docstring(return_values={'name': 'score',
                                       'type': 'float',
                                       'description': 'Opposite of the value \
                                                        of X on the K-means \
                                                        objective.'})
    @mlfunc(convert_output=False)
    def score(self, X, y=None, sample_weight=None):
        """
        Opposite of the value of X on the K-means objective.

        """

        inertia = self._predict_labels_inertia(X, sample_weight=sample_weight)[1]
        return -1 * inertia

    @generate_docstring(return_values={'name': 'X_new',
                                       'type': 'dense',
                                       'description': 'Transformed data',
                                       'shape': '(n_samples, n_clusters)'})
    @mlfunc(preserve_index=True)
    def fit_transform(self, X, y=None, sample_weight=None):
        """
        Compute clustering and transform X to cluster-distance space.

        """
        self.fit(X, sample_weight=sample_weight)
        return self.transform(X)
