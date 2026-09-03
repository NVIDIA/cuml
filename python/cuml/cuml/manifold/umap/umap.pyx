#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import ctypes
import warnings
from collections import deque

import cupy as cp
import cupyx.scipy.sparse
import joblib
import numpy as np
import scipy.sparse
import scipy.spatial
from sklearn.base import ClassNamePrefixFeaturesOutMixin

from cuml.common.doc_utils import generate_docstring
from cuml.common.sparse import is_sparse
from cuml.internals import logger
from cuml.internals.base import Base, get_handle
from cuml.internals.interop import InteropMixin, UnsupportedOnGPU
from cuml.internals.mixins import CMajorInputTagMixin, SparseInputTagMixin
from cuml.internals.outputs import ArrayIndexPair, ReflectedAttr, mlfunc
from cuml.internals.validation import (
    check_array,
    check_consistent_length,
    check_inputs,
    check_is_fitted,
    check_random_seed,
    check_y,
)
from cuml.manifold.utils import extract_knn_graph

from cuda.bindings.cyruntime cimport cudaStream_t
from libc.stdint cimport int64_t, uintptr_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibraft.common.handle cimport handle_t
from rmm.librmm.device_buffer cimport device_buffer
from rmm.librmm.memory_resource cimport any_resource, device_accessible
from rmm.pylibrmm.device_buffer cimport DeviceBuffer
from rmm.pylibrmm.memory_resource cimport get_current_device_resource

cimport cuml.manifold.umap.lib as lib
from cuml.metrics.distance_type cimport DistanceType


def _joblib_hash(X):
    """A thin shim around joblib.hash"""
    if cupyx.scipy.sparse.issparse(X) or isinstance(X, cp.ndarray):
        X = X.get()
    if scipy.sparse.issparse(X):
        # XXX: joblib.hash doesn't special case sparse inputs, meaning that
        # it's sensitive to what should be irrelevant internal state. For now
        # we trigger a cached attribute to ensure state is always fully filled
        # in so hashing is consistent. This is a relatively cheap operation and
        # has no measurable impact on performance.
        X.has_sorted_indices
    return joblib.hash(X)


def find_ab_params(spread=1.0, min_dist=0.1):
    """Fit a & b parameters for UMAP.

    Selects `a` and `b` for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.

    Borrowed from upstream umap: https://github.com/lmcinnes/umap.

    Parameters
    ----------
    spread: float (optional, default 1.0)
        The effective scale of embedded points.
    min_dist: float (optional, default 0.1)
        The effective minimum distance between embedded points.

    Returns
    -------
    a, b : float
        The `a` and `b` parameters for `UMAP`.
    """
    from scipy.optimize import curve_fit

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, _ = curve_fit(curve, xv, yv)
    return params[0], params[1]


def _breadth_first_search(adjmat, start, min_vertices):
    """Perform breadth-first search on an adjacency matrix.

    Parameters
    ----------
    adjmat : scipy.sparse.csr_matrix
        The adjacency matrix to search.
    start : int
        The starting vertex index.
    min_vertices : int
        Minimum number of vertices to explore.

    Returns
    -------
    explored : np.ndarray
        Array of explored vertex indices.
    """
    explored = []
    queue = deque([start])
    levels = {start: 0}
    max_level = float('inf')
    visited = {start}

    while queue:
        node = queue.popleft()
        explored.append(node)
        if max_level == float('inf') and len(explored) > min_vertices:
            max_level = max(levels.values())

        if levels[node] + 1 < max_level:
            neighbors = adjmat[node].indices
            for neighbour in neighbors:
                if neighbour not in visited:
                    queue.append(neighbour)
                    visited.add(neighbour)
                    levels[neighbour] = levels[node] + 1

    return np.array(explored)


def _compute_inverse_neighborhoods(embedding_np, X_np, min_vertices):
    """Compute neighborhoods for inverse transform using Delaunay triangulation.

    This is inherently CPU-bound as it uses scipy's Delaunay triangulation
    and BFS which are sequential algorithms.

    Parameters
    ----------
    embedding_np : np.ndarray
        The embedding coordinates (n_embedding, n_components).
    X_np : np.ndarray
        Points to inverse transform (n_samples, n_components).
    min_vertices : int
        Minimum number of neighbors to find per point.

    Returns
    -------
    neighborhoods : list of np.ndarray
        Variable-length neighbor indices for each sample.
    """
    n_embedding = embedding_np.shape[0]

    # Build Delaunay triangulation
    deltri = scipy.spatial.Delaunay(
        embedding_np, incremental=True, qhull_options="QJ"
    )

    # Find starting vertices (first vertex of simplex containing each point)
    simplex_indices = deltri.find_simplex(X_np)
    out_of_hull_mask = simplex_indices == -1

    start_vertices = np.empty(X_np.shape[0], dtype=np.intp)
    in_hull = ~out_of_hull_mask
    start_vertices[in_hull] = deltri.simplices[simplex_indices[in_hull]][:, 0]

    # For points outside the convex hull (can happen due to floating-point
    # precision even when inverse-transforming the training embedding),
    # fall back to the nearest embedding vertex.
    if np.any(out_of_hull_mask):
        ooh_points = X_np[out_of_hull_mask]
        dists = np.linalg.norm(
            embedding_np[np.newaxis, :, :] - ooh_points[:, np.newaxis, :],
            axis=2,
        )
        start_vertices[out_of_hull_mask] = np.argmin(dists, axis=1)

    # Build adjacency matrix from simplices
    simplices = deltri.simplices
    valid_mask = simplices < n_embedding
    rows_list, cols_list = [], []
    for i in range(simplices.shape[0]):
        valid_verts = simplices[i][valid_mask[i]]
        for v in valid_verts:
            rows_list.extend([v] * len(valid_verts))
            cols_list.extend(valid_verts)

    adjmat = scipy.sparse.csr_matrix(
        (np.ones(len(rows_list), dtype=np.int32),
         (np.array(rows_list), np.array(cols_list))),
        shape=(n_embedding, n_embedding)
    )

    # BFS from each starting vertex
    return [
        _breadth_first_search(adjmat, v, min_vertices=min_vertices)
        for v in start_vertices
    ]


def _build_inverse_graph(X_np, embedding_np, raw_data_np, neighborhoods, min_vertices, a, b):
    """Build inverse transform graph and compute initial points on GPU.

    Parameters
    ----------
    X_np : np.ndarray
        Points to inverse transform (n_samples, n_components).
    embedding_np : np.ndarray
        Embedding coordinates (n_embedding, n_components).
    raw_data_np : np.ndarray
        Original training data (n_orig, n_features).
    neighborhoods : list of np.ndarray
        Variable-length neighbor indices for each sample.
    min_vertices : int
        Number of closest neighbors to use per sample.
    a, b : float
        UMAP curve parameters.

    Returns
    -------
    inv_transformed : cp.ndarray
        Initial inverse transformed points (n_samples, n_features).
    rows, cols, weights : cp.ndarray
        COO graph arrays (on GPU).
    raw_data_gpu : cp.ndarray
        Original training data on GPU (n_orig, n_features).
    """
    n_samples = X_np.shape[0]

    # Pad neighborhoods to uniform length for GPU processing
    hood_lengths = np.array([len(h) for h in neighborhoods], dtype=np.int32)
    max_hood_len = int(hood_lengths.max())
    hoods_padded = np.zeros((n_samples, max_hood_len), dtype=np.int32)
    for i, hood in enumerate(neighborhoods):
        hoods_padded[i, :len(hood)] = hood

    # Transfer to GPU (raw_data with C-contiguous layout for CUDA kernels)
    X_gpu = cp.asarray(X_np, dtype=cp.float32)
    embedding_gpu = cp.asarray(embedding_np, dtype=cp.float32)
    raw_data_gpu = cp.asarray(raw_data_np, dtype=cp.float32, order="C")
    hoods_gpu = cp.asarray(hoods_padded, dtype=cp.int32)
    lengths_gpu = cp.asarray(hood_lengths, dtype=cp.int32)

    # Gather neighbor embeddings: (n_samples, max_hood_len, n_components)
    neighbor_embs = embedding_gpu[hoods_gpu]

    # Compute distances: (n_samples, max_hood_len)
    diffs = X_gpu[:, None, :] - neighbor_embs
    dists = cp.linalg.norm(diffs, axis=2)

    # Mask invalid entries with infinity
    valid_mask = cp.arange(max_hood_len)[None, :] < lengths_gpu[:, None]
    dists = cp.where(valid_mask, dists, cp.inf)

    # Get top-k closest indices
    order = cp.argsort(dists, axis=1)[:, :min_vertices]

    # Gather indices and distances
    row_idx = cp.arange(n_samples, dtype=cp.int32)[:, None]
    indices = hoods_gpu[row_idx, order]
    distances = dists[row_idx, order]

    # Compute membership strengths
    weights_2d = 1.0 / (1.0 + a * distances ** (2 * b))

    # Build COO graph
    rows = cp.repeat(cp.arange(n_samples, dtype=cp.int32), min_vertices)
    cols = indices.ravel()
    weights = weights_2d.ravel().astype(cp.float32)

    # Initialize via L1-normalized weighted average
    weights_norm = weights_2d / weights_2d.sum(axis=1, keepdims=True)
    neighbor_data = raw_data_gpu[indices]  # (n_samples, min_vertices, n_features)
    inv_transformed = cp.sum(
        weights_norm[:, :, None] * neighbor_data, axis=1
    ).astype(cp.float32)

    return inv_transformed, rows, cols, weights, raw_data_gpu


cdef class RaftCOO:
    """A wrapper around a `raft::sparse::COO`"""
    cdef unique_ptr[lib.COO] ptr

    def __dealloc__(self):
        self.ptr.reset(NULL)

    @staticmethod
    cdef RaftCOO from_ptr(unique_ptr[lib.COO]& ptr):
        """Create a new instance, taking ownership of an existing `unique_ptr[COO]`"""
        cdef RaftCOO self = RaftCOO.__new__(RaftCOO)
        self.ptr = move(ptr)
        return self

    @staticmethod
    cdef RaftCOO from_cupy_coo(handle, arr):
        """Create a new instance as a copy of a `cupyx.scipy.sparse.coo_matrix"""
        def copy_from_cupy(dst_ptr, src, dtype):
            src = src.astype(dtype, copy=False)
            size = src.size * src.dtype.itemsize
            dest_mem = cp.cuda.UnownedMemory(dst_ptr, size, owner=None)
            dest_mptr = cp.cuda.memory.MemoryPointer(dest_mem, 0)
            src_mem = cp.cuda.UnownedMemory(src.data.ptr, size, owner=None)
            src_mptr = cp.cuda.memory.MemoryPointer(src_mem, 0)
            dest_mptr.copy_from_device(src_mptr, size)

        cdef RaftCOO self = RaftCOO.__new__(RaftCOO)
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
        cdef lib.COO* coo = new lib.COO(handle_.get_stream())
        self.ptr.reset(coo)
        coo.allocate(arr.nnz, arr.shape[0], False, handle_.get_stream())
        handle_.sync_stream()

        copy_from_cupy(<uintptr_t>coo.vals(), arr.data, np.float32)
        copy_from_cupy(<uintptr_t>coo.rows(), arr.row, np.int32)
        copy_from_cupy(<uintptr_t>coo.cols(), arr.col, np.int32)

        return self

    def view_cupy_coo(self):
        """Create a new `cupyx.scipy.sparse.coo_matrix` as a view of this COO"""
        cdef lib.COO* coo = self.get()

        def view_as_cupy(ptr, dtype):
            dtype = np.dtype(dtype)
            mem = cp.cuda.UnownedMemory(ptr, (coo.nnz * dtype.itemsize), owner=self)
            memptr = cp.cuda.memory.MemoryPointer(mem, 0)
            return cp.ndarray(coo.nnz, dtype=dtype, memptr=memptr)

        vals = view_as_cupy(<uintptr_t>coo.vals(), np.float32)
        rows = view_as_cupy(<uintptr_t>coo.rows(), np.int32)
        cols = view_as_cupy(<uintptr_t>coo.cols(), np.int32)

        return cupyx.scipy.sparse.coo_matrix(((vals, (rows, cols))))

    cdef inline lib.COO* get(self) noexcept nogil:
        return self.ptr.get()


cdef copy_raft_host_coo_to_scipy_coo(lib.HostCOO &coo):
    """Copy a `raft::host_coo_matrix` to a `scipy.sparse.coo_matrix`"""
    nnz = coo.get_nnz()
    i32 = ctypes.POINTER(ctypes.c_int)
    f32 = ctypes.POINTER(ctypes.c_float)
    vals = np.ctypeslib.as_array(ctypes.cast(<uintptr_t>coo.vals(), f32), shape=(nnz,))
    rows = np.ctypeslib.as_array(ctypes.cast(<uintptr_t>coo.rows(), i32), shape=(nnz,))
    cols = np.ctypeslib.as_array(ctypes.cast(<uintptr_t>coo.cols(), i32), shape=(nnz,))
    return scipy.sparse.coo_matrix((vals.copy(), (rows.copy(), cols.copy())))


_BUILD_ALGOS = {"auto", "brute_force_knn", "nn_descent"}

_INITS = {"random": 0, "spectral": 1}

# Upper bound for n_components when force_serial_epochs is enabled. Matches
# SERIAL_PER_WARP_MAX_NC in cpp/src/umap/simpl_set_embed/optimize_batch_kernel.cuh
# Update both together.
_FORCE_SERIAL_EPOCHS_MAX_N_COMPONENTS = 512

_TARGET_METRICS = {
    "euclidean": lib.MetricType.EUCLIDEAN,
    "l2": lib.MetricType.EUCLIDEAN,
    "categorical": lib.MetricType.CATEGORICAL,
}

_METRICS = {
    "l2": DistanceType.L2SqrtExpanded,
    "euclidean": DistanceType.L2SqrtExpanded,
    "sqeuclidean": DistanceType.L2Expanded,
    "cityblock": DistanceType.L1,
    "l1": DistanceType.L1,
    "manhattan": DistanceType.L1,
    "taxicab": DistanceType.L1,
    "minkowski": DistanceType.LpUnexpanded,
    "chebyshev": DistanceType.Linf,
    "linf": DistanceType.Linf,
    "cosine": DistanceType.CosineExpanded,
    "correlation": DistanceType.CorrelationExpanded,
    "hellinger": DistanceType.HellingerExpanded,
    "hamming": DistanceType.HammingUnexpanded,
    "jaccard": DistanceType.JaccardExpanded,
    "canberra": DistanceType.Canberra
}

_SUPPORTED_METRICS = {
    "nn_descent": {
        "sparse": frozenset(),
        "dense": frozenset((
            DistanceType.L2SqrtExpanded,
            DistanceType.L2Expanded,
            DistanceType.CosineExpanded,
        ))
    },
    "brute_force_knn": {
        "sparse": frozenset((
            DistanceType.Canberra,
            DistanceType.CorrelationExpanded,
            DistanceType.CosineExpanded,
            DistanceType.HammingUnexpanded,
            DistanceType.HellingerExpanded,
            DistanceType.JaccardExpanded,
            DistanceType.L1,
            DistanceType.L2SqrtExpanded,
            DistanceType.L2Expanded,
            DistanceType.Linf,
            DistanceType.LpUnexpanded,
        )),
        "dense": frozenset((
            DistanceType.Canberra,
            DistanceType.CorrelationExpanded,
            DistanceType.CosineExpanded,
            DistanceType.HammingUnexpanded,
            DistanceType.HellingerExpanded,
            # DistanceType.JaccardExpanded,  # not supported
            DistanceType.L1,
            DistanceType.L2SqrtExpanded,
            DistanceType.L2Expanded,
            DistanceType.Linf,
            DistanceType.LpUnexpanded,
        ))
    }
}


def coerce_metric(metric, sparse=False, build_algo="brute_force_knn"):
    """Coerce a metric string to a `DistanceType`.

    Also checks that the metric is valid and supported.
    """
    if not isinstance(metric, str):
        raise TypeError(f"Expected `metric` to be a str, got {type(metric).__name__}")

    try:
        out = _METRICS[metric.lower()]
    except KeyError:
        raise ValueError(f"Invalid value for metric: {metric!r}")

    kind = "sparse" if sparse else "dense"
    supported = _SUPPORTED_METRICS[build_algo][kind]
    if out not in supported:
        raise NotImplementedError(
            f"Metric {metric!r} not supported for {kind} inputs with {build_algo=}"
        )

    return out


cdef init_params(self, lib.UMAPParams &params, n_rows, is_sparse=False, is_fit=True):
    """Initialize a UMAPParams instance from a UMAP model.

    This would be a method, except cdef methods aren't allowed on non cdef classes.

    Parameters
    ----------
    self : UMAP
        The UMAP model.
    params : lib.UMAPParams
        The params to instantiate.
    n_rows : int
        The number of rows in X for either fit or transform.
    is_sparse : bool
        Whether X is sparse for either fit or transform.
    is_fit : bool
        Whether these parameters are for a fit call.
    """
    if self.n_components < 1:
        raise ValueError(f"Expected `n_components >= 1`, got {self.n_components}")
    if self.n_neighbors < 1:
        raise ValueError(f"Expected `n_neighbors >= 1`, got {self.n_neighbors}")
    if self.min_dist > self.spread:
        raise ValueError(f"Expected min_dist ({self.min_dist}) <= spread ({self.spread})")

    if is_fit:
        build_algo = self.build_algo

        # Compute and stash some inferred params when fitting
        if self.a is None or self.b is None:
            self._a, self._b = find_ab_params(self.spread, self.min_dist)
        else:
            self._a, self._b = self.a, self.b
        self._n_neighbors = min(n_rows, self.n_neighbors)
        if self._n_neighbors != self.n_neighbors:
            warnings.warn(
                f"n_neighbors ({self.n_neighbors}) is larger than n_samples ({n_rows}) "
                f"truncating to {self._n_neighbors}"
            )
    else:
        # Only brute_force_knn supported for transform
        build_algo = "brute_force_knn"
        # Use the larger of the input shapes when inferring deterministic behavior
        n_rows = max(self._raw_data.shape[0], n_rows)

    if build_algo == "auto":
        if self.random_state is not None:
            # TODO: for now, users should be able to see the same results
            # as previous version (i.e. running brute force knn) when they
            # explicitly pass random_state
            # https://github.com/NVIDIA/cuml/issues/5985
            build_algo ="brute_force_knn"
        elif n_rows <= 50_000 or is_sparse:
            # brute force is faster for small datasets
            build_algo = "brute_force_knn"
        else:
            build_algo = "nn_descent"
        logger.debug(f"Building knn graph using build_algo={build_algo!r}")
    elif build_algo not in _BUILD_ALGOS:
        raise ValueError(
            f"Expected `build_algo` to be one of {list(_BUILD_ALGOS)}, "
            f"got {build_algo!r}"
        )

    if build_algo == "nn_descent" and n_rows < 150:
        # https://github.com/NVIDIA/cuvs/issues/184
        warnings.warn(
            "using build_algo='nn_descent' on a small dataset (< 150 samples) "
            "is unstable"
        )

    if build_algo == "nn_descent" and self.random_state is not None:
        warnings.warn("build_algo='nn_descent' is not deterministic. Please use "
                      "build_algo='brute_force_knn' instead with random_state set.")

    params.n_neighbors = self._n_neighbors
    params.n_components = self.n_components
    params.n_epochs = self.n_epochs or 0
    params.learning_rate = self.learning_rate
    params.initial_alpha = self.learning_rate
    params.min_dist = self.min_dist
    params.spread = self.spread
    params.set_op_mix_ratio = self.set_op_mix_ratio
    params.local_connectivity = self.local_connectivity
    params.repulsion_strength = self.repulsion_strength
    params.negative_sample_rate = self.negative_sample_rate
    params.transform_queue_size = self.transform_queue_size
    params.verbosity = self._verbose_level
    params.a = self._a
    params.b = self._b
    params.target_n_neighbors = self.target_n_neighbors
    params.target_weight = self.target_weight
    params.metric = coerce_metric(self.metric, sparse=is_sparse, build_algo=build_algo)
    params.p = (self.metric_kwds or {}).get("p", 2.0)
    params.random_state = check_random_seed(self.random_state)

    # deterministic if a random_state provided or when run on very small inputs
    params.deterministic = self.random_state is not None or n_rows < 300

    if isinstance(self.init, str):
        if self.init not in _INITS:
            raise ValueError(
                f"Expected `init` to be an array or one of {list(_INITS)}, "
                f"got {self.init!r}"
            )
        params.init = _INITS[self.init]
    else:
        params.init = 2

    if self.force_serial_epochs is None:
        # Only auto-enable for spectral fit. Also skip when n_components > 512 since
        # the warp-based serial kernel only supports up to 512 components
        params.force_serial_epochs = (
            is_fit
            and params.init == 1
            and self.n_components <= _FORCE_SERIAL_EPOCHS_MAX_N_COMPONENTS
        )
    else:
        if (
            self.force_serial_epochs
            and self.n_components > _FORCE_SERIAL_EPOCHS_MAX_N_COMPONENTS
        ):
            raise ValueError(
                f"force_serial_epochs=True is only supported for "
                f"n_components <= {_FORCE_SERIAL_EPOCHS_MAX_N_COMPONENTS}, "
                f"got n_components={self.n_components}. Pass "
                f"force_serial_epochs=False or None to use the parallel "
                f"batch kernel."
            )
        params.force_serial_epochs = self.force_serial_epochs

    if self.target_metric in _TARGET_METRICS:
        params.target_metric = _TARGET_METRICS[self.target_metric]
    else:
        raise ValueError(
            f"Expected `target_metric` to be one of {list(_TARGET_METRICS)}, "
            f"got {self.target_metric!r}"
        )

    if self.callback is not None:
        params.callback = (
            <lib.GraphBasedDimRedCallback*><uintptr_t>self.callback.get_native_callback()
        )

    build_kwds = self.build_kwds or {}
    n_clusters = build_kwds.get("knn_n_clusters", 1)
    overlap_factor = build_kwds.get("knn_overlap_factor", 2)

    if n_clusters < 1:
        raise ValueError(f"Expected `knn_n_clusters >= 1`, got {n_clusters}")
    elif n_clusters > 1 and overlap_factor >= n_clusters:
        raise ValueError(
            f"`knn_n_clusters > 1` requires `knn_n_clusters ({n_clusters}) > "
            f"knn_overlap_factor ({overlap_factor})`"
        )

    all_neighbors_supported_metrics = [
        'l2', 'euclidean', 'sqeuclidean', 'cosine', 'inner_product'
    ]
    if (
        build_algo == "brute_force_knn" and
        n_clusters > 1 and
        self.metric.lower() not in all_neighbors_supported_metrics
    ):
        warnings.warn(
            f"metric='{self.metric}' is not supported for batched knn build with "
            f"knn_n_clusters > 1. Supported metrics are: {all_neighbors_supported_metrics}. "
            f"The knn_n_clusters parameter will be ignored and regular brute force knn "
            f"(without batching) will be used instead."
        )

    params.build_params.n_clusters = n_clusters
    params.build_params.overlap_factor = overlap_factor

    if build_algo == "brute_force_knn":
        params.build_algo = lib.graph_build_algo.BRUTE_FORCE_KNN
    else:
        params.build_algo = lib.graph_build_algo.NN_DESCENT

        max_iterations = build_kwds.get("nnd_max_iterations", 20)
        termination_threshold = build_kwds.get("nnd_termination_threshold", 0.0001)
        graph_degree = build_kwds.get("nnd_graph_degree", 64)
        intermediate_graph_degree = build_kwds.get("nnd_intermediate_graph_degree", 128)

        if graph_degree < self._n_neighbors:
            logger.warn(
                f"build_algo='nn_descent' requires `nnd_graph_degree >= n_neighbors`. "
                f"Setting nnd_graph_degree to {self._n_neighbors}"
            )
            graph_degree = self._n_neighbors

        if intermediate_graph_degree < graph_degree:
            logger.warn(
                f"build_algo='nn_descent' requires `nnd_intermediate_graph_degree >= "
                f"nnd_graph_degree`. Setting nnd_intermediate_graph_degree to "
                f"{graph_degree}"
            )
            intermediate_graph_degree = graph_degree

        params.build_params.nnd.max_iterations = max_iterations
        params.build_params.nnd.termination_threshold = termination_threshold
        params.build_params.nnd.graph_degree = graph_degree
        params.build_params.nnd.intermediate_graph_degree = intermediate_graph_degree


class UMAP(
    InteropMixin,
    CMajorInputTagMixin,
    SparseInputTagMixin,
    ClassNamePrefixFeaturesOutMixin,
    Base,
):
    """Uniform Manifold Approximation and Projection

    Finds a low dimensional embedding of the data that approximates
    an underlying manifold.

    Adapted from https://github.com/lmcinnes/umap/blob/master/umap/umap_.py

    The UMAP algorithm is outlined in [1]. This implementation follows the
    GPU-accelerated version as described in [2].

    Parameters
    ----------
    n_neighbors: float (optional, default 15)
        The size of local neighborhood (in terms of number of neighboring
        sample points) used for manifold approximation. Larger values
        result in more global views of the manifold, while smaller
        values result in more local data being preserved. In general
        values should be in the range 2 to 100.
    n_components: int (optional, default 2)
        The dimension of the space to embed into. This defaults to 2 to
        provide easy visualization, but can reasonably be set to any
    metric: string (default='euclidean').
        Distance metric to use. Supported distances are ['l1, 'cityblock',
        'taxicab', 'manhattan', 'euclidean', 'l2', 'sqeuclidean', 'canberra',
        'minkowski', 'chebyshev', 'linf', 'cosine', 'correlation', 'hellinger',
        'hamming', 'jaccard']
        Metrics that take arguments (such as minkowski) can have arguments
        passed via the metric_kwds dictionary.
        Note: The 'jaccard' distance metric is only supported for sparse
        inputs.
        Note: If build_algo=`brute_force_knn` and `knn_n_clusters > 1`, the metric
        must be one of ['l2', 'sqeuclidean', 'euclidean', 'cosine', 'inner_product'].
    metric_kwds: dict (optional, default=None)
        Metric argument
    n_epochs: int (optional, default None)
        The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings. If None is specified a value will be selected based on
        the size of the input dataset (200 for large datasets, 500 for small).
    learning_rate: float (optional, default 1.0)
        The initial learning rate for the embedding optimization.
    init: string (optional, default 'spectral')
        How to initialize the low dimensional embedding. Options are:

        * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
        * 'random': assign initial embedding positions at random.
        * An array-like with initial embedding positions.

        Note: When ``init='spectral'`` and ``n_components <= 512``,
        ``force_serial_epochs`` defaults to ``True`` because spectral
        initialization is more susceptible to outlier artifacts. Pass
        ``force_serial_epochs=False`` explicitly to disable and use the
        faster parallel batch kernel.

    min_dist: float (optional, default 0.1)
        The effective minimum distance between embedded points. Smaller values
        will result in a more clustered/clumped embedding where nearby points
        on the manifold are drawn closer together, while larger values will
        result on a more even dispersal of points. The value should be set
        relative to the ``spread`` value, which determines the scale at which
        embedded points will be spread out.
    spread: float (optional, default 1.0)
        The effective scale of embedded points. In combination with
        ``min_dist`` this determines how clustered/clumped the embedded
        points are.
    set_op_mix_ratio: float (optional, default 1.0)
        Interpolate between (fuzzy) union and intersection as the set operation
        used to combine local fuzzy simplicial sets to obtain a global fuzzy
        simplicial sets. Both fuzzy set operations use the product t-norm.
        The value of this parameter should be between 0.0 and 1.0; a value of
        1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
        intersection.
    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.
    repulsion_strength: float (optional, default 1.0)
        Weighting applied to negative samples in low dimensional embedding
        optimization. Values higher than one will result in greater weight
        being given to negative samples.
    negative_sample_rate: int (optional, default 5)
        The number of negative samples to select per positive sample
        in the optimization process. Increasing this value will result
        in greater repulsive force being applied, greater optimization
        cost, but slightly more accuracy.
    transform_queue_size: float (optional, default 4.0)
        For transform operations (embedding new points using a trained model
        this will control how aggressively to search for nearest neighbors.
        Larger values will result in slower performance but more accurate
        nearest neighbor evaluation.
    a: float (optional, default None)
        More specific parameters controlling the embedding. If None these
        values are set automatically as determined by ``min_dist`` and
        ``spread``.
    b: float (optional, default None)
        More specific parameters controlling the embedding. If None these
        values are set automatically as determined by ``min_dist`` and
        ``spread``.
    target_n_neighbors: int (optional, default=-1)
        The number of nearest neighbors to use to construct the target
        simplicial set. If set to -1 use the ``n_neighbors`` value.
    target_metric: string (optional, default='categorical')
        The metric used to measure distance for a target array when using
        supervised dimension reduction. By default this is 'categorical'
        which will measure distance in terms of whether categories match
        or are different. Furthermore, if semi-supervised is required
        target values of -1 will be treated as unlabelled under the
        'categorical' metric. If the target array takes continuous values
        (e.g. for a regression problem) then metric of 'euclidean' or 'l2'
        is probably more appropriate.
    target_weight: float (optional, default=0.5)
        Weighting factor between data topology and target topology. A
        value of 0.0 weights predominantly on data, a value of 1.0 places
        a strong emphasis on target. The default of 0.5 balances the
        weighting equally between data and target.
    hash_input: bool, optional (default = False)
        UMAP can hash the training input so that exact embeddings
        are returned when transform is called on the same data upon
        which the model was trained. This enables consistent
        behavior between calling ``model.fit_transform(X)`` and
        calling ``model.fit(X).transform(X)``. Note that the CPU-based
        UMAP reference implementation does this by default. This
        feature is made optional in the GPU version due to the
        significant overhead in copying memory to the host for
        computing the hash.
    precomputed_knn : tuple[array, array], sparse-matrix, array, optional
        This feature allows the precomputation of the KNN outside of UMAP.
        Options are:

        - A tuple (indices, distances) of dense arrays of shape (n_samples,
          n_neighbors), where n_neighbors is >= the ``n_neighbors`` parameter.
          Self references should be included (i.e. the first column of
          `indices` should be [0, 1, ...], denotating that the nearest neighbor
          to each row is itself). This is the most efficient representation.
          Note that providing on CPU may result in lower peak GPU memory usage.

        - A sparse matrix KNN graph, as may be output by
          ``cuml.neighbors.kneighbors_graph`` with ``mode="distance"`` and
          ``include_self=True``. The ``n_neighbors`` used to calculate the
          graph must be >= the ``n_neighbors`` parameter.

        - A pairwise distances dense array of shape (n_samples, n_samples).

        In all cases the KNN should be computed using the same ``metric`` as
        provided to ``UMAP``.

    random_state : int, RandomState instance or None, optional (default=None)
        Seed used by the random number generator for embedding initialization
        and optimizer sampling. Setting a random_state enables reproducible
        embeddings, but at the cost of slower training and increased memory
        usage. This is because high parallelism during optimization involves
        non-deterministic floating-point addition ordering.

        Note: Explicitly setting ``build_algo='nn_descent'`` will break
        reproducibility, as NN Descent produces non-deterministic KNN graphs.
    force_serial_epochs: bool or None, optional (default=None)
        Controls whether optimization epochs use the sequential (reduced
        GPU parallelism) kernel. When ``None`` (the default), serial epochs
        are enabled automatically for ``init='spectral'`` with
        ``n_components <= 512`` because spectral initialization is more
        susceptible to outlier artifacts; for ``n_components > 512`` the
        auto-default falls back to ``False`` since the serial kernel does
        not support that range. Pass ``True`` to force serial epochs
        regardless of init (only supported for ``n_components <= 512``;
        otherwise a ``ValueError`` is raised), or ``False`` to disable them.
    callback: An instance of GraphBasedDimRedCallback class
        Used to intercept the internal state of embeddings while they are being
        trained. Example of callback usage:

        .. code-block:: python

            from cuml.internals import GraphBasedDimRedCallback

            class CustomCallback(GraphBasedDimRedCallback):
                def on_preprocess_end(self, embeddings):
                    print(embeddings.copy_to_host())

                def on_epoch_end(self, embeddings):
                    print(embeddings.copy_to_host())

                def on_train_end(self, embeddings):
                    print(embeddings.copy_to_host())

    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {None, 'input', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.
    build_algo: string (default='auto')
        How to build the knn graph. Supported build algorithms are ['auto', 'brute_force_knn',
        'nn_descent']. 'auto' chooses to run with brute force knn if number of data rows is
        smaller than or equal to 50K. Otherwise, runs with nn descent.
    build_kwds: dict (optional, default=None)
        Dictionary of parameters to configure the build algorithm. Default values:

        - `nnd_graph_degree` (int, default=64): Graph degree used for NN Descent.
          Must be ≥ `n_neighbors`.

        - `nnd_intermediate_graph_degree` (int, default=128): Intermediate graph degree for
          NN Descent. Must be > `nnd_graph_degree`.

        - `nnd_max_iterations` (int, default=20): Max NN Descent iterations.

        - `nnd_termination_threshold` (float, default=0.0001): Stricter threshold leads to
          better convergence but longer runtime.

        - `knn_n_clusters` (int, default=1): Number of clusters for data partitioning.
          Higher values reduce memory usage at the cost of accuracy. When `knn_n_clusters > 1`,
          UMAP can process data larger than device memory.

        - `knn_overlap_factor` (int, default=2): Number of clusters each data point belongs to.
          Valid only when `knn_n_clusters > 1`. Must be < 'knn_n_clusters'.

        Hints:

        - Increasing `nnd_graph_degree` and `nnd_max_iterations` may improve accuracy.

        - The ratio `knn_overlap_factor / knn_n_clusters` impacts memory usage.
          Approximately `(knn_overlap_factor / knn_n_clusters) * num_rows_in_entire_data`
          rows will be loaded onto device memory at once.  E.g., 2/20 uses less device
          memory than 2/10.

        - Larger `knn_overlap_factor` results in better accuracy of the final knn graph.
          E.g. While using similar amount of device memory,
          `(knn_overlap_factor / knn_n_clusters)` = 4/20 will have better accuracy
          than 2/10 at the cost of performance.

        - Start with `knn_overlap_factor = 2` and gradually increase (2->3->4 ...)
          for better accuracy.

        - Start with `knn_n_clusters = 4` and increase (4 → 8 → 16...) for less GPU
          memory usage. This is independent from knn_overlap_factor as long as
          'knn_overlap_factor' < 'knn_n_clusters'.

    device_ids : list[int], "all", or None, default=None
        The device IDs to use during fitting (only used when
        `build_algo=nn_descent` and `knn_n_clusters > 1`). May be a list of
        ids, ``"all"`` (to use all available devices), or ``None`` (to fit
        using a single GPU only). Default is None.

    Notes
    -----
    This module is heavily based on Leland McInnes' reference UMAP package.
    However, there are a number of differences and features that are not yet
    implemented in `cuml.umap`:

    * Using a pre-computed pairwise distance matrix (under consideration
      for future releases)
    * Manual initialization of initial embedding positions

    In addition to these missing features, you should expect to see
    the final embeddings differing between cuml.umap and the reference
    UMAP.

    References
    ----------
    .. [1] `Leland McInnes, John Healy, James Melville
       UMAP: Uniform Manifold Approximation and Projection for Dimension
       Reduction <https://arxiv.org/abs/1802.03426>`_

    .. [2] `Corey Nolet, Victor Lafargue, Edward Raff, Thejaswi Nanditale,
       Tim Oates, John Zedlewski, Joshua Patterson
       Bringing UMAP Closer to the Speed of Light with GPU Acceleration
       <https://arxiv.org/abs/2008.00325>`_
    """
    embedding_ = ReflectedAttr()

    _cpu_class_path = "umap.UMAP"

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "n_neighbors",
            "n_components",
            "n_epochs",
            "learning_rate",
            "min_dist",
            "spread",
            "set_op_mix_ratio",
            "local_connectivity",
            "repulsion_strength",
            "negative_sample_rate",
            "transform_queue_size",
            "init",
            "a",
            "b",
            "target_n_neighbors",
            "target_weight",
            "target_metric",
            "hash_input",
            "random_state",
            "force_serial_epochs",
            "callback",
            "metric",
            "metric_kwds",
            "precomputed_knn",
            "build_algo",
            "build_kwds",
            "device_ids",
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        if isinstance(model.init, str) and model.init not in _INITS:
            raise UnsupportedOnGPU(f"`init={model.init!r}` is not supported")

        try:
            coerce_metric(model.metric)
        except (ValueError, TypeError, NotImplementedError):
            raise UnsupportedOnGPU(f"`metric={model.metric!r}` is not supported")

        if model.target_metric not in _TARGET_METRICS:
            raise UnsupportedOnGPU(f"`target_metric={model.target_metric!r}` is not supported")

        if model.unique:
            raise UnsupportedOnGPU("`unique=True` is not supported")

        if model.densmap:
            raise UnsupportedOnGPU("`densmap=True` is not supported")

        precomputed_knn = model.precomputed_knn[:2]
        if all(item is None for item in precomputed_knn):
            precomputed_knn = None

        return {
            "n_neighbors": model.n_neighbors,
            "n_components": model.n_components,
            "metric": model.metric,
            "metric_kwds": model.metric_kwds,
            "n_epochs": model.n_epochs,
            "learning_rate": model.learning_rate,
            "min_dist": model.min_dist,
            "spread": model.spread,
            "set_op_mix_ratio": model.set_op_mix_ratio,
            "local_connectivity": model.local_connectivity,
            "repulsion_strength": model.repulsion_strength,
            "negative_sample_rate": model.negative_sample_rate,
            "transform_queue_size": model.transform_queue_size,
            "init": model.init,
            "a": model.a,
            "b": model.b,
            "target_n_neighbors": model.target_n_neighbors,
            "target_weight": model.target_weight,
            "target_metric": model.target_metric,
            "hash_input": True,
            "random_state": model.random_state,
            "force_serial_epochs": getattr(model, "force_serial_epochs", None),
            "precomputed_knn": precomputed_knn,
        }

    def _params_to_cpu(self):
        if (precomputed_knn := self.precomputed_knn) is None:
            precomputed_knn = (None, None, None)

        init = self.init
        if not isinstance(init, str):
            init = cp.asnumpy(init)

        return {
            "n_neighbors": self.n_neighbors,
            "n_components": self.n_components,
            "metric": self.metric,
            "metric_kwds": self.metric_kwds,
            "n_epochs": self.n_epochs,
            "learning_rate": self.learning_rate,
            "min_dist": self.min_dist,
            "spread": self.spread,
            "set_op_mix_ratio": self.set_op_mix_ratio,
            "local_connectivity": self.local_connectivity,
            "repulsion_strength": self.repulsion_strength,
            "negative_sample_rate": self.negative_sample_rate,
            "transform_queue_size": self.transform_queue_size,
            "init": init,
            "a": self.a,
            "b": self.b,
            "target_n_neighbors": self.target_n_neighbors,
            "target_weight": self.target_weight,
            "target_metric": self.target_metric,
            "random_state": self.random_state,
            "precomputed_knn": precomputed_knn,
        }

    def _attrs_from_cpu(self, model):
        if scipy.sparse.issparse(model._raw_data):
            raw_data = cupyx.scipy.sparse.csr_matrix(model._raw_data, dtype="float32")
        else:
            raw_data = cp.asarray(model._raw_data)

        attrs = {
            "embedding_": ArrayIndexPair(
                cp.asarray(model.embedding_, order="C"),
                None,
            ),
            "graph_": model.graph_.tocoo(),
            "_raw_data": raw_data,
            "_input_hash": model._input_hash,
            "_sparse_data": model._sparse_data,
            "_a": model._a,
            "_b": model._b,
            "_n_neighbors": model._n_neighbors,
            "n_features_in_": model._raw_data.shape[1],
            **super()._attrs_from_cpu(model),
        }

        # Transfer _sigmas and _rhos if available (needed for inverse_transform)
        if hasattr(model, "_sigmas") and model._sigmas is not None:
            attrs["_sigmas"] = cp.asarray(model._sigmas)
        if hasattr(model, "_rhos") and model._rhos is not None:
            attrs["_rhos"] = cp.asarray(model._rhos)

        return attrs

    def _attrs_to_cpu(self, model):
        from umap.umap_ import DISCONNECTION_DISTANCES

        disconnection_distance = DISCONNECTION_DISTANCES.get(self.metric, np.inf)

        raw_data = self._raw_data
        if cupyx.scipy.sparse.issparse(raw_data) or isinstance(raw_data, cp.ndarray):
            raw_data = raw_data.get()

        if (input_hash := getattr(self, "_input_hash", None)) is None:
            input_hash = _joblib_hash(raw_data)

        if (knn_dists := getattr(self, "_knn_dists", None)) is not None:
            knn_dists = cp.asnumpy(knn_dists)

        if (knn_indices := getattr(self, "_knn_indices", None)) is not None:
            knn_indices = cp.asnumpy(knn_indices)

        attrs = {
            "embedding_": self.embedding_.array.get(order="A"),
            "graph_": self.graph_.tocsr(),
            "graph_dists_": None,
            "_raw_data": raw_data,
            "_input_hash": input_hash,
            "_sparse_data": self._sparse_data,
            "_a": self._a,
            "_b": self._b,
            "_disconnection_distance": disconnection_distance,
            "_initial_alpha": self.learning_rate,
            "_n_neighbors": self._n_neighbors,
            "_n_features_out": self._n_features_out,
            "_supervised": self._supervised,
            "_small_data": False,
            "_knn_dists": knn_dists,
            "_knn_indices": knn_indices,
            # XXX: umap.UMAP requires _knn_search_index to transform on new data.
            # This is an instance of pynndescent.NNDescent, which can _currently_
            # only be recreated by rerunning the nndescent algorithm, effectively
            # repeating any work already done on GPU. Instead, we opt to set
            # _knn_search_index to None (which is allowed). In this case
            # calling umap.UMAP.transform on _new_ data will result in
            # a nicer NotImplementedError being raised informing the user
            # that that's not supported on the instance.
            "_knn_search_index": None,
            **super()._attrs_to_cpu(model),
        }

        # Transfer _sigmas and _rhos if available (needed for inverse_transform)
        if hasattr(self, "_sigmas") and self._sigmas is not None:
            attrs["_sigmas"] = self._sigmas.get()
        if hasattr(self, "_rhos") and self._rhos is not None:
            attrs["_rhos"] = self._rhos.get()

        return attrs

    def _sync_attrs_to_cpu(self, model):
        super()._sync_attrs_to_cpu(model)
        # _validate_parameters constructs the rest of umap.UMAP's internal state
        model._validate_parameters()

    def __init__(
        self,
        *,
        n_neighbors=15,
        n_components=2,
        metric="euclidean",
        metric_kwds=None,
        n_epochs=None,
        learning_rate=1.0,
        min_dist=0.1,
        spread=1.0,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        repulsion_strength=1.0,
        negative_sample_rate=5,
        transform_queue_size=4.0,
        init="spectral",
        a=None,
        b=None,
        target_n_neighbors=-1,
        target_weight=0.5,
        target_metric="categorical",
        hash_input=False,
        random_state=None,
        force_serial_epochs=None,
        precomputed_knn=None,
        callback=None,
        build_algo="auto",
        build_kwds=None,
        device_ids=None,
        verbose=False,
        output_type=None,
    ):
        super().__init__(verbose=verbose, output_type=output_type)

        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.metric = metric
        self.metric_kwds = metric_kwds
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.min_dist = min_dist
        self.spread = spread
        self.set_op_mix_ratio = set_op_mix_ratio
        self.local_connectivity = local_connectivity
        self.repulsion_strength = repulsion_strength
        self.negative_sample_rate = negative_sample_rate
        self.transform_queue_size = transform_queue_size
        self.init = init
        self.a = a
        self.b = b
        self.target_n_neighbors = target_n_neighbors
        self.target_weight = target_weight
        self.target_metric = target_metric
        self.hash_input = hash_input
        self.random_state = random_state
        self.force_serial_epochs = force_serial_epochs
        self.precomputed_knn = precomputed_knn
        self.callback = callback
        self.build_algo = build_algo
        self.build_kwds = build_kwds
        self.device_ids = device_ids

    @property
    @mlfunc(convert_output=False)
    def _n_features_out(self):
        return self.embedding_.array.shape[1]

    @generate_docstring(
        X="dense_sparse",
        skip_parameters_heading=True,
    )
    @mlfunc(set_input_type=True)
    def fit(self, X, y=None, *, knn_graph=None) -> "UMAP":
        """
        Fit X into an embedded space.

        Parameters
        ----------
        knn_graph: tuple[array, array], sparse-matrix, array, optional
            This feature allows the precomputation of the KNN outside of UMAP.

            This may take any of the valid forms accepted by the
            ``precomputed_knn`` parameter to ``UMAP``, and takes precedence
            over it. See the ``UMAP`` docstring on ``precomputed_knn`` for more
            information.
        """
        # Normalize X as cheaply as possible to minimize copies and work
        X, index = check_inputs(
            self,
            X,
            order=None,
            mem_type=None,
            accept_sparse=True,
            ensure_all_finite=False,
            return_index=True,
            reset=True,
        )
        if y is not None:
            y = check_y(y, dtype="float32", order="C")
            check_consistent_length(X, y)

        cdef int n_rows = X.shape[0]
        cdef int n_dims = X.shape[1]
        cdef bool X_is_sparse = is_sparse(X)

        cdef lib.UMAPParams params
        init_params(self, params, n_rows=n_rows, is_sparse=X_is_sparse)

        # Determine the required mem_type based on params and X
        if X_is_sparse:
            mem_type = "device"
        elif params.build_algo == lib.graph_build_algo.NN_DESCENT:
            mem_type = "host"
        elif (
            params.build_algo == lib.graph_build_algo.BRUTE_FORCE_KNN
            and params.build_params.n_clusters > 1
        ):
            mem_type = "host"
        elif knn_graph is not None or self.precomputed_knn is not None:
            # For dense inputs using a precomputed KNN, we leave the input in
            # its original mem_type so the device memory may be dropped earlier
            # if passed on host.
            mem_type = None
        else:
            mem_type = "device"

        # Now fully validate and coerce X to the required mem_type
        X = check_array(
            X,
            mem_type=mem_type,
            dtype="float32",
            order="C",
            accept_sparse="csr",
            ensure_min_samples=2,
            input_name="X",
        )

        cdef uintptr_t X_ptr = 0, X_indices_ptr = 0, X_indptr_ptr = 0
        cdef size_t X_nnz = 0

        if X_is_sparse:
            X_ptr = X.data.data.ptr
            X_indices_ptr = X.indices.data.ptr
            X_indptr_ptr = X.indptr.data.ptr
            X_nnz = X.nnz
        elif isinstance(X, np.ndarray):
            X_ptr = X.ctypes.data
        else:
            X_ptr = X.data.ptr

        cdef uintptr_t y_ptr = 0
        if y is not None:
            y_ptr = <uintptr_t>y.data.ptr

        cdef uintptr_t knn_dists_ptr = 0, knn_indices_ptr = 0
        if knn_graph is not None or self.precomputed_knn is not None:
            if y is not None and self.target_metric != "categorical":
                raise ValueError(
                    "Cannot provide a KNN graph when in semi-supervised mode "
                    "with categorical target_metric for now."
                )

            knn_indices, knn_dists = extract_knn_graph(
                (knn_graph if knn_graph is not None else self.precomputed_knn),
                X.shape[0],
                self._n_neighbors,
                mem_type=None,     # mirrors the input graph mem type
                indices_dtype=("int32" if X_is_sparse else "int64"),
            )
            if isinstance(knn_indices, cp.ndarray):
                knn_indices_ptr = <uintptr_t>knn_indices.data.ptr
                knn_dists_ptr = <uintptr_t>knn_dists.data.ptr
            else:
                knn_indices_ptr = <uintptr_t>knn_indices.ctypes.data
                knn_dists_ptr = <uintptr_t>knn_dists.ctypes.data
        else:
            knn_indices = knn_dists = None

        handle = get_handle(device_ids=self.device_ids)
        cdef handle_t * handle_ = <handle_t*> <size_t> handle.getHandle()
        cdef unique_ptr[device_buffer] embeddings_buffer
        cdef lib.HostCOO fss_graph = lib.HostCOO()
        handle_ = <handle_t*> <size_t> handle.getHandle()

        if not isinstance(self.init, str):
            init = check_array(
                self.init,
                dtype="float32",
                order="C",
                mem_type=None,
                input_name="init",
            )
            if init.shape != (n_rows, self.n_components):
                raise ValueError(
                    f"Expected `init` with shape {(n_rows, self.n_components)}, "
                    f"got {init.shape}"
                )
            embeddings_buffer.reset(
                new device_buffer(
                    <const void*><uintptr_t>(
                        init.data.ptr if isinstance(init, cp.ndarray) else init.ctypes.data
                    ),
                    <size_t> init.nbytes,
                    <cudaStream_t> handle_.get_stream(),
                    any_resource[device_accessible](
                        get_current_device_resource().get_mr()
                    )
                )
            )

        # Allocate device arrays for sigmas and rhos (needed for inverse_transform)
        # Note: fit_sparse in C++ doesn't output sigmas/rhos yet.
        cdef uintptr_t sigmas_ptr = 0
        cdef uintptr_t rhos_ptr = 0
        if not X_is_sparse:
            sigmas = cp.zeros(n_rows, dtype=np.float32)
            rhos = cp.zeros(n_rows, dtype=np.float32)
            sigmas_ptr = <uintptr_t>sigmas.data.ptr
            rhos_ptr = <uintptr_t>rhos.data.ptr
        else:
            sigmas = None
            rhos = None

        with nogil:
            if X_is_sparse:
                lib.fit_sparse(
                    handle_[0],
                    <int*><uintptr_t> X_indptr_ptr,
                    <int*><uintptr_t> X_indices_ptr,
                    <float*><uintptr_t> X_ptr,
                    X_nnz,
                    <float*> y_ptr,
                    n_rows,
                    n_dims,
                    <int*> knn_indices_ptr,
                    <float*> knn_dists_ptr,
                    &params,
                    embeddings_buffer,
                    fss_graph,
                )
            else:
                lib.fit(
                    handle_[0],
                    <float*> X_ptr,
                    <float*> y_ptr,
                    n_rows,
                    n_dims,
                    <int64_t*> knn_indices_ptr,
                    <float*> knn_dists_ptr,
                    &params,
                    embeddings_buffer,
                    fss_graph,
                    <float*> sigmas_ptr,
                    <float*> rhos_ptr,
                )
        handle.sync()

        buffer = DeviceBuffer.c_from_unique_ptr(move(embeddings_buffer))
        embedding = cp.ndarray(
            shape=(n_rows, self.n_components),
            dtype=np.float32,
            memptr=cp.cuda.MemoryPointer(
                cp.cuda.UnownedMemory(buffer.ptr, buffer.size, owner=buffer), 0
            ),
            order="C"
        )
        self.embedding_ = ArrayIndexPair(embedding, index)
        self.graph_ = copy_raft_host_coo_to_scipy_coo(fss_graph)
        self._raw_data = X
        self._sparse_data = X_is_sparse
        self._supervised = y is not None
        self._knn_indices = knn_indices
        self._knn_dists = knn_dists
        self._sigmas = sigmas
        self._rhos = rhos

        if self.hash_input:
            self._input_hash = _joblib_hash(X)
        else:
            self._input_hash = None

        return self

    @generate_docstring(
        skip_parameters_heading=True,
        return_values={
            "name": "X_new",
            "type": "dense",
            "description": "Embedding of the data in low-dimensional space.",
            "shape": "(n_samples, n_components)"
        }
    )
    @mlfunc(preserve_index=True, column_names="feature_names_out")
    def fit_transform(self, X, y=None, *, knn_graph=None):
        """
        Fit X into an embedded space and return that transformed
        output.

        There is a subtle difference between calling fit_transform(X)
        and calling fit().transform(). Calling fit_transform(X) will
        train the embeddings on X and return the embeddings. Calling
        fit(X).transform(X) will train the embeddings on X and then
        run a second optimization.

        Parameters
        ----------
        knn_graph : tuple[array, array], sparse-matrix, array, optional
            This feature allows the precomputation of the KNN outside of UMAP.

            This may take any of the valid forms accepted by the
            ``precomputed_knn`` parameter to ``UMAP``, and takes precedence
            over it. See the ``UMAP`` docstring on ``precomputed_knn`` for more
            information.
        """
        self.fit(X, y, knn_graph=knn_graph)
        return self.embedding_

    @generate_docstring(
        return_values={
            "name": "X_new",
            "type": "dense",
            "description": "Embedding of the data in low-dimensional space.",
            "shape": "(n_samples, n_components)"
        }
    )
    @mlfunc(preserve_index=True, column_names="feature_names_out")
    def transform(self, X):
        """
        Transform X into the existing embedded space and return that
        transformed output.

        Please refer to the reference UMAP implementation for information
        on the differences between fit_transform() and running fit()
        transform().

        Specifically, the transform() function is stochastic:
        https://github.com/lmcinnes/umap/issues/158
        """
        check_is_fitted(self)

        X = check_inputs(
            self,
            X,
            dtype="float32",
            order="C",
            accept_sparse="csr",
        )
        X_input_sparse = is_sparse(X)

        if self.hash_input and _joblib_hash(X) == self._input_hash:
            return self.embedding_

        if self._sparse_data and not X_input_sparse:
            logger.warn(
                "Model was trained on sparse data but dense data was provided to "
                "transform(). Converting to sparse."
            )
            X = cupyx.scipy.sparse.csr_matrix(X)
        elif not self._sparse_data and X_input_sparse:
            logger.warn(
                "Model was trained on dense data but sparse data was provided to "
                "transform(). Converting to dense."
            )
            X = cp.ascontiguousarray(X.toarray())

        cdef bool X_is_sparse = self._sparse_data
        cdef int n_rows = X.shape[0]
        cdef int n_cols = X.shape[1]
        cdef int orig_n_rows = self._raw_data.shape[0]

        cdef lib.UMAPParams params
        init_params(self, params, n_rows=n_rows, is_sparse=X_is_sparse, is_fit=False)

        out = cp.zeros(
            (n_rows, self.n_components),
            order="C",
            dtype=np.float32,
        )

        cdef uintptr_t X_ptr, X_indptr_ptr, X_indices_ptr
        cdef uintptr_t orig_ptr, orig_indptr_ptr, orig_indices_ptr
        cdef size_t X_nnz, orig_nnz
        if X_is_sparse:
            X_indptr_ptr = <uintptr_t>X.indptr.data.ptr
            X_indices_ptr = <uintptr_t>X.indices.data.ptr
            X_ptr = <uintptr_t>X.data.data.ptr
            X_nnz = X.nnz
            orig_indptr_ptr = self._raw_data.indptr.data.ptr
            orig_indices_ptr = self._raw_data.indices.data.ptr
            orig_ptr = self._raw_data.data.data.ptr
            orig_nnz = self._raw_data.nnz
        else:
            X_ptr = <uintptr_t>X.data.ptr
            orig_ptr = (
                self._raw_data.data.ptr
                if isinstance(self._raw_data, cp.ndarray)
                else self._raw_data.ctypes.data
            )

        cdef uintptr_t out_ptr = <uintptr_t>out.data.ptr
        cdef uintptr_t embedding_ptr = self.embedding_.array.data.ptr
        handle = get_handle(device_ids=self.device_ids)
        cdef handle_t* handle_ = <handle_t*><uintptr_t>handle.getHandle()

        with nogil:
            if X_is_sparse:
                lib.transform_sparse(
                    handle_[0],
                    <int*> X_indptr_ptr,
                    <int*> X_indices_ptr,
                    <float*> X_ptr,
                    X_nnz,
                    n_rows,
                    n_cols,
                    <int*> orig_indptr_ptr,
                    <int*> orig_indices_ptr,
                    <float*> orig_ptr,
                    orig_nnz,
                    orig_n_rows,
                    <float*> embedding_ptr,
                    orig_n_rows,
                    &params,
                    <float*> out_ptr,
                )
            else:
                lib.transform(
                    handle_[0],
                    <float*>X_ptr,
                    n_rows,
                    n_cols,
                    <float*>orig_ptr,
                    orig_n_rows,
                    <float*> embedding_ptr,
                    orig_n_rows,
                    &params,
                    <float*> out_ptr
                )
        handle.sync()

        return out

    @generate_docstring(
        X_shape="(n_samples, n_components)",
        return_values={
            "name": "X_new",
            "type": "dense",
            "description": "Generated data points in data space.",
            "shape": "(n_samples, n_features)"
        }
    )
    @mlfunc(preserve_index=True, column_names="feature_names_in")
    def inverse_transform(self, X):
        """Transform X in the existing embedded space back into the input
        data space and return that transformed output.
        """
        check_is_fitted(self)

        if self._sparse_data:
            raise ValueError("Inverse transform not available for sparse input.")
        if self.n_components >= 8:
            warnings.warn(
                "Inverse transform works best with low dimensional embeddings."
                " Results may be poor, or this approach to inverse transform"
                " may fail altogether! If you need a high dimensional latent"
                " space and inverse transform operations consider using an"
                " autoencoder."
            )

        # skip n_features_in_ validation
        X = check_array(X, dtype="float32", order="C")

        n_samples = X.shape[0]
        if X.shape[1] != self.n_components:
            raise ValueError(
                f"X has {X.shape[1]} components, but UMAP is expecting "
                f"{self.n_components} components as input"
            )

        # Get numpy arrays for preprocessing
        embedding_np = cp.asnumpy(self.embedding_.array, order="A")
        X_np = cp.asnumpy(X, order="A")
        raw_data_np = cp.asnumpy(self._raw_data, order="A")

        # Phase 1: Compute neighborhoods via Delaunay triangulation + BFS (CPU)
        # Cap neighborhood size to prevent explosion for high-dimensional data.
        # We use 3x n_neighbors as an upper bound since that's the scale at which
        # the manifold was learned. This avoids creating excessively dense graphs
        # when n_features >> n_neighbors (e.g., 10000 features).
        max_neighbors = 3 * self._n_neighbors
        min_vertices = min(raw_data_np.shape[1], raw_data_np.shape[0], max_neighbors)
        neighborhoods = _compute_inverse_neighborhoods(embedding_np, X_np, min_vertices)

        # Phase 2: Build graph and initial points (GPU-accelerated)
        inv_transformed_gpu, rows_gpu, cols_gpu, vals_gpu, raw_data_gpu = _build_inverse_graph(
            X_np, embedding_np, raw_data_np, neighborhoods, min_vertices, self._a, self._b
        )

        # Phase 3: CUDA optimization (call C++)
        cdef int n_epochs_inv
        if self.n_epochs is None:
            n_epochs_inv = 100 if n_samples <= 10000 else 30
        else:
            n_epochs_inv = int(self.n_epochs // 3)

        # Ensure C-contiguous layout for CUDA kernels
        inv_transformed_gpu = cp.ascontiguousarray(inv_transformed_gpu)

        cdef int c_n_samples = n_samples
        cdef int c_n_features = raw_data_np.shape[1]
        cdef int c_orig_n = raw_data_np.shape[0]
        cdef int c_nnz = vals_gpu.shape[0]

        cdef uintptr_t inv_ptr = inv_transformed_gpu.data.ptr
        cdef uintptr_t raw_ptr = raw_data_gpu.data.ptr
        cdef uintptr_t rows_ptr = rows_gpu.data.ptr
        cdef uintptr_t cols_ptr = cols_gpu.data.ptr
        cdef uintptr_t vals_ptr = vals_gpu.data.ptr

        # Check that sigmas and rhos are available (set during fit on dense data)
        if self._sigmas is None or self._rhos is None:
            raise ValueError(
                "inverse_transform requires sigmas and rhos arrays from fit. "
                "These may be missing if the model was loaded from a CPU UMAP "
                "model that did not have them, or if the model was not fitted."
            )
        cdef uintptr_t sigmas_ptr = self._sigmas.data.ptr
        cdef uintptr_t rhos_ptr = self._rhos.data.ptr

        cdef lib.UMAPParams params
        init_params(self, params, n_rows=n_samples, is_sparse=False, is_fit=False)

        handle = get_handle(device_ids=self.device_ids)
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

        lib.inverse_transform(
            handle_[0],
            <float*>inv_ptr, c_n_samples, c_n_features,
            <float*>raw_ptr, c_orig_n,
            <int*>rows_ptr, <int*>cols_ptr, <float*>vals_ptr, c_nnz,
            <float*>sigmas_ptr, <float*>rhos_ptr,
            &params, n_epochs_inv
        )
        handle.sync()

        return inv_transformed_gpu


def fuzzy_simplicial_set(
    X,
    n_neighbors,
    random_state=None,
    metric="euclidean",
    metric_kwds=None,
    knn_indices=None,
    knn_dists=None,
    set_op_mix_ratio=1.0,
    local_connectivity=1.0,
    verbose=False,
):
    """Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.

    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        The data to be modelled as a fuzzy simplicial set.
    n_neighbors: int
        The number of neighbors to use to approximate geodesic distance.
        Larger numbers induce more global estimates of the manifold that can
        miss finer detail, while smaller values will focus on fine manifold
        structure to the detriment of the larger picture.
    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.
    metric: string (default='euclidean').
        Distance metric to use. Supported distances are ['l1, 'cityblock',
        'taxicab', 'manhattan', 'euclidean', 'l2', 'sqeuclidean', 'canberra',
        'minkowski', 'chebyshev', 'linf', 'cosine', 'correlation', 'hellinger',
        'hamming', 'jaccard']
        Metrics that take arguments (such as minkowski) can have arguments
        passed via the metric_kwds dictionary.
        Note: The 'jaccard' distance metric is only supported for sparse
        inputs.
    metric_kwds: dict (optional, default=None)
        Metric argument
    knn_indices: array of shape (n_samples, n_neighbors) (optional)
        If the k-nearest neighbors of each point has already been calculated
        you can pass them in here to save computation time. This should be
        an array with the indices of the k-nearest neighbors as a row for
        each data point.
    knn_dists: array of shape (n_samples, n_neighbors) (optional)
        If the k-nearest neighbors of each point has already been calculated
        you can pass them in here to save computation time. This should be
        an array with the distances of the k-nearest neighbors as a row for
        each data point.
    set_op_mix_ratio: float (optional, default 1.0)
        Interpolate between (fuzzy) union and intersection as the set operation
        used to combine local fuzzy simplicial sets to obtain a global fuzzy
        simplicial sets. Both fuzzy set operations use the product t-norm.
        The value of this parameter should be between 0.0 and 1.0; a value of
        1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
        intersection.
    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.
    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.

    Returns
    -------
    fuzzy_simplicial_set: coo_matrix
        A fuzzy simplicial set represented as a sparse matrix. The (i, j) entry
        of the matrix represents the membership strength of the 1-simplex
        between the ith and jth sample points.
    """
    X = check_array(X, order="C", dtype="float32", input_name="X")

    cdef int n_rows = X.shape[0]
    cdef int n_cols = X.shape[1]

    cdef lib.UMAPParams params
    params.n_neighbors = n_neighbors
    params.random_state = check_random_seed(random_state)
    params.deterministic = (random_state is not None or n_rows < 300)
    params.set_op_mix_ratio = set_op_mix_ratio
    params.local_connectivity = local_connectivity
    params.metric = coerce_metric(metric)
    params.p = (metric_kwds or {}).get("p", 2.0)
    params.verbosity = logger._verbose_to_level(verbose)

    cdef uintptr_t X_ptr, knn_indices_ptr, knn_dists_ptr
    if knn_indices is not None and knn_dists is not None:
        knn_indices = check_array(
            knn_indices,
            dtype="int64",
            order="C",
            input_name="knn_indices",
        )
        knn_dists = check_array(
            knn_dists,
            dtype="float32",
            order="C",
            input_name="knn_dists",
        )
        X_ptr = 0
        knn_indices_ptr = knn_indices.data.ptr
        knn_dists_ptr = knn_dists.data.ptr
    else:
        X_ptr = X.data.ptr
        knn_indices_ptr = 0
        knn_dists_ptr = 0

    handle = get_handle()
    cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
    cdef unique_ptr[lib.COO] fss_graph_ptr = lib.get_graph(
        handle_[0],
        <float*> X_ptr,
        NULL,
        n_rows,
        n_cols,
        <int64_t*> knn_indices_ptr,
        <float*> knn_dists_ptr,
        &params
    )
    fss_graph = RaftCOO.from_ptr(fss_graph_ptr)
    return fss_graph.view_cupy_coo()


@mlfunc(preserve_index=True)
def simplicial_set_embedding(
    data,
    graph,
    n_components=2,
    initial_alpha=1.0,
    a=None,
    b=None,
    gamma=1.0,
    negative_sample_rate=5,
    n_epochs=None,
    init="spectral",
    random_state=None,
    force_serial_epochs=None,
    metric="euclidean",
    metric_kwds=None,
    output_metric="euclidean",
    output_metric_kwds=None,
    verbose=False,
):
    """Perform a fuzzy simplicial set embedding, using a specified
    initialisation method and then minimizing the fuzzy set cross entropy
    between the 1-skeletons of the high and low dimensional fuzzy simplicial
    sets.

    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The source data to be embedded by UMAP.
    graph: sparse matrix
        The 1-skeleton of the high dimensional fuzzy simplicial set as
        represented by a graph for which we require a sparse matrix for the
        (weighted) adjacency matrix.

        Note: When ``force_serial_epochs`` is enabled (either explicitly or
        via the auto-default for ``init='spectral'`` with
        ``n_components <= 512``), the COO is required to be sorted by row
        for internal CSR conversion. If it is not, it will be sorted internally.
        To avoid the extra sort, pass a row-sorted COO.
    n_components: int
        The dimensionality of the euclidean space into which to embed the data.
    initial_alpha: float
        Initial learning rate for the SGD.
    a: float
        Parameter of differentiable approximation of right adjoint functor
    b: float
        Parameter of differentiable approximation of right adjoint functor
    gamma: float
        Weight to apply to negative samples.
    negative_sample_rate: int (optional, default 5)
        The number of negative samples to select per positive sample
        in the optimization process. Increasing this value will result
        in greater repulsive force being applied, greater optimization
        cost, but slightly more accuracy.
    n_epochs: int (optional, default 0)
        The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings. If 0 is specified a value will be selected based on
        the size of the input dataset (200 for large datasets, 500 for small).
    init: string
        How to initialize the low dimensional embedding. Options are:
            * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
            * 'random': assign initial embedding positions at random.
            * An array-like with initial embedding positions.

        Note: When ``init='spectral'`` and ``n_components <= 512``,
        ``force_serial_epochs`` defaults to ``True`` because spectral
        initialization is more susceptible to outlier artifacts. Pass
        ``force_serial_epochs=False`` explicitly to disable and use the
        faster parallel batch kernel.
    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.
    force_serial_epochs: bool or None, optional (default=None)
        Controls whether optimization epochs use the sequential (reduced
        GPU parallelism) kernel. When ``None`` (the default), serial epochs
        are enabled automatically for ``init='spectral'`` with
        ``n_components <= 512`` because spectral initialization is more
        susceptible to outlier artifacts; for ``n_components > 512`` the
        auto-default falls back to ``False`` since the serial kernel does
        not support that range. Pass ``True`` to force serial epochs
        regardless of init (only supported for ``n_components <= 512``;
        otherwise a ``ValueError`` is raised), or ``False`` to disable them.
    metric: string (default='euclidean').
        Distance metric to use. Supported distances are ['l1, 'cityblock',
        'taxicab', 'manhattan', 'euclidean', 'l2', 'sqeuclidean', 'canberra',
        'minkowski', 'chebyshev', 'linf', 'cosine', 'correlation', 'hellinger',
        'hamming', 'jaccard']
        Metrics that take arguments (such as minkowski) can have arguments
        passed via the metric_kwds dictionary.
        Note: The 'jaccard' distance metric is only supported for sparse
        inputs.
    metric_kwds: dict (optional, default=None)
        Metric argument
    output_metric: function
        Function returning the distance between two points in embedding space
        and the gradient of the distance wrt the first argument.
    output_metric_kwds: dict
        Key word arguments to be passed to the output_metric function.
    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.

    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized of ``graph`` into an ``n_components`` dimensional
        euclidean space.
    """
    X = check_array(
        data,
        dtype="float32",
        order="C",
        input_name="X",
    )

    cdef int n_rows = X.shape[0]
    cdef int n_cols = X.shape[1]

    if a is None or b is None:
        a, b = find_ab_params()

    cdef lib.UMAPParams params
    params.n_components = n_components
    params.initial_alpha = initial_alpha
    params.learning_rate = initial_alpha
    params.a = a
    params.b = b
    params.repulsion_strength = gamma
    params.negative_sample_rate = negative_sample_rate
    params.n_epochs = n_epochs or 0
    params.random_state = check_random_seed(random_state)
    if force_serial_epochs is None:
        # Auto-enable only for spectral init within the serial kernel's supported
        # n_components range (the warp-based serial kernel supports up to 512).
        params.force_serial_epochs = (
            isinstance(init, str)
            and init == "spectral"
            and n_components <= _FORCE_SERIAL_EPOCHS_MAX_N_COMPONENTS
        )
    else:
        if force_serial_epochs and n_components > _FORCE_SERIAL_EPOCHS_MAX_N_COMPONENTS:
            raise ValueError(
                f"force_serial_epochs=True is only supported for "
                f"n_components <= {_FORCE_SERIAL_EPOCHS_MAX_N_COMPONENTS}, "
                f"got n_components={n_components}. Pass "
                f"force_serial_epochs=False or None to use the parallel "
                f"batch kernel."
            )
        params.force_serial_epochs = force_serial_epochs
    params.deterministic = (random_state is not None or n_rows < 300)
    params.metric = coerce_metric(metric)
    params.p = (metric_kwds or {}).get("p", 2.0)
    params.target_weight = (output_metric_kwds or {}).get("p", 0.5)
    params.verbosity = logger._verbose_to_level(verbose)

    if output_metric in _TARGET_METRICS:
        params.target_metric = _TARGET_METRICS[output_metric]
    else:
        raise ValueError(
            f"Expected `output_metric` to be one of {list(_TARGET_METRICS)}, "
            f"got {output_metric!r}"
        )

    cdef bool initialized = not isinstance(init, str)
    if initialized:
        embedding = check_array(
            init,
            dtype="float32",
            order="C",
            input_name="init",
        )
        if embedding.shape != (n_rows, n_components):
            raise ValueError(
                f"Expected `init` with shape {(n_rows, n_components)}, "
                f"got {embedding.shape}"
            )
    elif isinstance(init, str) and init in _INITS:
        params.init = _INITS[init]
        embedding = cp.zeros((n_rows, n_components), order="C", dtype="float32")
    else:
        raise ValueError(
            "Expected `init` to be an array or one of ['random', 'spectral'], "
            " got `{init!r}`"
        )

    graph = graph.tocoo()
    graph.sum_duplicates()
    if not isinstance(graph, cupyx.scipy.sparse.coo_matrix):
        graph = cupyx.scipy.sparse.coo_matrix(graph)

    handle = get_handle()
    cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
    cdef RaftCOO fss_graph = RaftCOO.from_cupy_coo(handle, graph)
    cdef uintptr_t embedding_ptr = embedding.data.ptr
    cdef uintptr_t X_ptr = X.data.ptr

    if initialized:
        lib.refine(
            handle_[0],
            <float*> X_ptr,
            n_rows,
            n_cols,
            fss_graph.get(),
            &params,
            <float*> embedding_ptr
        )
    else:
        lib.init_and_refine(
            handle_[0],
            <float*> X_ptr,
            n_rows,
            n_cols,
            fss_graph.get(),
            &params,
            <float*> embedding_ptr
        )
    return embedding
