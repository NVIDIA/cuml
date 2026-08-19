# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import cupy as cp
import numpy as np

import cuml
from cuml.common.doc_utils import generate_docstring
from cuml.internals import logger
from cuml.internals.base import Base, get_handle
from cuml.internals.interop import InteropMixin, UnsupportedOnGPU
from cuml.internals.mixins import ClusterMixin, CMajorInputTagMixin
from cuml.internals.outputs import ArrayIndexPair, ReflectedAttr, mlfunc
from cuml.internals.validation import check_inputs, check_is_fitted

from cython.operator cimport dereference as deref
from libc.stdint cimport int64_t, uint64_t, uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t
from rmm.librmm.device_uvector cimport device_uvector

cimport cuml.cluster.hdbscan.headers as lib
from cuml.metrics.distance_type cimport DistanceType


def import_hdbscan():
    """Import `hdbscan`, raising a nicer error if missing"""
    try:
        import hdbscan
        return hdbscan
    except ImportError as exc:
        raise ImportError(
            "The `hdbscan` library is required to use this functionality, "
            "but is not currently installed."
        ) from exc


def hdbscan_at_least(min_version):
    """Check if the version of `hdbscan` installed is >= min_version"""
    # hdbscan doesn't have `hdbscan.__version__`, so we have to
    # check the package metadata
    from importlib.metadata import version

    from packaging.version import Version
    return Version(version("hdbscan")) >= Version(min_version)


_metrics_mapping = {
    "l2": DistanceType.L2SqrtExpanded,
    "euclidean": DistanceType.L2SqrtExpanded,
}


def _cupy_array_from_ptr(ptr, shape, dtype, owner):
    """Create a cupy array from a pointer and metadata."""
    dtype = np.dtype(dtype)
    mem = cp.cuda.UnownedMemory(
        ptr=ptr, size=np.prod(shape) * dtype.itemsize, owner=owner,
    )
    mem_ptr = cp.cuda.memory.MemoryPointer(mem, 0)
    return cp.ndarray(shape=shape, dtype=dtype, memptr=mem_ptr)


cdef class _HDBSCANState:
    """Holds internal state of a fit HDBSCAN model.

    This class exists to decouple the c++ classes backing the results of an
    HDBSCAN fit from an `HDBSCAN` estimator itself. This helps ensure that
    refitting the same estimator doesn't invalidate old arrays that may still
    be lingering in memory.

    Further, it handles all the ways to coerce other values into internal state
    of `HDBSCAN`:

    - `init_and_fit`: for a direct call to `HDBSCAN.fit`
    - `from_dendrogram`: for testing comparisons to upstream `hdbscan`
    - `from_dict`: for unpickling
    - `from_sklearn`: for coercion from `hdbscan.HDBSCAN`

    The wrapper class helps isolate all `new`/`del` calls from the rest of the
    code. Any exposed output arrays have their lifetimes tied to it, ensuring
    that memory won't be freed until no python references still exist.
    """

    # A pointer to an `hdbscan_output` instance from a `fit` call, or `NULL` if
    # this state was initialized through a different method
    cdef lib.hdbscan_output *hdbscan_output

    # A pointer to a `CondensedHierarchy`, or `NULL` if this state was
    # initialized through a `fit` call.
    cdef lib.CondensedHierarchy[int64_t, float] *condensed_tree

    # The generated PredictionData, or NULL if prediction data was not yet generated.
    cdef lib.PredictionData[int64_t, float] *prediction_data

    # The number of clusters
    cdef public int n_clusters

    # A cupy array. Either a view of `hdbscan_output`, or memory allocated by
    # cupy (if not from a `fit` call). This is also passed to `PredictionData`,
    # which keeps a view of it (but doesn't do anything to keep it alive while
    # using it). We keep a reference here to work around that.
    cdef object core_dists

    # A cupy array. Either a view of `hdbscan_output`, or memory allocated by
    # cupy (if not from a `fit` call).
    cdef object inverse_label_map

    # A cached numpy array of the condensed tree, or None if a host-array version
    # of the tree hasn't been requested yet.
    cdef object cached_condensed_tree

    def __dealloc__(self):
        if self.prediction_data != NULL:
            del self.prediction_data
            self.prediction_data = NULL
        if self.condensed_tree != NULL:
            del self.condensed_tree
            self.condensed_tree = NULL
        if self.hdbscan_output != NULL:
            del self.hdbscan_output
            self.hdbscan_output = NULL

    def to_dict(self):
        """Returns a dict that can later be passed to `from_dict` to recreate state."""
        return {
            "n_leaves": self.get_condensed_tree().get_n_leaves(),
            "n_clusters": self.n_clusters,
            "core_dists": self.core_dists,
            "inverse_label_map": self.inverse_label_map,
            "condensed_tree": self.get_condensed_tree_array(),
        }

    def _init_from_condensed_tree_array(self, handle, tree, n_leaves):
        """Shared helper for initializing a `CondensedHierarchy` from a condensed_tree array"""
        self.cached_condensed_tree = tree

        parents = np.ascontiguousarray(tree["parent"], dtype=np.int64)
        children = np.ascontiguousarray(tree["child"], dtype=np.int64)
        lambdas = np.ascontiguousarray(tree["lambda_val"], dtype=np.float32)
        sizes = np.ascontiguousarray(tree["child_size"], dtype=np.int64)

        cdef int n_edges = len(tree)
        cdef handle_t *handle_ = <handle_t*> <size_t> handle.getHandle()
        self.condensed_tree = new lib.CondensedHierarchy[int64_t, float](
            handle_[0],
            n_leaves,
            n_edges,
            <int64_t*><uintptr_t>(parents.ctypes.data),
            <int64_t*><uintptr_t>(children.ctypes.data),
            <float*><uintptr_t>(lambdas.ctypes.data),
            <int64_t*><uintptr_t>(sizes.ctypes.data),
        )

    @staticmethod
    def from_dict(mapping):
        """Initialize internal state from the output of `to_dict`."""
        cdef _HDBSCANState self = _HDBSCANState.__new__(_HDBSCANState)
        self.n_clusters = mapping["n_clusters"]
        self.core_dists = mapping["core_dists"]
        self.inverse_label_map = mapping["inverse_label_map"]
        handle = get_handle()
        self._init_from_condensed_tree_array(
            handle, mapping["condensed_tree"], mapping["n_leaves"]
        )
        return self

    @staticmethod
    def from_sklearn(model, X, dendrogram=None):
        """Initialize internal state from a fitted CPU HDBSCAN instance."""
        cdef DistanceType metric = _metrics_mapping[model.metric]
        cdef lib.CLUSTER_SELECTION_METHOD cluster_selection_method = {
            "eom": lib.CLUSTER_SELECTION_METHOD.EOM,
            "leaf": lib.CLUSTER_SELECTION_METHOD.LEAF,
        }[model.cluster_selection_method]

        cdef _HDBSCANState self = _HDBSCANState.__new__(_HDBSCANState)

        cdef int n_rows = X.shape[0]
        cdef int n_cols = X.shape[1]

        handle = get_handle()
        if dendrogram is None:
            self._init_from_condensed_tree_array(
                handle, model._condensed_tree, n_rows
            )
        else:
            self._init_from_dendrogram(
                handle, dendrogram, model.min_cluster_size
            )

        self.core_dists = cp.empty(n_rows, dtype=np.float32)
        cdef handle_t *handle_ = <handle_t*> <size_t> handle.getHandle()
        cdef float* X_ptr = <float*><uintptr_t>X.data.ptr
        cdef float* core_dists_ptr = <float*><uintptr_t>self.core_dists.data.ptr
        cdef bool allow_single_cluster = model.allow_single_cluster
        cdef int64_t max_cluster_size = model.max_cluster_size
        cdef float cluster_selection_epsilon = model.cluster_selection_epsilon
        cdef int min_samples = model.min_samples or model.min_cluster_size
        cdef device_uvector[int64_t] *temp_buffer = new device_uvector[int64_t](
            0,
            handle_[0].get_stream(),
        )

        with nogil:
            lib.compute_core_dists(
                handle_[0],
                X_ptr,
                core_dists_ptr,
                n_rows,
                n_cols,
                metric,
                min_samples
            )
            lib.compute_inverse_label_map(
                handle_[0],
                deref(self.condensed_tree),
                n_rows,
                cluster_selection_method,
                deref(temp_buffer),
                allow_single_cluster,
                max_cluster_size,
                cluster_selection_epsilon
            )
        handle.sync()

        self.n_clusters = temp_buffer.size()

        if self.n_clusters > 0:
            self.inverse_label_map = _cupy_array_from_ptr(
                <size_t>temp_buffer.data(),
                (self.n_clusters,),
                np.int64,
                self
            ).copy()
        else:
            self.inverse_label_map = cp.empty((0,), dtype=np.int64)

        del temp_buffer

        return self

    def _init_from_dendrogram(
        self, handle, dendrogram, int min_cluster_size
    ):
        """Initialize the condensed hierarchy from a linkage dendrogram."""
        children = cp.asarray(
            dendrogram[:, 0:2], order="C", dtype="int64"
        )
        lambdas = cp.asarray(dendrogram[:, 2], order="C", dtype="float32")
        sizes = cp.asarray(dendrogram[:, 3], order="C", dtype="int64")

        cdef size_t n_leaves = dendrogram.shape[0] + 1
        cdef handle_t *handle_ = <handle_t*> <size_t> handle.getHandle()

        self.condensed_tree = new lib.CondensedHierarchy[int64_t, float](
            handle_[0], n_leaves
        )
        cdef int64_t* children_ptr = <int64_t*><uintptr_t>children.data.ptr
        cdef float* lambdas_ptr = <float*><uintptr_t>lambdas.data.ptr
        cdef int64_t* sizes_ptr = <int64_t*><uintptr_t>sizes.data.ptr
        with nogil:
            lib.build_condensed_hierarchy(
                handle_[0],
                children_ptr,
                lambdas_ptr,
                sizes_ptr,
                min_cluster_size,
                n_leaves,
                deref(self.condensed_tree)
            )

    cdef lib.CondensedHierarchy[int64_t, float]* get_condensed_tree(self) nogil:
        if self.hdbscan_output != NULL:
            return &(self.hdbscan_output.get_condensed_tree())
        return self.condensed_tree

    @staticmethod
    def from_dendrogram(dendrogram, int min_cluster_size):
        """Initialize internal state from a ScipPy dendrogram.

        Parameters
        ----------
        dendrogram : array-like (size n_samples, 4)
            Dendrogram in Scipy hierarchy format

        min_cluster_size : int
            Minimum number of children for a cluster to persist
        """
        cdef _HDBSCANState self = _HDBSCANState.__new__(_HDBSCANState)

        handle = get_handle()
        self._init_from_dendrogram(handle, dendrogram, min_cluster_size)
        return self

    @staticmethod
    cdef init_and_fit(
        X,
        lib.HDBSCANParams params,
        DistanceType metric,
        bool gen_min_span_tree,
        device_ids=None,
    ):
        """Initialize internal state from a new `fit`"""
        cdef _HDBSCANState self = _HDBSCANState.__new__(_HDBSCANState)

        cdef int n_rows = X.shape[0]
        cdef int n_cols = X.shape[1]

        # Allocate output structures
        labels = cp.empty(n_rows, dtype="int64")
        probabilities = cp.empty(n_rows, dtype="float32")

        children = cp.empty((2, n_rows), order="F", dtype="int64")
        lambdas = cp.empty(n_rows, dtype="float32")
        sizes = cp.empty(n_rows, dtype="int64")

        mst_src = cp.empty(n_rows - 1, dtype="int64")
        mst_dst = cp.empty(n_rows - 1, dtype="int64")
        mst_weights = cp.empty(n_rows - 1, dtype="float32")

        core_dists = cp.empty(n_rows, dtype="float32")

        handle = get_handle(device_ids=device_ids)
        cdef handle_t* handle_ = <handle_t*><uintptr_t>handle.getHandle()
        cdef float* X_ptr = <float*><uintptr_t>(
            X.data.ptr if isinstance(X, cp.ndarray) else X.ctypes.data
        )
        cdef float* core_dists_ptr = <float*><uintptr_t>core_dists.data.ptr

        self.hdbscan_output = new lib.hdbscan_output(
            handle_[0],
            n_rows,
            <int64_t*><uintptr_t>(labels.data.ptr),
            <float*><uintptr_t>(probabilities.data.ptr),
            <int64_t*><uintptr_t>(children.data.ptr),
            <int64_t*><uintptr_t>(sizes.data.ptr),
            <float*><uintptr_t>(lambdas.data.ptr),
            <int64_t*><uintptr_t>(mst_src.data.ptr),
            <int64_t*><uintptr_t>(mst_dst.data.ptr),
            <float*><uintptr_t>(mst_weights.data.ptr)
        )

        # Execute fit
        with nogil:
            lib.hdbscan(
                handle_[0],
                X_ptr,
                n_rows,
                n_cols,
                metric,
                params,
                deref(self.hdbscan_output),
                core_dists_ptr,
            )
        handle.sync()

        # Extract and store local state
        self.n_clusters = self.hdbscan_output.get_n_clusters()
        if self.n_clusters > 0:
            self.inverse_label_map = _cupy_array_from_ptr(
                <size_t>self.hdbscan_output.get_inverse_label_map(),
                (self.n_clusters,),
                np.int64,
                self
            )
        else:
            self.inverse_label_map = cp.empty((0,), dtype=np.int64)
        self.core_dists = core_dists

        # Extract and prepare results
        if self.n_clusters > 0:
            cluster_persistence = _cupy_array_from_ptr(
                <size_t>self.hdbscan_output.get_stabilities(),
                (self.n_clusters,),
                np.float32,
                self,
            )
        else:
            cluster_persistence = cp.empty((0,), dtype=np.float32)

        if gen_min_span_tree:
            min_span_tree = np.column_stack(
                (
                    mst_src.get().astype("float64"),
                    mst_dst.get().astype("float64"),
                    mst_weights.get().astype("float64"),
                )
            )
        else:
            min_span_tree = None

        single_linkage_tree = np.concatenate(
            (
                children.get().astype("float64"),
                lambdas.get().astype("float64")[None, :],
                sizes.get().astype("float64")[None, :],
            ),
        )[:, :n_rows - 1].T

        return (
            self,
            self.n_clusters,
            labels,
            probabilities,
            cluster_persistence,
            min_span_tree,
            single_linkage_tree,
        )

    def generate_prediction_data(self, X, labels):
        """Generate `prediction_data` if it hasn't already been generated."""
        if self.prediction_data != NULL:
            return

        handle = get_handle()

        cdef int n_rows = X.shape[0]
        cdef int n_cols = X.shape[1]
        cdef int64_t* labels_ptr = <int64_t*><uintptr_t>labels.data.ptr
        cdef int64_t* inverse_label_map_ptr = (
            <int64_t*><uintptr_t>self.inverse_label_map.data.ptr
        )
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

        self.prediction_data = new lib.PredictionData[int64_t, float](
            handle_[0],
            n_rows,
            n_cols,
            <float*><uintptr_t>(self.core_dists.data.ptr),
        )

        cdef lib.CondensedHierarchy[int64_t, float] *condensed_tree = self.get_condensed_tree()

        with nogil:
            lib.generate_prediction_data(
                handle_[0],
                deref(condensed_tree),
                labels_ptr,
                inverse_label_map_ptr,
                self.n_clusters,
                deref(self.prediction_data),
            )
        handle.sync()

    def get_condensed_tree_array(self):
        """Coerce `condensed_tree` to the same array layout that `hdbscan.HDBSCAN` uses."""
        if self.cached_condensed_tree is not None:
            # Cached, return the same result
            return self.cached_condensed_tree

        cdef lib.CondensedHierarchy[int64_t, float]* condensed_tree = self.get_condensed_tree()

        n_condensed_tree_edges = condensed_tree.get_n_edges()

        tree = np.recarray(
            shape=(n_condensed_tree_edges,),
            formats=[np.intp, np.intp, np.float64, np.intp],
            names=("parent", "child", "lambda_val", "child_size"),
        )

        parents = _cupy_array_from_ptr(
            <size_t>condensed_tree.get_parents(),
            (n_condensed_tree_edges,),
            np.int64,
            self,
        )

        children = _cupy_array_from_ptr(
            <size_t>condensed_tree.get_children(),
            (n_condensed_tree_edges,),
            np.int64,
            self,
        )

        lambdas = _cupy_array_from_ptr(
            <size_t>condensed_tree.get_lambdas(),
            (n_condensed_tree_edges,),
            np.float32,
            self,
        )

        sizes = _cupy_array_from_ptr(
            <size_t>condensed_tree.get_sizes(),
            (n_condensed_tree_edges,),
            np.int64,
            self,
        )

        tree["parent"] = parents.get()
        tree["child"] = children.get()
        tree["lambda_val"] = lambdas.get()
        tree["child_size"] = sizes.get()

        self.cached_condensed_tree = tree

        return tree


class HDBSCAN(InteropMixin, ClusterMixin, CMajorInputTagMixin, Base):
    """
    HDBSCAN Clustering

    Recursively merges the pair of clusters that minimally increases a
    given linkage distance.

    Note that while the algorithm is generally deterministic and should
    provide matching results between RAPIDS and the Scikit-learn Contrib
    versions, the construction of the k-nearest neighbors graph and
    minimum spanning tree can introduce differences between the two
    algorithms, especially when several nearest neighbors around a
    point might have the same distance. While the differences in
    the minimum spanning trees alone might be subtle, they can
    (and often will) lead to some points being assigned different
    cluster labels between the two implementations.

    Parameters
    ----------
    alpha : float, optional (default=1.0)
        A distance scaling parameter as used in robust single linkage.

    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.

    min_cluster_size : int, optional (default = 5)
        The minimum number of samples in a group for that group to be
        considered a cluster; groupings smaller than this size will be left
        as noise.

    min_samples : int, optional (default=None)
        The number of samples in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
        If 'None', it defaults to the min_cluster_size.

    cluster_selection_epsilon : float, optional (default=0.0)
        A distance threshold. Clusters below this value will be merged.
        Note that this should not be used
        if we want to predict the cluster labels for new points in future
        (e.g. using approximate_predict), as the approximate_predict function
        is not aware of this argument.

    max_cluster_size : int, optional (default=0)
        A limit to the size of clusters returned by the eom algorithm.
        Has no effect when using leaf clustering (where clusters are
        usually small regardless) and can also be overridden in rare
        cases by a high value for cluster_selection_epsilon. Note that
        this should not be used if we want to predict the cluster labels
        for new points in future (e.g. using approximate_predict), as
        the approximate_predict function is not aware of this argument.

    metric : string, optional (default='euclidean')
        The metric to use when calculating distance between instances in a
        feature array. Allowed values: 'euclidean'.

    p : int, optional (default=None)
        p value to use if using the minkowski metric.

    cluster_selection_method : string, optional (default='eom')
        The method used to select clusters from the condensed tree. The
        standard approach for HDBSCAN* is to use an Excess of Mass algorithm
        to find the most persistent clusters. Alternatively you can instead
        select the clusters at the leaves of the tree -- this provides the
        most fine grained and homogeneous clusters. Options are:

            * ``eom``
            * ``leaf``

    allow_single_cluster : bool, optional (default=False)
        By default HDBSCAN* will not produce a single cluster, setting this
        to True will override this and allow single cluster results in
        the case that you feel this is a valid result for your dataset.

    gen_min_span_tree : bool, optional (default=False)
        Whether to populate the `minimum_spanning_tree_` member for
        utilizing plotting tools. This requires the `hdbscan` CPU Python
        package to be installed.

    output_type : {None, 'input', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    prediction_data : bool, optional (default=False)
        Whether to generate extra cached data for predicting labels or
        membership vectors few new unseen points later. If you wish to
        persist the clustering object for later re-use you probably want
        to set this to True.

    build_algo: string (default='brute_force')
        How to build the knn graph. Supported build algorithms are ['brute_force',
        'nn_descent']. The 'nn_descent' algorithm is typically faster,
        but may result in a slight accuracy drop compared to 'brute_force'.

    build_kwds: dict (optional, default=None)
        Dictionary of parameters to configure the build algorithm. Default values:

        - `knn_n_clusters` (int, default=1): Number of clusters for data partitioning.
          Higher values reduce memory usage at the cost of accuracy. When `knn_n_clusters > 1`,
          HDBSCAN can process data larger than device memory.

        - `knn_overlap_factor` (int, default=2): Number of clusters each data point belongs to.
          Valid only when `knn_n_clusters > 1`. Must be < 'knn_n_clusters'.

        - `nnd_graph_degree` (int, default=64): Graph degree used for NN Descent when
          `build_algo=nn_descent`. Must be ≥ `min_samples+1`.

        - `nnd_intermediate_graph_degree` (int, default=128): Intermediate graph degree for
          NN Descent. Must be > `nnd_graph_degree`.

        - `nnd_max_iterations` (int, default=20): Max NN Descent iterations when
          `build_algo=nn_descent`.

        - `nnd_termination_threshold` (float, default=0.0001): Stricter threshold leads to
          better convergence but longer runtime.

        Hints:

        - Increasing `nnd_graph_degree` and `nnd_max_iterations` may improve accuracy
          when `build_algo=nn_descent`.

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
        `build_algo=nn_descent` and `nnd_n_clusters > 1`). May be a list of
        ids, ``"all"`` (to use all available devices), or ``None`` (to fit
        using a single GPU only). Default is None.

    Attributes
    ----------
    labels_ : ndarray, shape (n_samples, )
        Cluster labels for each point in the dataset given to fit().
        Noisy samples are given the label -1.

    probabilities_ : ndarray, shape (n_samples, )
        The strength with which each sample is a member of its assigned
        cluster. Noise points have probability zero; points in clusters
        have values assigned proportional to the degree that they
        persist as part of the cluster.

    cluster_persistence_ : ndarray, shape (n_clusters, )
        A score of how persistent each cluster is. A score of 1.0 represents
        a perfectly stable cluster that persists over all distance scales,
        while a score of 0.0 represents a perfectly ephemeral cluster. These
        scores can be used to gauge the relative coherence of the
        clusters output by the algorithm.

    condensed_tree_ : CondensedTree object
        The condensed tree produced by HDBSCAN. The object has methods
        for converting to pandas, networkx, and plotting.

    single_linkage_tree_ : SingleLinkageTree object
        The single linkage tree produced by HDBSCAN. The object has methods
        for converting to pandas, networkx, and plotting.

    minimum_spanning_tree_ : MinimumSpanningTree object
        The minimum spanning tree of the mutual reachability graph generated
        by HDBSCAN. Note that this is not generated by default and will only
        be available if `gen_min_span_tree` was set to True on object creation.
        Even then in some optimized cases a tree may not be generated.

    """
    labels_ = ReflectedAttr()
    probabilities_ = ReflectedAttr()
    cluster_persistence_ = ReflectedAttr()

    _cpu_class_path = "hdbscan.HDBSCAN"

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "min_cluster_size",
            "min_samples",
            "cluster_selection_epsilon",
            "max_cluster_size",
            "metric",
            "alpha",
            "p",
            "cluster_selection_method",
            "allow_single_cluster",
            "gen_min_span_tree",
            "prediction_data",
            "build_algo",
            "build_kwds",
            "device_ids",
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        if callable(model.metric) or model.metric not in _metrics_mapping:
            raise UnsupportedOnGPU(f"`metric={model.metric!r}` is not supported")

        if (
            isinstance(model.memory, str)
            or getattr(model.memory, "location", None) is not None
        ):
            # Result caching is unsupported
            raise UnsupportedOnGPU(f"`memory={model.memory!r}` is not supported")

        if model.match_reference_implementation:
            raise UnsupportedOnGPU("`match_reference_implementation=True` is not supported")

        if model.branch_detection_data:
            raise UnsupportedOnGPU("`branch_detection_data=True` is not supported")

        return {
            "min_cluster_size": model.min_cluster_size,
            "min_samples": model.min_samples,
            "cluster_selection_epsilon": model.cluster_selection_epsilon,
            "max_cluster_size": model.max_cluster_size,
            "metric": model.metric,
            "alpha": model.alpha,
            "p": model.p,
            "cluster_selection_method": model.cluster_selection_method,
            "allow_single_cluster": model.allow_single_cluster,
            "gen_min_span_tree": model.gen_min_span_tree,
            "prediction_data": model.prediction_data,
        }

    def _params_to_cpu(self):
        return {
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "cluster_selection_epsilon": self.cluster_selection_epsilon,
            "max_cluster_size": self.max_cluster_size,
            "metric": self.metric,
            "alpha": self.alpha,
            "p": self.p,
            "cluster_selection_method": self.cluster_selection_method,
            "allow_single_cluster": self.allow_single_cluster,
            "gen_min_span_tree": self.gen_min_span_tree,
            "prediction_data": self.prediction_data,
        }

    def _attrs_from_cpu(self, model):
        if (raw_data_cpu := getattr(model, "_raw_data", None)) is None:
            # Fit with precomputed metric
            raise UnsupportedOnGPU("Models fit with a precomputed metric are not supported")

        if not isinstance(raw_data_cpu, np.ndarray):
            # Sparse input
            raise UnsupportedOnGPU("Sparse inputs are not supported")

        raw_data = cp.asarray(raw_data_cpu, order="C", dtype="float32")
        labels = cp.asarray(model.labels_, order="C", dtype="int64")
        state = _HDBSCANState.from_sklearn(model, raw_data)
        if model._prediction_data is not None:
            state.generate_prediction_data(raw_data, labels)

        return {
            # XXX: `hdbscan.HDBSCAN` doesn't set `n_features_in_` currently, we need
            # to infer this ourselves from the raw data.
            "n_features_in_": raw_data_cpu.shape[1],
            "labels_": ArrayIndexPair(labels, None),
            "probabilities_": cp.asarray(model.probabilities_, dtype="float32"),
            "cluster_persistence_": cp.asarray(model.cluster_persistence_, dtype="float32"),
            "_raw_data": ArrayIndexPair(raw_data, None),
            "_raw_data_cpu": raw_data_cpu,
            "_state": state,
            "n_clusters_": state.n_clusters,
            "_single_linkage_tree": model._single_linkage_tree,
            "_min_spanning_tree": model._min_spanning_tree,
            "_prediction_data": model._prediction_data,
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        out = {
            "labels_": self.labels_.array.get(order="A"),
            "probabilities_": self.probabilities_.get(order="A"),
            "cluster_persistence_": self.cluster_persistence_.get(order="A"),
            "_condensed_tree": self._condensed_tree,
            "_single_linkage_tree": self._single_linkage_tree,
            "_min_spanning_tree": self._min_spanning_tree,
            "_raw_data": self._get_raw_data_cpu(),
            "_all_finite": True,
            **super()._attrs_to_cpu(model),
        }
        if self.prediction_data:
            out["_prediction_data"] = self.prediction_data_
        return out

    def __init__(
        self,
        *,
        min_cluster_size=5,
        min_samples=None,
        cluster_selection_epsilon=0.0,
        max_cluster_size=0,
        metric="euclidean",
        alpha=1.0,
        p=None,
        cluster_selection_method="eom",
        allow_single_cluster=False,
        gen_min_span_tree=False,
        verbose=False,
        output_type=None,
        prediction_data=False,
        build_algo="brute_force",
        build_kwds=None,
        device_ids=None,
    ):
        super().__init__(verbose=verbose, output_type=output_type)
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.max_cluster_size = max_cluster_size
        self.metric = metric
        self.p = p
        self.alpha = alpha
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster
        self.gen_min_span_tree = gen_min_span_tree
        self.prediction_data = prediction_data
        self.build_algo = build_algo
        self.build_kwds = build_kwds
        self.device_ids = device_ids

        self._single_linkage_tree = None
        self._min_spanning_tree = None
        self._prediction_data = None
        self._raw_data = None
        self._raw_data_cpu = None

    def _get_raw_data_cpu(self):
        if getattr(self, "_raw_data_cpu") is None:
            self._raw_data_cpu = self._raw_data.array.get(order="A")
        return self._raw_data_cpu

    @property
    def _condensed_tree(self):
        return self._state.get_condensed_tree_array()

    @property
    def condensed_tree_(self):
        if self._state is not None:
            hdbscan = import_hdbscan()
            if hdbscan_at_least("0.8.41"):
                with cuml.using_output_type("numpy"):
                    labels = self.labels_
                return hdbscan.plots.CondensedTree(self._condensed_tree, labels)
            else:
                return hdbscan.plots.CondensedTree(
                    self._condensed_tree,
                    self.cluster_selection_method,
                    self.allow_single_cluster
                )
        raise AttributeError("No condensed tree was generated; try running fit first.")

    @property
    def minimum_spanning_tree_(self):
        if self._min_spanning_tree is not None:
            hdbscan = import_hdbscan()
            return hdbscan.plots.MinimumSpanningTree(
                self._min_spanning_tree,
                self._get_raw_data_cpu(),
            )
        raise AttributeError(
            "No minimum spanning tree was generated. Set `gen_min_span_tree=True` and refit."
        )

    @property
    def single_linkage_tree_(self):
        if self._single_linkage_tree is not None:
            hdbscan = import_hdbscan()
            return hdbscan.plots.SingleLinkageTree(self._single_linkage_tree)
        raise AttributeError("No single linkage tree was generated; try running fit first.")

    @property
    def prediction_data_(self):
        if not self.prediction_data:
            raise AttributeError(
               "Prediction data not yet generated, please call "
               "`model.generate_prediction_data()`"
            )

        if self._prediction_data is None:
            hdbscan = import_hdbscan()
            self._prediction_data = hdbscan.prediction.PredictionData(
                self._get_raw_data_cpu(),
                self.condensed_tree_,
                self.min_samples or self.min_cluster_size,
                tree_type="kdtree",
                metric=self.metric
            )

        return self._prediction_data

    def generate_prediction_data(self):
        """
        Create data that caches intermediate results used for predicting
        the label of new/unseen points. This data is only useful if you
        are intending to use functions from hdbscan.prediction.
        """
        check_is_fitted(self)

        with cuml.using_output_type("cuml"):
            labels = self.labels_

        self._state.generate_prediction_data(self._raw_data.array, labels)
        self.prediction_data = True

    @generate_docstring()
    @mlfunc(set_input_type=True)
    def fit(self, X, y=None) -> "HDBSCAN":
        """
        Fit HDBSCAN model from features.
        """

        kwds = self.build_kwds or {}
        if kwds.get("knn_n_clusters", 1) > 1:
            logger.warn("Using data on host memory because knn_n_clusters > 1.")
            mem_type = "host"
        else:
            if self.build_algo == "nn_descent":
                # Keep original input data memory type to avoid unnecessary data transfer
                mem_type = None
            else:
                mem_type = "device"

        X, index = check_inputs(
            self,
            X,
            dtype="float32",
            mem_type=mem_type,
            order="C",
            ensure_min_samples=2,
            return_index=True,
            reset=True,
        )
        self._raw_data = ArrayIndexPair(X, index)
        self._raw_data_cpu = None

        # Validate and prepare hyperparameters
        if (min_samples := self.min_samples) is None:
            min_samples = self.min_cluster_size
        if not (1 <= min_samples <= 1023):
            raise ValueError(
                f"HDBSCAN requires `1 <= min_samples <= 1023`, got `{min_samples=}`"
            )

        cdef lib.HDBSCANParams params
        params.min_samples = min_samples
        params.alpha = self.alpha
        params.min_cluster_size = self.min_cluster_size
        params.max_cluster_size = self.max_cluster_size
        params.cluster_selection_epsilon = self.cluster_selection_epsilon
        params.allow_single_cluster = self.allow_single_cluster

        if self.cluster_selection_method == "eom":
            params.cluster_selection_method = lib.CLUSTER_SELECTION_METHOD.EOM
        elif self.cluster_selection_method == "leaf":
            params.cluster_selection_method = lib.CLUSTER_SELECTION_METHOD.LEAF
        else:
            raise ValueError(
                "`cluster_selection_method` must be one of {'eom', 'leaf'}, "
                f"got {self.cluster_selection_method!r}"
            )

        if self.build_algo == "brute_force":
            params.build_algo = lib.GRAPH_BUILD_ALGO.BRUTE_FORCE_KNN
            kwds = self.build_kwds or {}
            params.build_params.n_clusters = <uint64_t> kwds.get("knn_n_clusters", 1)
            params.build_params.overlap_factor = <uint64_t> kwds.get("knn_overlap_factor", 2)
        elif self.build_algo == "nn_descent":
            params.build_algo = lib.GRAPH_BUILD_ALGO.NN_DESCENT

            kwds = self.build_kwds or {}
            params.build_params.n_clusters = <uint64_t> kwds.get("knn_n_clusters", 1)
            params.build_params.overlap_factor = <uint64_t> kwds.get("knn_overlap_factor", 2)
            if (
                params.build_params.n_clusters > 1
                and params.build_params.overlap_factor >= params.build_params.n_clusters
            ):
                raise ValueError(
                    "If knn_n_clusters > 1, then knn_overlap_factor must be strictly "
                    "smaller than knn_n_clusters."
                )
            if params.build_params.n_clusters < 1:
                raise ValueError("knn_n_clusters must be >= 1")

            params.build_params.nn_descent_params.graph_degree = (
                <uint64_t> kwds.get("nnd_graph_degree", 64)
            )
            params.build_params.nn_descent_params.intermediate_graph_degree = (
                <uint64_t> kwds.get("nnd_intermediate_graph_degree", 128)
            )
            params.build_params.nn_descent_params.max_iterations = (
                <uint64_t> kwds.get("nnd_max_iterations", 20)
            )
            params.build_params.nn_descent_params.termination_threshold = (
                <float> kwds.get("nnd_termination_threshold", 0.0001)
            )

            if params.build_params.nn_descent_params.graph_degree < min_samples+1:
                logger.warn(
                    "to use nn descent as the build algo, nnd_graph_degree should be larger "
                    "than or equal to min_samples + 1. setting nnd_graph_degree to "
                    "min_samples + 1."
                )
                params.build_params.nn_descent_params.graph_degree = min_samples+1
            if (
                params.build_params.nn_descent_params.intermediate_graph_degree
                < params.build_params.nn_descent_params.graph_degree
            ):
                logger.warn(
                    "to use nn descent as the build algo, nnd_intermediate_graph_degree "
                    "should be larger than or equal to nnd_graph_degree. setting "
                    "nnd_intermediate_graph_degree to nnd_graph_degree"
                )
                params.build_params.nn_descent_params.intermediate_graph_degree = (
                    params.build_params.nn_descent_params.graph_degree
                )
        else:
            raise ValueError(
                "`build_algo` must be one of {'brute_force', 'nn_descent'}, "
                f"got {self.build_algo!r}"
            )

        cdef DistanceType metric
        if self.metric in _metrics_mapping:
            metric = _metrics_mapping[self.metric]
        else:
            raise ValueError(
                f"metric must be one of {sorted(_metrics_mapping)}, got {self.metric!r}"
            )

        # Execute fit
        (
            state,
            n_clusters,
            labels,
            probabilities,
            cluster_persistence,
            min_spanning_tree,
            single_linkage_tree,
        ) = _HDBSCANState.init_and_fit(
            X,
            params,
            metric,
            self.gen_min_span_tree,
            device_ids=self.device_ids,
        )

        # Store state on model
        self._state = state
        self.n_clusters_ = n_clusters
        self.labels_ = ArrayIndexPair(labels, index)
        self.probabilities_ = probabilities
        self.cluster_persistence_ = cluster_persistence
        self._min_spanning_tree = min_spanning_tree
        self._single_linkage_tree = single_linkage_tree

        if self.prediction_data:
            self.generate_prediction_data()

        return self

    @generate_docstring(return_values={"name": "preds",
                                       "type": "dense",
                                       "description": "Cluster indexes",
                                       "shape": "(n_samples, 1)"})
    @mlfunc(preserve_index=True)
    def fit_predict(self, X, y=None):
        """
        Fit the HDBSCAN model from features and return
        cluster labels.
        """
        return self.fit(X).labels_

    def __getstate__(self):
        out = self.__dict__.copy()
        out["_raw_data_cpu"] = None
        if (state := out.pop("_state", None)) is not None:
            out["_state_dict"] = state.to_dict()
        return out

    def __setstate__(self, state):
        state_dict = state.pop("_state_dict", None)
        self.__dict__.update(state)
        if state_dict is not None:
            self._state = _HDBSCANState.from_dict(state_dict)
        if self.prediction_data:
            self.generate_prediction_data()


###########################################################
#                  Prediction Functions                   #
###########################################################


def _check_clusterer(clusterer):
    """Validate an HDBSCAN instance is fit and has prediction data"""
    if not isinstance(clusterer, HDBSCAN):
        raise TypeError(
            f"Expected an instance of `HDBSCAN`, got {type(clusterer).__name__}"
        )

    check_is_fitted(clusterer)

    cdef _HDBSCANState state = <_HDBSCANState?>clusterer._state

    if state.prediction_data == NULL:
        raise ValueError(
            "Prediction data not yet generated, please call "
            "`clusterer.generate_prediction_data()`"
        )

    return state


@mlfunc(model_arg="clusterer", array_arg=None)
def all_points_membership_vectors(clusterer, int batch_size=4096):
    """
    Predict soft cluster membership vectors for all points in the
    original dataset the clusterer was trained on. This function is more
    efficient by making use of the fact that all points are already in the
    condensed tree, and processing in bulk.

    Parameters
    ----------
    clusterer : HDBSCAN
        A clustering object that has been fit to the data and
        had ``prediction_data=True`` set.

    batch_size : int, optional, default=min(4096, n_rows)
        Lowers memory requirement by computing distance-based membership
        in smaller batches of points in the training data. For example, a batch
        size of 1,000 computes distance based memberships for 1,000 points at a
        time. The default batch size is 4,096.

    Returns
    -------
    membership_vectors : array (n_samples, n_clusters)
        The probability that point ``i`` of the original dataset is a member of
        cluster ``j`` is in ``membership_vectors[i, j]``.
    """
    _check_clusterer(clusterer)

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    n_rows = clusterer._raw_data.array.shape[0]

    if clusterer.n_clusters_ == 0:
        return ArrayIndexPair(
            cp.zeros(n_rows, dtype=np.float32, order="C"),
            clusterer._raw_data.index,
        )

    membership_vec = cp.empty(
        (n_rows, clusterer.n_clusters_,), dtype="float32", order="C",
    )

    raw_data = clusterer._raw_data.array
    cdef _HDBSCANState state = <_HDBSCANState?>clusterer._state
    cdef float* X_ptr = <float*><uintptr_t>raw_data.data.ptr
    cdef float* membership_vec_ptr = <float*><uintptr_t>membership_vec.data.ptr
    cdef DistanceType metric = _metrics_mapping[clusterer.metric]
    handle = get_handle()
    cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

    with nogil:
        lib.compute_all_points_membership_vectors(
            handle_[0],
            deref(state.get_condensed_tree()),
            deref(state.prediction_data),
            X_ptr,
            metric,
            membership_vec_ptr,
            batch_size
        )
    handle.sync()

    return ArrayIndexPair(membership_vec, clusterer._raw_data.index)


@mlfunc(model_arg="clusterer", array_arg="points_to_predict", preserve_index=True)
def membership_vector(clusterer, points_to_predict, int batch_size=4096):
    """
    Predict soft cluster membership. The result produces a vector
    for each point in ``points_to_predict`` that gives a probability that
    the given point is a member of a cluster for each of the selected clusters
    of the ``clusterer``.

    Parameters
    ----------
    clusterer : HDBSCAN
        A clustering object that has been fit to the data and
        either had ``prediction_data=True`` set, or called the
        ``generate_prediction_data`` method after the fact.

    points_to_predict : array, or array-like (n_samples, n_features)
        The new data points to predict cluster labels for. They should
        have the same dimensionality as the original dataset over which
        clusterer was fit.

    batch_size : int, optional, default=min(4096, n_points_to_predict)
        Lowers memory requirement by computing distance-based membership
        in smaller batches of points in the prediction data. For example, a
        batch size of 1,000 computes distance based memberships for 1,000
        points at a time. The default batch size is 4,096.

    Returns
    -------
    membership_vectors : array (n_samples, n_clusters)
        The probability that point ``i`` is a member of cluster ``j`` is
        in ``membership_vectors[i, j]``.
    """
    _check_clusterer(clusterer)
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    points_to_predict = check_inputs(
        clusterer,
        points_to_predict,
        dtype="float32",
        order="C",
    )
    cdef int n_prediction_points = points_to_predict.shape[0]

    membership_vec = cp.empty(
        (n_prediction_points, clusterer.n_clusters_,), dtype="float32", order="C"
    )
    if clusterer.n_clusters_ == 0:
        return membership_vec

    raw_data = clusterer._raw_data.array
    cdef _HDBSCANState state = <_HDBSCANState?>clusterer._state
    cdef float* X_ptr = <float*><uintptr_t>raw_data.data.ptr
    cdef float* points_to_predict_ptr = <float*><uintptr_t>points_to_predict.data.ptr
    cdef float* membership_vec_ptr = <float*><uintptr_t>membership_vec.data.ptr
    cdef int min_samples = clusterer.min_samples or clusterer.min_cluster_size
    cdef DistanceType metric = _metrics_mapping[clusterer.metric]
    handle = get_handle()
    cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

    with nogil:
        lib.compute_membership_vector(
            handle_[0],
            deref(state.get_condensed_tree()),
            deref(state.prediction_data),
            X_ptr,
            points_to_predict_ptr,
            n_prediction_points,
            min_samples,
            metric,
            membership_vec_ptr,
            batch_size
        )
    handle.sync()

    return membership_vec


@mlfunc(model_arg="clusterer", array_arg="points_to_predict", preserve_index=True)
def approximate_predict(clusterer, points_to_predict):
    """Predict the cluster label of new points. The returned labels
    will be those of the original clustering found by ``clusterer``,
    and therefore are not (necessarily) the cluster labels that would
    be found by clustering the original data combined with
    ``points_to_predict``, hence the 'approximate' label.

    If you simply wish to assign new points to an existing clustering
    in the 'best' way possible, this is the function to use. If you
    want to predict how ``points_to_predict`` would cluster with
    the original data under HDBSCAN the most efficient existing approach
    is to simply recluster with the new point(s) added to the original dataset.

    Parameters
    ----------
    clusterer : HDBSCAN
        A clustering object that has been fit to the data and
        had ``prediction_data=True`` set.

    points_to_predict : array, or array-like (n_samples, n_features)
        The new data points to predict cluster labels for. They should
        have the same dimensionality as the original dataset over which
        clusterer was fit.

    Returns
    -------
    labels : array (n_samples,)
        The predicted labels of the ``points_to_predict``

    probabilities : array (n_samples,)
        The soft cluster scores for each of the ``points_to_predict``
    """
    _check_clusterer(clusterer)

    if clusterer.n_clusters_ == 0:
        logger.warn(
            "Clusterer does not have any defined clusters, new data "
            "will be automatically predicted as outliers."
        )

    points_to_predict = check_inputs(
        clusterer,
        points_to_predict,
        dtype="float32",
        order="C",
    )
    cdef int n_prediction_points = points_to_predict.shape[0]

    prediction_labels = cp.empty(n_prediction_points, dtype="int64")
    prediction_probs = cp.empty(n_prediction_points, dtype="float32")

    raw_data = clusterer._raw_data.array
    cdef _HDBSCANState state = <_HDBSCANState?>clusterer._state
    cdef float* X_ptr = <float*><uintptr_t>raw_data.data.ptr
    cdef int64_t* labels_ptr = <int64_t*><uintptr_t>clusterer.labels_.array.data.ptr
    cdef float* points_to_predict_ptr = <float*><uintptr_t>points_to_predict.data.ptr
    cdef int64_t* prediction_labels_ptr = <int64_t*><uintptr_t>prediction_labels.data.ptr
    cdef float* prediction_probs_ptr = <float*><uintptr_t>prediction_probs.data.ptr
    cdef DistanceType metric = _metrics_mapping[clusterer.metric]
    cdef int min_samples = clusterer.min_samples or clusterer.min_cluster_size,
    handle = get_handle()
    cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

    with nogil:
        lib.out_of_sample_predict(
            handle_[0],
            deref(state.get_condensed_tree()),
            deref(state.prediction_data),
            X_ptr,
            labels_ptr,
            points_to_predict_ptr,
            n_prediction_points,
            metric,
            min_samples,
            prediction_labels_ptr,
            prediction_probs_ptr,
        )
    handle.sync()

    return prediction_labels, prediction_probs


###########################################################
#           Functions exposed for testing only            #
###########################################################

def _condense_hierarchy(dendrogram, min_cluster_size):
    """
    Accepts a dendrogram in the Scipy hierarchy format, condenses the
    dendrogram to collapse subtrees containing less than min_cluster_size
    leaves, and returns a ``condensed_tree``.

    Exposed for testing only.

    Parameters
    ----------
    dendrogram : array-like (n_samples, 4)
        Dendrogram in Scipy hierarchy format.

    min_cluster_size : int
        Minimum number of children for a cluster to persist.

    Returns
    -------
    condensed_tree : np.ndarray
    """
    state = _HDBSCANState.from_dendrogram(dendrogram, min_cluster_size)
    return state.get_condensed_tree_array()


def _extract_clusters(
    condensed_tree,
    allow_single_cluster=False,
    max_cluster_size=0,
    cluster_selection_method="eom",
    cluster_selection_epsilon=0.0,
):
    """Extract clusters from a condensed_tree.

    Exposed for testing only"""
    cdef size_t n_leaves = condensed_tree["parent"].min()
    cdef int n_edges = len(condensed_tree)

    parents = cp.asarray(condensed_tree["parent"], order="C", dtype="int64")
    children = cp.asarray(condensed_tree["child"], order="C", dtype="int64")
    lambdas = cp.asarray(condensed_tree["lambda_val"], order="C", dtype="float32")
    sizes = cp.asarray(condensed_tree["child_size"], order="C", dtype="int64")

    labels = cp.empty(n_leaves, dtype="int64")
    probabilities = cp.empty(n_leaves, dtype="float32")

    cdef lib.CLUSTER_SELECTION_METHOD cluster_selection_method_val = {
        "eom": lib.CLUSTER_SELECTION_METHOD.EOM,
        "leaf": lib.CLUSTER_SELECTION_METHOD.LEAF,
    }[cluster_selection_method]

    handle = get_handle()
    cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

    lib._extract_clusters(
        handle_[0],
        n_leaves,
        n_edges,
        <int64_t*><uintptr_t>(parents.data.ptr),
        <int64_t*><uintptr_t>(children.data.ptr),
        <float*><uintptr_t>(lambdas.data.ptr),
        <int64_t*><uintptr_t>(sizes.data.ptr),
        <int64_t*><uintptr_t>(labels.data.ptr),
        <float*><uintptr_t>(probabilities.data.ptr),
        cluster_selection_method_val,
        <bool> allow_single_cluster,
        <int64_t> max_cluster_size,
        <float> cluster_selection_epsilon,
    )
    handle.sync()

    return labels.get(), probabilities.get()
