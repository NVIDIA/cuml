#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import cupyx.scipy.sparse as sp

from cuml.internals.base import Base, get_handle
from cuml.internals.interop import InteropMixin, UnsupportedOnGPU
from cuml.internals.mixins import ClusterMixin, CMajorInputTagMixin
from cuml.internals.outputs import ReflectedAttr, mlfunc
from cuml.internals.validation import check_inputs, check_random_seed

from libc.stdint cimport uint64_t, uintptr_t
from libcpp cimport bool
from pylibraft.common.cpp.mdspan cimport (
    device_matrix_view,
    device_vector_view,
    make_device_matrix_view,
    make_device_vector_view,
    row_major,
)
from pylibraft.common.handle cimport device_resources


cdef extern from "cuml/cluster/spectral_clustering.hpp" \
        namespace "ML::SpectralClustering" nogil:

    cdef cppclass params:
        int n_clusters
        int n_components
        int n_neighbors
        int n_init
        float eigen_tol
        uint64_t seed

    cdef void fit_predict(
        const device_resources &handle,
        params config,
        device_matrix_view[float, int, row_major] dataset,
        device_vector_view[int, int] labels) except +

    cdef void fit_predict(
        const device_resources &handle,
        params config,
        device_vector_view[int, int] rows,
        device_vector_view[int, int] cols,
        device_vector_view[float, int] vals,
        device_vector_view[int, int] labels) except +


_AFFINITIES = ["nearest_neighbors", "precomputed"]


class SpectralClustering(InteropMixin,
                         ClusterMixin,
                         CMajorInputTagMixin,
                         Base):
    """Apply spectral clustering from the normalized Laplacian.

    In practice spectral clustering is very useful when the structure of
    the individual clusters is highly non-convex, or when a measure of
    the center and spread of the cluster is not a suitable description
    of the complete cluster, such as when clusters are nested circles on
    the 2D plane.

    If the affinity matrix is the adjacency matrix of a graph, this method
    can be used to find normalized graph cuts.

    When calling ``fit``, an affinity matrix is constructed using a
    k-nearest neighbors connectivity matrix.

    Alternatively, a user-provided affinity matrix can be specified by
    setting ``affinity='precomputed'``.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form.
    n_components : int or None, default=None
        Number of eigenvectors to use for the spectral embedding. If None,
        defaults to n_clusters.
    random_state : int, RandomState instance or None, default=None
        A pseudo random number generator used for the initialization of the
        k-means clustering and the eigendecomposition. Use an int to make the
        results deterministic across calls.
    n_neighbors : int, default=10
        Number of neighbors to use when constructing the affinity matrix using
        the nearest neighbors method. Ignored for ``affinity='precomputed'``.
    n_init : int, default=10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of n_init
        consecutive runs in terms of inertia. Only used for the k-means step.
    eigen_tol : float or 'auto', default='auto'
        Tolerance for the eigensolver. If 'auto', a tolerance values of
        0.0 is used.
    affinity : {'nearest_neighbors', 'precomputed'}, default='nearest_neighbors'
        How to construct the affinity matrix.
         - 'nearest_neighbors' : construct the affinity matrix by computing a
           graph of nearest neighbors from the input data.
         - 'precomputed' : interpret X as a precomputed affinity matrix,
           where larger values indicate greater similarity between instances.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {None, 'input', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used.
        See :ref:`output-data-type-configuration` for more info.

    Attributes
    ----------
    labels_ : cupy.ndarray or np.ndarray of shape (n_samples,)
        Cluster labels for each sample.

    Examples
    --------
    >>> import cupy as cp
    >>> from sklearn.datasets import make_blobs
    >>> from cuml.cluster import SpectralClustering
    >>> X, y = make_blobs(n_samples=100, centers=3, n_features=10,
    ...                   cluster_std=0.5, random_state=42)
    >>> X = cp.asarray(X, dtype=cp.float32)
    >>> sc = SpectralClustering(n_clusters=3, affinity='nearest_neighbors',
    ...                         n_neighbors=10, random_state=42)
    >>> sc.fit(X)
    SpectralClustering()
    >>> sc.labels_[:10]
    array([2, 0, 1, 1, 2, 2, 1, 0, 2, 0])

    Notes
    -----
    The eigensolver uses the Lanczos approach from the raft implementation
    https://docs.nvidia.com/raft/latest/pylibraft_api/sparse/#pylibraft.sparse.linalg.eigsh.

    Kmeans is used for assigning labels.

    References
    ----------
    - Normalized cuts and image segmentation, 2000
      Jianbo Shi, Jitendra Malik
      http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.160.2324

    - A Tutorial on Spectral Clustering, 2007
      Ulrike von Luxburg
      http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.165.9323

    - Multiclass spectral clustering, 2003
      Stella X. Yu, Jianbo Shi
      https://www1.icsi.berkeley.edu/~stellayu/publication/doc/2003kwayICCV.pdf
    """
    labels_ = ReflectedAttr()

    _cpu_class_path = "sklearn.cluster.SpectralClustering"

    _SUPPORTED_AFFINITIES = frozenset(("nearest_neighbors", "precomputed"))

    @classmethod
    def _params_from_cpu(cls, model):
        if model.affinity not in _AFFINITIES:
            raise UnsupportedOnGPU(
                f"`affinity={model.affinity!r}` is not supported"
            )
        if model.assign_labels != "kmeans":
            raise UnsupportedOnGPU(
                f"`assign_labels={model.assign_labels!r}` is not supported"
            )
        return {
            "n_clusters": model.n_clusters,
            "n_components": model.n_components,
            "random_state": model.random_state,
            "n_neighbors": model.n_neighbors,
            "n_init": model.n_init,
            "eigen_tol": model.eigen_tol,
            "affinity": model.affinity,
        }

    def _params_to_cpu(self):
        return {
            "n_clusters": self.n_clusters,
            "n_components": self.n_components,
            "random_state": self.random_state,
            "n_neighbors": self.n_neighbors,
            "n_init": self.n_init,
            "eigen_tol": self.eigen_tol,
            "affinity": self.affinity,
            "assign_labels": "kmeans",
        }

    def _attrs_from_cpu(self, model):
        return {
            "labels_": cp.asarray(model.labels_, order="C"),
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        return {
            "labels_": self.labels_.get(order="C"),
            **super()._attrs_to_cpu(model),
        }

    def __init__(
        self,
        n_clusters=8,
        *,
        n_components=None,
        random_state=None,
        n_neighbors=10,
        n_init=10,
        eigen_tol="auto",
        affinity="nearest_neighbors",
        verbose=False,
        output_type=None,
    ):
        super().__init__(verbose=verbose, output_type=output_type)
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.n_init = n_init
        self.eigen_tol = eigen_tol
        self.affinity = affinity

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "n_clusters",
            "n_components",
            "random_state",
            "n_neighbors",
            "n_init",
            "eigen_tol",
            "affinity",
        ]

    @mlfunc(preserve_index=True)
    def fit_predict(self, X, y=None):
        """Perform spectral clustering on ``X`` and return cluster labels.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features) or \
            (n_samples, n_samples)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features. If affinity is
            'precomputed', X is the affinity matrix. Supported formats for
            precomputed affinity: scipy sparse (CSR, CSC, COO), cupy sparse
            (CSR, CSC, COO), dense numpy arrays, or dense cupy arrays.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : cupy.ndarray of shape (n_samples,)
            Cluster labels.
        """
        self.fit(X, y=y)
        return self.labels_

    @mlfunc(set_input_type=True)
    def fit(self, X, y=None) -> "SpectralClustering":
        """Perform spectral clustering on ``X``.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features) or \
            (n_samples, n_samples)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features. If affinity is
            'precomputed', X is the affinity matrix. Supported formats for
            precomputed affinity: scipy sparse (CSR, CSC, COO), cupy sparse
            (CSR, CSC, COO), dense numpy arrays, or dense cupy arrays.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if self.affinity not in _AFFINITIES:
            raise ValueError(
                f"Expected `affinity` to be one of {_AFFINITIES}, got {self.affinity!r}"
            )
        cdef bool precomputed = self.affinity == "precomputed"

        X = check_inputs(
            self,
            X,
            dtype="float32",
            order="C",
            accept_sparse=("coo" if precomputed else False),
            ensure_min_samples=2,
            reset=True,
        )
        cdef int n_samples, n_features
        n_samples, n_features = X.shape

        cdef float* affinity_data_ptr = NULL
        cdef int* affinity_rows_ptr = NULL
        cdef int* affinity_cols_ptr = NULL
        cdef int affinity_nnz = 0

        if precomputed:
            if n_samples != n_features:
                raise ValueError(
                    f"X must be 2-dimensional and square, (got shape={X.shape})"
                )
            if not sp.issparse(X):
                X = sp.coo_matrix(X)
            X.sum_duplicates()

            affinity_data = X.data
            affinity_rows = X.row.astype(cp.int32)
            affinity_cols = X.col.astype(cp.int32)
            affinity_nnz = X.nnz

            # Remove diagonal elements
            valid = affinity_rows != affinity_cols
            if not valid.all():
                affinity_data = affinity_data[valid]
                affinity_rows = affinity_rows[valid]
                affinity_cols = affinity_cols[valid]
                affinity_nnz = len(affinity_data)

            affinity_data_ptr = <float*><uintptr_t>affinity_data.data.ptr
            affinity_rows_ptr = <int*><uintptr_t>affinity_rows.data.ptr
            affinity_cols_ptr = <int*><uintptr_t>affinity_cols.data.ptr
        else:
            affinity_data_ptr = <float*><uintptr_t>X.data.ptr

        # Allocate output array
        labels = cp.empty(n_samples, dtype=cp.int32)

        cdef params config
        config.seed = check_random_seed(self.random_state)
        config.n_clusters = self.n_clusters
        cdef int effective_n_components = (
            self.n_components if self.n_components is not None else self.n_clusters
        )
        config.n_components = max(1, min(effective_n_components, (n_samples - 1) // 3))
        config.n_neighbors = min(self.n_neighbors, n_samples - 1)
        config.n_init = self.n_init
        if self.eigen_tol == "auto":
            config.eigen_tol = 0.0
        else:
            config.eigen_tol = self.eigen_tol

        cdef int* labels_ptr = <int*><uintptr_t>labels.data.ptr
        handle = get_handle()
        cdef device_resources *handle_ = <device_resources*><size_t>handle.getHandle()

        try:
            with nogil:
                if precomputed:
                    fit_predict(
                        handle_[0],
                        config,
                        make_device_vector_view(affinity_rows_ptr, affinity_nnz),
                        make_device_vector_view(affinity_cols_ptr, affinity_nnz),
                        make_device_vector_view(affinity_data_ptr, affinity_nnz),
                        make_device_vector_view(labels_ptr, n_samples)
                    )
                else:
                    fit_predict(
                        handle_[0],
                        config,
                        make_device_matrix_view[float, int, row_major](
                            affinity_data_ptr, n_samples, n_features
                        ),
                        make_device_vector_view(labels_ptr, n_samples)
                    )
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "eigensolver couldn't converge" in error_msg:
                raise RuntimeError(
                    "Spectral clustering failed to converge. "
                    "Ensure the input data has clear clustering structure."
                ) from None
            else:
                raise

        self.labels_ = labels
        return self


@mlfunc(preserve_index=True)
def spectral_clustering(
    X,
    *,
    int n_clusters=8,
    random_state=None,
    n_components=None,
    n_neighbors=10,
    n_init=10,
    eigen_tol="auto",
    affinity="nearest_neighbors",
):
    """Apply clustering to a projection of the normalized Laplacian.

    In practice Spectral Clustering is very useful when the structure of
    the individual clusters is highly non-convex or more generally when
    a measure of the center and spread of the cluster is not a suitable
    description of the complete cluster. For instance, when clusters are
    nested circles on the 2D plane.

    If affinity is the adjacency matrix of a graph, this method can be
    used to find normalized graph cuts.

    Parameters
    ----------
    X : array-like or sparse matrix of shape (n_samples, n_features) or \
        (n_samples, n_samples)
        If affinity is 'nearest_neighbors', this is the input data and a k-NN
        graph will be constructed. If affinity is 'precomputed', this is the
        affinity matrix. Supported formats for precomputed affinity: scipy
        sparse (CSR, CSC, COO), cupy sparse (CSR, CSC, COO), dense numpy
        arrays, or dense cupy arrays.
    n_clusters : int, default=8
        The number of clusters to form.
    random_state : int, RandomState instance or None, default=None
        A pseudo random number generator used for the initialization of the
        k-means clustering and the eigendecomposition. Use an int to make the
        results deterministic across calls.
    n_components : int or None, default=None
        Number of eigenvectors to use for the spectral embedding. If None,
        defaults to n_clusters.
    n_neighbors : int, default=10
        Number of nearest neighbors for nearest_neighbors graph building.
        Only used when affinity='nearest_neighbors'.
    n_init : int, default=10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of n_init
        consecutive runs in terms of inertia.
    eigen_tol : float or 'auto', default='auto'
        Convergence tolerance passed to the eigensolver. If set to 'auto',
        a default value of currently 0.0 will be used.
    affinity : {'nearest_neighbors', 'precomputed'}, default='nearest_neighbors'
        How to construct the affinity matrix.
         - 'nearest_neighbors' : construct the affinity matrix by computing a
           graph of nearest neighbors.
         - 'precomputed' : interpret ``A`` as a precomputed affinity matrix.

    Returns
    -------
    labels : cupy.ndarray or np.ndarray of shape (n_samples,)
        Cluster labels for each sample.

    Notes
    -----
    The graph should contain only one connected component, otherwise the
    results make little sense.

    This algorithm solves the normalized cut for k=2: it is a normalized
    spectral clustering.

    Examples
    --------
    >>> import numpy as np
    >>> from cuml.cluster import spectral_clustering
    >>> X = np.random.rand(100, 10).astype(np.float32)
    >>> labels = spectral_clustering(X, n_clusters=5, n_neighbors=10, random_state=42)
    """
    return SpectralClustering(
        n_clusters=n_clusters,
        random_state=random_state,
        n_components=n_components,
        n_neighbors=n_neighbors,
        n_init=n_init,
        eigen_tol=eigen_tol,
        affinity=affinity,
    ).fit_predict(X)
