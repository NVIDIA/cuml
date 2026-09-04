#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import cupyx.scipy.sparse as cp_sp
from sklearn.base import ClassNamePrefixFeaturesOutMixin

from cuml.internals.base import Base, get_handle
from cuml.internals.interop import InteropMixin, UnsupportedOnGPU
from cuml.internals.mixins import CMajorInputTagMixin
from cuml.internals.outputs import ReflectedAttr, mlfunc
from cuml.internals.validation import check_inputs, check_random_seed

from libc.stdint cimport int64_t, uint64_t, uintptr_t
from libcpp cimport bool
from libcpp.optional cimport nullopt, optional
from pylibraft.common.cpp.mdspan cimport (
    col_major,
    device_matrix_view,
    device_vector_view,
    make_device_matrix_view,
    make_device_vector_view,
    row_major,
)
from pylibraft.common.handle cimport device_resources


cdef extern from "cuml/manifold/spectral_embedding.hpp" \
        namespace "ML::SpectralEmbedding" nogil:

    cdef cppclass params:
        int n_components
        int n_neighbors
        bool norm_laplacian
        bool drop_first
        optional[uint64_t] seed

    cdef void transform(
        const device_resources &handle,
        params config,
        device_matrix_view[float, int, row_major] dataset,
        device_matrix_view[float, int, col_major] embedding) except +

    cdef void transform(
        const device_resources &handle,
        params config,
        device_vector_view[int, int64_t] rows,
        device_vector_view[int, int64_t] cols,
        device_vector_view[float, int64_t] vals,
        device_matrix_view[float, int, col_major] embedding) except +


class SpectralEmbedding(
    InteropMixin,
    CMajorInputTagMixin,
    ClassNamePrefixFeaturesOutMixin,
    Base,
):
    """Spectral embedding for non-linear dimensionality reduction.

    Forms an affinity matrix given by the specified function and
    applies spectral decomposition to the corresponding graph laplacian.
    The resulting transformation is given by the value of the
    eigenvectors for each data point.

    Note : Laplacian Eigenmaps is the actual algorithm implemented here.

    Parameters
    ----------
    n_components : int, default=2
        The dimension of the projected subspace.
    affinity : {'nearest_neighbors', 'precomputed'}, default='nearest_neighbors'
        How to construct the affinity matrix.
         - 'nearest_neighbors' : construct the affinity matrix by computing a
           graph of nearest neighbors.
         - 'precomputed' : interpret ``X`` as a precomputed affinity matrix.
    random_state : int, RandomState instance or None, default=None
        A pseudo random number generator used for the initialization.
        Use an int to make the results deterministic across calls.
    n_neighbors : int or None, default=2
        Number of nearest neighbors for nearest_neighbors graph building.
        If None, n_neighbors will be set to max(n_samples/10, 1).
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
    embedding_ : cupy.ndarray of shape (n_samples, n_components)
        Spectral embedding of the training matrix.
    n_neighbors_ : int
        Number of nearest neighbors effectively used.

    Notes
    -----
    Spectral Embedding (Laplacian Eigenmaps) is most useful when the graph
    has one connected component. If there graph has many components, the first
    few eigenvectors will simply uncover the connected components of the graph.

    Examples
    --------
    >>> import cupy as cp
    >>> from cuml.manifold import SpectralEmbedding
    >>> X = cp.random.rand(100, 20, dtype=cp.float32)
    >>> embedding = SpectralEmbedding(n_components=2, random_state=42)
    >>> X_transformed = embedding.fit_transform(X)
    >>> X_transformed.shape
    (100, 2)
    """
    _cpu_class_path = "sklearn.manifold.SpectralEmbedding"
    embedding_ = ReflectedAttr()

    # Private so that `spectral_embedding` can share the same code
    _drop_first = True
    _norm_laplacian = True

    def __init__(
        self,
        n_components=2,
        affinity="nearest_neighbors",
        random_state=None,
        n_neighbors=None,
        verbose=False,
        output_type=None,
    ):
        super().__init__(verbose=verbose, output_type=output_type)
        self.n_components = n_components
        self.affinity = affinity
        self.random_state = random_state
        self.n_neighbors = n_neighbors

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + [
            "n_components",
            "affinity",
            "random_state",
            "n_neighbors"
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        if model.affinity not in ("nearest_neighbors", "precomputed"):
            raise UnsupportedOnGPU(f"affinity={model.affinity!r} is not supported on GPU")
        params = {
            "n_components": model.n_components,
            "affinity": model.affinity,
            "random_state": model.random_state,
            "n_neighbors": model.n_neighbors
        }
        return params

    def _params_to_cpu(self):
        params = {
            "n_components": self.n_components,
            "affinity": self.affinity,
            "random_state": self.random_state,
            "n_neighbors": self.n_neighbors
        }
        return params

    def _attrs_from_cpu(self, model):
        return {
            "n_neighbors_": model.n_neighbors_,
            "embedding_": cp.asarray(model.embedding_, order="F"),
            **super()._attrs_from_cpu(model)
        }

    def _attrs_to_cpu(self, model):
        return {
            "n_neighbors_": self.n_neighbors_,
            "embedding_": self.embedding_.get(order="F"),
            **super()._attrs_to_cpu(model),
        }

    @property
    @mlfunc(convert_output=False)
    def _n_features_out(self):
        """Number of transformed output features."""
        # Exposed to support sklearn's `get_feature_names_out`
        return self.embedding_.shape[1]

    @mlfunc(preserve_index=True, column_names="feature_names_out")
    def fit_transform(self, X, y=None):
        """Fit the model from data in X and transform X.

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
            Not used, present for API consistency by convention.

        Returns
        -------
        X_new : cupy.ndarray of shape (n_samples, n_components)
            Spectral embedding of the training matrix.
        """
        self.fit(X, y)
        return self.embedding_

    @mlfunc(set_input_type=True)
    def fit(self, X, y=None) -> "SpectralEmbedding":
        """Fit the model from data in X.

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
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = check_inputs(
            self,
            X,
            dtype="float32",
            order="C",
            accept_sparse="coo" if self.affinity == "precomputed" else False,
            ensure_min_samples=2,
            ensure_min_features=2,
            reset=True,
        )

        cdef float* affinity_data_ptr = NULL
        cdef int* affinity_rows_ptr = NULL
        cdef int* affinity_cols_ptr = NULL
        cdef int64_t affinity_nnz = 0

        if self.affinity == "nearest_neighbors":
            affinity_data_ptr = <float*><uintptr_t>X.data.ptr
        elif self.affinity == "precomputed":
            # Coerce `X` to a canonical float32 COO sparse matrix
            if isinstance(X, cp.ndarray):
                X = cp_sp.coo_matrix(X)
            X.sum_duplicates()

            if X.shape[0] != X.shape[1]:
                raise ValueError(
                    f"Expected precomputed `X` to be square, got shape = {X.shape}"
                )

            affinity_data = X.data
            affinity_rows = X.row
            affinity_cols = X.col
            affinity_nnz = X.nnz

            # laplacian kernel expects diagonal to be zero
            # remove diagonal elements since they are ignored in laplacian calculation anyway
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
            raise ValueError(
                f"`affinity={self.affinity!r}` is not supported, expected one of "
                "['nearest_neighbors', 'precomputed']"
            )

        self.n_neighbors_ = (
            self.n_neighbors
            if self.n_neighbors is not None
            else max(int(X.shape[0] / 10), 1)
        )

        cdef int n_samples = X.shape[0]
        cdef int n_features = X.shape[1]
        cdef int n_components = self.n_components

        # Allocate output array
        embedding = cp.empty((n_samples, n_components), dtype="float32", order="F")

        cdef params config
        # No seed use nullopt (non-deterministic) or set user seed (deterministic)
        if self.random_state is None:
            config.seed = nullopt
        else:
            config.seed = <uint64_t>check_random_seed(self.random_state)
        config.norm_laplacian = self._norm_laplacian
        config.drop_first = self._drop_first
        config.n_components = n_components + 1 if self._drop_first else n_components
        config.n_neighbors = self.n_neighbors_
        cdef float* embedding_ptr = <float *><uintptr_t>embedding.data.ptr
        cdef bool precomputed = self.affinity == "precomputed"
        handle = get_handle()
        cdef device_resources *handle_ = <device_resources*><size_t>handle.getHandle()

        with nogil:
            if precomputed:
                transform(
                    handle_[0],
                    config,
                    make_device_vector_view[int, int64_t](affinity_rows_ptr, affinity_nnz),
                    make_device_vector_view[int, int64_t](affinity_cols_ptr, affinity_nnz),
                    make_device_vector_view[float, int64_t](affinity_data_ptr, affinity_nnz),
                    make_device_matrix_view[float, int, col_major](
                        embedding_ptr, n_samples, n_components,
                    )
                )
            else:
                transform(
                    handle_[0],
                    config,
                    make_device_matrix_view[float, int, row_major](
                        affinity_data_ptr, n_samples, n_features,
                    ),
                    make_device_matrix_view[float, int, col_major](
                        embedding_ptr, n_samples, n_components,
                    )
                )
        handle.sync()

        self.embedding_ = embedding

        return self


@mlfunc(preserve_index=True)
def spectral_embedding(
    A,
    *,
    int n_components=8,
    affinity="nearest_neighbors",
    random_state=None,
    n_neighbors=None,
    norm_laplacian=True,
    drop_first=True,
):
    """Project the sample on the first eigenvectors of the graph Laplacian.

    The adjacency matrix is used to compute a normalized graph Laplacian
    whose spectrum (especially the eigenvectors associated to the
    smallest eigenvalues) has an interpretation in terms of minimal
    number of cuts necessary to split the graph into comparably sized
    components.

    Note : Laplacian Eigenmaps is the actual algorithm implemented here.

    Parameters
    ----------
    A : array-like or sparse matrix of shape (n_samples, n_features) or \
        (n_samples, n_samples)
        If affinity is 'nearest_neighbors', this is the input data and a k-NN
        graph will be constructed. If affinity is 'precomputed', this is the
        affinity matrix. Supported formats for precomputed affinity: scipy
        sparse (CSR, CSC, COO), cupy sparse (CSR, CSC, COO), dense numpy
        arrays, or dense cupy arrays.
    n_components : int, default=8
        The dimension of the projection subspace.
    affinity : {'nearest_neighbors', 'precomputed'}, default='nearest_neighbors'
        How to construct the affinity matrix.
         - 'nearest_neighbors' : construct the affinity matrix by computing a
           graph of nearest neighbors.
         - 'precomputed' : interpret ``A`` as a precomputed affinity matrix.
    random_state : int, RandomState instance or None, default=None
        A pseudo random number generator used for the initialization.
        Use an int to make the results deterministic across calls.
    n_neighbors : int or None, default=None
        Number of nearest neighbors for nearest_neighbors graph building.
        If None, n_neighbors will be set to max(n_samples/10, 1).
        Only used when A has shape (n_samples, n_features).
    norm_laplacian : bool, default=True
        If True, then compute symmetric normalized Laplacian.
    drop_first : bool, default=True
        Whether to drop the first eigenvector. For spectral embedding, this
        should be True as the first eigenvector should be constant vector for
        connected graph, but for spectral clustering, this should be kept as
        False to retain the first eigenvector.

    Returns
    -------
    embedding : cupy.ndarray of shape (n_samples, n_components)
        The reduced samples.

    Notes
    -----
    Spectral Embedding (Laplacian Eigenmaps) is most useful when the graph
    has one connected component. If there graph has many components, the first
    few eigenvectors will simply uncover the connected components of the graph.

    Examples
    --------
    >>> import cupy as cp
    >>> from cuml.manifold import spectral_embedding
    >>> X = cp.random.rand(100, 20, dtype=cp.float32)
    >>> embedding = spectral_embedding(X, n_components=2, random_state=42)
    >>> embedding.shape
    (100, 2)
    """
    model = SpectralEmbedding(
        n_components=n_components,
        affinity=affinity,
        random_state=random_state,
        n_neighbors=n_neighbors,
    )
    model._drop_first = drop_first
    model._norm_laplacian = norm_laplacian
    return model.fit_transform(A)
