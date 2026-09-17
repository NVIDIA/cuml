# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import warnings

import cupy as cp
from sklearn.base import ClassNamePrefixFeaturesOutMixin

from cuml.common.doc_utils import generate_docstring
from cuml.common.sparse import is_sparse
from cuml.internals.base import Base, get_handle
from cuml.internals.interop import InteropMixin, UnsupportedOnGPU
from cuml.internals.mixins import CMajorInputTagMixin, SparseInputTagMixin
from cuml.internals.outputs import ArrayIndexPair, ReflectedAttr, mlfunc
from cuml.internals.validation import check_inputs, check_random_seed
from cuml.manifold.utils import extract_knn_graph

from libc.stdint cimport int64_t, uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t

from cuml.internals cimport logger
from cuml.metrics.distance_type cimport DistanceType


cdef extern from "cuml/manifold/tsne.h" namespace "ML" nogil:

    enum TSNE_ALGORITHM:
        EXACT = 0,
        BARNES_HUT = 1,
        FFT = 2

    enum TSNE_INIT:
        RANDOM = 0,
        PCA = 1

    cdef cppclass TSNEParams:
        int dim,
        int n_neighbors,
        float theta,
        float epssq,
        float perplexity,
        int perplexity_max_iter,
        float perplexity_tol,
        float early_exaggeration,
        float late_exaggeration,
        int exaggeration_iter,
        float min_gain,
        float pre_learning_rate,
        float post_learning_rate,
        int max_iter,
        float min_grad_norm,
        float pre_momentum,
        float post_momentum,
        long long random_state,
        logger.level_enum verbosity,
        TSNE_INIT init,
        bool square_distances,
        DistanceType metric,
        float p,
        TSNE_ALGORITHM algorithm


cdef extern from "cuml/manifold/tsne.h" namespace "ML" nogil:

    cdef void TSNE_fit(
        handle_t &handle,
        float *X,
        float *Y,
        int n,
        int p,
        int64_t* knn_indices,
        float* knn_dists,
        TSNEParams &params,
        float* kl_div,
        int* n_iter) except +

    cdef void TSNE_fit_sparse(
        const handle_t &handle,
        int *indptr,
        int *indices,
        float *data,
        float *Y,
        int nnz,
        int n,
        int p,
        int* knn_indices,
        float* knn_dists,
        TSNEParams &params,
        float* kl_div,
        int* n_iter) except +


_SUPPORTED_METRICS = {
    "l2": DistanceType.L2SqrtExpanded,
    "euclidean": DistanceType.L2SqrtExpanded,
    "sqeuclidean": DistanceType.L2Expanded,
    "cityblock": DistanceType.L1,
    "l1": DistanceType.L1,
    "manhattan": DistanceType.L1,
    "minkowski": DistanceType.LpUnexpanded,
    "chebyshev": DistanceType.Linf,
    "cosine": DistanceType.CosineExpanded,
    "correlation": DistanceType.CorrelationExpanded
}

_SUPPORTED_METHODS = {
    "barnes_hut": TSNE_ALGORITHM.BARNES_HUT,
    "exact": TSNE_ALGORITHM.EXACT,
    "fft": TSNE_ALGORITHM.FFT,
}

_SUPPORTED_INITS = {
    "random": TSNE_INIT.RANDOM,
    "pca": TSNE_INIT.PCA,
}


def _check_numeric(estimator, name, gt=None, ge=None, lt=None, le=None):
    """Check that a numeric parameter `name` is within valid bounds"""
    value = getattr(estimator, name)
    cls_name = type(estimator).__name__
    if gt is not None and value <= gt:
        raise ValueError(f"{cls_name} requires `{name} > {gt}`, got {value}")
    if ge is not None and value < ge:
        raise ValueError(f"{cls_name} requires `{name} >= {ge}`, got {value}")
    if lt is not None and value >= lt:
        raise ValueError(f"{cls_name} requires `{name} < {lt}`, got {value}")
    if le is not None and value > le:
        raise ValueError(f"{cls_name} requires `{name} <= {le}`, got {value}")
    return value


def _check_mapping(estimator, name, mapping):
    """Check that a parameter `name` contained within a valid mapping"""
    value = getattr(estimator, name)
    cls_name = type(estimator).__name__
    try:
        return mapping[value]
    except KeyError:
        raise ValueError(
            f"{cls_name} expects `{name}` to be one of {sorted(mapping)}, got {value}"
        ) from None


cdef _init_params(self, int n_samples, TSNEParams &params):
    """Validate TSNE parameters and initialize a TSNEParams instance."""
    if (n_components := self.n_components) != 2:
        raise ValueError(
            f"Currently TSNE only supports n_components = 2, got {self.n_components}"
        )
    perplexity = _check_numeric(self, "perplexity", gt=0)
    early_exaggeration = _check_numeric(self, "early_exaggeration", ge=1.0)
    late_exaggeration = _check_numeric(self, "late_exaggeration", ge=1.0)
    learning_rate = _check_numeric(self, "learning_rate", gt=0)
    adaptive_learning = _check_mapping(
        self, "learning_rate_method", {"adaptive": True, "none": False, None: False}
    )
    min_grad_norm = _check_numeric(self, "min_grad_norm", ge=0)
    angle = _check_numeric(self, "angle", ge=0, le=1)
    n_neighbors = _check_numeric(self, "n_neighbors", gt=0)
    perplexity_max_iter = _check_numeric(self, "perplexity_max_iter", ge=0)
    exaggeration_iter = _check_numeric(self, "exaggeration_iter", ge=0)
    pre_momentum = _check_numeric(self, "pre_momentum", gt=0, lt=1)
    post_momentum = _check_numeric(self, "post_momentum", gt=0, lt=1)
    init = _check_mapping(self, "init", _SUPPORTED_INITS)
    algo = _check_mapping(self, "method", _SUPPORTED_METHODS)
    metric = _check_mapping(self, "metric", _SUPPORTED_METRICS)

    if n_samples < 2:
        raise ValueError("TSNE requires >= 2 samples")

    max_iter = _check_numeric(self, "max_iter", gt=0)

    exaggeration_iter = min(exaggeration_iter, max_iter)
    if n_neighbors > 1023:
        warnings.warn(
            f"n_neighbors ({n_neighbors}) should be < 1024, "
            "thresholding n_neighbors to 1023"
        )
        n_neighbors = 1023
    n_neighbors = min(n_neighbors, n_samples)

    if perplexity > n_samples:
        warnings.warn(
            f"perplexity ({perplexity}) should be less than n_samples, "
            f"thresholding perplexity to {n_samples}"
        )
        perplexity = n_samples

    if adaptive_learning and algo is not TSNE_ALGORITHM.EXACT:
        # Adjust parameters when using adaptive learning
        if n_samples <= 2000:
            n_neighbors = min(max(n_neighbors, 90), n_samples)
        else:
            # A linear trend from (n=2000, neigh=100) to (n=60000,neigh=30)
            n_neighbors = max(int(102 - 0.0012 * n_samples), 30)

        pre_learning_rate = max(n_samples / 3.0, 1)
        post_learning_rate = pre_learning_rate
        early_exaggeration = 24.0 if n_samples > 10000 else 12.0
    else:
        pre_learning_rate = learning_rate
        post_learning_rate = learning_rate * 2

    cdef long long seed = (
        -1 if self.random_state is None
        else check_random_seed(self.random_state)
    )

    params.dim = n_components
    params.n_neighbors = n_neighbors
    params.theta = angle
    params.epssq = 0.0025
    params.perplexity = perplexity
    params.perplexity_max_iter = perplexity_max_iter
    params.perplexity_tol = 1e-5
    params.early_exaggeration = early_exaggeration
    params.late_exaggeration = late_exaggeration
    params.exaggeration_iter = exaggeration_iter
    params.min_gain = 0.01
    params.pre_learning_rate = pre_learning_rate
    params.post_learning_rate = post_learning_rate
    params.max_iter = max_iter
    params.min_grad_norm = min_grad_norm
    params.pre_momentum = pre_momentum
    params.post_momentum = post_momentum
    params.random_state = seed
    params.verbosity = self._verbose_level
    params.square_distances = self.square_distances
    params.algorithm = algo
    params.init = init
    params.metric = metric
    params.p = (self.metric_params or {}).get("p", 2.0)


class TSNE(InteropMixin,
           CMajorInputTagMixin,
           SparseInputTagMixin,
           ClassNamePrefixFeaturesOutMixin,
           Base):
    """
    t-SNE (T-Distributed Stochastic Neighbor Embedding) is an extremely
    powerful dimensionality reduction technique that aims to maintain
    local distances between data points. It is extremely robust to whatever
    dataset you give it, and is used in many areas including cancer research,
    music analysis and neural network weight visualizations.

    cuML's t-SNE supports three algorithms: the original exact algorithm, the
    Barnes-Hut approximation and the fast Fourier transform interpolation
    approximation. The latter two are derived from CannyLabs' open-source CUDA
    code and produce extremely fast embeddings when n_components = 2. The exact
    algorithm is more accurate, but too slow to use on large datasets.

    Parameters
    ----------
    n_components : int (default 2)
        The output dimensionality size. Currently only 2 is supported.
    perplexity : float (default 30.0)
        Larger datasets require a larger value. Consider choosing different
        perplexity values from 5 to 50 and see the output differences.
    early_exaggeration : float (default 12.0)
        Controls the space between clusters. Not critical to tune this.
    late_exaggeration : float (default 1.0)
        Controls the space between clusters. It may be beneficial to increase
        this slightly to improve cluster separation. This will be applied
        after `exaggeration_iter` iterations (FFT only).
    learning_rate : float (default 200.0)
        The learning rate usually between (10, 1000). If this is too high,
        t-SNE could look like a cloud / ball of points.
    max_iter : int (default 1000)
        The more epochs, the more stable/accurate the final embedding.
    n_iter_without_progress : int (default 300)
        Currently unused. When the KL Divergence becomes too small after some
        iterations, terminate t-SNE early.
    min_grad_norm : float (default 1e-07)
        The minimum gradient norm for when t-SNE will terminate early.
        Used in the 'exact' and 'fft' algorithms. Consider reducing if
        the embeddings are unsatisfactory. It's recommended to use a
        smaller value for smaller datasets.
    metric : str (default='euclidean').
        Distance metric to use. Supported distances are ['l1, 'cityblock',
        'manhattan', 'euclidean', 'l2', 'sqeuclidean', 'minkowski',
        'chebyshev', 'cosine', 'correlation']
    init : str 'random' or 'pca' (default 'random')
        Currently supports random or pca initialization.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    random_state : int (default None)
        Controls random initialization. For the FFT method, setting this value
        makes repeated runs with the same inputs deterministic.
    method : str 'fft', 'barnes_hut' or 'exact' (default 'fft')
        'barnes_hut' and 'fft' are fast approximations. 'exact' is more
        accurate but slower.
    angle : float (default 0.5)
        Valid values are between 0.0 and 1.0, which trade off speed and
        accuracy, respectively. Generally, these values are set between 0.2 and
        0.8. (Barnes-Hut only.)
    learning_rate_method : str 'adaptive', 'none' or None (default 'adaptive')
        Either adaptive or None. 'adaptive' tunes the learning rate, early
        exaggeration, perplexity and n_neighbors automatically based on
        input size.
    n_neighbors : int (default 90)
        The number of datapoints you want to use in the
        attractive forces. Smaller values are better for preserving
        local structure, whilst larger values can improve global structure
        preservation. Default is 3 * 30 (perplexity)
    perplexity_max_iter : int (default 100)
        The number of epochs the best gaussian bands are found for.
    exaggeration_iter : int (default 250)
        To promote the growth of clusters, set this higher.
    pre_momentum : float (default 0.5)
        During the exaggeration iteration, more forcefully apply gradients.
    post_momentum : float (default 0.8)
        During the late phases, less forcefully apply gradients.
    square_distances : boolean, default=True
        Whether TSNE should square the distance values.
        Internally, this will be used to compute a kNN graph using the provided
        metric and then squaring it when True. If a `knn_graph` is passed
        to `fit` or `fit_transform` methods, all the distances will be
        squared when True. For example, if a `knn_graph` was obtained using
        'sqeuclidean' metric, the distances will still be squared when True.
        Note: This argument should likely be set to False for distance metrics
        other than 'euclidean' and 'l2'.
    precomputed_knn : tuple[array, array], sparse-matrix, array, optional
        This feature allows the precomputation of the KNN outside of TSNE.
        Options are:

        - A tuple (indices, distances) of dense arrays of shape (n_samples,
          n_neighbors), where n_neighbors is >= the ``n_neighbors`` parameter.
          Self references should be included (i.e. the first column of
          `indices` should be [0, 1, ...], denotating that the nearest neighbor
          to each row is itself). This is the most efficient representation.

        - A sparse matrix KNN graph, as may be output by
          ``cuml.neighbors.kneighbors_graph`` with ``mode="distance"`` and
          ``include_self=True``. The ``n_neighbors`` used to calculate the
          graph must be >= the ``n_neighbors`` parameter.

        - A pairwise distances dense array of shape (n_samples, n_samples).

        In all cases the KNN should be computed using the same ``metric`` as
        provided to ``TSNE``.

    output_type : {None, 'input', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Attributes
    ----------
    embedding_ : array
        Stores the embedding vectors.
    kl_divergence_ : float
        Kullback-Leibler divergence after optimization. An experimental
        feature at this time.
    learning_rate_ : float
        Effective learning rate.
    n_iter_ : int
        Number of iterations run.

    References
    ----------
    .. [1] `van der Maaten, L.J.P.
       t-Distributed Stochastic Neighbor Embedding
       <https://lvdmaaten.github.io/tsne/>`_

    .. [2] van der Maaten, L.J.P.; Hinton, G.E.
       Visualizing High-Dimensional Data
       Using t-SNE. Journal of Machine Learning Research 9:2579-2605, 2008.

    .. [3] George C. Linderman, Manas Rachh, Jeremy G. Hoskins,
        Stefan Steinerberger, Yuval Kluger Efficient Algorithms for
        t-distributed Stochastic Neighborhood Embedding

    .. tip::
        Maaten and Linderman showcased how t-SNE can be very sensitive to both
        the starting conditions (i.e. random initialization) and its
        hyperparameters. You can run t-SNE with different random seeds to
        compare embeddings and settle on the best configuration. For the FFT
        method, reusing the same `random_state` and inputs reproduces the same
        result.

    .. note::
        The CUDA implementation is derived from the excellent CannyLabs open
        source implementation here: https://github.com/CannyLab/tsne-cuda/. The
        CannyLabs code is licensed according to the conditions in
        cuml/cpp/src/tsne/cannylabs_tsne_license.txt. A full description of
        their approach is available in their article t-SNE-CUDA:
        GPU-Accelerated t-SNE and its Applications to Modern Data
        (https://arxiv.org/abs/1807.11824).

    """
    embedding_ = ReflectedAttr()

    _cpu_class_path = "sklearn.manifold.TSNE"

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + [
            "n_components",
            "perplexity",
            "early_exaggeration",
            "late_exaggeration",
            "learning_rate",
            "max_iter",
            "n_iter_without_progress",
            "min_grad_norm",
            "metric",
            "metric_params",
            "init",
            "random_state",
            "method",
            "angle",
            "learning_rate_method",
            "n_neighbors",
            "perplexity_max_iter",
            "exaggeration_iter",
            "pre_momentum",
            "post_momentum",
            "square_distances",
            "precomputed_knn"
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        if model.n_components != 2:
            raise UnsupportedOnGPU("Only `n_components=2` is supported")

        # Our barnes_hut implementation can sometimes hang, see #3865 and #3360.
        # fft should be at least as good, and doesn't have this issue.
        method = {"exact": "exact", "barnes_hut": "fft"}.get(model.method, None)
        if method is None:
            raise UnsupportedOnGPU(f"`method={model.method!r}` is not supported")

        if not (isinstance(model.init, str) and model.init in _SUPPORTED_INITS):
            raise UnsupportedOnGPU(f"`init={model.init!r}` is not supported")

        if not (isinstance(model.metric, str) and model.metric in _SUPPORTED_METRICS):
            raise UnsupportedOnGPU(f"`metric={model.metric!r}` is not supported")

        params = {
            "n_components": model.n_components,
            "perplexity": model.perplexity,
            "early_exaggeration": model.early_exaggeration,
            "n_iter_without_progress": model.n_iter_without_progress,
            "min_grad_norm": model.min_grad_norm,
            "metric": model.metric,
            "metric_params": model.metric_params,
            "init": model.init,
            "random_state": model.random_state,
            "method": method,
        }
        if model.learning_rate != "auto":
            params["learning_rate"] = model.learning_rate
            params["learning_rate_method"] = "none"

        if (max_iter := getattr(model, "max_iter", None)) is not None:
            params["max_iter"] = max_iter

        return params

    def _params_to_cpu(self):
        method = "exact" if self.method == "Exact" else "barnes_hut"

        params = {
            "n_components": self.n_components,
            "perplexity": self.perplexity,
            "early_exaggeration": self.early_exaggeration,
            "learning_rate": self.learning_rate,
            "n_iter_without_progress": self.n_iter_without_progress,
            "min_grad_norm": self.min_grad_norm,
            "metric": self.metric,
            "metric_params": self.metric_params,
            "init": self.init,
            "random_state": self.random_state,
            "method": method,
            "max_iter": self.max_iter,
        }
        return params

    def _attrs_from_cpu(self, model):
        return {
            "embedding_": ArrayIndexPair(cp.asarray(model.embedding_, order="F"), None),
            "kl_divergence_": model.kl_divergence_,
            "learning_rate_": model.learning_rate_,
            "n_iter_": model.n_iter_,
            **super()._attrs_from_cpu(model)
        }

    def _attrs_to_cpu(self, model):
        return {
            "embedding_": self.embedding_.array.get(order="A"),
            "kl_divergence_": self.kl_divergence_,
            "learning_rate_": self.learning_rate_,
            "n_iter_": self.n_iter_,
            **super()._attrs_to_cpu(model)
        }

    def __init__(
        self,
        *,
        n_components=2,
        perplexity=30.0,
        early_exaggeration=12.0,
        late_exaggeration=1.0,
        learning_rate=200.0,
        max_iter=1000,
        n_iter_without_progress=300,
        min_grad_norm=1e-07,
        metric='euclidean',
        metric_params=None,
        init='random',
        random_state=None,
        method='fft',
        angle=0.5,
        n_neighbors=90,
        perplexity_max_iter=100,
        exaggeration_iter=250,
        pre_momentum=0.5,
        post_momentum=0.8,
        learning_rate_method='adaptive',
        square_distances=True,
        precomputed_knn=None,
        verbose=False,
        output_type=None,
    ):
        super().__init__(verbose=verbose, output_type=output_type)
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.late_exaggeration = late_exaggeration
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.metric_params = metric_params
        self.init = init
        self.random_state = random_state
        self.method = method
        self.angle = angle
        self.n_neighbors = n_neighbors
        self.perplexity_max_iter = perplexity_max_iter
        self.exaggeration_iter = exaggeration_iter
        self.pre_momentum = pre_momentum
        self.post_momentum = post_momentum
        self.learning_rate_method = learning_rate_method
        self.square_distances = square_distances
        self.precomputed_knn = precomputed_knn

    @property
    @mlfunc(convert_output=False)
    def _n_features_out(self):
        """Number of transformed output features."""
        # Exposed to support sklearn's `get_feature_names_out`
        return self.embedding_.array.shape[1]

    @generate_docstring(skip_parameters_heading=True,
                        X='dense_sparse')
    @mlfunc(set_input_type=True)
    def fit(self, X, y=None, *, knn_graph=None) -> "TSNE":
        """
        Fit X into an embedded space.

        Parameters
        ----------
        knn_graph : tuple[array, array], sparse-matrix, array, optional
            This feature allows the precomputation of the KNN outside of TSNE.

            This may take any of the valid forms accepted by the
            ``precomputed_knn`` parameter to ``TSNE``, and takes precedence
            over it. See the ``TSNE`` docstring on ``precomputed_knn`` for more
            information.
        """
        cdef int n_samples, n_features
        cdef uintptr_t X_ptr = 0
        cdef uintptr_t X_indptr_ptr = 0
        cdef uintptr_t X_indices_ptr = 0
        cdef int X_nnz = 0
        cdef bool sparse_fit = is_sparse(X)

        X, index = check_inputs(
            self,
            X,
            dtype="float32",
            order="F",
            accept_sparse="csr",
            ensure_min_samples=2,
            ensure_min_features=2,
            reset=True,
            return_index=True,
        )

        if sparse_fit:
            n_samples, n_features = X.shape
            X_ptr = <uintptr_t>X.data.data.ptr
            X_indptr_ptr = <uintptr_t>X.indptr.data.ptr
            X_indices_ptr = <uintptr_t>X.indices.data.ptr
            X_nnz = X.nnz
        else:
            n_samples, n_features = X.shape
            X_ptr = <uintptr_t>X.data.ptr

        # Initialize TSNEParams
        cdef TSNEParams params
        _init_params(self, n_samples, params)

        # Normalize precomputed knn graph if provided
        cdef uintptr_t knn_dists_ptr = 0
        cdef uintptr_t knn_indices_ptr = 0
        if knn_graph is None:
            knn_graph = self.precomputed_knn
        if knn_graph is not None:
            knn_indices, knn_dists = extract_knn_graph(
                knn_graph,
                n_samples,
                params.n_neighbors,
                indices_dtype="int32" if sparse_fit else "int64",
            )
            knn_dists_ptr = <uintptr_t>knn_dists.data.ptr
            knn_indices_ptr = <uintptr_t>knn_indices.data.ptr

        # Allocate output array
        embedding = cp.zeros(
            (n_samples, self.n_components),
            order="F",
            dtype=cp.float32,
        )
        cdef uintptr_t embed_ptr = <uintptr_t>embedding.data.ptr

        # Execute fit
        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
        cdef float kl_divergence = 0
        cdef int n_iter = 0

        with nogil:
            if sparse_fit:
                TSNE_fit_sparse(
                    handle_[0],
                    <int*>X_indptr_ptr,
                    <int*>X_indices_ptr,
                    <float*>X_ptr,
                    <float*>embed_ptr,
                    X_nnz,
                    n_samples,
                    n_features,
                    <int*>knn_indices_ptr,
                    <float*>knn_dists_ptr,
                    params,
                    &kl_divergence,
                    &n_iter,
                )
            else:
                TSNE_fit(
                    handle_[0],
                    <float*>X_ptr,
                    <float*>embed_ptr,
                    n_samples,
                    n_features,
                    <int64_t*> knn_indices_ptr,
                    <float*> knn_dists_ptr,
                    params,
                    &kl_divergence,
                    &n_iter,
                )
        handle.sync()

        # Store fitted attributes
        self._kl_divergence_ = kl_divergence
        self.n_iter_ = n_iter
        self.learning_rate_ = params.pre_learning_rate
        self.embedding_ = ArrayIndexPair(embedding, index)

        return self

    @generate_docstring(return_values={'name': 'X_new',
                                       'type': 'dense',
                                       'description': 'Embedding of the \
                                                       data in \
                                                       low-dimensional space.',
                                       'shape': '(n_samples, n_components)'})
    @mlfunc(preserve_index=True, column_names="feature_names_out")
    def fit_transform(self, X, y=None, *, knn_graph=None):
        """
        Fit X into an embedded space and return that transformed output.
        """
        self.fit(X, knn_graph=knn_graph)
        return self.embedding_

    @property
    def kl_divergence_(self):
        if self.method == 'barnes_hut':
            warnings.warn("The calculation of the Kullback-Leibler "
                          "divergence is still an experimental feature "
                          "while using the Barnes Hut algorithm.")
        return self._kl_divergence_

    @kl_divergence_.setter
    def kl_divergence_(self, value):
        self._kl_divergence_ = value
