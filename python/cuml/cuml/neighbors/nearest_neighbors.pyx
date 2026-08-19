#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import warnings

import cupy as cp
import cupyx.scipy.sparse as cp_sp
import scipy.sparse as sp

import cuml
from cuml.common.doc_utils import generate_docstring, insert_into_docstring
from cuml.common.sparse import is_sparse
from cuml.internals.base import Base, get_handle
from cuml.internals.interop import InteropMixin, UnsupportedOnGPU
from cuml.internals.mixins import CMajorInputTagMixin, SparseInputTagMixin
from cuml.internals.outputs import ReflectedAttr, mlfunc
from cuml.internals.validation import (
    check_array,
    check_features,
    check_inputs,
    check_is_fitted,
)

from libc.stdint cimport int64_t, uint32_t, uintptr_t
from libcpp cimport bool
from libcpp.vector cimport vector
from pylibraft.common.handle cimport handle_t

from cuml.metrics.distance_type cimport DistanceType


cdef extern from "cuml/neighbors/knn.hpp" namespace "ML" nogil:
    void brute_force_knn(
        const handle_t &handle,
        vector[float*] &inputs,
        vector[int] &sizes,
        int D,
        float *search_items,
        int n,
        int64_t *res_I,
        float *res_D,
        int k,
        bool rowMajorIndex,
        bool rowMajorQuery,
        DistanceType metric,
        float metric_arg
    ) except +

    void rbc_build_index(
        const handle_t &handle,
        uintptr_t &rbc_index,
        float *X,
        int64_t n_rows,
        int64_t n_cols,
        DistanceType metric
    ) except +

    void rbc_knn_query(
        const handle_t &handle,
        const uintptr_t &rbc_index,
        uint32_t k,
        float *search_items,
        uint32_t n_search_items,
        int64_t dim,
        int64_t *out_inds,
        float *out_dists
    ) except +

    void rbc_radius_neighbors_graph(
        const handle_t& handle,
        const uintptr_t& rbc_index,
        const float* query,
        int64_t n_query,
        int64_t dim,
        float radius,
        int64_t *adj_rows,
        int64_t *adj_cols,
        int64_t nnz
    ) except +

    void rbc_free_index(
        uintptr_t rbc_index
    ) except +

    void approx_knn_build_index(
        handle_t &handle,
        knnIndex* index,
        knnIndexParam* params,
        DistanceType metric,
        float metricArg,
        float *index_array,
        int n,
        int D
    ) except +

    void approx_knn_search(
        handle_t &handle,
        float *distances,
        int64_t* indices,
        knnIndex* index,
        int k,
        const float *query_array,
        int n
    ) except +

    cdef cppclass knnIndex:
        pass

    cdef cppclass knnIndexParam:
        pass

    cdef cppclass IVFParam(knnIndexParam):
        int nlist
        int nprobe

    cdef cppclass IVFFlatParam(IVFParam):
        pass

    cdef cppclass IVFPQParam(IVFParam):
        int M
        int n_bits
        bool usePrecomputedTables


cdef extern from "cuml/neighbors/knn_sparse.hpp" namespace "ML::Sparse" nogil:
    void brute_force_knn(handle_t &handle,
                         const int *idxIndptr,
                         const int *idxIndices,
                         const float *idxData,
                         size_t idxNNZ,
                         int n_idx_rows,
                         int n_idx_cols,
                         const int *queryIndptr,
                         const int *queryIndices,
                         const float *queryData,
                         size_t queryNNZ,
                         int n_query_rows,
                         int n_query_cols,
                         int *output_indices,
                         float *output_dists,
                         int k,
                         size_t batch_size_index,
                         size_t batch_size_query,
                         DistanceType metric,
                         float metricArg) except +

# Kernel to check for zeros in the second column of distances
check_zero_kernel = cp.RawKernel(r'''
extern "C" __global__
void check_zero_kernel(float* D, int n_rows, int n_cols, int* zero_found) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= n_rows) return;

    int index = row * n_cols + 1;

    // Check if the second column has a zero
    if (D[index] == 0.0f) {
        *zero_found = 1;
    }
}
''', 'check_zero_kernel')

# Kernel to swap self index to the first column
swap_kernel = cp.RawKernel(r'''
extern "C" __global__
void swap_kernel(long long int* I, float* D, int n_rows, int n_cols) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= n_rows) return;

    int base_idx = row * n_cols;

    for (int j = 1; j < n_cols; ++j) {
        int idx = base_idx + j;
        if (I[idx] == row) {
            // Swap I
            int tmp_I = I[base_idx];
            I[base_idx] = I[idx];
            I[idx] = tmp_I;

            // Swap D
            float tmp_D = D[base_idx];
            D[base_idx] = D[idx];
            D[idx] = tmp_D;

            break; // found self index
        }
    }
}
''', 'swap_kernel')


def _drop_self_edges(distances, indices):
    """Drop edges between a point and itself in the knn graph"""
    rows, cols = indices.shape

    # Launch config
    threads_per_block = 32
    blocks = (rows + threads_per_block - 1) // threads_per_block

    # Allocate memory for the flag
    zero_found = cp.zeros(1, dtype=cp.int32)

    # Run the kernel to check for zeros
    check_zero_kernel(
        (blocks,),
        (threads_per_block,),
        (distances.ravel(), rows, cols, zero_found.ravel())
    )

    # only run kernel if there are multiple zero distances
    if zero_found.item():
        threads_per_block = 32
        blocks = (rows + threads_per_block - 1) // threads_per_block
        swap_kernel(
            (blocks,),
            (threads_per_block,),
            (indices.ravel(), distances.ravel(), rows, cols)
        )

    # Drop first column, ensuring C contiguous and correct dtypes
    indices = cp.ascontiguousarray(indices[:, 1:], dtype=cp.int64)
    distances = cp.ascontiguousarray(distances[:, 1:], dtype=cp.float32)

    return distances, indices


METRICS = {
    "euclidean": DistanceType.L2SqrtExpanded,
    "l2": DistanceType.L2SqrtExpanded,
    "sqeuclidean": DistanceType.L2Expanded,
    "cityblock": DistanceType.L1,
    "l1": DistanceType.L1,
    "manhattan": DistanceType.L1,
    "taxicab": DistanceType.L1,
    "braycurtis": DistanceType.BrayCurtis,
    "canberra": DistanceType.Canberra,
    "minkowski": DistanceType.LpUnexpanded,
    "lp": DistanceType.LpUnexpanded,
    "chebyshev": DistanceType.Linf,
    "linf": DistanceType.Linf,
    "jensenshannon": DistanceType.JensenShannon,
    "cosine": DistanceType.CosineExpanded,
    "correlation": DistanceType.CorrelationExpanded,
    "inner_product": DistanceType.InnerProduct,
    "jaccard": DistanceType.JaccardExpanded,
    "hellinger": DistanceType.HellingerExpanded,
    "haversine": DistanceType.Haversine,
}


cdef DistanceType _metric_to_distance_type(str metric):
    try:
        return METRICS[metric]
    except Exception:
        raise ValueError(f"Metric {metric} is not supported") from None


cdef class RBCIndex:
    """An RBC index."""
    cdef uintptr_t index
    cdef int64_t n_samples

    def __dealloc__(self):
        if self.index != 0:
            rbc_free_index(self.index)
            self.index = 0

    def __reduce__(self):
        raise TypeError("Instances of RBCIndex aren't pickleable")

    @staticmethod
    def build(X, metric):
        """Build a new RBC index."""
        cdef RBCIndex self = RBCIndex.__new__(RBCIndex)

        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><uintptr_t>handle.getHandle()
        cdef float* X_ptr = <float*><uintptr_t>X.data.ptr
        cdef int64_t n_rows = X.shape[0]
        cdef int64_t n_cols = X.shape[1]
        cdef DistanceType distance_type = _metric_to_distance_type(metric)

        with nogil:
            rbc_build_index(
                handle_[0],
                self.index,
                X_ptr,
                n_rows,
                n_cols,
                distance_type,
            )
        handle.sync()
        self.n_samples = n_rows
        return self

    def radius_neighbors_graph(
        RBCIndex self,
        X,
        float radius,
    ):
        """Query the index for neighbors within a radius"""
        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><uintptr_t>handle.getHandle()
        cdef int64_t n_query = X.shape[0]

        indptr = cp.empty(n_query + 1, dtype=cp.int64)
        cdef float* X_ptr = <float*><uintptr_t>X.data.ptr
        cdef int64_t n_rows = X.shape[0]
        cdef int64_t n_cols = X.shape[1]
        cdef int64_t* indptr_ptr = <int64_t*><uintptr_t>indptr.data.ptr

        with nogil:
            rbc_radius_neighbors_graph(
                handle_[0],
                self.index,
                X_ptr,
                n_rows,
                n_cols,
                radius,
                indptr_ptr,
                NULL,
                0,
            )

        cdef int64_t nnz = indptr[-1].item()
        indices = cp.empty(nnz, dtype=cp.int64)
        cdef int64_t* indices_ptr = <int64_t*><uintptr_t>indices.data.ptr

        with nogil:
            rbc_radius_neighbors_graph(
                handle_[0],
                self.index,
                X_ptr,
                n_rows,
                n_cols,
                radius,
                indptr_ptr,
                indices_ptr,
                nnz,
            )

        data = cp.ones(nnz)
        return cp_sp.csr_matrix(
            (data, indices, indptr),
            shape=(n_rows, self.n_samples),
        )

    def kneighbors(RBCIndex self, X, uint32_t n_neighbors):
        """Query the index for the k nearest neighbors."""
        if X.shape[1] > 3:
            raise ValueError(
                "The rbc algorithm is not supported for >3 dimensions currently."
            )
        distances = cp.empty((X.shape[0], n_neighbors), dtype=cp.float32, order="C")
        indices = cp.empty((X.shape[0], n_neighbors), dtype=cp.int64, order="C")

        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><uintptr_t>handle.getHandle()
        cdef float* X_ptr = <float*><uintptr_t>X.data.ptr
        cdef uint32_t n_rows = X.shape[0]
        cdef int64_t n_cols = X.shape[1]
        cdef int64_t* indices_ptr = <int64_t*><uintptr_t>indices.data.ptr
        cdef float* distances_ptr = <float*><uintptr_t>distances.data.ptr

        with nogil:
            rbc_knn_query(
                handle_[0],
                self.index,
                n_neighbors,
                X_ptr,
                n_rows,
                n_cols,
                indices_ptr,
                distances_ptr,
            )
        handle.sync()
        return distances, indices


cdef class ApproxIndex:
    """An approximate nearest neighbors index (IVFQ or IVFFlat)"""
    cdef knnIndex* index

    def __dealloc__(self):
        if self.index != NULL:
            del self.index

    cdef _init_ivfflat(self, IVFFlatParam *out, params):
        if params is None:
            params = {"nlist": 8, "nprobe": 2}
        out.nlist = params["nlist"]
        out.nprobe = params["nprobe"]

    cdef _init_ivfpq(self, IVFPQParam *out, params):
        # TODO: These parameter defaults don't work for all datasets and
        # don't match the defaults in cuVS. The defaults here should be
        # redone to match cuVS.
        if params is None:
            params = {
                "nlist": 8,
                "nprobe": 3,
                "M": 0,
                "n_bits": 8,
            }
        out.nlist = params["nlist"]
        out.nprobe = params["nprobe"]
        out.M = params["M"]
        out.n_bits = params["n_bits"]

    @staticmethod
    def build(X, metric, algorithm, params=None, float p=2):
        """Build a new approx index."""
        cdef ApproxIndex self = ApproxIndex.__new__(ApproxIndex)

        cdef IVFFlatParam flat_params
        cdef IVFPQParam pq_params
        cdef knnIndexParam *build_params

        if algorithm == "ivfflat":
            self._init_ivfflat(&flat_params, params)
            build_params = &flat_params
        elif algorithm == "ivfpq":
            self._init_ivfpq(&pq_params, params)
            build_params = &pq_params
        else:
            raise ValueError("algorithm must be one of {'ivfflat', 'ivfpq'}")

        handle = get_handle()
        cdef DistanceType distance_type = _metric_to_distance_type(metric)
        cdef handle_t* handle_ = <handle_t*><uintptr_t>handle.getHandle()
        cdef float* X_ptr = <float*><uintptr_t>X.data.ptr
        cdef int n_rows = X.shape[0]
        cdef int n_cols = X.shape[1]

        with nogil:
            self.index = new knnIndex()
            approx_knn_build_index(
                handle_[0],
                self.index,
                build_params,
                distance_type,
                p,
                X_ptr,
                n_rows,
                n_cols,
            )
        handle.sync()

        return self

    def kneighbors(ApproxIndex self, X, int n_neighbors):
        """Query the index for the k nearest neighbors."""
        distances = cp.empty((X.shape[0], n_neighbors), dtype=cp.float32, order="C")
        indices = cp.empty((X.shape[0], n_neighbors), dtype=cp.int64, order="C")

        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><uintptr_t>handle.getHandle()
        cdef float* distances_ptr = <float*><uintptr_t>distances.data.ptr
        cdef int64_t* indices_ptr = <int64_t*><uintptr_t>indices.data.ptr
        cdef float* X_ptr = <float*><uintptr_t>X.data.ptr
        cdef int n_rows = X.shape[0]

        with nogil:
            approx_knn_search(
                handle_[0],
                distances_ptr,
                indices_ptr,
                self.index,
                n_neighbors,
                X_ptr,
                n_rows,
            )
        handle.sync()
        return distances, indices


class NeighborsBase(InteropMixin, CMajorInputTagMixin, SparseInputTagMixin, Base):
    """Base class for `cuml.neighbors` models"""
    _fit_X = ReflectedAttr()

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "n_neighbors",
            "algorithm",
            "metric",
            "p",
            "metric_params",
            "algo_params",
            "n_jobs",  # Ignored, here for sklearn API compatibility
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        if not (
            isinstance(model.metric, str) and
            model.metric in cuml.neighbors.VALID_METRICS["brute"]
        ):
            raise UnsupportedOnGPU(f"`metric={model.metric!r}` is not supported")

        return {
            "n_neighbors": model.n_neighbors,
            "algorithm": "auto" if model.algorithm == "auto" else "brute",
            "metric": model.metric,
            "p": model.p,
            "metric_params": model.metric_params,
        }

    def _params_to_cpu(self):
        return {
            "n_neighbors": self.n_neighbors,
            "algorithm": "auto" if self.algorithm == "auto" else "brute",
            "metric": self.metric,
            "p": self.p,
            "metric_params": self.metric_params,
        }

    def _attrs_from_cpu(self, model):
        if sp.issparse(model._fit_X):
            fit_X = cp_sp.csr_matrix(model._fit_X, dtype="float32")
        else:
            fit_X = cp.asarray(model._fit_X, order="C", dtype="float32")

        return {
            "n_samples_fit_": model.n_samples_fit_,
            "effective_metric_": model.effective_metric_,
            "effective_metric_params_": model.effective_metric_params_,
            "_fit_X": fit_X,
            "_fit_method": "brute",
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        return {
            "n_samples_fit_": self.n_samples_fit_,
            "effective_metric_": self.effective_metric_,
            "effective_metric_params_": self.effective_metric_params_,
            "_fit_X": (
                self._fit_X.get()
                if cp_sp.issparse(self._fit_X)
                else self._fit_X.get(order="A")
            ),
            "_fit_method": "brute",
            "_tree": None,
            **super()._attrs_to_cpu(model),
        }

    def __init__(
        self,
        *,
        n_neighbors=5,
        algorithm="auto",
        metric="euclidean",
        p=2,
        algo_params=None,
        metric_params=None,
        n_jobs=None,  # Ignored, here for sklearn API compatibility
        verbose=False,
        output_type=None,
    ):
        super().__init__(verbose=verbose, output_type=output_type)
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_params = metric_params
        self.algo_params = algo_params
        self.p = p
        self.algorithm = algorithm
        self.algo_params = algo_params
        self.n_jobs = n_jobs  # Ignored, here for sklearn API compatibility

    @property
    def _effective_p(self):
        """The `p` value to use, based on `effective_metric_params_` or `p`"""
        return self.effective_metric_params_.get("p", self.p)

    def __getstate__(self):
        state = self.__dict__.copy()
        # TODO: Indices currently aren't pickleable. For now we drop them and
        # recreate them on load.
        state.pop("_index", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if (fit_method := state.get("_fit_method")) in ("rbc", "ivfpq", "ivfflat"):
            # TODO: These index types currently aren't pickleable. For now we
            # recreate them on load.
            fit_X = getattr(self, "_fit_X", None)
            X = cp.asarray(fit_X) if fit_X is not None else None
            if fit_method == "rbc":
                self._index = RBCIndex.build(X, self.effective_metric_)
            else:
                self._index = ApproxIndex.build(
                    X,
                    self.effective_metric_,
                    fit_method,
                    params=self.algo_params,
                    p=self._effective_p,
                )

    @generate_docstring(X='dense_sparse')
    @mlfunc(set_input_type=True)
    def fit(self, X, y=None) -> "NearestNeighbors":
        """
        Fit GPU index for performing nearest neighbor queries.

        """
        sparse = is_sparse(X)
        valid_metrics = (
            cuml.neighbors.VALID_METRICS_SPARSE if sparse else cuml.neighbors.VALID_METRICS
        )

        X = check_inputs(
            self,
            X,
            dtype="float32",
            accept_sparse=["csr"],
            order="C",
            reset=True,
        )
        self._fit_X = X

        # Normalize metric, and simplify for common cases
        self.effective_metric_ = self.metric
        self.effective_metric_params_ = (
            {} if self.metric_params is None else self.metric_params.copy()
        )
        # Drop "p" and simplify metric, unless minkowski is necessary
        p = self.effective_metric_params_.pop("p", self.p)
        if self.effective_metric_ in ("minkowski", "lp"):
            if p == 1:
                self.effective_metric_ = "manhattan"
            elif p == 2:
                self.effective_metric_ = "euclidean"
            elif p == cp.inf:
                self.effective_metric_ = "chebyshev"
            else:
                self.effective_metric_params_["p"] = p

        self.n_samples_fit_ = self._fit_X.shape[0]

        if self.algorithm == "auto":
            if (
                self.n_features_in_ in (2, 3)
                and not sparse
                and self.effective_metric_ in cuml.neighbors.VALID_METRICS["rbc"]
                and self._fit_X.shape[0]**0.5 >= self.n_neighbors
            ):
                self._fit_method = "rbc"
            else:
                self._fit_method = "brute"
        elif self.algorithm in {"brute", "ivfpq", "ivfflat", "rbc"}:
            self._fit_method = self.algorithm
        else:
            raise ValueError(f"algorithm={self.algorithm!r} is not supported")

        if sparse and self._fit_method != "brute":
            raise ValueError(
                f"algorithm={self._fit_method!r} doesn't support sparse data"
            )

        if self.effective_metric_ not in valid_metrics[self._fit_method]:
            raise ValueError(
                f"Metric {self.effective_metric_} is not supported. See "
                f"`cuml.neighbors.VALID_METRICS{'_SPARSE' * sparse}[{self._fit_method!r}]`"
                f"for a list of valid options."
            )

        if self._fit_method in ('ivfflat', 'ivfpq'):
            self._index = ApproxIndex.build(
                X,
                self.effective_metric_,
                self._fit_method,
                params=self.algo_params,
                p=self._effective_p,
            )
        elif self._fit_method == "rbc":
            self._index = RBCIndex.build(X, self.effective_metric_)

        return self

    @insert_into_docstring(parameters=[('dense', '(n_samples, n_features)')],
                           return_values=[('dense', '(n_samples, n_features)'),
                                          ('dense',
                                           '(n_samples, n_features)')])
    @mlfunc(preserve_index=True)
    def kneighbors(
        self,
        X=None,
        n_neighbors=None,
        return_distance=True,
        *,
        two_pass_precision=False
    ):
        """
        Query the GPU index for the k nearest neighbors of column vectors in X.

        Parameters
        ----------
        X : {}

        n_neighbors : Integer
            Number of neighbors to search. If not provided, the n_neighbors
            from the model instance is used (default=10)

        return_distance: Boolean
            If False, distances will not be returned

        two_pass_precision : bool, optional (default = False)
            When set to True, a slow second pass will be used to improve the
            precision of results returned for searches using L2-derived
            metrics. FAISS uses the Euclidean distance decomposition trick to
            compute distances in this case, which may result in numerical
            errors for certain data. In particular, when several samples
            are close to the query sample (relative to typical inter-sample
            distances), numerical instability may cause the computed distance
            between the query and itself to be larger than the computed
            distance between the query and another sample. As a result, the
            query is not returned as the nearest neighbor to itself.  If this
            flag is set to true, distances to the query vectors will be
            recomputed with high precision for all retrieved samples, and the
            results will be re-sorted accordingly. Note that for large values
            of k or large numbers of query vectors, this correction becomes
            impractical in terms of both runtime and memory. It should be used
            with care and only when strictly necessary (when precise results
            are critical and samples may be tightly clustered).

        Returns
        -------
        distances : {}
            The distances of the k-nearest neighbors for each column vector
            in X

        indices : {}
            The indices of the k-nearest neighbors for each column vector in X
        """
        check_is_fitted(self)

        n_neighbors = self.n_neighbors if n_neighbors is None else n_neighbors

        if use_training_data := (X is None):
            X = self._fit_X
            n_neighbors += 1
        else:
            check_features(self, X)

        if (n_neighbors is None and self.n_neighbors is None) or n_neighbors <= 0:
            raise ValueError("k or n_neighbors must be a positive integers")

        if n_neighbors > self.n_samples_fit_:
            raise ValueError("n_neighbors must be <= number of samples in index")

        if cp_sp.issparse(self._fit_X):
            distances, indices = self._kneighbors_sparse(X, n_neighbors)
        else:
            distances, indices = self._kneighbors_dense(
                X, n_neighbors, two_pass_precision
            )

        if use_training_data:
            distances, indices = _drop_self_edges(distances, indices)

        return (distances, indices) if return_distance else indices

    def _kneighbors_dense(self, X, int n_neighbors, two_pass_precision=False):
        if is_sparse(X):
            raise ValueError("A NearestNeighbors model trained on dense "
                             "data requires dense input to kneighbors()")

        X = check_array(
            X,
            dtype="float32",
            order="C",
            input_name="X",
        )
        cdef int n_rows = X.shape[0]
        cdef int n_cols = X.shape[1]

        use_index = self._fit_method != "brute"
        if self._fit_method == "rbc" and n_neighbors > self.n_samples_fit_**0.5:
            warnings.warn(
                f"algorithm='rbc' requires sqrt(n_samples) >= n_neighbors "
                f"({self.n_samples_fit_**0.5:.2f} > {n_neighbors}). "
                f"Falling back to algorithm='brute'."
            )
            use_index = False

        if use_index:
            distances, indices = self._index.kneighbors(X, n_neighbors)
            return distances, indices

        distances = cp.empty((n_rows, n_neighbors), dtype=cp.float32, order="C")
        indices = cp.empty((n_rows, n_neighbors), dtype=cp.int64, order="C")

        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
        cdef vector[float*] inputs
        cdef vector[int] sizes
        inputs.push_back(<float*><uintptr_t>self._fit_X.data.ptr)
        sizes.push_back(<int>self.n_samples_fit_)
        cdef DistanceType distance_type = _metric_to_distance_type(self.effective_metric_)
        cdef float* X_ptr = <float*><uintptr_t>X.data.ptr
        cdef int64_t* indices_ptr = <int64_t*><uintptr_t>indices.data.ptr
        cdef float* distances_ptr = <float*><uintptr_t>distances.data.ptr
        cdef float metric_arg = self._effective_p

        with nogil:
            brute_force_knn(
                handle_[0],
                inputs,
                sizes,
                n_cols,
                X_ptr,
                n_rows,
                indices_ptr,
                distances_ptr,
                n_neighbors,
                True,
                True,
                distance_type,
                metric_arg
            )
        handle.sync()

        if two_pass_precision:
            distances, indices = self._maybe_apply_two_pass_precision(
                X, distances, indices
            )

        return distances, indices

    def _maybe_apply_two_pass_precision(self, X, distances, indices):
        # FAISS employs imprecise distance algorithm only for L2-based
        # expanded metrics. This code corrects numerical instabilities
        # that could arise.
        metric = _metric_to_distance_type(self.effective_metric_)
        if not (
            metric == DistanceType.L2SqrtExpanded or
            metric == DistanceType.L2Expanded or
            (metric == DistanceType.LpUnexpanded and self._effective_p == 2)
        ):
            # Nothing to do
            return distances, indices

        self_diff = X[indices] - X[:, cp.newaxis, :]
        distances = cp.sum(self_diff * self_diff, axis=2)
        correct_order = cp.argsort(distances, axis=1)

        distances = cp.take_along_axis(distances, correct_order, axis=1)
        indices = cp.take_along_axis(indices, correct_order, axis=1)

        distances = cp.ascontiguousarray(distances, dtype=cp.float32)
        indices = cp.ascontiguousarray(indices, dtype=cp.int64)
        return distances, indices

    def _kneighbors_sparse(self, X, int n_neighbors):
        if not is_sparse(X):
            raise ValueError("A NearestNeighbors model trained on sparse "
                             "data requires sparse input to kneighbors()")

        algo_params = self.algo_params or {}
        cdef size_t batch_size_index = algo_params.get("batch_size_index", 10000)
        cdef size_t batch_size_query = algo_params.get("batch_size_query", 10000)

        cdef DistanceType metric = _metric_to_distance_type(self.effective_metric_)
        cdef float metric_arg = self._effective_p

        X = check_array(
            X,
            dtype="float32",
            accept_sparse=["csr"],
            input_name="X",
        )
        cdef int* X_indptr = <int *><uintptr_t>X.indptr.data.ptr
        cdef int* X_indices = <int *><uintptr_t>X.indices.data.ptr
        cdef float* X_data = <float *><uintptr_t>X.data.data.ptr
        cdef size_t X_nnz = X.nnz
        cdef int X_n_rows = X.shape[0]
        cdef int X_n_cols = X.shape[1]

        cdef int* idx_indptr = <int *><uintptr_t>self._fit_X.indptr.data.ptr
        cdef int* idx_indices = <int *><uintptr_t>self._fit_X.indices.data.ptr
        cdef float* idx_data = <float *><uintptr_t>self._fit_X.data.data.ptr
        cdef size_t idx_nnz = self._fit_X.nnz
        cdef int idx_n_rows = self._fit_X.shape[0]
        cdef int idx_n_cols = self._fit_X.shape[1]

        # Allocate outputs
        indices = cp.empty((X_n_rows, n_neighbors), dtype=cp.int32, order="C")
        distances = cp.empty((X_n_rows, n_neighbors), dtype=cp.float32, order="C")
        cdef int* indices_ptr = <int *><uintptr_t>indices.data.ptr
        cdef float* distances_ptr = <float *><uintptr_t>distances.data.ptr

        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
        with nogil:
            brute_force_knn(
                handle_[0],
                idx_indptr,
                idx_indices,
                idx_data,
                idx_nnz,
                idx_n_rows,
                idx_n_cols,
                X_indptr,
                X_indices,
                X_data,
                X_nnz,
                X_n_rows,
                X_n_cols,
                indices_ptr,
                distances_ptr,
                n_neighbors,
                batch_size_index,
                batch_size_query,
                metric,
                metric_arg,
            )
        handle.sync()

        return distances, indices

    @insert_into_docstring(parameters=[('dense', '(n_samples, n_features)')])
    @mlfunc
    def kneighbors_graph(self, X=None, n_neighbors=None, mode='connectivity'):
        """
        Find the k nearest neighbors of column vectors in X and return as
        a sparse matrix in CSR format.

        Parameters
        ----------
        X : {}

        n_neighbors : Integer
            Number of neighbors to search. If not provided, the n_neighbors
            from the model instance is used

        mode : string (default='connectivity')
            Values in connectivity matrix: 'connectivity' returns the
            connectivity matrix with ones and zeros, 'distance' returns the
            edges as the distances between points with the requested metric.

        Returns
        -------
        A : sparse graph in CSR format, shape = (n_samples, n_samples_fit)
            n_samples_fit is the number of samples in the fitted data where
            A[i, j] is assigned the weight of the edge that connects i to j.
            Values will either be ones/zeros or the selected distance metric.
            Return types are either cupy's CSR sparse graph (device) or
            numpy's CSR sparse graph (host)

        """
        check_is_fitted(self)

        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        if mode == 'connectivity':
            indices = self.kneighbors(X, n_neighbors, return_distance=False)
            n_samples = indices.shape[0]
            distances = cp.ones(n_samples * n_neighbors, dtype=cp.float32)

        elif mode == 'distance':
            distances, indices = self.kneighbors(X, n_neighbors)
            distances = cp.ravel(distances)

        else:
            raise ValueError('Unsupported mode, must be one of "connectivity"'
                             ' or "distance" but got "%s" instead' % mode)

        n_samples = indices.shape[0]
        indices = cp.ravel(indices)

        n_nonzero = n_samples * n_neighbors
        rowptr = cp.arange(0, n_nonzero + 1, n_neighbors)

        return cp_sp.csr_matrix(
            (distances, indices, rowptr),
            shape=(n_samples, self.n_samples_fit_)
        )


class NearestNeighbors(NeighborsBase):
    """
    NearestNeighbors is an queries neighborhoods from a given set of
    datapoints. Currently, cuML supports k-NN queries, which define
    the neighborhood as the closest `k` neighbors to each query point.

    Parameters
    ----------
    n_neighbors : int (default=5)
        Default number of neighbors to query
    radius : float (default=1.0)
        Range of parameter space to use by default for ``radius_neighbors``
        queries.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    algorithm : string (default='auto')
        The query algorithm to use. Valid options are:

        - ``'auto'``: to automatically select brute-force or
          random ball cover based on data shape and metric
        - ``'rbc'``: for the random ball algorithm, which partitions
          the data space and uses the triangle inequality to lower the
          number of potential distances. Currently, this algorithm
          supports Haversine (2d) and Euclidean in 2d and 3d.
        - ``'brute'``: for brute-force, slow but produces exact results
        - ``'ivfflat'``: for inverted file, divide the dataset in partitions
          and perform search on relevant partitions only
        - ``'ivfpq'``: for inverted file and product quantization,
          same as inverted list, in addition the vectors are broken
          in n_features/M sub-vectors that will be encoded thanks
          to intermediary k-means clusterings. This encoding provide
          partial information allowing faster distances calculations

    metric : string (default='euclidean').
        Distance metric to use. Supported metrics include: 'l1', 'cityblock',
        'taxicab', 'manhattan', 'euclidean', 'l2', 'sqeuclidean', 'canberra',
        'minkowski', 'lp', 'chebyshev', 'linf', 'jensenshannon', 'cosine',
        'braycurtis', 'jaccard', 'hellinger', 'correlation', 'inner_product'.
        The ``'ivfflat'`` and ``'ivfpq'``
        algorithms only support: 'euclidean', 'l2', 'sqeuclidean', 'cosine',
        'correlation', 'inner_product', whereas the ``'rbc'`` algorithm only
        supports 'euclidean', 'l2', and 'haversine' (≤3 dimensions only).
        For sparse inputs, only the ``'brute'`` algorithm is supported, with
        metrics: 'l1', 'cityblock', 'taxicab', 'manhattan', 'euclidean', 'l2',
        'canberra', 'minkowski', 'lp', 'chebyshev', 'linf', 'cosine',
        'inner_product', 'jaccard', 'hellinger'.
    p : float (default=2)
        Parameter for the Minkowski metric. When p = 1, this is equivalent to
        manhattan distance (l1), and euclidean distance (l2) for p = 2. For
        arbitrary p, minkowski distance (lp) is used.
    algo_params : dict, optional (default=None)
        Used to configure the nearest neighbor algorithm to be used.
        If set to None, parameters will be generated automatically.
        Parameters for algorithm ``'brute'`` when inputs are sparse:

            - batch_size_index : (int) number of rows in each batch of \
                                 index array
            - batch_size_query : (int) number of rows in each batch of \
                                 query array

        Parameters for algorithm ``'ivfflat'``:

            - nlist: (int) number of cells to partition dataset into
            - nprobe: (int) at query time, number of cells used for search

        Parameters for algorithm ``'ivfpq'``:

            - nlist: (int) number of cells to partition dataset into
            - nprobe: (int) at query time, number of cells used for search
            - M: (int) number of subquantizers
            - n_bits: (int) bits allocated per subquantizer
            - usePrecomputedTables : (bool) whether to use precomputed tables
    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.
    n_jobs : int (default = None)
        Ignored, here for scikit-learn API compatibility.
    output_type : {None, 'input', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Examples
    --------

    .. code-block:: python

        >>> import cudf
        >>> from cuml.neighbors import NearestNeighbors
        >>> from cuml.datasets import make_blobs

        >>> X, _ = make_blobs(n_samples=5, centers=5,
        ...                   n_features=10, random_state=42)

        >>> # build a cudf Dataframe
        >>> X_cudf = cudf.DataFrame(X)

        >>> # fit model
        >>> model = NearestNeighbors(n_neighbors=3)
        >>> model.fit(X)
        NearestNeighbors(n_neighbors=3)

        >>> # get 3 nearest neighbors
        >>> distances, indices = model.kneighbors(X_cudf)

        >>> # print results
        >>> print(indices)  # doctest: +SKIP
        0  1  2
        0  0  3  1
        1  1  3  0
        2  2  4  0
        3  3  0  1
        4  4  2  0
        >>> print(distances) # doctest: +SKIP
                0          1          2
        0  0.007812  24.786566  26.399996
        1  0.000000  24.786566  30.045017
        2  0.007812   5.458400  27.051241
        3  0.000000  26.399996  27.543869
        4  0.000000   5.458400  29.583437

    Notes
    -----
    For an additional example see `the NearestNeighbors notebook
    <https://github.com/rapidsai/cuml/blob/main/notebooks/nearest_neighbors_demo.ipynb>`_.

    For additional docs, see `scikit-learn's NearestNeighbors
    <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors>`_.

    Pickling ``NearestNeighbors`` instances is supported for all algorithms.
    However, for RBC, IVFPQ or IVFFlat the index will currently be rebuilt upon
    load rather than serialized as part of the pickled binary. For approximate
    indices like IVFPQ or IVFFlat this may result in small differences between
    the original and reloaded models, as the generated indices may differ.
    """
    _cpu_class_path = "sklearn.neighbors.NearestNeighbors"

    def __init__(
        self,
        *,
        n_neighbors=5,
        radius=1.0,
        algorithm="auto",
        metric="euclidean",
        p=2,
        algo_params=None,
        metric_params=None,
        n_jobs=None,  # Ignored, here for sklearn API compatibility
        verbose=False,
        output_type=None,
    ):
        self.radius = radius
        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            metric=metric,
            p=p,
            algo_params=algo_params,
            metric_params=metric_params,
            n_jobs=n_jobs,
            verbose=verbose,
            output_type=output_type
        )

    @classmethod
    def _get_param_names(cls):
        return ["radius", *super()._get_param_names()]

    @classmethod
    def _params_from_cpu(cls, model):
        return {
            "radius": model.radius,
            **super()._params_from_cpu(model),
        }

    def _params_to_cpu(self):
        return {
            "radius": self.radius,
            **super()._params_to_cpu(),
        }

    @insert_into_docstring(parameters=[('dense', '(n_samples, n_features)')])
    @mlfunc
    def radius_neighbors_graph(self, X=None, radius=None):
        """Compute the (weighted) graph of neighbors within a radius.

        Parameters
        ----------
        X : array-like, default=None
            The query point or points. If not provided, neighbors of each indexed
            point are returned. In this case, the query point is not considered its
            own neighbor.

        radius : float, default=None
            Radius of neighborhoods. The default is the value passed to the
            constructor.

        Returns
        -------
        A : sparse-matrix of shape (n_queries, n_samples_fit)
            The neighborhood graph, in CSR format.

        Notes
        -----
        This method is most efficient when the instance is fit with
        `algorithm="rbc"`. Other algorithms will build a temporary RBC index
        per-call, which adds a small overhead.

        Only euclidean/l2 metrics and dense inputs are currently supported.

        Examples
        --------
        >>> import cupy as cp
        >>> from cuml.neighbors import NearestNeighbors
        >>> X = cp.array([[0], [3], [1]])
        >>> nn = NearestNeighbors().fit(X)
        >>> A = nn.radius_neighbors_graph(X, radius=1.5)
        >>> A.toarray()
        array([[1., 0., 1.],
               [0., 1., 0.],
               [1., 0., 1.]])
        """
        check_is_fitted(self)

        if is_sparse(self._fit_X) or is_sparse(X):
            raise TypeError("`radius_neighbors_graph` doesn't support sparse inputs")

        if self.effective_metric_ not in ["l2", "euclidean"]:
            raise ValueError(
                f"`radius_neighbors_graph` doesn't support "
                f"metric={self.effective_metric_!r}"
            )

        if radius is None:
            radius = self.radius

        if radius <= 0:
            raise ValueError(f"Expected `radius > 0`, got {radius}")

        if (using_fit_X := (X is None)):
            X = self._fit_X
        else:
            check_features(self, X)

        X = check_array(
            X,
            dtype="float32",
            order="C",
            input_name="X",
        )

        if hasattr(self, "_index") and isinstance(self._index, RBCIndex):
            # Already fit with RBC, reuse the index
            index = self._index
        else:
            # Fit with another method, build a temporary index
            index = RBCIndex.build(self._fit_X, self.effective_metric_)

        out = index.radius_neighbors_graph(X, radius)
        if using_fit_X:
            # When using the training data, the diagonal elements aren't included
            out.setdiag(cp.int64(0))
            out.eliminate_zeros()

        return out


@mlfunc
def kneighbors_graph(
    X=None,
    n_neighbors=5,
    mode='connectivity',
    verbose=False,
    algorithm="brute",
    metric="euclidean",
    p=2,
    include_self=False,
    metric_params=None,
):
    """
    Computes the (weighted) graph of k-Neighbors for points in X.

    Parameters
    ----------
    X : array-like (device or host) shape = (n_samples, n_features)
        Dense matrix (floats or doubles) of shape (n_samples, n_features).
        Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
        ndarray, cuda array interface compliant array like CuPy

    n_neighbors : Integer
        Number of neighbors to search. If not provided, the n_neighbors
        from the model instance is used (default=5)

    mode : string (default='connectivity')
        Values in connectivity matrix: 'connectivity' returns the
        connectivity matrix with ones and zeros, 'distance' returns the
        edges as the distances between points with the requested metric.

    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.

    algorithm : string (default='brute')
        The query algorithm to use. Valid options are:

        - ``'auto'``: to automatically select brute-force or
          random ball cover based on data shape and metric
        - ``'rbc'``: for the random ball algorithm, which partitions
          the data space and uses the triangle inequality to lower the
          number of potential distances. Currently, this algorithm
          supports 2d Euclidean and Haversine.
        - ``'brute'``: for brute-force, slow but produces exact results
        - ``'ivfflat'``: for inverted file, divide the dataset in partitions
          and perform search on relevant partitions only
        - ``'ivfpq'``: for inverted file and product quantization,
          same as inverted list, in addition the vectors are broken
          in n_features/M sub-vectors that will be encoded thanks
          to intermediary k-means clusterings. This encoding provide
          partial information allowing faster distances calculations

    metric : string (default='euclidean').
        Distance metric to use. Supported distances are ['l1, 'cityblock',
        'taxicab', 'manhattan', 'euclidean', 'l2', 'braycurtis', 'canberra',
        'minkowski', 'chebyshev', 'jensenshannon', 'cosine', 'correlation']

    p : float (default=2) Parameter for the Minkowski metric. When p = 1, this
        is equivalent to manhattan distance (l1), and euclidean distance (l2)
        for p = 2. For arbitrary p, minkowski distance (lp) is used.

    include_self : bool or 'auto' (default=False)
        Whether or not to mark each sample as the first nearest neighbor to
        itself. If 'auto', then True is used for mode='connectivity' and False
        for mode='distance'.

    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.

    Returns
    -------
    A : sparse graph in CSR format, shape = (n_samples, n_samples_fit)
        n_samples_fit is the number of samples in the fitted data where
        A[i, j] is assigned the weight of the edge that connects i to j.
        Values will either be ones/zeros or the selected distance metric.
        Return types are either cupy's CSR sparse graph (device) or
        numpy's CSR sparse graph (host)

    """
    model = NearestNeighbors(
        n_neighbors=n_neighbors,
        verbose=verbose,
        algorithm=algorithm,
        metric=metric,
        p=p,
        metric_params=metric_params,
        output_type="cupy",
    ).fit(X)

    if include_self == 'auto':
        include_self = mode == 'connectivity'

    if not include_self:
        query = None
    else:
        query = model._fit_X

    return model.kneighbors_graph(X=query, n_neighbors=n_neighbors, mode=mode)
