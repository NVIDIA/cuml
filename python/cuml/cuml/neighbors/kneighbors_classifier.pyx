#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp

from cuml.common.doc_utils import generate_docstring
from cuml.internals import get_handle
from cuml.internals.interop import UnsupportedOnGPU
from cuml.internals.mixins import ClassifierMixin, FMajorInputTagMixin
from cuml.internals.outputs import ClassLabels, mlfunc
from cuml.internals.validation import check_consistent_length, check_y
from cuml.neighbors.nearest_neighbors import NeighborsBase
from cuml.neighbors.weights import compute_weights

from libc.stdint cimport int64_t, uintptr_t
from libcpp.vector cimport vector
from pylibraft.common.handle cimport handle_t


cdef extern from "cuml/neighbors/knn.hpp" namespace "ML" nogil:

    void knn_classify(
        handle_t &handle,
        int* out,
        int64_t *knn_indices,
        vector[int*] &y,
        size_t n_index_rows,
        size_t n_samples,
        int k,
        float *sample_weight
    ) except +

    void knn_class_proba(
        handle_t &handle,
        vector[float*] &out,
        int64_t *knn_indices,
        vector[int*] &y,
        size_t n_index_rows,
        size_t n_samples,
        int k,
        float *sample_weight
    ) except +


class KNeighborsClassifier(ClassifierMixin, FMajorInputTagMixin, NeighborsBase):
    """
    K-Nearest Neighbors Classifier is an instance-based learning technique,
    that keeps training samples around for prediction, rather than trying
    to learn a generalizable set of model parameters.

    Parameters
    ----------
    n_neighbors : int (default=5)
        Default number of neighbors to query
    algorithm : string (default='auto')
        The query algorithm to use. Currently, only 'brute' is supported.
    metric : string (default='euclidean').
        Distance metric to use.
    weights : {'uniform', 'distance'} or callable, default='uniform'
        Weight function used in prediction. Possible values:

        - 'uniform' : uniform weights. All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          In this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.
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
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {None, 'input', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Examples
    --------

    .. code-block:: python

        >>> from cuml.neighbors import KNeighborsClassifier
        >>> from cuml.datasets import make_blobs
        >>> from cuml.model_selection import train_test_split

        >>> X, y = make_blobs(n_samples=100, centers=5,
        ...                   n_features=10, random_state=5)
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, train_size=0.80, random_state=5)

        >>> knn = KNeighborsClassifier(n_neighbors=10)

        >>> knn.fit(X_train, y_train)
        KNeighborsClassifier(n_neighbors=10)
        >>> knn.predict(X_test) # doctest: +SKIP
        array([1., 2., 2., 3., 4., 2., 4., 4., 2., 3., 1., 4., 3., 1., 3., 4., 3., # noqa: E501
            4., 1., 3.], dtype=float32)

    Notes
    -----

    For additional docs, see `scikitlearn's KNeighborsClassifier
    <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>`_.
    """
    _cpu_class_path = "sklearn.neighbors.KNeighborsClassifier"

    @classmethod
    def _get_param_names(cls):
        return [*super()._get_param_names(), "weights"]

    @classmethod
    def _params_from_cpu(cls, model):
        if callable(model.weights):
            raise UnsupportedOnGPU(
                "Callable weights are not supported for CPU model conversion"
            )

        return {
            "weights": model.weights,
            **super()._params_from_cpu(model),
        }

    def _params_to_cpu(self):
        return {
            "weights": self.weights,
            **super()._params_to_cpu(),
        }

    def _attrs_from_cpu(self, model):
        return {
            "classes_": model.classes_,
            "_y": cp.asarray(model._y, dtype=cp.int32, order="F"),
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        return {
            "classes_": self.classes_,
            "_y": cp.asnumpy(self._y),
            "outputs_2d_": self.outputs_2d_,
            **super()._attrs_to_cpu(model),
        }

    def __init__(
        self,
        *,
        n_neighbors=5,
        algorithm="auto",
        metric="euclidean",
        weights="uniform",
        p=2,
        algo_params=None,
        metric_params=None,
        n_jobs=None,  # Ignored, here for sklearn API compatibility
        verbose=False,
        output_type=None,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            metric=metric,
            p=p,
            algo_params=algo_params,
            metric_params=metric_params,
            n_jobs=n_jobs,
            verbose=verbose,
            output_type=output_type,
        )
        self.weights = weights

    @generate_docstring()
    @mlfunc(set_input_type=True)
    def fit(self, X, y) -> "KNeighborsClassifier":
        """
        Fit a GPU index for k-nearest neighbors classifier model.

        """
        if self.weights not in ('uniform', 'distance', None) and not callable(self.weights):
            raise ValueError(
                f"weights must be 'uniform', 'distance', or a callable, got {self.weights}"
            )

        super().fit(X)
        y, classes = check_y(
            y,
            dtype="int32",
            order="F",
            accept_multi_output=True,
            return_classes=True,
        )
        check_consistent_length(self._fit_X, y)
        self.classes_ = classes
        self._y = y
        return self

    @property
    def outputs_2d_(self):
        """Whether the output is 2d"""
        return self._y.ndim == 2 and self._y.shape[1] != 1

    @generate_docstring(return_values={'name': 'X_new',
                                       'type': 'dense',
                                       'description': 'Labels predicted',
                                       'shape': '(n_samples, 1)'})
    @mlfunc(preserve_index=True)
    def predict(self, X):
        """
        Use the trained k-nearest neighbors classifier to
        predict the labels for X

        """
        # Get KNN results - always get distances to compute weights
        distances, indices = self.kneighbors(X, return_distance=True)
        indices = cp.ascontiguousarray(indices, dtype=cp.int64)
        cdef size_t n_rows = indices.shape[0]

        # Allocate array for predictions
        out_cols = self._y.shape[1] if self._y.ndim == 2 else 1
        out_shape = (n_rows, out_cols) if out_cols > 1 else n_rows
        out = cp.empty(out_shape, dtype=cp.int32, order="C")
        cdef int* out_ptr = <int*><uintptr_t>out.data.ptr

        # Compose vector of y columns
        cdef vector[int*] y_vec
        for i in range(out_cols):
            col = self._y if out_cols == 1 else self._y[:, i]
            y_vec.push_back(<int*><uintptr_t>col.data.ptr)

        # Compute weights (returns None for uniform weights)
        weights = compute_weights(distances, self.weights)
        cdef float* weights_ptr = <float*><uintptr_t>(
            0 if weights is None else weights.data.ptr
        )

        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
        cdef int64_t* inds_ptr = <int64_t*><uintptr_t>indices.data.ptr
        cdef size_t n_samples_fit = self._y.shape[0]
        cdef int n_neighbors = self.n_neighbors
        with nogil:
            knn_classify(
                handle_[0],
                out_ptr,
                inds_ptr,
                y_vec,
                n_samples_fit,
                n_rows,
                n_neighbors,
                weights_ptr
            )

        handle.sync()

        return ClassLabels(out, self.classes_)

    @generate_docstring(return_values={'name': 'X_new',
                                       'type': 'dense',
                                       'description': 'Labels probabilities',
                                       'shape': '(n_samples, 1)'})
    @mlfunc(preserve_index=True)
    def predict_proba(self, X):
        """
        Use the trained k-nearest neighbors classifier to
        predict the label probabilities for X

        """
        # Get KNN results - always get distances to compute weights
        distances, indices = self.kneighbors(X, return_distance=True)
        indices = cp.ascontiguousarray(indices, dtype=cp.int64)
        cdef size_t n_rows = indices.shape[0]

        if self._y.ndim == 1 or self._y.shape[1] == 1:
            n_classes = [len(self.classes_)]
            ys = [self._y]
        else:
            n_classes = [len(c) for c in self.classes_]
            ys = [self._y[:, i] for i in range(self._y.shape[1])]

        # Construct vectors of y columns and output probas
        probas = []
        cdef vector[float*] out_vec
        cdef vector[int*] y_vec
        for n, y in zip(n_classes, ys):
            proba = cp.zeros((n_rows, n), dtype=cp.float32, order="C")
            probas.append(proba)
            out_vec.push_back(<float*><uintptr_t>proba.data.ptr)
            y_vec.push_back(<int*><uintptr_t>y.data.ptr)

        # Compute weights (returns None for uniform weights)
        weights = compute_weights(distances, self.weights)
        cdef float* weights_ptr = <float*><uintptr_t>(
            0 if weights is None else weights.data.ptr
        )

        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
        cdef int64_t* inds_ptr = <int64_t*><uintptr_t>indices.data.ptr
        cdef size_t n_samples_fit = self._y.shape[0]
        cdef int n_neighbors = self.n_neighbors
        with nogil:
            knn_class_proba(
                handle_[0],
                out_vec,
                inds_ptr,
                y_vec,
                n_samples_fit,
                n_rows,
                n_neighbors,
                weights_ptr
            )
        handle.sync()
        return probas[0] if len(probas) == 1 else probas
