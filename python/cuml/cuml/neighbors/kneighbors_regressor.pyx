#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp

from cuml.common.doc_utils import generate_docstring
from cuml.internals import get_handle, mlfunc
from cuml.internals.interop import UnsupportedOnGPU
from cuml.internals.mixins import FMajorInputTagMixin, RegressorMixin
from cuml.internals.validation import check_consistent_length, check_y
from cuml.neighbors.nearest_neighbors import NeighborsBase
from cuml.neighbors.weights import compute_weights

from libc.stdint cimport int64_t, uintptr_t
from libcpp cimport bool
from libcpp.vector cimport vector
from pylibraft.common.handle cimport handle_t


cdef extern from "cuml/neighbors/knn.hpp" namespace "ML" nogil:

    void knn_regress(
        handle_t &handle,
        float *out,
        int64_t *knn_indices,
        vector[float *] &y,
        size_t n_index_rows,
        size_t n_query_rows,
        int k,
        float *sample_weight
    ) except +

    void knn_regress(
        handle_t &handle,
        double *out,
        int64_t *knn_indices,
        vector[double *] &y,
        size_t n_index_rows,
        size_t n_query_rows,
        int k,
        float *sample_weight
    ) except +


class KNeighborsRegressor(RegressorMixin, FMajorInputTagMixin, NeighborsBase):
    """
    K-Nearest Neighbors Regressor is an instance-based learning technique,
    that keeps training samples around for prediction, rather than trying
    to learn a generalizable set of model parameters.

    The K-Nearest Neighbors Regressor will compute the average of the
    labels for the k closest neighbors and use it as the label.

    Parameters
    ----------
    n_neighbors : int (default=5)
        Default number of neighbors to query
    algorithm : string (default='auto')
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

        >>> from cuml.neighbors import KNeighborsRegressor
        >>> from cuml.datasets import make_regression
        >>> from cuml.model_selection import train_test_split

        >>> X, y = make_regression(n_samples=100, n_features=10,
        ...                        random_state=5)
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...   X, y, train_size=0.80, random_state=5)

        >>> knn = KNeighborsRegressor(n_neighbors=10)
        >>> knn.fit(X_train, y_train)
        KNeighborsRegressor(n_neighbors=10)
        >>> knn.predict(X_test) # doctest: +SKIP
        array([ 14.770798  ,  51.8834    ,  66.15657   ,  46.978275  ,
            21.589611  , -14.519918  , -60.25534   , -20.856869  ,
            29.869623  , -34.83317   ,   0.45447388, 120.39675   ,
            109.94834   ,  63.57794   , -17.956171  ,  78.77663   ,
            30.412262  ,  32.575233  ,  74.72834   , 122.276855  ],
        dtype=float32)

    Notes
    -----

    For additional docs, see `scikitlearn's KNeighborsClassifier
    <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>`_.
    """
    _cpu_class_path = "sklearn.neighbors.KNeighborsRegressor"

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
        y = cp.asarray(
            model._y,
            dtype="float64" if model._y.dtype == "float64" else "float32",
            order="F",
        )
        return {"_y": y, **super()._attrs_from_cpu(model)}

    def _attrs_to_cpu(self, model):
        return {
            "_y": cp.asnumpy(self._y),
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

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = True
        return tags

    @generate_docstring()
    @mlfunc(set_input_type=True)
    def fit(self, X, y) -> "KNeighborsRegressor":
        """
        Fit a GPU index for k-nearest neighbors regression model.

        """
        if self.weights not in ('uniform', 'distance', None) and not callable(self.weights):
            raise ValueError(
                f"weights must be 'uniform', 'distance', or a callable, got {self.weights}"
            )
        super().fit(X)

        y = check_y(y, dtype=("float32", "float64"), order="F", accept_multi_output=True)
        check_consistent_length(self._fit_X, y)
        self._y = y

        return self

    @generate_docstring(return_values={'name': 'X_new',
                                       'type': 'dense',
                                       'description': 'Predicted values',
                                       'shape': '(n_samples, n_features)'})
    @mlfunc(preserve_index=True)
    def predict(self, X):
        """
        Use the trained k-nearest neighbors regression model to
        predict the labels for X

        """
        # Get KNN results - always get distances to compute weights
        distances, indices = self.kneighbors(X, return_distance=True)
        indices = cp.ascontiguousarray(indices, dtype=cp.int64)
        cdef size_t n_rows = indices.shape[0]
        cdef int64_t* inds_ptr = <int64_t*><uintptr_t>indices.data.ptr

        res_cols = 1 if self._y.ndim == 1 else self._y.shape[1]
        res_shape = n_rows if res_cols == 1 else (n_rows, res_cols)

        out = cp.zeros(res_shape, dtype=self._y.dtype, order="C")
        cdef uintptr_t out_ptr = <uintptr_t>out.data.ptr
        cdef bool use_f32 = self._y.dtype == "float32"

        cdef vector[float*] y_f32_vec
        cdef vector[double*] y_f64_vec
        for col_num in range(res_cols):
            col = self._y if res_cols == 1 else self._y[:, col_num]
            if use_f32:
                y_f32_vec.push_back(<float*><uintptr_t>col.data.ptr)
            else:
                y_f64_vec.push_back(<double*><uintptr_t>col.data.ptr)

        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

        # Compute weights (returns None for uniform weights)
        weights = compute_weights(distances, self.weights)
        cdef float* weights_ptr = <float*><uintptr_t>(
            0 if weights is None else weights.data.ptr
        )

        cdef size_t n_samples_fit = self._y.shape[0]
        cdef int n_neighbors = self.n_neighbors
        with nogil:
            if use_f32:
                knn_regress(
                    handle_[0],
                    <float *>out_ptr,
                    inds_ptr,
                    y_f32_vec,
                    n_samples_fit,
                    n_rows,
                    n_neighbors,
                    weights_ptr
                )
            else:
                knn_regress(
                    handle_[0],
                    <double *>out_ptr,
                    inds_ptr,
                    y_f64_vec,
                    n_samples_fit,
                    n_rows,
                    n_neighbors,
                    weights_ptr
                )

        handle.sync()

        return out
