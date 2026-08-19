#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import re

import cudf
import cupy as cp
import numpy as np
import pandas as pd
import treelite

import cuml
from cuml.internals.treelite import safe_treelite_call
from cuml.internals.validation import check_array

from libc.stdint cimport uintptr_t

from cuml.internals.treelite cimport *


cdef extern from "cuml/explainer/tree_shap.hpp" namespace "ML::Explainer" nogil:
    cdef cppclass TreePathHandle:
        pass

    cdef cppclass FloatPointer:
        pass

    cdef TreePathHandle extract_path_info(TreeliteModelHandle model) except +
    cdef void gpu_treeshap(TreePathHandle  path_info,
                           const FloatPointer data,
                           size_t n_rows,
                           size_t n_cols,
                           FloatPointer out_preds,
                           size_t out_preds_size) except +

    cdef void gpu_treeshap_interventional(TreePathHandle path_info,
                                          const FloatPointer data,
                                          size_t n_rows,
                                          size_t n_cols,
                                          const FloatPointer background_data,
                                          size_t background_n_rows,
                                          size_t background_n_cols,
                                          FloatPointer out_preds,
                                          size_t out_preds_size) except +

    cdef void gpu_treeshap_interactions(TreePathHandle  path_info,
                                        const FloatPointer data,
                                        size_t n_rows,
                                        size_t n_cols,
                                        FloatPointer out_preds,
                                        size_t out_preds_size) except +

    cdef void gpu_treeshap_taylor_interactions(TreePathHandle  path_info,
                                               const FloatPointer data,
                                               size_t n_rows,
                                               size_t n_cols,
                                               FloatPointer out_preds,
                                               size_t out_preds_size) except +
cdef FloatPointer type_erase_float_ptr(array):
    cdef FloatPointer ptr
    if array.dtype == np.float32:
        ptr = <FloatPointer > <float*> < uintptr_t > array.data.ptr
    elif array.dtype == np.float64:
        ptr = <FloatPointer > <double*> < uintptr_t > array.data.ptr
    else:
        raise ValueError("Unsupported dtype")
    return ptr


def _is_host_container(x):
    """Returns if the data is a host-side container type"""
    if isinstance(x, (pd.DataFrame, pd.Series, np.ndarray)):
        return True
    elif isinstance(x, (cudf.DataFrame, cudf.Series, cp.ndarray)):
        return False
    # All other cases cuda memory types should have this
    return not hasattr(x, "__cuda_array_interface__")


cdef class TreeExplainer:
    """
    Model explainer that calculates Shapley values for the predictions of
    tree-based models. Shapley values are a method of attributing various input
    features to a given model prediction.

    Uses GPUTreeShap [1]_ as a back-end to accelerate computation using GPUs.

    Different variants of Shapley values exist based on different
    interpretations of marginalising out (or conditioning on) features. For the
    "tree_path_dependent" approach, see [2]_.

    For the "interventional" approach, see [3]_.

    We also provide two variants of feature interactions. For the
    "shapley-interactions" variant of interactions, see [2]_, for
    the "shapley-taylor" variant, see [4]_.


    .. [1] Mitchell, Rory, Eibe Frank, and Geoffrey Holmes. "GPUTreeShap:
        massively parallel exact calculation of SHAP scores for tree
        ensembles." PeerJ Computer Science 8 (2022): e880.

    .. [2] Lundberg, Scott M., et al. "From local explanations to global
        understanding with explainable AI for trees." Nature machine
        intelligence 2.1 (2020): 56-67.

    .. [3] Janzing, Dominik, Lenon Minorics, and Patrick Blöbaum. "Feature
        relevance quantification in explainable AI: A causal problem."
        International Conference on artificial intelligence and statistics.
        PMLR, 2020.

    .. [4] Sundararajan, Mukund, Kedar Dhamdhere, and Ashish Agarwal.
        "The Shapley Taylor Interaction Index." International Conference
        on Machine Learning. PMLR, 2020.


    Parameters
    ----------
    model : model object
        The tree based machine learning model. XGBoost, LightGBM, cuml random
        forest and sklearn random forest models are supported. Categorical
        features in XGBoost or LightGBM models are natively supported.
    data : array or DataFrame
        Optional background dataset to use for marginalising out features.
        If this argument is supplied, an "interventional" approach is used.
        Computation time increases with the size of this background data set,
        consider starting with between 100-1000 examples. If this argument is
        not supplied, statistics from the tree model are used to marginalise
        out features ("tree_path_dependent").

    Attributes
    ----------
    expected_value :
        Model prediction when all input features are marginalised out. Is a
        vector for multiclass problems.

    Examples
    --------

    .. code-block:: python

        >>> import numpy as np
        >>> import cuml
        >>> from cuml.explainer import TreeExplainer
        >>> X = np.array([[0.0, 2.0], [1.0, 0.5]])
        >>> y = np.array([0, 1])
        >>> model = cuml.ensemble.RandomForestRegressor().fit(X, y)
        >>> explainer = TreeExplainer(model=model)
        >>> shap_values = explainer.shap_values(X)

    """
    cdef public object expected_value
    cdef TreePathHandle path_info
    cdef object num_class
    cdef object data

    def __init__(self, *, model, data=None):
        if data is not None:
            data = check_array(
                data,
                dtype=("float32", "float64"),
                order="C",
                ensure_all_finite=False,
            )
        self.data = data

        # Handle various kinds of tree model objects
        cls = model.__class__
        cls_module, cls_name = cls.__module__, cls.__name__
        # XGBoost model object
        if re.match(r'xgboost.*$', cls_module):
            if cls_name != 'Booster':
                model = model.get_booster()
            tl_model = treelite.frontend.from_xgboost(model)
        # LightGBM model object
        elif re.match(r'lightgbm.*$', cls_module):
            if cls_name != 'Booster':
                model = model.booster_
            tl_model = treelite.frontend.from_lightgbm(model)
        # cuML RF model object
        elif isinstance(model, (cuml.RandomForestClassifier, cuml.RandomForestRegressor)):
            tl_model = model.as_treelite()
        # scikit-learn RF model object
        elif isinstance(model, treelite.Model):
            tl_model = model
        else:
            from sklearn.ensemble import (
                RandomForestClassifier,
                RandomForestRegressor,
            )
            if isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
                tl_model = treelite.sklearn.import_model(model)
            else:
                raise ValueError(f"Unrecognized model object type: {type(model)}")

        # Get num_class
        self.num_class = tl_model.get_header_accessor().get_field("num_class").copy()
        if len(self.num_class) > 1:
            raise NotImplementedError("TreeExplainer does not support multi-target models")

        # Serialize Treelite model object and de-serialize again,
        # to get around C++ ABI incompatibilities (due to different compilers
        # being used to build cuML pip wheel vs. Treelite pip wheel)
        bytes_seq = tl_model.serialize_bytes()
        cdef TreeliteModelHandle tl_handle = NULL
        safe_treelite_call(
            TreeliteDeserializeModelFromBytes(bytes_seq, len(bytes_seq), &tl_handle),
            "Failed to load Treelite model from bytes"
        )

        # Process Treelite model to extract path info
        self.path_info = extract_path_info(tl_handle)

    def shap_values(self, X):
        """
        Estimate the SHAP values for a set of samples. For a given row, the
        SHAP values plus the `expected_value` attribute sum up to the raw
        model prediction. 'Raw model prediction' means before the application
        of a link function, for example, the SHAP values of an XGBoost binary
        classification will be in the additive logit space as opposed to
        probability space.

        Parameters
        ----------
        X :
            A matrix of samples (# samples x # features) on which to explain
            the model's output.

        Returns
        -------
        array
            Returns a matrix of SHAP values of shape
            (# classes x # samples x # features).
        """
        return_numpy = _is_host_container(X)
        X = check_array(
            X,
            dtype=("float32", "float64"),
            order="C",
            ensure_all_finite=False,
        )

        n_rows, n_cols = X.shape
        dtype = X.dtype

        preds = cp.empty(
            (n_rows, self.num_class[0] * (n_cols + 1)),
            dtype=dtype,
            order="C",
        )

        if self.data is None:
            gpu_treeshap(self.path_info, type_erase_float_ptr(X),
                         < size_t > n_rows, < size_t > n_cols,
                         type_erase_float_ptr(preds), preds.size)
        else:
            if self.data.dtype != dtype:
                raise ValueError(
                    "Expected background data to have the same dtype as X.")
            gpu_treeshap_interventional(
                self.path_info,
                type_erase_float_ptr(X),
                < size_t > n_rows, < size_t > n_cols,
                type_erase_float_ptr(self.data),
                < size_t > self.data.shape[0], < size_t > self.data.shape[1],
                type_erase_float_ptr(preds), preds.size)

        # Reshape to 3D as appropriate
        # To follow the convention of the SHAP package:
        # 1. Store the bias term in the 'expected_value' attribute.
        # 2. Transpose SHAP values in dimension (row_id, feature_id, group_id)
        # Note. The layout of the SHAP values from the `shap` package changed
        #       in version 0.45.0. We use the changed layout.
        if return_numpy:
            preds = preds.get()
        if self.num_class[0] > 1:
            preds = preds.reshape(
                (n_rows, self.num_class[0], n_cols + 1))
            preds = preds.transpose((0, 2, 1))
            self.expected_value = preds[0, -1, :]
            preds = preds[:, :-1, :]
        else:
            assert self.num_class[0] == 1
            self.expected_value = preds[0, -1]
            preds = preds[:, :-1]
        return preds

    def shap_interaction_values(self, X, method='shapley-interactions'):
        """
        Estimate the SHAP interaction values for a set of samples. For a
        given row, the SHAP values plus the `expected_value` attribute sum
        up to the raw model prediction. 'Raw model prediction' means before
        the application of a link function, for example, the SHAP values of
        an XGBoost binary classification are in the additive logit space as
        opposed to probability space.

        Interventional feature marginalisation is not supported.

        Parameters
        ----------
        X :
            A matrix of samples (# samples x # features) on which to explain
            the model's output.
        method :
            One of ['shapley-interactions', 'shapley-taylor']

        Returns
        -------
        array
            Returns a matrix of SHAP values of shape
            (# classes x # samples x # features x # features).
        """
        return_numpy = _is_host_container(X)
        X = check_array(
            X,
            dtype=("float32", "float64"),
            order="C",
            ensure_all_finite=False,
        )

        n_rows, n_cols = X.shape
        dtype = X.dtype

        preds = cp.empty(
            (n_rows, self.num_class[0] * (n_cols + 1)**2),
            dtype=dtype,
            order="C",
        )

        if self.data is None:
            if method == 'shapley-interactions':
                gpu_treeshap_interactions(
                    self.path_info,
                    type_erase_float_ptr(X),
                    < size_t > n_rows, < size_t > n_cols,
                    type_erase_float_ptr(preds), preds.size)
            elif method == 'shapley-taylor':
                gpu_treeshap_taylor_interactions(
                    self.path_info, type_erase_float_ptr(X),
                    < size_t > n_rows, < size_t > n_cols,
                    type_erase_float_ptr(preds), preds.size)
            else:
                raise ValueError("Unknown interactions method.")
        else:
            raise ValueError(
                "Interventional algorithm not supported for interactions."
                " Please specify data as None in constructor.")

        if return_numpy:
            preds = preds.get()
        if self.num_class[0] > 1:
            preds = preds.reshape(
                (n_rows, self.num_class[0], n_cols + 1, n_cols + 1))
            preds = preds.transpose((1, 0, 2, 3))
            self.expected_value = preds[:, 0, -1, -1]
            preds = preds[:, :, :-1, :-1]
        else:
            assert self.num_class[0] == 1
            preds = preds.reshape(
                (n_rows,  n_cols + 1, n_cols + 1))
            self.expected_value = preds[0, -1, -1]
            preds = preds[:, :-1, :-1]
        return preds
