#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import builtins
import math
import numbers
import warnings
from typing import Literal

import cupy as cp
import numpy as np
import treelite.sklearn

from cuml.internals.base import Base, get_handle
from cuml.internals.interop import (
    InteropMixin,
    UnsupportedOnCPU,
    UnsupportedOnGPU,
)
from cuml.internals.treelite import safe_treelite_call
from cuml.internals.validation import check_is_fitted, check_random_seed
from cuml.metrics import accuracy_score, r2_score

from libc.stdint cimport uint64_t, uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t
import nvforest

from cuml.internals.logger cimport level_enum
from cuml.internals.treelite cimport (
    TreeliteFreeModel,
    TreeliteModelHandle,
    TreeliteSerializeModelToBytes,
)


cdef extern from "cuml/ensemble/randomforest.hpp" namespace "ML" nogil:
    cdef enum CRITERION:
        GINI,
        ENTROPY,
        MSE,
        MAE,
        POISSON,
        GAMMA,
        INVERSE_GAUSSIAN,
        CRITERION_END

    cdef struct RF_params:
        pass

    cdef RF_params set_rf_params(
        int max_depth,
        int max_leaves,
        float max_features,
        int max_n_bins,
        int min_samples_leaf,
        int min_samples_split,
        float min_impurity_decrease,
        bool bootstrap,
        int n_trees,
        float max_samples,
        uint64_t seed,
        CRITERION split_criterion,
        int cfg_n_streams,
        int max_batch_size
    ) except +

    cdef void fit_treelite[T, L](
        handle_t& handle,
        TreeliteModelHandle* model,
        T* values,
        int n_rows,
        int n_cols,
        L* labels,
        int n_unique_labels,
        RF_params params,
        bool* bootstrap_masks,
        T* feature_importances,
        level_enum verbosity,
        const double* sample_weight,
        bool input_row_major
    ) except +

    cdef void fit_treelite[T, L](
        handle_t& handle,
        TreeliteModelHandle* model,
        T* values,
        int n_rows,
        int n_cols,
        L* labels,
        RF_params params,
        bool* bootstrap_masks,
        T* feature_importances,
        level_enum verbosity,
        const double* sample_weight,
        bool input_row_major
    ) except +


_split_criterion_lookup = {
    "0": GINI,
    "gini": GINI,
    "1": ENTROPY,
    "entropy": ENTROPY,
    "2": MSE,
    "mse": MSE,
    "3": MAE,
    "mae": MAE,
    "4": POISSON,
    "poisson": POISSON,
    "5": GAMMA,
    "gamma": GAMMA,
    "6": INVERSE_GAUSSIAN,
    "inverse_gaussian": INVERSE_GAUSSIAN,
    "7": CRITERION_END
}

_criterion_to_split_criterion = {
    "gini": "gini",
    "entropy": "entropy",
    "squared_error": "mse",
    "poisson": "poisson",
}

_split_criterion_to_criterion = {
    "gini": "gini",
    0: "gini",
    "entropy": "entropy",
    1: "entropy",
    "mse": "squared_error",
    2: "squared_error",
    "poisson": "poisson",
    4: "poisson",
}


def _normalize_split_criterion(split_criterion):
    if (out := _split_criterion_lookup.get(str(split_criterion))) is None:
        warnings.warn(
            "The split criterion chosen was not present in the list of options accepted "
            "by the model and so the CRITERION_END option has been chosen."
        )
        return CRITERION_END
    if out == MAE:
        raise NotImplementedError(
            "cuML does not currently support mean average error as a"
            " RandomForest split criterion"
        )
    return out


def compute_max_features(
    max_features: Literal["sqrt", "log2", None] | int | float,
    n_cols: int,
) -> float:
    if isinstance(max_features, int):
        return max_features / n_cols
    elif isinstance(max_features, float):
        return max_features
    elif max_features == 'sqrt':
        return math.sqrt(n_cols) / n_cols
    elif max_features == 'log2':
        return math.log2(n_cols) / n_cols
    elif max_features is None:
        return 1.0
    else:
        raise ValueError(
            f"Expected `max_features` to be an int, float, None, or one of "
            f"['sqrt', 'log2']. Got {max_features!r} instead."
        )


class BaseRandomForestModel(InteropMixin, Base):

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "split_criterion",
            "n_estimators",
            "bootstrap",
            "max_samples",
            "max_depth",
            "max_leaves",
            "max_features",
            "n_bins",
            "min_samples_leaf",
            "min_samples_split",
            "min_impurity_decrease",
            "max_batch_size",
            "random_state",
            "n_streams",
            "oob_score",
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        if model.oob_score and callable(model.oob_score):
            raise UnsupportedOnGPU("callable `oob_score` is not supported")

        if model.warm_start:
            raise UnsupportedOnGPU("`warm_start=True` is not supported")

        if model.monotonic_cst is not None:
            raise UnsupportedOnGPU(f"`monotonic_cst={model.monotonic_cst!r} is not supported")

        if model.ccp_alpha != 0:
            raise UnsupportedOnGPU(f"`ccp_alpha={model.ccp_alpha}` is not supported")

        if model.min_weight_fraction_leaf != 0:
            raise UnsupportedOnGPU(
                f"`min_weight_fraction_leaf={model.min_weight_fraction_leaf}` is not supported"
            )

        if (split_criterion := _criterion_to_split_criterion.get(model.criterion)) is None:
            raise UnsupportedOnGPU(f"`criterion={model.criterion!r}` is not supported")

        # We only forward some parameters, falling back to cuml defaults otherwise
        conditional_params = {}

        if isinstance(model.max_samples, int):
            raise UnsupportedOnGPU("`int` values for `max_samples` are not supported")
        elif model.max_samples is not None:
            conditional_params["max_samples"] = model.max_samples

        return {
            "n_estimators": model.n_estimators,
            "split_criterion": split_criterion,
            "max_depth": model.max_depth,
            "min_samples_split": model.min_samples_split,
            "min_samples_leaf": model.min_samples_leaf,
            "max_features": model.max_features,
            "max_leaves": -1 if model.max_leaf_nodes is None else model.max_leaf_nodes,
            "min_impurity_decrease": model.min_impurity_decrease,
            "bootstrap": model.bootstrap,
            "random_state": model.random_state,
            "oob_score": model.oob_score,
            **conditional_params
        }

    def _params_to_cpu(self):
        if (criterion := _split_criterion_to_criterion.get(self.split_criterion)) is None:
            raise UnsupportedOnCPU(
                f"`split_criterion={self.split_criterion!r}` is not supported"
            )

        return {
            "n_estimators": self.n_estimators,
            "criterion": criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "max_leaf_nodes": None if self.max_leaves == -1 else self.max_leaves,
            "min_impurity_decrease": self.min_impurity_decrease,
            "bootstrap": self.bootstrap,
            "random_state": self.random_state,
            "max_samples": self.max_samples,
            "oob_score": self.oob_score,
        }

    def _attrs_from_cpu(self, model):
        tl_model = treelite.sklearn.import_model(model)
        attrs = {
            "_treelite_model_bytes": tl_model.serialize_bytes(),
            "n_outputs_": model.n_outputs_,
            "_n_samples": model._n_samples,
            "_n_samples_bootstrap": model._n_samples_bootstrap,
            **super()._attrs_from_cpu(model)
        }
        # Transfer OOB attributes if present
        if hasattr(model, 'oob_score_'):
            attrs["oob_score_"] = model.oob_score_
        if hasattr(model, 'oob_decision_function_'):
            attrs["oob_decision_function_"] = cp.asarray(model.oob_decision_function_)
        if hasattr(model, 'oob_prediction_'):
            attrs["oob_prediction_"] = cp.asarray(model.oob_prediction_)
        # Note: feature_importances_ is NOT transferred from sklearn to cuML
        # because cuML caches the impurity decrease directly (BestMetric()) which
        # is not available in sklearn models created via treelite export
        return attrs

    def _attrs_to_cpu(self, model):
        tl_model = treelite.Model.deserialize_bytes(self._treelite_model_bytes)
        sk_model = treelite.sklearn.export_model(tl_model)
        attrs = {
            "estimator_": sk_model.estimator,
            "estimators_": sk_model.estimators_,
            "n_outputs_": self.n_outputs_,
            "_n_samples": self._n_samples,
            "_n_samples_bootstrap": self._n_samples_bootstrap,
            **super()._attrs_to_cpu(model)
        }
        # Transfer OOB attributes if present
        if hasattr(self, 'oob_score_'):
            attrs["oob_score_"] = self.oob_score_
        if hasattr(self, 'oob_decision_function_'):
            attrs["oob_decision_function_"] = self.oob_decision_function_.get(order="A")
        if hasattr(self, 'oob_prediction_'):
            attrs["oob_prediction_"] = self.oob_prediction_.get(order="A")
        # Note: feature_importances_ is NOT transferred from cuML to sklearn
        # because sklearn's computation requires tree impurities that aren't
        # available in sklearn models created via treelite export
        return attrs

    def __init__(
        self,
        *,
        split_criterion,
        n_estimators=100,
        bootstrap=True,
        max_samples=1.0,
        max_depth=None,
        max_leaves=-1,
        max_features='sqrt',
        n_bins=128,
        min_samples_leaf=1,
        min_samples_split=2,
        min_impurity_decrease=0.0,
        max_batch_size=4096,
        random_state=None,
        n_streams=4,
        oob_score=False,
        verbose=False,
        output_type=None,
    ):
        super().__init__(verbose=verbose, output_type=output_type)

        self.split_criterion = split_criterion
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.max_features = max_features
        self.n_bins = n_bins
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_batch_size = max_batch_size
        self.random_state = random_state
        self.n_streams = n_streams
        self.oob_score = oob_score

    def __getstate__(self):
        state = self.__dict__.copy()
        # nvForest model isn't currently pickleable
        state.pop("_nvforest_model", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __len__(self):
        """Return the number of estimators in the ensemble."""
        return self.n_estimators

    def as_treelite(self):
        """
        Converts this estimator to a Treelite model.

        Returns
        -------
        treelite.Model
        """
        check_is_fitted(self)

        return treelite.Model.deserialize_bytes(self._treelite_model_bytes)

    def as_nvforest(
        self, layout="depth_first", default_chunk_size=None, align_bytes=None,
    ):
        """
        Create a nvForest model from the cuML model.

        Parameters
        ----------
        layout : string (default = 'depth_first')
            Specifies the in-memory layout of nodes in forests. Options:
            'depth_first', 'layered', 'breadth_first'.
        default_chunk_size : int, optional (default = None)
            Determines how batches are further subdivided for parallel processing.
            The optimal value depends on hardware, model, and batch size.
            If None, will be automatically determined.
        align_bytes : int, optional (default = None)
            If specified, trees will be padded such that their in-memory size is
            a multiple of this value. This can improve performance by guaranteeing
            that memory reads from trees begin on a cache line boundary.
            Typical values are 0 or 128 on GPU and 0 or 64 on CPU.

        Returns
        -------
        nvforest_model : nvforest.ForestInference
            A forest inference model which can be used to perform
            inferencing on the random forest model.
        """
        check_is_fitted(self)

        return nvforest.load_from_treelite_model(
            tl_model=treelite.Model.deserialize_bytes(self._treelite_model_bytes),
            device="gpu",
            layout=layout,
            default_chunk_size=default_chunk_size,
            align_bytes=align_bytes,
            handle=get_handle(),
        )

    def _fit_forest(self, X, y, sample_weight=None):
        cdef bool is_classifier = self._estimator_type == "classifier"
        cdef bool is_float32 = X.dtype == np.float32

        cdef uintptr_t X_ptr = X.data.ptr
        cdef uintptr_t y_ptr = y.data.ptr
        cdef uintptr_t sample_weight_ptr = (
            0 if sample_weight is None else sample_weight.data.ptr
        )
        cdef int n_rows = X.shape[0]
        cdef int n_cols = X.shape[1]
        cdef level_enum verbose = <level_enum> self._verbose_level
        cdef int n_classes = self.n_classes_ if is_classifier else 0
        cdef bool input_row_major = not X.flags.f_contiguous

        cdef int max_depth_c
        max_depth = self.max_depth

        if max_depth is None:
            max_depth_c = np.iinfo(np.int32).max
        elif not isinstance(max_depth, int) or max_depth <= 0:
            raise ValueError(
                f"max_depth must be a positive integer or None (unlimited); "
                f"got {max_depth!r}"
            )
        else:
            max_depth_c = max_depth

        # Validate OOB score parameter
        if callable(self.oob_score):
            raise ValueError(
                "oob_score must be a boolean. "
                "Custom scorer functions are not supported."
            )

        # Validate OOB score requirements
        if self.oob_score and not self.bootstrap:
            raise ValueError("Out of bag estimation only available if bootstrap=True")

        n_streams = self.n_streams
        if (
            isinstance(n_streams, (builtins.bool, np.bool_))
            or not isinstance(n_streams, numbers.Integral)
        ):
            raise TypeError(
                f"n_streams must be a positive integer; got {n_streams!r}"
            )
        if n_streams <= 0:
            raise ValueError(
                f"n_streams must be a positive integer; got {n_streams!r}"
            )

        cdef int n_streams_c = n_streams

        cdef float max_features = compute_max_features(self.max_features, n_cols)
        cdef uint64_t seed = (
            0 if self.random_state is None
            else check_random_seed(self.random_state)
        )
        cdef int min_samples_leaf = (
            self.min_samples_leaf if isinstance(self.min_samples_leaf, int)
            else math.ceil(self.min_samples_leaf * n_rows)
        )
        cdef int min_samples_split = (
            self.min_samples_split if isinstance(self.min_samples_split, int)
            else max(2, math.ceil(self.min_samples_split * n_rows))
        )

        cdef int n_bins
        if self.n_bins > n_rows:
            warnings.warn("The number of bins, `n_bins` is greater than "
                          "the number of samples used for training. "
                          "Changing `n_bins` to number of training samples.")
            n_bins = n_rows
        else:
            n_bins = self.n_bins

        cdef RF_params params = set_rf_params(
            max_depth_c,
            self.max_leaves,
            max_features,
            n_bins,
            min_samples_leaf,
            min_samples_split,
            self.min_impurity_decrease,
            self.bootstrap,
            self.n_estimators,
            self.max_samples,
            seed,
            _normalize_split_criterion(self.split_criterion),
            n_streams_c,
            self.max_batch_size,
        )

        cdef TreeliteModelHandle tl_handle
        handle = get_handle(n_streams=n_streams_c)
        cdef handle_t* handle_ = <handle_t*><uintptr_t>handle.getHandle()

        # Store oob_score in C variable for nogil block
        cdef bool use_oob_score = self.oob_score

        # Allocate buffer for bootstrap masks if OOB score is enabled
        bootstrap_masks_cp = None
        cdef bool* bootstrap_masks_ptr = NULL
        cdef uintptr_t masks_ptr_val = 0
        if use_oob_score:
            bootstrap_masks_cp = cp.zeros((self.n_estimators, n_rows), dtype=cp.bool_)
            masks_ptr_val = bootstrap_masks_cp.data.ptr
            bootstrap_masks_ptr = <bool*> masks_ptr_val

        cdef object feature_importances = np.empty(n_cols, dtype=X.dtype)
        cdef uintptr_t feature_importances_ptr = <uintptr_t> feature_importances.ctypes.data

        with nogil:
            if is_classifier:
                if is_float32:
                    fit_treelite(
                        handle_[0],
                        &tl_handle,
                        <float*> X_ptr,
                        n_rows,
                        n_cols,
                        <int*> y_ptr,
                        n_classes,
                        params,
                        bootstrap_masks_ptr,
                        <float*> feature_importances_ptr,
                        verbose,
                        <const double*> sample_weight_ptr,
                        input_row_major
                    )
                else:
                    fit_treelite(
                        handle_[0],
                        &tl_handle,
                        <double*> X_ptr,
                        n_rows,
                        n_cols,
                        <int*> y_ptr,
                        n_classes,
                        params,
                        bootstrap_masks_ptr,
                        <double*> feature_importances_ptr,
                        verbose,
                        <const double*> sample_weight_ptr,
                        input_row_major
                    )
            else:
                if is_float32:
                    fit_treelite(
                        handle_[0],
                        &tl_handle,
                        <float*> X_ptr,
                        n_rows,
                        n_cols,
                        <float*> y_ptr,
                        params,
                        bootstrap_masks_ptr,
                        <float*> feature_importances_ptr,
                        verbose,
                        <const double*> sample_weight_ptr,
                        input_row_major
                    )
                else:
                    fit_treelite(
                        handle_[0],
                        &tl_handle,
                        <double*> X_ptr,
                        n_rows,
                        n_cols,
                        <double*> y_ptr,
                        params,
                        bootstrap_masks_ptr,
                        <double*> feature_importances_ptr,
                        verbose,
                        <const double*> sample_weight_ptr,
                        input_row_major
                    )

        # XXX: Theoretically we could wrap `tl_handle` with `treelite.Model` to
        # manage ownership, and keep the loaded model around. However, this
        # only works if the `libtreelite` is ABI compatible with the one used
        # by `cuml`. This is currently true for conda environments, but not for
        # wheels where `cuml` and `treelite` use different manylinux ABIs. So
        # for now we need to do this serialize-and-reload dance. If/when this
        # is fixed we could instead store the loaded model and use that instead.
        cdef const char* tl_bytes = NULL
        cdef size_t tl_bytes_len
        safe_treelite_call(
            TreeliteSerializeModelToBytes(tl_handle, &tl_bytes, &tl_bytes_len),
            "Failed to serialize Treelite model to bytes:"
        )
        safe_treelite_call(
            TreeliteFreeModel(tl_handle), "Failed to free Treelite model:"
        )

        self._n_samples = y.shape[0]
        self._n_samples_bootstrap = (
            self._n_samples if self.max_samples is None
            else max(round(self._n_samples * self.max_samples), 1)
        )
        self.n_outputs_ = 1
        self._treelite_model_bytes = <bytes>(tl_bytes[:tl_bytes_len])
        # Reload nvforest model
        self._nvforest_model = self.as_nvforest()

        # Compute OOB score if requested
        if self.oob_score:
            self._compute_oob_score(X, y, bootstrap_masks_cp)

        self.feature_importances_ = feature_importances
        return self

    def _get_inference_nvforest_model(
        self,
        layout="depth_first",
        default_chunk_size=None,
        align_bytes=None,
    ):
        if (
            layout == "depth_first" and default_chunk_size is None and align_bytes is None
        ):
            # default parameters, get (or create) the cached nvforest model
            if (nvforest_model := getattr(self, "_nvforest_model", None)) is None:
                nvforest_model = self._nvforest_model = self.as_nvforest()
        else:
            nvforest_model = self.as_nvforest(
                layout=layout,
                default_chunk_size=default_chunk_size,
                align_bytes=align_bytes,
            )
        return nvforest_model

    def _compute_oob_score(self, X, y, bootstrap_masks_cp):
        """
        Compute OOB score using per-tree predictions and bootstrap masks.
        """
        # Get per-tree predictions using nvForest
        nvforest_model = self._get_inference_nvforest_model()
        # Per tree predictions shape: (n_samples, n_trees) for regression
        #                          or (n_samples, n_trees, n_classes) for classification
        per_tree_preds = nvforest_model.predict_per_tree(X)

        n_samples = X.shape[0]

        # Determine output shape based on prediction dimensionality
        # For regression: (n_samples,)
        # For classification: (n_samples, n_classes)
        output_shape = (n_samples,) + per_tree_preds.shape[2:]
        oob_predictions = cp.zeros(output_shape, dtype=cp.float64, order="C")
        oob_counts = cp.zeros(n_samples, dtype=cp.int32, order="C")

        # For each tree, accumulate predictions for OOB samples
        for tree_idx in range(self.n_estimators):
            # Get OOB mask for this tree (samples NOT in bootstrap)
            in_bag_mask = bootstrap_masks_cp[tree_idx]
            oob_mask = ~in_bag_mask

            # Accumulate predictions for OOB samples
            oob_predictions[oob_mask] += per_tree_preds[oob_mask, tree_idx]
            oob_counts[oob_mask] += 1

        # Average OOB predictions (broadcasting handles both 1D and 2D cases)
        valid_oob = oob_counts > 0

        # Warn if some samples don't have OOB predictions
        if not valid_oob.all():
            warnings.warn(
                "Some inputs do not have OOB scores. This probably means "
                "too few trees were used to compute any reliable OOB estimates.",
                UserWarning
            )

        if oob_predictions.ndim > 1:
            oob_predictions[valid_oob] /= oob_counts[valid_oob, cp.newaxis]
        else:
            oob_predictions[valid_oob] /= oob_counts[valid_oob]

        # Assign OOB predictions to the appropriate attribute based on estimator
        # type and compute the OOB score
        if self._estimator_type == "classifier":
            self.oob_decision_function_ = oob_predictions

            # Compute accuracy score for classification
            oob_pred_classes = cp.argmax(oob_predictions[valid_oob], axis=1)
            y_valid = y[valid_oob]
            self.oob_score_ = float(accuracy_score(y_valid, oob_pred_classes))
        else:
            self.oob_prediction_ = oob_predictions

            # Compute R² score for regression
            self.oob_score_ = float(r2_score(y[valid_oob], oob_predictions[valid_oob]))
