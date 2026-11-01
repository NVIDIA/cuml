# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""GPU-backed proxy objects that mimic sklearn DecisionTree estimators.

These are exposed via ``RandomForest{Classifier,Regressor}.estimators_``
and route inference through FIL while exposing the full tree structure
(topology, node values, impurity) extracted from the treelite model.
"""

import cupy as cp
import numpy as np
import treelite
from scipy.sparse import csr_matrix

from cuml.common.classification import decode_labels
from cuml.internals.base import Base
from cuml.internals.mixins import ClassifierMixin, RegressorMixin
from cuml.internals.outputs import (
    exit_internal_context,
    reflect,
    run_in_internal_context,
)


def _compute_max_depth(children_left, children_right):
    """Compute the max depth of a tree from its children arrays."""
    n_nodes = len(children_left)
    if n_nodes == 0:
        return 0
    depths = np.zeros(n_nodes, dtype=np.intp)
    for i in range(n_nodes):
        left = children_left[i]
        if left != -1:
            depths[left] = depths[i] + 1
            depths[children_right[i]] = depths[i] + 1
    return int(depths.max())


def _extract_tree_from_treelite(tl_model, tree_idx, n_classes, is_classifier):
    """Extract sklearn-compatible tree arrays from a treelite tree accessor.

    Returns a dict with keys matching sklearn Tree attributes.
    """
    ta = tl_model.get_tree_accessor(tree_idx)
    header = tl_model.get_header_accessor()

    node_type = ta.get_field("node_type")  # 1=internal, 0=leaf
    n_nodes = int(ta.get_field("num_nodes")[0])
    cleft = ta.get_field("cleft").astype(np.intp)
    cright = ta.get_field("cright").astype(np.intp)
    split_index = ta.get_field("split_index").astype(np.intp)
    threshold_raw = ta.get_field("threshold")
    data_count = ta.get_field("data_count").astype(np.intp)
    leaf_vector = ta.get_field("leaf_vector")

    n_features = int(header.get_field("num_feature")[0])
    is_leaf = node_type == 0

    # Convert to sklearn conventions
    children_left = cleft.copy()
    children_right = cright.copy()
    children_left[is_leaf] = -1
    children_right[is_leaf] = -1

    feature = split_index.copy()
    feature[is_leaf] = -2

    threshold = threshold_raw.copy().astype(np.float64)
    threshold[is_leaf] = -2.0

    n_node_samples = data_count.copy()

    n_outputs = 1
    if is_classifier:
        leaf_probs = leaf_vector.reshape(-1, n_classes)

        value = np.zeros((n_nodes, n_outputs, n_classes), dtype=np.float64)
        leaf_counter = 0
        for i in range(n_nodes):
            if is_leaf[i]:
                value[i, 0, :] = leaf_probs[leaf_counter] * data_count[i]
                leaf_counter += 1

        for i in range(n_nodes - 1, -1, -1):
            if not is_leaf[i]:
                value[i, 0, :] = value[cleft[i], 0, :] + value[cright[i], 0, :]
    else:
        n_classes = 1
        leaf_vals = ta.get_field("leaf_value").astype(np.float64)

        value = np.zeros((n_nodes, n_outputs, 1), dtype=np.float64)
        for i in range(n_nodes):
            if is_leaf[i]:
                value[i, 0, 0] = leaf_vals[i] * data_count[i]

        for i in range(n_nodes - 1, -1, -1):
            if not is_leaf[i]:
                value[i, 0, 0] = value[cleft[i], 0, 0] + value[cright[i], 0, 0]

    # Impurity: Gini for classification, variance for regression
    impurity = np.zeros(n_nodes, dtype=np.float64)
    if is_classifier:
        for i in range(n_nodes):
            total = value[i, 0, :].sum()
            if total > 0:
                p = value[i, 0, :] / total
                impurity[i] = 1.0 - np.sum(p**2)
    else:
        pass  # TODO: compute variance-based impurity for regression

    max_depth = _compute_max_depth(children_left, children_right)

    return {
        "children_left": children_left,
        "children_right": children_right,
        "feature": feature,
        "threshold": threshold,
        "value": value,
        "impurity": impurity,
        "n_node_samples": n_node_samples,
        "weighted_n_node_samples": n_node_samples.astype(np.float64),
        "node_count": n_nodes,
        "n_features": n_features,
        "n_classes": np.array(
            [n_classes] if is_classifier else [1], dtype=np.intp
        ),
        "n_outputs": n_outputs,
        "max_depth": max_depth,
    }


class GPUTree:
    """Duck-typed proxy for ``sklearn.tree._tree.Tree``.

    Exposes tree structure arrays and routes ``predict``/``apply`` through FIL.
    """

    def __init__(self, tree_data, fil_model, tree_idx, parent_estimator):
        self._fil_model = fil_model
        self._tree_idx = tree_idx
        self._parent_estimator = parent_estimator

        self.children_left = tree_data["children_left"]
        self.children_right = tree_data["children_right"]
        self.feature = tree_data["feature"]
        self.threshold = tree_data["threshold"]
        self.value = tree_data["value"]
        self.impurity = tree_data["impurity"]
        self.n_node_samples = tree_data["n_node_samples"]
        self.weighted_n_node_samples = tree_data["weighted_n_node_samples"]
        self.node_count = tree_data["node_count"]
        self.capacity = tree_data["node_count"]
        self.n_features = tree_data["n_features"]
        self.n_classes = tree_data["n_classes"]
        self.n_outputs = tree_data["n_outputs"]
        self.max_depth = tree_data["max_depth"]
        self.max_n_classes = int(self.n_classes.max())

    @property
    def n_leaves(self):
        return int(np.sum(self.children_left == -1))

    def _coerce_output(self, gpu_array, X):
        """Convert a cupy array to the appropriate output type based on the parent estimator."""
        output_type = self._parent_estimator._get_output_type(X)
        if output_type in (None, "cupy"):
            return gpu_array
        return cp.asnumpy(gpu_array)

    def predict(self, X):
        """Return per-class probabilities via FIL predict_per_tree."""
        X_gpu = cp.asarray(X, dtype=cp.float32)
        per_tree = self._fil_model.predict_per_tree(X_gpu)
        result = per_tree[:, self._tree_idx]
        if result.ndim == 1:
            result = result[:, np.newaxis]
        return self._coerce_output(result, X)

    def apply(self, X):
        """Return leaf node IDs via FIL apply."""
        X_gpu = cp.asarray(X, dtype=cp.float32)
        leaf_ids = self._fil_model.apply(X_gpu)
        result = leaf_ids[:, self._tree_idx].astype(cp.intp)
        return self._coerce_output(result, X)

    def decision_path(self, X):
        """CPU fallback: traverse the stored tree structure."""
        X = np.asarray(X, dtype=np.float32)
        n_samples = X.shape[0]
        indptr = [0]
        indices = []

        for sample_idx in range(n_samples):
            node = 0
            while node != -1:
                indices.append(node)
                if self.children_left[node] == -1:
                    break
                if X[sample_idx, self.feature[node]] <= self.threshold[node]:
                    node = self.children_left[node]
                else:
                    node = self.children_right[node]
            indptr.append(len(indices))

        return csr_matrix(
            (np.ones(len(indices), dtype=np.uint8), indices, indptr),
            shape=(n_samples, self.node_count),
        )

    def compute_feature_importances(self, normalize=True):
        importances = np.zeros(self.n_features, dtype=np.float64)
        is_leaf = self.children_left == -1
        for i in range(self.node_count):
            if is_leaf[i]:
                continue
            left = self.children_left[i]
            right = self.children_right[i]
            w = self.weighted_n_node_samples
            importances[self.feature[i]] += (
                w[i] * self.impurity[i]
                - w[left] * self.impurity[left]
                - w[right] * self.impurity[right]
            )
        if normalize:
            total = importances.sum()
            if total > 0:
                importances /= total
        return importances


def _build_gpu_estimators(rf_model):
    """Build list of GPU-backed DecisionTree proxy objects from a fitted RF."""
    is_classifier = rf_model._estimator_type == "classifier"
    tl_model = treelite.Model.deserialize_bytes(rf_model._treelite_model_bytes)
    fil_model = rf_model._get_inference_nvforest_model()

    header = tl_model.get_header_accessor()
    n_classes = int(header.get_field("num_class")[0])
    n_features = int(header.get_field("num_feature")[0])
    n_trees = tl_model.num_tree

    common_params = dict(
        split_criterion=rf_model.split_criterion,
        max_depth=rf_model.max_depth,
        min_samples_split=rf_model.min_samples_split,
        min_samples_leaf=rf_model.min_samples_leaf,
        max_features=rf_model.max_features,
        max_leaves=rf_model.max_leaves,
        min_impurity_decrease=rf_model.min_impurity_decrease,
        random_state=rf_model.random_state,
    )

    estimators = []
    for tree_idx in range(n_trees):
        tree_data = _extract_tree_from_treelite(
            tl_model,
            tree_idx,
            n_classes,
            is_classifier,
        )
        if is_classifier:
            est = GPUDecisionTreeClassifier(**common_params)
            est.classes_ = rf_model.classes_
            est.n_classes_ = rf_model.n_classes_
        else:
            est = GPUDecisionTreeRegressor(**common_params)

        est.tree_ = GPUTree(
            tree_data, fil_model, tree_idx, parent_estimator=est
        )
        est.n_features_in_ = n_features
        est.n_outputs_ = 1
        est.max_features_ = n_features
        est._fil_model = fil_model
        est._tree_idx = tree_idx
        estimators.append(est)

    return estimators


class GPUDecisionTreeClassifier(Base, ClassifierMixin):
    """GPU-backed proxy for a single decision tree classifier in a forest.

    This class is used to represent a fitted decision tree in the estimators_
    attribute of a RandomForestClassifier. The goal is to make the internal
    structure of the forest accessible. It is not a fully functional estimator.
    For example, it does not have a `fit` method.
    """

    def __init__(
        self,
        split_criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        max_leaves=-1,
        min_impurity_decrease=0.0,
        random_state=None,
        verbose=False,
        output_type=None,
    ):
        super().__init__(verbose=verbose, output_type=output_type)
        self.split_criterion = split_criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaves = max_leaves
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "split_criterion",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "max_leaves",
            "min_impurity_decrease",
            "random_state",
        ]

    @run_in_internal_context
    def predict(self, X):
        X_gpu = cp.asarray(X, dtype=cp.float32)
        per_tree = self._fil_model.predict_per_tree(X_gpu)
        proba = per_tree[:, self._tree_idx]
        inds = cp.argmax(proba, axis=1)
        with exit_internal_context():
            output_type = self._get_output_type(X)
        return decode_labels(inds, self.classes_, output_type=output_type)

    @reflect
    def predict_proba(self, X):
        X_gpu = cp.asarray(X, dtype=cp.float32)
        per_tree = self._fil_model.predict_per_tree(X_gpu)
        return per_tree[:, self._tree_idx]

    @reflect
    def apply(self, X):
        X_gpu = cp.asarray(X, dtype=cp.float32)
        leaf_ids = self._fil_model.apply(X_gpu)
        return leaf_ids[:, self._tree_idx]

    @property
    def feature_importances_(self):
        return self.tree_.compute_feature_importances()


class GPUDecisionTreeRegressor(Base, RegressorMixin):
    """GPU-backed proxy for a single decision tree regressor in a forest.

    This class is used to represent a fitted decision tree in the estimators_
    attribute of a RandomForestRegressor. The goal is to make the internal
    structure of the forest accessible. It is not a fully functional estimator.
    For example, it does not have a `fit` method.
    """

    def __init__(
        self,
        split_criterion="mse",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        max_leaves=-1,
        min_impurity_decrease=0.0,
        random_state=None,
        verbose=False,
        output_type=None,
    ):
        super().__init__(verbose=verbose, output_type=output_type)
        self.split_criterion = split_criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaves = max_leaves
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "split_criterion",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "max_leaves",
            "min_impurity_decrease",
            "random_state",
        ]

    @reflect
    def predict(self, X):
        X_gpu = cp.asarray(X, dtype=cp.float32)
        per_tree = self._fil_model.predict_per_tree(X_gpu)
        result = per_tree[:, self._tree_idx]
        if result.ndim > 1:
            result = result.squeeze(axis=-1)
        return result

    @reflect
    def apply(self, X):
        X_gpu = cp.asarray(X, dtype=cp.float32)
        leaf_ids = self._fil_model.apply(X_gpu)
        return leaf_ids[:, self._tree_idx]

    @property
    def feature_importances_(self):
        return self.tree_.compute_feature_importances()
