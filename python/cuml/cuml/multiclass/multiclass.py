# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import numpy as np

from cuml.common.doc_utils import generate_docstring
from cuml.internals.base import Base
from cuml.internals.mixins import ClassifierMixin
from cuml.internals.outputs import exit_internal_context, mlfunc
from cuml.internals.validation import check_inputs


def _fit_weighted_ovo(wrapper, X, y, sample_weight):
    """Fit one weighted estimator for each pair of classes."""
    from sklearn.base import clone
    from sklearn.utils.multiclass import check_classification_targets

    check_classification_targets(y)
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError(
            "OneVsOneClassifier can not be fit when only one class is present."
        )

    estimators = []
    pairwise_indices = []
    pairwise = wrapper.__sklearn_tags__().input_tags.pairwise
    for i, class_i in enumerate(classes):
        for class_j in classes[i + 1 :]:
            mask = (y == class_i) | (y == class_j)
            indices = np.flatnonzero(mask)
            X_binary = X[indices]
            if pairwise:
                X_binary = X_binary[:, indices]
            y_binary = (y[indices] == class_j).astype(np.int32)
            estimators.append(
                clone(wrapper.estimator).fit(
                    X_binary,
                    y_binary,
                    sample_weight=sample_weight[indices],
                )
            )
            pairwise_indices.append(indices)

    wrapper.classes_ = classes
    wrapper.estimators_ = estimators
    wrapper.pairwise_indices_ = pairwise_indices if pairwise else None
    return wrapper


def _fit_weighted_ovr(wrapper, X, y, sample_weight):
    """Fit one weighted estimator for each class against all other classes."""
    from sklearn.base import clone
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.utils.multiclass import check_classification_targets

    check_classification_targets(y)
    label_binarizer = LabelBinarizer(sparse_output=False)
    Y = label_binarizer.fit_transform(y)

    wrapper.label_binarizer_ = label_binarizer
    wrapper.classes_ = label_binarizer.classes_
    wrapper.estimators_ = [
        clone(wrapper.estimator).fit(
            X,
            y_binary,
            sample_weight=sample_weight,
        )
        for y_binary in Y.T
    ]
    return wrapper


class _BaseMulticlassClassifier(ClassifierMixin, Base):
    """Shared base class for multiclass classifiers"""

    def __init__(
        self,
        estimator,
        *,
        verbose=False,
        output_type=None,
    ):
        super().__init__(verbose=verbose, output_type=output_type)
        self.estimator = estimator

    @classmethod
    def _get_param_names(cls):
        return [*super()._get_param_names(), "estimator"]

    @property
    def classes_(self):
        return self.multiclass_estimator.classes_

    @generate_docstring(y="dense_anydtype")
    @mlfunc(set_input_type=True)
    def fit(self, X, y, sample_weight=None) -> "_BaseMulticlassClassifier":
        """
        Fit a multiclass classifier.
        """
        import sklearn.multiclass

        opts = {
            "ovo": sklearn.multiclass.OneVsOneClassifier,
            "ovr": sklearn.multiclass.OneVsRestClassifier,
        }
        if (cls := opts.get(self.strategy)) is None:
            raise ValueError(
                f"Expected `strategy` to be one of {list(opts)}, got {self.strategy}"
            )

        X, y, sample_weight = check_inputs(
            self,
            X,
            y,
            sample_weight,
            dtype=("float32", "float64"),
            y_dtype=None,
            accept_sparse=True,
            reset=True,
            mem_type="host",
        )

        with exit_internal_context():
            wrapper = cls(self.estimator, n_jobs=None)
            if sample_weight is None:
                wrapper.fit(X, y)
            elif self.strategy == "ovo":
                wrapper = _fit_weighted_ovo(wrapper, X, y, sample_weight)
            else:
                wrapper = _fit_weighted_ovr(wrapper, X, y, sample_weight)

            if hasattr(wrapper.estimators_[0], "n_features_in_"):
                wrapper.n_features_in_ = wrapper.estimators_[0].n_features_in_
            if hasattr(wrapper.estimators_[0], "feature_names_in_"):
                wrapper.feature_names_in_ = wrapper.estimators_[
                    0
                ].feature_names_in_

        self.multiclass_estimator = wrapper
        return self

    @generate_docstring(
        return_values={
            "name": "preds",
            "type": "dense",
            "description": "Predicted values",
            "shape": "(n_samples, 1)",
        }
    )
    @mlfunc(preserve_index=True)
    def predict(self, X):
        """
        Predict using multi class classifier.
        """
        X = check_inputs(
            self,
            X,
            dtype=("float32", "float64"),
            accept_sparse=True,
            mem_type="host",
        )

        with exit_internal_context():
            return cp.asarray(self.multiclass_estimator.predict(X))

    @generate_docstring(
        return_values={
            "name": "results",
            "type": "dense",
            "description": "Decision function values",
            "shape": "(n_samples, 1)",
        }
    )
    @mlfunc(preserve_index=True)
    def decision_function(self, X):
        """
        Calculate the decision function.
        """
        X = check_inputs(
            self,
            X,
            dtype=("float32", "float64"),
            accept_sparse=True,
            mem_type="host",
        )
        with exit_internal_context():
            return cp.asarray(self.multiclass_estimator.decision_function(X))


class OneVsRestClassifier(_BaseMulticlassClassifier):
    """
    Fit one binary classifier per class. The input can be any kind of cuML
    compatible array, and the output type follows cuML's output type
    configuration rules.

    The input is converted to a host (NumPy) array and partitioned into binary
    classification problems. Each cuML estimator transforms its partition
    back to the device. These host/device copies have some overhead. For more
    details see issue https://github.com/NVIDIA/cuml/issues/2876.

    For documentation see `scikit-learn's OneVsRestClassifier
    <https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html>`_.

    Parameters
    ----------
    estimator : cuML estimator
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
    >>> from cuml.linear_model import LogisticRegression
    >>> from cuml.multiclass import OneVsRestClassifier
    >>> from cuml.datasets.classification import make_classification

    >>> X, y = make_classification(n_samples=10, n_features=6,
    ...                            n_informative=4, n_classes=3,
    ...                            random_state=137)

    >>> cls = OneVsRestClassifier(LogisticRegression())
    >>> cls.fit(X, y)
    OneVsRestClassifier(estimator=LogisticRegression())
    >>> cls.predict(X)
    array([1, 1, 0, 1, 1, 1, 2, 2, 1, 2])
    """

    strategy = "ovr"


class OneVsOneClassifier(_BaseMulticlassClassifier):
    """
    Fit one binary classifier per pair of classes. The input can be any kind
    of cuML compatible array, and the output type follows cuML's output type
    configuration rules.

    The input is converted to a host (NumPy) array and partitioned into binary
    classification problems. Each cuML estimator transforms its partition
    back to the device. These host/device copies have some overhead. For more
    details see issue https://github.com/NVIDIA/cuml/issues/2876.

    For documentation see `scikit-learn's OneVsOneClassifier
    <https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsOneClassifier.html>`_.

    Parameters
    ----------
    estimator : cuML estimator
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
    >>> from cuml.linear_model import LogisticRegression
    >>> from cuml.multiclass import OneVsOneClassifier
    >>> from cuml.datasets.classification import make_classification

    >>> X, y = make_classification(n_samples=10, n_features=6,
    ...                            n_informative=4, n_classes=3,
    ...                            random_state=137)

    >>> cls = OneVsOneClassifier(LogisticRegression())
    >>> cls.fit(X, y)
    OneVsOneClassifier(estimator=LogisticRegression())
    >>> cls.predict(X)
    array([1, 1, 0, 1, 1, 1, 2, 2, 1, 2])
    """

    strategy = "ovo"
