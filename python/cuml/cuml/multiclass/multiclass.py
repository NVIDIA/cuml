# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import warnings

import cupy as cp
import cupyx.scipy.sparse as cp_sp

from cuml.common.doc_utils import generate_docstring
from cuml.internals.base import Base
from cuml.internals.mixins import ClassifierMixin
from cuml.internals.outputs import ClassLabels, mlfunc
from cuml.internals.validation import check_inputs, check_is_fitted


class _ConstantPredictor:
    def predict(self, X):
        return cp.zeros(X.shape[0], dtype=cp.int32)

    def decision_function(self, X):
        return cp.zeros(X.shape[0], dtype=cp.int32)


class _BaseMulticlassClassifier(ClassifierMixin, Base):
    """Shared base class for multiclass classifiers."""

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
        check_is_fitted(self)
        return self._classes

    @staticmethod
    def _predict_binary(est, X):
        from sklearn.base import is_regressor

        if is_regressor(est):
            return cp.asarray(est.predict(X)).ravel()

        try:
            return cp.asarray(est.decision_function(X)).ravel()
        except (AttributeError, NotImplementedError):
            return cp.asarray(est.predict_proba(X))[:, 1]

    @staticmethod
    def _threshold_for_binary_predict(est):
        from sklearn.base import is_classifier

        if hasattr(est, "decision_function") and is_classifier(est):
            return 0.0
        return 0.5

    def _fit_ovr(self, X, y):
        from sklearn.base import clone

        X, y, self._classes = check_inputs(
            self,
            X,
            y,
            dtype=("float32", "float64"),
            y_dtype=None,
            accept_sparse=True,
            reset=True,
            mem_type="device",
            return_classes=True,
        )

        n_cls = len(self._classes)

        if n_cls == 1:
            warnings.warn(
                f"Label not {self._classes[0]} is present in all "
                "training examples.",
                stacklevel=2,
            )
            self.estimators_ = [_ConstantPredictor()]
            return self

        ids = (1,) if n_cls == 2 else range(n_cls)

        self.estimators_ = [
            clone(self.estimator).fit(X, (y == i).astype(cp.int32))
            for i in ids
        ]
        return self

    def _fit_ovo(self, X, y):
        from sklearn.base import clone
        from sklearn.utils import get_tags

        X, y, self._classes = check_inputs(
            self,
            X,
            y,
            dtype=("float32", "float64"),
            y_dtype=None,
            accept_sparse=True,
            reset=True,
            mem_type="device",
            return_classes=True,
        )

        n_cls = len(self._classes)

        if n_cls == 1:
            raise ValueError(
                "OneVsOneClassifier can not be fit when only one class is "
                "present."
            )

        pw = get_tags(self.estimator).input_tags.pairwise

        if cp_sp.issparse(X):
            X = X.tocsr()

        self.estimators_ = []
        self.pairwise_indices_ = [] if pw else None

        for i in range(n_cls):
            for j in range(i + 1, n_cls):
                idx = cp.flatnonzero((y == i) | (y == j))
                Xi = X[idx]

                if pw:
                    Xi = Xi[:, idx]
                    self.pairwise_indices_.append(idx)

                yi = (y[idx] == j).astype(cp.int32)

                self.estimators_.append(clone(self.estimator).fit(Xi, yi))

        return self

    @staticmethod
    def _ovr_decision_function(pred, score, n_cls):
        n = pred.shape[0]

        conf = cp.zeros((n, n_cls))
        votes = cp.zeros((n, n_cls))

        k = 0

        for i in range(n_cls):
            for j in range(i + 1, n_cls):
                conf[:, i] -= score[:, k]
                conf[:, j] += score[:, k]

                votes[pred[:, k] == 0, i] += 1
                votes[pred[:, k] == 1, j] += 1

                k += 1

        conf /= 3 * (cp.abs(conf) + 1)

        return votes + conf

    @generate_docstring(y="dense_anydtype")
    @mlfunc(set_input_type=True)
    def fit(self, X, y) -> "_BaseMulticlassClassifier":
        """
        Fit a multiclass classifier.
        """
        if self.strategy == "ovr":
            return self._fit_ovr(X, y)

        if self.strategy == "ovo":
            return self._fit_ovo(X, y)

        raise ValueError(
            f"Expected `strategy` to be one of ['ovo', 'ovr'], "
            f"got {self.strategy}"
        )

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
        check_is_fitted(self)

        if self.strategy == "ovr":
            X = check_inputs(
                self,
                X,
                dtype=("float32", "float64"),
                accept_sparse=True,
                mem_type="device",
            )

            if len(self.estimators_) == 1:
                est = self.estimators_[0]
                scr = self._predict_binary(est, X)
                cut = self._threshold_for_binary_predict(est)
                idx = (scr > cut).astype(cp.intp)
            else:
                scr = cp.column_stack(
                    [self._predict_binary(est, X) for est in self.estimators_]
                )
                idx = cp.argmax(scr, axis=1)

            return ClassLabels(idx, self._classes)

        if self.strategy == "ovo":
            scr = self.decision_function(X)

            if len(self._classes) == 2:
                cut = self._threshold_for_binary_predict(self.estimators_[0])
                idx = (scr > cut).astype(cp.intp)
            else:
                idx = cp.argmax(scr, axis=1)

            return ClassLabels(idx, self._classes)

        raise ValueError(
            f"Expected `strategy` to be one of ['ovo', 'ovr'], "
            f"got {self.strategy}"
        )

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
        check_is_fitted(self)

        if self.strategy == "ovr":
            X = check_inputs(
                self,
                X,
                dtype=("float32", "float64"),
                accept_sparse=True,
                mem_type="device",
            )

            if len(self.estimators_) == 1:
                return cp.asarray(
                    self.estimators_[0].decision_function(X)
                ).ravel()

            return cp.column_stack(
                [
                    cp.asarray(est.decision_function(X)).ravel()
                    for est in self.estimators_
                ]
            )

        if self.strategy == "ovo":
            X = check_inputs(
                self,
                X,
                dtype=("float32", "float64"),
                accept_sparse=True,
                mem_type="device",
            )

            ids = self.pairwise_indices_

            if ids is None:
                Xs = [X] * len(self.estimators_)
            else:
                Xs = [X[:, idx] for idx in ids]

            pred = []
            conf = []

            for est, Xi in zip(self.estimators_, Xs):
                p = est.predict(Xi)

                if isinstance(p, ClassLabels):
                    p = p.indices

                pred.append(cp.asarray(p).ravel())
                conf.append(self._predict_binary(est, Xi))

            pred = cp.column_stack(pred)
            conf = cp.column_stack(conf)

            scr = self._ovr_decision_function(
                pred,
                conf,
                len(self._classes),
            )

            if len(self._classes) == 2:
                return scr[:, 1]

            return scr

        raise ValueError(
            f"Expected `strategy` to be one of ['ovo', 'ovr'], "
            f"got {self.strategy}"
        )


class OneVsRestClassifier(_BaseMulticlassClassifier):
    """
    One-vs-rest multiclass classifier using device-resident multiclass
    orchestration. The input can be any kind of cuML compatible array, and
    the output type follows cuML's output type configuration rules.

    The data and generated binary targets remain on the device while fitting
    the underlying binary estimators, avoiding unnecessary device-to-host
    transfers.

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
    One-vs-one multiclass classifier using device-resident multiclass
    orchestration. The input can be any kind of cuML compatible array, and
    the output type follows cuML's output type configuration rules.

    Each pairwise binary problem is constructed on the device and the
    resulting votes and confidence values are combined with CuPy, avoiding
    unnecessary device-to-host transfers.

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
