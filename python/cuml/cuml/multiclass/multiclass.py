# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp

from cuml.common.doc_utils import generate_docstring
from cuml.internals.base import Base
from cuml.internals.mixins import ClassifierMixin
from cuml.internals.outputs import exit_internal_context, mlfunc
from cuml.internals.validation import check_inputs


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
    def fit(self, X, y) -> "_BaseMulticlassClassifier":
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
        X, y = check_inputs(
            self,
            X,
            y,
            dtype=("float32", "float64"),
            y_dtype=None,
            accept_sparse=True,
            reset=True,
            mem_type="host",
        )

        with exit_internal_context():
            wrapper = cls(self.estimator, n_jobs=None).fit(X, y)

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
    Wrapper around Sckit-learn's class with the same name. The input can be
    any kind of cuML compatible array, and the output type follows cuML's
    output type configuration rules.

    Before passing the data to scikit-learn, it is converted to host (numpy)
    array. Under the hood the data is partitioned for binary classification,
    and it is transformed back to the device by the cuML estimator. These
    copies back and forth the device and the host have some overhead. For more
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
    Wrapper around Sckit-learn's class with the same name. The input can be
    any kind of cuML compatible array, and the output type follows cuML's
    output type configuration rules.

    Before passing the data to scikit-learn, it is converted to host (numpy)
    array. Under the hood the data is partitioned for binary classification,
    and it is transformed back to the device by the cuML estimator. These
    copies back and forth the device and the host have some overhead. For more
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
