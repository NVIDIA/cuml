# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import numbers

import cupy as cp

import cuml.svm.linear
from cuml.common.doc_utils import generate_docstring
from cuml.internals.base import Base
from cuml.internals.interop import InteropMixin, UnsupportedOnGPU
from cuml.internals.mixins import ClassifierMixin
from cuml.internals.outputs import ClassLabels, ReflectedAttr, mlfunc
from cuml.linear_model.base import LinearClassifierMixin

__all__ = ("LinearSVC",)


class LinearSVC(InteropMixin, LinearClassifierMixin, ClassifierMixin, Base):
    """
    Linear Support Vector Classification.

    Similar to SVC with parameter kernel='linear', but implemented using a
    linear solver. This enables flexibility in penalties and loss functions,
    and can scale better for larger problems.

    Parameters
    ----------
    penalty : {'l1', 'l2'}, default = 'l2'
        The norm used in the penalization.
    loss : {'hinge', 'squared_hinge'}, default='squared_hinge'
        The loss function.
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.
    fit_intercept : bool, default=True
        Whether to fit the bias term. Set to False if you expect that the
        data is already centered.
    penalized_intercept : bool, default=False
        When true, the bias term is treated the same way as other features;
        i.e. it's penalized by the regularization term of the target function.
        Enabling this feature forces an extra copying the input data X.
    class_weight : dict or string, default=None
        Weights to modify the parameter C for class i to ``class_weight[i]*C``.
        The string 'balanced' is also accepted, in which case
        ``class_weight[i] = n_samples / (n_classes * n_samples_of_class[i])``
    tol : float, default=1e-4
        Tolerance for the stopping criterion.
    max_iter : int, default=1000
        Maximum number of iterations for the underlying solver.
    linesearch_max_iter : int, default=100
        Maximum number of linesearch (inner loop) iterations for
        the underlying (QN) solver.
    lbfgs_memory : int, default=5
        Number of vectors approximating the hessian for the underlying QN
        solver (l-bfgs).
    n_streams : int (default = 1)
        Number of parallel streams used for fitting.
    multi_class : {'ovr'}, default='ovr'
        Multiclass classification strategy. Currently only 'ovr' is supported.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {None, 'input', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Attributes
    ----------
    coef_ : array, shape (1, n_features) if n_classes == 2 else (n_classes, n_features)
        Weights assigned to the features (coefficients in the primal problem).
    intercept_ : array or float, shape (1,) if n_classes == 2 else (n_classes,)
        The constant factor in the decision function. If
        ``fit_intercept=False`` this is instead a float with value 0.0.
    classes_ : np.ndarray, shape=(n_classes,)
        A sorted array of the class labels.
    n_iter_ : int
        The maximum number of iterations run across all classes during the fit.

    Notes
    -----
    The model uses the quasi-newton (QN) solver to find the solution in the
    primal space. Thus, in contrast to generic :class:`SVC<cuml.svm.SVC>`
    model, it does not compute the support coefficients/vectors.

    Check the solver's documentation for more details
    :class:`Quasi-Newton (L-BFGS/OWL-QN)<cuml.QN>`.

    For additional docs, see `scikitlearn's LinearSVC
    <https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html>`_.

    Examples
    --------
    >>> import cupy as cp
    >>> from cuml.svm import LinearSVC
    >>> X = cp.array([[1,1], [2,1], [1,2], [2,2], [1,3], [2,3]],
    ...              dtype=cp.float32);
    >>> y = cp.array([0, 0, 1, 0, 1, 1], dtype=cp.float32)
    >>> clf = LinearSVC(penalty='l1', C=1).fit(X, y)
    >>> print("Predicted labels:", clf.predict(X))  # doctest: +SKIP
    Predicted labels: [0 0 1 0 1 1]
    """

    coef_ = ReflectedAttr()
    intercept_ = ReflectedAttr()

    _cpu_class_path = "sklearn.svm.LinearSVC"

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "penalty",
            "loss",
            "C",
            "fit_intercept",
            "penalized_intercept",
            "class_weight",
            "tol",
            "max_iter",
            "linesearch_max_iter",
            "lbfgs_memory",
            "n_streams",
            "multi_class",
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        if model.intercept_scaling != 1:
            raise UnsupportedOnGPU(
                f"`intercept_scaling={model.intercept_scaling}` is not supported"
            )
        if model.multi_class != "ovr":
            raise UnsupportedOnGPU(
                f"`multi_class={model.multi_class}` is not supported"
            )

        return {
            "penalty": model.penalty,
            "loss": model.loss,
            "C": model.C,
            "fit_intercept": model.fit_intercept,
            "class_weight": model.class_weight,
            "tol": model.tol,
            "max_iter": model.max_iter,
            "multi_class": model.multi_class,
        }

    def _params_to_cpu(self):
        return {
            "penalty": self.penalty,
            "loss": self.loss,
            "C": self.C,
            "fit_intercept": self.fit_intercept,
            "class_weight": self.class_weight,
            "tol": self.tol,
            "max_iter": self.max_iter,
            "multi_class": self.multi_class,
        }

    def _attrs_from_cpu(self, model):
        return {
            "coef_": cp.asarray(model.coef_, order="A", dtype="float64"),
            "intercept_": (
                model.intercept_
                if cp.isscalar(model.intercept_)
                else cp.asarray(model.intercept_, dtype="float64")
            ),
            "classes_": model.classes_,
            "n_iter_": model.n_iter_,
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        return {
            "coef_": self.coef_.get(order="A").astype("f8", copy=False),
            "intercept_": (
                self.intercept_
                if cp.isscalar(self.intercept_)
                else self.intercept_.get(order="A").astype("f8", copy=False)
            ),
            "classes_": self.classes_,
            "n_iter_": self.n_iter_,
            **super()._attrs_to_cpu(model),
        }

    def __init__(
        self,
        *,
        penalty="l2",
        loss="squared_hinge",
        C=1.0,
        fit_intercept=True,
        penalized_intercept=False,
        class_weight=None,
        tol=1e-4,
        max_iter=1000,
        linesearch_max_iter=100,
        lbfgs_memory=5,
        n_streams=1,
        multi_class="ovr",
        verbose=False,
        output_type=None,
    ):
        super().__init__(verbose=verbose, output_type=output_type)

        self.penalty = penalty
        self.loss = loss
        self.C = C
        self.fit_intercept = fit_intercept
        self.penalized_intercept = penalized_intercept
        self.class_weight = class_weight
        self.tol = tol
        self.max_iter = max_iter
        self.linesearch_max_iter = linesearch_max_iter
        self.lbfgs_memory = lbfgs_memory
        self.n_streams = n_streams
        self.multi_class = multi_class

    @generate_docstring()
    @mlfunc(set_input_type=True)
    def fit(self, X, y, sample_weight=None) -> "LinearSVC":
        """Fit the model according to the given training data."""
        n_streams = self.n_streams
        if isinstance(n_streams, bool) or not isinstance(
            n_streams, numbers.Integral
        ):
            raise TypeError(
                f"n_streams must be a positive integer; got {n_streams!r}"
            )
        if n_streams <= 0:
            raise ValueError(
                f"n_streams must be a positive integer; got {n_streams!r}"
            )
        n_streams = int(n_streams)

        coef, intercept, n_iter, classes = cuml.svm.linear.fit(
            self,
            X,
            y,
            sample_weight,
            is_classifier=True,
            n_streams=n_streams,
            class_weight=self.class_weight,
            loss=self.loss,
            penalty=self.penalty,
            fit_intercept=self.fit_intercept,
            penalized_intercept=self.penalized_intercept,
            max_iter=self.max_iter,
            linesearch_max_iter=self.linesearch_max_iter,
            lbfgs_memory=self.lbfgs_memory,
            C=self.C,
            tol=self.tol,
            epsilon=0.0,
            verbose=self._verbose_level,
        )
        self.coef_ = coef
        self.intercept_ = intercept
        self.n_iter_ = n_iter
        self.classes_ = classes
        return self

    @generate_docstring(
        return_values={
            "name": "y_pred",
            "type": "dense",
            "description": "Predicted class labels.",
            "shape": "(n_samples,)",
        },
    )
    @mlfunc(preserve_index=True)
    def predict(self, X):
        """Predict class labels for samples in X."""
        scores = self.decision_function(X)
        if scores.ndim == 1:
            indices = (scores >= 0).view(cp.int8)
        else:
            indices = scores.argmax(axis=1)

        return ClassLabels(indices, self.classes_)
