# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import numpy as np

from cuml.common.classification import process_class_weight
from cuml.common.doc_utils import generate_docstring
from cuml.common.sparse import is_sparse
from cuml.internals.interop import UnsupportedOnCPU, UnsupportedOnGPU
from cuml.internals.logger import warn
from cuml.internals.mixins import ClassifierMixin
from cuml.internals.outputs import ClassLabels, mlfunc
from cuml.internals.validation import check_inputs, check_is_fitted
from cuml.multiclass import OneVsOneClassifier, OneVsRestClassifier
from cuml.svm.svm_base import SVMBase


class SVC(ClassifierMixin, SVMBase):
    """
    SVC (C-Support Vector Classification)

    Construct an SVC classifier for training and predictions.

    Examples
    --------
    .. code-block:: python

        >>> import cupy as cp
        >>> from cuml.svm import SVC
        >>> X = cp.array([[1,1], [2,1], [1,2], [2,2], [1,3], [2,3]],
        ...              dtype=cp.float32);
        >>> y = cp.array([-1, -1, 1, -1, 1, 1], dtype=cp.float32)
        >>> clf = SVC(kernel='poly', degree=2, gamma='auto', C=1)
        >>> clf.fit(X, y)
        SVC(C=1, degree=2, gamma='auto', kernel='poly')
        >>> print("Predicted labels:", clf.predict(X))
        Predicted labels: [-1. -1.  1. -1.  1.  1.]

    Parameters
    ----------
    C : float (default = 1.0)
        Penalty parameter C
    kernel : string (default='rbf')
        Specifies the kernel function. Possible options: 'linear', 'poly',
        'rbf', 'sigmoid', 'precomputed'. When using 'precomputed', X is
        expected to be a precomputed kernel matrix of shape
        (n_samples, n_samples) at fit time, and (n_samples_test,
        n_samples_train) at predict time. A valid kernel matrix should be
        symmetric and positive semi-definite; cuML does not validate these
        properties.
    degree : int (default=3)
        Degree of polynomial kernel function.
    gamma : float or string (default = 'scale')
        Coefficient for rbf, poly, and sigmoid kernels. You can specify the
        numeric value, or use one of the following options:

        - 'auto': gamma will be set to ``1 / n_features``
        - 'scale': gamma will be se to ``1 / (n_features * X.var())``

    coef0 : float (default = 0.0)
        Independent term in kernel function, only significant for poly and
        sigmoid
    tol : float (default = 1e-3)
        Tolerance for stopping criterion.
    cache_size : float (default = 1024.0)
        Size of the kernel cache during training in MiB. Increase it to improve
        the training time, at the cost of higher memory footprint. After
        training the kernel cache is deallocated.
        During prediction, we also need a temporary space to store kernel
        matrix elements (this can be significant if n_support is large).
        The cache_size variable sets an upper limit to the prediction
        buffer as well.
    class_weight : dict or string (default=None)
        Weights to modify the parameter C for class i to class_weight[i]*C. The
        string 'balanced' is also accepted, in which case ``class_weight[i] =
        n_samples / (n_classes * n_samples_of_class[i])``
    max_iter : int (default = -1)
        Limit the number of total iterations in the solver. Default of -1 for
        "no limit".
    decision_function_shape : str ('ovo' or 'ovr', default 'ovo')
        Multiclass classification strategy. ``'ovo'`` uses `OneVsOneClassifier
        <https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsOneClassifier.html>`_
        while ``'ovr'`` selects `OneVsRestClassifier
        <https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html>`_
    nochange_steps : int (default = 1000)
        We monitor how much our stopping criteria changes during outer
        iterations. If it does not change (changes less then 1e-3*tol)
        for nochange_steps consecutive steps, then we stop training.
    output_type : {None, 'input', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.
    random_state: int (default = None)
        Seed for random number generator (used only when ``probability=True``).
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.

    Attributes
    ----------
    n_support_ : int
        The total number of support vectors. Note: this will change in the
        future to represent number support vectors for each class (like
        in Sklearn, see https://github.com/NVIDIA/cuml/issues/956 )
    support_ : int, shape = (n_support)
        Device array of support vector indices
    support_vectors_ : float, shape (n_support, n_cols)
        Device array of support vectors. For kernel='precomputed', this
        attribute is empty (shape (0, 0)) since the original feature vectors
        are not available.
    dual_coef_ : float, shape = (1, n_support)
        Device array of coefficients for support vectors
    intercept_ : float
        The constant in the decision function
    fit_status_ : int
        0 if SVM is correctly fitted
    n_iter_ : array
        Number of outer iterations run by the solver for each model fit.
    coef_ : float, shape (1, n_cols)
        Only available for linear kernels. It is the normal of the
        hyperplane.
    classes_ : np.ndarray, shape=(n_classes,)
        A sorted array of the class labels.
    class_weight_ : np.ndarray of shape (n_classes,)
        Class weight multipliers, computed based on the ``class_weight``
        parameter.
    n_classes_ : int
        Number of classes

    Notes
    -----
    The solver uses the SMO method to fit the classifier. We use the Optimized
    Hierarchical Decomposition [1]_ variant of the SMO algorithm, similar to
    [2]_.

    For additional docs, see `scikitlearn's SVC
    <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_.

    References
    ----------
    .. [1] J. Vanek et al. A GPU-Architecture Optimized Hierarchical
       Decomposition Algorithm for Support VectorMachine Training, IEEE
       Transactions on Parallel and Distributed Systems, vol 28, no 12, 3330,
       (2017)

    .. [2] `Z. Wen et al. ThunderSVM: A Fast SVM Library on GPUs and CPUs,
       Journal of Machine Learning Research, 19, 1-5 (2018)
       <https://github.com/Xtra-Computing/thundersvm>`_

    """

    _cpu_class_path = "sklearn.svm.SVC"

    @classmethod
    def _get_param_names(cls):
        params = super()._get_param_names()
        # SVC doesn't expose `epsilon` in the constructor
        params.remove("epsilon")
        params.extend(
            ["random_state", "class_weight", "decision_function_shape"]
        )
        return params

    @classmethod
    def _params_from_cpu(cls, model):
        # TODO: remove when we only support sklearn >= 1.9
        if getattr(model, "probability", None) is True:
            raise UnsupportedOnGPU("`probability=True` is not supported")

        params = super()._params_from_cpu(model)
        # SVC doesn't expose `epsilon` in the constructor
        params.pop("epsilon")
        params.update(
            {
                "random_state": model.random_state,
                "class_weight": model.class_weight,
                "decision_function_shape": model.decision_function_shape,
            }
        )
        return params

    def _params_to_cpu(self):
        params = super()._params_to_cpu()
        # SVC doesn't expose `epsilon` in the constructor
        params.pop("epsilon")
        params.update(
            {
                "random_state": self.random_state,
                "class_weight": self.class_weight,
                "decision_function_shape": self.decision_function_shape,
            }
        )
        return params

    def _attrs_from_cpu(self, model):
        n_classes = len(model.classes_)
        if n_classes > 2:
            raise UnsupportedOnGPU("multiclass models are not supported")
        return {
            "n_classes_": n_classes,
            "classes_": model.classes_,
            "class_weight_": model.class_weight_,
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        if hasattr(self, "_multiclass"):
            raise UnsupportedOnCPU(
                "Converting multiclass models from GPU is not yet supported"
            )

        out = super()._attrs_to_cpu(model)

        # sklearn's binary classification expects inverted
        # _dual_coef_ and _intercept_.
        out.update(
            _dual_coef_=-out["dual_coef_"],
            _intercept_=-out["intercept_"],
            classes_=self.classes_,
            class_weight_=self.class_weight_,
        )
        return out

    def __init__(
        self,
        *,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        tol=1e-3,
        cache_size=1024.0,
        max_iter=-1,
        nochange_steps=1000,
        verbose=False,
        output_type=None,
        random_state=None,
        class_weight=None,
        decision_function_shape="ovo",
    ):
        super().__init__(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            cache_size=cache_size,
            max_iter=max_iter,
            nochange_steps=nochange_steps,
            verbose=verbose,
            output_type=output_type,
        )
        self.random_state = random_state
        self.class_weight = class_weight
        self.decision_function_shape = decision_function_shape

    @property
    @mlfunc
    def support_(self):
        if hasattr(self, "_multiclass"):
            estimators = self._multiclass.multiclass_estimator.estimators_
            return cp.concatenate(
                [cp.asarray(cls._support_) for cls in estimators]
            )
        else:
            return self._support_

    @support_.setter
    def support_(self, value):
        self._support_ = value

    @property
    @mlfunc
    def intercept_(self):
        if hasattr(self, "_multiclass"):
            estimators = self._multiclass.multiclass_estimator.estimators_
            return cp.concatenate(
                [cp.asarray(cls._intercept_) for cls in estimators]
            )
        else:
            return self._intercept_

    @intercept_.setter
    def intercept_(self, value):
        self._intercept_ = value

    def _fit_multiclass(self, X, y, sample_weight):
        if sample_weight is not None:
            warn(
                "Sample weights are currently ignored for multi class classification"
            )

        params = self.get_params()
        decision_function_shape = params.pop("decision_function_shape")
        wrappers = {"ovo": OneVsOneClassifier, "ovr": OneVsRestClassifier}
        if (multiclass_cls := wrappers.get(decision_function_shape)) is None:
            raise ValueError(
                f"Expected `decision_function_shape` to be one of "
                f"{list(wrappers)}, got {decision_function_shape}"
            )
        self._multiclass = multiclass_cls(
            estimator=SVC(**params),
            verbose=self.verbose,
            output_type=self.output_type,
        )
        self._multiclass.fit(X, y)

        # if using one-vs-one we align support_ indices to those of
        # full dataset
        if decision_function_shape == "ovo":
            classes = cp.unique(y)
            n_classes = len(classes)
            estimator_index = 0
            # Loop through multiclass estimators and re-align support_ indices
            for i in range(n_classes):
                for j in range(i + 1, n_classes):
                    cond = cp.logical_or(y == classes[i], y == classes[j])
                    ovo_support = cp.array(
                        self._multiclass.multiclass_estimator.estimators_[
                            estimator_index
                        ].support_
                    )
                    self._multiclass.multiclass_estimator.estimators_[
                        estimator_index
                    ].support_ = cp.nonzero(cond)[0][ovo_support]
                    estimator_index += 1

        self.shape_fit_ = X.shape
        self.fit_status_ = 0
        self.n_iter_ = np.concatenate(
            [
                est.n_iter_
                for est in self._multiclass.multiclass_estimator.estimators_
            ]
        )
        return self

    @generate_docstring(y="dense_anydtype")
    @mlfunc(set_input_type=True)
    def fit(self, X, y, sample_weight=None) -> "SVC":
        """
        Fit the model with X and y.

        """
        if hasattr(self, "_multiclass"):
            del self._multiclass

        if self.kernel == "precomputed" and is_sparse(X):
            raise TypeError("Sparse precomputed kernels are not supported.")

        X, y, sample_weight, classes = check_inputs(
            self,
            X,
            y,
            sample_weight,
            dtype=("float32", "float64"),
            order="F",
            accept_sparse="csr",
            ensure_min_samples=2,
            y_dtype=None,
            return_classes=True,
            reset=True,
        )

        if len(classes) == 1:
            raise ValueError(
                "This solver needs samples of at least 2 classes in the data, but "
                "the data contains only 1 class"
            )

        if self.kernel == "precomputed" and X.shape[0] != X.shape[1]:
            raise ValueError(
                f"Precomputed kernel matrix must be square, got shape {X.shape}"
            )

        self.n_classes_ = len(classes)
        self.classes_ = classes
        self.class_weight_, sample_weight = process_class_weight(
            classes,
            y,
            class_weight=self.class_weight,
            sample_weight=sample_weight,
            dtype=X.dtype,
            balanced_with_sample_weight=False,
        )

        if len(classes) > 2:
            return self._fit_multiclass(X, y, sample_weight)

        # Encode y to -1/1 (like [0, 1, 0, 1] -> [-1, 1, -1, 1])
        y = cp.array([-1, 1], dtype=X.dtype).take(y)
        self._fit(X, y, sample_weight)
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
        Predicts the class labels for X. The returned y values are the class
        labels associated to sign(decision_function(X)).
        """
        check_is_fitted(self)

        if hasattr(self, "_multiclass"):
            indices = self._multiclass.predict(X)
        else:
            res = self.decision_function(X)
            indices = (res >= 0).view(cp.int8)

        return ClassLabels(indices, self.classes_)

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
        Calculates the decision function values for X.

        For precomputed kernels, X should be a kernel matrix of shape
        (n_samples_test, n_samples_train) where n_samples_train is the
        number of samples used during fit.

        """
        check_is_fitted(self)

        if hasattr(self, "_multiclass"):
            return self._multiclass.decision_function(X)

        return self._predict(X)
