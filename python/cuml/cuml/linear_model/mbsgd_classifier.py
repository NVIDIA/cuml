#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp

from cuml.common.doc_utils import generate_docstring
from cuml.internals.base import Base
from cuml.internals.mixins import ClassifierMixin, FMajorInputTagMixin
from cuml.internals.outputs import ClassLabels, ReflectedAttr, mlfunc
from cuml.linear_model.base import LinearClassifierMixin
from cuml.solvers.sgd import fit_sgd


class MBSGDClassifier(
    LinearClassifierMixin, ClassifierMixin, FMajorInputTagMixin, Base
):
    """
    Linear models (linear SVM, logistic regression, or linear regression)
    fitted by minimizing a regularized empirical loss with mini-batch SGD.
    The MBSGD Classifier implementation is experimental and and it uses a
    different algorithm than sklearn's SGDClassifier. In order to improve
    the results obtained from cuML's MBSGDClassifier:

    * Reduce the batch size
    * Increase the eta0
    * Increase the number of iterations

    Since cuML is analyzing the data in batches using a small eta0 might
    not let the model learn as much as scikit learn does. Furthermore,
    decreasing the batch size might seen an increase in the time required
    to fit the model.

    Parameters
    ----------
    loss : {'hinge', 'log', 'squared_loss'} (default = 'hinge')
       'hinge' uses linear SVM

       'log' uses logistic regression

       'squared_loss' uses linear regression

    penalty : {'l1', 'l2', 'elasticnet', None} (default = 'l2')
        The penalty (aka regularization term) to apply.

        - 'l1': L1 norm (Lasso) regularization
        - 'l2': L2 norm (Ridge) regularization (the default)
        - 'elasticnet': Elastic Net regularization, a weighted average of L1 and L2
        - None: no penalty is added

    alpha : float (default = 0.0001)
        The constant value which decides the degree of regularization
    l1_ratio : float (default=0.15)
        The l1_ratio is used only when `penalty = elasticnet`. The value for
        l1_ratio should be `0 <= l1_ratio <= 1`. When `l1_ratio = 0` then the
        `penalty = 'l2'` and if `l1_ratio = 1` then `penalty = 'l1'`
    batch_size : int (default = 32)
        It sets the number of samples that will be included in each batch.
    fit_intercept : boolean (default = True)
       If True, the model tries to correct for the global mean of y.
       If False, the model expects that you have centered the data.
    epochs : int (default = 1000)
        The number of times the model should iterate through the entire dataset
        during training (default = 1000)
    tol : float (default = 1e-3)
       The training process will stop if current_loss > previous_loss - tol
    shuffle : boolean (default = True)
       True, shuffles the training data after each epoch
       False, does not shuffle the training data after each epoch
    eta0 : float (default = 0.001)
        Initial learning rate
    power_t : float (default = 0.5)
        The exponent used for calculating the invscaling learning rate
    learning_rate : {'constant', 'invscaling', 'adaptive'} (default = 'constant')
        `constant` keeps the learning rate constant

        `adaptive` changes the learning rate if the training loss or the
        validation accuracy does not improve for `n_iter_no_change` epochs.
        The old learning rate is generally divided by 5
    n_iter_no_change : int (default = 5)
        the number of epochs to train without any improvement in the model
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
    coef_: array, shape=(n_features,)
        The model coefficients.
    intercept_: float
        The independent term. If `fit_intercept` is False, will be 0.
    classes_ : np.ndarray, shape=(n_classes,)
        Array of the class labels.

    Notes
    -----
    For additional docs, see `scikitlearn's SGDClassifier
    <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html>`_.

    Examples
    --------
    >>> import cupy as cp
    >>> import cuml
    >>> X = cp.array([[1, 1], [1, 2], [2, 2], [2, 3]], dtype=cp.float32)
    >>> y = cp.array([1, 1, 2, 2])
    >>> X_test = cp.asarray([[3, 5], [2, 5]], dtype=cp.float32)
    >>> model = cuml.MBSGDClassifier().fit(X, y)
    >>> model.predict(X_test)
    array([2, 2])
    """

    coef_ = ReflectedAttr()

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "loss",
            "penalty",
            "alpha",
            "l1_ratio",
            "fit_intercept",
            "epochs",
            "tol",
            "shuffle",
            "learning_rate",
            "eta0",
            "power_t",
            "batch_size",
            "n_iter_no_change",
        ]

    def __init__(
        self,
        *,
        loss="hinge",
        penalty="l2",
        alpha=0.0001,
        l1_ratio=0.15,
        fit_intercept=True,
        epochs=1000,
        tol=1e-3,
        shuffle=True,
        learning_rate="constant",
        eta0=0.001,
        power_t=0.5,
        batch_size=32,
        n_iter_no_change=5,
        verbose=False,
        output_type=None,
    ):
        super().__init__(verbose=verbose, output_type=output_type)
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.epochs = epochs
        self.tol = tol
        self.shuffle = shuffle
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.batch_size = batch_size
        self.n_iter_no_change = n_iter_no_change

    @generate_docstring()
    @mlfunc(set_input_type=True)
    def fit(self, X, y) -> "MBSGDClassifier":
        """
        Fit the model with X and y.

        """
        coef, intercept, classes = fit_sgd(
            self,
            X,
            y,
            loss=self.loss,
            penalty=self.penalty,
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=self.fit_intercept,
            epochs=self.epochs,
            tol=self.tol,
            shuffle=self.shuffle,
            learning_rate=self.learning_rate,
            eta0=self.eta0,
            power_t=self.power_t,
            batch_size=self.batch_size,
            n_iter_no_change=self.n_iter_no_change,
            return_classes=True,
        )
        self.coef_ = coef
        self.intercept_ = intercept
        self.classes_ = classes
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
        Predicts the y for X.

        """
        scores = self.decision_function(X)
        thresh = 0 if self.loss == "hinge" else 0.5
        indices = (scores > thresh).view(cp.int8)
        return ClassLabels(indices, self.classes_)
