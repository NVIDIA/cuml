#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import cupyx.scipy.sparse

from cuml.common.doc_utils import generate_docstring
from cuml.common.sparse import is_sparse
from cuml.internals.base import Base
from cuml.internals.interop import InteropMixin, UnsupportedOnGPU
from cuml.internals.mixins import (
    FMajorInputTagMixin,
    RegressorMixin,
    SparseInputTagMixin,
)
from cuml.internals.outputs import ReflectedAttr, mlfunc
from cuml.linear_model.base import LinearPredictMixin
from cuml.solvers.cd import fit_cd
from cuml.solvers.qn import fit_qn


class ElasticNet(
    InteropMixin,
    LinearPredictMixin,
    RegressorMixin,
    SparseInputTagMixin,
    FMajorInputTagMixin,
    Base,
):
    """
    Linear regression with combined L1 and L2 priors as regularizer.

    Parameters
    ----------
    alpha : float, default=1.0
        Constant that multiplies the L1 term.
        alpha = 0 is equivalent to an ordinary least square, solved by the
        LinearRegression object.
        For numerical reasons, using alpha = 0 with the Lasso object is not
        advised.
        Given this, you should use the LinearRegression object.
    l1_ratio : float, default=0.5
        The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1.
        For l1_ratio = 0 the penalty is an L2 penalty. For l1_ratio = 1 it is
        an L1 penalty.
        For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
    fit_intercept : boolean, default=True
        If True, Lasso tries to correct for the global mean of y.
        If False, the model expects that you have centered the data.
    max_iter : int, default=1000
        The maximum number of iterations
    tol : float, default=1e-3
        The tolerance for the optimization: if the updates are smaller than
        tol, the optimization code checks the dual gap for optimality and
        continues until it is smaller than tol.
    solver : {'auto', 'cd', 'qn'}, default='auto'
        The solver to use.

        - 'auto': uses 'cd' for dense inputs, and 'qn' for sparse inputs
        - 'cd': uses coordinate descent. Only supports dense inputs.
        - 'qn': uses quasi-newton methods. Supports sparse and dense inputs.

        You may find the alternative 'qn' algorithm is faster when the number
        of features is sufficiently large but the sample size is small.
    selection : {'cyclic', 'random'}, default='cyclic'
        How selections are made when `solver="cd"`. If set to 'random', a
        random coefficient is updated every iteration rather than looping over
        features sequentially by default. This (setting to 'random') often
        leads to significantly faster convergence especially when tol is higher
        than 1e-4.
    output_type : {None, 'input', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.

    Attributes
    ----------
    coef_ : array, shape (n_features)
        The estimated coefficients for the linear regression model.
    sparse_coef_ : sparse matrix, shape (n_targets, n_features)
        Sparse matrix representation of `coef_`.
    intercept_ : float
        The independent term, will be 0 if `fit_intercept` is False.
    n_iter_ : int
        The number of iterations taken by the solver.

    Notes
    -----
    For additional docs, see `scikitlearn's ElasticNet
    <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html>`_.

    Examples
    --------
    >>> import cupy as cp
    >>> import cudf
    >>> from cuml.linear_model import ElasticNet
    >>> enet = ElasticNet(alpha = 0.1, l1_ratio=0.5, solver='qn')
    >>> X = cudf.DataFrame()
    >>> X['col1'] = cp.array([0, 1, 2], dtype = cp.float32)
    >>> X['col2'] = cp.array([0, 1, 2], dtype = cp.float32)
    >>> y = cudf.Series(cp.array([0.0, 1.0, 2.0], dtype = cp.float32) )
    >>> result_enet = enet.fit(X, y)
    >>> print(result_enet.coef_)
    0    0.445...
    1    0.445...
    dtype: float32
    >>> print(result_enet.intercept_)
    0.108433...
    >>> X_new = cudf.DataFrame()
    >>> X_new['col1'] = cp.array([3,2], dtype = cp.float32)
    >>> X_new['col2'] = cp.array([5,5], dtype = cp.float32)
    >>> preds = result_enet.predict(X_new)
    >>> print(preds)
    0    3.674...
    1    3.228...
    dtype: float32
    """

    coef_ = ReflectedAttr()

    _cpu_class_path = "sklearn.linear_model.ElasticNet"

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "alpha",
            "l1_ratio",
            "fit_intercept",
            "max_iter",
            "tol",
            "solver",
            "selection",
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        if model.positive:
            raise UnsupportedOnGPU("`positive=True` is not supported")

        if model.warm_start:
            raise UnsupportedOnGPU("`warm_start=True` is not supported")

        if model.precompute is not False:
            raise UnsupportedOnGPU("`precompute` is not supported")

        # We use different algorithms than sklearn, adjust tolerance by a
        # factor empirically determined to be ~equivalent.
        tol = 10 * model.tol

        return {
            "alpha": model.alpha,
            "l1_ratio": model.l1_ratio,
            "fit_intercept": model.fit_intercept,
            "tol": tol,
            "max_iter": model.max_iter,
            "selection": model.selection,
        }

    def _params_to_cpu(self):
        # We use different algorithms than sklearn, adjust tolerance by a
        # factor empirically determined to be ~equivalent.
        tol = 0.1 * self.tol

        return {
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            "fit_intercept": self.fit_intercept,
            "tol": tol,
            "max_iter": self.max_iter,
            "selection": self.selection,
        }

    def _attrs_from_cpu(self, model):
        return {
            "intercept_": model.intercept_,
            "coef_": cp.asarray(model.coef_),
            "n_iter_": model.n_iter_,
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        return {
            "intercept_": self.intercept_,
            "coef_": cp.asnumpy(self.coef_, order="A"),
            "n_iter_": self.n_iter_,
            **super()._attrs_to_cpu(model),
        }

    def __init__(
        self,
        alpha=1.0,
        *,
        l1_ratio=0.5,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        solver="auto",
        selection="cyclic",
        output_type=None,
        verbose=False,
    ):
        super().__init__(verbose=verbose, output_type=output_type)

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.selection = selection

    @property
    @mlfunc
    def sparse_coef_(self):
        """Sparse representation of the fitted `coef_`."""
        return cupyx.scipy.sparse.csr_matrix(self.coef_)

    @generate_docstring()
    @mlfunc(set_input_type=True)
    def fit(self, X, y, sample_weight=None) -> "ElasticNet":
        """
        Fit the model with X and y.

        """
        if self.alpha < 0.0:
            raise ValueError(f"Expected alpha >= 0, got {self.alpha}")
        if self.selection not in ["cyclic", "random"]:
            raise ValueError(f"selection {self.selection!r} is not supported")
        if self.l1_ratio < 0.0 or self.l1_ratio > 1.0:
            raise ValueError(
                f"Expected 0.0 <= l1_ratio <= 1.0, got {self.l1_ratio}"
            )

        solver = self.solver
        if solver == "auto":
            solver = "qn" if is_sparse(X) else "cd"

        if solver == "qn":
            coef, intercept, n_iter, _ = fit_qn(
                self,
                X,
                y,
                sample_weight=sample_weight,
                loss="l2",
                fit_intercept=self.fit_intercept,
                l1_strength=self.alpha * self.l1_ratio,
                l2_strength=self.alpha * (1.0 - self.l1_ratio),
                max_iter=self.max_iter,
                tol=self.tol,
                penalty_normalized=False,
                verbose=self._verbose_level,
            )
            coef = coef.flatten()
            intercept = intercept.item()
        elif solver == "cd":
            if is_sparse(X):
                raise ValueError(
                    "solver='cd' doesn't support sparse inputs, please use "
                    "solver='auto' or solver='qn' instead"
                )
            coef, intercept, n_iter = fit_cd(
                self,
                X,
                y,
                sample_weight=sample_weight,
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                l1_ratio=self.l1_ratio,
                shuffle=self.selection == "random",
                max_iter=self.max_iter,
                tol=self.tol,
            )
        else:
            raise ValueError(f"solver={solver!r} is not supported")

        self.coef_ = coef
        self.intercept_ = intercept
        self.n_iter_ = n_iter

        return self
