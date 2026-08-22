#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from cuml.dask.common.base import BaseEstimator
from cuml.dask.solvers import CD


class ElasticNet(BaseEstimator):
    """
    ElasticNet extends LinearRegression with combined L1 and L2 regularizations
    on the coefficients when predicting response y with a linear combination of
    the predictors in X. It can reduce the variance of the predictors, force
    some coefficients to be small, and improves the conditioning of the
    problem.

    cuML's ElasticNet an array-like object or cuDF DataFrame, uses coordinate
    descent to fit a linear model.

    Parameters
    ----------
    alpha : float (default = 1.0)
        Constant that multiplies the L1 term.
        alpha = 0 is equivalent to an ordinary least square, solved by the
        LinearRegression object.
        For numerical reasons, using alpha = 0 with the Lasso object is not
        advised.
        Given this, you should use the LinearRegression object.
    l1_ratio: float (default = 0.5)
        The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1.
        For l1_ratio = 0 the penalty is an L2 penalty. For l1_ratio = 1 it is
        an L1 penalty.
        For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
    fit_intercept : boolean (default = True)
        If True, Lasso tries to correct for the global mean of y.
        If False, the model expects that you have centered the data.
    max_iter : int (default = 1000)
        The maximum number of iterations
    tol : float (default = 1e-3)
        The tolerance for the optimization: if the updates are smaller than
        tol, the optimization code checks the dual gap for optimality and
        continues until it is smaller than tol.
    selection : {'cyclic', 'random'} (default='cyclic')
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default.
        This (setting to 'random') often leads to significantly faster
        convergence especially when tol is higher than 1e-4.
    output_type : {None, 'input', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Attributes
    ----------
    coef_ : array, shape (n_features)
        The estimated coefficients for the linear regression model.
    intercept_ : array
        The independent term. If `fit_intercept` is False, will be 0.
    n_iter_ : int
        The number of iterations taken by the solver.

    For additional docs, see `scikitlearn's ElasticNet
    <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html>`_.
    """

    def __init__(self, *, client=None, **kwargs):
        super().__init__(client=client, **kwargs)

        kwargs["shuffle"] = False

        if "selection" in kwargs:
            if kwargs["selection"] == "random":
                kwargs["shuffle"] = True

            del kwargs["selection"]

        self.solver = CD(client=client, **kwargs)

    def fit(self, X, y):
        """
        Fit the model with X and y.

        Parameters
        ----------
        X : Dask cuDF DataFrame or CuPy backed Dask Array
            Dense matrix (floats or doubles) of shape (n_samples, n_features).

        y : Dask cuDF DataFrame or CuPy backed Dask Array
            Dense matrix (floats or doubles) of shape (n_samples, n_features).

        """
        self.solver.fit(X, y)
        return self

    def __getattr__(self, attr):
        if attr in ("coef_", "intercept_", "n_iter_"):
            try:
                return getattr(self.solver, attr)
            except AttributeError as e:
                raise AttributeError(
                    "%s is not fitted" % type(self).__name__
                ) from e
        return super().__getattr__(attr)

    def predict(self, X, delayed=True):
        """
        Predicts the y for X.

        Parameters
        ----------
        X : Dask cuDF DataFrame or CuPy backed Dask Array
            Dense matrix (floats or doubles) of shape (n_samples, n_features).

        delayed : bool (default = True)
            Whether to do a lazy prediction (and return Delayed objects) or an
            eagerly executed one.


        Returns
        -------
        y : Dask cuDF DataFrame or CuPy backed Dask Array
            Dense matrix (floats or doubles) of shape (n_samples, n_features).

        """

        return self.solver.predict(X)
