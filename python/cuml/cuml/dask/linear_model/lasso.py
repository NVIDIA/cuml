#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from cuml.dask.common.base import BaseEstimator
from cuml.dask.solvers import CD


class Lasso(BaseEstimator):
    """
    Lasso extends LinearRegression by providing L1 regularization on the
    coefficients when predicting response y with a linear combination of the
    predictors in X. It can zero some of the coefficients for feature
    selection and improves the conditioning of the problem.

    cuML's Lasso an array-like object or cuDF DataFrame and
    uses coordinate descent to fit a linear model.

    Parameters
    ----------
    alpha : float (default = 1.0)
        Constant that multiplies the L1 term.
        alpha = 0 is equivalent to an ordinary least square, solved by the
        LinearRegression class.
        For numerical reasons, using alpha = 0 with the Lasso class is not
        advised.
        Given this, you should use the LinearRegression class.
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

    Attributes
    ----------
    coef_ : array, shape (n_features)
        The estimated coefficients for the linear regression model.
    intercept_ : array
        The independent term. If `fit_intercept` is False, will be 0.
    n_iter_ : int
        The number of iterations taken by the solver.

    For additional docs, see `scikitlearn's Lasso
    <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html>`_.
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

        return self.solver.predict(X, delayed=delayed)
