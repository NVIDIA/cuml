# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from dask.distributed import get_worker
from raft_dask.common.comms import get_raft_comm_state

from cuml.dask.common.base import (
    BaseEstimator,
    DelayedPredictionMixin,
    SyncFitMixinLinearModel,
    mnmg_import,
)


class Ridge(BaseEstimator, SyncFitMixinLinearModel, DelayedPredictionMixin):
    """
    Ridge extends LinearRegression by providing L2 regularization on the
    coefficients when predicting response y with a linear combination of the
    predictors in X. It can reduce the variance of the predictors, and improves
    the conditioning of the problem.

    cuML's Dask Ridge (multi-node multi-GPU) expects Dask cuDF DataFrame and
    provides an eigendecomposition-based algorithm (Eig) to fit a linear model.
    The Eig algorithm is usually preferred when X is a tall and skinny matrix.
    As the number of features in X increases, the accuracy of the Eig algorithm
    may decrease.

    Parameters
    ----------
    alpha : float (default = 1.0)
        Regularization strength - must be a positive float. Larger values
        specify stronger regularization.
    solver : {'eig'}
        Eig uses an eigendecomposition of the covariance matrix.
    fit_intercept : boolean (default = True)
        If True, Ridge adds an additional term c to correct for the global
        mean of y, modeling the response as "x * beta + c".
        If False, the model expects that you have centered the data.

    Attributes
    ----------
    coef_ : array, shape (n_features)
        The estimated coefficients for the linear regression model.
    intercept_ : array
        The independent term. If `fit_intercept` is False, will be 0.

    """

    def __init__(self, *, client=None, verbose=False, **kwargs):
        super().__init__(client=client, verbose=verbose, **kwargs)

        self._model_fit = False
        self._consec_call = 0

    def fit(self, X, y):
        """
        Fit the model with X and y.

        Parameters
        ----------
        X : Dask cuDF DataFrame or CuPy backed Dask Array (n_rows, n_features)
            Features for regression
        y : Dask cuDF DataFrame or CuPy backed Dask Array (n_rows, 1)
            Labels (outcome values)
        """
        models = self._fit(model_func=Ridge._create_model, data=(X, y))

        self._set_internal_model(models[0])

        return self

    def predict(self, X, delayed=True):
        """
        Make predictions for X and returns a dask collection.

        Parameters
        ----------
        X : Dask cuDF DataFrame or CuPy backed Dask Array (n_rows, n_features)
            Distributed dense matrix (floats or doubles) of shape
            (n_samples, n_features).

        delayed : bool (default = True)
            Whether to do a lazy prediction (and return Delayed objects) or an
            eagerly executed one.

        Returns
        -------
        y : Dask cuDF DataFrame or CuPy backed Dask Array (n_rows, 1)
        """
        return self._predict(X, delayed=delayed)

    def _get_param_names(self):
        return list(self.kwargs.keys())

    @staticmethod
    @mnmg_import
    def _create_model(sessionId, datatype, **kwargs):
        from cuml.linear_model.ridge_mg import RidgeMG

        handle = get_raft_comm_state(sessionId, get_worker())["handle"]
        return RidgeMG(handle=handle, output_type=datatype, **kwargs)
