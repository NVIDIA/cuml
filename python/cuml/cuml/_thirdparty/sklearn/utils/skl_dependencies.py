# SPDX-FileCopyrightText: Gael Varoquaux <gael.varoquaux@normalesup.org>
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

# Original authors from Sckit-Learn:
#         Gael Varoquaux <gael.varoquaux@normalesup.org>
# License: BSD 3 clause


# This code originates from the Scikit-Learn library,
# it was since modified to allow GPU acceleration.
# This code is under BSD 3 clause license.
# Authors mentioned above do not endorse or promote this production.
from collections import defaultdict

from cuml.internals.base import Base


class BaseEstimator(Base):
    """Base class for all estimators in scikit-learn

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    def __init_subclass__(cls):
        orig_init = cls.__init__

        def init(self, *args, **kwargs):
            verbose = kwargs.pop("verbose", False)
            output_type = kwargs.pop("output_type", None)
            Base.__init__(self, verbose=verbose, output_type=output_type)
            orig_init(self, *args, **kwargs)

        cls.__init__ = init


class TransformerMixin:
    """Mixin class for all transformers in scikit-learn."""

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit to data, then transform it.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        X : {array-like, sparse matrix, dataframe} of shape \
                (n_samples, n_features)

        y : ndarray of shape (n_samples,), default=None
            Target values.

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        X_new : ndarray array of shape (n_samples, n_features_new)
            Transformed array.
        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params).transform(X)


class BaseComposition:
    """Handles parameter management for classifiers composed of named estimators.
    """

    def _get_params(self, attr, deep=True):
        out = super().get_params(deep=deep)
        if not deep:
            return out
        estimators = getattr(self, attr)
        out.update(estimators)
        for name, estimator in estimators:
            if hasattr(estimator, 'get_params'):
                for key, value in estimator.get_params(deep=True).items():
                    out['%s__%s' % (name, key)] = value
        return out

    def _set_params(self, attr, **params):
        # Ensure strict ordering of parameter setting:
        # 1. All steps
        if attr in params:
            setattr(self, attr, params.pop(attr))
        # 2. Step replacement
        items = getattr(self, attr)
        names = []
        if items:
            names, _ = zip(*items)
        for name in list(params.keys()):
            if '__' not in name and name in names:
                self._replace_estimator(attr, name, params.pop(name))
        # 3. Nested step parameters: route ``<name>__<param>`` to the estimator
        nested_params = defaultdict(dict)
        for key in list(params.keys()):
            name, delim, sub_key = key.partition('__')
            if delim and name in names:
                nested_params[name][sub_key] = params.pop(key)
        estimators = dict(getattr(self, attr))
        for name, sub_params in nested_params.items():
            estimators[name].set_params(**sub_params)
        # 4. Other initialisation arguments
        super().set_params(**params)
        return self

    def _replace_estimator(self, attr, name, new_val):
        # assumes `name` is a valid estimator name
        new_estimators = list(getattr(self, attr))
        for i, (estimator_name, _) in enumerate(new_estimators):
            if estimator_name == name:
                new_estimators[i] = (name, new_val)
                break
        setattr(self, attr, new_estimators)

    def _validate_names(self, names):
        if len(set(names)) != len(names):
            raise ValueError('Names provided are not unique: '
                             '{0!r}'.format(list(names)))
        invalid_names = set(names).intersection(self.get_params(deep=False))
        if invalid_names:
            raise ValueError('Estimator names conflict with constructor '
                             'arguments: {0!r}'.format(sorted(invalid_names)))
        invalid_names = [name for name in names if '__' in name]
        if invalid_names:
            raise ValueError('Estimator names must not contain __: got '
                             '{0!r}'.format(invalid_names))
