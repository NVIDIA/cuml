#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import cupyx
import numpy as np

import cuml.internals.nvtx as nvtx
from cuml.common.doc_utils import generate_docstring
from cuml.internals.base import Base
from cuml.internals.mixins import ClassifierMixin, SparseInputTagMixin
from cuml.internals.outputs import ClassLabels, ReflectedAttr, mlfunc
from cuml.internals.validation import (
    check_array,
    check_inputs,
    check_is_fitted,
)

_binarize = cp.ElementwiseKernel(
    "T x, float32 threshold",
    "T out",
    "out = x > threshold ? 1 : 0",
    "binarize",
)


def _count_classes(y, n_classes, dtype):
    """
    Count samples per class.

    Parameters
    ----------
    y : cupy.ndarray
        Class labels (monotonic, 0 to n_classes-1)
    n_classes : int
        Number of classes
    dtype : dtype
        Output dtype

    Returns
    -------
    class_counts : cupy.ndarray of shape (n_classes,)
        Count of samples per class
    """
    return cp.bincount(y, minlength=n_classes).astype(dtype, copy=False)


class _BaseNB(ClassifierMixin, SparseInputTagMixin, Base):
    """A base class for all naive-bayes estimators"""

    class_count_ = ReflectedAttr()
    feature_count_ = ReflectedAttr()
    # a tuple of dtypes to coerce X to
    _supported_dtypes = ("float32", "float64")

    def _transform_X(self, X):
        """An optional transform to apply to X after it's been validated"""
        return X

    def _check_predict(self, X):
        """Validate and return X for predict."""
        X = check_inputs(
            self,
            X,
            dtype=self._supported_dtypes,
            sample_weight_dtype=("float32", "float64"),
            accept_sparse=["coo", "csr"],
            ensure_non_negative=self.__sklearn_tags__().input_tags.positive_only,
        )
        return self._transform_X(X)

    def _check_fit(
        self,
        X,
        y,
        classes=None,
        sample_weight=None,
        *,
        reset=False,
    ):
        """Validate and return (X, y, classes, sample_weight) for fit."""
        if reset:
            self._set_output_type(X)
        elif classes is None:
            # Not the first call, and no `classes` passed: encode `y`
            # against the already-established `self.classes_` rather than
            # letting `check_inputs` derive classes from just this batch,
            # which would silently corrupt updates for any class not
            # present in the current batch.
            classes = self.classes_
        X, y, sample_weight, classes = check_inputs(
            self,
            X,
            y,
            sample_weight=sample_weight,
            dtype=self._supported_dtypes,
            y_dtype=None,
            sample_weight_dtype=("float32", "float64"),
            accept_sparse=["coo", "csr"],
            ensure_non_negative=self.__sklearn_tags__().input_tags.positive_only,
            return_classes=(True if classes is None else classes),
            reset=reset,
        )
        X = self._transform_X(X)
        y = y.astype(
            "int32" if len(classes) < (2**31 - 1) else "int64", copy=False
        )
        return X, y, classes, sample_weight

    def _check_classes(self, classes, reset=False):
        """Validate `classes`, and return (classes, reset)."""
        classes_ = getattr(self, "classes_", None)

        if classes is not None:
            classes = np.unique(
                check_array(
                    classes,
                    ensure_2d=False,
                    mem_type="host",
                    input_name="classes",
                )
            )
            if classes_ is not None:
                if not np.array_equal(classes_, classes):
                    raise ValueError(
                        "`classes=%r` is not the same as on last call "
                        "to partial_fit, was: %r" % (classes, classes_)
                    )
        elif classes_ is None and not reset:
            raise ValueError(
                "classes must be passed on the first call to partial_fit."
            )

        return classes, (reset or classes_ is None)

    @generate_docstring(
        X="dense_sparse",
        return_values={
            "name": "y_hat",
            "type": "dense",
            "description": "Predicted values",
            "shape": "(n_rows, 1)",
        },
    )
    @mlfunc(preserve_index=True)
    def predict(self, X):
        """
        Perform classification on an array of test vectors X.

        """
        check_is_fitted(self)

        X = self._check_predict(X)
        jll = self._joint_log_likelihood(X)
        indices = cp.argmax(jll, axis=1)
        return ClassLabels(indices, self.classes_)

    @generate_docstring(
        X="dense_sparse",
        return_values={
            "name": "C",
            "type": "dense",
            "description": (
                "Returns the log-probability of the samples for each class in "
                "the model. The columns correspond to the classes in sorted "
                "order, as they appear in the attribute `classes_`."
            ),
            "shape": "(n_rows, 1)",
        },
    )
    @mlfunc(preserve_index=True)
    def predict_log_proba(self, X):
        """
        Return log-probability estimates for the test vector X.

        """
        check_is_fitted(self)
        X = self._check_predict(X)
        jll = self._joint_log_likelihood(X)

        # normalize by P(X) = P(f_1, ..., f_n)
        # Compute log(sum(exp()))
        # Subtract max in exp to prevent inf
        a_max = cp.amax(jll, axis=1, keepdims=True)
        exp = cp.exp(jll - a_max)
        logsumexp = cp.log(cp.sum(exp, axis=1))
        log_prob_x = cp.squeeze(a_max, axis=1) + logsumexp
        if log_prob_x.ndim < 2:
            log_prob_x = log_prob_x.reshape((1, log_prob_x.shape[0]))
        return jll - log_prob_x.T

    @generate_docstring(
        X="dense_sparse",
        return_values={
            "name": "C",
            "type": "dense",
            "description": (
                "Returns the probability of the samples for each class in the "
                "model. The columns correspond to the classes in sorted order,"
                " as they appear in the attribute `classes_`."
            ),
            "shape": "(n_rows, 1)",
        },
    )
    @mlfunc(preserve_index=True)
    def predict_proba(self, X):
        """
        Return probability estimates for the test vector X.
        """
        log_proba = self.predict_log_proba(X)
        cp.exp(log_proba, out=log_proba)
        return log_proba


class GaussianNB(_BaseNB):
    """
    Gaussian Naive Bayes (GaussianNB)

    Can perform online updates to model parameters via :meth:`partial_fit`.
    For details on algorithm used to update feature means and variance online,
    see Stanford CS tech report STAN-CS-79-773 by Chan, Golub, and LeVeque:

        http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf

    Parameters
    ----------
    priors : array-like of shape (n_classes,)
        Prior probabilities of the classes. If specified the priors are not
        adjusted according to the data.
    var_smoothing : float, default=1e-9
        Portion of the largest variance of all features that is added to
        variances for calculation stability.
    output_type : {None, 'input', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.

    Examples
    --------
    >>> import cupy as cp
    >>> from cuml.naive_bayes import GaussianNB
    >>> X = cp.array(
    ...     [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]],
    ...     dtype=cp.float32
    ... )
    >>> y = cp.array([1, 1, 1, 2, 2, 2])
    >>> clf = GaussianNB().fit(X, y)
    >>> print(clf.predict(cp.array([[-0.8, -1]], cp.float32)))
    [1]
    """

    class_prior_ = ReflectedAttr()

    def __init__(
        self,
        *,
        priors=None,
        var_smoothing=1e-9,
        output_type=None,
        verbose=False,
    ):
        super().__init__(verbose=verbose, output_type=output_type)
        self.priors = priors
        self.var_smoothing = var_smoothing

    @mlfunc(convert_output=False)
    def fit(self, X, y, sample_weight=None) -> "GaussianNB":
        """
        Fit Gaussian Naive Bayes classifier according to X, y

        Parameters
        ----------
        X : {array-like, cupy sparse matrix} of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like shape (n_samples)
            Target values.
        sample_weight : array-like of shape (n_samples)
            Weights applied to individual samples.
        """
        return self._partial_fit(
            X,
            y,
            reset=True,
            sample_weight=sample_weight,
        )

    @nvtx.annotate(
        message="naive_bayes.GaussianNB._partial_fit", domain="cuml_python"
    )
    def _partial_fit(
        self,
        X,
        y,
        classes=None,
        sample_weight=None,
        reset=False,
    ) -> "GaussianNB":
        classes, reset = self._check_classes(classes, reset)
        X, y, classes, sample_weight = self._check_fit(
            X,
            y,
            classes=classes,
            sample_weight=sample_weight,
            reset=reset,
        )

        self.epsilon_ = self.var_smoothing * (
            ((X - X.mean(axis=0)) ** 2).mean(axis=0).max()
        )

        if reset:
            self.classes_ = classes
            n_features = X.shape[1]
            n_classes = len(self.classes_)

            self.n_classes_ = n_classes
            self.n_features_ = n_features

            self.theta_ = cp.zeros((n_classes, n_features))
            self.sigma_ = cp.zeros((n_classes, n_features))

            self.class_count_ = cp.zeros(n_classes, dtype=X.dtype)

            if self.priors is not None:
                priors = check_array(
                    self.priors, dtype=X.dtype, ensure_2d=False
                )
                if priors.shape[0] != n_classes:
                    raise ValueError(
                        "Number of priors must match number of classes."
                    )
                if not cp.isclose(priors.sum(), 1.0):
                    raise ValueError("The sum of the priors should be 1.")
                if (priors < 0).any():
                    raise ValueError("Priors must be non-negative.")
                self.class_prior_ = priors
        else:
            self.sigma_[:, :] -= self.epsilon_

        # Convert sparse matrices to CSR for efficient row indexing
        if cupyx.scipy.sparse.issparse(X):
            X = X.tocsr()

        # Update mean and variance for each class
        # Following scikit-learn's approach: iterate through unique labels
        for i in cp.unique(y):
            # Explicit indices can index sparse arrays
            mask = y == i
            indices = cp.where(mask)[0]

            # Index X using integer indices (works efficiently with CSR)
            X_i = X[indices, :]

            if sample_weight is not None:
                sw_i = sample_weight[indices]
                N_i = float(sw_i.sum())
            else:
                sw_i = None
                N_i = X_i.shape[0]

            # Update mean and variance for this class
            new_theta, new_sigma = self._update_mean_variance(
                self.class_count_[i],
                self.theta_[i, :],
                self.sigma_[i, :],
                X_i,
                sw_i,
            )

            self.theta_[i, :] = new_theta
            self.sigma_[i, :] = new_sigma
            self.class_count_[i] += N_i

        self.sigma_[:, :] += self.epsilon_

        if self.priors is None:
            self.class_prior_ = self.class_count_ / self.class_count_.sum()

        return self

    @mlfunc(convert_output=False)
    def partial_fit(
        self, X, y, classes=None, sample_weight=None
    ) -> "GaussianNB":
        """
        Incremental fit on a batch of samples.

        This method is expected to be called several times consecutively on
        different chunks of a dataset so as to implement out-of-core or online
        learning.

        This is especially useful when the whole dataset is too big to fit in
        memory at once.

        This method has some performance overhead hence it is better to call
        partial_fit on chunks of data that are as large as possible (as long
        as fitting in the memory budget) to hide the overhead.

        Parameters
        ----------
        X : {array-like, cupy sparse matrix} of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features. A sparse matrix in COO
            format is preferred, other formats will go through a conversion
            to COO.
        y : array-like of shape (n_samples)
            Target values.
        classes : array-like of shape (n_classes)
            List of all the classes that can possibly appear in the y vector.
            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.
        sample_weight : array-like of shape (n_samples)
            Weights applied to individual samples.

        Returns
        -------
        self : object
        """
        return self._partial_fit(
            X, y, classes=classes, sample_weight=sample_weight
        )

    @staticmethod
    def _update_mean_variance(n_past, mu, var, X, sample_weight=None):
        """Compute online update of Gaussian mean and variance.

        This is a direct port of scikit-learn's implementation to CuPy,
        with efficient handling of both dense and sparse matrices.
        Given starting sample count, mean, and variance, a new set of
        points X, and optionally sample weights, return the updated mean and
        variance.

        Parameters
        ----------
        n_past : float
            Number of samples represented in old mean and variance.
        mu : cupy.ndarray of shape (n_features,)
            Means for Gaussians in original set.
        var : cupy.ndarray of shape (n_features,)
            Variances for Gaussians in original set.
        X : cupy.ndarray or cupyx.scipy.sparse matrix of shape (n_samples, n_features)
            New data points. Can be dense or sparse.
        sample_weight : cupy.ndarray of shape (n_samples,), optional
            Weights applied to individual samples.

        Returns
        -------
        total_mu : cupy.ndarray of shape (n_features,)
            Updated mean for each Gaussian over the combined set.
        total_var : cupy.ndarray of shape (n_features,)
            Updated variance for each Gaussian over the combined set.
        """
        if X.shape[0] == 0:
            return mu, var

        # Compute (potentially weighted) mean and variance of new datapoints
        if sample_weight is not None:
            n_new = float(sample_weight.sum())
            if cp.isclose(n_new, 0.0):
                return mu, var

            # Handle sparse vs dense differently for efficiency
            if cupyx.scipy.sparse.issparse(X):
                # Sparse weighted mean - avoid densification
                # X.T @ sample_weight gives sum of weighted features
                new_mu = cp.asarray((X.T.dot(sample_weight) / n_new)).ravel()
                # Sparse weighted variance using E[X²] - E[X]²
                # This avoids creating a dense difference matrix
                X_squared = X.power(2)
                new_var = (
                    cp.asarray(
                        (X_squared.T.dot(sample_weight) / n_new)
                    ).ravel()
                    - new_mu**2
                )
            else:
                # Dense weighted case
                new_mu = cp.average(X, axis=0, weights=sample_weight)
                new_var = cp.average(
                    (X - new_mu) ** 2, axis=0, weights=sample_weight
                )
        else:
            n_new = X.shape[0]

            # Handle sparse vs dense differently for efficiency
            if cupyx.scipy.sparse.issparse(X):
                # Sparse unweighted - efficient for sparse matrices
                # mean() works efficiently on sparse matrices
                new_mu = cp.asarray(X.mean(axis=0)).ravel()
                # Variance: E[X²] - E[X]² (avoids creating dense diff matrix)
                X_squared = X.power(2)
                new_var = (
                    cp.asarray(X_squared.mean(axis=0)).ravel() - new_mu**2
                )
            else:
                # Dense unweighted case
                new_var = cp.var(X, axis=0)
                new_mu = cp.mean(X, axis=0)

        if n_past == 0:
            return new_mu, new_var

        n_total = float(n_past + n_new)

        # Combine mean of old and new data
        total_mu = (n_new * new_mu + n_past * mu) / n_total

        # Combine variance using sum-of-squared-differences
        old_ssd = n_past * var
        new_ssd = n_new * new_var
        total_ssd = (
            old_ssd + new_ssd + (n_new * n_past / n_total) * (mu - new_mu) ** 2
        )
        total_var = total_ssd / n_total

        return total_mu, total_var

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of samples for each class."""
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        # Pre-compute log of 2*pi*sigma for all classes at once
        # Shape: (n_classes, n_features)
        log_2pi_sigma = cp.log(2.0 * cp.pi * self.sigma_)

        # Initialize joint log likelihood array
        joint_log_likelihood = cp.zeros((n_samples, n_classes))

        for i in range(n_classes):
            jointi = cp.log(self.class_prior_[i])

            # Compute the constant term for this class
            n_ij = -0.5 * cp.sum(log_2pi_sigma[i, :])

            # Compute squared Mahalanobis distance efficiently
            # (X - theta[i])^2 / sigma[i]
            diff = (
                X - self.theta_[i, :]
            )  # Broadcasting: (n_samples, n_features)
            scaled_diff_sq = (diff**2) / self.sigma_[i, :]

            # Sum over features for each sample
            mahalanobis = -0.5 * cp.sum(scaled_diff_sq, axis=1)

            # Combine all terms
            joint_log_likelihood[:, i] = jointi + n_ij + mahalanobis

        return joint_log_likelihood

    @classmethod
    def _get_param_names(cls):
        return [*super()._get_param_names(), "priors", "var_smoothing"]


class _BaseDiscreteNB(_BaseNB):
    class_log_prior_ = ReflectedAttr()
    feature_log_prob_ = ReflectedAttr()
    _supported_dtypes = ("float32", "float64", "int32", "int64")

    def __init__(
        self,
        *,
        alpha=1.0,
        fit_prior=True,
        class_prior=None,
        verbose=False,
        output_type=None,
    ):
        super().__init__(verbose=verbose, output_type=output_type)
        self.class_prior = class_prior
        self.alpha = alpha
        self.fit_prior = fit_prior

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        return tags

    def _update_class_log_prior(self, class_prior=None):
        if class_prior is not None:
            if class_prior.shape[0] != self.n_classes_:
                raise ValueError(
                    "Number of classes must match number of priors"
                )

            self.class_log_prior_ = cp.log(cp.asarray(class_prior))

        elif self.fit_prior:
            log_class_count = cp.log(self.class_count_)
            self.class_log_prior_ = log_class_count - cp.log(
                self.class_count_.sum()
            )
        else:
            self.class_log_prior_ = cp.full(
                self.n_classes_, -np.log(self.n_classes_)
            )

    @mlfunc(convert_output=False)
    def partial_fit(
        self,
        X,
        y,
        classes=None,
    ) -> "_BaseDiscreteNB":
        """
        Incremental fit on a batch of samples.

        This method is expected to be called several times consecutively on
        different chunks of a dataset so as to implement out-of-core or online
        learning.

        This is especially useful when the whole dataset is too big to fit in
        memory at once.

        This method has some performance overhead hence it is better to call
        partial_fit on chunks of data that are as large as possible (as long
        as fitting in the memory budget) to hide the overhead.

        Parameters
        ----------
        X : {array-like, cupy sparse matrix} of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features
        y : array-like of shape (n_samples)
            Target values.
        classes : array-like of shape (n_classes)
            List of all the classes that can possibly appear in the y vector.
            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.

        Returns
        -------
        self : object
        """
        return self._partial_fit(X, y, classes=classes)

    @nvtx.annotate(
        message="naive_bayes._BaseDiscreteNB._partial_fit",
        domain="cuml_python",
    )
    def _partial_fit(
        self,
        X,
        y,
        classes=None,
        reset=False,
    ) -> "_BaseDiscreteNB":
        if self.alpha < 0:
            raise ValueError(f"Expected alpha >= 0, got {self.alpha}")

        classes, reset = self._check_classes(classes, reset)
        X, y, classes, _ = self._check_fit(
            X,
            y,
            classes=classes,
            reset=reset,
        )

        if reset:
            self.classes_ = classes
            self.n_classes_ = self.classes_.shape[0]
            self.n_features_ = X.shape[1]
            self._init_counters(self.n_classes_, self.n_features_, X.dtype)

        if cupyx.scipy.sparse.issparse(X):
            X = X.tocoo()
            self._count_sparse(X.row, X.col, X.data, X.shape, y)
        else:
            self._count(X, y)

        self._update_feature_log_prob(self.alpha)
        self._update_class_log_prior(class_prior=self.class_prior)
        return self

    @mlfunc(convert_output=False)
    def fit(self, X, y) -> "_BaseDiscreteNB":
        """
        Fit Naive Bayes classifier according to X, y

        Parameters
        ----------
        X : {array-like, cupy sparse matrix} of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like shape (n_samples)
            Target values.
        """
        return self._partial_fit(X, y, reset=True)

    def _init_counters(self, n_effective_classes, n_features, dtype):
        self.class_count_ = cp.zeros(
            n_effective_classes, order="F", dtype=dtype
        )
        self.feature_count_ = cp.zeros(
            (n_effective_classes, n_features), order="F", dtype=dtype
        )

    def update_log_probs(self):
        """
        Updates the log probabilities. This enables lazy update for
        applications like distributed Naive Bayes, so that the model
        can be updated incrementally without incurring this cost each
        time.
        """
        self._update_feature_log_prob(self.alpha)
        self._update_class_log_prior(class_prior=self.class_prior)

    def _count(self, X, y):
        """
        Sum feature counts & class prior counts and add to current model.
        Parameters
        ----------
        X : cupy.ndarray or cupyx.scipy.sparse matrix, shape (n_rows, n_features)
        y : cupy.array of monotonic class labels
        """
        n_features = X.shape[1]

        # Sum features per class
        # counts[class, feature] = sum of X[i, feature] where y[i] == class
        counts = cp.zeros(
            (self.n_classes_, n_features), dtype=X.dtype, order="F"
        )

        # Vectorized approach using advanced indexing
        for class_idx in range(self.n_classes_):
            mask = y == class_idx
            if cp.any(mask):
                counts[class_idx] = X[mask].sum(axis=0)

        # Count samples per class
        class_c = _count_classes(y, self.n_classes_, X.dtype)

        self.feature_count_ += counts
        self.class_count_ += class_c

    def _count_sparse(self, x_coo_rows, x_coo_cols, x_coo_data, x_shape, y):
        """
        Sum feature counts & class prior counts and add to current model.

        Parameters
        ----------
        x_coo_rows : cupy.ndarray of size (nnz)
        x_coo_cols : cupy.ndarray of size (nnz)
        x_coo_data : cupy.ndarray of size (nnz)
        y : cupy.array of monotonic class labels
        """
        n_rows, n_cols = x_shape
        counts = cp.zeros(
            (self.n_classes_, n_cols), dtype=x_coo_data.dtype, order="F"
        )

        # For each non-zero element, add its value to the appropriate (class,
        # feature) bin. Get the class label for each non-zero element.
        labels_for_nnz = y[x_coo_rows]

        # Create flat indices for the counts array: col * n_classes + label
        flat_indices = x_coo_cols * self.n_classes_ + labels_for_nnz

        # Use bincount to accumulate values - this is the key GPU-efficient operation
        flat_counts = cp.bincount(
            flat_indices,
            weights=x_coo_data,
            minlength=n_cols * self.n_classes_,
        )

        # Reshape to (n_classes, n_cols) but bincount gives us col-major order
        # so we need to transpose
        counts = flat_counts.reshape(n_cols, self.n_classes_).T.astype(
            x_coo_data.dtype
        )

        class_c = _count_classes(y, self.n_classes_, x_coo_data.dtype)

        self.feature_count_ = self.feature_count_ + counts
        self.class_count_ = self.class_count_ + class_c

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "alpha",
            "fit_prior",
            "class_prior",
        ]


class MultinomialNB(_BaseDiscreteNB):
    """
    Naive Bayes classifier for multinomial models.

    The multinomial Naive Bayes classifier is suitable for classification
    with discrete features (e.g., word counts for text classification).

    The multinomial distribution normally requires integer feature counts.
    However, in practice, fractional counts such as tf-idf may also work.

    Parameters
    ----------
    alpha : float (default=1.0)
        Additive (Laplace/Lidstone) smoothing parameter (0 for no
        smoothing).
    fit_prior : boolean (default=True)
        Whether to learn class prior probabilities or no. If false, a
        uniform prior will be used.
    class_prior : array-like, size (n_classes) (default=None)
        Prior probabilities of the classes. If specified, the priors are
        not adjusted according to the data.
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
    class_count_ : ndarray of shape (n_classes)
        Number of samples encountered for each class during fitting.
    class_log_prior_ : ndarray of shape (n_classes)
        Log probability of each class (smoothed).
    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier
    feature_count_ : ndarray of shape (n_classes, n_features)
        Number of samples encountered for each (class, feature)
        during fitting.
    feature_log_prob_ : ndarray of shape (n_classes, n_features)
        Empirical log probability of features given a class, P(x_i|y).
    n_features_ : int
        Number of features of each sample.

    Examples
    --------
    Load the 20 newsgroups dataset from Scikit-learn and train a
    Naive Bayes classifier.

    >>> from sklearn.datasets import fetch_20newsgroups
    >>> from sklearn.feature_extraction.text import CountVectorizer
    >>> from cuml.naive_bayes import MultinomialNB
    >>> data = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
    >>> X = CountVectorizer().fit_transform(data.data)
    >>> model = MultinomialNB().fit(X, data.target)
    >>> model.score(X, y)  # doctest: +SKIP
    0.9245...
    """

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.positive_only = True
        return tags

    def _update_feature_log_prob(self, alpha):
        """
        Apply add-lambda smoothing to raw counts and recompute
        log probabilities

        Parameters
        ----------
        alpha : float amount of smoothing to apply (0. means no smoothing)
        """
        smoothed_fc = self.feature_count_ + alpha
        smoothed_cc = smoothed_fc.sum(axis=1).reshape(-1, 1)
        self.feature_log_prob_ = cp.log(smoothed_fc) - cp.log(
            smoothed_cc.reshape(-1, 1)
        )

    def _joint_log_likelihood(self, X):
        """
        Calculate the posterior log probability of the samples X

        Parameters
        ----------
        X : array-like of size (n_samples, n_features)
        """
        ret = X.dot(self.feature_log_prob_.T)
        ret += self.class_log_prior_
        return ret


class BernoulliNB(_BaseDiscreteNB):
    """
    Naive Bayes classifier for multivariate Bernoulli models.

    Like MultinomialNB, this classifier is suitable for discrete data. The
    difference is that while MultinomialNB works with occurrence counts,
    BernoulliNB is designed for binary/boolean features.

    Parameters
    ----------
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter
        (0 for no smoothing).
    binarize : float or None, default=0.0
        Threshold for binarizing (mapping to booleans) of sample features.
        If None, input is presumed to already consist of binary vectors.
    fit_prior : bool, default=True
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.
    class_prior : array-like of shape (n_classes,), default=None
        Prior probabilities of the classes. If specified the priors are not
        adjusted according to the data.
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
    class_count_ : ndarray of shape (n_classes)
        Number of samples encountered for each class during fitting.
    class_log_prior_ : ndarray of shape (n_classes)
        Log probability of each class (smoothed).
    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier
    feature_count_ : ndarray of shape (n_classes, n_features)
        Number of samples encountered for each (class, feature)
        during fitting.
    feature_log_prob_ : ndarray of shape (n_classes, n_features)
        Empirical log probability of features given a class, P(x_i|y).
    n_features_ : int
        Number of features of each sample.

    Examples
    --------
    >>> import cupy as cp
    >>> from cuml.naive_bayes import BernoulliNB
    >>> rng = cp.random.RandomState(1)
    >>> X = rng.randint(5, size=(6, 100), dtype=cp.int32)
    >>> y = cp.array([1, 2, 3, 4, 4, 5])
    >>> clf = BernoulliNB()
    >>> clf.fit(X, y)
    BernoulliNB()
    >>> print(clf.predict(X[2:3]))
    [3]

    References
    ----------
    C.D. Manning, P. Raghavan and H. Schuetze (2008). Introduction to
    Information Retrieval. Cambridge University Press, pp. 234-265.
    https://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html
    A. McCallum and K. Nigam (1998). A comparison of event models for naive
    Bayes text classification. Proc. AAAI/ICML-98 Workshop on Learning for
    Text Categorization, pp. 41-48.
    V. Metsis, I. Androutsopoulos and G. Paliouras (2006). Spam filtering with
    naive Bayes -- Which naive Bayes? 3rd Conf. on Email and Anti-Spam (CEAS).
    """

    def __init__(
        self,
        *,
        alpha=1.0,
        binarize=0.0,
        fit_prior=True,
        class_prior=None,
        output_type=None,
        verbose=False,
    ):
        super().__init__(
            alpha=alpha,
            fit_prior=fit_prior,
            class_prior=class_prior,
            output_type=output_type,
            verbose=verbose,
        )
        self.binarize = binarize

    def _transform_X(self, X):
        if self.binarize is None:
            return X
        elif cupyx.scipy.sparse.issparse(X):
            X = X.copy()
            X.data = _binarize(X.data, self.binarize)
            return X
        else:
            return _binarize(X, self.binarize)

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""
        n_classes, n_features = self.feature_log_prob_.shape
        n_samples, n_features_X = X.shape

        neg_prob = cp.log(1 - cp.exp(self.feature_log_prob_))

        # Compute  neg_prob · (1 - X).T  as  ∑neg_prob - X · neg_prob
        jll = X.dot((self.feature_log_prob_ - neg_prob).T)
        jll += self.class_log_prior_ + neg_prob.sum(axis=1)

        return jll

    def _update_feature_log_prob(self, alpha):
        """
        Apply add-lambda smoothing to raw counts and recompute
        log probabilities

        Parameters
        ----------
        alpha : float amount of smoothing to apply (0. means no smoothing)
        """
        smoothed_fc = self.feature_count_ + alpha
        smoothed_cc = self.class_count_ + alpha * 2
        self.feature_log_prob_ = cp.log(smoothed_fc) - cp.log(
            smoothed_cc.reshape(-1, 1)
        )

    @classmethod
    def _get_param_names(cls):
        return [*super()._get_param_names(), "binarize"]


class ComplementNB(_BaseDiscreteNB):
    """
    The Complement Naive Bayes classifier described in Rennie et al. (2003).

    The Complement Naive Bayes classifier was designed to correct the "severe
    assumptions" made by the standard Multinomial Naive Bayes classifier. It is
    particularly suited for imbalanced data sets.

    Parameters
    ----------
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter
        (0 for no smoothing).
    fit_prior : bool, default=True
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.
    class_prior : array-like of shape (n_classes,), default=None
        Prior probabilities of the classes. If specified the priors are not
        adjusted according to the data.
    norm : bool, default=False
        Whether or not a second normalization of the weights is performed.
        The default behavior mirrors the implementation found in Mahout and
        Weka, which do not follow the full algorithm described in Table 9 of
        the paper.
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
    class_count_ : ndarray of shape (n_classes)
        Number of samples encountered for each class during fitting.
    class_log_prior_ : ndarray of shape (n_classes)
        Log probability of each class (smoothed).
    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier
    feature_count_ : ndarray of shape (n_classes, n_features)
        Number of samples encountered for each (class, feature)
        during fitting.
    feature_log_prob_ : ndarray of shape (n_classes, n_features)
        Empirical log probability of features given a class, P(x_i|y).
    n_features_ : int
        Number of features of each sample.

    Examples
    --------
    >>> import cupy as cp
    >>> from cuml.naive_bayes import ComplementNB
    >>> rng = cp.random.RandomState(1)
    >>> X = rng.randint(5, size=(6, 100), dtype=cp.int32)
    >>> y = cp.array([1, 2, 3, 4, 4, 5])
    >>> clf = ComplementNB()
    >>> clf.fit(X, y)
    ComplementNB()
    >>> print(clf.predict(X[2:3]))
    [3]

    References
    ----------
    Rennie, J. D., Shih, L., Teevan, J., & Karger, D. R. (2003).
    Tackling the poor assumptions of naive bayes text classifiers. In ICML
    (Vol. 3, pp. 616-623).
    https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf
    """

    def __init__(
        self,
        *,
        alpha=1.0,
        fit_prior=True,
        class_prior=None,
        norm=False,
        output_type=None,
        verbose=False,
    ):
        super().__init__(
            alpha=alpha,
            fit_prior=fit_prior,
            class_prior=class_prior,
            output_type=output_type,
            verbose=verbose,
        )
        self.norm = norm

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.positive_only = True
        return tags

    def _count(self, X, y):
        super()._count(X, y)
        self.feature_all_ = self.feature_count_.sum(axis=0)

    def _count_sparse(self, x_coo_rows, x_coo_cols, x_coo_data, x_shape, y):
        super()._count_sparse(x_coo_rows, x_coo_cols, x_coo_data, x_shape, y)
        self.feature_all_ = self.feature_count_.sum(axis=0)

    def _joint_log_likelihood(self, X):
        """Calculate the class scores for the samples in X."""
        jll = X.dot(self.feature_log_prob_.T)
        if len(self.class_count_) == 1:
            jll += self.class_log_prior_
        return jll

    def _update_feature_log_prob(self, alpha):
        """
        Apply smoothing to raw counts and compute the weights.

        Parameters
        ----------
        alpha : float amount of smoothing to apply (0. means no smoothing)
        """
        comp_count = self.feature_all_ + alpha - self.feature_count_
        logged = cp.log(comp_count / comp_count.sum(axis=1, keepdims=True))
        if self.norm:
            summed = logged.sum(axis=1, keepdims=True)
            feature_log_prob = logged / summed
        else:
            feature_log_prob = -logged
        self.feature_log_prob_ = feature_log_prob

    @classmethod
    def _get_param_names(cls):
        return [*super()._get_param_names(), "norm"]


class CategoricalNB(_BaseDiscreteNB):
    """
    Naive Bayes classifier for categorical features.

    The categorical Naive Bayes classifier is suitable for classification with
    discrete features that are categorically distributed. The categories of
    each feature are drawn from a categorical distribution.

    Parameters
    ----------
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter
        (0 for no smoothing).
    fit_prior : bool, default=True
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.
    class_prior : array-like of shape (n_classes,), default=None
        Prior probabilities of the classes. If specified the priors are not
        adjusted according to the data.
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
    category_count_ : ndarray of shape (n_features, n_classes, n_categories)
        With n_categories being the highest category of all the features.
        This array provides the number of samples encountered for each feature,
        class and category of the specific feature.
    class_count_ : ndarray of shape (n_classes,)
        Number of samples encountered for each class during fitting.
    class_log_prior_ : ndarray of shape (n_classes,)
        Smoothed empirical log probability for each class.
    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier
    feature_log_prob_ : ndarray of shape (n_features, n_classes, n_categories)
        With n_categories being the highest category of all the features.
        Each array of shape (n_classes, n_categories) provides the empirical
        log probability of categories given the respective feature
        and class, ``P(x_i|y)``.
        This attribute is not available when the model has been trained with
        sparse data.
    n_features_ : int
        Number of features of each sample.

    Examples
    --------
    >>> import cupy as cp
    >>> from cuml.naive_bayes import CategoricalNB
    >>> rng = cp.random.RandomState(1)
    >>> X = rng.randint(5, size=(6, 100), dtype=cp.int32)
    >>> y = cp.array([1, 2, 3, 4, 5, 6])
    >>> clf = CategoricalNB()
    >>> clf.fit(X, y)
    CategoricalNB()
    >>> print(clf.predict(X[2:3]))
    [3]
    """

    def __init__(
        self,
        *,
        alpha=1.0,
        fit_prior=True,
        class_prior=None,
        output_type=None,
        verbose=False,
    ):
        super().__init__(
            alpha=alpha,
            fit_prior=fit_prior,
            class_prior=class_prior,
            output_type=output_type,
            verbose=verbose,
        )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.positive_only = True
        return tags

    def _count_sparse(self, x_coo_rows, x_coo_cols, x_coo_data, x_shape, y):
        """
        Sum feature counts & class prior counts and add to current model.

        Parameters
        ----------
        x_coo_rows : cupy.ndarray of size (nnz)
        x_coo_cols : cupy.ndarray of size (nnz)
        x_coo_data : cupy.ndarray of size (nnz)
        y : cupy.array of monotonic class labels
        """
        n_classes = self.n_classes_
        n_cols = x_shape[1]
        x_coo_nnz = x_coo_rows.shape[0]
        x_coo_data = x_coo_data.astype("int32", copy=False)

        # Count samples per class
        class_c = _count_classes(y, n_classes, self.class_count_.dtype)

        highest_feature = int(x_coo_data.max()) + 1
        feature_diff = highest_feature - self.category_count_.shape[1]
        # In case of a partial fit, pad the array to have the highest feature
        if not cupyx.scipy.sparse.issparse(self.category_count_):
            self.category_count_ = cupyx.scipy.sparse.coo_matrix(
                (self.n_features_ * n_classes, highest_feature)
            )
        elif feature_diff > 0:
            self.category_count_ = cupyx.scipy.sparse.coo_matrix(
                self.category_count_,
                shape=(self.n_features_ * n_classes, highest_feature),
            )
        highest_feature = self.category_count_.shape[1]

        # Map sparse data to categorical counts
        # For each non-zero element at (row, col) with value val:
        # - Get the class label for that row
        # - Create an entry at (col + n_cols * label, val)
        labels_for_nnz = y[x_coo_rows]
        counts_rows = x_coo_cols + n_cols * labels_for_nnz
        counts_cols = x_coo_data

        # Create the sparse category count matrix
        counts = cupyx.scipy.sparse.coo_matrix(
            (cp.ones(x_coo_nnz), (counts_rows, counts_cols)),
            shape=(self.n_features_ * n_classes, highest_feature),
        ).tocsr()

        # Adjust with the missing (zeros) data of the sparse matrix
        for i in range(n_classes):
            counts[i * n_cols : (i + 1) * n_cols, 0] = (y == i).sum() - counts[
                i * n_cols : (i + 1) * n_cols
            ].sum(1)
        self.category_count_ = (self.category_count_ + counts).tocoo()
        self.class_count_ = self.class_count_ + class_c

    def _count(self, X, y):
        n_classes = self.n_classes_
        n_features = X.shape[1]
        X = X.astype("int32", copy=False)

        highest_feature = int(X.max()) + 1
        feature_diff = highest_feature - self.category_count_.shape[2]
        # In case of a partial fit, pad the array to have the highest feature
        if feature_diff > 0:
            self.category_count_ = cp.pad(
                self.category_count_,
                [(0, 0), (0, 0), (0, feature_diff)],
                "constant",
            )
        highest_feature = self.category_count_.shape[2]

        # Categorical counting
        # count occurrences of each category per class per feature
        counts_raw = cp.zeros(
            (n_features, n_classes, highest_feature), dtype=X.dtype, order="F"
        )
        for feature_idx in range(n_features):
            feature_vals = X[:, feature_idx].astype(cp.int32)
            # Create flat indices into counts_raw[feature_idx]
            # Index is: class * highest_feature + category
            flat_indices = y.astype(cp.int32) * highest_feature + feature_vals
            flat_counts = cp.bincount(
                flat_indices, minlength=n_classes * highest_feature
            )
            counts_raw[feature_idx] = flat_counts.reshape(
                n_classes, highest_feature
            ).astype(X.dtype)

        # Pad or trim to match expected highest_feature size
        if counts_raw.shape[2] < highest_feature:
            counts = cp.zeros(
                (self.n_features_, n_classes, highest_feature),
                order="F",
                dtype=X.dtype,
            )
            counts[:, :, : counts_raw.shape[2]] = counts_raw
        else:
            counts = counts_raw[:, :, :highest_feature]

        self.category_count_ += counts

        class_c = _count_classes(y, n_classes, self.class_count_.dtype)
        self.class_count_ += class_c

    def _init_counters(self, n_effective_classes, n_features, dtype):
        self.class_count_ = cp.zeros(
            n_effective_classes, order="F", dtype=cp.float64
        )
        self.category_count_ = cp.zeros(
            (n_features, n_effective_classes, 0), order="F", dtype="int32"
        )

    def _update_feature_log_prob(self, alpha):
        highest_feature = cp.zeros(self.n_features_, dtype=cp.float64)
        if cupyx.scipy.sparse.issparse(self.category_count_):
            # For sparse data we avoid the creation of the dense matrix
            # feature_log_prob_. This can be created on the fly during
            # the prediction without using as much memory.
            features = self.category_count_.row % self.n_features_
            cp.maximum.at(highest_feature, features, self.category_count_.col)
            highest_feature = (highest_feature + 1) * alpha

            smoothed_class_count = self.category_count_.sum(axis=1)
            smoothed_class_count = smoothed_class_count.reshape(
                (self.n_classes_, self.n_features_)
            ).T
            smoothed_class_count += highest_feature[:, cp.newaxis]
            smoothed_cat_count = cupyx.scipy.sparse.coo_matrix(
                self.category_count_
            )
            smoothed_cat_count.data = cp.log(smoothed_cat_count.data + alpha)
            self.smoothed_cat_count = smoothed_cat_count.tocsr()
            self.smoothed_class_count = cp.log(smoothed_class_count)
        else:
            indices = self.category_count_.nonzero()
            cp.maximum.at(highest_feature, indices[0], indices[2])
            highest_feature = (highest_feature + 1) * alpha

            smoothed_class_count = (
                self.category_count_.sum(axis=2)
                + highest_feature[:, cp.newaxis]
            )
            smoothed_cat_count = self.category_count_ + alpha
            self.feature_log_prob_ = cp.log(smoothed_cat_count) - cp.log(
                smoothed_class_count[:, :, cp.newaxis]
            )

    def _joint_log_likelihood(self, X):
        n_rows = X.shape[0]
        if cupyx.scipy.sparse.issparse(X):
            # For sparse data we assume that most categories will be zeros,
            # so we first compute the jll for categories 0
            features_zeros = self.smoothed_cat_count[:, 0].todense()
            features_zeros = features_zeros.reshape(
                self.n_classes_, self.n_features_
            ).T
            if self.alpha != 1.0:
                features_zeros[cp.where(features_zeros == 0)] += cp.log(
                    self.alpha
                )
            features_zeros -= self.smoothed_class_count
            features_zeros = features_zeros.sum(0)
            jll = cp.repeat(features_zeros[cp.newaxis, :], n_rows, axis=0)

            X = X.tocoo()
            col_indices = X.col
            X_data = X.data.astype("int32", copy=False)

            # Adjust with the non-zeros data by adding jll_data (non-zeros)
            # and subtracting jll_zeros which are the zeros
            # that were first computed
            for i in range(self.n_classes_):
                jll_data = self.smoothed_cat_count[
                    col_indices + i * self.n_features_, X_data
                ].ravel()
                jll_zeros = self.smoothed_cat_count[
                    col_indices + i * self.n_features_, 0
                ].todense()[:, 0]
                if self.alpha != 1.0:
                    jll_data[cp.where(jll_data == 0)] += cp.log(self.alpha)
                    jll_zeros[cp.where(jll_zeros == 0)] += cp.log(self.alpha)
                jll_data -= jll_zeros
                cp.add.at(jll[:, i], X.row, jll_data)

        else:
            col_indices = cp.indices(X.shape)[1].flatten()
            jll = self.feature_log_prob_[
                col_indices, :, X.ravel().astype("int32", copy=False)
            ]
            jll = jll.reshape((n_rows, self.n_features_, self.n_classes_))
            jll = jll.sum(1)
        jll += self.class_log_prior_
        return jll
