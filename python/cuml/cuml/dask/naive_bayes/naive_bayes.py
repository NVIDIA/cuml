#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import dask
import dask.array
from toolz import first

from cuml.dask.common.base import BaseEstimator, DelayedPredictionMixin
from cuml.dask.common.func import reduce, tree_reduce
from cuml.dask.common.input_utils import DistributedDataHandler
from cuml.dask.common.utils import wait_and_raise_from_futures
from cuml.naive_bayes import MultinomialNB as MNB


def _count_accurate_predictions(y_hat, y):
    y_hat = cp.asarray(y_hat, dtype=y_hat.dtype)
    y = cp.asarray(y, dtype=y.dtype)
    return y.shape[0] - cp.count_nonzero(y - y_hat)


class MultinomialNB(BaseEstimator, DelayedPredictionMixin):
    """
    Distributed Naive Bayes classifier for multinomial models

    Examples
    --------

    Load the 20 newsgroups dataset from Scikit-learn and train a
    Naive Bayes classifier.

    .. code-block:: python

        >>> import cupy as cp

        >>> from sklearn.datasets import fetch_20newsgroups
        >>> from sklearn.feature_extraction.text import CountVectorizer

        >>> from dask_cuda import LocalCUDACluster
        >>> from dask.distributed import Client
        >>> import dask
        >>> from cuml.dask.common import to_sparse_dask_array
        >>> from cuml.dask.naive_bayes import MultinomialNB

        >>> # Create a local CUDA cluster
        >>> cluster = LocalCUDACluster()
        >>> client = Client(cluster)

        >>> # Load corpus
        >>> twenty_train = fetch_20newsgroups(subset='train',
        ...                           shuffle=True, random_state=42)

        >>> cv = CountVectorizer()
        >>> xformed = cv.fit_transform(twenty_train.data).astype(cp.float32)
        >>> X = to_sparse_dask_array(xformed, client)
        >>> y = dask.array.from_array(twenty_train.target, asarray=False,
        ...                       fancy=False).astype(cp.int32)

        >>> # Train model
        >>> model = MultinomialNB()
        >>> model.fit(X, y)
        <cuml.dask.naive_bayes.naive_bayes.MultinomialNB object at 0x...>

        >>> # Compute accuracy on training set
        >>> model.score(X, y)  # doctest: +SKIP
        array(0.924...)
        >>> client.close()
        >>> cluster.close()

    """

    def __init__(self, *, client=None, verbose=False, **kwargs):
        """
        Create new multinomial distributed Naive Bayes classifier instance

        Parameters
        ----------
        client : dask.distributed.Client, optional
            Dask client to use
        """
        super().__init__(client=client, verbose=verbose, **kwargs)

        self.datatype = "cupy"

        # Make any potential model args available and catch any potential
        # ValueErrors before distributed training begins.
        self._set_internal_model(MNB(**kwargs))

    @staticmethod
    def _fit(Xy, classes, kwargs):
        X, y = Xy

        model = MNB(**kwargs)
        model.partial_fit(X, y, classes=classes)

        return model

    @staticmethod
    def _merge_counts_to_model(models):
        modela = first(models)

        for model in models[1:]:
            modela.feature_count_ += model.feature_count_
            modela.class_count_ += model.class_count_
        return modela

    @staticmethod
    def _update_log_probs(model):
        model.update_log_probs()
        return model

    def fit(self, X, y, classes=None):
        """
        Fit distributed Naive Bayes classifier model

        Parameters
        ----------

        X : dask.Array with blocks containing dense or sparse cupy arrays
        y : dask.Array with blocks containing cupy.ndarray
        classes : array-like containing unique class labels

        Returns
        -------

        cuml.dask.naive_bayes.MultinomialNB current model instance
        """

        # Only Dask.Array supported for now
        if not isinstance(X, dask.array.core.Array):
            raise ValueError("Only dask.Array is supported for X")

        if not isinstance(y, dask.array.core.Array):
            raise ValueError("Only dask.Array is supported for y")

        if len(X.chunks[1]) != 1:
            raise ValueError(
                "X must be chunked by row only. "
                "Multi-dimensional chunking is not supported"
            )

        futures = DistributedDataHandler.create([X, y], self.client)

        classes = (
            cp.unique(y.map_blocks(cp.unique).compute())
            if classes is None
            else classes
        )

        models = [
            self.client.submit(
                self._fit, part, classes, self.kwargs, pure=False
            )
            for w, part in futures.gpu_futures
        ]

        models = reduce(
            models, self._merge_counts_to_model, client=self.client
        )

        models = self.client.submit(self._update_log_probs, models, pure=False)

        wait_and_raise_from_futures([models])

        self._set_internal_model(models)

        return self

    @staticmethod
    def _get_part(parts, idx):
        return parts[idx]

    @staticmethod
    def _get_size(arrs):
        return arrs.shape[0]

    def predict(self, X):
        """
        Use distributed Naive Bayes model to predict the classes for a
        given set of data samples.

        Parameters
        ----------

        X : dask.Array with blocks containing dense or sparse cupy arrays


        Returns
        -------

        dask.Array containing predicted classes

        """
        # TODO: This could be refactored to use DelayedPredictionMixin
        # Ref: https://github.com/NVIDIA/cuml/issues/1834
        # Ref: https://github.com/NVIDIA/cuml/issues/1387
        if not isinstance(X, dask.array.core.Array):
            raise ValueError("Only dask.Array is supported for X")

        return self._predict(X, delayed=True, output_dtype=cp.int32)

    def score(self, X, y):
        """
        Compute accuracy score

        Parameters
        ----------

        X : Dask.Array
            Features to predict. Note- it is assumed that chunk sizes and
            shape of X are known. This can be done for a fully delayed
            Array by calling X.compute_chunks_sizes()
        y : Dask.Array
            Labels to use for computing accuracy. Note- it is assumed that
            chunk sizes and shape of X are known. This can be done for a fully
            delayed Array by calling X.compute_chunks_sizes()

        Returns
        -------
        score : float the resulting accuracy score
        """

        y_hat = self.predict(X)

        delayed_parts = zip(y_hat.to_delayed(), y.to_delayed())

        func = dask.delayed(_count_accurate_predictions)

        accuracy_parts = [func(*p) for p in delayed_parts]

        reduced = first(dask.compute(tree_reduce(accuracy_parts)))

        return reduced / X.shape[0]
