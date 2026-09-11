#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
import cupyx.scipy.sparse as cp_sp
import dask
import dask.array

import cuml.feature_extraction.text
from cuml.dask.common.base import BaseEstimator, DelayedTransformMixin
from cuml.dask.common.func import reduce
from cuml.dask.common.input_utils import DistributedDataHandler
from cuml.dask.common.utils import wait_and_raise_from_futures


def _get_df_and_n_samples(X):
    """Compute doc frequencies and n_samples for X"""
    X = cp_sp.csr_matrix(X)
    df = cp.bincount(X.indices, minlength=X.shape[1]).astype(
        X.dtype, copy=False
    )
    return df, X.shape[0]


def _merge_df_and_n_samples(parts):
    """Merge doc frequencies and n_samples for X"""
    dfs, ns = zip(*parts)
    df = cp.vstack(dfs).sum(axis=0)
    n_samples = sum(ns)
    return df, n_samples


def _build_fit_tfidf_transformer(df_and_n_samples, kwargs):
    """Build a fit TfidfTransformer from df, n_samples, and kwargs"""
    df, n_samples = df_and_n_samples
    model = cuml.feature_extraction.text.TfidfTransformer(**kwargs)
    model._set_idf(df, n_samples)
    model.n_features_in_ = len(df)
    return model


class TfidfTransformer(BaseEstimator, DelayedTransformMixin):
    """
    Distributed TF-IDF transformer

    Examples
    --------
    .. code-block:: python

        >>> import cupy as cp
        >>> from sklearn.datasets import fetch_20newsgroups
        >>> from sklearn.feature_extraction.text import CountVectorizer
        >>> from dask_cuda import LocalCUDACluster
        >>> from dask.distributed import Client
        >>> from cuml.dask.common import to_sparse_dask_array
        >>> from cuml.dask.naive_bayes import MultinomialNB
        >>> import dask
        >>> from cuml.dask.feature_extraction.text import TfidfTransformer

        >>> # Create a local CUDA cluster
        >>> cluster = LocalCUDACluster()
        >>> client = Client(cluster)

        >>> # Load corpus
        >>> twenty_train = fetch_20newsgroups(subset='train',
        ...                         shuffle=True, random_state=42)
        >>> cv = CountVectorizer()
        >>> xformed = cv.fit_transform(twenty_train.data).astype(cp.float32)
        >>> X = to_sparse_dask_array(xformed, client)

        >>> y = dask.array.from_array(twenty_train.target, asarray=False,
        ...                     fancy=False).astype(cp.int32)

        >>> multi_gpu_transformer = TfidfTransformer()
        >>> X_transformed = multi_gpu_transformer.fit_transform(X)
        >>> X_transformed.compute_chunk_sizes()
        dask.array<...>

        >>> model = MultinomialNB()
        >>> model.fit(X_transformed, y)
        <cuml.dask.naive_bayes.naive_bayes.MultinomialNB object at 0x...>
        >>> result = model.score(X_transformed, y)
        >>> print(result) # doctest: +SKIP
        array(0.93264981)
        >>> client.close()
        >>> cluster.close()

    """

    def __init__(self, *, client=None, verbose=False, **kwargs):
        """
        Create new distributed TF-IDF transformer instance

        Parameters
        ----------
        client : dask.distributed.Client, optional
            Dask client to use
        """
        super().__init__(client=client, verbose=verbose, **kwargs)

        self.datatype = "cupy"

        # Make any potential model args available and catch any potential
        # ValueErrors before distributed training begins.
        model = cuml.feature_extraction.text.TfidfTransformer(**kwargs)
        model._check_params()
        self._set_internal_model(model)

    def fit(self, X, y=None):
        """
        Fit distributed TFIDF Transformer

        Parameters
        ----------

        X : dask.Array with blocks containing dense or sparse cupy arrays

        Returns
        -------

        cuml.dask.feature_extraction.text.TfidfTransformer instance
        """
        # Only Dask.Array supported for now
        if not isinstance(X, dask.array.core.Array):
            raise ValueError("Only dask.Array is supported for X")

        if len(X.chunks[1]) != 1:
            raise ValueError(
                "X must be chunked by row only. "
                "Multi-dimensional chunking is not supported"
            )

        # No need to compute if we don't need idf
        if not self.internal_model.use_idf:
            self.internal_model.n_features_in_ = X.shape[1]
            return self

        handler = DistributedDataHandler.create(X, self.client)
        chunks = [chunk for _, chunk in handler.gpu_futures]
        df_and_n_samples = reduce(
            self.client.map(_get_df_and_n_samples, chunks, pure=False),
            _merge_df_and_n_samples,
            client=self.client,
        )
        model = self.client.submit(
            _build_fit_tfidf_transformer,
            df_and_n_samples,
            self.kwargs,
            pure=False,
        )
        wait_and_raise_from_futures([model])
        self._set_internal_model(model)

        return self

    def fit_transform(self, X, y=None):
        """
        Fit distributed TFIDFTransformer and then transform
        the given set of data samples.

        Parameters
        ----------

        X : dask.Array with blocks containing dense or sparse cupy arrays

        Returns
        -------

        dask.Array with blocks containing transformed sparse cupy arrays

        """
        return self.fit(X).transform(X)

    def transform(self, X, y=None):
        """
        Use distributed TFIDFTransformer to transform the
        given set of data samples.

        Parameters
        ----------

        X : dask.Array with blocks containing dense or sparse cupy arrays

        Returns
        -------

        dask.Array with blocks containing transformed sparse cupy arrays

        """
        if not isinstance(X, dask.array.core.Array):
            raise ValueError("Only dask.Array is supported for X")

        return self._transform(
            X, n_dims=2, delayed=True, output_collection_type="cupy"
        )
