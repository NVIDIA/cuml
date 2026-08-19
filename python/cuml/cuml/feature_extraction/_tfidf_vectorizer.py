# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Portions of this code are derived from the scikit-learn feature_extraction
# package, which has the following license:
#
#    -*- coding: utf-8 -*-
#    Authors: Olivier Grisel <olivier.grisel@ensta.org>
#             Mathieu Blondel <mathieu@mblondel.org>
#             Lars Buitinck
#             Robert Layton <robertlayton@gmail.com>
#             Jochen Wersdörfer <jochen@wersdoerfer.de>
#             Roman Sinayev <roman.sinayev@gmail.com>
#
#    License: BSD 3 clause
#

import cupy as cp

from cuml.feature_extraction._tfidf import TfidfTransformer
from cuml.feature_extraction._vectorizers import CountVectorizer


class TfidfVectorizer(CountVectorizer):
    """Convert a collection of raw documents to a matrix of TF-IDF features.

    Equivalent to :class:`CountVectorizer` followed by
    :class:`TfidfTransformer`.

    Parameters
    ----------
    lowercase : boolean, True by default
        Convert all characters to lowercase before tokenizing.
    preprocessor : callable or None (default)
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.
    stop_words : string {'english'}, list, or None (default)
        If 'english', a built-in stop word list for English is used.
        If a list, that list is assumed to contain stop words, all of which
        will be removed from the input documents.
        If None, no stop words will be used. max_df can be set to a value
        to automatically detect and filter stop words based on intra corpus
        document frequency of terms.
    ngram_range : tuple (min_n, max_n), default=(1, 1)
        The lower and upper boundary of the range of n-values for different
        word n-grams or char n-grams to be extracted. All values of n such
        such that min_n <= n <= max_n will be used. For example an
        ``ngram_range`` of ``(1, 1)`` means only unigrams, ``(1, 2)`` means
        unigrams and bigrams, and ``(2, 2)`` means only bigrams.
    analyzer : string, {'word', 'char', 'char_wb'}, default='word'
        Whether the feature should be made of word n-gram or character
        n-grams.
        Option 'char_wb' creates character n-grams only from text inside
        word boundaries; n-grams at the edges of words are padded with space.
    max_df : float in range [0.0, 1.0] or int, default=1.0
        When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.
    min_df : float in range [0.0, 1.0] or int, default=1
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.
    max_features : int or None, default=None
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus.
        This parameter is ignored if vocabulary is not None.
    vocabulary : cudf.Series, optional
        If not given, a vocabulary is determined from the input documents.
    binary : boolean, default=False
        If True, all non zero counts are set to 1. This is useful for discrete
        probabilistic models that model binary events rather than integer
        counts.
    dtype : type, optional
        Type of the matrix returned by fit_transform() or transform().
    delimiter : str, whitespace by default
        String used as a replacement for stop words if stop_words is not None.
        Typically the delimiting character between words is a good choice.
    norm : {'l1', 'l2'}, default='l2'
        Each output row will have unit norm, either:
         * 'l2': Sum of squares of vector elements is 1. The cosine similarity
           between two vectors is their dot product when l2 norm has been
           applied.
         * 'l1': Sum of absolute values of vector elements is 1.

    use_idf : bool, default=True
        Enable inverse-document-frequency reweighting.
    smooth_idf : bool, default=True
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.
    sublinear_tf : bool, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

    Attributes
    ----------
    idf_ : array of shape (n_features)
        The inverse document frequency (IDF) vector; only defined
        if  `use_idf` is True.
    vocabulary_ : cudf.Series[str]
        Array mapping from feature integer indices to feature name.
    stop_words_ : cudf.Series[str]
        Terms that were ignored because they either:
          - occurred in too many documents (`max_df`)
          - occurred in too few documents (`min_df`)
          - were cut off by feature selection (`max_features`).

        This is only available if no vocabulary was given.

    Notes
    -----
    The ``stop_words_`` attribute can get large and increase the model size
    when pickling. This attribute is provided only for introspection and can
    be safely removed using delattr or set to None before pickling.

    This class is largely based on scikit-learn 0.23.1's TfIdfVectorizer code,
    which is provided under the BSD-3 license.
    """

    def __init__(
        self,
        input=None,
        encoding=None,
        decode_error=None,
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        stop_words=None,
        token_pattern=None,
        ngram_range=(1, 1),
        analyzer="word",
        max_df=1.0,
        min_df=1,
        max_features=None,
        vocabulary=None,
        binary=False,
        dtype=cp.float32,
        delimiter=" ",
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    ):
        super().__init__(
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            analyzer=analyzer,
            stop_words=stop_words,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            binary=binary,
            dtype=dtype,
            delimiter=delimiter,
        )

        self._tfidf = TfidfTransformer(
            norm=norm,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf,
        )

    # Broadcast the TF-IDF parameters to the underlying transformer instance
    # for easy grid search and repr

    @property
    def norm(self):
        return self._tfidf.norm

    @norm.setter
    def norm(self, value):
        self._tfidf.norm = value

    @property
    def use_idf(self):
        return self._tfidf.use_idf

    @use_idf.setter
    def use_idf(self, value):
        self._tfidf.use_idf = value

    @property
    def smooth_idf(self):
        return self._tfidf.smooth_idf

    @smooth_idf.setter
    def smooth_idf(self, value):
        self._tfidf.smooth_idf = value

    @property
    def sublinear_tf(self):
        return self._tfidf.sublinear_tf

    @sublinear_tf.setter
    def sublinear_tf(self, value):
        self._tfidf.sublinear_tf = value

    @property
    def idf_(self):
        return self._tfidf.idf_

    @idf_.setter
    def idf_(self, value):
        if hasattr(self, "vocabulary_"):
            if len(self.vocabulary_) != len(value):
                raise ValueError(
                    "idf length = %d must be equal "
                    "to vocabulary size = %d"
                    % (len(value), len(self.vocabulary))
                )
        self._tfidf.idf_ = value

    def fit(self, raw_documents):
        """Learn vocabulary and idf from training set.

        Parameters
        ----------
        raw_documents : cudf.Series or pd.Series
           A Series of string documents

        Returns
        -------
        self : object
            Fitted vectorizer.
        """
        X = super().fit_transform(raw_documents)
        self._tfidf.fit(X)
        return self

    def fit_transform(self, raw_documents, y=None):
        """Learn vocabulary and idf, return document-term matrix.
        This is equivalent to fit followed by transform, but more efficiently
        implemented.

        Parameters
        ----------
        raw_documents : cudf.Series or pd.Series
           A Series of string documents
        y : None
            Ignored.

        Returns
        -------
        X : cupy csr array of shape (n_samples, n_features)
            Tf-idf-weighted document-term matrix.
        """
        X = super().fit_transform(raw_documents)
        self._tfidf.fit(X)
        # X is already a transformed view of raw_documents so
        # we set copy to False
        return self._tfidf.transform(X, copy=False)

    def transform(self, raw_documents):
        """Transform documents to document-term matrix.
        Uses the vocabulary and document frequencies (df) learned by fit (or
        fit_transform).

        Parameters
        ----------
        raw_documents : cudf.Series or pd.Series
           A Series of string documents

        Returns
        -------
        X : cupy csr array of shape (n_samples, n_features)
            Tf-idf-weighted document-term matrix.
        """
        X = super().transform(raw_documents)
        return self._tfidf.transform(X, copy=False)
