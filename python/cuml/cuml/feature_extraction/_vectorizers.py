# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import numbers
from functools import partial

import cudf
import cupy as cp
import cupyx.scipy.sparse as cp_sp
import numpy as np
import pandas as pd
from cudf import Series
from sklearn.exceptions import NotFittedError

import cuml.internals.logger as logger
from cuml.common.sparse import csr_row_normalize_l1, csr_row_normalize_l2
from cuml.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from cuml.internals.mixins import DeprecatedGetFeatureNamesMixin

CUPY_SPARSE_DTYPES = [cp.float32, cp.float64, cp.complex64, cp.complex128]


def min_signed_type(n):
    for int_dtype in (np.int8, np.int16, np.int32, np.int64):
        dtype = np.dtype(int_dtype)
        if (dtype.itemsize * 8) >= 8:
            if np.iinfo(int_dtype).min <= n <= np.iinfo(int_dtype).max:
                return dtype
    # resort to using `int64` and let numpy raise appropriate exception:
    return np.int64(n).dtype


def _insert_zeros(ary, zero_indices):
    """
    Create a new array of len(ary + zero_indices) where zero_indices
    indicates indexes of 0s in the new array. Ary is used to fill the rest.

    Examples
    --------
    _insert_zeros([1, 2, 3], [1, 3]) => [1, 0, 2, 0, 3]
    """
    if len(zero_indices) == 0:
        return ary.values

    new_ary = cp.zeros((len(ary) + len(zero_indices)), dtype=cp.int32)

    # getting mask of non-zeros
    data_mask = ~cp.in1d(
        cp.arange(0, len(new_ary), dtype=cp.int32), zero_indices
    )

    new_ary[data_mask] = ary
    return new_ary


def _create_csr_matrix_from_count_df(
    count_df, empty_doc_ids, n_doc, n_features, dtype=np.float32
):
    """
    Create a sparse matrix from the count of tokens by document

    Parameters
    ----------
    count_df = cudf.DataFrame({'count':..., 'doc_id':.., 'token':.. })
                sorted by doc_id and token
    empty_doc_ids = cupy array containing doc_ids with no tokens
    n_doc: Total number of documents
    n_features: Number of features
    dtype: Output dtype
    """
    data = count_df["count"].values
    indices = count_df["token"].values

    doc_token_counts = count_df["doc_id"].value_counts().reset_index()
    del count_df

    doc_token_counts = doc_token_counts.rename(
        {"count": "token_counts"}, axis=1
    ).sort_values(by="doc_id")

    token_counts = _insert_zeros(
        doc_token_counts["token_counts"], empty_doc_ids
    )
    indptr = token_counts.cumsum()
    indptr = cp.pad(indptr, (1, 0), "constant")

    return cp_sp.csr_matrix(
        arg1=(data, indices, indptr), dtype=dtype, shape=(n_doc, n_features)
    )


def _preprocess(
    doc,
    lower=False,
    remove_non_alphanumeric=False,
    delimiter=" ",
    keep_underscore_char=True,
    remove_single_token_len=True,
):
    """
    Chain together an optional series of text preprocessing steps to
    apply to a document.

    Parameters
    ----------
    doc: cudf.Series[str] or pd.Series[str]
        The string to preprocess
    lower: bool
        Whether to use str.lower to lowercase all of the text
    remove_non_alphanumeric: bool
        Whether or not to remove non-alphanumeric characters.
    keep_underscore_char: bool
        Whether or not to keep the underscore character

    Returns
    -------
    doc: cudf.Series[str]
        preprocessed string
    """
    if isinstance(doc, pd.Series):
        doc = Series(doc)
    if lower:
        doc = doc.str.lower()
    if remove_non_alphanumeric:
        if keep_underscore_char:
            # why: sklearn by default keeps `_` char along with  alphanumerics
            # currently we dont have a easy way of removing
            # all chars but `_`
            # in cudf.Series[str] below works around it
            temp_string = "cumlSt"
            doc = doc.str.replace("_", temp_string, regex=False)
            doc = doc.str.filter_alphanum(" ", keep=True)
            doc = doc.str.replace(temp_string, "_", regex=False)
        else:
            doc = doc.str.filter_alphanum(" ", keep=True)

        # sklearn by default removes tokens of
        # length 1, if its remove alphanumerics
        if remove_single_token_len:
            doc = doc.str.filter_tokens(2)

    return doc


class _VectorizerMixin:
    """
    Provides common code for text vectorizers (tokenization logic).
    """

    def _remove_stop_words(self, doc):
        """
        Remove stop words only if needed.
        """
        if self.analyzer == "word" and self.stop_words is not None:
            stop_words = Series(self._get_stop_words())
            doc = doc.str.replace_tokens(
                stop_words,
                replacements=self.delimiter,
                delimiter=self.delimiter,
            )
        return doc

    def build_preprocessor(self):
        """
        Return a function to preprocess the text before tokenization.

        If analyzer == 'word' and stop_words is not None, stop words are
        removed from the input documents after preprocessing.

        Returns
        -------
        preprocessor: callable
              A function to preprocess the text before tokenization.
        """
        if self.preprocessor is not None:
            preprocess = self.preprocessor
        else:
            remove_non_alpha = self.analyzer == "word"
            preprocess = partial(
                _preprocess,
                lower=self.lowercase,
                remove_non_alphanumeric=remove_non_alpha,
                delimiter=self.delimiter,
            )
        return lambda doc: self._remove_stop_words(preprocess(doc))

    def _get_stop_words(self):
        """
        Build or fetch the effective stop words list.

        Returns
        -------
        stop_words: list or None
                A list of stop words.
        """
        if self.stop_words == "english":
            return list(ENGLISH_STOP_WORDS)
        elif isinstance(self.stop_words, str):
            raise ValueError("not a built-in stop list: %s" % self.stop_words)
        elif self.stop_words is None:
            return None
        else:  # assume it's a collection
            return list(self.stop_words)

    def get_char_ngrams(self, ngram_size, str_series, doc_id_sr):
        """
        Handles ngram generation for characters analyzers.

        When analyzer is 'char_wb', we generate ngrams within word boundaries,
        meaning we need to first tokenize and pad each token with a delimiter.
        """
        if self.analyzer == "char_wb" and ngram_size != 1:
            token_count = str_series.str.token_count(delimiter=self.delimiter)
            tokens = str_series.str.tokenize(self.delimiter)
            del str_series

            padding = Series(self.delimiter).repeat(len(tokens))
            tokens = tokens.str.cat(padding)
            padding = padding.reset_index(drop=True)
            tokens = padding.str.cat(tokens)
            tokens = tokens.reset_index(drop=True)

            ngram_sr = tokens.str.character_ngrams(n=ngram_size)

            doc_id_df = cudf.DataFrame(
                {
                    "doc_id": doc_id_sr.repeat(token_count).reset_index(
                        drop=True
                    ),
                    # formula to count ngrams given number of letters per token:
                    "ngram_count": tokens.str.len() - (ngram_size - 1),
                }
            )
            del tokens
            ngram_count = doc_id_df.groupby("doc_id", sort=True).sum()[
                "ngram_count"
            ]
            return ngram_sr, ngram_count, token_count

        if ngram_size == 1:
            token_count = str_series.str.len()
            ngram_sr = str_series.str.character_tokenize()
            del str_series
        elif self.analyzer == "char":
            token_count = str_series.str.len()
            ngram_sr = str_series.str.character_ngrams(n=ngram_size)
            del str_series

        ngram_count = token_count - (ngram_size - 1)

        return ngram_sr, ngram_count, token_count

    def get_ngrams(self, str_series, ngram_size, doc_id_sr):
        """
        This returns the ngrams for the string series

        Parameters
        ----------
        str_series : (cudf.Series)
            String series to tokenize
        ngram_size : int
            Gram level to get (1 for unigram, 2 for bigram etc)
        doc_id_sr : cudf.Series
            Int series containing documents ids
        """

        if self.analyzer == "word":
            token_count_sr = str_series.str.token_count(self.delimiter)
            ngram_sr = str_series.str.ngrams_tokenize(
                n=ngram_size, separator=" ", delimiter=self.delimiter
            )
            # formula to count ngrams given number of tokens x per doc: x-(n-1)
            ngram_count = token_count_sr - (ngram_size - 1)
        else:
            ngram_sr, ngram_count, token_count_sr = self.get_char_ngrams(
                ngram_size, str_series, doc_id_sr
            )

        not_empty_docs = token_count_sr > 0
        doc_id_sr = doc_id_sr[not_empty_docs]
        ngram_count = ngram_count[not_empty_docs]

        doc_id_sr = doc_id_sr.repeat(ngram_count).reset_index(drop=True)
        tokenized_df = cudf.DataFrame()
        tokenized_df["doc_id"] = doc_id_sr

        ngram_sr = ngram_sr.reset_index(drop=True)
        tokenized_df["token"] = ngram_sr
        return tokenized_df

    def _create_tokenized_df(self, docs):
        """
        Creates a tokenized DataFrame from a string Series.
        Each row describes the token string and the corresponding document id.
        """
        min_n, max_n = self.ngram_range

        doc_id = cp.arange(start=0, stop=len(docs), dtype=cp.int32)
        doc_id = Series(doc_id)

        tokenized_df_ls = [
            self.get_ngrams(docs, n, doc_id) for n in range(min_n, max_n + 1)
        ]
        del docs
        tokenized_df = cudf.concat(tokenized_df_ls)
        tokenized_df = tokenized_df.reset_index(drop=True)

        return tokenized_df

    def _compute_empty_doc_ids(self, count_df, n_doc):
        """
        Compute empty docs ids using the remaining docs, given the total number
        of documents.
        """
        remaining_docs = count_df["doc_id"].unique()
        dtype = min_signed_type(n_doc)
        doc_ids = cudf.DataFrame(
            data={"all_ids": cp.arange(0, n_doc, dtype=dtype)}, dtype=dtype
        )

        empty_docs = doc_ids - doc_ids.iloc[remaining_docs]
        empty_ids = empty_docs[empty_docs["all_ids"].isnull()].index.values
        return empty_ids

    def _validate_params(self):
        """
        Check validity of ngram_range parameter
        """
        min_n, max_m = self.ngram_range
        msg = ""
        if min_n < 1:
            msg += "lower boundary must be >= 1. "
        if min_n > max_m:
            msg += "lower boundary larger than the upper boundary. "
        if msg != "":
            msg = f"Invalid value for ngram_range={self.ngram_range} {msg}"
            raise ValueError(msg)

        if hasattr(self, "n_features"):
            if not isinstance(self.n_features, numbers.Integral):
                raise TypeError(
                    f"n_features must be integral, got {self.n_features}\
                    ({type(self.n_features)})."
                )

    def _warn_for_unused_params(self):
        if self.analyzer != "word" and self.stop_words is not None:
            logger.warn(
                "The parameter 'stop_words' will not be used"
                " since 'analyzer' != 'word'"
            )

    def _check_sklearn_params(self, analyzer, sklearn_params):
        if callable(analyzer):
            raise ValueError(
                "cuML does not support callable analyzer,"
                " please refer to the cuML documentation for"
                " more information."
            )

        for key, vals in sklearn_params.items():
            if vals is not None:
                raise TypeError(
                    "The Scikit-learn variable",
                    key,
                    " is not supported in cuML,"
                    " please read the cuML documentation for"
                    " more information.",
                )


def _document_frequency(X):
    """
    Count the number of non-zero values for each feature in X.
    """
    doc_freq = X[["token", "doc_id"]].groupby(["token"], sort=True).count()
    return doc_freq["doc_id"].values


def _term_frequency(X):
    """
    Count the number of occurrences of each term in X.
    """
    term_freq = X[["token", "count"]].groupby(["token"], sort=True).sum()
    return term_freq["count"].values


class CountVectorizer(DeprecatedGetFeatureNamesMixin, _VectorizerMixin):
    """
    Convert a collection of text documents to a matrix of token counts

    If you do not provide an a-priori dictionary then the number of features
    will be equal to the vocabulary size found by analyzing the data.

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
    analyzer : string, {'word', 'char', 'char_wb'}
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

    Attributes
    ----------
    vocabulary_ : cudf.Series[str]
        Array mapping from feature integer indices to feature name.
    stop_words_ : cudf.Series[str]
        Terms that were ignored because they either:
          - occurred in too many documents (`max_df`)
          - occurred in too few documents (`min_df`)
          - were cut off by feature selection (`max_features`).

        This is only available if no vocabulary was given.

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
    ):
        self.preprocessor = preprocessor
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.stop_words = stop_words
        self.max_df = max_df
        self.min_df = min_df
        if max_df < 0 or min_df < 0:
            raise ValueError("negative value for max_df or min_df")
        self.max_features = max_features
        if max_features is not None:
            if not isinstance(max_features, int) or max_features <= 0:
                raise ValueError(
                    "max_features=%r, neither a positive integer nor None"
                    % max_features
                )
        self.ngram_range = ngram_range
        self.vocabulary = vocabulary
        self.binary = binary
        self.dtype = dtype
        self.delimiter = delimiter
        if dtype not in CUPY_SPARSE_DTYPES:
            msg = f"Expected dtype in {CUPY_SPARSE_DTYPES}, got {dtype}"
            raise ValueError(msg)

        sklearn_params = {
            "input": input,
            "encoding": encoding,
            "decode_error": decode_error,
            "strip_accents": strip_accents,
            "tokenizer": tokenizer,
            "token_pattern": token_pattern,
        }
        self._check_sklearn_params(analyzer, sklearn_params)

    def _count_vocab(self, tokenized_df):
        """
        Count occurrences of tokens in each document.
        """
        # Transform string tokens into token indexes from 0 to len(vocab)
        # The indexes are based on lexicographical ordering.
        tokenized_df["token"] = tokenized_df["token"].astype("category")
        tokenized_df["token"] = (
            tokenized_df["token"]
            .cat.set_categories(self.vocabulary_)
            .cat.codes
        )

        # Count of each token in each document
        count_df = (
            tokenized_df[["doc_id", "token"]]
            .groupby(["doc_id", "token"], sort=True)
            .size()
            .reset_index()
            .rename({0: "count"}, axis=1)
        )

        return count_df

    def _filter_and_renumber(self, df, keep_values, column):
        """
        Filter dataframe to keep only values from column matching
        keep_values.
        """
        df[column] = (
            df[column]
            .astype("category")
            .cat.set_categories(keep_values)
            .cat.codes
        )
        df = df.dropna(subset=column)
        return df

    def _limit_features(self, count_df, vocab, high, low, limit):
        """
        Remove too rare or too common features.

        Prune features that are non zero in more samples than high or less
        documents than low, modifying the vocabulary, and restricting it to
        at most the limit most frequent.

        Sets `self.vocabulary_` and `self.stop_words_` with the new values.
        """
        if high is None and low is None and limit is None:
            self.stop_words_ = None
            return count_df

        document_frequency = _document_frequency(count_df)

        mask = cp.ones(len(document_frequency), dtype=bool)
        if high is not None:
            mask &= document_frequency <= high
        if low is not None:
            mask &= document_frequency >= low
        if limit is not None and mask.sum() > limit:
            term_frequency = _term_frequency(count_df)
            mask_inds = (-term_frequency[mask]).argsort()[:limit]
            new_mask = cp.zeros(len(document_frequency), dtype=bool)
            new_mask[cp.where(mask)[0][mask_inds]] = True
            mask = new_mask

        keep_idx = cp.where(mask)[0].astype(cp.int32)
        keep_num = keep_idx.shape[0]

        if keep_num == 0:
            raise ValueError(
                "After pruning, no terms remain. Try a lower"
                " min_df or a higher max_df."
            )

        if len(vocab) - keep_num != 0:
            count_df = self._filter_and_renumber(count_df, keep_idx, "token")

        self.stop_words_ = vocab[~mask].reset_index(drop=True)
        self.vocabulary_ = vocab[mask].reset_index(drop=True)

        return count_df

    def _preprocess(self, raw_documents):
        preprocess = self.build_preprocessor()
        return preprocess(raw_documents)

    def fit(self, raw_documents, y=None):
        """
        Build a vocabulary of all tokens in the raw documents.

        Parameters
        ----------

        raw_documents : cudf.Series or pd.Series
            A Series of string documents
        y : None
            Ignored.

        Returns
        -------
        self

        """
        self.fit_transform(raw_documents)
        return self

    def fit_transform(self, raw_documents, y=None):
        """
        Build the vocabulary and return document-term matrix.

        Equivalent to ``self.fit(X).transform(X)`` but preprocess `X` only
        once.

        Parameters
        ----------
        raw_documents : cudf.Series or pd.Series
           A Series of string documents
        y : None
            Ignored.

        Returns
        -------
        X : cupy csr array of shape (n_samples, n_features)
            Document-term matrix.
        """
        self._warn_for_unused_params()
        self._validate_params()

        self._fixed_vocabulary = self.vocabulary is not None

        docs = self._preprocess(raw_documents)
        n_doc = len(docs)

        tokenized_df = self._create_tokenized_df(docs)

        if self._fixed_vocabulary:
            self.vocabulary_ = self.vocabulary
        else:
            self.vocabulary_ = (
                tokenized_df["token"].drop_duplicates().sort_values()
            )

        count_df = self._count_vocab(tokenized_df)

        if not self._fixed_vocabulary:
            max_doc_count = (
                self.max_df
                if isinstance(self.max_df, numbers.Integral)
                else self.max_df * n_doc
            )
            min_doc_count = (
                self.min_df
                if isinstance(self.min_df, numbers.Integral)
                else self.min_df * n_doc
            )
            if max_doc_count < min_doc_count:
                raise ValueError(
                    "max_df corresponds to < documents than min_df"
                )
            count_df = self._limit_features(
                count_df,
                self.vocabulary_,
                max_doc_count,
                min_doc_count,
                self.max_features,
            )

        empty_doc_ids = self._compute_empty_doc_ids(count_df, n_doc)

        X = _create_csr_matrix_from_count_df(
            count_df,
            empty_doc_ids,
            n_doc,
            len(self.vocabulary_),
            dtype=self.dtype,
        )
        if self.binary:
            X.data.fill(1)
        return X

    def transform(self, raw_documents):
        """
        Transform documents to document-term matrix.

        Extract token counts out of raw text documents using the vocabulary
        fitted with fit or the one provided to the constructor.

        Parameters
        ----------
        raw_documents : cudf.Series or pd.Series
           A Series of string documents

        Returns
        -------
        X : cupy csr array of shape (n_samples, n_features)
            Document-term matrix.
        """
        if not hasattr(self, "vocabulary_"):
            if self.vocabulary is not None:
                self.vocabulary_ = self.vocabulary
            else:
                raise NotFittedError()

        docs = self._preprocess(raw_documents)
        n_doc = len(docs)
        tokenized_df = self._create_tokenized_df(docs)
        count_df = self._count_vocab(tokenized_df)
        empty_doc_ids = self._compute_empty_doc_ids(count_df, n_doc)
        X = _create_csr_matrix_from_count_df(
            count_df,
            empty_doc_ids,
            n_doc,
            len(self.vocabulary_),
            dtype=self.dtype,
        )
        if self.binary:
            X.data.fill(1)
        return X

    def inverse_transform(self, X):
        """
        Return terms per document with nonzero entries in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Document-term matrix.

        Returns
        -------
        X_inv : list of cudf.Series of shape (n_samples,)
            List of Series of terms.
        """
        vocab = Series(self.vocabulary_)
        return [vocab[X[i, :].indices] for i in range(X.shape[0])]

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Not used, present here for API consistency by convention.

        Returns
        -------
        feature_names_out : numpy.ndarray of str objects.
            Transformed feature names.
        """
        # TODO: use `check_is_fitted` once this class subclasses from `Base`
        return self.vocabulary_.to_numpy(dtype=object)


class HashingVectorizer(_VectorizerMixin):
    """
    Convert a collection of text documents to a matrix of token occurrences

    It turns a collection of text documents into a cupyx.scipy.sparse matrix
    holding token occurrence counts (or binary occurrence information),
    possibly normalized as token frequencies if norm='l1' or projected on the
    euclidean unit sphere if norm='l2'.

    This text vectorizer implementation uses the hashing trick to find the
    token string name to feature integer index mapping.

    This strategy has several advantages:

     - it is very low memory scalable to large datasets as there is no need to
       store a vocabulary dictionary in memory which is even more important
       as GPU's that are often memory constrained
     - it is fast to pickle and un-pickle as it holds no state besides the
       constructor parameters
     - it can be used in a streaming (partial fit) or parallel pipeline as
       there is no state computed during fit.

    There are also a couple of cons (vs using a CountVectorizer with an
    in-memory vocabulary):

     - there is no way to compute the inverse transform (from feature indices
       to string feature names) which can be a problem when trying to
       introspect which features are most important to a model.
     - there can be collisions: distinct tokens can be mapped to the same
       feature index. However in practice this is rarely an issue if n_features
       is large enough (e.g. 2 ** 18 for text classification problems).
     - no IDF weighting as this would render the transformer stateful.

    The hash function employed is the signed 32-bit version of Murmurhash3.

    Parameters
    ----------
    lowercase : bool, default=True
        Convert all characters to lowercase before tokenizing.
    preprocessor : callable or None (default)
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.
    stop_words : string {'english'}, list, default=None
        If 'english', a built-in stop word list for English is used.
        There are several known issues with 'english' and you should
        consider an alternative.
        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.
        Only applies if ``analyzer == 'word'``.
    ngram_range : tuple (min_n, max_n), default=(1, 1)
        The lower and upper boundary of the range of n-values for different
        word n-grams or char n-grams to be extracted. All values of n such
        such that min_n <= n <= max_n will be used. For example an
        ``ngram_range`` of ``(1, 1)`` means only unigrams, ``(1, 2)`` means
        unigrams and bigrams, and ``(2, 2)`` means only bigrams.
    analyzer : string, {'word', 'char', 'char_wb'}
        Whether the feature should be made of word n-gram or character
        n-grams.
        Option 'char_wb' creates character n-grams only from text inside
        word boundaries; n-grams at the edges of words are padded with space.
    n_features : int, default=(2 ** 20)
        The number of features (columns) in the output matrices. Small numbers
        of features are likely to cause hash collisions, but large numbers
        will cause larger coefficient dimensions in linear learners.
    binary : bool, default=False.
        If True, all non zero counts are set to 1. This is useful for discrete
        probabilistic models that model binary events rather than integer
        counts.
    norm : {'l1', 'l2'}, default='l2'
        Norm used to normalize term vectors. None for no normalization.
    alternate_sign : bool, default=True
        When True, an alternating sign is added to the features as to
        approximately conserve the inner product in the hashed space even for
        small n_features. This approach is similar to sparse random projection.
    dtype : type, optional
        Type of the matrix returned by fit_transform() or transform().
    delimiter : str, whitespace by default
        String used as a replacement for stop words if `stop_words` is not
        None. Typically the delimiting character between words is a good
        choice.

    Examples
    --------

    .. code-block:: python

        >>> from cuml.feature_extraction.text import HashingVectorizer
        >>> import pandas as pd
        >>> corpus = [
        ...     'This is the first document.',
        ...     'This document is the second document.',
        ...     'And this is the third one.',
        ...     'Is this the first document?',
        ... ]
        >>> vectorizer = HashingVectorizer(n_features=2**4)
        >>> X = vectorizer.fit_transform(pd.Series(corpus))
        >>> print(X.shape)
        (4, 16)

    See Also
    --------
    CountVectorizer, TfidfVectorizer
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
        n_features=(2**20),
        binary=False,
        norm="l2",
        alternate_sign=True,
        dtype=cp.float32,
        delimiter=" ",
    ):
        self.preprocessor = preprocessor
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.stop_words = stop_words
        self.n_features = n_features
        self.ngram_range = ngram_range
        self.binary = binary
        self.norm = norm
        self.alternate_sign = alternate_sign
        self.dtype = dtype
        self.delimiter = delimiter

        if dtype not in CUPY_SPARSE_DTYPES:
            msg = f"Expected dtype in {CUPY_SPARSE_DTYPES}, got {dtype}"
            raise ValueError(msg)

        if self.norm not in ("l1", "l2", None):
            raise ValueError(f"{self.norm} is not a supported norm")

        sklearn_params = {
            "input": input,
            "encoding": encoding,
            "decode_error": decode_error,
            "strip_accents": strip_accents,
            "tokenizer": tokenizer,
            "token_pattern": token_pattern,
        }
        self._check_sklearn_params(analyzer, sklearn_params)

    def partial_fit(self, X, y=None):
        """
        Does nothing: This transformer is stateless
        This method is just there to mark the fact that this transformer
        can work in a streaming setup.

        Parameters
        ----------
        X : cudf.Series(A Series of string documents).
        """
        return self

    def fit(self, X, y=None):
        """
        This method only checks the input type and the model parameter.
        It does not do anything meaningful as this transformer is stateless

        Parameters
        ----------
        X : cudf.Series or pd.Series
             A Series of string documents
        """
        self._warn_for_unused_params()
        self._validate_params()
        return self

    def _preprocess(self, raw_documents):
        preprocess = self.build_preprocessor()
        return preprocess(raw_documents)

    def _count_hash(self, tokenized_df):
        """
        Count occurrences of tokens in each document.
        """
        # Transform string tokens into token indexes from 0 to n_features
        tokenized_df["token"] = tokenized_df["token"].hash_values()
        if self.alternate_sign:
            # below logic is equivalent to: value *= ((h >= 0) * 2) - 1
            tokenized_df["value"] = (tokenized_df["token"] >= 0).astype(
                cp.int8
            ) * 2 - 1
            tokenized_df["token"] = (
                tokenized_df["token"].abs() % self.n_features
            )
            count_ser = tokenized_df.groupby(
                ["doc_id", "token"], sort=True
            ).value.sum()
            count_ser.name = "count"
        else:
            tokenized_df["token"] = (
                tokenized_df["token"].abs() % self.n_features
            )
            count_ser = tokenized_df.groupby(
                ["doc_id", "token"], sort=True
            ).size()
            count_ser.name = "count"

        count_df = count_ser.reset_index(drop=False)
        del count_ser, tokenized_df
        return count_df

    def fit_transform(self, X, y=None):
        """
        Transform a sequence of documents to a document-term matrix.

        Parameters
        ----------
        X : iterable over raw text documents, length = n_samples
            Samples. Each sample must be a text document (either bytes or
            unicode strings, file name or file object depending on the
            constructor argument) which will be tokenized and hashed.
        y : any
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.

        Returns
        -------
        X : sparse CuPy CSR matrix of shape (n_samples, n_features)
            Document-term matrix.
        """
        return self.fit(X, y).transform(X)

    def transform(self, raw_documents):
        """
        Transform documents to document-term matrix.

        Extract token counts out of raw text documents using the vocabulary
        fitted with fit or the one provided to the constructor.

        Parameters
        ----------
        raw_documents : cudf.Series or pd.Series
            A Series of string documents

        Returns
        -------
        X : sparse CuPy CSR matrix of shape (n_samples, n_features)
            Document-term matrix.
        """
        docs = self._preprocess(raw_documents)
        del raw_documents
        n_doc = len(docs)
        tokenized_df = self._create_tokenized_df(docs)
        del docs
        count_df = self._count_hash(tokenized_df)
        del tokenized_df
        empty_doc_ids = self._compute_empty_doc_ids(count_df, n_doc)
        X = _create_csr_matrix_from_count_df(
            count_df, empty_doc_ids, n_doc, self.n_features, dtype=self.dtype
        )

        if self.binary:
            X.data.fill(1)
        if self.norm:
            if self.norm == "l1":
                csr_row_normalize_l1(X, inplace=True)
            elif self.norm == "l2":
                csr_row_normalize_l2(X, inplace=True)

        return X
