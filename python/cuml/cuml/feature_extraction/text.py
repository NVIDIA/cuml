# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import warnings
from collections.abc import Mapping
from numbers import Integral

import cudf
import cupy as cp
import cupyx.scipy.sparse as cp_sp
from sklearn.base import OneToOneFeatureMixin

from cuml.common.sparse import csr_row_normalize_l1, csr_row_normalize_l2
from cuml.internals.base import Base
from cuml.internals.mixins import DeprecatedGetFeatureNamesMixin
from cuml.internals.outputs import ReflectedAttr, mlfunc
from cuml.internals.validation import (
    check_array,
    check_cudf,
    check_inputs,
    check_is_fitted,
)

__all__ = (
    "HashingVectorizer",
    "CountVectorizer",
    "TfidfTransformer",
    "TfidfVectorizer",
)


def _check_vocabulary(vocab):
    """Validate and normalizer a user-provided vocabulary.

    Parameters
    ----------
    vocab : Iterable[str], Mapping[str, int], cudf.Series
        The user provided vocabulary. See the docstring for `CountVectorizer`
        for more info.

    Returns
    -------
    vocabulary_ : cudf.Series
        The validated and normalized vocabulary.
    """
    if isinstance(vocab, set):
        vocab = sorted(vocab)

    if isinstance(vocab, Mapping):
        vocab = cudf.Series(vocab.keys(), index=vocab.values())
        if not vocab.index.is_unique:
            raise ValueError("Vocabulary contains repeated indices.")
        vocab.sort_index(inplace=True)
        missing = cudf.RangeIndex(len(vocab)).difference(vocab.index)
        if len(missing):
            raise ValueError(
                f"Vocabulary of size {len(vocab)} doesn't contain index "
                f"{missing[0]}"
            )
        vocab = vocab.reset_index(drop=True)
    else:
        vocab = cudf.Series(vocab)
        duplicates = vocab[vocab.duplicated()]
        if len(duplicates):
            raise ValueError(
                f"Duplicate term in vocabulary: {duplicates.iloc[0]}"
            )

    if not len(vocab):
        raise ValueError("empty vocabulary passed to fit")

    return vocab


def _check_oneof(estimator, name, options):
    """A helper for validating estimator parameters are within a certain set of
    options"""
    value = getattr(estimator, name)
    if value not in options:
        raise ValueError(
            f"Expected `{name}` to be one of {options!r}, got {value!r}"
        )


class _BaseVectorizer(Base):
    """A base class for all vectorizers"""

    def __init__(
        self,
        *,
        lowercase=True,
        preprocessor=None,
        stop_words=None,
        ngram_range=(1, 1),
        analyzer="word",
        delimiter=" ",
        binary=False,
        dtype=cp.float32,
        verbose=False,
        output_type=None,
    ):
        super().__init__(verbose=verbose, output_type=output_type)
        self.lowercase = lowercase
        self.preprocessor = preprocessor
        self.stop_words = stop_words
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self.delimiter = delimiter
        self.binary = binary
        self.dtype = dtype

    def _get_param_names(self):
        return [
            "lowercase",
            "preprocessor",
            "stop_words",
            "ngram_range",
            "analyzer",
            "delimiter",
            "binary",
            "dtype",
            *super()._get_param_names(),
        ]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.string = True
        tags.input_tags.one_d_array = True
        tags.input_tags.two_d_array = False
        return tags

    def _check_params(self):
        min_n, max_n = self.ngram_range
        if min_n < 1:
            raise ValueError(
                f"Invalid value for ngram_range={self.ngram_range} "
                "lower boundary must be >= 1."
            )
        elif min_n > max_n:
            raise ValueError(
                f"Invalid value for ngram_range={self.ngram_range} "
                "lower boundary larger than the upper boundary."
            )
        _check_oneof(self, "analyzer", ["word", "char", "char_wb"])

        if self.analyzer != "word" and self.stop_words is not None:
            warnings.warn(
                "The parameter 'stop_words' will not be used"
                " since 'analyzer' != 'word'"
            )

    def _get_stop_words(self):
        """Validate and normalize the specified `stop_words`"""
        if self.stop_words == "english":
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

            return cudf.Series(ENGLISH_STOP_WORDS)
        elif isinstance(self.stop_words, str):
            raise ValueError("not a built-in stop list: %s" % self.stop_words)
        return cudf.Series(self.stop_words)

    def _preprocess(self, X):
        """Preprocess `X`.

        The default preprocessor lowercases all inputs.

        Additionally, if `analyzer="word"`, it:
        - Removes all non-alphanumeric characters (excluding "_")
        - Normalizes whitespace to " "
        - Removes any single character tokens
        - Removes any tokens specified by `stop_words`.

        The whole step may be overridden by specifying `preprocessor`
        explicitly.
        """
        if self.preprocessor is not None:
            return self.preprocessor(X)

        if self.lowercase:
            X = X.str.lower()

        if self.analyzer == "word":
            # XXX: a filler string to take the place of _ temporarily
            # since `filter_alphanum` strips `_` but sklearn keeps `_`.
            # We can use `X` in the common case of lowercase normalization
            # since no uppercase letters will remain. Otherwise pick
            # an unlikely key of unicode characters"
            flag = "X" if self.lowercase else "cuᵐl"
            X = (
                X.str.replace("_", flag, regex=False)
                .str.filter_alphanum(self.delimiter, keep=True)
                .str.replace(flag, "_", regex=False)
            )

            # sklearn by default removes single char tokens
            X = X.str.filter_tokens(2, delimiter=self.delimiter)

            # Drop any stop words specified
            if self.stop_words is not None:
                X = X.str.replace_tokens(
                    self._get_stop_words(),
                    " ",
                    delimiter=self.delimiter,
                )

        return X

    def _to_tokens(self, X):
        """The main tokenization step.

        Parameters
        ----------
        X : cudf.Series
            A series of strings

        Returns
        -------
        tokens : cudf.DataFrame
            A dataframe with schema ``{"doc_id": int, "token": str}``, where
            `doc_id` is the integer id of the document, and `token` is a token
            present in that doc.
        """
        X = self._preprocess(X)
        parts = []

        if self.analyzer == "word":
            token_counts = X.str.token_count(delimiter=self.delimiter)
            for ngram_size in range(
                self.ngram_range[0], self.ngram_range[1] + 1
            ):
                ngrams = X.str.ngrams_tokenize(
                    n=ngram_size,
                    delimiter=self.delimiter,
                    separator=" ",
                )
                ngram_count = (token_counts - (ngram_size - 1)).clip(0)
                parts.append(
                    cudf.DataFrame(
                        {
                            "doc_id": X.index.repeat(ngram_count),
                            "token": ngrams,
                        }
                    )
                )
        else:
            if self.analyzer == "char_wb":
                words = X.str.tokenize()
                padding = cudf.Series(" ", index=words.index)
                X = padding.str.cat([words, padding])
            for ngram_size in range(
                self.ngram_range[0], self.ngram_range[1] + 1
            ):
                if ngram_size == 1:
                    ngrams = X.str.character_tokenize()
                else:
                    ngrams = X.str.character_ngrams(ngram_size)
                ngrams.index.name = "doc_id"
                ngrams.name = "token"
                parts.append(ngrams.reset_index())

        if len(parts) == 1:
            return parts[0]
        return cudf.concat(parts)

    def _to_sparse(self, X, values, n_features):
        """Create a sparse matrix from vectorizer output.

        Parameters
        ----------
        X : cudf.Series
            The original X input series
        values : cudf.DataFrame
            A dataframe with schema ``{"doc_id": int, "feature_id": int,
            "value": float}``. The input  is assumed to have already been
            sorted by ``(doc_id, feature_id)``.
        n_features : int
            The number of features (columns) in the output.

        Returns
        -------
        out : cupyx.scipy.sparse.csr_matrix
        """
        data = values["value"].values.astype(self.dtype, copy=False)
        indices = values.feature_id.values
        doc_id_counts = values.doc_id.value_counts().reindex(
            X.index, fill_value=0
        )
        indptr = cp.zeros(len(doc_id_counts) + 1, dtype="int32")
        cp.cumsum(doc_id_counts.values, out=indptr[1:])

        return cp_sp.csr_matrix(
            (data, indices, indptr),
            shape=(len(X), n_features),
        )


class HashingVectorizer(_BaseVectorizer):
    """Convert a collection of text documents to a matrix of token occurrences.

    It turns a collection of text documents into a sparse matrix holding
    token occurrence counts (or binary occurrence information), possibly
    normalized as token frequencies if norm='l1' or projected on the euclidean
    unit sphere if norm='l2'.

    This text vectorizer implementation uses the hashing trick to find the
    token string name to feature integer index mapping.

    This strategy has several advantages:

    - it is very low memory scalable to large datasets as there is no need to
      store a vocabulary dictionary in memory.

    - it is fast to pickle and un-pickle as it holds no state besides the
      constructor parameters.

    - it can be used in a streaming (partial fit) or parallel pipeline as there
      is no state computed during fit.

    There are also a couple of cons (vs using a CountVectorizer with an
    in-memory vocabulary):

    - there is no way to compute the inverse transform (from feature indices to
      string feature names) which can be a problem when trying to introspect
      which features are most important to a model.

    - there can be collisions: distinct tokens can be mapped to the same
      feature index. However in practice this is rarely an issue if n_features
      is large enough (e.g. 2 ** 18 for text classification problems).

    - no IDF weighting as this would render the transformer stateful.

    The hash function employed is the signed 32-bit version of Murmurhash3.

    Parameters
    ----------
    lowercase : bool, default=True
        Convert all characters to lowercase before tokenizing.

    preprocessor : callable, default=None
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.

    stop_words : {'english'}, list, default=None
        If 'english', a built-in stop word list for English is used. If a list,
        that list is assumed to contain stop words, all of which will be
        removed from the resulting tokens. If None, no stop words will be used.
        Only applies if ``analyzer == 'word'``.

    ngram_range : tuple (min_n, max_n), default=(1, 1)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used. For example an ``ngram_range`` of ``(1, 1)`` means only
        unigrams, ``(1, 2)`` means unigrams and bigrams, and ``(2, 2)`` means
        only bigrams.

    analyzer : {'word', 'char', 'char_wb'}, default='word'
        Whether the feature should be made of word or character n-grams.
        Option 'char_wb' creates character n-grams only from text inside
        word boundaries; n-grams at the edges of words are padded with space.

    n_features : int, default=(2 ** 20)
        The number of features (columns) in the output matrices. Small numbers
        of features are likely to cause hash collisions, but large numbers
        will cause larger coefficient dimensions in linear learners.

    binary : bool, default=False
        If True, all non zero counts are set to 1. This is useful for discrete
        probabilistic models that model binary events rather than integer
        counts.

    norm : {'l1', 'l2', None}, default='l2'
        Norm used to normalize term vectors. None for no normalization.

    alternate_sign : bool, default=True
        When True, an alternating sign is added to the features as to
        approximately conserve the inner product in the hashed space even for
        small n_features. This approach is similar to sparse random projection.

    dtype : type, default=np.float32
        Type of the matrix returned by fit_transform() or transform().

    delimiter : str, default=" "
        String used as a replacement for stop words if `stop_words` is not
        None. Typically the delimiting character between words is a good
        choice.

    Examples
    --------
    >>> from cuml.feature_extraction.text import HashingVectorizer
    >>> corpus = [
    ...     'This is the first document.',
    ...     'This document is the second document.',
    ...     'And this is the third one.',
    ...     'Is this the first document?',
    ... ]
    >>> vectorizer = HashingVectorizer(n_features=2**4)
    >>> X = vectorizer.fit_transform(corpus)
    >>> X.shape
    (4, 16)
    """

    def __init__(
        self,
        *,
        lowercase=True,
        preprocessor=None,
        stop_words=None,
        ngram_range=(1, 1),
        analyzer="word",
        delimiter=" ",
        alternate_sign=True,
        n_features=2**20,
        dtype=cp.float32,
        binary=False,
        norm="l2",
        verbose=False,
        output_type=None,
    ):
        super().__init__(
            lowercase=lowercase,
            preprocessor=preprocessor,
            stop_words=stop_words,
            ngram_range=ngram_range,
            analyzer=analyzer,
            delimiter=delimiter,
            binary=binary,
            dtype=dtype,
            verbose=verbose,
            output_type=output_type,
        )
        self.alternate_sign = alternate_sign
        self.n_features = n_features
        self.norm = norm

    def _get_param_names(self):
        return [
            "alternate_sign",
            "n_features",
            "norm",
            *super()._get_param_names(),
        ]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        return tags

    def _check_params(self):
        super()._check_params()
        _check_oneof(self, "norm", ["l1", "l2", None])
        if self.n_features < 1:
            raise ValueError(
                f"Expected `n_features >= 1`, got {self.n_features!r}"
            )

    @mlfunc(set_input_type=True)
    def fit(self, X, y=None):
        """Only validate's the estimator's parameters.

        This estimator is stateless, ``fit`` is a no-op.

        Parameters
        ----------
        X : Iterable[str]
            Training samples. Each sample must be a text document
            which will be tokenized and hashed.
        y : None
            Ignored. Exists for API compatibility only.

        Returns
        -------
        self : object
            The instance itself.
        """
        self._check_params()
        return self

    def partial_fit(self, X, y=None):
        """Only validate's the estimator's parameters.

        This estimator is stateless, ``fit`` is a no-op.

        Parameters
        ----------
        X : Iterable[str]
            Training samples. Each sample must be a text document
            which will be tokenized and hashed.
        y : None
            Ignored. Exists for API compatibility only.

        Returns
        -------
        self : object
            The instance itself.
        """
        if not hasattr(self, "_input_type"):
            self._set_output_type(X)
        self._check_params()
        return self

    @mlfunc
    def transform(self, X):
        """Transform a sequence of documents to a document-term matrix.

        Parameters
        ----------
        X : Iterable[str]
            Training samples. Each sample must be a text document
            which will be tokenized and hashed.

        Returns
        -------
        X : sparse matrix of shape (n_samples, n_features)
            Document-term matrix.
        """
        X = check_cudf(X, ensure_ndim=1, input_name="X").reset_index(drop=True)
        tokens = self._to_tokens(X)
        hashes = tokens.token.hash_values().to_cupy().view("int32")

        if self.binary:
            tokens = (
                tokens.assign(feature_id=cp.abs(hashes) % self.n_features)
                .drop(columns="token")
                .drop_duplicates()
                .assign(value=cp.dtype(self.dtype).type(1))
                .sort_values(["doc_id", "feature_id"])
            )
        elif self.alternate_sign:
            tokens = (
                tokens.assign(
                    value=cp.sign(hashes, dtype=self.dtype),
                    feature_id=cp.abs(hashes) % self.n_features,
                )
                .drop(columns="token")
                .groupby(["doc_id", "feature_id"], sort=True)
                .value.sum()
                .reset_index(name="value")
            )
        else:
            tokens = (
                tokens.assign(feature_id=cp.abs(hashes) % self.n_features)
                .drop(columns="token")
                .groupby(["doc_id", "feature_id"], sort=True)
                .size()
                .reset_index(name="value")
            )

        out = self._to_sparse(X, tokens, n_features=self.n_features)

        if self.norm:
            if self.norm == "l1":
                csr_row_normalize_l1(out, inplace=True)
            elif self.norm == "l2":
                csr_row_normalize_l2(out, inplace=True)

        return out

    @mlfunc(preserve_index=True)
    def fit_transform(self, X, y=None):
        """Transform a sequence of documents to a document-term matrix.

        Parameters
        ----------
        X : Iterable[str]
            Training samples. Each sample must be a text document
            which will be tokenized and hashed.
        y : None
            Ignored. Exists for API compatibility only.

        Returns
        -------
        X : sparse matrix of shape (n_samples, n_features)
            Document-term matrix.
        """
        return self.fit(X, y).transform(X)


class CountVectorizer(DeprecatedGetFeatureNamesMixin, _BaseVectorizer):
    """Convert a collection of text documents to a matrix of token counts.

    If you do not provide an a-priori dictionary then the number of features
    will be equal to the vocabulary size found by analyzing the data.

    Parameters
    ----------
    lowercase : bool, default=True
        Convert all characters to lowercase before tokenizing.

    preprocessor : callable, default=None
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.

    stop_words : {'english'}, list, default=None
        If 'english', a built-in stop word list for English is used. If a list,
        that list is assumed to contain stop words, all of which will be
        removed from the resulting tokens. If None, no stop words will be used.
        Only applies if ``analyzer == 'word'``.

    ngram_range : tuple (min_n, max_n), default=(1, 1)
        The lower and upper boundary of the range of n-values for different
        word n-grams or char n-grams to be extracted. All values of n such
        such that min_n <= n <= max_n will be used. For example an
        ``ngram_range`` of ``(1, 1)`` means only unigrams, ``(1, 2)`` means
        unigrams and bigrams, and ``(2, 2)`` means only bigrams.

    analyzer : {'word', 'char', 'char_wb'}, default='word'
        Whether the feature should be made of word or character n-grams.
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

    max_features : int, default=None
        If not None, build a vocabulary that only consider the top
        `max_features` ordered by term frequency across the corpus.
        Otherwise, all features are used.

        This parameter is ignored if vocabulary is not None.

    vocabulary : array-like or mapping, default=None
        Either an array-like of terms, or a mapping where keys are terms and
        values are indices in the feature matrix. If not given, a vocabulary is
        determined from the input documents.

    binary : bool, default=False
        If True, all non zero counts are set to 1. This is useful for discrete
        probabilistic models that model binary events rather than integer
        counts.

    dtype : dtype, default=np.float32
        Type of the matrix returned by fit_transform() or transform().

    delimiter : str, default=" "
        String used as a replacement for stop words if `stop_words` is not
        None. Typically the delimiting character between words is a good
        choice.

    Attributes
    ----------
    vocabulary_ : cudf.Series
        The vocabulary used to map terms to feature indices.

    fixed_vocabulary_ : bool
        True if a fixed vocabulary of term to indices mapping
        is provided by the user.

    Examples
    --------
    >>> from cuml.feature_extraction.text import CountVectorizer
    >>> corpus = [
    ...     'This is the first document.',
    ...     'This document is the second document.',
    ...     'And this is the third one.',
    ...     'Is this the first document?',
    ... ]
    >>> vectorizer = CountVectorizer()
    >>> X = vectorizer.fit_transform(corpus)
    >>> vectorizer.get_feature_names_out()
    array(['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third',
           'this'], ...)
    >>> X.shape
    (4, 9)
    """

    def __init__(
        self,
        *,
        lowercase=True,
        preprocessor=None,
        stop_words=None,
        ngram_range=(1, 1),
        analyzer="word",
        delimiter=" ",
        max_df=1.0,
        min_df=1,
        max_features=None,
        vocabulary=None,
        binary=False,
        dtype=cp.float32,
        verbose=False,
        output_type=None,
    ):
        super().__init__(
            lowercase=lowercase,
            preprocessor=preprocessor,
            stop_words=stop_words,
            ngram_range=ngram_range,
            analyzer=analyzer,
            delimiter=delimiter,
            binary=binary,
            dtype=dtype,
            verbose=verbose,
            output_type=output_type,
        )
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.vocabulary = vocabulary

    def _get_param_names(self):
        return [
            "max_df",
            "min_df",
            "max_features",
            "vocabulary",
            *super()._get_param_names(),
        ]

    def _check_params(self):
        super()._check_params()
        for name in ["max_df", "min_df"]:
            val = getattr(self, name)
            if not (
                val >= 1 if isinstance(val, Integral) else (0 <= val <= 1)
            ):
                raise ValueError(
                    f"Expected a float `(0.0 <= {name} <= 1.0)` or an int "
                    f"`({name} >= 1)`, got {val}."
                )
        if self.max_features is not None and self.max_features < 1:
            raise ValueError(
                f"Expected `max_features >= 1`, got {self.max_features}"
            )

    def _fit(self, X, tokens):
        self.fixed_vocabulary_ = self.vocabulary is not None

        if self.vocabulary is not None:
            self.vocabulary_ = _check_vocabulary(self.vocabulary)
        else:
            n_doc = X.shape[0]
            max_features = self.max_features
            max_doc_count = (
                self.max_df
                if isinstance(self.max_df, Integral)
                else self.max_df * n_doc
            )
            min_doc_count = (
                self.min_df
                if isinstance(self.min_df, Integral)
                else self.min_df * n_doc
            )
            if max_doc_count < min_doc_count:
                raise ValueError(
                    "max_df corresponds to < documents than min_df"
                )

            if (
                max_doc_count < n_doc
                or min_doc_count > 1
                or max_features is not None
            ):
                counts = tokens.token.value_counts()
                if max_doc_count < n_doc:
                    counts = counts[counts <= max_doc_count]
                if min_doc_count > 1:
                    counts = counts[counts >= min_doc_count]
                if max_features is not None:
                    counts = counts.iloc[:max_features]
                self.vocabulary_ = cudf.Series(counts.index.sort_values())
                if not len(self.vocabulary_):
                    raise ValueError(
                        "After pruning, no terms remain. Try a lower min_df or "
                        "a higher max_df."
                    )
            else:
                self.vocabulary_ = (
                    tokens.token.drop_duplicates()
                    .sort_values()
                    .reset_index(drop=True)
                )
                if not len(self.vocabulary_):
                    raise ValueError(
                        "empty vocabulary; perhaps the documents only contain stop words"
                    )
        return self

    def _transform(self, X, tokens):
        tokens = (
            tokens.assign(
                token=tokens.token.astype(
                    cudf.CategoricalDtype(self.vocabulary_)
                ).cat.codes
            )
            .rename(columns={"token": "feature_id"})
            .dropna()
        )

        if self.binary:
            tokens = (
                tokens.drop_duplicates()
                .assign(value=cp.dtype(self.dtype).type(1))
                .sort_values(["doc_id", "feature_id"])
            )
        else:
            tokens = (
                tokens.groupby(["doc_id", "feature_id"], sort=True)
                .size()
                .reset_index(name="value")
            )

        return self._to_sparse(X, tokens, n_features=len(self.vocabulary_))

    @mlfunc(set_input_type=True)
    def fit(self, X, y=None):
        """Fit the vectorizer.

        Parameters
        ----------
        X : Iterable[str]
            Training samples. Each sample must be a text document
            which will be tokenized and hashed.
        y : None
            Ignored. Exists for API compatibility only.

        Returns
        -------
        self : object
            The instance itself.
        """
        self._check_params()
        X = check_cudf(X, ensure_ndim=1, input_name="X").reset_index(drop=True)
        tokens = self._to_tokens(X)
        self._fit(X, tokens)
        return self

    @mlfunc(set_input_type=True)
    def fit_transform(self, X, y=None):
        """Fit the vectorizer and return a document-term matrix.

        Parameters
        ----------
        X : Iterable[str]
            Training samples. Each sample must be a text document
            which will be tokenized and hashed.
        y : None
            Ignored. Exists for API compatibility only.

        Returns
        -------
        X : sparse matrix of shape (n_samples, n_features)
            Document-term matrix.
        """
        self._check_params()
        X = check_cudf(X, ensure_ndim=1, input_name="X").reset_index(drop=True)
        tokens = self._to_tokens(X)
        self._fit(X, tokens)
        return self._transform(X, tokens)

    @mlfunc
    def transform(self, X):
        """Transform documents to document-term matrix.

        Parameters
        ----------
        X : Iterable[str]
            Training samples. Each sample must be a text document
            which will be tokenized and hashed.

        Returns
        -------
        X : sparse matrix of shape (n_samples, n_features)
            Document-term matrix.
        """
        check_is_fitted(self)
        X = check_cudf(X, ensure_ndim=1, input_name="X").reset_index(drop=True)
        tokens = self._to_tokens(X)
        return self._transform(X, tokens)

    @mlfunc(preserve_index=True)
    def inverse_transform(self, X):
        """Return terms per document with nonzero entries in X.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Document-term matrix.

        Returns
        -------
        X_original : list of arrays of shape (n_samples,)
            List of arrays of terms.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse="csr", mem_type="host")
        n_samples = X.shape[0]
        vocab = self.vocabulary_.to_numpy()
        return [vocab[X[i, :].nonzero()[-1]].ravel() for i in range(n_samples)]

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
        check_is_fitted(self)
        return self.vocabulary_.to_numpy(dtype=object)


class TfidfTransformer(OneToOneFeatureMixin, Base):
    """Transform a count matrix to a normalized tf or tf-idf representation.

    Tf means term-frequency while tf-idf means term-frequency times inverse
    document-frequency. This is a common term weighting scheme in information
    retrieval, that has also found good use in document classification.

    The goal of using tf-idf instead of the raw frequencies of occurrence of a
    token in a given document is to scale down the impact of tokens that occur
    very frequently in a given corpus and that are hence empirically less
    informative than features that occur in a small fraction of the training
    corpus.

    The formula that is used to compute the tf-idf for a term t of a document d
    in a document set is tf-idf(t, d) = tf(t, d) * idf(t), and the idf is
    computed as idf(t) = log [ n / df(t) ] + 1 (if ``smooth_idf=False``), where
    n is the total number of documents in the document set and df(t) is the
    document frequency of t; the document frequency is the number of documents
    in the document set that contain the term t. The effect of adding "1" to
    the idf in the equation above is that terms with zero idf, i.e., terms
    that occur in all documents in a training set, will not be entirely
    ignored.
    (Note that the idf formula above differs from the standard textbook
    notation that defines the idf as
    idf(t) = log [ n / (df(t) + 1) ]).

    If ``smooth_idf=True`` (the default), the constant "1" is added to the
    numerator and denominator of the idf as if an extra document was seen
    containing every term in the collection exactly once, which prevents
    zero divisions: idf(t) = log [ (1 + n) / (1 + df(t)) ] + 1.

    Furthermore, the formulas used to compute tf and idf depend
    on parameter settings that correspond to the SMART notation used in IR
    as follows:

    Tf is "n" (natural) by default, "l" (logarithmic) when
    ``sublinear_tf=True``.
    Idf is "t" when use_idf is given, "n" (none) otherwise.
    Normalization is "c" (cosine) when ``norm='l2'``, "n" (none)
    when ``norm=None``.

    Parameters
    ----------
    norm : {'l1', 'l2', None}, default='l2'
        Norm used to normalize term vectors. None for no normalization.

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
        if  ``use_idf`` is True.

    Examples
    --------
    >>> from cuml.feature_extraction.text import TfidfTransformer
    >>> from cuml.feature_extraction.text import CountVectorizer
    >>> from sklearn.pipeline import Pipeline
    >>> corpus = ['this is the first document',
    ...           'this document is the second document',
    ...           'and this is the third one',
    ...           'is this the first document']
    >>> pipe = Pipeline([('count', CountVectorizer()),
    ...                  ('tfid', TfidfTransformer())])
    >>> X = pipe.fit_transform(corpus)
    >>> X.shape
    (4, 9)
    """

    idf_ = ReflectedAttr()

    def __init__(
        self,
        *,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
        verbose=False,
        output_type=None,
    ):
        super().__init__(verbose=verbose, output_type=output_type)
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

    def _get_param_names(self):
        return [
            "norm",
            "use_idf",
            "smooth_idf",
            "sublinear_tf",
            *super()._get_param_names(),
        ]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        tags.transformer_tags.preserves_dtype = ["float64", "float32"]
        return tags

    def _check_params(self):
        _check_oneof(self, "norm", ["l1", "l2", None])

    def _check_X(self, X, reset=False, copy=True):
        """Validate and normalize X to a CSR sparse matrix"""
        X = check_inputs(
            self,
            X,
            accept_sparse="csr",
            dtype=("float32", "float64"),
            reset=reset,
            copy=copy,
        )
        if not cp_sp.issparse(X):
            X = cp_sp.csr_matrix(X)
        return X

    def _fit(self, X):
        assert cp_sp.issparse(X)
        assert X.format == "csr"
        if self.use_idf:
            n_samples, _ = X.shape

            # Compute document frequency
            df = cp.bincount(X.indices, minlength=X.shape[1]).astype(
                X.dtype, copy=False
            )

            # perform idf smoothing if required
            if self.smooth_idf:
                df += 1.0
                n_samples += 1

            # log + 1 instead of log makes sure terms with zero idf don't get
            # suppressed entirely.
            idf = cp.full_like(df, fill_value=n_samples, dtype=X.dtype)
            idf /= df
            cp.log(idf, out=idf)
            idf += 1.0
            self.idf_ = idf

        return self

    def _transform(self, X):
        assert cp_sp.issparse(X)
        assert X.format == "csr"
        if self.sublinear_tf:
            cp.log(X.data, out=X.data)
            X.data += 1.0

        if self.use_idf:
            # the columns of X (CSR matrix) can be accessed with `X.indices `and
            # multiplied with the corresponding `idf` value
            X.data *= self.idf_[X.indices]

        if self.norm is not None:
            if self.norm == "l1":
                csr_row_normalize_l1(X, inplace=True)
            elif self.norm == "l2":
                csr_row_normalize_l2(X, inplace=True)

        return X

    @mlfunc(set_input_type=True)
    def fit(self, X, y=None):
        """Fit the transformer.

        Parameters
        ----------
        X : sparse matrix of shape (n_samples, n_features)
            A matrix of term/token counts.

        y : None
            Ignored. Exists for API compatibility only.

        Returns
        -------
        self : object
            The instance itself.
        """
        self._check_params()
        X = self._check_X(X, reset=True)
        return self._fit(X)

    @mlfunc(set_input_type=True)
    def fit_transform(self, X, y=None):
        """Fit the transformer, then transform X.

        Parameters
        ----------
        X : sparse matrix of shape (n_samples, n_features)
            A matrix of term/token counts.

        y : None
            Ignored. Exists for API compatibility only.

        Returns
        -------
        X : sparse matrix of shape (n_samples, n_features)
            Tf-idf weighted document-term matrix.
        """
        self._check_params()
        X = self._check_X(X, reset=True)
        return self._fit(X)._transform(X)

    @mlfunc
    def transform(self, X, copy=True):
        """Transform a count matrix to tf or tf-idf representation.

        Parameters
        ----------
        X : sparse matrix of shape (n_samples, n_features)
            A matrix of term/token counts.

        copy : bool, default=True
            If `copy=False,` then `transform` may mutate `X` in-place when
            possible.

        Returns
        -------
        X : sparse matrix of shape (n_samples, n_features)
            Tf-idf weighted document-term matrix.
        """
        check_is_fitted(self)
        X = self._check_X(X, copy=copy)
        return self._transform(X)


class TfidfVectorizer(CountVectorizer):
    """Convert a collection of raw documents to a matrix of TF-IDF features.

    Equivalent to `CountVectorizer` followed by `TfidfTransformer`.

    Parameters
    ----------
    lowercase : bool, default=True
        Convert all characters to lowercase before tokenizing.

    preprocessor : callable, default=None
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.

    stop_words : {'english'}, list, default=None
        If 'english', a built-in stop word list for English is used. If a list,
        that list is assumed to contain stop words, all of which will be
        removed from the resulting tokens. If None, no stop words will be used.
        Only applies if ``analyzer == 'word'``.

    ngram_range : tuple (min_n, max_n), default=(1, 1)
        The lower and upper boundary of the range of n-values for different
        word n-grams or char n-grams to be extracted. All values of n such
        such that min_n <= n <= max_n will be used. For example an
        ``ngram_range`` of ``(1, 1)`` means only unigrams, ``(1, 2)`` means
        unigrams and bigrams, and ``(2, 2)`` means only bigrams.

    analyzer : {'word', 'char', 'char_wb'}, default='word'
        Whether the feature should be made of word or character n-grams.
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

    max_features : int, default=None
        If not None, build a vocabulary that only consider the top
        `max_features` ordered by term frequency across the corpus.
        Otherwise, all features are used.

        This parameter is ignored if vocabulary is not None.

    vocabulary : array-like or mapping, default=None
        Either an array-like of terms, or a mapping where keys are terms and
        values are indices in the feature matrix. If not given, a vocabulary is
        determined from the input documents.

    binary : bool, default=False
        If True, all non zero counts are set to 1. This is useful for discrete
        probabilistic models that model binary events rather than integer
        counts.

    dtype : dtype, default=np.float32
        Type of the matrix returned by fit_transform() or transform().

    delimiter : str, default=" "
        String used as a replacement for stop words if `stop_words` is not
        None. Typically the delimiting character between words is a good
        choice.

    norm : {'l1', 'l2', None}, default='l2'
        Norm used to normalize term vectors. None for no normalization.

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
    vocabulary_ : cudf.Series
        The vocabulary used to map terms to feature indices.

    fixed_vocabulary_ : bool
        True if a fixed vocabulary of term to indices mapping
        is provided by the user.

    idf_ : array of shape (n_features)
        The inverse document frequency (IDF) vector; only defined
        if  ``use_idf`` is True.

    Examples
    --------
    >>> from cuml.feature_extraction.text import TfidfVectorizer
    >>> corpus = [
    ...     'This is the first document.',
    ...     'This document is the second document.',
    ...     'And this is the third one.',
    ...     'Is this the first document?',
    ... ]
    >>> vectorizer = TfidfVectorizer()
    >>> X = vectorizer.fit_transform(corpus)
    >>> vectorizer.get_feature_names_out()
    array(['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third',
           'this'], ...)
    >>> X.shape
    (4, 9)
    """

    def __init__(
        self,
        *,
        lowercase=True,
        preprocessor=None,
        stop_words=None,
        ngram_range=(1, 1),
        analyzer="word",
        delimiter=" ",
        max_df=1.0,
        min_df=1,
        max_features=None,
        vocabulary=None,
        binary=False,
        dtype=cp.float32,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
        verbose=False,
        output_type=None,
    ):
        super().__init__(
            lowercase=lowercase,
            preprocessor=preprocessor,
            stop_words=stop_words,
            ngram_range=ngram_range,
            analyzer=analyzer,
            delimiter=delimiter,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            binary=binary,
            dtype=dtype,
            verbose=verbose,
            output_type=output_type,
        )
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

    def _get_param_names(self):
        return [
            "norm",
            "use_idf",
            "smooth_idf",
            "sublinear_tf",
            *super()._get_param_names(),
        ]

    @property
    def idf_(self):
        return self._tfidf.idf_

    def _check_params(self):
        super()._check_params()
        _check_oneof(self, "norm", ["l1", "l2", None])

    def _fit_transform(self, X, transform=True):
        self._check_params()
        self._tfidf = TfidfTransformer(
            norm=self.norm,
            use_idf=self.use_idf,
            smooth_idf=self.smooth_idf,
            sublinear_tf=self.sublinear_tf,
        )
        X = super().fit_transform(X)
        self._tfidf.fit(X)
        if transform:
            return self._tfidf.transform(X)
        return self

    @mlfunc(set_input_type=True)
    def fit_transform(self, X, y=None):
        """Fit the vectorizer and return a document-term matrix.

        Parameters
        ----------
        X : Iterable[str]
            Training samples. Each sample must be a text document
            which will be tokenized and hashed.
        y : None
            Ignored. Exists for API compatibility only.

        Returns
        -------
        X : sparse matrix of shape (n_samples, n_features)
            Tf-idf weighted document-term matrix.
        """
        return self._fit_transform(X)

    @mlfunc(set_input_type=True)
    def fit(self, X, y=None):
        """Fit the vectorizer.

        Parameters
        ----------
        X : Iterable[str]
            Training samples. Each sample must be a text document
            which will be tokenized and hashed.
        y : None
            Ignored. Exists for API compatibility only.

        Returns
        -------
        self : object
            The instance itself.
        """
        return self._fit_transform(X, transform=False)

    @mlfunc
    def transform(self, X):
        """Transform documents to document-term matrix.

        Parameters
        ----------
        X : Iterable[str]
            Training samples. Each sample must be a text document
            which will be tokenized and hashed.

        Returns
        -------
        X : sparse matrix of shape (n_samples, n_features)
            Tf-idf weighted document-term matrix.
        """
        check_is_fitted(self)

        X = super().transform(X)
        return self._tfidf.transform(X, copy=False)
