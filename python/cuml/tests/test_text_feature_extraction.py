#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from sklearn.feature_extraction.text import CountVectorizer as SkCountVect
from sklearn.feature_extraction.text import HashingVectorizer as SkHashVect
from sklearn.feature_extraction.text import TfidfVectorizer as SkTfidfVect

from cuml.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)


def test_count_vectorizer():
    corpus = [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?",
    ]

    res = CountVectorizer().fit_transform(corpus)
    ref = SkCountVect().fit_transform(corpus)
    cp.testing.assert_array_equal(res.toarray(), ref.toarray())


JUNK_FOOD_DOCS = (
    "the pizza pizza beer copyright",
    "the pizza burger beer copyright",
    "the the pizza beer beer copyright",
    "the burger beer beer copyright",
    "the coke burger coke copyright",
    "the coke burger burger",
)

NOTJUNK_FOOD_DOCS = (
    "the salad celeri copyright",
    "the salad salad sparkling water copyright",
    "the the celeri celeri copyright",
    "the tomato tomato salad water",
    "the tomato salad water copyright",
)

EMPTY_DOCS = ("",)

DOCS = JUNK_FOOD_DOCS + EMPTY_DOCS + NOTJUNK_FOOD_DOCS + EMPTY_DOCS

NGRAM_RANGES = [(1, 1), (1, 2), (2, 3)]
NGRAM_IDS = [f"ngram_range={str(r)}" for r in NGRAM_RANGES]


@pytest.mark.parametrize("ngram_range", NGRAM_RANGES, ids=NGRAM_IDS)
def test_word_analyzer(ngram_range):
    v = CountVectorizer(ngram_range=ngram_range).fit(DOCS)
    ref = SkCountVect(ngram_range=ngram_range).fit(DOCS)
    assert_array_equal(
        ref.get_feature_names_out(),
        v.get_feature_names_out(),
    )


def test_preprocessor():
    corpus = ["aa bb cc", "aa bb ee", "cc dd ff"]
    vec = CountVectorizer(
        preprocessor=lambda s: s.str.upper(),
        stop_words=["EE"],
    ).fit(corpus)
    res = vec.get_feature_names_out()
    np.testing.assert_array_equal(
        res,
        ["AA", "BB", "CC", "DD", "FF"],
    )


def test_delimiter():
    corpus = ["aa0bb0cc", "aa 0 bb0ee", "c0d0f"]
    vec = CountVectorizer(
        delimiter="0",
        stop_words=["ee"],
    ).fit(corpus)
    res = vec.get_feature_names_out()
    np.testing.assert_array_equal(
        res,
        [" bb", "aa", "aa ", "bb", "c", "cc", "d", "f"],
    )


def test_tokenizer():
    corpus = [
        "filler<term1>filler <term2>",
        "filler<term1>filler  <term3 >",
        "<term4> but not <term5>no terms here",
    ]
    vec = CountVectorizer(
        tokenizer=lambda s: s.str.findall(r"<([\w\s]*)>"),
        delimiter="|",
        stop_words=["term5"],
    ).fit(corpus)
    res = vec.get_feature_names_out()
    np.testing.assert_array_equal(
        res,
        ["term1", "term2", "term3 ", "term4"],
    )


def test_countvectorizer_custom_vocabulary():
    vocab = {"pizza": 0, "beer": 1}

    ref = SkCountVect(vocabulary=vocab).fit_transform(DOCS)
    X = CountVectorizer(vocabulary=vocab).fit_transform(DOCS)
    cp.testing.assert_array_equal(X.toarray(), ref.toarray())


def test_countvectorizer_stop_words():
    ref = SkCountVect(stop_words="english").fit_transform(DOCS)
    X = CountVectorizer(stop_words="english").fit_transform(DOCS)
    cp.testing.assert_array_equal(X.toarray(), ref.toarray())


def test_countvectorizer_empty_vocabulary():
    v = CountVectorizer(max_df=1.0, stop_words="english")
    # fitting only on stopwords will result in an empty vocabulary
    with pytest.raises(ValueError):
        v.fit(["to be or not to be", "and me too", "and so do you"])


def test_countvectorizer_stop_words_ngrams():
    stop_words_doc = ["and me too andy andy too"]
    expected_vocabulary = ["andy andy"]

    v = CountVectorizer(ngram_range=(2, 2), stop_words="english")
    v.fit(stop_words_doc)

    assert_array_equal(v.get_feature_names_out(), expected_vocabulary)


def test_countvectorizer_max_features():
    expected_vocabulary = {"burger", "beer", "salad", "pizza"}

    # test bounded number of extracted features
    vec = CountVectorizer(max_df=0.6, max_features=4)
    vec.fit(DOCS)
    assert set(vec.get_feature_names_out()) == expected_vocabulary


def test_countvectorizer_max_features_counts():
    cv_1 = CountVectorizer(max_features=1)
    cv_3 = CountVectorizer(max_features=3)
    cv_None = CountVectorizer(max_features=None)

    counts_1 = cv_1.fit_transform(JUNK_FOOD_DOCS).sum(axis=0)
    counts_3 = cv_3.fit_transform(JUNK_FOOD_DOCS).sum(axis=0)
    counts_None = cv_None.fit_transform(JUNK_FOOD_DOCS).sum(axis=0)

    features_1 = cv_1.get_feature_names_out()
    features_3 = cv_3.get_feature_names_out()
    features_None = cv_None.get_feature_names_out()

    # The most common feature is "the", with frequency 7.
    assert 7 == counts_1.max()
    assert 7 == counts_3.max()
    assert 7 == counts_None.max()

    # The most common feature should be the same
    def as_index(x):
        return x.astype(cp.int32).item()

    assert "the" == features_1[as_index(counts_1.argmax())]
    assert "the" == features_3[as_index(counts_3.argmax())]
    assert "the" == features_None[as_index(counts_None.argmax())]


def test_countvectorizer_max_df():
    test_data = ["abc", "dea", "eat"]
    vect = CountVectorizer(analyzer="char", max_df=1.0)
    vect.fit(test_data)
    assert "a" in vect.vocabulary_.to_arrow().to_pylist()
    assert len(vect.vocabulary_.to_arrow().to_pylist()) == 6

    vect.max_df = 0.5  # 0.5 * 3 documents -> max_doc_count == 1.5
    vect.fit(test_data)
    assert "a" not in vect.vocabulary_.to_arrow().to_pylist()  # {ae} ignored
    assert len(vect.vocabulary_.to_arrow().to_pylist()) == 4  # {bcdt} remain

    vect.max_df = 1
    vect.fit(test_data)
    assert "a" not in vect.vocabulary_.to_arrow().to_pylist()  # {ae} ignored
    assert len(vect.vocabulary_.to_arrow().to_pylist()) == 4  # {bcdt} remain


def test_vectorizer_min_df():
    test_data = ["abc", "dea", "eat"]
    vect = CountVectorizer(analyzer="char", min_df=1)
    vect.fit(test_data)
    assert "a" in vect.vocabulary_.to_arrow().to_pylist()
    assert len(vect.vocabulary_.to_arrow().to_pylist()) == 6

    vect.min_df = 2
    vect.fit(test_data)
    assert "c" not in vect.vocabulary_.to_arrow().to_pylist()  # {bcdt} ignored
    assert len(vect.vocabulary_.to_arrow().to_pylist()) == 2  # {ae} remain

    vect.min_df = 0.8  # 0.8 * 3 documents -> min_doc_count == 2.4
    vect.fit(test_data)
    # {bcdet} ignored
    assert "c" not in vect.vocabulary_.to_arrow().to_pylist()
    assert len(vect.vocabulary_.to_arrow().to_pylist()) == 1  # {a} remains


def test_count_binary_occurrences():
    # by default multiple occurrences are counted as longs
    test_data = ["aaabc", "abbde"]
    vect = CountVectorizer(analyzer="char", max_df=1.0)
    X = cp.asnumpy(vect.fit_transform(test_data).toarray())
    assert_array_equal(["a", "b", "c", "d", "e"], vect.get_feature_names_out())
    assert_array_equal([[3, 1, 1, 0, 0], [1, 2, 0, 1, 1]], X)

    # using boolean features, we can fetch the binary occurrence info
    # instead.
    vect = CountVectorizer(analyzer="char", max_df=1.0, binary=True)
    X = cp.asnumpy(vect.fit_transform(test_data).toarray())
    assert_array_equal([[1, 1, 1, 0, 0], [1, 1, 0, 1, 1]], X)

    # check the ability to change the dtype
    vect = CountVectorizer(
        analyzer="char", max_df=1.0, binary=True, dtype=cp.float32
    )
    X = vect.fit_transform(test_data)
    assert X.dtype == cp.float32


def test_vectorizer_inverse_transform():
    vectorizer = CountVectorizer()
    transformed_data = vectorizer.fit_transform(DOCS)
    inversed_data = vectorizer.inverse_transform(transformed_data)

    sk_vectorizer = SkCountVect()
    sk_transformed_data = sk_vectorizer.fit_transform(DOCS)
    sk_inversed_data = sk_vectorizer.inverse_transform(sk_transformed_data)

    for doc, sk_doc in zip(inversed_data, sk_inversed_data):
        doc = np.sort(doc)
        sk_doc = np.sort(sk_doc)
        assert_array_equal(doc, sk_doc)


@pytest.mark.skip(
    reason="scikit-learn replaced get_feature_names with "
    "get_feature_names_out"
    "https://github.com/NVIDIA/cuml/issues/5159"
)
@pytest.mark.parametrize("ngram_range", NGRAM_RANGES, ids=NGRAM_IDS)
def test_space_ngrams(ngram_range):
    data = ["abc      def. 123 456    789"]
    vec = CountVectorizer(ngram_range=ngram_range).fit(data)
    ref = SkCountVect(ngram_range=ngram_range).fit(data)
    assert_array_equal(
        ref.get_feature_names_out(),
        vec.get_feature_names_out(),
    )


def test_empty_doc_after_limit_features():
    data = ["abc abc def", "def abc", "ghi"]
    count = CountVectorizer(min_df=2).fit_transform(data)
    ref = SkCountVect(min_df=2).fit_transform(data)
    cp.testing.assert_array_equal(count.toarray(), ref.toarray())


def test_countvectorizer_separate_fit_transform():
    res = CountVectorizer().fit(DOCS).transform(DOCS)
    ref = SkCountVect().fit(DOCS).transform(DOCS)
    cp.testing.assert_array_equal(res.toarray(), ref.toarray())


def test_non_ascii():
    non_ascii = ("This is ascii,", "but not this Αγγλικά.")

    cv = CountVectorizer()
    res = cv.fit_transform(non_ascii)
    ref = SkCountVect().fit_transform(non_ascii)

    assert "αγγλικά" in set(cv.get_feature_names_out())
    cp.testing.assert_array_equal(res.toarray(), ref.toarray())


def test_single_token_length():
    data = ["S I N G L E T 0 K E N Example", "1 2 3 4 5 eg"]

    cv = CountVectorizer()
    res = cv.fit_transform(data)
    ref = SkCountVect().fit_transform(data)

    cp.testing.assert_array_equal(res.toarray(), ref.toarray())


def test_only_delimiters():
    data = ["abc def. 123", "   ", "456 789"]
    res = CountVectorizer().fit_transform(data)
    ref = SkCountVect().fit_transform(data)
    cp.testing.assert_array_equal(res.toarray(), ref.toarray())


@pytest.mark.skip(
    reason="scikit-learn replaced get_feature_names with "
    "get_feature_names_out"
    "https://github.com/NVIDIA/cuml/issues/5159"
)
@pytest.mark.parametrize("analyzer", ["char", "char_wb"])
@pytest.mark.parametrize("ngram_range", NGRAM_RANGES, ids=NGRAM_IDS)
def test_character_ngrams(analyzer, ngram_range):
    data = ["ab c", "edf gh"]

    res = CountVectorizer(analyzer=analyzer, ngram_range=ngram_range).fit(data)
    ref = SkCountVect(analyzer=analyzer, ngram_range=ngram_range).fit(data)

    assert_array_equal(
        res.get_feature_names_out(),
        ref.get_feature_names_out(),
    )


@pytest.mark.parametrize(
    "query",
    [
        ["science aa", "", "a aa aaa"],
        ["science aa", ""],
        ["science"],
    ],
)
def test_transform_unsigned_categories(query):
    token = "a"
    thousand_tokens = list()
    for i in range(1000):
        thousand_tokens.append(token)
        token += "a"
    thousand_tokens[128] = "science"

    vec = CountVectorizer().fit(thousand_tokens)
    res = vec.transform(query)

    assert res.shape[0] == len(query)


# ----------------------------------------------------------------
# TfidfVectorizer tests are already covered by CountVectorizer and
# TfidfTransformer so we only do the bare minimum tests here
# ----------------------------------------------------------------


@pytest.mark.parametrize("norm", ["l1", "l2", None])
@pytest.mark.parametrize("use_idf", [True, False])
@pytest.mark.parametrize("smooth_idf", [True, False])
@pytest.mark.parametrize("sublinear_tf", [True, False])
def test_tfidf_vectorizer(norm, use_idf, smooth_idf, sublinear_tf):
    tfidf_mat = TfidfVectorizer(
        norm=norm,
        use_idf=use_idf,
        smooth_idf=smooth_idf,
        sublinear_tf=sublinear_tf,
    ).fit_transform(DOCS)

    ref = SkTfidfVect(
        norm=norm,
        use_idf=use_idf,
        smooth_idf=smooth_idf,
        sublinear_tf=sublinear_tf,
    ).fit_transform(DOCS)

    cp.testing.assert_array_almost_equal(tfidf_mat.toarray(), ref.toarray())


def test_tfidf_vectorizer_get_feature_names_out():
    corpus = [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?",
    ]
    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(corpus)
    output = [
        "and",
        "document",
        "first",
        "is",
        "one",
        "second",
        "the",
        "third",
        "this",
    ]
    assert_array_equal(vectorizer.get_feature_names_out(), output)


@pytest.mark.parametrize("cls", [TfidfVectorizer, CountVectorizer])
def test_vectorizer_get_feature_names_deprecated(cls):
    X = [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?",
    ]
    model = cls().fit(X)
    with pytest.warns(FutureWarning, match="get_feature_names"):
        res = model.get_feature_names()

    np.testing.assert_array_equal(res, model.get_feature_names_out())


def test_tfidf_vectorizer_char_wb_ngrams():
    # Regression test for #8416: get_char_ngrams misaligned padded tokens
    # across documents once index alignment relied on the original
    # per-document index instead of a reset range index.
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6))
    tfidf_mat = vectorizer.fit_transform(DOCS)

    ref_vectorizer = SkTfidfVect(analyzer="char_wb", ngram_range=(2, 6))
    ref = ref_vectorizer.fit_transform(DOCS)

    cp.testing.assert_array_almost_equal(tfidf_mat.todense(), ref.toarray())
    assert_array_equal(
        vectorizer.get_feature_names_out(),
        ref_vectorizer.get_feature_names_out(),
    )


# ----------------------------------------------------------------
# HashingVectorizer tests
# ----------------------------------------------------------------


def test_hashingvectorizer():
    corpus = [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?",
    ]

    res = HashingVectorizer().fit_transform(corpus)
    ref = SkHashVect().fit_transform(corpus)
    np.testing.assert_allclose(res.toarray(), ref.toarray())


@pytest.mark.xfail
@pytest.mark.filterwarnings(
    "ignore:The parameter 'token_pattern' will not be used:UserWarning:sklearn"
)
def test_vectorizer_empty_token_case():
    """
    We ignore empty tokens right now but sklearn treats them as a character
    we might want to look into this more but
    this should not be a concern for most pipelines
    """
    corpus = [
        "a b ",
    ]

    # we have extra null token here
    # we slightly diverge from sklearn here as not treating it as a token
    res = CountVectorizer(preprocessor=lambda s: s).fit_transform(corpus)
    ref = SkCountVect(
        preprocessor=lambda s: s, tokenizer=lambda s: s.split(" ")
    ).fit_transform(corpus)
    cp.testing.assert_array_equal(res.toarray(), ref.toarray())

    res = HashingVectorizer(preprocessor=lambda s: s).fit_transform(corpus)
    ref = SkHashVect(
        preprocessor=lambda s: s, tokenizer=lambda s: s.split(" ")
    ).fit_transform(corpus)
    np.testing.assert_allclose(res.toarray(), ref.toarray())


@pytest.mark.parametrize("lowercase", [False, True])
def test_hashingvectorizer_lowercase(lowercase):
    corpus = [
        "This Is DoC",
        "this DoC is the second DoC.",
        "And this document is the third one.",
        "and Is this the first document?",
    ]
    res = HashingVectorizer(lowercase=lowercase).fit_transform(corpus)
    ref = SkHashVect(lowercase=lowercase).fit_transform(corpus)
    np.testing.assert_allclose(res.toarray(), ref.toarray())


def test_hashingvectorizer_stop_word():
    ref = SkHashVect(stop_words="english").fit_transform(DOCS)
    res = HashingVectorizer(stop_words="english").fit_transform(DOCS)
    np.testing.assert_allclose(res.toarray(), ref.toarray())


def test_hashingvectorizer_n_features():
    n_features = 10
    res = (
        HashingVectorizer(n_features=n_features).fit_transform(DOCS).toarray()
    )
    ref = SkHashVect(n_features=n_features).fit_transform(DOCS).toarray()
    assert res.shape == ref.shape


@pytest.mark.parametrize("norm", ["l1", "l2", None, "max"])
def test_hashingvectorizer_norm(norm):
    if norm not in ["l1", "l2", None]:
        with pytest.raises(ValueError):
            res = HashingVectorizer(norm=norm).fit_transform(DOCS)
    else:
        res = HashingVectorizer(norm=norm).fit_transform(DOCS)
        ref = SkHashVect(norm=norm).fit_transform(DOCS)
        np.testing.assert_allclose(res.toarray(), ref.toarray())


def test_hashingvectorizer_alternate_sign():
    # if alternate_sign = True
    # we should have some negative and positive values
    res = HashingVectorizer(alternate_sign=True).fit_transform(DOCS)
    res_f_array = res.toarray().flatten()
    assert np.sum(res_f_array > 0, axis=0) > 0
    assert np.sum(res_f_array < 0, axis=0) > 0

    # if alternate_sign = False
    # we should have no negative values and some positive values
    res = HashingVectorizer(alternate_sign=False).fit_transform(DOCS)
    res_f_array = res.toarray().flatten()
    assert np.sum(res_f_array > 0, axis=0) > 0
    assert np.sum(res_f_array < 0, axis=0) == 0


@pytest.mark.parametrize("dtype", [np.float32, np.float64, cp.float64])
def test_hashingvectorizer_dtype(dtype):
    res = HashingVectorizer(dtype=dtype).fit_transform(DOCS)
    assert res.dtype == dtype


@pytest.mark.parametrize("vectorizer", ["tfidf", "hash_vec", "count_vec"])
def test_vectorizer_with_pandas_series(vectorizer):
    corpus = [
        "This Is DoC",
        "this DoC is the second DoC.",
        "And this document is the third one.",
        "and Is this the first document?",
    ]
    cuml_vec, sklearn_vec = {
        "tfidf": (TfidfVectorizer, SkTfidfVect),
        "hash_vec": (HashingVectorizer, SkHashVect),
        "count_vec": (CountVectorizer, SkCountVect),
    }[vectorizer]
    raw_documents = pd.Series(corpus)
    res = cuml_vec(dtype=np.float32).fit_transform(raw_documents)
    ref = sklearn_vec(dtype=np.float32).fit_transform(raw_documents)
    np.testing.assert_allclose(res.toarray(), ref.toarray())
