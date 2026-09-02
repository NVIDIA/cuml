#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
import numpy as np
import pandas as pd
import pytest
from cudf import Series
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

    res = CountVectorizer().fit_transform(Series(corpus))
    ref = SkCountVect().fit_transform(corpus)
    cp.testing.assert_array_equal(res.todense(), ref.toarray())


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
DOCS_GPU = Series(DOCS)

NGRAM_RANGES = [(1, 1), (1, 2), (2, 3)]
NGRAM_IDS = [f"ngram_range={str(r)}" for r in NGRAM_RANGES]


@pytest.mark.parametrize("ngram_range", NGRAM_RANGES, ids=NGRAM_IDS)
def test_word_analyzer(ngram_range):
    v = CountVectorizer(ngram_range=ngram_range).fit(DOCS_GPU)
    ref = SkCountVect(ngram_range=ngram_range).fit(DOCS)
    assert_array_equal(
        ref.get_feature_names_out(),
        v.get_feature_names_out(),
    )


def test_countvectorizer_custom_vocabulary():
    vocab = {"pizza": 0, "beer": 1}
    vocab_gpu = Series(vocab.keys())

    ref = SkCountVect(vocabulary=vocab).fit_transform(DOCS)
    X = CountVectorizer(vocabulary=vocab_gpu).fit_transform(DOCS_GPU)
    cp.testing.assert_array_equal(X.todense(), ref.toarray())


def test_countvectorizer_stop_words():
    ref = SkCountVect(stop_words="english").fit_transform(DOCS)
    X = CountVectorizer(stop_words="english").fit_transform(DOCS_GPU)
    cp.testing.assert_array_equal(X.todense(), ref.toarray())


def test_countvectorizer_empty_vocabulary():
    v = CountVectorizer(max_df=1.0, stop_words="english")
    # fitting only on stopwords will result in an empty vocabulary
    with pytest.raises(ValueError):
        v.fit(Series(["to be or not to be", "and me too", "and so do you"]))


def test_countvectorizer_stop_words_ngrams():
    stop_words_doc = Series(["and me too andy andy too"])
    expected_vocabulary = ["andy andy"]

    v = CountVectorizer(ngram_range=(2, 2), stop_words="english")
    v.fit(stop_words_doc)

    assert_array_equal(v.get_feature_names_out(), expected_vocabulary)


def test_countvectorizer_max_features():
    expected_vocabulary = {"burger", "beer", "salad", "pizza"}
    expected_stop_words = {
        "celeri",
        "tomato",
        "copyright",
        "coke",
        "sparkling",
        "water",
        "the",
    }

    # test bounded number of extracted features
    vec = CountVectorizer(max_df=0.6, max_features=4)
    vec.fit(DOCS_GPU)
    assert set(vec.get_feature_names_out()) == expected_vocabulary
    assert set(vec.stop_words_.to_arrow().to_pylist()) == expected_stop_words


def test_countvectorizer_max_features_counts():
    JUNK_FOOD_DOCS_GPU = Series(JUNK_FOOD_DOCS)

    cv_1 = CountVectorizer(max_features=1)
    cv_3 = CountVectorizer(max_features=3)
    cv_None = CountVectorizer(max_features=None)

    counts_1 = cv_1.fit_transform(JUNK_FOOD_DOCS_GPU).sum(axis=0)
    counts_3 = cv_3.fit_transform(JUNK_FOOD_DOCS_GPU).sum(axis=0)
    counts_None = cv_None.fit_transform(JUNK_FOOD_DOCS_GPU).sum(axis=0)

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

    assert "the" == features_1[as_index(cp.argmax(counts_1))]
    assert "the" == features_3[as_index(cp.argmax(counts_3))]
    assert "the" == features_None[as_index(cp.argmax(counts_None))]


def test_countvectorizer_max_df():
    test_data = Series(["abc", "dea", "eat"])
    vect = CountVectorizer(analyzer="char", max_df=1.0)
    vect.fit(test_data)
    assert "a" in vect.vocabulary_.to_arrow().to_pylist()
    assert len(vect.vocabulary_.to_arrow().to_pylist()) == 6
    assert len(vect.stop_words_) == 0

    vect.max_df = 0.5  # 0.5 * 3 documents -> max_doc_count == 1.5
    vect.fit(test_data)
    assert "a" not in vect.vocabulary_.to_arrow().to_pylist()  # {ae} ignored
    assert len(vect.vocabulary_.to_arrow().to_pylist()) == 4  # {bcdt} remain
    assert "a" in vect.stop_words_.to_arrow().to_pylist()
    assert len(vect.stop_words_) == 2

    vect.max_df = 1
    vect.fit(test_data)
    assert "a" not in vect.vocabulary_.to_arrow().to_pylist()  # {ae} ignored
    assert len(vect.vocabulary_.to_arrow().to_pylist()) == 4  # {bcdt} remain
    assert "a" in vect.stop_words_.to_arrow().to_pylist()
    assert len(vect.stop_words_) == 2


def test_vectorizer_min_df():
    test_data = Series(["abc", "dea", "eat"])
    vect = CountVectorizer(analyzer="char", min_df=1)
    vect.fit(test_data)
    assert "a" in vect.vocabulary_.to_arrow().to_pylist()
    assert len(vect.vocabulary_.to_arrow().to_pylist()) == 6
    assert len(vect.stop_words_) == 0

    vect.min_df = 2
    vect.fit(test_data)
    assert "c" not in vect.vocabulary_.to_arrow().to_pylist()  # {bcdt} ignored
    assert len(vect.vocabulary_.to_arrow().to_pylist()) == 2  # {ae} remain
    assert "c" in vect.stop_words_.to_arrow().to_pylist()
    assert len(vect.stop_words_) == 4

    vect.min_df = 0.8  # 0.8 * 3 documents -> min_doc_count == 2.4
    vect.fit(test_data)
    # {bcdet} ignored
    assert "c" not in vect.vocabulary_.to_arrow().to_pylist()
    assert len(vect.vocabulary_.to_arrow().to_pylist()) == 1  # {a} remains
    assert "c" in vect.stop_words_.to_arrow().to_pylist()
    assert len(vect.stop_words_) == 5


def test_count_binary_occurrences():
    # by default multiple occurrences are counted as longs
    test_data = Series(["aaabc", "abbde"])
    vect = CountVectorizer(analyzer="char", max_df=1.0)
    X = cp.asnumpy(vect.fit_transform(test_data).todense())
    assert_array_equal(["a", "b", "c", "d", "e"], vect.get_feature_names_out())
    assert_array_equal([[3, 1, 1, 0, 0], [1, 2, 0, 1, 1]], X)

    # using boolean features, we can fetch the binary occurrence info
    # instead.
    vect = CountVectorizer(analyzer="char", max_df=1.0, binary=True)
    X = cp.asnumpy(vect.fit_transform(test_data).todense())
    assert_array_equal([[1, 1, 1, 0, 0], [1, 1, 0, 1, 1]], X)

    # check the ability to change the dtype
    vect = CountVectorizer(
        analyzer="char", max_df=1.0, binary=True, dtype=cp.float32
    )
    X = vect.fit_transform(test_data)
    assert X.dtype == cp.float32


def test_vectorizer_inverse_transform():
    vectorizer = CountVectorizer()
    transformed_data = vectorizer.fit_transform(DOCS_GPU)
    inversed_data = vectorizer.inverse_transform(transformed_data)

    sk_vectorizer = SkCountVect()
    sk_transformed_data = sk_vectorizer.fit_transform(DOCS)
    sk_inversed_data = sk_vectorizer.inverse_transform(sk_transformed_data)

    for doc, sk_doc in zip(inversed_data, sk_inversed_data):
        doc = np.sort(doc.to_arrow().to_pylist())
        sk_doc = np.sort(sk_doc)
        if len(doc) + len(sk_doc) == 0:
            continue
        assert_array_equal(doc, sk_doc)


@pytest.mark.skip(
    reason="scikit-learn replaced get_feature_names with "
    "get_feature_names_out"
    "https://github.com/NVIDIA/cuml/issues/5159"
)
@pytest.mark.parametrize("ngram_range", NGRAM_RANGES, ids=NGRAM_IDS)
def test_space_ngrams(ngram_range):
    data = ["abc      def. 123 456    789"]
    data_gpu = Series(data)
    vec = CountVectorizer(ngram_range=ngram_range).fit(data_gpu)
    ref = SkCountVect(ngram_range=ngram_range).fit(data)
    assert_array_equal(
        ref.get_feature_names_out(),
        vec.get_feature_names_out(),
    )


def test_empty_doc_after_limit_features():
    data = ["abc abc def", "def abc", "ghi"]
    data_gpu = Series(data)
    count = CountVectorizer(min_df=2).fit_transform(data_gpu)
    ref = SkCountVect(min_df=2).fit_transform(data)
    cp.testing.assert_array_equal(count.todense(), ref.toarray())


def test_countvectorizer_separate_fit_transform():
    res = CountVectorizer().fit(DOCS_GPU).transform(DOCS_GPU)
    ref = SkCountVect().fit(DOCS).transform(DOCS)
    cp.testing.assert_array_equal(res.todense(), ref.toarray())


def test_non_ascii():
    non_ascii = ("This is ascii,", "but not this Αγγλικά.")
    non_ascii_gpu = Series(non_ascii)

    cv = CountVectorizer()
    res = cv.fit_transform(non_ascii_gpu)
    ref = SkCountVect().fit_transform(non_ascii)

    assert "αγγλικά" in set(cv.get_feature_names_out())
    cp.testing.assert_array_equal(res.todense(), ref.toarray())


def test_sngle_len():
    single_token_ser = ["S I N G L E T 0 K E N Example", "1 2 3 4 5 eg"]
    single_token_gpu = Series(single_token_ser)

    cv = CountVectorizer()
    res = cv.fit_transform(single_token_gpu)
    ref = SkCountVect().fit_transform(single_token_ser)

    cp.testing.assert_array_equal(res.todense(), ref.toarray())


def test_only_delimiters():
    data = ["abc def. 123", "   ", "456 789"]
    data_gpu = Series(data)
    res = CountVectorizer().fit_transform(data_gpu)
    ref = SkCountVect().fit_transform(data)
    cp.testing.assert_array_equal(res.todense(), ref.toarray())


@pytest.mark.skip(
    reason="scikit-learn replaced get_feature_names with "
    "get_feature_names_out"
    "https://github.com/NVIDIA/cuml/issues/5159"
)
@pytest.mark.parametrize("analyzer", ["char", "char_wb"])
@pytest.mark.parametrize("ngram_range", NGRAM_RANGES, ids=NGRAM_IDS)
def test_character_ngrams(analyzer, ngram_range):
    data = ["ab c", "edf gh"]

    res = CountVectorizer(analyzer=analyzer, ngram_range=ngram_range)
    res.fit(Series(data))

    ref = SkCountVect(analyzer=analyzer, ngram_range=ngram_range).fit(data)

    assert_array_equal(
        res.get_feature_names_out(),
        ref.get_feature_names_out(),
    )


@pytest.mark.parametrize(
    "query",
    [
        Series(["science aa", "", "a aa aaa"]),
        Series(["science aa", ""]),
        Series(["science"]),
    ],
)
def test_transform_unsigned_categories(query):
    token = "a"
    thousand_tokens = list()
    for i in range(1000):
        thousand_tokens.append(token)
        token += "a"
    thousand_tokens[128] = "science"

    vec = CountVectorizer().fit(Series(thousand_tokens))
    res = vec.transform(query)

    assert res.shape[0] == len(query)


# ----------------------------------------------------------------
# TfidfVectorizer tests are already covered by CountVectorizer and
# TfidfTransformer so we only do the bare minimum tests here
# ----------------------------------------------------------------


def test_tfidf_vectorizer_setters():
    tv = TfidfVectorizer(
        norm="l2", use_idf=False, smooth_idf=False, sublinear_tf=False
    )
    tv.norm = "l1"
    assert tv._tfidf.norm == "l1"
    tv.use_idf = True
    assert tv._tfidf.use_idf
    tv.smooth_idf = True
    assert tv._tfidf.smooth_idf
    tv.sublinear_tf = True
    assert tv._tfidf.sublinear_tf


def test_tfidf_vectorizer_idf_setter():
    orig = TfidfVectorizer(use_idf=True)
    orig.fit(DOCS_GPU)
    copy = TfidfVectorizer(vocabulary=orig.vocabulary_, use_idf=True)
    copy.idf_ = orig.idf_[0]
    cp.testing.assert_array_almost_equal(
        copy.transform(DOCS_GPU).todense(), orig.transform(DOCS_GPU).todense()
    )


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
    ).fit_transform(DOCS_GPU)

    ref = SkTfidfVect(
        norm=norm,
        use_idf=use_idf,
        smooth_idf=smooth_idf,
        sublinear_tf=sublinear_tf,
    ).fit_transform(DOCS)

    cp.testing.assert_array_almost_equal(tfidf_mat.todense(), ref.toarray())


def test_tfidf_vectorizer_get_feature_names_out():
    corpus = [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?",
    ]
    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(Series(corpus))
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
    X = Series(
        [
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
        ]
    )
    model = cls().fit(X)
    with pytest.warns(FutureWarning, match="get_feature_names"):
        res = model.get_feature_names()

    np.testing.assert_array_equal(res, model.get_feature_names_out())


# ----------------------------------------------------------------
# HashingVectorizer tests
# ----------------------------------------------------------------
def assert_almost_equal_hash_matrices(mat_1, mat_2, ignore_sign=True):
    """
    Currently if all the sorted values in the row is equal we
    assume equality
    TODO: Find better way to test ig hash matrices are equal
    """
    assert mat_1.shape == mat_2.shape
    for row_id in range(mat_1.shape[0]):
        row_m1 = mat_1[row_id]
        row_m2 = mat_2[row_id]
        nz_row_m1 = np.sort(row_m1[row_m1 != 0])
        nz_row_m2 = np.sort(row_m2[row_m2 != 0])
        # print(nz_row_m1)
        # print(nz_row_m2)
        if ignore_sign:
            nz_row_m1 = np.abs(nz_row_m1)
            nz_row_m2 = np.abs(nz_row_m2)
        nz_row_m1.sort()
        nz_row_m2.sort()
        np.testing.assert_almost_equal(nz_row_m1, nz_row_m2)


def test_hashingvectorizer():
    corpus = [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?",
    ]

    res = HashingVectorizer().fit_transform(Series(corpus))
    ref = SkHashVect().fit_transform(corpus)
    assert_almost_equal_hash_matrices(res.todense().get(), ref.toarray())


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
    res = CountVectorizer(preprocessor=lambda s: s).fit_transform(
        Series(corpus)
    )
    ref = SkCountVect(
        preprocessor=lambda s: s, tokenizer=lambda s: s.split(" ")
    ).fit_transform(corpus)
    cp.testing.assert_array_equal(res.todense(), ref.toarray())

    res = HashingVectorizer(preprocessor=lambda s: s).fit_transform(
        Series(corpus)
    )
    ref = SkHashVect(
        preprocessor=lambda s: s, tokenizer=lambda s: s.split(" ")
    ).fit_transform(corpus)
    assert_almost_equal_hash_matrices(res.todense().get(), ref.toarray())


@pytest.mark.parametrize("lowercase", [False, True])
def test_hashingvectorizer_lowercase(lowercase):
    corpus = [
        "This Is DoC",
        "this DoC is the second DoC.",
        "And this document is the third one.",
        "and Is this the first document?",
    ]
    res = HashingVectorizer(lowercase=lowercase).fit_transform(Series(corpus))
    ref = SkHashVect(lowercase=lowercase).fit_transform(corpus)
    assert_almost_equal_hash_matrices(res.todense().get(), ref.toarray())


def test_hashingvectorizer_stop_word():
    ref = SkHashVect(stop_words="english").fit_transform(DOCS)
    res = HashingVectorizer(stop_words="english").fit_transform(DOCS_GPU)
    assert_almost_equal_hash_matrices(res.todense().get(), ref.toarray())


def test_hashingvectorizer_n_features():
    n_features = 10
    res = (
        HashingVectorizer(n_features=n_features)
        .fit_transform(DOCS_GPU)
        .todense()
        .get()
    )
    ref = SkHashVect(n_features=n_features).fit_transform(DOCS).toarray()
    assert res.shape == ref.shape


@pytest.mark.parametrize("norm", ["l1", "l2", None, "max"])
def test_hashingvectorizer_norm(norm):
    if norm not in ["l1", "l2", None]:
        with pytest.raises(ValueError):
            res = HashingVectorizer(norm=norm).fit_transform(DOCS_GPU)
    else:
        res = HashingVectorizer(norm=norm).fit_transform(DOCS_GPU)
        ref = SkHashVect(norm=norm).fit_transform(DOCS)
        assert_almost_equal_hash_matrices(res.todense().get(), ref.toarray())


@pytest.mark.xfail(reason="https://github.com/NVIDIA/cuml/issues/4721")
def test_hashingvectorizer_alternate_sign():
    # if alternate_sign = True
    # we should have some negative and positive values
    res = HashingVectorizer(alternate_sign=True).fit_transform(DOCS_GPU)
    res_f_array = res.todense().get().flatten()
    assert np.sum(res_f_array > 0, axis=0) > 0
    assert np.sum(res_f_array < 0, axis=0) > 0

    # if alternate_sign = False
    # we should have no negative values and some positive values
    res = HashingVectorizer(alternate_sign=False).fit_transform(DOCS_GPU)
    res_f_array = res.todense().get().flatten()
    assert np.sum(res_f_array > 0, axis=0) > 0
    assert np.sum(res_f_array < 0, axis=0) == 0


@pytest.mark.parametrize("dtype", [np.float32, np.float64, cp.float64])
def test_hashingvectorizer_dtype(dtype):
    res = HashingVectorizer(dtype=dtype).fit_transform(DOCS_GPU)
    assert res.dtype == dtype


def test_hashingvectorizer_delimiter():
    corpus = ["a0b0c", "a 0 b0e", "c0d0f"]
    res = HashingVectorizer(
        delimiter="0", norm=None, preprocessor=lambda s: s
    ).fit_transform(Series(corpus))
    # equivalent logic for sklearn
    ref = SkHashVect(
        tokenizer=lambda s: s.split("0"),
        norm=None,
        token_pattern=None,
        preprocessor=lambda s: s,
    ).fit_transform(corpus)
    assert_almost_equal_hash_matrices(res.todense().get(), ref.toarray())


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
    res = cuml_vec().fit_transform(raw_documents)
    ref = sklearn_vec().fit_transform(raw_documents)
    assert_almost_equal_hash_matrices(res.todense().get(), ref.toarray())
