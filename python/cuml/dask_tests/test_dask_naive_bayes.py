#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
import dask.array
import pytest
from sklearn.metrics import accuracy_score

from cuml.dask.naive_bayes import MultinomialNB
from cuml.naive_bayes.naive_bayes import MultinomialNB as SGNB
from cuml.testing.dask.utils import load_text_corpus


def test_basic_fit_predict(client):
    X, y = load_text_corpus(client)

    model = MultinomialNB()

    model.fit(X, y)

    y_hat = model.predict(X)

    y_hat = y_hat.compute()
    y = y.compute()

    assert accuracy_score(y_hat.get(), y) > 0.97


def test_single_distributed_exact_results(client):
    X, y = load_text_corpus(client)

    sgX, sgy = (X.compute(), y.compute())

    model = MultinomialNB()
    model.fit(X, y)

    sg_model = SGNB()
    sg_model.fit(sgX, sgy)

    y_hat = model.predict(X)
    sg_y_hat = sg_model.predict(sgX).get()

    y_hat = y_hat.compute().get()

    assert accuracy_score(y_hat, sg_y_hat) == 1.0


def test_score(client):
    X, y = load_text_corpus(client)

    model = MultinomialNB()
    model.fit(X, y)

    y_hat = model.predict(X)

    score = model.score(X, y)

    y_hat_local = y_hat.compute()
    y_local = y.compute()

    assert accuracy_score(y_hat_local.get(), y_local) == score


@pytest.mark.parametrize("dtype", [cp.float32, cp.float64, cp.int32])
def test_model_multiple_chunks(client, dtype):
    # tests naive_bayes with n_chunks being greater than one, related to issue
    # https://github.com/NVIDIA/cuml/issues/3150
    X = cp.array([[0, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 0]])

    X = dask.array.from_array(X, chunks=((1, 1, 1), -1)).astype(dtype)
    y = dask.array.from_array(
        [1, 0, 0], asarray=False, fancy=False, chunks=(1)
    ).astype(cp.int32)

    model = MultinomialNB()
    model.fit(X, y)

    # this test is a code coverage test, it is too small to be a numeric test,
    # but we call score here to exercise the whole model.
    assert 0 <= model.score(X, y) <= 1
