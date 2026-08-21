# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import sklearn.preprocessing

from cuml.preprocessing import OneHotEncoder


@pytest.mark.parametrize("kind", ["array", "dataframe"])
@pytest.mark.parametrize("drop", [None, "first", [2, 2, 1]])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("sparse_output", [True, False])
def test_onehot_encoder(kind, drop, dtype, sparse_output):
    X = np.array(
        [
            [2, 2, 2, 2],
            [1, 2, 1, 2],
            [3, 2, 1, 1],
        ]
    ).T
    if kind == "dataframe":
        X = pd.DataFrame(X, columns=["a", "b", "c"])

    kwargs = {
        "drop": drop,
        "dtype": dtype,
        "sparse_output": sparse_output,
    }
    sk_enc = sklearn.preprocessing.OneHotEncoder(**kwargs).fit(X)
    cu_enc = OneHotEncoder(output_type="numpy", **kwargs).fit(X)

    # Check fitted attributes
    assert len(cu_enc.categories_) == len(sk_enc.categories_)
    for res, sol in zip(cu_enc.categories_, sk_enc.categories_):
        np.testing.assert_array_equal(res, sol)

    if drop is not None:
        np.testing.assert_array_equal(cu_enc.drop_idx_, sk_enc.drop_idx_)

    # Check transform
    res = cu_enc.transform(X)
    Xt = sol = sk_enc.transform(X)
    assert res.dtype == sol.dtype
    if sparse_output:
        np.testing.assert_array_equal(res.toarray(), sol.toarray())
    else:
        np.testing.assert_array_equal(res, sol)

    # Check inverse_transform
    res = pd.DataFrame(cu_enc.inverse_transform(Xt))
    sol = pd.DataFrame(sk_enc.inverse_transform(Xt))
    pd.testing.assert_frame_equal(res, sol)


@pytest.mark.parametrize("kind", ["numpy", "pandas"])
def test_onehot_encoder_fit_transform(kind):
    X = np.array(
        [
            [2, 2, 2, 2],
            [1, 2, 1, 2],
            [3, 2, 1, 1],
        ]
    ).T
    if kind == "pandas":
        X = pd.DataFrame(X, columns=["a", "b", "c"])
    enc1 = OneHotEncoder(sparse_output=False).fit(X)
    Xt1 = enc1.transform(X)
    enc2 = OneHotEncoder(sparse_output=False)
    Xt2 = enc2.fit_transform(X)
    assert enc1._input_type == kind
    assert enc2._input_type == kind
    if kind == "pandas":
        pd.testing.assert_frame_equal(Xt1, Xt2)
    else:
        np.testing.assert_array_equal(Xt1, Xt2)


@pytest.mark.parametrize(
    "drop", [None, "first", [2, 2, float("nan"), 2, "banana", "b"]]
)
def test_onehot_encoder_all_dtypes(drop):
    X = pd.DataFrame(
        {
            "int32": pd.Series([1, 2, 1, 2, 1], dtype="int32"),
            "int64": pd.Series([1, 2, 1, 2, 1], dtype="int64"),
            "float32": pd.Series([1, 2, float("nan"), 2, 1], dtype="float32"),
            "float64": pd.Series([1, 2, float("nan"), 2, 1], dtype="float64"),
            "string": pd.Series(["apple", "banana", "carrot", "apple", None]),
            "category": pd.Series(
                ["a", "b", "a", "b", None], dtype="category"
            ),
        }
    )
    cu_enc = OneHotEncoder(drop=drop).fit(X)
    sk_enc = sklearn.preprocessing.OneHotEncoder(drop=drop).fit(X)

    # Check fitted attributes
    assert len(cu_enc.categories_) == len(sk_enc.categories_)
    for res, sol in zip(cu_enc.categories_, sk_enc.categories_):
        assert res.dtype == sol.dtype
        pd.testing.assert_series_equal(pd.Series(res), pd.Series(sol))

    if drop is not None:
        np.testing.assert_array_equal(cu_enc.drop_idx_, sk_enc.drop_idx_)

    # Check transform
    res = cu_enc.transform(X)
    Xt = sol = sk_enc.transform(X)
    np.testing.assert_array_equal(res.toarray(), sol.toarray())

    # Check inverse_transform on sparse
    res = pd.DataFrame(cu_enc.inverse_transform(Xt))
    sol = pd.DataFrame(sk_enc.inverse_transform(Xt))
    pd.testing.assert_frame_equal(res, sol)

    # Check inverse_transform on dense
    res = pd.DataFrame(cu_enc.inverse_transform(Xt.toarray()))
    sol = pd.DataFrame(sk_enc.inverse_transform(Xt.toarray()))
    pd.testing.assert_frame_equal(res, sol)


@pytest.mark.parametrize(
    "cardinalities",
    [
        (1, 2),
        (2, 1, 1, 2),
        (2, 256),
        (2, 65536),
        (256, 1, 65536),
    ],
)
@pytest.mark.parametrize("drop", [None, "first"])
def test_onehot_encoder_cardinalities(cardinalities, drop):
    """A stress test around mixed high and low cardinalities"""
    n_samples = max(cardinalities)
    X = np.empty(shape=(n_samples, len(cardinalities)), dtype="int32")
    col = np.empty(n_samples, dtype="int32")
    rng = np.random.default_rng(42)
    for i, n_cats in enumerate(cardinalities):
        # Pre-fill first n_cats to ensure 1 of each category present
        col[:n_cats] = np.arange(n_cats)
        col[n_cats:] = rng.choice(n_cats, n_samples - n_cats)
        rng.shuffle(col)
        X[:, i] = col

    cu_enc = OneHotEncoder(drop=drop).fit(X)
    sk_enc = sklearn.preprocessing.OneHotEncoder(drop=drop).fit(X)

    # Check fitted attributes
    assert len(cu_enc.categories_) == len(sk_enc.categories_)
    for res, sol in zip(cu_enc.categories_, sk_enc.categories_):
        np.testing.assert_array_equal(res, sol)

    if drop is not None:
        np.testing.assert_array_equal(cu_enc.drop_idx_, sk_enc.drop_idx_)

    # Check transform
    res = cu_enc.transform(X)
    Xt = sol = sk_enc.transform(X)
    # efficient equality check for sparse data
    assert (res != sol).count_nonzero() == 0

    # Check inverse_transform
    res = pd.DataFrame(cu_enc.inverse_transform(Xt))
    pd.testing.assert_frame_equal(res, pd.DataFrame(X))


def test_onehot_encoder_invalid_parameters():
    X = pd.DataFrame(
        {
            "x": [1.0, 2.0, 1.0, 2.0],
            "y": [1.0, 2.0, 3.0, 1.0],
            "z": [2.0, 2.0, float("nan"), 2.0],
        }
    )

    # Invalid `handle_unknown` errors
    with pytest.raises(
        ValueError, match="Expected `handle_unknown` .* got 'bad'"
    ):
        OneHotEncoder(handle_unknown="bad").fit(X)

    # Invalid `drop` errors
    with pytest.raises(ValueError, match="Expected `drop` .* got 'bad'"):
        OneHotEncoder(drop="bad").fit(X)

    with pytest.raises(
        ValueError, match="`drop` should have length .* \\(3\\), got 2"
    ):
        OneHotEncoder(drop=[2, 2]).fit(X)

    with pytest.raises(ValueError, match="The following categories") as rec:
        OneHotEncoder(drop=[10, 1, 9]).fit(X)
    assert "Category: 0, Feature: 10" in str(rec.value)
    assert "Category: 2, Feature: 9" in str(rec.value)

    # Invalid `categories` errors
    with pytest.raises(ValueError, match="Expected `categories` .* got 'bad'"):
        OneHotEncoder(categories="bad").fit(X)

    with pytest.raises(ValueError, match="Shape mismatch"):
        OneHotEncoder(categories=[[2], [1, 2]]).fit(X)

    with pytest.raises(ValueError, match="Nan should be the last element"):
        OneHotEncoder(categories=[[1, 2], [1, 2, 3], [float("nan"), 2]]).fit(X)

    X2 = pd.DataFrame(
        {
            "x": [1.0, 2.0, 1.0, 2.0],
            "y": ["a", None, "b", None],
        }
    )
    with pytest.raises(ValueError, match="Nan should be the last element"):
        OneHotEncoder(categories=[[1, 2], ["a", float("nan"), "b"]]).fit(X2)
    with pytest.raises(ValueError, match="Nan should be the last element"):
        OneHotEncoder(categories=[[1, 2], ["a", float("nan"), "b"]]).fit(X2)

    with pytest.raises(ValueError, match="In column 1, .* duplicate elements"):
        OneHotEncoder(
            categories=[[1, 2], [1, 2, 3, 3], [2, float("nan")]]
        ).fit(X)


def test_onehot_encoder_unknown_categories_in_fit():
    X = np.array([[1, 2, float("nan"), 2]]).T

    with pytest.raises(ValueError, match="Found unknown categories \\[nan\\]"):
        OneHotEncoder(categories=[[1, 2]]).fit(X)

    with pytest.raises(
        ValueError, match="Found unknown categories \\[1.0, 2.0\\]"
    ):
        OneHotEncoder(categories=[[float("nan")]]).fit(X)

    enc = OneHotEncoder(categories=[[1, 2, float("nan")]]).fit(X)
    np.testing.assert_array_equal(enc.categories_[0], [1, 2, float("nan")])


@pytest.mark.parametrize("unknown_val", ["c", float("nan")])
def test_onehot_encoder_transform_unknown(unknown_val):
    X1 = pd.DataFrame({"x": ["a", "b", "a"]})
    X2 = pd.DataFrame({"x": ["b", unknown_val]})

    enc = OneHotEncoder().fit(X1)

    # Unknown value errors by default
    with pytest.raises(
        ValueError,
        match=f".* categories \\[{unknown_val!r}\\] in column 0 during transform",
    ):
        enc.transform(X2)

    # Passing `handle_unknown="ignore"` fixes things
    kwargs = {"handle_unknown": "ignore"}
    cu_enc = OneHotEncoder(**kwargs).fit(X1)
    sk_enc = sklearn.preprocessing.OneHotEncoder(**kwargs).fit(X1)
    res = cu_enc.transform(X2)
    sol = sk_enc.transform(X2)
    np.testing.assert_array_equal(res.toarray(), sol.toarray())

    # Explicitly passing categories also fixes things
    kwargs = {"categories": [["a", "b", unknown_val]]}
    cu_enc = OneHotEncoder(**kwargs).fit(X1)
    sk_enc = sklearn.preprocessing.OneHotEncoder(**kwargs).fit(X1)
    res = cu_enc.transform(X2)
    sol = sk_enc.transform(X2)
    np.testing.assert_array_equal(res.toarray(), sol.toarray())


@pytest.mark.parametrize("drop", [None, "first", ["b", 3, 1]])
@pytest.mark.parametrize("handle_unknown", ["error", "ignore"])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("unknown", [False, True])
def test_onehot_encoder_inverse_transform(
    drop, handle_unknown, sparse, unknown
):
    X = pd.DataFrame({"x": ["a", "b", "b"], "y": [1, 3, 2], "z": [1, 1, 1]})

    kwargs = {"handle_unknown": handle_unknown, "drop": drop}
    cu_enc = OneHotEncoder(**kwargs).fit(X)
    sk_enc = sklearn.preprocessing.OneHotEncoder(**kwargs).fit(X)

    if drop is None:
        Xt = np.array(
            [
                [0, 1, 1, 0, 0, 1],
                [1, 0, 0, 0, 1, 1],
                [1, 0, 0, 1, 0, 1],
            ]
        )
    else:
        Xt = np.array(
            [
                [0, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
            ]
        )

    if unknown:
        Xt[1, 0] = 0

    if sparse:
        Xt = sp.csr_matrix(Xt)

    if handle_unknown == "error" and unknown and drop is None:
        with pytest.raises(ValueError, match="Samples .* can not be inverted"):
            cu_enc.inverse_transform(Xt)
    else:
        res = pd.DataFrame(cu_enc.inverse_transform(Xt))
        sol = pd.DataFrame(sk_enc.inverse_transform(Xt))
        pd.testing.assert_frame_equal(res, sol)


@pytest.mark.parametrize("drop", [None, "first"])
def test_onehot_encoder_inverse_transform_errors(drop):
    X = np.array([[1, 2, 1], [3, 1, 2]]).T

    enc = OneHotEncoder(drop=drop)
    Xt = enc.fit_transform(X)
    with pytest.raises(ValueError, match="Shape of the passed X data"):
        enc.inverse_transform(Xt[:, :-1])


@pytest.mark.parametrize("named", [True, False])
@pytest.mark.parametrize("drop", [None, "first"])
def test_onehot_encoder_get_feature_names_out(named, drop):
    columns = [["apple", "banana", "strawberry"], [0, 1, 2]]
    names = ["fruits", "sizes"] if named else [0, 1]
    X = pd.DataFrame(dict(zip(names, columns)))

    cu_model = OneHotEncoder(drop=drop).fit(X)
    sk_model = sklearn.preprocessing.OneHotEncoder(drop=drop).fit(X)

    res = cu_model.get_feature_names_out()
    sol = sk_model.get_feature_names_out()
    assert np.array_equal(res, sol)

    if not named:
        res = cu_model.get_feature_names_out(["fruit", "size"])
        sol = sk_model.get_feature_names_out(["fruit", "size"])
        assert np.array_equal(res, sol)


def test_onehot_encoder_get_feature_names_deprecated():
    X = pd.DataFrame(
        {"fruits": ["apple", "banana", "strawberry"], "sizes": [0, 1, 2]}
    )
    model = OneHotEncoder().fit(X)
    with pytest.warns(FutureWarning, match="get_feature_names"):
        res = model.get_feature_names()

    np.testing.assert_array_equal(res, model.get_feature_names_out())
