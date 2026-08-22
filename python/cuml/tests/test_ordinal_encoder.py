# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pandas as pd
import pytest
import sklearn.preprocessing

from cuml.preprocessing import OrdinalEncoder


@pytest.mark.parametrize("kind", ["array", "dataframe"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_ordinal_encoder(kind, dtype):
    X = np.array(
        [
            [2, 2, 2, 2],
            [1, 2, 1, 2],
            [3, 2, 1, 1],
        ]
    ).T
    if kind == "dataframe":
        X = pd.DataFrame(X, columns=["a", "b", "c"])

    sk_enc = sklearn.preprocessing.OrdinalEncoder(dtype=dtype).fit(X)
    cu_enc = OrdinalEncoder(output_type="numpy", dtype=dtype).fit(X)

    # Check fitted attributes
    assert len(cu_enc.categories_) == len(sk_enc.categories_)
    for res, sol in zip(cu_enc.categories_, sk_enc.categories_):
        np.testing.assert_array_equal(res, sol)

    # Check transform
    res = cu_enc.transform(X)
    Xt = sol = sk_enc.transform(X)
    assert res.dtype == sol.dtype
    np.testing.assert_array_equal(res, sol)

    # Check inverse_transform
    res = pd.DataFrame(cu_enc.inverse_transform(Xt))
    sol = pd.DataFrame(sk_enc.inverse_transform(Xt))
    pd.testing.assert_frame_equal(res, sol)


@pytest.mark.parametrize("kind", ["numpy", "pandas"])
def test_ordinal_encoder_fit_transform(kind):
    X = np.array(
        [
            [2, 2, 2, 2],
            [1, 2, 1, 2],
            [3, 2, 1, 1],
        ]
    ).T
    if kind == "pandas":
        X = pd.DataFrame(X, columns=["a", "b", "c"])
    enc1 = OrdinalEncoder().fit(X)
    Xt1 = enc1.transform(X)
    enc2 = OrdinalEncoder()
    Xt2 = enc2.fit_transform(X)
    assert enc1._input_type == kind
    assert enc2._input_type == kind
    if kind == "pandas":
        pd.testing.assert_frame_equal(Xt1, Xt2)
    else:
        np.testing.assert_array_equal(Xt1, Xt2)


def test_ordinal_encoder_all_dtypes():
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
    cu_enc = OrdinalEncoder(output_type="numpy")
    sk_enc = sklearn.preprocessing.OrdinalEncoder()
    cu_enc.fit(X)
    sk_enc.fit(X)

    # Check fitted attributes
    assert len(cu_enc.categories_) == len(sk_enc.categories_)
    for res, sol in zip(cu_enc.categories_, sk_enc.categories_):
        assert res.dtype == sol.dtype
        pd.testing.assert_series_equal(pd.Series(res), pd.Series(sol))

    # Check transform
    res = cu_enc.transform(X)
    Xt = sol = sk_enc.transform(X)
    np.testing.assert_array_equal(res, sol)

    # Check inverse_transform
    res = pd.DataFrame(cu_enc.inverse_transform(Xt))
    sol = pd.DataFrame(sk_enc.inverse_transform(Xt))
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
def test_ordinal_encoder_cardinalities(cardinalities):
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

    cu_enc = OrdinalEncoder().fit(X)
    sk_enc = sklearn.preprocessing.OrdinalEncoder().fit(X)

    # Check fitted attributes
    assert len(cu_enc.categories_) == len(sk_enc.categories_)
    for res, sol in zip(cu_enc.categories_, sk_enc.categories_):
        np.testing.assert_array_equal(res, sol)

    # Check transform
    res = cu_enc.transform(X)
    Xt = sol = sk_enc.transform(X)
    np.testing.assert_array_equal(res, sol)

    # Check inverse_transform
    res = pd.DataFrame(cu_enc.inverse_transform(Xt))
    sol = pd.DataFrame(sk_enc.inverse_transform(Xt))
    pd.testing.assert_frame_equal(res, sol)


def test_ordinal_encoder_invalid_parameters():
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
        OrdinalEncoder(handle_unknown="bad").fit(X)

    with pytest.raises(
        ValueError,
        match=(
            "When handle_unknown='ignore', the dtype parameter should be a "
            "float dtype. Got int32."
        ),
    ):
        OrdinalEncoder(dtype="int32", handle_unknown="ignore").fit(X)

    # Invalid `categories` errors
    with pytest.raises(ValueError, match="Expected `categories` .* got 'bad'"):
        OrdinalEncoder(categories="bad").fit(X)

    with pytest.raises(ValueError, match="Shape mismatch"):
        OrdinalEncoder(categories=[[2], [1, 2]]).fit(X)

    with pytest.raises(ValueError, match="Nan should be the last element"):
        OrdinalEncoder(categories=[[1, 2], [1, 2, 3], [float("nan"), 2]]).fit(
            X
        )

    with pytest.raises(ValueError, match="In column 1, .* duplicate elements"):
        OrdinalEncoder(
            categories=[[1, 2], [1, 2, 3, 3], [2, float("nan")]]
        ).fit(X)


def test_ordinal_encoder_int_dtype():
    X1 = pd.DataFrame({"x": [1, 2, np.nan]})
    X2 = pd.DataFrame({"x": [1, 2, 1], "y": [1, 3, 2]})

    with pytest.raises(
        ValueError,
        match="There are missing values in features \\[0\\].",
    ):
        OrdinalEncoder(dtype="int32").fit(X1)

    res = OrdinalEncoder(dtype="int32").fit_transform(X2)
    np.testing.assert_array_equal(
        res, np.array([[0, 0], [1, 2], [0, 1]], dtype="int32")
    )


def test_ordinal_encoder_unknown_categories_in_fit():
    X = np.array([[1, 2, float("nan"), 2]]).T

    with pytest.raises(ValueError, match="Found unknown categories \\[nan\\]"):
        OrdinalEncoder(categories=[[1, 2]]).fit(X)

    with pytest.raises(
        ValueError, match="Found unknown categories \\[1.0, 2.0\\]"
    ):
        OrdinalEncoder(categories=[[float("nan")]]).fit(X)

    enc = OrdinalEncoder(categories=[[1, 2, float("nan")]]).fit(X)
    np.testing.assert_array_equal(enc.categories_[0], [1, 2, float("nan")])


def test_ordinal_encoder_transform_missing():
    X1 = pd.DataFrame({"x": [np.nan, "b", "a"], "y": [1, 2, np.nan]})
    X2 = pd.DataFrame({"x": ["b", np.nan], "y": [np.nan, 1]})

    cu_enc = OrdinalEncoder().fit(X1)
    sk_enc = sklearn.preprocessing.OrdinalEncoder().fit(X1)

    res = cu_enc.transform(X2)
    sol = sk_enc.transform(X2)
    np.testing.assert_array_equal(res.to_numpy(), sol)


def test_ordinal_encoder_transform_missing_unknown():
    """Check error raised if unknown category is NaN"""
    X1 = pd.DataFrame({"x": ["a", "b", "a"], "y": [1, 2, 1]})
    X2 = pd.DataFrame({"x": ["b", None], "y": [2, 1]})
    X3 = pd.DataFrame({"x": ["b", "a"], "y": [2, np.nan]})

    enc = OrdinalEncoder().fit(X1)

    with pytest.raises(
        ValueError,
        match="Found unknown categories \\[nan\\] in column 0 during transform",
    ):
        enc.transform(X2)
    with pytest.raises(
        ValueError,
        match="Found unknown categories \\[nan\\] in column 1 during transform",
    ):
        enc.transform(X3)


def test_ordinal_encoder_transform_unknown():
    X1 = pd.DataFrame({"x": ["a", "b", "a"]})
    X2 = pd.DataFrame({"x": ["b", "c"]})

    enc = OrdinalEncoder().fit(X1)

    # Unknown value errors by default
    with pytest.raises(
        ValueError,
        match=".* categories \\['c'\\] in column 0 during transform",
    ):
        enc.transform(X2)

    # Passing `handle_unknown="ignore"` fixes things
    cu_enc = OrdinalEncoder(
        output_type="numpy",
        handle_unknown="ignore",
    ).fit(X1)
    sk_enc = sklearn.preprocessing.OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
    ).fit(X1)
    res = cu_enc.transform(X2)
    sol = sk_enc.transform(X2)
    np.testing.assert_array_equal(res, sol)

    # Explicitly passing categories also fixes things
    kwargs = {"categories": [["a", "b", "c"]]}
    cu_enc = OrdinalEncoder(output_type="numpy", **kwargs).fit(X1)
    sk_enc = sklearn.preprocessing.OrdinalEncoder(**kwargs).fit(X1)
    res = cu_enc.transform(X2)
    sol = sk_enc.transform(X2)
    np.testing.assert_array_equal(res, sol)


def test_ordinal_encoder_inverse_transform():
    Xt = pd.DataFrame({"x": [np.nan, 0], "y": [0, 1], "z": [0, np.nan]})

    # No unknown elements, fully invertible
    X = pd.DataFrame(
        {"x": ["a", "b", None], "y": [1, 3, 2], "z": [1, 1, np.nan]}
    )
    enc = OrdinalEncoder().fit(X)
    res = enc.inverse_transform(Xt)
    sol = pd.DataFrame({"x": [np.nan, "a"], "y": [1, 2], "z": [1, np.nan]})
    pd.testing.assert_frame_equal(res, sol)

    # Incorrect input dimensions errors
    with pytest.raises(ValueError, match="Shape of the passed X data"):
        enc.inverse_transform(Xt[["x", "y"]])

    # handle_unknown="ignore", fully invertible
    X = pd.DataFrame(
        {"x": ["a", "b", "b"], "y": [1, 3, 2], "z": [1, 1, np.nan]}
    )
    enc = OrdinalEncoder(handle_unknown="ignore").fit(X)
    res = enc.inverse_transform(Xt)
    sol = pd.DataFrame({"x": [np.nan, "a"], "y": [1, 2], "z": [1, np.nan]})
    pd.testing.assert_frame_equal(res, sol)

    # handle_unknown="error", errors on unknown values
    X = pd.DataFrame({"x": ["a", "b", "b"], "y": [1, 3, 2], "z": [1, 1, 1]})
    enc = OrdinalEncoder().fit(X)
    with pytest.raises(
        ValueError, match="Samples \\[0\\] can not be inverted"
    ):
        enc.inverse_transform(Xt)


def test_ordinal_encoder_get_feature_names_out():
    X = pd.DataFrame(
        {
            "fruits": ["apple", "banana", "apple"],
            "counts": [0, 1, 2],
        }
    )
    cu_model = OrdinalEncoder().fit(X)
    sk_model = sklearn.preprocessing.OrdinalEncoder().fit(X)
    res = cu_model.get_feature_names_out()
    sol = sk_model.get_feature_names_out()
    assert np.array_equal(res, sol)
