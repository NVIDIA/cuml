# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cudf
import cupy as cp
import numpy as np
import pandas
import pytest

from cuml.preprocessing._target_encoder import TargetEncoder
from cuml.testing.utils import array_equal

# TODO: many of these tests use `output_type="numpy"` to work around
# https://github.com/NVIDIA/cuml/issues/7893. These can be
# reverted once that's resolved.


def test_targetencoder_fit_transform():
    train = cudf.DataFrame({"category": ["a", "b", "b", "a"]})
    label = cudf.Series([1, 0, 1, 1])
    encoder = TargetEncoder(output_type="numpy")
    train_encoded = encoder.fit_transform(train, label)
    answer = np.array([1.0, 1.0, 0.0, 1.0])[:, None]
    assert array_equal(train_encoded, answer)

    encoder = TargetEncoder(output_type="numpy")
    encoder.fit(train, label)
    train_encoded = encoder.transform(train)

    assert array_equal(train_encoded, answer)


def test_targetencoder_transform():
    train = cudf.DataFrame({"category": ["a", "b", "b", "a"]})
    label = cudf.Series([1, 0, 1, 1])
    test = cudf.DataFrame({"category": ["b", "b", "a", "b"]})
    encoder = TargetEncoder(output_type="numpy")
    encoder.fit_transform(train, label)
    test_encoded = encoder.transform(test)
    answer = np.array([0.5, 0.5, 1.0, 0.5])[:, None]
    assert array_equal(test_encoded, answer)

    encoder = TargetEncoder(output_type="numpy")
    encoder.fit(train, label)
    test_encoded = encoder.transform(test)
    assert array_equal(test_encoded, answer)


@pytest.mark.parametrize("n_samples", [5000, 500000])
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("stat", ["mean", "var", "median"])
def test_targetencoder_random(n_samples, dtype, stat):
    x = cp.random.randint(0, 1000, n_samples).astype(dtype)
    y = cp.random.randint(0, 2, n_samples).astype(dtype)
    xt = cp.random.randint(0, 1000, n_samples).astype(dtype)

    encoder = TargetEncoder(stat=stat)
    encoder.fit_transform(x[:, None], y)
    test_encoded = encoder.transform(xt[:, None])

    df_train = cudf.DataFrame({"x": x, "y": y})
    dg = df_train.groupby("x", as_index=False).agg({"y": stat})
    df_test = cudf.DataFrame({"x": xt})
    df_test["row_id"] = cp.arange(len(df_test))
    df_test = df_test.merge(dg, on="x", how="left")
    df_test = df_test.sort_values("row_id")
    answer = df_test["y"].fillna(getattr(cp, stat)(y).item()).values[:, None]
    assert array_equal(test_encoded, answer)


def test_targetencoder_multi_column():
    """
    Test jointly encoding multiple columns
    """
    train = cudf.DataFrame(
        {
            "cat_1": ["a", "b", "b", "a", "a", "b"],
            "cat_2": [1, 1, 2, 2, 1, 2],
            "label": [1, 0, 1, 1, 0, 1],
        }
    )
    test = cudf.DataFrame(
        {"cat_1": ["b", "b", "a", "b"], "cat_2": [1, 2, 1, 2]}
    )
    encoder = TargetEncoder(output_type="numpy")
    train_encoded = encoder.fit_transform(
        train[["cat_1", "cat_2"]], train.label
    )
    test_encoded = encoder.transform(test[["cat_1", "cat_2"]])
    train_answer = np.array([2.0 / 3, 2.0 / 3, 1.0, 2.0 / 3, 2.0 / 3, 1.0])[
        :, None
    ]
    test_answer = np.array([0.0, 1.0, 0.5, 1.0])[:, None]
    assert array_equal(train_encoded, train_answer)
    assert array_equal(test_encoded, test_answer)

    encoder = TargetEncoder(output_type="numpy")
    encoder.fit(train[["cat_1", "cat_2"]], train.label)
    train_encoded = encoder.transform(train[["cat_1", "cat_2"]])
    test_encoded = encoder.transform(test[["cat_1", "cat_2"]])
    assert array_equal(train_encoded, train_answer)
    assert array_equal(test_encoded, test_answer)


def test_targetencoder_newly_encountered():
    """
    Note that there are newly-encountered values in test,
    namely, 'c' and 'd'.
    """
    train = cudf.DataFrame({"category": ["a", "b", "b", "a"]})
    label = cudf.Series([1, 0, 1, 1])
    test = cudf.DataFrame({"category": ["c", "b", "a", "d"]})
    encoder = TargetEncoder(output_type="numpy")
    encoder.fit_transform(train, label)
    test_encoded = encoder.transform(test)
    answer = np.array([0.75, 0.5, 1.0, 0.75])[:, None]
    assert array_equal(test_encoded, answer)

    encoder = TargetEncoder(output_type="numpy")
    encoder.fit(train, label)
    test_encoded = encoder.transform(test)
    assert array_equal(test_encoded, answer)


def test_one_category():
    train = cudf.DataFrame({"category": ["a", "a", "a", "a"]})
    label = cudf.Series([3, 0, 0, 3])
    test = cudf.DataFrame({"category": ["c", "b", "a", "d"]})

    encoder = TargetEncoder(output_type="numpy")
    train_encoded = encoder.fit_transform(train, label)
    answer = np.array([1 / 3, 2 / 3, 2 / 3, 1 / 3])[:, None]
    assert array_equal(train_encoded, answer)

    test_encoded = encoder.transform(test)
    answer = np.array([0.5, 0.5, 0.5, 0.5])[:, None]
    assert array_equal(test_encoded, answer)


def test_targetencoder_pandas():
    """
    Note that there are newly-encountered values in test,
    namely, 'c' and 'd'.
    """
    train = pandas.DataFrame({"category": ["a", "b", "b", "a"]})
    label = pandas.Series([1, 0, 1, 1])
    test = pandas.DataFrame({"category": ["c", "b", "a", "d"]})
    encoder = TargetEncoder(output_type="numpy")
    encoder.fit_transform(train, label)
    test_encoded = encoder.transform(test)
    answer = np.array([0.75, 0.5, 1.0, 0.75])[:, None]
    assert array_equal(test_encoded, answer)


def test_targetencoder_numpy():
    """
    Note that there are newly-encountered values in x_test,
    namely, 3 and 4.
    """
    x_train = np.array([1, 2, 2, 1])[:, None]
    y_train = np.array([1, 0, 1, 1])
    x_test = np.array([1, 2, 3, 4])[:, None]
    encoder = TargetEncoder()
    encoder.fit_transform(x_train, y_train)
    test_encoded = encoder.transform(x_test)
    answer = np.array([1.0, 0.5, 0.75, 0.75])[:, None]
    assert array_equal(test_encoded, answer)
    assert isinstance(test_encoded, np.ndarray)


def test_targetencoder_cupy():
    """
    Note that there are newly-encountered values in x_test,
    namely, 3 and 4.
    """
    x_train = cp.array([1, 2, 2, 1])[:, None]
    y_train = cp.array([1, 0, 1, 1])
    x_test = cp.array([1, 2, 3, 4])[:, None]
    encoder = TargetEncoder()
    encoder.fit_transform(x_train, y_train)
    test_encoded = encoder.transform(x_test)
    answer = np.array([1.0, 0.5, 0.75, 0.75])[:, None]
    assert array_equal(test_encoded, answer)
    assert isinstance(test_encoded, cp.ndarray)


def test_targetencoder_smooth():
    train = cudf.DataFrame({"category": ["a", "b", "b", "a"]})
    label = cudf.Series([1, 0, 1, 1])
    answers = np.array(
        [
            [1.0, 1.0, 0.0, 1.0],
            [0.875, 0.875, 0.375, 0.875],
            [0.8333, 0.8333, 0.5, 0.8333],
            [0.75, 0.75, 0.75, 0.75],
        ]
    )
    smooths = [0, 1, 2, 10000]
    for smooth, answer in zip(smooths, answers):
        answer = np.array(answer)[:, None]
        encoder = TargetEncoder(smooth=smooth)
        train_encoded = encoder.fit_transform(train, label)
        assert array_equal(train_encoded, answer)

        encoder = TargetEncoder(smooth=smooth)
        encoder.fit(train, label)
        train_encoded = encoder.transform(train)

        assert array_equal(train_encoded, answer)


def test_targetencoder_customized_fold_id():
    """
    use customized `fold_ids` array to split data.
    in this example, the 1st sample belongs to `fold 0`
    the 2nd and 3rd sample belongs to `fold 1`
    and the 4th sample belongs to `fold 2`
    """
    train = cudf.DataFrame({"category": ["a", "b", "b", "a"]})
    label = cudf.Series([1, 0, 1, 1])
    fold_ids = [0, 1, 1, 2]
    encoder = TargetEncoder(split_method="customize", output_type="numpy")
    train_encoded = encoder.fit_transform(train, label, fold_ids=fold_ids)
    answer = np.array([1.0, 0.75, 0.75, 1.0])[:, None]
    assert array_equal(train_encoded, answer)

    encoder = TargetEncoder(split_method="customize", output_type="numpy")
    encoder.fit(train, label, fold_ids=fold_ids)
    train_encoded = encoder.transform(train)

    assert array_equal(train_encoded, answer)


def test_targetencoder_var():
    train = cudf.DataFrame({"category": ["a", "b", "b", "b"]})
    label = cudf.Series([1, 0, 1, 1])
    encoder = TargetEncoder(stat="var", output_type="numpy")
    train_encoded = encoder.fit_transform(train, label)
    answer = np.array([0.25, 0.0, 0.5, 0.5])[:, None]
    assert array_equal(train_encoded, answer)

    encoder = TargetEncoder(stat="var", output_type="numpy")
    encoder.fit(train, label)
    train_encoded = encoder.transform(train)

    assert array_equal(train_encoded, answer)


def test_transform_with_index():
    df = cudf.DataFrame(
        {"a": [1, 1, 2, 3], "b": [True, False, False, True]},
        index=[9, 4, 5, 3],
    )
    X = df[["a"]]
    y = df["b"]

    t_enc = TargetEncoder(output_type="numpy")

    t_enc.fit(X, y)
    train_encoded = t_enc.transform(X)
    ans = cp.asarray([0, 1, 0.5, 0.5])[:, None]
    assert array_equal(train_encoded, ans)


def test_get_params():
    params = {
        "n_folds": 5,
        "smooth": 1,
        "seed": 49,
        "split_method": "customize",
    }
    encoder = TargetEncoder(**params)
    p2 = encoder.get_params()
    for k, v in params.items():
        assert v == p2[k]


def test_targetencoder_median():
    train = cudf.DataFrame(
        {"category": ["a", "a", "a", "a", "b", "b", "b", "b"]}
    )
    label = cudf.Series([1.5, 22.0, 15.0, 17.0, 70.0, 9.0, 99.0, 56.0])
    encoder = TargetEncoder(stat="median", output_type="numpy")
    train_encoded = encoder.fit_transform(train, label)
    answer = np.array([17.0, 15.0, 17.0, 15.0, 56.0, 70.0, 56.0, 70.0])[
        :, None
    ]
    assert array_equal(train_encoded, answer)

    encoder = TargetEncoder(stat="median", output_type="numpy")
    encoder.fit(train, label)
    train_encoded = encoder.transform(train)

    assert array_equal(train_encoded, answer)


def test_target_encoder_target_type_and_classes():
    X = np.array([[0, 1], [1, 2], [2, 3]])

    # String labels
    y = np.array(["a", "b", "a"], dtype="object")
    enc = TargetEncoder().fit(X, y)
    assert enc.target_type_ == "binary"
    np.testing.assert_array_equal(
        enc.classes_, np.array(["a", "b"], dtype="object")
    )

    # Integral labels
    y = np.array([20, 10, 20])
    enc = TargetEncoder().fit(X, y)
    assert enc.target_type_ == "binary"
    np.testing.assert_array_equal(enc.classes_, np.array([10, 20]))

    # Floating labels
    y = np.array([20, 10, 20], dtype="float64")
    enc = TargetEncoder().fit(X, y)
    assert enc.target_type_ == "binary"
    np.testing.assert_array_equal(enc.classes_, np.array([10.0, 20.0]))

    # Continuous
    y = np.array([1.5, 2, 3])
    enc = TargetEncoder().fit(X, y)
    assert enc.target_type_ == "continuous"
    assert enc.classes_ is None


def test_get_feature_names_out():
    X = np.array([[0, 1], [1, 2], [2, 3]])
    y = np.array(["a", "b", "a"], dtype="object")

    model = TargetEncoder(multi_feature_mode="combination").fit(X, y)
    res = model.get_feature_names_out()
    sol = np.array(["targetencoder0"], dtype=object)
    np.testing.assert_array_equal(res, sol)

    model = TargetEncoder(multi_feature_mode="independent").fit(X, y)
    res = model.get_feature_names_out(["a", "b"])
    sol = np.array(["a", "b"], dtype=object)
    np.testing.assert_array_equal(res, sol)
