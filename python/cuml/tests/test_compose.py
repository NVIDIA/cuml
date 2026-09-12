# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#


import cudf
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone as sk_clone
from sklearn.compose import ColumnTransformer as skColumnTransformer
from sklearn.compose import make_column_selector as sk_make_column_selector
from sklearn.compose import (
    make_column_transformer as sk_make_column_transformer,
)
from sklearn.impute import SimpleImputer as skSimpleImputer
from sklearn.preprocessing import Normalizer as skNormalizer
from sklearn.preprocessing import OneHotEncoder as skOneHotEncoder
from sklearn.preprocessing import PolynomialFeatures as skPolynomialFeatures
from sklearn.preprocessing import StandardScaler as skStandardScaler

from cuml.compose import ColumnTransformer as cuColumnTransformer
from cuml.compose import make_column_selector as cu_make_column_selector
from cuml.compose import make_column_transformer as cu_make_column_transformer
from cuml.preprocessing import Normalizer as cuNormalizer
from cuml.preprocessing import OneHotEncoder as cuOneHotEncoder
from cuml.preprocessing import PolynomialFeatures as cuPolynomialFeatures
from cuml.preprocessing import SimpleImputer as cuSimpleImputer
from cuml.preprocessing import StandardScaler as cuStandardScaler
from cuml.testing.test_preproc_utils import (  # noqa: F401
    assert_allclose,
    clf_dataset,
    sparse_clf_dataset,
)


@pytest.mark.parametrize("remainder", ["drop", "passthrough"])
@pytest.mark.parametrize(
    "transformer_weights", [None, {"scaler": 2.4, "normalizer": 1.8}]
)
def test_column_transformer(
    clf_dataset,
    remainder,
    transformer_weights,  # noqa: F811
):
    X_np, X = clf_dataset

    sk_selec1 = [0, 2]
    sk_selec2 = [1, 3]
    cu_selec1 = sk_selec1
    cu_selec2 = sk_selec2
    if isinstance(X, (pd.DataFrame, cudf.DataFrame)):
        cu_selec1 = ["c" + str(i) for i in sk_selec1]
        cu_selec2 = ["c" + str(i) for i in sk_selec2]

    cu_transformers = [
        ("scaler", cuStandardScaler(), cu_selec1),
        ("normalizer", cuNormalizer(), cu_selec2),
    ]

    transformer = cuColumnTransformer(
        cu_transformers,
        remainder=remainder,
        transformer_weights=transformer_weights,
    )
    ft_X = transformer.fit_transform(X)
    t_X = transformer.transform(X)
    assert type(t_X) is type(X)

    sk_transformers = [
        ("scaler", skStandardScaler(), sk_selec1),
        ("normalizer", skNormalizer(), sk_selec2),
    ]

    transformer = skColumnTransformer(
        sk_transformers,
        remainder=remainder,
        transformer_weights=transformer_weights,
    )
    sk_t_X = transformer.fit_transform(X_np)

    assert_allclose(ft_X, sk_t_X)
    assert_allclose(t_X, sk_t_X)


@pytest.mark.parametrize("remainder", ["drop", "passthrough"])
@pytest.mark.parametrize(
    "transformer_weights", [None, {"scaler": 2.4, "normalizer": 1.8}]
)
@pytest.mark.parametrize("sparse_threshold", [0.2, 0.8])
def test_column_transformer_sparse(
    sparse_clf_dataset,
    remainder,  # noqa: F811
    transformer_weights,
    sparse_threshold,
):
    X_np, X = sparse_clf_dataset

    if X.format == "csc":
        pytest.xfail()
    dataset_density = X.nnz / X.size

    cu_transformers = [
        ("scaler", cuStandardScaler(with_mean=False), [0, 2]),
        ("normalizer", cuNormalizer(), [1, 3]),
    ]

    transformer = cuColumnTransformer(
        cu_transformers,
        remainder=remainder,
        transformer_weights=transformer_weights,
        sparse_threshold=sparse_threshold,
    )
    ft_X = transformer.fit_transform(X)
    t_X = transformer.transform(X)
    if dataset_density < sparse_threshold:
        # Sparse input -> sparse output if dataset_density > sparse_threshold
        # else sparse input -> dense output
        assert type(t_X) is type(X)

    sk_transformers = [
        ("scaler", skStandardScaler(with_mean=False), [0, 2]),
        ("normalizer", skNormalizer(), [1, 3]),
    ]

    transformer = skColumnTransformer(
        sk_transformers,
        remainder=remainder,
        transformer_weights=transformer_weights,
        sparse_threshold=sparse_threshold,
    )
    sk_t_X = transformer.fit_transform(X_np)

    assert_allclose(ft_X, sk_t_X)
    assert_allclose(t_X, sk_t_X)


@pytest.mark.parametrize("remainder", ["drop", "passthrough"])
def test_make_column_transformer(clf_dataset, remainder):  # noqa: F811
    X_np, X = clf_dataset

    sk_selec1 = [0, 2]
    sk_selec2 = [1, 3]
    cu_selec1 = sk_selec1
    cu_selec2 = sk_selec2
    if isinstance(X, (pd.DataFrame, cudf.DataFrame)):
        cu_selec1 = ["c" + str(i) for i in sk_selec1]
        cu_selec2 = ["c" + str(i) for i in sk_selec2]

    transformer = cu_make_column_transformer(
        (cuStandardScaler(), cu_selec1),
        (cuNormalizer(), cu_selec2),
        remainder=remainder,
    )

    ft_X = transformer.fit_transform(X)
    t_X = transformer.transform(X)
    assert type(t_X) is type(X)

    transformer = sk_make_column_transformer(
        (skStandardScaler(), sk_selec1),
        (skNormalizer(), sk_selec2),
        remainder=remainder,
    )
    sk_t_X = transformer.fit_transform(X_np)

    assert_allclose(ft_X, sk_t_X)
    assert_allclose(t_X, sk_t_X)


@pytest.mark.parametrize("remainder", ["drop", "passthrough"])
@pytest.mark.parametrize("sparse_threshold", [0.2, 0.8])
def test_make_column_transformer_sparse(
    sparse_clf_dataset,
    remainder,
    sparse_threshold,  # noqa: F811
):
    X_np, X = sparse_clf_dataset

    if X.format == "csc":
        pytest.xfail()
    dataset_density = X.nnz / X.size

    transformer = cu_make_column_transformer(
        (cuStandardScaler(with_mean=False), [0, 2]),
        (cuNormalizer(), [1, 3]),
        remainder=remainder,
        sparse_threshold=sparse_threshold,
    )

    ft_X = transformer.fit_transform(X)
    t_X = transformer.transform(X)
    if dataset_density < sparse_threshold:
        # Sparse input -> sparse output if dataset_density > sparse_threshold
        # else sparse input -> dense output
        assert type(t_X) is type(X)

    transformer = sk_make_column_transformer(
        (skStandardScaler(with_mean=False), [0, 2]),
        (skNormalizer(), [1, 3]),
        remainder=remainder,
        sparse_threshold=sparse_threshold,
    )

    sk_t_X = transformer.fit_transform(X_np)

    assert_allclose(ft_X, sk_t_X)
    assert_allclose(t_X, sk_t_X)


@pytest.mark.parametrize("remainder", ["drop", "passthrough"])
def test_column_transformer_get_feature_names_out(clf_dataset, remainder):
    X_np, X = clf_dataset

    bool_mask = [False] * X_np.shape[1]
    bool_mask[0] = True
    bool_mask[-1] = True

    cu_transformer = cuColumnTransformer(
        [
            ("t1", cuPolynomialFeatures(), slice(0, 2)),
            ("t2", cuPolynomialFeatures(), [0, 2]),
            ("t3", cuPolynomialFeatures(), lambda X: [1]),
            ("t4", cuPolynomialFeatures(), bool_mask),
        ],
        remainder=remainder,
    ).fit(X)
    cu_transformer.fit_transform(X)

    sk_transformer = skColumnTransformer(
        [
            ("t1", skPolynomialFeatures(), slice(0, 2)),
            ("t2", skPolynomialFeatures(), [0, 2]),
            ("t3", skPolynomialFeatures(), lambda X: [1]),
            ("t4", skPolynomialFeatures(), bool_mask),
        ],
        remainder=remainder,
    ).fit(X_np)

    res = cu_transformer.get_feature_names_out()
    sol = sk_transformer.get_feature_names_out()
    np.testing.assert_array_equal(res, sol)

    # If the input data lacks feature names, also check the generated output
    # names if feature names are provided explicitly.
    if not hasattr(cu_transformer, "feature_names_in_"):
        input_features = [f"c{i}" for i in range(X.shape[1])]
        res = cu_transformer.get_feature_names_out(input_features)
        sol = sk_transformer.get_feature_names_out(input_features)
        np.testing.assert_array_equal(res, sol)


def test_column_transformer_get_feature_names_deprecated():
    X = np.array([[1.5, 2.5, 3.5], [1.6, 2.4, 3.7]])
    model = cuColumnTransformer([("t1", cuPolynomialFeatures(), [0, 2])])
    model.fit(X)
    with pytest.warns(FutureWarning, match="get_feature_names"):
        res = model.get_feature_names()

    np.testing.assert_array_equal(res, model.get_feature_names_out())


def test_column_transformer_named_transformers_(clf_dataset):  # noqa: F811
    X_np, X = clf_dataset

    cu_transformers = [("PolynomialFeatures", cuPolynomialFeatures(), [0, 2])]
    transformer = cuColumnTransformer(cu_transformers)
    transformer.fit_transform(X)
    cu_named_transformers = transformer.named_transformers_

    sk_transformers = [("PolynomialFeatures", skPolynomialFeatures(), [0, 2])]
    transformer = skColumnTransformer(sk_transformers)
    transformer.fit_transform(X_np)
    sk_named_transformers = transformer.named_transformers_

    assert cu_named_transformers.keys() == sk_named_transformers.keys()


def test_column_transformer_sklearn_clone_preserves_transformers():
    transformer = cuColumnTransformer(
        [("one_hot_encoder", skOneHotEncoder(), ["a", "b"])]
    )

    cloned = sk_clone(transformer)

    assert len(cloned.transformers) == 1
    assert isinstance(cloned.transformers[0][1], skOneHotEncoder)
    assert cloned.transformers[0][2] == ["a", "b"]


def test_column_transformer_sklearn_clone_default_transformers():
    transformer = cuColumnTransformer()

    assert transformer.get_params(deep=True)["transformers"] is None
    cloned = sk_clone(transformer)

    assert cloned.transformers is None


def test_column_transformer_set_transformers_from_empty_list():
    transformer = cuColumnTransformer([])

    transformer._transformers = [("one_hot_encoder", skOneHotEncoder())]

    assert len(transformer.transformers) == 1
    assert isinstance(transformer.transformers[0][1], skOneHotEncoder)
    assert transformer.transformers[0][2] is None


def test_make_column_selector():
    X_np = pd.DataFrame(
        {
            "city": ["London", "London", "Paris", "Sallisaw"],
            "rating": [5, 3, 4, 5],
            "temperature": [21.0, 21.0, 24.0, 28.0],
        }
    )
    X = cudf.from_pandas(X_np)

    cu_transformers = [
        (
            "ohe",
            cuOneHotEncoder(),
            cu_make_column_selector(dtype_exclude=np.number),
        ),
        (
            "scaler",
            cuStandardScaler(),
            cu_make_column_selector(dtype_include=np.integer),
        ),
        (
            "normalizer",
            cuNormalizer(),
            cu_make_column_selector(pattern="temp"),
        ),
    ]
    transformer = cuColumnTransformer(cu_transformers, remainder="drop")
    t_X = transformer.fit_transform(X)

    sk_transformers = [
        (
            "ohe",
            skOneHotEncoder(),
            sk_make_column_selector(dtype_exclude=np.number),
        ),
        (
            "scaler",
            skStandardScaler(),
            sk_make_column_selector(dtype_include=np.integer),
        ),
        (
            "normalizer",
            skNormalizer(),
            sk_make_column_selector(pattern="temp"),
        ),
    ]
    transformer = skColumnTransformer(sk_transformers, remainder="drop")
    sk_t_X = transformer.fit_transform(X_np)

    assert_allclose(t_X, sk_t_X)
    assert type(t_X) is type(X)


def test_column_transformer_index(clf_dataset):  # noqa: F811
    X_np, X = clf_dataset

    if not isinstance(X, (pd.DataFrame, cudf.DataFrame)):
        pytest.skip()

    cu_transformers = [("scaler", cuStandardScaler(), X.columns)]

    transformer = cuColumnTransformer(cu_transformers)
    transformer.fit_transform(X)


def test_column_transform_properly_handles_sub_output_type():
    """Check that ColumnTransformer properly handles child estimators
    with different output types configured"""
    df = cudf.DataFrame({"x": ["a", "b", "a", "b"], "y": [1, 10, 100, 5]})

    transformer = cuColumnTransformer(
        [
            ("x_enc", cuOneHotEncoder(sparse_output=False), ["x"]),
            ("y_enc", cuStandardScaler(output_type="numpy"), ["y"]),
        ]
    ).fit(df)
    transformer.transform(df)


def test_column_transformer_simple_imputer_categorical_cudf():
    """Regression test for https://github.com/rapidsai/cuml/issues/6183

    ColumnTransformer + SimpleImputer on a native cuDF DataFrame with
    categorical/string columns used to raise (KeyError / AttributeError on
    ``.dtype``, later ``ValueError: Unsupported dtype object`` after
    unrelated refactors). impute-then-transform is one of the most common
    sklearn pipeline shapes, so this must work end to end.
    """
    df = cudf.DataFrame(
        {
            "num1": [1.0, np.nan, 3.0],
            "num2": [4.0, 5.0, np.nan],
            "cat1": ["a", None, "c"],
            "cat2": ["x", "y", None],
        }
    )
    df_np = df.to_pandas()

    num_cols = ["num1", "num2"]
    cat_cols = ["cat1"]
    mode_cols = ["cat2"]

    cu_transformer = cuColumnTransformer(
        transformers=[
            (
                "num",
                cuSimpleImputer(strategy="constant", fill_value=0),
                num_cols,
            ),
            (
                "cat",
                cuSimpleImputer(
                    strategy="constant",
                    fill_value="missing",
                    missing_values=pd.NA,
                ),
                cat_cols,
            ),
            (
                "mod",
                cuSimpleImputer(
                    strategy="most_frequent", missing_values=pd.NA
                ),
                mode_cols,
            ),
        ]
    )
    cu_result = cu_transformer.fit_transform(df)

    sk_transformer = skColumnTransformer(
        transformers=[
            (
                "num",
                skSimpleImputer(strategy="constant", fill_value=0),
                num_cols,
            ),
            (
                "cat",
                skSimpleImputer(
                    strategy="constant",
                    fill_value="missing",
                    missing_values=pd.NA,
                ),
                cat_cols,
            ),
            (
                "mod",
                skSimpleImputer(
                    strategy="most_frequent", missing_values=pd.NA
                ),
                mode_cols,
            ),
        ]
    )
    sk_result = sk_transformer.fit_transform(df_np)

    np.testing.assert_array_equal(cu_result.to_numpy(), sk_result)


def test_simple_imputer_add_indicator_object_cudf():
    """Regression: SimpleImputer(add_indicator=True) on string/object columns
    must stack the imputed (host object) data with the indicator mask on host
    instead of routing the host array through cupy.hstack (issue #6183 follow-up).
    """
    df = cudf.DataFrame(
        {
            "cat1": ["a", None, "c", "a"],
            "cat2": ["x", "y", None, "x"],
        }
    )
    df_np = df.to_pandas()
    cu_imp = cuSimpleImputer(
        strategy="most_frequent", missing_values=pd.NA, add_indicator=True
    )
    sk_imp = skSimpleImputer(
        strategy="most_frequent", missing_values=pd.NA, add_indicator=True
    )

    cu_result = cu_imp.fit_transform(df)
    sk_result = sk_imp.fit_transform(df_np)

    np.testing.assert_array_equal(cu_result.to_numpy(), np.asarray(sk_result))


def test_simple_imputer_drops_fully_missing_object_column_cudf():
    df = cudf.DataFrame(
        {
            "all_missing": [None, None, None],
            "observed": ["a", "b", "a"],
        }
    )
    df_np = df.to_pandas()
    cu_imp = cuSimpleImputer(strategy="most_frequent", missing_values=pd.NA)
    sk_imp = skSimpleImputer(strategy="most_frequent", missing_values=pd.NA)

    cu_result = cu_imp.fit_transform(df)
    sk_result = sk_imp.fit_transform(df_np)

    assert cu_result.shape == (3, 1)
    np.testing.assert_array_equal(cu_result.to_numpy(), sk_result)


def test_simple_imputer_add_indicator_clone_params():
    imputer = cuSimpleImputer(add_indicator=True, missing_values=pd.NA)

    cloned = sk_clone(imputer)

    params = cloned.get_params()
    assert params["add_indicator"] is True
    assert params["missing_values"] is pd.NA


@pytest.mark.parametrize("input_type", ["pandas", "numpy"])
@pytest.mark.parametrize("missing_values", [pd.NA, None, np.nan])
def test_simple_imputer_missing_sentinel_parity(input_type, missing_values):
    if input_type == "pandas":
        X = pd.DataFrame({"cat": ["x", missing_values, "x", "y"]})
    else:
        X = np.array([["x"], [np.nan], ["x"], ["y"]], dtype=object)

    cu_imputer = cuSimpleImputer(
        strategy="most_frequent", missing_values=missing_values
    )
    sk_imputer = skSimpleImputer(
        strategy="most_frequent", missing_values=missing_values
    )

    if missing_values is None:
        with pytest.raises(ValueError) as sk_error:
            sk_imputer.fit_transform(X)
        with pytest.raises(ValueError) as cu_error:
            cu_imputer.fit_transform(X)

        assert str(cu_error.value) == str(sk_error.value)
    else:
        cu_result = cu_imputer.fit_transform(X)
        sk_result = sk_imputer.fit_transform(X)

        np.testing.assert_array_equal(np.asarray(cu_result), sk_result)


@pytest.mark.parametrize("input_type", ["pandas", "readonly-numpy"])
@pytest.mark.parametrize("copy", [False, True])
@pytest.mark.parametrize("missing_values", [pd.NA, np.nan])
def test_simple_imputer_copy_object_parity(input_type, copy, missing_values):
    def make_input():
        if input_type == "pandas":
            return pd.DataFrame(
                {
                    "cat": pd.Series(
                        ["x", missing_values, "x", "y"], dtype=object
                    )
                }
            )

        X = np.array([["x"], [np.nan], ["x"], ["y"]], dtype=object)
        X.flags.writeable = False
        return X

    cu_imputer = cuSimpleImputer(
        strategy="most_frequent", missing_values=missing_values, copy=copy
    )
    sk_imputer = skSimpleImputer(
        strategy="most_frequent", missing_values=missing_values, copy=copy
    )

    cu_result = cu_imputer.fit_transform(make_input())
    sk_result = sk_imputer.fit_transform(make_input())

    np.testing.assert_array_equal(np.asarray(cu_result), sk_result)


@pytest.mark.parametrize("add_indicator", [False, True])
def test_simple_imputer_string_get_feature_names_out(add_indicator):
    df = pd.DataFrame(
        {
            "cat": pd.Series(["x", pd.NA, "x"], dtype=object),
            "other": pd.Series(["a", "b", "a"], dtype=object),
        }
    )
    cu_imputer = cuSimpleImputer(
        strategy="most_frequent",
        missing_values=pd.NA,
        add_indicator=add_indicator,
    ).fit(df)
    sk_imputer = skSimpleImputer(
        strategy="most_frequent",
        missing_values=pd.NA,
        add_indicator=add_indicator,
    ).fit(df)

    np.testing.assert_array_equal(
        cu_imputer.get_feature_names_out(),
        sk_imputer.get_feature_names_out(),
    )


def test_is_object_dtype_handles_series_and_extension_dtypes():
    from cuml.internals.outputs import _is_object_dtype

    pa = pytest.importorskip("pyarrow")

    assert _is_object_dtype(pd.Series(["a", "b"])) is True
    assert _is_object_dtype(pd.Series([1, 2, 3])) is False
    assert _is_object_dtype(pd.Series(pd.Categorical(["a", "b"]))) is False
    assert _is_object_dtype(pd.Series(["a"], dtype="string")) is True
    assert (
        _is_object_dtype(pd.Series(["a"], dtype=pd.ArrowDtype(pa.string())))
        is True
    )
    assert _is_object_dtype(pd.DataFrame({"a": ["x"], "b": [1]})) is True
    assert _is_object_dtype(np.array(["a", "b"], dtype=object)) is True
    assert _is_object_dtype(np.array([1, 2, 3])) is False
