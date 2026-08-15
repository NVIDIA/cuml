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
