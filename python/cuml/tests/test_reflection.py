# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pickle

import cudf
import cudf.pandas
import cupy as cp
import cupyx.scipy.sparse
import numpy as np
import pandas as pd
import pytest
import scipy.sparse
from sklearn.base import ClassNamePrefixFeaturesOutMixin

import cuml
from cuml.internals.base import Base
from cuml.internals.global_settings import GlobalSettings
from cuml.internals.outputs import (
    ArrayIndexPair,
    ClassLabels,
    ReflectedAttr,
    convert_arrays,
    infer_output_type,
    mlfunc,
)
from cuml.internals.validation import check_inputs

OUTPUT_TYPES = ["numpy", "cupy", "cudf", "pandas"]


@pytest.fixture(autouse=True)
def reset_global_output_type():
    yield
    # Ensure we reset the type at the end of the test
    cuml.set_global_output_type(None)


def assert_output_type(arr, output_type):
    cls = {
        "numpy": np.ndarray,
        "cupy": cp.ndarray,
        "cudf": (cudf.Series, cudf.DataFrame),
        "pandas": (pd.Series, pd.DataFrame),
    }[output_type]
    assert isinstance(arr, cls)


def rand_array(output_type, *, shape=(8, 4), seed=42):
    X = cp.random.default_rng(seed).uniform(
        low=0.0, high=10.0, size=shape, dtype="float32"
    )
    if output_type == "cupy":
        return X
    elif output_type == "numpy":
        return cp.asnumpy(X)
    elif output_type == "pandas":
        X = X.get()
        return pd.DataFrame(X) if X.ndim == 2 else pd.Series(X)
    else:
        assert output_type == "cudf"
        return cudf.DataFrame(X) if X.ndim == 2 else cudf.Series(X)


class ImplementsArray:
    def __init__(self, x):
        self.x = x

    def __array__(self, dtype=None, copy=None):
        return self.x


class ImplementsArrayInterface:
    def __init__(self, x):
        self.x = x

    @property
    def __array_interface__(self):
        return self.x.__array_interface__


class ImplementsCudaArrayInterface:
    def __init__(self, x):
        self.x = x

    @property
    def __cuda_array_interface__(self):
        return self.x.__cuda_array_interface__


class DummyEstimator(Base):
    X_ = ReflectedAttr()

    @mlfunc(set_input_type=True)
    def fit(self, X, y=None):
        self.X_ = check_inputs(self, X, reset=True)
        return self

    @mlfunc
    def example(self, X):
        return cp.zeros(3)

    @mlfunc
    def example_no_args(self):
        return cp.zeros(3)

    @mlfunc
    def check_descriptor(self):
        # When run in an internal context, a descriptor returns its original
        # internal value.
        assert isinstance(self.X_, cp.ndarray)

        with cuml.using_output_type("numpy"):
            # Can override with using_output_type
            assert_output_type(self.X_, "numpy")


@mlfunc
def returns_cupy(X):
    return cp.asarray(X)


@mlfunc
def returns_array_no_args():
    return cp.ones(3)


@mlfunc(array_arg=None)
def returns_array_one_arg(n):
    return cp.ones(n)


def test_set_global_output_type():
    gs = GlobalSettings()
    assert gs.output_type is None

    cuml.set_global_output_type("cupy")
    assert gs.output_type == "cupy"

    cuml.set_global_output_type(None)
    assert gs.output_type is None

    with pytest.raises(ValueError, match="`output_type='bad'`"):
        cuml.set_global_output_type("bad")


def test_using_output_type():
    gs = GlobalSettings()
    assert gs.output_type is None

    cuml.set_global_output_type("cupy")
    assert gs.output_type == "cupy"

    with cuml.using_output_type("cudf"):
        assert gs.output_type == "cudf"
    assert gs.output_type == "cupy"

    with cuml.using_output_type(None):
        assert gs.output_type is None
    assert gs.output_type == "cupy"

    with pytest.raises(ValueError, match="`output_type='bad'`"):
        with cuml.using_output_type("bad"):
            pass


@pytest.mark.parametrize("input_type", OUTPUT_TYPES)
def test_infer_output_type(input_type):
    X = rand_array(input_type)
    output_type = infer_output_type(X)
    assert output_type == input_type


@pytest.mark.parametrize("kind", ["cudf", "pandas"])
def test_infer_output_type_dataframes(kind):
    ns = cudf if kind == "cudf" else pd
    df = ns.DataFrame({"x": [1, 2, 3]}, index=[10, 20, 30])
    assert infer_output_type(df) == kind
    assert infer_output_type(df.x) == kind
    assert infer_output_type(df.index) == kind


def test_infer_output_type_cuda_array_interface():
    x = ImplementsCudaArrayInterface(cp.array([1, 2, 3]))
    assert infer_output_type(x) == "cupy"


@pytest.mark.parametrize(
    "obj",
    [
        pytest.param([1, 2], id="list"),
        pytest.param((1, 2), id="tuple"),
        pytest.param([[1, 2], [3, 4]], id="nested-list"),
        pytest.param(ImplementsArray(np.array([1, 2])), id="__array__"),
        pytest.param(
            ImplementsArrayInterface(np.array([1, 2])),
            id="__array_interface__",
        ),
    ],
)
def test_infer_output_type_array_like(obj):
    assert infer_output_type(obj) == "numpy"
    assert infer_output_type(obj, array_like="fizz") == "fizz"


@pytest.mark.parametrize(
    "obj",
    [
        pytest.param(None, id="none"),
        pytest.param(1, id="scalar"),
        pytest.param(np.int32(1), id="numpy-scalar"),
        pytest.param("abc", id="string"),
        pytest.param(b"abc", id="bytes"),
        pytest.param({"a": 1, "b": 2}, id="dict"),
        pytest.param(object(), id="arbitrary-object"),
    ],
)
def test_infer_output_type_non_arrays(obj):
    assert infer_output_type(obj) is None


@pytest.mark.parametrize("input_type", OUTPUT_TYPES)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_default_output_type(input_type):
    X = rand_array(input_type)
    model = cuml.DBSCAN(eps=1.0, min_samples=1)
    labels = model.fit_predict(X)
    assert_output_type(labels, input_type)
    assert_output_type(model.components_, input_type)


@pytest.mark.parametrize("input_type", OUTPUT_TYPES)
@pytest.mark.parametrize("output_type", OUTPUT_TYPES)
def test_estimator_output_type(input_type, output_type):
    X = rand_array(input_type)
    model = cuml.DBSCAN(eps=1.0, min_samples=1, output_type=output_type)
    labels = model.fit_predict(X)
    assert_output_type(labels, output_type)
    assert_output_type(model.components_, output_type)


@pytest.mark.parametrize("input_type", OUTPUT_TYPES)
@pytest.mark.parametrize("output_type", OUTPUT_TYPES)
def test_global_output_type(input_type, output_type):
    cuml.set_global_output_type(output_type)
    X = rand_array(input_type)
    model = cuml.DBSCAN(eps=1.0, min_samples=1)
    labels = model.fit_predict(X)
    assert_output_type(labels, output_type)
    assert_output_type(model.components_, output_type)


def test_invalid_estimator_output_type():
    X = rand_array("numpy")
    model = cuml.DBSCAN(eps=1.0, min_samples=1)
    model.fit(X)
    model.output_type = "invalid"

    # Descriptor raises appropriately
    with pytest.raises(ValueError, match="`output_type='invalid'`"):
        model.components_

    # Method raises appropriately
    with pytest.raises(ValueError, match="`output_type='invalid'`"):
        model.fit_predict(X)


def test_global_overrides_estimator_output_type():
    cuml.set_global_output_type("numpy")
    X = rand_array("pandas")
    model = cuml.DBSCAN(eps=1.0, min_samples=1, output_type="cupy")
    labels = model.fit_predict(X)
    assert_output_type(labels, "numpy")
    assert_output_type(model.components_, "numpy")


def test_global_input_with_estimator_output_type():
    cuml.set_global_output_type("input")
    X = rand_array("pandas")
    model = cuml.DBSCAN(eps=1.0, min_samples=1, output_type="cupy")
    labels = model.fit_predict(X)
    # The difference here is probably a bug, but it's been the behavior for a
    # long time. Methods respect `estimator.output_type` if the global
    # `output_type` is 'input', while attributes respect the global
    # `output_type`.
    assert_output_type(labels, "cupy")
    assert_output_type(model.components_, "pandas")


@pytest.mark.parametrize(
    "input_type, order",
    [
        ("numpy", "C"),
        ("numpy", "F"),
        ("cupy", "C"),
        ("cupy", "F"),
        ("pandas", "F"),
        ("cudf", "F"),
    ],
)
@pytest.mark.parametrize("output_type", ["numpy", "cupy"])
def test_convert_arrays_dense_array(input_type, order, output_type):
    X = rand_array(input_type)
    if input_type == "cupy":
        X = cp.asarray(X, order=order)
    elif input_type == "numpy":
        X = np.asarray(X, order=order)

    out = convert_arrays(X, output_type)
    sol = X.to_numpy() if hasattr(X, "to_numpy") else cp.asnumpy(X)

    assert_output_type(out, output_type)
    np.testing.assert_array_equal(cp.asnumpy(out), sol)
    assert out.flags.c_contiguous if order == "C" else out.flags.f_contiguous


@pytest.mark.parametrize("input_type", ["numpy", "cupy", "pandas", "cudf"])
@pytest.mark.parametrize("output_type", ["numpy", "cupy"])
def test_convert_arrays_dense_array_1d(input_type, output_type):
    X = rand_array(input_type, shape=8)
    out = convert_arrays(X, output_type)
    sol = X.to_numpy() if hasattr(X, "to_numpy") else cp.asnumpy(X)

    assert_output_type(out, output_type)
    np.testing.assert_array_equal(cp.asnumpy(out), sol)


@pytest.mark.parametrize("input_type", ["scipy", "cupyx"])
@pytest.mark.parametrize("format", ["coo", "csc", "csr"])
@pytest.mark.parametrize("output_type", OUTPUT_TYPES)
def test_convert_arrays_sparse_array(input_type, output_type, format):
    if input_type == "cupyx":
        X = cupyx.scipy.sparse.random(
            5, 5, random_state=42, density=0.5, format=format
        )
    else:
        X = scipy.sparse.random(
            5, 5, random_state=42, density=0.5, format=format
        )

    out = convert_arrays(X, output_type)

    if output_type in ["cupy", "cudf"]:
        assert cupyx.scipy.sparse.issparse(out)
    else:
        assert scipy.sparse.issparse(out)

    assert out.format == format

    np.testing.assert_array_equal(
        cp.asnumpy(X.todense()),
        cp.asnumpy(out.todense()),
    )


@pytest.mark.parametrize("kind", ["dataframe", "series"])
@pytest.mark.parametrize("input_type", ["cupy", "numpy", "pandas", "cudf"])
@pytest.mark.parametrize("output_type", ["pandas", "cudf"])
def test_convert_arrays_dataframe(kind, input_type, output_type):
    arr = rand_array(input_type, shape=((8, 4) if kind == "dataframe" else 8))
    res = convert_arrays(arr, output_type)
    assert_output_type(res, output_type)

    if kind == "dataframe":
        cudf.testing.assert_frame_equal(
            cudf.DataFrame(res), cudf.DataFrame(arr)
        )
    else:
        cudf.testing.assert_series_equal(cudf.Series(res), cudf.Series(arr))


@pytest.mark.parametrize("xdf", [pd, cudf])
@pytest.mark.parametrize("kind", ["dataframe", "series"])
@pytest.mark.parametrize("use_pair", [False, True])
@pytest.mark.parametrize("input_type", ["cupy", "numpy", "pandas", "cudf"])
@pytest.mark.parametrize("output_type", ["pandas", "cudf"])
def test_convert_arrays_dataframe_with_index(
    xdf, kind, use_pair, input_type, output_type
):
    arr = rand_array(input_type, shape=((8, 4) if kind == "dataframe" else 8))
    index = xdf.Index(["a", "b", "c", "d", "e", "f", "g", "h"])

    if use_pair:
        res = convert_arrays(ArrayIndexPair(arr, index), output_type)
    else:
        res = convert_arrays(arr, output_type, index=index)

    if kind == "dataframe":
        sol = cudf.DataFrame(arr)
        sol.index = index
        cudf.testing.assert_frame_equal(cudf.DataFrame(res), sol)
    else:
        sol = cudf.Series(arr)
        sol.index = index
        cudf.testing.assert_series_equal(cudf.Series(res), sol)


@pytest.mark.parametrize(
    "construct",
    [
        pytest.param(lambda x, y: [x, 1, y], id="list"),
        pytest.param(lambda x, y: (x, 1, y), id="tuple"),
        pytest.param(lambda x, y: {"x": x, "y": y, "z": 1}, id="dict"),
        pytest.param(lambda x, y: {"a": [(x, 1), (y, 2)]}, id="nested"),
    ],
)
@pytest.mark.parametrize("output_type", ["numpy", "cupy", "pandas", "cudf"])
def test_convert_arrays_nested_values(construct, output_type):
    cuml.set_global_output_type(output_type)
    x = rand_array("cupy", shape=(4, 8))
    y = rand_array("cupy", shape=(4, 4))
    res = convert_arrays(construct(x, y), output_type)
    sol = construct("x", "y")

    def check_nested_types(res, sol):
        """Check types match, using `x` and `y` as placeholders for arrays"""
        if sol in ("x", "y"):
            assert_output_type(res, output_type)
        else:
            assert type(res) is type(sol)
            if isinstance(res, dict):
                assert set(res) == set(sol)
                for k in res:
                    check_nested_types(res[k], sol[k])
            elif isinstance(res, (tuple, list)):
                assert len(res) == len(sol)
                for r, s in zip(res, sol):
                    check_nested_types(r, s)

    check_nested_types(res, sol)


@pytest.mark.parametrize("output_type", [None, *OUTPUT_TYPES])
def test_mlfunc_dense_outputs(output_type):
    cuml.set_global_output_type(output_type)
    X = rand_array("cupy")

    # Reflected functions treat None/"input" the same
    assert_output_type(
        returns_cupy(X),
        "cupy" if output_type in (None, "input") else output_type,
    )

    # With no array argument functions default to 'cupy' unless
    # a concrete type is configured
    expected = "cupy" if output_type in (None, "input") else output_type
    assert_output_type(returns_array_no_args(), expected)
    assert_output_type(returns_array_one_arg(3), expected)


@pytest.mark.parametrize("output_type", [None, *OUTPUT_TYPES])
def test_mlfunc_sparse_outputs(output_type):
    @mlfunc
    def make_sparse():
        return cupyx.scipy.sparse.random(5, 5, random_state=42)

    cuml.set_global_output_type(output_type)
    res = make_sparse()

    if output_type in [None, "input", "cupy", "cudf", "cuml"]:
        assert cupyx.scipy.sparse.issparse(res)
    else:
        assert scipy.sparse.issparse(res)


@pytest.mark.parametrize("output_type", ["input", "cupy", "cudf"])
def test_mlfunc_nested_output(output_type):
    expected_type = "cupy" if output_type == "input" else output_type
    cuml.set_global_output_type(output_type)
    x = rand_array("cupy")

    @mlfunc
    def myfunc(x):
        return [{"x": x}, (1, x + 1)]

    res = myfunc(x)
    assert isinstance(res, list)
    assert isinstance(res[0], dict)
    assert_output_type(res[0]["x"], expected_type)
    assert isinstance(res[1], tuple)
    assert res[1][0] == 1
    assert_output_type(res[1][1], expected_type)


@pytest.mark.parametrize(
    "obj",
    [
        pytest.param([1, 2, 3], id="list"),
        pytest.param((1, 2, 3), id="tuple"),
        pytest.param(ImplementsArray(np.array([1, 2, 3])), id="__array__"),
    ],
)
def test_mlfunc_dont_convert_array_like(obj):
    @mlfunc
    def make_array_like():
        return obj

    cuml.set_global_output_type("numpy")
    res = make_array_like()
    assert type(res) is type(obj)


@pytest.mark.parametrize("output_type", [None, "input", "numpy"])
def test_mlfunc_internal_calls(output_type):
    @mlfunc(array_arg="X")
    def apply(func, X):
        result = func(X)
        # Internal calls return internal types by default
        assert isinstance(result, cp.ndarray)

        with cuml.using_output_type("numpy"):
            temp = func(X)

        # Internal calls can configure output type to get
        # something specific when needed
        assert isinstance(temp, np.ndarray)

        return result

    cuml.set_global_output_type(output_type)
    X = rand_array("pandas")
    res = apply(returns_cupy, X)
    expected = "pandas" if output_type in (None, "input") else output_type
    assert_output_type(res, expected)


@pytest.mark.parametrize("input_type", ["pandas", "cudf"])
@pytest.mark.parametrize("output_type", ["pandas", "cudf"])
def test_mlfunc_preserve_index(input_type, output_type):
    xdf = cudf if input_type == "cudf" else pd
    df = xdf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}, index=["a", "c", "b"])

    @mlfunc(preserve_index=True)
    def myfunc(df):
        return cp.asarray(df.x + 1)

    with cuml.using_output_type(output_type):
        res = myfunc(df)

    assert_output_type(res, output_type)

    cudf.testing.assert_series_equal(
        cudf.Series(res), cudf.Series([2, 3, 4], index=df.index)
    )


@pytest.mark.parametrize("input_type", ["pandas", "cudf"])
def test_mlfunc_column_names(input_type):
    xdf = cudf if input_type == "cudf" else pd
    X = xdf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    class MyEstimator(ClassNamePrefixFeaturesOutMixin, Base):
        @mlfunc(set_input_type=True)
        def fit(self, X):
            X = check_inputs(self, X, reset=True)
            self._n_features_out = X.shape[1] * 2
            return self

        @mlfunc(column_names="feature_names_out")
        def fit_transform(self, X):
            self.fit(X)
            return cp.ones((X.shape[0], self._n_features_out))

        @mlfunc(column_names="feature_names_out")
        def returns_sparse_matrix(self, X):
            return cupyx.scipy.sparse.eye(
                X.shape[0],
                self._n_features_out,
                format="csr",
            )

        @mlfunc
        def no_names(self, X):
            return cp.ones((X.shape[0], self.n_features_in_))

        @mlfunc(column_names="feature_names_in")
        def inverse_transform(self, X):
            return cp.ones((X.shape[0], self.n_features_in_))

    # Works with feature names
    model = MyEstimator()
    Xt = model.fit_transform(X)
    np.testing.assert_array_equal(Xt.columns, model.get_feature_names_out())

    X2 = model.inverse_transform(Xt)
    np.testing.assert_array_equal(X2.columns, X.columns)

    # column_names=None doesn't add anything
    res = model.no_names(X)
    np.testing.assert_array_equal(res.columns, [0, 1])

    # No error applying names if not a dataframe output
    res = model.returns_sparse_matrix(X)
    assert cupyx.scipy.sparse.issparse(res) or scipy.sparse.issparse(res)

    # Works if no feature names
    X = X.to_numpy()
    model = MyEstimator(output_type=input_type)
    Xt = model.fit_transform(X)
    np.testing.assert_array_equal(Xt.columns, model.get_feature_names_out())

    X2 = model.inverse_transform(Xt)
    np.testing.assert_array_equal(X2.columns, [0, 1])


@pytest.mark.parametrize("dtype", ["int32", "object", "U"])
@pytest.mark.parametrize("output_type", OUTPUT_TYPES)
def test_class_labels(dtype, output_type):
    if dtype in ("object", "U"):
        classes = np.array(["a", "b", "c"], dtype=dtype)
    else:
        classes = np.array([10, 20, 30], dtype=dtype)

    indices = cp.array([0, 2, 1, 2, 1, 0])

    @mlfunc
    def myfunc():
        return ClassLabels(indices, classes)

    if dtype in ("object", "U") and output_type == "cupy":
        with pytest.raises(
            TypeError,
            match=f"output_type={output_type!r} doesn't support outputs of dtype",
        ):
            with cuml.using_output_type(output_type):
                res = myfunc()
    else:
        with cuml.using_output_type(output_type):
            res = myfunc()

        assert_output_type(res, output_type)
        res = res.to_numpy() if hasattr(res, "to_numpy") else cp.asnumpy(res)
        np.testing.assert_array_equal(
            res,
            classes.take(indices.get()),
        )


def test_mlfunc_convert_output_false():
    @mlfunc(convert_output=False)
    def always_returns_numpy(X):
        result = returns_cupy(X)
        assert isinstance(result, cp.ndarray)
        return result.get(order="A")

    cuml.set_global_output_type("cudf")
    X = rand_array("pandas")
    res = always_returns_numpy(X)
    assert_output_type(res, "numpy")


def test_decorator_sets_input_type():
    X = rand_array("numpy", shape=(10, 5))
    model = DummyEstimator().fit(X)
    assert model.n_features_in_ == 5
    assert model._input_type == "numpy"


def test_estimator_method_with_array_input():
    X = rand_array("numpy", shape=(10, 5))
    X2 = rand_array("cudf", shape=(10, 5))
    model = DummyEstimator().fit(X)

    # Reflects method input by default
    assert_output_type(model.example(X2), "cudf")

    # Estimator output_type can override
    model.output_type = "pandas"
    assert_output_type(model.example(X2), "pandas")

    # Global output type overrides
    with cuml.using_output_type("cupy"):
        assert_output_type(model.example(X2), "cupy")


def test_array_like_inputs_treated_as_numpy_by_reflection():
    X_cupy = rand_array("cupy", shape=(10, 5))
    X_list = rand_array("numpy", shape=(10, 5)).tolist()

    model_fit_list = DummyEstimator().fit(X_list)
    model_fit_cupy = DummyEstimator().fit(X_cupy)

    # Fitting on array-likes stores `numpy` as input-type
    assert model_fit_list._input_type == "numpy"

    # Inferring on array-likes uses `numpy` as output type
    assert_output_type(model_fit_list.example(X_list), "numpy")
    assert_output_type(model_fit_cupy.example(X_list), "numpy")

    # Methods with no args use input type
    assert_output_type(model_fit_list.example_no_args(), "numpy")
    assert_output_type(model_fit_cupy.example_no_args(), "cupy")


@pytest.mark.parametrize("output_type", ["pandas", "cudf"])
def test_one_col_2d_array_only_coerced_to_series_for_predict(output_type):
    """For legacy reasons, cuml will coerce a 1 column 2D output to a Series
    instead of a DataFrame when outputting pandas/cudf types. In the long run
    we want to deprecate and remove this (See #7893). However, the output of
    `transform`/`inverse_transform`/... should _always_ be a 2D output. Here we
    test that this coercion is only enabled for `predict*` methods."""

    class MyEstimator(Base):
        @mlfunc
        def transform(self, X):
            return cp.ones((X.shape[0], 1))

        @mlfunc
        def fit_transform(self, X):
            return cp.ones((X.shape[0], 1))

        @mlfunc
        def fit_predict(self, X):
            return cp.ones((X.shape[0], 1))

        @mlfunc
        def predict(self, X):
            return cp.ones((X.shape[0], 1))

        @mlfunc
        def predict_proba(self, X):
            return cp.ones((X.shape[0], 1))

    model = MyEstimator(output_type=output_type)
    X = cp.ones((3, 4))
    assert model.fit_transform(X).ndim == 2
    assert model.transform(X).ndim == 2
    assert model.fit_predict(X).ndim == 1
    assert model.predict(X).ndim == 1
    assert model.predict_proba(X).ndim == 1


def test_estimator_method_with_no_array_input():
    X = rand_array("numpy", shape=(10, 5))
    model = DummyEstimator().fit(X)

    # Reflects fit input by default
    assert_output_type(model.example_no_args(), "numpy")

    # Estimator output_type can override
    model.output_type = "cupy"
    assert_output_type(model.example_no_args(), "cupy")

    # Global output type overrides
    with cuml.using_output_type("pandas"):
        assert_output_type(model.example_no_args(), "pandas")


@pytest.mark.parametrize("output_type", [None, *OUTPUT_TYPES])
def test_reflected_attr(output_type):
    cuml.set_global_output_type(output_type)
    X = rand_array("pandas")

    model = DummyEstimator().fit(X)

    # Reflected attributes treat None/"input" the same
    assert_output_type(
        model.X_,
        "pandas" if output_type in (None, "input") else output_type,
    )

    # With no array argument functions default to 'cupy' unless
    # a concrete type is configured
    expected = "cupy" if output_type in (None, "input") else output_type
    assert_output_type(returns_array_no_args(), expected)
    assert_output_type(returns_array_one_arg(3), expected)


def test_reflected_attr_type_in_internal_context():
    X = rand_array("numpy")
    model = DummyEstimator().fit(X)
    assert_output_type(model.X_, "numpy")
    model.check_descriptor()


def test_reflected_attr_cache_behavior():
    X = rand_array("cupy")
    model = DummyEstimator().fit(X)
    assert_output_type(model.X_, "cupy")
    # Instance is cached
    assert model.X_ is model.X_
    assert len(model.__dict__["X_"]._cache) == 1  # cupy

    with cuml.using_output_type("pandas"):
        assert_output_type(model.X_, "pandas")
        # Instance is cached, but original cache isn't wiped
        assert model.X_ is model.X_
        assert len(model.__dict__["X_"]._cache) == 2  # cupy + pandas

    msg = pickle.dumps(model)
    model2 = pickle.loads(msg)
    # Only one array is serialized when pickled, and the cache is reset in the
    # new instance.
    assert b"pandas" not in msg
    assert_output_type(model2.X_, "cupy")
    assert len(model2.__dict__["X_"]._cache) == 1  # cupy
    assert len(model.__dict__["X_"]._cache) == 2  # cupy + pandas

    # Assigning an attribute resets the cache
    model.X_ = X
    assert_output_type(model.X_, "cupy")
    assert len(model.__dict__["X_"]._cache) == 1  # cupy


def test_decorators_set_cupy_ptds():
    class MyEstimator(Base):
        @mlfunc(set_input_type=True)
        def fit(self, X, y=None):
            assert cp.cuda.get_current_stream() is cp.cuda.Stream.ptds
            return self

        @mlfunc
        def direct_call(self, X):
            assert cp.cuda.get_current_stream() is cp.cuda.Stream.ptds
            return cp.zeros(3)

        @mlfunc
        def nested_call(self, X):
            assert cp.cuda.get_current_stream() is cp.cuda.Stream.ptds
            return self.direct_call(X)

        @mlfunc(convert_output=False)
        def no_reflection(self, X):
            assert cp.cuda.get_current_stream() is cp.cuda.Stream.ptds
            return cp.zeros(3)

    X = cp.ones(3)

    # Check that ptds is used instead of the default stream
    model = MyEstimator()
    model.fit(X)
    model.direct_call(X)
    model.nested_call(X)
    model.no_reflection(X)

    # Check that ptds is used instead of a custom stream
    with cp.cuda.Stream():
        model = MyEstimator()
        model.fit(X)
        model.direct_call(X)
        model.nested_call(X)
        model.no_reflection(X)
