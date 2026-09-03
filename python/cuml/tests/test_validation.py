# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import re
import warnings
from contextlib import contextmanager

import cudf
import cupy as cp
import cupyx.scipy.sparse as cp_sp
import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import sklearn
from hypothesis import assume, example, given
from packaging.version import Version
from sklearn.exceptions import DataConversionWarning

from cuml.internals.validation import (
    _get_feature_names,
    _get_n_features,
    check_all_finite,
    check_array,
    check_consistent_length,
    check_cudf,
    check_features,
    check_input_features,
    check_inputs,
    check_non_negative,
    check_random_seed,
    check_sample_weight,
    check_y,
)

CUPY_SUPPORTS_LARGE_SPARSE = Version(cp.__version__) >= Version("14.1.0")

DTYPES = ("i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f2", "f4", "f8")


def is_cuda_output(mem_type, value=..., kind=...):
    """Infer if cuda output given `mem_type` and a value or kind"""
    if mem_type is None:
        if value is not ...:
            if cudf.pandas.LOADED:
                if isinstance(value, pd.DataFrame):
                    return not any(d.kind == "O" for d in value.dtypes)
                elif isinstance(value, pd.Series):
                    return value.dtype.kind != "O"
            return isinstance(
                value,
                (cp.ndarray, cp_sp.spmatrix, cudf.Series, cudf.DataFrame),
            )
        elif kind is not ...:
            if cudf.pandas.LOADED:
                return kind in ("cupy", "cudf", "pandas")
            return kind in ("cupy", "cudf")
        else:
            raise ValueError("Expected `value` or `kind`")
    return mem_type == "device"


def gen_dense_array(
    *,
    kind="cupy",
    dtype="float32",
    order="C",
    ndim=2,
    n_samples=5,
    n_features=4,
):
    """Generate a suitable dense array input."""
    shape = (n_samples, n_features) if ndim == 2 else (n_samples,)
    dtype = np.dtype(dtype)

    if kind in ("cupy", "numpy", "pandas") and order is None:
        # generate a non-contiguous strided input if order=None
        shape = (n_samples * 2, n_features) if ndim == 2 else (n_samples * 2,)

    rng = np.random.default_rng(42)
    if dtype.kind == "f":
        data = rng.uniform(0, 100, size=shape).astype(dtype)
    else:
        data = rng.integers(0, 100, size=shape).astype(dtype)

    if kind == "cupy":
        out = cp.asarray(data, dtype=dtype, order=order)
        return out[::2] if order is None else out
    elif kind == "numpy":
        out = np.asarray(data, dtype=dtype, order=order)
        return out[::2] if order is None else out
    elif kind == "list":
        return data.tolist()
    elif kind == "pandas":
        out = pd.Series(data) if ndim == 1 else pd.DataFrame(data)
        return out[::2] if order is None else out
    elif kind == "cudf":
        return cudf.Series(data) if ndim == 1 else cudf.DataFrame(data)


@st.composite
def dense_arrays(
    draw,
    kind=None,
    dtype=None,
    order=None,
    ndim=2,
    n_samples=5,
    n_features=4,
):
    """A strategy for generating dense array inputs.

    Parameters
    ----------
    kind : {'cupy', 'numpy', 'list', 'pandas', 'cudf'} or list
        The input kind(s) to select from.
    dtype : dtype-like or list[dtype]
        The dtype(s) to select from.
    order : {'C', 'F', None} or list
        The contiguity order requirement(s) to select from.
    ndim : {1, 2} or list
        The number of dimensions to select from.
    n_samples : int
        The number of samples to generate
    n_features : int
        The number of features to generate (if ndim=2).
    """

    def select(value, choices, cast=None):
        if value is None:
            value = choices
        if isinstance(value, (list, tuple)):
            value = draw(st.sampled_from(value))
        if cast is not None:
            value = cast(value)
        assert value in choices
        return value

    kind = select(kind, ("cupy", "numpy", "list", "pandas", "cudf"))
    dtype = select(dtype, DTYPES, cast=np.dtype)
    assume(not (kind == "cudf" and dtype == "float16"))

    ndim = select(ndim, (1, 2))

    if kind in ("cupy", "numpy", "pandas"):
        order = select(order, ("C", "F", None) if ndim == 2 else ("C", None))
    else:
        # Other containers don't have flexible contiguity
        order = None

    return gen_dense_array(
        kind=kind,
        dtype=dtype,
        order=order,
        ndim=ndim,
        n_samples=n_samples,
        n_features=n_features,
    )


@st.composite
def sparse_arrays(
    draw,
    kind=None,
    dtype=None,
    format=None,
    n_samples=5,
    n_features=4,
    density=0.5,
):
    """A strategy for generating sparse array inputs.

    Parameters
    ----------
    kind : {'cupy', 'scipy', 'scipy-array'} or list
        The input kind(s) to select from.
    dtype : dtype-like or list[dtype]
        The dtype(s) to select from.
    format : sparse format or list
        The format(s) to select from.
    n_samples : int
        The number of samples to generate
    n_features : int
        The number of features to generate
    density : float
        The density of the sparse matrix to generate
    """

    def select(value, choices, cast=None):
        if value is None:
            value = choices
        if isinstance(value, (list, tuple)):
            value = draw(st.sampled_from(value))
        if cast is not None:
            value = cast(value)
        assume(value in choices)
        return value

    kind = select(kind, ("cupy", "scipy", "scipy-array"))
    mem_type = "device" if kind == "cupy" else "host"

    DTYPES = ("f4", "f8")
    FORMATS = ("csr", "csc", "coo", "dia")
    if mem_type == "host":
        DTYPES += ("i4", "i8")
        FORMATS += ("bsr", "dok", "lil")

    dtype = select(dtype, DTYPES, cast=np.dtype)
    format = select(format, FORMATS)

    rng = np.random.default_rng(42)
    array = sp.random(
        n_samples,
        n_features,
        density=density,
        format=format,
        dtype=dtype,
        data_rvs=(
            (lambda n: rng.uniform(0, 100, size=n))
            if dtype.kind == "f"
            else (lambda n: rng.integers(0, 100, size=n))
        ),
        random_state=42,
    )
    if kind == "cupy":
        array = getattr(cp_sp, f"{format}_matrix")(array)
    elif kind == "scipy-array":
        array = getattr(sp, f"{format}_array")(array)

    return array


def as_cupy(array, dtype=None, order=None):
    """Coerce an array to cupy"""
    if isinstance(array, (cudf.Series, cudf.DataFrame)):
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        array = array.to_cupy()
    return cp.asarray(array, dtype=dtype, order=order)


def assert_contiguity(array, order):
    """Assert an array has the proper contiguity"""
    if order == "A":
        assert array.flags["C_CONTIGUOUS"] or array.flags["F_CONTIGUOUS"]
    elif order == "C":
        assert array.flags["C_CONTIGUOUS"]
    elif order == "F":
        assert array.flags["F_CONTIGUOUS"]


@contextmanager
def assert_no_warnings():
    """Small helper for asserting no warnings raised"""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


@pytest.mark.parametrize(
    "seed",
    [
        pytest.param(None, id="none"),
        pytest.param(42, id="int"),
        pytest.param(np.random.RandomState(42), id="numpy"),
        pytest.param(cp.random.RandomState(42), id="cupy"),
    ],
)
def test_check_random_seed(seed):
    res = check_random_seed(seed)
    assert isinstance(res, int)
    assert 0 <= res <= (2**32 - 1)  # in range for uint32
    if isinstance(seed, int):
        assert check_random_seed(seed) == res


def test_check_random_seed_errors():
    for bad in [-1, 2**32]:
        with pytest.raises(
            ValueError, match=r"Expected `0 <= random_state <= 2\*\*32 - 1`"
        ):
            check_random_seed(bad)
    for ok in [0, 2**32 - 1]:
        check_random_seed(ok)
    with pytest.raises(TypeError, match="`random_state` must be"):
        check_random_seed("incorrect type")


def test_get_n_features():
    class ArrayInterfaceOnly:
        def __init__(self, array):
            self.array = array

        @property
        def __array_interface__(self):
            return self.array.__array_interface__

        @property
        def __cuda_array_interface__(self):
            return self.array.__cuda_array_interface__

    x = np.ones((3, 2))
    cx = cp.ones((3, 2))

    assert _get_n_features(x) == 2
    assert _get_n_features(ArrayInterfaceOnly(x)) == 2
    assert _get_n_features(cx) == 2
    assert _get_n_features(ArrayInterfaceOnly(cx)) == 2

    assert _get_n_features([[1, 2, 3], [3, 4, 5]]) == 3
    assert _get_n_features([np.array([1, 2])]) == 2
    assert _get_n_features([]) == 0

    with pytest.raises(ValueError, match="Expected 2D array, got 3D array"):
        _get_n_features(np.ones((3, 2, 1)))


@pytest.mark.parametrize(
    "X",
    [
        [1, 2, 3],
        ["a", "b", "c"],
        [b"a", b"b", b"c"],
        [{"a": 1, "b": 2}, {"c": 3}],
        np.array([1, 2, 3]),
        cp.array([1, 2, 3]),
        pd.Series([1, 2, 3]),
        cudf.Series([1, 2, 3]),
    ],
)
def test_get_n_features_1D(X):
    if isinstance(X, (pd.Series, cudf.Series)):
        match = "Expected a 2-dimensional container"
    else:
        match = "Expected 2D array"

    with pytest.raises(ValueError, match=match):
        _get_n_features(X)


def test_get_feature_names():
    def assert_names(df, sol):
        names = _get_feature_names(df)
        if sol is None:
            assert names is None
        else:
            assert isinstance(names, np.ndarray)
            assert names.dtype == "object"
            assert (names == sol).all()

    class DataFrameInterfaceOnly:
        def __init__(self, df):
            self.df = df

        def __dataframe__(self):
            return self.df.__dataframe__()

    # Non-dataframe-like objects return None
    assert_names(None, None)
    assert_names([1, 2], None)
    assert_names(np.array([[1, 2], [3, 4]]), None)

    # 0-column frames return None
    empty = pd.DataFrame({})
    assert_names(empty, None)
    assert_names(cudf.DataFrame(empty), None)
    assert_names(DataFrameInterfaceOnly(empty), None)

    # str-names return object arrays
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert_names(df, ["a", "b"])
    assert_names(cudf.DataFrame(df), ["a", "b"])
    assert_names(DataFrameInterfaceOnly(df), ["a", "b"])

    # Fully non-str names are fully ignored
    df_int = pd.DataFrame({0: [1, 2], 1: [3, 4]})
    assert_names(df_int, None)

    # Mixed str & non-str names warn
    df_mixed = pd.DataFrame({"a": [1, 2], 1: [3, 4]})
    with pytest.raises(
        TypeError, match="all input features have string names"
    ):
        _get_feature_names(df_mixed)


class MyModel:
    def fit(self, X, y=None):
        check_features(self, X, reset=True)
        return self

    def predict(self, X):
        check_features(self, X)


def test_check_features_reset_true():
    X = np.ones((3, 2))
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    model = MyModel().fit(X)
    assert model.n_features_in_ == 2
    assert not hasattr(model, "feature_names_in_")

    model = MyModel().fit(df)
    assert model.n_features_in_ == 2
    assert (model.feature_names_in_ == ["a", "b"]).all()

    # refitting without feature names clears
    model.fit(X)
    assert model.n_features_in_ == 2
    assert not hasattr(model, "feature_names_in_")


def test_check_n_features_in():
    X = np.ones((3, 2))
    model = MyModel().fit(X)
    # no error
    model.predict(X)

    with pytest.raises(
        ValueError,
        match="X has 3 features, but MyModel is expecting 2 features as input",
    ):
        model.predict(np.ones((3, 3)))

    # If the model doesn't have `n_features_in_` stored, the check is skipped
    # (and this doesn't error). This is necessary for some stateless
    # transformers.
    MyModel().predict(X)


def test_fit_and_predict_with_and_without_feature_names_warnings():
    X_unnamed = np.ones((3, 2))
    X_named = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    est_unnamed = MyModel().fit(X_unnamed)
    est_named = MyModel().fit(X_named)

    with assert_no_warnings():
        est_unnamed.predict(X_unnamed)

    with assert_no_warnings():
        est_named.predict(X_named)

    with pytest.warns(UserWarning, match="X has feature names"):
        est_unnamed.predict(X_named)

    with pytest.warns(
        UserWarning, match="X does not have valid feature names"
    ):
        est_named.predict(X_unnamed)


def test_feature_names_mismatch_errors():
    X = pd.DataFrame({"a": [1], "b": [2], "c": [3]})

    # Correct names are fine
    model = MyModel().fit(X)
    with assert_no_warnings():
        model.predict(X)

    # Missing column
    bad = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError, match="The feature names") as rec:
        model.predict(bad)
    assert "Feature names seen at fit time" in str(rec.value)

    # Extra column
    bad = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
    with pytest.raises(ValueError, match="The feature names") as rec:
        model.predict(bad)
    assert "Feature names unseen at fit time" in str(rec.value)

    # Reordered columns
    bad = pd.DataFrame({"a": [1], "c": [3], "b": [2]})
    with pytest.raises(ValueError, match="The feature names") as rec:
        model.predict(bad)
    assert "Feature names must be in the same order" in str(rec.value)


def test_check_input_features():
    X = pd.DataFrame({"a": [1], "b": [2], "c": [3]})

    # Correct names are fine
    model_named = MyModel().fit(X)
    model_unnamed = MyModel().fit(X.to_numpy())

    # no-args call returns feature_names_in_
    np.testing.assert_array_equal(
        check_input_features(model_named),
        model_named.feature_names_in_,
    )

    # if no feature_names_in_, names are generated like x0, x1, ...
    np.testing.assert_array_equal(
        check_input_features(model_unnamed),
        np.array(["x0", "x1", "x2"], dtype="object"),
    )

    # If `input_features` provided, that's returned if valid
    np.testing.assert_array_equal(
        check_input_features(model_named, ["a", "b", "c"]),
        model_named.feature_names_in_,
    )
    np.testing.assert_array_equal(
        check_input_features(model_unnamed, ["x", "y", "z"]),
        np.array(["x", "y", "z"]),
    )

    # Errors if feature_names_in_ defined and doesn't match
    with pytest.raises(ValueError, match="input_features is not equal to"):
        check_input_features(model_named, ["x", "y", "z"])

    # If no `feature_names_in_`, errors if input_features isn't the right size
    with pytest.raises(ValueError, match=r".*number of features \(3\), got 2"):
        check_input_features(model_unnamed, ["x", "y"])


def test_check_consistent_length():
    y3 = np.empty(3)
    y4 = np.empty(4)
    x34 = np.empty((3, 4))

    check_consistent_length()
    check_consistent_length(None)
    check_consistent_length(x34)
    check_consistent_length(x34, y3)
    check_consistent_length(x34, None, y3)

    # Supports non-normalized inputs too
    check_consistent_length(x34.tolist(), y3, None)
    check_consistent_length(pd.DataFrame(x34), cudf.Series(y3), None)

    with pytest.raises(
        ValueError,
        match=r"Found input variables with inconsistent number of samples: \[3, 4\]",
    ):
        check_consistent_length(x34, y4)

    with pytest.raises(
        TypeError, match="Expected array-like, got scalar instead"
    ):
        check_consistent_length(x34, None, np.array(1.5))

    with pytest.raises(
        TypeError, match=re.escape(f"Expected array-like, got {int} instead")
    ):
        check_consistent_length(x34, None, 1)


@pytest.mark.parametrize("device", [True, False])
@pytest.mark.parametrize("sparse_format", [None, "csr", "coo"])
def test_check_all_finite(device, sparse_format):
    def array(values, dtype=None):
        x = cp.array(values, dtype=dtype)
        if sparse_format is not None:
            x = getattr(cp_sp, f"{sparse_format}_matrix")(x)
        if not device:
            x = x.get()
        return x

    non_floating = array([True, False, True], dtype="bool")
    f32_empty = array([], dtype="float32")
    f32_good = array([1.5, -1.5, 2.5], dtype="float32")
    f32_nan = array([1.5, float("nan"), 2.5], dtype="float32")
    f32_inf = array([1.5, float("inf"), 2.5], dtype="float32")
    f32_pos_neg_inf = array(
        [1.5, -float("inf"), float("inf")], dtype="float32"
    )
    f64_both = array([[1.5, float("inf"), float("nan")]], dtype="float64")

    check_all_finite(non_floating)
    check_all_finite(f32_empty)
    check_all_finite(f32_good, allow_nan=False)
    check_all_finite(f32_good, allow_nan=True)
    check_all_finite(f32_nan, allow_nan=True)

    with pytest.raises(ValueError, match="Input X contains NaN."):
        check_all_finite(f32_nan, allow_nan=False, input_name="X")

    with pytest.raises(
        ValueError,
        match=(
            r"Input array contains infinity or a value too large for "
            r"dtype\('float32'\)."
        ),
    ):
        check_all_finite(f32_inf, allow_nan=True)

    with pytest.raises(
        ValueError,
        match=(
            r"Input array contains infinity or a value too large for "
            r"dtype\('float32'\)."
        ),
    ):
        check_all_finite(f32_pos_neg_inf)

    with pytest.raises(ValueError, match="Input array contains NaN."):
        check_all_finite(f64_both)


def test_check_all_finite_host_fallback():
    x_good = np.array([1e307] * 100, dtype="float64")
    x_nan = np.array([1e307] * 99 + [float("nan")], dtype="float64")
    x_inf = np.array([1e307] * 99 + [float("inf")], dtype="float64")
    x_both = np.array(
        [1e307] * 98 + [float("inf"), float("nan")], dtype="float64"
    )

    check_all_finite(x_good)
    check_all_finite(x_good, allow_nan=True)
    check_all_finite(x_nan, allow_nan=True)

    with pytest.raises(ValueError, match="Input array contains NaN."):
        check_all_finite(x_nan)

    with pytest.raises(ValueError, match="Input array contains infinity"):
        check_all_finite(x_inf)

    with pytest.raises(ValueError, match="Input array contains NaN."):
        check_all_finite(x_both)

    with pytest.raises(ValueError, match="Input array contains infinity"):
        check_all_finite(x_both, allow_nan=True)


def test_check_all_finite_assume_finite():
    bad = cp.array([1.5, float("nan"), 2.5], dtype="float32")
    # No errors for bad inputs if `assume_finite=True` configured
    with sklearn.config_context(assume_finite=True):
        check_all_finite(bad)


@pytest.mark.parametrize("device", [True, False])
@pytest.mark.parametrize("sparse_format", [None, "csr", "coo"])
def test_check_non_negative(device, sparse_format):
    def array(values, dtype=None):
        x = cp.array(values, dtype=dtype)
        if sparse_format is not None:
            x = getattr(cp_sp, f"{sparse_format}_matrix")(x)
        if not device:
            x = x.get()
        return x

    f32_empty = array([], dtype="float32")
    f32_good = array([0, 1, 2], dtype="float32")
    f64_good_nan = array([0, float("nan"), 1], dtype="float64")
    f32_bad = array([-1, 1, 2], dtype="float32")
    f64_bad_nan = array([-1, float("nan"), 1], dtype="float64")

    check_non_negative(f32_empty)
    check_non_negative(f32_good)
    check_non_negative(f64_good_nan)

    with pytest.raises(ValueError, match="Negative values in data"):
        check_non_negative(f32_bad)

    with pytest.raises(
        ValueError, match="Negative values in data passed to X"
    ):
        check_non_negative(f64_bad_nan, input_name="X")


def test_check_array_bad_args():
    with pytest.raises(ValueError, match="Unsupported mem_type='bad'"):
        check_array([1, 2, 3], mem_type="bad")

    with pytest.raises(ValueError, match="Unsupported order='bad'"):
        check_array([1, 2, 3], order="bad")

    # Exception raised by `np.dtype`, we don't really care what it is
    with pytest.raises(Exception, match="'bad'"):
        check_array([1, 2, 3], dtype="bad")
    with pytest.raises(Exception, match="'bad'"):
        check_array([1, 2, 3], dtype=("int32", "bad"))


@pytest.mark.parametrize(
    "func",
    [
        pytest.param(lambda x: x, id="list"),
        pytest.param(np.array, id="numpy"),
        pytest.param(cp.array, id="cupy"),
        pytest.param(
            lambda x: cp_sp.csr_matrix(cp.asarray(x)), id="cupyx.sparse"
        ),
        pytest.param(
            lambda x: sp.csr_matrix(np.asarray(x)), id="scipy.sparse"
        ),
    ],
)
def test_check_array_complex_errors(func):
    array = func([[complex(1), complex(2, 3)]])
    with pytest.raises(ValueError, match="Complex data not supported"):
        check_array(array)


@example(array=np.asarray([1, 2, 3]), mem_type=None)
@example(array=np.asarray([[1, 2], [3, 4]]), mem_type="device")
@example(array=cp.asarray([[1, 2], [3, 4]]), mem_type="host")
@given(
    array=dense_arrays(dtype="float32", order="C"),
    mem_type=st.sampled_from(["device", "host", None]),
)
def test_check_array_mem_type(array, mem_type):
    out = check_array(array, mem_type=mem_type, ensure_2d=False)
    cls = cp.ndarray if is_cuda_output(mem_type, array) else np.ndarray
    assert isinstance(out, cls)
    cp.testing.assert_allclose(as_cupy(array), as_cupy(out))


@example(
    array=cp.asarray([[1, 2], [3, 4]], order="C"),
    order="F",
    mem_type="device",
)
@example(
    array=cp.asarray([[1, 2], [3, 4]], order="F"),
    order="C",
    mem_type="device",
)
@example(
    array=np.array([[1, 2], [3, 4], [5, 6]])[::2],
    order="A",
    mem_type="host",
)
@given(
    array=dense_arrays(dtype="float32"),
    order=st.sampled_from(["C", "F", "A"]),
    mem_type=st.sampled_from(["device", "host", None]),
)
def test_check_array_order(array, order, mem_type):
    out = check_array(array, ensure_2d=False, order=order, mem_type=mem_type)
    cp.testing.assert_allclose(cp.asarray(out), as_cupy(array))
    assert_contiguity(out, order)


@example(array=[[1, 2]], mem_type="device")
@example(array=cp.array([[1, 2]]), mem_type="host")
@example(array=np.array([[1, 2]]), mem_type="device")
@example(array=cudf.DataFrame([[1, 2]]), mem_type="device")
@example(array=pd.DataFrame([[1, 2]]), mem_type="device")
@given(
    array=dense_arrays(ndim=2, dtype="int32"),
    mem_type=st.sampled_from(["device", "host", None]),
)
def test_check_array_dtype(array, mem_type):
    """Generate arrays with definitive different dtype, and ensure
    they're all cast appropriately. Checks that `dtype` is passed properly
    in all code paths."""

    # By default the input dtype is kept the same
    out = check_array(array, dtype=None, mem_type=mem_type)
    if hasattr(array, "dtype"):
        assert out.dtype == array.dtype

        out = check_array(array, dtype=array.dtype, mem_type=mem_type)
        assert out.dtype == array.dtype

        # If a sequence, no coercion done if dtype already valid
        out = check_array(
            array, dtype=("float32", array.dtype), mem_type=mem_type
        )
        assert out.dtype == array.dtype

    # Coercion to a specific dtype
    out = check_array(array, dtype="float32", mem_type=mem_type)
    assert out.dtype == "float32"
    cp.testing.assert_allclose(cp.asarray(out), as_cupy(array, "float32"))

    out = check_array(array, dtype="float64", mem_type=mem_type)
    assert out.dtype == "float64"
    cp.testing.assert_allclose(cp.asarray(out), as_cupy(array, "float64"))

    # If a sequence, the first dtype is used when coercion needed
    out = check_array(array, dtype=("float32", "float64"), mem_type=mem_type)
    assert out.dtype == "float32"


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("mem_type", ["device", "host", None])
def test_check_array_array_protocol_preserves_dtype(dtype, mem_type):
    class ArrayLike:
        def __init__(self, array):
            self.array = array

        def __array__(self, dtype=None, copy=None):
            return self.array

    array = ArrayLike(np.array([[1, 2, 3]], dtype=dtype))
    out = check_array(array, dtype=("float32", "float64"), mem_type=mem_type)
    assert out.dtype == dtype


@pytest.mark.parametrize("mem_type", ["host", "device"])
def test_check_array_non_numpy_dtype_attribute(mem_type):
    """Some array implementations (pytorch) have non-numpy-compatible `dtype`
    attributes. This test checks that these objects may still be ingested
    through other protocols, and that their unsupported `dtype` attribute
    doesn't break things."""

    class Float32:
        pass

    if mem_type == "host":

        class ArrayLike:
            """A array-like class like torch-cpu"""

            def __init__(self, array):
                self.array = np.asarray(array, dtype="float32")
                self.dtype = Float32()

            def __array__(self, dtype=None, copy=None):
                return self.array

    else:

        class ArrayLike:
            """A array-like class like torch-gpu"""

            def __init__(self, array):
                self.array = cp.asarray(array, dtype="float32")
                self.dtype = Float32()

            def __array__(self, dtype=None, copy=None):
                raise ValueError("This can't actually be called")

            @property
            def __cuda_array_interface__(self):
                return self.array.__cuda_array_interface__

    source_cls = np.ndarray if mem_type == "host" else cp.ndarray

    array = ArrayLike([[1, 2, 3]])

    out = check_array(array, dtype=("float64", "float32"), mem_type=None)
    assert out.dtype == "float32"
    assert isinstance(out, source_cls)
    out = check_array(array, dtype="float64", mem_type=None)
    assert out.dtype == "float64"
    assert isinstance(out, source_cls)


@example(mem_type="device", dtype="int32", order="C", shape=(3, 4))
@example(mem_type="host", dtype="float32", order="F", shape=(3,))
@given(
    mem_type=st.sampled_from(["device", "host"]),
    dtype=st.sampled_from(DTYPES),
    order=st.sampled_from(["C", "F"]),
    shape=st.sampled_from([(3, 4), (3,)]),
)
def test_check_array_no_copy_needed(mem_type, dtype, order, shape):
    """Ensure no copy made for fast paths."""
    xp = cp if mem_type == "device" else np
    array = xp.ones(shape, dtype=dtype, order=order)

    if len(shape) == 1:
        # all orders are equivalent for 1D inputs
        orders = ("C", "F", "A", None)
    else:
        orders = (order, "A", None)

    for dtype, mem_type, order in zip((dtype, None), (mem_type, None), orders):
        out = check_array(
            array, dtype=dtype, mem_type=mem_type, order=order, ensure_2d=False
        )
        assert xp.may_share_memory(out, array), (
            f"{dtype=}, {mem_type=}, {order=}"
        )


@example(array=[1, 2], mem_type="device")
@example(array=[1, 2], mem_type="host")
@example(array=cp.array([1, 2]), mem_type="device")
@example(array=np.array([1, 2]), mem_type=None)
@example(array=cudf.Series([1, 2]), mem_type="device")
@example(array=pd.Series([1, 2]), mem_type="host")
@example(array=pd.Series([1, 2]), mem_type=None)
@given(
    array=dense_arrays(ndim=1, dtype="int32"),
    mem_type=st.sampled_from(["device", "host", None]),
)
def test_check_array_copy(array, mem_type):
    sol = check_array(array, mem_type="host", ensure_2d=False).copy()
    out = check_array(array, mem_type=mem_type, ensure_2d=False, copy=True)
    # mutate the output
    out[0] -= 1
    res = check_array(array, mem_type="host", ensure_2d=False)
    np.testing.assert_array_equal(res, sol)


@pytest.mark.parametrize("mem_type", ["host", "device"])
@pytest.mark.parametrize("format", ["csr", "csc", "coo"])
def test_check_array_sparse_copy(mem_type, format):
    array = (cp_sp if mem_type == "device" else sp).random(
        10,
        5,
        density=0.5,
        random_state=42,
        format=format,
    )
    res = check_array(array, accept_sparse=True, copy=True)
    assert res is not array

    def ptr(x):
        return x.ctypes.data if isinstance(x, np.ndarray) else x.data.ptr

    assert ptr(res.data) != ptr(array.data)
    if format == "coo":
        assert ptr(res.row) != ptr(array.row)
        assert ptr(res.col) != ptr(array.col)
    else:
        assert ptr(res.indices) != ptr(array.indices)
        assert ptr(res.indptr) != ptr(array.indptr)


@pytest.mark.parametrize("kind", ["cudf", "pandas"])
@pytest.mark.parametrize("mem_type", ["device", "host", None])
def test_check_array_dataframe_mixed_dtypes(kind, mem_type):
    xdf = cudf if kind == "cudf" else pd

    df = xdf.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [2.5, 3.5, 4.5, 5.5, 6.5],
            "z": ["1", "2", "3.5", "4", "5"],
        }
    )
    # Non-numeric columns -> object dtype by default
    if is_cuda_output(mem_type, df):
        # cupy doesn't support object dtypes. We don't care what the exception
        # is here, just that one is raised.
        with pytest.raises(Exception, match="object"):
            check_array(df, mem_type=mem_type)
    else:
        # dtype=None does no conversion by default
        out = check_array(df, mem_type=mem_type)
        assert out.dtype == "object"

    # Can coerce all columns to specified dtype
    out = check_array(df, mem_type=mem_type, dtype=("float32", "float64"))
    assert out.dtype == "float32"
    np.testing.assert_allclose(cp.asnumpy(out), df.to_numpy(dtype="float32"))

    # Subset of numeric columns -> numeric by default
    df2 = df[["x", "y"]]
    out = check_array(df2, mem_type=mem_type)
    assert out.dtype == df2.y.dtype
    np.testing.assert_allclose(cp.asnumpy(out), df2.to_numpy())


@pytest.mark.parametrize("kind", ["cudf", "pandas"])
@pytest.mark.parametrize("mem_type", ["device", "host", None])
def test_check_array_dataframe_non_numpy_dtype(kind, mem_type):
    xdf = cudf if kind == "cudf" else pd

    df = xdf.DataFrame({"x": ["1", "2", "1", "3"]}).astype("category")
    # Can coerce all columns to specified dtype
    out = check_array(df, mem_type=mem_type, dtype=("float32", "float64"))
    assert out.dtype == "float32"
    np.testing.assert_allclose(cp.asnumpy(out), df.to_numpy(dtype="float32"))


@pytest.mark.parametrize("kind", ["cudf", "pandas"])
@pytest.mark.parametrize("ndim", [1, 2])
@pytest.mark.parametrize("mem_type", ["device", "host", None])
def test_check_array_return_index(kind, ndim, mem_type):
    xdf = cudf if kind == "cudf" else pd
    if ndim == 2:
        array = xdf.DataFrame({"x": [1, 2, 3]}, index=[1, 3, 5])
    else:
        array = xdf.Series([1, 2, 3], index=[1, 3, 5])

    out, index = check_array(
        array, ensure_2d=False, return_index=True, mem_type=mem_type
    )
    if mem_type is None:
        out_xdf = xdf
    else:
        out_xdf = cudf if mem_type == "device" else pd
    assert isinstance(index, out_xdf.Index)
    assert (cp.asnumpy(index) == np.array([1, 3, 5])).all()


def xfail_cudf_ext_dtype_bug(*args, strict=True):
    # TODO: when the test starts xpassing once cudf fixes the issue, remove and
    # simplify the parametrization below.
    return pytest.param(
        *args,
        marks=pytest.mark.xfail(
            reason=(
                "cudf's to_cupy/to_numpy fails on extension dtypes, see "
                "https://github.com/NVIDIA/cudf/issues/23723"
            ),
            strict=strict,
        ),
    )


@pytest.mark.parametrize(
    "kind,mem_type,ndim,dtype",
    [
        xfail_cudf_ext_dtype_bug("cudf", "host", 1, None, strict=False),
        xfail_cudf_ext_dtype_bug("cudf", "device", 1, None),
        xfail_cudf_ext_dtype_bug("cudf", None, 1, None),
        ("pandas", "host", 1, None),
        ("pandas", "device", 1, None),
        ("pandas", None, 1, None),
        xfail_cudf_ext_dtype_bug("cudf", "host", 1, "float32"),
        xfail_cudf_ext_dtype_bug("cudf", "device", 1, "float32"),
        xfail_cudf_ext_dtype_bug("cudf", None, 1, "float32"),
        ("pandas", "host", 1, "float32"),
        ("pandas", "device", 1, "float32"),
        ("pandas", None, 1, "float32"),
        xfail_cudf_ext_dtype_bug("cudf", "host", 2, None),
        xfail_cudf_ext_dtype_bug("cudf", "device", 2, None),
        xfail_cudf_ext_dtype_bug("cudf", None, 2, None),
        ("pandas", "host", 2, None),
        ("pandas", "device", 2, None),
        ("pandas", None, 2, None),
        xfail_cudf_ext_dtype_bug("cudf", "host", 2, "float32"),
        xfail_cudf_ext_dtype_bug("cudf", "device", 2, "float32"),
        xfail_cudf_ext_dtype_bug("cudf", None, 2, "float32"),
        ("pandas", "host", 2, "float32"),
        ("pandas", "device", 2, "float32"),
        ("pandas", None, 2, "float32"),
    ],
)
def test_check_array_dataframe_extension_dtype(kind, mem_type, ndim, dtype):
    xdf = cudf if kind == "cudf" else pd
    array = xdf.Series([1, 2, None], index=[1, 3, 5], dtype="Int32")
    if ndim == 2:
        array = array.to_frame()

    out, index = check_array(
        array,
        dtype=dtype,
        mem_type=mem_type,
        ensure_2d=False,
        ensure_all_finite="allow-nan",
        return_index=True,
    )
    if mem_type is None:
        out_xdf = xdf
    else:
        out_xdf = cudf if mem_type == "device" else pd

    if xdf is cudf:
        sol = array.to_pandas().to_numpy(dtype=dtype)
    else:
        sol = array.to_numpy(dtype=dtype)

    np.testing.assert_array_equal(cp.asnumpy(out), sol)

    assert isinstance(index, out_xdf.Index)
    assert (cp.asnumpy(index) == np.array([1, 3, 5])).all()


@pytest.mark.parametrize("kind", ["numpy", "cudf", "pandas"])
@pytest.mark.parametrize("mem_type", ["device", "host", None])
def test_check_array_object_dtype(kind, mem_type):
    array = np.array(["1.5", "2.5", "3.5"], dtype="object")
    if kind == "cudf":
        array = cudf.Series(array)
    elif kind == "pandas":
        array = pd.Series(array)

    if is_cuda_output(mem_type, array):
        # cupy doesn't support object dtypes
        with pytest.raises((ValueError, TypeError), match="object|str"):
            check_array(array, mem_type=mem_type, ensure_2d=False)
    else:
        out = check_array(array, mem_type=mem_type, ensure_2d=False)
        assert out.dtype == "object"

    # Can coerce to numeric if specified
    out = check_array(
        array, mem_type=mem_type, dtype=("float32", "float64"), ensure_2d=False
    )
    assert out.dtype == "float32"
    cp.testing.assert_allclose(cp.asarray(out), as_cupy(array, "float32"))


@pytest.mark.parametrize(
    "array",
    [
        pytest.param(cp_sp.csr_matrix(cp.array([[1.0, 0]])), id="cupy matrix"),
        pytest.param(sp.csr_matrix(np.array([[1.0, 0]])), id="scipy matrix"),
        pytest.param(sp.csr_array(np.array([[1.0, 0]])), id="scipy array"),
    ],
)
def test_check_array_sparse_not_supported(array):
    with pytest.raises(
        TypeError, match="Sparse data was passed, but dense data is required"
    ):
        check_array(array)

    with pytest.raises(
        TypeError,
        match="Sparse data was passed for X, but dense data is required",
    ):
        check_array(array, input_name="X")


@example(array=cp_sp.csr_matrix(cp.array([[1.0, 0], [0, 0]])), mem_type="host")
@example(array=sp.csr_matrix(np.array([[1.0, 0], [0, 0]])), mem_type="device")
@given(
    array=sparse_arrays(),
    mem_type=st.sampled_from(["device", "host", None]),
)
def test_check_array_sparse_input(array, mem_type):
    ns = cp_sp if is_cuda_output(mem_type, array) else sp
    # dtype=None case
    if mem_type == "device" and array.dtype.kind != "f":
        # coercing to device causes a dtype conversion even if dtype=None if
        # the input is an unsupported type. In that case we chose the float
        # type with the nearest itemsize.
        out = check_array(array, accept_sparse=True, mem_type=mem_type)
        coerced_dtype = "float32" if array.dtype.itemsize <= 4 else "float64"
        assert ns.issparse(out)
        assert out.dtype == coerced_dtype
    else:
        out = check_array(array, accept_sparse=True, mem_type=mem_type)
        assert ns.issparse(out)
        assert out.dtype == array.dtype

    # Coerce to specified dtypes
    out = check_array(
        array, accept_sparse=True, dtype="float32", mem_type=mem_type
    )
    assert ns.issparse(out)
    assert out.dtype == "float32"


def test_check_array_sparse_input_unsupported_dtype():
    """Cupy doesn't support integral types while scipy does, this checks a few
    edge cases for how we handle that"""
    host_i2 = sp.rand(100, 100, density=0.5).astype("i2")
    host_i4 = sp.rand(100, 100, density=0.5).astype("i4")
    host_i8 = sp.rand(100, 100, density=0.5).astype("i8")
    device_f4 = cp_sp.rand(100, 100, density=0.5).astype("f4")

    # Unsupported dtypes are coerced to floats by default
    for array in [host_i2, host_i4, host_i8]:
        out = check_array(array, accept_sparse=True)
        dtype = "f4" if array.dtype.itemsize <= 4 else "f8"
        assert out.dtype == dtype

    # Unsupported dtypes skipped when selecting conversion
    out = check_array(host_i4, dtype=["i4", "f8", "f4"], accept_sparse=True)
    assert out.dtype == "f8"

    # When coercing host->host, ints are supported
    out = check_array(host_i4, accept_sparse=True, mem_type=None)
    assert out.dtype == "i4"
    out = check_array(host_i2, dtype=["i4"], accept_sparse=True, mem_type=None)
    assert out.dtype == "i4"

    # When coercing device->host, ints are supported
    out = check_array(
        device_f4, dtype=["i4", "i8"], mem_type="host", accept_sparse=True
    )
    assert out.dtype == "i4"

    # When coercing to device, error if only unsupported dtypes specified
    with pytest.raises(ValueError, match="is supported by cupyx.scipy.sparse"):
        check_array(host_i4, dtype=["i4", "i8"], accept_sparse=True)
    with pytest.raises(ValueError, match="is supported by cupyx.scipy.sparse"):
        check_array(device_f4, dtype=["i4", "i8"], accept_sparse=True)
    with pytest.raises(ValueError, match="is supported by cupyx.scipy.sparse"):
        check_array(
            device_f4, mem_type=None, dtype=["i4", "i8"], accept_sparse=True
        )


@example(
    array=cp_sp.csr_matrix(cp.array([[1.0, 0], [0, 0]])),
    mem_type="host",
    format="coo",
)
@example(
    array=sp.csr_matrix(np.array([[1.0, 0], [0, 0]])),
    mem_type="device",
    format="coo",
)
@given(
    array=sparse_arrays(dtype="float32"),
    mem_type=st.sampled_from(["device", "host", None]),
    format=st.sampled_from(["csr", "csc", "coo"]),
)
def test_check_array_sparse_input_format(array, mem_type, format):
    ns = cp_sp if is_cuda_output(mem_type, array) else sp
    out = check_array(array, accept_sparse=True, mem_type=mem_type)
    assert ns.issparse(out)
    assert out.dtype == array.dtype
    # Format unchanged unless not a format cupy supports
    if array.format in ["csr", "coo", "csc", "dia"]:
        assert out.format == array.format
    else:
        assert out.format == "csr"

    # Coerce to specified formats
    out = check_array(array, accept_sparse=format, mem_type=mem_type)
    assert ns.issparse(out)
    assert out.dtype == array.dtype
    assert out.format == format

    out = check_array(array, accept_sparse=[format, "csc"], mem_type=mem_type)
    assert ns.issparse(out)
    assert out.dtype == array.dtype
    assert out.format == (
        array.format if array.format in [format, "csc"] else format
    )


@pytest.mark.parametrize("format", ["coo", "csr", "csc", "bsr", "dia"])
@pytest.mark.parametrize("mem_type", ["host", "device"])
def test_check_array_coerce_large_sparse(format, mem_type):
    array = sp.random(20, 10, density=0.5, format=format, random_state=42)
    if array.format == "coo":
        array.coords = tuple(v.astype("int64") for v in array.coords)
    elif array.format in ["csr", "csc", "bsr"]:
        array.indices = array.indices.astype("int64")
        array.indptr = array.indptr.astype("int64")
    else:  # dia
        array.offsets = array.offsets.astype("int64")

    # Sparse matrices with indices > int32 but _could_ fit in int32 are supported
    out = check_array(array, accept_sparse=True, mem_type=mem_type)
    if mem_type == "device":
        assert cp_sp.issparse(out)
        out = out.get()
    assert (out != array).nnz == 0


def test_check_array_large_sparse_errors():
    # Large sparse matrices that truly cannot fit in int32 error by default
    # This is only possible to efficiently test in CI for COO, other large
    # sparse matrices also allocate large arrays.
    array = sp.coo_matrix(
        (
            np.array([1.5]),
            (np.array([0], dtype="int64"), np.array([0], dtype="int64")),
        ),
        shape=(2**32, 10),
    )

    with pytest.raises(ValueError, match="sparse matrices with int32 indices"):
        check_array(array, accept_sparse=True)

    # No error when large sparse matrices are accepted
    out = check_array(
        array, accept_sparse=True, accept_large_sparse=True, mem_type="host"
    )
    assert out is array


@pytest.mark.skipif(
    not CUPY_SUPPORTS_LARGE_SPARSE, reason="requires cupy >= 14.1.0"
)
def test_check_array_large_sparse_cupy_supported():
    x_host = sp.coo_matrix(
        (
            np.array([1.5]),
            (np.array([0], dtype="int64"), np.array([0], dtype="int64")),
        ),
        shape=(2**32, 10),
    )
    x_device = cp_sp.coo_matrix(x_host)

    out = check_array(x_host, accept_sparse=True, accept_large_sparse=True)
    assert isinstance(out, cp_sp.coo_matrix)
    assert out.shape == x_host.shape

    out = check_array(x_device, accept_sparse=True, accept_large_sparse=True)
    assert out is x_device


@pytest.mark.skipif(
    CUPY_SUPPORTS_LARGE_SPARSE, reason="requires cupy < 14.1.0"
)
def test_check_array_large_sparse_cupy_not_supported():
    array = sp.coo_matrix(
        (
            np.array([1.5]),
            (np.array([0], dtype="int64"), np.array([0], dtype="int64")),
        ),
        shape=(2**32, 10),
    )
    with pytest.raises(
        ValueError,
        match="Sparse matrices with int64 indices require cupy >= 14.1.0",
    ):
        check_array(array, accept_sparse=True, accept_large_sparse=True)


@example(array=np.ones((3, 2)))
@example(array=cp_sp.csr_matrix(cp.ones((3, 2))))
@given(
    array=st.one_of(
        dense_arrays(dtype="float32", ndim=2, n_samples=3, n_features=2),
        sparse_arrays(dtype="float32", n_samples=3, n_features=2),
    )
)
def test_check_array_ensure_min_samples_and_ensure_min_features(array):
    with pytest.raises(
        ValueError,
        match=(
            r"Found array with 3 sample\(s\) \(shape=\(3, 2\)\) while a "
            "minimum of 6 is required"
        ),
    ):
        check_array(array, ensure_min_samples=6, accept_sparse=True)

    with pytest.raises(
        ValueError,
        match=(
            r"Found array with 2 feature\(s\) \(shape=\(3, 2\)\) while a "
            "minimum of 5 is required"
        ),
    ):
        check_array(array, ensure_min_features=5, accept_sparse=True)


@pytest.mark.parametrize(
    "func",
    [
        pytest.param(lambda x: x, id="list"),
        pytest.param(np.asarray, id="numpy"),
        pytest.param(cp.asarray, id="cupy"),
        pytest.param(cudf.Series, id="cudf"),
        pytest.param(pd.Series, id="pandas"),
        pytest.param(sp.csr_array, id="scipy.sparse.csr_array"),
    ],
)
def test_check_array_ensure_2d(func):
    array = func([1.5, 2.5, 3.5])
    if isinstance(array, (pd.Series, cudf.Series)):
        err_msg = "Expected a 2-dimensional container"
    else:
        err_msg = "Expected 2D array, got 1D array instead"
    with pytest.raises(ValueError, match=err_msg):
        check_array(array, accept_sparse=True)


def test_check_array_ensure_all_finite():
    """Tests plumbing of check_array -> check_all_finite"""
    f32_nan = np.array([[1.5, float("nan"), 2.5]], dtype="float32")
    f64_both = np.array([[1.5, float("inf"), float("nan")]], dtype="float64")

    # No errors
    check_array(f32_nan, ensure_all_finite="allow-nan")
    check_array(f64_both, ensure_all_finite=False)

    with pytest.raises(ValueError, match="Input X contains NaN."):
        check_array(f32_nan, input_name="X")

    with pytest.raises(ValueError, match="Input array contains infinity."):
        check_array(f64_both, ensure_all_finite="allow-nan")


def test_check_array_ensure_non_negative():
    """Tests plumbing of check_array -> check_non_negative"""
    array = np.array([[-1, 1, 2]], dtype="float32")

    # No error, check disabled by default
    check_array(array)

    with pytest.raises(
        ValueError, match="Negative values in data passed to X"
    ):
        check_array(array, input_name="X", ensure_non_negative=True)


@example(y=cp.asarray([[1, 2], [3, 4]], order="F"), order="C", mem_type="host")
@example(
    y=np.asarray([[1, 2], [3, 4]], order="C"), order="F", mem_type="device"
)
@given(
    y=dense_arrays(ndim=(1, 2), dtype="int64"),
    order=st.sampled_from(["C", "F", "A"]),
    mem_type=st.sampled_from(["device", "host", None]),
)
def test_check_y(y, mem_type, order):
    exp_type = cp.ndarray if is_cuda_output(mem_type, y) else np.ndarray
    out = check_y(y, mem_type=mem_type, order=order, accept_multi_output=True)
    if hasattr(y, "dtype"):
        assert out.dtype == y.dtype
    assert isinstance(out, exp_type)
    assert_contiguity(out, order)
    cp.testing.assert_allclose(as_cupy(out), as_cupy(y))

    # Check that if dtype specified it's properly converted
    out = check_y(
        y,
        mem_type=mem_type,
        order=order,
        accept_multi_output=True,
        dtype="float32",
    )
    assert out.dtype == "float32"
    assert isinstance(out, exp_type)
    assert_contiguity(out, order)
    cp.testing.assert_allclose(as_cupy(out), as_cupy(y, "float32"))

    # Sequence of dtypes also works
    out = check_y(
        y,
        accept_multi_output=True,
        dtype=("int32", "int64"),
    )
    assert out.dtype in ("int32", "int64")


@pytest.mark.parametrize("kind", ["array", "dataframe"])
def test_check_y_accept_multi_output(kind):
    if kind == "array":
        y_2d_1col = np.array([1, 2, 3])[:, None]
        y_2d = np.array([[1, 2, 3], [4, 5, 6]])
    else:
        y_2d_1col = pd.DataFrame({"x": [1, 2, 3]})
        y_2d = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    # 2d with 1 column warns, but still returns 2D
    with pytest.warns(DataConversionWarning, match="A column-vector y"):
        out = check_y(y_2d_1col)
    assert out.ndim == 2

    # 2d with multiple columns just errors
    with pytest.raises(ValueError, match="y should be a 1d"):
        check_y(y_2d)

    # With accept_multi_output=True, no cases error or warn
    out = check_y(y_2d_1col, accept_multi_output=True)
    assert out.ndim == 2
    out = check_y(y_2d, accept_multi_output=True)
    assert out.ndim == 2


@example(
    kind="cupy",
    label_dtype="int32",
    n_classes=2,
    mem_type="host",
    dtype=None,
    order=None,
)
@example(
    kind="pandas",
    label_dtype="O",
    n_classes=(2, 4),
    mem_type="device",
    dtype="int32",
    order="F",
)
@example(
    kind="pandas",
    label_dtype="string",
    n_classes=2,
    mem_type=None,
    dtype=None,
    order=None,
)
@example(
    kind="cudf",
    label_dtype="string",
    n_classes=3,
    mem_type="device",
    dtype=None,
    order=None,
)
@example(
    kind="list",
    label_dtype="O",
    n_classes=2,
    mem_type=None,
    dtype=None,
    order="A",
)
@given(
    kind=st.sampled_from(["cupy", "numpy", "pandas", "cudf", "list"]),
    label_dtype=st.sampled_from(
        ["int32", "int64", "bool", "O", "U", "string"]
    ),
    n_classes=st.sampled_from([2, 4, (3,), (2, 4)]),
    mem_type=st.sampled_from(["device", "host", None]),
    dtype=st.sampled_from([None, "int32", "int64", "float32"]),
    order=st.sampled_from(["C", "F", "A", None]),
)
def test_check_y_return_classes(
    kind, label_dtype, n_classes, mem_type, dtype, order
):
    # Construct input data
    if label_dtype in ("O", "U"):
        # cupy doesn't support these types
        assume(not (kind == "cupy" or is_cuda_output(mem_type, kind=kind)))
        labels = np.array(["a", "b", "c", "d"], dtype=label_dtype)
    elif label_dtype == "bool":
        labels = np.array([True, False])
    elif label_dtype == "string":
        # only pandas and cudf support this type
        assume(kind in ("pandas", "cudf"))
        labels = np.array(["a", "b", "c", "d"], dtype="object")
    else:
        labels = np.array([5, 10, 15, 20], dtype=label_dtype)

    rng = np.random.default_rng(42)
    if isinstance(n_classes, int):
        assume(n_classes < len(labels))
        inds = rng.integers(n_classes, size=100)
    else:
        assume(all(n < len(labels) for n in n_classes))
        inds = np.stack([rng.integers(n, size=100) for n in n_classes]).T

    y = labels.take(inds)
    if kind == "cupy":
        y = cp.asarray(y)
    elif kind == "pandas":
        y = pd.DataFrame(y) if inds.ndim == 2 else pd.Series(y)
        if label_dtype == "string":
            y = y.astype("string")
    elif kind == "cudf":
        y = cudf.DataFrame(y) if inds.ndim == 2 else cudf.Series(y)
        if label_dtype == "string":
            y = y.astype("string")
    elif kind == "list":
        y = y.tolist()

    # Construct the expected outputs
    sol = inds.flatten() if inds.ndim == 2 and inds.shape[1] == 1 else inds
    if dtype is not None:
        sol = sol.astype(dtype)
    if is_cuda_output(mem_type, value=y):
        sol = cp.asarray(sol)

    y2 = check_array(y, mem_type="host", ensure_2d=False)
    if inds.ndim == 2 and inds.shape[1] != 1:
        sol_classes = [np.unique(y2[:, i]) for i in range(y2.shape[1])]
    else:
        sol_classes = np.unique(y2)

    # Helper function so we don't have to repeat args below
    def check(accept_multi_output=False):
        return check_y(
            y,
            order=order,
            dtype=dtype,
            mem_type=mem_type,
            accept_multi_output=accept_multi_output,
            return_classes=True,
        )

    # Check the calls warn/raise appropriately and return expected types
    if inds.ndim == 2:
        if inds.shape[1] == 1:
            # For classifiers this always warns, even if multi-output accepted
            with pytest.warns(
                DataConversionWarning, match="A column-vector y"
            ):
                out, classes = check()
            assert out.ndim == 1
            assert isinstance(classes, np.ndarray)
            with pytest.warns(
                DataConversionWarning, match="A column-vector y"
            ):
                out, classes = check(True)
        else:
            with pytest.raises(ValueError, match="y should be a 1d"):
                check()
            out, classes = check(True)
    else:
        out, classes = check()

    # Assert encoded y is correct
    assert_contiguity(out, order)
    if dtype is None:
        assert out.dtype.kind in "iu"  # default to some integral type
    else:
        assert out.dtype == dtype
    assert out.shape == sol.shape
    assert isinstance(out, type(sol))
    np.testing.assert_allclose(cp.asnumpy(out), cp.asnumpy(sol))

    # Assert classes are correct
    if isinstance(sol_classes, list):
        assert isinstance(classes, list)
        assert len(classes) == len(sol_classes)
        for c, s in zip(classes, sol_classes):
            assert c.dtype == s.dtype
            assert (c == s).all()
    else:
        assert classes.dtype == sol_classes.dtype
        assert (classes == sol_classes).all()


@pytest.mark.parametrize("kind", ["numpy", "cudf"])
@pytest.mark.parametrize("label_dtype", ["int32", "O", "U"])
@pytest.mark.parametrize("multi_output", [False, True])
def test_check_y_return_classes_classes_specified(
    kind, label_dtype, multi_output
):
    # Deliberately choose non-sorted labels to ensure the provided encoding is
    # used rather than the default ordered encoding.
    if label_dtype in ("O", "U"):
        labels = np.array(["d", "c", "a", "b"], dtype=label_dtype)
    else:
        labels = np.array([20, 15, 5, 10], dtype=label_dtype)

    rng = np.random.default_rng(42)
    if multi_output:
        inds = np.stack(
            [rng.integers(3, size=100), rng.integers(4, size=100)]
        ).T
        y = labels.take(inds)
        classes = [labels[:3], labels[:4]]
        missing = [labels[:3], labels[:3]]
    else:
        inds = rng.integers(4, size=100)
        y = labels.take(inds)
        classes = labels
        missing = labels[:3]
    sol = cp.asarray(inds)

    if kind == "cudf":
        y = cudf.DataFrame(y) if multi_output else cudf.Series(y)
    else:
        assert kind == "numpy"

    res, res_classes = check_y(
        y, return_classes=classes, accept_multi_output=multi_output
    )

    # Assert encoded y is correct
    assert res.dtype.kind in "iu"  # default to some integral type
    assert res.shape == sol.shape
    assert isinstance(res, cp.ndarray)
    cp.testing.assert_allclose(res, sol)

    # Assert classes is returned as is
    if multi_output:
        assert all(a is b for a, b in zip(res_classes, classes))
    else:
        assert res_classes is classes

    # Missing a class raises
    if label_dtype == "U":
        # cudf normalizes numpy unicode labels through string categories.
        msg = (
            r"The target label\(s\) \[.*b.*\] in y do not exist in the initial"
        )
    else:
        msg = re.escape(
            f"The target label(s) {labels[3:]!s} in y do not exist in the initial "
            f"classes {labels[:3]!s}"
        )
    with pytest.raises(ValueError, match=msg):
        check_y(y, return_classes=missing, accept_multi_output=multi_output)


def test_check_y_return_classes_bad_type():
    y = np.array(["a", "b", "a", "b"])
    y2 = np.array([["a", "x"], ["b", "y"], ["c", "x"]])
    msg_1d = "Expected `return_classes` to be a numpy array"
    msg_2d = "Expected `return_classes` to be a list of 2 numpy arrays"

    with pytest.raises(ValueError, match=msg_1d):
        check_y(y, return_classes="bad")

    with pytest.raises(ValueError, match=msg_2d):
        check_y(y2, return_classes="bad", accept_multi_output=True)

    with pytest.raises(ValueError, match=msg_2d):
        check_y(y2, return_classes=["bad", "bad"], accept_multi_output=True)

    with pytest.raises(ValueError, match=msg_2d):
        check_y(
            y2,
            return_classes=[np.array(["a", "b", "c"])],
            accept_multi_output=True,
        )


@pytest.mark.parametrize(
    "array",
    [
        pytest.param(
            np.array([1.0, 2.0, 1.0], dtype="float32"), id="small-float32"
        ),
        pytest.param(
            np.array([2**24, 2**24 + 1, 2**24 + 2], dtype="float64"),
            id="big-float64",
        ),
    ],
)
def test_check_y_classifier_floating_input_accepted(array):
    # integral floating values are accepted
    out, classes = check_y(array, return_classes=True)
    assert classes.dtype == array.dtype
    sol_classes, sol_ind = np.unique(array, return_inverse=True)
    np.testing.assert_array_equal(classes, sol_classes)
    np.testing.assert_array_equal(cp.asnumpy(out), sol_ind)


def test_check_y_classifier_floating_input_errors():
    # Non integral values error
    has_nan = cp.array([1.0, float("nan"), 3.0])
    has_inf = np.array([1.0, float("inf"), 3.0])
    non_integral = np.array([1.5, 2.5, 3.5])
    with pytest.raises(ValueError, match="Input y contains NaN."):
        check_y(has_nan, return_classes=True)
    with pytest.raises(ValueError, match="Input y contains infinity"):
        check_y(has_inf, return_classes=True)
    with pytest.raises(ValueError, match="Unknown label type: continuous"):
        check_y(non_integral, return_classes=True)
    # No issue if ensure_discrete_classes=False
    check_y(has_nan, return_classes=True, ensure_discrete_classes=False)
    check_y(has_inf, return_classes=True, ensure_discrete_classes=False)
    check_y(non_integral, return_classes=True, ensure_discrete_classes=False)


def test_check_y_classifier_on_non_str_object():
    bad = np.array([1, 2, 0], dtype=object)
    with pytest.raises(ValueError, match="Unknown label type: unknown"):
        check_y(bad, return_classes=True)


@pytest.mark.parametrize("return_classes", [False, True])
@pytest.mark.parametrize("mem_type", ["device", "host"])
def test_check_y_return_index(return_classes, mem_type):
    y = pd.Series([1, 2, 1, 3], index=[10, 20, 30, 40])

    out = check_y(
        y, return_index=True, return_classes=return_classes, mem_type=mem_type
    )

    if return_classes:
        y, classes, index = out
        np.testing.assert_array_equal(classes, np.array([1, 2, 3]))
    else:
        y, index = out

    if mem_type == "host":
        assert isinstance(index, pd.Index)
    else:
        assert isinstance(index, cudf.Index)

    assert (cudf.Index(index) == cudf.Index([10, 20, 30, 40])).all()


@pytest.mark.parametrize(
    "mem_type, dtype",
    [
        ("device", "int32"),
        ("host", "int32"),
        ("host", "object"),
    ],
)
@pytest.mark.parametrize("return_classes", [False, True])
def test_check_y_empty(mem_type, dtype, return_classes):
    xp = cp if mem_type == "device" else np
    array = xp.array([], dtype=dtype)
    if return_classes:
        y, classes = check_y(array, mem_type=None, return_classes=True)
        assert classes.dtype == dtype
        assert classes.size == 0
        assert y.dtype.kind in "iu"
        assert y.size == 0
    else:
        y = check_y(array, mem_type=None)
        assert y.dtype == array.dtype
        assert y.size == 0


def test_check_y_none():
    with pytest.raises(ValueError, match="This estimator requires y"):
        check_y(None)


@example(
    sample_weight=cp.array([1.5, 2.5, 3.5], dtype="float32"),
    dtype=None,
    order="C",
    mem_type="host",
)
@example(
    sample_weight=np.array([1.5, 2.5, 3.5], dtype="float64")[::2],
    dtype="float32",
    order="A",
    mem_type="device",
)
@given(
    sample_weight=dense_arrays(ndim=1, dtype=("int32", "float32", "float64")),
    dtype=st.sampled_from(["float32", "float64", None]),
    order=st.sampled_from(["C", "F", "A", None]),
    mem_type=st.sampled_from(["device", "host", None]),
)
def test_check_sample_weight(sample_weight, dtype, order, mem_type):
    exp_type = (
        cp.ndarray if is_cuda_output(mem_type, sample_weight) else np.ndarray
    )
    out = check_sample_weight(
        sample_weight, dtype=dtype, order=order, mem_type=mem_type
    )
    if dtype is None and hasattr(sample_weight, "dtype"):
        assert out.dtype == sample_weight.dtype
    elif dtype is not None:
        assert out.dtype == dtype
    assert isinstance(out, exp_type)
    assert_contiguity(out, order)
    cp.testing.assert_allclose(as_cupy(out), as_cupy(sample_weight))


@pytest.mark.parametrize("shape", [(4, 5), (4, 1)])
def test_check_sample_weight_errors_2d(shape):
    bad = np.ones(shape)
    with pytest.raises(
        ValueError, match="Sample weights must be 1D array or scalar"
    ):
        check_sample_weight(bad)


@pytest.mark.parametrize("mem_type", ["device", "host"])
@pytest.mark.parametrize("ensure_non_negative", [False, True])
def test_check_sample_weight_empty(mem_type, ensure_non_negative):
    xp = cp if mem_type == "device" else np
    array = xp.array([], dtype="float32")
    out = check_sample_weight(array, ensure_non_negative=ensure_non_negative)
    assert out.dtype == array.dtype
    assert out.size == 0


def test_check_sample_weight_scalar_or_none():
    assert check_sample_weight(None) is None
    assert check_sample_weight(1.5) is None
    assert check_sample_weight(np.float32(1.0)) is None


@pytest.mark.parametrize(
    "sample_weight",
    [
        pytest.param(-1, id="scalar"),
        pytest.param(np.full(3, -1), id="numpy"),
        pytest.param(cp.full(3, -1), id="cupy"),
    ],
)
def test_check_sample_weight_ensure_non_negative(sample_weight):
    with pytest.raises(
        ValueError,
        match="Negative values in data passed to sample_weight",
    ):
        check_sample_weight(sample_weight, ensure_non_negative=True)


@pytest.mark.parametrize(
    "sample_weight",
    [
        pytest.param(0, id="scalar"),
        pytest.param(np.zeros(3), id="numpy"),
        pytest.param(cp.zeros(3), id="cupy"),
    ],
)
def test_check_sample_weight_all_zero(sample_weight):
    with pytest.raises(
        ValueError,
        match="Sample weights must contain at least one non-zero number",
    ):
        check_sample_weight(sample_weight)


@pytest.mark.parametrize("value", ["NaN", "infinity"])
def test_check_sample_weight_non_finite(value):
    scalar = float(value)
    array = np.array([1.5, scalar, 2.5])
    msg = f"Input sample_weight contains {value}"
    with pytest.raises(ValueError, match=msg):
        check_sample_weight(scalar)

    with pytest.raises(ValueError, match=msg):
        check_sample_weight(array)


def test_check_inputs_X():
    model = MyModel()
    X = np.arange(6).reshape((3, 2))

    X2 = check_inputs(model, X, reset=True)
    assert model.n_features_in_ == 2
    cp.testing.assert_array_equal(X2, cp.asarray(X))

    with pytest.raises(
        ValueError,
        match="X has 3 features, but MyModel is expecting 2 features as input",
    ):
        check_inputs(model, np.ones((3, 3)))


def test_check_inputs_X_y():
    model = MyModel()

    X = np.arange(6, dtype="float32").reshape((3, 2))
    y = cp.arange(3, dtype="int32")

    X2, y2 = check_inputs(model, X, y, reset=True)
    assert model.n_features_in_ == 2
    assert X2.dtype == "float32"
    # y defaults to X output dtype
    assert y2.dtype == "float32"
    cp.testing.assert_array_equal(X2, cp.asarray(X))
    cp.testing.assert_array_equal(y2, y.astype("float32"))

    # check y_dtype overrides
    _, y2 = check_inputs(model, X, y, y_dtype="int32", reset=True)
    assert y2.dtype == "int32"


def test_check_inputs_X_y_sample_weight():
    model = MyModel()
    X = np.arange(6, dtype="float32").reshape((3, 2))
    y = cp.ones(3, dtype="int32")
    sample_weight = [10, 20, 30]

    X2, y2, sample_weight2 = check_inputs(
        model, X, y, sample_weight, reset=True
    )
    assert model.n_features_in_ == 2
    assert X2.dtype == "float32"
    assert y2.dtype == "float32"
    # sample_weight defaults to X output dtype
    assert sample_weight2.dtype == "float32"
    cp.testing.assert_array_equal(X2, cp.asarray(X))
    cp.testing.assert_array_equal(y2, y.astype("float32"))
    cp.testing.assert_array_equal(
        sample_weight2, cp.array(sample_weight, dtype="float32")
    )

    # check sample_weight_dtype overrides
    _, _, sample_weight2 = check_inputs(
        model, X, y, sample_weight, sample_weight_dtype="float64"
    )
    assert sample_weight2.dtype == "float64"

    # sample_weight=None is supported
    _, _, sample_weight2 = check_inputs(model, X, y, None)
    assert sample_weight2 is None


def test_check_inputs_check_consistent_length():
    model = MyModel()

    x34 = np.ones((3, 4))
    y3 = np.ones(3)
    y4 = np.ones(4)

    with pytest.raises(
        ValueError,
        match=r"Found input variables with inconsistent number of samples: \[3, 4\]",
    ):
        check_inputs(model, x34, y4, None, reset=True)

    with pytest.raises(
        ValueError,
        match=r"Found input variables with inconsistent number of samples: \[3, 4\]",
    ):
        check_inputs(model, x34, sample_weight=y4, reset=True)

    with pytest.raises(
        ValueError,
        match=r"Found input variables with inconsistent number of samples: \[3, 3, 4\]",
    ):
        check_inputs(model, x34, y3, sample_weight=y4, reset=True)


def test_check_inputs_return_classes():
    model = MyModel()
    X = cp.ones((3, 2), "float32")
    y = np.array(["a", "b", "a"], dtype="O")

    _, y2, classes = check_inputs(model, X, y, return_classes=True, reset=True)
    # y defaults to X dtype
    assert y2.dtype == "float32"
    cp.testing.assert_array_equal(y2, cp.array([0, 1, 0]))
    np.testing.assert_array_equal(classes, np.array(["a", "b"], dtype="O"))

    # check y_dtype=None results in an integral type
    _, y2, classes = check_inputs(
        model, X, y, y_dtype=None, return_classes=True, reset=True
    )
    assert y2.dtype.kind in "iu"
    cp.testing.assert_array_equal(y2, cp.array([0, 1, 0]))
    np.testing.assert_array_equal(classes, np.array(["a", "b"], dtype="O"))

    # check y_dtype overrides
    _, y2, classes = check_inputs(
        model, X, y, y_dtype="float64", return_classes=True, reset=True
    )
    assert y2.dtype == "float64"
    cp.testing.assert_array_equal(y2, cp.array([0, 1, 0], dtype="float64"))

    # Can explicitly specify classes
    _, y2, classes = check_inputs(
        model, X, y, return_classes=np.array(["b", "a"], dtype="O"), reset=True
    )
    cp.testing.assert_array_equal(y2, cp.array([1, 0, 1]))
    np.testing.assert_array_equal(classes, np.array(["b", "a"], dtype="O"))


def test_check_inputs_return_index():
    model = MyModel()
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=[10, 20, 30])
    sample_weight = [0, 1, 0]

    X2, sample_weight2, index = check_inputs(
        model, X, sample_weight=sample_weight, return_index=True, reset=True
    )
    assert X2.dtype == sample_weight2.dtype
    cp.testing.assert_array_equal(X2, cp.asarray(X.to_numpy()))
    cp.testing.assert_array_equal(
        sample_weight2, cp.asarray(sample_weight, dtype=X2.dtype)
    )
    assert isinstance(index, cudf.Index)
    assert (index == cudf.Index([10, 20, 30])).all()


@example(kind="numpy", dtype="U", shape=10)
@example(kind="numpy", dtype="f2", shape=(10, 1))
@example(kind="numpy", dtype="U", shape=(10, 2))
@example(kind="list", dtype="i8", shape=10)
@example(kind="cupy", dtype="i4", shape=(10, 1))
@example(kind="cupy", dtype="f2", shape=(10, 2))
@example(kind="cudf", dtype="f4", shape=10)
@example(kind="cudf", dtype="i4", shape=(10, 1))
@example(kind="cudf", dtype="i4", shape=(10, 2))
@example(kind="pandas", dtype="O", shape=(10, 1))
@example(kind="pandas", dtype="O", shape=(10, 2))
@example(kind="pandas", dtype="f2", shape=10)
@example(kind="pandas", dtype="f2", shape=(10, 2))
@given(
    kind=st.sampled_from(["cudf", "pandas", "numpy", "cupy", "list"]),
    dtype=st.sampled_from(["i4", "i8", "f2", "f4", "f8", "U", "O"]),
    shape=st.sampled_from([10, (10, 1), (10, 2)]),
)
@pytest.mark.parametrize("ensure_ndim", [1, 2, None])
@pytest.mark.parametrize("coerce_ndim", [True, False, "warn"])
def test_check_cudf(kind, ensure_ndim, coerce_ndim, dtype, shape):
    rng = np.random.default_rng(42)
    dtype = np.dtype(dtype)

    # Construct input data
    if dtype in ("O", "U"):
        # cupy doesn't support these types
        assume(kind != "cupy")
        data = rng.choice(["a", "b", "c", "d"], size=shape).astype(dtype)
    elif dtype.kind == "i":
        data = rng.integers(0, 100, size=shape, dtype=dtype)
    else:
        # cudf doesn't support float16
        assume(not (kind == "cudf" and dtype == "float16"))
        data = rng.random(shape).astype(dtype)

    if kind == "cupy":
        array = cp.asarray(data)
    elif kind == "numpy":
        array = data
    elif kind == "list":
        array = data.tolist()
    elif kind == "pandas":
        array = pd.Series(data) if data.ndim == 1 else pd.DataFrame(data)
    elif kind == "cudf":
        array = cudf.Series(data) if data.ndim == 1 else cudf.DataFrame(data)

    should_error = False
    ctx = assert_no_warnings()
    if ensure_ndim == 1 and data.ndim == 2:
        if not coerce_ndim or data.shape[1] != 1:
            ctx = pytest.raises(
                ValueError, match="myarray should be a 1d array"
            )
            should_error = True
        elif coerce_ndim == "warn":
            ctx = pytest.warns(
                DataConversionWarning, match="A column-vector myarray"
            )
    elif ensure_ndim == 2 and data.ndim == 1:
        if not coerce_ndim:
            if isinstance(array, (cudf.Series, pd.Series)):
                ctx = pytest.raises(
                    ValueError, match="Expected a 2-dimensional container"
                )
            else:
                ctx = pytest.raises(ValueError, match="Expected 2D array")
            should_error = True
        elif coerce_ndim == "warn":
            ctx = pytest.warns(
                DataConversionWarning, match="A 1d array myarray"
            )

    with ctx:
        out = check_cudf(
            array,
            ensure_ndim=ensure_ndim,
            coerce_ndim=coerce_ndim,
            input_name="myarray",
        )
    if not should_error:
        if ensure_ndim == 2:
            assert isinstance(out, cudf.DataFrame)
            if data.ndim != 2:
                data = data.reshape(-1, 1)
        elif ensure_ndim == 1:
            assert isinstance(out, cudf.Series)
            if data.ndim != 1:
                data = data.ravel()
        else:
            assert isinstance(out, (cudf.Series, cudf.DataFrame))

        res = out.to_numpy(
            dtype=("float32" if data.dtype == "float16" else data.dtype)
        )
        np.testing.assert_array_equal(res, data)


def test_check_cudf_bad_args():
    with pytest.raises(ValueError, match="Unsupported ensure_ndim='bad'"):
        check_cudf([1, 2, 3], ensure_ndim="bad")

    with pytest.raises(ValueError, match="Unsupported coerce_ndim='bad'"):
        check_cudf([1, 2, 3], coerce_ndim="bad")

    with pytest.raises(
        ValueError, match="ensure_min_features=2 requires ensure_ndim=2"
    ):
        check_cudf([1, 2, 3], ensure_ndim=1, ensure_min_features=2)


@pytest.mark.parametrize(
    "xdf", [pytest.param(pd, id="pandas"), pytest.param(cudf, id="cudf")]
)
@pytest.mark.parametrize("ensure_ndim", [1, 2, None])
def test_check_cudf_persists_index(xdf, ensure_ndim):
    index = xdf.Index([20, 10, 30])
    df = xdf.DataFrame({"x": [1, 2, 3]}, index=index)

    out = check_cudf(df, ensure_ndim=ensure_ndim, coerce_ndim=True)
    assert (out.index == cudf.Index(index)).all()


def test_check_cudf_coerces_numeric_objects():
    x = np.array([1.0, 2.0, 3.0], dtype=object)
    s = check_cudf(x, ensure_ndim=1)
    assert (s == cudf.Series([1.0, 2.0, 3.0])).all()
    assert s.dtype == "float64"


@pytest.mark.parametrize("kind", ["list", "array", "array-like"])
def test_check_cudf_mixed_dtype_array_like_inputs(kind):
    class ArrayLike:
        def __init__(self, array):
            self.array = array

        def __array__(self, dtype=None, copy=None):
            return self.array

    data = [
        [1, 2.0, "x", "a", None, np.nan],
        [2, 4.0, "y", None, None, np.nan],
    ]
    sol = cudf.DataFrame(data, nan_as_null=True)
    if kind == "list":
        X = data
    elif kind == "array":
        X = np.array(data, dtype=object)
    else:
        assert kind == "array-like"
        X = ArrayLike(np.array(data, dtype=object))

    res = check_cudf(X)
    cudf.testing.assert_frame_equal(res, sol)


@pytest.mark.parametrize("kind", ["list", "numpy-object", "numpy-float"])
def test_check_cudf_nan_as_null(kind):
    """cudf's default is to treat NaN as NULL in inputs, but that default
    changes when `cudf.pandas` is active. Here we check that the code paths in
    `check_cudf` hardcode the `nan_as_null` configuration so that behavior
    doesn't change if cudf.pandas is active."""

    if kind == "list":
        X = [[1, None], [np.nan, 1]]
    elif kind == "numpy-object":
        X = np.array([[1, None], [np.nan, 1]], dtype=object)
    else:
        assert kind == "numpy-float"
        X = np.array([[1, np.nan], [np.nan, 1]], dtype="float32")

    res = check_cudf(X, ensure_ndim=None)
    vals = cudf.Series([1, np.nan], nan_as_null=True)
    assert res.iloc[:, 0].isin(vals).all()
    assert res.iloc[:, 1].isin(vals).all()


@pytest.mark.parametrize(
    "kind, dtype",
    [
        ("list", None),
        ("numpy", "O"),
        ("numpy", "S"),
        ("pandas", "O"),
        ("pandas", "S"),
    ],
)
@pytest.mark.parametrize("ndim", [1, 2])
def test_check_cudf_bytes_dtype_errors(kind, dtype, ndim):
    X = [b"a", b"b"] if ndim == 1 else [[b"a", b"b"], [b"c", b"d"]]
    if kind == "numpy":
        X = np.array(X, dtype=dtype)
    elif kind == "pandas":
        X = (pd.DataFrame if ndim == 2 else pd.Series)(X).astype(dtype)

    with pytest.raises(TypeError, match="X with bytes dtype is not supported"):
        check_cudf(X, ensure_ndim=None, input_name="X")


@pytest.mark.parametrize("kind", ["list", "array"])
def test_check_cudf_unsupported_object_inputs(kind):
    X = [[{"x": 1}, 1], [1, 2]]
    if kind == "array":
        X = np.array(X, dtype=object)

    with pytest.raises(
        TypeError,
        match="An object dtype X argument must be composed of",
    ):
        check_cudf(X, input_name="X")
