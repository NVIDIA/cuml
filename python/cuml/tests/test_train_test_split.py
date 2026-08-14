# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import cudf
import cupy as cp
import numpy as np
import pandas as pd
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from cuml.datasets import make_classification
from cuml.model_selection import train_test_split
from cuml.testing.strategies import cuml_array_inputs

test_seeds = ["int", "cupy", "numpy"]


@pytest.fixture(
    params=[
        ("cupy", cp.asarray),
        ("cudf", cudf),
        ("pandas", pd),
    ],
    ids=["to_cupy", "to_cudf", "to_pandas"],
)
def convert_to_type(request):
    backend_name, array_constructor = request.param
    if array_constructor in (cudf, pd):

        def ctor(X):
            if isinstance(X, cp.ndarray) and array_constructor == pd:
                X = X.get()

            if X.ndim > 1:
                return array_constructor.DataFrame(X)
            else:
                return array_constructor.Series(X)

        yield (backend_name, ctor)

    else:
        yield (backend_name, array_constructor)


@pytest.mark.parametrize("train_size", [0.2, 0.6, 0.8])
@pytest.mark.parametrize("shuffle", [True, False])
def test_split_dataframe(train_size, shuffle):
    X = cudf.DataFrame({"x": range(100)})
    y = cudf.Series(([0] * (100 // 2)) + ([1] * (100 // 2)))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, shuffle=shuffle
    )
    assert len(X_train) == len(y_train) == pytest.approx(train_size * len(X))
    assert (
        len(X_test) == len(y_test) == pytest.approx((1 - train_size) * len(X))
    )
    assert all(X_train.index.to_pandas() == y_train.index.to_pandas())
    assert all(X_test.index.to_pandas() == y_test.index.to_pandas())

    X_reconstructed = cudf.concat([X_train, X_test]).sort_values(by=["x"])
    y_reconstructed = cudf.concat([y_train, y_test]).sort_values()

    assert all(X_reconstructed.reset_index(drop=True) == X)
    out = y_reconstructed.reset_index(drop=True).to_numpy() == y.to_numpy()
    assert all(out)


@pytest.mark.parametrize("y_type", ["cudf", "cupy"])
def test_split_dataframe_array(y_type):
    X = cudf.DataFrame({"x": range(100)})
    y = cudf.Series(([0] * (100 // 2)) + ([1] * (100 // 2)))
    if y_type == "cupy":
        X_train, X_test, y_train, y_test = train_test_split(X, y.values)
        assert isinstance(X_train, cudf.DataFrame)
        assert isinstance(X_test, cudf.DataFrame)
        assert isinstance(y_train, cp.ndarray)
        assert isinstance(y_test, cp.ndarray)
    elif y_type == "cudf":
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        assert isinstance(X_train, cudf.DataFrame)
        assert isinstance(X_test, cudf.DataFrame)
        assert isinstance(y_train, cudf.Series)
        assert isinstance(y_test, cudf.Series)


def test_split_size_mismatch():
    X = cudf.DataFrame({"x": range(3)})
    y = cudf.Series([0, 1])

    with pytest.raises(ValueError):
        train_test_split(X, y)


@pytest.mark.parametrize("train_size", [1.2, 100])
def test_split_invalid_proportion(train_size):
    X = cudf.DataFrame({"x": range(10)})
    y = cudf.Series([0] * 10)

    with pytest.raises(ValueError):
        train_test_split(X, y, train_size=train_size)


@pytest.mark.parametrize("seed_type", test_seeds)
def test_random_state(seed_type):
    for i in range(10):
        seed_n = np.random.randint(0, int(1e9))
        if seed_type == "int":
            seed = seed_n
        if seed_type == "cupy":
            seed = cp.random.RandomState(seed=seed_n)
        if seed_type == "numpy":
            seed = np.random.RandomState(seed=seed_n)
        X = cudf.DataFrame({"x": range(100)})
        y = cudf.Series(([0] * (100 // 2)) + ([1] * (100 // 2)))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=seed
        )

        if seed_type == "cupy":
            seed = cp.random.RandomState(seed=seed_n)
        if seed_type == "numpy":
            seed = np.random.RandomState(seed=seed_n)

        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            X, y, random_state=seed
        )

        assert X_train.equals(X_train2)
        assert X_test.equals(X_test2)
        assert y_train.equals(y_train2)
        assert y_test.equals(y_test2)


@pytest.mark.parametrize(
    "X, y",
    [
        (np.arange(-100, 0), np.arange(100)),
        (
            np.zeros((100, 10)) + np.arange(100).reshape(100, 1),
            np.arange(100).reshape(100, 1),
        ),
    ],
)
@pytest.mark.parametrize("sizes", [(0.6, 0.4), (0.8, 0.2), (None, 0.25)])
@pytest.mark.parametrize("shuffle", [True, False])
def test_array_split(X, y, convert_to_type, sizes, shuffle):
    test_size, train_size = sizes

    backend_name, array_constructor = convert_to_type

    X = array_constructor(X)
    y = array_constructor(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_size,
        test_size=test_size,
        shuffle=shuffle,
        random_state=0,
    )

    if backend_name == "cupy":
        assert isinstance(X_train, cp.ndarray)
        assert isinstance(X_test, cp.ndarray)
        assert isinstance(y_train, cp.ndarray)
        assert isinstance(y_test, cp.ndarray)

    if backend_name == "cudf":
        # cudf input should produce cudf output
        expected_x_type = cudf.DataFrame if X.ndim > 1 else cudf.Series
        expected_y_type = cudf.DataFrame if y.ndim > 1 else cudf.Series
        assert isinstance(X_train, expected_x_type)
        assert isinstance(X_test, expected_x_type)
        assert isinstance(y_train, expected_y_type)
        assert isinstance(y_test, expected_y_type)

    if backend_name == "pandas":
        # pandas input should produce pandas output (preserves input type)
        expected_x_type = pd.DataFrame if X.ndim > 1 else pd.Series
        expected_y_type = pd.DataFrame if y.ndim > 1 else pd.Series
        assert isinstance(X_train, expected_x_type)
        assert isinstance(X_test, expected_x_type)
        assert isinstance(y_train, expected_y_type)
        assert isinstance(y_test, expected_y_type)

    if train_size is not None:
        assert X_train.shape[0] == X.shape[0] * train_size
        assert y_train.shape[0] == y.shape[0] * train_size

    if test_size is not None:
        assert X_test.shape[0] == X.shape[0] * test_size
        assert y_test.shape[0] == y.shape[0] * test_size

    if shuffle is None:
        assert X_train == X[0:train_size]
        assert y_train == y[0:train_size]
        assert X_test == X[-1 * test_size :]
        assert y_test == y[-1 * test_size :]

        X_rec = cp.sort(cp.concatenate(X_train, X_test))
        y_rec = cp.sort(cp.concatenate(y_train, y_test))

        assert X_rec == X
        assert y_rec == y


def test_default_values():
    X = np.zeros((100, 10)) + np.arange(100).reshape(100, 1)
    y = np.arange(100).reshape(100, 1)

    X = cp.asarray(X)
    y = cp.asarray(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    assert isinstance(X_train, cp.ndarray)
    assert isinstance(X_test, cp.ndarray)
    assert isinstance(y_train, cp.ndarray)
    assert isinstance(y_test, cp.ndarray)

    assert X_train.shape[0] == X.shape[0] * 0.75
    assert y_train.shape[0] == y.shape[0] * 0.75

    assert X_test.shape[0] == X.shape[0] * 0.25
    assert y_test.shape[0] == y.shape[0] * 0.25


@pytest.mark.parametrize("sizes", [(0.6, 0.4), (0.8, 0.2), (None, 0.25)])
@pytest.mark.parametrize("shuffle", [True, False])
def test_split_df_single_argument(sizes, shuffle):
    test_size, train_size = sizes

    X = cudf.DataFrame({"x": range(50)})
    X_train, X_test = train_test_split(
        X,
        train_size=train_size,
        test_size=test_size,
        shuffle=shuffle,
        random_state=0,
    )
    if train_size is not None:
        assert X_train.shape[0] == (int)(X.shape[0] * train_size)

    if test_size is not None:
        assert X_test.shape[0] == (int)(X.shape[0] * test_size)


@pytest.mark.parametrize(
    "X",
    [np.arange(-100, 0), np.zeros((100, 10)) + np.arange(100).reshape(100, 1)],
)
@pytest.mark.parametrize("sizes", [(0.6, 0.4), (0.8, 0.2), (None, 0.25)])
@pytest.mark.parametrize("shuffle", [True, False])
def test_split_array_single_argument(X, convert_to_type, sizes, shuffle):
    test_size, train_size = sizes
    backend_name, array_constructor = convert_to_type

    X = array_constructor(X)

    X_train, X_test = train_test_split(
        X,
        train_size=train_size,
        test_size=test_size,
        shuffle=shuffle,
        random_state=0,
    )

    if backend_name == "cupy":
        assert isinstance(X_train, cp.ndarray)
        assert isinstance(X_test, cp.ndarray)

    if backend_name == "cudf":
        # cudf input should produce cudf output
        expected_type = cudf.DataFrame if X.ndim > 1 else cudf.Series
        assert isinstance(X_train, expected_type)
        assert isinstance(X_test, expected_type)

    if backend_name == "pandas":
        # pandas input should produce pandas output (preserves input type)
        expected_type = pd.DataFrame if X.ndim > 1 else pd.Series
        assert isinstance(X_train, expected_type)
        assert isinstance(X_test, expected_type)

    if train_size is not None:
        assert X_train.shape[0] == (int)(X.shape[0] * train_size)

    if test_size is not None:
        assert X_test.shape[0] == (int)(X.shape[0] * test_size)

    if shuffle is None:
        assert X_train == X[0:train_size]
        assert X_test == X[-1 * test_size :]

        X_rec = cp.sort(cp.concatenate(X_train, X_test))

        assert X_rec == X


@pytest.mark.parametrize("sizes", [(0.6, 0.4), (0.8, 0.2), (None, 0.25)])
def test_stratified_split(convert_to_type, sizes):
    test_size, train_size = sizes

    backend_name, array_constructor = convert_to_type

    # For more tolerance and reliable estimates
    X, y = make_classification(n_samples=10000)

    X = array_constructor(X)
    y = array_constructor(y)

    def counts(y):
        _, y_indices = cp.unique(y, return_inverse=True)
        class_counts = cp.bincount(y_indices)
        total = cp.sum(class_counts)
        percent_counts = []
        for count in class_counts:
            percent_counts.append(
                cp.around(float(count) / total.item(), decimals=2).item()
            )
        return percent_counts

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, test_size=test_size, stratify=y
    )

    original_counts = counts(y)
    split_counts = counts(y_train)
    assert cp.isclose(
        original_counts, split_counts, equal_nan=False, rtol=0.1
    ).all()
    if backend_name == "cupy":
        assert isinstance(X_train, cp.ndarray)
        assert isinstance(X_test, cp.ndarray)

    if backend_name == "cudf":
        # cudf input should produce cudf output
        expected_type = cudf.DataFrame if X.ndim > 1 else cudf.Series
        assert isinstance(X_train, expected_type)
        assert isinstance(X_test, expected_type)

    if backend_name == "pandas":
        # pandas input should produce pandas output (preserves input type)
        expected_type = pd.DataFrame if X.ndim > 1 else pd.Series
        assert isinstance(X_train, expected_type)
        assert isinstance(X_test, expected_type)


@pytest.mark.parametrize("seed_type", test_seeds)
def test_stratified_random_seed(seed_type):
    for i in range(10):
        seed_n = np.random.randint(0, int(1e9))
        if seed_type == "int":
            seed = seed_n
        if seed_type == "cupy":
            seed = cp.random.RandomState(seed=seed_n)
        if seed_type == "numpy":
            seed = np.random.RandomState(seed=seed_n)
        X = cudf.DataFrame({"x": range(100)})
        y = cudf.Series(([0] * (100 // 2)) + ([1] * (100 // 2)))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=seed, stratify=y
    )

    if seed_type == "cupy":
        seed = cp.random.RandomState(seed=seed_n)
    if seed_type == "numpy":
        seed = np.random.RandomState(seed=seed_n)

    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X, y, random_state=seed, stratify=y
    )

    assert X_train.equals(X_train2)
    assert X_test.equals(X_test2)
    assert y_train.equals(y_train2)
    assert y_test.equals(y_test2)

    # Ensure that data is shuffled
    assert not (X.head().index.values == X_train.head().index.values).all()

    def monotonic_inc(x):
        dx = cp.diff(x.values, axis=0)
        return cp.all(dx == 1)

    assert not monotonic_inc(X_train)


@pytest.mark.parametrize("sizes", [(0.6, 0.4), (0.8, 0.2), (None, 0.25)])
def test_stratify_retain_index(sizes):
    test_size, train_size = sizes

    X = cudf.DataFrame({"x": range(10)})
    y = cudf.Series(([0] * (10 // 2)) + ([1] * (10 // 2)))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_size,
        test_size=test_size,
        shuffle=True,
        stratify=y,
        random_state=15,
    )
    assert (X_train["x"].to_numpy() == X_train.index.to_numpy()).all()
    assert (X_test["x"].to_numpy() == X_test.index.to_numpy()).all()

    if train_size is not None:
        assert X_train.shape[0] == (int)(X.shape[0] * train_size)

    elif test_size is not None:
        assert X_test.shape[0] == (int)(X.shape[0] * test_size)


def test_stratified_binary_classification():
    X = cp.array(
        [
            [0.37487513, -2.3031888, 1.662633, 0.7671007],
            [-0.49796826, -1.0621182, -0.32518214, -0.20583323],
            [-1.0104885, -2.4997945, 2.8952584, 1.4712684],
            [2.008748, -2.4520662, 0.5557737, 0.07749569],
            [0.97350526, -0.3403474, -0.58081895, -0.23199573],
        ]
    )

    # Needs to fail when we have just 1 occurrence of a label
    y = cp.array([0, 0, 0, 0, 1])
    with pytest.raises(ValueError):
        train_test_split(X, y, train_size=0.75, stratify=y, shuffle=True)

    y = cp.array([0, 0, 0, 1, 1])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, stratify=y, random_state=15
    )

    _, y_counts = cp.unique(y, return_counts=True)
    _, train_counts = cp.unique(y_train, return_counts=True)
    _, test_counts = cp.unique(y_test, return_counts=True)

    # Ensure we have preserve the number of labels
    cp.testing.assert_array_equal(train_counts + test_counts, y_counts)


@pytest.mark.parametrize("sizes", [(0.6, 0.4), (0.8, 0.2), (None, 0.25)])
def test_stratify_any_input(sizes):
    test_size, train_size = sizes

    X = cudf.DataFrame({"x": range(10)})
    X["test_col"] = cudf.Series([10, 0, 0, 10, 10, 10, 0, 0, 10, 10])
    y = cudf.Series(([0] * (10 // 2)) + ([1] * (10 // 2)))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_size,
        test_size=test_size,
        shuffle=True,
        stratify=X["test_col"],
        random_state=15,
    )
    assert (X_train["x"].to_numpy() == X_train.index.to_numpy()).all()
    assert (X_test["x"].to_numpy() == X_test.index.to_numpy()).all()

    if train_size is not None:
        assert X_train.shape[0] == (int)(X.shape[0] * train_size)

    elif test_size is not None:
        assert X_test.shape[0] == (int)(X.shape[0] * test_size)


# =============================================================================
# Regression tests for sklearn shim refactoring
# =============================================================================


def test_sklearn_signature_keyword_args():
    """Test that sklearn-style keyword arguments work correctly."""
    X = cp.random.rand(100, 10).astype(cp.float32)
    y = cp.random.randint(0, 2, 100)

    # Test with sklearn parameter order (random_state before shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        train_size=None,
        random_state=42,
        shuffle=True,
        stratify=None,
    )

    assert X_train.shape[0] == 80
    assert X_test.shape[0] == 20


def test_multiple_arrays_same_split():
    """Verify that multiple arrays are split consistently."""
    X = cp.random.rand(100, 10).astype(cp.float32)
    y = cp.random.randint(0, 2, 100)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Verify sizes match
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[0] + X_test.shape[0] == 100


def test_no_data_loss_cupy():
    """Verify no data is lost during split with cupy arrays."""
    X = cp.arange(100).reshape(100, 1)
    y = cp.arange(100)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Combine and sort to check all values are preserved
    X_combined = cp.sort(cp.concatenate([X_train.flatten(), X_test.flatten()]))
    y_combined = cp.sort(cp.concatenate([y_train, y_test]))

    cp.testing.assert_array_equal(X_combined, cp.arange(100))
    cp.testing.assert_array_equal(y_combined, cp.arange(100))


def test_no_data_loss_cudf():
    """Verify no data is lost during split with cudf DataFrames."""
    df = cudf.DataFrame({"a": range(100), "b": range(100, 200)})
    y = cudf.Series(range(100))

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, shuffle=False
    )

    # Combine and verify all values present
    X_combined = cudf.concat([X_train, X_test]).sort_values("a")
    assert len(X_combined) == 100
    assert X_combined["a"].to_numpy().tolist() == list(range(100))


def test_integer_sizes():
    """Test with integer train_size and test_size."""
    X = cp.random.rand(100, 5)

    X_train, X_test = train_test_split(X, train_size=70, test_size=30)

    assert X_train.shape[0] == 70
    assert X_test.shape[0] == 30


def test_single_array_split():
    """Test splitting a single array without y."""
    X = cp.random.rand(100, 10)

    X_train, X_test = train_test_split(X, test_size=0.2)

    assert X_train.shape[0] == 80
    assert X_test.shape[0] == 20


def test_empty_stratify_class_error():
    """Test that stratification with insufficient samples raises error."""
    X = cp.random.rand(5, 2)
    y = cp.array([0, 0, 0, 0, 1])  # Only 1 sample of class 1

    with pytest.raises(ValueError):
        train_test_split(X, y, stratify=y, test_size=0.4)


def test_reproducibility_with_seed():
    """Test that same seed produces same results."""
    X = cp.random.rand(100, 10)
    y = cp.random.randint(0, 2, 100)

    X_train1, X_test1, y_train1, y_test1 = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    cp.testing.assert_array_equal(X_train1, X_train2)
    cp.testing.assert_array_equal(X_test1, X_test2)
    cp.testing.assert_array_equal(y_train1, y_train2)
    cp.testing.assert_array_equal(y_test1, y_test2)


def test_different_seeds_different_results():
    """Test that different seeds produce different results."""
    X = cp.random.rand(100, 10)
    y = cp.random.randint(0, 2, 100)

    X_train1, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train2, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=43)

    # Results should be different (with high probability)
    assert not cp.allclose(X_train1, X_train2)


@pytest.mark.parametrize("backend", ["cudf", "pandas"])
def test_dataframe_column_preservation(backend):
    """Test that DataFrame column names and types are preserved."""
    if backend == "cudf":
        df = cudf.DataFrame({"col_a": range(100), "col_b": range(100, 200)})
        expected_type = cudf.DataFrame
    else:
        df = pd.DataFrame({"col_a": range(100), "col_b": range(100, 200)})
        expected_type = pd.DataFrame

    train, test = train_test_split(df, test_size=0.2)

    # Check column names are preserved
    assert list(train.columns) == ["col_a", "col_b"]
    assert list(test.columns) == ["col_a", "col_b"]

    # Check return type matches input type
    assert isinstance(train, expected_type)
    assert isinstance(test, expected_type)


def test_shuffle_false_preserves_order():
    """Test that shuffle=False preserves data order."""
    X = cp.arange(100).reshape(100, 1)

    X_train, X_test = train_test_split(X, test_size=0.2, shuffle=False)

    # First 80 should be in train, last 20 in test
    cp.testing.assert_array_equal(X_train.flatten(), cp.arange(80))
    cp.testing.assert_array_equal(X_test.flatten(), cp.arange(80, 100))


def test_zero_arrays_raises():
    """Test that train_test_split with 0 arrays raises ValueError."""
    with pytest.raises(ValueError, match="At least one array required"):
        train_test_split()


@example(arrays=[cp.arange(100).reshape(100, 1)])
@example(arrays=[cp.arange(100).reshape(100, 1), cudf.Series(range(100))])
@given(
    arrays=st.lists(
        cuml_array_inputs(
            dtypes=st.just(np.float32),
            shapes=st.just((100,)),
        ),
        min_size=1,
        max_size=5,
    )
)
def test_variadic_input_type_preservation(arrays):
    """Test train_test_split with variable array inputs of mixed types."""

    n_samples = len(arrays[0])
    test_size = 0.2
    train_samples = int(n_samples * (1 - test_size))
    test_samples = n_samples - train_samples

    result = train_test_split(*arrays, test_size=test_size, shuffle=False)

    # Check number of outputs (2 per input array)
    assert len(result) == len(arrays) * 2

    # Check sizes and types of each output pair
    for input_arr, train, test in zip(arrays, result[::2], result[1::2]):
        expected_type = type(input_arr)

        assert isinstance(train, expected_type)
        assert isinstance(test, expected_type)

        if hasattr(train, "shape"):
            assert train.shape[0] == train_samples
            assert test.shape[0] == test_samples
        else:
            assert len(train) == train_samples
            assert len(test) == test_samples
