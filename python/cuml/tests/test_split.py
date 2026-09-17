# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cudf
import cupy as cp
import numpy as np
import pytest

from cuml.datasets import make_regression
from cuml.model_selection import KFold, StratifiedKFold


def get_x_y(n_samples, n_classes):
    X = cudf.DataFrame({"x": range(n_samples)})
    y = cp.arange(n_samples) % n_classes
    cp.random.shuffle(y)
    y = cudf.Series(y)
    return X, y


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("n_splits", [5, 10])
@pytest.mark.parametrize("n_samples", [10000])
@pytest.mark.parametrize("n_classes", [2, 10])
def test_split_dataframe(n_samples, n_classes, n_splits, shuffle):
    X, y = get_x_y(n_samples, n_classes)

    kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle)
    assert kf.get_n_splits(X, y) == kf.get_n_splits() == n_splits

    for train_index, test_index in kf.split(X, y):
        assert len(train_index) + len(test_index) == n_samples
        assert len(train_index) == len(test_index) * (n_splits - 1)
        for i in range(n_classes):
            ratio_tr = (y[train_index] == i).sum() / len(train_index)
            ratio_te = (y[test_index] == i).sum() / len(test_index)
            assert ratio_tr == ratio_te


def test_stratified_kfold_n_splits_invalid():
    X, y = get_x_y(n_samples=1000, n_classes=1)
    kf = StratifiedKFold(n_splits=5)

    with pytest.raises(
        ValueError, match="number of unique classes cannot be less than 2"
    ):
        list(kf.split(X, y))

    y = cp.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2])
    X = cp.zeros((len(y), 3))
    with pytest.raises(
        ValueError,
        match=(
            "n_splits=5 cannot be greater than the number of members "
            "in each class"
        ),
    ):
        list(kf.split(X, y))


@pytest.mark.parametrize("n_splits", [0, 1, "bad"])
@pytest.mark.parametrize("cls", [KFold, StratifiedKFold])
def test_invalid_folds(cls, n_splits):
    X, y = get_x_y(n_samples=1000, n_classes=2)

    err_msg = f"Expected an integral n_splits >= 2, got {n_splits=!r}"
    with pytest.raises(ValueError, match=err_msg):
        kf = cls(n_splits=n_splits)
        for train_index, test_index in kf.split(X, y):
            break


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("n_splits", [5, 10])
@pytest.mark.parametrize(
    "random_state",
    [
        1,
        np.random.RandomState(1),
        cp.random.RandomState(1),
        None,
    ],
)
def test_kfold(shuffle, n_splits, random_state) -> None:
    n_samples = 256
    n_features = 16
    X, y = make_regression(n_samples, n_features, random_state=1)
    kfold = KFold(
        n_splits=n_splits, shuffle=shuffle, random_state=random_state
    )
    assert kfold.get_n_splits(X, y) == kfold.get_n_splits() == n_splits
    n_test_total = 0

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        n_test_total += test_idx.size

        assert train_idx.shape[0] + test_idx.shape[0] == n_samples
        fold_size = X.shape[0] // n_splits
        # We assign the remainder to the beginning folds.
        if fold_idx < n_samples % n_splits:
            assert test_idx.shape[0] == fold_size + 1
        else:
            assert test_idx.shape[0] == fold_size
        assert cp.all(train_idx >= 0)
        assert cp.all(test_idx >= 0)
        indices = cp.concatenate([train_idx, test_idx])
        assert len(indices.shape) == 1
        assert indices.size == n_samples
        uniques = cp.unique(indices)
        sorted_uniques = cp.sort(uniques)

        assert uniques.size == n_samples, indices
        arr = cp.arange(n_samples)
        cp.testing.assert_allclose(sorted_uniques, arr)

    assert n_test_total == n_samples


# Since the kfold only uses the shape of the input, not the actual data, we only have a
# small test for dataframe.
def test_kfold_dataframe() -> None:
    n_samples = 4096
    X, y = get_x_y(n_samples, 2)
    kfold = KFold(n_splits=5, shuffle=True)
    for train_idx, test_idx in kfold.split(X, y):
        assert train_idx.shape[0] + test_idx.shape[0] == n_samples




def test_kfold_sort_indices_matches_sklearn_order() -> None:
    """sort_indices=True yields scikit-learn-like sorted index order.

    Coverage for NVIDIA/cuml#8631: with shuffle=True,
    KFold(sort_indices=True) must return the train and test indices in
    sorted order like scikit-learn - the shuffle only determines fold
    membership. Membership itself is not asserted here because cuML
    shuffles with CuPy's RNG, which is not guaranteed to generate the
    same permutation as NumPy's RNG.
    """
    n_samples = 3
    X = np.arange(n_samples * 2).reshape(n_samples, 2)
    kfold = KFold(n_splits=3, shuffle=True, random_state=1, sort_indices=True)

    for train_idx, test_idx in kfold.split(X):
        # Indices must come out sorted, exactly like scikit-learn.
        cp.testing.assert_array_equal(train_idx, cp.sort(train_idx))
        cp.testing.assert_array_equal(test_idx, cp.sort(test_idx))
        # Train and test indices must partition the sample set.
        combined = cp.sort(cp.concatenate([train_idx, test_idx]))
        cp.testing.assert_array_equal(combined, cp.arange(n_samples))

    # Identical parameters must produce identical splits.
    first = [(t.copy(), s.copy()) for t, s in kfold.split(X)]
    for (train_idx, test_idx), (exp_train, exp_test) in zip(
        kfold.split(X), first
    ):
        cp.testing.assert_array_equal(train_idx, exp_train)
        cp.testing.assert_array_equal(test_idx, exp_test)


@pytest.mark.parametrize("n_samples", [10, 11, 100])
@pytest.mark.parametrize("n_splits", [2, 3, 5])
@pytest.mark.parametrize("random_state", [0, 1, 42, 123])
def test_kfold_sort_indices_sweep(
    n_samples, n_splits, random_state
) -> None:
    """KFold(sort_indices=True) always yields sorted train/test indices."""
    X = cp.arange(n_samples * 2).reshape(n_samples, 2)
    kfold = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
        sort_indices=True,
    )

    for train_idx, test_idx in kfold.split(X):
        assert train_idx.shape[0] + test_idx.shape[0] == n_samples
        cp.testing.assert_array_equal(train_idx, cp.sort(train_idx))
        cp.testing.assert_array_equal(test_idx, cp.sort(test_idx))

    # Splits must be deterministic for identical parameters.
    first = [(t.copy(), s.copy()) for t, s in kfold.split(X)]
    for (train_idx, test_idx), (exp_train, exp_test) in zip(
        kfold.split(X), first
    ):
        cp.testing.assert_array_equal(train_idx, exp_train)
        cp.testing.assert_array_equal(test_idx, exp_test)


def test_kfold_default_behavior_unchanged() -> None:
    """The default path (sort_indices=False) must stay exactly as before.

    Uses a fixed, deliberately unsorted index array so the test does not
    depend on CuPy's RNG: the default path preserves the shuffled order
    untouched, while sort_indices=True only reorders the same indices.
    """
    X = np.arange(6).reshape(3, 2)
    indices = cp.asarray([2, 0, 1])

    default = KFold(n_splits=3, sort_indices=False)
    opt_in = KFold(n_splits=3, sort_indices=True)

    default_splits = list(default._split(X, None, indices.copy()))
    optin_splits = list(opt_in._split(X, None, indices.copy()))

    assert len(default_splits) == len(optin_splits) == 3
    for (d_train, d_test), (o_train, o_test) in zip(
        default_splits, optin_splits
    ):
        # Opt-in only reorders: fold membership must be identical.
        cp.testing.assert_array_equal(cp.sort(d_train), o_train)
        cp.testing.assert_array_equal(cp.sort(d_test), o_test)
        # Opt-in output is sorted.
        cp.testing.assert_array_equal(o_train, cp.sort(o_train))
        cp.testing.assert_array_equal(o_test, cp.sort(o_test))

    # The default path keeps the unsorted shuffle order ([2, 1] and [2, 0]
    # for folds 1 and 2 with these indices), i.e. behavior is unchanged.
    cp.testing.assert_array_equal(default_splits[1][0], cp.asarray([2, 1]))
    cp.testing.assert_array_equal(default_splits[2][0], cp.asarray([2, 0]))
    cp.testing.assert_array_equal(optin_splits[1][0], cp.asarray([1, 2]))
    cp.testing.assert_array_equal(optin_splits[2][0], cp.asarray([0, 2]))
