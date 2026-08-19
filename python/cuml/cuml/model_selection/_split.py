# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import cudf
import cupy as cp
from sklearn.model_selection import (
    train_test_split as sklearn_train_test_split,
)

from cuml.internals.validation import (
    _get_n_samples,
    check_array,
    check_consistent_length,
    check_cudf,
    check_random_seed,
)


def train_test_split(
    *arrays,
    test_size=None,
    train_size=None,
    random_state=None,
    shuffle=True,
    stratify=None,
):
    """
    Split arrays or matrices into random train and test subsets.

    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
        Allowed inputs are cudf DataFrames/Series, cupy arrays, numpy arrays,
        pandas DataFrames/Series, or any array-like objects with a shape
        attribute.

    test_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If train_size is also None, it will
        be set to 0.25.

    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.

    shuffle : bool, default=True
        Whether or not to shuffle the data before splitting.

    stratify : array-like, default=None
        If not None, data is split in a stratified fashion, using this as
        the class labels.

    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs. Output types match
        input types (cudf inputs return cudf outputs, cupy inputs return
        cupy outputs, etc.)

    Examples
    --------
    >>> import cupy as cp
    >>> from cuml.model_selection import train_test_split
    >>> X = cp.arange(10).reshape((5, 2))
    >>> y = cp.array([0, 0, 1, 1, 1])
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.2, random_state=42
    ... )
    """
    if len(arrays) == 0:
        raise ValueError("At least one array required as input")
    check_consistent_length(*arrays)

    # CuPy RandomState isn't supported by sklearn, so extract an int seed.
    # int, None, and np.random.RandomState are all accepted by sklearn as-is.
    sklearn_random_state = (
        int(check_random_seed(random_state))
        if isinstance(random_state, cp.random.RandomState)
        else random_state
    )

    # Convert stratify to numpy, other arrays types are not supported by scikit-learn
    if stratify is not None:
        stratify = check_array(stratify, ensure_2d=False, mem_type="host")

    return sklearn_train_test_split(
        *arrays,
        test_size=test_size,
        train_size=train_size,
        random_state=sklearn_random_state,
        shuffle=shuffle,
        stratify=stratify,
    )


class _KFoldBase:
    """Base class for k-fold split."""

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        if not (isinstance(n_splits, int) and n_splits >= 2):
            raise ValueError(
                f"Expected an integral n_splits >= 2, got {n_splits=!r}"
            )

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.seed = random_state

    def split(self, X, y=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.

        Yields
        ------
        train : CuPy ndarray
            The training set indices for that split.

        test : CuPy ndarray
            The testing set indices for that split.
        """
        check_consistent_length(X, y)
        n_samples = _get_n_samples(X)
        if n_samples < self.n_splits:
            raise ValueError(
                f"Cannot have number of splits n_splits={self.n_splits} greater "
                f"than the number of samples: {n_samples=}"
            )
        indices = cp.arange(n_samples)

        if self.shuffle:
            cp.random.RandomState(check_random_seed(self.seed)).shuffle(
                indices
            )
        yield from self._split(X, y, indices)

    def _split(self, X, y, indices):
        """Generate the split indices given X, y, and indices"""
        raise NotImplementedError

    def get_n_splits(self, X=None, y=None):
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits


class KFold(_KFoldBase):
    """K-Folds cross-validator.

    Provides train/test indices to split data in train/test sets. Split dataset into k
    consecutive folds (without shuffling by default).

    Each fold is then used once as a validation set while the k - 1 remaining folds form
    the training set.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

    shuffle : bool, default=False
        Whether to shuffle the samples before splitting. Note that the samples within
        each split will not be shuffled.

    random_state : int, CuPy RandomState, NumPy RandomState, or None, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold. Otherwise, this
        parameter has no effect. Pass an int for reproducible output across
        multiple function calls.

    Examples
    --------
    >>> import cupy as cp
    >>> from cuml.model_selection import KFold
    >>> X = cp.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = cp.array([0, 0, 1, 1])
    >>> kf = KFold(n_splits=2)
    >>> kf.get_n_splits()
    2
    >>> for i, (train_index, test_index) in enumerate(kf.split(X, y)):
    ...     print(f"Fold{i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[2 3]
      Test:  index=[0 1]
    Fold 1:
      Train: index=[0 1]
      Test:  index=[2 3]
    """

    def _split(self, X, y, indices):
        n_samples = len(indices)
        fold_sizes = cp.full(
            self.n_splits, n_samples // self.n_splits, dtype=cp.int64
        )
        fold_sizes[: n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test = indices[start:stop]

            mask = cp.zeros(n_samples, dtype=cp.bool_)
            mask[start:stop] = True

            train = indices[cp.logical_not(mask)]
            yield train, test
            current = stop


class StratifiedKFold(_KFoldBase):
    """Class-wise stratified K-fold cross-validator.

    Provides train/test indices to split data into stratified K folds. The
    percentage of samples for each class are maintained in each fold.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

    shuffle : bool, default=False
        Whether to shuffle the samples before splitting. Note that the samples within
        each split will not be shuffled.

    random_state : int, CuPy RandomState, NumPy RandomState, or None, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold. Otherwise, this
        parameter has no effect. Pass an int for reproducible output across
        multiple function calls.

    Examples
    --------
    >>> import cupy as cp
    >>> from cuml.model_selection import StratifiedKFold
    >>> X = cp.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = cp.array([0, 0, 1, 1])
    >>> kf = StratifiedKFold(n_splits=2)
    >>> kf.get_n_splits()
    2
    >>> for i, (train_index, test_index) in enumerate(kf.split(X, y)):
    ...     print(f"Fold{i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[1 3]
      Test:  index=[0 2]
    Fold 1:
      Train: index=[0 2]
      Test:  index=[1 3]
    """

    def _split(self, X, y, indices):
        y = check_cudf(y, ensure_ndim=1)
        if y.nunique() < 2:
            raise ValueError("number of unique classes cannot be less than 2")

        df = cudf.DataFrame({"y": y[indices], "ids": indices})
        gb = df.groupby(["y"])

        if self.n_splits > gb.y.count().min():
            raise ValueError(
                f"n_splits={self.n_splits} cannot be greater "
                f"than the number of members in each class."
            )

        df = df.assign(order=gb.cumcount()).sort_values("ids")
        ids = df["ids"].to_cupy()

        for i in range(self.n_splits):
            mask = (df["order"] % self.n_splits == i).to_cupy()
            train = ids[~mask]
            test = ids[mask]
            yield train, test
