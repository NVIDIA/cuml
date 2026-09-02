#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import warnings

import cudf
import cupy as cp
import numpy as np

from cuml.internals.base import Base
from cuml.internals.interop import (
    InteropMixin,
    UnsupportedOnCPU,
    UnsupportedOnGPU,
)
from cuml.internals.outputs import mlfunc
from cuml.internals.validation import (
    check_classification_targets,
    check_consistent_length,
    check_cudf,
    check_features,
    check_input_features,
    check_is_fitted,
    check_random_seed,
)


class TargetEncoder(InteropMixin, Base):
    """
    A cudf based implementation of target encoding [1]_, which converts
    one or multiple categorical variables, 'Xs', with the average of
    corresponding values of the target variable, 'Y'. The input data is
    grouped by the columns `Xs` and the aggregated mean value of `Y` of
    each group is calculated to replace each value of `Xs`. Several
    optimizations are applied to prevent label leakage and parallelize
    the execution.

    Parameters
    ----------
    n_folds : int (default=4)
        Default number of folds for fitting training data. To prevent
        label leakage in `fit`, we split data into `n_folds` and
        encode one fold using the target variables of the remaining folds.
    smooth : int or float (default=0)
        Count of samples to smooth the encoding. 0 means no smoothing.
    seed : int (default=42)
        Random seed
    split_method : {'random', 'continuous', 'interleaved'}, \
        (default='interleaved')
        Method to split train data into `n_folds`.
        'random': random split.
        'continuous': consecutive samples are grouped into one folds.
        'interleaved': samples are assign to each fold in a round robin way.
        'customize': customize splitting by providing a `fold_ids` array
        in `fit()` or `fit_transform()` functions.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {None, 'input', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.
    stat : {'mean','var','median'}, default = 'mean'
        The statistic used in encoding, mean, variance or median of the
        target.
    multi_feature_mode : {'combination', 'independent'}, default='combination'
        How to handle multiple input features:

        - ``'combination'``: Encode all feature combinations together
          (cuML native behavior). Produces a single output column with
          encodings based on the joint distribution of all features.
        - ``'independent'``: Encode each feature independently (sklearn
          behavior). Produces N output columns, one per input feature,
          each containing encodings based only on that feature's
          relationship with the target.

        For single-feature input, both modes produce identical results.

    Attributes
    ----------
    categories_ : list of cupy.ndarray
        The categories of each input feature determined during fitting.
        Each element is an array of unique category values for that feature,
        sorted in ascending order.
    target_type_ : str
        Type of target.
    classes_ : numpy.ndarray or None
        The labels for each class if `target_type_` is 'binary', `None`
        otherwise.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    encode_all : cudf.DataFrame or list[cudf.DataFrame]
        DataFrame containing the learned encodings for all category
        combinations, or a list of such dataframes if fit with
        `multi_feature_mode="independent"`. Used internally for transforming
        new data.
    mean : float
        The overall mean of the target variable, computed during fitting.
        Used for smoothing and imputing unseen categories.
    y_stat_val : float
        The statistic value (mean, variance, or median) of the target
        variable, depending on the ``stat`` parameter. Used to impute
        encodings for unseen categories.
    train : cudf.DataFrame or None
        The training DataFrame used during fitting, containing the original
        features, target values, and fold assignments. Set to ``None`` if
        the encoder was loaded from a sklearn model via :meth:`from_sklearn`.
    train_encode : cupy array or None
        The encoded values for the training data, computed during
        :meth:`fit` or :meth:`fit_transform`. Set to ``None`` if the
        encoder was loaded from a sklearn model via :meth:`from_sklearn`.

    Notes
    -----
    **sklearn Conversion Limitations**

    When converting between cuML and sklearn via :meth:`as_sklearn` and
    :meth:`from_sklearn`, be aware of the following semantic differences:

    - **Training data behavior**: cuML's :meth:`transform` returns
      cross-validated (regularized) encodings when called on training data
      to prevent data leakage. sklearn's ``transform`` always returns global
      mean encodings regardless of whether the input is training or test data.
      After roundtrip conversion, the cuML model will return global encodings
      for all data since the training data reference is not preserved.

    - **Multi-feature encoding**: cuML's default ``multi_feature_mode='combination'``
      encodes feature combinations jointly, while sklearn always encodes features
      independently. Multi-feature models fitted with ``'combination'`` mode
      cannot be converted to sklearn; use ``multi_feature_mode='independent'``
      for sklearn compatibility.

    **Cross-Validation Differences**

    cuML and sklearn use different cross-validation fold assignment strategies
    during ``fit_transform``. Both are valid target encoding implementations,
    but they produce different encoded values for the same input:

    - **sklearn**: Uses ``KFold``/``StratifiedKFold`` with specific sample-to-fold
      assignments based on ``random_state``.
    - **cuML**: Uses configurable ``split_method`` (``'interleaved'``, ``'random'``,
      ``'continuous'``, or ``'customize'``) with different fold assignment logic.

    Because samples are assigned to different folds, the leave-fold-out encoding
    for each sample is computed from different data subsets. For example::

        # Same data, same random_state, different encoded values:
        # sklearn fit_transform: [0.52, 0.48, 0.51, 0.49, ...]
        # cuML fit_transform:    [0.50, 0.51, 0.49, 0.52, ...]

    This difference only affects ``fit_transform`` on training data. The
    ``transform`` method on test data produces equivalent results since it
    uses global statistics computed from all training samples.

    References
    ----------
    .. [1] https://maxhalford.github.io/blog/target-encoding/

    Examples
    --------
    Converting a categorical implementation to a numerical one

    >>> from cudf import DataFrame, Series
    >>> from cuml.preprocessing import TargetEncoder
    >>> train = DataFrame({'category': ['a', 'b', 'b', 'a'],
    ...                    'label': [1, 0, 1, 1]})
    >>> test = DataFrame({'category': ['a', 'c', 'b', 'a']})

    >>> encoder = TargetEncoder(output_type='numpy')
    >>> encoded = encoder.fit_transform(train[["category"]], train.label)
    >>> encoded
    array([[1.],
           [1.],
           [0.],
           [1.]])
    """

    # InteropMixin requirements
    _cpu_class_path = "sklearn.preprocessing.TargetEncoder"

    def __init__(
        self,
        *,
        n_folds=4,
        smooth=0,
        seed=42,
        split_method="interleaved",
        verbose=False,
        output_type=None,
        stat="mean",
        multi_feature_mode="combination",
    ):
        super().__init__(verbose=verbose, output_type=output_type)
        self.n_folds = n_folds
        self.seed = seed
        self.smooth = smooth
        self.split_method = split_method
        self.stat = stat
        self.multi_feature_mode = multi_feature_mode

    @mlfunc(set_input_type=True)
    def fit(self, X, y, *, fold_ids=None):
        """
        Fit a TargetEncoder instance to a set of categories

        Parameters
        ----------
        X : cudf.Series or cudf.DataFrame or cupy.ndarray
           categories to be encoded. It's elements may or may
           not be unique
        y : cudf.Series or cupy.ndarray
            Series containing the target variable.
        fold_ids : cudf.Series or cupy.ndarray
            Series containing the indices of the customized
            folds. Its values should be integers in range
            `[0, N-1]` to split data into `N` folds. If None,
            fold_ids is generated based on `split_method`.

        Returns
        -------
        self : TargetEncoder
            A fitted instance of itself to allow method chaining
        """
        res, train = self._fit_transform(X, y, fold_ids=fold_ids)
        self.train_encode = res
        self.train = train
        return self

    @mlfunc(preserve_index=True, column_names="feature_names_out")
    def fit_transform(self, X, y, *, fold_ids=None):
        """
        Simultaneously fit and transform an input

        This is functionally equivalent to (but faster than)
        `TargetEncoder().fit(y).transform(y)`

        Parameters
        ----------
        X : cudf.Series or cudf.DataFrame or cupy.ndarray
           categories to be encoded. It's elements may or may
           not be unique
        y : cudf.Series or cupy.ndarray
            Series containing the target variable.
        fold_ids : cudf.Series or cupy.ndarray
            Series containing the indices of the customized
            folds. Its values should be integers in range
            `[0, N-1]` to split data into `N` folds. If None,
            fold_ids is generated based on `split_method`.

        Returns
        -------
        encoded : cupy.ndarray
            The ordinally encoded input series

        """
        self.fit(X, y, fold_ids=fold_ids)
        return self.train_encode

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input feature names.

        Returns
        -------
        feature_names_out : numpy.ndarray of str objects.
            Transformed feature names.
        """
        check_is_fitted(self)
        input_features = check_input_features(self, input_features)
        if self.multi_feature_mode == "independent":
            return input_features
        return np.asarray(["targetencoder0"], dtype=object)

    @mlfunc(preserve_index=True, column_names="feature_names_out")
    def transform(self, X):
        """
        Transform an input into its categorical keys.

        This is intended for test data. For fitting and transforming
        the training data, prefer `fit_transform`.

        Parameters
        ----------
        X : cudf.Series
            Input keys to be transformed. Its values doesn't have to
            match the categories given to `fit`

        Returns
        -------
        encoded : cupy.ndarray
            The ordinally encoded input series

        """
        check_is_fitted(self)
        df = self._check_X(X)

        if self._is_train_df(df):
            return self.train_encode

        x_cols = [n for n in df.columns.tolist() if n.startswith("X_")]

        if isinstance(self.encode_all, list):
            return self._transform_independent(df, x_cols)

        df = df.merge(self.encode_all, on=x_cols, how="left")
        return self._impute_and_sort(df)

    def _transform_independent(self, test, x_cols):
        """Transform using independent per-feature encodings."""
        result_cols = []
        for i, col in enumerate(x_cols):
            out_col_i = f"out_{i}"
            encode_all_i = self.encode_all[i]

            test = test.merge(
                encode_all_i.rename(columns={"out": out_col_i}),
                on=col,
                how="left",
            )
            test[out_col_i] = test[out_col_i].nans_to_nulls()
            test[out_col_i] = test[out_col_i].fillna(self.y_stat_val)
            result_cols.append(out_col_i)

        test = test.sort_values("id")
        res = test[result_cols].to_cupy()
        return res

    def _fit_transform(self, X, y, fold_ids):
        if self.smooth < 0:
            raise ValueError(f"smooth {self.smooth} is not zero or positive")

        if self.stat not in {"mean", "var", "median"}:
            raise ValueError(
                f"Expected `stat` in ['mean', 'var', 'median'], got {self.stat!r}"
            )

        df = self._check_X_y(X, y)
        x_cols = [n for n in df.columns.tolist() if n.startswith("X_")]

        # Extract unique categories for each feature
        self.categories_ = []
        for col in x_cols:
            cats = df[col].drop_duplicates().sort_values().to_numpy()
            self.categories_.append(cats)

        if self.multi_feature_mode not in {"combination", "independent"}:
            raise ValueError(
                f"multi_feature_mode should be 'combination' or 'independent', "
                f"got '{self.multi_feature_mode}'."
            )

        # Add fold column
        df = df.assign(fold=self._make_fold_column(len(df), fold_ids))

        # Compute stats
        self.mean = df.y.mean()
        self.y_stat_val = getattr(df.y, self.stat)()

        # Delegate to appropriate method based on mode and stat
        if self.multi_feature_mode == "independent":
            self._n_features_out = self.n_features_in_
            return self._fit_transform_independent(df, x_cols, fold_ids)
        elif self.stat == "median":
            self._n_features_out = 1
            return self._fit_transform_median(df, x_cols, fold_ids)
        else:
            self._n_features_out = 1
            return self._fit_transform_combination(df, x_cols, fold_ids)

    def _fit_transform_combination(self, df, x_cols, fold_ids):
        """
        Fit-transform with combination encoding (cuML native behavior).

        Encodes all feature combinations together, producing a single output
        column with encodings based on the joint distribution of all features.
        """
        self._encodings_per_feature = []
        if self.stat == "var":
            y_cols = ["y", "y2"]
            df = df.assign(y2=df.y * df.y)
            self.mean2 = df.y2.mean()
        else:
            y_cols = ["y"]

        y_count_each_fold, y_count_all = self._groupby_agg(
            df, x_cols, op="count", y_cols=y_cols
        )

        y_sum_each_fold, y_sum_all = self._groupby_agg(
            df, x_cols, op="sum", y_cols=y_cols
        )

        # encode_each_fold is used to encode training data
        # encode_all is used to encode testing data
        cols = ["fold", *x_cols]
        encode_each_fold = self._compute_output(
            y_sum_each_fold,
            y_count_each_fold,
            cols,
            "y_x",
            "y2_x",
        )
        self.encode_all = self._compute_output(
            y_sum_all, y_count_all, x_cols, "y", "y2"
        )

        df = df.merge(encode_each_fold, on=cols, how="left")
        return self._impute_and_sort(df), df

    def _fit_transform_median(self, df, x_cols, fold_ids):
        """
        Fit-transform with median stat using a for-loop approach.

        Median requires computing the statistic per fold separately,
        which cannot be vectorized like mean/var.
        """
        self._encodings_per_feature = []

        def _rename_col(df, col):
            df.columns = [col]
            return df.reset_index()

        res = []
        unq_vals = df.fold.drop_duplicates().sort_values().to_numpy()
        for f in unq_vals:
            mask = df.fold.values == f
            dg = df.loc[~mask].groupby(x_cols).agg({"y": self.stat})
            dg = _rename_col(dg, "out")
            res.append(df.loc[mask].merge(dg, on=x_cols, how="left"))
        res = cudf.concat(res, axis=0)
        self.encode_all = df.groupby(x_cols).agg({"y": self.stat})
        self.encode_all = _rename_col(self.encode_all, "out")
        return self._impute_and_sort(res), df

    def _fit_transform_independent(self, df, x_cols, fold_ids):
        """
        Fit-transform with independent per-feature encoding (sklearn-like).

        Each feature is encoded independently based on its relationship with
        the target, producing N output columns for N input features.
        """
        self._encodings_per_feature = []
        self.encode_all = []
        result_cols = []

        for i, col in enumerate(x_cols):
            out_col_i = f"out_{i}"

            if self.stat in ["median"]:
                encode_all_i = self._compute_single_feature_encoding_median(
                    df, col
                )
            else:
                encode_all_i = self._compute_single_feature_encoding(
                    df, col, out_col_i
                )

            self.encode_all.append(encode_all_i)

            # Extract encodings in category order for sklearn compatibility
            feature_encodings = []
            for cat_val in self.categories_[i]:
                mask = encode_all_i[col] == cat_val
                if mask.any():
                    enc_val = float(encode_all_i.loc[mask, "out"].iloc[0])
                else:
                    enc_val = float(self.mean)
                feature_encodings.append(enc_val)
            self._encodings_per_feature.append(np.array(feature_encodings))

            # Merge encoding into df for this feature
            df = df.merge(
                encode_all_i.rename(columns={"out": out_col_i}),
                on=col,
                how="left",
            )
            df[out_col_i] = df[out_col_i].nans_to_nulls()
            df[out_col_i] = df[out_col_i].fillna(self.y_stat_val)
            result_cols.append(out_col_i)

        # Sort by original index and return results
        df = df.sort_values("id")
        res = df[result_cols].to_cupy()
        return res, df

    def _compute_single_feature_encoding(self, train, col, out_col):
        """Compute target encoding for a single feature column."""
        # Group by single feature and compute stats
        df_count = train.groupby(col, as_index=False).agg({"y": "count"})
        df_count.columns = [col, "y_count"]

        df_sum = train.groupby(col, as_index=False).agg({"y": "sum"})
        df_sum.columns = [col, "y_sum"]

        df = df_sum.merge(df_count, on=col, how="left")

        # Apply smoothing
        smooth = self.smooth
        df["out"] = (df.y_sum + smooth * self.mean) / (df.y_count + smooth)

        return df[[col, "out"]]

    def _compute_single_feature_encoding_median(self, train, col):
        """Compute median target encoding for a single feature column."""
        encode_all = train.groupby(col, as_index=False).agg({"y": self.stat})
        encode_all.columns = [col, "out"]
        return encode_all

    def _make_fold_column(self, n_samples, fold_ids):
        """
        Create a fold id column for each split
        """
        if self.n_folds < 1:
            raise ValueError("n_folds >= 1 is required")

        n_folds = min(self.n_folds, n_samples)

        if fold_ids is not None or self.split_method == "customize":
            if fold_ids is None:
                raise ValueError(
                    "`fold_ids` is required if split_method='customize'"
                )
            elif self.split_method != "customize":
                warnings.warn(
                    "Using `split_method='customize'` since `fold_ids` are provided"
                )
            if len(fold_ids) != n_samples:
                raise ValueError(
                    f"`fold_ids` length {len(fold_ids)} is different from input "
                    f"data length {n_samples}"
                )
            return cp.asarray(fold_ids)
        if self.split_method == "random":
            rng = cp.random.default_rng(check_random_seed(self.seed))
            return rng.integers(0, n_folds, n_samples)
        elif self.split_method == "continuous":
            return (cp.arange(n_samples) / (n_samples / n_folds)) % n_folds
        elif self.split_method == "interleaved":
            return cp.arange(n_samples) % n_folds
        else:
            raise ValueError(
                f"Expected `split_method` in ['random', 'continuous', "
                f"'interleaved', 'customize'], got {self.split_method!r}"
            )

    def _compute_output(self, df_sum, df_count, cols, y_col, y_col2):
        """
        Compute the output encoding based on aggregated sum and count
        """
        df_sum = df_sum.merge(df_count, on=cols, how="left")
        smooth = self.smooth
        df_sum["out"] = (df_sum[f"{y_col}_x"] + smooth * self.mean) / (
            df_sum[f"{y_col}_y"] + smooth
        )
        if self.stat == "var":
            df_sum["out2"] = (df_sum[f"{y_col2}_x"] + smooth * self.mean2) / (
                df_sum[f"{y_col2}_y"] + smooth
            )
            df_sum["out"] = df_sum["out2"] - df_sum["out"] ** 2
            df_sum["out"] = (
                df_sum["out"]
                * df_sum[f"{y_col2}_y"]
                / (df_sum[f"{y_col2}_y"] - 1)
            )
        return df_sum

    def _groupby_agg(self, train, x_cols, op, y_cols):
        """
        Compute aggregated value of each fold and overall dataframe
        grouped by `x_cols` and agg by `op`
        """
        cols = ["fold", *x_cols]
        df_each_fold = train.groupby(cols, as_index=False).agg(
            {y_col: op for y_col in y_cols}
        )
        df_all = df_each_fold.groupby(x_cols, as_index=False).agg(
            {y_col: "sum" for y_col in y_cols}
        )

        df_each_fold = df_each_fold.merge(df_all, on=x_cols, how="left")
        for y_col in y_cols:
            df_each_fold[f"{y_col}_x"] = (
                df_each_fold[f"{y_col}_y"] - df_each_fold[f"{y_col}_x"]
            )
        return df_each_fold, df_all

    def _is_train_df(self, df):
        """
        Return True if the dataframe `df` is the training dataframe, which
        is used in `fit_transform`
        """
        # If train is None (e.g., loaded from sklearn), we can't compare
        if getattr(self, "train", None) is None:
            return False
        if len(df) != len(self.train):
            return False
        self.train = self.train.sort_values("id").reset_index(drop=True)
        for col in df.columns:
            if not (df[col] == self.train[col]).all():
                return False
        return True

    def _impute_and_sort(self, df):
        """
        Impute and sort the result encoding in the same row order as input.
        """

        df["out"] = df["out"].nans_to_nulls().fillna(self.y_stat_val)
        df = df.sort_values("id")
        res = df["out"].to_cupy().reshape(-1, 1)
        return res

    def _check_X(self, X, reset=False):
        # Check features
        check_features(self, X, reset=reset)
        # Coerce to a cudf.DataFrame
        df = check_cudf(X, input_name="X")
        # Rename columns uniformly
        df = df.rename({c: f"X_{i}" for i, c in enumerate(df.columns)}, axis=1)
        # Add an id column
        df = df.assign(id=cp.arange(len(df)))
        # Drop the index
        df = df.reset_index(drop=True)
        return df

    def _check_X_y(self, X, y):
        X = self._check_X(X, reset=True)
        if y is None:
            raise ValueError(
                "This estimator requires y to be passed, but the target y is None"
            )
        y = check_cudf(y, ensure_ndim=1, coerce_ndim=True, input_name="y")
        check_consistent_length(X, y)

        # Drop the index from y
        y = y.reset_index(drop=True)

        # Infer the type of target and transform y
        continuous = False
        if cudf.api.types.is_float_dtype(y):
            # Floating input. Check if it's a valid classification target.
            try:
                check_classification_targets(y)
            except ValueError:
                continuous = True
        if continuous:
            y = y.astype("float64")
            self.target_type_ = "continuous"
            self.classes_ = None
        elif y.nunique() <= 2:
            y = y.astype("category")
            self.target_type_ = "binary"
            self.classes_ = y.cat.categories.to_numpy()
            y = y.cat.codes.astype("float64")
        else:
            raise ValueError(
                "TargetEncoder currently only supports 'continuous' and 'binary' "
                "target types, but got 'multiclass'."
            )

        df = X.assign(**{"y": y})
        return df

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + [
            "n_folds",
            "smooth",
            "seed",
            "split_method",
            "stat",
            "multi_feature_mode",
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        """Convert sklearn TargetEncoder hyperparameters to cuML format."""
        if model.categories != "auto":
            raise UnsupportedOnGPU("Only categories='auto' is supported")
        if not (
            model.random_state is None or isinstance(model.random_state, int)
        ):
            raise UnsupportedOnGPU("Only integral random seeds are supported")

        params = {
            "n_folds": model.cv,
            "seed": model.random_state,
            "smooth": 1.0 if model.smooth == "auto" else float(model.smooth),
            "split_method": "random" if model.shuffle else "continuous",
            "stat": "mean",
            "multi_feature_mode": "independent",
        }
        return params

    def _params_to_cpu(self):
        return {
            "cv": self.n_folds,
            "random_state": self.seed,
            "smooth": self.smooth,
            "shuffle": self.split_method == "random",
        }

    def _attrs_from_cpu(self, model):
        """Convert sklearn TargetEncoder attributes to cuML format.

        sklearn always uses independent per-feature encoding, so we set up
        cuML to use independent mode as well for exact compatibility.
        """
        # Handle string categories (object dtype) - keep as numpy arrays
        # since cupy doesn't support object dtype
        categories_gpu = []
        for cat in model.categories_:
            if cat.dtype == np.object_:
                categories_gpu.append(cat)  # Keep as numpy array
            else:
                categories_gpu.append(cp.asarray(cat))
        n_features = len(model.categories_)

        # Generate column names matching cuML's internal format
        x_cols = [f"X_{i}" for i in range(n_features)]

        # sklearn uses independent encoding, so we always use independent mode
        # This gives exact compatibility with no approximation
        encode_all = []
        for i, col in enumerate(x_cols):
            encode_all_i = cudf.DataFrame(
                {
                    col: model.categories_[i],
                    "out": model.encodings_[i],
                }
            )
            encode_all.append(encode_all_i)

        return {
            "encode_all": encode_all,
            "_encodings_per_feature": [
                cp.asarray(enc) for enc in model.encodings_
            ],
            "categories_": categories_gpu,
            "classes_": model.classes_,
            "_n_features_out": n_features,  # sklearn always uses independent mode
            "mean": float(model.target_mean_),
            "y_stat_val": float(model.target_mean_),
            "train": None,
            "train_encode": None,
            "target_type_": model.target_type_,
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        """Convert cuML TargetEncoder attributes to sklearn format.

        sklearn expects independent per-feature encodings. If cuML was fitted
        in independent mode, we have exact encodings. Multi-feature combination
        mode cannot be converted to sklearn.
        """
        # Handle categories that may be numpy arrays or cupy arrays
        categories_cpu = []
        for cat in self.categories_:
            categories_cpu.append(cp.asnumpy(cat))
        n_features = len(self.categories_)

        # Use per-feature encodings if available (from independent mode or sklearn)
        if hasattr(self, "_encodings_per_feature"):
            encodings_list = []
            for enc in self._encodings_per_feature:
                encodings_list.append(cp.asnumpy(enc))
        elif n_features == 1:
            # Single feature: extract encodings directly (exact conversion)
            cats = self.categories_[0]
            feature_encodings = []
            for cat_val in cats:
                mask = self.encode_all["X_0"] == cat_val
                if mask.any():
                    enc_val = float(self.encode_all.loc[mask, "out"].iloc[0])
                else:
                    enc_val = float(self.mean)
                feature_encodings.append(enc_val)
            encodings_list = [np.array(feature_encodings)]
        else:
            # Multi-feature combination mode cannot be converted to sklearn
            raise UnsupportedOnCPU(
                f"`multi_feature_mode={self.multi_feature_mode!r}` is not supported"
            )

        return {
            "encodings_": encodings_list,
            "categories_": categories_cpu,
            "classes_": self.classes_,
            "target_mean_": float(self.mean),
            # sklearn internal attributes needed for transform
            "_infrequent_enabled": False,
            "target_type_": self.target_type_,
            **super()._attrs_to_cpu(model),
        }
