# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import cudf
import cupy as cp
import numpy as np
import pandas as pd
import pytest
import scipy
import sklearn
from hypothesis import assume, example, given
from hypothesis import strategies as st
from hypothesis import target
from packaging.version import Version
from scipy.sparse import csr_matrix
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.linear_model import LogisticRegression as skLog
from sklearn.model_selection import train_test_split

import cuml
from cuml import LogisticRegression as cuLog
from cuml.testing.datasets import (
    make_classification,
    make_classification_dataset,
    small_classification_dataset,
    split_datasets,
    standard_classification_datasets,
)
from cuml.testing.strategies import dataset_dtypes
from cuml.testing.utils import array_equal

SKLEARN_110 = Version(sklearn.__version__) >= Version("1.10.0.dev0")


def _filter_scipy_lbfgsb_deprecation(func):
    if Version("1.16") <= Version(scipy.__version__) < Version("1.18"):
        return pytest.mark.filterwarnings(
            "ignore:.*The `disp` and `iprint` options.*:DeprecationWarning"
        )(func)
    return func


@given(
    num_classes=st.integers(min_value=2, max_value=10),
    dtype=dataset_dtypes(),
    l1_ratio=st.floats(min_value=0.0, max_value=1.0),
    fit_intercept=st.booleans(),
    nrows=st.integers(min_value=1000, max_value=5000),
    C=st.floats(min_value=0.5, max_value=2.0),
    tol=st.floats(min_value=1e-8, max_value=1e-3),
)
@example(
    num_classes=2,
    dtype=np.float32,
    l1_ratio=1.0,
    fit_intercept=True,
    nrows=1000,
    C=np.inf,
    tol=1e-3,
)
@example(
    num_classes=10,
    dtype=np.float64,
    l1_ratio=0.0,
    fit_intercept=True,
    nrows=5000,
    C=1.0,
    tol=1e-8,
)
# ignoring UserWarnings in sklearn about setting unused parameters
# like l1 for none penalty
@pytest.mark.filterwarnings("ignore::UserWarning:sklearn[.*]")
def test_logistic_regression(
    num_classes,
    dtype,
    l1_ratio,
    fit_intercept,
    nrows,
    C,
    tol,
):
    ncols, n_info = 20, 10

    X_train, X_test, y_train, y_test = make_classification_dataset(
        datatype=dtype,
        nrows=nrows,
        ncols=ncols,
        n_info=n_info,
        num_classes=num_classes,
    )
    y_train = y_train.astype(dtype)
    y_test = y_test.astype(dtype)

    culog = cuLog(
        l1_ratio=l1_ratio,
        C=C,
        fit_intercept=fit_intercept,
        tol=tol,
    )
    culog.fit(X_train, y_train)

    sklog = skLog(
        solver="saga",
        l1_ratio=l1_ratio,
        # `C` was deprecated in sklearn 1.10 and will be removed in 1.14
        **({"alpha": 1 / C} if SKLEARN_110 else {"C": C}),
        fit_intercept=fit_intercept,
    )

    sklog.fit(X_train, y_train)
    cu_preds = culog.predict(X_test)

    tol_score = 1e-1
    assert (
        culog.score(X_train, y_train)
        >= sklog.score(X_train, y_train) - tol_score
    )
    assert (
        culog.score(X_test, y_test) >= sklog.score(X_test, y_test) - tol_score
    )
    assert len(np.unique(cu_preds)) == len(np.unique(y_test))

    if fit_intercept is False:
        assert np.array_equal(culog.intercept_, sklog.intercept_)


@given(
    dtype=dataset_dtypes(),
    l1_ratio=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0)),
    C=st.sampled_from([1.0, np.inf]),
)
@example(dtype=np.float32, l1_ratio=None, C=np.inf)
@example(dtype=np.float64, l1_ratio=1.0, C=1.0)
@example(dtype=np.float64, l1_ratio=0.0, C=1.0)
def test_logistic_regression_unscaled(dtype, l1_ratio, C):
    # Test logistic regression on the breast cancer dataset. We do not scale
    # the dataset which could lead to numerical problems (fixed in PR #2543).
    X, y = load_breast_cancer(return_X_y=True)
    X = X.astype(dtype)
    y = y.astype(dtype)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    params = {
        "C": C,
        "tol": 1e-4,
        "fit_intercept": True,
        "max_iter": 5000,
        "l1_ratio": l1_ratio,
    }
    culog = cuLog(**params)
    culog.fit(X_train, y_train)

    score_train = culog.score(X_train, y_train)
    score_test = culog.score(X_test, y_test)

    target(1 / score_train, label="inverse train score")
    target(1 / score_test, label="inverse test score")

    # TODO: Use a more rigorous approach to determine expected minimal scores
    # here. The values here are selected empirically and passed during test
    # development.
    assert score_train >= 0.94
    assert score_test >= 0.94


@given(dtype=dataset_dtypes())
@example(dtype=np.float32)
@example(dtype=np.float64)
@_filter_scipy_lbfgsb_deprecation
def test_logistic_regression_model_default(dtype):
    # Ignore SciPy 1.16/1.17 deprecation warnings emitted through sklearn's
    # default LogisticRegression L-BFGS-B path.
    X_train, X_test, y_train, y_test = small_classification_dataset(dtype)
    y_train = y_train.astype(dtype)
    y_test = y_test.astype(dtype)
    culog = cuLog()
    culog.fit(X_train, y_train)
    sklog = skLog()

    sklog.fit(X_train, y_train)

    assert culog.score(X_test, y_test) >= sklog.score(X_test, y_test) - 0.022


@given(
    dtype=dataset_dtypes(),
    order=st.sampled_from(["C", "F"]),
    sparse_input=st.booleans(),
    fit_intercept=st.booleans(),
    penalty=st.sampled_from([None, "l1", "l2"]),
)
@example(
    dtype=np.float32,
    order="C",
    sparse_input=False,
    fit_intercept=True,
    penalty=None,
)
@example(
    dtype=np.float64,
    order="F",
    sparse_input=True,
    fit_intercept=False,
    penalty="l1",
)
@example(
    dtype=np.float32,
    order="C",
    sparse_input=True,
    fit_intercept=False,
    penalty="l2",
)
def test_logistic_regression_model_digits(
    dtype, order, sparse_input, fit_intercept, penalty
):
    # smallest sklearn score with max_iter = 10000
    # put it as a constant here, because sklearn 0.23.1 needs a lot of iters
    # to converge and has a bug returning an unrelated error if not converged.
    acceptable_score = 0.9

    digits = load_digits()

    X_dense = digits.data.astype(dtype)
    X_dense.reshape(X_dense.shape, order=order)
    X = csr_matrix(X_dense) if sparse_input else X_dense

    y = digits.target.astype(dtype)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    culog = cuLog(fit_intercept=fit_intercept, penalty=penalty)
    culog.fit(X_train, y_train)
    score = culog.score(X_test, y_test)

    assert score >= acceptable_score


@given(dtype=dataset_dtypes())
@example(dtype=np.float32)
@example(dtype=np.float64)
def test_logistic_regression_sparse_only(dtype, sparse_text_dataset):
    # sklearn score with max_iter = 10000
    sklearn_score = 0.878
    acceptable_score = sklearn_score - 0.01

    X, y = sparse_text_dataset

    X = csr_matrix(X.astype(dtype))
    y = y.get().astype(dtype)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    culog = cuLog()
    culog.fit(X_train, y_train)
    score = culog.score(X_test, y_test)

    assert score >= acceptable_score


@given(
    dataset=split_datasets(
        standard_classification_datasets(
            n_classes=st.sampled_from((2, 10)),
            n_features=st.just(20),
            n_informative=st.just(10),
        )
    ),
    fit_intercept=st.booleans(),
    sparse_input=st.booleans(),
)
@example(
    dataset=small_classification_dataset(np.float32),
    fit_intercept=True,
    sparse_input=False,
)
@example(
    dataset=small_classification_dataset(np.float64),
    fit_intercept=False,
    sparse_input=True,
)
def test_logistic_regression_decision_function(
    dataset, fit_intercept, sparse_input
):
    X_train, X_test, y_train, y_test = dataset

    # Assumption needed to avoid qn.h: logistic loss invalid C error.
    assume(set(np.unique(y_train)) == set(np.unique(y_test)))
    num_classes = len(np.unique(np.concatenate((y_train, y_test))))

    if sparse_input:
        X_train = csr_matrix(X_train)
        X_test = csr_matrix(X_test)

    culog = cuLog(fit_intercept=fit_intercept, output_type="numpy")
    culog.fit(X_train, y_train)

    sklog = skLog(fit_intercept=fit_intercept)
    sklog.coef_ = culog.coef_
    sklog.intercept_ = culog.intercept_ if fit_intercept else 0
    sklog.classes_ = np.arange(num_classes)

    cu_dec_func = culog.decision_function(X_test)
    sk_dec_func = sklog.decision_function(X_test)

    assert array_equal(cu_dec_func, sk_dec_func)


@given(
    dataset=split_datasets(
        standard_classification_datasets(
            n_classes=st.sampled_from((2, 10)),
            n_features=st.just(20),
            n_informative=st.just(10),
        )
    ),
    fit_intercept=st.booleans(),
    sparse_input=st.booleans(),
)
@example(
    dataset=small_classification_dataset(np.float32),
    fit_intercept=True,
    sparse_input=False,
)
@example(
    dataset=small_classification_dataset(np.float64),
    fit_intercept=False,
    sparse_input=True,
)
def test_logistic_regression_predict_proba(
    dataset, fit_intercept, sparse_input
):
    X_train, X_test, y_train, y_test = dataset

    # Assumption needed to avoid qn.h: logistic loss invalid C error.
    assume(set(np.unique(y_train)) == set(np.unique(y_test)))
    num_classes = len(np.unique(y_train))

    if sparse_input:
        X_train = csr_matrix(X_train)
        X_test = csr_matrix(X_test)

    culog = cuLog(fit_intercept=fit_intercept, output_type="numpy")
    culog.fit(X_train, y_train)

    sklog = skLog(
        fit_intercept=fit_intercept,
        **({"solver": "lbfgs"} if num_classes > 2 else {}),
    )
    sklog.coef_ = culog.coef_
    sklog.intercept_ = culog.intercept_ if fit_intercept else 0
    sklog.classes_ = np.arange(num_classes)

    cu_proba = culog.predict_proba(X_test)
    sk_proba = sklog.predict_proba(X_test)

    cu_log_proba = culog.predict_log_proba(X_test)
    sk_log_proba = sklog.predict_log_proba(X_test)

    assert array_equal(cu_proba, sk_proba)

    # if the probabilities pass test, then the margin of the logarithm
    # of the probabilities can be relaxed to avoid false positives.
    assert array_equal(
        cu_log_proba, sk_log_proba, unit_tol=1e-2, total_tol=1e-3
    )


@pytest.mark.parametrize("constructor", [np.array, cp.array, cudf.DataFrame])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_logistic_regression_input_type_consistency(constructor, dtype):
    X = constructor([[5, 10], [3, 1], [7, 8]]).astype(dtype)
    y = constructor([0, 1, 1]).astype(dtype)
    clf = cuLog().fit(X, y)

    assert isinstance(clf.predict_proba(X), type(X))
    expected_type = cudf.Series if constructor == cudf.DataFrame else type(X)
    assert isinstance(clf.predict(X), expected_type)


@pytest.mark.parametrize(
    "y_kind", ["object", "fixed-string", "int32", "float32", "float16"]
)
@pytest.mark.parametrize("output_type", ["numpy", "cupy", "cudf", "pandas"])
@_filter_scipy_lbfgsb_deprecation
def test_logistic_regression_complex_classes(y_kind, output_type):
    # Ignore SciPy 1.16/1.17 deprecation warnings emitted through sklearn's
    # default LogisticRegression L-BFGS-B path.
    """Test that LogisticRegression handles non-numeric or non-monotonically
    increasing classes properly in both `fit` and `predict`"""
    if output_type == "cupy" and y_kind in ("object", "fixed-string"):
        pytest.skip("cupy doesn't support strings!")
    elif output_type in ("cudf", "pandas") and y_kind == "float16":
        pytest.skip("float16 dtype not supported")

    X, y_inds = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=10,
        n_classes=3,
        random_state=0,
    )
    if y_kind == "object":
        classes = np.array(["apple", "banana", "carrot"], dtype="object")
        df_dtype = "object"
    elif y_kind == "fixed-string":
        classes = np.array(["apple", "banana", "carrot"], dtype="U")
        df_dtype = "object"
    else:
        classes = np.array([10, 20, 30], dtype=y_kind)
        df_dtype = classes.dtype

    y = classes.take(y_inds)

    cu_model = cuLog(output_type=output_type)
    sk_model = skLog()

    cu_model.fit(X, y)
    sk_model.fit(X, y)

    np.testing.assert_array_equal(
        cu_model.classes_, sk_model.classes_, strict=True
    )

    res = cu_model.predict(X)
    sol = sk_model.predict(X)
    if output_type == "numpy":
        assert res.dtype == sol.dtype
        assert isinstance(res, np.ndarray)
    elif output_type == "cupy":
        assert res.dtype == sol.dtype
        assert isinstance(res, cp.ndarray)
    elif output_type == "pandas":
        assert res.dtype == df_dtype
        assert isinstance(res, pd.Series)
    elif output_type == "cudf":
        assert res.dtype == df_dtype
        assert isinstance(res, cudf.Series)


@pytest.mark.parametrize("y_kind", ["pandas", "cudf"])
def test_logistic_regression_categorical_y(y_kind):
    X, y_inds = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=10,
        n_classes=3,
        random_state=0,
    )
    categories = np.array(["apple", "banana", "carrot"], dtype="object")
    y = pd.Series(pd.Categorical.from_codes(y_inds, categories))
    if y_kind == "cudf":
        y = cudf.Series(y)

    model = cuLog(output_type="numpy")
    model.fit(X, y)
    np.testing.assert_array_equal(model.classes_, categories, strict=True)
    res = model.predict(X)
    assert isinstance(res, np.ndarray)
    assert res.dtype == "object"


@given(
    dataset=split_datasets(
        standard_classification_datasets(),
    ),
    test_dtype=dataset_dtypes(),
)
@example(
    dataset=small_classification_dataset(np.float32), test_dtype=np.float32
)
@example(
    dataset=small_classification_dataset(np.float32), test_dtype=np.float64
)
@example(
    dataset=small_classification_dataset(np.float64), test_dtype=np.float32
)
@example(
    dataset=small_classification_dataset(np.float64), test_dtype=np.float64
)
def test_logistic_predict_output_dtype(dataset, test_dtype):
    X_train, X_test, y_train, y_test = dataset

    # Assumption needed to avoid qn.h: logistic loss invalid C error.
    assume(set(np.unique(y_train)) == set(np.unique(y_test)))

    clf = cuLog()
    clf.fit(X_train, y_train)
    clf.predict(X_test.astype(test_dtype))


@given(
    dataset=standard_classification_datasets(),
    use_sample_weight=st.booleans(),
    class_weight_option=st.sampled_from([None, "balanced", "dict"]),
)
@example(
    dataset=(
        *make_classification(
            n_samples=100000,
            n_features=5,
            n_informative=4,
            n_classes=2,
            n_redundant=0,
            random_state=0,
        ),
    ),
    use_sample_weight=True,
    class_weight_option=None,
)
@example(
    dataset=(
        *make_classification(
            n_samples=100000,
            n_features=5,
            n_informative=4,
            n_classes=7,
            n_redundant=0,
            random_state=1,
        ),
    ),
    use_sample_weight=False,
    class_weight_option="balanced",
)
@example(
    dataset=(
        *make_classification(
            n_samples=100000,
            n_features=5,
            n_informative=4,
            n_classes=3,
            n_redundant=0,
            random_state=2,
        ),
    ),
    use_sample_weight=True,
    class_weight_option="dict",
)
@example(
    dataset=(
        *make_classification(
            n_samples=100000,
            n_features=5,
            n_informative=4,
            n_classes=2,
            n_redundant=0,
            random_state=3,
        ),
    ),
    use_sample_weight=False,
    class_weight_option=None,
)
@_filter_scipy_lbfgsb_deprecation
def test_logistic_regression_weighting(
    dataset, use_sample_weight, class_weight_option
):
    # Ignore SciPy 1.16/1.17 deprecation warnings emitted through sklearn's
    # default LogisticRegression L-BFGS-B path.
    X, y = dataset

    num_classes = len(np.unique(y))

    # Set up sample_weight
    sample_weight = None
    if use_sample_weight:
        n_samples = X.shape[0]
        sample_weight = np.abs(np.random.rand(n_samples))

    # Set up class_weight
    class_weight = None
    match class_weight_option:
        case "dict":
            weights = np.random.rand(num_classes)
            class_weight = {i: weights[i] for i in range(num_classes)}
        case "balanced":
            class_weight = "balanced"
        case None:
            pass
        case _:
            raise ValueError(
                f"Unknown class_weight_option: {class_weight_option}"
            )

    # Use higher max_iter for better convergence with complex weighting
    max_iter = (
        1000
        if (use_sample_weight and class_weight_option is not None)
        else 500
    )

    culog = cuLog(
        fit_intercept=False, class_weight=class_weight, max_iter=max_iter
    )
    culog.fit(X, y, sample_weight=sample_weight)

    sklog = skLog(
        fit_intercept=False, class_weight=class_weight, max_iter=max_iter
    )
    sklog.fit(X, y, sample_weight=sample_weight)

    skcoef = np.squeeze(sklog.coef_)
    cucoef = np.squeeze(culog.coef_)

    # Normalize coefficients
    if num_classes == 2:
        skcoef /= np.linalg.norm(skcoef)
        cucoef /= np.linalg.norm(cucoef)
    else:
        skcoef /= np.linalg.norm(skcoef, axis=1, keepdims=True)
        cucoef /= np.linalg.norm(cucoef, axis=1, keepdims=True)

    # Set tolerances based on empirical analysis
    has_weights = use_sample_weight or class_weight_option is not None
    harder_problem = has_weights or num_classes > 3
    unit_tol = 0.25 if harder_problem else 0.04
    total_tol = 0.40 if harder_problem else 0.08
    assert array_equal(skcoef, cucoef, unit_tol=unit_tol, total_tol=total_tol)

    cuOut = culog.predict(X)
    skOut = sklog.predict(X)
    assert array_equal(skOut, cuOut, unit_tol=unit_tol, total_tol=total_tol)


@pytest.mark.parametrize("penalty", ["l1", "l2"])
def test_logistic_regression_max_iter_n_iter(penalty):
    X, y = make_classification(random_state=42)
    model = cuml.LogisticRegression(penalty=penalty).fit(X, y)
    assert model.n_iter_.max() <= model.max_iter

    model = cuml.LogisticRegression(penalty=penalty, max_iter=10).fit(X, y)
    assert model.n_iter_.max() == 10
