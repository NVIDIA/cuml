# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import platform

import cudf
import cupy as cp
import numpy as np
import pytest
import scipy.sparse as scipy_sparse
from cudf.pandas import LOADED as cudf_pandas_active
from sklearn import svm
from sklearn.datasets import (
    load_iris,
    make_blobs,
    make_classification,
    make_friedman1,
    make_gaussian_quantiles,
    make_regression,
)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import cuml
import cuml.svm as cu_svm
from cuml.testing.utils import (
    compare_svm,
    quality_param,
    stress_param,
    svm_array_equal,
    unit_param,
)

IS_ARM = platform.processor() == "aarch64"


def make_dataset(dataset, n_rows, n_cols, n_classes=2, n_informative=2):
    np.random.seed(137)
    if n_rows * 0.25 < 4000:
        # Use at least 4000 test samples
        n_test = 4000
        if n_rows > 1000:
            # To avoid a large increase in test time (which is between
            # O(n_rows^2) and O(n_rows^3)).
            n_rows = int(n_rows * 0.75)
        n_rows += n_test
    else:
        n_test = n_rows * 0.25
    if dataset == "classification1":
        X, y = make_classification(
            n_rows,
            n_cols,
            n_informative=n_informative,
            n_redundant=0,
            n_classes=n_classes,
            n_clusters_per_class=1,
        )
    elif dataset == "classification2":
        X, y = make_classification(
            n_rows,
            n_cols,
            n_informative=n_informative,
            n_redundant=0,
            n_classes=n_classes,
            n_clusters_per_class=2,
        )
    elif dataset == "gaussian":
        X, y = make_gaussian_quantiles(
            n_samples=n_rows, n_features=n_cols, n_classes=n_classes
        )
    elif dataset == "blobs":
        X, y = make_blobs(
            n_samples=n_rows, n_features=n_cols, centers=n_classes
        )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test)
    # correct case when not all classes made it into the training set
    if np.unique(y_train).size < n_classes:
        for i in range(n_classes):
            y_train[i] = i
    return X_train, X_test, y_train, y_test


def get_binary_iris_dataset():
    iris = load_iris()
    X = iris.data
    y = iris.target
    y = (y > 0).astype(X.dtype)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y


@pytest.mark.parametrize(
    "params",
    [
        {"kernel": "linear", "C": 1},
        {"kernel": "linear", "C": 1, "tol": 1e-6},
        {"kernel": "linear", "C": 10},
        {"kernel": "rbf", "C": 1, "gamma": 1},
        {"kernel": "rbf", "C": 1, "gamma": "auto"},
        {"kernel": "rbf", "C": 0.1, "gamma": "auto"},
        {"kernel": "rbf", "C": 10, "gamma": "auto"},
        {"kernel": "rbf", "C": 1, "gamma": "scale"},
        {"kernel": "poly", "C": 1, "gamma": 1},
        {"kernel": "poly", "C": 1, "gamma": "auto"},
        {"kernel": "poly", "C": 1, "gamma": "scale"},
        {"kernel": "poly", "C": 1, "gamma": "auto", "degree": 2},
        {"kernel": "poly", "C": 1, "gamma": "auto", "coef0": 1.37},
        {"kernel": "sigmoid", "C": 1, "gamma": "auto"},
        {"kernel": "sigmoid", "C": 1, "gamma": "scale", "coef0": 0.42},
    ],
)
def test_svm_skl_cmp_kernels(params):
    # X_train, X_test, y_train, y_test = make_dataset('gaussian', 1000, 4)
    X_train, y_train = get_binary_iris_dataset()
    cuSVC = cu_svm.SVC(**params)
    cuSVC.fit(X_train, y_train)

    sklSVC = svm.SVC(**params)
    sklSVC.fit(X_train, y_train)

    compare_svm(cuSVC, sklSVC, X_train, y_train, cmp_decision_func=True)


@pytest.mark.parametrize(
    "params",
    [
        {"kernel": "linear", "C": 1},
        {"kernel": "rbf", "C": 1, "gamma": 1},
        {"kernel": "poly", "C": 1, "gamma": 1},
    ],
)
@pytest.mark.parametrize("dataset", ["classification2", "gaussian", "blobs"])
@pytest.mark.parametrize(
    "n_rows", [3, unit_param(100), quality_param(1000), stress_param(5000)]
)
@pytest.mark.parametrize(
    "n_cols", [2, unit_param(100), quality_param(1000), stress_param(1000)]
)
def test_svm_skl_cmp_datasets(params, dataset, n_rows, n_cols):
    if (
        params["kernel"] == "linear"
        and dataset in ["gaussian", "classification2"]
        and n_rows > 1000
        and n_cols >= 1000
    ):
        # linear kernel will not fit the gaussian dataset, but takes very long
        return
    X_train, X_test, y_train, y_test = make_dataset(dataset, n_rows, n_cols)

    # Default to numpy for testing
    with cuml.using_output_type("numpy"):
        cuSVC = cu_svm.SVC(**params)
        cuSVC.fit(X_train, y_train)

        sklSVC = svm.SVC(**params)
        sklSVC.fit(X_train, y_train)

        compare_svm(
            cuSVC, sklSVC, X_test, y_test, coef_tol=1e-5, report_summary=True
        )


@pytest.mark.parametrize("params", [{"kernel": "rbf", "C": 1, "gamma": 1}])
@pytest.mark.parametrize("sparse", [True, False])
def test_svm_skl_cmp_multiclass(
    params, sparse, dataset="classification2", n_rows=100, n_cols=6
):
    X_train, X_test, y_train, y_test = make_dataset(
        dataset, n_rows, n_cols, n_classes=3, n_informative=6
    )

    if sparse:
        X_train = scipy_sparse.csr_matrix(X_train)
        X_test = scipy_sparse.csr_matrix(X_test)

    # Default to numpy for testing
    with cuml.using_output_type("numpy"):
        cuSVC = cu_svm.SVC(**params)
        cuSVC.fit(X_train, y_train)

        sklSVC = svm.SVC(**params)
        sklSVC.fit(X_train, y_train)

        compare_svm(
            cuSVC, sklSVC, X_test, y_test, coef_tol=1e-5, report_summary=True
        )


def test_svm_skl_cmp_decision_function():
    X_train, X_test, y_train, _ = make_dataset("classification1", 4000, 20)
    y_train = y_train.astype("int32")

    params = {"kernel": "rbf", "C": 5, "gamma": 0.005}
    cu_model = cu_svm.SVC(**params).fit(X_train, y_train)
    sk_model = svm.SVC(**params).fit(X_train, y_train)

    cu_res = cu_model.decision_function(X_test)
    assert cu_res.dtype == X_train.dtype
    sk_res = sk_model.decision_function(X_test)
    assert mean_squared_error(cu_res, sk_res) < 1e-5


@pytest.mark.parametrize(
    "params",
    [
        {"kernel": "linear", "C": 1},
        {"kernel": "rbf", "C": 1, "gamma": 1},
        {"kernel": "poly", "C": 1, "gamma": 1},
        {"kernel": "sigmoid", "C": 1, "gamma": 1},
    ],
)
@pytest.mark.parametrize(
    "n_pred", [unit_param(5000), quality_param(100000), stress_param(1000000)]
)
def test_svm_predict(params, n_pred):
    n_rows = 500
    n_cols = 2
    X, y = make_blobs(
        n_samples=n_rows + n_pred,
        n_features=n_cols,
        centers=[[-5, -5], [5, 5]],
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=n_rows
    )
    cuSVC = cu_svm.SVC(**params)
    cuSVC.fit(X_train, y_train)
    y_pred = cuSVC.predict(X_test)
    n_correct = np.sum(y_test == y_pred)
    accuracy = n_correct * 100 / n_pred
    assert accuracy > 99


@pytest.mark.parametrize("class_weight", [None, {1: 10}, "balanced"])
@pytest.mark.parametrize("sample_weight", [None, True])
def test_svc_weights(class_weight, sample_weight):
    # We are using the following example as a test case
    # https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane_unbalanced.html
    X, y = make_blobs(
        n_samples=[1000, 100],
        centers=[[0.0, 0.0], [2.0, 2.0]],
        cluster_std=[1.5, 0.5],
        random_state=137,
        shuffle=False,
    )
    if sample_weight:
        # Put large weight on class 1
        sample_weight = y * 9 + 1

    params = {
        "kernel": "linear",
        "C": 1,
        "gamma": "scale",
        "class_weight": class_weight,
    }
    cu_model = cu_svm.SVC(**params).fit(X, y, sample_weight=sample_weight)

    if class_weight is not None or sample_weight is not None:
        # Standalone test: check if smaller blob is correctly classified in the
        # presence of class weights
        X_1 = X[y == 1, :]
        y_1 = np.ones(X_1.shape[0])
        cu_score = cu_model.score(X_1, y_1)
        assert cu_score > 0.9

    sk_model = svm.SVC(**params).fit(X, y, sample_weight=sample_weight)
    compare_svm(cu_model, sk_model, X, y, coef_tol=1e-5, report_summary=True)


@pytest.mark.parametrize(
    "params",
    [
        pytest.param(
            {"kernel": "poly", "degree": 40, "C": 1, "gamma": "auto"},
            marks=pytest.mark.xfail(
                reason="fp overflow in kernel "
                "function due to non scaled input "
                "features"
            ),
        ),
        pytest.param(
            {
                "kernel": "poly",
                "degree": 40,
                "C": 1,
                "gamma": "scale",
                "x_arraytype": "numpy",
            }
        ),
        pytest.param(
            {
                "kernel": "poly",
                "degree": 40,
                "C": 1,
                "gamma": "scale",
                "x_arraytype": "dataframe",
            }
        ),
        pytest.param(
            {
                "kernel": "poly",
                "degree": 40,
                "C": 1,
                "gamma": "scale",
                "x_arraytype": "cupy",
            }
        ),
    ],
)
def test_svm_gamma(params):
    # Note: we test different array types to make sure that the X.var() is
    # calculated correctly for gamma == 'scale' option.
    x_arraytype = params.pop("x_arraytype", "numpy")
    n_rows = 500
    n_cols = 380
    centers = [10 * np.ones(380), -10 * np.ones(380)]
    X, y = make_blobs(
        n_samples=n_rows, n_features=n_cols, random_state=137, centers=centers
    )
    X = X.astype(np.float32)
    if x_arraytype == "dataframe":
        y = cudf.Series(y)
    # Using degree 40 polynomials and fp32 training would fail with
    # gamma = 1/(n_cols*X.std()), but it works with the correct implementation:
    # gamma = 1/(n_cols*X.var())
    cuSVC = cu_svm.SVC(**params)
    cuSVC.fit(X, y)
    accuracy = cuSVC.score(X, y) * 100
    assert accuracy > 70


@pytest.mark.parametrize("x_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("y_dtype", [np.float32, np.float64, np.int32])
@pytest.mark.xfail(reason="SVC testing inflexibility (see issue #6575)")
def test_svm_numeric_arraytype(x_dtype, y_dtype):
    X, y = get_binary_iris_dataset()
    X = X.astype(x_dtype, order="F")
    y = y.astype(y_dtype)

    params = {"kernel": "rbf", "C": 1, "gamma": 0.25}
    cuSVC = cu_svm.SVC(**params)
    cuSVC.fit(X, y)
    intercept_exp = 0.23468959692060373
    n_sv_exp = 15
    assert abs(cuSVC.intercept_ - intercept_exp) / intercept_exp < 1e-3
    assert cuSVC.n_support_ == n_sv_exp
    n_pred_wrong = np.sum(cuSVC.predict(X) - y)
    assert n_pred_wrong == 0


def get_memsize(svc):
    """Calculates the memory occupied by the parameters of an SVC object

    Parameters
    ----------
    svc : cuML SVC classifier object

    Return
    ------
    The GPU memory usage in bytes.
    """
    ms = 0
    for a in ["dual_coef_", "support_", "support_vectors_"]:
        x = getattr(svc, a)
        ms += np.prod(x[0].shape) * x[0].dtype.itemsize
    return ms


def make_regression_dataset(dataset, n_rows, n_cols):
    np.random.seed(137)
    if dataset == "reg1":
        X, y = make_regression(
            n_rows, n_cols, n_informative=2, n_targets=1, random_state=137
        )
    elif dataset == "reg2":
        X, y = make_regression(
            n_rows,
            n_cols,
            n_informative=2,
            n_targets=1,
            random_state=137,
            noise=10,
        )
    elif dataset == "Friedman":
        X, y = make_friedman1(
            n_samples=n_rows, n_features=n_cols, noise=0.0, random_state=137
        )
    else:
        raise ValueError("Wrong option for dataste: ", dataset)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    dtype = np.float32
    X = X.astype(dtype)
    y = y.astype(dtype)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test


def compare_svr(svr1, svr2, X_test, y_test, tol=1e-3):
    if X_test.shape[0] > 1:
        score1 = svr1.score(X_test, y_test)
        score2 = svr2.score(X_test, y_test)
        assert abs(score1 - score2) < tol
    else:
        y_pred1 = svr1.predict(X_test)
        y_pred2 = svr2.predict(X_test)
        mse1 = mean_squared_error(y_test, y_pred1)
        mse2 = mean_squared_error(y_test, y_pred2)
        assert (mse1 - mse2) / mse2 < tol


@pytest.mark.parametrize(
    "params",
    [
        {"kernel": "linear", "C": 1, "gamma": 1},
        {"kernel": "rbf", "C": 1, "gamma": 1},
        {"kernel": "poly", "C": 1, "gamma": 1},
    ],
)
@pytest.mark.parametrize("dataset", ["reg1", "reg2", "Friedman"])
@pytest.mark.parametrize(
    "n_rows",
    [unit_param(3), unit_param(100), quality_param(1000), stress_param(5000)],
)
@pytest.mark.parametrize(
    "n_cols",
    [unit_param(5), unit_param(100), quality_param(1000), stress_param(1000)],
)
def test_svr_skl_cmp(params, dataset, n_rows, n_cols):
    """Compare to Sklearn SVR"""
    if dataset == "Friedman" and n_cols < 5:
        # We need at least 5 feature columns for the Friedman dataset
        return
    X_train, X_test, y_train, y_test = make_regression_dataset(
        dataset, n_rows, n_cols
    )
    cuSVR = cu_svm.SVR(**params)
    cuSVR.fit(X_train, y_train)

    sklSVR = svm.SVR(**params)
    sklSVR.fit(X_train, y_train)

    compare_svr(cuSVR, sklSVR, X_test, y_test)


def test_svr_skl_cmp_weighted():
    """Compare to Sklearn SVR, use sample weights"""
    X, y = make_regression(
        n_samples=100,
        n_features=5,
        n_informative=2,
        n_targets=1,
        random_state=137,
        noise=10,
    )
    sample_weights = 10 * np.sin(np.linspace(0, 2 * np.pi, len(y))) + 10.1

    params = {"kernel": "linear", "C": 10, "gamma": 1}
    cuSVR = cu_svm.SVR(**params)
    cuSVR.fit(X, y, sample_weights)

    sklSVR = svm.SVR(**params)
    sklSVR.fit(X, y, sample_weights)

    compare_svr(cuSVR, sklSVR, X, y)


@pytest.mark.parametrize("classifier", [True, False])
@pytest.mark.parametrize("train_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("test_dtype", [np.float64, np.float32])
def test_svm_predict_mixed_dtypes(train_dtype, test_dtype, classifier):
    X, y = make_classification(n_samples=50, random_state=0)

    X = X.astype(train_dtype)
    y = y.astype(train_dtype)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )

    if classifier:
        clf = cu_svm.SVC()
    else:
        clf = cu_svm.SVR()
    clf.fit(X_train, y_train)
    clf.predict(X_test.astype(test_dtype))


@pytest.mark.skipif(
    IS_ARM,
    reason="Test fails unexpectedly on ARM. "
    "github.com/rapidsai/cuml/issues/5100",
)
@pytest.mark.skipif(
    cudf_pandas_active,
    reason="cudf.pandas causes small numeric issues in this test only ",
)
def test_svm_no_support_vectors():
    n_rows = 10
    n_cols = 3
    X = cp.random.uniform(size=(n_rows, n_cols), dtype=cp.float64)
    y = cp.ones((n_rows, 1))
    model = cuml.svm.SVR(kernel="linear", C=10)
    model.fit(X, y)
    pred = model.predict(X)

    assert svm_array_equal(pred, y, 0)

    assert model.n_support_ == 0
    assert abs(model.intercept_ - 1) <= 1e-6
    assert svm_array_equal(model.coef_, cp.zeros((1, n_cols)))
    assert model.dual_coef_.shape == (1, 0)
    assert model.support_.shape == (0,)
    assert model.support_vectors_.shape[0] == 0
    # Check disabled due to https://github.com/rapidsai/cuml/issues/4095
    # assert model.support_vectors_.shape[1] == n_cols


@pytest.mark.parametrize("classifier", [False, True])
def test_max_iter_n_iter(classifier):
    if classifier:
        cls = cuml.SVC
        X, y = make_classification(random_state=42)
    else:
        cls = cuml.SVR
        X, y = make_regression(random_state=42)

    # max_iter limit is respected
    model = cls(max_iter=5).fit(X, y)
    assert (model.n_iter_.item() if classifier else model.n_iter_) == 5


def test_svc_multiclass_n_iter():
    X, y = make_classification(random_state=42, n_classes=3, n_informative=4)
    model = cuml.SVC().fit(X, y)
    assert model.n_iter_.dtype == np.int32
    assert model.n_iter_.shape == (3,)


# Tests for kernel='precomputed'
@pytest.mark.parametrize("kernel_func", ["linear", "rbf"])
def test_svc_precomputed_kernel(kernel_func):
    """Test SVC with precomputed kernel matrix matches sklearn."""
    np.random.seed(42)
    n_samples, n_features = 100, 10

    # Generate linearly separable data
    X = np.random.randn(n_samples, n_features).astype(np.float64)
    y = np.array(
        [0] * (n_samples // 2) + [1] * (n_samples // 2), dtype=np.int32
    )

    # Compute kernel matrix
    if kernel_func == "linear":
        K = X @ X.T
    else:  # rbf
        gamma = 0.1
        sq_dists = (
            np.sum(X**2, axis=1, keepdims=True)
            + np.sum(X**2, axis=1)
            - 2 * X @ X.T
        )
        K = np.exp(-gamma * sq_dists)

    # Convert to cupy
    K_cp = cp.asarray(K, order="F")
    y_cp = cp.asarray(y)

    # Fit sklearn
    sklearn_clf = svm.SVC(kernel="precomputed", C=1.0)
    sklearn_clf.fit(K, y)
    sklearn_preds = sklearn_clf.predict(K)

    # Fit cuML
    cuml_clf = cu_svm.SVC(kernel="precomputed", C=1.0)
    cuml_clf.fit(K_cp, y_cp)
    cuml_preds = cp.asnumpy(cuml_clf.predict(K_cp))

    # Predictions should match
    assert np.array_equal(sklearn_preds, cuml_preds), (
        f"Predictions don't match for {kernel_func} kernel"
    )


def test_svc_precomputed_kernel_train_test_split():
    """Test SVC with precomputed kernel on train/test split."""
    np.random.seed(42)
    n_train, n_test, n_features = 80, 20, 5

    # Generate data
    X_train = np.random.randn(n_train, n_features).astype(np.float32)
    X_test = np.random.randn(n_test, n_features).astype(np.float32)
    y_train = np.array(
        [0] * (n_train // 2) + [1] * (n_train // 2), dtype=np.int32
    )

    # Compute kernel matrices
    K_train = X_train @ X_train.T
    K_test = X_test @ X_train.T

    # Fit and predict with cuML
    cuml_clf = cu_svm.SVC(kernel="precomputed", C=1.0)
    cuml_clf.fit(cp.asarray(K_train, order="F"), cp.asarray(y_train))
    cuml_preds = cp.asnumpy(cuml_clf.predict(cp.asarray(K_test, order="F")))

    # Fit and predict with sklearn
    sklearn_clf = svm.SVC(kernel="precomputed", C=1.0)
    sklearn_clf.fit(K_train, y_train)
    sklearn_preds = sklearn_clf.predict(K_test)

    # Predictions should match
    assert np.array_equal(sklearn_preds, cuml_preds)


def test_svc_precomputed_decision_function():
    """Test decision_function with precomputed kernel."""
    np.random.seed(42)
    n_samples, n_features = 50, 5

    # Generate well-separated data
    X = np.vstack(
        [
            np.random.randn(n_samples // 2, n_features) - 2,
            np.random.randn(n_samples // 2, n_features) + 2,
        ]
    ).astype(np.float32)
    y = np.array(
        [0] * (n_samples // 2) + [1] * (n_samples // 2), dtype=np.int32
    )

    # Compute kernel matrix
    K = X @ X.T

    # Fit cuML with precomputed
    cuml_clf = cu_svm.SVC(kernel="precomputed", C=1.0)
    cuml_clf.fit(cp.asarray(K, order="F"), cp.asarray(y))
    cuml_decision = cp.asnumpy(
        cuml_clf.decision_function(cp.asarray(K, order="F"))
    )

    # Fit cuML with linear kernel (should give same results)
    cuml_linear = cu_svm.SVC(kernel="linear", C=1.0)
    cuml_linear.fit(cp.asarray(X, order="F"), cp.asarray(y))
    linear_decision = cp.asnumpy(
        cuml_linear.decision_function(cp.asarray(X, order="F"))
    )

    # Decision functions should be close
    assert np.allclose(cuml_decision, linear_decision, rtol=1e-3, atol=1e-3), (
        f"Max diff: {np.max(np.abs(cuml_decision - linear_decision))}"
    )


def test_svc_precomputed_kernel_validation():
    """Test that precomputed kernel validates input shape."""
    np.random.seed(42)
    n_samples = 50

    # Non-square kernel matrix should raise error
    K_nonsquare = np.random.randn(n_samples, n_samples + 1).astype(np.float32)
    y = np.array(
        [0] * (n_samples // 2) + [1] * (n_samples // 2), dtype=np.int32
    )

    cuml_clf = cu_svm.SVC(kernel="precomputed")
    with pytest.raises(ValueError, match="square"):
        cuml_clf.fit(cp.asarray(K_nonsquare), cp.asarray(y))


def test_svc_precomputed_vs_linear():
    """Test that precomputed linear kernel gives same result as kernel='linear'."""
    np.random.seed(123)
    n_samples, n_features = 60, 4

    # Generate separable data
    X = np.vstack(
        [
            np.random.randn(n_samples // 2, n_features) - 1.5,
            np.random.randn(n_samples // 2, n_features) + 1.5,
        ]
    ).astype(np.float32)
    y = np.array(
        [0] * (n_samples // 2) + [1] * (n_samples // 2), dtype=np.int32
    )

    # Compute linear kernel
    K = X @ X.T

    # Fit with linear kernel
    clf_linear = cu_svm.SVC(kernel="linear", C=1.0)
    clf_linear.fit(cp.asarray(X, order="F"), cp.asarray(y))

    # Fit with precomputed kernel
    clf_precomputed = cu_svm.SVC(kernel="precomputed", C=1.0)
    clf_precomputed.fit(cp.asarray(K, order="F"), cp.asarray(y))

    # Should have same number of support vectors
    assert clf_linear.n_support_ == clf_precomputed.n_support_

    # Predictions should match
    preds_linear = cp.asnumpy(clf_linear.predict(cp.asarray(X, order="F")))
    preds_precomputed = cp.asnumpy(
        clf_precomputed.predict(cp.asarray(K, order="F"))
    )
    assert np.array_equal(preds_linear, preds_precomputed)


# Tests for SVR with kernel='precomputed'
@pytest.mark.parametrize("kernel_func", ["linear", "rbf"])
def test_svr_precomputed_kernel(kernel_func):
    """Test SVR with precomputed kernel matrix matches sklearn."""
    np.random.seed(42)
    n_samples, n_features = 100, 5

    # Generate regression data
    X = np.random.randn(n_samples, n_features).astype(np.float64)
    y = np.sin(X[:, 0]) + 0.1 * np.random.randn(n_samples)
    y = y.astype(np.float64)

    # Compute kernel matrix
    if kernel_func == "linear":
        K = X @ X.T
    else:  # rbf
        gamma = 0.1
        sq_dists = (
            np.sum(X**2, axis=1, keepdims=True)
            + np.sum(X**2, axis=1)
            - 2 * X @ X.T
        )
        K = np.exp(-gamma * sq_dists)

    # Convert to cupy
    K_cp = cp.asarray(K, order="F")
    y_cp = cp.asarray(y)

    # Fit sklearn
    sklearn_reg = svm.SVR(kernel="precomputed", C=1.0, epsilon=0.1)
    sklearn_reg.fit(K, y)
    sklearn_preds = sklearn_reg.predict(K)

    # Fit cuML
    cuml_reg = cu_svm.SVR(kernel="precomputed", C=1.0, epsilon=0.1)
    cuml_reg.fit(K_cp, y_cp)
    cuml_preds = cp.asnumpy(cuml_reg.predict(K_cp))

    # Predictions should be close
    assert np.allclose(sklearn_preds, cuml_preds, rtol=1e-2, atol=1e-2), (
        f"Max diff: {np.max(np.abs(sklearn_preds - cuml_preds))}"
    )


def test_svr_precomputed_kernel_train_test_split():
    """Test SVR with precomputed kernel on train/test split."""
    np.random.seed(42)
    n_train, n_test, n_features = 80, 20, 5

    # Generate data
    X_train = np.random.randn(n_train, n_features).astype(np.float32)
    X_test = np.random.randn(n_test, n_features).astype(np.float32)
    y_train = np.sin(X_train[:, 0]).astype(np.float32)

    # Compute kernel matrices (linear)
    K_train = X_train @ X_train.T
    K_test = X_test @ X_train.T

    # Fit and predict with cuML
    cuml_reg = cu_svm.SVR(kernel="precomputed", C=1.0, epsilon=0.1)
    cuml_reg.fit(cp.asarray(K_train, order="F"), cp.asarray(y_train))
    cuml_preds = cp.asnumpy(cuml_reg.predict(cp.asarray(K_test, order="F")))

    # Fit and predict with sklearn
    sklearn_reg = svm.SVR(kernel="precomputed", C=1.0, epsilon=0.1)
    sklearn_reg.fit(K_train, y_train)
    sklearn_preds = sklearn_reg.predict(K_test)

    # Predictions should be close
    assert np.allclose(sklearn_preds, cuml_preds, rtol=1e-2, atol=1e-2)


def test_svr_precomputed_vs_linear():
    """Test that precomputed linear kernel gives same result as kernel='linear'."""
    np.random.seed(123)
    n_samples, n_features = 60, 4

    # Generate data
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = (2 * X[:, 0] + X[:, 1] + 0.1 * np.random.randn(n_samples)).astype(
        np.float32
    )

    # Compute linear kernel
    K = X @ X.T

    # Fit with linear kernel
    reg_linear = cu_svm.SVR(kernel="linear", C=1.0, epsilon=0.1)
    reg_linear.fit(cp.asarray(X, order="F"), cp.asarray(y))

    # Fit with precomputed kernel
    reg_precomputed = cu_svm.SVR(kernel="precomputed", C=1.0, epsilon=0.1)
    reg_precomputed.fit(cp.asarray(K, order="F"), cp.asarray(y))

    # Should have same number of support vectors
    assert reg_linear.n_support_ == reg_precomputed.n_support_

    # Predictions should be close
    preds_linear = cp.asnumpy(reg_linear.predict(cp.asarray(X, order="F")))
    preds_precomputed = cp.asnumpy(
        reg_precomputed.predict(cp.asarray(K, order="F"))
    )
    assert np.allclose(preds_linear, preds_precomputed, rtol=1e-3, atol=1e-3)


def test_svr_precomputed_kernel_validation():
    """Test that precomputed kernel validates input shape."""
    np.random.seed(42)
    n_samples = 50

    # Non-square kernel matrix should raise error
    K_nonsquare = np.random.randn(n_samples, n_samples + 1).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32)

    cuml_reg = cu_svm.SVR(kernel="precomputed")
    with pytest.raises(ValueError, match="square"):
        cuml_reg.fit(cp.asarray(K_nonsquare), cp.asarray(y))


@pytest.mark.parametrize("n_classes", [3, 4])
@pytest.mark.parametrize("kernel_func", ["linear", "rbf"])
def test_svc_precomputed_multiclass_ovr(n_classes, kernel_func):
    """Test SVC with precomputed kernel for multiclass using OneVsRest.

    OneVsRest uses all samples for each binary classifier (one class vs rest),
    so the kernel matrix is NOT subsetted and remains square. This works with
    precomputed kernels, unlike OneVsOne which subsets to class pairs.
    """
    X, y = make_blobs(
        n_samples=100, n_features=10, centers=n_classes, random_state=42
    )
    X = X.astype(np.float64)
    y = y.astype(np.int32)

    # Compute kernel matrix
    if kernel_func == "linear":
        K = X @ X.T
    else:  # rbf
        gamma = 0.1
        sq_dists = (
            np.sum(X**2, axis=1, keepdims=True)
            + np.sum(X**2, axis=1)
            - 2 * X @ X.T
        )
        K = np.exp(-gamma * sq_dists)

    # Convert to cupy
    K_cp = cp.asarray(K, order="F")
    y_cp = cp.asarray(y)

    # Fit sklearn with OvR
    sklearn_clf = svm.SVC(
        kernel="precomputed", C=1.0, decision_function_shape="ovr"
    )
    sklearn_clf.fit(K, y)
    sklearn_preds = sklearn_clf.predict(K)
    sklearn_accuracy = np.mean(sklearn_preds == y)

    # Fit cuML with OvR (uses OneVsRestClassifier which doesn't subset samples)
    cuml_clf = cu_svm.SVC(
        kernel="precomputed", C=1.0, decision_function_shape="ovr"
    )
    cuml_clf.fit(K_cp, y_cp)
    cuml_preds = cp.asnumpy(cuml_clf.predict(K_cp))
    cuml_accuracy = np.mean(cuml_preds == y)

    # Both should achieve high accuracy on well-separated data
    assert sklearn_accuracy > 0.9, (
        f"sklearn accuracy too low: {sklearn_accuracy}"
    )
    assert cuml_accuracy > 0.9, f"cuML accuracy too low: {cuml_accuracy}"

    # Predictions should match
    assert np.array_equal(sklearn_preds, cuml_preds), (
        f"Predictions don't match for {n_classes}-class {kernel_func} kernel. "
        f"sklearn acc: {sklearn_accuracy}, cuML acc: {cuml_accuracy}"
    )


def test_svc_precomputed_multiclass_ovo_fails():
    """Test that precomputed kernel with OvO multiclass fails as expected.

    OneVsOne subsets samples to class pairs, resulting in non-square kernel
    matrices which are invalid for precomputed kernels.
    """
    X, y = make_blobs(n_samples=90, n_features=5, centers=3, random_state=42)
    X = X.astype(np.float64)
    y = y.astype(np.int32)

    # Compute linear kernel matrix
    K = X @ X.T
    K_cp = cp.asarray(K, order="F")
    y_cp = cp.asarray(y)

    # OvO with precomputed kernel should fail because OneVsOneClassifier
    # subsets rows but not columns, resulting in non-square matrices
    cuml_clf = cu_svm.SVC(
        kernel="precomputed", C=1.0, decision_function_shape="ovo"
    )
    with pytest.raises(ValueError, match="square"):
        cuml_clf.fit(K_cp, y_cp)


def test_svr_fit_homogeneous_y():
    """SVR fit with homogeneous y has empty coefs and just an intercept"""
    X = np.eye(3)
    y = np.full(3, 1.23)
    model = cuml.SVR().fit(X, y)
    assert (model.intercept_ == 1.23).all()
    assert model.dual_coef_.shape == (1, 0)
    assert (model.predict(X) == y).all()
