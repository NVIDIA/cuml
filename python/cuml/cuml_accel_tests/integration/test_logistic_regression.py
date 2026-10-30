#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import sklearn
from packaging import version
from packaging.version import Version
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

SKLEARN_18 = Version(sklearn.__version__) >= Version("1.8.0.dev0")
SKLEARN_110 = Version(sklearn.__version__) >= Version("1.10.0.dev0")


@pytest.fixture(scope="module")
def classification_data():
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_classes=3,
        n_informative=10,
        random_state=42,
    )
    return X, y


# TODO: Once sklearn <1.10 is no longer supported, replace this legacy penalty test
# with l1_ratio/alpha cases and remove its warning filters.
@pytest.mark.filterwarnings(
    "ignore:The solver 'liblinear' was deprecated in version 1.10.*:FutureWarning"
)
@pytest.mark.parametrize(
    "penalty, solver",
    [
        ("l1", "liblinear"),
        ("l1", "saga"),
        ("l2", "lbfgs"),
        ("l2", "liblinear"),
        ("l2", "sag"),
        ("l2", "saga"),
        ("elasticnet", "saga"),
        (None, "lbfgs"),
        (None, "saga"),
    ],
)
@pytest.mark.filterwarnings(
    "ignore:'penalty' was deprecated in version 1.8.*:FutureWarning"
)
def test_logistic_regression_penalty(classification_data, penalty, solver):
    X, y = classification_data
    kwargs = {"penalty": penalty, "solver": solver, "max_iter": 200}
    if penalty == "elasticnet":
        kwargs["l1_ratio"] = 0.5  # l1_ratio is required for elasticnet
    clf = LogisticRegression(**kwargs).fit(X, y)
    y_pred = clf.predict(X)
    accuracy_score(y, y_pred)


# TODO: Once sklearn <1.12 is no longer supported, remove this legacy dual test.
@pytest.mark.filterwarnings(
    "ignore:The solver 'liblinear' was deprecated in version 1.10.*:FutureWarning"
)
@pytest.mark.filterwarnings(
    "ignore:The parameter 'dual' was deprecated in version 1.10.*:FutureWarning"
)
@pytest.mark.parametrize("dual", [True, False])
def test_logistic_regression_dual(classification_data, dual):
    X, y = classification_data
    # dual=True is only supported with L2 regularization and solver='liblinear'.
    clf = LogisticRegression(solver="liblinear", dual=dual, max_iter=200).fit(
        X, y
    )
    y_pred = clf.predict(X)
    accuracy_score(y, y_pred)


@pytest.mark.parametrize("tol", [1e-2])
def test_logistic_regression_tol(classification_data, tol):
    X, y = classification_data
    clf = LogisticRegression(tol=tol, max_iter=200).fit(X, y)
    y_pred = clf.predict(X)
    accuracy_score(y, y_pred)


@pytest.mark.parametrize("strength", [0.0, 0.01, 0.1, 1.0, 10.0, 100.0])
def test_logistic_regression_regularization(classification_data, strength):
    X, y = classification_data
    C = float("inf") if strength == 0 else 1.0 / strength
    # TODO: Once sklearn <1.10 is no longer supported, pass alpha directly.
    kwargs = {"alpha": strength} if SKLEARN_110 else {"C": C}
    model = LogisticRegression(**kwargs, max_iter=200).fit(X, y)
    assert model._gpu is not None
    assert model._gpu.C == C
    assert model.predict(X).shape == y.shape


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_logistic_regression_fit_intercept(classification_data, fit_intercept):
    X, y = classification_data
    clf = LogisticRegression(fit_intercept=fit_intercept, max_iter=200).fit(
        X, y
    )
    y_pred = clf.predict(X)
    accuracy_score(y, y_pred)


# TODO: Once sklearn <1.12 is no longer supported, remove this legacy scaling test.
@pytest.mark.filterwarnings(
    "ignore:The solver 'liblinear' was deprecated in version 1.10.*:FutureWarning"
)
@pytest.mark.filterwarnings(
    "ignore:The parameter 'intercept_scaling' was deprecated in version 1.10.*:FutureWarning"
)
@pytest.mark.parametrize("intercept_scaling", [0.5, 1.0, 2.0])
def test_logistic_regression_intercept_scaling(intercept_scaling):
    # Cannot use multi-class data for this test as using the liblinear solver
    # with multi-class data is deprecated in scikit-learn version 1.5 and will
    # be removed in version 1.8.
    X, y = make_classification(random_state=42)
    # 'intercept_scaling' is only used when solver='liblinear' and fit_intercept=True
    clf = LogisticRegression(
        solver="liblinear",
        fit_intercept=True,
        intercept_scaling=intercept_scaling,
        max_iter=200,
    ).fit(X, y)
    assert (clf._gpu is not None) == (intercept_scaling == 1.0)
    assert hasattr(clf._cpu, "coef_") == (intercept_scaling != 1.0)
    y_pred = clf.predict(X)
    accuracy_score(y, y_pred)


def test_logistic_regression_default_intercept_scaling_uses_gpu():
    X, y = make_classification(random_state=42)
    model = LogisticRegression().fit(X, y)
    assert model._gpu is not None
    assert not hasattr(model._cpu, "coef_")


@pytest.mark.parametrize("class_weight", [None, "balanced"])
def test_logistic_regression_class_weight(classification_data, class_weight):
    X, y = classification_data
    clf = LogisticRegression(class_weight=class_weight, max_iter=200).fit(X, y)
    y_pred = clf.predict(X)
    accuracy_score(y, y_pred)


def test_logistic_regression_class_weight_custom(classification_data):
    X, y = classification_data
    class_weights = {0: 1, 1: 2, 2: 1}
    clf = LogisticRegression(class_weight=class_weights, max_iter=200).fit(
        X, y
    )
    y_pred = clf.predict(X)
    accuracy_score(y, y_pred)


# TODO: Once sklearn <1.12 is no longer supported, remove the liblinear case
# and filter.
@pytest.mark.filterwarnings(
    "ignore:The solver 'liblinear' was deprecated in version 1.10.*:FutureWarning"
)
@pytest.mark.parametrize(
    "solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
)
def test_logistic_regression_solver(classification_data, solver):
    X, y = classification_data
    clf = LogisticRegression(solver=solver, max_iter=200).fit(X, y)
    y_pred = clf.predict(X)
    accuracy_score(y, y_pred)


@pytest.mark.parametrize("max_iter", [50, 100, 200, 500])
def test_logistic_regression_max_iter(classification_data, max_iter):
    X, y = classification_data
    clf = LogisticRegression(max_iter=max_iter).fit(X, y)
    y_pred = clf.predict(X)
    accuracy_score(y, y_pred)


# TODO: Once sklearn <1.8 is no longer supported, remove this entire multi_class test
# together with its warning filters, xfail, and skip guard.
@pytest.mark.filterwarnings(
    "ignore:.*multi_class.*was deprecated.*:FutureWarning"
)
# Ignore DeprecationWarning from sklearn's internal call to scipy.optimize.minimize
# with the L-BFGS-B method. sklearn passes deprecated 'disp' and 'iprint' parameters
# to scipy, which is not in our control. This filter can be removed once the minimum
# supported sklearn version is 1.7.1+ (where this was fixed).
# TODO: Once sklearn <1.7.1 is no longer supported, remove this SciPy warning filter.
@pytest.mark.filterwarnings(
    "ignore:scipy.optimize.*disp.*iprint.*:DeprecationWarning"
)
@pytest.mark.xfail(
    version.parse(sklearn.__version__) >= version.parse("1.8.0"),
    reason="multi_class parameter removed in scikit-learn 1.8",
)
@pytest.mark.parametrize(
    "multi_class, solver",
    [
        ("ovr", "liblinear"),
        ("ovr", "lbfgs"),
        ("multinomial", "lbfgs"),
        ("multinomial", "newton-cg"),
        ("multinomial", "sag"),
        ("multinomial", "saga"),
        ("auto", "lbfgs"),
        ("auto", "liblinear"),
    ],
)
@pytest.mark.skipif(SKLEARN_18, reason="parameter removed in sklearn 1.8")
def test_logistic_regression_multi_class(
    classification_data, multi_class, solver
):
    X, y = classification_data
    if (
        version.parse(sklearn.__version__) >= version.parse("1.7.0")
        and solver == "liblinear"
    ):
        pytest.skip(
            "liblinear does not support multinomial multi_class in scikit-learn 1.7+"
        )
    elif solver == "liblinear" and multi_class == "multinomial":
        pytest.skip("liblinear does not support multinomial multi_class")
    clf = LogisticRegression(
        multi_class=multi_class, solver=solver, max_iter=200
    ).fit(X, y)
    y_pred = clf.predict(X)
    accuracy_score(y, y_pred)


# Ignore DeprecationWarning from sklearn's internal call to scipy.optimize.minimize
# with the L-BFGS-B method (default solver). sklearn passes deprecated 'disp' and
# 'iprint' parameters to scipy, which is not in our control. This filter can be
# removed once the minimum supported sklearn version is 1.7.1+ (where this was fixed).
# TODO: Once sklearn <1.7.1 is no longer supported, remove this SciPy warning filter.
@pytest.mark.filterwarnings(
    "ignore:scipy.optimize.*disp.*iprint.*:DeprecationWarning"
)
@pytest.mark.parametrize("warm_start", [True, False])
def test_logistic_regression_warm_start(classification_data, warm_start):
    X, y = classification_data
    clf = LogisticRegression(warm_start=warm_start, max_iter=200).fit(X, y)
    y_pred = clf.predict(X)
    accuracy_score(y, y_pred)


@pytest.mark.parametrize("l1_ratio", [0.0, 0.5, 1.0])
def test_logistic_regression_l1_ratio(classification_data, l1_ratio):
    X, y = classification_data
    # TODO: Once sklearn <1.8 is no longer supported, remove kwargs and pass l1_ratio
    # directly without the legacy penalty parameter.
    kwargs = {} if SKLEARN_18 else {"penalty": "elasticnet"}
    clf = LogisticRegression(
        **kwargs, solver="saga", l1_ratio=l1_ratio, max_iter=200
    ).fit(X, y)
    y_pred = clf.predict(X)
    accuracy_score(y, y_pred)


# TODO: Once sklearn <1.10 is no longer supported, remove the alpha availability guard.
# Once sklearn <1.14 is no longer supported, remove this entire C/alpha conflict test.
@pytest.mark.skipif(not SKLEARN_110, reason="alpha added in sklearn 1.10")
def test_logistic_regression_C_and_alpha(classification_data):
    # Falls back to CPU so that sklearn raises its error
    X, y = classification_data
    with pytest.raises(ValueError, match="not both"):
        LogisticRegression(C=2.0, alpha=0.5).fit(X, y)
