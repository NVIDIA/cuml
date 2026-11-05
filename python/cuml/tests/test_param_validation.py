#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import math

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cuml
from cuml.internals.base import Base
from cuml.internals.outputs import (
    FIT_LIKE_METHODS,
    exit_internal_context,
    mlfunc,
)
from cuml.internals.param_validation import (
    AnyOf,
    ArrayLike,
    Bool,
    Callable,
    Float,
    Int,
    InvalidParameterError,
    Options,
    RandomState,
    as_constraint,
    get_parameter_constraints,
    validate_params,
)
from cuml.linear_model import LogisticRegression
from cuml.linear_model.ridge_mg import RidgeMG
from cuml.multiclass import OneVsOneClassifier, OneVsRestClassifier
from cuml.testing.utils import get_all_base_subclasses


@pytest.mark.parametrize(
    "constraint, valid, invalid",
    [
        (Int(), [0, -3, np.int64(2), np.uint8(1)], [1.0, True, "1", None]),
        (Int(ge=1), [1, 10], [0, -1]),
        (Int(gt=0, le=3), [1, 3], [0, 4]),
        (Int(ge=0, lt=3), [0, 2], [-1, 3]),
        (
            Float(),
            [0.5, 1, np.float32(0.5), np.int32(1), math.inf],
            [math.nan, True, np.bool_(True), "0.5", None],
        ),
        (Float(ge=0), [0, 0.0, 2.5], [-1e-12, -1]),
        (Float(gt=0, lt=1), [0.5], [0, 1, 0.0, 1.0]),
        (
            Options("auto", "eig"),
            ["auto", "eig", np.str_("auto")],
            ["AUTO", "svd", None, np.array([1, 2])],
        ),
        (as_constraint(dict), [{}], [[], None]),
        (Bool, [True, False, np.bool_(True)], [0, 1, "True"]),
        (as_constraint(None), [None], [0, False, ""]),
        (
            ArrayLike,
            [
                [1, 2],
                (1, 2),
                np.zeros(3),
                cp.zeros(3),
                pd.Series([1.0]),
                np.array(1.0),
            ],
            [1.0, np.float64(1.0), "abc", None],
        ),
        (Callable, [len, lambda: 1], [1, "len"]),
        (
            RandomState,
            [
                None,
                0,
                2**32 - 1,
                np.random.RandomState(0),
                cp.random.RandomState(0),
            ],
            [-1, 2**32, 1.0, np.random.default_rng(0)],
        ),
    ],
)
def test_constraint_accepts(constraint, valid, invalid):
    for value in valid:
        assert constraint.accepts(value)
    for value in invalid:
        assert not constraint.accepts(value)


@pytest.mark.parametrize(
    "constraint",
    [
        Int(),
        Int(ge=1),
        Int(gt=0, lt=10),
        Int(ge=-1, le=5),
        Float(ge=0),
        Float(gt=0, le=1),
        Float(ge=-1, lt=1e20),
        Options("a", "b"),
        Bool,
        AnyOf(Int(ge=1), None),
        AnyOf(Float(ge=0), ArrayLike),
        AnyOf(Bool, Int()),
        RandomState,
    ],
)
def test_invalid_examples_are_rejected(constraint):
    for value in constraint.invalid_examples():
        assert not constraint.accepts(value)


def test_anyof_invalid_examples_filters_overlap():
    # `True` is an invalid `Int`, but a valid bool
    constraint = AnyOf(Bool, Int(ge=0))
    assert True not in constraint.invalid_examples()
    assert -1 in constraint.invalid_examples()


@pytest.mark.parametrize(
    "constraint, expected",
    [
        (Int(), "an instance of 'int'"),
        (Int(ge=1), "an int in the range [1, inf)"),
        (Int(gt=0, lt=5), "an int in the range (0, 5)"),
        (Float(), "an instance of 'float'"),
        (Float(gt=0, le=1), "a float in the range (0.0, 1.0]"),
        (Float(le=1), "a float in the range (-inf, 1.0]"),
        (Options("auto", "eig"), "a str among {'auto', 'eig'}"),
        (
            AnyOf(Options("auto"), Int(ge=1)),
            "a str among {'auto'} or an int in the range [1, inf)",
        ),
        (as_constraint(dict), "an instance of 'dict'"),
        (
            Bool,
            "an instance of 'bool' or an instance of 'numpy.bool'",
        ),
        (AnyOf(Int(ge=1), None), "an int in the range [1, inf) or None"),
        (
            AnyOf(Float(ge=0), ArrayLike, None),
            "a float in the range [0.0, inf), an array-like or None",
        ),
        (
            RandomState,
            "an int in the range [0, 4294967295], an instance of "
            "'numpy.random.mtrand.RandomState', an instance of "
            "'cupy.random._generator.RandomState' or None",
        ),
    ],
)
def test_constraint_str(constraint, expected):
    assert str(constraint) == expected


@pytest.mark.parametrize(
    "spec, replacement",
    [
        (bool, "Bool"),
        (int, r"Int\(\)"),
        (float, r"Float\(\)"),
    ],
)
def test_builtin_types_are_rejected(spec, replacement):
    with pytest.raises(TypeError, match=f"use {replacement} instead"):
        as_constraint(spec)


@pytest.mark.parametrize("spec", ["random_state", [Int()], (Int(), None)])
def test_unsupported_spec(spec):
    with pytest.raises(TypeError, match="Unsupported parameter constraint"):
        as_constraint(spec)


@pytest.mark.parametrize("spec", [Int(), bool, (Int(), None)])
def test_constraints_must_be_a_list(spec):
    class _Estimator(Base):
        _parameter_constraints = {"x": spec}

    with pytest.raises(TypeError, match="_Estimator.x must be a list"):
        get_parameter_constraints(_Estimator)


@pytest.mark.parametrize("kwargs", [{"ge": 0, "gt": 0}, {"le": 1, "lt": 1}])
def test_number_bounds_exclusive(kwargs):
    with pytest.raises(ValueError, match="Only one of"):
        Int(**kwargs)


def test_invalid_parameter_error_is_value_and_type_error():
    assert issubclass(InvalidParameterError, ValueError)
    assert issubclass(InvalidParameterError, TypeError)


class _Parent(Base):
    _parameter_constraints = {"a": [Int(ge=0)], "b": [Bool]}

    def __init__(self, *, a=1, b=True, verbose=False, output_type=None):
        super().__init__(verbose=verbose, output_type=output_type)
        self.a = a
        self.b = b

    @classmethod
    def _get_param_names(cls):
        return [*super()._get_param_names(), "a", "b"]


class _Child(_Parent):
    # Overrides `a`, inherits `b`, adds `c`
    _parameter_constraints = {"a": [Int(ge=10)], "c": [Float()]}

    def __init__(self, *, c=0.5, a=10, **kwargs):
        super().__init__(a=a, **kwargs)
        self.c = c

    @classmethod
    def _get_param_names(cls):
        return [*super()._get_param_names(), "c"]


def test_constraints_inherited():
    constraints = get_parameter_constraints(_Child)
    assert set(constraints) == {"verbose", "output_type", "a", "b", "c"}
    assert str(constraints["a"]) == "an int in the range [10, inf)"


def test_validate_params():
    validate_params(_Child())

    with pytest.raises(
        InvalidParameterError,
        match=(
            r"^The 'a' parameter of _Child must be an int in the range "
            r"\[10, inf\). "
            r"Got 1 instead.$"
        ),
    ):
        validate_params(_Child(a=1))

    with pytest.raises(
        InvalidParameterError,
        match=(
            "The 'b' parameter of _Child must be an instance of 'bool' or an "
            "instance of 'numpy.bool'. Got 'yes' instead."
        ),
    ):
        validate_params(_Child(b="yes"))


@pytest.mark.parametrize(
    "params",
    [
        {"verbose": "loud"},
        {"verbose": 1.5},
        {"output_type": "invalid"},
        {"output_type": "NumPy"},
    ],
)
def test_validate_base_params(params):
    with pytest.raises(InvalidParameterError):
        validate_params(_Parent(**params))


@pytest.mark.parametrize(
    "params",
    [
        {"verbose": True},
        {"verbose": 5},
        {"output_type": None},
        {"output_type": "input"},
        {"output_type": "cupy"},
    ],
)
def test_validate_base_params_valid(params):
    validate_params(_Parent(**params))


def test_params_without_constraints_are_not_checked():
    class _NoConstraints(Base):
        def __init__(self, *, x=1, verbose=False, output_type=None):
            super().__init__(verbose=verbose, output_type=output_type)
            self.x = x

        @classmethod
        def _get_param_names(cls):
            return [*super()._get_param_names(), "x"]

    validate_params(_NoConstraints(x=object()))


class _Fittable(Base):
    _parameter_constraints = {"a": [Int(ge=0)]}

    def __init__(self, *, a=1, inner=None, verbose=False, output_type=None):
        super().__init__(verbose=verbose, output_type=output_type)
        self.a = a
        self.inner = inner

    @classmethod
    def _get_param_names(cls):
        return [*super()._get_param_names(), "a", "inner"]

    @mlfunc(set_input_type=True)
    def fit(self, X, y=None):
        if self.inner is not None:
            # Use a user provided estimator like a meta-estimator should
            # (see the developer guide): exit the internal context so the
            # inner estimator is validated, and pin the output type
            with exit_internal_context(), cuml.using_output_type("numpy"):
                self.inner.fit(X)
                pred = self.inner.predict(X)
            # numpy-only code, like sklearn's `OneVsRestClassifier.predict`
            maxima = np.zeros(X.shape[0])
            np.maximum(maxima, pred, out=maxima)
        return self

    @mlfunc(preserve_index=True)
    def predict(self, X):
        return cp.zeros(X.shape[0])

    @mlfunc(set_input_type=True)
    def fit_transform(self, X, y=None):
        return self.fit(X)

    @mlfunc
    def fit_predict(self, X, y=None):
        return self.fit(X)


def test_fit_validates_params():
    X = np.zeros((3, 2))
    _Fittable(a=0).fit(X)
    with pytest.raises(
        InvalidParameterError, match="'a' parameter of _Fittable"
    ):
        _Fittable(a=-1).fit(X)
    # Base parameters are validated too
    with pytest.raises(
        InvalidParameterError, match="'output_type' parameter of _Fittable"
    ):
        _Fittable(output_type="invalid").fit(X)


@pytest.mark.parametrize("method", ["fit_transform", "fit_predict"])
def test_fit_like_methods_validate_params(method):
    # `fit_predict` isn't decorated with `set_input_type=True`, it is
    # validated because of its name. Both call `fit` internally.
    X = np.zeros((3, 2))
    getattr(_Fittable(), method)(X)
    with pytest.raises(
        InvalidParameterError, match="'a' parameter of _Fittable"
    ):
        getattr(_Fittable(a=-1), method)(X)


def test_exit_internal_context_validates_inner_estimator():
    # The outer estimator is valid, the error comes from the inner one
    inner = _Fittable(a=-1)
    with pytest.raises(
        InvalidParameterError,
        match="'a' parameter of _Fittable must .* Got -1",
    ):
        _Fittable(inner=inner).fit(np.zeros((3, 2)))


@pytest.mark.parametrize("output_type", [None, "cupy", "cudf", "pandas"])
def test_inner_estimator_output_type_does_not_leak(output_type):
    # A user provided estimator's `output_type` must not change what the
    # meta-estimator gets back from it
    inner = _Fittable(output_type=output_type)
    _Fittable(inner=inner).fit(np.zeros((3, 2)))


# Estimators whose parameters don't all have constraints yet. Remove an
# estimator from this list once it declares `_parameter_constraints` for all
# its parameters.
NOT_YET_CONVERTED = {
    "AgglomerativeClustering",
    "BernoulliNB",
    "CategoricalNB",
    "ColumnTransformer",
    "ComplementNB",
    "CountVectorizer",
    "DBSCAN",
    "ElasticNet",
    "EmpiricalCovariance",
    "GaussianNB",
    "GaussianRandomProjection",
    "HDBSCAN",
    "HashingVectorizer",
    "IncrementalPCA",
    "IsolationForest",
    "KBinsDiscretizer",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    "KernelDensity",
    "KernelRidge",
    "LabelBinarizer",
    "LabelEncoder",
    "Lars",
    "Lasso",
    "LedoitWolf",
    "LinearRegression",
    "LinearSVC",
    "LinearSVR",
    "LogisticRegression",
    "MBSGDClassifier",
    "MBSGDRegressor",
    "MaxAbsScaler",
    "MinMaxScaler",
    "MissingIndicator",
    "MultinomialNB",
    "NearestNeighbors",
    "Normalizer",
    "OneHotEncoder",
    "OneVsOneClassifier",
    "OneVsRestClassifier",
    "OrdinalEncoder",
    "PCA",
    "PolynomialFeatures",
    "PowerTransformer",
    "QuantileTransformer",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "RobustScaler",
    "SVC",
    "SVR",
    "SimpleImputer",
    "SparseRandomProjection",
    "SpectralClustering",
    "SpectralEmbedding",
    "StandardScaler",
    "TSNE",
    "TargetEncoder",
    "TfidfTransformer",
    "TfidfVectorizer",
    "TruncatedSVD",
    "UMAP",
}

ESTIMATORS = {
    name: cls
    for name, cls in get_all_base_subclasses().items()
    if "Base" not in name
    and cls.__module__.startswith("cuml.")
    and not cls.__module__.startswith("cuml.solvers")
}
CONVERTED = sorted(set(ESTIMATORS) - NOT_YET_CONVERTED)


def test_ridge_mg_narrows_constraints():
    # RidgeMG supports a scalar alpha and a subset of Ridge's solvers
    validate_params(RidgeMG(handle=None, alpha=0.5, solver="svd"))
    with pytest.raises(
        InvalidParameterError, match="'alpha' parameter of RidgeMG"
    ):
        validate_params(RidgeMG(handle=None, alpha=[1.0, 2.0]))
    with pytest.raises(
        InvalidParameterError, match="'solver' parameter of RidgeMG"
    ):
        validate_params(RidgeMG(handle=None, solver="lsmr"))


# Estimators that can't be constructed with default arguments
_INSTANCES = {
    OneVsOneClassifier: lambda: OneVsOneClassifier(LogisticRegression()),
    OneVsRestClassifier: lambda: OneVsRestClassifier(LogisticRegression()),
}


def _default_instance(cls):
    return _INSTANCES.get(cls, cls)()


def test_not_yet_converted_is_up_to_date():
    assert NOT_YET_CONVERTED <= set(ESTIMATORS)


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            name,
            marks=pytest.mark.xfail(
                name in NOT_YET_CONVERTED,
                reason=(
                    "Not converted to declarative parameter validation yet, "
                    "see https://github.com/NVIDIA/cuml/issues/8498"
                ),
                strict=True,
            ),
        )
        for name in sorted(ESTIMATORS)
    ],
)
def test_all_params_have_constraints(name):
    cls = ESTIMATORS[name]
    params = set(_default_instance(cls).get_params(deep=False))
    missing = params - set(get_parameter_constraints(cls))
    assert not missing, f"{name} has no constraints for {sorted(missing)}"


@pytest.mark.parametrize("name", sorted(ESTIMATORS))
def test_no_constraints_for_unknown_params(name):
    cls = ESTIMATORS[name]
    own = set(vars(cls).get("_parameter_constraints", {}))
    params = set(_default_instance(cls).get_params(deep=False))
    assert own <= params, f"{name} has constraints for {sorted(own - params)}"


class _BadType:
    def __repr__(self):
        return "<bad type>"


@pytest.mark.parametrize("name", CONVERTED)
def test_invalid_params_raise_in_fit(name):
    cls = ESTIMATORS[name]
    X = np.arange(20, dtype="float32").reshape(10, 2)
    y = np.array([0, 1] * 5)
    methods = [m for m in FIT_LIKE_METHODS if hasattr(cls, m)]
    assert methods

    for param, constraint in get_parameter_constraints(cls).items():
        for value in [_BadType(), *constraint.invalid_examples()]:
            for method in methods:
                est = _default_instance(cls).set_params(**{param: value})
                with pytest.raises(
                    InvalidParameterError,
                    match=rf"The '{param}' parameter of {name} must be .* Got .* instead.",
                ):
                    getattr(est, method)(X, y)
