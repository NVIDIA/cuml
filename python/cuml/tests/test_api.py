#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import inspect

import cupy as cp
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.utils import get_tags

import cuml
import cuml.internals.mixins as cumix
from cuml.internals.base import Base
from cuml.internals.mixins import CumlTags
from cuml.testing.utils import ClassEnumerator

###############################################################################
#                        Helper functions and classes                         #
###############################################################################


def func_positional_arg(func):
    if hasattr(func, "__wrapped__"):
        return func_positional_arg(func.__wrapped__)

    elif hasattr(func, "__code__"):
        all_args = func.__code__.co_argcount
        if func.__defaults__ is not None:
            kwargs = len(func.__defaults__)
        else:
            kwargs = 0
        return all_args - kwargs
    return 2


@pytest.fixture(scope="session")
def dataset():
    X, y = make_classification(100, 5, random_state=42)
    X = X.astype(np.float64)
    y = y.astype(np.float64)
    return X, y


models = ClassEnumerator(module=cuml).get_models()

cuml_tags_mixins = {
    cumix.FMajorInputTagMixin: {"preferred_input_order": "F"},
    cumix.CMajorInputTagMixin: {"preferred_input_order": "C"},
    cumix.SparseInputTagMixin: {
        "X_types_gpu": ["2darray", "sparse"],
    },
    cumix.StringInputTagMixin: {
        "X_types_gpu": ["2darray", "string"],
    },
}


class dummy_regressor_estimator(cumix.RegressorMixin, Base):
    pass


class dummy_classifier_estimator(cumix.ClassifierMixin, Base):
    pass


class dummy_cluster_estimator(cumix.ClusterMixin, Base):
    pass


class dummy_class_with_tags(
    cumix.FMajorInputTagMixin, cumix.CMajorInputTagMixin, cumix.TagsMixin
):
    pass


class dummy_sparse_nan_estimator(
    cumix.SparseInputTagMixin, cumix.AllowNaNTagMixin, Base
):
    pass


def uninitialized_model(model):
    return model.__new__(model)


###############################################################################
#                               Tags Tests                                    #
###############################################################################


@pytest.mark.parametrize("model", list(models.values()))
def test_tags_api(model):
    assert not hasattr(model, "_get_tags")
    assert not hasattr(model, "_get_" + "cuml_tags")
    assert hasattr(model, "__sklearn_tags__")

    model_tags = uninitialized_model(model).__sklearn_tags__()
    sklearn_tags = get_tags(uninitialized_model(model))

    for tags in [model_tags, sklearn_tags]:
        assert isinstance(tags, CumlTags)
        if tags.preferred_input_order is not None:
            assert isinstance(tags.preferred_input_order, str)
        assert isinstance(tags.X_types_gpu, list)


def test_cuml_tags_and_composition():
    tags = dummy_class_with_tags().__sklearn_tags__()
    print(dummy_class_with_tags.__mro__)

    # Under cooperative super, the leftmost mixin in MRO applies its mutation
    # last (on the way back up the super-chain), so FMajorInputTagMixin
    # overrides CMajorInputTagMixin.
    assert tags.preferred_input_order == "F"


@pytest.mark.parametrize("mixin", cuml_tags_mixins.keys())
def test_cuml_tag_mixins(mixin):
    class TaggedEstimator(mixin, Base):
        pass

    tags = get_tags(TaggedEstimator())
    assert isinstance(tags, CumlTags)
    for tag, value in cuml_tags_mixins[mixin].items():
        assert getattr(tags, tag) == value


def test_sklearn_tag_mixins():
    tags = get_tags(dummy_sparse_nan_estimator())
    assert isinstance(tags, CumlTags)
    assert tags.input_tags.sparse
    assert tags.input_tags.allow_nan
    assert tags.X_types_gpu == ["2darray", "sparse"]


@pytest.mark.parametrize(
    "model",
    [
        dummy_cluster_estimator,
        dummy_regressor_estimator,
        dummy_classifier_estimator,
    ],
)
def test_estimator_type_mixins(model):
    assert hasattr(model, "_estimator_type")
    tags = model().__sklearn_tags__()
    if model._estimator_type in ["regressor", "classifier"]:
        assert tags.target_tags.required
    else:
        assert not tags.target_tags.required


@pytest.mark.parametrize("model", list(models.values()))
def test_mro(model):
    found_base = False
    for cl in reversed(inspect.getmro(model.__class__)):
        if cl == Base:
            if found_base:
                pytest.fail("Found Base class twice in the MRO")
            else:
                found_base = True


###############################################################################
#                            Fit Function Tests                               #
###############################################################################


@pytest.mark.parametrize("model_name", list(models.keys()))
# ignore random forest float64 warnings
@pytest.mark.filterwarnings("ignore:To use pickling or GPU-based")
@pytest.mark.filterwarnings(
    "ignore:`cuml.tsa.*`, along with the entire `cuml.tsa` module, was "
    "deprecated:FutureWarning"
)
def test_fit_function(dataset, model_name):
    # This test ensures that our estimators return self after a call to fit
    if model_name in [
        "TSNE",
        "TruncatedSVD",
        "AutoARIMA",
        "MultinomialNB",
        "LabelEncoder",
    ]:
        pytest.xfail("These models are not tested yet")

    n_pos_args_constr = func_positional_arg(models[model_name].__init__)

    if model_name in ["SparseRandomProjection", "GaussianRandomProjection"]:
        model = models[model_name](n_components=2)
    elif model_name in ["ARIMA", "AutoARIMA", "ExponentialSmoothing"]:
        model = models[model_name](np.random.normal(0.0, 1.0, (10,)))
    elif model_name in ["RandomForestClassifier", "RandomForestRegressor"]:
        model = models[model_name](n_bins=32)
    elif model_name == "KMeans":
        model = models[model_name](n_init="auto")
    else:
        if n_pos_args_constr == 1:
            model = models[model_name]()
        elif n_pos_args_constr == 2:
            model = models[model_name](5)
        else:
            model = models[model_name](5, 5)

    if hasattr(model, "fit"):
        # Unfortunately co_argcount doesn't work with decorated functions,
        # and the inspect module doesn't work with Cython. Therefore we need
        # to register the number of arguments manually if `fit` is decorated
        pos_args_spec = {
            "ARIMA": 1,
            "ElasticNet": 3,
            "Lasso": 3,
            "LinearRegression": 3,
            "LogisticRegression": 3,
            "NearestNeighbors": 2,
            "PCA": 2,
            "Ridge": 3,
            "UMAP": 2,
        }
        n_pos_args_fit = (
            pos_args_spec[model_name]
            if model_name in pos_args_spec
            else func_positional_arg(models[model_name].fit)
        )

        X, y = dataset

        if model_name == "RandomForestClassifier":
            y = y.astype(np.int32)
            assert model.fit(X, y) is model
        else:
            if n_pos_args_fit == 1:
                assert model.fit() is model
            elif n_pos_args_fit == 2:
                assert model.fit(X) is model
            else:
                assert model.fit(X, y) is model

        # test classifiers correctly set self.classes_ during fit
        if hasattr(model, "_estimator_type"):
            if model._estimator_type == "classifier":
                cp.testing.assert_array_almost_equal(
                    model.classes_, np.unique(y)
                )
