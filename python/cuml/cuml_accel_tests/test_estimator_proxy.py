# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import inspect
import os
import pickle
import subprocess
import sys
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest
import scipy.sparse
import sklearn
from packaging.version import Version
from sklearn.base import check_is_fitted, is_classifier, is_regressor
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import (
    ElasticNet,
    LinearRegression,
    LogisticRegression,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from cuml.accel import is_proxy
from cuml.accel.estimator_proxy import ProxyBase

SKLEARN_18 = Version(sklearn.__version__) >= Version("1.8.0.dev0")
SKLEARN_19 = Version(sklearn.__version__) >= Version("1.9.0.dev0")
SKLEARN_110 = Version(sklearn.__version__) >= Version("1.10.0.dev0")

requires_array_api = pytest.mark.skipif(
    not SKLEARN_18,
    reason="scikit-learn >= 1.8 required for array-api enabled estimators",
)


def test_is_proxy():
    class Foo:
        pass

    assert is_proxy(LogisticRegression)
    assert is_proxy(LogisticRegression())
    assert not is_proxy(Foo)
    assert not is_proxy(Foo())


def test_class_metadata():
    cpu_cls = LogisticRegression._cpu_class

    assert LogisticRegression.__name__ == cpu_cls.__name__
    assert LogisticRegression.__qualname__ == cpu_cls.__qualname__
    assert LogisticRegression.__doc__ == cpu_cls.__doc__

    cls = importlib.import_module(
        LogisticRegression.__module__
    ).LogisticRegression
    assert cls is LogisticRegression

    cpu_sig = inspect.signature(cpu_cls)
    assert inspect.signature(LogisticRegression) == cpu_sig


def test_method_metadata():
    cpu_cls = LogisticRegression._cpu_class

    assert LogisticRegression.fit.__name__ == cpu_cls.fit.__name__
    assert LogisticRegression.fit.__doc__ == cpu_cls.fit.__doc__

    cpu_sig = inspect.signature(cpu_cls.fit)
    assert inspect.signature(LogisticRegression.fit) == cpu_sig


def test_init_no_parameters_signature():
    """On Python < 3.13 `inspect.signature` has a bug that led to `__init__`s
    of proxies with no parameters mistakenly having `*args, **kwargs` in
    the signature. This test checks that we successfully work around that."""
    assert not inspect.signature(LabelEncoder).parameters


def test_sklearn_introspect_estimator_type():
    if not SKLEARN_18:
        assert LogisticRegression._estimator_type == "classifier"
        assert LogisticRegression()._estimator_type == "classifier"
    assert is_classifier(LogisticRegression())
    assert is_regressor(LinearRegression())


def test_sklearn_introspect_parameter_constraints():
    assert isinstance(LogisticRegression._parameter_constraints, dict)
    assert isinstance(LogisticRegression()._parameter_constraints, dict)


def test_sklearn_utils_get_tags():
    from sklearn.utils import get_tags

    model = LogisticRegression()
    assert get_tags(model) == get_tags(model._cpu)


def test_BaseEstimator__validate_params():
    model = LogisticRegression()
    model._validate_params()

    model.C = "oops"
    with pytest.raises(Exception, match="C"):
        model._validate_params()


def test_BaseEstimator__get_param_names():
    names = LogisticRegression._get_param_names()
    sol = LogisticRegression._cpu_class._get_param_names()
    assert names == sol


def test_init_positional_and_keyword():
    model = PCA()
    assert model.n_components is None

    model = PCA(n_components=10)
    assert model.n_components == 10

    model = PCA(10)
    assert model.n_components == 10

    with pytest.raises(TypeError):
        # Can't pass keyword-only parameters in as positional
        PCA(10, False)

    model = IncrementalPCA()
    assert model.n_components is None

    model = IncrementalPCA(n_components=10)
    assert model.n_components == 10

    model = IncrementalPCA(10)
    assert model.n_components == 10


def test_repr():
    model = LogisticRegression(C=1.5)
    assert str(model) == str(model._cpu)
    assert repr(model) == repr(model._cpu)
    # smoketest _repr_mimebundle_. It changes per-call, so can't directly compare
    # TODO: On some runs this `_repr_mimebundle_` can error with a
    # `UnicodeDecodeError`. I suspect a misconfigured `LC_ALL`/`LANG` on the CI
    # machine, but :shrug:. All we care about is that things are plumbed
    # properly, so ignoring this error here for now. This is repeated twice
    # below as well.
    # See https://github.com/rapidsai/cuml/issues/8212.
    try:
        mimebundle = model._repr_mimebundle_()
    except UnicodeDecodeError:
        pass
    else:
        assert isinstance(mimebundle, dict)


def test_repr_mimebundle():
    model = LogisticRegression(C=1.5)
    X, y = make_classification()
    # TODO: see https://github.com/rapidsai/cuml/issues/8212
    try:
        html_repr = model._repr_mimebundle_()["text/html"]
    except UnicodeDecodeError:
        pass
    else:
        assert "<span>Not fitted</span>" in html_repr

    model.fit(X, y)
    try:
        html_repr = model._repr_mimebundle_()["text/html"]
    except UnicodeDecodeError:
        pass
    else:
        assert "<span>Fitted</span>" in html_repr


def test_repr_mimebundle_display_text():
    """If `display="text"` is configured, no host sync is needed"""
    X, y = make_classification()
    model = LogisticRegression(C=1.5).fit(X, y)
    with sklearn.config_context(display="text"):
        # The conditional definition of `_repr_html_` is proxied through
        assert not hasattr(model, "_repr_html_")
        mimebundle = model._repr_mimebundle_()
    # No host sync needed to repr
    assert not hasattr(model._cpu, "n_features_in_")
    assert "text/html" not in mimebundle


def test_pipeline_repr():
    """sklearn's pretty printer requires you not override __repr__
    for pipelines to repr properly"""
    model = LogisticRegression(C=1.5)
    pipe = Pipeline([("cls", model)])
    native = Pipeline([("cls", model._cpu)])
    assert str(pipe) == str(native)
    assert repr(pipe) == repr(native)
    # TODO: see https://github.com/rapidsai/cuml/issues/8212
    try:
        mimebundle = pipe._repr_mimebundle_()
    except UnicodeDecodeError:
        pass
    else:
        assert isinstance(mimebundle, dict)


def test_dir():
    params = {"C", "fit_intercept"}
    attrs = {"n_features_in_", "coef_", "intercept_"}
    methods = {"fit", "predict"}
    private = {"_cpu", "_gpu", "_synced", "_call_method"}

    model = LogisticRegression(C=1.5)
    names = set(dir(model))
    assert names.issuperset(params | methods)
    assert names.isdisjoint(attrs | private)

    X, y = make_classification()
    model.fit(X, y)
    names = set(dir(model))
    assert names.issuperset(params | attrs | methods)
    assert names.isdisjoint(private)


def test_getattr():
    model = LogisticRegression(C=1.5, fit_intercept=False)
    assert model.C == 1.5
    assert model.fit_intercept is False

    # Never proxy through private attributes
    model._cpu._xxx_private_attr = "some_value"
    with pytest.raises(AttributeError):
        model._xxx_private_attr

    # Unfit, no fit attributes
    with pytest.raises(AttributeError):
        model.coef_

    X, y = make_classification()
    model.fit(X, y)
    # Fit attributes now available
    assert model.coef_ is model._cpu.coef_


def test_getattr_supports_select_private_attributes():
    X, y = make_blobs(random_state=42)

    # Private attributes error on unfit models
    model = NearestNeighbors()
    assert not hasattr(model, "_tree")

    # Some estimators can still expose select private attrs
    model = NearestNeighbors().fit(X)
    assert model._gpu is not None
    assert model._tree is None
    assert model._fit_method == "brute"

    # But missing attributes still error appropriately
    assert not hasattr(model, "_oops_not_a_real_attr")


def test_not_implemented_attr_error():
    X, y = make_regression()
    model = ElasticNet()

    msg = (
        "The `ElasticNet.dual_gap_` attribute is not yet "
        "implemented in `cuml.accel`"
    )

    # For unfit models the original error is raised
    with pytest.raises(AttributeError) as rec:
        model.dual_gap_
    assert msg not in str(rec.value)

    model.fit(X, y)
    # Fit models raise the nicer error message
    with pytest.raises(AttributeError, match=msg):
        model.dual_gap_

    # If trained on CPU though then there's no error
    model2 = ElasticNet(positive=True).fit(X, y)
    assert hasattr(model2, "dual_gap_")


def test_setattr():
    model = LinearRegression()

    # Hyperparameters are forwarded to CPU
    model.fit_intercept = False
    assert not model._cpu.fit_intercept
    assert model._gpu is None

    # Fit uses current hyperparameters
    X, y = make_regression(n_samples=10)
    model.fit(X, y)
    assert not model._gpu.fit_intercept

    # Changing hyperparameters forwards to both
    model.fit_intercept = True
    assert model._cpu.fit_intercept
    assert model._gpu.fit_intercept

    # But changing to an unsupported value causes fallback to CPU,
    # ensuring fit attributes are forwarded to CPU
    model.positive = True
    assert model._cpu.positive
    assert model._cpu.coef_ is not None
    assert model._gpu is None

    # Changing the value of fit attributes causes fallback to CPU.
    # This should never happen in normal workflows, but we want
    # to handle even the weird things.
    model = LinearRegression().fit(X, y)
    model.coef_ = [1, 2, 3]
    assert model._gpu is None
    assert model.coef_ == [1, 2, 3]


def test_delattr():
    X, y = make_regression(n_samples=10)
    model = LinearRegression().fit(X, y)
    del model.n_features_in_
    # Deleting an attribute causes fallback to CPU. This should never happen in
    # normal workflows, but we want to handle even the weird things.
    assert model._gpu is None
    assert model._cpu.coef_ is not None
    assert not hasattr(model._cpu, "n_features_in_")
    # Smoketest that deleting internal state works. We never do this internally.
    del model._synced
    assert not hasattr(model, "_synced")


def test_get_params():
    model = LogisticRegression(
        C=1.5,
        fit_intercept=False,
        solver="newton-cholesky",
    )
    params = model.get_params()
    assert params["C"] == 1.5
    assert not params["fit_intercept"]
    assert params["solver"] == "newton-cholesky"
    assert params == model._cpu.get_params()

    params2 = model.get_params(deep=False)
    assert params2 == model._cpu.get_params(deep=False)
    assert model.get_params.__doc__ == model._cpu.get_params.__doc__


def test_set_params():
    model = LinearRegression()

    # Hyperparameters are forwarded to CPU
    assert model.set_params(fit_intercept=False) is model
    assert not model._cpu.fit_intercept
    assert model._gpu is None

    # Fit uses current hyperparameters
    X, y = make_regression(n_samples=10)
    model.fit(X, y)
    assert not model._gpu.fit_intercept

    # Changing hyperparameters forwards to both
    model.set_params(fit_intercept=True)
    assert model._cpu.fit_intercept
    assert model._gpu.fit_intercept

    # But changing to an unsupported value causes fallback to CPU,
    # ensuring fit attributes are forwarded to CPU
    model.set_params(positive=True)
    assert model._cpu.positive
    assert model._cpu.coef_ is not None
    assert model._gpu is None


@pytest.mark.parametrize("fit", [False, True])
def test_clone(fit):
    model = LogisticRegression(
        C=1.5,
        fit_intercept=False,
        solver="newton-cholesky",
    )
    params = model.get_params()

    if fit:
        X, y = make_classification(n_samples=10)
        model.fit(X, y)

    model2 = sklearn.clone(model)

    # GPU state not initialized
    assert model2._gpu is None
    assert not model2._synced
    # Parameters copied
    assert model2.get_params() == params
    # Fit attributes not copied
    assert not hasattr(model2, "n_features_in_")


def test_pickle_proxy_estimator_class():
    """Check that the *class* can be roundtripped via pickle"""
    cls = pickle.loads(pickle.dumps(LogisticRegression))
    assert cls is LogisticRegression


def test_pickle_unfit():
    model = LogisticRegression(C=1.5, solver="newton-cholesky")

    model2 = pickle.loads(pickle.dumps(model))

    assert model2.get_params() == model.get_params()
    assert model2._gpu is None
    assert not model2._synced


def test_pickle_gpu_fit():
    X, y = make_classification(n_samples=10)
    model = LogisticRegression(C=1.5, solver="newton-cholesky")
    model.fit(X, y)
    # Ensure the test case is one that is GPU accelerated
    assert model._gpu is not None

    model2 = pickle.loads(pickle.dumps(model))

    assert model2.get_params() == model.get_params()
    # GPU model exists and is fit
    assert model2._gpu is not None
    assert model2._gpu.coef_ is not None
    # CPU model has fit attributes cleared to reduce host memory
    assert not hasattr(model2._cpu, "coef_")
    assert not model2._synced


def test_pickle_cpu_fit():
    X, y = make_regression(n_samples=10)
    model = LinearRegression(positive=True)
    model.fit(X, y)
    # Ensure the test case is one that isn't GPU accelerated
    assert model._gpu is None

    model2 = pickle.loads(pickle.dumps(model))

    assert model2.get_params() == model.get_params()
    # GPU model doesn't exist
    assert model2._gpu is None
    # CPU model has fit attributes
    assert model2._cpu.coef_ is not None
    assert not model2._synced


def test_pickle_cpu_fit_but_params_supported():
    """Test that a model fit on CPU isn't reloaded to the GPU on unpickle, even
    if the hyperparameters are supported."""
    X, y = make_regression(n_samples=10)
    X[0, 0] = np.nan
    model = RandomForestRegressor().fit(X, y)
    # ensure the test case is one that isn't GPU accelerated
    assert model._gpu is None

    model2 = pickle.loads(pickle.dumps(model))
    # GPU model doesn't exist
    assert model2._gpu is None
    # CPU model has fit attributes
    assert model2._cpu.n_features_in_
    assert not model2._synced


def test_unpickle_cuml_accel_not_active():
    """Unpickling in an process without cuml.accel enabled uses the CPU model"""
    X, y = make_classification(n_samples=10)
    model = LogisticRegression(C=1.5, solver="newton-cholesky")
    model.fit(X, y)

    buf = pickle.dumps((model, X, y))
    script = dedent(
        f"""
        import pickle

        model, X, y = pickle.loads({buf!r})

        assert model.C == 1.5
        assert model.solver == "newton-cholesky"

        from cuml.accel import enabled
        from cuml.accel.estimator_proxy import ProxyBase
        from sklearn.linear_model import LogisticRegression

        # Unpickling hasn't installed the accelerator or patched sklearn
        assert not enabled()
        assert not issubclass(LogisticRegression, ProxyBase)

        # Is a scikit-learn model of expected type
        assert type(model) is LogisticRegression
        assert not hasattr(model, "_cpu")

        # Can run inference
        model.score(X, y)
        """
    )
    env = os.environ.copy()
    env.pop("CUML_ACCEL_ENABLED", None)
    res = subprocess.run(
        [sys.executable, "-c", script],
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    # Pull out attributes before assert for nicer error reporting on failure
    returncode = res.returncode
    stdout = res.stdout
    assert returncode == 0, stdout


def test_unpickle_cuml_not_installed():
    """Unpickling in an environment without cuml installed uses the CPU model"""
    X, y = make_classification(n_samples=10)
    model = LogisticRegression(C=1.5, solver="newton-cholesky")
    model.fit(X, y)

    buf = pickle.dumps((model, X, y))
    script = dedent(
        f"""
        import pickle
        import sys

        # This prevents cuml from being imported
        sys.modules["cuml"] = None

        model, X, y = pickle.loads({buf!r})

        assert model.C == 1.5
        assert model.solver == "newton-cholesky"

        # Is a scikit-learn model of expected type
        from sklearn.linear_model import LogisticRegression
        assert type(model) is LogisticRegression
        assert not hasattr(model, "_cpu")

        # Can run inference
        model.score(X, y)
        """
    )
    env = os.environ.copy()
    env.pop("CUML_ACCEL_ENABLED", None)
    res = subprocess.run(
        [sys.executable, "-c", script],
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    # Pull out attributes before assert for nicer error reporting on failure
    returncode = res.returncode
    stdout = res.stdout
    assert returncode == 0, stdout


def test_fit_gpu():
    X, y = make_regression(n_samples=10)
    model = LinearRegression()

    # Model isn't fit
    with pytest.raises(NotFittedError):
        check_is_fitted(model)

    assert model.fit(X, y) is model
    # Fit happened on GPU, attrs kept on GPU
    check_is_fitted(model)
    assert model._gpu is not None
    assert not hasattr(model._cpu, "n_features_in_")

    # Access something that requires moving to CPU
    assert model.coef_ is not None
    assert hasattr(model._cpu, "n_features_in_")

    # Refitting resets CPU estimator
    assert model.fit(X, y) is model
    assert not hasattr(model._cpu, "n_features_in_")


def test_fit_unsupported_params():
    """Hyperparameters not supported on GPU"""
    X, y = make_regression(n_samples=10)
    model = LinearRegression(positive=True)
    assert model.fit(X, y) is model
    # Fit happened on CPU
    check_is_fitted(model)
    assert model._gpu is None
    assert hasattr(model._cpu, "n_features_in_")


def test_fit_unsupported_args():
    """Hyperparameters supported on GPU, but X/y type isn't"""
    X_dense, y = make_regression(
        n_samples=100, n_features=200, random_state=42
    )
    X_dense[X_dense < 2.5] = 0.0
    X = scipy.sparse.coo_matrix(X_dense)
    model = RandomForestRegressor()
    assert model.fit(X, y) is model
    # Fit happened on CPU
    check_is_fitted(model)
    assert model._gpu is None
    assert hasattr(model._cpu, "n_features_in_")


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_fit_warm_start():
    """Some sklearn estimators support a `warm_start` parameter where `fit`
    updates an existing model rather than resetting it. cuml doesn't
    support this. Here we check that fitting on CPU in this case doesn't
    clear the prior CPU model but updates it"""
    X, y = make_regression(
        n_samples=50,
        n_features=200,
        n_informative=10,
        random_state=42,
    )
    # 2 fits of 5 iters each w/ warm start should be equal to 1 fit of 10 iters
    m1 = ElasticNet(random_state=42, max_iter=5, warm_start=True)
    m1.fit(X, y).fit(X, y)
    m2 = ElasticNet(random_state=42, max_iter=10)
    m2.fit(X, y)
    np.testing.assert_allclose(m1.coef_, m2.coef_)


def test_fit_gpu_predict_gpu():
    X, y = make_regression(n_samples=10)
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    assert isinstance(y_pred, np.ndarray)
    # fit attributes not transferred to host
    assert not hasattr(model._cpu, "n_features_in_")


def test_fit_cpu_predict_cpu():
    X, y = make_regression(n_samples=10)
    model = LinearRegression(positive=True).fit(X, y)
    y_pred = model.predict(X)
    assert isinstance(y_pred, np.ndarray)
    # cpu estimator used
    assert model._gpu is None


def test_common_fit_attributes():
    X, y = make_regression(n_samples=10, n_features=5)
    model = LinearRegression()
    assert not hasattr(model, "n_features_in_")
    model.fit(X, y)
    assert model.n_features_in_ == X.shape[1]


def test_fit_validates_params():
    X, y = make_classification()
    model = LogisticRegression(C="oops")

    with pytest.raises(Exception, match="C"):
        model.fit(X, y)


def test_method_that_only_exists_on_cpu_estimator():
    """For methods that cuml doesn't implement, we fallback to CPU
    before executing."""
    X, y = make_classification(n_samples=10)
    model = LogisticRegression().fit(X, y)
    # Fit on GPU, no host transfer
    assert model._gpu is not None
    assert not hasattr(model._cpu, "n_features_in_")
    # Sanity check in case this method is implemented in cuml in the future
    assert not hasattr(model._gpu, "sparsify")
    assert model.sparsify() is model
    # Unknown method caused host transfer
    assert hasattr(model._cpu, "n_features_in_")
    # Method was run on host
    assert scipy.sparse.issparse(model.coef_)


@pytest.mark.parametrize(
    "methods",
    [
        "set_output,fit,transform",
        "fit,set_output,transform",
        "set_output,fit_transform",
    ],
)
def test_set_output(methods):
    X, _ = make_classification(n_samples=100, random_state=42)

    model = PCA(n_components=5)

    # Test calling methods in different order
    for method in methods.split(","):
        if method == "set_output":
            # set_output returns self
            assert model.set_output(transform="pandas") is model
        elif method == "fit":
            model.fit(X)
        elif method == "fit_transform":
            out = model.fit_transform(X)
        else:
            assert method == "transform"
            out = model.transform(X)

    # Transform outputs a pandas dataframe with the proper columns
    assert isinstance(out, pd.DataFrame)
    assert out.columns[0].startswith("pca")
    # No host transfer required
    assert not hasattr(model._cpu, "n_features_in_")

    # If input has an index, the output has an aligned index
    X_df = pd.DataFrame(
        X,
        columns=[f"x{i}" for i in range(X.shape[1])],
        index=[f"row{i}" for i in range(X.shape[0])],
    )
    out = model.transform(X_df)
    assert (out.index == X_df.index).all()

    # Can change back to default without requiring a host transfer either
    model.set_output(transform="default")
    out = model.transform(X)
    assert isinstance(out, np.ndarray)
    # No host transfer required
    assert not hasattr(model._cpu, "n_features_in_")


def test_incremental_pca_partial_fit():
    X, _ = make_classification(n_samples=40, n_features=8, random_state=42)
    model = IncrementalPCA(n_components=3, batch_size=10)

    model.partial_fit(X[:20])
    gpu = model._gpu

    assert gpu is not None
    assert int(gpu.n_samples_seen_) == 20

    model.partial_fit(X[20:])

    assert model._gpu is gpu
    assert int(model._gpu.n_samples_seen_) == 40
    assert not hasattr(model._cpu, "components_")

    out = model.transform(X)
    assert isinstance(out, np.ndarray)
    assert out.shape[1] == 3


def test_incremental_pca_set_output():
    X, _ = make_classification(n_samples=40, n_features=8, random_state=42)
    model = IncrementalPCA(n_components=3, batch_size=10)

    assert model.set_output(transform="pandas") is model
    model.partial_fit(X)

    out = model.transform(X)
    assert isinstance(out, pd.DataFrame)
    assert out.columns[0].startswith("incrementalpca")


@pytest.mark.parametrize("method", ["fit_transform", "partial_fit"])
def test_incremental_pca_rejects_unsupported_fit_params(method):
    X, _ = make_classification(n_samples=40, n_features=8, random_state=42)
    model = IncrementalPCA(n_components=3, batch_size=10)

    with pytest.raises(TypeError):
        getattr(model, method)(X, sample_weight=np.ones(X.shape[0]))


def test_get_feature_names_out():
    model = PCA(n_components=5)

    # Calling on an unfit model raises appropriately
    with pytest.raises(NotFittedError):
        model.get_feature_names_out()

    X, _ = make_classification(random_state=42)
    model.fit(X)

    # Calling on fit model returns appropriate column names
    res = model.get_feature_names_out()
    assert isinstance(res, np.ndarray)
    assert res.dtype == "object"
    assert res[0].startswith("pca")

    # No host transfer required
    assert not hasattr(model._cpu, "n_features_in_")


@pytest.fixture
def metadata_routing():
    with sklearn.config_context(enable_metadata_routing=True):
        yield


@pytest.mark.parametrize("fitted", [False, True])
def test_metadata_routing(metadata_routing, fitted):
    model = LogisticRegression()

    if fitted:
        X, y = make_classification(n_samples=10)
        model.fit(X, y)

    # Check default metadata
    routing = model.get_metadata_routing()
    assert routing.fit.requests.get("sample_weight") is None
    assert routing.score.requests.get("sample_weight") is None

    # Check can set and get metadata
    assert model.set_fit_request(sample_weight=True) is model
    assert model.set_score_request(sample_weight=False) is model

    # Check all different ways sklearn accesses metadata report the same
    def check(routing):
        assert routing.fit.requests["sample_weight"] is True
        assert routing.score.requests["sample_weight"] is False

    check(model.get_metadata_routing())
    check(model._get_metadata_request())
    check(model._metadata_request)

    # Smoketest internal methods
    if SKLEARN_18:
        requests = LogisticRegression._get_class_level_metadata_request_values(
            "fit"
        )
        assert requests.get("sample_weight") is None
    else:
        defaults = LogisticRegression._get_default_requests()
        assert defaults.fit.requests.get("sample_weight") is None

    # No method caused host transfer
    assert not hasattr(model._cpu, "n_features_in_")


def test_metadata_routing_consumed(metadata_routing):
    """Test that a proxy estimator is a valid consumer of metadata"""
    X, y = make_classification(random_state=42)
    weights = np.random.default_rng(42).uniform(low=0.5, size=y.shape)

    lr = LogisticRegression()
    lr.set_fit_request(sample_weight=True)
    lr.set_score_request(sample_weight=False)

    search = GridSearchCV(estimator=lr, param_grid={"C": [0.9, 1]})
    search.fit(X, y, sample_weight=weights)


@requires_array_api
def test_array_api_proxy_pandas_inputs():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    df = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
    model = StandardScaler().fit(df)

    # get_feature_names_out works as expected
    assert (model.get_feature_names_out() == df.columns).all()

    # Can transform pandas input
    out = model.transform(df)
    assert isinstance(out, np.ndarray)

    # set_output works as expected
    model.set_output(transform="pandas")
    out = model.transform(df)
    assert isinstance(out, pd.DataFrame)
    assert (out.columns == df.columns).all()

    # No host transfer required
    assert not hasattr(model._cpu, "n_features_in_")


@requires_array_api
def test_array_api_proxy_global_transform_output():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    model = StandardScaler().fit(X)

    with sklearn.config_context(transform_output="pandas"):
        out = model.transform(X)
    assert isinstance(out, pd.DataFrame)


@requires_array_api
def test_array_api_proxy_partial_fit():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    model = StandardScaler().fit(X)

    model2 = StandardScaler()
    model2.partial_fit(X[:25])
    assert model2._gpu._internal_model.n_samples_seen_ == 25

    # 2nd partial fit doesn't reset internal state
    model2.partial_fit(X[25:])
    assert model2._gpu._internal_model.n_samples_seen_ == X.shape[0]

    # fit and partial_fit result in close values
    sol = model.transform(X)
    res = model2.transform(X)
    np.testing.assert_allclose(sol, res)

    # No host transfer required
    assert not hasattr(model._cpu, "n_features_in_")


@pytest.mark.parametrize(
    "transform",
    [
        pytest.param(lambda x: x.astype(str).astype(object), id="object"),
        pytest.param(lambda x: x.tolist(), id="list"),
    ],
)
@requires_array_api
def test_array_api_proxy_coerce_inputs(transform):
    X, _ = make_blobs(n_features=5, n_samples=10, random_state=42)
    X = transform(X)
    model = StandardScaler().fit(X)
    assert model.n_features_in_ == 5


@requires_array_api
def test_array_api_proxy_pickle():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    df = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
    model = StandardScaler().fit(df)

    model2 = pickle.loads(pickle.dumps(model))

    # hyperparameters roundtrip
    assert model2.get_params() == model.get_params()

    # GPU model exists and is fit
    assert model2._gpu is not None
    assert model2._gpu._internal_model.mean_ is not None

    # CPU model has fit attributes cleared to reduce host memory
    assert not hasattr(model2._cpu, "mean_")
    assert not model2._synced

    # Check location of attribute storage on _ArrayAPIWrapper
    # - n_features_in_ set on both wrapper and internal model
    assert model2._gpu.n_features_in_ == model.n_features_in_
    assert model2._gpu._internal_model.n_features_in_ == model.n_features_in_
    # - feature_names_in_ only set on wrapper
    assert (model2._gpu.feature_names_in_ == model.feature_names_in_).all()
    assert not hasattr(model2._gpu._internal_model, "feature_names_in_")
    # - Other attributes only set on internal model
    assert not hasattr(model2._gpu, "mean_")
    assert hasattr(model2._gpu._internal_model, "mean_")


@pytest.mark.skipif(
    SKLEARN_18, reason="Test checks fallback for scikit-learn < 1.8"
)
def test_array_api_proxy_fallback_older_sklearn():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    model = StandardScaler().fit(X)
    # Ran on CPU
    assert model._gpu is None


@pytest.mark.parametrize(
    "Base",
    [
        pytest.param(LinearRegression, id="ProxyBase"),
        pytest.param(RandomForestRegressor, id="ProxyBase-with-ABCMeta"),
        pytest.param(StandardScaler, id="ArrayAPIProxyBase"),
    ],
)
def test_subclass_of_proxy_isnt_accelerated(Base):
    class Sub(Base):
        pass

    X, y = make_regression(n_samples=200, random_state=42)

    base_model = Base().fit(X, y)
    sub_model = Sub().fit(X, y)

    # A base class and instance are proxies
    assert is_proxy(Base)
    assert is_proxy(base_model)

    # issubclass/isinstance work on true proxy classes/instances
    assert issubclass(Base, ProxyBase)
    assert isinstance(base_model, Base)
    assert isinstance(base_model, ProxyBase)

    # A subclass (and instance) are not proxies
    assert not is_proxy(Sub)
    assert not is_proxy(sub_model)

    # A subclass doesn't have any concrete proxy bases in its mro:
    assert Base not in Sub.__mro__
    # A subclass does have the original CPU model in its mro:
    assert Base._cpu_class in Sub.__mro__

    # A subclass still identifies as a subclass
    assert issubclass(Sub, Base)
    assert issubclass(Sub, ProxyBase)
    # A subclass instance still identifies as an instance
    assert isinstance(sub_model, Base)
    assert isinstance(sub_model, ProxyBase)


@pytest.mark.skipif(not SKLEARN_19, reason="scikit-learn >= 1.9 required")
def test_set_callbacks(capsys):
    from sklearn.callback import ProgressBar

    bar = ProgressBar()

    X, y = make_classification()

    # Initial fit on GPU
    model = LogisticRegression().fit(X, y)
    assert model._gpu is not None

    # set_callbacks runs on CPU but doesn't trigger a host migration
    model.set_callbacks(bar)
    assert model._cpu._skl_callbacks
    assert model._gpu is not None

    # inference still runs on GPU, callbacks only affect fit
    y = model.predict(X)
    assert model._gpu is not None
    assert not hasattr(model._cpu, "n_features_in_")

    # refitting with a callback configured runs on CPU
    model.fit(X, y)
    assert model._gpu is None

    # Check the ProgressBar output something
    captured = capsys.readouterr()
    assert "LogisticRegression" in captured.out

    # resetting callbacks allows GPU execution again
    model.set_callbacks()
    model.fit(X, y)
    assert model._gpu is not None


@pytest.mark.skipif(not SKLEARN_19, reason="scikit-learn >= 1.9 required")
def test_set_callbacks_partial_fit(capsys):
    from sklearn.callback import ProgressBar

    bar = ProgressBar()

    X, _ = make_regression(n_samples=200)
    X1, X2 = X[:100], X[100:200]

    # Initial fit on GPU
    model = StandardScaler().partial_fit(X1)
    assert model._gpu is not None
    assert model._gpu._internal_model.n_samples_seen_ == 100

    # set_callbacks runs on CPU but doesn't trigger a host migration
    model.set_callbacks(bar)
    assert model._cpu._skl_callbacks
    assert model._gpu is not None

    # partial_fit leads to a host migration and running on CPU
    model.partial_fit(X2)
    assert model._gpu is None

    # Check the ProgressBar output something
    captured = capsys.readouterr()
    assert "StandardScaler" in captured.out

    # host migration didn't drop fitted state
    assert model._cpu.n_samples_seen_ == 200


@pytest.mark.skipif(not SKLEARN_19, reason="scikit-learn >= 1.9 required")
def test_set_callbacks_meta_estimator(capsys):
    from sklearn.callback import ProgressBar

    X, y = make_classification()
    scaler = StandardScaler()
    model = LogisticRegression()
    pipeline = Pipeline([("scaler", scaler), ("model", model)])

    # Non-propagated callbacks on a pipeline don't lead to fallback
    bar = ProgressBar(0)
    pipeline.set_callbacks(bar)
    pipeline.fit(X, y)
    assert scaler._gpu is not None
    assert model._gpu is not None

    # Check the ProgressBar output something
    captured = capsys.readouterr()
    assert "Pipeline" in captured.out

    # Propagated callbacks on a pipeline work, but can lead to fallback
    bar = ProgressBar(None)
    pipeline.set_callbacks(bar)
    pipeline.fit(X, y)
    assert scaler._gpu is None
    assert model._gpu is None

    # Check the ProgressBar output something
    captured = capsys.readouterr()
    assert "StandardScaler" in captured.out
    assert "LogisticRegression" in captured.out


@pytest.mark.skipif(not SKLEARN_110, reason="scikit-learn >= 1.10 required")
def test_private_set_callbacks():
    from sklearn.callback import ProgressBar

    bar = ProgressBar()

    X, y = make_classification()

    model = LogisticRegression().fit(X, y)
    assert model._gpu is not None

    # Setting callbacks through the private API returns the proxy,
    # which makes this different from the other methods we forward.
    assert model._set_callbacks([bar]) is model
    assert model._cpu._skl_callbacks == [bar]
    assert model._gpu is not None

    assert model._set_callbacks([]) is model
    assert not hasattr(model._cpu, "_skl_callbacks")
