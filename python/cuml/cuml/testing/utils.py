# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import inspect
from textwrap import dedent, indent

import cupy as cp
import numpy as np
import pytest
from numba import cuda

from cuml.internals.base import Base
from cuml.internals.outputs import convert_arrays


def array_difference(a, b, with_sign=True):
    """
    Utility function to compute the difference between 2 arrays.
    """
    a = as_numpy(a)
    b = as_numpy(b)

    if len(a) == 0 and len(b) == 0:
        return 0

    if not with_sign:
        a, b = np.abs(a), np.abs(b)
    return np.sum(np.abs(a - b))


class array_equal:
    """
    Utility functor to compare 2 numpy arrays and optionally show a meaningful
    error message in case they are not. Two individual elements are assumed
    equal if they are within `unit_tol` of each other, and two arrays are
    considered equal if less than `total_tol` percentage of elements are
    different.

    """

    def __init__(self, a, b, unit_tol=1e-4, total_tol=1e-4, with_sign=True):
        self.a = np.atleast_1d(as_numpy(a))
        self.b = np.atleast_1d(as_numpy(b))
        self.unit_tol = unit_tol
        self.total_tol = total_tol
        self.with_sign = with_sign

    def compute_difference(self):
        return array_difference(self.a, self.b, with_sign=self.with_sign)

    def __bool__(self):
        if len(self.a) == len(self.b) == 0:
            return True

        if self.with_sign:
            a, b = self.a, self.b
        else:
            a, b = np.abs(self.a), np.abs(self.b)

        res = (np.sum(np.abs(a - b) > self.unit_tol)) / a.size < self.total_tol
        return bool(res)

    def __eq__(self, other):
        if isinstance(other, bool):
            return bool(self) == other
        return super().__eq__(other)

    def _repr(self, threshold=None):
        name = self.__class__.__name__

        return (
            [f"<{name}: "]
            + f"{np.array2string(self.a, threshold=threshold)} ".splitlines()
            + f"{np.array2string(self.b, threshold=threshold)} ".splitlines()
            + [
                f"unit_tol={self.unit_tol} ",
                f"total_tol={self.total_tol} ",
                f"with_sign={self.with_sign}",
                ">",
            ]
        )

    def __repr__(self):
        return "".join(self._repr(threshold=5))

    def __str__(self):
        tokens = self._repr(threshold=1000)
        return "\n".join(
            f"{'    ' if 0 < n < len(tokens) - 1 else ''}{token}"
            for n, token in enumerate(tokens)
        )


def assert_array_equal(a, b, unit_tol=1e-4, total_tol=1e-4, with_sign=True):
    """
    Raises an AssertionError if arrays are not considered equal.

    Uses the same arguments as array_equal(), but raises an AssertionError in
    case that the test considers the arrays to not be equal.

    This function will generate a nicer error message in the context of pytest
    compared to a plain `assert array_equal(...)`.
    """
    # Determine array equality.
    equal = array_equal(
        a, b, unit_tol=unit_tol, total_tol=total_tol, with_sign=with_sign
    )
    if not equal:
        # Generate indented array string representation ...
        str_a = indent(np.array2string(a), "   ").splitlines()
        str_b = indent(np.array2string(b), "   ").splitlines()
        # ... and add labels
        str_a[0] = f"a: {str_a[0][3:]}"
        str_b[0] = f"b: {str_b[0][3:]}"

        # Create assertion error message and raise exception.
        assertion_error_msg = dedent(
            f"""
        Arrays are not equal

        unit_tol:  {unit_tol}
        total_tol: {total_tol}
        with_sign: {with_sign}

        """
        ) + "\n".join(str_a + str_b)
        raise AssertionError(assertion_error_msg)


def as_type(output_type, *arrays):
    """Convert input arrays (cupy or numpy) to the requested output type"""
    out = convert_arrays(
        arrays, output_type=output_type, one_col_2d_as_series=False
    )
    return out[0] if len(out) == 1 else tuple(out)


def as_numpy(*arrays):
    """Convert input array-likes to numpy arrays"""
    out = []
    for x in arrays:
        if hasattr(x, "to_numpy"):
            # XXX: ensure copy for pandas, some downstream sklearn functions
            # don't work for non-writable buffers.
            out.append(x.to_numpy(copy=True))
        elif x is None or np.isscalar(x):
            out.append(x)
        else:
            out.append(cp.asnumpy(x))
    return out[0] if len(out) == 1 else tuple(out)


def as_cupy(*arrays):
    """Convert input array-likes to cupy arrays"""
    out = []
    for x in arrays:
        if hasattr(x, "to_cupy"):
            out.append(x.to_cupy())
        elif hasattr(x, "to_numpy"):
            out.append(cp.asarray(x.to_numpy()))
        elif x is None or cp.isscalar(x):
            out.append(x)
        else:
            out.append(cp.asarray(x))
    return out[0] if len(out) == 1 else tuple(out)


def assert_dbscan_equal(ref, actual, X, core_indices, eps):
    """
    Utility function to compare two numpy label arrays.
    The labels of core/noise points are expected to be equal, and the labels
    of border points are verified by finding a neighboring core point with the
    same label.
    """
    core_set = set(core_indices)
    N, _ = X.shape
    eps2 = eps**2

    def sqnorm(x):
        return np.inner(x, x)

    for i in range(N):
        la, lb = ref[i], actual[i]

        if i in core_set:  # core point
            assert la == lb, (
                "Core point mismatch at #{}: {} (expected {})".format(
                    i, lb, la
                )
            )
        elif la == -1:  # noise point
            assert lb == -1, "Noise mislabelled at #{}: {}".format(i, lb)
        else:  # border point
            found = False
            for j in range(N):
                # Check if j is a core point with the same label
                if j in core_set and lb == actual[j]:
                    # Check if j is a neighbor of i
                    if sqnorm(X[i] - X[j]) <= eps2:
                        found = True
                        break
            assert found, (
                "Border point not connected to cluster at #{}: "
                "{} (reference: {})".format(i, lb, la)
            )

    # Note: we can also do it in a rand score fashion by checking that pairs
    # correspond in both label arrays for core points, if we need to drop the
    # requirement of minimality for core points


def unit_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.unit)


def quality_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.quality)


def stress_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.stress)


class ClassEnumerator:
    """Helper class to automatically pick up every models classes in a module.
    Filters out classes not inheriting from cuml.Base.

    Parameters
    ----------
    module: python module (ex: cuml.linear_regression)
        The module for which to retrieve models.
    exclude_classes: list of classes (optional)
        Those classes will be filtered out from the retrieved models.
    custom_constructors: dictionary of {class_name: lambda}
        Custom constructors to use instead of the default one.
        ex: {'LinearSVC': lambda: cuml.LinearSVC(n_streams=2)}
    recursive: bool, default=False
        Instructs the class to recursively search submodules when True,
        otherwise only classes in the specified model will be enumerated
    """

    def __init__(
        self,
        module,
        exclude_classes=None,
        custom_constructors=None,
        recursive=False,
    ):
        self.module = module
        self.exclude_classes = exclude_classes or []
        self.custom_constructors = custom_constructors or []
        self.recursive = recursive

    def _get_classes(self):
        def recurse_module(module):
            classes = {}

            modules = []

            if self.recursive:
                modules = inspect.getmembers(module, inspect.ismodule)

            # Enumerate child modules only if they are a submodule of the
            # current one. i.e. `{parent_module}.{submodule}`
            for _, m in modules:
                if module.__name__ + "." in m.__name__:
                    classes.update(recurse_module(m))

            # Ensure we only get classes that are part of this module
            classes.update(
                {
                    (".".join((klass.__module__, klass.__qualname__))): klass
                    for name, klass in inspect.getmembers(
                        module, inspect.isclass
                    )
                    if module.__name__ + "."
                    in ".".join((klass.__module__, klass.__qualname__))
                }
            )

            return classes

        return [
            (val.__name__, val)
            for key, val in recurse_module(self.module).items()
        ]

    def get_models(self):
        """Picks up every models classes from self.module.
        Filters out classes not inheriting from cuml.Base.

        Returns
        -------
        models: dictionary of {class_name: class|class_constructor}
            Dictionary of models in the module, except when a
            custom_constructor is specified, in that case the value is the
            specified custom_constructor.
        """
        classes = self._get_classes()
        models = {
            name: cls
            for name, cls in classes
            if cls not in self.exclude_classes and issubclass(cls, Base)
        }
        models.update(self.custom_constructors)
        return models


def get_all_base_subclasses() -> dict[str, type]:
    """
    Returns all currently defined subclasses of `cuml.Base`,
    excluding any with a leading `_` in their name.
    """
    out = {}

    def _collect(cls):
        if not cls.__name__.startswith("_"):
            out[cls.__name__] = cls
        for c in cls.__subclasses__():
            _collect(c)

    _collect(Base)

    return out


def generate_random_labels(random_generation_lambda, seed=1234, as_cupy=False):
    """
    Generates random labels to act as ground_truth and predictions for tests.

    Parameters
    ----------
    random_generation_lambda : lambda function [numpy.random] -> ndarray
        A lambda function used to generate labels for either y_true or y_pred
        using a seeded numpy.random object.
    seed : int
        Seed for the numpy.random object.
    as_cupy : bool
        Choose return type of y_true and y_pred.
        True: returns Cupy ndarray
        False: returns Numba cuda DeviceNDArray

    Returns
    -------
    y_true, y_pred, np_y_true, np_y_pred : tuple
        y_true : Numba cuda DeviceNDArray or Cupy ndarray
            Random target values.
        y_pred : Numba cuda DeviceNDArray or Cupy ndarray
            Random predictions.
        np_y_true : Numpy ndarray
            Same as y_true but as a numpy ndarray.
        np_y_pred : Numpy ndarray
            Same as y_pred but as a numpy ndarray.
    """
    rng = np.random.RandomState(seed)  # makes it reproducible
    a = random_generation_lambda(rng)
    b = random_generation_lambda(rng)

    if as_cupy:
        return cp.array(a), cp.array(b), a, b
    else:
        return cuda.to_device(a), cuda.to_device(b), a, b


def get_shap_values(
    model,
    explainer,
    background_dataset,
    explained_dataset,
    api_type="shap_values",
):
    # function to get shap values from an explainer using SHAP style API.
    # This function allows isolating all calls in test suite for the case of
    # API changes.
    explainer = explainer(model=model, data=background_dataset)
    if api_type == "shap_values":
        shap_values = explainer.shap_values(explained_dataset)
    elif api_type == "__call__":
        shap_values = explainer(explained_dataset)

    return explainer, shap_values


def compare_svm(
    svm1,
    svm2,
    X,
    y,
    b_tol=None,
    coef_tol=None,
    report_summary=False,
    cmp_decision_func=False,
):
    """Compares two svm classifiers
    Parameters:
    -----------
    svm1 : svm classifier to be tested
    svm2 : svm classifier, the correct model
    b_tol : float
        tolerance while comparing the constant in the decision functions
    coef_tol: float
        tolerance used while comparing coef_ attribute for linear SVM

    Support vector machines have a decision function:

    F(x) = sum_{i=1}^{n_sv} d_i K(x_i, x) + b,

    where n_sv is the number of support vectors, K is the kernel function, x_i
    are the support vectors, d_i are the dual coefficients (more precisely
    d = alpha_i * y_i, where alpha_i is the dual coef), and b is the intercept.

    For linear svms K(x_i, x) = x_i * x, and we can simplify F by introducing
    w = sum_{i=1}^{n_sv} d_i x_i, the normal vector of the separating
    hyperplane:

    F(x) = w * x + b.

    Mathematically the solution of the optimization should be unique, which
    means w and b should be unique.

    There could be multiple set of vectors that lead to the same w, therefore
    comparing parameters d_k, n_sv or the support vector indices can lead to
    false positives.

    We can only evaluate w for linear models, for nonlinear models we can only
    test model accuracy and intercept.
    """

    n = X.shape[0]
    accuracy1 = svm1.score(X, y)
    accuracy2 = svm2.score(X, y)

    # We use at least 0.1% tolerance for accuracy comparison
    accuracy_tol_min = 0.001
    if accuracy2 < 1:
        # Set tolerance to include the 95% confidence interval of svm2's
        # accuracy. In practice this gives 0.9% tolerance for a 90% accurate
        # model (assuming n_test = 4000).
        accuracy_tol = 1.96 * np.sqrt(accuracy2 * (1 - accuracy2) / n)
        if accuracy_tol < accuracy_tol_min:
            accuracy_tol = accuracy_tol_min
    else:
        accuracy_tol = accuracy_tol_min

    assert accuracy1 >= accuracy2 - accuracy_tol

    if b_tol is None:
        b_tol = 100 * svm1.tol  # Using default tol=1e-3 leads to b_tol=0.1

    if accuracy2 < 0.5:
        # Increase error margin for classifiers that are not accurate.
        # Although analytically the classifier should always be the same,
        # we fit only until we reach a certain numerical tolerance, and
        # therefore the resulting SVM's can be different. We increase the
        # tolerance in these cases.
        #
        # A good example is the gaussian dataset with linear classifier:
        # the classes are concentric blobs, and we cannot separate that with a
        # straight line. When we have a large number of data points, then
        # any separating hyperplane that goes through the center would be good.
        b_tol *= 10
        if n >= 250:
            coef_tol = 2  # allow any direction
        else:
            coef_tol *= 10

    # Compare model parameter b (intercept). In practice some models can have
    # some differences in the model parameters while still being within
    # the accuracy tolerance.
    #
    # We skip this test for multiclass (when intercept_ is an array). Apart
    # from the larger discrepancies in multiclass case, sklearn also uses a
    # different sign convention for intercept in that case.
    if cp.isscalar(svm2.intercept_) or svm2.intercept_.shape[0] == 1:
        if abs(svm2.intercept_) > 1e-6:
            assert (
                abs((svm1.intercept_ - svm2.intercept_) / svm2.intercept_)
                <= b_tol
            )
        else:
            assert abs((svm1.intercept_ - svm2.intercept_)) <= b_tol

    # For linear kernels we can compare the normal vector of the separating
    # hyperplane w, which is stored in the coef_ attribute.
    if svm1.kernel == "linear":
        if coef_tol is None:
            coef_tol = 1e-5
        cs = np.dot(svm1.coef_, svm2.coef_.T) / (
            np.linalg.norm(svm1.coef_) * np.linalg.norm(svm2.coef_)
        )
        assert cs > 1 - coef_tol

    if cmp_decision_func:
        if accuracy2 > 0.9 and svm1.kernel != "sigmoid":
            df1 = svm1.decision_function(X)
            df2 = svm2.decision_function(X)
            # For classification, the class is determined by
            # sign(decision function). We should not expect tight match for
            # the actual value of the function, therefore we set large tolerance
            assert svm_array_equal(
                df1, df2, tol=1e-1, relative_diff=True, report_summary=True
            )
        else:
            print(
                "Skipping decision function test due to low  accuracy",
                accuracy2,
            )

    # Compare `class_weight_` attribute for classifiers, if present
    if hasattr(svm1, "class_weight_"):
        np.testing.assert_allclose(svm1.class_weight_, svm2.class_weight_)

    # Compare support_ (dataset indices of points that form the support
    # vectors) and ensure that some overlap (~1/8) between two exists
    support1 = set(svm1.support_)
    support2 = set(svm2.support_)
    intersection_len = len(support1.intersection(support2))
    average_len = (len(support1) + len(support2)) / 2
    assert intersection_len > average_len / 8


def svm_array_equal(a, b, tol=1e-6, relative_diff=True, report_summary=False):
    diff = np.abs(a - b)
    if relative_diff:
        idx = np.nonzero(abs(b) > tol)
        diff[idx] = diff[idx] / abs(b[idx])
    equal = np.all(diff <= tol)
    if not equal and report_summary:
        idx = np.argsort(diff)
        print("Largest diffs")
        a = a.ravel()
        b = b.ravel()
        diff = diff.ravel()
        for i in idx[-5:]:
            if diff[i] > tol:
                print(diff[i], "at", i, "values", a[i], b[i])
        print(
            "Avgdiff:",
            np.mean(diff),
            "stddiyy:",
            np.std(diff),
            "avgval:",
            np.mean(b),
        )
    return equal
