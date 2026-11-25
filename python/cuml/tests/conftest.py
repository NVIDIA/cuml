#
# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import os
from datetime import timedelta

import cudf.pandas
import cupy as cp
import hypothesis
import numpy as np
import pytest
from sklearn import datasets

from cuml.testing.datasets import make_text_classification_dataset

# =============================================================================
# Pytest Configuration
# =============================================================================

# Add the import here for any plugins that should be loaded EVERY TIME
pytest_plugins = [
    "cuml.testing.plugins.categories",
    "cuml.testing.plugins.gpu_test_config",
    "cuml.testing.plugins.quick_run_plugin",
    "cuml.testing.plugins.restart_corrupted_xdist_worker",
]


# =============================================================================
# Test Configuration Constants
# =============================================================================

# Many CI systems set the "magic" CI variable to let code know it is running
# in a CI environment.
CI = os.environ.get("CI") in ("true", "1")
HYPOTHESIS_ENABLED = os.environ.get("HYPOTHESIS_ENABLED") in (
    "true",
    "1",
)


# =============================================================================
# Hypothesis Configuration
# =============================================================================

HEALTH_CHECKS_SUPPRESSED_BY_DEFAULT = (
    list(hypothesis.HealthCheck)
    if CI
    else [
        hypothesis.HealthCheck.data_too_large,
        hypothesis.HealthCheck.too_slow,
    ]
)

HYPOTHESIS_DEFAULT_PHASES = (
    (
        hypothesis.Phase.explicit,
        hypothesis.Phase.reuse,
        hypothesis.Phase.generate,
        hypothesis.Phase.target,
        hypothesis.Phase.shrink,
    )
    if HYPOTHESIS_ENABLED
    else (hypothesis.Phase.explicit,)
)

hypothesis.settings.register_profile(
    name="unit",
    # CI runners have varying performance, so we do not set a deadline for CI
    # jobs to avoid flaky tests.
    deadline=None if CI else timedelta(milliseconds=5000),
    parent=hypothesis.settings.get_profile("default"),
    phases=HYPOTHESIS_DEFAULT_PHASES,
    max_examples=20,
    report_multiple_bugs=False,
    suppress_health_check=HEALTH_CHECKS_SUPPRESSED_BY_DEFAULT,
)

hypothesis.settings.register_profile(
    name="quality",
    parent=hypothesis.settings.get_profile("unit"),
    max_examples=100,
)

hypothesis.settings.register_profile(
    name="stress",
    parent=hypothesis.settings.get_profile("quality"),
    max_examples=200,
)

# =============================================================================
# Pytest Hooks and Configuration
# =============================================================================


def pytest_collection_modifyitems(config, items):
    """Check for hypothesis tests without explicit examples."""
    # Check for hypothesis tests without examples
    tests_without_examples = []
    for item in items:
        if isinstance(item, pytest.Function):
            # Check if function has @given decorator
            has_given = hasattr(item.obj, "hypothesis")
            # Check if function has @example decorator
            has_example = hasattr(item.obj, "hypothesis_explicit_examples")

            if has_given and not has_example:
                tests_without_examples.append(
                    f"Test {item.name} uses @given but has no @example cases."
                )

    if tests_without_examples:
        msg = (
            "\nCollection failed because the following tests lack examples:\n"
            + "\n".join(f"  - {e}" for e in tests_without_examples)
        )
        raise pytest.UsageError(msg)


def pytest_configure(config):
    """Configure pytest settings and load hypothesis profiles.

    This function:
    1. Adds custom markers
    2. Loads appropriate hypothesis profiles based on test execution context
    """
    config.addinivalue_line(
        "markers",
        "cudf_pandas: mark test as requiring the cudf.pandas wrapper",
    )
    # Load special hypothesis profiles for either quality or stress tests.
    # Note that the profile can be manually overwritten with the
    # --hypothesis-profile command line option in which case the settings
    # specified here will be ignored.
    if config.getoption("--run_stress"):
        hypothesis.settings.load_profile("stress")
    elif config.getoption("--run_quality"):
        hypothesis.settings.load_profile("quality")
    else:
        hypothesis.settings.load_profile("unit")


def pytest_pyfunc_call(pyfuncitem):
    """Skip tests that require the cudf.pandas accelerator.

    Tests marked with `@pytest.mark.cudf_pandas` will only be run if the
    cudf.pandas accelerator is enabled via the `cudf.pandas` plugin.
    """
    if "cudf_pandas" in pyfuncitem.keywords and not cudf.pandas.LOADED:
        pytest.skip("Test requires cudf.pandas accelerator")


# =============================================================================
# Test Report and Logging Fixtures
# =============================================================================


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Create a test report and attach it to the test item.

    This function is used to create detailed test reports and attach them
    to the test items for later use in fixtures like failure_logger.
    """
    outcome = yield
    rep = outcome.get_result()
    setattr(item, "rep_" + rep.when, rep)


@pytest.fixture(scope="function")
def failure_logger(request):
    """Log the random seed used in failing tests.

    This fixture is used to log the random seed when a test fails,
    which helps in reproducing test failures.
    """
    yield
    if request.node.rep_call.failed:
        error_msg = " {} failed with seed: {}"
        error_msg = error_msg.format(
            request.node.nodeid, os.getenv("PYTEST_RANDOM_SEED")
        )
        print(error_msg)


# =============================================================================
# Random Seed Fixture
# =============================================================================


@pytest.fixture(scope="session")
def random_seed(request):
    """Generate or retrieve a random seed for test reproducibility.

    This fixture improves reproducibility of tests using random numbers
    by using the same seed throughout the test session.
    """
    current_random_seed = os.getenv("PYTEST_RANDOM_SEED")
    if current_random_seed is not None and current_random_seed.isdigit():
        random_seed = int(current_random_seed)
    else:
        random_seed = np.random.randint(0, 1e6)
        os.environ["PYTEST_RANDOM_SEED"] = str(random_seed)
    print("\nRandom seed value:", random_seed)
    return random_seed


# =============================================================================
# Dataset Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def sparse_text_dataset():
    """Return a sparse text-like dataset and a CuPy target vector."""
    X, y = make_text_classification_dataset()
    return X, cp.array(y)


@pytest.fixture(
    scope="session",
    params=["digits", "diabetes", "cancer"],
)
def supervised_learning_dataset(request):
    """Provide various supervised learning datasets for testing.

    This fixture provides access to multiple standard supervised learning
    datasets. It is parameterized to allow testing with different datasets.
    """
    datasets_dict = {
        "digits": datasets.load_digits(),
        "diabetes": datasets.load_diabetes(),
        "cancer": datasets.load_breast_cancer(),
    }

    return datasets_dict[request.param].data
