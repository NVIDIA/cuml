#
# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import os
import tempfile
from datetime import timedelta
from math import ceil

import cudf.pandas
import cupy as cp
import hypothesis
import numpy as np
import pynvml
import pytest
from sklearn import datasets

from cuml.testing.datasets import make_text_classification_dataset


def _set_cuda_cache_path_per_xdist_worker(config):
    """Use a dedicated CUDA JIT cache directory per pytest-xdist worker.

    Avoids concurrent access to the same cache from multiple worker processes.
    Must run before any CUDA runtime initialization.
    """
    workerinput = getattr(config, "workerinput", None)
    if workerinput is None:
        return
    workerid = workerinput["workerid"]
    safe = "".join(
        c if c.isalnum() or c in "-_" else "_" for c in str(workerid)
    )
    if existing := os.environ.get("CUDA_CACHE_PATH"):
        cache_dir = os.path.join(existing, safe)
    else:
        cache_dir = os.path.join(
            tempfile.gettempdir(),
            "cuml-pytest-cuda-cache",
            safe,
        )
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["CUDA_CACHE_PATH"] = cache_dir


# =============================================================================
# Pytest Configuration
# =============================================================================

# Add the import here for any plugins that should be loaded EVERY TIME
pytest_plugins = [
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


def pytest_addoption(parser):
    """Add custom command line options for pytest.

    This function adds three custom options to control test execution:
    - --run_stress: Run stress tests
    - --run_quality: Run quality tests
    - --run_unit: Run unit tests
    - --run_memleak: Run memleak tests
    """
    group = parser.getgroup("cuML Custom Options")

    group.addoption(
        "--run_stress",
        action="store_true",
        default=False,
        help=(
            "Runs tests marked with 'stress'. These are the most intense "
            "tests that take the longest to run and are designed to stress "
            "the hardware's compute resources."
        ),
    )

    group.addoption(
        "--run_quality",
        action="store_true",
        default=False,
        help=(
            "Runs tests marked with 'quality'. These tests are more "
            "computationally intense than 'unit', but less than 'stress'"
        ),
    )

    group.addoption(
        "--run_unit",
        action="store_true",
        default=False,
        help=(
            "Runs tests marked with 'unit'. These are the quickest tests "
            "that are focused on accuracy and correctness."
        ),
    )

    group.addoption(
        "--run_memleak",
        action="store_true",
        default=False,
        help="Runs tests marked with 'memleak'. These test for memory leaks.",
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options.

    This function:
    1. Checks for hypothesis tests without examples
    2. Selectively skip tests based on selected categories (unit/quality/stress)
    """
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

    # Handle test categories (unit/quality/stress)
    should_run_quality = config.getoption("--run_quality")
    should_run_stress = config.getoption("--run_stress")
    should_run_memleak = config.getoption("--run_memleak")

    # Run unit is implied if no --run_XXX is set
    should_run_unit = config.getoption("--run_unit") or not (
        should_run_quality or should_run_stress
    )

    # Mark the tests as "skip" if needed
    if not should_run_unit:
        skip_unit = pytest.mark.skip(
            reason="Unit tests run with --run_unit flag."
        )
        for item in items:
            if "unit" in item.keywords:
                item.add_marker(skip_unit)

    if not should_run_quality:
        skip_quality = pytest.mark.skip(
            reason="Quality tests run with --run_quality flag."
        )
        for item in items:
            if "quality" in item.keywords:
                item.add_marker(skip_quality)

    if not should_run_stress:
        skip_stress = pytest.mark.skip(
            reason="Stress tests run with --run_stress flag."
        )
        for item in items:
            if "stress" in item.keywords:
                item.add_marker(skip_stress)

    if not should_run_memleak:
        skip_memleak = pytest.mark.skip(
            reason="Memory leak tests run with --run_memleak flag."
        )
        for item in items:
            if "memleak" in item.keywords:
                item.add_marker(skip_memleak)


def pytest_configure(config):
    """Configure pytest settings and load hypothesis profiles.

    This function:
    1. Adds custom markers
    2. Records the available GPU memory
    3. Loads appropriate hypothesis profiles based on test execution context
    """
    _set_cuda_cache_path_per_xdist_worker(config)
    config.addinivalue_line(
        "markers",
        "cudf_pandas: mark test as requiring the cudf.pandas wrapper",
    )
    # max_gpu_memory: Capacity of the GPU memory in GB
    pytest.max_gpu_memory = _get_gpu_memory()
    pytest.adapt_stress_test = "CUML_ADAPT_STRESS_TESTS" in os.environ

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


def _get_pynvml_device_handle(device_id=0):
    """Get GPU handle from device index or UUID.

    Parameters
    ----------
    device_id: int or str
        The index or UUID of the device from which to obtain the handle.

    Raises
    ------
    ValueError
        If acquiring the device handle for the device specified failed.
    pynvml.NVMLError
        If any NVML error occurred while initializing.

    Returns
    -------
    A pynvml handle to the device.

    Examples
    --------
    >>> _get_pynvml_device_handle(device_id=0)

    >>> _get_pynvml_device_handle(device_id="GPU-9fb42d6f-7d6b-368f-f79c-3c3e784c93f6")
    """
    pynvml.nvmlInit()

    try:
        if device_id and not str(device_id).isnumeric():
            # This means device_id is UUID.
            # This works for both MIG and non-MIG device UUIDs.
            handle = pynvml.nvmlDeviceGetHandleByUUID(str.encode(device_id))
            if pynvml.nvmlDeviceIsMigDeviceHandle(handle):
                # Additionally get parent device handle
                # if the device itself is a MIG instance
                handle = pynvml.nvmlDeviceGetDeviceHandleFromMigDeviceHandle(
                    handle
                )
        else:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        return handle
    except pynvml.NVMLError:
        raise ValueError(f"Invalid device index or UUID: {device_id}")


def _get_gpu_memory(device_index=0):
    """Return total memory of CUDA device with index or with device identifier UUID.

    Parameters
    ----------
    device_index: int or str
        The index or UUID of the device from which to obtain the CPU affinity.

    Returns
    -------
    The total memory of the CUDA Device in GB, or ``None`` for devices that do not
    have a dedicated memory resource, as is usually the case for system on a chip (SoC)
    devices.
    """
    handle = _get_pynvml_device_handle(device_index)

    try:
        # Return total memory in GB
        return ceil(pynvml.nvmlDeviceGetMemoryInfo(handle).total / 2**30)
    except pynvml.NVMLError_NotSupported:
        return None


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
    """Generate a sparse text-like dataset similar to 20 newsgroups.

    Returns
    -------
    tuple
        (X, Y) where X is a sparse feature matrix and Y is a cupy target vector
    """
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
