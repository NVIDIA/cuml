#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Implementation of cuML health checks for rapids doctor and standalone use."""


def import_check(verbose=False, **kwargs):
    """Check that cuML can be imported.

    Mainly useful when invoked programmatically; when run via rapids doctor,
    cuml is typically already loaded. On failure, use the RAPIDS install docs.
    """
    try:
        import cuml
    except ImportError as e:
        raise ImportError(
            "cuML could not be imported. Install cuML with conda or pip as "
            "described at https://docs.nvidia.com/datascience/install/"
        ) from e
    if verbose:
        return f"cuML {cuml.__version__} is available"


def functional_check(verbose=False, **kwargs):
    """Check that a basic cuML estimator can fit and predict."""
    import numpy as np

    from cuml.linear_model import LinearRegression

    X = np.array([[1], [2], [3], [4]], dtype=np.float32)
    y = np.array([1, 2, 3, 4], dtype=np.float32)
    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict(X)
    if pred.shape != (4,):
        raise AssertionError(
            f"Expected predictions of shape (4,), got {pred.shape}"
        )
    pred = np.asarray(pred, dtype=np.float32)
    if not np.allclose(pred, y, atol=0.1):
        raise AssertionError(
            f"LinearRegression predictions differ from expected: "
            f"got {pred.tolist()}, expected {y.tolist()}"
        )
    if verbose:
        return "LinearRegression fit/predict succeeded"


_SUBPROCESS_TIMEOUT = 120


def accel_basic_check(verbose=False, **kwargs):
    """Check that cuml.accel can be installed and intercepts sklearn."""
    import subprocess
    import sys

    script = (
        "import cuml.accel; cuml.accel.install(); "
        "from sklearn.ensemble import RandomForestClassifier; "
        "assert cuml.accel.is_proxy(RandomForestClassifier), "
        "'RandomForestClassifier is not a cuml.accel proxy'; "
        "from sklearn.datasets import make_classification; "
        "X, y = make_classification(n_samples=100, random_state=0); "
        "RandomForestClassifier(n_estimators=10).fit(X, y)"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"cuml.accel subprocess check timed out after "
            f"{_SUBPROCESS_TIMEOUT}s"
        )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        detail = (
            "\n".join(stderr.splitlines()[-5:]) if stderr else "unknown error"
        )
        raise RuntimeError(f"cuml.accel subprocess check failed:\n{detail}")
    if verbose:
        return (
            "cuml.accel intercepted sklearn and fit a RandomForestClassifier"
        )


def accel_cli_check(verbose=False, **kwargs):
    """Check that python -m cuml.accel runs sklearn code on the GPU."""
    import os
    import subprocess
    import sys
    import tempfile

    script_content = (
        "from sklearn.datasets import make_classification\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "X, y = make_classification(n_samples=200, random_state=0)\n"
        "clf = RandomForestClassifier(n_estimators=10, random_state=0)\n"
        "clf.fit(X, y)\n"
        "clf.predict(X)\n"
    )
    fd, script_path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(script_content)

        try:
            result = subprocess.run(
                [sys.executable, "-m", "cuml.accel", "--verbose", script_path],
                capture_output=True,
                text=True,
                timeout=_SUBPROCESS_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"python -m cuml.accel --verbose timed out after "
                f"{_SUBPROCESS_TIMEOUT}s"
            )
    finally:
        os.unlink(script_path)

    if result.returncode != 0:
        stderr = result.stderr.strip()
        detail = (
            "\n".join(stderr.splitlines()[-5:]) if stderr else "unknown error"
        )
        raise RuntimeError(f"python -m cuml.accel --verbose failed:\n{detail}")

    output = result.stdout
    if "ran on GPU" not in output:
        raise AssertionError(
            "cuml.accel --verbose output missing 'ran on GPU':\n" + output
        )
    if "falling back to CPU" in output or "ran on CPU" in output:
        raise AssertionError(
            "cuml.accel --verbose reported CPU fallbacks:\n" + output
        )
    if verbose:
        return (
            "python -m cuml.accel --verbose ran sklearn code on GPU "
            "with no fallbacks"
        )
