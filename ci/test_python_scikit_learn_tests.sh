#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Support invoking test script outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../ || exit 1

export DEPENDENCY_FILE_KEY=test_python_accel_sklearn
source ./ci/test_python_common.sh

rapids-logger "Running scikit-learn tests with cuML acceleration"

# Do not immediately exit on error
set +e

timeout -v --signal=SIGINT --kill-after=60s 1h ./python/cuml/cuml_accel_tests/upstream/scikit-learn/run-tests.sh \
    --numprocesses=8 \
    --dist=worksteal \
    --durations=100 --durations-min=1 \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml-accel-scikit-learn.xml"
TEST_EXITCODE=$?

rapids-logger "Analyzing test results"
./python/cuml/cuml_accel_tests/upstream/summarize-results.py \
    --config ./python/cuml/cuml_accel_tests/upstream/scikit-learn/test_config.yaml \
    "${RAPIDS_TESTS_DIR}/junit-cuml-accel-scikit-learn.xml"

EXITCODE=$TEST_EXITCODE

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
