#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Support invoking test_python_singlegpu.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../ || exit 1

# Common setup steps shared by Python test jobs
source ./ci/test_python_common.sh

# We want to error if dask is installed in this environment so
if python -c 'import dask' 2>/dev/null; then
  echo "ERROR: dask is installed in this environment! This means we are not \
  testing that cuml works without dask. Please adjust the environment so the \
  main test environment doesn't install dask."
  exit 1
fi

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest cuml single GPU"
timeout -v --signal=SIGINT --kill-after=60s 1h python -m pytest --cuml-suite=standard \
  --cache-clear --durations=100 --durations-min=1 \
  --numprocesses=8 \
  --dist=worksteal \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml.xml"

  rapids-logger "pytest cuml accelerator"
timeout -v --signal=SIGINT --kill-after=60s 15m python -m pytest --cuml-suite=accel \
  --cache-clear --durations=100 --durations-min=1 \
  --numprocesses=8 \
  --dist=worksteal \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml-accel.xml"

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
