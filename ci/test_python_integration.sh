#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Support invoking test_python_singlegpu.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../ || exit

# Common setup steps shared by Python test jobs
source ./ci/test_python_common.sh

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest cuml integration"
timeout -v --signal=SIGINT --kill-after=60s 1h python -m pytest --cuml-suite=cudf-pandas \
  --cache-clear --durations=100 --durations-min=1 --quick_run \
  --numprocesses=8 \
  --dist=worksteal \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml.xml"

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
