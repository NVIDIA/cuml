#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Support invoking test_python_dask.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../ || exit

# Common setup steps shared by Python test jobs
export DEPENDENCY_FILE_KEY=test_python_dask
source ./ci/test_python_common.sh

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Keep both transport reports; the UCXX run must not overwrite the TCP run.
rapids-logger "pytest cuml-dask (No UCXX)"
timeout -v --signal=SIGINT --kill-after=60s 1h ./ci/run_cuml_dask_pytests.sh \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml-dask.xml"

rapids-logger "pytest cuml-dask (UCXX only)"
timeout -v --signal=SIGINT --kill-after=60s 10m ./ci/run_cuml_dask_pytests.sh \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml-dask-ucxx.xml" --run_ucx

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
