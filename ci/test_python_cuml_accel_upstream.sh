#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Support invoking test script outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../ || exit 1

# Common setup steps shared by Python test jobs
source ./ci/test_python_common.sh

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Run UMAP tests
rapids-logger "UMAP test suite"
timeout -v --signal=SIGINT --kill-after=60s 15m ./python/cuml/cuml_accel_tests/upstream/umap/run-tests.sh \
  --durations=100 --durations-min=1

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
