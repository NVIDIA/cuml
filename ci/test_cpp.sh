#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking test_cpp.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcuml cuml --cuda "$RAPIDS_CUDA_VERSION")")

rapids-logger "Generate C++ testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_cpp \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" \
  --prepend-channel "${CPP_CHANNEL}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-print-env

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Run gtests"
export GTEST_OUTPUT=xml:${RAPIDS_TESTS_DIR}/
# Run libcuml gtests from libcuml-tests package
timeout -v 20m ./ci/run_ctests.sh -j9 && EXITCODE=$? || EXITCODE=$?;

rapids-logger "Test script exiting with value: $EXITCODE"
exit "${EXITCODE}"
