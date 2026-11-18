#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# CPU-only tests of pytest entrypoint behavior; no cuML build or GPU is needed.
set -euo pipefail
REPO=$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")
cd "${REPO}"

source /opt/conda/etc/profile.d/conda.sh
rapids-logger "Create minimal pytest-entrypoint-test environment"
rapids-mamba-retry create --yes --name pytest-entrypoint-tests --override-channels \
    --channel conda-forge python=3.12 pytest packaging pyyaml cmake git
conda activate pytest-entrypoint-tests

# The shared CPU workflow uploads this directory even when the script fails.
REPORT_DIR="${RAPIDS_ARTIFACTS_DIR:-${REPO}/artifacts}/pytest-entrypoint-tests"
mkdir -p "${REPORT_DIR}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest ci/tests --junitxml="${REPORT_DIR}/junit.xml"
