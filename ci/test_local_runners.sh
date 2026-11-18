#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# CPU-only tests of local runner behavior; no cuML build or GPU is needed.
set -euo pipefail
REPO=$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")
cd "${REPO}"

source /opt/conda/etc/profile.d/conda.sh
rapids-logger "Create minimal runner-test environment"
rapids-mamba-retry create --yes --name local-runner-tests --override-channels \
    --channel conda-forge python=3.12 pytest packaging pyyaml cmake git
conda activate local-runner-tests

# The shared CPU workflow uploads this directory even when the script fails.
REPORT_DIR="${RAPIDS_ARTIFACTS_DIR:-${REPO}/artifacts}/local-runner-tests"
mkdir -p "${REPORT_DIR}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest ci/tests --junitxml="${REPORT_DIR}/junit.xml"
