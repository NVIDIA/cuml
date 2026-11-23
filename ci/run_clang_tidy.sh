#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

rapids-logger "Create clang_tidy conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

source ./ci/use_conda_packages_from_prs.sh

rapids-dependency-file-generator \
  --output conda \
  --file-key clang_tidy \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
  --prepend-channel "${LIBNVFOREST_CHANNEL}" \
  --prepend-channel "${NVFOREST_CHANNEL}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n clang_tidy
# Temporarily allow unbound variables for conda activation.
set +u && conda activate clang_tidy && set -u

./build.sh --configure-only libcuml

rapids-logger "Run clang-tidy"

python cpp/scripts/run-clang-tidy.py --config pyproject.toml
