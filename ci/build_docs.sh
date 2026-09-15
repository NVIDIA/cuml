#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcuml cuml --cuda "$RAPIDS_CUDA_VERSION")")
PYTHON_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_python cuml cuml --stable --cuda "$RAPIDS_CUDA_VERSION")")


rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

RAPIDS_VERSION_MAJOR_MINOR="$(rapids-version-major-minor)"
export RAPIDS_VERSION_MAJOR_MINOR

rapids-dependency-file-generator \
  --output conda \
  --file-key docs \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
  --prepend-channel "${CPP_CHANNEL}" \
  --prepend-channel "${PYTHON_CHANNEL}" \
  | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n docs
conda activate docs

rapids-print-env

RAPIDS_DOCS_DIR="$(mktemp -d)"
export RAPIDS_DOCS_DIR

rapids-logger "Generate C++ API XML for Breathe"
pushd cpp
doxygen Doxyfile.in
popd

rapids-logger "Build the combined Python and C++ Sphinx documentation"
pushd docs
sphinx-build -b dirhtml ./source _html -W
mkdir -p "${RAPIDS_DOCS_DIR}/cuml/html"
mv _html/* "${RAPIDS_DOCS_DIR}/cuml/html"
popd

RAPIDS_VERSION_NUMBER="${RAPIDS_VERSION_MAJOR_MINOR}" rapids-upload-docs
