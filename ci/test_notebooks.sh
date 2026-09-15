#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Downloading artifacts from previous jobs"
source ./ci/use_conda_packages_from_prs.sh
CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcuml cuml --cuda "$RAPIDS_CUDA_VERSION")")
PYTHON_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_python cuml cuml --stable --cuda "$RAPIDS_CUDA_VERSION")")

rapids-logger "Generate Notebook testing dependencies"
source ./ci/use_conda_packages_from_prs.sh
rapids-dependency-file-generator \
  --output conda \
  --file-key test_notebooks \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
  --prepend-channel "${LIBNVFOREST_CHANNEL}" \
  --prepend-channel "${NVFOREST_CHANNEL}" \
  --prepend-channel "${CPP_CHANNEL}" \
  --prepend-channel "${PYTHON_CHANNEL}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi

NOTEBOOKS_EXITCODE=0
trap "NOTEBOOKS_EXITCODE=1" ERR
set +e

rapids-logger "notebook tests cuml"

# Add notebooks that should be skipped here
# (space-separated list of filenames without paths)
SKIPNBS="cuml_benchmarks.ipynb hdbscan_soft_clustering_benchmark.ipynb"
NBTEST="$(realpath "$(dirname "$0")/utils/nbtest.sh")"
REPO_ROOT="$(realpath "$(dirname "$0")/..")"
CI_NOTEBOOKS_DIR="${REPO_ROOT}/ci/notebooks"
USER_NOTEBOOKS_DIR="${REPO_ROOT}/notebooks"

# Function to run notebooks in a directory
run_notebooks() {
    local nb_dir=$1
    local nb_dir_name=$2
    if [ ! -d "${nb_dir}" ]; then
        echo "--------------------------------------------------------------------------------"
        echo "SKIPPING: ${nb_dir_name} (directory does not exist)"
        echo "--------------------------------------------------------------------------------"
        return
    fi
    cd "${nb_dir}"
    # shellcheck disable=SC2044
    for nb in $(find . -name "*.ipynb"); do
        nbBasename=$(basename "${nb}")
        # Skip all NBs that use dask (in the code or even in their name)
        if (echo "${nb}" | grep -qi dask) || \
            ( grep -q dask "${nb}" && [ "${nbBasename}" != 'forest_inference_demo.ipynb' ] ); then
            echo "--------------------------------------------------------------------------------"
            echo "SKIPPING: ${nb} (suspected Dask usage, not currently automatable)"
            echo "--------------------------------------------------------------------------------"
        elif (echo " ${SKIPNBS} " | grep -q " ${nbBasename} "); then
            echo "--------------------------------------------------------------------------------"
            echo "SKIPPING: ${nb} (listed in skip list)"
            echo "--------------------------------------------------------------------------------"
        else
            nvidia-smi
            ${NBTEST} "${nb}"
        fi
    done
}

rapids-logger "Running user-facing notebooks"
run_notebooks "${USER_NOTEBOOKS_DIR}" "notebooks"

rapids-logger "Running CI-only notebooks"
run_notebooks "${CI_NOTEBOOKS_DIR}" "ci/notebooks"

rapids-logger "Test script exiting with value: $NOTEBOOKS_EXITCODE"
exit ${NOTEBOOKS_EXITCODE}
