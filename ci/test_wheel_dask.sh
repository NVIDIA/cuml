#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

LIBCUML_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcuml cuml --cuda "$RAPIDS_CUDA_VERSION")")
CUML_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python cuml cuml --stable --cuda "$RAPIDS_CUDA_VERSION")")
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
mkdir -p "${RAPIDS_TESTS_DIR}"

# generate constraints, the constraints will limit the version of the
# dependencies that can be installed later on when installing the wheel
rapids-generate-pip-constraints test_python "${PIP_CONSTRAINT}"

# notes:
#
#   * echo to expand wildcard before adding `[test]` requires for pip
#   * just providing --constraint="${PIP_CONSTRAINT}" to be explicit, and because
#     that environment variable is ignored if any other --constraint are passed via the CLI
#
rapids-pip-retry install \
  --prefer-binary \
  --constraint "${PIP_CONSTRAINT}" \
   "${LIBCUML_WHEELHOUSE}"/libcuml*.whl \
  "$(echo "${CUML_WHEELHOUSE}"/cuml*.whl)[dask,test,test-dask]"

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "TEMP: WSL diagnostics"
nproc; free -g; nvidia-smi topo -m || true; ls -la /dev/shm; df -h /dev/shm
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
python -c "
from raft_dask.common.nccl import nccl, unique_id
n = nccl()
uid = nccl.get_unique_id()
n.init(1, uid, 0)
print('single-rank NCCL init OK')
" || true
timeout -v --signal=SIGINT --kill-after=60s 10m ./ci/run_cuml_dask_pytests.sh -x -k 'test_get_combined_model' || true
unset NCCL_DEBUG NCCL_DEBUG_SUBSYS
rapids-logger "pytest cuml dask"
timeout -v --signal=SIGINT --kill-after=60s 1h ./ci/run_cuml_dask_pytests.sh --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml-dask.xml"

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
