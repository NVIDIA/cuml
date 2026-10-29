#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

python -m venv libcuml-env
. libcuml-env/bin/activate

rapids-pip-retry install \
    -v \
    --prefer-binary \
    --constraint "${PIP_CONSTRAINT}" \
    "${LIBCUML_WHEELHOUSE}"/libcuml*.whl
python -c "import libcuml; assert (libraries := libcuml.load_library()) and all(libraries)"
deactivate

# notes:
#
#   * just providing --constraint="${PIP_CONSTRAINT}" to be explicit, and because
#     that environment variable is ignored if any other --constraint are passed via the CLI
#
rapids-pip-retry install \
  --prefer-binary \
  --constraint "${PIP_CONSTRAINT}" \
  "${LIBCUML_WHEELHOUSE}"/libcuml*.whl \
  "${CUML_WHEELHOUSE}"/cuml*.whl


# Try to import cuml with just a minimal install"
rapids-logger "Importing cuml with minimal dependencies"
python -c "import cuml"

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
  "$(echo "${CUML_WHEELHOUSE}"/cuml*.whl)[test]"

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Run linkage test for libcuml
rapids-logger "Testing libcuml linkage"
python -m pytest --cache-clear python/libcuml/tests/test_libcuml_linkage.py -v

rapids-logger "pytest cuml single GPU"
timeout -v --signal=SIGINT --kill-after=60s 1h ./ci/run_cuml_singlegpu_pytests.sh \
  --numprocesses=8 \
  --dist=worksteal \
  -k 'not test_sparse_pca_inputs' \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml.xml"

if [[ "${RAPIDS_DEPENDENCIES:-}" == "nightly" ]]; then
  rapids-logger "pytest cuml accelerator"
  timeout -v --signal=SIGINT --kill-after=60s 15m ./ci/run_cuml_singlegpu_accel_pytests.sh \
    --numprocesses=8 \
    --dist=worksteal \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml-accel.xml"
fi

# Run test_sparse_pca_inputs separately
timeout -v --signal=SIGINT --kill-after=60s 10m ./ci/run_cuml_singlegpu_pytests.sh \
  -k 'test_sparse_pca_inputs' \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml-sparse-pca.xml"

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
