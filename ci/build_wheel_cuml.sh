#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

package_name="cuml"
package_dir="python/cuml"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Download the libcuml wheel built in the previous step and make it
# available for pip to find.
#
# env variable 'PIP_CONSTRAINT' is set up by rapids-init-pip. It constrains all subsequent
# 'pip install', 'pip download', etc. calls (except the isolated wheel build,
# which is handled separately by ci/build_wheel.sh)
LIBCUML_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcuml cuml --cuda "$RAPIDS_CUDA_VERSION")")
echo "libcuml-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBCUML_WHEELHOUSE}"/libcuml_*.whl)" >> "${PIP_CONSTRAINT}"

EXCLUDE_ARGS=(
  --exclude "libcublas.so.*"
  --exclude "libcublasLt.so.*"
  --exclude "libcufft.so.*"
  --exclude "libcuml.so"
  --exclude "libcurand.so.*"
  --exclude "libcusolver.so.*"
  --exclude "libcusparse.so.*"
  --exclude "libcuvs.so"
  --exclude "libnvforest.so"
  --exclude "libnvJitLink.so.*"
  --exclude "libraft.so"
  --exclude "librapids_logger.so"
  --exclude "librmm.so"
)

# TODO: move this variable into `ci-wheel`
# Format Python limited API version string
RAPIDS_PY_API="cp${RAPIDS_PY_VERSION//./}"
export RAPIDS_PY_API

export SKBUILD_CMAKE_ARGS="-DDISABLE_DEPRECATION_WARNINGS=ON;-DSINGLEGPU=OFF;-DUSE_LIBCUML_WHEEL=ON"
./ci/build_wheel.sh "${package_name}" "${package_dir}" --stable

# repair wheels and write to the location that artifact-uploading code expects to find them
python -m auditwheel repair \
    "${EXCLUDE_ARGS[@]}" \
    -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
    ${package_dir}/dist/*

./ci/validate_wheel.sh ${package_dir} "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

RAPIDS_PACKAGE_NAME="$(rapids-artifact-name wheel_python cuml cuml --stable --cuda "$RAPIDS_CUDA_VERSION")"
export RAPIDS_PACKAGE_NAME
