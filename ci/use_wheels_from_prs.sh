#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Temporary overrides to test https://github.com/rapidsai/nvforest/pull/294.
source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX=$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")
LIBNVFOREST_WHEELHOUSE=$(rapids-get-pr-artifact nvforest 294 cpp wheel)
NVFOREST_WHEELHOUSE=$(rapids-get-pr-artifact nvforest 294 python wheel)

# Preserve existing constraints and force both build and runtime dependencies
# to use the PR wheels.
cat >> "${PIP_CONSTRAINT}" <<EOF
libnvforest-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBNVFOREST_WHEELHOUSE}"/libnvforest_*.whl)
nvforest-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${NVFOREST_WHEELHOUSE}"/nvforest_*.whl)
EOF
