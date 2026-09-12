#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# initialize PIP_CONSTRAINT
source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX=$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")

# download wheels, store the directories holding them in variables
LIBNVFOREST_WHEELHOUSE=$(rapids-get-pr-artifact nvforest 219 cpp wheel)
NVFOREST_WHEELHOUSE=$(rapids-get-pr-artifact nvforest 219 python wheel)

# write a pip constraints file saying e.g. "whenever you encounter a requirement for 'librmm-cu12', use this wheel"
cat > "${PIP_CONSTRAINT}" <<EOF
libnvforest-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBNVFOREST_WHEELHOUSE}"/libnvforest_*.whl)
nvforest-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${NVFOREST_WHEELHOUSE}"/nvforest_*.whl)
EOF
