#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Temporary overrides to test https://github.com/rapidsai/nvforest/pull/294.
LIBNVFOREST_CHANNEL=$(rapids-get-pr-artifact nvforest 294 cpp conda)
NVFOREST_CHANNEL=$(rapids-get-pr-artifact nvforest 294 python conda)

# Prefer PR artifacts over nightly packages in rattler builds.
RAPIDS_PREPENDED_CONDA_CHANNELS=(
    "${LIBNVFOREST_CHANNEL}"
    "${NVFOREST_CHANNEL}"
)
export RAPIDS_PREPENDED_CONDA_CHANNELS

# Also make the channels available to conda/mamba installs.
for _channel in "${RAPIDS_PREPENDED_CONDA_CHANNELS[@]}"; do
    conda config --system --add channels "${_channel}"
done
