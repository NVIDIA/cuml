#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# download CI artifacts
LIBNVFOREST_CHANNEL=$(rapids-get-pr-artifact nvforest 219 cpp conda)
NVFOREST_CHANNEL=$(rapids-get-pr-artifact nvforest 219 python conda)

# For `rattler` builds:
#
# Add these channels to the array checked by 'rapids-rattler-channel-string'.
# This ensures that when conda packages are built with strict channel priority enabled,
# the locally-downloaded packages will be preferred to remote packages (e.g. nightlies).
#
RAPIDS_PREPENDED_CONDA_CHANNELS=(
    "${LIBNVFOREST_CHANNEL}"
    "${NVFOREST_CHANNEL}"
)
export RAPIDS_PREPENDED_CONDA_CHANNELS

# For tests and `conda-build` builds:
#
# Add these channels to the system-wide conda configuration.
# This results in PREPENDING them to conda's channel list, so
# these packages should be found first if strict channel priority is enabled.
#
for _channel in "${RAPIDS_PREPENDED_CONDA_CHANNELS[@]}"
do
   conda config --system --add channels "${_channel}"
done
