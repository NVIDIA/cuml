#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Support invoking run_cuml_singlegpu_accel_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cuml/cuml_accel_tests || exit 1

python -m pytest --cache-clear --durations=100 --durations-min=1 "$@" .
