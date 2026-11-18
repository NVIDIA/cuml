#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
REPO=$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")
exec python -m pytest -c "${REPO}/pyproject.toml" --cuml-suite=standard \
    --cache-clear --durations=100 --durations-min=1 "$@"
