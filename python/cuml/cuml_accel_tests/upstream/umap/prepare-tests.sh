#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Explicit source provisioning, separate from local test execution.
set -euo pipefail
UMAP_VERSION=$(python -c "import umap; print(umap.__version__)")
UMAP_REPO="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/umap-upstream"
if [[ ! -d "${UMAP_REPO}" ]]; then
    git clone https://github.com/lmcinnes/umap.git "${UMAP_REPO}"
fi
git -C "${UMAP_REPO}" fetch --tags
git -C "${UMAP_REPO}" checkout "release-${UMAP_VERSION}"
