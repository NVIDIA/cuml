#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
########################
# cuML Version Updater #
########################

## Usage
# NOTE: This script must be run from the repository root, not from the ci/release/ directory
# Primary interface:   bash ci/release/update-version.sh <new_version> [--run-context=main|release]
# Fallback interface:  [RAPIDS_RUN_CONTEXT=main|release] bash ci/release/update-version.sh <new_version>
# CLI arguments take precedence over environment variables
# Defaults to main when no run-context is specified


# Parse command line arguments
CLI_RUN_CONTEXT=""
VERSION_ARG=""

for arg in "$@"; do
    case $arg in
        --run-context=*)
            CLI_RUN_CONTEXT="${arg#*=}"
            shift
            ;;
        *)
            if [[ -z "$VERSION_ARG" ]]; then
                VERSION_ARG="$arg"
            fi
            ;;
    esac
done

# Format is YY.MM.PP - no leading 'v' or trailing 'a'
NEXT_FULL_TAG="$VERSION_ARG"

# Determine RUN_CONTEXT with CLI precedence over environment variable, defaulting to main
if [[ -n "$CLI_RUN_CONTEXT" ]]; then
    RUN_CONTEXT="$CLI_RUN_CONTEXT"
    echo "Using run-context from CLI: $RUN_CONTEXT"
elif [[ -n "${RAPIDS_RUN_CONTEXT}" ]]; then
    RUN_CONTEXT="$RAPIDS_RUN_CONTEXT"
    echo "Using run-context from environment: $RUN_CONTEXT"
else
    RUN_CONTEXT="main"
    echo "No run-context provided, defaulting to: $RUN_CONTEXT"
fi

# Validate RUN_CONTEXT value
if [[ "${RUN_CONTEXT}" != "main" && "${RUN_CONTEXT}" != "release" ]]; then
    echo "Error: Invalid run-context value '${RUN_CONTEXT}'"
    echo "Valid values: main, release"
    exit 1
fi

# Validate version argument
if [[ -z "$NEXT_FULL_TAG" ]]; then
    echo "Error: Version argument is required"
    echo "Usage: $0 <new_version> [--run-context=<context>]"
    echo "   or: [RAPIDS_RUN_CONTEXT=<context>] $0 <new_version>"
    echo "Note: Defaults to main when run-context is not specified"
    exit 1
fi

# Get current version
CURRENT_TAG=$(git tag --merged HEAD | grep -xE '^v.*' | sort --version-sort | tail -n 1 | tr -d 'v')
CURRENT_MAJOR=$(echo "$CURRENT_TAG" | awk '{split($0, a, "."); print a[1]}')
CURRENT_MINOR=$(echo "$CURRENT_TAG" | awk '{split($0, a, "."); print a[2]}')
CURRENT_SHORT_TAG=${CURRENT_MAJOR}.${CURRENT_MINOR}

# Get <major>.<minor> for next version
NEXT_MAJOR=$(echo "$NEXT_FULL_TAG" | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo "$NEXT_FULL_TAG" | awk '{split($0, a, "."); print a[2]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}

# Need to distutils-normalize the original version
NEXT_SHORT_TAG_PEP440=$(python -c "from packaging.version import Version; print(Version('${NEXT_SHORT_TAG}'))")

# Set branch references based on RUN_CONTEXT
if [[ "${RUN_CONTEXT}" == "main" ]]; then
    RAPIDS_BRANCH_NAME="main"
    echo "Preparing development branch update $CURRENT_TAG => $NEXT_FULL_TAG (targeting main branch)"
elif [[ "${RUN_CONTEXT}" == "release" ]]; then
    RAPIDS_BRANCH_NAME="release/${NEXT_SHORT_TAG}"
    echo "Preparing release branch update $CURRENT_TAG => $NEXT_FULL_TAG (targeting release/${NEXT_SHORT_TAG} branch)"
fi


# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' "$2" && rm -f "${2}".bak
}



# Centralized version file update
echo "${NEXT_FULL_TAG}" > VERSION
echo "${RAPIDS_BRANCH_NAME}" > RAPIDS_BRANCH

DEPENDENCIES=(
  cudf
  cuml
  cuvs
  dask-cuda
  dask-cudf
  libcuml
  libcuml-tests
  libcuvs
  libnvforest
  libraft-headers
  libraft
  librmm
  nvforest
  pylibraft
  raft-dask
  rapids-dask-dependency
  rmm
)
for DEP in "${DEPENDENCIES[@]}"; do
  for FILE in dependencies.yaml conda/environments/*.yaml; do
    sed_runner "/-.* ${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}\(\[.*\]\)\{0,1\}==/ s/==.*/==${NEXT_SHORT_TAG_PEP440}.*,>=0.0.0a0/g" "${FILE}"
  done
  for FILE in python/*/pyproject.toml; do
    sed_runner "/\"${DEP}==/ s/==.*\"/==${NEXT_SHORT_TAG_PEP440}.*,>=0.0.0a0\"/g" "${FILE}"
  done
done

# CI image tags of the form {rapids_version}-{something}
sed_runner "s/:[0-9]*\\.[0-9]*-/:${NEXT_SHORT_TAG}-/g" ./CONTRIBUTING.md

# branch references in docs
sed_runner "s|/release/[^/]*/|/${RAPIDS_BRANCH_NAME}/|g" README.md
sed_runner "s|/main/|/${RAPIDS_BRANCH_NAME}/|g" README.md
sed_runner "s|/release/[^/]*/|/${RAPIDS_BRANCH_NAME}/|g" python/cuml/README.md
sed_runner "s|/main/|/${RAPIDS_BRANCH_NAME}/|g" python/cuml/README.md

# CI files
for FILE in .github/workflows/*.yaml .github/workflows/*.yml; do
  sed_runner "/shared-workflows/ s|@.*|@${RAPIDS_BRANCH_NAME}|g" "${FILE}"
  # CI image tags of the form {rapids_version}-{something}
  sed_runner "s|:[0-9]*\\.[0-9]*-|:${NEXT_SHORT_TAG}-|g" "${FILE}"
done

# .devcontainer files
find .devcontainer/ -type f -name devcontainer.json -print0 | while IFS= read -r -d '' filename; do
    sed_runner "s@rapidsai/devcontainers:[0-9.]*@rapidsai/devcontainers:${NEXT_SHORT_TAG}@g" "${filename}"
    sed_runner "s@ghcr.io/rapidsai/cuml/devcontainer:[0-9.]*@ghcr.io/rapidsai/cuml/devcontainer:${NEXT_SHORT_TAG}@g" "${filename}"
    sed_runner "s@rapidsai/devcontainers/features/ucx:[0-9.]*@rapidsai/devcontainers/features/ucx:${NEXT_SHORT_TAG_PEP440}@" "${filename}"
    sed_runner "s@rapidsai/devcontainers/features/cuda:[0-9.]*@rapidsai/devcontainers/features/cuda:${NEXT_SHORT_TAG_PEP440}@" "${filename}"
    sed_runner "s@rapidsai/devcontainers/features/rapids-build-utils:[0-9.]*@rapidsai/devcontainers/features/rapids-build-utils:${NEXT_SHORT_TAG_PEP440}@" "${filename}"
    sed_runner "s@rapids-\${localWorkspaceFolderBasename}-${CURRENT_SHORT_TAG}@rapids-\${localWorkspaceFolderBasename}-${NEXT_SHORT_TAG}@g" "${filename}"
done
