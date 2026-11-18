#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This script runs all scikit-learn examples with the cuml.accel plugin.
# Any arguments passed to this script are forwarded directly to pytest.
#
# Example usage:
#   ./run-examples.sh                          # Run all examples
#   ./run-examples.sh -v -k plot_kmeans        # Run specific example with verbosity
#   ./run-examples.sh -x                       # Stop on first failure
#   ./run-examples.sh -n auto --dist worksteal # Run in parallel

set -eu

SKLEARN_VERSION=$(python -c "import sklearn; print(sklearn.__version__)")
SKLEARN_TAG="${SKLEARN_VERSION}"

THIS_DIRECTORY=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
SKLEARN_REPO="${THIS_DIRECTORY}/sklearn-upstream"

# Clone if not already present, then check out the matching tag
if [ ! -d "$SKLEARN_REPO" ]; then
    git clone "https://github.com/scikit-learn/scikit-learn.git" "$SKLEARN_REPO"
fi
git -C "$SKLEARN_REPO" fetch --tags
git -C "$SKLEARN_REPO" checkout "$SKLEARN_TAG"

# Run scikit-learn examples under cuml.accel.
# The example_collector plugin is loaded via -p to register the custom
# collector hooks regardless of which directory pytest collects from.
PYTHONPATH="${THIS_DIRECTORY}:${PYTHONPATH:-}" \
pytest -p example_collector -p cuml.accel \
    "${SKLEARN_REPO}/examples" \
    --rootdir="${THIS_DIRECTORY}" \
    --config-file="${THIS_DIRECTORY}/../pytest-upstream.ini" \
    --xfail-list="${THIS_DIRECTORY}/xfail-examples.yaml" \
    --examples-dir="${SKLEARN_REPO}/examples" \
    "$@"
