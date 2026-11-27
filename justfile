# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set minimum-version := '1.55.0'
set positional-arguments

# List available recipes.
default:
    @just --list

# Build cuML.
build *args:
    @exec ./build.sh "$@"

# Run standard tests.
test *args:
    @exec python -m pytest "$@"

# Run cuml.accel tests.
test-accel *args:
    @exec python -m pytest python/cuml/cuml_accel_tests "$@"

# Run upstream scikit-learn tests with cuml.accel.
test-accel-sklearn *args:
    @exec python -m pytest -p cuml.accel --pyargs sklearn \
        --rootdir=python/cuml/cuml_accel_tests/upstream/scikit-learn \
        -c python/cuml/cuml_accel_tests/upstream/pytest.ini \
        --xfail-list=python/cuml/cuml_accel_tests/upstream/scikit-learn/xfail-list.yaml "$@"

# Run Dask tests.
test-dask *args:
    @exec python -m pytest python/cuml/dask_tests "$@"

# Run standard tests with cudf.pandas.
test-cudf-pandas *args:
    @exec python -m pytest -p cudf.pandas python/cuml/tests "$@"
