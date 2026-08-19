#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import cudf
import cupy as cp
import numpy as np
import pytest

import cuml
from cuml.datasets import (
    make_arima,
    make_blobs,
    make_classification,
    make_regression,
)

TEST_OUTPUT_TYPES = (
    (None, (cp.ndarray, cp.ndarray)),  # Default is cupy if None is used
    ("numpy", (np.ndarray, np.ndarray)),
    ("cupy", (cp.ndarray, cp.ndarray)),
    ("cudf", (cudf.DataFrame, cudf.Series)),
)

GENERATORS = (make_blobs, make_classification, make_regression)


@pytest.mark.parametrize("generator", GENERATORS)
@pytest.mark.parametrize("output_str,output_types", TEST_OUTPUT_TYPES)
def test_xy_output_type(generator, output_str, output_types):
    # Set the output type and ensure data of that type is generated
    with cuml.using_output_type(output_str):
        data = generator(n_samples=10, random_state=0)

    for data, type_ in zip(data, output_types):
        assert isinstance(data, type_)


@pytest.mark.parametrize("output_str,output_types", TEST_OUTPUT_TYPES)
@pytest.mark.filterwarnings(
    "ignore:`cuml.datasets.make_arima`, along with the entire `cuml.tsa` module, "
    "was deprecated:FutureWarning"
)
def test_time_series_label_output_type(output_str, output_types):
    # Set the output type and ensure data of that type is generated
    with cuml.using_output_type(output_str):
        data = make_arima(n_obs=10, random_state=0)[0]

    assert isinstance(data, output_types[1])
