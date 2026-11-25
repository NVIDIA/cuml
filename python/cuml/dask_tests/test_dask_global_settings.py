#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable=no-member

from time import sleep

import pytest
from dask import delayed

import cuml
from cuml import set_global_output_type, using_output_type
from cuml.internals.global_settings import (
    GlobalSettings,
    _global_settings_data,
    _GlobalSettingsData,
)

test_output_types_str = ("numpy", "cupy", "cudf")
test_global_settings_data_obj = _GlobalSettingsData()


def test_set_global_output_type():
    """Ensure that set_global_output_type is thread-safe"""

    def check_correct_type(index):
        output_type = test_output_types_str[index]
        # Force a race condition
        if index == 0:
            sleep(0.1)
        set_global_output_type(output_type)
        sleep(0.5)
        return cuml.global_settings.output_type == output_type

    results = [
        delayed(check_correct_type)(index)
        for index in range(len(test_output_types_str))
    ]

    assert (delayed(all)(results)).compute()


def test_using_output_type():
    """Ensure that using_output_type is thread-safe"""

    def check_correct_type(index):
        output_type = test_output_types_str[index]
        # Force a race condition
        if index == 0:
            sleep(0.1)
        with using_output_type(output_type):
            sleep(0.5)
            return cuml.global_settings.output_type == output_type

    results = [
        delayed(check_correct_type)(index)
        for index in range(len(test_output_types_str))
    ]

    assert (delayed(all)(results)).compute()


def test_global_settings_data():
    """Ensure that GlobalSettingsData objects are properly initialized
    per-thread"""

    def check_initialized(index):
        if index == 0:
            sleep(0.1)

        with pytest.raises(AttributeError):
            _global_settings_data.testing_index  # pylint: disable=W0104
        _global_settings_data.testing_index = index

        sleep(0.5)
        return (
            test_global_settings_data_obj.shared_state["_output_type"] is None
            and test_global_settings_data_obj.shared_state[
                "_external_output_type"
            ]
            is False
            and _global_settings_data.testing_index == index
        )

    results = [delayed(check_initialized)(index) for index in range(5)]

    assert (delayed(all)(results)).compute()


def test_global_settings():
    """Ensure that GlobalSettings acts as a proper thread-local borg"""

    def check_settings(index):
        # Force a race condition
        if index == 0:
            sleep(0.1)
        cuml.global_settings.index = index
        sleep(0.5)
        return (
            cuml.global_settings.index == index
            and cuml.global_settings.index == GlobalSettings().index
        )

    results = [delayed(check_settings)(index) for index in range(5)]

    assert (delayed(all)(results)).compute()
