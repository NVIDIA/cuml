# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

set(cython_sources "")
add_module_gpu_default("arima.pyx" ${arima_algo} ${datasets_algo})
add_module_gpu_default("_blobs.pyx" ${datasets_algo})
add_module_gpu_default("regression.pyx" ${regression_algo} ${datasets_algo})

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${cuml_sg_libraries}" MODULE_PREFIX datasets_
)
