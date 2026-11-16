# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

set(cython_sources cd.pyx qn.pyx sgd.pyx nnls.pyx)

if(NOT SINGLEGPU)
  list(APPEND cython_sources cd_mg.pyx)
endif()

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${cuml_mg_libraries}" MODULE_PREFIX solvers_
)
