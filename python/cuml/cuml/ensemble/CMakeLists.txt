# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

set(cython_sources "")
add_module_gpu_default(
  "randomforest_common.pyx" ${randomforestclassifier_algo} ${randomforestregressor_algo}
  ${ensemble_algo}
)
add_module_gpu_default("isolation_forest.pyx" ${isolationforest_algo} ${ensemble_algo})

set(linked_libraries ${cuml_sg_libraries} ${CUML_PYTHON_TREELITE_TARGET})

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}" MODULE_PREFIX ensemble_
)
