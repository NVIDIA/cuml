# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

set(cython_sources "")
add_module_gpu_default("adjusted_mutual_info_score.pyx")
add_module_gpu_default("adjusted_rand_index.pyx" ${adjusted_rand_index_algo} ${metrics_algo})
add_module_gpu_default("completeness_score.pyx" ${completeness_score_algo} ${metrics_algo})
add_module_gpu_default("entropy.pyx" ${entropy_algo} ${metrics_algo})
add_module_gpu_default("homogeneity_score.pyx" ${homogeneity_score_algo} ${metrics_algo})
add_module_gpu_default("mutual_info_score.pyx" ${mutual_info_score_algo} ${metrics_algo})
add_module_gpu_default("silhouette_score.pyx" ${silhouette_score_algo} ${metrics_algo})
add_module_gpu_default("v_measure.pyx" ${v_measure_algo} ${metrics_algo})

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${cuml_sg_libraries}"
)
