# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# ##################################################################################################
# * build cuml bench executable ----------------------------------------------

if(BUILD_CUML_BENCH)
  # (please keep the filenames in alphabetical order)
  add_executable(
    ${CUML_CPP_BENCH_TARGET} sg/dbscan.cu sg/kmeans.cu sg/linkage.cu sg/main.cpp
                             sg/rf_classifier.cu sg/rf_regressor.cu sg/svc.cu sg/svr.cu sg/umap.cu
  )
  if(CUML_ENABLE_GPU)
    target_compile_definitions(${CUML_CPP_BENCH_TARGET} PUBLIC CUML_ENABLE_GPU)
  endif()

  target_compile_options(
    ${CUML_CPP_BENCH_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUML_CXX_FLAGS}>"
                                     "$<$<COMPILE_LANGUAGE:CUDA>:${CUML_CUDA_FLAGS}>"
  )

  target_link_libraries(
    ${CUML_CPP_BENCH_TARGET} PUBLIC cuml::${CUML_CPP_TARGET} benchmark::benchmark ${TREELITE_LIBS}
                                    ${CUVS_LIB} raft::raft
  )

  target_include_directories(
    ${CUML_CPP_BENCH_TARGET} PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../src_prims>
  )

  set_target_properties(
    ${CUML_CPP_BENCH_TARGET}
    PROPERTIES INSTALL_RPATH "\$ORIGIN/../../../lib"
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
  )

  install(
    TARGETS ${CUML_CPP_BENCH_TARGET}
    COMPONENT testing
    DESTINATION bin/benchmarks/libcuml
    EXCLUDE_FROM_ALL
  )
endif()
