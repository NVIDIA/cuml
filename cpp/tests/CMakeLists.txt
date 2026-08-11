# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
enable_testing()

# We use rapids-cmake testing infrastructure to allow us to run multiple GPU tests concurrently
# without causing OOM issues. Use the `GPUS` and `PERCENT` options to control how 'much' of the GPUs
# you need for a test:
#
# GPUS 1 PERCENT 25 -> I need 25% of a single GPU GPUS 1 PERCENT 100 -> all of 1 GPU GPUS 2 PERCENT
# 200 -> all of 2 GPUs (will only run this test on 2 GPU machines)
include(rapids-test)
rapids_test_init()

function(ConfigureTest)

  set(options MPI ML_INCLUDE RAFT_DISTRIBUTED NO_GTEST_MAIN)
  set(one_value PREFIX NAME GPUS PERCENT)
  set(multi_value TARGETS CONFIGURATIONS)
  cmake_parse_arguments(_CUML_TEST "${options}" "${one_value}" "${multi_value}" ${ARGN})
  if(NOT DEFINED _CUML_TEST_GPUS AND NOT DEFINED _CUML_TEST_PERCENT)
    set(_CUML_TEST_GPUS 1)
    set(_CUML_TEST_PERCENT 15)
  endif()
  if(NOT DEFINED _CUML_TEST_GPUS)
    set(_CUML_TEST_GPUS 1)
  endif()
  if(NOT DEFINED _CUML_TEST_PERCENT)
    set(_CUML_TEST_PERCENT 100)
  endif()
  string(PREPEND _CUML_TEST_NAME "${_CUML_TEST_PREFIX}_")

  add_executable(${_CUML_TEST_NAME} ${_CUML_TEST_UNPARSED_ARGUMENTS})
  target_link_libraries(
    ${_CUML_TEST_NAME}
    PRIVATE ${CUML_CPP_TARGET}
            CUDA::cublas${_ctk_static_suffix}
            CUDA::curand${_ctk_static_suffix}
            CUDA::cusolver${_ctk_static_suffix}
            CUDA::cusparse${_ctk_static_suffix}
            $<$<BOOL:${LINK_CUFFT}>:CUDA::cufft${_ctk_static_suffix_cufft}>
            $<$<BOOL:${LINK_NVFOREST}>:${CUML_NVFOREST_TARGET}>
            CUDA::cudart_static
            rmm::rmm
            raft::raft
            GTest::gtest
            $<$<NOT:$<BOOL:${_CUML_TEST_NO_GTEST_MAIN}>>:GTest::gtest_main>
            GTest::gmock
            ${OpenMP_CXX_LIB_NAMES}
            Threads::Threads
            $<$<BOOL:${_CUML_TEST_MPI}>:${MPI_CXX_LIBRARIES}>
            $<$<BOOL:${_CUML_TEST_RAFT_DISTRIBUTED}>:raft::distributed>
            ${TREELITE_LIBS}
            ${CUVS_LIB}
            $<TARGET_NAME_IF_EXISTS:conda_env>
  )

  target_compile_options(
    ${_CUML_TEST_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUML_CXX_FLAGS}>"
                               "$<$<COMPILE_LANGUAGE:CUDA>:${CUML_CUDA_FLAGS}>"
  )

  target_include_directories(
    ${_CUML_TEST_NAME}
    PRIVATE
      $<$<BOOL:${_CUML_TEST_ML_INCLUDE}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>>
      $<$<BOOL:${_CUML_TEST_ML_INCLUDE}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../src>>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../src_prims>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/prims>
  )
  set_target_properties(
    ${_CUML_TEST_NAME}
    PROPERTIES INSTALL_RPATH "\$ORIGIN/../../../lib"
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
  )

  set(_CUML_TEST_COMPONENT_NAME testing)
  if(_CUML_TEST_PREFIX STREQUAL "PRIMS")
    set(_CUML_TEST_COMPONENT_NAME cumlprims_testing)
  endif()

  rapids_test_add(
    NAME ${_CUML_TEST_NAME}
    COMMAND ${_CUML_TEST_NAME}
    GPUS ${_CUML_TEST_GPUS}
    PERCENT ${_CUML_TEST_PERCENT}
    INSTALL_COMPONENT_SET ${_CUML_TEST_COMPONENT_NAME}
  )

endfunction()

# ##################################################################################################
# * build ml_test executable -------------------------------------------------
if(BUILD_CUML_TESTS AND (all_algo OR dbscan_algo))
  ConfigureTest(PREFIX SG NAME DBSCAN_TEST sg/dbscan_test.cu ML_INCLUDE)
endif()

if(BUILD_CUML_TESTS AND (all_algo OR explainer_algo))
  ConfigureTest(PREFIX SG NAME SHAP_KERNEL_TEST sg/shap_kernel.cu ML_INCLUDE)
endif()

# todo: organize linear models better
if(BUILD_CUML_TESTS
   AND (all_algo
        OR linearregression_algo
        OR ridge_algo
        OR lasso_algo
        OR logisticregression_algo
       )
)
  ConfigureTest(PREFIX SG NAME OLS_TEST sg/ols.cu ML_INCLUDE)
  ConfigureTest(PREFIX SG NAME RIDGE_TEST sg/ridge.cu ML_INCLUDE)
endif()

if(BUILD_CUML_TESTS AND (all_algo OR genetic_algo))
  ConfigureTest(PREFIX SG NAME GENETIC_NODE_TEST sg/genetic/node_test.cpp ML_INCLUDE)
  ConfigureTest(PREFIX SG NAME GENETIC_PARAM_TEST sg/genetic/param_test.cu ML_INCLUDE)
endif()

if(BUILD_CUML_TESTS AND (all_algo OR hdbscan_algo))
  ConfigureTest(PREFIX SG NAME HDBSCAN_TEST sg/hdbscan_test.cu ML_INCLUDE)
  # When using GCC 13, some maybe-uninitialized warnings appear from CCCL and are treated as errors.
  # See this issue: https://github.com/rapidsai/cuml/issues/6225
  set_property(
    SOURCE sg/hdbscan_test.cu
    APPEND_STRING
    PROPERTY COMPILE_FLAGS " -Xcompiler=-Wno-maybe-uninitialized"
  )
endif()

if(BUILD_CUML_TESTS AND (all_algo OR holtwinters_algo))
  ConfigureTest(PREFIX SG NAME HOLTWINTERS_TEST sg/holtwinters_test.cu ML_INCLUDE)
endif()

if(BUILD_CUML_TESTS AND (all_algo OR knn_algo))
  ConfigureTest(PREFIX SG NAME KNN_TEST sg/knn_test.cu ML_INCLUDE)
endif()

if(BUILD_CUML_TESTS AND (all_algo OR metrics_algo))
  ConfigureTest(PREFIX SG NAME TRUSTWORTHINESS_TEST sg/trustworthiness_test.cu ML_INCLUDE)
endif()

if(BUILD_CUML_TESTS AND (all_algo OR pca_algo))
  ConfigureTest(PREFIX SG NAME PCA_TEST sg/pca_test.cu ML_INCLUDE)
endif()

if(BUILD_CUML_TESTS AND (all_algo OR randomforest_algo))
  ConfigureTest(
    PREFIX SG
    NAME RF_TEST sg/rf_test.cu ML_INCLUDE
    GPUS 1
    PERCENT 100
  )
endif()

if(all_algo OR isolationforest_algo)
  ConfigureTest(
    PREFIX SG
    NAME ISOLATION_FOREST_TEST sg/isolation_forest_test.cu ML_INCLUDE
    GPUS 1
    PERCENT 100
  )
endif()

# todo: separate solvers better
if(BUILD_CUML_TESTS AND (all_algo OR solvers_algo))
  ConfigureTest(PREFIX SG NAME CD_TEST sg/cd_test.cu ML_INCLUDE)
  ConfigureTest(PREFIX SG NAME LARS_TEST sg/lars_test.cu ML_INCLUDE)
  ConfigureTest(PREFIX SG NAME QUASI_NEWTON sg/quasi_newton.cu ML_INCLUDE)
  ConfigureTest(PREFIX SG NAME SGD_TEST sg/sgd.cu ML_INCLUDE)
endif()

if(BUILD_CUML_TESTS AND (all_algo OR svm_algo))
  ConfigureTest(
    PREFIX SG
    NAME SVC_TEST sg/svc_test.cu ML_INCLUDE
    GPUS 1
    PERCENT 100
  )
  # The SVC Test tries to verify it has no memory leaks by checking how much free memory on the GPU
  # exists after execution. This check requires no other GPU tests to be running or it fails since
  # it thinks it has a memory leak
  set_tests_properties(SG_SVC_TEST PROPERTIES RUN_SERIAL ON)
endif()

if(BUILD_CUML_TESTS AND (all_algo OR tsne_algo))
  ConfigureTest(PREFIX SG NAME TSNE_TEST sg/tsne_test.cu ML_INCLUDE)
endif()

if(BUILD_CUML_TESTS AND (all_algo OR tsvd_algo))
  ConfigureTest(PREFIX SG NAME TSVD_TEST sg/tsvd_test.cu ML_INCLUDE)
endif()

if(BUILD_CUML_TESTS AND (all_algo OR umap_algo))
  ConfigureTest(PREFIX SG NAME UMAP_PARAMETRIZABLE_TEST sg/umap_parametrizable_test.cu ML_INCLUDE)
endif()

# ##################################################################################################
# * build test_ml_mg executable ----------------------------------------------

if(BUILD_CUML_MG_TESTS)

  # This test needs to be rewritten to use the MPI comms, not the std comms, and moved to RAFT:
  # https://github.com/rapidsai/cuml/issues/5058 ConfigureTest(PREFIX MG NAME KMEANS_TEST
  # mg/kmeans_test.cu NCCL ML_INCLUDE)
  if(MPI_CXX_FOUND)
    # (please keep the filenames in alphabetical order)
    ConfigureTest(
      PREFIX MG NAME KNN_TEST mg/knn.cu mg/main.cu MPI RAFT_DISTRIBUTED ML_INCLUDE NO_GTEST_MAIN
    )
    ConfigureTest(
      PREFIX MG NAME KNN_CLASSIFY_TEST mg/knn_classify.cu mg/main.cu MPI RAFT_DISTRIBUTED
                                       ML_INCLUDE NO_GTEST_MAIN
    )
    ConfigureTest(
      PREFIX MG NAME KNN_REGRESS_TEST mg/knn_regress.cu mg/main.cu MPI RAFT_DISTRIBUTED ML_INCLUDE
                                      NO_GTEST_MAIN
    )
    ConfigureTest(PREFIX MG NAME MAIN_TEST mg/main.cu MPI RAFT_DISTRIBUTED ML_INCLUDE)
    ConfigureTest(
      PREFIX MG NAME PCA_TEST mg/pca.cu mg/main.cu MPI RAFT_DISTRIBUTED ML_INCLUDE NO_GTEST_MAIN
    )
    ConfigureTest(
      PREFIX MG NAME RF_QUANTILE_TEST mg/rf_quantile_test.cu MPI RAFT_DISTRIBUTED ML_INCLUDE
    )
    ConfigureTest(PREFIX MG NAME RF_TEST mg/rf_test.cu MPI RAFT_DISTRIBUTED ML_INCLUDE)
  else(MPI_CXX_FOUND)
    message("OpenMPI not found. Skipping MultiGPU tests '${CUML_MG_TEST_TARGET}'")
  endif()
endif()

# ##################################################################################################
# * build prims_test executable ----------------------------------------------

if(BUILD_PRIMS_TESTS)
  # (please keep the filenames in alphabetical order)
  ConfigureTest(PREFIX PRIMS NAME ADD_SUB_DEV_SCALAR_TEST prims/add_sub_dev_scalar.cu)
  ConfigureTest(PREFIX PRIMS NAME BATCHED_CSR_TEST prims/batched/csr.cu)
  ConfigureTest(PREFIX PRIMS NAME BATCHED_GEMV_TEST prims/batched/gemv.cu)
  ConfigureTest(PREFIX PRIMS NAME BATCHED_MAKE_SYMM_TEST prims/batched/make_symm.cu)
  ConfigureTest(PREFIX PRIMS NAME BATCHED_MATRIX_TEST prims/batched/matrix.cu)
  ConfigureTest(PREFIX PRIMS NAME CHECKED_ARITHMETIC_TEST prims/checked_arithmetic.cpp ML_INCLUDE)
  ConfigureTest(PREFIX PRIMS NAME DECOUPLED_LOOKBACK_TEST prims/decoupled_lookback.cu)
  ConfigureTest(PREFIX PRIMS NAME DEVICE_UTILS_TEST prims/device_utils.cu)
  ConfigureTest(PREFIX PRIMS NAME ELTWISE2D_TEST prims/eltwise2d.cu)
  ConfigureTest(PREFIX PRIMS NAME FAST_INT_DIV_TEST prims/fast_int_div.cu)
  ConfigureTest(PREFIX PRIMS NAME FILLNA_TEST prims/fillna.cu)
  ConfigureTest(PREFIX PRIMS NAME GRID_SYNC_TEST prims/grid_sync.cu)
  ConfigureTest(PREFIX PRIMS NAME HINGE_TEST prims/hinge.cu)
  ConfigureTest(PREFIX PRIMS NAME JONES_TRANSFORM_TEST prims/jones_transform.cu)
  ConfigureTest(PREFIX PRIMS NAME KNN_CLASSIFY_TEST prims/knn_classify.cu)
  ConfigureTest(PREFIX PRIMS NAME KNN_REGRESSION_TEST prims/knn_regression.cu)
  ConfigureTest(PREFIX PRIMS NAME KSELECTION_TEST prims/kselection.cu)
  ConfigureTest(PREFIX PRIMS NAME LINALG_BLOCK_TEST prims/linalg_block.cu)
  ConfigureTest(PREFIX PRIMS NAME LINEARREG_TEST prims/linearReg.cu)
  ConfigureTest(PREFIX PRIMS NAME LOG_TEST prims/log.cu)
  ConfigureTest(PREFIX PRIMS NAME LOGISTICREG_TEST prims/logisticReg.cu)
  ConfigureTest(PREFIX PRIMS NAME MAKE_ARIMA_TEST prims/make_arima.cu)
  ConfigureTest(PREFIX PRIMS NAME PENALTY_TEST prims/penalty.cu)
  ConfigureTest(PREFIX PRIMS NAME SIGMOID_TEST prims/sigmoid.cu)

  rapids_test_install_relocatable(
    INSTALL_COMPONENT_SET cumlprims_testing DESTINATION bin/gtests/libcuml_prims
  )
endif()

rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing DESTINATION bin/gtests/libcuml)
