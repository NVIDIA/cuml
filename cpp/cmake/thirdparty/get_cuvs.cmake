#=============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#=============================================================================

set(CUML_MIN_VERSION_cuvs "${CUML_VERSION_MAJOR}.${CUML_VERSION_MINOR}.00")

# Prefix CMake messages with the project context to make CI logs easier to diagnose.
list(APPEND CMAKE_MESSAGE_CONTEXT "CUML")

function(find_and_configure_cuvs)
    set(oneValueArgs VERSION FORK PINNED_TAG EXCLUDE_FROM_ALL USE_CUVS_STATIC COMPILE_LIBRARY CLONE_ON_PIN)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN} )

    if(PKG_CLONE_ON_PIN AND NOT PKG_PINNED_TAG STREQUAL "${rapids-cmake-checkout-tag}")
        message(STATUS "CUVS pinned tag found: ${PKG_PINNED_TAG}. Cloning cuvs locally.")
        set(CPM_DOWNLOAD_cuvs ON)
    elseif(PKG_USE_CUVS_STATIC AND (NOT CPM_cuvs_SOURCE))
        message(STATUS "Cloning cuvs locally to build static libraries.")
        set(CPM_DOWNLOAD_cuvs ON)
    else()
        message(STATUS "Not cloning cuvs locally")
    endif()

    if(PKG_USE_CUVS_STATIC)
      set(CUVS_LIB cuvs::cuvs_static PARENT_SCOPE)
    else()
      set(CUVS_LIB cuvs::cuvs PARENT_SCOPE)
    endif()

    set(CUVS_BUILD_MG_ALGOS ON)
    if(SINGLEGPU)
      set(CUVS_BUILD_MG_ALGOS OFF)
    endif()

    rapids_cpm_find(cuvs ${PKG_VERSION}
      GLOBAL_TARGETS      cuvs::cuvs
      CPM_ARGS
        GIT_REPOSITORY         https://github.com/${PKG_FORK}/cuvs.git
        GIT_TAG                ${PKG_PINNED_TAG}
        SOURCE_SUBDIR          cpp
        EXCLUDE_FROM_ALL       ${PKG_EXCLUDE_FROM_ALL}
        OPTIONS
          "BUILD_TESTS OFF"
          "BUILD_CAGRA_HNSWLIB OFF"
          "BUILD_CUVS_BENCH OFF"
          "BUILD_MG_ALGOS ${CUVS_BUILD_MG_ALGOS}"

    )

    if(cuvs_ADDED)
        message(VERBOSE "Using CUVS located in ${cuvs_SOURCE_DIR}")
    else()
        message(VERBOSE "Using CUVS located in ${cuvs_DIR}")
    endif()


endfunction()

# Change pinned tag here to test a commit in CI
# To use a different CUVS locally, set the CMake variable
# CPM_cuvs_SOURCE=/path/to/local/cuvs
find_and_configure_cuvs(VERSION          ${CUML_MIN_VERSION_cuvs}
      FORK             rapidsai
      PINNED_TAG       ${rapids-cmake-checkout-tag}
      EXCLUDE_FROM_ALL ${CUML_EXCLUDE_CUVS_FROM_ALL}
      # When PINNED_TAG above doesn't match cuml,
      # force local cuvs clone in build directory
      # even if it's already installed.
      CLONE_ON_PIN     ${CUML_CUVS_CLONE_ON_PIN}
      COMPILE_LIBRARY  ${CUML_CUVS_COMPILED}
      USE_CUVS_STATIC  ${CUML_USE_CUVS_STATIC}
      )

# Restore the outer CMAKE_MESSAGE_CONTEXT.
list(POP_BACK CMAKE_MESSAGE_CONTEXT)
