#=============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#=============================================================================

set(CUML_MIN_VERSION_nvforest "${CUML_VERSION_MAJOR}.${CUML_VERSION_MINOR}.00")

function(find_and_configure_nvforest)
    set(oneValueArgs VERSION FORK PINNED_TAG EXCLUDE_FROM_ALL CLONE_ON_PIN)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN} )

    if(PKG_CLONE_ON_PIN AND NOT PKG_PINNED_TAG STREQUAL "${rapids-cmake-checkout-tag}")
        message(STATUS "CUML: nvForest pinned tag found: ${PKG_PINNED_TAG}. Cloning nvForest locally.")
        set(CPM_DOWNLOAD_nvforest ON)
    endif()

    rapids_cpm_find(nvforest ${PKG_VERSION}
            GLOBAL_TARGETS      nvforest::nvforest
            BUILD_EXPORT_SET    cuml-exports
            INSTALL_EXPORT_SET  cuml-exports
            CPM_ARGS
              GIT_REPOSITORY         https://github.com/${PKG_FORK}/nvforest.git
              GIT_TAG                ${PKG_PINNED_TAG}
              SOURCE_SUBDIR          cpp
              EXCLUDE_FROM_ALL       ${PKG_EXCLUDE_FROM_ALL}
              OPTIONS
                "BUILD_NVFOREST_TESTS OFF"
                "NVFOREST_USE_TREELITE_STATIC ${CUML_USE_TREELITE_STATIC}"
    )

    if(nvforest_ADDED)
        message(VERBOSE "CUML: Using nvForest located in ${nvforest_SOURCE_DIR}")
    else()
        message(VERBOSE "CUML: Using nvForest located in ${nvforest_DIR}")
    endif()

endfunction()

# Change pinned tag here to test a commit in CI
# To use a different nvForest locally, set the CMake variable
# CPM_nvforest_SOURCE=/path/to/local/nvforest
find_and_configure_nvforest(VERSION          ${CUML_MIN_VERSION_nvforest}
        FORK             rapidsai
        PINNED_TAG       ${rapids-cmake-checkout-tag}
        EXCLUDE_FROM_ALL ${CUML_EXCLUDE_NVFOREST_FROM_ALL}
        # When PINNED_TAG above doesn't match cuml,
        # force local nvforest clone in build directory
        # even if it's already installed.
        CLONE_ON_PIN     ${CUML_NVFOREST_CLONE_ON_PIN}
)
