# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)

include(rapids-cuda)
rapids_cuda_init_architectures(libcuml-python)

project(
  libcuml-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

# Find C++ libraries from RAPIDS wheels in site-packages.
set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE)

# Check if cuml is already available. If so, it is the user's responsibility to ensure that the
# CMake package is also available at build time of the Python cuml package.
find_package(cuml "${RAPIDS_VERSION}")

if(cuml_FOUND)
  return()
endif()

unset(cuml_FOUND)

# --- cuML --- #
set(BUILD_CUML_TESTS OFF)
set(BUILD_PRIMS_TESTS OFF)
set(BUILD_CUML_EXAMPLES OFF)
set(BUILD_CUML_BENCH OFF)

# In libcuml wheels, we always want to build in all cuML algorithms. This is the default in
# cpp/CMakeLists.txt, but just making that choice for wheels explicit here.
set(CUML_ALGORITHMS
    "ALL"
    CACHE
      STRING
      "Choose which algorithms are built cuML. Can specify individual algorithms or groups in a semicolon-separated list."
)

# for libcuml wheels, always compile in the multi-node, multi-GPU functionality
set(SINGLEGPU OFF)

set(CUML_CPP_TARGET "cuml")
set(CUML_CPP_SRC "../../cpp")

# --- cuVS --- #
set(CUML_EXCLUDE_CUVS_FROM_ALL ON)

# --- raft --- #
set(CUML_RAFT_CLONE_ON_PIN OFF)
set(CUML_EXCLUDE_RAFT_FROM_ALL ON)

# --- treelite --- #
set(CUML_PYTHON_TREELITE_TARGET treelite::treelite_static)
set(CUML_USE_TREELITE_STATIC ON)
set(CUML_EXCLUDE_TREELITE_FROM_ALL ON)

# --- CUDA --- #
set(CUDA_STATIC_MATH_LIBRARIES OFF)

add_subdirectory(../../cpp cuml-cpp)

# RPATHs below assume the shared library is installed in a wheel layout:
# .../site-packages/libcuml/lib64/libcuml.so (see python/libcuml/libcuml/load.py and
# python/libcuml/tests/test_libcuml_linkage.py, which resolve libcuml.so under the libcuml package's
# lib64 directory).
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
  # starting with CTK 13 wheels, all libraries except NCCL are grouped in this 'nvidia/cu13/lib'
  # directory
  set(rpaths "$ORIGIN/../../nvidia/cu13/lib")
else()
  set(rpaths
      "$ORIGIN/../../nvidia/cublas/lib"
      "$ORIGIN/../../nvidia/cufft/lib"
      "$ORIGIN/../../nvidia/curand/lib"
      "$ORIGIN/../../nvidia/cusolver/lib"
      "$ORIGIN/../../nvidia/cusparse/lib"
      "$ORIGIN/../../nvidia/nvjitlink/lib"
  )
endif()

set_property(
  TARGET ${CUML_CPP_TARGET}
  PROPERTY INSTALL_RPATH ${rpaths}
  APPEND
)
