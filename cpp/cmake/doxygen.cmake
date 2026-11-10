# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#

find_package(Doxygen 1.18.0)

function(add_doxygen_target)
  if(Doxygen_FOUND)
    set(options "")
    set(oneValueArgs IN_DOXYFILE OUT_DOXYFILE CWD)
    set(multiValueArgs "")
    cmake_parse_arguments(dox "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    configure_file(${dox_IN_DOXYFILE} ${dox_OUT_DOXYFILE} @ONLY)
    add_custom_target(
      docs_cuml
      ${CMAKE_COMMAND}
      -E
      env
      "RAPIDS_VERSION=${RAPIDS_VERSION}"
      "RAPIDS_VERSION_MAJOR_MINOR=${RAPIDS_VERSION_MAJOR_MINOR}"
      ${DOXYGEN_EXECUTABLE}
      ${dox_OUT_DOXYFILE}
      WORKING_DIRECTORY ${dox_CWD}
      VERBATIM
      COMMENT "Generate doxygen docs"
    )
  else()
    message("add_doxygen_target: doxygen exe not found")
  endif()
endfunction(add_doxygen_target)
