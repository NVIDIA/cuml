#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# cuml build script

# This script is used to build the component(s) in this repo from
# source, and can be called with various options to customize the
# build as needed (see the help output for details)

# Abort script on first error
set -e

NUMARGS=$#
ARGS=$*

# NOTE: ensure all dir changes are relative to the location of this
# script, and that this script resides in the repo dir!
REPODIR=$(cd "$(dirname "$0")"; pwd)

VALIDTARGETS="clean libcuml cuml cpp-mgtests prims bench prims-bench cppdocs pydocs"
VALIDFLAGS="-v -g -n --allgpuarch --singlegpu --nolibcumltest --nvtx --show_depr_warn --codecov --ccache --configure-only --static-treelite --build-metrics --incl-cache-stats -h --help "
VALIDARGS="${VALIDTARGETS} ${VALIDFLAGS}"
HELP="$0 [<target> ...] [<flag> ...]
 where <target> is:
   clean              - remove all existing build artifacts and configuration (start over)
   libcuml            - build the cuml C++ code only.
   cuml               - build the cuml Python package
   cpp-mgtests        - build libcuml MNMG tests. Requires MPI and RAFT distributed dependencies.
   prims              - build the ml-prims tests
   bench              - build the libcuml C++ benchmark
   prims-bench        - build the ml-prims C++ benchmark
   cppdocs            - build the C++ API doxygen documentation
   pydocs             - build the general and Python API documentation
 and <flag> is:
   -v                 - verbose build mode
   -g                 - build for debug
   -n                 - no install step
   -h                 - print this text
   --allgpuarch       - build for all supported GPU architectures
   --singlegpu        - Build libcuml and cuml without multigpu components
   --nolibcumltest    - disable building libcuml C++ tests for a faster build
   --nvtx             - Enable nvtx for profiling support
   --show_depr_warn   - show cmake deprecation warnings
   --codecov          - Enable code coverage support by compiling with Cython linetracing
                        and profiling enabled (WARNING: Impacts performance)
   --ccache           - Use ccache to cache previous compilations
   --configure-only   - Invoke CMake without actually building
   --static-treelite  - Force CMake to use the Treelite static libs, cloning and building them if necessary
   --build-metrics    - filename for generating build metrics report for libcuml
   --incl-cache-stats - include cache statistics in build metrics report

 default action (no args) is to build and install 'libcuml', 'cuml', and 'prims' targets only for the detected GPU arch

 The following environment variables are also accepted to allow further customization:
   PARALLEL_LEVEL         - Number of parallel threads to use in compilation.
   CUML_EXTRA_CMAKE_ARGS  - Extra arguments to pass directly to cmake. Values listed in environment
                            variable will override existing arguments.
   CUML_EXTRA_PYTHON_ARGS - Extra arguments to pass directly to pip install
"
LIBCUML_BUILD_DIR=${LIBCUML_BUILD_DIR:=${REPODIR}/cpp/build}
CUML_BUILD_DIR=${REPODIR}/python/cuml/build
PYTHON_DEPS_CLONE=${REPODIR}/python/external_repositories
BUILD_DIRS="${LIBCUML_BUILD_DIR} ${CUML_BUILD_DIR} ${PYTHON_DEPS_CLONE}"

# Set defaults for vars modified by flags to this script
BUILD_TYPE=Release
INSTALL_TARGET=install
BUILD_ALL_GPU_ARCH=0
SINGLEGPU_CPP_FLAG=""
NVTX=OFF
CCACHE=OFF
CLEAN=0
BUILD_DISABLE_DEPRECATION_WARNINGS=ON
BUILD_CUML_TESTS=ON
BUILD_CUML_MG_TESTS=OFF
BUILD_STATIC_TREELITE=OFF
CMAKE_LOG_LEVEL=WARNING
BUILD_REPORT_METRICS=OFF
BUILD_REPORT_INCL_CACHE_STATS=OFF

# Set defaults for vars that may not have been defined externally
INSTALL_PREFIX=${INSTALL_PREFIX:=${PREFIX:=${CONDA_PREFIX:=$LIBCUML_BUILD_DIR/install}}}
PARALLEL_LEVEL=${PARALLEL_LEVEL:=$(nproc)}
PYTHON_ARGS_FOR_INSTALL=(
    -v
    --no-build-isolation
    --no-deps
    --config-settings
    "rapidsai.disable-cuda=true"
)

# Default to Ninja if generator is not specified
export CMAKE_GENERATOR="${CMAKE_GENERATOR:=Ninja}"

# Allow setting arbitrary cmake args via CUML_EXTRA_CMAKE_ARGS. Any
# values listed there will override existing arguments.
CUML_EXTRA_CMAKE_ARGS=${CUML_EXTRA_CMAKE_ARGS:=""}

read -ra CUML_EXTRA_CMAKE_ARGS <<< "$CUML_EXTRA_CMAKE_ARGS"

CUML_EXTRA_PYTHON_ARGS=${CUML_EXTRA_PYTHON_ARGS:=""}

read -ra CUML_EXTRA_PYTHON_ARGS <<< "$CUML_EXTRA_PYTHON_ARGS"

function hasArg {
    (( NUMARGS != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Check for valid usage
if (( NUMARGS != 0 )); then
    for a in ${ARGS}; do
    if ! (echo " ${VALIDARGS} " | grep -q " ${a} "); then
        echo "Invalid option or formatting, check --help: ${a}"
        exit 1
    fi
    done
fi

function completeBuild {
    (( NUMARGS == 0 )) && return
    for a in ${ARGS}; do
        if (echo " ${VALIDTARGETS} " | grep -q " ${a} "); then
          false; return
        fi
    done
    true
}

if hasArg -h || hasArg --help; then
    echo "${HELP}"
    exit 0
fi

if hasArg clean; then
    CLEAN=1
fi

if hasArg cpp-mgtests; then
    BUILD_CUML_MG_TESTS=ON
fi

# Long arguments
LONG_ARGUMENT_LIST=(
    "verbose"
    "debug"
    "no-install"
    "allgpuarch"
    "singlegpu"
    "nvtx"
    "show_depr_warn"
    "codecov"
    "ccache"
    "nolibcumltest"
    "configure-only"
    "static-treelite"
    "build-metrics"
    "incl-cache-stats"
)

# Short arguments
ARGUMENT_LIST=(
    "v"
    "g"
    "n"
)

# read arguments
opts=$(getopt \
    --longoptions "$(printf "%s," "${LONG_ARGUMENT_LIST[@]}")" \
    --name "$(basename "$0")" \
    --options "$(printf "%s" "${ARGUMENT_LIST[@]}")" \
    -- "$@"
)

# shellcheck disable=SC2181
if [ $? != 0 ] ; then echo "Terminating..." >&2 ; exit 1 ; fi

eval set -- "$opts"

while true; do
    case "$1" in
        -h)
            show_help
            exit 0
            ;;
        -v | --verbose )
            VERBOSE_FLAG="-v"
            CMAKE_LOG_LEVEL=VERBOSE
            ;;
        -g | --debug )
            BUILD_TYPE=RelWithDebInfo
            ;;
        -n | --no-install )
            INSTALL_TARGET=""
            ;;
        --allgpuarch )
            BUILD_ALL_GPU_ARCH=1
            ;;
        --singlegpu )
            CUML_EXTRA_PYTHON_ARGS+=("--singlegpu")
            SINGLEGPU_CPP_FLAG=ON
            ;;
        --nvtx )
            NVTX=ON
            ;;
        --show_depr_warn )
            BUILD_DISABLE_DEPRECATION_WARNINGS=OFF
            ;;
        --codecov )
            CUML_EXTRA_PYTHON_ARGS+=("--linetrace=1" "--profile")
            ;;
        --ccache )
            CCACHE=ON
            ;;
        --nolibcumltest )
            BUILD_CUML_TESTS=OFF
            ;;
        --static-treelite )
            BUILD_STATIC_TREELITE=ON
            ;;
        --build-metrics )
            BUILD_REPORT_METRICS=ON
            ;;
        --incl-cache-stats )
            BUILD_REPORT_INCL_CACHE_STATS=ON
            ;;
        --)
            shift
            break
            ;;
    esac
    shift
done

# If clean given, run it prior to any other steps
if (( CLEAN == 1 )); then
    # If the dirs to clean are mounted dirs in a container, the
    # contents should be removed but the mounted dirs will remain.
    # The find removes all contents but leaves the dirs, the rmdir
    # attempts to remove the dirs but can fail safely.
    for bd in ${BUILD_DIRS}; do
        if [ -d "${bd}" ]; then
            find "${bd}" -mindepth 1 -delete
            rmdir "${bd}" || true
        fi
    done

    # Clean up python artifacts
    find "${REPODIR}"/python/ | grep -E "(__pycache__|\.pyc|\.pyo|\.so|\_skbuild)$"  | xargs rm -rf

    # Remove Doxyfile
    rm -rf "${REPODIR}"/cpp/Doxyfile

    # Remove .benchmark dirs and .pytest_cache
    find "${REPODIR}"/ | grep -E "(\.pytest_cache|\.benchmarks)$"  | xargs rm -rf
fi


################################################################################
# Configure for building all C++ targets
if completeBuild || hasArg libcuml || hasArg prims || hasArg bench || hasArg prims-bench || hasArg cppdocs || hasArg cpp-mgtests; then
    if (( BUILD_ALL_GPU_ARCH == 0 )); then
        CUML_CMAKE_CUDA_ARCHITECTURES="NATIVE"
        echo "Building for the architecture of the GPU in the system..."
    else
        CUML_CMAKE_CUDA_ARCHITECTURES="RAPIDS"
        echo "Building for *ALL* supported GPU architectures..."
    fi

    cmake -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
          -DCMAKE_CUDA_ARCHITECTURES=${CUML_CMAKE_CUDA_ARCHITECTURES} \
          -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
          -DSINGLEGPU=${SINGLEGPU_CPP_FLAG} \
          -DCUML_ALGORITHMS="ALL" \
          -DBUILD_CUML_TESTS=${BUILD_CUML_TESTS} \
          -DBUILD_CUML_MG_TESTS=${BUILD_CUML_MG_TESTS} \
          -DCUML_USE_TREELITE_STATIC=${BUILD_STATIC_TREELITE} \
          -DNVTX=${NVTX} \
          -DUSE_CCACHE=${CCACHE} \
          -DDISABLE_DEPRECATION_WARNINGS=${BUILD_DISABLE_DEPRECATION_WARNINGS} \
          -DCMAKE_PREFIX_PATH="${INSTALL_PREFIX}" \
          -DCMAKE_MESSAGE_LOG_LEVEL=${CMAKE_LOG_LEVEL} \
          "${CUML_EXTRA_CMAKE_ARGS[@]}" \
          -S "${REPODIR}"/cpp \
          -B "${LIBCUML_BUILD_DIR}"
fi

# If `./build.sh cuml` is called, don't build C/C++ components
if (! hasArg --configure-only) && (completeBuild || hasArg libcuml || hasArg prims || hasArg bench || hasArg cpp-mgtests); then
    # get the current count before the compile starts
    CACHE_TOOL=${CACHE_TOOL:-sccache}
    if [[ "$BUILD_REPORT_INCL_CACHE_STATS" == "ON" && -x "$(command -v "${CACHE_TOOL}")" ]]; then
        "${CACHE_TOOL}" --zero-stats
    fi

    if [ -n "${INSTALL_TARGET}" ]; then
      cmake --build "${LIBCUML_BUILD_DIR}" -j"${PARALLEL_LEVEL}" --target ${INSTALL_TARGET} "${VERBOSE_FLAG}"
    else
      cmake --build "${LIBCUML_BUILD_DIR}" -j"${PARALLEL_LEVEL}" "${VERBOSE_FLAG}"
    fi

    if [[ "$BUILD_REPORT_METRICS" == "ON" && -f "${LIBCUML_BUILD_DIR}/.ninja_log" ]]; then
      if ! rapids-build-metrics-reporter.py 2> /dev/null && [ ! -f rapids-build-metrics-reporter.py ]; then
          echo "Downloading rapids-build-metrics-reporter.py"
          curl -sO https://raw.githubusercontent.com/rapidsai/build-metrics-reporter/v1/rapids-build-metrics-reporter.py
      fi

      echo "Formatting build metrics"
      MSG=""
      # get some sccache/ccache stats after the compile
      if [[ "$BUILD_REPORT_INCL_CACHE_STATS" == "ON" ]]; then
          if [[ ${CACHE_TOOL} == "sccache" && -x "$(command -v sccache)" ]]; then
              COMPILE_REQUESTS=$(sccache -s | grep "Compile requests \+ [0-9]\+$" | awk '{ print $NF }')
              CACHE_HITS=$(sccache -s | grep "Cache hits \+ [0-9]\+$" | awk '{ print $NF }')
              HIT_RATE=$(COMPILE_REQUESTS="${COMPILE_REQUESTS}" CACHE_HITS="${CACHE_HITS}" python3 -c "import os; print(f'{int(os.getenv(\"CACHE_HITS\")) / int(os.getenv(\"COMPILE_REQUESTS\")):.2f}' if int(os.getenv(\"COMPILE_REQUESTS\")) else 'nan')")
              MSG="${MSG}<br/>cache hit rate ${HIT_RATE} %"
          elif [[ ${CACHE_TOOL} == "ccache" && -x "$(command -v ccache)" ]]; then
              CACHE_STATS_LINE=$(ccache -s | grep "Hits: \+ [0-9]\+ / [0-9]\+" | tail -n1)
              if [[ -n "$CACHE_STATS_LINE" ]]; then
                  CACHE_HITS=$(echo "$CACHE_STATS_LINE" - | awk '{ print $2 }')
                  COMPILE_REQUESTS=$(echo "$CACHE_STATS_LINE" - | awk '{ print $4 }')
                  HIT_RATE=$(COMPILE_REQUESTS="${COMPILE_REQUESTS}" CACHE_HITS="${CACHE_HITS}" python3 -c "import os; print(f'{int(os.getenv(\"CACHE_HITS\")) / int(os.getenv(\"COMPILE_REQUESTS\")):.2f}' if int(os.getenv(\"COMPILE_REQUESTS\")) else 'nan')")
                  MSG="${MSG}<br/>cache hit rate ${HIT_RATE} %"
              fi
          fi
        fi
        MSG="${MSG}<br/>parallel setting: $PARALLEL_LEVEL"
        if [[ -f "${LIBCUML_BUILD_DIR}/libcuml.so" ]]; then
            LIBCUML_FS=$(find "${LIBCUML_BUILD_DIR}" -name libcuml.so -printf '%s'| awk '{printf "%.2f MB", $1/1024/1024}')
            MSG="${MSG}<br/>libcuml.so size: $LIBCUML_FS"
        fi
        BMR_DIR=${RAPIDS_ARTIFACTS_DIR:-"${LIBCUML_BUILD_DIR}"}
        echo "The HTML report can be found at [${BMR_DIR}/ninja_log.html]. In CI, this report"
        echo "will also be uploaded to the appropriate subdirectory of https://downloads.rapids.ai/ci/cuml/, and"
        echo "the entire URL can be found in \"conda-cpp-build\" runs under the task \"Upload additional artifacts\""
        echo "Metrics output dir: [$BMR_DIR]"
        mkdir -p "${BMR_DIR}"
        MSG_OUTFILE="$(mktemp)"
        echo "$MSG" > "${MSG_OUTFILE}"
        PATH=".:$PATH" python rapids-build-metrics-reporter.py "${LIBCUML_BUILD_DIR}"/.ninja_log --fmt html --msg "${MSG_OUTFILE}" > "${BMR_DIR}"/ninja_log.html
        cp "${LIBCUML_BUILD_DIR}"/.ninja_log "${BMR_DIR}"/ninja.log
      fi
fi

if (! hasArg --configure-only) && hasArg cppdocs; then
    cmake --build "${LIBCUML_BUILD_DIR}" --target docs_cuml
fi


# Build and (optionally) install the cuml Python package
if (! hasArg --configure-only) && (completeBuild || hasArg cuml || hasArg pydocs); then
    # If `RAPIDS_PY_VERSION` is set, use that as the lower-bound for the stable ABI CPython version
    if [ -n "${RAPIDS_PY_VERSION}" ]; then
        RAPIDS_PY_API="cp${RAPIDS_PY_VERSION//./}"
        PYTHON_ARGS_FOR_INSTALL+=("--config-settings" "skbuild.wheel.py-api=${RAPIDS_PY_API}")
    fi

    # Replace spaces with semicolons in SKBUILD_EXTRA_CMAKE_ARGS
    SKBUILD_EXTRA_CMAKE_ARGS=${SKBUILD_EXTRA_CMAKE_ARGS// /;}

    SKBUILD_CMAKE_ARGS="-DCMAKE_MESSAGE_LOG_LEVEL=${CMAKE_LOG_LEVEL};${SKBUILD_EXTRA_CMAKE_ARGS}" \
        python -m pip install "${PYTHON_ARGS_FOR_INSTALL[@]}" "${CUML_EXTRA_PYTHON_ARGS[@]}" "${REPODIR}"/python/cuml

    if hasArg pydocs; then
        cd "${REPODIR}"/docs
        make html
    fi
fi
