# cuML C++

This folder contains the C++ and CUDA code of the algorithms and ML primitives of cuML. The build system uses CMake for build configuration, and an out-of-source build is recommended.

## Source Code Folders

The source code of cuML is divided mainly into `src` and `src_prims`.

- `src` contains the source code of the Machine Learning algorithms, and the main cuML C++ API. The main consumable is the shared library `libcuml`, that can be used stand alone by C++ consumers or is consumed by our Python package `cuml` to provide a Python API.
- `src_prims` contains most of the common components and computational primitives that form part of the machine learning algorithms in cuML, and can be used individually as well in the form of a header only library.

Multi-GPU communication is provided through RAFT communicator APIs; cuML does not build separate `std` or `mpi` communicator libraries. The `tests` directory contains single-GPU, multi-GPU, and primitive tests.

## Setup
### Dependencies

1. cmake (>= 3.26.4)
2. CUDA (>= 12.2)
3. gcc (>=13.0)
4. clang-format (= 20.1.8) - enforces uniform C++ coding style; required to build cuML from source. The packages `clang=20` and `clang-tools=20` from the conda-forge channel should be sufficient, if you are on conda. If not using conda, install the right version using your OS package manager.

### Building cuML:

The main artifact produced by the build system is the shared library `libcuml`. Additionally, executables to run tests for the algorithms can be built. To see detailed steps see the [BUILD](../BUILD.md) document of the repository.

Current cmake offers the following configuration options:

- Build Configuration Options:

| Flag | Possible Values | Default Value | Behavior |
| --- | --- | --- | --- |
| BUILD_CUML_CPP_LIBRARY | [ON, OFF]  | ON  | Enable/disable building the `libcuml` shared library. Setting this variable to `OFF` also forces `BUILD_CUML_TESTS`, `BUILD_CUML_MG_TESTS`, `BUILD_CUML_EXAMPLES`, `BUILD_PRIMS_TESTS`, and `BUILD_CUML_BENCH` to `OFF` |
| BUILD_CUML_TESTS | [ON, OFF]  | ON  |  Enable/disable building cuML single-GPU C++ test targets.  |
| BUILD_CUML_MG_TESTS | [ON, OFF]  | OFF | Enable/disable building cuML multi-GPU C++ test targets. Requires MPI and RAFT distributed dependencies. See section about additional requirements. |
| BUILD_PRIMS_TESTS | [ON, OFF]  | ON  | Enable/disable building cuML primitive C++ test targets.  |
| BUILD_CUML_EXAMPLES | [ON, OFF]  | ON  | Enable/disable building cuML C++ API usage examples.  |
| BUILD_CUML_BENCH | [ON, OFF]  | ON  | Enable/disable building of cuML C++ benchmark. |
| SINGLEGPU | [ON, OFF] | OFF | Disable cuML MNMG C++ sources and tests, and build cuVS without multi-GPU algorithms. Forces `BUILD_CUML_MG_TESTS` to `OFF`. |
| DISABLE_OPENMP | [ON, OFF]  | OFF  | Set to `ON` to disable OpenMP  |
| CMAKE_CUDA_ARCHITECTURES |  List of GPU architectures, semicolon-separated | Empty  | List the GPU architectures to compile the GPU targets for. Set to "NATIVE" to auto detect GPU architecture of the system, set to "ALL" to compile for all supported archs. |
| USE_CCACHE | [ON, OFF]  | ON  | Cache build artifacts with ccache. |

- Debug configuration options:

| Flag | Possible Values | Default Value | Behavior |
| --- | --- | --- | --- |
| KERNEL_INFO | [ON, OFF]  | OFF  | Enable/disable kernel resource usage info in nvcc. |
| LINE_INFO | [ON, OFF]  | OFF  | Enable/disable lineinfo in nvcc.  |
| NVTX | [ON, OFF]  | OFF  | Enable/disable nvtx markers in libcuml.  |

After running CMake in a `build` directory, if the `BUILD_*` options were not turned `OFF`, the following targets can be built:

```bash
$ cmake --build . -j                        # Build libcuml and enabled C++ test targets
$ cmake --build . -j --target  sg_benchmark # Build C++ cuML single-GPU benchmark
$ cmake --build . -j --target  cuml         # Build libcuml

# Test executables are generated as individual CTest targets with SG_, MG_, or PRIMS_ prefixes.
```

### MultiGPU Tests Requirements Note:

To build the MultiGPU tests (CMake option `BUILD_CUML_MG_TESTS`), the following dependencies are required:

- MPI (OpenMPI recommended)
- RAFT distributed dependencies, including NCCL and UCXX/UCX. See RAFT's build documentation for the current requirements.

### Third Party Modules

The external folder contains submodules that cuML depends on.

Current external submodules are:

1. [CUB](https://github.com/NVlabs/cub)
2. [Faiss](https://github.com/facebookresearch/faiss)
3. [Google Test](https://github.com/google/googletest)

## Using cuML libraries

After building cuML, you can use its functionality in other C++ applications by
linking against the generated libraries, or from Python via the `cuml` package.
The following trivial example shows
how to make external use of cuML's logger:

```cpp
// main.cpp
#include <cuml/common/logger.hpp>

int main(int argc, char *argv[]) {
  CUML_LOG_WARN("This is a warning from the cuML logger!");
  return 0;
}
```

To compile this example, we must point the compiler to where cuML was
installed. Assuming you did not provide a custom `$CMAKE_INSTALL_PREFIX`, this
will default to the `$CONDA_PREFIX` environment variable.

```bash
$ export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib"
$ nvcc \
       main.cpp \
       -o cuml_logger_example \
       "-L${CONDA_PREFIX}/lib" \
       "-I${CONDA_PREFIX}/include" \
       "-I${CONDA_PREFIX}/include/cuml/raft" \
       -lcuml
$ ./cuml_logger_example
[W] [13:26:43.503068] This is a warning from the cuML logger!
```
