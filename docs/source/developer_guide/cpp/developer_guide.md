# cuML developer guide
This document summarizes rules and best practices for contributions to the cuML C++ component of NVIDIA/cuml. This is a living document and contributions for clarifications or fixes and issue reports are highly welcome.

## General
Please start by reading [`CONTRIBUTING.md`](https://github.com/rapidsai/cuml/blob/main/CONTRIBUTING.md).

## Performance
1. In performance critical sections of the code, favor `cudaDeviceGetAttribute` over `cudaDeviceGetProperties`. See corresponding CUDA devblog [here](https://devblogs.nvidia.com/cuda-pro-tip-the-fast-way-to-query-device-properties/) to know more.
2. If an algorithm requires multiple CUDA streams, do not create a separate `raft::handle_t` for each stream. Use the streams exposed by the handle and keep stream ordering explicit. See [CUDA Resources](#cuda-resources) and [Asynchronous operations and stream ordering](#asynchronous-operations-and-stream-ordering).

## Threading Model

With the exception of the raft::handle_t, cuML algorithms should maintain thread-safety and are, in general,
assumed to be single threaded. This means they should be able to be called from multiple host threads so
long as different instances of `raft::handle_t` are used.

Exceptions are made for algorithms that can take advantage of multiple CUDA streams within multiple host threads
in order to oversubscribe or increase occupancy on a single GPU. In these cases, the use of multiple host
threads within cuML algorithms should be used only to maintain concurrency of the underlying CUDA streams.
Multiple host threads should be used sparingly, be bounded, and should steer clear of performing CPU-intensive
computations.

A good example of an acceptable use of host threads within a cuML algorithm might look like the following

```
handle.sync_stream();

int n_streams = handle.get_num_internal_streams();

#pragma omp parallel for num_threads(n_threads)
for(int i = 0; i < n; i++) {
    int thread_num = omp_get_thread_num() % n_threads;
    cudaStream_t s = handle.get_stream_from_stream_pool(thread_num);
    ... possible light cpu pre-processing ...
    my_kernel1<<<b, tpb, 0, s>>>(...);
    ...
    ... some possible async d2h / h2d copies ...
    my_kernel2<<<b, tpb, 0, s>>>(...);
    ...
    handle.sync_stream(s);
    ... possible light cpu post-processing ...
}
```

In the example above, if there is no CPU pre-processing at the beginning of the for-loop, an event can be registered in
each of the streams within the for-loop to make them wait on the stream from the handle. If there is no CPU post-processing
at the end of each for-loop iteration, `handle.sync_stream(s)` can be replaced with a single `handle.sync_stream_pool()`
after the for-loop.

To avoid compatibility issues between different threading models, the only threading programming allowed in cuML is OpenMP.
Though cuML's build enables OpenMP by default, cuML algorithms should still function properly even when OpenMP has been
disabled. If the CPU pre- and post-processing were not needed in the example above, OpenMP would not be needed.

The use of threads in third-party libraries is allowed, though they should still avoid depending on a specific OpenMP runtime.

## Public cuML interface
### Terminology
We have the following supported APIs:
1. Core cuML interface aka stateless C++ API aka C++ API aka `libcuml.so`

### Motivation
The cuML C++ API is stateless so that algorithm state (models, hyper-parameters, and similar data) can be serialized in a straightforward way, which supports features such as pickling in the Python layer, and so that a small, explicit surface is presented to the bindings above this library.

This section lays out guidelines for managing state along the API of cuML.

### General guideline
As mentioned before, functions exposed via the C++ API must be stateless. Things that are OK to be exposed on the interface:
1. Any [POD](https://en.wikipedia.org/wiki/Passive_data_structure) - see [std::is_pod](https://en.cppreference.com/w/cpp/types/is_pod) as a reference for C++11  POD types.
2. `raft::handle_t` - since it stores GPU-related state which has nothing to do with the model/algo state.
3. Pointers to POD types (explicitly putting it out, even though it can be considered as a POD).
Internal to the C++ API, these stateless functions are free to use their own temporary classes, as long as they are not exposed on the interface.

### Stateless C++ API
Using the Decision Tree Classifier algorithm as an example, the following way of exposing its API would be wrong according to the guidelines in this section, since it exposes a non-POD C++ class object in the C++ API:
```cpp
template <typename T>
class DecisionTreeClassifier {
  TreeNode<T>* root;
  DTParams params;
  const raft::handle_t &handle;
public:
  DecisionTreeClassifier(const raft::handle_t &handle, DTParams& params, bool verbose=false);
  void fit(const T *input, int n_rows, int n_cols, const int *labels);
  void predict(const T *input, int n_rows, int n_cols, int *predictions);
};

void decisionTreeClassifierFit(const raft::handle_t &handle, const float *input, int n_rows, int n_cols,
                               const int *labels, DecisionTreeClassifier<float> *model, DTParams params,
                               bool verbose=false);
void decisionTreeClassifierPredict(const raft::handle_t &handle, const float* input,
                                   DecisionTreeClassifier<float> *model, int n_rows,
                                   int n_cols, int* predictions, bool verbose=false);
```

An alternative correct way to expose this could be:
```cpp
// NOTE: this example assumes that TreeNode and DTParams are the model/state that need to be stored
// and passed between fit and predict methods
template <typename T> struct TreeNode { /* nested tree-like data structure, but written as a POD! */ };
struct DTParams { /* hyper-params for building DT */ };
typedef TreeNode<float> TreeNodeF;
typedef TreeNode<double> TreeNodeD;

void decisionTreeClassifierFit(const raft::handle_t &handle, const float *input, int n_rows, int n_cols,
                               const int *labels, TreeNodeF *&root, DTParams params,
                               bool verbose=false);
void decisionTreeClassifierPredict(const raft::handle_t &handle, const double* input, int n_rows,
                                   int n_cols, const TreeNodeD *root, int* predictions,
                                   bool verbose=false);
```
The above example understates the complexity involved with exposing a tree-like data structure across the interface! However, this example should be simple enough to drive the point across.

### Other functions on state
These guidelines also mean that it is the responsibility of C++ API to expose methods to load and store (aka marshalling) such a data structure. Further continuing the Decision Tree Classifier example,  the following methods could achieve this:
```cpp
void storeTree(const TreeNodeF *root, std::ostream &os);
void storeTree(const TreeNodeD *root, std::ostream &os);
void loadTree(TreeNodeF *&root, std::istream &is);
void loadTree(TreeNodeD *&root, std::istream &is);
```
It is also worth noting that for algorithms such as the members of GLM, where models consist of an array of weights and are therefore easy to manipulate directly by the users, such custom load/store methods might not be explicitly needed.

### File naming convention
1. An ML algorithm `<algo>` is to be contained inside the folder named `src/<algo>`.
2. `<algo>.hpp` and `<algo>.[cpp|cu]` contain C++ API declarations and definitions respectively.

## Coding style

## Code format
### Introduction
cuML relies on `clang-format` to enforce code style across all C++ and CUDA source code. The coding style is based on the [Google style guide](https://google.github.io/styleguide/cppguide.html#Formatting). The only digressions from this style are the following.
1. Do not split empty functions/records/namespaces.
2. Two-space indentation everywhere, including the line continuations.
3. Disable reflowing of comments.
The reasons behind these deviations from the Google style guide are given in comments [here](https://github.com/rapidsai/cuml/blob/main/cpp/.clang-format).

### How is the check done?
Formatting is checked by CI and by the repository's pre-commit configuration.
Run `pre-commit run --all-files clang-format` from the repository root, or
format only changed files with the corresponding `--files` option.

### clang-format version?
Use the clang-format version specified by the C++ build dependencies (currently `20.1.8`). See the [C++ build dependencies](https://github.com/rapidsai/cuml/blob/main/cpp/README.md#dependencies) for the current requirement.

### Additional scripts
Along with clang, there are are the include checker and copyright checker scripts for checking style, which can be performed as part of CI, as well as manually.

#### #include style
[include_checker.py](https://github.com/rapidsai/cuml/blob/main/cpp/scripts/include_checker.py) is used to enforce the include style as follows:
1. `#include "..."` should be used for referencing local files only. It is acceptable to be used for referencing files in a sub-folder/parent-folder of the same algorithm, but should never be used to include files in other algorithms or between algorithms and the primitives or other dependencies.
2. `#include <...>` should be used for referencing everything else

Manually, run the following to bulk-fix include style issues:
```bash
python ./cpp/scripts/include_checker.py --inplace [cpp/include cpp/src cpp/src_prims cpp/tests ... list of folders which you want to fix]
```

#### Copyright header
RAPIDS [pre-commit-hooks](https://github.com/rapidsai/pre-commit-hooks) checks the Copyright
header for all git-modified files.

Manually, you can run the following to bulk-fix the header on all files in the repository:
```bash
pre-commit run -a verify-copyright
```
Keep in mind that this only applies to files tracked by git that have been modified.

## Error handling
Call CUDA APIs via the provided helper macros `RAFT_CUDA_TRY`, `RAFT_CUBLAS_TRY` and `RAFT_CUSOLVER_TRY`. These macros take care of checking the return values of the used API calls and generate an exception when the command is not successful. If you need to avoid an exception, e.g. inside a destructor, use `RAFT_CUDA_TRY_NO_THROW`, `RAFT_CUBLAS_TRY_NO_THROW ` and `RAFT_CUSOLVER_TRY_NO_THROW ` (currently not available, see https://github.com/NVIDIA/cuml/issues/229). These macros log the error but do not throw an exception.

## Logging
### Introduction
Anything and everything about logging is defined inside [logger.hpp](https://github.com/rapidsai/cuml/blob/main/cpp/include/cuml/common/logger.hpp).

### Usage
```cpp
#include <cuml/common/logger.hpp>

// Inside your method or function, use any of these macros
CUML_LOG_TRACE("Hello %s!", "world");
CUML_LOG_DEBUG("Hello %s!", "world");
CUML_LOG_INFO("Hello %s!", "world");
CUML_LOG_WARN("Hello %s!", "world");
CUML_LOG_ERROR("Hello %s!", "world");
CUML_LOG_CRITICAL("Hello %s!", "world");
```

### Changing logging level and pattern
The global logger is available through `ML::default_logger()` and uses the
rapids logger API. For example:
```cpp
ML::default_logger().set_level(rapids_logger::level_enum::warn);
ML::default_logger().set_pattern("[%l] %v");
```

### Tips
* Do NOT end your logging messages with a newline! It is automatically added by spdlog.
* The `CUML_LOG_TRACE()` is by default not compiled due to the `CUML_ACTIVE_LEVEL` macro setup, for performance reasons. If you need it to be enabled, change this macro accordingly during compilation time

## Documentation
All external interfaces need to have a complete [doxygen](http://www.doxygen.nl) API documentation. This is also recommended for internal interfaces.

## Testing and unit testing
Add or update focused C++ tests under `cpp/tests` with implementation changes.
Follow the repository's normal build and test instructions in the
[C++ README](https://github.com/rapidsai/cuml/blob/main/cpp/README.md), and
run the relevant test target locally before submitting a change.

## Device and host memory allocations
Use the current RAFT and RMM resource interfaces for temporary allocations;
the older `ML::deviceAllocator`, `MLCommon::*_buffer`, and allocator-adapter
examples formerly documented here are no longer current cuML APIs. In code
that owns a `raft::handle_t`, obtain the relevant resource from the handle and
use RMM containers such as `rmm::device_uvector` or `rmm::device_buffer` with
the appropriate stream. Follow nearby current implementations and include
checked arithmetic for allocation sizes.
```cpp
#include <cstddef>
#include <rmm/device_uvector.hpp>

template<typename T>
void foo(const raft::handle_t& h, std::size_t n)
{
    rmm::device_uvector<T> temporary(n, h.get_stream());
}
```
## Asynchronous operations and stream ordering
All ML algorithms should be as asynchronous as possible avoiding the use of the default stream (aka as NULL or `0` stream). Implementations that require only one CUDA Stream should use the stream from `raft::handle_t`:
```cpp
void foo(const raft::handle_t& h, ...)
{
    cudaStream_t stream = h.get_stream();
}
```
When multiple streams are needed, e.g. to manage a pipeline, use the internal streams available in `raft::handle_t` (see [CUDA Resources](#cuda-resources)). If multiple streams are used all operations still must be ordered according to `raft::handle_t::get_stream()`. Before any operation in any of the internal CUDA streams is started, all previous work in `raft::handle_t::get_stream()` must have completed. Any work enqueued in `raft::handle_t::get_stream()` after a cuML function returns should not start before all work enqueued in the internal streams has completed. E.g. if a cuML algorithm is called like this:
```cpp
void foo(const double* const srcdata, double* const result)
{
    cudaStream_t stream;
    CUDA_RT_CALL( cudaStreamCreate( &stream ) );
    raft::handle_t raftHandle( stream );

    ...

    RAFT_CUDA_TRY( cudaMemcpyAsync( srcdata, h_srcdata.data(), n*sizeof(double), cudaMemcpyHostToDevice, stream ) );

    ML::algo(raft::handle_t, dopredict, srcdata, result, ... );

    RAFT_CUDA_TRY( cudaMemcpyAsync( h_result.data(), result, m*sizeof(int), cudaMemcpyDeviceToHost, stream ) );

    ...
}
```
No work in any stream should start in `ML::algo` before the `cudaMemcpyAsync` in `stream` launched before the call to `ML::algo` is done. And all work in all streams used in `ML::algo` should be done before the `cudaMemcpyAsync` in `stream` launched after the call to `ML::algo` starts.

This can be ensured by introducing interstream dependencies with CUDA events and `cudaStreamWaitEvent`. For convenience, the header `raft/core/handle.hpp` provides the class `raft::stream_syncer` which lets all `raft::handle_t` internal CUDA streams wait on `raft::handle_t::get_stream()` in its constructor and in its destructor and lets `raft::handle_t::get_stream()` wait on all work enqueued in the `raft::handle_t` internal CUDA streams. The intended use would be to create a `raft::stream_syncer` object as the first thing in a entry function of the public cuML API:

```cpp
void cumlAlgo(const raft::handle_t& handle, ...)
{
    raft::stream_syncer _(handle);
}
```
This ensures the stream ordering behavior described above.

### Using Thrust
Use a Thrust execution policy bound to the intended CUDA stream when calling
Thrust algorithms.

## CUDA Resources

Do not create reusable CUDA resources directly in implementations of ML algorithms. Instead, use the existing resources in `raft::handle_t` to avoid constant creation and deletion of reusable resources such as CUDA streams, CUDA events or library handles. Please file a feature request if a resource handle is missing in `raft::handle_t`.
The resources can be obtained like this
```cpp
void foo(const raft::handle_t& h, ...)
{
    cublasHandle_t cublasHandle = h.get_cublas_handle();
    const int num_streams       = h.get_num_internal_streams();
    const int stream_idx        = ...
    cudaStream_t stream         = h.get_internal_stream(stream_idx);
    ...
}
```

The example below shows one way to create `nStreams` number of internal cuda streams which can later be used by the algos inside cuML. For a full working example of how to use internal streams to schedule work on a single GPU, the reader is further referred to [this PR](https://github.com/NVIDIA/cuml/pull/1015). In this PR, the internal streams inside `raft::handle_t` are used to schedule more work onto a GPU for Random Forest building.
```cpp
int main(int argc, char** argv)
{
    int nStreams = argc > 1 ? atoi(argv[1]) : 0;
    raft::handle_t handle(nStreams);
    foo(handle, ...);
}
```

## Multi-GPU

The multi GPU paradigm of cuML is **O**ne **P**rocess per **G**PU (OPG). Each algorithm should be implemented in a way that it can run with a single GPU without any specific dependencies to a particular communication library. A multi-GPU implementation should use the methods offered by the class `raft::comms::comms_t` from [`raft/core/comms.hpp`](https://github.com/rapidsai/raft/blob/main/cpp/include/raft/core/comms.hpp) for inter-rank/GPU communication. It is the responsibility of the user of cuML to create an initialized instance of `raft::comms::comms_t`.

E.g. with a CUDA-aware MPI, a cuML user could use code like this to inject an initialized instance of `raft::comms::mpi_comms` into a `raft::handle_t`:

```cpp
#include <mpi.h>
#include <raft/core/handle.hpp>
#include <raft/comms/mpi_comms.hpp>
#include <mlalgo/mlalgo.hpp>
...
int main(int argc, char * argv[])
{
    MPI_Init(&argc, &argv);
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int local_rank = -1;
    {
        MPI_Comm local_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);

        MPI_Comm_rank(local_comm, &local_rank);

        MPI_Comm_free(&local_comm);
    }

    cudaSetDevice(local_rank);

    mpi_comms raft_mpi_comms;
    MPI_Comm_dup(MPI_COMM_WORLD, &raft_mpi_comms);

    {
        raft::handle_t raftHandle;
        initialize_mpi_comms(raftHandle, raft_mpi_comms);

        ...

        ML::mlalgo(raftHandle, ... );
    }

    MPI_Comm_free(&raft_mpi_comms);

    MPI_Finalize();
    return 0;
}
```

A cuML developer can assume the following:
 * A instance of `raft::comms::comms_t` was correctly initialized.
 * All processes that are part of `raft::comms::comms_t` call into the ML algorithm cooperatively.

The initialized instance of `raft::comms::comms_t` can be accessed from the `raft::handle_t` instance:

```cpp
void foo(const raft::handle_t& h, ...)
{
    const raft::comms::comms_t& communicator = h.get_comms();
    const int rank = communicator.get_rank();
    const int size = communicator.get_size();
    ...
}
```
