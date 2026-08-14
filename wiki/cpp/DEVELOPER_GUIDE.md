# cuML developer guide
This document summarizes rules and best practices for contributions to the cuML C++ component of NVIDIA/cuml. This is a living document and contributions for clarifications or fixes and issue reports are highly welcome.

## General
Please start by reading [CONTRIBUTING.md](../../CONTRIBUTING.md).

## Performance
1. In performance critical sections of the code, favor `cudaDeviceGetAttribute` over `cudaDeviceGetProperties`. See corresponding CUDA devblog [here](https://devblogs.nvidia.com/cuda-pro-tip-the-fast-way-to-query-device-properties/) to know more.
2. If an algo requires you to launch GPU work in multiple cuda streams, do not create multiple `raft::handle_t` objects, one for each such work stream. Instead, expose a `n_streams` parameter in that algo's cuML C++ interface and then rely on `raft::handle_t::get_internal_stream()` to pick up the right cuda stream. Refer to the section on [CUDA Resources](#cuda-resources) and the section on [Threading](#TBD) for more details. TIP: use `raft::handle_t::get_num_internal_streams` to know how many such streams are available at your disposal.

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
2. Stateful convenience C++ API - wrapper around core API (WIP)

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

### Stateful C++ API
This scikit-learn-esq C++ API should always be a wrapper around the stateless C++ API, NEVER the other way around. The design discussion about the right way to expose such a wrapper around `libcuml.so` is [still going on](https://github.com/NVIDIA/cuml/issues/456)  So, stay tuned for more details.

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
The reasons behind these deviations from the Google style guide are given in comments [here](../../cpp/.clang-format).

### How is the check done?
All formatting checks are done by this python script: [run-clang-format.py](../../cpp/scripts/run-clang-format.py) which is effectively a wrapper over `clang-format`. An error is raised if the code diverges from the format suggested by clang-format. It is expected that the developers run this script to detect and fix formatting violations before creating PR.

#### As part of CI
[run-clang-format.py](../../cpp/scripts/run-clang-format.py) is executed as part of our CI tests. If there are any formatting violations, PR author is expected to fix those to get CI passing. Steps needed to fix the formatting violations are described in the subsequent sub-section.

#### Manually
Developers can also manually (or setup this command as part of git pre-commit hook) run this check by executing:
```bash
python ./cpp/scripts/run-clang-format.py
```
From the root of the cuML repository.

### How to know the formatting violations?
When there are formatting errors, [run-clang-format.py](../../cpp/scripts/run-clang-format.py) prints a `diff` command, showing where there are formatting differences. Unfortunately, unlike `flake8`, `clang-format` does NOT print descriptions of the violations, but instead directly formats the code. So, the only way currently to know about formatting differences is to run the diff command as suggested by this script against each violating source file.

### How to fix the formatting violations?
When there are formatting violations, [run-clang-format.py](../../cpp/scripts/run-clang-format.py) prints at the end, the exact command that can be run by developers to fix them. This is the easiest way to fix formatting errors. [This screencast](https://asciinema.org/a/287367) shows how developers can check for formatting violations in their branches and also how to fix those, before sending out PRs.

In short, to bulk-fix all the formatting violations, execute the following command:
```bash
python ./cpp/scripts/run-clang-format.py -inplace
```
From the root of the cuML repository.

### clang-format version?
To avoid spurious code style violations we specify the exact clang-format version required, currently `8.0.0`. This is enforced by the [run-clang-format.py](../../cpp/scripts/run-clang-format.py) script itself. Refer [here](../../cpp/README.md#dependencies) for the list of build-time dependencies.

### Additional scripts
Along with clang, there are are the include checker and copyright checker scripts for checking style, which can be performed as part of CI, as well as manually.

#### #include style
[include_checker.py](../../cpp/scripts/include_checker.py) is used to enforce the include style as follows:
1. `#include "..."` should be used for referencing local files only. It is acceptable to be used for referencing files in a sub-folder/parent-folder of the same algorithm, but should never be used to include files in other algorithms or between algorithms and the primitives or other dependencies.
2. `#include <...>` should be used for referencing everything else

Manually, run the following to bulk-fix include style issues:
```bash
python ./cpp/scripts/include_checker.py --inplace [cpp/include cpp/src cpp/src_prims cpp/test ... list of folders which you want to fix]
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
Anything and everything about logging is defined inside [logger.hpp](../../cpp/include/cuml/common/logger.hpp). It uses [spdlog](https://github.com/gabime/spdlog) underneath, but this information is transparent to all.

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

### Changing logging level
There are 7 logging levels with each successive level becoming quieter:
1. CUML_LEVEL_TRACE
2. CUML_LEVEL_DEBUG
3. CUML_LEVEL_INFO
4. CUML_LEVEL_WARN
5. CUML_LEVEL_ERROR
6. CUML_LEVEL_CRITICAL
7. CUML_LEVEL_OFF
Pass one of these as per your needs into the `setLevel()` method as follows:
```cpp
ML::Logger::get.setLevel(CUML_LEVEL_WARN);
// From now onwards, this will print only WARN and above kind of messages
```

### Changing logging pattern
Pass the [format string](https://github.com/gabime/spdlog/wiki/3.-Custom-formatting) as follows in order use a different logging pattern than the default.
```cpp
ML::Logger::get.setPattern(YourFavoriteFormat);
```
One can also use the corresponding `getPattern()` method to know the current format as well.

### Temporarily changing the logging pattern
Sometimes, we need to temporarily change the log pattern (eg: for reporting decision tree structure). This can be achieved in a RAII-like approach as follows:
```cpp
{
  PatternSetter _(MyNewTempFormat);
  // new log format is in effect from here onwards
  doStuff();
  // once the above temporary object goes out-of-scope, the old format will be restored
}
```

### Tips
* Do NOT end your logging messages with a newline! It is automatically added by spdlog.
* The `CUML_LOG_TRACE()` is by default not compiled due to the `CUML_ACTIVE_LEVEL` macro setup, for performance reasons. If you need it to be enabled, change this macro accordingly during compilation time

## Documentation
All external interfaces need to have a complete [doxygen](http://www.doxygen.nl) API documentation. This is also recommended for internal interfaces.

## Testing and Unit Testing
TODO: Add this

## Device and Host memory allocations
To enable `libcuml` users to control how memory for temporary data is allocated, allocate device memory using the allocator provided:
```cpp
template<typename T>
void foo(const raft::handle_t& h, cudaStream_t stream, ... )
{
    T* temp_h = h.get_device_allocator()->allocate(n*sizeof(T), stream);
    ...
    h.get_device_allocator()->deallocate(temp_h, n*sizeof(T), stream);
}
```
The same rule applies to larger amounts of host heap memory:
```cpp
template<typename T>
void foo(const raft::handle_t& h, cudaStream_t stream, ... )
{
    T* temp_h = h.get_host_allocator()->allocate(n*sizeof(T), stream);
    ...
    h.get_host_allocator()->deallocate(temp_h, n*sizeof(T), stream);
}
```
Small host memory heap allocations, e.g. as internally done by STL containers, are fine, e.g. an `std::vector` managing only a handful of integers.
Both the Host and the Device Allocators might allow asynchronous stream ordered allocation and deallocation. This can provide significant performance benefits so a stream always needs to be specified when allocating or deallocating (see [Asynchronous operations and stream ordering](#asynchronous-operations-and-stream-ordering)). `ML::deviceAllocator` returns pinned device memory on the current device, while `ML::hostAllocator` returns host memory. A user of cuML can write customized allocators and pass them into cuML. If a cuML user does not provide custom allocators default allocators will be used. For `ML::deviceAllocator` the default is to use `cudaMalloc`/`cudaFree`. For `ML::hostAllocator` the default is to use `cudaMallocHost`/`cudaFreeHost`.
There are two simple container classes compatible with the allocator interface `MLCommon::device_buffer` available in `src_prims/common/device_buffer.hpp` and `MLCommon::host_buffer` available in `src_prims/common/host_buffer.hpp`. These allow to follow the [RAII idiom](https://en.wikipedia.org/wiki/Resource_acquisition_is_initialization) to avoid resources leaks and enable exception safe code. These containers also allow asynchronous allocation and deallocation using the `resize` and `release` member functions:
```cpp
template<typename T>
void foo(const raft::handle_t& h, ..., cudaStream_t stream )
{
    ...
    MLCommon::device_buffer<T> temp( h.get_device_allocator(), stream, 0 )

    temp.resize(n, stream);
    kernelA<<<grid, block, 0, stream>>>(..., temp.data(), ...);
    kernelB<<<grid, block, 0, stream>>>(..., temp.data(), ...);
    temp.release(stream);
}
```
The motivation for `MLCommon::host_buffer` and `MLCommon::device_buffer` over using `std::vector` or `thrust::device_vector` (which would require thrust 1.9.4 or later) is to enable exception safe asynchronous allocation and deallocation following stream semantics with an explicit interface while avoiding the overhead of implicitly initializing the underlying allocation.
To use `ML::hostAllocator` with a STL container the header `src/common/allocatorAdapter.hpp` provides `ML::stdAllocatorAdapter`:
```cpp
template<typename T>
void foo(const raft::handle_t& h, ..., cudaStream_t stream )
{
    ...
    std::vector<T,ML::stdAllocatorAdapter<T> > temp( n, val, ML::stdAllocatorAdapter<T>(h.get_host_allocator(), stream) )
    ...
}
```
If thrust 1.9.4 or later is available for use in cuML a similar allocator can be provided for `thrust::device_vector`.

### <a name="allocationsthrust"></a>Using Thrust
To ensure that thrust algorithms allocate temporary memory via the provided device memory allocator, use the `ML::thrustAllocatorAdapter` available in `src/common/allocatorAdapter.hpp` with the `thrust::cuda::par` execution policy:
```cpp
void foo(const raft::handle_t& h, ..., cudaStream_t stream )
{
    ML::thrustAllocatorAdapter alloc( h.get_device_allocator(), stream );
    auto execution_policy = thrust::cuda::par(alloc).on(stream);
    thrust::for_each(execution_policy, ... );
}
```
The header `src/common/allocatorAdapter.hpp` also provides a helper function to create an execution policy:
```cpp
void foo(const raft::handle_t& h, ... , cudaStream_t stream )
{
    auto execution_policy = ML::thrust_exec_policy(h.get_device_allocator(),stream);
    thrust::for_each(execution_policy->on(stream), ... );
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
    raft::streamSyncer _(handle);
}
```
This ensures the stream ordering behavior described above.

### Using Thrust
To ensure that thrust algorithms are executed in the intended stream the `thrust::cuda::par` execution policy should be used (see [Using Thrust](#allocationsthrust) in [Device and Host memory allocations](#device-and-host-memory-allocations)).

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

The multi GPU paradigm of cuML is **O**ne **P**rocess per **G**PU (OPG). Each algorithm should be implemented in a way that it can run with a single GPU without any specific dependencies to a particular communication library. A multi-GPU implementation should use the methods offered by the class `raft::comms::comms_t` from [raft/core/comms.hpp] for inter-rank/GPU communication. It is the responsibility of the user of cuML to create an initialized instance of `raft::comms::comms_t`.

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
    const MLCommon::cumlCommunicator& communicator = h.get_comms();
    const int rank = communicator.get_rank();
    const int size = communicator.get_size();
    ...
}
```
