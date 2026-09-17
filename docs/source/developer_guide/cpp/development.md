# C++ and CUDA Developer Guide

This guide summarizes current conventions for contributions to cuML's C++ and
CUDA implementation. Read the repository
[contribution guidelines](https://github.com/NVIDIA/cuml/blob/main/CONTRIBUTING.md)
before starting.

## Source and API layout

Installed libcuml headers live under
[`cpp/include/cuml`](https://github.com/NVIDIA/cuml/tree/main/cpp/include/cuml).
Algorithm implementations and internal headers live under
[`cpp/src`](https://github.com/NVIDIA/cuml/tree/main/cpp/src). Keep a public
declaration in the appropriate installed header and place implementation details
with the corresponding algorithm in `cpp/src`.

The C++ interfaces are used by cuML's bindings and by some direct libcuml
consumers. The C++ API currently has no backward-compatibility or deprecation
guarantee. Describe behavior and preconditions precisely, keep changes focused,
and clearly describe consequential API changes during review.

## Formatting and implementation style

The configured pre-commit hooks and
[`CONTRIBUTING.md`](https://github.com/NVIDIA/cuml/blob/main/CONTRIBUTING.md#code-formatting)
are the formatting authority. Install pre-commit and run the hooks on changed
files before opening a pull request:

```bash
pre-commit run --files cpp/include/cuml/example.hpp cpp/src/example.cu
```

Follow neighboring code and use existing RAFT primitives rather than creating
local alternatives. Factor generally reusable low-level operations into the
appropriate primitive layer rather than duplicating them inside individual
algorithms. Use the RAFT error-checking facilities appropriate to the CUDA
library call. Avoid unnecessary host/device transfers and synchronization. Keep
algorithm array inputs and outputs device-accessible; do not require host
staging unless the API contract requires it.

## Memory and streams

Use RMM RAII containers for temporary allocations, such as
`rmm::device_uvector`, `rmm::device_scalar`, and `rmm::host_uvector`. Construct
and use them with the operation's explicit stream so allocation, work, and
lifetime follow the same ordering. Do not introduce raw `cudaMalloc` ownership
when an RMM container expresses the lifetime.

A `raft::handle_t` is a RAFT resource container. Obtain the caller's stream from
it (for example, `handle.get_stream()` in handle-based code or
`raft::resource::get_cuda_stream(resources)` for `raft::resources`) and enqueue
work on that stream. Avoid the default CUDA stream and avoid synchronizing
unless the API contract requires host-visible completion.

When concurrency is useful, use the stream pool supplied by the RAFT resource
container rather than creating handles or reusable CUDA resources per stream.
Current handle-based code uses `get_stream_pool_size()` and
`get_stream_from_stream_pool(index)`. Preserve ordering between the caller's
primary stream and pool work, and do not assume that a pool exists or has a
particular size. Follow a nearby implementation using the same RAFT resource
type because RAFT resource APIs evolve.

Algorithms should be safe to invoke concurrently when each invocation has its
own resources and output storage. Shared process-wide state and unnecessary CPU
threading should be avoided.

## Logging

Include [`cuml/common/logger.hpp`](https://github.com/NVIDIA/cuml/blob/main/cpp/include/cuml/common/logger.hpp)
and use `CUML_LOG_TRACE`, `CUML_LOG_DEBUG`, `CUML_LOG_INFO`, `CUML_LOG_WARN`,
`CUML_LOG_ERROR`, or `CUML_LOG_CRITICAL` as appropriate. The logger is the
RAPIDS Logger instance returned by `ML::default_logger()`; its levels use
`rapids_logger::level_enum`. Do not append a newline to log messages, and avoid
formatting expensive diagnostic values unless the level will be logged.

## Multi-GPU communication

cuML's distributed C++ algorithms use one process per GPU. Communication is
provided through `raft::comms` attached to the RAFT handle. With CUDA-aware MPI,
current tests initialize the handle as follows:

```cpp
#include <cuda_runtime_api.h>
#include <mpi.h>
#include <raft/comms/mpi_comms.hpp>
#include <raft/core/handle.hpp>

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int local_rank = 0;
  MPI_Comm local_comm;
  MPI_Comm_split_type(
    MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
  MPI_Comm_rank(local_comm, &local_rank);
  cudaSetDevice(local_rank);

  {
    raft::handle_t handle;
    raft::comms::initialize_mpi_comms(&handle, MPI_COMM_WORLD);
    auto const& comm = handle.get_comms();
    // All ranks in comm must enter collective algorithm calls cooperatively.
  }

  MPI_Comm_free(&local_comm);
  MPI_Finalize();
  return 0;
}
```

Check every MPI and CUDA return value in production code. The snippet focuses
on the current `initialize_mpi_comms` signature and resource lifetime; consult
[`cpp/tests/mg`](https://github.com/NVIDIA/cuml/tree/main/cpp/tests/mg) for
complete test setup and error handling.

## Testing

Add focused GoogleTest coverage alongside the corresponding tests under
`cpp/tests`. Use focused GoogleTests for reusable primitives and end-to-end
GoogleTests for algorithms, covering representative inputs and datasets. Add
every new test source to the appropriate `CMakeLists.txt` so it is built and
run. Configure and build the relevant targets, then run CTest from the build
tree:

```bash
./build.sh libcuml
ctest --test-dir cpp/build --output-on-failure
```

To select a subset while iterating:

```bash
ctest --test-dir cpp/build --output-on-failure -R '<test-name-regex>'
```

Installed CI test packages can also be exercised through `ci/run_ctests.sh`.
Tests should cover meaningful shapes, dtypes, failure conditions, and stream or
distributed behavior affected by the change without inflating the fast suite
unnecessarily.

## Doxygen documentation

Document interfaces in installed headers with Doxygen comments. State parameter
and output shapes, dtypes, host or device memory location, ownership and
lifetime, stream behavior, preconditions, errors, and algorithm references.
Do not imply that the generated C++ reference makes an interface stable or that
libcuml validates every input.

Build Doxygen XML before Sphinx so Breathe can resolve declarations:

```bash
./build.sh cppdocs pydocs
```

The published C++ API reference is part of the Sphinx Developer Guide. Doxygen
XML is an intermediate input, not a separately published API site.
