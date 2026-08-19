# cuML Python Developer Guide

This document provides comprehensive guidelines and best practices for contributing to the cuML Python library, the machine learning library within the CUDA and RAPIDS ecosystem. As an evolving document, we welcome contributions, clarifications, and issue reports to help maintain and improve these guidelines.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Guide Map](#guide-map)
3. [Getting Started](#getting-started)
4. [Coding Style](#coding-style)
5. [Documentation](#documentation)
6. [Testing and Unit Testing](#testing-and-unit-testing)
7. [Input Validation](#input-validation)
8. [Memory Management](#memory-management)
9. [Thread Safety](#thread-safety)
10. [Creating New Estimators](#creating-new-estimators)
11. [Deprecation Policy](#deprecation-policy)
12. [Logging](#logging)
13. [Multi-GPU Support](#multi-gpu-support)
14. [Benchmarking](#benchmarking)

## Prerequisites

Before diving into Python development for cuML, please ensure you have:

1. Reviewed our [contribution guidelines](../../CONTRIBUTING.md) for general project standards
2. Read the [Python cuML README](../../python/README.md) for setup and installation instructions

If you are working on C++/CUDA code or need to understand the underlying implementation details, you should also familiarize yourself with the [C++ Developer Guide](../cpp/DEVELOPER_GUIDE.md).

## Guide Map

Use this document for repository-wide Python development policy: style, docstrings, testing, memory management, deprecations, logging, multi-GPU structure, and benchmarking.

Use [Estimator Guide](ESTIMATOR_GUIDE.md) when creating or modifying a `cuml.Base` estimator. It contains the estimator contract, copyable estimator skeleton, input validation, array descriptor guidance, reflection guidance, and estimator-specific do's and don'ts.

## Getting Started

The cuML Python library provides a scikit-learn style API for GPU-accelerated machine learning algorithms. This guide focuses on Python-specific development practices, while maintaining consistency with the underlying C++/CUDA implementations.

## Coding Style

The majority of style guidelines are enforced through pre-commit hooks. Run the configured hooks before submitting changes so formatting and lint checks match the repository's current toolchain.

See [Documentation](#documentation) for guidelines on doc-string formatting.

## Documentation
Doc-string documentation should follow the [NumPy docstring style guide](https://numpydoc.readthedocs.io/en/stable/format.html) for documenting interfaces. This provides a consistent format that is both human-readable and machine-parseable.

### Docstring Sections
The docstring should include the following sections in order:

1. **Short Summary**
   - A one-line summary that does not use variable names or the function name
   - Example: `"""The sum of two numbers."""`

2. **Extended Summary**
   - A few sentences giving an extended description
   - Should clarify functionality, not implementation details

3. **Parameters**
   - Description of function arguments, keywords and their types
   - Format:
   ```python
   Parameters
   ----------
   x : type
       Description of parameter `x`.
   y : type, optional
       Description of parameter `y` (with type not specified).
   ```

4. **Returns**
   - Description of returned values and their types
   - Format:
   ```python
   Returns
   -------
   int
       Description of anonymous integer return value.
   ```

5. **Other Sections** (as needed)
   - Raises
   - See Also
   - Notes
   - References
   - Examples

### Examples
Examples must be written in the form of an interactive REPL. The examples in the
documentation are checked through doctest. To skip the check for an example's
output, use the command `# doctest: +SKIP`. Examples subject to numerical
imprecision, or that can't be reproduced consistently should be skipped.

Example docstring format:
```python
def function_name(param1, param2):
    """Short summary.

    Extended summary if needed.

    Parameters
    ----------
    param1 : type
        Description of param1
    param2 : type, optional
        Description of param2

    Returns
    -------
    type
        Description of return value

    Examples
    --------
    >>> function_name(1, 2)
    result
    """
    pass
```

### Best Practices
1. Use reStructuredText (reST) syntax for formatting
2. Keep docstring lines to 75 characters for readability in text terminals
3. Enclose parameter names in single backticks when referring to them
4. Use double backticks for inline code
5. Include type information for all parameters and return values
6. Document default values for optional parameters
7. Use the `@generate_docstring` decorator for common parameter documentation
8. Include examples that demonstrate typical usage

For more details, refer to the [NumPy docstring style guide](https://numpydoc.readthedocs.io/en/stable/format.html).

## Testing and Unit Testing
We use [pytest](https://docs.pytest.org/en/latest/) for writing and running tests. To see existing examples, refer to any of the `test_*.py` files in the folder `cuml/tests`.

### Test Organization
- Keep all tests for a single estimator in one file, with exceptions for:
  - Performance testing/benchmarking
  - Generic estimator checks (e.g., `test_base.py`)
- Use small, focused datasets for correctness testing
- Only explicitly parametrize dataset size when it triggers alternate code paths

### Test Input Generation
We support three main approaches for test input generation:

1. **Fixtures** (`@pytest.fixture`):
   - For shared setup/teardown code and resources
   - Examples: random seeds, clients, loading test datasets
   ```python
   @pytest.fixture(scope="module")
   def random_state():
       return 42
   ```

2. **Parametrization** (`@pytest.mark.parametrize`):
   - For testing specific input combinations
   - Good for hyperparameters and configurations that we need to test _exhaustively_
   ```python
   @pytest.mark.parametrize("solver", ["svd", "eig"])
   def test_estimator(solver):
       pass
   ```

3. **Hypothesis** (`@given`):
   - For property-based testing with random inputs
   - Must include at least one `@example` for deterministic testing
   - Preferred for dataset generation and most hyperparameter testing
   ```python
   @example(dataset=small_regression_dataset(np.float32), alpha=floats(0.1, 10.0))
   @given(dataset=standard_regression_datasets(), alpha=1.0)
   def test_estimator(dataset, alpha):
       pass
   ```

### Test Parameter Levels

You can mark test parameters for different scales with (`unit_param`, `quality_param`, and `stress_param`).

_Note: For dataset scaling, prefer using hypothesis, e.g. with `standard_regression_datasets()`._

We provide three test parameter levels:

1. **Unit Tests** (`unit_param`): Small values for quick, basic functionality testing
   ```python
   unit_param(2)  # For number of components
   ```

2. **Quality Tests** (`quality_param`): Medium values for thorough testing
   ```python
   quality_param(10)  # For number of components
   ```

3. **Stress Tests** (`stress_param`): Large values for performance testing
   ```python
   stress_param(100)  # For number of components
   ```

Control via these pytest options:
- `--run_unit`: Unit tests (default)
- `--run_quality`: Quality tests
- `--run_stress`: Stress tests

### Testing Guidelines

1. **Accuracy Testing**
   - Compare against recorded reference values when possible
   - Document origin of reference values
   - Use appropriate quality metrics for equivalent but different results
   - Ensure reproducibility rather than using retry logic

2. **Minimize resources**
   - Use minimal dataset sizes
   - Only test different scales if they would actually hit different code paths

3. **Best Practices**
   - Write small, focused tests
   - Avoid duplication between test files
   - Choose appropriate input generation method
   - Make tests reproducible
   - Document test assumptions and requirements

### Running Tests
Build from the repository root. Run Python tests from `python/cuml/` or one of its subdirectories so pytest picks up the package configuration.

```bash
# From the repository root
./build.sh

# Then run Python tests
cd python/cuml
python -m pytest  # Run all configured Python tests
```

Common options:
- `python -m pytest cuml/tests/test_kmeans.py` - Run a specific file
- `python -m pytest -k "test_kmeans"` - Run tests matching a pattern
- `python -m pytest --run_unit` - Run only unit tests
- `python -m pytest -v` - Verbose output

Running pytest from outside `python/cuml/` can result in import errors or missed pytest configuration.

## Input Validation

Code should use `cuml.internals.validation` for user-facing input validation.
These helpers are the standard path for matching scikit-learn validation
behavior, simplifying input ingest, and avoiding module-specific validation
pipelines. See the [Estimator Guide](ESTIMATOR_GUIDE.md#input-validation) for
estimator-specific patterns and examples.

Prefer `check_inputs` for estimator methods that validate `X` and optional `y`
/ `sample_weight` values. Use lower-level helpers directly only when a method
has a non-standard shape that the higher-level helper cannot express.

Validation helpers should be configured to describe what the estimator actually
supports. Set `dtype`, `mem_type`, `order`, `accept_sparse`,
`ensure_all_finite`, `ensure_non_negative`, and minimum shape requirements
explicitly when the defaults are not correct. Do not hand-roll equivalent
checks unless the common helpers cannot express the estimator's requirements.

The validation pipeline normalizes inputs to standard array containers:

- `cupy.ndarray` or `cupyx.scipy.sparse.spmatrix` for device outputs
- `numpy.ndarray` or `scipy.sparse.spmatrix` for host outputs

Fit-like methods should validate with `reset=True` so feature metadata is set
from the training input. Inference methods should usually call
`check_is_fitted` and then validate with the default `reset=False` so the input
is checked against the fitted feature metadata.

Tests for validation changes should cover both accepted and rejected inputs,
including dtype conversion, sparse support, finite/non-negative requirements,
feature-count and feature-name checks, and any classifier class-encoding
behavior. When changing validation behavior for an estimator, also check the
scikit-learn compatibility tests and the `cuml.accel` upstream xfail list.

## Memory Management

cuML uses RMM (RAPIDS Memory Manager) for GPU memory management and configures
CuPy to allocate through RMM when `cuml` is imported. Validated user inputs
should generally be processed as standard arrays (`cupy`, `numpy`,
`cupyx.scipy.sparse`, or `scipy.sparse`) returned by the input validation
helpers.

Use standard array libraries for allocations and conversions in new internal
code:

```python
import cupy as cp
import numpy as np

device_arr = cp.empty((1000,), dtype=cp.float32)
host_arr = np.zeros((1000,), dtype=np.float32)

device_arr = cp.asarray(host_arr)
host_arr = cp.asnumpy(device_arr)
```

Additional considerations:

- Minimize memory transfers between host and device
- Prefer device memory for GPU kernels and host memory only when a CPU API needs it
- Consider memory layout (C/F order) for optimal performance
- Let `@mlfunc` and `ReflectedAttr` handle user-facing output type conversion
  for estimator methods and fitted array attributes

## Thread Safety

Algorithms implemented in C++/CUDA should be implemented in a thread-safe manner. The Python code is generally not thread safe.
Refer to the section on thread safety in [C++ DEVELOPER_GUIDE.md](../cpp/DEVELOPER_GUIDE.md#thread-safety)

## Creating New Estimators

When implementing a new estimator in cuML, follow these key steps:

1. Choose an implementation strategy appropriate for the algorithm. C++/CUDA implementations are still preferred for shared, performance-critical primitives, but CuPy-only implementations are permitted when they provide a significant speedup and do not need a reusable C++ algorithm first.
2. Create a Python wrapper class that:
   - Follows scikit-learn's API design patterns
   - Inherits from `cuml.Base`
   - Is placed in the appropriate subdirectory matching scikit-learn's structure
   - Uses `cuml.internals.validation` for public input validation

For detailed implementation guidelines, including file organization, API design, output type handling, and a copyable estimator skeleton, refer to the [Estimator Guide](ESTIMATOR_GUIDE.md).

## Deprecation Policy

cuML follows the policy of deprecating code for one release prior to removal. This applies
to publicly accessible functions, classes, methods, attributes and parameters. During the
deprecation cycle the old name or value is still supported, but will raise a deprecation
warning when it is used.

Code in cuML should not use deprecated cuML APIs.

```python
warnings.warn(
    (
        "Attribute `foo` was deprecated in version 25.06 and will be"
        " removed in 25.08. Use `metric` instead."
    ),
    FutureWarning,
)
```

The warning message should always give both the version in which the deprecation was introduced
and the version in which the old behavior will be removed. The message should also include
a brief explanation of the change and point users to an alternative.

In addition, a deprecation note should be added in the docstring, repeating the information
from the warning message:

```
.. deprecated:: 25.06
    Attribute `foo` was deprecated in version 25.06 and will be removed
    in 25.08. Use `metric` instead.
```

A deprecation requires a test which ensures that the warning is raised in relevant cases
but not in other cases. The warning should be caught in all other tests (using e.g., ``@pytest.mark.filterwarnings``).

### Public vs Private APIs
The following rules determine whether an API is considered public and therefore subject to the deprecation policy:

1. Any API prefixed with `_` is considered private and not subject to deprecation rules
2. APIs within the following namespaces are private and not subject to deprecation rules:
   - `internals`
   - `utils`
   - `common`
3. APIs within the `experimental` namespace are not subject to deprecation rules
4. For all other APIs, the determination of public vs private status is based on:
   - Presence in public documentation
   - Usage in public examples
   - Explicit declaration as part of the public API

When in doubt, treat an API as public to ensure proper deprecation cycles.

## Logging

cuML uses the [rapids-logger library](https://github.com/rapidsai/rapids-logger) for logging which in turn is built on top of [spdlog](https://github.com/gabime/spdlog). This provides fast, asynchronous logging with support for multiple sinks and formatting options.

### Usage

To emit log messages within the Python library, use the functions provided within the `cuml.internals.logger` module:

```python
from cuml.internals.logger import debug, info, warn, error, critical

# Example usage
debug("Detailed debug information")
info("General information")
warn("Warning message")
error("Error message")
critical("Critical error message")
```

### Log Levels

Use the appropriate log level based on the message's importance and target audience:

- **DEBUG**: Detailed information for debugging primarily aimed at developers
  - Use for detailed execution flow, variable values, and internal state
  - Example: "Initializing CUDA stream with device 0"

- **INFO**: General information about program execution primarily aimed at users
  - Use for messages that could be of interest to users, but usually don't require any further action
  - Example: "Building knn graph using nn descent"

- **WARNING**: Indicate a potential problem or other potentially surprising behavior
  - Use for ignored parameters, performance issues, or unexpected but handled conditions
  - Use for messages that we typically expect users to take action on
  - Example: "Parameter 'n_neighbors' was ignored as it's not supported in this mode"

- **ERROR**: Indicates a problem that prevents program execution
  - Use for recoverable errors that prevent the current operation
  - Example: "Failed to allocate GPU memory for input data"

- **CRITICAL**: A critical problem that is expected to lead to immediate program termination
  - Use sparingly, only for unrecoverable errors
  - Example: "CUDA device became unresponsive"

### Best Practices

1. **Message Content**
   - Be specific and include relevant context
   - Include error codes or exception details when applicable
   - Use consistent terminology

2. **Performance**
   - Avoid expensive string operations in debug messages
   - Do not emit log messages within loops that are input size dependent
   - Use lazy evaluation for debug messages:
   ```python
   from cuml.internals import logger

   if logger.should_log_for(logging.DEBUG):
       logger.debug(f"Expensive operation result: {expensive_operation()}")
   ```

3. **Formatting**
   - Use proper punctuation and capitalization
   - Keep messages concise but informative
   - Include relevant numeric values with units

4. **Context**
   - Include relevant parameters or state information
   - For errors, include the operation that failed
   - For warnings, explain why the behavior might be unexpected

### Configuration

The logging level can be configured using the `set_level` function from `cuml.internals.logger`:

```python
from cuml.internals import logger

# Set log level directly
logger.set_level(logger.level_enum.debug)

# Or use as a context manager for temporary level changes
with logger.set_level(logger.level_enum.debug):
    # Operations with debug logging
    pass
```

Available log levels:
- `logger.level_enum.debug`
- `logger.level_enum.info`
- `logger.level_enum.warn`
- `logger.level_enum.error`
- `logger.level_enum.critical`

## Multi-GPU Support

cuML provides limited multi-GPU support through Dask. Here is a basic example:

```python
import cudf
import dask_cudf
import numpy as np
from cuml.dask.linear_model import LogisticRegression
from dask.distributed import Client
from dask_cuda import LocalCUDACluster


def main():
    # Create sample data
    X = cudf.DataFrame(
        {
            "col1": np.array([1, 1, 2, 2], dtype=np.float32),
            "col2": np.array([1, 2, 2, 3], dtype=np.float32),
        }
    )
    y = cudf.Series(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32))

    # Convert to distributed dataframes
    X_ddf = dask_cudf.from_cudf(X, npartitions=2)
    y_ddf = dask_cudf.from_cudf(y, npartitions=2)

    # Train distributed model
    model = LogisticRegression()
    model.fit(X_ddf, y_ddf)

    # Make predictions
    prediction = model.predict(X_ddf)
    print(prediction.compute())


if __name__ == "__main__":
    # Create a local cluster with 2 GPUs
    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES="0,1")
    client = Client(cluster)
    main()
    client.close()
    cluster.close()
```

Key points for implementing multi-GPU estimators:

- Dask-based estimators should be implemented within the cuml.dask namespace
- The dask layer should focus on distributed computation, with base algorithms implemented in standard estimators
- See currently implemented estimators, e.g., LogisticRegression for examples on how to implement dask-based Multi-GPU estimators

## Benchmarking

The cuML code including its Python operations can be profiled. The `nvtx_benchmark.py` is a helper script that produces a simple benchmark summary. To use it, run `python nvtx_benchmark.py "python test.py"`.

Here is an example with the following script:
```python
from cuml.datasets import make_blobs
from cuml.manifold import UMAP

X, y = make_blobs(n_samples=1000, n_features=30)

model = UMAP()
model.fit(X)
embeddings = model.transform(X)
```

that once benchmarked can have its profiling summarized:
```
datasets.make_blobs                                          :   1.3571 s

manifold.umap.fit [0x7f10eb69d4f0]                           :   0.6629 s
    |> umap::unsupervised::fit                               :   0.6611 s
    |==> umap::knnGraph                                      :   0.4693 s
    |==> umap::simplicial_set                                :   0.0015 s
    |==> umap::embedding                                     :   0.1902 s

manifold.umap.transform [0x7f10eb69d4f0]                     :   0.0934 s
    |> umap::transform                                       :   0.0925 s
    |==> umap::knnGraph                                      :   0.0909 s
    |==> umap::smooth_knn                                    :   0.0002 s
    |==> umap::optimization                                  :   0.0011 s
```
