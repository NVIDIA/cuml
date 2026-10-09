# cuML Benchmark Suite

Use `python -m cuml.benchmark` to run estimator workloads and save warmup and
measurement observations in JSON. A YAML suite defines shared workloads, their
supported providers, and execution profiles.

## Run a suite

Run in an environment with cuML installed, the selected providers' dependencies,
and enough host/device memory. Suite loading requires PyYAML and msgspec
(`python -m pip install pyyaml msgspec` if they are missing).

```bash
# Compare the cuML and scikit-learn providers at reduced workload sizes.
python -m cuml.benchmark --suite estimators --profile smoke \
  --provider cuml --provider scikit-learn --output comparison

# Run the cuml.accel provider with per-repetition logging.
python -m cuml.benchmark --suite estimators --profile smoke \
  --provider cuml.accel --output accelerated -v
```

The `estimators` built-in (the default suite) is packaged in
`python/cuml/cuml/benchmark/suites/estimators.yaml`. Refer to the manifest for
its workloads, supported providers, and profiles.

By default, the CLI runs benchmarks using only `cuml`. Use `--provider`
to select another provider and compare implementations of the same algorithms.
Repeat the option to select multiple providers; explicit selections replace the
default. Selected providers run sequentially in separate processes, in manifest
order. Execution continues after a provider failure; the command exits nonzero
if any selected provider fails.

The default profile is `standard`. Select a profile with `--profile`.
Copy the manifest and edit its cases or profiles to customize workloads,
parameters, and execution counts.

Provider requirements:
- `cuml.accel` requires at least one warmup. Warmups collect dispatch evidence;
  any recorded CPU fallback fails the case and stops its repetitions.
- `cuml.dask` requires at least two visible GPUs and Dask/Dask-CUDA dependencies.
  The harness creates one local cluster for the provider run. Select GPUs with
  `CUDA_VISIBLE_DEVICES` (for example, `CUDA_VISIBLE_DEVICES=4,5`).
- Estimator availability and accepted constructor parameters depend on installed
  packages.

## Define a suite

Save this manifest as `my-suite.yaml`:

```yaml
version: 2
name: small-kmeans
providers: [cuml, scikit-learn, cuml.accel]
profiles:
  standard:
    warmups: 1
    repetitions: 3
    size_scale: 1.0
    timeout_sec: 60
cases:
  - estimator: KMeans
    dataset:
      generator: blobs
      shape: {rows: 1024, features: 8}
      dtype: float32
      format: dense
      parameters: {centers: 5, cluster_std: 1.0}
    operation: fit_predict
    input_selection: [X]
    parameters: {n_clusters: 5, random_state: 42}
```

A suite brings together:

- **Providers:** a nonempty, unique list of Python-registered providers.
- **Profiles:** named warmup/measurement counts, dataset size scaling, and optional
  timeouts.
- **Cases:** estimator workloads with constructor parameters and measured
  operations, such as `fit_predict`.
- **Datasets:** synthetic inputs defined by generator, shape, data type, and
  representation. Generator parameters are separate from estimator parameters.
- **Input selection:** arguments passed to the operation (`X`, `y`, or both).
  Inference cases also specify training rows and inputs for the initial fit.

Cases inherit the suite's providers. To restrict a case, give it a
nonempty subset, for example:

```yaml
  - estimator: AgglomerativeClustering
    providers: [cuml, scikit-learn]
    # ... dataset, operation, input_selection, and parameters ...
```

### Compare providers of the same algorithm

Providers are defined in Python by associating an execution backend with an
estimator registry. Suites declare which providers they support.

| Provider | Estimators | Execution |
| --- | --- | --- |
| `scikit-learn` | sklearn estimators, including sklearn's HDBSCAN | CPU |
| `umap-learn` | UMAP | CPU |
| `hdbscan` | Standalone HDBSCAN | CPU |
| `cuml` | Native cuML estimators | Single GPU |
| `cuml.accel` | Supported accelerated sklearn, umap-learn, and hdbscan estimators | Single GPU (transparent dispatch) |
| `cuml.dask` | Distributed cuML estimators | Multi-GPU Dask |

For example, compare both CPU HDBSCAN implementations against cuML:

```yaml
version: 2
name: hdbscan-providers
providers: [scikit-learn, hdbscan, cuml]
profiles:
  standard:
    warmups: 1
    repetitions: 3
    size_scale: 1.0
cases:
  - estimator: HDBSCAN
    dataset:
      generator: blobs
      shape: {rows: 1024, features: 8}
      parameters: {centers: 5}
    operation: fit_predict
    input_selection: [X]
    parameters: {min_cluster_size: 5}
```

Select providers with `--provider scikit-learn --provider hdbscan`.
Each provider produces its own file (`scikit-learn.json` or `hdbscan.json`), with
matching workload IDs. Constructor parameters must be accepted by each selected
provider; parameter names alone do not imply identical algorithm semantics.
Use separate cases restricted to particular providers when parameters differ.

A suite spanning algorithms from different libraries can narrow case
applicability with `providers: [umap-learn, cuml]` for UMAP and
`providers: [scikit-learn, hdbscan, cuml]` for HDBSCAN. An omitted case list
inherits all suite providers; incompatible estimator/provider pairs are rejected.

The `cuml.accel` provider includes multiple upstream distribution packages.
Its HDBSCAN binding is the standalone hdbscan implementation, not sklearn's
HDBSCAN.

Suite loading validates the manifest structure, inference inputs, provider
compatibility, and workload IDs before execution.

## Timing and results

The output directory contains one JSON results file per selected provider:

```text
results/
  cuml.json
  scikit-learn.json
  cuml.accel.json
  cuml.dask.json
```

Without `--output`, the CLI chooses an unused timestamped directory under
`.benchmarks/` in the current working directory, for example
`.benchmarks/estimators-smoke-20260813T000000Z/`. The repository's `.gitignore`
ignores `.benchmarks/` directories. An explicit output directory may already
exist. **Without
`--resume`, the harness refuses to run if a selected provider's results file
already exists.** Use `--resume` to continue the run or choose a new output
directory. Unrelated files and unselected providers' results files are left
untouched. If startup fails before a
results file is created, the failure is reported in the console and exit status.
Check the command's exit status to confirm the invocation succeeded.

Each results file contains:

- **Run metadata:** command, suite/profile, provider execution plan, system, and
  software used for that provider run.
- **Case results:** workload descriptors, implementation, and success/failure.
  Workload IDs are provider-independent, so matching cases can be joined across
  results files. Each provider run has its own run ID.
- **Observations:** individual warmup and measurement repetitions. Use successful
  measurement observations to calculate performance summaries from raw timings.
- **Timing:** synchronized wall time in seconds for the selected estimator
  operation. Preparation and setup are excluded, so these timings differ from
  total elapsed console time.

Each provider saves its results file after every case, including failures.
The results JSON Schema is packaged at
`cuml/benchmark/schemas/benchmark-result.schema.json`.

## Resume a run

```bash
python -m cuml.benchmark --suite estimators --profile smoke \
  --output results --resume
```

`--resume` requires an explicit, existing output directory. For each selected
provider, it retains successful cases, retries failed cases from scratch, and
runs missing cases. A missing provider results file starts a new provider run. Other
providers' results files are preserved when selecting a subset or adding another
provider.

Existing results files must be compatible: schema version, methodology, suite
metadata (including path, provider, profile, and execution plan), software, and
system metadata must match. Original provider run IDs are preserved. Software
metadata includes the installed Python distribution snapshot and, for GPU
providers, CUDA driver and runtime versions. Changing
counts, timeouts, suite location, installed package versions (including generator
and backend dependencies), CUDA versions, or hardware can prevent resume.
Use a new output directory for a changed experiment.

## Add or modify cases

Start from `suites/estimators.yaml` or the smaller `suites/example.yaml`.
The example exercises matching workloads across native cuML, CPU, and accel,
including supervised fitting, data types, sparse inputs, and inference. Use it
as a template and execution check.

Each module in `providers/` defines one provider's estimator catalog and backend
binding; `providers/__init__.py` registers the selectable names. Shared types live
in `providers/base.py`. Check generator constraints in `datasets.py`.
Extend packaged manifests and provider catalogs together; the tests in
`python/cuml/tests/test_benchmark.py` validate packaged suites, workload
identity, and exhaustive cuML and cuml.accel coverage in the `estimators` suite.

## Run from Python

Use `cuml.benchmark.run()` to run a suite from Python with the same sequential
isolated workers as the CLI:

```python
from cuml import benchmark

results = benchmark.run(
    "my-suite.yaml", providers=["cuml", "scikit-learn"]
)
```

The return value is a `BenchmarkResults` dataclass whose `artifacts` field holds
JSON dictionaries keyed by provider. Temporary output is cleaned by default.
Pass a directory path to `output`, such as `output="./results"`, to save results
and checkpoints. Use the same directory with `resume=True` to
continue a run. The CLI defaults to a timestamped directory.
See the [API reference](../api/cuml.benchmark) for failure diagnostics,
temporary cleanup and per-provider resume semantics.

## CPU-only execution

From a source checkout, run without cuML installed:

```bash
python -m pip install scikit-learn pandas scipy pyyaml msgspec
python python/cuml/cuml/benchmark/run_benchmarks.py \
  --profile smoke --provider scikit-learn --output cpu-results
```

For `umap-learn` or `hdbscan`, install the package and select its provider.

## NVTX profiling

`cuml.benchmark.nvtx_benchmark` is a standalone Nsight Systems utility. It
requires `nsys` version 2021.4 or later and profiles NVTX ranges in the command
supplied as its single argument:

```bash
python -m cuml.benchmark.nvtx_benchmark \
  "python -m cuml.benchmark --provider cuml --profile smoke --output results"
```

It is independent of the harness's observation timing and JSON serialization.
