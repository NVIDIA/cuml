# cuML Benchmark Suite

This document describes how to run the cuML benchmark suite. The tools support two execution modes:

- **Full mode**: `python -m cuml.benchmark` — requires cuML and GPU dependencies; runs GPU + CPU benchmarks.
- **Standalone mode**: `python run_benchmarks.py` — works without cuML installed (e.g. from the repo); runs CPU-only benchmarks when cuML is not available.

The benchmark runner also supports YAML manifests. A manifest is the declarative source of truth for a benchmark suite, while CLI flags can be used to filter or override selected fields at runtime.

## Contents

- [Running the benchmarks](#running-the-benchmarks)
- [Common options](#common-options)
- [Examples](#examples)
- [YAML manifests](#yaml-manifests)
  - [Manifest structure](#top-level-schema)
  - [`suite`](#suite)
  - [`profiles`](#profiles)
  - [Compact `variants`](#compact-variants)
  - [`defaults`](#defaults)
  - [Benchmark entry schema](#benchmark-entry-schema)
  - [Choosing dimensions](#choosing-dimensions)
  - [Creating your own manifest](#creating-your-own-manifest)
  - [CLI overrides in config mode](#cli-overrides-in-config-mode)
- [Input types](#input-types)
- [Manifest structural validation](#manifest-structural-validation)

## Running the benchmarks

### Full mode (cuML installed)

When cuML is installed, use the module entry point:

```bash
python -m cuml.benchmark --dataset classification LogisticRegression --csv results.csv
```

If a GPU is available, this runs both GPU (cuML) and CPU implementations and reports speedup.
Use `--output` to write the canonical JSON benchmark artifact:

```bash
python -m cuml.benchmark --dataset classification LogisticRegression --output results.json
```

To run a YAML-defined suite:

```bash
python -m cuml.benchmark \
  --config python/cuml/cuml/benchmark/configs/single_gpu.yaml \
  --profile default \
  --backends gpu \
  --output results.json
```

To run the tiny harness-validation manifest:

```bash
python -m cuml.benchmark \
  --config python/cuml/cuml/benchmark/configs/test.yaml \
  --profile default \
  --backends cpu
```

To run a YAML-defined suite:

```bash
python -m cuml.benchmark \
  --config python/cuml/cuml/benchmark/configs/single_gpu.yaml \
  --profile default \
  --backends gpu \
  --csv results.csv
```

To run the tiny harness-validation manifest:

```bash
python -m cuml.benchmark \
  --config python/cuml/cuml/benchmark/configs/test.yaml \
  --profile default \
  --backends cpu
```

### Standalone mode (from the repository)

From the `python/cuml/cuml/benchmark/` directory, you can run without installing cuML:

```bash
cd python/cuml/cuml/benchmark/
python run_benchmarks.py --dataset classification LogisticRegression
```

When cuML is not installed, only CPU (e.g. scikit-learn) benchmarks run. The same script works with cuML installed for full GPU+CPU runs.

Standalone mode also supports YAML manifests:

```bash
cd python/cuml/cuml/benchmark/
python run_benchmarks.py \
  --config configs/test.yaml \
  --profile default \
  --backends cpu
```

The following Python packages are required for standalone mode:

```bash
pip install numpy pandas scikit-learn scipy
```

YAML manifests require PyYAML and msgspec. If either is not installed, the
benchmark CLI will print install instructions. You can install them with:

```bash
conda install -c conda-forge pyyaml msgspec
# or
python -m pip install pyyaml msgspec
```

### CPU-only mode

To run only CPU benchmarks (e.g. when no GPU or to compare CPU implementations only):

```bash
python -m cuml.benchmark --skip-gpu --dataset classification LogisticRegression
# or: python run_benchmarks.py --skip-gpu --dataset classification LogisticRegression
```

### GPU-only mode

To run only GPU (cuML) benchmarks:

```bash
python -m cuml.benchmark --skip-cpu --dataset classification LogisticRegression
```

**Note:** Do not use both `--skip-gpu` and `--skip-cpu`; that would run no benchmarks. The script will report an error.

## Common options

| Option | Description |
|--------|-------------|
| `--dataset` | Dataset name: e.g. `blobs`, `classification`, `regression`, `higgs`. Use `--print-datasets` to list all. |
| `--config` | Path to a YAML benchmark manifest. |
| `--profile` | Named profile to select from a YAML benchmark manifest. |
| `--backends` | Comma-separated backends to run (`cpu`, `gpu`). |
| `--skip-gpu` | Skip GPU/cuML benchmarks (CPU only); compatibility shortcut. |
| `--skip-cpu` | Skip CPU benchmarks (GPU/cuML only); compatibility shortcut. |
| `--output FILE` | Save the canonical JSON artifact with metadata and grouped benchmark results. |
| `--csv [FILE]` | Save a legacy flat CSV export. |
| `--min-rows`, `--max-rows`, `--num-sizes` | Control sample sizes for scaling benchmarks. |
| `--input-dimensions` | Feature dimensions to test (e.g. `16 256`). |
| `--metadata-override KEY=VALUE` | Override a JSON metadata field using dotted paths, e.g. `hardware.label=test-node`. May be repeated. |
| `--print-algorithms` | List available algorithms and exit. |
| `--print-datasets` | List available datasets and exit. |
| `--print-status` | Print GPU/cuML availability and exit. |

## Examples

Run a single algorithm at default sizes:

```bash
python -m cuml.benchmark --dataset classification LogisticRegression
```

Run with parameter sweeps and save a JSON artifact:

```bash
python -m cuml.benchmark --dataset classification \
  --max-rows 100000 --min-rows 10000 \
  --dataset-param-sweep n_classes=[2,4] \
  --cuml-param-sweep n_estimators=[10,100] \
  --output results.json \
  RandomForestClassifier
```

Run multiple algorithms (use `--` before algorithm names when passing `--input-dimensions`):

```bash
python -m cuml.benchmark --dataset blobs --num-sizes 1 \
  --input-dimensions 16 256 -- \
  DBSCAN KMeans PCA UMAP
```

Run with a real dataset at its default size:

```bash
python -m cuml.benchmark --dataset higgs --default-size \
  RandomForestClassifier LogisticRegression
```

Run a manifest but restrict execution to one algorithm:

```bash
python -m cuml.benchmark \
  --config python/cuml/cuml/benchmark/configs/single_gpu.yaml \
  --profile default \
  LogisticRegression
```

Run a manifest and override only the row count from the CLI:

```bash
python -m cuml.benchmark \
  --config python/cuml/cuml/benchmark/configs/test.yaml \
  --profile default \
  --num-rows 500 \
  --backends cpu
```

In config mode, only explicitly provided CLI flags override the manifest. Parser defaults do not silently replace YAML values.

## Output

The recommended output artifact is a single JSON file written with `--output`:

```bash
python -m cuml.benchmark \
  --config python/cuml/cuml/benchmark/configs/single_gpu.yaml \
  --profile default \
  --backends gpu \
  --output results.json
```

The JSON file contains:

- `metadata`: run context, command, Python/platform info, cuML/git identity, GPU availability, hardware details, and config/profile provenance
- `results`: grouped benchmark results, with raw backend timings and statuses under each logical benchmark variation

For lab or CI machines where detected hardware names need a stable display label,
use repeated metadata overrides:

```bash
python -m cuml.benchmark \
  --config python/cuml/cuml/benchmark/configs/single_gpu.yaml \
  --profile default \
  --backends gpu \
  --output results.json \
  --metadata-override hardware.label=dgx-ci-01 \
  --metadata-override hardware.gpu.effective.name=H100
```

Example shape:

```json
{
  "metadata": {
    "result_schema_version": 1,
    "config": {
      "path": "python/cuml/cuml/benchmark/configs/single_gpu.yaml",
      "profile": "default"
    }
  },
  "results": [
    {
      "benchmark_id": "logreg_fit_narrow_default",
      "algorithm": "LogisticRegression",
      "shape": {
        "rows": 84000000,
        "features": 16,
        "estimated_input_size_gb": 5.376
      },
      "params": {
        "declared": {},
        "effective": {
          "gpu": {},
          "cpu": null
        }
      },
      "backends": {
        "gpu": {
          "status": "success",
          "time_sec": 0.91,
          "accuracy": 0.995
        }
      }
    }
  ]
}
```

Derived comparisons such as GPU speedup are intentionally not stored in JSON. They are computed from raw backend timings for terminal display and downstream analysis.

### JSON result schema

The JSON artifact is versioned separately from the YAML manifest schema. The current output schema is identified by:

```json
{
  "metadata": {
    "result_schema_version": 1
  }
}
```

Top-level fields:

- `results`: list of benchmark result records
- `metadata`: run-level provenance and environment information

Each entry in `results` represents one logical benchmark variation:

- `benchmark_id`: stable ID from YAML configs, or `null` for ad hoc CLI runs
- `algorithm`: benchmark algorithm name
- `dataset`: dataset generator name
- `operation`: operation name when defined by YAML, otherwise `null`
- `shape`: row/feature dimensions and estimated dense input size
- `data`: input type, dtype, and repetition count
- `params.declared`: parameters explicitly supplied through YAML or CLI sweeps
- `params.effective`: estimator parameters reported by `get_params()` for each backend when available
- `backends`: backend-specific result records keyed by `cpu` and/or `gpu`

Backend result records use these fields:

- `status`: `success` or `skipped`
- `time_sec`: elapsed benchmark time in seconds, present for successful runs
- `accuracy`: accuracy or score metric when available
- `reason`: explanation for skipped backends, such as `GPU unavailable`

Run metadata includes:

- `command`: argv and current working directory
- `host`: hostname captured with `platform.node()`
- `python`: Python executable, version, and platform
- `cuml`: cuML version, current Git repo path, SHA, and dirty-state flag when available
- `runtime`: GPU/cuML availability status
- `config`: selected config path, profile, and backend override
- `hardware`: detected CPU/GPU/OS/memory metadata plus optional effective labels from metadata overrides
- `environment`: package snapshot from `conda list` or `pip list`

The environment package snapshot is intentionally compact. Conda entries include only `name`, `version`, `build`, and `channel`; pip entries include only `name` and `version`.

Terminal output is a concise progress table, for example:

```text
 progress  algorithm                            shape        data   gpu_time   cpu_time  details
--------------------------------------------------------------------------------------------
   [1/96]  LogisticRegression              84.0M x 16    ~5.38 GB     0.91s          -  acc=0.9950
```

When multiple backends are present, timings are grouped on one row:

```text
   [1/96]  LogisticRegression              50.0K x 16    ~0.00 GB    12.3ms    18.5ms  gpu_speedup=1.50x acc=0.9944
```

CSV output remains available through `--csv`, but it is a flat compatibility export. Prefer JSON for regression tracking and reproducibility.

## YAML manifests

A manifest defines a benchmark suite, default settings for the suite, and the individual benchmark entries to run.

The repository currently includes:

- `python/cuml/cuml/benchmark/configs/test.yaml`: tiny smoke/harness-validation suite
- `python/cuml/cuml/benchmark/configs/single_gpu.yaml`: canonical single-GPU regression suite with multiple profiles

### Top-level schema

Each manifest is a YAML mapping with these top-level fields:

- `version`: integer schema version. Currently `1`.
- `suite`: metadata about the suite.
- `profiles`: optional named selectors that include benchmarks by tag.
- `defaults`: optional fields applied to every benchmark unless overridden.
- `benchmarks`: list of benchmark entries.

Example:

```yaml
version: 1

suite:
  name: single_gpu
  tier: single_gpu
  description: Canonical single-GPU regression manifest

profiles:
  default:
    include_tags: [default]

defaults:
  input_type: cupy
  dtype: fp32
  n_reps: 3
  random_state: 42
  test_split: 0.1
  backends: [cpu, gpu]
  raise_on_error: true

benchmarks:
  - id: logreg_fit_small
    algorithm: LogisticRegression
    dataset: classification
    operation: fit
    rows: [100000]
    features: [16]
    tags: [default, linear, classification]
```

### `suite`

`suite` must contain:

- `name`: short suite identifier
- `tier`: suite category such as `test` or `single_gpu`
- `description`: human-readable description

### `profiles`

Profiles allow a single manifest to define multiple run surfaces such as default or nightly.

In `single_gpu.yaml`, the manifest is organized along two axes:

- width class: `narrow`, `medium`, `wide`
- runtime tier: `default`, `nightly`

The width class changes the feature shape of the workload. The runtime tier changes only the row count. Every algorithm should therefore appear as a full `width x tier` grid.

Each profile currently has:

- `include_tags`: list of tags; a benchmark is included when it shares at least one tag with the profile

Example:

```yaml
profiles:
  default:
    include_tags: [default]
  nightly:
    include_tags: [nightly]
```

This means:

- `default` includes `narrow`, `medium`, and `wide` workloads for every algorithm at the smallest row counts
- `nightly` includes the same `narrow`, `medium`, and `wide` workloads with the largest row counts

Example:

```yaml
- id: logreg_fit_narrow_default
  algorithm: LogisticRegression
  dataset: classification
  operation: fit
  rows: [500000]
  features: [16]
  tags: [default, narrow, linear]

- id: logreg_fit_narrow_nightly
  algorithm: LogisticRegression
  dataset: classification
  operation: fit
  rows: [1500000]
  features: [16]
  tags: [nightly, narrow, linear]

- id: logreg_fit_wide_default
  algorithm: LogisticRegression
  dataset: classification
  operation: fit
  rows: [150000]
  features: [512]
  tags: [default, wide, linear]
```

### Compact `variants`

To avoid repeating the same benchmark definition for every `width x tier` combination, a benchmark entry can define `variants`. This expands one compact entry into multiple resolved benchmark entries while preserving the same runtime behavior.

Example:

```yaml
- id: logreg_fit
  algorithm: LogisticRegression
  dataset: classification
  operation: fit
  tags: [linear, classification]
  variants:
    narrow:
      features: [16]
      tiers:
        default:
          rows: [500000]
        nightly:
          rows: [1500000]
    wide:
      features: [512]
      tiers:
        default:
          rows: [150000]
```

How this works:

- each key under `variants` becomes a width-class suffix and tag, such as `narrow` or `wide`
- each key under `tiers` becomes a runtime-tier suffix and tag, such as `default` or `nightly`
- the example above expands to benchmark IDs like `logreg_fit_narrow_default`, `logreg_fit_narrow_nightly`, and `logreg_fit_wide_default`
- fields from the outer benchmark entry are shared across all expanded entries
- width-specific fields such as `features` live under the variant
- tier-specific fields such as `rows` live under the tier

Flat benchmark entries using explicit `rows`, `features`, or `shapes` are still supported. `variants` is just a more compact way to describe repeated width/tier grids.

### `defaults`

`defaults` can provide common values for benchmark entries, including:

- `dataset`
- `input_type`
- `dtype`
- `n_reps`
- `random_state`
- `test_split`
- `backends`: execution backends to run, selected from `cpu` and `gpu`
- `raise_on_error`
- `enabled`
- `tags`
- `params`, `cuml_params`, `cpu_params`, `dataset_params`
- `param_grid`, `cuml_param_grid`, `cpu_param_grid`, `dataset_param_grid`
- `comparison`
- `metadata`

Benchmark entries override scalar defaults and merge dictionary-style fields.

For GPU-oriented suites such as `single_gpu.yaml`, prefer a GPU-native `input_type` such as `cupy` or `cudf`. This avoids benchmarking large host/device or cuDF-to-NumPy conversion costs when the goal is to measure the estimator itself on GPU inputs.

### Benchmark entry schema

Each entry in `benchmarks` must define:

- `algorithm`: algorithm name from the benchmark registry

It can also define:

- `id`: stable benchmark identifier; defaults to `algorithm` if omitted
- `dataset`: dataset generator name such as `classification`, `regression`, or `blobs`
- `operation`: benchmarked operation such as `fit`, `predict`, `transform`, `fit_transform`, `fit_predict`, `fit_kneighbors`, or `kneighbors`
- `rows` and `features`: lists used as a Cartesian product
- `shapes`: explicit paired `(rows, features)` combinations when you do not want a Cartesian product
- `default_size`: use the dataset's natural default size instead of explicit dimensions
- `variants`: compact expansion for repeated width/tier families
- `params`: shared estimator parameters
- `cuml_params`: GPU-only estimator parameters
- `cpu_params`: CPU-only estimator parameters
- `dataset_params`: dataset generator parameters
- `param_grid`, `cuml_param_grid`, `cpu_param_grid`, `dataset_param_grid`: parameter sweeps expanded as Cartesian products
- `n_reps`, `input_type`, `dtype`, `random_state`, `test_split`, `backends`, `raise_on_error`
- `tags`: labels used by profiles
- `enabled`, `skip_reason`
- `comparison`, `metadata`

### Choosing dimensions

Use `rows` plus `features` when you want the Cartesian product:

```yaml
- id: pca_fit_small
  algorithm: PCA
  dataset: blobs
  operation: fit
  rows: [100000]
  features: [32, 256]
```

This runs `(100000, 32)` and `(100000, 256)`.

Use `shapes` when you want explicit pairs instead:

```yaml
- id: paired_shapes_example
  algorithm: LogisticRegression
  dataset: classification
  operation: fit
  shapes:
    - rows: 10000
      features: 32
    - rows: 50000
      features: 256
```

This runs only `(10000, 32)` and `(50000, 256)`.

### Creating your own manifest

Start from `test.yaml` if you want a tiny suite, or from `single_gpu.yaml` if you want a richer example with profiles.

This minimal custom manifest is a good starting point:

```yaml
version: 1

suite:
  name: my_benchmarks
  tier: custom
  description: Example custom benchmark suite

profiles:
  default:
    include_tags: [default]
  nightly:
    include_tags: [nightly]

defaults:
  input_type: cupy
  dtype: fp32
  n_reps: 2
  random_state: 42
  test_split: 0.1
  backends: [cpu, gpu]
  raise_on_error: true

benchmarks:
  - id: my_logreg_fit
    algorithm: LogisticRegression
    dataset: classification
    operation: fit
    tags: [linear]
    variants:
      narrow:
        features: [16]
        tiers:
          default:
            rows: [500000]
          nightly:
            rows: [1500000]
      wide:
        features: [512]
        tiers:
          default:
            rows: [150000]

  - id: my_kmeans_fitpredict
    algorithm: KMeans
    dataset: blobs
    operation: fit_predict
    dataset_params:
      centers: 8
    tags: [clustering]
    variants:
      medium:
        features: [128]
        tiers:
          default:
            rows: [50000]
```

Save it anywhere and run it with:

```bash
python -m cuml.benchmark --config /path/to/my_benchmarks.yaml --profile default
```

If you also define a `nightly` profile, keep the same algorithm and width-class entries, and increase only `rows` for the companion nightly tiers.

### CLI overrides in config mode

When `--config` is provided, the manifest remains the primary definition of the benchmark suite. CLI flags can still override selected fields for ad hoc runs.

Common useful overrides are:

- `--profile`
- positional algorithm names to restrict which manifest entries run
- `--backends`, `--skip-gpu`, or `--skip-cpu`
- `--num-rows`, `--num-features`, `--input-dimensions`, `--default-size`
- `--n-reps`, `--dtype`, `--input-type`, `--test-split`
- `--param-sweep`, `--cuml-param-sweep`, `--cpu-param-sweep`, `--dataset-param-sweep`

Example:

```bash
python -m cuml.benchmark \
  --config python/cuml/cuml/benchmark/configs/single_gpu.yaml \
  --profile default \
  --backends cpu \
  --num-rows 50000 \
  LogisticRegression
```

This keeps the manifest-selected benchmark entry but overrides the row count and execution mode for that run.

## Input types

With GPU/cuML you can use `--input-type` such as `numpy`, `pandas`, `cupy`, or `cudf`. Without GPU, only `numpy` and `pandas` are valid; the script will warn and switch to `numpy` if needed.

## Manifest structural validation

The benchmark manifest structure is validated by typed `msgspec` models in
`python/cuml/cuml/benchmark/config.py`. The prose in this document explains the
fields and gives examples, while the `msgspec` models provide the structural
contract for parsing YAML manifests.

To generate a JSON Schema from the `msgspec` manifest model:

```bash
python - <<'PY'
import json
from cuml.benchmark.config import benchmark_manifest_json_schema

print(json.dumps(benchmark_manifest_json_schema(), indent=2))
PY
```

Some semantic checks, such as post-default required fields, unknown algorithm
names, and compact `variants` conflicts, are still enforced by custom Python
validation after the structural conversion step.
