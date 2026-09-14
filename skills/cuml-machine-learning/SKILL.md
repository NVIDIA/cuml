---
name: cuml-machine-learning
version: "1.0.0"
description: Use for cuML model design, evaluation, cuml.accel migration, and multi-GPU estimators. Do NOT use for data-only ETL or deep-learning training.
license: Apache-2.0
metadata:
  author: "NVIDIA <opensource@nvidia.com>"
  tags:
    - cuml
    - machine-learning
    - scikit-learn
    - prediction
    - clustering
---

# cuML Machine Learning

## Purpose

Build machine-learning workflows that are statistically valid, measurably useful, and demonstrably GPU-accelerated. This skill covers task design, interface selection, implementation, evaluation, and result handoff for cuML's supervised and unsupervised capabilities.

It does not choose cuML merely because a table is large. Use cuDF for data-only ETL or descriptive aggregation, cuGraph for topology questions, and deep-learning frameworks for neural-network training.

## Prerequisites

- A supported NVIDIA GPU environment with version-matched cuML and RAPIDS dependencies; consult the RAPIDS installation guide for current platform requirements.
- A defined learning unit, feature population, evaluation objective, and—when supervised—target and deployment-aligned split boundary.
- scikit-learn/UMAP/HDBSCAN for the corresponding `cuml.accel` path, or Dask dependencies only for a supported distributed estimator. No API key is required by cuML itself.

## Instructions

## 1. Define the learning task

Before selecting an estimator, write down:

| Item | Supervised task | Unsupervised/representation task |
|---|---|---|
| Unit | One labeled example at a declared grain | One object or observation to group/embed |
| Goal | Target, prediction horizon, and decision use | Structural property the representation should preserve |
| Inputs | Features available at inference time | Features, metric, scaling, and neighborhood meaning |
| Evaluation | Holdout design, baseline, primary metric, cohort gates | Stability plus task-appropriate intrinsic/extrinsic quality |
| Output | Stable keys, prediction/score, uncertainty if used | Stable keys, cluster/embedding/neighbor output |

A successful `fit()` is not evidence that the model answers the intended question.

### Supervised tasks

- Freeze the target definition, row grain, output population, and prediction origin.
- Permit only features available at that origin. Historical event time is not enough if data arrived or was revised later.
- Choose a split that matches deployment: time split for future scoring, group/entity split for unseen entities, stratification for class balance only when it does not violate time/group boundaries.
- Define a simple baseline under the same information boundary.
- Choose metrics from the cost of errors and class/target distribution, not estimator defaults.

### Clustering and dimensionality reduction

- State what similarity means and scale/encode features accordingly.
- Select the method from cluster shape, noise, density, dimensionality, and scale—not from a desire to exercise a GPU.
- Evaluate with multiple signals: stability across seeds/samples, intrinsic scores where meaningful, known-label metrics only when labels are legitimate, and downstream task quality.
- For UMAP/t-SNE, compare neighborhood or trustworthiness quality rather than exact coordinates. Rotations, reflections, label permutations, and stochastic variation can make elementwise parity meaningless.

## 2. Choose the cuML path

### Direct `cuml` Python API

Use for new GPU-native code, explicit estimator control, GPU-resident cuDF/CuPy inputs, or functionality beyond the compatibility layer. cuML estimators follow the familiar fit/predict/transform pattern.

Inputs can include cuDF, CuPy, NumPy, pandas, PyTorch tensors, and other supported array-interface objects. By default, outputs mirror the input type; control this deliberately with `cuml.set_global_output_type`, `cuml.using_output_type`, or estimator output settings when a downstream boundary requires a specific type.

### `cuml.accel`

Use when retaining existing scikit-learn, UMAP, or HDBSCAN code matters. Activate it **before** importing the libraries to accelerate:

```bash
python -m cuml.accel -v training.py
python -m cuml.accel --profile training.py
```

or in a fresh notebook:

```python
%load_ext cuml.accel
```

Supported operations run on GPU; unsupported estimators, methods, parameter combinations, input types, or dependency versions can use the original CPU implementation. A successful run is therefore not proof of GPU execution. Inspect INFO logs or the function/line profiler and retain fallback reasons.

### `cuml.dask`

Use only when the selected estimator exists in the distributed API and a single GPU cannot meet capacity or justified throughput needs. Multi-GPU support covers a subset of direct cuML. Establish a Dask CUDA cluster, keep partitions on workers, and validate a smaller single-GPU equivalent before scaling.

## 3. Build a leakage-safe pipeline

1. Preserve stable business/example keys separately from the numeric feature matrix.
2. Create train, validation, and final test membership from the frozen split contract.
3. Fit every learned transform—imputation, scaling, encoding, feature selection, target encoding—only on the permitted training rows.
4. Fit candidate models and tune on training/validation data. Do not use the final test set to choose features, algorithm, hyperparameters, thresholds, or tolerances.
5. Freeze the selected pipeline, then evaluate once on the final test population.
6. Materialize predictions at the exact key grain and reject missing, extra, duplicate, stale, or non-finite outputs.

`cuml.model_selection.train_test_split` implements a random split and supports `stratify`; it does not create a deployment-aware temporal or entity-group split for you. Build those masks from keys/timestamps before fitting. `KFold` and `StratifiedKFold` are available, but random folds are invalid when rows from the same entity or future period must not cross the boundary.

### Compact direct-cuML example

```python
import cudf
import pandas as pd
from cuml.linear_model import LogisticRegression

FEATURES = ["tenure_days", "orders_90d", "support_cases_30d"]
ORIGIN = pd.Timestamp("2026-04-01")

train_mask = customers["label_available_at"] < ORIGIN
score_mask = customers["score_period"] == ORIGIN

X_train = customers.loc[train_mask, FEATURES].astype("float32")
y_train = customers.loc[train_mask, "churned"]
X_score = customers.loc[score_mask, FEATURES].astype("float32")

model = LogisticRegression()
model.fit(X_train, y_train)
probability = model.predict_proba(X_score)[:, 1]

predictions = customers.loc[score_mask, ["customer_id"]].reset_index(drop=True)
predictions["churn_probability"] = probability
assert not predictions.duplicated(subset=["customer_id"]).any()
assert predictions["churn_probability"].notna().all()
```

The example shows mechanics, not a complete experiment. A held-out labeled set, baseline, threshold policy, and cohort evaluation are still required. See [execution and evaluation patterns](references/execution-and-evaluation-patterns.md).

## 4. Select and compare models responsibly

Check the current API for estimator and parameter support rather than relying on a scikit-learn class name. Direct cuML and `cuml.accel` may use algorithms or solvers that differ from CPU implementations.

- For classification, consider discrimination (ROC AUC or PR AUC), thresholded errors, calibration, and cohort performance as appropriate.
- For regression, report an error metric in target units plus a relative/scale-free metric when useful.
- For clustering, compare partition quality independent of cluster-label integers.
- For embeddings, evaluate neighborhood preservation or downstream usefulness; do not require coordinate equality.
- Report confidence intervals or repeated-split variability when sampling uncertainty can change conclusions.

Use the same feature bytes, split membership, preprocessing, metric implementation, and acceptance gates for CPU/GPU comparisons. Compare model quality, not coefficients or cluster labels by default. Solver order, floating-point reduction order, randomization, and algorithmic differences can produce different fitted attributes while preserving acceptable quality.

Do not dismiss materially worse held-out performance as floating-point noise. Investigate preprocessing, dtype, regularization, solver, convergence, and fallback. Freeze any development-time repair before the final evaluation.

## 5. Verify acceleration and benchmark fairly

For `cuml.accel`, capture profiler/log evidence identifying GPU calls, CPU calls, and fallback reasons. The profiler tracks potentially accelerated methods; a method absent from the table may be unsupported rather than free.

For direct cuML, record package/CUDA/device identity and keep GPU-resident inputs when practical. Synchronize asynchronous GPU work around timing boundaries. Warm up runtime-compiled or initialization-heavy paths before measurement, but do not omit startup from an end-to-end latency claim unless the boundary explicitly excludes it.

Benchmark at representative rows, features, sparsity, and dtype. Small data can be slower on GPU. Report:

- setup, transfer, preprocessing, fit, inference, and metric boundaries;
- warmup and synchronization policy;
- hardware/software versions;
- model-quality gates for every timed arm;
- CPU wins and `cuml.accel` fallbacks rather than hiding them.

A faster run that fails the common quality gate is not an accepted speedup.

## 6. Validate and hand off outputs

For every accepted run, retain:

- source/feature/split identities and stable output keys;
- estimator, parameters, seed, dtype, preprocessing, and output type;
- train/validation/test counts and target/class distributions;
- baseline and model metrics, including declared cohorts;
- acceleration/fallback evidence and measured boundary;
- full keyed predictions, clusters, embeddings, or neighbors;
- package versions and unresolved limitations.

For probabilistic outputs, verify finite values and expected ranges. For multiclass output, retain class order. For cluster output, preserve noise-label semantics and avoid interpreting arbitrary numeric labels as ordered categories.

Model serialization is a security boundary. Only load pickle/joblib artifacts from trusted, integrity-checked sources; these formats can execute code during deserialization. Record the environment needed to restore the model and test predictions after reload.

## Examples

- A monthly classifier uses an availability-aware time split, training-only preprocessing, a baseline, and keyed probabilities; see [execution and evaluation patterns](references/execution-and-evaluation-patterns.md#stable-keys-and-temporal-split).
- A `cuml.accel` migration captures GPU/CPU dispatch and fallback reasons before any speedup claim.
- A clustering comparison uses permutation-invariant quality and stability checks rather than literal cluster IDs.

## Limitations

- Direct cuML, `cuml.accel`, and `cuml.dask` expose different and evolving estimator/parameter coverage; check the installed release before promising an execution path.
- `cuml.accel` may fall back to CPU, and its profiler does not currently observe accelerated calls in subprocesses.
- GPU and CPU algorithms need not produce identical coefficients, labels, or coordinates; acceptance requires a task-specific quality contract.
- This guide does not supply missing labels, eliminate sampling bias, establish causality, or replace deployment monitoring and approval.
- GPU startup, transfer, and compilation overhead can make small workloads slower than CPU execution.

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| Implausibly high test score | Leakage or duplicate entities across split | Audit feature availability and split membership by key/time/group |
| `cuml.accel` shows CPU calls | Unsupported estimator/configuration/input | Read the logged reason; adjust only if semantics permit, or report fallback |
| GPU and CPU coefficients differ | Different solver/algorithm/numerics | Compare held-out quality under identical data and preprocessing |
| GPU is slower | Input too small, transfers, first-call compilation, fallback | Profile end to end, warm up for steady-state claims, retain CPU result |
| Output type surprises downstream code | Input-driven output mirroring | Set and document the required cuDF/CuPy/NumPy output boundary |
| Clusters differ by label number | Labels are arbitrary permutations | Use ARI/AMI or canonicalized memberships, not elementwise label equality |
| OOM | Data/features/model exceed device capacity | Reduce copies/features, use suitable dtypes, or choose a supported distributed estimator |
| Reload is unsafe or fails | Untrusted artifact or environment mismatch | Do not deserialize untrusted files; restore pinned environment and validate inference |

## Deliverable

Return the task contract, selected cuML path, implementation, split/preprocessing/model settings, baseline and quality evidence, acceleration/fallback evidence, keyed outputs, and limitations. Distinguish a retrospective experiment from a deployment-ready model when point-in-time availability, calibration, monitoring, or approval is unresolved.

## References

- [Execution and evaluation patterns](references/execution-and-evaluation-patterns.md)
- [cuML documentation](https://docs.nvidia.com/cuml/)
- [cuML API reference](https://docs.nvidia.com/cuml/api/)
- [`cuml.accel` compatibility](https://docs.nvidia.com/cuml/cuml-accel/compatibility/)
- [`cuml.accel` logging and profiling](https://docs.nvidia.com/cuml/cuml-accel/logging-and-profiling/)
