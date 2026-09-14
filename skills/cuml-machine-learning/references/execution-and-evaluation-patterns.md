# cuML Execution and Evaluation Patterns

These patterns are grounded in the current cuML source tree. Check the installed API and [`cuml.accel` compatibility documentation](https://docs.nvidia.com/cuml/cuml-accel/compatibility/) because estimator coverage, fallback conditions, and supported dependency versions evolve.

## Path selection

| Situation | Preferred path |
|---|---|
| New GPU-native Python workflow with cuDF/CuPy data | Direct `cuml` |
| Existing scikit-learn, UMAP, or HDBSCAN application | `cuml.accel` |
| Selected estimator exceeds one GPU and has distributed support | `cuml.dask` |
| Deep neural network training | A deep-learning framework, not cuML |
| Data preparation without a learning task | cuDF |

Do not rewrite working scikit-learn code before profiling `cuml.accel`. Do not stay on the compatibility layer when the task needs direct cuML-only controls or when repeated CPU fallback dominates the hot path.

## Activate and observe `cuml.accel`

Activation must precede imports of the intercepted libraries.

```bash
# INFO dispatch messages
python -m cuml.accel -v pipeline.py

# Per-method GPU/CPU call counts, time, and fallback reasons
python -m cuml.accel --profile pipeline.py

# Per-line attribution; use for diagnosis, not timing comparison
python -m cuml.accel --line-profile pipeline.py
```

Notebook:

```python
%load_ext cuml.accel
%cuml.accel.log_level info
```

or:

```python
import cuml
cuml.accel.install(log_level="info")  # before importing sklearn/umap/hdbscan
```

The function and line profilers do not currently track GPU calls made in subprocesses. Avoid `n_jobs > 1` while collecting profiler evidence, or disclose the blind spot. Line profiling adds overhead and must not be used as the benchmark timing.

A listed estimator can still fall back because of a method, hyperparameter, input type, or dependency version. An estimator not represented by the profiler may be entirely unsupported, not absent from runtime cost.

## Stable keys and temporal split

Do not put identifiers into the feature matrix merely to recover row identity later.

```python
import pandas as pd

ORIGIN = pd.Timestamp("2026-07-01")
FEATURES = ["balance", "payments_30d", "days_since_contact"]

train = (
    (examples["label_time"] < ORIGIN)
    & (examples["available_at"] < ORIGIN)
)
test = examples["prediction_origin"] == ORIGIN

X_train = examples.loc[train, FEATURES].astype("float32")
y_train = examples.loc[train, "target"]
X_test = examples.loc[test, FEATURES].astype("float32")
test_keys = examples.loc[test, ["account_id", "prediction_origin"]].reset_index(
    drop=True
)
```

Check that keys are unique at the declared example grain and that no entity crosses a group holdout when the deployment scenario is unseen entities. A random `train_test_split` cannot express this time contract by itself.

## Fit preprocessing only on training rows

```python
from cuml.preprocessing import SimpleImputer, StandardScaler

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_train_imputed = imputer.fit_transform(X_train)
X_train_scaled = scaler.fit_transform(X_train_imputed)

X_test_imputed = imputer.transform(X_test)
X_test_scaled = scaler.transform(X_test_imputed)
```

Never call `fit_transform` separately on test data. Target encoding and feature selection are learned transforms too. If cross-validation is used, fit each transform inside each training fold rather than once on the full development set.

The `cuml.pipeline` namespace exposes scikit-learn's `Pipeline` and `make_pipeline`; the `cuml.compose` namespace exposes compatible column-transformer helpers. Verify estimator/transformer interoperability for the installed versions and include the entire fitted pipeline in evaluation and persistence.

## Classification evaluation

```python
from cuml.linear_model import LogisticRegression
from cuml.metrics import accuracy_score, roc_auc_score

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

p_test = model.predict_proba(X_test_scaled)[:, 1]
yhat_test = model.predict(X_test_scaled)

auc = float(roc_auc_score(y_test, p_test))
accuracy = float(accuracy_score(y_test, yhat_test))

predictions = test_keys.copy()
predictions["score"] = p_test
predictions["prediction"] = yhat_test

assert len(predictions) == len(test_keys)
assert not predictions.duplicated(
    subset=["account_id", "prediction_origin"]
).any()
assert predictions[["score", "prediction"]].notna().all().all()
assert ((predictions["score"] >= 0) & (predictions["score"] <= 1)).all()
```

Accuracy by itself can conceal failure on an imbalanced class. Choose additional measures from the use case: PR AUC, confusion-matrix counts at a predeclared threshold, calibration, and cohort metrics. Fit and evaluate a baseline on the same rows and under the same information boundary.

For multiclass probabilities, retain the model's class ordering with the output columns. Verify every column position against an explicit business-label mapping.

## Regression evaluation

Use a baseline such as training mean/median or the accepted incumbent model, then compare on identical test rows.

```python
from cuml.linear_model import Ridge
from cuml.metrics import mean_absolute_error, r2_score

model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

mae = float(mean_absolute_error(y_test, y_pred))
r2 = float(r2_score(y_test, y_pred))
```

Report MAE or RMSE in target units and inspect error by decision-relevant cohorts. R² alone does not communicate operational error magnitude.

## Clustering comparison

Cluster IDs are nominal labels and may be permuted. Compare partitions with permutation-invariant metrics such as adjusted Rand index or adjusted mutual information when reference labels are valid. Without labels, combine intrinsic quality with stability and domain checks.

```python
from cuml.cluster import KMeans
from cuml.metrics.cluster import adjusted_rand_score

first = KMeans(n_clusters=8, random_state=7).fit_predict(X)
second = KMeans(n_clusters=8, random_state=19).fit_predict(X)
stability = float(adjusted_rand_score(first, second))
```

Confirm exact metric availability in the installed release. A high intrinsic score does not prove business usefulness, and a low seed-to-seed agreement is evidence that the partition is unstable.

For density clustering, preserve the documented noise label rather than treating it as an ordinary ordered cluster. Validate sensitivity to distance metric, scaling, and neighborhood parameters.

## Embedding comparison

For UMAP/t-SNE/PCA migrations:

- compare explained variance where applicable;
- compare trustworthiness or neighbor recall for nonlinear embeddings;
- evaluate downstream model quality if that is the purpose;
- test stability under allowed randomness;
- never require exact coordinate equality for methods invariant to rotation/reflection or using different parallel algorithms.

`cuml.accel` documents estimator-specific differences and fallbacks. For example, some methods ignore CPU-oriented parameters or support only selected metrics. Treat the installed compatibility page as the current source of truth.

## Output type control

cuML normally mirrors the input type in outputs. Keep the default when it makes downstream GPU composition easy; otherwise make the boundary explicit.

```python
import cuml

with cuml.using_output_type("cudf"):
    labels = model.predict(X_gpu)
```

Global output-type changes affect subsequent operations. Prefer a scoped context manager in libraries and tests to avoid hidden process-wide state.

## Fair CPU/GPU comparison

Freeze before timing:

- exact train/test key membership and feature column order;
- preprocessing and its fitted training population;
- estimator objective, effective regularization, and parameters;
- dtype and missing/category policy;
- metric code and acceptance thresholds;
- timing boundary, warmup, synchronization, and repetitions.

GPU and CPU estimators can differ algorithmically. Compare task quality rather than expecting fitted coefficients, tree structures, cluster IDs, or embedding coordinates to match. If quality fails, the GPU timing is diagnostic—not an accepted speedup.

For `cuml.accel`, preserve profile output separately from benchmark output: profiling adds overhead. Include data transfer and fallback in end-to-end timings. For steady-state fit/inference timings, warm up and synchronize GPU work around each measurement.

## Multi-GPU gate

Before selecting `cuml.dask`:

1. Verify the estimator exists in the current distributed API.
2. Demonstrate the single-GPU capacity or throughput reason.
3. Define Dask cluster, partition count/size, worker/GPU mapping, and data residency.
4. Validate quality and keyed output on a tractable single-GPU slice.
5. Measure scheduler, communication, fit, inference, and collection boundaries.

Do not collect a result larger than one GPU into a single local object. Persist distributed output or aggregate it at a safe grain.

## Artifact safety and replay

Persist the fitted preprocessing and estimator together with feature order, package versions, dtype, and training/split identity. Re-read the saved artifact in the target environment and compare predictions on a fixed fixture.

Pickle, joblib, and cloudpickle loading can compromise the process when an artifact is malicious. Deserialize only trusted, integrity-checked artifacts. A hash checks that bytes did not change; it does not make an untrusted artifact safe or prove model quality.
