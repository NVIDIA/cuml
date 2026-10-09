---
name: cuml
description: >-
  Use when writing, running, debugging or upgrading Python code that uses NVIDIA cuML
  (GPU machine learning), cuml.accel (zero-code-change GPU acceleration of scikit-learn,
  umap-learn and hdbscan), or GPU UMAP/HDBSCAN/RandomForest/KMeans/SVC, including through
  libraries that wrap cuML (BERTopic, rapids-singlecell). Covers APIs removed or renamed in
  2025-2026 releases, verifying code really ran on the GPU, large-data UMAP/HDBSCAN memory
  recipes, and exporting trained models. Not for cuDF-only dataframe work, PyTorch or XGBoost.
---

# cuML

cuML releases every two months and removes deprecated APIs, so your training data is likely
to contain APIs that no longer exist. Look things up in the installed cuML instead of
recalling them.

## Ground facts

- Install from pypi.org: `pip install cuml-cu13` (CUDA 13 driver, >= 580) or `cuml-cu12`.
  No extra index URL is needed. `pip install cuml` installs an unrelated placeholder.
- Linux only; on Windows use WSL2.
- Google Colab GPU runtimes ship cuML preinstalled: check `pip list` before installing anything there.
- `cuml.accel` runs unmodified scikit-learn / umap-learn / hdbscan code on the GPU and falls
  back to the CPU for anything unsupported. It accelerates RandomForest, LogisticRegression,
  Ridge, KMeans, DBSCAN, PCA, SVC, KNN, UMAP, HDBSCAN, scalers, encoders and more.
- Enable it with `python -m cuml.accel script.py`, `%load_ext cuml.accel` (first notebook
  cell), `CUML_ACCEL_ENABLED=1`, or `cuml.accel.install()` before importing sklearn.

## Step 1: look it up in the installed cuML

- Check the versions and answer for them:
  `python -c "import cuml, sklearn; print(cuml.__version__, sklearn.__version__)"`.
- Before using a cuML class, function or argument, confirm it exists and read its
  documentation: `python -c "import inspect, cuml; print(inspect.signature(cuml.svm.SVC))"`,
  `help(cuml.manifold.UMAP)`. Docstrings list valid values, for example the accepted
  `build_kwds` keys and `output_type` values.
- Run the code and read the errors: removed arguments raise `TypeError`, removed modules
  `ModuleNotFoundError`, invalid values usually list the valid ones.
- Treat deprecated APIs as removed: do not recommend anything that emits a `FutureWarning` or
  is marked deprecated in its docstring, even if it still works in the installed version.
  Deprecation warnings only appear when the code runs, so run the code you suggest (on small
  data if needed) with `python -W always::FutureWarning` and read the warnings; they name the
  replacement.
- For `cuml.accel`, see below for how to find out what runs on the GPU.

## Step 2: check code for removed APIs

Run `python scripts/check_removed_apis.py <files or dirs>` (path relative to this skill) on
the user's code and on code you write, before presenting it. It reports removed APIs whose
replacement the installed cuML does not reveal, with the replacement, for example
`cuml.fil` (use nvForest), `SVC(probability=True)` (use `CalibratedClassifierCV`) and
renamed UMAP/HDBSCAN `build_kwds` keys that are silently ignored.

The checker only knows about these few APIs. An empty result does not mean the code is
current, and it says nothing about whether `cuml.accel` runs it on the GPU; still do Step 1
and check `cuml.accel` fallbacks as described below.

Behavior that changed without an API change:
- RandomForest is reproducible with `random_state` alone (since 25.10). Do not set
  `n_streams=1` "for determinism"; that advice is outdated and only slows training.
- `KMeans.transform` returns Euclidean, not squared, distances (since 26.10).

## cuml.accel

- Activate before sklearn is imported. In a notebook that already imported sklearn,
  restart the kernel and make `%load_ext cuml.accel` the first cell.
- With cudf.pandas: `python -m cudf.pandas -m cuml.accel script.py`, or
  `%load_ext cudf.pandas` then `%load_ext cuml.accel`.
- Prove where code ran instead of assuming: `python -m cuml.accel -v script.py` logs
  "ran on GPU" / "falling back to CPU: <reason>"; `--profile` (or `%%cuml.accel.profile`,
  or `with cuml.accel.profile():`) prints a per-call GPU/CPU table.
  `cuml.accel.is_proxy(est)` only says the estimator is wrapped, not that it ran on the GPU.
- Unsupported parameters and inputs fall back to the CPU silently. Do not recall fallback
  lists: run with `-v` or `--profile` and read the reasons, or, without running, read the
  documentation for the installed release from GitHub:
  `https://raw.githubusercontent.com/NVIDIA/cuml/v<cuml.__version__>/docs/source/cuml-accel/compatibility.rst`
  (for example `v26.08.00`; for 26.06 and earlier the file is `limitations.rst`; for nightly
  builds such as `26.12.00a5` use `main` instead of the tag).
- Single GPU only. To use several GPUs, run separate processes with `CUDA_VISIBLE_DEVICES`.
- Models trained under accel pickle as plain sklearn/umap/hdbscan objects and load without
  cuML. The pickle bytes still contain the string `cuml` (a reconstructor), so do not use
  "no `cuml` in the bytes" as a check; load it in a CPU-only environment instead.
- Results are comparable in quality, not numerically identical, to the CPU libraries.
  See `references/parity-and-numerics.md` before treating a difference as a bug.

## Large data (UMAP, HDBSCAN, KMeans)

- UMAP `build_algo="auto"` already uses NN-Descent above 50,000 rows (unless `random_state`
  is set, which forces brute force). HDBSCAN defaults to `build_algo="brute_force"`.
- Out of memory on the kNN build: batch it with
  `build_algo="nn_descent", build_kwds={"knn_n_clusters": 4, "knn_overlap_factor": 2}`
  (increase `knn_n_clusters` to 8, 16, ... to use less memory). Unknown keys are ignored
  without a warning, so spell them exactly.
- Oversubscribe GPU memory with managed memory:
  `import rmm; rmm.reinitialize(managed_memory=True)` before creating estimators.
  `cuml.accel` enables managed memory by default (not on WSL2).
- KMeans on host data larger than the GPU: `KMeans(device_buffer_samples=...)` streams host
  batches (26.08+).
- HDBSCAN soft clustering: fit with `prediction_data=True`, then use
  `cuml.cluster.hdbscan.all_points_membership_vectors`, `membership_vector` and
  `approximate_predict`. There is no `HDBSCAN.predict` and no `.prediction` submodule.

## Exporting models

| Have | Want | Use |
|---|---|---|
| cuML estimator | scikit-learn object for a CPU-only machine | `est.as_sklearn()` |
| scikit-learn estimator | cuML estimator | `cuml.<Estimator>.from_sklearn(skl)` |
| cuML RandomForest | Treelite / fast GPU or CPU inference | `rf.as_treelite()`, `rf.as_nvforest()` |
| sklearn / XGBoost / LightGBM forest | fast inference | `nvforest.load_from_sklearn(model)`, `nvforest.load_model(path)` |
| accel-trained model | ONNX | `skl2onnx` on the proxy (not DBSCAN, TSNE, UMAP, HDBSCAN, NearestNeighbors) |

## Outputs

Outputs mirror the input container: NumPy in, NumPy out; cuDF in, cuDF out. With cuDF or
dask_cudf input, fitted attributes such as `cluster_centers_` are cuDF objects, so call
`.to_pandas()` / `.to_numpy()` before handing them to pandas or matplotlib. Force a type
with `output_type="cupy"` on the estimator, `cuml.set_global_output_type("cupy")`, or
`with cuml.using_output_type("cupy"):` (global and context settings take precedence).
Non-float input is converted to float32 by default.
