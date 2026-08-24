# Regenerating the cuml.accel benchmark page

The checked-in `benchmark-data.json` is the compact, non-published input for
the Sphinx page. Raw mlbench results stay outside the documentation tree.

From the repository root, import updated paired results with:

```console
python docs/benchmarks/generate_cuml_accel_benchmarks.py import \
  --reference /path/to/core/cpu.json \
  --candidate /path/to/core/accelerated.json \
  --supplemental-reference /path/to/pca/cpu.json \
  --supplemental-candidate /path/to/pca/accelerated.json
```

The import validates case pairing, backend identity, package and image parity,
complete-case timeout policy, repetition counts, correctness completion,
GPU-only accelerated execution, and the PCA publication gate. It then replaces
the operation-specific PCA `large` cell with the validated 2,048-feature case.

Render the page and heatmaps, or verify that they are current, with:

```console
python docs/benchmarks/generate_cuml_accel_benchmarks.py render
python docs/benchmarks/generate_cuml_accel_benchmarks.py import --check \
  --reference /path/to/core/cpu.json \
  --candidate /path/to/core/accelerated.json \
  --supplemental-reference /path/to/pca/cpu.json \
  --supplemental-candidate /path/to/pca/accelerated.json
python docs/benchmarks/generate_cuml_accel_benchmarks.py render --check
```

Edit `docs/source/cuml-accel/benchmarks.rst.in` for narrative or structural
changes. Do not edit the generated `benchmarks.rst` or SVG files directly.
