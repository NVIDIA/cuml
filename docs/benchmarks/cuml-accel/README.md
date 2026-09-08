# Synchronizing the cuml.accel benchmark page

The checked-in `benchmark-data.json` is the compact input for the Sphinx page.
It is produced by `cumlbench-dash`; raw benchmark observations stay outside
this repository. The file stores only case labels, shapes, median timings,
CPU timeout limits when applicable, and PCA component counts that cannot be
derived from labels. Speedups, classifications, summaries, input sizes, and
display units are derived while rendering.

From the repository root, synchronize an updated publication artifact with:

```console
python docs/benchmarks/generate_cuml_accel_benchmarks.py sync \
  --data /path/to/benchmark-data.json
```

The sync command accepts publication schema version 1. The artifact contains
168 cases: the selected 165-case performance grid plus three additional
medium-wide PCA component-rank measurements. The existing medium-wide PCA case
supplies the rank-1,024 point. The command validates and copies the artifact
unchanged, then renders the RST page and SVG heatmaps.

Render the page and heatmaps, or verify that they are current, with:

```console
python docs/benchmarks/generate_cuml_accel_benchmarks.py render
python docs/benchmarks/generate_cuml_accel_benchmarks.py sync --check \
  --data /path/to/benchmark-data.json
python docs/benchmarks/generate_cuml_accel_benchmarks.py render --check
```

`sync --check` validates both the supplied and checked-in artifacts, verifies
that they are byte-for-byte identical, and checks the rendered files without
writing.

Edit `docs/source/cuml-accel/benchmarks.rst.in` for narrative or structural
changes. Do not edit the generated `benchmarks.rst` or SVG files directly.
