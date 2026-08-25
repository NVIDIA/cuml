# Synchronizing the cuml.accel benchmark page

The checked-in `benchmark-data.json` is the portable, non-published input for
the Sphinx page. It is produced and schema-validated by `cumlbench-dash`; raw
mlbench results stay outside this repository.

From the repository root, synchronize an updated publication artifact with:

```console
python docs/benchmarks/generate_cuml_accel_benchmarks.py sync \
  --data /path/to/benchmark-data.json
```

The sync command validates publication schema version 1, content-addressed
source provenance, record source references, and the 147-record transport
shape, including each case's backend-neutral declared estimator parameters. It
copies the artifact unchanged, selects the widest available PCA `fit_transform`
result for PCA's canonical large case, and renders the resulting 145-record RST
page and SVG heatmaps.

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
