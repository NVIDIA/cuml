# Upstream Acceleration Tests

Run upstream project tests with `cuml.accel` directly through pytest. From the
repository root, in an active development environment:

```bash
pytest --cuml-suite=accel/sklearn
pytest --cuml-suite=accel/sklearn -k kmeans
pytest --cuml-suite=accel/sklearn --pyargs sklearn.cluster.tests.test_k_means
pytest --cuml-suite=accel/hdbscan
```

The option selects acceleration, upstream warning policy, and the official
xfail list. Explicit targets replace the default suite; `-k` and `-m` filter
after collection. Paths and output filenames remain relative to the caller.
These commands do not install packages. UMAP can obtain missing test sources
with `--cuml-fetch-upstream` as described below; sklearn and HDBSCAN use their
installed tests.

The examples runner is separate and provisions its own source checkout.

Most test suites use a suite-local `xfail-list.yaml` file. The scikit-learn
examples runner is separate from the scikit-learn test-suite runner: it collects
example scripts as pytest tests and uses `scikit-learn/xfail-examples.yaml`.

Run commands below from the repository root in an environment
where `cuml.accel` and the target upstream package are importable.

## Run upstream tests

```bash
pytest --cuml-suite=accel/sklearn --junitxml=report-sklearn.xml
pytest --cuml-suite=accel/umap --cuml-fetch-upstream --junitxml=report-umap.xml
pytest --cuml-suite=accel/hdbscan --junitxml=report-hdbscan.xml
./python/cuml/cuml_accel_tests/upstream/scikit-learn/run-examples.sh --junitxml=report-sklearn-examples.xml
```

Available suites:

| Suite | Selection or runner | Xfail file |
| ----- | ------ | ---------- |
| scikit-learn tests | `--cuml-suite=accel/sklearn` | `scikit-learn/xfail-list.yaml` |
| UMAP tests | `--cuml-suite=accel/umap` | `umap/xfail-list.yaml` |
| HDBSCAN tests | `--cuml-suite=accel/hdbscan` | `hdbscan/xfail-list.yaml` |
| scikit-learn examples | `scikit-learn/run-examples.sh` | `scikit-learn/xfail-examples.yaml` |

UMAP does not ship its tests in the installed package. To allow pytest to obtain
missing sources, opt in explicitly:

```bash
pytest --cuml-suite=accel/umap --cuml-fetch-upstream
```

Without the flag, missing sources produce a usage error with setup guidance.
With the flag, pytest shallow-clones the matching `release-<installed UMAP version>`
tag from GitHub into `umap/umap-upstream`. This requires Git and network access.
Existing valid checkouts are reused offline, even with the flag.

Pytest checks that an existing default checkout is at the matching tag and has
no tracked local changes. It does not reset or update existing checkouts. To
prepare sources in advance or update them after changing the installed version,
run the explicit preparation script (from the repository root):

```bash
./python/cuml/cuml_accel_tests/upstream/umap/prepare-tests.sh
pytest --cuml-suite=accel/umap
```

The preparation script fetches tags and checks out the matching release. Unlike
pytest, it explicitly changes the checkout. Automatic setup failures include
manual preparation and explicit-source commands.

To test another checkout or modified sources, pass its test path explicitly:

```bash
pytest --cuml-suite=accel/umap /path/to/umap-checkout/umap/tests
```

Explicit targets bypass managed-checkout validation and do not enable full-list
xfail validation automatically. Use a compatible source version; the accepted
xfails still come from this repository.

To run a single file from the managed checkout, specify its path relative to
your current directory. For example, from the repository root:

```bash
pytest --cuml-suite=accel/umap python/cuml/cuml_accel_tests/upstream/umap/umap-upstream/umap/tests/test_umap_ops.py
```

The scikit-learn examples runner still clones the matching upstream project tag
into its suite directory when needed.

## CI-like runs

Use bounded xdist parallelism for GPU-backed test runs. Avoid `-n auto` on large
machines because it can spawn too many workers and exhaust GPU memory.

### scikit-learn tests

```bash
pytest --cuml-suite=accel/sklearn \
    --numprocesses=8 \
    --dist=worksteal \
    --junitxml=report-sklearn.xml
```

CI summarizes this report with the scikit-learn test config:

```bash
./python/cuml/cuml_accel_tests/upstream/summarize-results.py \
    --config python/cuml/cuml_accel_tests/upstream/scikit-learn/test_config.yaml \
    report-sklearn.xml
```

### scikit-learn examples

```bash
./python/cuml/cuml_accel_tests/upstream/scikit-learn/run-examples.sh \
    -vv --durations=0 --durations-min=0 \
    -n 4 --dist worksteal \
    --junitxml=report-sklearn-examples.xml
```

### UMAP tests

```bash
pytest --cuml-suite=accel/umap --cuml-fetch-upstream --junitxml=report-umap.xml
```

The current CI wrapper runs UMAP with a 15-minute outer timeout. Add local pytest
filters or bounded xdist options as needed while debugging.

### HDBSCAN tests

```bash
pytest --cuml-suite=accel/hdbscan --junitxml=report-hdbscan.xml
```

HDBSCAN tests can be run locally; there is no dedicated HDBSCAN CI job.

## Debugging tests

Use standard pytest options to debug upstream tests. The examples runner forwards
these options too:

```bash
pytest --cuml-suite=accel/sklearn -k "pattern" --junitxml=report.xml
pytest --cuml-suite=accel/umap -x --tb=short --junitxml=report.xml
pytest --cuml-suite=accel/hdbscan --runxfail --junitxml=report.xml
./python/cuml/cuml_accel_tests/upstream/scikit-learn/run-examples.sh -n 4 --dist worksteal --junitxml=report.xml
```

Useful options:

- `-k "pattern"`: Run tests or examples whose pytest node ID matches a pattern.
- `-x --tb=short`: Stop at the first failure with shorter tracebacks.
- `--runxfail`: Run xfailed tests as ordinary tests to see current behavior.
- `--junitxml=FILE`: Save a JUnit XML report for later analysis.
- `-n N --dist worksteal`: Run with bounded xdist parallelism. Prefer a small
  fixed `N`, especially for GPU-backed tests or when the selected subset is
  small.

The examples runner also supports `--example-timeout=SECONDS` to bound each
example subprocess runtime. Examples that exceed the timeout are reported as
xfailed instead of failed. The default timeout is 1200 seconds.

## scikit-learn examples environment

The scikit-learn examples runner assumes a standard cuML test environment with
`cuml.accel`, scikit-learn, pytest, pytest-xdist, and the usual cuML test
dependencies already installed. The examples suite needs a few additional
plotting and data-loading dependencies that are not required by every upstream
test runner.

After activating the cuML test environment, install the additional dependencies:

```bash
mamba install -c conda-forge "plotly>=6,<7" "polars>=1,<2" "pooch>=1,<2" scikit-image
```

`scikit-learn/run-examples.sh` clones `scikit-learn` into
`scikit-learn/sklearn-upstream` if needed, checks out the tag matching the
installed scikit-learn version, and collects upstream example files as pytest
tests. Expected example failures are read from
`scikit-learn/xfail-examples.yaml`.

To debug expected example failures as ordinary failures:

```bash
./python/cuml/cuml_accel_tests/upstream/scikit-learn/run-examples.sh \
    --runxfail \
    --example-timeout=1200 \
    --junitxml=report-sklearn-examples-runxfail.xml
```

## Analyze results

Generate a summary from a JUnit XML report:

```bash
./python/cuml/cuml_accel_tests/upstream/summarize-results.py -v -f 80 report.xml
```

View tracebacks for specific failures:

```bash
./python/cuml/cuml_accel_tests/upstream/summarize-results.py --format=traceback -k "logistic" report.xml
```

Useful `summarize-results.py` options:

- `-v, --verbose`: Display detailed failure information.
- `-f, --fail-below VALUE`: Set a minimum pass-rate threshold from 0 to 100.
- `--total-fail-below VALUE`: Set a minimum total pass-rate threshold,
  including outcomes excluded by `--exclude-xfail-reason`.
- `--format FORMAT`: Output `summary`, `xfail_list`, or `traceback`.
- `--limit N`: Limit output to the first `N` entries.
- `--test-id-prefix PREFIX`: Prefix added to test IDs in generated output.
- `-k, --filter PATTERN`: Filter tests by ID substring, case-insensitively.
- `--exclude-xfail-reason TEXT`: Exclude xfails whose JUnit reason contains
  `TEXT` from the pass-rate denominator. May be specified multiple times.
- `--config FILE`: Load summary defaults from a config file, such as
  `scikit-learn/test_config.yaml` for scikit-learn tests.

## Xfail lists

Xfail lists mark tests that are expected to fail. They are used to track known
issues, manage version-specific failures, and identify tests that are flaky or
not currently expected to match upstream behavior.

The suite selector supplies the relevant xfail file; the examples runner supplies
its separate list:

- `scikit-learn/xfail-list.yaml` for scikit-learn tests.
- `umap/xfail-list.yaml` for UMAP tests.
- `hdbscan/xfail-list.yaml` for HDBSCAN tests.
- `scikit-learn/xfail-examples.yaml` for scikit-learn examples.

### Generate an xfail list

Generate xfail YAML from a report:

```bash
./python/cuml/cuml_accel_tests/upstream/summarize-results.py --format=xfail_list report.xml > python/cuml/cuml_accel_tests/upstream/umap/xfail-list.yaml
```

Review generated entries before using them. Use `xfail_manager.py` for
formatting and bulk metadata changes so xfail files stay consistently sorted.

### Format

An xfail list is a YAML file containing groups of related expected failures.
Each group can include:

- `reason`: Description of why the tests in this group are expected to fail.
- `marker`: Optional pytest marker name for selecting or filtering tests.
- `condition`: Optional version requirement, such as `scikit-learn>=1.5.2`.
- `strict`: Whether to enforce xfail. The default is `true`.
- `run`: Whether pytest should run the xfailed tests. The default is `true`.
- `tests`: Test IDs in pytest node ID format.

Example:

```yaml
- reason: "Known issues with sparse inputs"
  marker: cuml_accel_sparse_inputs
  tests:
    - "upstream_package.tests.test_sparse::test_sparse_input"
    - "upstream_package.tests.test_sparse::test_sparse_precomputed"

- reason: "Unsupported hyperparameters for older dependency versions"
  condition: "scikit-learn<1.5.2"
  tests:
    - "upstream_package.tests.test_estimator::test_old_version_behavior"

- reason: "Flaky tests due to random seed sensitivity"
  strict: false
  tests:
    - "upstream_package.tests.test_randomized::test_random_seed_sensitive"
```

Use `strict: false` only for tests that are genuinely non-deterministic or
intermittent. Each non-strict group should have a clear reason and should be
reviewed periodically.

### Modify an xfail list

Use `xfail_manager.py` from the repository root:

```bash
# Format an xfail list
./python/cuml/cuml_accel_tests/upstream/xfail_manager.py format python/cuml/cuml_accel_tests/upstream/umap/xfail-list.yaml

# Format the scikit-learn examples xfail file
./python/cuml/cuml_accel_tests/upstream/xfail_manager.py format python/cuml/cuml_accel_tests/upstream/scikit-learn/xfail-examples.yaml

# Set metadata for specific tests
./python/cuml/cuml_accel_tests/upstream/xfail_manager.py set python/cuml/cuml_accel_tests/upstream/hdbscan/xfail-list.yaml \
    "hdbscan.tests.test_hdbscan::test_missing_data" \
    --reason "Known issue with missing-data handling"

# Mark an existing test as flaky
./python/cuml/cuml_accel_tests/upstream/xfail_manager.py set python/cuml/cuml_accel_tests/upstream/scikit-learn/xfail-list.yaml \
    "sklearn.tests.test_example::test_flaky" \
    --no-strict

# Add a version condition
./python/cuml/cuml_accel_tests/upstream/xfail_manager.py set python/cuml/cuml_accel_tests/upstream/umap/xfail-list.yaml \
    "umap.tests.test_umap_ops::test_new_behavior" \
    --reason "Version-specific upstream behavior" \
    --condition "umap-learn>=0.5.8"
```

Before adding tests, check existing reasons and markers so related failures stay
grouped:

```bash
grep "^- reason:" python/cuml/cuml_accel_tests/upstream/scikit-learn/xfail-list.yaml | sort -u
grep "marker:" python/cuml/cuml_accel_tests/upstream/scikit-learn/xfail-list.yaml | sort -u
```

### Version-conditional xfails

Use `--condition` when failures apply only to specific dependency versions:

```bash
./python/cuml/cuml_accel_tests/upstream/xfail_manager.py set python/cuml/cuml_accel_tests/upstream/scikit-learn/xfail-list.yaml \
    "sklearn.tests.test_new_api::test_behavior" \
    --reason "Test uses behavior introduced in scikit-learn 1.8" \
    --condition "scikit-learn>=1.8"

./python/cuml/cuml_accel_tests/upstream/xfail_manager.py set python/cuml/cuml_accel_tests/upstream/umap/xfail-list.yaml \
    "umap.tests.test_umap_ops::test_version_specific" \
    --reason "Version-specific upstream behavior" \
    --condition "umap-learn<=0.5.8 and scikit-learn>=1.6"
```

Conditions can combine multiple version requirements with `and`.

### Handling unmatched test IDs

Focused collection applies matching xfails without requiring unrelated entries
to be collected. Full default upstream collection enables `--strict-xfail-list`
automatically, including when using `-k` or `-m`: validation runs before
deselection. This preserves stale-entry checks in full-suite CI.

For a custom full-suite target, pass `--strict-xfail-list` explicitly. Unknown
active entries produce an `UnmatchedXfailTests` warning, treated as an error by
the upstream policy. The examples runner also retains strict validation.
`--runxfail` changes test execution, not list validation.

Common causes:

- Version-specific tests that only exist for some dependency versions.
- Tests renamed or removed upstream.
- Typographical errors in test IDs.

For focused investigation, select a module or file rather than collecting the
whole suite. Do not remove the official xfail list just to narrow collection,
and do not leave stale unmatched IDs in a committed xfail list.

## Parity workflow

1. Run the relevant upstream suite and write a JUnit XML report for the current
   branch.
2. Summarize the report with `summarize-results.py`.
3. Group failures by root cause, estimator, upstream feature, or dependency
   version.
4. Fix one group at a time, or update the relevant suite-local xfail list when a
   failure is known and intentionally tracked.
5. Re-run the focused test selection, then re-run the broader suite when the
   focused group is resolved.

Keep xfail groups small and descriptive. Prefer suite-local commands and paths
in notes and PR descriptions so it is clear which upstream integration is being
changed.
