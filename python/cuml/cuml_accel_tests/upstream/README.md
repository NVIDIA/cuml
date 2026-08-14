# Upstream Acceleration Tests

This directory contains wrappers for running upstream project test suites under
`cuml.accel`. Each upstream integration has a wrapper script under
`python/cuml/cuml_accel_tests/upstream/<suite>/`, and pytest arguments passed to
the wrapper are forwarded to pytest.

Most test suites use a suite-local `xfail-list.yaml` file. The scikit-learn
examples runner is separate from the scikit-learn test-suite runner: it collects
example scripts as pytest tests and uses `scikit-learn/xfail-examples.yaml`.

Run commands below from `python/cuml/cuml_accel_tests/upstream` in an environment
where `cuml.accel` and the target upstream package are importable.

## Run upstream tests

```bash
./scikit-learn/run-tests.sh --junitxml=report-sklearn.xml
./umap/run-tests.sh --junitxml=report-umap.xml
./hdbscan/run-tests.sh --junitxml=report-hdbscan.xml
./scikit-learn/run-examples.sh --junitxml=report-sklearn-examples.xml
```

Available runners:

| Suite | Runner | Xfail file |
| ----- | ------ | ---------- |
| scikit-learn tests | `scikit-learn/run-tests.sh` | `scikit-learn/xfail-list.yaml` |
| UMAP tests | `umap/run-tests.sh` | `umap/xfail-list.yaml` |
| HDBSCAN tests | `hdbscan/run-tests.sh` | `hdbscan/xfail-list.yaml` |
| scikit-learn examples | `scikit-learn/run-examples.sh` | `scikit-learn/xfail-examples.yaml` |

The UMAP and scikit-learn examples runners clone the matching upstream project
tag into their suite directory when needed.

## CI-like runs

Use bounded xdist parallelism for GPU-backed test runs. Avoid `-n auto` on large
machines because it can spawn too many workers and exhaust GPU memory.

### scikit-learn tests

```bash
./scikit-learn/run-tests.sh \
    --numprocesses=8 \
    --dist=worksteal \
    --junitxml=report-sklearn.xml
```

CI summarizes this report with the scikit-learn test config:

```bash
./summarize-results.py \
    --config scikit-learn/test_config.yaml \
    report-sklearn.xml
```

### scikit-learn examples

```bash
./scikit-learn/run-examples.sh \
    -vv --durations=0 --durations-min=0 \
    -n 4 --dist worksteal \
    --junitxml=report-sklearn-examples.xml
```

### UMAP tests

```bash
./umap/run-tests.sh --junitxml=report-umap.xml
```

The current CI wrapper runs UMAP with a 15-minute outer timeout. Add local pytest
filters or bounded xdist options as needed while debugging.

### HDBSCAN tests

```bash
./hdbscan/run-tests.sh --junitxml=report-hdbscan.xml
```

This repository has a local HDBSCAN upstream runner. At the time this README was
updated, there was no dedicated CI script invoking it.

## Debugging runners

All wrappers forward arguments to pytest, so the same basic debugging options
work across the pytest-based runners:

```bash
./scikit-learn/run-tests.sh -k "pattern" --junitxml=report.xml
./umap/run-tests.sh -x --tb=short --junitxml=report.xml
./hdbscan/run-tests.sh --runxfail --junitxml=report.xml
./scikit-learn/run-examples.sh -n 4 --dist worksteal --junitxml=report.xml
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
./scikit-learn/run-examples.sh \
    --runxfail \
    --example-timeout=1200 \
    --junitxml=report-sklearn-examples-runxfail.xml
```

## Analyze results

Generate a summary from any wrapper's JUnit XML report:

```bash
./summarize-results.py -v -f 80 report.xml
```

View tracebacks for specific failures:

```bash
./summarize-results.py --format=traceback -k "logistic" report.xml
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

The upstream wrappers pass the relevant xfail file to pytest:

- `scikit-learn/xfail-list.yaml` for scikit-learn tests.
- `umap/xfail-list.yaml` for UMAP tests.
- `hdbscan/xfail-list.yaml` for HDBSCAN tests.
- `scikit-learn/xfail-examples.yaml` for scikit-learn examples.

### Generate an xfail list

Generate xfail YAML from a report:

```bash
./summarize-results.py --format=xfail_list report.xml > umap/xfail-list.yaml
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

Use `xfail_manager.py` from this directory:

```bash
# Format an xfail list
./xfail_manager.py format umap/xfail-list.yaml

# Format the scikit-learn examples xfail file
./xfail_manager.py format scikit-learn/xfail-examples.yaml

# Set metadata for specific tests
./xfail_manager.py set hdbscan/xfail-list.yaml \
    "hdbscan.tests.test_hdbscan::test_missing_data" \
    --reason "Known issue with missing-data handling"

# Mark an existing test as flaky
./xfail_manager.py set scikit-learn/xfail-list.yaml \
    "sklearn.tests.test_example::test_flaky" \
    --no-strict

# Add a version condition
./xfail_manager.py set umap/xfail-list.yaml \
    "umap.tests.test_umap_ops::test_new_behavior" \
    --reason "Version-specific upstream behavior" \
    --condition "umap-learn>=0.5.8"
```

Before adding tests, check existing reasons and markers so related failures stay
grouped:

```bash
grep "^- reason:" scikit-learn/xfail-list.yaml | sort -u
grep "marker:" scikit-learn/xfail-list.yaml | sort -u
```

### Version-conditional xfails

Use `--condition` when failures apply only to specific dependency versions:

```bash
./xfail_manager.py set scikit-learn/xfail-list.yaml \
    "sklearn.tests.test_new_api::test_behavior" \
    --reason "Test uses behavior introduced in scikit-learn 1.8" \
    --condition "scikit-learn>=1.8"

./xfail_manager.py set umap/xfail-list.yaml \
    "umap.tests.test_umap_ops::test_version_specific" \
    --reason "Version-specific upstream behavior" \
    --condition "umap-learn<=0.5.8 and scikit-learn>=1.6"
```

Conditions can combine multiple version requirements with `and`.

### Handling unmatched test IDs

The pytest plugin validates that xfail entries correspond to collected tests.
When tests do not exist, an `UnmatchedXfailTests` warning is issued and the
shared pytest config treats that warning as an error.

Common causes:

- Version-specific tests that only exist for some dependency versions.
- Tests renamed or removed upstream.
- Typographical errors in test IDs.

For temporary local investigation, remove the pytest argument or warning filter
that elevates unmatched xfails to an error. Do not leave stale unmatched IDs in a
committed xfail list.

## Parity workflow

1. Run the relevant upstream wrapper and write a JUnit XML report for the current
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
