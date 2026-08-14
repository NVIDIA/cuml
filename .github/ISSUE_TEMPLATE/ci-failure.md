---
name: CI Failure
about: Report a continuous integration test failure
title: '[CI] Brief description of the failing test/component'
labels: ['ci', 'bug']
assignees: ''
---

## Summary

<!-- Provide a brief description of the CI failure -->

**Failing test/component:** `[test_name_or_component]`

**Failure observed in:**
- https://github.com/NVIDIA/cuml/actions/runs/XXXX

## Environment

<!-- Fill in the relevant environment details -->

* **OS:** [e.g., ubuntu-20.04, windows-latest, macos-latest]
* **Python version:** [e.g., 3.8, 3.9, 3.10, 3.11]
* **CUDA version:** [if applicable, e.g., 11.8, 12.1]
* **GPU:** [if applicable, e.g., V100, A100, CPU-only]
* **Dependencies:** [e.g., oldest-deps, latest-deps, specific versions]

## Test Details

<!-- Please provide the following information about the test failure. -->

- **Test file:** `[path/to/test_file.py]`
- **Test name:** `[test_function_name]`
- **Error message:**
  ```
  # Paste the relevant error message or traceback here
  ```

<!-- Add any other details about the failure if helpful (e.g., summary, logs, etc.) -->

## Root Cause Analysis

<!-- If known, describe any suspected causes or contributing factors to the failure.
     This section can be left blank if the root cause is unknown. -->
