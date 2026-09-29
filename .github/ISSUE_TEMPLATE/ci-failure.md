---
name: CI failure
about: Report a failure observed in continuous integration
title: '[CI] Brief description of the failure'
labels: ['ci', 'bug']
assignees: ''
---

## Summary

<!-- Describe what failed and what CI reported. Keep the description objective; avoid speculating about the root cause unless it has been confirmed. -->

**Failing job/test/component:**

**Failure observed in:**
<!-- Link directly to the failing job. Add other jobs if the same failure was observed more than once. -->
- https://github.com/NVIDIA/cuml/actions/runs/XXXX/job/YYYY

## Failure details

<!-- Include a short excerpt of the relevant error, not the full log. Use N/A or remove fields that do not apply. -->

- **Failing step or test:**
- **File or test name (if applicable):**
- **Error message:**
  ```text
  Paste the relevant error or traceback here.
  ```

## Environment

<!-- Copy the failing job's matrix values where available. Omit details that do not apply or are not known. -->

- **Branch or PR:**
- **OS/architecture:**
- **Python:**
- **CUDA:**
- **GPU:**
- **Driver/dependency variant:**

<!-- Add investigation, confirmed causes, fixes, and verification in comments as they become available.

This CI failure issue may be closed once the failure is mitigated, for example by an xfail or skip. If the underlying bug remains unresolved, make sure it is tracked in a separate issue and link that issue when closing this one. -->
