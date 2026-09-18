---
name: Bug report
about: Create a report to help us improve cuML
title: "[BUG]"
labels: "? - Needs Triage, bug"
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**User impact and discovery**
Describe how this affects your use of cuML, including whether it blocks current
work and the scale or frequency of the impact. If you found the problem through
automated analysis, fuzzing, systematic testing, or another process rather
than while using cuML, say so. This context helps maintainers assess impact and
prioritize work.

**Steps/Code to reproduce bug**
Follow this guide http://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports to craft a minimal bug report. This helps us reproduce the issue you're having and resolve the issue more quickly.

**Expected behavior**
A clear and concise description of what you expected to happen.

**Environment details (please complete the following information):**
 - Environment location: [Bare-metal, Docker, Cloud(specify cloud provider)]
 - Linux Distro/Architecture: [Ubuntu 16.04 amd64]
 - GPU Model/Driver: [V100 and driver 396.44]
 - CUDA: [9.2]
 - Method of cuDF & cuML install: [conda, Docker, or from source]
   - If method of install is [conda], run `conda list` and include results here
   - If method of install is [Docker], provide `docker pull` & `docker run` commands used
   - If method of install is [from source], provide versions of `cmake` & `gcc/g++` and commit hash of build

**Additional context**
Add any other context about the problem here.

<!--
By submitting this report, you confirm that you reviewed and understand its
contents and will respond personally to maintainer questions. Fully autonomous
submissions may be closed without investigation. See CONTRIBUTING.md.
-->
