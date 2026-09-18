# Code quality checks

Install and use cuML's [pre-commit](https://pre-commit.com) hooks to check
formatting, lint, and spelling before submitting changes. CI runs the same
hooks. The repository's
[`.pre-commit-config.yaml`](https://github.com/NVIDIA/cuml/blob/main/.pre-commit-config.yaml)
is the source of truth for the configured checks, versions, and file selection.

For contribution requirements, see
[`CONTRIBUTING.md`](https://github.com/NVIDIA/cuml/blob/main/CONTRIBUTING.md).
For local clang-tidy checks, see the
[C++ and CUDA Developer Guide](cpp/development.md#clang-tidy).

## Install pre-commit hooks

Install pre-commit in your development environment with either Conda or pip:

```bash
conda install -c conda-forge pre-commit
```

Alternatively:

```bash
pip install pre-commit
```

From the repository root, install the Git hooks:

```bash
pre-commit install
```

The hooks run automatically on staged files when you commit. Review any
formatter changes, stage them, and retry the commit. If checks modify files
again, repeat until they pass.

## Run checks manually

From the repository root, check specific files while iterating:

```bash
pre-commit run --files path/to/changed_file.py path/to/another_file.md
```

To check all files:

```bash
pre-commit run --all-files
```

Some hooks inspect repository-wide configuration rather than only the selected
files. Read the hook output, fix reported problems, and rerun the checks before
requesting review. Formatting and lint checks do not replace the tests relevant
to your change.

## Spelling checks

The hooks run [codespell](https://github.com/codespell-project/codespell) to
check spelling. To interactively apply suggested fixes, install codespell in
your development environment and run it from the repository root:

```bash
codespell --toml pyproject.toml -i 3 -w path/to/changed_file.md
```

Review suggested edits before committing. For false positives, use the
narrowest appropriate exception:

- Ignore a specific line using a
  [codespell inline directive](https://github.com/codespell-project/codespell#ignoring-words).
- Add a legitimate project-wide term to `ignore-words-list` in the root
  `pyproject.toml`.
- Exclude a file from the codespell hook in `.pre-commit-config.yaml` only when
  checking the file is inappropriate, such as generated content.
