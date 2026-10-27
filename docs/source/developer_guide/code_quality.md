# Code quality checks

Install and use cuML's [pre-commit](https://pre-commit.com) hooks to check
formatting, lint, and spelling before submitting changes. CI runs the same
hooks. The repository's
[`.pre-commit-config.yaml`](https://github.com/NVIDIA/cuml/blob/main/.pre-commit-config.yaml)
is the source of truth for the configured checks, versions, and file selection.

For contribution requirements, see
[`CONTRIBUTING.md`](https://github.com/NVIDIA/cuml/blob/main/CONTRIBUTING.md).

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

## Clang-tidy

CI runs clang-tidy to detect potential C++ issues beyond the pre-commit checks.
Running it locally is optional, but useful when investigating CI failures.
Use either Docker or Conda on a Linux development machine with the build
prerequisites described in
[`BUILD.md`](https://github.com/NVIDIA/cuml/blob/main/BUILD.md).
Run the commands below from the repository root.

### Docker

Use the CI image matching the branch's cuML version (shown here for 26.12):

```bash
docker run --rm --pull always \
    --mount type=bind,source="$(pwd)",target=/opt/repo --workdir /opt/repo \
    -e SCCACHE_S3_NO_CREDENTIALS=1 \
    rapidsai/ci-conda:26.12-latest /opt/repo/ci/run_clang_tidy.sh
```

The CI script creates its environment, configures the build, and runs
clang-tidy.

### Conda

Choose an existing `clang_tidy_*.yaml` environment file from
`conda/environments/` matching your CUDA version and architecture. For example,
on Linux x86_64 with CUDA 13.3:

```bash
conda env create -n cuml-clang-tidy \
    -f conda/environments/clang_tidy_cuda-133_arch-x86_64.yaml
conda activate cuml-clang-tidy
./build.sh --configure-only libcuml
python cpp/scripts/run-clang-tidy.py --config pyproject.toml
```

The configure step generates the compilation database used by clang-tidy.

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
