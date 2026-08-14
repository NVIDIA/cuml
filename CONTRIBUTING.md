# Contributing to cuML

If you are interested in contributing to cuML, your contributions will fall
into three categories:
1. You want to report a bug, feature request, or documentation issue
    - File an [issue](https://github.com/NVIDIA/cuml/issues/new/choose)
    describing what you encountered or what you want to see changed.
    - Please run and paste the output of the `cuml/print_env.sh` script while
    reporting a bug to gather and report relevant environment details.
    - The RAPIDS team will evaluate the issues and triage them, scheduling
    them for a release. If you believe the issue needs priority attention
    comment on the issue to notify the team.
2. You want to propose a new Feature and implement it
    - Post about your intended feature, and we shall discuss the design and
    implementation.
    - Once we agree that the plan looks good, go ahead and implement it, using
    the [code contributions](#code-contributions) guide below.
3. You want to implement a feature or bug-fix for an outstanding issue
    - Follow the [code contributions](#code-contributions) guide below.
    - If you need more context on a particular issue, please ask and we shall
    provide.

## Code contributions

### Your first issue

1. Read the project's [README.md](https://github.com/NVIDIA/cuml/blob/main/README.md)
    to learn how to setup the development environment.
2. Find an issue to work on. The best way is to look for the [good first issue](https://github.com/NVIDIA/cuml/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
    or [help wanted](https://github.com/NVIDIA/cuml/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) labels
3. Comment on the issue saying you are going to work on it.
4. Get familiar with the developer guide relevant for you:
    * For C++ developers it is available here [DEVELOPER_GUIDE.md](wiki/cpp/DEVELOPER_GUIDE.md)
    * For Python developers, a [Python DEVELOPER_GUIDE.md](wiki/python/DEVELOPER_GUIDE.md) is available as well.
5. Code! Make sure to update unit tests!
6. When done, [create your pull request](https://github.com/NVIDIA/cuml/compare).
7. Verify that CI passes all [status checks](https://help.github.com/articles/about-status-checks/), or fix if needed.
8. Wait for other developers to review your code and update code as needed.
9. Once reviewed and approved, a RAPIDS developer will merge your pull request.

Remember, if you are unsure about anything, don't hesitate to comment on issues and ask for clarifications!


## Code Formatting

Consistent code formatting is important in the cuML project to ensure
readability, maintainability, and thus simplifies collaboration.

### Using pre-commit hooks

cuML uses [pre-commit](https://pre-commit.com) to execute code linters and
formatters that check the code for common issues, such as syntax errors, code
style violations, and help to detect bugs. Using pre-commit ensures that linter
versions and options are aligned for all developers. The same hooks are executed
as part of the CI checks. This means running pre-commit checks locally avoids
unnecessary CI iterations.

To use `pre-commit`, install the tool via `conda` or `pip` into your development
environment:

```console
conda install -c conda-forge pre-commit
```
Alternatively:
```console
pip install pre-commit
```

After installing pre-commit, it is recommended to install pre-commit hooks to
run automatically before creating a git commit. In this way, it is less likely
that style checks will fail as part of CI checks. To install pre-commit hooks,
simply run the following command within the repository root directory:

```console
pre-commit install
```

By default, pre-commit runs on staged files only, meaning only on changes that
are about to be committed. To run pre-commit checks on all files, execute:

```bash
pre-commit run --all-files
```

To skip the checks temporarily, use `git commit --no-verify` or its short form
`-n`.

_Note_: If the auto-formatters' changes affect each other, you may need to go
through multiple iterations of `git commit` and `git add -u`.

cuML also uses [codespell](https://github.com/codespell-project/codespell) to find spelling
mistakes, and this check is run as part of the pre-commit hook. To apply the suggested spelling
fixes, you can run  `codespell -i 3 -w .` from the command-line in the cuML root directory.
This will bring up an interactive prompt to select which spelling fixes to apply.

If you want to ignore errors highlighted by codespell you can:
 * Add the word to the ignore-words-list in pyproject.toml, to exclude for all of cuML
 * Exclude the entire file from spellchecking, by adding to the `exclude` regex in .pre-commit-config.yaml
 * Ignore only specific lines as shown in https://github.com/codespell-project/codespell/issues/1212#issuecomment-654191881

### Summary of pre-commit hooks

The pre-commit hooks configured for this repository consist of a number of
linters and auto-formatters that we summarize here. For a full and current list,
please see the `.pre-commit-config.yaml` file.

- `clang-format`: Formats C++ and CUDA code for consistency and readability.
- `black`: Auto-formats Python code to conform to the PEP 8 style guide.
- `flake8`: Lints Python code for syntax errors and common code style issues.
- `cython-lint`: Lints Cython code for syntax errors and common code style issues.
- _`DeprecationWarning` checker_: Checks for new `DeprecationWarning` being
  introduced in Python code, and instead `FutureWarning` should be used.
- _`#include` syntax checker_: Ensures consistent syntax for C++ `#include` statements.
- _Copyright header checker and auto-formatter_: Ensures the copyright headers
  of files are up-to-date and in the correct format.
- `codespell`: Checks for spelling mistakes

### Clang-tidy

In order to maintain high-quality code, cuML uses not only pre-commit hooks
featuring various formatters and linters but also the clang-tidy tool.
Clang-tidy is designed to detect potential issues within the C and C++ code. It
is typically run as part of our continuous integration (CI) process.

While it's generally unnecessary for contributors to run clang-tidy locally,
there might be cases where you would want to do so. There are two primary
methods to run clang-tidy on your local machine: using Docker or Conda.

* **Docker**

    1. Navigate to the repository root directory.
    2. Run the following Docker command:

        ```bash
        docker run --rm --pull always \
            --mount type=bind,source="$(pwd)",target=/opt/repo --workdir /opt/repo \
            -e SCCACHE_S3_NO_CREDENTIALS=1 \
            rapidsai/ci-conda:26.10-latest /opt/repo/ci/run_clang_tidy.sh
        ```


* **Conda**

    1. Navigate to the repository root directory.
    2. Create and activate the needed conda environment:
        ```bash
        conda env create --yes -n cuml-clang-tidy -f conda/environments/clang_tidy_cuda-133_arch-$(uname -m).yaml
        conda activate cuml-clang-tidy
        ```
    3. Generate the compile command database with
        ```bash
        ./build.sh --configure-only libcuml
        ```
    3. Run clang-tidy with the following command:
        ```bash
        python cpp/scripts/run-clang-tidy.py --config pyproject.toml
        ```

### Managing PR labels

Each PR must be labeled according to whether it is a "breaking" or "non-breaking" change (using Github labels). This is used to highlight changes that users should know about when upgrading.

For cuML, a "breaking" change is one that modifies the public, non-experimental, Python API in a
non-backward-compatible way. The C++ API does not have an expectation of backward compatibility at this
time, so changes to it are not typically considered breaking. Backward-compatible API changes to the Python
API (such as adding a new keyword argument to a function) do not need to be labeled.

Additional labels must be applied to indicate whether the change is a feature, improvement, bugfix, or documentation change. See the shared RAPIDS documentation for these labels: https://github.com/rapidsai/kb/issues/42.

### Seasoned developers

Once you have gotten your feet wet and are more comfortable with the code, you
can look at the prioritized issues of our next release in our [project boards](https://github.com/NVIDIA/cuml/projects).

> **Pro Tip:** Always look at the release board with the highest number for
issues to work on. This is where RAPIDS developers also focus their efforts.

Look at the unassigned issues, and find an issue you are comfortable with
contributing to. Start with _Step 3_ from above, commenting on the issue to let
others know you are working on it. If you have any questions related to the
implementation of the issue, ask them in the issue instead of the PR.

### Branches and Versions

The cuML repository has two main branches:

1. `main` branch: primary development for the next release
2. `release/YY.MM` (e.g. `release/26.02`): release branch for a given release. Once that release is completed, only hotfixes should be targeted at this branch.

### Additional details

For all development, your changes should be pushed into a branch (created using the naming instructions below) in your own fork of cuML and then create a pull request when the code is ready.

PRs should target `main` by default, except in the following situations:

* changes target a soon-to-be-released version: `release/YY.MM`
* hotfixes targeting critical issues: `hotfix/YY.MM.patch-version`

For more details, see https://docs.rapids.ai/releases/process/

### Branch naming

Branches used to create PRs should have a name of the form `<type>-<name>`
which conforms to the following conventions:
- Type:
    - fea - For if the branch is for a new feature(s)
    - enh - For if the branch is an enhancement of an existing feature(s)
    - bug - For if the branch is for fixing a bug(s) or regression(s)
- Name:
    - A name to convey what is being worked on
    - Please use dashes or underscores between words as opposed to spaces.

## Attribution
Portions adopted from https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md
