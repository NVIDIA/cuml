# Contributing to cuML

Contribute to cuML by reporting problems, proposing improvements, or submitting
code and documentation changes. Start by describing the problem and agreeing
on the scope; use the developer guides for implementation details.

## Report a bug or request a change

Open an [issue](https://github.com/NVIDIA/cuml/issues/new/choose) using the
appropriate template. For bugs, include a minimal reproducer, expected and
actual behavior, and relevant environment details, including the output of
[`print_env.sh`](print_env.sh) from the repository root.

Explain the user impact: how the problem affects your use of cuML. If you found
it through automated analysis rather than actual use, say so. The cuML team
uses this context to triage and prioritize issues. If an issue needs priority
attention, comment with concrete impact information.

## Agree on the scope

Discuss new features in an issue and agree on the design and implementation
plan with maintainers before starting work. For an existing issue, comment
with your proposed scope and ask to be assigned. Wait for assignment before
starting work, and ask on the issue if you need clarification.

Except for trivial changes, pull requests should close an issue to which the
author is assigned. Maintainers may close a PR without review if it does not
meet this requirement.

To find a first contribution, look for issues labeled
[good first issue](https://github.com/NVIDIA/cuml/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
or [help wanted](https://github.com/NVIDIA/cuml/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22).

## Prepare your contribution

Work on a branch in your own fork. Follow the [build-from-source guide](BUILD.md)
to set up your development environment, then consult the relevant guides:

- [Python development](docs/source/developer_guide/python/development.md)
- [Python estimator development](docs/source/developer_guide/python/estimators.md)
- [C++ and CUDA development](docs/source/developer_guide/cpp/development.md)

Keep changes focused on the agreed scope. Add or update tests for changed
behavior and update affected documentation. Install and use the
[pre-commit hooks](docs/source/developer_guide/code_quality.md) to check
formatting, lint, and spelling. Run relevant tests and checks locally before
requesting review; identify anything you could not validate.

## Open a pull request and work through review

Open a [pull request](https://github.com/NVIDIA/cuml/compare) when your change
is ready for review. Follow the PR template: link the issue it closes, explain
the problem and key implementation choices, and summarize the validation
performed and any gaps.

### Target branch

Target `main` by default. Changes for a soon-to-be-released version target
`release/YY.MM`; critical hotfixes target `hotfix/YY.MM.patch-version`. Once a
release is complete, its release branch is for hotfixes only. See the
[RAPIDS release process](https://docs.nvidia.com/datascience/releases/process/)
for details, and ask maintainers if the appropriate target is unclear.

### PR labels

Each PR needs the labels described in the
[RAPIDS label checker documentation](https://docs.nvidia.com/datascience/resources/label-checker/):
a `breaking` or `non-breaking` label and a label identifying it as a feature,
improvement, bugfix, or documentation change. If you cannot apply labels,
comment on the PR to request them.

A breaking change modifies the public, non-experimental Python API in a
backward-incompatible way. Backward-compatible additions do not require a
`breaking` label. The C++ API currently has no backward-compatibility guarantee,
so C++ API changes are not typically considered breaking.

### Review and merge

Check CI results and address failures. Respond to review feedback and update
your contribution as needed. A cuML maintainer will merge the PR once it is
reviewed and approved and the required checks pass.

## Automated and AI-assisted contributions

Tools that help analyze code, draft text, or implement changes are welcome.
They do not replace human judgment and communication.

- Do not submit issues, pull requests, or review responses through a fully
  autonomous process. A human contributor must remain available and engaged.
- Understand and take responsibility for everything submitted under your
  account, including generated descriptions and replies. Review or otherwise
  validate every change, and disclose the scope and basis of any non-line-by-line
  validation. Communication must accurately represent your judgment. Be able to
  explain the problem, the implementation, its fit with cuML's goals, and the
  validation performed.
- Do not use GitHub review as the validation loop for speculative or
  bulk-generated changes. Run relevant checks, remove irrelevant generated
  content, and keep contributions focused before requesting maintainer attention.

Maintainers may close submissions without technical review when they appear
fully autonomous, unvalidated, misleading, or otherwise impose disproportionate
review cost. Repeated submissions of this kind may result in account blocking.

## Attribution

Portions adopted from [PyTorch's contribution guidelines](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md).
