# Contributing to cuML

Contribute to cuML by reporting problems, proposing improvements, or submitting
code and documentation changes. Start by describing the problem and, before
implementing a nontrivial change, agreeing on the scope with maintainers. Use
the developer guides for implementation details.

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
plan with maintainers before starting work.

Prefer issues labeled
[Contributions welcome](https://github.com/NVIDIA/cuml/issues?q=is%3Aissue+is%3Aopen+label%3A%22Contributions+welcome%22).
Anyone may open a pull request addressing the stated scope of one of these
issues without prior assignment. Consider commenting before starting to reduce
the risk of duplicate work.

For other existing issues, consider volunteering only when the issue affects
your work or you have another concrete reason to take it on; for example, you
bring relevant domain expertise or have a very specific learning goal. Access
to AI or other automated tools is not by itself a reason to take on an issue.
Comment with your proposed scope and rationale, ask to be assigned, and wait
for assignment before starting work.

Except for trivial changes, pull requests should close an issue to which the
author is assigned, unless the issue is labeled `Contributions welcome`.
Maintainers may close a PR without review if it does not meet this requirement.

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

Target `main` by default. Changes for a soon-to-be-released version may target
`release/YY.MM`. See the
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
your contribution as needed. Do not use GitHub's **Update branch** button only
to bring the PR up to date with its base branch. Resolve conflicts when needed;
otherwise, let maintainers update the branch when necessary. Unnecessary base
branch merges trigger costly CI runs and can delay the PR.

A cuML maintainer will merge the PR once it is reviewed and approved and the
required checks pass.

## Automated and AI-assisted contributions

Tools that help analyze code, draft text, or implement changes are welcome.
They do not replace human judgment and communication.

- Do not submit issues, pull requests, or review responses through a fully
  autonomous process. A human contributor must remain available and engaged.
- Understand and take responsibility for everything submitted under your
  account, including generated descriptions and replies. Review or otherwise
  validate every change, and disclose which parts you did not review line by
  line and how you validated them. Communication must accurately represent your
  judgment. Be able to explain the problem, the implementation, its fit with
  cuML's goals, and the validation performed.
- Run relevant checks, remove irrelevant generated content, and keep
  contributions narrowly scoped before requesting maintainer attention.

Maintainers may close submissions without technical review when they appear
unvetted, misleading, or otherwise impose disproportionate review cost. Repeated
submissions of this kind may result in account blocking.

## Attribution

Portions adopted from [PyTorch's contribution guidelines](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md). The automated-contribution guidance was informed by the contribution policies of [scikit-learn](https://scikit-learn.org/stable/developers/contributing.html#automated-contributions-policy) and [Dask](https://github.com/dask/dask/pull/12320).
