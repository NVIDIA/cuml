# Agent skill

cuML ships an [Agent Skill](https://agentskills.io/) for users who write cuML code with coding
agents such as Claude Code, Codex or Cursor. It lives in
[`skills/cuml/`](https://github.com/NVIDIA/cuml/tree/main/skills/cuml) and focuses on what
current models get wrong: APIs removed or renamed in recent releases, enabling and verifying
`cuml.accel`, large-data UMAP/HDBSCAN settings, and model export. This page explains how to keep
the skill correct and how to measure whether it helps.

## Layout

| Path | Contents |
|---|---|
| `SKILL.md` | The skill. Keep it short (under about 8 KB); agents load all of it. |
| `references/parity-and-numerics.md` | How to compare cuML results with scikit-learn; read on demand. |
| `scripts/removed_apis.json` | Removed APIs whose replacement cannot be found by inspecting the installed cuML. |
| `scripts/check_removed_apis.py` | Scans user code for the APIs in `removed_apis.json`. |
| `evals/evals.json` | Evaluation cases (prompt, expected output, assertions). |
| `evals/config.yml`, `evals/environment/Dockerfile` | How the evaluation runs. |

## Keeping the skill up to date

The skill tells agents to look things up in the installed cuML (signatures, docstrings, error
messages, `python -m cuml.accel -v`) and in the documentation for the installed release, rather
than carrying copies of that information. Keeping cuML's docstrings, error messages and
`docs/source/cuml-accel/compatibility.rst` accurate therefore keeps the skill accurate. Fix
mistakes there, not in the skill.

When you deprecate or remove a public API:

- Name the replacement in the deprecation warning and docstring.
- If, after the removal, an agent could not find the replacement by inspecting the installed
  cuML (for example a removed module, or a dictionary key that is silently ignored), add an
  entry to `scripts/removed_apis.json` (`deprecated_in`, `removed_in`, `replacement`, and the
  fields used to verify it). Remove entries once the replacement becomes discoverable.

`python/cuml/tests/test_agent_skill.py` checks that every API listed in `removed_apis.json` is
really gone, that every recommended replacement exists, that `compatibility.rst` has not moved
(the skill links to it), and that `evals.json` is well formed. Run it with:

```bash
pytest python/cuml/tests/test_agent_skill.py
```

Write `SKILL.md` for things models get wrong, not as a tutorial. General knowledge that models
already have (what cuML is, KMeans differs numerically from scikit-learn, CuPy interop) costs
context without improving answers.

## Evaluation cases

`evals/evals.json` uses the [agentskills.io](https://agentskills.io/) evaluation format:

```json
{
  "skill_name": "cuml",
  "evals": [
    {
      "id": "p18",
      "name": "svc-proba",
      "prompt": "Using cuML's SVC directly ...",
      "expected_output": "cuML SVC no longer has a `probability` argument ...",
      "assertions": ["Does not pass `probability=True` to cuML SVC or LinearSVC", "..."]
    }
  ]
}
```

`expected_output` describes a correct answer; each assertion is one checkable statement an LLM
judge evaluates against the agent's transcript. Extra fields are kept as metadata: `source`
(`brian` for the original cases, `probe` for cases from a no-skill probe of Claude models),
`track`, and for probe cases `probe_verdicts`, the grades models received without the skill
(C correct, P partly wrong, W wrong).

Good cases are ones where agents without the skill fail: removed APIs, silent fallbacks, new
recipes. Cases that every model already passes only check that the skill does no harm. Keep
assertions specific and verifiable, for example "Does not pass `data_on_host` to
`fit_transform`" rather than "Gives good UMAP advice".

All current cases were seen by the skill's author. A held-out set written by someone else is
still needed to measure the skill without overfitting.

## Running the evaluation

The evaluation uses [NVIDIA SkillEvaluator](https://github.com/NVIDIA/SkillEvaluator) Tier 3. It
runs each case twice, with and without the skill, with a coding agent inside a Docker container,
grades both transcripts with an LLM judge, and reports the difference ("Skill Lift"). The setup
below uses [NVIDIA Build](https://build.nvidia.com/) models for both the agent and the judge.

### Set up

You need Docker (with Compose v2), an NVIDIA API key from
[build.nvidia.com](https://build.nvidia.com/), and SkillEvaluator in its own Python 3.12 or 3.13
environment. The environment name follows the cuML convention, `skilleval-YYYYMMDD`:

```bash
conda create -n skilleval-20261005 python=3.13
conda activate skilleval-20261005
pip install "skillevaluator[tier3] @ git+https://github.com/NVIDIA/SkillEvaluator.git@6bab56e9d2e28f178bd413ff731b3d27291371e4"
```

Configure the provider. With NVIDIA Build, one key is used by the judge and by the agent. The
default agent is [OpenCode](https://opencode.ai/), and agent and judge both default to
`nvidia/nemotron-3-super-120b-a12b`.

```bash
export SKILL_EVAL_LLM_PROVIDER=nv_build
export NVIDIA_API_KEY=nvapi-...
export SKILLEVALUATOR_RESULTS_DIR=~/skilleval-results   # keep results out of the repository
```

Check the setup and the dataset:

```bash
skillevaluator doctor --agents opencode --env-mode docker --verify-models
skillevaluator tier3 validate skills/cuml --strict
skillevaluator models --limit 100    # model IDs in the catalog
```

`doctor` and `models` only read the public model catalog, so they pass even with an invalid key.
Check that the key can run inference before starting an evaluation; this should print `200`:

```bash
curl -s -o /dev/null -w '%{http_code}\n' https://integrate.api.nvidia.com/v1/chat/completions \
  -H "Authorization: Bearer $NVIDIA_API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "nvidia/nemotron-3-super-120b-a12b", "messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 5}'
```

Other providers work too: `SKILL_EVAL_LLM_PROVIDER=anthropic` with `ANTHROPIC_API_KEY` runs Claude
Code with Claude models, and `openai` runs Codex. See the SkillEvaluator configuration docs.

### Run

```bash
skillevaluator tier3 evaluate skills/cuml --agents opencode --env-mode docker --n-attempts 3
```

Pick a different agent model with `--agent-model opencode=nvidia/<publisher>/<model>` (the extra
`nvidia/` prefix is OpenCode's adapter namespace). Claude Code can also run against NVIDIA Build
models through an experimental SkillEvaluator bridge: `--agents claude-code --agent-model
claude-code=<publisher>/<model>`.

`--n-attempts 1` is enough for a smoke test; use 3 or more for numbers you report.

SkillEvaluator has no option to select cases. To run a subset, copy the skill and keep only the
cases you want:

```bash
tmp=$(mktemp -d) && cp -r skills/cuml "$tmp/cuml"
jq '.evals |= map(select(.id == "p18" or .id == "b01"))' skills/cuml/evals/evals.json \
  > "$tmp/cuml/evals/evals.json"
skillevaluator tier3 evaluate "$tmp/cuml" --agents opencode --env-mode docker \
  --n-attempts 1 --harbor-keep-jobs
```

`--harbor-keep-jobs` keeps the agent transcripts; browse them with
`skillevaluator harbor-view <run-dir>/_harbor-jobs`. Read a few: a run where the agent did
nothing still gets graded.

### Results

`skillevaluator view skills/cuml` opens the HTML report of the latest run. Each run directory
contains `result.json` and, per agent, `lift.json` and `pass_at_k_lift.json`. The report scores
five dimensions (security, correctness, discoverability, effectiveness, efficiency); Skill Lift
is the with-skill score minus the without-skill score. Record the image's package versions
(`pip freeze` inside the container) next to any results you report.

## Evaluation environment

`evals/environment/Dockerfile` installs a pinned cuML release and the newest scikit-learn,
umap-learn and hdbscan versions inside that release's `cuml.accel` tested window, so rebuilding
the image gives the same versions. SkillEvaluator replaces its `FROM` line with its own base image
and keeps the rest. When a new cuML release is on PyPI, update all four pins together using that
release's `_CONSTRAINTS` in `python/cuml/cuml/accel/core.py`.

### GPU access

SkillEvaluator does not give containers a GPU, and rejects GPU settings in its configuration
files as sandbox escapes. Without a GPU, cuML still imports and agents can inspect signatures,
but every fit fails with `cudaErrorNoDevice`, which changes how agents behave. To give the
evaluation containers a GPU, make NVIDIA the default Docker runtime and expose a device in the
image:

1. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/),
   add `"default-runtime": "nvidia"` to `/etc/docker/daemon.json`, and restart Docker.
2. Add `ENV NVIDIA_VISIBLE_DEVICES=0 NVIDIA_DRIVER_CAPABILITIES=compute,utility` to
   `evals/environment/Dockerfile`, using the index or UUID from `nvidia-smi -L`.
3. Set `n_concurrent: 1` in `evals/config.yml`, because trials share the GPU.

This deliberately bypasses a SkillEvaluator safety rule and changes the default runtime for every
container on the machine. Only do it on a machine you control, for skills you trust.

## Limitations

- The judge sees the agent's transcript, not files it wrote or code it ran.
- Prompts about very large data (out-of-memory at 10M rows) or multiple GPUs cannot be
  reproduced faithfully in the evaluation container; they are graded on the answer.
- There is no held-out case set yet.
- Results depend on the agent and model. The `probe_verdicts` baselines in `evals.json` come from
  Claude models; other models fail different cases.
