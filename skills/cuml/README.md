# cuML agent skill

An [Agent Skill](https://agentskills.io/) that helps coding agents (Claude Code, Codex, Cursor,
...) write correct code for cuML users. It targets what current models get wrong: APIs that
were removed or renamed in 2025-2026 releases, `cuml.accel` activation and verification,
large-data UMAP/HDBSCAN memory settings, and model export.

- `SKILL.md`: the skill itself.
- `references/parity-and-numerics.md`: how to compare results with scikit-learn, loaded on
  demand.
- `scripts/check_removed_apis.py`: scans code for removed cuML APIs whose replacement cannot
  be found by inspecting the installed cuML, using `scripts/removed_apis.json`.
- `evals/`: evaluation cases and environment for
  [NVIDIA SkillEvaluator](https://github.com/NVIDIA/SkillEvaluator).

To use it, copy or symlink this directory into your agent's skills directory, for example
`~/.claude/skills/cuml`.

Maintainers: see the developer guide page "Agent skill"
(`docs/source/developer_guide/agent_skill.md`) for how to update and evaluate the skill.
The entries in `removed_apis.json` are checked by
`python/cuml/tests/test_agent_skill.py`.
