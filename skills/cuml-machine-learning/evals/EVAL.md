# cuML machine-learning evaluation guidance

Assessment evidence is produced with [NVIDIA SkillEvaluator](https://github.com/NVIDIA/SkillEvaluator); CPU semantic checks remain distinct from cuML/GPU execution evidence.

## Questions
- Include one coding-agent task that repairs temporal leakage, training-only preprocessing, and keyed output.
- Include accelerator fallback, unsupervised evaluation, and negative data-only cases.

## Behaviors
- Reward deployment-aligned splits, baselines, task-quality metrics, output keys, and actual acceleration/fallback evidence.
- Require agents to distinguish CPU semantic/syntax checks from cuML/GPU execution.
- Penalize final-test tuning, coefficient/label equality as universal parity, hidden fallback, and invented ML tasks.

## Notes
- Keep four buckets: explicit coding, implicit, contextual, and negative.
- Baseline and with-skill arms receive identical selected fixtures.
- The default sandbox need not contain a GPU; an honest runtime blocker is preferable to a fabricated cuML run.
