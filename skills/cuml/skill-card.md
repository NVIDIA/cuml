## Description: <br>
Use when writing, running, debugging or upgrading Python code that uses NVIDIA cuML (GPU machine learning), cuml.accel (zero-code-change GPU acceleration of scikit-learn, umap-learn and hdbscan), or GPU UMAP/HDBSCAN/RandomForest/KMeans/SVC, including through libraries that wrap cuML (BERTopic, rapids-singlecell), covering APIs removed or renamed in 2025-2026 releases, verifying code really ran on the GPU, large-data UMAP/HDBSCAN memory recipes, and exporting trained models. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and data scientists use this skill with coding agents (such as Claude Code, Codex, or Cursor) to write, debug, and upgrade Python code that uses cuML or cuml.accel, including checking for removed APIs, confirming GPU execution, tuning large-data UMAP/HDBSCAN/KMeans memory use, and exporting trained models. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Not Specified] <br>
**Credential Type(s):** [None identified] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Comparing cuML with scikit-learn (references/parity-and-numerics.md)](references/parity-and-numerics.md) <br>
- [RAPIDS installation guide](https://docs.rapids.ai/install/) <br>
- [cuML cuml.accel benchmarks](https://docs.nvidia.com/cuml/latest/cuml-accel/benchmarks/) <br>
- [Agent Skills](https://agentskills.io/) <br>
- [NVIDIA SkillEvaluator](https://github.com/NVIDIA/SkillEvaluator) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Shell commands, Analysis] <br>
**Output Format:** [Markdown with inline Python and bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Bundled checker script scripts/check_removed_apis.py prints findings as text or JSON (--json) and exits with status 1 when a removed API is found.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
32 positive evaluation tasks, 1 attempt per task, each run in an isolated k8s-sandbox pod and compared against a no-skill baseline (evaluator version 1.6.6, evaluated 2026-11-02). <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from the security signal. <br>
- Correctness: Is the answer correct? Scored from the accuracy signal. <br>
- Discoverability: Was the right skill loaded when needed? Scored from the skill_execution signal. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal_accuracy and behavior_check. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? 50% skill_efficiency plus 50% token_efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 93.6% (uplift unavailable) | 82.3% (uplift unavailable) |
| Security | 100.0% → 95.3% (-4.7 points) | 100.0% → 90.6% (-9.4 points) |
| Correctness | 78.8% → 95.6% (+16.8 points) | 93.1% → 95.0% (+1.9 points) |
| Discoverability | 98.3% (uplift unavailable) | 87.3% (uplift unavailable) |
| Effectiveness | 66.0% → 86.6% (+20.6 points) | 71.7% → 81.1% (+9.4 points) |
| Efficiency | 92.1% (uplift unavailable) | 57.3% (uplift unavailable) |

## Skill Version(s): <br>
26.12.00 (source: repo VERSION file; git SHA 871ee48, committed 2026-11-02) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
