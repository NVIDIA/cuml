# Skill Benchmark: cuml

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cuml`
- Validation status: `passed`
- Evaluation date: 2026-11-04
- Evaluator version: `1.6.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 32 evaluation tasks (32 positive)
- Dataset digest: `sha256:8efd78ed52747d376b59182ebeb97a529c0cfa2167658e8a2efb1c39eb4452f4` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| cuml | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.3` | nondeterministic / provider_defaults |
| cuml | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.3` | nondeterministic / provider_defaults |
| cuml | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.3` | nondeterministic / provider_defaults |
| cuml | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.3` | nondeterministic / provider_defaults |
| cuml | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.3` | nondeterministic / provider_defaults |

- Controls for `cuml`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 95.5% — baseline ran, but no comparable score was available; uplift unavailable | 83.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 93.8% (-6.2 points) |
| Correctness | 81.3% → 98.1% (+16.8 points) | 91.9% → 94.4% (+2.5 points) |
| Discoverability | 98.2% — baseline ran, but no comparable score was available; uplift unavailable | 89.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 66.3% → 87.4% (+21.1 points) | 70.5% → 83.7% (+13.2 points) |
| Efficiency | 94.0% — baseline ran, but no comparable score was available; uplift unavailable | 56.8% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.6% → 94.5% (-3.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 80.4% → 73.9% (-6.5 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 6,891,762 | 2,179,539 | +4,712,223 | +216.20% | skill 32/32; base 32/32 |
| claude-code | b01 | 239,106 | 38,621 | +200,485 | +519.11% | skill 1/1; base 1/1 |
| claude-code | b02 | 246,666 | 39,894 | +206,772 | +518.30% | skill 1/1; base 1/1 |
| claude-code | b03 | 199,311 | 38,086 | +161,225 | +423.32% | skill 1/1; base 1/1 |
| claude-code | b04 | 195,783 | 36,873 | +158,910 | +430.97% | skill 1/1; base 1/1 |
| claude-code | b05 | 155,758 | 37,090 | +118,668 | +319.95% | skill 1/1; base 1/1 |
| claude-code | b06 | 75,110 | 36,448 | +38,662 | +106.07% | skill 1/1; base 1/1 |
| claude-code | b07 | 388,915 | 38,222 | +350,693 | +917.52% | skill 1/1; base 1/1 |
| claude-code | b08 | 241,742 | 74,214 | +167,528 | +225.74% | skill 1/1; base 1/1 |
| claude-code | b09 | 301,405 | 111,649 | +189,756 | +169.96% | skill 1/1; base 1/1 |
| claude-code | b10 | 566,836 | 76,588 | +490,248 | +640.11% | skill 1/1; base 1/1 |
| claude-code | p01 | 74,108 | 71,970 | +2,138 | +2.97% | skill 1/1; base 1/1 |
| claude-code | p02 | 75,471 | 36,937 | +38,534 | +104.32% | skill 1/1; base 1/1 |
| claude-code | p03 | 154,593 | 73,530 | +81,063 | +110.24% | skill 1/1; base 1/1 |
| claude-code | p04 | 200,784 | 37,363 | +163,421 | +437.39% | skill 1/1; base 1/1 |
| claude-code | p05 | 195,572 | 37,546 | +158,026 | +420.89% | skill 1/1; base 1/1 |
| claude-code | p06 | 196,595 | 344,194 | -147,599 | -42.88% | skill 1/1; base 1/1 |
| claude-code | p07 | 283,283 | 38,268 | +245,015 | +640.26% | skill 1/1; base 1/1 |
| claude-code | p08 | 364,733 | 37,552 | +327,181 | +871.27% | skill 1/1; base 1/1 |
| claude-code | p09 | 75,082 | 36,808 | +38,274 | +103.98% | skill 1/1; base 1/1 |
| claude-code | p10 | 333,008 | 39,344 | +293,664 | +746.40% | skill 1/1; base 1/1 |
| claude-code | p11 | 241,667 | 74,551 | +167,116 | +224.16% | skill 1/1; base 1/1 |
| claude-code | p12 | 197,090 | 36,928 | +160,162 | +433.71% | skill 1/1; base 1/1 |
| claude-code | p13 | 113,265 | 35,616 | +77,649 | +218.02% | skill 1/1; base 1/1 |
| claude-code | p14 | 480,379 | 151,806 | +328,573 | +216.44% | skill 1/1; base 1/1 |
| claude-code | p15 | 74,104 | 35,704 | +38,400 | +107.55% | skill 1/1; base 1/1 |
| claude-code | p16 | 74,664 | 36,017 | +38,647 | +107.30% | skill 1/1; base 1/1 |
| claude-code | p17 | 154,030 | 36,246 | +117,784 | +324.96% | skill 1/1; base 1/1 |
| claude-code | p18 | 447,854 | 344,795 | +103,059 | +29.89% | skill 1/1; base 1/1 |
| claude-code | p19 | 151,994 | 35,927 | +116,067 | +323.06% | skill 1/1; base 1/1 |
| claude-code | p20 | 74,690 | 36,898 | +37,792 | +102.42% | skill 1/1; base 1/1 |
| claude-code | p21 | 242,934 | 36,746 | +206,188 | +561.12% | skill 1/1; base 1/1 |
| claude-code | p22 | 75,230 | 37,108 | +38,122 | +102.73% | skill 1/1; base 1/1 |
| codex | All cases | 8,209,804 | 1,661,718 | +6,548,086 | +394.06% | skill 32/32; base 32/32 |
| codex | b01 | 391,778 | 50,076 | +341,702 | +682.37% | skill 1/1; base 1/1 |
| codex | b02 | 349,425 | 67,545 | +281,880 | +417.32% | skill 1/1; base 1/1 |
| codex | b03 | 780,328 | 49,205 | +731,123 | +1485.87% | skill 1/1; base 1/1 |
| codex | b04 | 162,172 | 36,557 | +125,615 | +343.61% | skill 1/1; base 1/1 |
| codex | b05 | 326,664 | 39,695 | +286,969 | +722.93% | skill 1/1; base 1/1 |
| codex | b06 | 136,168 | 43,416 | +92,752 | +213.64% | skill 1/1; base 1/1 |
| codex | b07 | 324,658 | 44,878 | +279,780 | +623.42% | skill 1/1; base 1/1 |
| codex | b08 | 144,187 | 31,284 | +112,903 | +360.90% | skill 1/1; base 1/1 |
| codex | b09 | 370,987 | 82,095 | +288,892 | +351.90% | skill 1/1; base 1/1 |
| codex | b10 | 302,500 | 45,197 | +257,303 | +569.29% | skill 1/1; base 1/1 |
| codex | p01 | 41,728 | 40,432 | +1,296 | +3.21% | skill 1/1; base 1/1 |
| codex | p02 | 240,179 | 39,816 | +200,363 | +503.22% | skill 1/1; base 1/1 |
| codex | p03 | 186,142 | 54,187 | +131,955 | +243.52% | skill 1/1; base 1/1 |
| codex | p04 | 473,039 | 65,784 | +407,255 | +619.08% | skill 1/1; base 1/1 |
| codex | p05 | 303,495 | 39,316 | +264,179 | +671.94% | skill 1/1; base 1/1 |
| codex | p06 | 153,331 | 52,274 | +101,057 | +193.32% | skill 1/1; base 1/1 |
| codex | p07 | 242,297 | 64,786 | +177,511 | +274.00% | skill 1/1; base 1/1 |
| codex | p08 | 262,575 | 64,945 | +197,630 | +304.30% | skill 1/1; base 1/1 |
| codex | p09 | 158,583 | 39,150 | +119,433 | +305.07% | skill 1/1; base 1/1 |
| codex | p10 | 478,238 | 68,070 | +410,168 | +602.57% | skill 1/1; base 1/1 |
| codex | p11 | 275,380 | 31,891 | +243,489 | +763.50% | skill 1/1; base 1/1 |
| codex | p12 | 182,244 | 38,493 | +143,751 | +373.45% | skill 1/1; base 1/1 |
| codex | p13 | 97,498 | 17,655 | +79,843 | +452.24% | skill 1/1; base 1/1 |
| codex | p14 | 357,922 | 159,292 | +198,630 | +124.70% | skill 1/1; base 1/1 |
| codex | p15 | 41,738 | 17,883 | +23,855 | +133.39% | skill 1/1; base 1/1 |
| codex | p16 | 53,247 | 39,029 | +14,218 | +36.43% | skill 1/1; base 1/1 |
| codex | p17 | 358,377 | 48,491 | +309,886 | +639.06% | skill 1/1; base 1/1 |
| codex | p18 | 347,392 | 95,610 | +251,782 | +263.34% | skill 1/1; base 1/1 |
| codex | p19 | 161,148 | 26,396 | +134,752 | +510.50% | skill 1/1; base 1/1 |
| codex | p20 | 50,431 | 46,695 | +3,736 | +8.00% | skill 1/1; base 1/1 |
| codex | p21 | 256,736 | 65,203 | +191,533 | +293.75% | skill 1/1; base 1/1 |
| codex | p22 | 199,217 | 56,372 | +142,845 | +253.40% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 15,101,566 | 3,841,257 | +11,260,309 | +293.14% | skill 64/64; base 64/64 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 20 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 32 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/cuml/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/cuml/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/cuml/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/cuml/SKILL.md`)
- **MEDIUM** QUALITY/quality_discoverability: Skill name very short: 'cuml' (`skills/cuml/SKILL.md`)
- 15 additional finding(s) are available in the full evaluation artifacts.

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
