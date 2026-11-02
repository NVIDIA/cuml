# Skill Benchmark: cuml

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cuml`
- Validation status: `passed`
- Evaluation date: 2026-11-02
- Evaluator version: `1.6.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 32 evaluation tasks (32 positive)
- Dataset digest: `sha256:4dad3d83e6a81208bd4d1bb359ae83e0666d9ab17ea5d4df92b5a3c1b825229f` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 93.6% — baseline ran, but no comparable score was available; uplift unavailable | 82.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 95.3% (-4.7 points) | 100.0% → 90.6% (-9.4 points) |
| Correctness | 78.8% → 95.6% (+16.8 points) | 93.1% → 95.0% (+1.9 points) |
| Discoverability | 98.3% — baseline ran, but no comparable score was available; uplift unavailable | 87.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 66.0% → 86.6% (+20.6 points) | 71.7% → 81.1% (+9.4 points) |
| Efficiency | 92.1% — baseline ran, but no comparable score was available; uplift unavailable | 57.3% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.0% → 94.5% (-3.5 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 81.7% → 74.6% (-7.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 6,838,215 | 1,525,320 | +5,312,895 | +348.31% | skill 32/32; base 32/32 |
| claude-code | b01 | 238,605 | 37,700 | +200,905 | +532.90% | skill 1/1; base 1/1 |
| claude-code | b02 | 247,764 | 39,196 | +208,568 | +532.12% | skill 1/1; base 1/1 |
| claude-code | b03 | 244,351 | 38,946 | +205,405 | +527.41% | skill 1/1; base 1/1 |
| claude-code | b04 | 154,676 | 37,009 | +117,667 | +317.94% | skill 1/1; base 1/1 |
| claude-code | b05 | 155,753 | 72,066 | +83,687 | +116.13% | skill 1/1; base 1/1 |
| claude-code | b06 | 75,261 | 36,292 | +38,969 | +107.38% | skill 1/1; base 1/1 |
| claude-code | b07 | 434,154 | 38,332 | +395,822 | +1032.62% | skill 1/1; base 1/1 |
| claude-code | b08 | 77,137 | 73,897 | +3,240 | +4.38% | skill 1/1; base 1/1 |
| claude-code | b09 | 393,659 | 75,536 | +318,123 | +421.15% | skill 1/1; base 1/1 |
| claude-code | b10 | 544,592 | 75,428 | +469,164 | +622.00% | skill 1/1; base 1/1 |
| claude-code | p01 | 74,220 | 71,519 | +2,701 | +3.78% | skill 1/1; base 1/1 |
| claude-code | p02 | 153,783 | 37,043 | +116,740 | +315.15% | skill 1/1; base 1/1 |
| claude-code | p03 | 114,968 | 38,360 | +76,608 | +199.71% | skill 1/1; base 1/1 |
| claude-code | p04 | 244,710 | 37,450 | +207,260 | +553.43% | skill 1/1; base 1/1 |
| claude-code | p05 | 153,984 | 36,803 | +117,181 | +318.40% | skill 1/1; base 1/1 |
| claude-code | p06 | 196,323 | 36,867 | +159,456 | +432.52% | skill 1/1; base 1/1 |
| claude-code | p07 | 276,950 | 38,197 | +238,753 | +625.06% | skill 1/1; base 1/1 |
| claude-code | p08 | 406,653 | 37,281 | +369,372 | +990.78% | skill 1/1; base 1/1 |
| claude-code | p09 | 74,653 | 37,101 | +37,552 | +101.22% | skill 1/1; base 1/1 |
| claude-code | p10 | 376,174 | 37,859 | +338,315 | +893.62% | skill 1/1; base 1/1 |
| claude-code | p11 | 238,441 | 74,020 | +164,421 | +222.13% | skill 1/1; base 1/1 |
| claude-code | p12 | 195,352 | 36,607 | +158,745 | +433.65% | skill 1/1; base 1/1 |
| claude-code | p13 | 113,173 | 36,236 | +76,937 | +212.32% | skill 1/1; base 1/1 |
| claude-code | p14 | 201,645 | 150,773 | +50,872 | +33.74% | skill 1/1; base 1/1 |
| claude-code | p15 | 74,372 | 35,776 | +38,596 | +107.88% | skill 1/1; base 1/1 |
| claude-code | p16 | 74,613 | 36,058 | +38,555 | +106.92% | skill 1/1; base 1/1 |
| claude-code | p17 | 158,214 | 36,477 | +121,737 | +333.74% | skill 1/1; base 1/1 |
| claude-code | p18 | 521,158 | 39,764 | +481,394 | +1210.63% | skill 1/1; base 1/1 |
| claude-code | p19 | 190,869 | 36,064 | +154,805 | +429.25% | skill 1/1; base 1/1 |
| claude-code | p20 | 74,661 | 36,897 | +37,764 | +102.35% | skill 1/1; base 1/1 |
| claude-code | p21 | 242,964 | 36,642 | +206,322 | +563.08% | skill 1/1; base 1/1 |
| claude-code | p22 | 114,383 | 37,124 | +77,259 | +208.11% | skill 1/1; base 1/1 |
| codex | All cases | 8,661,616 | 1,707,063 | +6,954,553 | +407.40% | skill 32/32; base 32/32 |
| codex | b01 | 259,780 | 48,100 | +211,680 | +440.08% | skill 1/1; base 1/1 |
| codex | b02 | 433,205 | 62,567 | +370,638 | +592.39% | skill 1/1; base 1/1 |
| codex | b03 | 774,030 | 55,239 | +718,791 | +1301.24% | skill 1/1; base 1/1 |
| codex | b04 | 200,534 | 30,571 | +169,963 | +555.96% | skill 1/1; base 1/1 |
| codex | b05 | 238,691 | 48,257 | +190,434 | +394.62% | skill 1/1; base 1/1 |
| codex | b06 | 110,216 | 14,256 | +95,960 | +673.12% | skill 1/1; base 1/1 |
| codex | b07 | 289,772 | 53,269 | +236,503 | +443.98% | skill 1/1; base 1/1 |
| codex | b08 | 124,429 | 35,456 | +88,973 | +250.94% | skill 1/1; base 1/1 |
| codex | b09 | 707,457 | 16,616 | +690,841 | +4157.69% | skill 1/1; base 1/1 |
| codex | b10 | 314,754 | 40,647 | +274,107 | +674.36% | skill 1/1; base 1/1 |
| codex | p01 | 50,129 | 34,527 | +15,602 | +45.19% | skill 1/1; base 1/1 |
| codex | p02 | 220,793 | 49,408 | +171,385 | +346.88% | skill 1/1; base 1/1 |
| codex | p03 | 72,297 | 60,104 | +12,193 | +20.29% | skill 1/1; base 1/1 |
| codex | p04 | 496,447 | 82,416 | +414,031 | +502.37% | skill 1/1; base 1/1 |
| codex | p05 | 431,683 | 39,845 | +391,838 | +983.41% | skill 1/1; base 1/1 |
| codex | p06 | 166,468 | 47,933 | +118,535 | +247.29% | skill 1/1; base 1/1 |
| codex | p07 | 303,826 | 73,366 | +230,460 | +314.12% | skill 1/1; base 1/1 |
| codex | p08 | 394,129 | 66,749 | +327,380 | +490.46% | skill 1/1; base 1/1 |
| codex | p09 | 417,090 | 36,290 | +380,800 | +1049.32% | skill 1/1; base 1/1 |
| codex | p10 | 410,937 | 62,352 | +348,585 | +559.06% | skill 1/1; base 1/1 |
| codex | p11 | 263,413 | 278,281 | -14,868 | -5.34% | skill 1/1; base 1/1 |
| codex | p12 | 214,047 | 31,367 | +182,680 | +582.40% | skill 1/1; base 1/1 |
| codex | p13 | 93,934 | 21,725 | +72,209 | +332.38% | skill 1/1; base 1/1 |
| codex | p14 | 258,033 | 56,850 | +201,183 | +353.88% | skill 1/1; base 1/1 |
| codex | p15 | 41,755 | 17,829 | +23,926 | +134.20% | skill 1/1; base 1/1 |
| codex | p16 | 54,593 | 36,412 | +18,181 | +49.93% | skill 1/1; base 1/1 |
| codex | p17 | 338,675 | 56,958 | +281,717 | +494.60% | skill 1/1; base 1/1 |
| codex | p18 | 313,260 | 67,094 | +246,166 | +366.90% | skill 1/1; base 1/1 |
| codex | p19 | 139,339 | 25,429 | +113,910 | +447.95% | skill 1/1; base 1/1 |
| codex | p20 | 118,893 | 47,289 | +71,604 | +151.42% | skill 1/1; base 1/1 |
| codex | p21 | 186,589 | 61,597 | +124,992 | +202.92% | skill 1/1; base 1/1 |
| codex | p22 | 222,418 | 48,264 | +174,154 | +360.84% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 15,499,831 | 3,232,383 | +12,267,448 | +379.52% | skill 64/64; base 64/64 |

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
