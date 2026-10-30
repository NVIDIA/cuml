# Skill Benchmark: cuml

> **Tier 3 verdict: PASS**

> ⚪ **Overall verdict: NOT ASSESSED — Publication gate was not assessed**

Tier 1 and Tier 2 were not requested. This report covers Tier 3 only; the full publication gate was not assessed.

- Publication gate: NOT ASSESSED

## Evaluation Metadata

- Report scope: Tier 3 only (requested tiers: 3)
- Skill: `cuml`
- Validation status: `passed`
- Evaluation date: 2026-10-22
- Evaluator version: `1.8.0`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 32 evaluation tasks (32 positive)
- Dataset digest: `sha256:4dad3d83e6a81208bd4d1bb359ae83e0666d9ab17ea5d4df92b5a3c1b825229f` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 3
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

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
| Overall | 96.1% — full-score comparison unavailable; shared-metric lift reported below | 83.9% — full-score comparison unavailable; shared-metric lift reported below |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 89.4% (-10.6 points) |
| Correctness | 89.7% → 99.4% (+9.7 points) | 92.2% → 99.4% (+7.2 points) |
| Discoverability | 96.9% — baseline ran, but no comparable score was available; uplift unavailable | 88.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 64.7% → 91.2% (+26.5 points) | 64.0% → 84.3% (+20.3 points) |
| Efficiency | 93.1% — baseline ran, but no comparable score was available; uplift unavailable | 58.3% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

- claude-code: Shared-metric lift **+15.64 percentage points**. The native shared-metric basket uses the same scored evaluator metrics in both conditions. Diagnostic only; separate from the full dimension scores and their comparison.
- codex: Shared-metric lift **+9.27 percentage points**. The native shared-metric basket uses the same scored evaluator metrics in both conditions. Diagnostic only; separate from the full dimension scores and their comparison.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Dimension comparison | N/A | The with-skill and baseline conditions have different attempt counts. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Dimension comparison | N/A | The with-skill and baseline conditions have different attempt counts. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 6,668,703 | 1,820,871 | N/A | N/A | skill 32/32; base 35/35; The with-skill and baseline conditions have different attempt counts. |
| claude-code | b01 | 209,937 | 26,949 | +182,988 | +679.02% | skill 1/1; base 1/1 |
| claude-code | b02 | 246,067 | 30,434 | +215,633 | +708.53% | skill 1/1; base 1/1 |
| claude-code | b03 | 241,459 | 27,005 | +214,454 | +794.13% | skill 1/1; base 1/1 |
| claude-code | b04 | 137,876 | 25,562 | +112,314 | +439.38% | skill 1/1; base 1/1 |
| claude-code | b05 | 201,809 | 50,053 | +151,756 | +303.19% | skill 1/1; base 1/1 |
| claude-code | b06 | 51,870 | 25,345 | +26,525 | +104.66% | skill 1/1; base 1/1 |
| claude-code | b07 | 315,838 | 26,751 | +289,087 | +1080.66% | skill 1/1; base 1/1 |
| claude-code | b08 | 200,015 | 27,823 | +172,192 | +618.88% | skill 1/1; base 1/1 |
| claude-code | b09 | 469,354 | 76,960 | +392,394 | +509.87% | skill 1/1; base 1/1 |
| claude-code | b10 | 322,361 | 52,747 | +269,614 | +511.15% | skill 1/1; base 1/1 |
| claude-code | p01 | 51,208 | 98,864 | -47,656 | -48.20% | skill 1/1; base 1/1 |
| claude-code | p02 | 109,132 | 25,465 | +83,667 | +328.56% | skill 1/1; base 1/1 |
| claude-code | p03 | 337,802 | 77,128 | +260,674 | +337.98% | skill 1/1; base 1/1 |
| claude-code | p04 | 277,884 | 185,575 | +92,309 | +49.74% | skill 1/1; base 1/1 |
| claude-code | p05 | 137,915 | 77,936 | N/A | N/A | skill 1/1; base 3/3; The with-skill and baseline conditions have different attempt counts. |
| claude-code | p06 | 108,334 | 74,732 | +33,602 | +44.96% | skill 1/1; base 1/1 |
| claude-code | p07 | 231,600 | 26,447 | +205,153 | +775.71% | skill 1/1; base 1/1 |
| claude-code | p08 | 304,748 | 26,395 | +278,353 | +1054.57% | skill 1/1; base 1/1 |
| claude-code | p09 | 52,170 | 25,827 | +26,343 | +102.00% | skill 1/1; base 1/1 |
| claude-code | p10 | 487,934 | 27,527 | +460,407 | +1672.57% | skill 1/1; base 1/1 |
| claude-code | p11 | 206,364 | 53,141 | +153,223 | +288.33% | skill 1/1; base 1/1 |
| claude-code | p12 | 172,641 | 25,264 | +147,377 | +583.35% | skill 1/1; base 1/1 |
| claude-code | p13 | 78,696 | 24,733 | +53,963 | +218.18% | skill 1/1; base 1/1 |
| claude-code | p14 | 229,360 | 103,485 | +125,875 | +121.64% | skill 1/1; base 1/1 |
| claude-code | p15 | 51,338 | 24,670 | +26,668 | +108.10% | skill 1/1; base 1/1 |
| claude-code | p16 | 51,978 | 25,050 | +26,928 | +107.50% | skill 1/1; base 1/1 |
| claude-code | p17 | 136,719 | 25,565 | +111,154 | +434.79% | skill 1/1; base 1/1 |
| claude-code | p18 | 491,016 | 321,586 | N/A | N/A | skill 1/1; base 2/2; The with-skill and baseline conditions have different attempt counts. |
| claude-code | p19 | 134,033 | 24,504 | +109,529 | +446.98% | skill 1/1; base 1/1 |
| claude-code | p20 | 51,881 | 125,949 | -74,068 | -58.81% | skill 1/1; base 1/1 |
| claude-code | p21 | 209,900 | 25,632 | +184,268 | +718.90% | skill 1/1; base 1/1 |
| claude-code | p22 | 359,464 | 25,767 | +333,697 | +1295.06% | skill 1/1; base 1/1 |
| codex | All cases | 8,528,108 | 1,987,780 | N/A | N/A | skill 33/33; base 36/36; The with-skill and baseline conditions have different attempt counts. |
| codex | b01 | 306,738 | 88,876 | N/A | N/A | skill 1/1; base 2/2; The with-skill and baseline conditions have different attempt counts. |
| codex | b02 | 476,535 | 73,054 | +403,481 | +552.31% | skill 1/1; base 1/1 |
| codex | b03 | 1,191,071 | 59,607 | N/A | N/A | skill 2/2; base 1/1; The with-skill and baseline conditions have different attempt counts. |
| codex | b04 | 165,896 | 29,265 | +136,631 | +466.88% | skill 1/1; base 1/1 |
| codex | b05 | 69,331 | 61,119 | +8,212 | +13.44% | skill 1/1; base 1/1 |
| codex | b06 | 56,087 | 49,049 | +7,038 | +14.35% | skill 1/1; base 1/1 |
| codex | b07 | 344,121 | 51,571 | +292,550 | +567.28% | skill 1/1; base 1/1 |
| codex | b08 | 52,140 | 38,384 | +13,756 | +35.84% | skill 1/1; base 1/1 |
| codex | b09 | 579,577 | 65,080 | +514,497 | +790.56% | skill 1/1; base 1/1 |
| codex | b10 | 290,202 | 52,451 | +237,751 | +453.28% | skill 1/1; base 1/1 |
| codex | p01 | 46,644 | 28,837 | +17,807 | +61.75% | skill 1/1; base 1/1 |
| codex | p02 | 233,559 | 93,561 | N/A | N/A | skill 1/1; base 2/2; The with-skill and baseline conditions have different attempt counts. |
| codex | p03 | 65,284 | 51,475 | +13,809 | +26.83% | skill 1/1; base 1/1 |
| codex | p04 | 368,372 | 186,944 | +181,428 | +97.05% | skill 1/1; base 1/1 |
| codex | p05 | 421,577 | 43,794 | +377,783 | +862.64% | skill 1/1; base 1/1 |
| codex | p06 | 162,390 | 46,393 | +115,997 | +250.03% | skill 1/1; base 1/1 |
| codex | p07 | 330,629 | 63,677 | +266,952 | +419.23% | skill 1/1; base 1/1 |
| codex | p08 | 262,141 | 81,559 | +180,582 | +221.41% | skill 1/1; base 1/1 |
| codex | p09 | 170,528 | 40,480 | +130,048 | +321.26% | skill 1/1; base 1/1 |
| codex | p10 | 947,812 | 71,105 | +876,707 | +1232.98% | skill 1/1; base 1/1 |
| codex | p11 | 172,903 | 146,236 | +26,667 | +18.24% | skill 1/1; base 1/1 |
| codex | p12 | 227,597 | 29,241 | +198,356 | +678.35% | skill 1/1; base 1/1 |
| codex | p13 | 89,647 | 15,938 | +73,709 | +462.47% | skill 1/1; base 1/1 |
| codex | p14 | 60,883 | 63,551 | -2,668 | -4.20% | skill 1/1; base 1/1 |
| codex | p15 | 38,770 | 28,595 | +10,175 | +35.58% | skill 1/1; base 1/1 |
| codex | p16 | 51,642 | 37,591 | +14,051 | +37.38% | skill 1/1; base 1/1 |
| codex | p17 | 353,056 | 51,159 | +301,897 | +590.12% | skill 1/1; base 1/1 |
| codex | p18 | 286,009 | 93,173 | +192,836 | +206.97% | skill 1/1; base 1/1 |
| codex | p19 | 141,253 | 77,908 | N/A | N/A | skill 1/1; base 3/3; The with-skill and baseline conditions have different attempt counts. |
| codex | p20 | 47,320 | 41,433 | +5,887 | +14.21% | skill 1/1; base 1/1 |
| codex | p21 | 229,504 | 67,245 | +162,259 | +241.30% | skill 1/1; base 1/1 |
| codex | p22 | 288,890 | 59,429 | +229,461 | +386.11% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 15,196,811 | 3,808,651 | N/A | N/A | skill 65/65; base 71/71; The with-skill and baseline conditions have different attempt counts. |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **NOT RUN** | No result was recorded |
| Tier 2 | Semantic deduplication | **NOT RUN** | No result was recorded |
| Tier 3 | Live agent evaluation | **PASS** | Selected Tier 3 evaluation; native coverage and quality apply |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- AGENT_EVAL: Tier 3 evaluation complete: verdict PASS; best agent claude-code

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
