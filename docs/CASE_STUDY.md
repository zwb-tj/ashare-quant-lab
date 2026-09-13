# Case Study: Validating an A-Share Screening Rule Set

<!-- keywords: a-share, cross-sectional-screening, event-study, excess-return, paired-benchmark,
lookahead-bias, execution-delay, transaction-cost-model, out-of-sample, stratified-sampling,
reproducible-research, pytest, github-actions, llm-agent-evaluation, grounded-number-rate -->

## Overview

`aqlab` is a from-scratch, reproducible lab for China A-share screening and backtesting: explicit
transaction-cost modelling, lookahead-free execution, cross-sectional factor scoring, an auditable
LLM research agent behind read-only tools, and a testable CLI. The exhibit is the research loop -
hypothesis, experiment, measured result, decision - with negative results kept: three of the four
hypotheses below were rejected by the project's own data.

## Research questions

1. Does the screening rule (`b1_graded`) carry forward-return information beyond a same-horizon benchmark?
2. Does a market-regime gate improve outcomes?
3. Do structural exit rules (moving-average break, trend-line break) add or destroy value?
4. Do opening-window features (volume ratio, opening strength, volume slope) admit a hard filter?
5. Do the results survive a longer sample, or are they window artefacts?

## Method

**Data.** Full-market daily bars, 5,424 symbols, 2025-01-01 to 2026-09-11, forward-adjusted. Minute
bars for 2026-06-01 to 2026-09-10 cover 995 of 1,030 symbols; the long re-check uses monthly
stratified sampling (1,100 symbols attempted, 809 with long-history minute data).

**Conventions.** Signal at the close of day T, entry at the open of T+1, exits judged bar by bar,
regime state from the previous close only, every horizon paired against a benchmark over the same
holding period.

**Benchmarks.** Equal-weighted full-market index (universe study); an equal-weighted basket of 148
non-selected symbols entered the same day at the same open (published-list study), with excess on
paired observations only (n = 332 / 322 / 311 / 272 at 1 / 3 / 5 / 10 days).

**Statistics.** Mean and median forward return, win rate, mean excess, t-statistics on excess (Welch
t across groups); n < 20 is flagged, never aggregated away.

| Sample | Cohort | n | Mean % | Excess % | Excess t |
| --- | --- | ---: | ---: | ---: | ---: |
| Full market 2025-01 to 2026-09 | hold 20 days | 58,682 | +1.42 | - | - |
| Full market | hard stop/take-profit only (-7%/+15%) | 58,682 | +1.65 | +0.02 | 0.58 |
| Full market | structural exit only | 58,682 | +0.27 | -0.17 | -4.61 |
| Full market | regime open | 32,161 | +0.69 | -0.21 | -7.54 |
| Full market | regime closed | 26,521 | -0.35 | -0.04 | -1.18 |
| Long re-check 2025-01 to 2026-09 | all | 13,492 | +0.14 | -0.03 | -0.54 |
| Long re-check | volume slope > 0 | 4,621 | +0.21 | +0.08 | 1.00 |
| Long re-check | volume ratio >= 4 | 1,918 | +0.37 | +0.21 | 1.38 |
| Published list 2026-06 to 09 | all, 10-day | 334 | -3.06 | -4.23 | -5.58 |
| Published list | `b1`, 10-day | 146 | -2.11 | -3.55 | -3.41 |

## Findings

**1. Regime gating is the one stable effect - holds.** Regime-open trades averaged +0.69% versus
-0.35% closed over 58,682 trades; in 2026-06 to 09 the gap was +0.78% excess (t = 3.44) versus -0.81%
(t = -6.08), and the long re-check kept the direction (+0.58% versus -0.31%).

**2. Structural exits destroy return - holds, as a rejection.** Holding fell from 10-20 days to 3-5
and excess moved from about zero to -0.13/-0.17 (t = -4.6 to -6.3); in the volume-slope subset the
rules gave +0.21% against mechanical holds of +0.39% / +0.62% / +1.42% at 5 / 10 / 20 days.

**3. The screening rule loses to the benchmark - holds, as a rejection.** Over 334 published records
excess was -0.40% / -1.18% / -2.15% / -4.23% at 1 / 3 / 5 / 10 days (t = -2.25 to -5.58); 13 of 24
bucket-by-horizon cells were significantly negative, none significantly positive.

**4. Opening-window features have no reproducible edge - rejected.** Volume-slope quartiles looked
monotone in the 3.5-month window (-0.90%, t = -3.68 at Q1 to -0.01%, t = -0.04 at Q4); over 21 months
and 13,492 trades the advantage shrank to +0.16pp (+0.08% excess, t = 1.00), and a high volume ratio,
harmful at -0.53% (t = -1.57) short-window, reversed to +0.21% (t = 1.38).

**5. Only the left tail was controlled - holds.** The -7% stop cut the worst trade to -10.4% with none
losing more than 10%, against -22.5% and 3.8% losing more than 10% for a plain 10-day hold, costing
0.8pp of mean return.

## What was rejected and why

- **"High volume ratio plus an opening push signals fresh money."** The relative-caliber gate
  (ratio >= 1) selected 26 of 146 published signals at -11.19% 10-day excess, against -2.17% for the
  120 excluded (-9.02pp, t = -3.26). Ratio >= 4 and up at 09:37 left 9 trades at -12.51%; all 24
  window-length / direction / threshold combinations were negative (mean -11.69%, median -11.38%); a
  threshold scan ran -4.6 / -7.5 / -11.2 / -15.3% at theta = 0.5 / 0.8 / 1.0 / 1.5. The shipped
  `min_ratio = 1.0` is a negative contribution in this sample.
- **"Higher volume ratio means accumulation."** A -11.2% short-window figure became +0.21%
  (t = 1.38) over 21 months: one weak window, not a regularity.
- **"Structural exits protect profit."** Moving-average breaks alone were 33,440 trades (57%)
  averaging -1.55%; the trend-line rule added 15,325 at -0.48%.
- **"Relaxing the entry threshold fixes concentration."** Raising `j_max` from 13 to 30 lifted average
  holdings from 3.04 to 3.51 but moved excess from +0.10% to +0.08% and excess win rate from -2.01% to
  -2.22%; on a 40-symbol pool the same comparison was +2.26%. Pool size, not the threshold, was the
  lever.

## Engineering

271 pytest cases in 28 modules run offline with no network or API key; GitHub Actions adds Python
3.10 / 3.11 / 3.12 plus CLI smoke tests. Lookahead is prevented structurally (`positions.shift(1)`)
and disproved constructively: a test asserts that a signal peeking at the same day's move loses
money. Costs are explicit (`BacktestConfig(fee_bps=3, slippage_bps=2)`) with per-bar cost and turnover
persisted; quality auditing covers gaps, zero-volume bars, jumps, cross-source diffs and sha256
fingerprints. The LLM layer uses six read-only JSON-schema tools, returns failures as `ok=false` to
force abstention, traces every step, and scores offline: `grounded_number_rate` 0.900,
`hallucinated_tasks` 1, `abstain_accuracy` 1.000, `regression_consistency` 1.000,
`tool_success_rate` 0.667 (5 tasks, 6 tool calls, 10 numbers).

## Limitations

Sample periods: 2025-01-01 to 2026-09-11 (universe study) and one 55-trading-day cross-section,
2026-06-17 to 2026-09-11 (published list). Opening features are not full-market (2,597 trades short
window; 809-symbol stratified sample long re-check). Benchmarks are equal-weighted and small-cap
tilted, not cap-weighted. No costs are deducted (roughly 0.15% round trip); market impact and
liquidity beyond fixed slippage are not measured; limit-up, limit-down and suspensions are unmodelled;
prices are forward-adjusted. Excess t-statistics mostly fall in 0.2 to 1.7, and the 146 signals are a
subset of published picks rather than all candidates.

## Reproduction

```bash
python scripts/fetch_universe_daily.py
aqlab universe-study --daily-dir data/universe/daily --rule b1_graded --start 2025-01-01 \
  --end 2026-09-11 --exit-mode entry_low --stop-pct 0.03 --take-profit 0.15 --min-holding 3 --out output
python scripts/opening_filter_study.py
aqlab picks-backtest --archive ./picks_archive --buckets b1,b2,n20,n30,v3 --dedupe-window 5 \
  --horizons 1,3,5,10 --benchmark-sample 0 --out output
aqlab picks-backtest --archive ./picks_archive --buckets b1 --horizons 1,3,5,10 --confirm \
  --confirm-buckets b1 --confirm-entry window_close --benchmark-sample 0 --out output
pytest -q
```
