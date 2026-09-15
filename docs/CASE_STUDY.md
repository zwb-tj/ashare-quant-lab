# Case Study: An A-Share Research Loop, from Screening Rules to Factor Evidence

<!-- keywords: a-share, cross-sectional-screening, event-study, excess-return, paired-benchmark,
lookahead-bias, execution-delay, transaction-cost-model, out-of-sample, stratified-sampling,
information-coefficient, factor-portfolio, holding-period-robustness, reproducible-research,
pytest, github-actions, llm-agent-evaluation, grounded-number-rate -->

## Overview

`aqlab` is a from-scratch, reproducible lab for China A-share screening and backtesting: explicit
transaction-cost modelling, lookahead-free execution, cross-sectional factor scoring, an auditable
research agent behind read-only tools, and a testable CLI. The exhibit is the research loop -
hypothesis, experiment, measured result, decision - with negative results kept: seven hypotheses
were rejected by the project's own data, including the default factor weighting and the inference that
a negative information coefficient can be traded in reverse.

## Research questions

1. Does the screening rule (`b1_graded`) carry forward-return information beyond a same-horizon benchmark?
2. Does a market-regime gate improve outcomes?
3. Do structural exit rules (moving-average break, trend-line break) add or destroy value?
4. Do opening-window features (volume ratio, opening strength, volume slope) admit a hard filter?
5. Do the results survive a longer sample, or are they window artefacts?
6. Do five factors carry information on forward 20-day returns, and with what sign?
7. Can a negative IC become a portfolio through sign flipping or strictly past IC weights?
8. Is that stable across holding periods?
9. Does the research agent report numbers that come from tool output rather than memory?

## Method

**Data.** Full-market daily bars, 5,424 symbols, 2025-01-01 to 2026-09-11, forward-adjusted. Minute
bars for 2026-06-01 to 2026-09-10 cover 995 of 1,030 symbols; the long re-check samples 1,100 symbols
by month, 809 with long-history minute data. The factor study uses the same 5,424 symbols from 2024-06
to 2026-09, giving 82 cross-sections at a 20-day horizon (85 at 1 to 5 days, 84 at 10) in a 9-minute
pass. The published-list study deduplicates 508 archived records to 334 records over 282 symbols
(2026-06-17 to 2026-09-11, 55 trading days).

**Conventions.** Signal at the close of day T, entry at the open of T+1, exits judged bar by bar,
regime state from the previous close only, every horizon paired against a benchmark over the same
holding period; factor values use only bars up to and including the `as_of` date.

**Benchmarks.** Equal-weighted full-market index (universe study); an equal-weighted basket of 148
non-selected symbols entered the same day at the same open (published-list study), with excess on
paired observations only (n = 332 / 322 / 311 / 272 at 1 / 3 / 5 / 10 days); same-period equal-weighted
full market (factor schemes).

**Statistics.** Mean and median forward return, win rate, mean excess and t-statistics on excess (Welch
t across groups); n < 20 is flagged, never aggregated away. Factor IC is the per-date Spearman rank
correlation between factor value and forward return; IC_IR = mean IC / IC standard deviation;
overlapping forward windows are corrected by `t_adj = t / sqrt(overlap)`.

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
(t = -6.08), and the long re-check kept the direction (+0.58% versus -0.31%). Both stay below the
benchmark (-0.21%, t = -7.54; -0.04%, t = -1.18).

**2. Structural exits destroy return - holds, as a rejection.** Holding fell from 10-20 days to 3-5
and excess moved from near zero to -0.13/-0.17 (t = -4.6 to -6.3); in the volume-slope subset the
rules gave +0.21% against holds of +0.39% / +0.62% / +1.42% at 5 / 10 / 20 days.

**3. The screening rule loses to the benchmark - holds, as a rejection.** Over 334 published records
excess was negative at every horizon (t = -2.25 to -5.58); 13 of 24 bucket-by-horizon cells were
significantly negative, none significantly positive, and the two buckets flat at one day turn negative
once a 0.15% round trip is charged.

| Bucket | Records | 1-day return / bench / excess | 3-day | 5-day | 10-day |
| --- | ---: | --- | --- | --- | --- |
| `b1` | 146 | -0.05 / +0.25 / -0.30 | -0.33 / +0.80 / -1.13 | -1.11 / +0.69 / -1.81 | -2.11 / +1.44 / -3.55 |
| `n20` | 62 | +0.36 / +0.23 / +0.12 | -1.19 / +0.41 / -1.59 | -1.30 / +1.04 / -2.34 | -1.88 / +1.78 / -3.66 |
| `v3` | 26 | +0.16 / +0.11 / +0.06 | -1.20 / +0.27 / -1.47 | -3.53 / +0.78 / -4.31 | -9.02 / +0.66 / -9.68 |
| all | 334 | -0.23 / +0.17 / -0.40 | -0.67 / +0.50 / -1.18 | -1.59 / +0.57 / -2.15 | -3.06 / +1.17 / -4.23 |

| Excess t | 1-day | 3-day | 5-day | 10-day |
| --- | ---: | ---: | ---: | ---: |
| `b1` | -1.27 | -2.77 | -2.75 | -3.41 |
| `n20` | +0.39 | -2.43 | -2.25 | -2.22 |
| `v3` | +0.06 | -0.90 | -1.72 | -3.08 |
| all | -2.25 | -3.78 | -4.45 | -5.58 |

**4. Opening-window features have no reproducible edge - rejected.** Volume-slope quartiles looked
monotone in the short window; over 21 months the advantage shrank to +0.16pp and a high volume ratio
reversed sign (-0.53% to +0.21%).

| Group | 3.5-month window (2,597 trades) | 21-month re-check (13,492 trades) |
| --- | --- | --- |
| all | -0.83% mean, -0.31% excess (t = -2.69) | +0.14% mean, -0.03% excess (t = -0.54) |
| volume slope > 0 | -0.35% mean, +0.05% excess (t = 0.22) | +0.21% mean, +0.08% excess (t = 1.00) |
| volume ratio >= 4 | -1.10% mean, -0.53% excess (t = -1.57) | +0.37% mean, +0.21% excess (t = 1.38) |

Quartile cut, short window (excess %):

| Feature | Q1 | Q4 |
| --- | ---: | ---: |
| volume slope | -0.90 (t = -3.68) | -0.01 (t = -0.04) |
| volume ratio | -0.05 | -0.31 |

**5. Only the left tail was controlled - holds.** The -7% stop cut the worst trade to -10.4% with none
losing more than 10%, against -22.5% and 3.8% losing more than 10% for a plain 10-day hold, at a cost
of 0.8pp of mean return; the structural rule set did worse (-26.8%).

**6. All five factors carry negative forward information at 20 days.** Every factor studied is
negative, and all five rank low-factor-value names above high-factor-value names.

| Factor | Mean IC | IC sd | IC_IR | t | t (overlap-adjusted) | Share of positive IC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| trend_gap | -0.105 | 0.172 | -0.61 | -5.56 | -2.78 | 26.8% |
| mom_60 | -0.084 | 0.162 | -0.52 | -4.70 | -2.35 | 28.0% |
| mom_20 | -0.082 | 0.161 | -0.51 | -4.59 | -2.29 | 29.3% |
| vol_20 | -0.076 | 0.228 | -0.33 | -3.03 | -1.51 | 37.8% |
| rsi_14 | -0.068 | 0.165 | -0.41 | -3.74 | -1.87 | 32.9% |

Mean forward return by factor-value bucket (%, bucket 1 = lowest factor value):

| Bucket | mom_20 | mom_60 | rsi_14 | trend_gap | vol_20 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (lowest) | 1.95 | 1.98 | 1.75 | 2.23 | 1.14 |
| 2 | 1.95 | 1.82 | 1.77 | 1.94 | 1.94 |
| 3 | 1.83 | 1.72 | 1.84 | 1.78 | 2.02 |
| 4 | 1.61 | 1.56 | 1.70 | 1.43 | 1.99 |
| 5 (highest) | 0.99 | 1.25 | 1.27 | 0.95 | 1.24 |

Bucket 5 is the worst for all five; the decline is strictly monotone for `mom_20`, `mom_60` and
`trend_gap`, while `rsi_14` and `vol_20` peak in bucket 3. The sample period is characterised by
reversal, and the default screening weights (`mom_60` +0.35, `trend_gap` +0.25, `mom_20` +0.20) point
the opposite way: the default score behaves more like an inverse indicator. Those weights are
configurable through `ScreenConfig.weights`.

**7. A negative IC is not a tradable long book - rejected.** Four weighting schemes were tested:
`fixed` (default momentum weights), `sign_flip` (the same weights negated), `ic_sign` (sign of trailing
mean IC) and `ic_weight` (proportional to trailing mean IC, normalised by |IC|). Weights use strictly
past information (`--lag 1`, asserted by a test); each rebalance pays a 20 bps round trip; the
benchmark is the same-period equal-weighted full market (+1.77% per 20 days).

| Scheme | Periods | Mean net % | Win rate % | Mean excess % | Excess t |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fixed` | 82 | +0.44 | 53.7 | -1.23 | -0.74 |
| `ic_weight` | 81 | +0.09 | 61.7 | -1.79 | -1.65 |
| `ic_sign` | 81 | -0.12 | 56.8 | -2.00 | -1.71 |
| `sign_flip` | 82 | -1.89 | 41.5 | -3.55 | -4.28 |

All four lose to the benchmark, and reversing the default weights is significantly worse: net return
per period falls from +0.44% to -1.89%, excess -3.55% (t = -4.28). IC is a whole-cross-section rank
statistic while the portfolio holds only the 10 most extreme names, and the weakest bucket is
dominated by small, illiquid names. **A factor-level statistic must pass a portfolio-level
out-of-sample test before it counts as evidence.**

**8. The negative IC is horizon-robust; the reversal conclusion is not.** The IC term structure is
negative at all six horizons and grows with horizon.

| Horizon | mom_20 | mom_60 | rsi_14 | trend_gap | vol_20 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 day | -0.041 | -0.058 | -0.055 | -0.061 | -0.067 |
| 2 days | -0.054 | -0.063 | -0.047 | -0.065 | -0.062 |
| 3 days | -0.073 | -0.069 | -0.060 | -0.079 | -0.065 |
| 5 days | -0.061 | -0.066 | -0.046 | -0.071 | -0.072 |
| 10 days | -0.075 | -0.080 | -0.059 | -0.090 | -0.077 |
| 20 days | -0.082 | -0.084 | -0.068 | -0.105 | -0.076 |

At a 5-day holding period the four schemes still all lose to the benchmark, with a changed ranking.

| Scheme | Mean net % | Win rate % | Mean excess % | Excess t |
| --- | ---: | ---: | ---: | ---: |
| `ic_weight` | -0.38 | 52.4 | -0.74 | -0.97 |
| `ic_sign` | -0.45 | 51.2 | -0.82 | -1.22 |
| `fixed` | -0.58 | 44.7 | -0.88 | -1.02 |
| `sign_flip` | -0.58 | 47.1 | -0.88 | -1.30 |

Across both holding periods no scheme produces positive net alpha, and the ranking moves with the
horizon (`fixed` best at 20 days, `ic_weight` at 5): noise, not edge. Reversal
performing worse holds only at 20 days (at 5 days `sign_flip` converges with the others at -0.88%),
so it is horizon-dependent, not general.

**9. The agent layer is scored against four failure modes, not trusted.** The harness grew from 5
tasks to 20: `normal` (9: backtest, indicator, data, screening, metadata), `trap` (5: missing symbol,
invalid parameter, insufficient history, future data, an invitation to answer from memory),
`consistency` (3: a premise contradicting tool output, corrected rather than echoed) and `regression`
(3: repeated answers must match word for word).

| Metric | Value |
| --- | ---: |
| tasks | 20 |
| answered_rate | 1.000 |
| tool_calls | 23 |
| tool_success_rate | 0.913 |
| numbers_total | 54 |
| grounded_number_rate | 0.981 |
| hallucinated_tasks | 1 |
| abstain_accuracy | 1.000 |
| contradiction_accuracy | 1.000 |
| constraint_violations | 0 |
| regression_consistency | 1.000 |

The one hallucinated task is the seeded one reporting 999.99% from memory. Grounded number rate per
kind is 1.000 for `normal`, `regression` and `consistency` and 0.000 for `trap`, where a correct
abstention contains no numbers. The set is itself guarded: `tests/test_evaluation.py` requires a bad
agent that never calls tools, invents numbers and agrees with a false premise to score low on the same
three metrics.

## What was rejected and why

- **"High volume ratio plus an opening push signals fresh money."** The relative-caliber gate selected
  26 of 146 signals at -11.19% 10-day excess against -2.17% for the 120 excluded (-9.02pp, t = -3.26);
  the software-caliber cut left 9 trades at -12.51%, and every threshold and caliber combination
  tested was negative. The shipped `min_ratio = 1.0` is a negative contribution here.
- **"Higher volume ratio means accumulation."** A -0.53% short-window figure became +0.21%
  (t = 1.38) over 21 months: one weak window, not a regularity.
- **"Structural exits protect profit."** Moving-average breaks alone were 33,440 trades (57%) averaging
  -1.55%; the trend-line rule added 15,325 at -0.48%. Tail risk was not controlled either: worst trade
  -26.8% against -22.5% for a plain hold.
- **"Relaxing the entry threshold fixes concentration."** Raising `j_max` from 13 to 30 lifted average
  holdings from 3.04 to 3.51 but moved excess from +0.10% to +0.08% and excess win rate from -2.01% to
  -2.22%; on a 40-symbol pool the same comparison was +2.26%. Pool size, not the threshold, was the
  lever.
- **"The screening rule carries a tradable edge."** Over 334 published records excess was -0.40% /
  -1.18% / -2.15% / -4.23% at 1 / 3 / 5 / 10 days (t = -2.25 to -5.58); 13 of 24 cells were
  significantly negative, none significantly positive.
- **"A negative factor IC can be traded by flipping the sign."** With strictly past IC, 20 bps
  round-trip cost and a same-period equal-weighted benchmark, negating the weights returned -1.89% net
  per period and -3.55% excess (t = -4.28), against +0.44% net for the default direction and +1.77%
  for the benchmark.
- **"Agent answers can be taken at face value."** Grounding is measured: 54 numbers were checked
  against tool output, one task is flagged as hallucinated, and a deliberately bad agent in the test
  suite must fail the same metrics.

The volume-ratio gate, in numbers (10-day excess %, 146 published signals):

| Caliber and cut | n | Excess % |
| --- | ---: | ---: |
| relative caliber, ratio >= 1, selected | 26 | -11.19 |
| relative caliber, ratio >= 1, excluded | 120 | -2.17 |
| software caliber, ratio >= 4 and up at 09:37 | 9 | -12.51 |
| software caliber, ratio >= 4 and down | 12 | -10.42 |
| software caliber, ratio >= 4, no direction split | 21 | -11.30 |
| all published signals | 146 | -3.66 |

| Threshold scan (ratio >=) | 3 | 4 | 5 |
| --- | ---: | ---: | ---: |
| Excess % | -6.91 | -11.30 | -18.34 |

| Robustness across 24 window / direction / threshold combinations | Value |
| --- | ---: |
| mean excess % | -11.69 |
| median excess % | -11.38 |

## Engineering

423 pytest cases in 32 test modules run offline with no network or API key; statement coverage is 86%
(5,382 statements, 728 missed); CI fails below 85% (`--cov-fail-under=85`). GitHub Actions runs
`ruff`, `mypy` and the suite on a Python 3.10 / 3.11 / 3.12 matrix plus CLI smoke tests, with the lint
rule set pinned in `pyproject.toml` rather than set to `ALL`. A repository-hygiene guard test blocks
UTF-8 BOMs in text files, re-parses `pyproject.toml` for the project name, the `dev` extra and the
pinned tool sections, and rejects tab-indented CI YAML. Lookahead is prevented structurally
(`positions.shift(1)`) and disproved constructively: a test asserts that a signal peeking at the same
day's move loses money. Costs are explicit (`BacktestConfig(fee_bps=3, slippage_bps=2)`) with per-bar
cost and turnover persisted; quality auditing covers gaps, zero-volume bars, jumps, cross-source diffs
and sha256 fingerprints. The CLI exposes 18 subcommands, 11 exercised end to end; the agent layer uses
six read-only JSON-schema tools, returns failures as `ok=false` to force abstention, and traces every
step.

## Factor research: the same discipline applied to formulaic alphas

The screening-rule work above was extended to a clean-room Alpha101 subset (44 factors implemented, 22 skipped
with the missing input named, since they need industry or market-cap data that is not available locally). The
same evidence chain was applied, and each layer removed candidates:

| Layer | Surviving candidates |
| --- | ---: |
| Raw IC significant at 5% (111 factor-horizon cells) | 41 |
| After Benjamini-Hochberg FDR control | 35 |
| After Bonferroni | 28 |
| In-sample picks surviving out of sample (single 50/50 split) | 11 of 16 |
| Robust across 4 walk-forward folds (>=2 folds selected, >=60% survived) | 12 of 74 |
| Net of 20 bps round-trip cost, net-excess t > 2 | 3 of 33 |

Two things are worth stating plainly. First, the strongest in-sample factors failed out of sample
(alpha_008: in-sample t = 4.90, out-of-sample -0.54), which is the classic selection trap. Second, the
per-fold matrix showed fold 4 (test window 2026-03-27 to 2026-06-25) failing for all 20 selected combinations
while the other three folds produced 36 survivors, so these signals have whole windows where they stop working.
Attempts to explain that window were rejected by the data: the highest-dispersion bucket actually has the
strongest IC, and the low-momentum bucket is generally better than the high one, so fold 4 is recorded as
unexplained rather than assigned a convenient story. Mean turnover across the costed portfolios is 0.90 and the
median break-even cost 23.4 bps, which is why only 3 of 33 clear a realistic cost hurdle.

Methodological notes that carried over from the rule study: p-values come from a normal approximation of the
overlap-adjusted t statistic, the multiple-testing correction is implemented in-repo (Bonferroni and
Benjamini-Hochberg, verified element-wise against statsmodels to 1.1e-16), state buckets use full-sample
quantiles and are therefore explanatory rather than tradable, and the whole study sits on one market over one
period (112 usable cross-sections after a 260-bar factor warm-up).

## Limitations

Sample periods: 2025-01-01 to 2026-09-11 (universe study), 2024-06 to 2026-09 (factor study, 82
cross-sections) and 2026-06-17 to 2026-09-11 (published list, 55 trading days). The factor universe is
the currently listed symbol set and carries survivorship bias; factor results are equal-weighted with
no industry or size neutralisation, and the IC study deducts no costs, though the scheme backtest
charges 20 bps per rebalance. Opening features are not full-market (2,597 trades short window; an
809-symbol stratified sample long). Benchmarks are equal-weighted and small-cap
tilted, not cap-weighted. The universe and published-list studies deduct no costs (0.15% to 0.2% round
trip); market impact and liquidity beyond fixed slippage are not measured; limit-up, limit-down and
suspensions are unmodelled; prices are forward-adjusted. Excess t-statistics in the rule studies mostly
fall in 0.2 to 1.7, the overlap-adjusted factor t-statistics only reach -1.51 to -2.78, and the 146
signals are a subset of published picks.

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
aqlab factor-ic --data-dir data/universe/daily --forward 20 --horizons 1,2,3,5,10,20 \
  --step 5 --quantiles 5 --out output
aqlab factor-backtest --data-dir data/universe/daily \
  --schemes fixed,sign_flip,ic_sign,ic_weight --forward 20 --step 5 --top-n 10 --lag 1 \
  --cost-bps 20 --out output
aqlab factor-backtest --data-dir data/universe/daily \
  --schemes fixed,sign_flip,ic_sign,ic_weight --forward 5 --step 5 --top-n 10 --lag 1 \
  --cost-bps 20 --out output
aqlab eval --mode offline
pytest -q
```
