# Phase 6 pre-registration: lookback robustness

Written and committed on 2026-11-18, before any Phase 6 variant was computed. The
frozen main specification (the project spec file) and every result in `results/phase1_*` to
`results/phase5*` are left exactly as they are. Nothing below replaces them.

## What this phase is, and what it is not

These are **robustness checks, not new headline results.** The headline is still
the pre-specified main test: 12-1 momentum against 1-week reversal, Method A, 20 bp,
long-short, point-in-time (Version 2) universe. The question here is only whether
that result, and the way it fails to replicate held out, depends on the 12-1 and
1-week lookbacks.

Because several variants are tried, some of them will look better or worse than the
main spec purely by chance. So:

- No variant will be promoted to a headline result, whatever its p-value.
- No new claim of statistical significance will be made from any variant. A variant
  p-value below the threshold is reported as "agrees with the main test", not as a
  finding.
- The verdict is about **consistency**: does each variant give the same sign and
  roughly the same size for H3 in-sample, and does it fail held out in the same way?

## Variants (all set before running)

Momentum, J-1 months with the skip month kept. The signal runs from the last
trading day of month t-J to the last trading day of month t-1, exactly as the
existing 12-1 code does with J = 12. Rebalanced monthly, held one month.

| label | formation window                 |
|-------|----------------------------------|
| 3-1   | month-end t-3 to month-end t-1   |
| 6-1   | month-end t-6 to month-end t-1   |
| 12-1  | month-end t-12 to month-end t-1 (main spec) |

Reversal, K-week lookback, still rebalanced weekly and held one week; only the
signal window changes. The signal is the return from the week end K weeks back to
this week end, as in the main spec with K = 1.

| label   | formation window                       |
|---------|----------------------------------------|
| 1 week  | previous week end to this week end (main spec) |
| 2 weeks | week end two weeks back to this week end |
| 1 month | week end four weeks back to this week end (4 weeks is about 20 trading days, the usual one-month reversal window, and keeps the signal on the same weekly grid as the trade) |

Combinations. H3 needs one momentum and one reversal series, so the variants are
paired in two sets:

1. **Primary robustness set: one change at a time.** Five pairs, each changing one
   lookback and holding the other at the main spec: (3-1, 1w), (6-1, 1w),
   (12-1, 1w) = main spec, (12-1, 2w), (12-1, 1m).
2. **Supplementary: the full 3 x 3 grid** of H3, all nine pairs. Reported in full,
   but only as a picture of how H3 moves; it is not used for the verdict.

## Pipeline (unchanged from Phases 1-5)

Every variant goes through the same code path as the main spec, with only the
formation window changed:

- Version 2 point-in-time Nifty 50 membership (headline). Version 1 (today's
  members) is run too, only for the survivorship comparison below.
- The same cleaning rules, applied over each variant's own formation window: 252
  days of history, at most 5% of the formation window missing, no 10-day flat run
  touching the window, a price on the rebalance day.
- Quintile sorts, equal weights, long-short, holdings drifting within the period.
- Regimes: Method A (main), India VIX (main robustness), Method B (HMM,
  underpowered in-sample; reported with its sample counts).
- Costs: gross, 10, 20 (base) and 30 bp one-way on turnover.
- In-sample 2010-2019 and held-out 2020-2025, the same split.
- Tests: stationary block bootstrap, 10,000 resamples, fixed seed 20100101,
  Politis-White block length, the same H1, H2, H3 statistics and one-sided p-values;
  Newey-West regression on the high-vol dummy.
- Long-only portfolios are not part of this phase; H1-H3 are defined on the
  long-short books.

## Statistics reported per variant

- Sharpe ratio by regime (high-vol, low-vol, all) for momentum and reversal, with
  the number of periods in each regime.
- The momentum - reversal Sharpe gap in each regime, and H3 = gap(low) - gap(high),
  with its 95% bootstrap interval and one-sided p-value.
- H1 and H2 with their p-values.
- Newey-West slope and p-value for each strategy.
- In-sample and held-out, gross and net.

## Multiple testing

- Within each variant: Bonferroni over the three hypotheses, 0.05 / 3 = 0.0167,
  exactly as in the main spec, with the same caveat that H3 = H1 + H2 so the three
  are not independent.
- Across the five primary pairs: Holm-adjusted H3 p-values, reported alongside. This
  is a check against reading too much into any one variant, not a new test.

## The three parts of the main conclusion being checked

1. **In-sample:** H3 (Method A, 20 bp, Version 2) is positive and its one-sided
   p-value is below 0.0167. A variant "agrees" if H3 is positive with p < 0.0167;
   "same sign, weaker" if positive with p >= 0.0167; "contradicts" if H3 <= 0.
2. **Held out:** the in-sample result fails to replicate (H3 p >= 0.05 held out). A
   variant "agrees" if held-out H3 p >= 0.05.
3. **Survivorship and momentum crashes:** under the survivorship-biased Version 1,
   momentum's losses in high-volatility months are roughly twice those under
   Version 2. Measured as the ratio of mean high-vol monthly momentum return,
   Version 1 / Version 2 (Method A, 20 bp, in-sample); the main spec gives about
   2.0. Also shown: the worst high-vol month and the high-vol short-leg return in
   each universe. A variant "agrees" if the ratio is between 1.5 and 3. Held-out
   ratios are reported too. This applies to momentum variants only.

## What would change the conclusion

If the in-sample H3 result holds only at 12-1 and 1 week, the paper will say that
the main finding depends on the lookback choice. If a variant reverses the sign of
H3, that is reported as a contradiction. Either outcome is written up as it comes
out.
