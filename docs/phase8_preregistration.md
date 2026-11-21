# Phase 8 pre-registration: a one-day execution lag

Written and committed on 2026-11-21, before any Phase 8 number was computed. The
frozen main specification (the project spec file) and every result in `results/phase1_*` to
`results/phase7_*` are left exactly as they are. Nothing below replaces them.

## The concern being tested

In the main specification, weekly reversal is **traded at the same close that
defines its signal**: the week's return is measured to the week-end close t, and
the portfolio is formed at that same close t. This is a known limitation (see the
spec file, "Known limitations").

Why it matters: a closing price is sometimes a bid and sometimes an ask. A stock
whose close at t happened to print at the bid looks like a "loser" this week, and
its next close is more likely to be back nearer the middle of the spread, so it
looks like it "reversed". That is **bid-ask bounce**: a mechanical negative
autocorrelation in measured returns that no one could actually trade on, because
buying at t means paying the ask, not the bid that made the stock look cheap. It is
the standard reason short-term reversal looks profitable before costs. If bid-ask
spreads widen in volatile markets, which they usually do, the bounce would also be
larger in high-vol weeks, and that would inflate H2 (reversal does better in high
vol) and, through it, H3.

The standard fix is to skip one day between the signal and the trade, so that the
price which defines the signal is never the price the trade is done at.

## What this phase is, and what it is not

This is a **robustness check on the main result, not a new hypothesis test.**

- No new claim of statistical significance will be made. The lagged H1, H2 and H3
  are read as "the main result survives the lag or not", under the rules fixed
  below.
- The headline stays as it is: Method A, 20 bp, long-short, Version 2 universe,
  momentum 12-1 against reversal 1 week, in-sample 2010-2019.
- The held-out period (2020-2025) was already looked at in Phases 4, 6 and 7, so
  nothing in it is protected; it is run here for completeness and read under the
  same rules.

## The change (and only this change)

**Signal at close t, trade at close t+1**, where t+1 is the next trading day on the
index calendar after t. Applied to both strategies:

- **Weekly reversal** (the case that matters): t is the week-end trading day, as
  now. The signal is unchanged: the return from the previous week end to the close
  at t. The portfolio is entered at the close of the next trading day, t+1 (usually
  the Monday), and held until the close of the trading day after the next week end.
- **Monthly momentum** (for consistency): t is the month-end trading day, as now.
  The signal is unchanged: the 12-1 return from month-end t-12 to month-end t-1.
  The portfolio is entered at the close of the first trading day of the next month
  and held until the close of the first trading day of the month after.
  Momentum already has a month between the end of its signal and the trade, so the
  lag is not expected to matter much for it; it is applied so that both strategies
  are executed the same way when H3 compares them.

In both cases the holding periods remain back-to-back: each one runs from one trade
date to the next, with no gap and no overlap, so the strategy is always invested and
every holding period is still one week or one month long.

### Everything else is decided at the signal date t

- **Who can be selected.** The eligibility rules (252 days of history, at most 5%
  of the formation window missing, no 10-day flat run touching the formation
  window, a price on t) and Version 2 index membership are all evaluated at the
  signal date t, exactly as now. The quintile sort and the selected names are fixed
  at t, using only data up to the close at t.
- **A name with no price on t+1.** It cannot be bought at t+1, so it is dropped
  from its leg and the rest of the leg is equal-weighted. It is not replaced, and
  the quintile breakpoints are not recomputed, so no information from t+1 enters the
  sort. The number of such cases is reported (expected to be close to zero).
- **Leaving the index.** As in the main spec, a stock is held to the end of its
  holding period even if it leaves the index in the meantime.
- **Costs.** The same turnover formula and the same 10 / 20 (base) / 30 bp one-way
  charges. Turnover is measured against the weights drifted over the lagged holding
  period.
- **Which period a holding belongs to.** Periods are dated and assigned to the
  in-sample or held-out sample by their **signal date t**, exactly as the unlagged
  periods are by their rebalance date. This keeps a one-to-one match between each
  lagged period and the unlagged period with the same signal, which is what makes a
  paired comparison possible. The last period(s) whose t+1 trade or holding end falls
  after 2025-12-31 are dropped; period counts are reported for both versions.

### Regime labels

**Primary comparison: every lagged period is labelled by the Method A regime at the
signal date t**, the same label its unlagged counterpart has. Each lagged period
therefore falls in exactly the same regime as the unlagged period with the same
signal, and **execution timing is the only thing that differs between the two
runs**. That is the cleanest way to answer the question this phase asks: how much
of the result comes from trading at the signal close. The label at t uses data up to
and including t, so it is causal.

The spec says to label each holding period by the regime **at its start**, and with
the lag the holding period starts at t+1, one trading day later. Labelling at t is a
deliberate choice for this robustness phase, made before any Phase 8 number was
computed, so that the comparison isolates execution. It does not change the main
spec, which remains as written for every other result.

**Sensitivity: every lagged period labelled at t+1**, the start of the lagged
holding period, as the spec wording reads (21-day realised volatility of `^NSEI`
above its expanding median, using data up to and including t+1, known at the moment
of the trade). This can move a few periods that sit on a regime boundary into the
other regime, so it mixes a relabelling effect into the lag effect. Reported
alongside it: the number of periods whose label differs between t and t+1.

## What is reported

Main pair (momentum 12-1, reversal 1 week), Method A, Version 2 (headline), then
Version 1 (comparison); in-sample 2010-2019 and held-out 2020-2025; **without the lag
and with the lag, side by side**; at 20 bp (main) and gross (shown alongside,
because bid-ask bounce is a gross-return effect); 10 and 30 bp in the CSVs.

Per strategy, per regime: Sharpe ratio, mean return, volatility, number of periods,
turnover. Then:

- **H1** = Sharpe(momentum, low vol) - Sharpe(momentum, high vol);
- **H2** = Sharpe(reversal, high vol) - Sharpe(reversal, low vol);
- **H3** = [Sharpe(mom, low) - Sharpe(rev, low)] - [Sharpe(mom, high) - Sharpe(rev, high)];

each with its 95% bootstrap interval and one-sided p-value, and Bonferroni 0.0167 as
in the main spec (with the same caveat that H3 = H1 + H2, so the three are not
independent tests).

The unlagged numbers must **reproduce the Phase 6 main-pair rows exactly**; a
reproduction check is saved, as in Phase 7.

### The same bootstrap

- The same functions as Phases 4 and 6 (`E.bootstrap_sharpe_gap` for H1 and H2,
  `E.bootstrap_difference_in_differences` for H3): stationary block bootstrap,
  10,000 resamples, seed 20100101.
- **The same block length for both versions**: the Politis-White length chosen on
  the unlagged series is used for the lagged series too, so that the two intervals
  differ only because the returns differ, not because the resampling does.
- **The change caused by the lag, with its own interval.** For H2 and H3, a paired
  bootstrap over calendar months of the signal date: a drawn month brings its
  momentum observation and all of its reversal weeks with it, **in both versions at
  once**, and the statistic is (lagged - unlagged). Same seed, same block length,
  10,000 resamples. Reported: the change, its 95% interval, and the fraction kept,
  f = lagged / unlagged, as a point estimate (no interval for the ratio, for the
  same reason as in Phase 7: it is unstable when the denominator is near zero in a
  resample).
- Newey-West slope on the high-vol dummy for each strategy, both versions, as in
  Phase 4.

### Supporting descriptives (not part of the verdict)

- Gross reversal Sharpe and mean weekly return, all periods and by regime, both
  versions, and the **share of the gross mean weekly return that the lag removes**,
  overall and separately in high-vol and low-vol weeks. If bounce is what is
  driving H2, this share should be clearly larger in high-vol weeks.
- The same for momentum.
- Turnover, both versions.

## How the results will be read (fixed now)

The verdict uses the main pair, Method A, 20 bp, **Version 2, in-sample**, with
regime labels at the signal date t (the primary comparison). Unlagged reference
values (Phase 6): H2 = +1.39 (p = 0.043), H3 = +2.54 (p = 0.006).

**Reversal result (H2).** Let f2 = lagged H2 / unlagged H2.

| outcome | condition | what the paper will say |
|---|---|---|
| **Survives** | lagged H2 > 0, f2 >= 2/3, and lagged one-sided p < 0.05 | The reversal half of the regime effect is not an artefact of trading at the signal close. |
| **Weakened** | lagged H2 > 0 and either 1/3 <= f2 < 2/3, or f2 >= 2/3 with p >= 0.05 | Part of the reversal regime effect is consistent with bid-ask bounce or very-short-horizon reversal; what is left is weaker or less certain. |
| **Does not survive** | lagged H2 <= 0, or f2 < 1/3 | The reversal half of the regime effect is mostly an execution artefact. |

The significance condition for H2 is p < 0.05, not Bonferroni, because 0.05 is the
level the unlagged H2 met (p = 0.043; it never met 0.0167), and the lagged version is
not held to a stricter standard than the original.

**Main result (H3).** Let f3 = lagged H3 / unlagged H3. The same three bands, with
the significance condition set at the main spec's Bonferroni level, p < 0.0167,
which the unlagged H3 met:

| outcome | condition |
|---|---|
| **Survives** | lagged H3 > 0, f3 >= 2/3, and lagged one-sided p < 0.0167 |
| **Weakened** | lagged H3 > 0 and either 1/3 <= f3 < 2/3, or f3 >= 2/3 with p >= 0.0167 |
| **Does not survive** | lagged H3 <= 0, or f3 < 1/3 |

**Momentum (H1).** No verdict, because there is no bounce reason for momentum to
move. But if lagged H1 differs from unlagged H1 by more than one third of the
unlagged value, that is unexpected and will be flagged and looked into in the write-up.

**Reading the size of the drop honestly.** The lag removes the first day of every
holding period. For reversal, that day contains bid-ask bounce **and** any genuine
very-short-horizon price reversal, and the two cannot be told apart with daily
closes. So the drop caused by the lag is an **upper bound** on the bounce component,
not an estimate of it, and the paper will say so.

**Other runs.** Version 1, the held-out period, gross returns, the t+1-labelled
sensitivity run and India VIX (supplementary table) are read by the same bands and
reported, but do not change the verdict. Held out, the unlagged H2 and H3 are
already not significant (p = 0.306 and 0.210), so f there is a ratio of two noisy
numbers and is reported but not interpreted on its own.

**Whatever comes out is written up as it comes out**, including the case where the
reversal half of H3 disappears once the trade is lagged. In that case the paper says
so in the main text, not only in a robustness note.

## Tests (fixed now)

- **The existing no-look-ahead truncation tests keep passing** (the whole `pytest`
  suite is run before the Phase 8 commit, and the result is reported).
- New file `tests/test_phase8_execution_lag.py`:
  1. **The lagged signal never uses the trade-day close.** For every lagged window,
     the last price the signal reads is strictly before the trade date. And on the
     data itself: overwrite every stock's close on each trade day t+1 (and every
     later day) with random numbers, recompute the signals and the selected long
     and short names at t, and check they are identical to those from the
     untouched data.
  2. **Truncation.** Cutting the data at a signal date t leaves the signal, the
     selected names and the Method A label at t unchanged.
  3. **Timing.** The trade date is the next trading day after the signal date; the
     holding periods chain with no gap and no overlap; on synthetic prices with a
     jump on day t+1, the jump is in the unlagged holding return and not in the
     lagged one.
  4. **A name with no price on t+1** is dropped from its leg without changing the
     other selections.
  5. **No lag reproduces the main spec.** With the lag set to zero days, the new
     code gives the Phase 6 main-pair return series exactly.

## Outputs

- `src/run_phase8_execution_lag.py`: the script; writes only new `results/phase8_*`
  files.
- `results/phase8_*.csv` and `results/phase8_tables.md`: every table above.
- A short "Execution lag" paragraph in the paper.
- `tests/test_phase8_execution_lag.py`.
