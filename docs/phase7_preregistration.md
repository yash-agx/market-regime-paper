# Phase 7 pre-registration: is H3 a market-beta effect counted twice?

Written and committed on 2026-11-20, before any Phase 7 number was computed. The
frozen main specification (the project spec file) and every result in `results/phase1_*` to
`results/phase6_*` are left exactly as they are. Nothing below replaces them.

## The alternative explanation being tested

H3 says the momentum - reversal Sharpe gap is larger in calm markets than in
high-volatility markets. An alternative explanation raised in review is that H1 and
H2 are **one effect counted twice**:

- High-volatility months are often months in which the market rebounds sharply.
- In a rebound, momentum's **short** leg (past losers) is high-beta, so momentum
  loses: the long-short spread has a negative market beta exactly when the market
  goes up.
- Reversal's **long** leg (recent losers) is also high-beta, so reversal gains.
- If that is all that is going on, then "momentum is worse in high vol" and
  "reversal is better in high vol" are the same bet on the market, seen twice, and
  H3 adds them together.

The standard way to test this is the one in Daniel and Moskowitz (2016, "Momentum
crashes"): let the spread's market beta change with the state of the market, and
see how much of the high-volatility effect is left once that is allowed for.

## What this phase is, and what it is not

This is a **mechanism check on an existing result, not a new hypothesis test.**

- No new claim of statistical significance will be made. The reported p-values
  and intervals describe how much of H3 is left after the beta terms; they do not
  create new findings.
- The headline stays as it is: Method A, 20 bp, long-short, Version 2 universe,
  momentum 12-1 against reversal 1 week, in-sample 2010-2019.
- The held-out period (2020-2025) has already been looked at in Phases 4 and 6;
  nothing in it is protected any more, so it is run here for completeness, and
  read under the same rules.

## Inputs (all built by the existing code, nothing re-estimated)

**Spreads.** The long-short return series of Phases 3 and 6, built with
`src/run_phase6_robustness.build_variants()` on the existing cleaned data:

- momentum 12-1 (main), 3-1, 6-1 - monthly, one observation per month-end rebalance;
- reversal 1 week (main), 2 weeks, 1 month - weekly, one observation per week-end
  rebalance.

Net of 20 bp (main) and gross (shown alongside). Long-short spreads are
zero-investment, so no risk-free rate is subtracted from them, as in the main spec.

**Market return, Mkt_t.** The equal-weighted Nifty 50 benchmark of Phase 3, from
the **same universe** as the spread (V1 benchmark for V1 spreads, V2 for V2), over
**exactly the same holding window** as the spread (the monthly benchmark for
momentum, the weekly benchmark for reversal), gross of costs, **minus the 91-day
T-bill rate over that window**. Excess over the T-bill because the beta question is
about exposure to the market risk premium, as in Daniel and Moskowitz.

**High-vol dummy, HV_t.** The existing Method A label (21-day realised volatility of
`^NSEI` above its expanding median) at the **start** of each holding period, i.e.
on the rebalance date, exactly as in Phases 4 and 6.

**Universes and periods.** V2 (point-in-time, headline) and V1 (today's members,
comparison), each in-sample 2010-2019 and held-out 2020-2025. Eight runs of every
table.

### One choice of frequency, stated in advance

Reversal is traded weekly. **The reversal regressions are run at the weekly
frequency**, with the weekly benchmark and the weekly regime labels. Reason:
that is the frequency at which the main paper defines the reversal Sharpe ratio,
the regime labels and H3, so the beta-adjusted H3 below is the same quantity as
the published H3, only hedged. Compounding weeks into calendar months would mix
weeks that start in different regimes into one "month" and would no longer
reproduce the published H3.

As a **secondary check**, regression A is also run on reversal compounded to
calendar months (weekly returns compounded within each calendar month, matched to
the monthly benchmark over the same month; the month is labelled by the regime on
the first reversal rebalance date in it). This is reported in its own table and is
not used for the verdict.

## 1. Descriptives

For each universe and period, on the monthly (momentum) grid and on the weekly
(reversal) grid:

- the share of high-vol periods in which Mkt_t > 0, and the same share in calm
  periods;
- mean Mkt_t in high-vol and in calm periods, and its standard deviation;
- the number of periods in each regime;
- the correlation between the momentum spread and the reversal spread (reversal
  compounded to calendar months), in each regime, before and after the regression A
  beta terms are removed. If the "counted twice" story is right, this correlation
  should be strongly negative in high-vol months before hedging, and much closer to
  zero after.

## 2. Regression A: state-dependent beta on the regime dummy

For each of the six spreads:

    spread_t = a0 + a1*HV_t + b0*Mkt_t + b1*HV_t*Mkt_t + e_t

OLS, Newey-West standard errors with **3 lags** (the automatic rule used in Phase 4,
4(n/100)^(2/9), is reported alongside as a sensitivity check).

In plain words: b0 is the spread's market beta in calm periods, b0 + b1 its beta in
high-vol periods, and **a1 is the high-vol shift in mean return that is left after
the spread's market exposure is allowed to differ between regimes.**

Reported per spread: all four coefficients with t-statistics and p-values, R², n,
and the **raw high-vol difference** (the Phase 4 Newey-West slope: the same
regression with the two Mkt terms dropped). Key comparison: a1 against the raw
difference, and the ratio a1 / raw.

What the "counted twice" story predicts, written down now:

- momentum: b0 + b1 < b0 (beta falls in high vol; b1 < 0);
- reversal: b0 + b1 > b0 (beta rises in high vol; b1 > 0);
- for both, |a1| much smaller than the raw difference.

## 3. Regression B: Daniel-Moskowitz bear-market optionality

    spread_t = g0 + gB*B_t + b0*Mkt_t + bB*B_t*Mkt_t + bBU*B_t*U_t*Mkt_t + e_t

- **B_t = 1 if the past 24-month market return is negative**, measured at the
  rebalance date: the `^NSEI` close on the rebalance date divided by the last
  `^NSEI` close on or before the date 24 calendar months earlier, minus one. Only
  closes up to and including the rebalance date are used, so B_t is known when the
  portfolio is formed. `^NSEI` is used here, rather than the equal-weighted
  benchmark, because it has history back to January 2009, which the benchmark
  (built from the 2010 study start) does not. It is a price index; for a sign
  condition over two years the missing dividends do not matter much, and this is
  noted as a limitation.
- **U_t = 1 if Mkt_t > 0** (the contemporaneous market excess return of the same
  holding period; this is ex post on purpose - it is a description of the period,
  not a trading signal, exactly as in Daniel and Moskowitz).
- Newey-West, 3 lags, as in regression A.

The 24-month window means B_t is first defined in January 2011, so regression B
in-sample runs 2011-2019, not 2010-2019. This is reported in the table.

What the story predicts: bBU < 0 for momentum (in bear markets the spread's beta is
more negative when the market goes up - the "short a call" payoff), and bBU > 0 for
reversal.

**Power rule (fixed now).** If a period has fewer than 12 bear-state (B_t = 1)
monthly observations, regression B for momentum in that period is still reported
but marked "too few bear months to interpret", and no conclusion is drawn from it.
The same rule with 52 weekly observations applies to reversal.

## 4. H3: raw against beta-adjusted

**Raw H3** is the published statistic: [Sharpe(mom, calm) - Sharpe(rev, calm)] -
[Sharpe(mom, high) - Sharpe(rev, high)], momentum annualised with N = 12 and reversal
with N = 52. For the main pair this must reproduce the Phase 6 / Phase 4 number
exactly; a reproduction check is saved.

**Beta-adjusted H3** is the same statistic computed on the **hedged** series,
spread_t - b0*Mkt_t - b1*HV_t*Mkt_t = a0 + a1*HV_t + e_t, with b0 and b1 from
regression A on that period. Because the regression has a constant and HV_t, the
residual has mean zero in each regime, so the hedged series has mean a0 in calm
periods and a0 + a1 in high-vol periods: it is the spread with its
regime-dependent market exposure taken out.

**Inference.** The same stationary block bootstrap as `E.bootstrap_difference_in_differences`:
resampling over calendar months so that a drawn month brings its momentum
observation and all of its reversal weeks with it, 10,000 resamples, seed 20100101,
Politis-White block length. **Within each resample, regression A is re-estimated on
the resampled observations** before hedging, so the uncertainty in the betas is
carried into the interval. The raw and adjusted statistics use the same resample
draws and the same block length (the one chosen on the raw series), so they are
directly comparable. Reported:

- raw H3, adjusted H3, each with 95% interval and one-sided p-value (share of
  resamples at or below zero);
- the **explained part**, raw - adjusted, with its 95% interval;
- the **fraction surviving**, adjusted / raw, as a point estimate. (No interval is
  reported for the ratio: when raw H3 is near zero in a resample the ratio is
  unstable, so the interval for the difference is the one to read.)

The same decomposition is done for H1 (momentum half) and H2 (reversal half), so it
is clear which half carries any beta effect.

Pairs: the main pair (12-1, 1w) is the one the verdict uses. The other four Phase 6
primary pairs, (3-1, 1w), (6-1, 1w), (12-1, 2w), (12-1, 1m), are reported as
robustness.

**A note on look-ahead.** The hedge betas are estimated over the whole period they
are applied to. That would be look-ahead in a trading strategy, but this is an
attribution exercise - "how much of this realised difference is market exposure" -
not something anyone could trade, and full-period betas are the standard way to do
it (as in Daniel and Moskowitz). B_t and HV_t themselves use only data up to the
rebalance date.

## 5. Robustness: a slower regime definition

Regression A and the H3 decomposition are repeated with a second high-vol label:
**the 126-day (about six months) realised variance of daily `^NSEI` returns above its
expanding median**, built the same way as Method A (right-aligned window, expanding
median including the current day, label at the start of each holding period). The
126-day window follows the six-month realised-variance estimator of Barroso and
Santa-Clara (2015). To be precise about the citation: they measure the realised
variance of the **momentum portfolio itself**, and use it to scale the position;
here the same window is applied to the **market**, as a regime label. Variance above
its median and volatility above its median give the same label, since one is the
square of the other.

This is a robustness check on the regime definition only. It is not a new main
regime method and will not replace Method A anywhere.

## How the results will be read (fixed now)

The verdict uses the main pair, 20 bp, Version 2, in-sample. Let f = adjusted H3 /
raw H3.

| outcome | condition | what the paper will say |
|---|---|---|
| **H3 survives** | f >= 2/3 and the adjusted H3 interval excludes zero | Time-varying market beta does not explain the regime interaction. |
| **Partly explained** | 1/3 <= f < 2/3, or f >= 2/3 but the adjusted interval includes zero | Market beta explains part of H3; a regime effect beyond beta remains but is weaker or less certain. |
| **Mostly explained** | f < 1/3, or adjusted H3 <= 0 | H3 is largely the same market-beta bet counted twice; the paper's central result is a beta-timing effect, not a separate regime effect in the two strategies. |

Supporting evidence, read alongside the verdict but not changing it: whether b1 has
the predicted sign for momentum (negative) and reversal (positive); whether the
high-vol share of up-market periods is clearly above the calm share; whether the
momentum - reversal correlation in high-vol periods moves towards zero after
hedging; and whether the slower regime definition gives the same verdict.

V1 is read the same way and reported second. Held-out numbers are read the same way
too, with the reminder that raw H3 is already not significant held out, so f there
is a ratio of two noisy numbers and is reported but not interpreted on its own.

**Whatever comes out is written up as it comes out**, including the case where H3
disappears entirely after hedging. In that case the paper says so in the abstract,
not only in this phase's note.

## Multiple testing and power

- Six spreads, two regressions, two universes, two periods and two regime
  definitions produce many coefficients. None of them is a new hypothesis; they are
  diagnostics. No individual coefficient will be described as a finding because of
  its p-value.
- In-sample there are only about 25 high-vol momentum months out of 120, so b1 for
  momentum is estimated on few points and its interval will be wide. A beta
  difference that is economically large but statistically imprecise will be
  described as exactly that.

## Outputs

- `src/run_phase7_mechanism.py`: the script; writes only new `results/phase7_*`
  files.
- `results/phase7_*.csv` and `results/phase7_tables.md`: every table above.
- `paper/mechanism_note.md`: a plain-English summary, at most one page.
- `tests/test_phase7_mechanism.py`: regression A on synthetic data with a known
  state-dependent beta recovers it; truncating the data at a date leaves B_t and the
  126-day regime label up to that date unchanged; the hedged series has mean a0 in
  calm and a0 + a1 in high-vol periods; the main-pair raw H3 equals the Phase 6
  value.
