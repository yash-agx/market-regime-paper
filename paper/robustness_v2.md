## Robustness: signal lookback

The main test fixes two lookbacks: 12-1 months for momentum and one week for reversal. Both are
conventional, but they are still choices. If the result only showed up at those two, it would
say more about the choice than about volatility regimes. So I re-ran the whole pipeline with
momentum at 3-1, 6-1 and 12-1 months (the skip month kept in every case) and reversal at one
week, two weeks and one month (four weeks, still rebalanced weekly and held one week). Nothing
else changed: the point-in-time universe, the cleaning rules (applied over each variant's own
formation window), the regime definitions, the costs, the 2010-2019 / 2020-2025 split and the
tests are all as in the main analysis. The variants, the statistics and the rules for reading
them were written down and committed before any variant was computed
(`docs/phase6_preregistration.md`). These are robustness checks, not new results: no variant
replaces the main specification, and no claim of significance is drawn from searching across
them. As a check on the code, the 12-1 / one-week pair reproduces every Phase 4 number exactly.

The primary set changes one lookback at a time and holds the other at the main specification,
which gives five pairs. Table R1 reports them under Method A at 20 bp, Version 2 universe.
"Gap" is the momentum Sharpe minus the reversal Sharpe within a regime, and H3 is the gap in
low-volatility periods minus the gap in high-volatility periods. p-values are one-sided, from
10,000 stationary block bootstrap resamples. The Holm column adjusts the H3 p-value across the
five pairs.

**Table R1. Sharpe ratios by regime and the momentum - reversal gap, by lookback (Method A, 20 bp, Version 2)**

| Variant | Period | Mom low | Mom high | Rev low | Rev high | Gap low | Gap high | H3 | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Mom 3-1, rev 1w | 2010-2019 | -0.07 | -1.19 | -1.12 | +0.28 | +1.05 | -1.47 | +2.51 | [+0.18, +5.06] | 0.017 | 0.018 |
| | 2020-2025 | -0.75 | -0.50 | -1.48 | -0.93 | +0.72 | +0.43 | +0.29 | [-2.65, +2.87] | 0.391 | 0.781 |
| Mom 6-1, rev 1w | 2010-2019 | +0.06 | -0.50 | -1.12 | +0.28 | +1.18 | -0.78 | +1.95 | [+0.51, +3.95] | 0.005 | 0.018 |
| | 2020-2025 | -0.74 | -0.50 | -1.48 | -0.93 | +0.73 | +0.43 | +0.30 | [-2.93, +2.71] | 0.404 | 0.781 |
| **Mom 12-1, rev 1w (main)** | 2010-2019 | +0.69 | -0.46 | -1.12 | +0.28 | +1.81 | -0.73 | +2.54 | [+0.58, +4.86] | 0.006 | 0.018 |
| | 2020-2025 | +0.04 | -0.60 | -1.48 | -0.93 | +1.52 | +0.33 | +1.19 | [-1.80, +3.51] | 0.210 | 0.699 |
| Mom 12-1, rev 2w | 2010-2019 | +0.69 | -0.46 | -0.99 | +0.30 | +1.68 | -0.76 | +2.44 | [+0.64, +4.42] | 0.004 | 0.018 |
| | 2020-2025 | +0.04 | -0.60 | -1.29 | -0.81 | +1.33 | +0.20 | +1.13 | [-1.05, +3.05] | 0.140 | 0.699 |
| Mom 12-1, rev 1m | 2010-2019 | +0.69 | -0.46 | -0.51 | +0.62 | +1.20 | -1.08 | +2.27 | [+1.21, +3.75] | 0.000 | 0.001 |
| | 2020-2025 | +0.04 | -0.60 | -1.15 | -0.69 | +1.19 | +0.08 | +1.11 | [-1.03, +2.82] | 0.152 | 0.699 |

In-sample: 120 momentum months (25 high-vol) and 521 reversal weeks (115 high-vol). Held out: 71
months (28 high-vol) and 313 weeks (119 high-vol). The Bonferroni threshold within a variant is
0.05/3 = 0.0167, as in the main test.

The survivorship claim is checked in Table R2, for the momentum lookbacks only. It compares the
mean monthly return of momentum in high-volatility months under today's membership (Version 1)
with the point-in-time membership (Version 2).

**Table R2. Momentum in high-volatility months, Version 1 against Version 2 (Method A, 20 bp)**

| Lookback | Period | Mean return V1 | Mean return V2 | Ratio V1/V2 | Short leg V1 (ann.) | Short leg V2 (ann.) |
|---|---|---|---|---|---|---|
| 3-1 | 2010-2019 | -3.43% | -2.04% | 1.69 | +47.8% | +30.1% |
| 6-1 | 2010-2019 | -1.64% | -1.18% | 1.39 | +39.1% | +21.6% |
| **12-1 (main)** | 2010-2019 | -2.56% | -1.28% | 2.00 | +42.7% | +22.6% |
| 3-1 | 2020-2025 | -1.14% | -0.98% | 1.16 | +48.8% | +35.0% |
| 6-1 | 2020-2025 | -1.17% | -0.94% | 1.25 | +56.2% | +32.6% |
| 12-1 (main) | 2020-2025 | -0.64% | -1.28% | 0.50 | +48.7% | +31.9% |

Short-leg returns are gross and annualised over the high-volatility months. A short leg that
earns more is worse for the long-short book.

Two of the three parts of the main conclusion hold across lookbacks; the third depends on the
12-1 choice. In-sample, the momentum - reversal gap moves the pre-specified way in all five pairs
(and in all nine pairs of the full 3 x 3 grid, where H3 runs from +1.69 to +2.54). Four of the
five clear the 0.0167 threshold, and 3-1 momentum misses it narrowly at p = 0.017. Held out, every
variant fails to replicate, just as the main specification does: H3 falls to between +0.21 and
+1.19, p is between 0.14 and 0.42, and every interval spans zero. The survivorship claim is the
fragile one. Version 1's mean high-volatility momentum loss is 2.0 times Version 2's at 12-1, but
1.7 times at 3-1 and 1.4 times at 6-1. Held out the ratio is 1.2, 1.2 and 0.5, so at 12-1 the bias
actually points the other way after 2019. What does hold at every lookback and in both periods is
the mechanism behind it: Version 1 inflates momentum's high-volatility short-leg return by 14 to 24
percentage points a year. "Survivorship bias roughly doubles momentum crashes" should therefore be
stated as an in-sample, 12-1 result. The robust version of the claim is that survivorship bias
flatters the short leg in volatile months.

Several results go against the main reading, and I report them as they are. First, the in-sample
consistency of H3 is partly built in. H3 is H1 plus H2, so every momentum variant shares the
same reversal half and every reversal variant shares the same momentum half, and the five pairs
are not five independent confirmations. Looking at the halves separately: at 3-1 and 6-1,
momentum earns roughly nothing in calm months (Sharpe -0.07 and +0.06), its H1 p-value at 6-1 is
0.125, and the Newey-West regime difference at 6-1 is insignificant (t = -1.00). The momentum half
of the story is clearly present only at 12-1 and, more weakly, at 3-1. Second, under the India VIX
regime definition the 3-1 pair fails in-sample (H3 = +1.21, p = 0.22), and the 6-1 pair weakens to
p = 0.025. Third, on gross returns held out, H3 turns negative for 3-1 and 6-1 (-0.21 and -0.33).
Fourth, the one place where a variant looks better held out is Method B: the 3-1 and 6-1 pairs give
p = 0.021 and 0.031. These rest on 7 high-volatility months and 29 high-volatility weeks. They come
almost entirely from the one-week reversal half, which Method B also gives the main specification
(p = 0.081). None of them clears either Bonferroni or Holm (0.11 and 0.12), so I do not read them
as a replication. Finally, the one-month reversal gives the smallest in-sample p-value of any
variant (0.0002). Because it was not the pre-specified choice, it is not promoted to the headline,
and held out it fails like the others (p = 0.15).

All tables, including every regime definition, the gross and 10 and 30 bp cases, the full 3 x 3
grid and the Newey-West regressions, are in `results/phase6_*.csv` and `results/phase6_tables.md`,
produced by `src/run_phase6_robustness.py`.
