# Is H3 a market-beta effect counted twice?

*Phase 7, pre-registered in `docs/phase7_preregistration.md`. Full tables:
`results/phase7_tables.md`.*

**The question.** High-volatility months are often rebounds. In a rebound,
momentum's short leg and reversal's long leg are both made of high-beta losers. If
that is the whole story, "momentum is worse in high vol" (H1) and "reversal is better
in high vol" (H2) are the same market bet seen twice, and H3 just adds the two
together. We test this the way Daniel and Moskowitz (2016) do: we let each spread's
market beta differ between regimes, take that exposure out, and recompute H3 on what
is left.

**Main result (Version 2, Method A, 20 bp, 2010-2019).**

| | H3 | 95% interval |
|---|---|---|
| Raw (as published) | +2.54 | [+0.58, +4.86] |
| Beta-adjusted | +1.90 | [-0.07, +4.50] |
| Explained by beta | +0.65 | [-0.71, +1.63] |

**About three-quarters of H3 survives (f = 0.75).** The pre-registered reading is
**"partly explained"**, not "survives". The share surviving is above the 2/3 line,
but the adjusted interval just reaches zero (one-sided p = 0.029, above the 0.0167
Bonferroni level).

**What the beta terms show.**

- **The betas move the way the explanation predicts.** For momentum 12-1, beta goes
  from -0.51 in calm months to -1.14 in high-vol months. For reversal 1 week, it goes
  from +0.11 to +0.45. All six spreads have the predicted sign in-sample.
- **Most of the explained part is momentum's.** H1 falls from 1.15 to 0.68 after
  hedging. H2 hardly moves (1.39 to 1.21).
- **The first step of the story fails in-sample.** High-vol months in 2010-2019 were
  *not* mostly up-markets: the market rose in 48% of them, against 51% of calm months.
  The market's mean return was higher in high-vol months (+1.2% against +0.0%), but
  that comes from bigger moves, not more frequent rebounds. Held out, the premise does
  hold: the market rose in 79% of high-vol months.
- **The two spreads stay linked after hedging.** In high-vol months the
  momentum-reversal correlation is -0.43 before hedging and -0.36 after. Market beta
  is not what ties them together.
- **Momentum shows the Daniel-Moskowitz bear-market effect; reversal does not.** In
  bear states (past 24-month index return below zero), momentum 12-1 loses beta
  sharply when the market goes up: bBU = -2.45, t = -2.76, on 14 bear months. That is
  the "short a call" payoff. Reversal's bBU is small and has the wrong sign.

**Robustness.**

- **Slower regime label** (126-day variance): f = 0.85, adjusted interval
  [-0.07, +4.60]. Same verdict, partly explained.
- **Version 1** (today's members): f = 0.82, adjusted H3 +2.45 [+0.46, +4.87].
  Verdict: survives.
- **Other lookback pairs, in-sample V2:** f is 0.67 to 0.85 for all five. 12-1/2w and
  12-1/1m survive; 3-1/1w and 6-1/1w are partly explained.
- **Held out (2020-2025):** f = 0.47. Raw H3 was already not significant there, so
  this ratio of two noisy numbers is not interpreted.

**Honest conclusion.** The beta-timing explanation is partly right. Both spreads'
betas shift between regimes in the predicted direction, and taking that out removes
about a quarter of H3, mostly from the momentum side. But H3 is not mostly a beta
effect counted twice:

- three-quarters of the point estimate remains after hedging;
- the high-vol months were not mainly rebound months in-sample;
- the two spreads remain negatively correlated after the beta terms are removed.

What changes is how strongly we can state the result. Once beta is allowed for, the
remaining regime interaction can no longer be distinguished from zero at the 95%
level in the headline specification. The paper should therefore say that H3 is
*largely, but not robustly,* separate from time-varying market beta. Two weaknesses
limit any stronger claim in either direction:

- only 25 high-vol months in-sample;
- held out, the premise holds but the result itself does not replicate.
