# Momentum vs Mean-Reversion Across Volatility Regimes (Nifty 50, 2010-2025)

## Research question

Does the relative performance of momentum and short-term mean-reversion strategies on Nifty 50
stocks (2010-2025) depend on the market's volatility regime, measured by the annualised Sharpe
ratio net of transaction costs?

## What I did

I build two long-short strategies on Nifty 50 stocks from daily data: a 12-1 month momentum sort
(top versus bottom quintile, rebalanced monthly) and a one-week reversal sort (bottom versus top
quintile, rebalanced weekly), then label every holding period by the volatility regime in force at
its start. Regimes come from 21-day realised volatility of the Nifty 50 index against an expanding
median, with a two-state hidden Markov model on filtered probabilities and India VIX as
alternatives; every regime label, signal and eligibility screen uses only data available at the
time, so the whole thing runs walk-forward with no look-ahead. To remove survivorship bias I
rebuild the index membership point-in-time from NSE index-change announcements, each with a source
URL, and report that universe as the headline alongside today's membership for comparison. All
returns are shown gross and net of transaction costs (20 bps one-way as the base case, with 10 and
30 bps as robustness), and 2020-2025 was held out and left untouched until the 2010-2019 analysis
was final.

Results will be added once the paper is published.

## Hypotheses

- **H1:** momentum earns a higher Sharpe ratio in low-volatility regimes than in high-volatility regimes.
- **H2:** short-term reversal earns a higher Sharpe ratio in high-volatility regimes than in low-volatility regimes.
- **H3:** the momentum minus reversal Sharpe gap changes across regimes, and the interaction is statistically significant. This is the primary test.

## How to run it

Python 3.12 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The price data is downloaded rather than committed, so the phases run in order:

```bash
python -m src.run_phase1                     # download and clean prices, eligibility screens
python -m src.run_phase2                     # volatility regimes (realised vol, HMM, VIX)
python -m src.run_phase3                     # strategy returns, costs, turnover
python -m src.run_phase5a                    # index changes -> point-in-time membership
python -m src.run_phase5b                    # membership coverage checks
python -m src.run_phase4                     # evaluation and tests, in-sample 2010-2019
python -m src.run_phase4 --holdout           # the held-out 2020-2025 period
```

`run_phase1` takes `--force` to re-download instead of using the cache. `run_phase4` takes
`--universe v2` (point-in-time membership, the default and the headline) or `--universe v1`
(today's members). The held-out period sits behind a flag on purpose.

Figures:

```bash
python -m analysis.fig_phase2_regimes
python -m analysis.fig_phase4_cumulative
python -m analysis.fig_phase4_cumulative --holdout
```

Tests, including the look-ahead checks that truncate the data at a date and confirm signals,
regimes and the universe up to that date do not change:

```bash
pytest
```

## Repo layout

```
config/      universe lists, index changes, corporate actions, ticker resolution
data/raw/    downloaded prices and the hand-added 91-day T-bill rate file
data/processed/
src/         data, regimes, strategies, membership and evaluation code, plus phase entry points
analysis/    one script per figure
tests/       look-ahead and validation tests (pytest)
results/     every table as a CSV, every figure under results/figures/
```

The risk-free rate file (`data/raw/tbill_91d.csv`) is the RBI 91-day T-bill yield and is added by
hand, since it cannot be re-downloaded by the scripts. Everything else in `data/` is rebuilt by
Phase 1.

## Reproducibility

Fixed random seeds throughout, every table written to `results/` as a CSV, and every figure
produced by a script in `analysis/`, so every number in the paper can be rebuilt from a clean
checkout by rerunning the phases above.

## Notes

Built with AI coding assistance. The research question, design and write-up are my own.
