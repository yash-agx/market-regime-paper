# Momentum vs Mean-Reversion Across Volatility Regimes

**Does the relative performance of momentum and short-term mean-reversion strategies on Nifty 50
stocks (2010–2025) depend on the market's volatility regime, measured by the annualised Sharpe
ratio net of transaction costs?**

A research project, not a trading bot. The point is not to find a profitable strategy but to run a
fair experiment: regime labels built only from past data, methods frozen before results are seen,
a held-out 2020–2025 period, realistic transaction costs, and formal tests of whether the
differences are real.

## Design at a glance

| Part | Choice |
|---|---|
| Universe | Nifty 50 stocks, today's members and historical point-in-time members |
| Period | 2010–2025 daily data (2010–2019 in-sample, 2020–2025 held out) |
| Momentum | 12-1 month signal, top vs bottom quintile, monthly rebalance |
| Mean reversion | 1-week reversal, bottom vs top quintile, weekly rebalance |
| Regimes | Realised volatility vs expanding median (main); 2-state HMM with filtered probabilities; India VIX |
| Costs | 20 bps one-way (also 10 and 30 bps) |
| Evaluation | Sharpe ratio per regime, block-bootstrap tests, Newey–West regression |

## Hypotheses

- **H1:** momentum does better in low-volatility regimes.
- **H2:** reversal does better in high-volatility regimes.
- **H3:** the gap between them changes across regimes, and the change is statistically significant.

## Status

Spec frozen. Code not started. See `the README` for the full spec and the five build phases: data,
regimes, strategies, evaluation, and survivorship-bias correction.

## Repo layout

```
config/      universe and membership lists
data/        raw and processed price data
src/         data, regime, strategy and evaluation code
tests/       look-ahead-bias and validation tests (pytest)
results/     every table, saved as CSV
analysis/    scripts that make every figure
```

## Reproducibility

Fixed random seeds and saved CSVs, so every table and figure in the paper can be rebuilt by
rerunning the code. The risk-free rate file (`data/raw/tbill_91d.csv`) comes from the RBI and is
added by hand.

## Stack

Python: pandas, NumPy, matplotlib, statsmodels, hmmlearn, yfinance, pytest.
