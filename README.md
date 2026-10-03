# Market-Regime & Statistical-Arbitrage Research Paper

**To what extent does market-regime variation affect the out-of-sample performance of momentum
and mean-reversion strategies?**

A research project, not a trading bot. The point is not to find a profitable strategy but to
design a fair experiment: regime labels built only from past data, walk-forward testing, a final
out-of-sample period left untouched until the end, and realistic transaction costs shown
alongside the gross numbers.

Python: pandas, NumPy, matplotlib, statsmodels, a free price source, pytest.

## Status

Not started. `the README` holds the plan and the rules — data and universe choice, the two
strategies, regime classification, walk-forward testing, per-regime metrics, then the paper.

## Rules that shape the code

- No look-ahead bias anywhere; anything that risks it gets flagged before it is written.
- Fixed seeds and saved CSVs, so every figure in the paper is reproducible.
- The writing is mine — the code and analysis are the deliverables here.
