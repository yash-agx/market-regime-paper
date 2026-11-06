"""Phase 4: regime-conditional performance and the tests of H1-H3.

What gets measured
------------------
Each holding period is labelled by the regime on its START date (the rebalance
day), then the periods are split into two groups and each group gets its own
annualised Sharpe ratio, mean, volatility, drawdown, hit rate and turnover.

Excess returns follow the spec correction: a long-short book is zero-investment
and self-financing, so its return is ALREADY an excess return and the T-bill
rate is not subtracted. Long-only portfolios and the benchmark do tie up
capital, so the T-bill rate is subtracted from those.

The three hypotheses
--------------------
  H1  momentum:  Sharpe(low-vol)  - Sharpe(high-vol)  > 0
  H2  reversal:  Sharpe(high-vol) - Sharpe(low-vol)   > 0
  H3  the momentum-minus-reversal Sharpe gap differs across regimes:
          [Sh_mom(low) - Sh_rev(low)] - [Sh_mom(high) - Sh_rev(high)]

Note that H3 is algebraically H1 + H2, so it cannot be independent evidence -
it is the two effects added together, and it is significant largely when they
are. Worth saying out loud in the paper rather than presenting three findings.

Why a stationary block bootstrap
--------------------------------
Volatility regimes are persistent: high-vol days cluster into episodes lasting
weeks. Resampling individual months would destroy that clustering and make the
regime difference look far better determined than it is. The stationary
bootstrap of Politis and Romano resamples blocks of consecutive periods, with
geometrically distributed block lengths, which keeps the clustering intact.
The regime label travels with its return, so a resampled sample has a realistic
mixture of calm and stormy stretches.

H1 and H2 are bootstrapped on their own series, at their own frequency. H3
mixes a monthly and a weekly strategy, so it is bootstrapped over calendar
MONTHS: a drawn month contributes its momentum observation and all of its
reversal weeks together. That preserves the correlation between the two
strategies, which resampling them independently would throw away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as cfg
from . import strategies as S
from . import windows as W

N_BOOTSTRAP = 10_000
MIN_REGIME_OBS = 3          # a Sharpe needs at least this many periods to mean anything


# --------------------------------------------------------------------------
# excess returns
# --------------------------------------------------------------------------


def period_riskfree(rf_daily: pd.Series, wins: list[W.Window]) -> pd.Series:
    """Risk-free return compounded over each holding period.

    The days used are strictly after the rebalance date, through the end of the
    holding period - exactly the days the portfolio is held.
    """
    out = {}
    for win in wins:
        seg = rf_daily.loc[win.rebalance_date:win.holding_end].iloc[1:]
        out[win.rebalance_date] = float((1 + seg.dropna()).prod() - 1)
    return pd.Series(out).rename("rf_period")


def excess_returns(
    returns: pd.Series, rf_period: pd.Series | None, zero_investment: bool
) -> pd.Series:
    """Subtract the risk-free rate only where capital is actually committed."""
    if zero_investment or rf_period is None:
        return returns
    return (returns - rf_period.reindex(returns.index)).rename(returns.name)


# --------------------------------------------------------------------------
# performance statistics
# --------------------------------------------------------------------------


def sharpe(r: pd.Series | np.ndarray, periods_per_year: int) -> float:
    """Annualised Sharpe ratio of an excess-return series."""
    a = np.asarray(r, dtype="float64")
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return np.nan
    sd = a.std(ddof=1)
    # Not "sd == 0": the standard deviation of a constant series comes out as
    # floating-point dust (around 1e-19), which would divide into a Sharpe of
    # 1e16. A resample that happens to draw one block repeatedly can do this,
    # and a single such draw would wreck the bootstrap percentiles.
    if not np.isfinite(sd) or sd < 1e-12:
        return np.nan
    return float(a.mean() / sd * np.sqrt(periods_per_year))


def regime_table(
    excess: pd.Series,
    flag: pd.Series,
    periods_per_year: int,
    turnover: pd.Series | None = None,
) -> pd.DataFrame:
    """Per-regime performance. Rows: high-vol, low-vol, all."""
    flag = flag.reindex(excess.index)
    groups = {
        "high-vol": excess[flag.fillna(False).astype(bool)],
        "low-vol": excess[(~flag.fillna(True).astype(bool))],
        "all": excess,
    }
    rows = []
    for name, r in groups.items():
        r = r.dropna()
        rec = {
            "regime": name,
            "n_periods": int(len(r)),
            "sharpe": round(sharpe(r, periods_per_year), 3),
            "mean_period_return_pct": round(100 * float(r.mean()), 4) if len(r) else np.nan,
            "ann_return_pct": round(100 * S.annualised_return(r, periods_per_year), 2)
            if len(r) else np.nan,
            "ann_vol_pct": round(100 * S.annualised_vol(r, periods_per_year), 2)
            if len(r) > 1 else np.nan,
            "max_drawdown_pct": round(100 * S.max_drawdown(r), 2) if len(r) else np.nan,
            "hit_rate_pct": round(100 * float((r > 0).mean()), 2) if len(r) else np.nan,
        }
        if turnover is not None:
            rec["mean_turnover"] = round(float(turnover.reindex(r.index).mean()), 3) \
                if len(r) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# the stationary block bootstrap
# --------------------------------------------------------------------------


def stationary_bootstrap_indices(
    n: int, mean_block: float, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano stationary bootstrap index matrix, shape (n_boot, n).

    Each step either continues the current block (wrapping round the end of the
    sample) or jumps to a fresh random start. The chance of jumping is
    1/mean_block, so block lengths are geometric with that mean.
    """
    if n == 0:
        return np.zeros((n_boot, 0), dtype=int)
    p = 1.0 / max(mean_block, 1.0)
    starts = rng.integers(0, n, size=(n_boot, n))
    carry_on = rng.random((n_boot, n)) >= p

    idx = np.empty((n_boot, n), dtype=int)
    current = starts[:, 0]
    idx[:, 0] = current
    for t in range(1, n):
        current = np.where(carry_on[:, t], (current + 1) % n, starts[:, t])
        idx[:, t] = current
    return idx


def default_block(n: int) -> int:
    """n^(1/3), the usual rule of thumb, floored at 2 periods."""
    return max(2, int(round(n ** (1 / 3))))


@dataclass
class BootstrapResult:
    name: str
    observed: float
    mean: float
    ci_low: float
    ci_high: float
    p_one_sided: float          # share of resamples at or below zero
    n_valid: int
    n_boot: int
    mean_block: float
    note: str = ""

    def as_row(self) -> dict:
        return {
            "hypothesis": self.name,
            "observed": round(self.observed, 4),
            "bootstrap_mean": round(self.mean, 4),
            "ci95_low": round(self.ci_low, 4),
            "ci95_high": round(self.ci_high, 4),
            "p_one_sided": round(self.p_one_sided, 4),
            "n_valid_resamples": self.n_valid,
            "mean_block": self.mean_block,
            "note": self.note,
        }


def _summarise_bootstrap(
    name: str, observed: float, draws: np.ndarray, mean_block: float, n_boot: int,
    note: str = "",
) -> BootstrapResult:
    valid = draws[np.isfinite(draws)]
    if len(valid) == 0:
        return BootstrapResult(name, observed, np.nan, np.nan, np.nan, np.nan,
                               0, n_boot, mean_block, note or "no valid resamples")
    return BootstrapResult(
        name=name,
        observed=observed,
        mean=float(valid.mean()),
        ci_low=float(np.percentile(valid, 2.5)),
        ci_high=float(np.percentile(valid, 97.5)),
        # One-sided: how often the resampled difference fails to be positive.
        p_one_sided=float((valid <= 0).mean()),
        n_valid=int(len(valid)),
        n_boot=n_boot,
        mean_block=mean_block,
        note=note,
    )


def bootstrap_sharpe_gap(
    excess: pd.Series,
    flag: pd.Series,
    periods_per_year: int,
    name: str,
    favour: str,                     # "low" for H1, "high" for H2
    n_boot: int = N_BOOTSTRAP,
    seed: int = cfg.RANDOM_SEED,
    mean_block: float | None = None,
) -> BootstrapResult:
    """Bootstrap Sharpe(favoured regime) - Sharpe(other regime).

    The return and its regime label are resampled together, in blocks, so the
    persistence of regimes survives the resampling.
    """
    df = pd.DataFrame({"r": excess, "high": flag.reindex(excess.index)}).dropna()
    r = df["r"].to_numpy(dtype="float64")
    hi = df["high"].to_numpy(dtype=bool)
    n = len(r)
    if mean_block is None:
        mean_block = default_block(n)

    def stat(rv: np.ndarray, hv: np.ndarray) -> float:
        a, b = rv[~hv], rv[hv]
        if len(a) < MIN_REGIME_OBS or len(b) < MIN_REGIME_OBS:
            return np.nan
        s_low, s_high = sharpe(a, periods_per_year), sharpe(b, periods_per_year)
        return s_low - s_high if favour == "low" else s_high - s_low

    observed = stat(r, hi)
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(n, mean_block, n_boot, rng)
    draws = np.array([stat(r[row], hi[row]) for row in idx])

    dropped = int(np.sum(~np.isfinite(draws)))
    note = (f"{dropped} resamples dropped for having under {MIN_REGIME_OBS} "
            f"periods in a regime") if dropped else ""
    return _summarise_bootstrap(name, observed, draws, mean_block, n_boot, note)


def bootstrap_difference_in_differences(
    mom_excess: pd.Series,
    mom_flag: pd.Series,
    rev_excess: pd.Series,
    rev_flag: pd.Series,
    n_boot: int = N_BOOTSTRAP,
    seed: int = cfg.RANDOM_SEED,
    mean_block: float | None = None,
) -> BootstrapResult:
    """H3, resampled over calendar months so the two strategies move together.

    A drawn month brings its momentum observation and all of its reversal weeks
    with it. Resampling the two series independently would assume momentum and
    reversal are unrelated, which they are not.
    """
    mom = pd.DataFrame({"r": mom_excess, "high": mom_flag.reindex(mom_excess.index)}).dropna()
    rev = pd.DataFrame({"r": rev_excess, "high": rev_flag.reindex(rev_excess.index)}).dropna()

    months = sorted(set(mom.index.to_period("M")) | set(rev.index.to_period("M")))
    mom_by_month = {m: g for m, g in mom.groupby(mom.index.to_period("M"))}
    rev_by_month = {m: g for m, g in rev.groupby(rev.index.to_period("M"))}

    # Pre-extract to arrays so the inner loop is cheap.
    packs = []
    for m in months:
        mg, rg = mom_by_month.get(m), rev_by_month.get(m)
        packs.append((
            (mg["r"].to_numpy(), mg["high"].to_numpy(dtype=bool)) if mg is not None
            else (np.empty(0), np.empty(0, dtype=bool)),
            (rg["r"].to_numpy(), rg["high"].to_numpy(dtype=bool)) if rg is not None
            else (np.empty(0), np.empty(0, dtype=bool)),
        ))

    def stat_from_months(order: np.ndarray) -> float:
        mr = np.concatenate([packs[i][0][0] for i in order])
        mh = np.concatenate([packs[i][0][1] for i in order])
        rr = np.concatenate([packs[i][1][0] for i in order])
        rh = np.concatenate([packs[i][1][1] for i in order])
        if (min(len(mr[~mh]), len(mr[mh])) < MIN_REGIME_OBS
                or min(len(rr[~rh]), len(rr[rh])) < MIN_REGIME_OBS):
            return np.nan
        gap_low = (sharpe(mr[~mh], cfg.MONTHS_PER_YEAR)
                   - sharpe(rr[~rh], cfg.WEEKS_PER_YEAR))
        gap_high = (sharpe(mr[mh], cfg.MONTHS_PER_YEAR)
                    - sharpe(rr[rh], cfg.WEEKS_PER_YEAR))
        return gap_low - gap_high

    n = len(months)
    if mean_block is None:
        mean_block = default_block(n)
    observed = stat_from_months(np.arange(n))
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(n, mean_block, n_boot, rng)
    draws = np.array([stat_from_months(row) for row in idx])

    dropped = int(np.sum(~np.isfinite(draws)))
    note = "resampled over calendar months to keep the two strategies aligned"
    if dropped:
        note += f"; {dropped} resamples dropped for a near-empty regime"
    return _summarise_bootstrap("H3", observed, draws, mean_block, n_boot, note)


# --------------------------------------------------------------------------
# Newey-West regression on a high-volatility dummy
# --------------------------------------------------------------------------


def newey_west_lags(n: int) -> int:
    """The usual automatic rule, 4*(n/100)^(2/9)."""
    return max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))


def newey_west_dummy(
    excess: pd.Series, flag: pd.Series, label: str, lags: int | None = None
) -> dict:
    """Regress the return on a high-volatility dummy with Newey-West errors.

    The slope is the difference in MEAN return between regimes (not Sharpe),
    with standard errors robust to the autocorrelation that regime persistence
    creates.
    """
    import statsmodels.api as sm

    df = pd.DataFrame({"r": excess, "high": flag.reindex(excess.index)}).dropna()
    if len(df) < 10 or df["high"].nunique() < 2:
        return {"strategy": label, "n": len(df), "note": "too few observations"}
    y = df["r"].to_numpy(dtype="float64")
    X = sm.add_constant(df["high"].to_numpy(dtype="float64"))
    if lags is None:
        lags = newey_west_lags(len(df))
    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {
        "strategy": label,
        "n": int(len(df)),
        "nw_lags": lags,
        "mean_low_vol_pct": round(100 * float(fit.params[0]), 4),
        "diff_high_minus_low_pct": round(100 * float(fit.params[1]), 4),
        "t_stat": round(float(fit.tvalues[1]), 3),
        "p_value_two_sided": round(float(fit.pvalues[1]), 4),
        "r_squared": round(float(fit.rsquared), 4),
    }
