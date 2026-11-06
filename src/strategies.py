"""Phase 3: the momentum and reversal strategies, and what trading them costs.

How a single rebalance works, in plain words:

  1.  On the rebalance day, work out which stocks are allowed (252 days of
      history, no more than 5% of the formation window missing, no 10-day flat
      run touching the window, and a price today so the trade can be done).
  2.  Compute each allowed stock's signal from its formation window only.
  3.  Sort on the signal and split into five equal groups. Momentum buys the
      top group; reversal buys the bottom group. The long-short version also
      sells the opposite group.
  4.  Hold until the next rebalance. Within the holding period nothing is
      traded, so the equal weights drift with prices - which is what a real
      portfolio does, and is what makes next month's turnover less than 100%
      when some names repeat.
  5.  Charge costs on everything actually traded.

Two conventions worth stating because papers differ:

  * The long-short book is $1 long and $1 short, so its return is
    (long leg) - (short leg). It is self-financing, so this is already an
    excess return and the risk-free rate is NOT subtracted from it. The
    risk-free rate is subtracted only from the long-only strategies and the
    benchmark, which do tie up capital.

  * Turnover is reported as the value traded per rupee of capital,
    sum_i |w_new,i - w_drifted,i| summed over both legs. Cost is then simply
    (one-way bps) x turnover - every rupee sold and every rupee bought is
    charged once. A long-short book that replaces all its holdings trades
    4.0 by this measure (sell $2, buy $2), costing 80bp at 20bp one-way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as cfg
from . import data as D
from . import windows as W

COST_BPS = (10, 20, 30)
BASE_COST_BPS = 20


# --------------------------------------------------------------------------
# signals
# --------------------------------------------------------------------------


def window_return(prices: pd.DataFrame, win: W.Window) -> pd.Series:
    """Return over the formation window, from its first price to its last.

    For momentum this is months t-12 to t-1 (the skip-a-month 12-1 signal);
    for reversal it is the one-week return. Both read prices only inside the
    window, which ends on or before the rebalance day.
    """
    p0 = prices.loc[win.formation_start]
    p1 = prices.loc[win.formation_end]
    return (p1 / p0 - 1).rename("signal")


def holding_return(prices: pd.DataFrame, win: W.Window) -> pd.Series:
    """Return from the rebalance day to the end of the holding period.

    If a stock stops trading part-way through, its last traded price is used -
    the position is held to the end of the period, as the spec requires, rather
    than being quietly dropped.
    """
    p0 = prices.loc[win.rebalance_date]
    held = prices.loc[win.rebalance_date:win.holding_end]
    p1 = held.ffill().iloc[-1]
    return (p1 / p0 - 1).rename("holding_return")


# --------------------------------------------------------------------------
# one leg of a portfolio
# --------------------------------------------------------------------------


def _drifted(weights: pd.Series, rets: pd.Series) -> pd.Series:
    """Where the weights end up after a holding period with no trading."""
    if weights.empty:
        return weights
    grown = weights * (1 + rets.reindex(weights.index).fillna(0.0))
    total = grown.sum()
    return grown / total if total != 0 else grown


def _traded(new: pd.Series, old: pd.Series) -> float:
    """Value traded per rupee of capital to move from `old` to `new`."""
    idx = new.index.union(old.index)
    return float((new.reindex(idx).fillna(0.0) - old.reindex(idx).fillna(0.0)).abs().sum())


# --------------------------------------------------------------------------
# the strategy engine
# --------------------------------------------------------------------------


@dataclass
class StrategyResult:
    name: str
    frequency: str                        # "monthly" or "weekly"
    gross: pd.Series                      # period return before costs
    turnover: pd.Series                   # value traded per rupee of capital
    n_long: pd.Series
    n_short: pd.Series
    n_eligible: pd.Series
    r_long_leg: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    r_short_leg: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    net: dict[int, pd.Series] = field(default_factory=dict)
    holdings: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def periods_per_year(self) -> int:
        return cfg.MONTHS_PER_YEAR if self.frequency == "monthly" else cfg.WEEKS_PER_YEAR

    def ret(self, cost_bps: int | None = BASE_COST_BPS) -> pd.Series:
        """Net return at the given cost, or the gross return when cost_bps is None."""
        return self.gross if cost_bps is None else self.net[cost_bps]


def run_strategy(
    cleaned: D.CleanedData,
    wins: list[W.Window],
    long_quintile: str,
    short_quintile: str | None,
    name: str,
    frequency: str,
    runs: pd.DataFrame | None = None,
    cost_bps: tuple[int, ...] = COST_BPS,
    equal_weight_all: bool = False,
) -> StrategyResult:
    """Run one cross-sectional strategy over a list of rebalance windows.

    long_quintile / short_quintile are "top" or "bottom" (or None for no short
    leg). equal_weight_all ignores the signal entirely and holds every eligible
    stock - that is the benchmark.
    """
    prices = cleaned.prices
    if runs is None:
        runs = D.flat_run_length(prices)

    rows, holdings = [], []
    prev_long = pd.Series(dtype="float64")
    prev_short = pd.Series(dtype="float64")

    for win in wins:
        allowed = D.eligible_at(
            cleaned, win.rebalance_date, win.formation_start, win.formation_end, runs=runs
        )
        signal = window_return(prices, win).reindex(allowed).dropna()
        hold = holding_return(prices, win)

        if equal_weight_all:
            long_names, short_names = signal.index, pd.Index([])
        elif len(signal) < 5:
            long_names, short_names = pd.Index([]), pd.Index([])
        else:
            # Five equal groups on the signal; group 4 is the highest.
            bucket = pd.qcut(signal.rank(method="first"), 5, labels=False)
            top = signal.index[bucket == 4]
            bottom = signal.index[bucket == 0]
            long_names = top if long_quintile == "top" else bottom
            if short_quintile is None:
                short_names = pd.Index([])
            else:
                short_names = top if short_quintile == "top" else bottom

        w_long = pd.Series(1.0 / len(long_names), index=long_names) if len(long_names) else pd.Series(dtype="float64")
        w_short = pd.Series(1.0 / len(short_names), index=short_names) if len(short_names) else pd.Series(dtype="float64")

        r_long = float((w_long * hold.reindex(w_long.index)).sum()) if len(w_long) else 0.0
        r_short = float((w_short * hold.reindex(w_short.index)).sum()) if len(w_short) else 0.0
        gross = r_long - r_short

        traded = _traded(w_long, prev_long) + _traded(w_short, prev_short)

        rows.append({
            "rebalance_date": win.rebalance_date,
            "holding_end": win.holding_end,
            "gross": gross,
            "turnover": traded,
            "n_long": len(long_names),
            "n_short": len(short_names),
            "n_eligible": len(signal),
            "r_long_leg": r_long,
            "r_short_leg": r_short,
        })
        for ticker in long_names:
            holdings.append({"rebalance_date": win.rebalance_date, "ticker": ticker,
                             "side": "long", "signal": signal.get(ticker, np.nan),
                             "holding_return": hold.get(ticker, np.nan)})
        for ticker in short_names:
            holdings.append({"rebalance_date": win.rebalance_date, "ticker": ticker,
                             "side": "short", "signal": signal.get(ticker, np.nan),
                             "holding_return": hold.get(ticker, np.nan)})

        prev_long = _drifted(w_long, hold)
        prev_short = _drifted(w_short, hold)

    df = pd.DataFrame(rows).set_index("rebalance_date")
    result = StrategyResult(
        name=name, frequency=frequency,
        gross=df["gross"], turnover=df["turnover"],
        n_long=df["n_long"], n_short=df["n_short"], n_eligible=df["n_eligible"],
        r_long_leg=df["r_long_leg"], r_short_leg=df["r_short_leg"],
        holdings=pd.DataFrame(holdings),
    )
    for bps in cost_bps:
        result.net[bps] = (df["gross"] - (bps / 10_000.0) * df["turnover"]).rename(
            f"net_{bps}bps")
    return result


# --------------------------------------------------------------------------
# the five portfolios in the spec
# --------------------------------------------------------------------------


def build_all(cleaned: D.CleanedData, study_only: bool = True) -> dict[str, StrategyResult]:
    """Momentum and reversal, long-short and long-only, plus the benchmark."""
    runs = D.flat_run_length(cleaned.prices)
    cal = cleaned.prices.index

    def in_study(wins):
        if not study_only:
            return wins
        lo, hi = pd.Timestamp(cfg.STUDY_START), pd.Timestamp(cfg.STUDY_END)
        return [w for w in wins if lo <= w.rebalance_date <= hi]

    mom_wins = in_study(W.momentum_windows(cal))
    rev_wins = in_study(W.reversal_windows(cal))

    out = {}
    out["momentum_ls"] = run_strategy(
        cleaned, mom_wins, "top", "bottom", "Momentum (long-short)", "monthly", runs)
    out["momentum_lo"] = run_strategy(
        cleaned, mom_wins, "top", None, "Momentum (long-only)", "monthly", runs)
    out["reversal_ls"] = run_strategy(
        cleaned, rev_wins, "bottom", "top", "Reversal (long-short)", "weekly", runs)
    out["reversal_lo"] = run_strategy(
        cleaned, rev_wins, "bottom", None, "Reversal (long-only)", "weekly", runs)
    out["benchmark_monthly"] = run_strategy(
        cleaned, mom_wins, "top", None, "Equal-weighted Nifty 50 (monthly)", "monthly",
        runs, equal_weight_all=True)
    out["benchmark_weekly"] = run_strategy(
        cleaned, rev_wins, "bottom", None, "Equal-weighted Nifty 50 (weekly)", "weekly",
        runs, equal_weight_all=True)
    return out


# --------------------------------------------------------------------------
# performance statistics
# --------------------------------------------------------------------------


def annualised_return(r: pd.Series, periods_per_year: int) -> float:
    """Geometric (compound) annual growth rate."""
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    total = float((1 + r).prod())
    if total <= 0:
        return np.nan
    return total ** (periods_per_year / len(r)) - 1


def annualised_vol(r: pd.Series, periods_per_year: int) -> float:
    r = r.dropna()
    return float(r.std() * np.sqrt(periods_per_year)) if len(r) > 1 else np.nan


def max_drawdown(r: pd.Series) -> float:
    """Worst peak-to-trough fall of the compounded series, as a negative number."""
    r = r.dropna()
    if r.empty:
        return np.nan
    curve = (1 + r).cumprod()
    return float((curve / curve.cummax() - 1).min())


def summarise(
    result: StrategyResult, cost_bps: int | None, label: str = ""
) -> dict:
    r = result.ret(cost_bps).dropna()
    ppy = result.periods_per_year
    return {
        "strategy": result.name,
        "period": label,
        "cost_bps": "gross" if cost_bps is None else cost_bps,
        "n_periods": int(len(r)),
        "ann_return_pct": round(100 * annualised_return(r, ppy), 2),
        "ann_vol_pct": round(100 * annualised_vol(r, ppy), 2),
        "mean_period_return_pct": round(100 * float(r.mean()), 4),
        "max_drawdown_pct": round(100 * max_drawdown(r), 2),
        "hit_rate_pct": round(100 * float((r > 0).mean()), 2),
        "mean_turnover": round(float(result.turnover.reindex(r.index).mean()), 3),
        "ann_turnover": round(float(result.turnover.reindex(r.index).mean()) * ppy, 2),
        "mean_n_long": round(float(result.n_long.reindex(r.index).mean()), 1),
        "mean_n_short": round(float(result.n_short.reindex(r.index).mean()), 1),
        "mean_n_eligible": round(float(result.n_eligible.reindex(r.index).mean()), 1),
    }


def slice_period(result: StrategyResult, lo: str, hi: str) -> StrategyResult:
    """A copy of the result restricted to rebalance dates in [lo, hi]."""
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    sel = (result.gross.index >= lo) & (result.gross.index <= hi)
    out = StrategyResult(
        name=result.name, frequency=result.frequency,
        gross=result.gross[sel], turnover=result.turnover[sel],
        n_long=result.n_long[sel], n_short=result.n_short[sel],
        n_eligible=result.n_eligible[sel],
        r_long_leg=result.r_long_leg[sel], r_short_leg=result.r_short_leg[sel],
        holdings=result.holdings,
    )
    out.net = {k: v[sel] for k, v in result.net.items()}
    return out
