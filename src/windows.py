"""Rebalance dates and formation windows.

Separated from the strategies themselves because the data-quality rules
(missing data, staleness) have to be applied over exactly the same windows the
signals are computed from, and Phase 1 needs to count exclusions before the
strategies exist.

Two schedules:

  * momentum - rebalance on the last trading day of each month. The signal uses
    months t-11 to t-1, i.e. the price span from the last trading day of month
    t-12 to the last trading day of month t-1. Month t is skipped, so the
    formation window ends a month before the trade.

  * reversal - rebalance on the last trading day of each week. The signal is
    that week's return, i.e. the price span from the previous week end to this
    week end. The formation window ends on the rebalance day itself.

A "formation window" below is the span of prices a signal reads. Returns need
one price before the first return, so the window starts at the base price.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Window:
    """One rebalance: when it happens and which price span its signal reads."""

    rebalance_date: pd.Timestamp   # when the portfolio is formed
    formation_start: pd.Timestamp  # base price of the signal
    formation_end: pd.Timestamp    # last price the signal reads
    holding_end: pd.Timestamp      # last day of the holding period

    @property
    def label(self) -> str:
        return self.rebalance_date.strftime("%Y-%m-%d")


def month_end_dates(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last actual trading day of each calendar month."""
    s = pd.Series(calendar, index=calendar)
    return pd.DatetimeIndex(s.groupby(calendar.to_period("M")).max().to_numpy())


def week_end_dates(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last actual trading day of each calendar week (Mon-Sun weeks)."""
    s = pd.Series(calendar, index=calendar)
    return pd.DatetimeIndex(s.groupby(calendar.to_period("W")).max().to_numpy())


def momentum_windows(calendar: pd.DatetimeIndex) -> list[Window]:
    """12-1 momentum: skip the most recent month, hold one month."""
    ends = month_end_dates(calendar)
    out = []
    for k in range(12, len(ends) - 1):
        out.append(Window(
            rebalance_date=ends[k],
            formation_start=ends[k - 12],
            formation_end=ends[k - 1],
            holding_end=ends[k + 1],
        ))
    return out


def reversal_windows(calendar: pd.DatetimeIndex) -> list[Window]:
    """One-week reversal: signal is this week's return, hold one week."""
    ends = week_end_dates(calendar)
    out = []
    for j in range(1, len(ends) - 1):
        out.append(Window(
            rebalance_date=ends[j],
            formation_start=ends[j - 1],
            formation_end=ends[j],
            holding_end=ends[j + 1],
        ))
    return out
