"""Phase 5b: point-in-time Nifty 50 membership (the Version 2 universe).

The idea is simple and the direction matters. We know exactly who is in the
index today, and we know every change NSE announced. So membership is rebuilt
by walking BACKWARDS from today, undoing one change at a time: to step back
over a replacement, put the excluded stock back and take the included one out.

Walking forwards from a guessed 2009 starting list would make the whole series
depend on that guess. Walking backwards anchors it to something checkable, and
gives a free error detector: if the list of changes is incomplete, the count
drifts away from 50 and says so.

What this does NOT fix: a stock that left the index and was later delisted may
have no price history available, so it cannot be traded in the backtest even
though the index held it. That residual is measured and reported rather than
hidden - it is the part of survivorship bias Version 2 cannot remove.
"""

from __future__ import annotations

import pandas as pd

from . import config as cfg
from . import index_changes as IC

MEMBERSHIP_CSV = cfg.PROC_DIR / "membership_v2.csv"


def load_aliases() -> dict[str, str]:
    """Old symbol -> current symbol, for listings that were merely renamed."""
    if not IC.TICKER_STATUS_CSV.exists():
        return {}
    ts = pd.read_csv(IC.TICKER_STATUS_CSV)
    return dict(zip(ts["old_symbol"].astype(str), ts["new_symbol"].astype(str)))


def canonical(symbol: str, aliases: dict[str, str], depth: int = 6) -> str:
    """Follow a chain of renames to the name the listing goes by now.

    Chains are real: SESAGOA -> SSLT -> VEDL, and LTI -> LTIM -> LTM.
    """
    seen = set()
    for _ in range(depth):
        if symbol in seen or symbol not in aliases:
            break
        seen.add(symbol)
        symbol = aliases[symbol]
    return symbol


def load_changes(resolved_csv=None) -> pd.DataFrame:
    """The index-change list, with tickers resolved and supersessions dropped."""
    path = resolved_csv or (cfg.CONFIG_DIR / "index_changes_resolved.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `python -m src.run_phase5a_validate` first")
    df = pd.read_csv(path)
    if "superseded" in df:
        df = df[~df["superseded"].astype(bool)]
    df["symbol_resolved"] = df["symbol_resolved"].fillna("").astype(str)
    return df[df["symbol_resolved"] != ""].copy()


def build_timeline(changes: pd.DataFrame, aliases: dict[str, str]
                   ) -> list[tuple[pd.Timestamp, set[str]]]:
    """Membership segments, newest first: (date the set starts, the set).

    The set returned for a date is who was in the index ON that date.
    """
    today_members = {canonical(s, aliases)
                     for s in pd.read_csv(cfg.UNIVERSE_CSV)["symbol"]}
    dates = sorted(changes["effective_date"].unique(), reverse=True)

    current = set(today_members)
    # The newest segment runs from the last change up to now.
    segments: list[tuple[pd.Timestamp, set[str]]] = []
    if dates:
        segments.append((pd.Timestamp(dates[0]), set(current)))

    for i, d in enumerate(dates):
        sub = changes[changes["effective_date"] == d]
        for s in sub[sub["action"] == "include"]["symbol_resolved"]:
            current.discard(canonical(s, aliases))
        for s in sub[sub["action"] == "exclude"]["symbol_resolved"]:
            current.add(canonical(s, aliases))
        # This set holds from the previous change date until d (exclusive).
        start = pd.Timestamp(dates[i + 1]) if i + 1 < len(dates) else pd.Timestamp("1990-01-01")
        segments.append((start, set(current)))
    return segments


def membership_matrix(calendar: pd.DatetimeIndex, changes: pd.DataFrame,
                      aliases: dict[str, str]) -> pd.DataFrame:
    """Daily boolean matrix: was this symbol in the Nifty 50 on this day?"""
    ordered = sorted(build_timeline(changes, aliases), key=lambda x: x[0])
    all_symbols = sorted({s for _, members in ordered for s in members})
    out = pd.DataFrame(False, index=calendar, columns=all_symbols)

    # Each segment holds from its own start until the next segment begins.
    for i, (start, members) in enumerate(ordered):
        end = ordered[i + 1][0] if i + 1 < len(ordered) else None
        mask = (calendar >= start)
        if end is not None:
            mask &= (calendar < end)
        if mask.any():
            out.loc[mask, sorted(members)] = True
    return out


def to_tickers(matrix: pd.DataFrame) -> pd.DataFrame:
    """Rename columns from NSE symbols to Yahoo tickers."""
    return matrix.rename(columns={c: f"{c}.NS" for c in matrix.columns})
