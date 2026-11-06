"""Sweep NSE's corporate-action record for demergers affecting either universe.

    python -m src.run_corporate_actions

Queries every ticker that appears in the Version 1 universe, the index-change
list, or the rename map, and keeps the actions that mechanically cut the price
without costing the holder anything.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from . import config as cfg
from . import corporate_actions as CA
from . import index_changes as IC


def symbols_in_scope() -> list[str]:
    """Every ticker either universe could ever need a price series for."""
    syms = set(pd.read_csv(cfg.UNIVERSE_CSV)["symbol"])
    changes = pd.read_csv(IC.CHANGES_CSV)
    syms |= {s for s in changes["symbol"].dropna().astype(str) if s and s != "nan"}
    if IC.TICKER_STATUS_CSV.exists():
        ts = pd.read_csv(IC.TICKER_STATUS_CSV)
        syms |= set(ts["old_symbol"]) | set(ts["new_symbol"])
    return sorted(syms)


def main() -> None:
    cfg.ensure_dirs()
    syms = symbols_in_scope()
    print(f"Querying NSE corporate actions for {len(syms)} symbols, 2009-2025...")
    raw = CA.fetch_actions(syms, dt.date(2009, 1, 1), dt.date(2025, 12, 31))
    raw.to_csv(CA.RAW_ACTIONS_CSV, index=False)
    print(f"\n{len(raw)} corporate actions on record in total")

    dem = CA.select_demergers(raw)
    dem.to_csv(CA.CORPORATE_ACTIONS_CSV, index=False)
    print(f"{len(dem)} are demergers, spin-offs or schemes of arrangement:\n")
    if not dem.empty:
        print(dem[["symbol", "ex_date", "subject", "company_name"]].to_string(index=False))
    print(f"\n  -> {CA.CORPORATE_ACTIONS_CSV}")


if __name__ == "__main__":
    main()
