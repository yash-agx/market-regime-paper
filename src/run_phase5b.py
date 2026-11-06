"""Phase 5b: download every historical member and build the Version 2 universe.

    python -m src.run_phase5b

Version 1 only ever needed the 50 stocks in the index today. Version 2 needs
every stock the index has EVER held, including ones that were dropped and later
delisted - which is exactly the population survivorship bias is made of.

Some of them will not be on Yahoo Finance at all. That is the part of the bias
Version 2 cannot remove, so it is measured here and reported rather than
quietly left out: for each month, how many of the index's actual members we
have a price for.
"""

from __future__ import annotations

import pandas as pd

from . import config as cfg
from . import data as D
from . import membership as M

V2_SYMBOLS_CSV = cfg.CONFIG_DIR / "universe_v2_symbols.csv"
RAW_V2 = "stocks_v2_raw.csv"
COVERAGE_CSV = cfg.RESULTS_DIR / "phase5b_v2_coverage.csv"


def main() -> None:
    cfg.ensure_dirs()
    cleaned = D.load_processed()
    changes = M.load_changes()
    aliases = M.load_aliases()

    matrix = M.membership_matrix(cleaned.prices.index, changes, aliases)
    ever = sorted(matrix.columns)
    pd.DataFrame({"symbol": ever, "ticker": [f"{s}.NS" for s in ever]}).to_csv(
        V2_SYMBOLS_CSV, index=False)
    print(f"{len(ever)} symbols have been in the Nifty 50 at some point")

    # ---- download the ones Version 1 never needed -------------------------
    have = set(cleaned.prices.columns)
    missing = [f"{s}.NS" for s in ever if f"{s}.NS" not in have]
    path = cfg.RAW_DIR / RAW_V2
    if path.exists():
        extra = pd.read_csv(path, index_col="date", parse_dates=["date"])
    else:
        print(f"Downloading {len(missing)} historical members...")
        extra = D._yf_download(missing, cfg.DATA_START, cfg.DATA_END)
        extra = extra.reindex(columns=missing)
        extra = D._repair_missing_columns(extra)
        extra.to_csv(path, index_label="date")

    got = [c for c in extra.columns if extra[c].notna().sum() > 0]
    gone = [c for c in extra.columns if c not in got]
    print(f"  {len(got)} downloaded, {len(gone)} unavailable on Yahoo: {gone}")

    # ---- how much of the real index can we actually price? ----------------
    all_prices = pd.concat([cleaned.prices, extra.reindex(cleaned.prices.index)], axis=1)
    all_prices = all_prices.loc[:, ~all_prices.columns.duplicated()]
    tickers = M.to_tickers(matrix)
    tickers = tickers.reindex(columns=all_prices.columns.intersection(tickers.columns))

    rows = []
    for date in pd.date_range(cfg.STUDY_START, cfg.STUDY_END, freq="ME"):
        day = all_prices.index[all_prices.index <= date]
        if len(day) == 0:
            continue
        d = day[-1]
        members = [c for c in M.to_tickers(matrix).columns if matrix.at[d, c[:-3]]]
        priced = [c for c in members
                  if c in all_prices.columns and pd.notna(all_prices.at[d, c])]
        rows.append({"date": d.date(), "index_members": len(members),
                     "with_a_price": len(priced),
                     "pct_covered": round(100 * len(priced) / max(len(members), 1), 1),
                     "missing": ", ".join(sorted(set(members) - set(priced)))})
    cov = pd.DataFrame(rows)
    cov.to_csv(COVERAGE_CSV, index=False)

    insample = cov[cov["date"].astype(str) <= cfg.INSAMPLE_END]
    print(f"\n  Version 2 price coverage of actual index members:")
    print(f"    2010-2019 (in-sample): mean {insample['pct_covered'].mean():.1f}%, "
          f"worst {insample['pct_covered'].min():.1f}%")
    print(f"    whole period:          mean {cov['pct_covered'].mean():.1f}%, "
          f"worst {cov['pct_covered'].min():.1f}%")
    never = sorted({t for row in cov["missing"] for t in row.split(", ") if t})
    print(f"\n  Members we can never price ({len(never)}): {never}")
    print(f"\n  Coverage by month -> {COVERAGE_CSV}")

    # ---- save the combined price panel and the membership matrix ----------
    all_prices.to_csv(cfg.PROC_DIR / "prices_all.csv", index_label="date")
    matrix.astype(int).to_csv(M.MEMBERSHIP_CSV, index_label="date")
    print(f"  Prices (both universes) -> {cfg.PROC_DIR / 'prices_all.csv'}")
    print(f"  Membership matrix       -> {M.MEMBERSHIP_CSV}")


if __name__ == "__main__":
    main()
