"""Audit the staleness rule: which stock-windows does it actually exclude?

    python -m src.run_staleness_report

Walks every momentum and reversal formation window in the sample, applies the
rule point-in-time, and reports every exclusion. The point is to confirm the
rule removes the vendor artefact and nothing else.
"""

from __future__ import annotations

import pandas as pd

from . import config as cfg
from . import data as D
from . import windows as W


def audit(cleaned: D.CleanedData) -> dict[str, pd.DataFrame]:
    runs = D.flat_run_length(cleaned.prices)
    out = {}

    for name, builder in (("momentum", W.momentum_windows), ("reversal", W.reversal_windows)):
        rows = []
        for win in builder(cleaned.prices.index):
            # Only stocks that would otherwise have been usable can be "excluded".
            has_history = cleaned.eligible.loc[:win.rebalance_date].iloc[-1]
            tradable = cleaned.prices.loc[:win.rebalance_date].iloc[-1].notna()
            missing = D.formation_missing_fraction(
                cleaned.prices, win.formation_start, win.formation_end
            )
            candidate = has_history & tradable & (missing <= cfg.MAX_MISSING_FRACTION)

            stale = D.stale_in_window(runs, win.formation_start, win.formation_end)
            strict = D.stale_in_window(
                runs, win.formation_start, win.formation_end, strict_containment=True
            )
            longest = runs.loc[win.formation_start:win.formation_end].max()

            for ticker in cleaned.prices.columns[(candidate & stale).to_numpy()]:
                rows.append({
                    "strategy": name,
                    "rebalance_date": win.rebalance_date,
                    "formation_start": win.formation_start,
                    "formation_end": win.formation_end,
                    "ticker": ticker,
                    "longest_flat_run_touching_window": int(longest[ticker]),
                    "window_trading_days": len(
                        cleaned.prices.loc[win.formation_start:win.formation_end]
                    ),
                    "also_excluded_under_strict_containment": bool(strict[ticker]),
                    "n_candidates_in_window": int(candidate.sum()),
                })
        out[name] = pd.DataFrame(rows)
    return out


def main() -> None:
    cfg.ensure_dirs()
    cleaned = D.load_processed()
    audits = audit(cleaned)

    combined = pd.concat(audits.values(), ignore_index=True)
    combined.to_csv(cfg.RESULTS_DIR / "phase1_staleness_exclusions.csv", index=False)

    print("=== Staleness rule audit (>= %d identical consecutive closes) ===\n"
          % cfg.MAX_FLAT_RUN_DAYS)

    for name, df in audits.items():
        total_windows = len(
            W.momentum_windows(cleaned.prices.index) if name == "momentum"
            else W.reversal_windows(cleaned.prices.index)
        )
        print(f"{name}: {len(df)} stock-windows excluded, over {total_windows} windows")
        if df.empty:
            print("  (none)\n")
            continue
        for ticker, grp in df.groupby("ticker"):
            print(f"  {ticker:16s} {len(grp):4d} windows excluded, "
                  f"rebalances {grp['rebalance_date'].min().date()} .. "
                  f"{grp['rebalance_date'].max().date()}")
        n_strict = int(df["also_excluded_under_strict_containment"].sum())
        print(f"  under strict containment instead: {n_strict} of {len(df)} "
              f"would still be excluded\n")

    print("Tickers excluded anywhere:", sorted(combined["ticker"].unique()) or "none")

    # When does each affected stock come back?
    print("\nFirst clean window after the last exclusion:")
    for name, df in audits.items():
        if df.empty:
            continue
        all_windows = (W.momentum_windows if name == "momentum" else W.reversal_windows)(
            cleaned.prices.index
        )
        for ticker, grp in df.groupby("ticker"):
            last_bad = grp["rebalance_date"].max()
            later = [w for w in all_windows if w.rebalance_date > last_bad]
            if later:
                print(f"  {name:9s} {ticker:16s} excluded through {last_bad.date()}, "
                      f"next eligible rebalance {later[0].rebalance_date.date()}")


if __name__ == "__main__":
    main()
