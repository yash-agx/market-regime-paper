"""Phase 1 entry point: build the cleaned dataset and write the data-quality reports.

    python -m src.run_phase1            # uses the cached download if present
    python -m src.run_phase1 --force    # re-downloads from Yahoo Finance
"""

from __future__ import annotations

import argparse

import pandas as pd

from . import config as cfg
from . import data as D


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1: data")
    ap.add_argument("--force", action="store_true", help="re-download instead of using cache")
    args = ap.parse_args()

    cfg.ensure_dirs()
    cleaned = D.build(force_download=args.force)
    D.save_processed(cleaned)

    # ---- reports -----------------------------------------------------------
    cleaned.listing.to_csv(cfg.RESULTS_DIR / "phase1_coverage_by_stock.csv")
    cleaned.flagged.to_csv(cfg.RESULTS_DIR / "phase1_flagged_returns.csv", index=False)

    eligible_count = cleaned.eligible.sum(axis=1).rename("n_eligible_stocks")
    eligible_count.to_csv(cfg.RESULTS_DIR / "phase1_eligible_count.csv", index_label="date")

    stale = D.staleness_report(cleaned.prices, cleaned.returns)
    stale.to_csv(cfg.RESULTS_DIR / "phase1_stale_prices.csv")

    study = cleaned.prices.loc[cfg.STUDY_START:cfg.STUDY_END]
    listing = cleaned.listing
    summary = {
        "n_tickers_in_universe": len(cleaned.prices.columns),
        "n_tickers_with_any_data": int((listing["n_obs"] > 0).sum()),
        "n_tickers_with_no_data": int((listing["n_obs"] == 0).sum()),
        "calendar_first_day": cleaned.prices.index.min().date(),
        "calendar_last_day": cleaned.prices.index.max().date(),
        "n_trading_days_total": len(cleaned.prices),
        "n_trading_days_study_period": len(study),
        "n_tickers_full_history_from_2009": int(
            (listing["first_date"] <= pd.Timestamp("2009-01-15")).sum()
        ),
        "median_pct_missing_in_span": round(float(listing["pct_missing_in_span"].median()), 3),
        "max_pct_missing_in_span": round(float(listing["pct_missing_in_span"].max()), 3),
        "eligible_stocks_on_study_start": int(
            cleaned.eligible.loc[:cfg.STUDY_START].iloc[-1].sum()
        ),
        "eligible_stocks_on_insample_end": int(
            cleaned.eligible.loc[:cfg.INSAMPLE_END].iloc[-1].sum()
        ),
        "eligible_stocks_on_last_day": int(cleaned.eligible.iloc[-1].sum()),
        "n_flagged_returns": len(cleaned.flagged),
        "n_stocks_with_flat_run_over_20_days": int((stale["longest_flat_run_days"] > 20).sum()),
        "worst_flat_run_days": int(stale["longest_flat_run_days"].max()),
        "worst_flat_run_ticker": stale["longest_flat_run_days"].idxmax(),
        "nsei_days": int(cleaned.nsei.notna().sum()),
        "india_vix_days": int(cleaned.vix.notna().sum()),
        "india_vix_first_day": cleaned.vix.first_valid_index().date(),
        "india_vix_pct_missing_on_calendar": round(
            100 * float(cleaned.vix.loc[cleaned.vix.first_valid_index():].isna().mean()), 3
        ),
    }
    pd.Series(summary).rename("value").to_csv(
        cfg.RESULTS_DIR / "phase1_summary.csv", index_label="metric"
    )

    # ---- risk-free rate (optional at this stage, needed from Phase 4) ------
    try:
        rf = D.load_riskfree(cleaned.prices.index)
        rf.rename("rf_daily").to_csv(cfg.PROC_DIR / "riskfree_daily.csv", index_label="date")
        rf_status = (
            f"loaded, {rf.notna().sum()} days, "
            f"annualised mean {100 * ((1 + rf.mean()) ** cfg.TRADING_DAYS - 1):.2f}%"
        )
    except (FileNotFoundError, ValueError) as err:
        rf_status = f"NOT AVAILABLE\n{err}"

    # ---- console summary ---------------------------------------------------
    print("\n=== Phase 1 summary ===")
    for k, v in summary.items():
        print(f"  {k:42s} {v}")
    print("\n  Stocks whose data starts after the study begins (index additions):")
    late = listing[listing["first_date"] > pd.Timestamp(cfg.STUDY_START)].sort_values("first_date")
    for tkr, row in late.iterrows():
        print(f"    {tkr:16s} first price {row['first_date'].date()}  "
              f"eligible from {row['first_eligible_date'].date() if pd.notna(row['first_eligible_date']) else 'never'}")
    print("\n  Worst missing-data offenders (% of days missing inside their own span):")
    for tkr, row in listing.nlargest(5, "pct_missing_in_span").iterrows():
        print(f"    {tkr:16s} {row['pct_missing_in_span']:.3f}%  ({row['n_missing_in_span']} days)")
    print(f"\n  Flagged returns beyond +/-50%: {len(cleaned.flagged)}")
    if not cleaned.flagged.empty:
        for _, row in cleaned.flagged.head(10).iterrows():
            print(f"    {row['date'].date()}  {row['ticker']:16s} {row['return_pct']:+.2f}%")
    print("\n  Longest runs of an unchanging price (diagnostic; nothing is filtered on it):")
    for tkr, row in stale.head(5).iterrows():
        span = ""
        if pd.notna(row["longest_flat_run_start"]):
            span = (f"  {row['longest_flat_run_start'].date()} to "
                    f"{row['longest_flat_run_end'].date()}")
        print(f"    {tkr:16s} {row['longest_flat_run_days']:5d} days"
              f"  zero-return days {row['pct_zero_return_days']:6.2f}%{span}")

    print(f"\n  Risk-free rate: {rf_status}")
    print(f"\n  Processed data -> {cfg.PROC_DIR}")
    print(f"  Reports        -> {cfg.RESULTS_DIR}")


if __name__ == "__main__":
    main()
