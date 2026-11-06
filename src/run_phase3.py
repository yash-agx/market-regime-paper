"""Phase 3 entry point: build the strategies and report returns, vol and turnover.

    python -m src.run_phase3
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from . import data as D
from . import strategies as S

RETURNS_CSV = "strategy_returns.csv"


def main() -> None:
    cfg.ensure_dirs()
    cleaned = D.load_processed()
    strats = S.build_all(cleaned)

    # ---- save the period return series ------------------------------------
    frames = []
    for key, res in strats.items():
        df = pd.DataFrame({"gross": res.gross, "turnover": res.turnover,
                           "n_long": res.n_long, "n_short": res.n_short,
                           "n_eligible": res.n_eligible,
                           "r_long_leg": res.r_long_leg,
                           "r_short_leg": res.r_short_leg})
        for bps, series in res.net.items():
            df[f"net_{bps}bps"] = series
        df["strategy"] = key
        df["frequency"] = res.frequency
        frames.append(df.reset_index())
    pd.concat(frames, ignore_index=True).to_csv(
        cfg.PROC_DIR / RETURNS_CSV, index=False)

    for key, res in strats.items():
        if not res.holdings.empty:
            res.holdings.to_csv(cfg.RESULTS_DIR / f"phase3_holdings_{key}.csv", index=False)

    # ---- summary table, before and after costs ----------------------------
    periods = [
        ("2010-2025 (all)", cfg.STUDY_START, cfg.STUDY_END),
        ("2010-2019 (in-sample)", cfg.STUDY_START, cfg.INSAMPLE_END),
        ("2020-2025 (held out)", cfg.HOLDOUT_START, cfg.STUDY_END),
    ]
    rows = []
    for key, res in strats.items():
        for label, lo, hi in periods:
            sub = S.slice_period(res, lo, hi)
            for bps in (None, 10, 20, 30):
                rec = S.summarise(sub, bps, label)
                rec["key"] = key
                rows.append(rec)
    summary = pd.DataFrame(rows)
    summary.to_csv(cfg.RESULTS_DIR / "phase3_performance.csv", index=False)

    # ---- console report ----------------------------------------------------
    pd.set_option("display.width", 220, "display.max_columns", 60)

    print("=== Phase 3: strategies ===\n")
    print("Conventions: the long-short book is $1 long and $1 short, so its return is")
    print("(long leg - short leg) and is already an excess return. Turnover is value")
    print("traded per rupee of capital, so cost = one-way bps x turnover.\n")

    for label, lo, hi in periods:
        print(f"\n{'=' * 110}\n{label}\n{'=' * 110}")
        block = summary[(summary["period"] == label)]
        show = block.pivot_table(
            index=["strategy"], columns="cost_bps",
            values=["ann_return_pct", "ann_vol_pct"], sort=False)
        print("\nAnnualised return (%) and volatility (%), by cost assumption:\n")
        print(show.to_string())

        print("\nTurnover and portfolio size:\n")
        t = block[block["cost_bps"] == "gross"][
            ["strategy", "n_periods", "mean_turnover", "ann_turnover",
             "mean_n_long", "mean_n_short", "mean_n_eligible"]]
        print(t.to_string(index=False))

    print(f"\n\nCost drag at 20bps (annualised return gross minus net), {periods[0][0]}:\n")
    allp = summary[summary["period"] == periods[0][0]]
    for key in strats:
        g = allp[(allp["key"] == key) & (allp["cost_bps"] == "gross")]["ann_return_pct"].iloc[0]
        n = allp[(allp["key"] == key) & (allp["cost_bps"] == 20)]["ann_return_pct"].iloc[0]
        print(f"  {strats[key].name:38s} {g:7.2f}%  ->  {n:7.2f}%   (drag {g - n:5.2f} pp)")

    # ---- break-even cost ---------------------------------------------------
    # The one-way cost at which the average gross edge is exactly eaten:
    # mean gross return / mean turnover. A strategy whose break-even is below
    # the 20bp base case cannot survive its own trading.
    be_rows = []
    for key, res in strats.items():
        rec = {"strategy": res.name}
        for label, lo, hi in periods:
            sub = S.slice_period(res, lo, hi)
            t = float(sub.turnover.mean())
            rec[label] = round(10_000 * float(sub.gross.mean()) / t, 1) if t > 0 else np.nan
        be_rows.append(rec)
    breakeven = pd.DataFrame(be_rows).set_index("strategy")
    breakeven.to_csv(cfg.RESULTS_DIR / "phase3_breakeven_costs.csv")
    print("\n\nBreak-even one-way cost (bps) - the cost at which the gross edge is gone:\n")
    print(breakeven.to_string())
    print("\n  (The base case is 20bp. Anything below that line loses money by trading.)")

    # ---- where does the long-short return come from? ----------------------
    print("\n\nLeg decomposition (annualised, gross, 2010-2025):\n")
    leg_rows = []
    for key in ("momentum_ls", "reversal_ls"):
        res = S.slice_period(strats[key], cfg.STUDY_START, cfg.STUDY_END)
        ppy = res.periods_per_year
        rec = {
            "strategy": res.name,
            "long_leg_pct": round(100 * S.annualised_return(res.r_long_leg, ppy), 2),
            "short_leg_pct": round(100 * S.annualised_return(res.r_short_leg, ppy), 2),
            "long_short_gross_pct": round(100 * S.annualised_return(res.gross, ppy), 2),
        }
        leg_rows.append(rec)
        print(f"  {rec['strategy']:24s} long leg {rec['long_leg_pct']:7.2f}%   "
              f"short leg {rec['short_leg_pct']:7.2f}%   "
              f"spread {rec['long_short_gross_pct']:7.2f}%")
    pd.DataFrame(leg_rows).to_csv(cfg.RESULTS_DIR / "phase3_leg_decomposition.csv",
                                  index=False)

    # ---- survivorship reality check ---------------------------------------
    nsei = cleaned.nsei.loc[cfg.STUDY_START:cfg.STUDY_END]
    years = (nsei.index[-1] - nsei.index[0]).days / 365.25
    nsei_cagr = 100 * ((nsei.iloc[-1] / nsei.iloc[0]) ** (1 / years) - 1)
    bench = summary[(summary["key"] == "benchmark_monthly")
                    & (summary["period"] == "2010-2025 (all)")
                    & (summary["cost_bps"] == "gross")]["ann_return_pct"].iloc[0]
    print(f"\n\nSurvivorship reality check:")
    print(f"  Real ^NSEI price index, 2010-2025:                  {nsei_cagr:6.2f}% a year")
    print(f"  Equal-weighted benchmark of TODAY'S 50 members:     {bench:6.2f}% a year")
    print(f"  Gap:                                                {bench - nsei_cagr:6.2f} pp")
    print("  Part of this is dividends (our prices are total-return, ^NSEI is not)")
    print("  and part is equal- vs cap-weighting, but most of it is survivorship")
    print("  bias. Phase 5 is what fixes it.")

    print(f"\n  Returns -> {cfg.PROC_DIR / RETURNS_CSV}")
    print(f"  Tables  -> {cfg.RESULTS_DIR}")


if __name__ == "__main__":
    main()
