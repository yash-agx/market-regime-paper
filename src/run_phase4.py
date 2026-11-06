"""Phase 4 entry point: regime-conditional performance and the tests of H1-H3.

    python -m src.run_phase4                 # in-sample 2010-2019 only
    python -m src.run_phase4 --holdout       # 2020-2025, ONLY once in-sample is final

The held-out period is behind a flag on purpose. Looking at 2020-2025 while
still deciding anything about method would destroy the one genuinely
out-of-sample test this project has.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as cfg
from . import data as D
from . import evaluation as E
from . import regimes as R
from . import strategies as S
from . import windows as W

# Which strategies tie up capital (so the T-bill rate comes off) and which do not.
ZERO_INVESTMENT = {
    "momentum_ls": True, "reversal_ls": True,
    "momentum_lo": False, "reversal_lo": False,
    "benchmark_monthly": False, "benchmark_weekly": False,
}
REGIME_METHODS = {
    "A: realised vol (main)": "high_vol_a",
    "VIX (main robustness)": "high_vol_vix",
    "B: HMM (underpowered)": "high_vol_b",
}


def build_inputs(lo: str, hi: str):
    """Everything Phase 4 reads, restricted to one date range."""
    cleaned = D.load_processed()
    reg = R.build(cleaned.nsei_returns, cleaned.vix)
    labels = reg.labels
    rf_daily = D.load_riskfree(cleaned.prices.index)
    strats = S.build_all(cleaned)

    mom_wins = W.momentum_windows(cleaned.prices.index)
    rev_wins = W.reversal_windows(cleaned.prices.index)
    rf_month = E.period_riskfree(rf_daily, mom_wins)
    rf_week = E.period_riskfree(rf_daily, rev_wins)

    sliced, excess = {}, {}
    for key, res in strats.items():
        sub = S.slice_period(res, lo, hi)
        sliced[key] = sub
        rf = rf_month if sub.frequency == "monthly" else rf_week
        excess[key] = {}
        for cost in (None, 20):
            excess[key][cost] = E.excess_returns(
                sub.ret(cost), rf, ZERO_INVESTMENT[key])
    return cleaned, labels, sliced, excess


def leg_regime_table(sub: S.StrategyResult, flag: pd.Series) -> pd.DataFrame:
    """Long leg and short leg separately, per regime - not just the spread."""
    f = flag.reindex(sub.gross.index)
    rows = []
    for regime, mask in (("high-vol", f.fillna(False).astype(bool)),
                         ("low-vol", ~f.fillna(True).astype(bool)),
                         ("all", pd.Series(True, index=sub.gross.index))):
        ppy = sub.periods_per_year
        for leg, series in (("long leg", sub.r_long_leg),
                            ("short leg", sub.r_short_leg),
                            ("spread (long-short)", sub.gross)):
            r = series[mask].dropna()
            rows.append({
                "regime": regime, "leg": leg, "n_periods": int(len(r)),
                "ann_return_pct": round(100 * S.annualised_return(r, ppy), 2)
                if len(r) else np.nan,
                "ann_vol_pct": round(100 * S.annualised_vol(r, ppy), 2)
                if len(r) > 1 else np.nan,
                "sharpe": round(E.sharpe(r, ppy), 3) if len(r) > 1 else np.nan,
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4: evaluation")
    ap.add_argument("--holdout", action="store_true",
                    help="run on 2020-2025 instead of 2010-2019")
    args = ap.parse_args()

    if args.holdout:
        lo, hi, tag, title = (cfg.HOLDOUT_START, cfg.STUDY_END, "holdout",
                              "2020-2025 (HELD OUT)")
    else:
        lo, hi, tag, title = (cfg.STUDY_START, cfg.INSAMPLE_END, "insample",
                              "2010-2019 (in-sample)")

    cfg.ensure_dirs()
    cleaned, labels, sliced, excess = build_inputs(lo, hi)
    pd.set_option("display.width", 220, "display.max_columns", 60)

    print(f"{'=' * 100}\nPHASE 4 - {title}\n{'=' * 100}")
    print("\nLong-short books are zero-investment, so their returns are already excess")
    print("returns. The T-bill rate is subtracted only from long-only and the benchmark.")
    print("Each holding period is labelled by the regime on its rebalance (start) date.\n")

    # ---- 1. regime tables, every strategy x every regime method x gross/net ----
    rows = []
    for key, sub in sliced.items():
        for method_name, col in REGIME_METHODS.items():
            for cost in (None, 20):
                tab = E.regime_table(excess[key][cost], labels[col],
                                     sub.periods_per_year, sub.turnover)
                tab.insert(0, "cost", "gross" if cost is None else f"{cost}bps")
                tab.insert(0, "regime_method", method_name)
                tab.insert(0, "strategy", sub.name)
                rows.append(tab)
    regime_stats = pd.concat(rows, ignore_index=True)
    regime_stats.to_csv(cfg.RESULTS_DIR / f"phase4_{tag}_regime_stats.csv", index=False)

    main_mask = ((regime_stats["regime_method"] == "A: realised vol (main)")
                 & (regime_stats["cost"] == "20bps")
                 & (regime_stats["strategy"].str.contains("long-short")))
    print(f"\n{'-' * 100}\nMAIN TEST (pre-specified): Method A, 20bps, long-short"
          f"\n{'-' * 100}\n")
    print(regime_stats[main_mask][
        ["strategy", "regime", "n_periods", "sharpe", "ann_return_pct",
         "ann_vol_pct", "max_drawdown_pct", "hit_rate_pct", "mean_turnover"]
    ].to_string(index=False))

    gross_mask = ((regime_stats["regime_method"] == "A: realised vol (main)")
                  & (regime_stats["cost"] == "gross")
                  & (regime_stats["strategy"].str.contains("long-short")))
    print(f"\n\nSECONDARY: the same thing on GROSS returns - does the SIGNAL behave")
    print("differently across regimes, setting aside whether it is tradable?\n")
    print(regime_stats[gross_mask][
        ["strategy", "regime", "n_periods", "sharpe", "ann_return_pct",
         "ann_vol_pct", "hit_rate_pct"]
    ].to_string(index=False))

    print("\n\nLong-only strategies and the benchmark (Method A, 20bps, "
          "net of the T-bill rate):\n")
    lo_mask = ((regime_stats["regime_method"] == "A: realised vol (main)")
               & (regime_stats["cost"] == "20bps")
               & (~regime_stats["strategy"].str.contains("long-short")))
    print(regime_stats[lo_mask][
        ["strategy", "regime", "n_periods", "sharpe", "ann_return_pct", "ann_vol_pct"]
    ].to_string(index=False))

    # ---- 2. leg decomposition by regime ------------------------------------
    legs = []
    for key in ("momentum_ls", "reversal_ls"):
        t = leg_regime_table(sliced[key], labels["high_vol_a"])
        t.insert(0, "strategy", sliced[key].name)
        legs.append(t)
    legs_df = pd.concat(legs, ignore_index=True)
    legs_df.to_csv(cfg.RESULTS_DIR / f"phase4_{tag}_legs_by_regime.csv", index=False)
    print(f"\n\n{'-' * 100}\nLONG LEG vs SHORT LEG BY REGIME "
          f"(Method A, gross)\n{'-' * 100}\n")
    print(legs_df.to_string(index=False))

    # ---- 3. the hypothesis tests -------------------------------------------
    print(f"\n\n{'-' * 100}\nH1-H3: stationary block bootstrap, "
          f"{E.N_BOOTSTRAP:,} resamples, seed {cfg.RANDOM_SEED}\n{'-' * 100}")
    print("\nH1: momentum Sharpe(low-vol) - Sharpe(high-vol) > 0")
    print("H2: reversal Sharpe(high-vol) - Sharpe(low-vol) > 0")
    print("H3: [mom-rev gap in low-vol] - [mom-rev gap in high-vol]  (= H1 + H2)")
    print("p is one-sided: the share of resamples where the difference is <= 0.\n")

    boot_rows = []
    for method_name, col in REGIME_METHODS.items():
        for cost in (None, 20):
            cost_label = "gross" if cost is None else f"{cost}bps"
            h1 = E.bootstrap_sharpe_gap(
                excess["momentum_ls"][cost], labels[col], cfg.MONTHS_PER_YEAR,
                "H1", favour="low")
            h2 = E.bootstrap_sharpe_gap(
                excess["reversal_ls"][cost], labels[col], cfg.WEEKS_PER_YEAR,
                "H2", favour="high")
            h3 = E.bootstrap_difference_in_differences(
                excess["momentum_ls"][cost], labels[col],
                excess["reversal_ls"][cost], labels[col])
            for res in (h1, h2, h3):
                rec = res.as_row()
                rec["regime_method"] = method_name
                rec["cost"] = cost_label
                boot_rows.append(rec)
    boot = pd.DataFrame(boot_rows)
    cols = ["regime_method", "cost", "hypothesis", "observed", "ci95_low", "ci95_high",
            "p_one_sided", "n_valid_resamples", "mean_block", "note"]
    boot = boot[cols]
    boot.to_csv(cfg.RESULTS_DIR / f"phase4_{tag}_bootstrap.csv", index=False)

    print("Block lengths are chosen by the Politis-White (2004) automatic rule,")
    print("applied to the returns AND the regime dummy, taking the longer. The")
    print("dummy is what drives it: returns are near white noise, regimes are not.\n")
    print("MAIN (Method A):\n")
    print(boot[boot["regime_method"] == "A: realised vol (main)"]
          [["cost", "hypothesis", "observed", "ci95_low", "ci95_high",
            "p_one_sided", "mean_block", "n_valid_resamples"]].to_string(index=False))

    # ---- multiple testing --------------------------------------------------
    bonf = 0.05 / 3
    print(f"\n\nMultiple testing: three hypotheses are tested, so the Bonferroni")
    print(f"threshold is 0.05/3 = {bonf:.4f}. Against that:\n")
    main_boot = boot[(boot["regime_method"] == "A: realised vol (main)")
                     & (boot["cost"] == "20bps")]
    mt_rows = []
    for _, row in main_boot.iterrows():
        survives = (row["p_one_sided"] < bonf) if pd.notna(row["p_one_sided"]) else False
        mt_rows.append({
            "hypothesis": row["hypothesis"],
            "p_one_sided": row["p_one_sided"],
            "bonferroni_threshold": round(bonf, 4),
            "survives_bonferroni": survives,
        })
        verdict = "SURVIVES" if survives else "does not survive"
        print(f"  {row['hypothesis']}: p = {row['p_one_sided']:.4f}  ->  {verdict}")
    pd.DataFrame(mt_rows).to_csv(
        cfg.RESULTS_DIR / f"phase4_{tag}_multiple_testing.csv", index=False)
    print("\n  Note that H3 = H1 + H2 arithmetically, so the three tests are not")
    print("  independent and Bonferroni is conservative here. H1 is the result")
    print("  that stands on its own.")
    print("\n\nRobustness across regime definitions:\n")
    print(boot[["regime_method", "cost", "hypothesis", "observed",
                "p_one_sided", "n_valid_resamples"]].to_string(index=False))

    # ---- 3b. does the answer depend on the block length? -------------------
    # The mean block is a judgment call (the n^(1/3) rule of thumb), so the
    # main test is repeated across a range of it. If H1 only survives at one
    # block length, it is an artefact of that choice.
    sens_rows = []
    mom_r = excess["momentum_ls"][20].dropna()
    rev_r = excess["reversal_ls"][20].dropna()
    pw_mom = E.politis_white_block(
        mom_r, labels["high_vol_a"].reindex(mom_r.index).astype("float64"))
    pw_rev = E.politis_white_block(
        rev_r, labels["high_vol_a"].reindex(rev_r.index).astype("float64"))
    settings = [
        (pw_mom * 0.5, pw_rev * 0.5, "half of Politis-White"),
        (pw_mom, pw_rev, "Politis-White (used in main tables)"),
        (pw_mom * 2, pw_rev * 2, "double Politis-White"),
        (pw_mom * 4, pw_rev * 4, "quadruple Politis-White"),
        (E.default_block(len(mom_r)), E.default_block(len(rev_r)),
         "old n^(1/3) rule of thumb"),
    ]
    for bm_raw, br_raw, tag_m in settings:
        # Keep the fractional block, so the Politis-White row reproduces the
        # main table exactly rather than a rounded-off version of it.
        bm, br = max(2.0, round(bm_raw, 2)), max(2.0, round(br_raw, 2))
        h1 = E.bootstrap_sharpe_gap(
            excess["momentum_ls"][20], labels["high_vol_a"], cfg.MONTHS_PER_YEAR,
            "H1", favour="low", mean_block=bm)
        h2 = E.bootstrap_sharpe_gap(
            excess["reversal_ls"][20], labels["high_vol_a"], cfg.WEEKS_PER_YEAR,
            "H2", favour="high", mean_block=br)
        sens_rows.append({
            "block_setting": tag_m, "mom_block_months": bm, "rev_block_weeks": br,
            "H1_observed": round(h1.observed, 3), "H1_p": round(h1.p_one_sided, 4),
            "H2_observed": round(h2.observed, 3), "H2_p": round(h2.p_one_sided, 4),
        })
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(cfg.RESULTS_DIR / f"phase4_{tag}_block_sensitivity.csv", index=False)
    print("\n\nBlock-length sensitivity (Method A, 20bps, long-short):\n")
    print(sens.to_string(index=False))

    # ---- 4. Newey-West check ------------------------------------------------
    nw_rows = []
    for method_name, col in REGIME_METHODS.items():
        for key in ("momentum_ls", "reversal_ls", "momentum_lo", "reversal_lo"):
            for cost in (None, 20):
                rec = E.newey_west_dummy(
                    excess[key][cost], labels[col],
                    f"{sliced[key].name}")
                rec["regime_method"] = method_name
                rec["cost"] = "gross" if cost is None else f"{cost}bps"
                nw_rows.append(rec)
    nw = pd.DataFrame(nw_rows)
    nw.to_csv(cfg.RESULTS_DIR / f"phase4_{tag}_newey_west.csv", index=False)
    print(f"\n\n{'-' * 100}\nNEWEY-WEST CHECK: return regressed on a high-vol dummy"
          f"\n{'-' * 100}")
    print("The slope is the difference in MEAN period return (high minus low),\n"
          "not a Sharpe difference. Standard errors are robust to the\n"
          "autocorrelation that persistent regimes create.\n")
    show = nw[nw["regime_method"] == "A: realised vol (main)"]
    print(show[["strategy", "cost", "n", "nw_lags", "mean_low_vol_pct",
                "diff_high_minus_low_pct", "t_stat", "p_value_two_sided"]]
          .to_string(index=False))

    print(f"\n\n  Tables -> {cfg.RESULTS_DIR}/phase4_{tag}_*.csv")
    if not args.holdout:
        print("\n  Held-out 2020-2025 NOT touched. Run with --holdout only once the")
        print("  in-sample analysis is final.")


if __name__ == "__main__":
    main()
