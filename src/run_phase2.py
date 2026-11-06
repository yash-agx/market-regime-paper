"""Phase 2 entry point: build the regime labels and report on them.

    python -m src.run_phase2
"""

from __future__ import annotations

import pandas as pd

from . import config as cfg
from . import data as D
from . import regimes as R

LABELS_CSV = "regime_labels.csv"


def main() -> None:
    cfg.ensure_dirs()
    cleaned = D.load_processed()

    reg = R.build(cleaned.nsei_returns, cleaned.vix)
    labels = reg.labels
    labels.to_csv(cfg.PROC_DIR / LABELS_CSV, index_label="date")

    # Everything below is reported on the study period only (2009 is warm-up).
    study = labels.loc[cfg.STUDY_START:cfg.STUDY_END]
    insample = labels.loc[cfg.STUDY_START:cfg.INSAMPLE_END]
    holdout = labels.loc[cfg.HOLDOUT_START:cfg.STUDY_END]

    rows = []
    for period_name, frame in (("2010-2025 (all)", study),
                               ("2010-2019 (in-sample)", insample),
                               ("2020-2025 (held out)", holdout)):
        for method, col in (("A: realised vol", "high_vol_a"),
                            ("B: HMM filtered", "high_vol_b"),
                            ("VIX (robustness)", "high_vol_vix")):
            summary = R.regime_summary(frame[col], method)
            summary["period"] = period_name
            rows.append(summary)
    summary_df = pd.DataFrame(rows).set_index(["period", "method"])
    summary_df.to_csv(cfg.RESULTS_DIR / "phase2_regime_summary.csv")

    agree_rows = []
    for period_name, frame in (("2010-2025 (all)", study),
                               ("2010-2019 (in-sample)", insample),
                               ("2020-2025 (held out)", holdout)):
        for other, col in (("B: HMM filtered", "high_vol_b"), ("VIX (robustness)", "high_vol_vix")):
            rec = {"period": period_name, "compared_with_A": other}
            rec.update(R.agreement(frame["high_vol_a"], frame[col]))
            agree_rows.append(rec)
    agree_df = pd.DataFrame(agree_rows).set_index(["period", "compared_with_A"])
    agree_df.to_csv(cfg.RESULTS_DIR / "phase2_agreement.csv")

    runs_a = R.regime_runs(study["high_vol_a"])
    runs_a.to_csv(cfg.RESULTS_DIR / "phase2_episodes_method_a.csv", index=False)
    R.regime_runs(study["high_vol_b"]).to_csv(
        cfg.RESULTS_DIR / "phase2_episodes_method_b.csv", index=False)

    fits = pd.DataFrame([{
        "fit_date": f.fit_date.date(), "n_train_days": f.n_train, "converged": f.converged,
        "high_state": f.high_state, "stationary_p_high": round(f.stationary_p_high, 4),
        "mean_low_state_pct": round(100 * f.means[1 - f.high_state], 4),
        "mean_high_state_pct": round(100 * f.means[f.high_state], 4),
        "ann_vol_low_state_pct": round(100 * f.sds[1 - f.high_state] * (cfg.TRADING_DAYS ** 0.5), 2),
        "ann_vol_high_state_pct": round(100 * f.sds[f.high_state] * (cfg.TRADING_DAYS ** 0.5), 2),
        "p_stay_low": round(f.transmat[1 - f.high_state, 1 - f.high_state], 4),
        "p_stay_high": round(f.transmat[f.high_state, f.high_state], 4),
    } for f in reg.fits])
    fits.to_csv(cfg.RESULTS_DIR / "phase2_hmm_fits.csv", index=False)

    # ---- how many rebalance periods land in each regime? -------------------
    # This is what Phase 4 actually tests on. Each holding period is labelled by
    # the regime on its rebalance date, i.e. at the START of the period.
    from . import windows as W

    period_rows = []
    for strat, builder in (("momentum", W.momentum_windows), ("reversal", W.reversal_windows)):
        wins = [w for w in builder(cleaned.prices.index)
                if pd.Timestamp(cfg.STUDY_START) <= w.rebalance_date <= pd.Timestamp(cfg.STUDY_END)]
        dates = pd.DatetimeIndex([w.rebalance_date for w in wins])
        for period_name, lo, hi in (("2010-2019 (in-sample)", cfg.STUDY_START, cfg.INSAMPLE_END),
                                    ("2020-2025 (held out)", cfg.HOLDOUT_START, cfg.STUDY_END)):
            sel = dates[(dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi))]
            for method, col in (("A: realised vol", "high_vol_a"),
                                ("B: HMM filtered", "high_vol_b"),
                                ("VIX (robustness)", "high_vol_vix")):
                lab = labels[col].reindex(sel)
                period_rows.append({
                    "strategy": strat, "period": period_name, "method": method,
                    "n_periods": int(len(sel)),
                    "n_high_vol": int(lab.sum()),
                    "n_low_vol": int((~lab.astype("boolean")).sum()),
                    "n_unlabelled": int(lab.isna().sum()),
                    "pct_high_vol": round(100 * float(lab.mean()), 1),
                })
    periods_df = pd.DataFrame(period_rows)
    periods_df.to_csv(cfg.RESULTS_DIR / "phase2_periods_per_regime.csv", index=False)

    # ---- how much does the HMM prior matter? -------------------------------
    # The frozen spec leaves the transition-matrix prior open. 1.0 is no
    # regularisation (plain maximum likelihood) and is what runs by default;
    # the rest are here so the cost of that choice is visible rather than
    # buried. Nothing downstream uses these columns.
    prior_rows = []
    for prior in (1.0, 1.5, 2.0, 5.0):
        b_p, fits_p = R.method_b(cleaned.nsei_returns, transmat_prior=prior)
        flag = b_p["high_vol"]
        degenerate = [f.fit_date.date() for f in fits_p
                      if not (f.stationary_p_high > 0.005)]
        prior_rows.append({
            "transmat_prior": prior,
            "pct_high_vol_2010_2025": round(
                100 * float(flag.loc[cfg.STUDY_START:cfg.STUDY_END].mean()), 2),
            "pct_high_vol_2010_2019": round(
                100 * float(flag.loc[cfg.STUDY_START:cfg.INSAMPLE_END].mean()), 2),
            "pct_high_vol_2020_2025": round(
                100 * float(flag.loc[cfg.HOLDOUT_START:cfg.STUDY_END].mean()), 2),
            "pct_agreement_with_A": R.agreement(
                labels["high_vol_a"].loc[cfg.STUDY_START:cfg.STUDY_END],
                flag.loc[cfg.STUDY_START:cfg.STUDY_END])["pct_agreement"],
            "mean_episode_days": R.regime_summary(
                flag.loc[cfg.STUDY_START:cfg.STUDY_END], "b")["mean_high_vol_length_days"],
            "n_fits_with_absorbing_low_state": len(degenerate),
            "absorbing_fit_dates": " ".join(str(d) for d in degenerate),
        })
    prior_df = pd.DataFrame(prior_rows)
    prior_df.to_csv(cfg.RESULTS_DIR / "phase2_methodb_prior_sensitivity.csv", index=False)

    # ---- console report ----------------------------------------------------
    pd.set_option("display.width", 200, "display.max_columns", 50)

    print("=== Phase 2: regimes ===\n")
    print("Share of time high-volatility, and how long episodes last:\n")
    show = summary_df[["pct_days_high_vol", "n_high_vol_episodes",
                       "mean_high_vol_length_days", "mean_low_vol_length_days",
                       "longest_high_vol_episode_days", "n_days"]]
    print(show.to_string())

    print("\n\nAgreement with Method A (the main method):\n")
    print(agree_df[["n_common_days", "pct_agreement", "pct_both_high",
                    "pct_a_high_only", "pct_b_high_only"]].to_string())

    print("\n\nHMM refits (expanding window, one per December):\n")
    print(fits.to_string(index=False))

    print("\n\nRebalance periods per regime (what Phase 4 tests on):\n")
    print(periods_df.set_index(["strategy", "period", "method"]).to_string())

    print("\n\nMethod B sensitivity to the transition-matrix prior "
          "(1.0 = no regularisation = the default):\n")
    print(prior_df.to_string(index=False))

    unlabelled = labels.loc[cfg.STUDY_START:cfg.STUDY_END, "high_vol_b"].isna().sum()
    print(f"\n  Method B unlabelled days in the study period: {unlabelled}")

    print("\n\nTen longest high-volatility episodes, Method A, 2010-2025:\n")
    top = runs_a[runs_a["high_vol"]].nlargest(10, "length_days")
    for _, row in top.iterrows():
        print(f"  {row['start'].date()} to {row['end'].date()}  "
              f"{row['length_days']:4d} trading days")

    n_hi_a = int(study['high_vol_a'].sum())
    print(f"\n  Method A high-vol days in 2010-2025: {n_hi_a} of {len(study)}")
    print(f"  Independent high-vol episodes (Method A): "
          f"{int((runs_a['high_vol']).sum())}")
    print(f"\n  Labels -> {cfg.PROC_DIR / LABELS_CSV}")
    print(f"  Tables -> {cfg.RESULTS_DIR}")


if __name__ == "__main__":
    main()
