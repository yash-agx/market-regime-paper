"""Phase 6: does the main result depend on the 12-1 and 1-week lookbacks?

    python -m src.run_phase6_robustness

Pre-registered in docs/phase6_preregistration.md before anything here was run.
These are robustness checks, not new headline results: no variant is promoted,
and a variant p-value is read as "agrees with the main test or not", never as a
new finding.

What changes, in plain words
----------------------------
Only the formation window - how far back the signal looks. Everything else is
the Phase 3 and Phase 4 code, called as it is:

  * momentum J-1: the signal runs from month-end t-J to month-end t-1, so the
    most recent month is still skipped. J = 12 is the main spec, and the window
    builder below reproduces windows.momentum_windows() exactly for J = 12.
  * reversal K weeks: the signal runs from the week end K weeks back to this
    week end, still traded weekly and held one week. K = 1 is the main spec;
    "1 month" is K = 4 (about 20 trading days), which keeps the signal on the
    same weekly grid as the trade.

The cleaning rules (5% missing, 10-day flat runs) are applied over each
variant's own formation window, because that is the window its signal reads.

Nothing from Phases 1-5 is overwritten: every output here is a new
results/phase6_* file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from . import data as D
from . import evaluation as E
from . import regimes as R
from . import strategies as S
from . import windows as W
from .run_phase4 import universe_mask

# --------------------------------------------------------------------------
# the variants (frozen in the pre-registration)
# --------------------------------------------------------------------------

MOM_LOOKBACKS = {"3-1": 3, "6-1": 6, "12-1": 12}       # months, skip month kept
REV_LOOKBACKS = {"1w": 1, "2w": 2, "1m": 4}             # weeks
MAIN_PAIR = ("12-1", "1w")

# One change at a time: the primary robustness set.
PRIMARY_PAIRS = [("3-1", "1w"), ("6-1", "1w"), ("12-1", "1w"),
                 ("12-1", "2w"), ("12-1", "1m")]
ALL_PAIRS = [(m, r) for m in MOM_LOOKBACKS for r in REV_LOOKBACKS]

COSTS = (None, 10, 20, 30)
PERIODS = {
    "insample": ("2010-2019 (in-sample)", cfg.STUDY_START, cfg.INSAMPLE_END),
    "holdout": ("2020-2025 (held out)", cfg.HOLDOUT_START, cfg.STUDY_END),
}
REGIME_METHODS = {
    "A: realised vol (main)": "high_vol_a",
    "VIX (main robustness)": "high_vol_vix",
    "B: HMM (underpowered)": "high_vol_b",
}
MAIN_METHOD = "A: realised vol (main)"
BONFERRONI = 0.05 / 3

# The verdict bands from the pre-registration.
HOLDOUT_FAIL_P = 0.05
CRASH_RATIO_BAND = (1.5, 3.0)


def cost_label(cost: int | None) -> str:
    return "gross" if cost is None else f"{cost}bps"


def pair_label(pair: tuple[str, str]) -> str:
    return f"mom {pair[0]} / rev {pair[1]}"


# --------------------------------------------------------------------------
# formation windows for each lookback
# --------------------------------------------------------------------------


def momentum_windows_lookback(calendar: pd.DatetimeIndex, months: int) -> list[W.Window]:
    """J-1 momentum: signal from month-end t-J to month-end t-1, hold one month.

    For months = 12 this is identical to windows.momentum_windows().
    """
    if months < 2:
        raise ValueError("J-1 momentum needs J >= 2 so that something is left after the skip")
    ends = W.month_end_dates(calendar)
    return [
        W.Window(rebalance_date=ends[k], formation_start=ends[k - months],
                 formation_end=ends[k - 1], holding_end=ends[k + 1])
        for k in range(months, len(ends) - 1)
    ]


def reversal_windows_lookback(calendar: pd.DatetimeIndex, weeks: int) -> list[W.Window]:
    """K-week reversal: signal from K week-ends back to this week end, hold one week.

    For weeks = 1 this is identical to windows.reversal_windows().
    """
    if weeks < 1:
        raise ValueError("the reversal lookback must be at least one week")
    ends = W.week_end_dates(calendar)
    return [
        W.Window(rebalance_date=ends[j], formation_start=ends[j - weeks],
                 formation_end=ends[j], holding_end=ends[j + 1])
        for j in range(weeks, len(ends) - 1)
    ]


def in_study(wins: list[W.Window]) -> list[W.Window]:
    """The same 2010-2025 rebalance-date filter as strategies.build_all()."""
    lo, hi = pd.Timestamp(cfg.STUDY_START), pd.Timestamp(cfg.STUDY_END)
    return [w for w in wins if lo <= w.rebalance_date <= hi]


def build_variants(cleaned: D.CleanedData, mask: pd.DataFrame,
                   runs: pd.DataFrame) -> dict[tuple[str, str], S.StrategyResult]:
    """Long-short momentum and reversal for every lookback, on one universe."""
    cal = cleaned.prices.index
    out = {}
    for lab, months in MOM_LOOKBACKS.items():
        out[("mom", lab)] = S.run_strategy(
            cleaned, in_study(momentum_windows_lookback(cal, months)), "top", "bottom",
            f"Momentum {lab} (long-short)", "monthly", runs, universe_mask=mask)
    for lab, weeks in REV_LOOKBACKS.items():
        out[("rev", lab)] = S.run_strategy(
            cleaned, in_study(reversal_windows_lookback(cal, weeks)), "bottom", "top",
            f"Reversal {lab} (long-short)", "weekly", runs, universe_mask=mask)
    return out


# --------------------------------------------------------------------------
# small helpers that are tested on their own
# --------------------------------------------------------------------------


def regime_sharpes(r: pd.Series, flag: pd.Series, ppy: int) -> dict:
    """Sharpe in each regime, on exactly the observations the bootstrap uses."""
    df = pd.DataFrame({"r": r, "high": flag.reindex(r.index)}).dropna()
    hi = df["high"].astype(bool)
    return {
        "sharpe_low": E.sharpe(df["r"][~hi], ppy),
        "sharpe_high": E.sharpe(df["r"][hi], ppy),
        "n_low": int((~hi).sum()),
        "n_high": int(hi.sum()),
    }


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (NaNs are left out and stay NaN)."""
    p = np.asarray(pvalues, dtype="float64")
    out = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    m = len(ok)
    if m == 0:
        return out.tolist()
    order = ok[np.argsort(p[ok], kind="stable")]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out.tolist()


def crash_ratio(mean_high_v1: float, mean_high_v2: float) -> float:
    """How many times bigger the mean high-vol momentum loss is under Version 1.

    Only meaningful when both are losses; otherwise NaN, because "twice as bad"
    has no meaning when one of them made money.
    """
    if not (np.isfinite(mean_high_v1) and np.isfinite(mean_high_v2)):
        return np.nan
    if mean_high_v1 >= 0 or mean_high_v2 >= 0:
        return np.nan
    return float(mean_high_v1 / mean_high_v2)


def insample_verdict(h3: float, p: float) -> str:
    if not np.isfinite(h3):
        return "no estimate"
    if h3 <= 0:
        return "contradicts (H3 <= 0)"
    return "agrees" if p < BONFERRONI else "same sign, weaker"


def holdout_verdict(p: float) -> str:
    if not np.isfinite(p):
        return "no estimate"
    return "agrees (fails to replicate)" if p >= HOLDOUT_FAIL_P else "differs (replicates)"


def crash_verdict(ratio: float) -> str:
    if not np.isfinite(ratio):
        return "differs (not both losses)"
    lo, hi = CRASH_RATIO_BAND
    return "agrees" if lo <= ratio <= hi else f"differs (ratio {ratio:.2f})"


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------


def regime_stats_table(variants, labels, universe: str) -> pd.DataFrame:
    rows = []
    for (kind, lab), res in variants.items():
        for pkey, (plabel, lo, hi) in PERIODS.items():
            sub = S.slice_period(res, lo, hi)
            for method, col in REGIME_METHODS.items():
                for cost in COSTS:
                    tab = E.regime_table(sub.ret(cost), labels[col],
                                         sub.periods_per_year, sub.turnover)
                    tab.insert(0, "cost", cost_label(cost))
                    tab.insert(0, "regime_method", method)
                    tab.insert(0, "lookback", lab)
                    tab.insert(0, "strategy", "momentum" if kind == "mom" else "reversal")
                    tab.insert(0, "period", pkey)
                    tab.insert(0, "universe", universe)
                    rows.append(tab)
    return pd.concat(rows, ignore_index=True)


def legs_table(variants, labels, universe: str) -> pd.DataFrame:
    """High-vol and low-vol leg returns (gross), for the survivorship comparison."""
    rows = []
    flag = labels["high_vol_a"]
    for (kind, lab), res in variants.items():
        if kind != "mom":
            continue
        for pkey, (_, lo, hi) in PERIODS.items():
            sub = S.slice_period(res, lo, hi)
            f = flag.reindex(sub.gross.index)
            for regime, sel in (("high-vol", f.fillna(False).astype(bool)),
                                ("low-vol", ~f.fillna(True).astype(bool))):
                rows.append({
                    "universe": universe, "period": pkey, "lookback": lab,
                    "regime": regime, "n_periods": int(sel.sum()),
                    "long_leg_ann_pct": round(100 * S.annualised_return(
                        sub.r_long_leg[sel], 12), 2),
                    "short_leg_ann_pct": round(100 * S.annualised_return(
                        sub.r_short_leg[sel], 12), 2),
                })
    return pd.DataFrame(rows)


def bootstraps(variants, labels) -> pd.DataFrame:
    """H1 per momentum lookback, H2 per reversal lookback, H3 per pair."""
    rows = []
    for pkey, (_, lo, hi) in PERIODS.items():
        sl = {k: S.slice_period(v, lo, hi) for k, v in variants.items()}
        for method, col in REGIME_METHODS.items():
            flag = labels[col]
            for cost in COSTS:
                base = {"period": pkey, "regime_method": method, "cost": cost_label(cost)}
                for lab in MOM_LOOKBACKS:
                    res = E.bootstrap_sharpe_gap(sl[("mom", lab)].ret(cost), flag,
                                                 cfg.MONTHS_PER_YEAR, "H1", favour="low")
                    rows.append({**base, "mom": lab, "rev": "", **res.as_row()})
                for lab in REV_LOOKBACKS:
                    res = E.bootstrap_sharpe_gap(sl[("rev", lab)].ret(cost), flag,
                                                 cfg.WEEKS_PER_YEAR, "H2", favour="high")
                    rows.append({**base, "mom": "", "rev": lab, **res.as_row()})
                for m, r in ALL_PAIRS:
                    res = E.bootstrap_difference_in_differences(
                        sl[("mom", m)].ret(cost), flag, sl[("rev", r)].ret(cost), flag)
                    rows.append({**base, "mom": m, "rev": r, **res.as_row()})
                print(f"  bootstraps done: {pkey:8s} {method:24s} {cost_label(cost)}",
                      flush=True)
    return pd.DataFrame(rows)


def gap_table(variants, labels, boot: pd.DataFrame) -> pd.DataFrame:
    """Per pair: regime Sharpes, the mom - rev gap in each regime, H3 and its p."""
    def p_of(pkey, method, cost, hyp, mom="", rev=""):
        b = boot[(boot["period"] == pkey) & (boot["regime_method"] == method)
                 & (boot["cost"] == cost) & (boot["hypothesis"] == hyp)
                 & (boot["mom"] == mom) & (boot["rev"] == rev)]
        return b.iloc[0]

    rows = []
    for pkey, (_, lo, hi) in PERIODS.items():
        sl = {k: S.slice_period(v, lo, hi) for k, v in variants.items()}
        for method, col in REGIME_METHODS.items():
            for cost in COSTS:
                cl = cost_label(cost)
                for m, r in ALL_PAIRS:
                    ms = regime_sharpes(sl[("mom", m)].ret(cost), labels[col], 12)
                    rs = regime_sharpes(sl[("rev", r)].ret(cost), labels[col], 52)
                    h1 = p_of(pkey, method, cl, "H1", mom=m)
                    h2 = p_of(pkey, method, cl, "H2", rev=r)
                    h3 = p_of(pkey, method, cl, "H3", mom=m, rev=r)
                    gap_low = ms["sharpe_low"] - rs["sharpe_low"]
                    gap_high = ms["sharpe_high"] - rs["sharpe_high"]
                    rows.append({
                        "period": pkey, "regime_method": method, "cost": cl,
                        "mom": m, "rev": r, "primary": (m, r) in PRIMARY_PAIRS,
                        "main_spec": (m, r) == MAIN_PAIR,
                        "mom_sharpe_low": round(ms["sharpe_low"], 3),
                        "mom_sharpe_high": round(ms["sharpe_high"], 3),
                        "rev_sharpe_low": round(rs["sharpe_low"], 3),
                        "rev_sharpe_high": round(rs["sharpe_high"], 3),
                        "mom_n_low": ms["n_low"], "mom_n_high": ms["n_high"],
                        "rev_n_low": rs["n_low"], "rev_n_high": rs["n_high"],
                        "gap_low": round(gap_low, 3), "gap_high": round(gap_high, 3),
                        "H3": h3["observed"], "H3_ci95_low": h3["ci95_low"],
                        "H3_ci95_high": h3["ci95_high"], "H3_p": h3["p_one_sided"],
                        "H1": h1["observed"], "H1_p": h1["p_one_sided"],
                        "H2": h2["observed"], "H2_p": h2["p_one_sided"],
                        "H3_survives_bonferroni": bool(h3["p_one_sided"] < BONFERRONI)
                        if np.isfinite(h3["p_one_sided"]) else False,
                    })
    df = pd.DataFrame(rows)

    # Holm across the five primary pairs, within each period x method x cost.
    df["H3_p_holm_primary"] = np.nan
    for _, g in df[df["primary"]].groupby(["period", "regime_method", "cost"]):
        df.loc[g.index, "H3_p_holm_primary"] = np.round(holm(g["H3_p"].tolist()), 4)
    return df


def newey_west_table(variants, labels) -> pd.DataFrame:
    rows = []
    for pkey, (_, lo, hi) in PERIODS.items():
        for (kind, lab), res in variants.items():
            sub = S.slice_period(res, lo, hi)
            for method, col in REGIME_METHODS.items():
                for cost in COSTS:
                    rec = E.newey_west_dummy(sub.ret(cost), labels[col], sub.name)
                    rows.append({"period": pkey, "lookback": lab,
                                 "regime_method": method, "cost": cost_label(cost), **rec})
    return pd.DataFrame(rows)


def survivorship_table(stats: pd.DataFrame, legs: pd.DataFrame) -> pd.DataFrame:
    """Momentum in high-vol months, Version 1 against Version 2, per lookback."""
    rows = []
    sel = stats[(stats["strategy"] == "momentum") & (stats["regime_method"] == MAIN_METHOD)
                & (stats["cost"] == "20bps") & (stats["regime"] == "high-vol")]
    for pkey in PERIODS:
        for lab in MOM_LOOKBACKS:
            rec = {"period": pkey, "lookback": lab}
            for uni in ("v1", "v2"):
                row = sel[(sel["universe"] == uni) & (sel["period"] == pkey)
                          & (sel["lookback"] == lab)].iloc[0]
                leg = legs[(legs["universe"] == uni) & (legs["period"] == pkey)
                           & (legs["lookback"] == lab) & (legs["regime"] == "high-vol")].iloc[0]
                rec[f"{uni}_n_high"] = int(row["n_periods"])
                rec[f"{uni}_mean_high_pct"] = row["mean_period_return_pct"]
                rec[f"{uni}_sharpe_high"] = row["sharpe"]
                rec[f"{uni}_mdd_high_pct"] = row["max_drawdown_pct"]
                rec[f"{uni}_short_leg_high_ann_pct"] = leg["short_leg_ann_pct"]
            rec["crash_ratio_v1_over_v2"] = round(
                crash_ratio(rec["v1_mean_high_pct"], rec["v2_mean_high_pct"]), 3)
            rows.append(rec)
    return pd.DataFrame(rows)


def worst_high_vol_month(variants_by_uni, labels) -> pd.DataFrame:
    rows = []
    flag = labels["high_vol_a"]
    for uni, variants in variants_by_uni.items():
        for lab in MOM_LOOKBACKS:
            res = variants[("mom", lab)]
            for pkey, (_, lo, hi) in PERIODS.items():
                r = S.slice_period(res, lo, hi).ret(20)
                hv = r[flag.reindex(r.index).fillna(False).astype(bool)]
                rows.append({"universe": uni, "period": pkey, "lookback": lab,
                             "worst_high_vol_month_pct": round(100 * float(hv.min()), 2),
                             "worst_month_date": hv.idxmin().date() if len(hv) else None})
    return pd.DataFrame(rows)


def verdicts(gaps: pd.DataFrame, surv: pd.DataFrame) -> pd.DataFrame:
    main = gaps[(gaps["regime_method"] == MAIN_METHOD) & (gaps["cost"] == "20bps")
                & gaps["primary"]]
    rows = []
    for m, r in PRIMARY_PAIRS:
        ins = main[(main["period"] == "insample") & (main["mom"] == m) & (main["rev"] == r)].iloc[0]
        hol = main[(main["period"] == "holdout") & (main["mom"] == m) & (main["rev"] == r)].iloc[0]
        if m != MAIN_PAIR[0] or (m, r) == MAIN_PAIR:
            s = surv[(surv["period"] == "insample") & (surv["lookback"] == m)].iloc[0]
            ratio = s["crash_ratio_v1_over_v2"]
            cv = crash_verdict(ratio)
        else:
            ratio, cv = np.nan, "n/a (momentum is the main 12-1)"
        rows.append({
            "pair": pair_label((m, r)), "main_spec": (m, r) == MAIN_PAIR,
            "insample_H3": ins["H3"], "insample_H3_p": ins["H3_p"],
            "insample_H3_p_holm": ins["H3_p_holm_primary"],
            "insample_verdict": insample_verdict(ins["H3"], ins["H3_p"]),
            "holdout_H3": hol["H3"], "holdout_H3_p": hol["H3_p"],
            "holdout_verdict": holdout_verdict(hol["H3_p"]),
            "crash_ratio_insample": ratio, "crash_verdict": cv,
        })
    return pd.DataFrame(rows)


def reproduction_check(gaps: pd.DataFrame) -> pd.DataFrame:
    """The main pair here must reproduce the Phase 4 Version 2 bootstrap exactly."""
    rows = []
    for pkey, tag in (("insample", "insample_v2"), ("holdout", "holdout_v2")):
        p4 = pd.read_csv(cfg.RESULTS_DIR / f"phase4_{tag}_bootstrap.csv")
        for method in REGIME_METHODS:
            for cost in ("gross", "20bps"):
                ours = gaps[(gaps["period"] == pkey) & (gaps["regime_method"] == method)
                            & (gaps["cost"] == cost) & gaps["main_spec"]].iloc[0]
                for hyp in ("H1", "H2", "H3"):
                    ref = p4[(p4["regime_method"] == method) & (p4["cost"] == cost)
                             & (p4["hypothesis"] == hyp)].iloc[0]
                    mine_obs, mine_p = ours[hyp], ours[f"{hyp}_p"]
                    same = (np.isclose(mine_obs, ref["observed"], equal_nan=True)
                            and np.isclose(mine_p, ref["p_one_sided"], equal_nan=True))
                    rows.append({"period": pkey, "regime_method": method, "cost": cost,
                                 "hypothesis": hyp, "phase4_observed": ref["observed"],
                                 "phase6_observed": mine_obs,
                                 "phase4_p": ref["p_one_sided"], "phase6_p": mine_p,
                                 "identical": bool(same)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# markdown tables for the paper
# --------------------------------------------------------------------------


def _f(x, nd=2):
    return "–" if not np.isfinite(x) else f"{x:+.{nd}f}"


def _p(x):
    return "–" if not np.isfinite(x) else f"{x:.3f}"


def markdown_tables(gaps: pd.DataFrame, surv: pd.DataFrame, cost: str = "20bps",
                    method: str = MAIN_METHOD) -> str:
    out = []
    sub = gaps[(gaps["regime_method"] == method) & (gaps["cost"] == cost) & gaps["primary"]]
    for m, r in PRIMARY_PAIRS:
        out.append(f"\n**{pair_label((m, r))}**"
                   + (" (main specification)" if (m, r) == MAIN_PAIR else "") + "\n")
        out.append("| Period | Mom SR low | Mom SR high | Rev SR low | Rev SR high "
                   "| Gap low | Gap high | H3 | 95% CI | p (H3) | p (H1) | p (H2) |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for pkey, (plabel, _, _) in PERIODS.items():
            g = sub[(sub["period"] == pkey) & (sub["mom"] == m) & (sub["rev"] == r)].iloc[0]
            out.append(
                f"| {plabel} | {_f(g.mom_sharpe_low)} | {_f(g.mom_sharpe_high)} "
                f"| {_f(g.rev_sharpe_low)} | {_f(g.rev_sharpe_high)} "
                f"| {_f(g.gap_low)} | {_f(g.gap_high)} | {_f(g.H3)} "
                f"| [{_f(g.H3_ci95_low)}, {_f(g.H3_ci95_high)}] "
                f"| {_p(g.H3_p)} | {_p(g.H1_p)} | {_p(g.H2_p)} |")
    return "\n".join(out)


def main() -> None:
    cfg.ensure_dirs()
    pd.set_option("display.width", 220, "display.max_columns", 60)
    print("=== Phase 6: lookback robustness (pre-registered; not new headline results) ===\n")

    cleaned = D.load_processed()
    labels = R.build(cleaned.nsei_returns, cleaned.vix).labels
    runs = D.flat_run_length(cleaned.prices)

    variants_by_uni = {}
    for uni in ("v2", "v1"):
        mask = universe_mask(uni, cleaned.prices.index)
        variants_by_uni[uni] = build_variants(cleaned, mask, runs)
        print(f"  strategies built: {uni}", flush=True)

    stats = pd.concat([regime_stats_table(v, labels, u) for u, v in variants_by_uni.items()],
                      ignore_index=True)
    stats.to_csv(cfg.RESULTS_DIR / "phase6_regime_stats.csv", index=False)
    legs = pd.concat([legs_table(v, labels, u) for u, v in variants_by_uni.items()],
                     ignore_index=True)
    legs.to_csv(cfg.RESULTS_DIR / "phase6_momentum_legs.csv", index=False)

    turnover = []
    for uni, v in variants_by_uni.items():
        for (kind, lab), res in v.items():
            for pkey, (_, lo, hi) in PERIODS.items():
                sub = S.slice_period(res, lo, hi)
                t = float(sub.turnover.mean())
                turnover.append({"universe": uni, "period": pkey, "strategy": res.name,
                                 "mean_turnover": round(t, 3),
                                 "mean_n_long": round(float(sub.n_long.mean()), 1),
                                 "breakeven_bps": round(10_000 * float(sub.gross.mean()) / t, 1)
                                 if t > 0 else np.nan})
    pd.DataFrame(turnover).to_csv(cfg.RESULTS_DIR / "phase6_turnover.csv", index=False)

    # Tests: Version 2 only (the headline universe), as pre-registered.
    v2 = variants_by_uni["v2"]
    boot = bootstraps(v2, labels)
    boot.to_csv(cfg.RESULTS_DIR / "phase6_bootstrap.csv", index=False)
    gaps = gap_table(v2, labels, boot)
    gaps.to_csv(cfg.RESULTS_DIR / "phase6_gap_table.csv", index=False)
    nw = newey_west_table(v2, labels)
    nw.to_csv(cfg.RESULTS_DIR / "phase6_newey_west.csv", index=False)

    surv = survivorship_table(stats, legs)
    surv.to_csv(cfg.RESULTS_DIR / "phase6_survivorship_crashes.csv", index=False)
    worst = worst_high_vol_month(variants_by_uni, labels)
    worst.to_csv(cfg.RESULTS_DIR / "phase6_worst_high_vol_month.csv", index=False)
    verd = verdicts(gaps, surv)
    verd.to_csv(cfg.RESULTS_DIR / "phase6_verdicts.csv", index=False)
    rep = reproduction_check(gaps)
    rep.to_csv(cfg.RESULTS_DIR / "phase6_reproduction_check.csv", index=False)

    md = ["# Phase 6 tables (generated by src/run_phase6_robustness.py)\n"]
    for cost in ("20bps", "gross"):
        md.append(f"\n## Method A, {cost}, Version 2\n")
        md.append(markdown_tables(gaps, surv, cost=cost))
    for method in ("VIX (main robustness)", "B: HMM (underpowered)"):
        md.append(f"\n## {method}, 20bps, Version 2\n")
        md.append(markdown_tables(gaps, surv, method=method))
    (cfg.RESULTS_DIR / "phase6_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- console ----------------------------------------------------------
    print(f"\nMain pair reproduces Phase 4 exactly: {bool(rep['identical'].all())} "
          f"({int(rep['identical'].sum())}/{len(rep)} rows)")
    print("\nVerdicts (Method A, 20bps, Version 2):\n")
    print(verd.to_string(index=False))
    print("\nH3 across the full 3x3 grid (Method A, 20bps, Version 2):\n")
    grid = gaps[(gaps["regime_method"] == MAIN_METHOD) & (gaps["cost"] == "20bps")]
    for pkey in PERIODS:
        g = grid[grid["period"] == pkey]
        print(f"  {pkey}: H3")
        print(g.pivot(index="mom", columns="rev", values="H3")
              .reindex(index=list(MOM_LOOKBACKS), columns=list(REV_LOOKBACKS)).to_string())
        print(f"  {pkey}: p")
        print(g.pivot(index="mom", columns="rev", values="H3_p")
              .reindex(index=list(MOM_LOOKBACKS), columns=list(REV_LOOKBACKS)).to_string())
    print("\nSurvivorship and momentum crashes (Method A, 20bps, high-vol months):\n")
    print(surv.to_string(index=False))
    print(f"\n  Tables -> {cfg.RESULTS_DIR}/phase6_*.csv and phase6_tables.md")


if __name__ == "__main__":
    main()
