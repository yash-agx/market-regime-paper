"""Phase 7: is H3 one market-beta effect counted twice?

    python -m src.run_phase7_mechanism

Pre-registered in docs/phase7_preregistration.md before anything here was run.
This is a mechanism check on an existing result, not a new hypothesis test.

The idea being tested
---------------------
High-volatility periods are often rebounds. In a rebound momentum's short leg
(past losers) and reversal's long leg (recent losers) are both high-beta, so
momentum loses and reversal gains for the same reason. If that is all there is,
H1 and H2 are the same market bet seen twice and H3 adds them up.

What is done
------------
  * Regression A: spread = a0 + a1*HV + b0*Mkt + b1*HV*Mkt + e. a1 is the
    high-vol shift left once the spread's beta may differ between regimes.
  * Regression B (Daniel-Moskowitz): bear-market state from the past 24 months
    of the index, with an extra beta term for up-markets inside a bear state.
  * H3 recomputed on the hedged series (spread - b0*Mkt - b1*HV*Mkt), with the
    paper's month-block bootstrap, betas re-estimated inside every resample.
  * The same with a slower regime label: 126-day realised market variance
    above its expanding median.

Market return: the equal-weighted Nifty 50 benchmark of the same universe, over
the same holding window, minus the T-bill rate. Hedge betas are full-period
betas - an attribution, not a tradable hedge. B_t and HV_t only use data up to
the rebalance date.

Nothing from Phases 1-6 is overwritten: every output here is a new
results/phase7_* file.
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
from . import run_phase6_robustness as P6
from .run_phase4 import universe_mask

NW_LAGS = 3
SLOW_WINDOW = 126                 # about six months of trading days
BEAR_LOOKBACK_MONTHS = 24
MIN_BEAR_OBS = {"monthly": 12, "weekly": 52}
COSTS = (20, None)                # 20 bp is the main case; gross alongside
PERIODS = P6.PERIODS
MAIN_PAIR = P6.MAIN_PAIR
PRIMARY_PAIRS = P6.PRIMARY_PAIRS
UNIVERSES = ("v2", "v1")          # V2 is the headline, V1 the comparison
REGIME_DEFS = {"A": "Method A (21-day vol)", "126d": "126-day variance"}
PPY = {"monthly": cfg.MONTHS_PER_YEAR, "weekly": cfg.WEEKS_PER_YEAR}


def cost_label(cost):
    return P6.cost_label(cost)


# --------------------------------------------------------------------------
# state variables
# --------------------------------------------------------------------------


def slow_regime(index_returns: pd.Series, window: int = SLOW_WINDOW) -> pd.Series:
    """High-vol if the trailing `window`-day realised variance is above its expanding median.

    Right-aligned window and an expanding median that includes the current day,
    exactly as Method A, only slower.
    """
    var = index_returns.astype("float64").rolling(window).var() * cfg.TRADING_DAYS
    flag, _ = R.expanding_median_flag(var)
    return flag.rename("high_vol_126d")


def bear_state(level: pd.Series, dates, months: int = BEAR_LOOKBACK_MONTHS) -> pd.Series:
    """B_t = 1 if the index fell over the `months` calendar months to date t.

    Uses the last close on or before t and the last close on or before
    t - months. NaN when the data do not reach back that far.
    """
    level = level.dropna().astype("float64")
    out = {}
    for d in pd.DatetimeIndex(dates):
        now = level.loc[:d]
        past = level.loc[:d - pd.DateOffset(months=months)]
        if now.empty or past.empty:
            out[d] = np.nan
        else:
            out[d] = float(now.iloc[-1] / past.iloc[-1] - 1.0 < 0)
    return pd.Series(out, dtype="float64", name="bear")


# --------------------------------------------------------------------------
# regressions
# --------------------------------------------------------------------------


def align(spread: pd.Series, mkt: pd.Series, flag: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"y": spread, "m": mkt.reindex(spread.index),
                       "hv": flag.reindex(spread.index)}).dropna()
    df["hv"] = df["hv"].astype("float64")
    return df


def design_a(hv: np.ndarray, m: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones_like(m), hv, m, hv * m])


def hedge(y: np.ndarray, hv: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Spread with its regime-dependent market exposure taken out: a0 + a1*HV + e."""
    b = np.linalg.lstsq(design_a(hv, m), y, rcond=None)[0]
    return y - b[2] * m - b[3] * hv * m


def nw_fit(y: np.ndarray, X: np.ndarray, names: list[str], lags: int) -> dict:
    import statsmodels.api as sm

    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    out = {}
    for i, nm in enumerate(names):
        out[nm] = float(fit.params[i])
        out[f"{nm}_t"] = float(fit.tvalues[i])
        out[f"{nm}_p"] = float(fit.pvalues[i])
    out["r2"] = float(fit.rsquared)
    return out


def regression_a(spread: pd.Series, mkt: pd.Series, flag: pd.Series,
                 lags: int = NW_LAGS) -> dict:
    """Regression A, plus the raw high-vol difference and an automatic-lag check."""
    df = align(spread, mkt, flag)
    y, hv, m = df["y"].to_numpy(), df["hv"].to_numpy(), df["m"].to_numpy()
    n = len(df)
    full = nw_fit(y, design_a(hv, m), ["a0", "a1", "b0", "b1"], lags)
    raw = nw_fit(y, np.column_stack([np.ones(n), hv]), ["c", "raw_diff"], lags)
    auto = nw_fit(y, design_a(hv, m), ["a0", "a1", "b0", "b1"], E.newey_west_lags(n))
    return {
        "n": n, "n_high": int(hv.sum()), "nw_lags": lags,
        **{k: full[k] for k in ("a0", "a1", "a1_t", "a1_p", "b0", "b0_t", "b0_p",
                                "b1", "b1_t", "b1_p", "r2")},
        "beta_calm": full["b0"], "beta_high": full["b0"] + full["b1"],
        "raw_diff": raw["raw_diff"], "raw_diff_t": raw["raw_diff_t"],
        "raw_diff_p": raw["raw_diff_p"],
        "a1_over_raw": full["a1"] / raw["raw_diff"] if raw["raw_diff"] != 0 else np.nan,
        "auto_lags": E.newey_west_lags(n), "a1_t_auto": auto["a1_t"],
        "b1_t_auto": auto["b1_t"],
    }


def regression_b(spread: pd.Series, mkt: pd.Series, bear: pd.Series,
                 frequency: str, lags: int = NW_LAGS) -> dict:
    """Daniel-Moskowitz: y = g0 + gB*B + b0*M + bB*B*M + bBU*B*U*M + e."""
    df = pd.DataFrame({"y": spread, "m": mkt.reindex(spread.index),
                       "b": bear.reindex(spread.index)}).dropna()
    y, m, b = df["y"].to_numpy(), df["m"].to_numpy(), df["b"].to_numpy()
    u = (m > 0).astype("float64")
    n_bear = int(b.sum())
    row = {"n": len(df), "n_bear": n_bear, "n_bear_up": int((b * u).sum()),
           "first_date": df.index.min().date() if len(df) else None, "nw_lags": lags}
    if n_bear < 3 or n_bear == len(df):
        row["note"] = "too few bear periods to estimate"
        return row
    X = np.column_stack([np.ones_like(m), b, m, b * m, b * u * m])
    fit = nw_fit(y, X, ["g0", "gB", "b0", "bB", "bBU"], lags)
    row.update({k: fit[k] for k in ("g0", "gB", "gB_t", "b0", "b0_t", "bB", "bB_t",
                                    "bBU", "bBU_t", "bBU_p", "r2")})
    row["note"] = ("too few bear periods to interpret"
                   if n_bear < MIN_BEAR_OBS[frequency] else "")
    return row


# --------------------------------------------------------------------------
# H3: raw against beta-adjusted, with the paper's month-block bootstrap
# --------------------------------------------------------------------------


def _month_table(month_idx: np.ndarray, n_months: int) -> np.ndarray:
    """Row positions for each month, padded with -1."""
    counts = np.bincount(month_idx, minlength=n_months)
    tab = np.full((n_months, max(1, counts.max())), -1, dtype=int)
    fill = np.zeros(n_months, dtype=int)
    for pos, mth in enumerate(month_idx):
        tab[mth, fill[mth]] = pos
        fill[mth] += 1
    return tab


def _regime_sharpes(r: np.ndarray, h: np.ndarray, ppy: int) -> tuple[float, float]:
    lo, hi = r[h == 0], r[h == 1]
    if len(lo) < E.MIN_REGIME_OBS or len(hi) < E.MIN_REGIME_OBS:
        return np.nan, np.nan
    return E.sharpe(lo, ppy), E.sharpe(hi, ppy)


STAT_NAMES = ["H1_raw", "H2_raw", "H3_raw", "H1_adj", "H2_adj", "H3_adj"]


def decompose_h3(mom: pd.Series, mom_mkt: pd.Series, mom_flag: pd.Series,
                 rev: pd.Series, rev_mkt: pd.Series, rev_flag: pd.Series,
                 n_boot: int = E.N_BOOTSTRAP, seed: int = cfg.RANDOM_SEED,
                 mean_block: float | None = None) -> dict:
    """Raw and beta-hedged H1, H2, H3 from one set of month-block resamples.

    The resampling is the one in E.bootstrap_difference_in_differences: calendar
    months are drawn, each bringing its momentum month and its reversal weeks,
    with the same seed and the same Politis-White block, so the raw H3 draws are
    identical to the published ones. Inside each resample regression A is fitted
    again on the drawn rows before hedging.
    """
    md, rd = align(mom, mom_mkt, mom_flag), align(rev, rev_mkt, rev_flag)
    months = sorted(set(md.index.to_period("M")) | set(rd.index.to_period("M")))
    pos = {mth: i for i, mth in enumerate(months)}
    n = len(months)
    m_month = np.array([pos[p] for p in md.index.to_period("M")], dtype=int)
    r_month = np.array([pos[p] for p in rd.index.to_period("M")], dtype=int)
    m_tab, r_tab = _month_table(m_month, n), _month_table(r_month, n)

    my, mh, mm = md["y"].to_numpy(), md["hv"].to_numpy(), md["m"].to_numpy()
    ry, rh, rm = rd["y"].to_numpy(), rd["hv"].to_numpy(), rd["m"].to_numpy()

    if mean_block is None:
        mom_m = np.full(n, np.nan)
        rev_m = np.full(n, np.nan)
        dum_m = np.full(n, np.nan)
        for i in range(n):
            mi, ri = m_tab[i][m_tab[i] >= 0], r_tab[i][r_tab[i] >= 0]
            if len(mi):
                mom_m[i] = my[mi].sum()
            if len(ri):
                rev_m[i] = np.prod(1 + ry[ri]) - 1
            flags = np.concatenate([mh[mi], rh[ri]])
            if len(flags):
                dum_m[i] = np.nanmean(flags)
        mean_block = E.politis_white_block(mom_m, rev_m, dum_m)

    def stat(order: np.ndarray) -> np.ndarray:
        mi = m_tab[order].ravel()
        mi = mi[mi >= 0]
        ri = r_tab[order].ravel()
        ri = ri[ri >= 0]
        y1, h1, x1 = my[mi], mh[mi], mm[mi]
        y2, h2, x2 = ry[ri], rh[ri], rm[ri]
        out = np.full(6, np.nan)
        ml, mhs = _regime_sharpes(y1, h1, cfg.MONTHS_PER_YEAR)
        rl, rhs = _regime_sharpes(y2, h2, cfg.WEEKS_PER_YEAR)
        if not np.all(np.isfinite([ml, mhs, rl, rhs])):
            return out
        out[0], out[1] = ml - mhs, rhs - rl
        out[2] = (ml - rl) - (mhs - rhs)
        al, ah = _regime_sharpes(hedge(y1, h1, x1), h1, cfg.MONTHS_PER_YEAR)
        bl, bh = _regime_sharpes(hedge(y2, h2, x2), h2, cfg.WEEKS_PER_YEAR)
        if np.all(np.isfinite([al, ah, bl, bh])):
            out[3], out[4] = al - ah, bh - bl
            out[5] = (al - bl) - (ah - bh)
        return out

    observed = stat(np.arange(n))
    rng = np.random.default_rng(seed)
    idx = E.stationary_bootstrap_indices(n, mean_block, n_boot, rng)
    draws = np.vstack([stat(row) for row in idx])

    res = {"mean_block": mean_block, "n_boot": n_boot,
           "n_mom": len(md), "n_mom_high": int(mh.sum()),
           "n_rev": len(rd), "n_rev_high": int(rh.sum())}
    for j, nm in enumerate(STAT_NAMES):
        d = draws[:, j]
        d = d[np.isfinite(d)]
        res[nm] = observed[j]
        res[f"{nm}_lo"] = np.percentile(d, 2.5) if len(d) else np.nan
        res[f"{nm}_hi"] = np.percentile(d, 97.5) if len(d) else np.nan
        res[f"{nm}_p"] = float((d <= 0).mean()) if len(d) else np.nan
        res[f"{nm}_valid"] = int(len(d))
    for h in ("H1", "H2", "H3"):
        diff = draws[:, STAT_NAMES.index(f"{h}_raw")] - draws[:, STAT_NAMES.index(f"{h}_adj")]
        diff = diff[np.isfinite(diff)]
        res[f"{h}_explained"] = res[f"{h}_raw"] - res[f"{h}_adj"]
        res[f"{h}_explained_lo"] = np.percentile(diff, 2.5) if len(diff) else np.nan
        res[f"{h}_explained_hi"] = np.percentile(diff, 97.5) if len(diff) else np.nan
        raw = res[f"{h}_raw"]
        res[f"{h}_fraction_surviving"] = res[f"{h}_adj"] / raw if raw else np.nan
    res["verdict"] = verdict(res["H3_raw"], res["H3_adj"], res["H3_adj_lo"])
    return res


def verdict(raw: float, adj: float, adj_lo: float) -> str:
    """The reading rule fixed in the pre-registration."""
    if not np.all(np.isfinite([raw, adj])):
        return "n/a"
    if raw <= 0:
        return "raw H3 not positive; not interpreted"
    if adj <= 0:
        return "mostly explained"
    f = adj / raw
    if f < 1 / 3:
        return "mostly explained"
    if f >= 2 / 3 and adj_lo > 0:
        return "survives"
    return "partly explained"


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def market_series(cleaned: D.CleanedData, mask: pd.DataFrame, runs: pd.DataFrame,
                  rf_daily: pd.Series) -> dict[str, pd.Series]:
    """Equal-weighted benchmark of this universe, minus T-bill, on both grids."""
    cal = cleaned.prices.index
    mom_w = P6.in_study(W.momentum_windows(cal))
    rev_w = P6.in_study(W.reversal_windows(cal))
    bm = S.run_strategy(cleaned, mom_w, "top", None, "EW benchmark (monthly)", "monthly",
                        runs, equal_weight_all=True, universe_mask=mask)
    bw = S.run_strategy(cleaned, rev_w, "bottom", None, "EW benchmark (weekly)", "weekly",
                        runs, equal_weight_all=True, universe_mask=mask)
    rf_m = E.period_riskfree(rf_daily, mom_w).reindex(bm.gross.index)
    rf_w = E.period_riskfree(rf_daily, rev_w).reindex(bw.gross.index)
    return {"monthly": (bm.gross - rf_m).rename("mkt"),
            "weekly": (bw.gross - rf_w).rename("mkt"),
            "weekly_gross": bw.gross, "weekly_rf": rf_w}


def to_calendar_months(rev: pd.Series, mkt_gross_w: pd.Series, rf_w: pd.Series,
                       flag: pd.Series) -> pd.DataFrame:
    """Weekly reversal compounded within each calendar month (secondary check).

    The market is compounded over exactly the same weeks (benchmark and T-bill
    separately, then differenced). The month takes the regime label of its first
    reversal rebalance date.
    """
    df = pd.DataFrame({"y": rev, "g": mkt_gross_w.reindex(rev.index),
                       "rf": rf_w.reindex(rev.index)}).dropna()
    rows = []
    for per, g in df.groupby(df.index.to_period("M")):
        first = g.index.min()
        rows.append({"month": per, "first_date": first,
                     "y": np.prod(1 + g["y"]) - 1,
                     "m": np.prod(1 + g["g"]) - np.prod(1 + g["rf"]),
                     "hv": flag.reindex([first]).iloc[0]})
    out = pd.DataFrame(rows).set_index("first_date")
    return out


# --------------------------------------------------------------------------
# descriptives
# --------------------------------------------------------------------------


def market_descriptives(mkt: pd.Series, flag: pd.Series) -> dict:
    df = pd.DataFrame({"m": mkt, "hv": flag.reindex(mkt.index)}).dropna()
    hv = df["hv"].astype(bool)
    out = {}
    for name, sel in (("high", hv), ("calm", ~hv)):
        m = df["m"][sel]
        out[f"n_{name}"] = int(len(m))
        out[f"share_up_{name}"] = float((m > 0).mean()) if len(m) else np.nan
        out[f"mean_mkt_{name}_pct"] = 100 * float(m.mean()) if len(m) else np.nan
        out[f"sd_mkt_{name}_pct"] = 100 * float(m.std(ddof=1)) if len(m) > 1 else np.nan
    return out


def mom_rev_correlation(mom: pd.Series, mom_mkt: pd.Series, mom_flag: pd.Series,
                        rev_monthly: pd.DataFrame) -> dict:
    """Correlation of the two spreads by regime, before and after hedging.

    Momentum formed at month-end t is held over calendar month t+1, so it is
    matched with reversal compounded over calendar month t+1. The regime is
    momentum's (the label at the start of the month).
    """
    md = align(mom, mom_mkt, mom_flag)
    md["y_h"] = hedge(md["y"].to_numpy(), md["hv"].to_numpy(), md["m"].to_numpy())
    rv = rev_monthly.dropna().copy()
    rv["hv"] = rv["hv"].astype("float64")
    rv["y_h"] = hedge(rv["y"].to_numpy(), rv["hv"].to_numpy(), rv["m"].to_numpy())
    rv = rv.set_index("month")
    md["month"] = md.index.to_period("M") + 1
    rv = rv[["y", "y_h"]].rename(columns={"y": "y_rev", "y_h": "y_h_rev"})
    j = md.join(rv, on="month").dropna()
    out = {}
    for name, sel in (("high", j["hv"] == 1), ("calm", j["hv"] == 0)):
        g = j[sel]
        out[f"n_{name}"] = int(len(g))
        out[f"corr_raw_{name}"] = float(g["y"].corr(g["y_rev"])) if len(g) > 2 else np.nan
        out[f"corr_hedged_{name}"] = (float(g["y_h"].corr(g["y_h_rev"]))
                                      if len(g) > 2 else np.nan)
    return out


# --------------------------------------------------------------------------
# reproduction check
# --------------------------------------------------------------------------


def reproduction_check(h3: pd.DataFrame) -> pd.DataFrame:
    """Raw H3 for V2 / Method A must equal the Phase 6 gap table exactly."""
    p6 = pd.read_csv(cfg.RESULTS_DIR / "phase6_gap_table.csv")
    p6 = p6[p6["regime_method"] == P6.MAIN_METHOD]
    sub = h3[(h3["universe"] == "v2") & (h3["regime"] == "A")]
    rows = []
    for _, r in sub.iterrows():
        q = p6[(p6["period"] == r["period"]) & (p6["cost"] == r["cost"])
               & (p6["mom"] == r["mom"]) & (p6["rev"] == r["rev"])].iloc[0]
        ours = [round(r["H3_raw"], 4), round(r["H3_raw_lo"], 4),
                round(r["H3_raw_hi"], 4), round(r["H3_raw_p"], 4)]
        theirs = [q["H3"], q["H3_ci95_low"], q["H3_ci95_high"], q["H3_p"]]
        rows.append({"period": r["period"], "cost": r["cost"], "mom": r["mom"],
                     "rev": r["rev"], "phase7": ours, "phase6": theirs,
                     "identical": bool(np.allclose(ours, theirs, atol=1e-9))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# markdown
# --------------------------------------------------------------------------


def _f(x, nd=2, signed=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def _ci(lo, hi):
    return f"[{_f(lo)}, {_f(hi)}]"


def h3_markdown(h3: pd.DataFrame, universe: str, regime: str, cost: str,
                pairs=None) -> str:
    sub = h3[(h3["universe"] == universe) & (h3["regime"] == regime)
             & (h3["cost"] == cost)]
    lines = ["| Pair | Period | Raw H3 | 95% CI | Adjusted H3 | 95% CI | Explained by beta | 95% CI | Fraction surviving | p (adj) | Verdict |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for pair in (pairs or PRIMARY_PAIRS):
        for pkey in PERIODS:
            r = sub[(sub["mom"] == pair[0]) & (sub["rev"] == pair[1])
                    & (sub["period"] == pkey)]
            if r.empty:
                continue
            r = r.iloc[0]
            lab = P6.pair_label(pair) + (" (main)" if pair == MAIN_PAIR else "")
            lines.append(
                f"| {lab} | {pkey} | {_f(r['H3_raw'])} | {_ci(r['H3_raw_lo'], r['H3_raw_hi'])} "
                f"| {_f(r['H3_adj'])} | {_ci(r['H3_adj_lo'], r['H3_adj_hi'])} "
                f"| {_f(r['H3_explained'])} | {_ci(r['H3_explained_lo'], r['H3_explained_hi'])} "
                f"| {_f(r['H3_fraction_surviving'], 2, False)} | {_f(r['H3_adj_p'], 3, False)} "
                f"| {r['verdict']} |")
    return "\n".join(lines)


def halves_markdown(h3: pd.DataFrame, universe: str, regime: str, cost: str) -> str:
    sub = h3[(h3["universe"] == universe) & (h3["regime"] == regime)
             & (h3["cost"] == cost) & (h3["mom"] == MAIN_PAIR[0])
             & (h3["rev"] == MAIN_PAIR[1])]
    lines = ["| Period | H1 raw | H1 adj | H1 explained [95% CI] | H2 raw | H2 adj | H2 explained [95% CI] |",
             "|---|---|---|---|---|---|---|"]
    for _, r in sub.iterrows():
        lines.append(
            f"| {r['period']} | {_f(r['H1_raw'])} | {_f(r['H1_adj'])} "
            f"| {_f(r['H1_explained'])} {_ci(r['H1_explained_lo'], r['H1_explained_hi'])} "
            f"| {_f(r['H2_raw'])} | {_f(r['H2_adj'])} "
            f"| {_f(r['H2_explained'])} {_ci(r['H2_explained_lo'], r['H2_explained_hi'])} |")
    return "\n".join(lines)


def rega_markdown(ra: pd.DataFrame, universe: str, regime: str, cost: str) -> str:
    sub = ra[(ra["universe"] == universe) & (ra["regime"] == regime) & (ra["cost"] == cost)]
    lines = ["| Spread | Period | n (high) | Raw HV diff, % (t) | a1, % (t) | a1 / raw | beta calm (b0) | beta high (b0+b1) | b1 (t) | R² |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in sub.iterrows():
        lines.append(
            f"| {r['spread']} | {r['period']} | {r['n']} ({r['n_high']}) "
            f"| {_f(100 * r['raw_diff'])} ({_f(r['raw_diff_t'])}) "
            f"| {_f(100 * r['a1'])} ({_f(r['a1_t'])}) | {_f(r['a1_over_raw'], 2, False)} "
            f"| {_f(r['beta_calm'])} | {_f(r['beta_high'])} | {_f(r['b1'])} ({_f(r['b1_t'])}) "
            f"| {_f(r['r2'], 2, False)} |")
    return "\n".join(lines)


def regb_markdown(rb: pd.DataFrame, universe: str, cost: str) -> str:
    sub = rb[(rb["universe"] == universe) & (rb["cost"] == cost)]
    lines = ["| Spread | Period | n | bear (bear & up) | gB, % (t) | b0 (t) | bB (t) | bBU (t) | note |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in sub.iterrows():
        gb = _f(100 * r["gB"]) if pd.notna(r.get("gB")) else "-"
        lines.append(
            f"| {r['spread']} | {r['period']} | {r['n']} | {r['n_bear']} ({r['n_bear_up']}) "
            f"| {gb} ({_f(r.get('gB_t'))}) | {_f(r.get('b0'))} ({_f(r.get('b0_t'))}) "
            f"| {_f(r.get('bB'))} ({_f(r.get('bB_t'))}) | {_f(r.get('bBU'))} ({_f(r.get('bBU_t'))}) "
            f"| {r['note'] if isinstance(r['note'], str) else ''} |")
    return "\n".join(lines)


def desc_markdown(desc: pd.DataFrame, corr: pd.DataFrame, universe: str) -> str:
    d = desc[desc["universe"] == universe]
    lines = ["| Regime def | Period | Grid | n high / calm | Share up, high | Share up, calm | Mean mkt % high | Mean mkt % calm | SD % high | SD % calm |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in d.iterrows():
        lines.append(
            f"| {r['regime']} | {r['period']} | {r['grid']} | {r['n_high']} / {r['n_calm']} "
            f"| {r['share_up_high']:.2f} | {r['share_up_calm']:.2f} "
            f"| {_f(r['mean_mkt_high_pct'])} | {_f(r['mean_mkt_calm_pct'])} "
            f"| {_f(r['sd_mkt_high_pct'], 2, False)} | {_f(r['sd_mkt_calm_pct'], 2, False)} |")
    c = corr[corr["universe"] == universe]
    lines += ["", "Correlation of momentum 12-1 with reversal 1w (compounded to the same calendar month), 20 bp:", "",
              "| Regime def | Period | n high / calm | corr high, raw | corr high, hedged | corr calm, raw | corr calm, hedged |",
              "|---|---|---|---|---|---|---|"]
    for _, r in c.iterrows():
        lines.append(
            f"| {r['regime']} | {r['period']} | {r['n_high']} / {r['n_calm']} "
            f"| {_f(r['corr_raw_high'])} | {_f(r['corr_hedged_high'])} "
            f"| {_f(r['corr_raw_calm'])} | {_f(r['corr_hedged_calm'])} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> None:
    cfg.ensure_dirs()
    pd.set_option("display.width", 220, "display.max_columns", 60)
    print("=== Phase 7: market-beta mechanism test (pre-registered) ===\n")

    cleaned = D.load_processed()
    labels = R.build(cleaned.nsei_returns, cleaned.vix).labels
    flags = {"A": labels["high_vol_a"], "126d": slow_regime(cleaned.nsei_returns)}
    runs = D.flat_run_length(cleaned.prices)
    rf_daily = D.load_riskfree(cleaned.prices.index)

    ra_rows, rb_rows, h3_rows, desc_rows, corr_rows, ram_rows = [], [], [], [], [], []
    for uni in UNIVERSES:
        mask = universe_mask(uni, cleaned.prices.index)
        variants = P6.build_variants(cleaned, mask, runs)
        mkt = market_series(cleaned, mask, runs, rf_daily)
        print(f"  strategies built: {uni}", flush=True)

        for pkey, (_, lo, hi) in PERIODS.items():
            sl = {k: S.slice_period(v, lo, hi) for k, v in variants.items()}
            mk = {g: mkt[g].loc[lo:hi] for g in ("monthly", "weekly")}
            bear = {g: bear_state(cleaned.nsei, mk[g].index) for g in mk}
            for rkey, flag in flags.items():
                base = {"universe": uni, "period": pkey, "regime": rkey}
                for g in ("monthly", "weekly"):
                    desc_rows.append({**base, "grid": g, **market_descriptives(mk[g], flag)})
                for cost in COSTS:
                    cl = cost_label(cost)
                    for (kind, lab), res in sl.items():
                        grid = "monthly" if kind == "mom" else "weekly"
                        name = f"{'Momentum' if kind == 'mom' else 'Reversal'} {lab}"
                        ra_rows.append({**base, "cost": cl, "spread": name,
                                        **regression_a(res.ret(cost), mk[grid], flag)})
                        if rkey == "A":
                            rb_rows.append({"universe": uni, "period": pkey, "cost": cl,
                                            "spread": name,
                                            **regression_b(res.ret(cost), mk[grid],
                                                           bear[grid], grid)})
                        if kind == "rev":
                            rmo = to_calendar_months(res.ret(cost), mkt["weekly_gross"],
                                                     mkt["weekly_rf"], flag)
                            ram_rows.append({**base, "cost": cl, "spread": name + " (monthly)",
                                             **regression_a(rmo["y"], rmo["m"], rmo["hv"])})
                            if lab == "1w" and cost == 20:
                                corr_rows.append({**base, **mom_rev_correlation(
                                    sl[("mom", "12-1")].ret(cost), mk["monthly"], flag, rmo)})
                    for m, r in PRIMARY_PAIRS:
                        d = decompose_h3(sl[("mom", m)].ret(cost), mk["monthly"], flag,
                                         sl[("rev", r)].ret(cost), mk["weekly"], flag)
                        h3_rows.append({**base, "cost": cl, "mom": m, "rev": r, **d})
                    print(f"  done: {uni} {pkey:8s} {rkey:4s} {cl}", flush=True)

    ra = pd.DataFrame(ra_rows)
    rb = pd.DataFrame(rb_rows)
    ram = pd.DataFrame(ram_rows)
    h3 = pd.DataFrame(h3_rows)
    desc = pd.DataFrame(desc_rows)
    corr = pd.DataFrame(corr_rows)
    rep = reproduction_check(h3)

    rnd = lambda df: df.round(6)
    rnd(ra).to_csv(cfg.RESULTS_DIR / "phase7_regression_a.csv", index=False)
    rnd(rb).to_csv(cfg.RESULTS_DIR / "phase7_regression_b.csv", index=False)
    rnd(ram).to_csv(cfg.RESULTS_DIR / "phase7_regression_a_reversal_monthly.csv", index=False)
    rnd(h3).to_csv(cfg.RESULTS_DIR / "phase7_h3_decomposition.csv", index=False)
    rnd(desc).to_csv(cfg.RESULTS_DIR / "phase7_market_descriptives.csv", index=False)
    rnd(corr).to_csv(cfg.RESULTS_DIR / "phase7_mom_rev_correlation.csv", index=False)
    rep.to_csv(cfg.RESULTS_DIR / "phase7_reproduction_check.csv", index=False)

    md = ["# Phase 7 tables (generated by src/run_phase7_mechanism.py)", "",
          "Pre-registered in docs/phase7_preregistration.md. Spreads are long-short; the "
          "market is the equal-weighted Nifty 50 benchmark of the same universe minus the "
          "T-bill rate. Newey-West with 3 lags. Bootstrap: month-block stationary bootstrap, "
          "10,000 resamples, seed 20100101, betas re-estimated in every resample. "
          "H1 and H2 here come from the same joint month-block resamples as H3, so their "
          "intervals differ slightly from the separate Phase 4 / Phase 6 H1 and H2 bootstraps.",
          "",
          f"Reproduction check (raw H3, V2, Method A, against Phase 6): "
          f"{int(rep['identical'].sum())}/{len(rep)} rows identical.", ""]
    for uni, uname in (("v2", "Version 2 (point-in-time, headline)"),
                       ("v1", "Version 1 (today's members, comparison)")):
        md.append(f"## {uname}\n")
        for rkey, rname in REGIME_DEFS.items():
            for cost in ("20bps", "gross"):
                md.append(f"### H3 raw vs beta-adjusted: {rname}, {cost}\n")
                md.append(h3_markdown(h3, uni, rkey, cost))
                md.append(f"\nMain pair, the two halves ({rname}, {cost}):\n")
                md.append(halves_markdown(h3, uni, rkey, cost) + "\n")
        for rkey, rname in REGIME_DEFS.items():
            md.append(f"### Regression A: {rname}, 20bps\n")
            md.append(rega_markdown(ra, uni, rkey, "20bps") + "\n")
        md.append("### Regression B (Daniel-Moskowitz), 20bps\n")
        md.append("B_t = 1 if `^NSEI` fell over the 24 months to the rebalance date; "
                  "first defined January 2011. Power rule: under 12 bear months "
                  "(52 bear weeks) the row is reported but not interpreted.\n")
        md.append(regb_markdown(rb, uni, "20bps") + "\n")
        md.append("### Market descriptives\n")
        md.append(desc_markdown(desc, corr, uni) + "\n")
        md.append("### Secondary: reversal compounded to calendar months, Method A, 20bps\n")
        md.append("Market compounded over exactly the same weeks (weekly benchmark and "
                  "T-bill compounded separately). Not used for the verdict.\n")
        md.append(rega_markdown(ram, uni, "A", "20bps") + "\n")
    (cfg.RESULTS_DIR / "phase7_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"\nRaw H3 reproduces Phase 6: {bool(rep['identical'].all())} "
          f"({int(rep['identical'].sum())}/{len(rep)})")
    main = h3[(h3["mom"] == MAIN_PAIR[0]) & (h3["rev"] == MAIN_PAIR[1])]
    cols = ["universe", "period", "regime", "cost", "H3_raw", "H3_raw_lo", "H3_raw_hi",
            "H3_adj", "H3_adj_lo", "H3_adj_hi", "H3_explained", "H3_fraction_surviving",
            "verdict"]
    print("\nMain pair, H3 raw vs beta-adjusted:\n")
    print(main[cols].round(3).to_string(index=False))
    print(f"\n  Tables -> {cfg.RESULTS_DIR}/phase7_*.csv and phase7_tables.md")


if __name__ == "__main__":
    main()
