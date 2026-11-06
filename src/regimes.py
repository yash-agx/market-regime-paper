"""Phase 2: volatility regime detection.

Three ways of answering "was the market calm or stormy on day t?", all of which
may use only data up to and including day t.

Method A (main) - realised volatility against an expanding median
    Take the last 21 daily index returns, compute their standard deviation, and
    annualise it by multiplying by sqrt(252). Call the day high-volatility if
    that number is above the median of every volatility reading seen so far.
    The expanding median matters: a full-sample median would let the code know
    how volatile 2008 or 2020 turned out to be before those years happened, and
    the regime labels would quietly encode the future.

Method B - a 2-state Gaussian hidden Markov model
    Assume the index return each day is drawn from one of two normal
    distributions - a calm one and a stormy one - and that the market switches
    between them with fixed probabilities. The model is refit every 12 months on
    an expanding window, and the state on day t is read off the FILTERED
    probability P(state_t | returns up to t). This is the important part: the
    usual output of an HMM fit is the *smoothed* probability, which uses the
    whole sample to decide what state day t was in. Smoothed probabilities are
    the right tool for describing history and completely wrong for a backtest,
    so the forward pass is implemented here by hand rather than taken from
    hmmlearn's predict_proba.

Robustness - India VIX against its expanding median
    Same expanding-median logic as Method A, but using the market's own implied
    volatility rather than realised volatility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as cfg

REALISED_VOL_WINDOW = 21
HMM_N_STATES = 2
HMM_REFIT_MONTHS = 12


# --------------------------------------------------------------------------
# Method A: realised volatility vs an expanding median
# --------------------------------------------------------------------------


def realised_vol(index_returns: pd.Series, window: int = REALISED_VOL_WINDOW) -> pd.Series:
    """Annualised standard deviation of the last `window` daily returns.

    Right-aligned, so the value on day t uses days t-20..t and nothing later.
    """
    r = index_returns.astype("float64")
    return (r.rolling(window).std() * np.sqrt(cfg.TRADING_DAYS)).rename("realised_vol")


def expanding_median_flag(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Flag days where `series` is above the median of all its values up to that day.

    Returns (flag, median). The median on day t includes day t, which is known
    at t, and no later value.

    A day with no volatility reading yet (the first 20 days of the sample) gets
    pd.NA, not False. Calling an unlabelled day "low volatility" would quietly
    invent data, and the nullable dtype forces later phases to deal with it.
    """
    s = series.astype("float64")
    med = s.expanding().median()
    flag = (s > med).astype("boolean")
    flag[s.isna()] = pd.NA
    return flag.rename("high_vol"), med.rename("expanding_median")


def method_a(index_returns: pd.Series) -> pd.DataFrame:
    vol = realised_vol(index_returns)
    flag, med = expanding_median_flag(vol)
    return pd.DataFrame({"realised_vol": vol, "expanding_median": med, "high_vol": flag})


# --------------------------------------------------------------------------
# Method B: 2-state Gaussian HMM, filtered probabilities only
# --------------------------------------------------------------------------


@dataclass
class HMMFit:
    """One refit of the HMM, with the data cut-off it was trained on."""

    fit_date: pd.Timestamp
    startprob: np.ndarray
    transmat: np.ndarray
    means: np.ndarray
    sds: np.ndarray
    high_state: int          # index of the higher-variance state
    n_train: int
    converged: bool
    loglik: float = np.nan
    transmat_prior: float = 1.0

    @property
    def stationary_p_high(self) -> float:
        """The model's own long-run share of high-volatility days.

        Useful as a sanity check: if this is 0, the low-volatility state is
        absorbing and the filter can never report high volatility again.
        """
        lo = 1 - self.high_state
        leave_low = 1 - self.transmat[lo, lo]
        leave_high = 1 - self.transmat[self.high_state, self.high_state]
        if leave_low + leave_high == 0:
            return np.nan
        return float(leave_low / (leave_low + leave_high))


def _fit_hmm(
    returns: np.ndarray,
    seed: int,
    transmat_prior: float = None,
    restarts: int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, float]:
    """Fit the 2-state Gaussian HMM, keeping the best of several restarts.

    The restarts are seeded deterministically from the master seed, so the
    result is reproducible. They guard against EM settling in a poor local
    optimum; the best log-likelihood wins.
    """
    import logging

    from hmmlearn.hmm import GaussianHMM

    # hmmlearn narrates every tolerance wobble to the log; the convergence flag
    # is captured per fit instead.
    logging.getLogger("hmmlearn").setLevel(logging.ERROR)

    transmat_prior = cfg.HMM_TRANSMAT_PRIOR if transmat_prior is None else transmat_prior
    restarts = cfg.HMM_RESTARTS if restarts is None else restarts
    X = returns.reshape(-1, 1)

    best = None
    for k in range(restarts):
        model = GaussianHMM(
            n_components=HMM_N_STATES,
            covariance_type="full",
            n_iter=1000,
            tol=1e-6,
            random_state=seed + k,
            transmat_prior=transmat_prior,
        )
        model.fit(X)
        loglik = float(model.score(X))
        if not np.isfinite(loglik):
            continue
        if best is None or loglik > best[0]:
            best = (loglik, model)
    if best is None:
        raise RuntimeError("every HMM restart failed")

    loglik, model = best
    means = model.means_.ravel()
    sds = np.sqrt(np.array([model.covars_[i].ravel()[0] for i in range(HMM_N_STATES)]))
    return (model.startprob_, model.transmat_, means, sds,
            bool(model.monitor_.converged), loglik)


def refit_dates(calendar: pd.DatetimeIndex, every_months: int = HMM_REFIT_MONTHS) -> list[pd.Timestamp]:
    """Last trading day of December, every `every_months` months.

    The model fitted on date R is used for the days after R, up to the next
    refit date - so a day is never labelled by a model that has seen it.
    """
    from .windows import month_end_dates

    ends = month_end_dates(calendar)
    decembers = [d for d in ends if d.month == 12]
    return list(decembers[::every_months // 12]) if every_months >= 12 else list(ends)


def _filtered_probabilities(
    returns: np.ndarray, fit: HMMFit
) -> np.ndarray:
    """Forward pass: P(state_t | returns up to and including t), for each t.

    This is the scaled forward algorithm. alpha_t is renormalised at every step,
    which is exactly what makes it a filtered probability rather than an
    unnormalised likelihood - and it means no information from day t+1 onwards
    can reach day t.
    """
    n, k = len(returns), HMM_N_STATES
    # Gaussian log-densities, written out rather than taken from the fitted
    # object, so it is obvious no backward pass is involved.
    z = (returns[:, None] - fit.means[None, :]) / fit.sds[None, :]
    log_b = -0.5 * z**2 - np.log(fit.sds)[None, :] - 0.5 * np.log(2 * np.pi)
    b = np.exp(log_b - log_b.max(axis=1, keepdims=True))   # scaled, cancels in the ratio

    alpha = np.zeros((n, k))
    a = fit.startprob * b[0]
    alpha[0] = a / a.sum()
    for t in range(1, n):
        a = (alpha[t - 1] @ fit.transmat) * b[t]
        total = a.sum()
        # If a day is wildly unlikely under both states, keep the previous belief
        # rather than dividing by zero.
        alpha[t] = alpha[t - 1] if total <= 0 or not np.isfinite(total) else a / total
    return alpha


def method_b(
    index_returns: pd.Series,
    seed: int = cfg.RANDOM_SEED,
    transmat_prior: float = None,
) -> tuple[pd.DataFrame, list[HMMFit]]:
    """Filtered high-volatility probability from a rolling-refit 2-state HMM.

    A day is labelled by the model fitted at the most recent refit date strictly
    before it, so no day is ever labelled by a model that has seen it. Days
    before the first usable fit are left as pd.NA - unlabelled, not low-vol.
    """
    r = index_returns.dropna().astype("float64")
    calendar = r.index
    fits: list[HMMFit] = []

    out = pd.DataFrame(index=calendar, columns=["model_fit_date"], dtype="object")
    out["p_high_vol"] = np.nan

    anchors = [d for d in refit_dates(calendar) if d <= calendar[-1]]
    if not anchors:
        raise ValueError("not enough history to fit the HMM even once")

    for i, fit_date in enumerate(anchors):
        train = r.loc[:fit_date]
        if len(train) < cfg.HMM_MIN_TRAIN_DAYS:
            continue              # not enough history to fit anything meaningful
        startprob, transmat, means, sds, converged, loglik = _fit_hmm(
            train.to_numpy(), seed, transmat_prior=transmat_prior
        )
        high_state = int(np.argmax(sds))  # the stormy state is the wider one
        fit = HMMFit(
            fit_date=fit_date, startprob=startprob, transmat=transmat, means=means,
            sds=sds, high_state=high_state, n_train=len(train), converged=bool(converged),
            loglik=loglik,
            transmat_prior=(cfg.HMM_TRANSMAT_PRIOR if transmat_prior is None
                            else transmat_prior),
        )
        fits.append(fit)

        # This model governs the days after its fit date, until the next refit.
        seg_end = anchors[i + 1] if i + 1 < len(anchors) else calendar[-1]
        segment = r.loc[fit_date:seg_end].index[1:]        # strictly after fit_date
        if len(segment) == 0:
            continue

        # Filter from the start of the sample up to the end of this segment using
        # this model, then keep only the segment's own days.
        upto = r.loc[:seg_end]
        alpha = _filtered_probabilities(upto.to_numpy(), fit)
        p_high = pd.Series(alpha[:, high_state], index=upto.index)
        out.loc[segment, "p_high_vol"] = p_high.reindex(segment).to_numpy()
        out.loc[segment, "model_fit_date"] = fit_date

    out["p_high_vol"] = out["p_high_vol"].astype("float64")
    flag = (out["p_high_vol"] > 0.5).astype("boolean")
    flag[out["p_high_vol"].isna()] = pd.NA     # no model yet means no label
    out["high_vol"] = flag
    return out.reindex(index_returns.index)[["p_high_vol", "high_vol", "model_fit_date"]], fits


# --------------------------------------------------------------------------
# Robustness: India VIX vs its expanding median
# --------------------------------------------------------------------------


def method_vix(vix: pd.Series) -> pd.DataFrame:
    """India VIX above its expanding median.

    VIX is missing on a handful of days; it is forward-filled, which only ever
    carries a past value forward.
    """
    v = vix.astype("float64").ffill()
    flag, med = expanding_median_flag(v)
    return pd.DataFrame({"india_vix": v, "expanding_median": med, "high_vol": flag})


# --------------------------------------------------------------------------
# summary statistics
# --------------------------------------------------------------------------


def regime_runs(flag: pd.Series) -> pd.DataFrame:
    """One row per unbroken stretch of the same label."""
    f = flag.dropna().astype(bool)
    if f.empty:
        return pd.DataFrame(columns=["start", "end", "high_vol", "length_days"])
    run_id = f.ne(f.shift()).cumsum()
    rows = []
    for _, grp in f.groupby(run_id):
        rows.append({"start": grp.index[0], "end": grp.index[-1],
                     "high_vol": bool(grp.iloc[0]), "length_days": len(grp)})
    return pd.DataFrame(rows)


def regime_summary(flag: pd.Series, name: str) -> dict:
    """Share of time high-vol, and how long regimes last."""
    f = flag.dropna().astype(bool)
    runs = regime_runs(f)
    hi = runs[runs["high_vol"]]
    lo = runs[~runs["high_vol"]]
    return {
        "method": name,
        "n_days": int(len(f)),
        "pct_days_high_vol": round(100 * float(f.mean()), 2),
        "n_high_vol_episodes": int(len(hi)),
        "n_low_vol_episodes": int(len(lo)),
        "mean_high_vol_length_days": round(float(hi["length_days"].mean()), 1) if len(hi) else np.nan,
        "mean_low_vol_length_days": round(float(lo["length_days"].mean()), 1) if len(lo) else np.nan,
        "median_high_vol_length_days": float(hi["length_days"].median()) if len(hi) else np.nan,
        "longest_high_vol_episode_days": int(hi["length_days"].max()) if len(hi) else 0,
        "mean_episode_length_days": round(float(runs["length_days"].mean()), 1),
    }


def agreement(a: pd.Series, b: pd.Series) -> dict:
    """How often two methods give the same label, on days where both have one."""
    joint = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if joint.empty:
        return {"n_common_days": 0, "pct_agreement": np.nan}
    same = joint["a"].astype(bool) == joint["b"].astype(bool)
    both_high = joint["a"].astype(bool) & joint["b"].astype(bool)
    return {
        "n_common_days": int(len(joint)),
        "pct_agreement": round(100 * float(same.mean()), 2),
        "pct_both_high": round(100 * float(both_high.mean()), 2),
        "pct_a_high_only": round(100 * float((joint["a"].astype(bool) & ~joint["b"].astype(bool)).mean()), 2),
        "pct_b_high_only": round(100 * float((~joint["a"].astype(bool) & joint["b"].astype(bool)).mean()), 2),
    }


# --------------------------------------------------------------------------
# build everything
# --------------------------------------------------------------------------


@dataclass
class Regimes:
    a: pd.DataFrame
    b: pd.DataFrame
    vix: pd.DataFrame
    fits: list[HMMFit]

    @property
    def labels(self) -> pd.DataFrame:
        """The three daily high-volatility flags, side by side."""
        return pd.DataFrame({
            "high_vol_a": self.a["high_vol"],
            "high_vol_b": self.b["high_vol"],
            "high_vol_vix": self.vix["high_vol"],
            "p_high_vol_b": self.b["p_high_vol"],
            "realised_vol": self.a["realised_vol"],
            "india_vix": self.vix["india_vix"],
        })

    def at(self, dates, method: str = "a") -> pd.Series:
        """Regime label on given dates - used to label each holding period by its start."""
        col = {"a": "high_vol_a", "b": "high_vol_b", "vix": "high_vol_vix"}[method]
        return self.labels[col].reindex(pd.DatetimeIndex(dates))


def build(
    index_returns: pd.Series,
    vix: pd.Series,
    seed: int = cfg.RANDOM_SEED,
    transmat_prior: float = None,
) -> Regimes:
    a = method_a(index_returns)
    b, fits = method_b(index_returns, seed=seed, transmat_prior=transmat_prior)
    v = method_vix(vix)
    return Regimes(a=a, b=b, vix=v, fits=fits)
