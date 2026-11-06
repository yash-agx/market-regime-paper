"""Look-ahead tests for Phase 2: the regime labels.

Regimes are the easiest place in this project to leak the future, in two
specific ways:

  * using a full-sample median as the high/low threshold, so the label for 2011
    already knows how volatile 2020 was;
  * taking an HMM's smoothed probabilities, which are computed with a backward
    pass over the whole sample, instead of filtered ones.

Both have a dedicated test below, on top of the general truncation checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src import data as D
from src import regimes as R

CUTS = ["2014-12-31", "2019-06-28", "2019-12-31"]


@pytest.fixture(scope="module")
def nsei():
    if not (cfg.PROC_DIR / D.PROC_FILES["nsei"]).exists():
        pytest.skip("run `python -m src.run_phase1` first")
    return D.load_processed()


# --------------------------------------------------------------------------
# Method A
# --------------------------------------------------------------------------


def test_realised_vol_uses_only_trailing_days():
    r = pd.Series(np.arange(30.0) / 100, index=pd.bdate_range("2010-01-04", periods=30))
    vol = R.realised_vol(r, window=21)
    assert vol.iloc[:20].isna().all()          # no reading before 21 days exist
    expected = r.iloc[:21].std() * np.sqrt(cfg.TRADING_DAYS)
    assert vol.iloc[20] == pytest.approx(expected)


def test_expanding_median_is_not_the_full_sample_median():
    """The whole point of the expanding median: early days must not see later ones."""
    s = pd.Series([1.0, 2.0, 3.0, 100.0, 200.0],
                  index=pd.bdate_range("2010-01-04", periods=5))
    flag, med = R.expanding_median_flag(s)
    assert list(med) == [1.0, 1.5, 2.0, 2.5, 3.0]
    # 3.0 beats the median of what came before it, but is far below the full-sample median
    assert bool(flag.iloc[2])
    assert s.median() == 3.0        # a full-sample rule would have called it NOT high


def test_days_without_a_volatility_reading_are_unlabelled_not_low():
    s = pd.Series([np.nan, np.nan, 1.0, 2.0], index=pd.bdate_range("2010-01-04", periods=4))
    flag, _ = R.expanding_median_flag(s)
    assert flag.isna().iloc[:2].all()
    assert flag.dtype == "boolean"


@pytest.mark.parametrize("cut", CUTS)
def test_method_a_unchanged_by_truncation(nsei, cut):
    cut = pd.Timestamp(cut)
    full = R.method_a(nsei.nsei_returns)
    part = R.method_a(nsei.nsei_returns.loc[:cut])
    pd.testing.assert_frame_equal(part, full.loc[:cut])


# --------------------------------------------------------------------------
# Method B
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cut", CUTS)
def test_method_b_unchanged_by_truncation(nsei, cut):
    """Refits, segment governance and the forward pass must all be causal."""
    cut = pd.Timestamp(cut)
    full, _ = R.method_b(nsei.nsei_returns)
    part, _ = R.method_b(nsei.nsei_returns.loc[:cut])
    pd.testing.assert_series_equal(part["p_high_vol"], full["p_high_vol"].loc[:cut])
    pd.testing.assert_series_equal(part["high_vol"], full["high_vol"].loc[:cut])


def test_no_day_is_labelled_by_a_model_that_has_seen_it(nsei):
    b, _ = R.method_b(nsei.nsei_returns)
    labelled = b.dropna(subset=["p_high_vol"])
    fit_dates = pd.DatetimeIndex(labelled["model_fit_date"].to_numpy())
    assert (fit_dates < labelled.index).all(), "a model labelled a day it was trained on"


def test_filtered_is_not_smoothed(nsei):
    """Guard against someone swapping in hmmlearn's predict_proba.

    The two are correlated but must not be equal: the smoothed probability for
    day t uses days after t, which is exactly what this project forbids.
    """
    from hmmlearn.hmm import GaussianHMM

    train = nsei.nsei_returns.dropna().loc[:"2019-12-31"]
    X = train.to_numpy().reshape(-1, 1)
    sp, tm, mu, sd, conv, ll = R._fit_hmm(train.to_numpy(), cfg.RANDOM_SEED)
    hi = int(np.argmax(sd))
    fit = R.HMMFit(pd.Timestamp("2019-12-31"), sp, tm, mu, sd, hi, len(train), conv, ll)
    filtered = R._filtered_probabilities(train.to_numpy(), fit)[:, hi]

    model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000,
                        tol=1e-6, random_state=cfg.RANDOM_SEED)
    model.fit(X)
    smoothed = model.predict_proba(X)[:, hi]

    assert not np.allclose(filtered, smoothed), "filtered probabilities look smoothed"
    assert np.corrcoef(filtered, smoothed)[0, 1] > 0.5, "these should still be related"


def test_filtered_probability_is_a_probability(nsei):
    b, _ = R.method_b(nsei.nsei_returns)
    p = b["p_high_vol"].dropna()
    assert (p >= 0).all() and (p <= 1).all()


def test_forward_pass_matches_a_hand_computed_two_step_example():
    """Pin the forward recursion down on numbers that can be checked by hand."""
    fit = R.HMMFit(
        fit_date=pd.Timestamp("2010-01-01"),
        startprob=np.array([0.5, 0.5]),
        transmat=np.array([[0.9, 0.1], [0.2, 0.8]]),
        means=np.array([0.0, 0.0]),
        sds=np.array([0.01, 0.04]),
        high_state=1, n_train=0, converged=True,
    )
    x = np.array([0.0, 0.10])        # a calm day, then a very large move
    alpha = R._filtered_probabilities(x, fit)

    from scipy.stats import norm
    b0 = norm.pdf(x[0], 0, fit.sds)
    a0 = fit.startprob * b0
    a0 = a0 / a0.sum()
    b1 = norm.pdf(x[1], 0, fit.sds)
    a1 = (a0 @ fit.transmat) * b1
    a1 = a1 / a1.sum()

    assert alpha[0] == pytest.approx(a0)
    assert alpha[1] == pytest.approx(a1)
    assert alpha[0, 1] < 0.5 < alpha[1, 1]      # the big move flips the belief


def test_hmm_is_reproducible(nsei):
    a, _ = R.method_b(nsei.nsei_returns, seed=cfg.RANDOM_SEED)
    b, _ = R.method_b(nsei.nsei_returns, seed=cfg.RANDOM_SEED)
    pd.testing.assert_series_equal(a["p_high_vol"], b["p_high_vol"])


# --------------------------------------------------------------------------
# VIX and summaries
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cut", CUTS)
def test_vix_method_unchanged_by_truncation(nsei, cut):
    cut = pd.Timestamp(cut)
    full = R.method_vix(nsei.vix)
    part = R.method_vix(nsei.vix.loc[:cut])
    pd.testing.assert_frame_equal(part, full.loc[:cut])


def test_vix_is_only_ever_filled_forward(nsei):
    """A back-fill would put tomorrow's VIX on a day it was not known."""
    v = R.method_vix(nsei.vix)["india_vix"]
    raw = nsei.vix
    for date in raw.index[raw.isna()][:20]:
        prior = raw.loc[:date].dropna()
        if prior.empty:
            continue
        assert v.loc[date] == prior.iloc[-1]


def test_regime_runs_partition_the_sample():
    flag = pd.Series([True, True, False, False, False, True],
                     index=pd.bdate_range("2010-01-04", periods=6), dtype="boolean")
    runs = R.regime_runs(flag)
    assert list(runs["length_days"]) == [2, 3, 1]
    assert list(runs["high_vol"]) == [True, False, True]
    assert runs["length_days"].sum() == len(flag)


def test_regime_summary_ignores_unlabelled_days():
    flag = pd.Series([pd.NA, pd.NA, True, False], dtype="boolean",
                     index=pd.bdate_range("2010-01-04", periods=4))
    s = R.regime_summary(flag, "test")
    assert s["n_days"] == 2
    assert s["pct_days_high_vol"] == 50.0
