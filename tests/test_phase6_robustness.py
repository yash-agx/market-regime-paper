"""Phase 6: the lookback variants.

Three things are checked here:

  * the variant window builders reproduce the main-spec windows exactly at
    12-1 and 1 week, so the "main pair" row in Phase 6 is the same experiment as
    Phase 4 and not a near-copy of it;
  * the variants keep the properties that matter - the momentum skip month,
    and formation windows that end on or before the rebalance day - and
    truncating the data at a date leaves every signal, eligible set and holding
    formed up to that date unchanged (no look-ahead);
  * the small helpers (Holm, the crash ratio, the verdict bands) do what the
    pre-registration says.

A last test reads the saved Phase 6 output, when it exists, and checks that the
main pair reproduces the Phase 4 Version 2 numbers to the last digit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src import data as D
from src import strategies as S
from src import windows as W
from src import run_phase6_robustness as P6


@pytest.fixture
def calendar():
    return pd.bdate_range("2008-06-02", "2012-12-31")


def make_cleaned(prices: pd.DataFrame) -> D.CleanedData:
    eligible = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    nsei = pd.Series(100.0, index=prices.index)
    return D.CleanedData(
        prices=prices, returns=prices.pct_change(), eligible=eligible,
        nsei=nsei, nsei_returns=nsei.pct_change(),
        vix=pd.Series(15.0, index=prices.index),
        listing=pd.DataFrame(), flagged=pd.DataFrame(),
    )


@pytest.fixture
def noisy():
    """20 stocks of random walks, so every lookback gives a different sort."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2009-01-01", periods=700)
    rets = rng.normal(0.0004, 0.02, size=(len(idx), 20))
    prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx,
                          columns=[f"S{i:02d}" for i in range(20)])
    return make_cleaned(prices)


# --------------------------------------------------------------------------
# the main spec is reproduced exactly
# --------------------------------------------------------------------------


def test_12_1_momentum_windows_are_the_main_spec(calendar):
    assert P6.momentum_windows_lookback(calendar, 12) == W.momentum_windows(calendar)


def test_1_week_reversal_windows_are_the_main_spec(calendar):
    assert P6.reversal_windows_lookback(calendar, 1) == W.reversal_windows(calendar)


def test_main_pair_is_one_of_the_primary_pairs():
    assert P6.MAIN_PAIR in P6.PRIMARY_PAIRS
    assert P6.MOM_LOOKBACKS[P6.MAIN_PAIR[0]] == 12
    assert P6.REV_LOOKBACKS[P6.MAIN_PAIR[1]] == 1


def test_primary_pairs_change_one_lookback_at_a_time():
    for m, r in P6.PRIMARY_PAIRS:
        assert m == P6.MAIN_PAIR[0] or r == P6.MAIN_PAIR[1]
    assert len(set(P6.PRIMARY_PAIRS)) == 5
    assert len(P6.ALL_PAIRS) == 9


# --------------------------------------------------------------------------
# window shapes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("months", [3, 6, 12])
def test_momentum_variants_keep_the_skip_month(calendar, months):
    ends = W.month_end_dates(calendar)
    pos = {d: i for i, d in enumerate(ends)}
    for w in P6.momentum_windows_lookback(calendar, months):
        k = pos[w.rebalance_date]
        assert w.formation_end == ends[k - 1]          # month t is skipped
        assert w.formation_start == ends[k - months]
        assert w.holding_end == ends[k + 1]
        assert w.formation_end < w.rebalance_date


@pytest.mark.parametrize("weeks", [1, 2, 4])
def test_reversal_variants_end_on_the_rebalance_day(calendar, weeks):
    ends = W.week_end_dates(calendar)
    pos = {d: i for i, d in enumerate(ends)}
    for w in P6.reversal_windows_lookback(calendar, weeks):
        j = pos[w.rebalance_date]
        assert w.formation_end == w.rebalance_date
        assert w.formation_start == ends[j - weeks]
        assert w.holding_end == ends[j + 1]             # still held one week


def test_one_month_reversal_is_about_twenty_trading_days(calendar):
    wins = P6.reversal_windows_lookback(calendar, P6.REV_LOOKBACKS["1m"])
    spans = [len(calendar[(calendar > w.formation_start) & (calendar <= w.formation_end)])
             for w in wins]
    assert 18 <= np.median(spans) <= 21


def test_lookbacks_below_the_minimum_are_refused(calendar):
    with pytest.raises(ValueError):
        P6.momentum_windows_lookback(calendar, 1)
    with pytest.raises(ValueError):
        P6.reversal_windows_lookback(calendar, 0)


# --------------------------------------------------------------------------
# no look-ahead: truncate at a date, nothing formed before it changes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("months", [3, 6, 12])
def test_momentum_windows_do_not_change_when_the_future_is_cut(calendar, months):
    cut = pd.Timestamp("2011-06-30")
    full = P6.momentum_windows_lookback(calendar, months)
    trunc = P6.momentum_windows_lookback(calendar[calendar <= cut], months)
    full_by_date = {w.rebalance_date: w for w in full}
    assert trunc, "the truncated calendar should still have windows"
    for w in trunc:
        assert full_by_date[w.rebalance_date] == w


@pytest.mark.parametrize("weeks", [1, 2, 4])
def test_reversal_windows_do_not_change_when_the_future_is_cut(calendar, weeks):
    cut = pd.Timestamp("2011-06-30")
    # The cut is a Thursday, so the truncated calendar ends in a part-week and
    # the last window's holding end moves. That is fine - a holding return is
    # allowed to need the future. What must not move is the signal window.
    full = {w.rebalance_date: w for w in P6.reversal_windows_lookback(calendar, weeks)}
    for w in P6.reversal_windows_lookback(calendar[calendar <= cut], weeks):
        f = full[w.rebalance_date]
        assert (f.formation_start, f.formation_end) == (w.formation_start, w.formation_end)


@pytest.mark.parametrize("kind,lookback", [("mom", 3), ("mom", 6), ("rev", 2), ("rev", 4)])
def test_variant_holdings_do_not_change_when_the_future_is_cut(noisy, kind, lookback):
    """Signals, eligibility and the chosen holdings, formed up to the cut, are identical."""
    cut = noisy.prices.index[500]
    short = make_cleaned(noisy.prices.loc[:cut])

    def holdings(cleaned):
        cal = cleaned.prices.index
        if kind == "mom":
            wins, freq = P6.momentum_windows_lookback(cal, lookback), "monthly"
            lq, sq = "top", "bottom"
        else:
            wins, freq = P6.reversal_windows_lookback(cal, lookback), "weekly"
            lq, sq = "bottom", "top"
        res = S.run_strategy(cleaned, wins, lq, sq, "x", freq)
        return res.holdings

    full, trunc = holdings(noisy), holdings(short)
    assert len(trunc) > 0
    cols = ["rebalance_date", "ticker", "side", "signal"]
    merged = trunc[cols].merge(full[cols], on=["rebalance_date", "ticker", "side"],
                               how="left", suffixes=("_trunc", "_full"))
    assert merged["signal_full"].notna().all()
    np.testing.assert_allclose(merged["signal_trunc"], merged["signal_full"])


def test_lookbacks_actually_produce_different_portfolios(noisy):
    """If the variants sorted identically the whole phase would be empty."""
    cal = noisy.prices.index
    sets = []
    for months in (3, 6, 12):
        res = S.run_strategy(noisy, P6.momentum_windows_lookback(cal, months),
                             "top", "bottom", "x", "monthly")
        h = res.holdings
        sets.append(set(zip(h["rebalance_date"], h["ticker"], h["side"])))
    assert sets[0] != sets[1] != sets[2]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def test_holm_matches_a_hand_calculation():
    # sorted: 0.01*4=0.04, 0.02*3=0.06, 0.03*2=0.06 (monotone), 0.5*1=0.5
    out = P6.holm([0.03, 0.01, 0.5, 0.02])
    np.testing.assert_allclose(out, [0.06, 0.04, 0.5, 0.06])


def test_holm_leaves_nans_out_and_caps_at_one():
    out = P6.holm([np.nan, 0.4, 0.6])
    assert np.isnan(out[0])
    np.testing.assert_allclose(out[1:], [0.8, 0.8])


def test_crash_ratio():
    assert P6.crash_ratio(-2.5576, -1.2768) == pytest.approx(2.003, abs=1e-3)
    assert np.isnan(P6.crash_ratio(0.5, -1.0))          # not both losses
    assert np.isnan(P6.crash_ratio(np.nan, -1.0))


def test_verdict_bands_follow_the_preregistration():
    assert P6.insample_verdict(2.5, 0.006) == "agrees"
    assert P6.insample_verdict(1.0, 0.03) == "same sign, weaker"
    assert P6.insample_verdict(-0.2, 0.6).startswith("contradicts")
    assert P6.holdout_verdict(0.32).startswith("agrees")
    assert P6.holdout_verdict(0.01).startswith("differs")
    assert P6.crash_verdict(2.0) == "agrees"
    assert P6.crash_verdict(0.6).startswith("differs")


def test_regime_sharpes_split_on_the_flag():
    idx = pd.date_range("2010-01-31", periods=8, freq="ME")
    r = pd.Series([0.01, 0.02, 0.03, 0.01, -0.01, -0.02, -0.03, -0.01], index=idx)
    flag = pd.Series([False] * 4 + [True] * 4, index=idx)
    out = P6.regime_sharpes(r, flag, 12)
    assert out["n_low"] == 4 and out["n_high"] == 4
    assert out["sharpe_low"] > 0 > out["sharpe_high"]


# --------------------------------------------------------------------------
# the saved run reproduces Phase 4
# --------------------------------------------------------------------------


def test_saved_main_pair_reproduces_phase4():
    path = cfg.RESULTS_DIR / "phase6_reproduction_check.csv"
    if not path.exists():
        pytest.skip("Phase 6 has not been run yet")
    rep = pd.read_csv(path)
    assert len(rep) == 2 * 3 * 2 * 3
    assert rep["identical"].all(), rep[~rep["identical"]]
