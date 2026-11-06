"""Phase 4: Sharpe ratios, excess returns, the block bootstrap and Newey-West.

Most of these run on constructed series where the right answer is known in
advance, so a wrong answer points at the code rather than at the market.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src import evaluation as E
from src import windows as W


# --------------------------------------------------------------------------
# Sharpe and excess returns
# --------------------------------------------------------------------------


def test_sharpe_annualises_by_the_square_root_of_frequency():
    r = pd.Series([0.01, 0.02, -0.01, 0.03, 0.00, 0.015])
    expected = r.mean() / r.std(ddof=1) * np.sqrt(12)
    assert E.sharpe(r, 12) == pytest.approx(expected)


def test_sharpe_of_a_constant_series_is_undefined_not_infinite():
    assert np.isnan(E.sharpe(pd.Series([0.01] * 10), 12))


def test_sharpe_needs_at_least_two_observations():
    assert np.isnan(E.sharpe(pd.Series([0.01]), 12))


def test_long_short_returns_are_not_reduced_by_the_risk_free_rate():
    """A zero-investment book has no capital tied up, so nothing is subtracted."""
    r = pd.Series([0.01, 0.02], index=pd.to_datetime(["2015-01-30", "2015-02-27"]))
    rf = pd.Series([0.005, 0.005], index=r.index)
    out = E.excess_returns(r, rf, zero_investment=True)
    pd.testing.assert_series_equal(out, r)


def test_long_only_returns_do_have_the_risk_free_rate_subtracted():
    r = pd.Series([0.01, 0.02], index=pd.to_datetime(["2015-01-30", "2015-02-27"]))
    rf = pd.Series([0.005, 0.005], index=r.index)
    out = E.excess_returns(r, rf, zero_investment=False)
    assert out.tolist() == pytest.approx([0.005, 0.015])


def test_period_riskfree_compounds_only_the_days_actually_held():
    cal = pd.bdate_range("2015-01-01", periods=20)
    rf_daily = pd.Series(0.0001, index=cal)
    win = W.Window(rebalance_date=cal[2], formation_start=cal[0],
                   formation_end=cal[2], holding_end=cal[7])
    out = E.period_riskfree(rf_daily, [win])
    # days strictly after the rebalance, through the end: cal[3]..cal[7] = 5 days
    assert out.iloc[0] == pytest.approx(1.0001 ** 5 - 1)


# --------------------------------------------------------------------------
# the stationary bootstrap
# --------------------------------------------------------------------------


def test_bootstrap_indices_are_in_range_and_right_shaped():
    rng = np.random.default_rng(0)
    idx = E.stationary_bootstrap_indices(50, 5, 200, rng)
    assert idx.shape == (200, 50)
    assert idx.min() >= 0 and idx.max() < 50


def test_bootstrap_preserves_blocks_of_consecutive_observations():
    """The whole point: neighbouring draws should usually be neighbours in time."""
    rng = np.random.default_rng(1)
    idx = E.stationary_bootstrap_indices(500, 20, 400, rng)
    steps = np.diff(idx, axis=1) % 500
    share_consecutive = float((steps == 1).mean())
    # with a mean block of 20, about 1 - 1/20 = 95% of steps continue a block
    assert 0.88 < share_consecutive < 0.99


def test_a_mean_block_of_one_is_an_ordinary_iid_bootstrap():
    rng = np.random.default_rng(2)
    idx = E.stationary_bootstrap_indices(300, 1, 300, rng)
    steps = np.diff(idx, axis=1) % 300
    assert float((steps == 1).mean()) < 0.05


def test_bootstrap_is_reproducible_with_a_fixed_seed():
    a = E.stationary_bootstrap_indices(40, 5, 50, np.random.default_rng(7))
    b = E.stationary_bootstrap_indices(40, 5, 50, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_default_block_follows_the_cube_root_rule():
    assert E.default_block(120) == 5
    assert E.default_block(521) == 8
    assert E.default_block(1) == 2       # floored


# --------------------------------------------------------------------------
# the hypothesis statistics
# --------------------------------------------------------------------------


@pytest.fixture
def two_regime_series():
    """Low-vol periods earn a steady 1%, high-vol periods lose noisily."""
    rng = np.random.default_rng(42)
    n = 240
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    high = pd.Series(False, index=idx)
    high.iloc[::4] = True                       # a quarter of periods are stormy
    r = pd.Series(np.where(high, -0.02, 0.01), index=idx)
    r += rng.normal(0, 0.005, n)
    return r, high.astype("boolean")


def test_h1_detects_momentum_doing_better_in_calm_markets(two_regime_series):
    r, high = two_regime_series
    res = E.bootstrap_sharpe_gap(r, high, 12, "H1", favour="low", n_boot=400)
    assert res.observed > 0                     # low-vol Sharpe is the higher one
    assert res.p_one_sided < 0.05
    assert res.ci_low > 0


def test_favour_high_flips_the_sign(two_regime_series):
    r, high = two_regime_series
    low = E.bootstrap_sharpe_gap(r, high, 12, "x", favour="low", n_boot=200)
    hi = E.bootstrap_sharpe_gap(r, high, 12, "x", favour="high", n_boot=200)
    assert low.observed == pytest.approx(-hi.observed)


def test_a_series_with_no_regime_effect_is_not_significant():
    rng = np.random.default_rng(3)
    n = 240
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    r = pd.Series(rng.normal(0.01, 0.04, n), index=idx)
    high = pd.Series(rng.random(n) < 0.25, index=idx).astype("boolean")
    res = E.bootstrap_sharpe_gap(r, high, 12, "H1", favour="low", n_boot=600)
    assert res.p_one_sided > 0.05
    assert res.ci_low < 0 < res.ci_high


def test_bootstrap_results_are_reproducible(two_regime_series):
    r, high = two_regime_series
    a = E.bootstrap_sharpe_gap(r, high, 12, "H1", favour="low", n_boot=300)
    b = E.bootstrap_sharpe_gap(r, high, 12, "H1", favour="low", n_boot=300)
    assert a.observed == b.observed
    assert a.p_one_sided == b.p_one_sided
    assert a.ci_low == b.ci_low


def test_h3_equals_h1_plus_h2_on_the_observed_statistic():
    """H3 is algebraically the sum of the other two. If it is not, something is wrong."""
    rng = np.random.default_rng(11)
    midx = pd.date_range("2005-01-31", periods=120, freq="ME")
    widx = pd.date_range("2005-01-07", periods=521, freq="W-FRI")
    mhigh = pd.Series(rng.random(120) < 0.25, index=midx).astype("boolean")
    whigh = pd.Series(rng.random(521) < 0.25, index=widx).astype("boolean")
    mr = pd.Series(rng.normal(0.01, 0.05, 120), index=midx)
    wr = pd.Series(rng.normal(0.002, 0.02, 521), index=widx)

    h1 = E.bootstrap_sharpe_gap(mr, mhigh, cfg.MONTHS_PER_YEAR, "H1",
                                favour="low", n_boot=10)
    h2 = E.bootstrap_sharpe_gap(wr, whigh, cfg.WEEKS_PER_YEAR, "H2",
                                favour="high", n_boot=10)
    h3 = E.bootstrap_difference_in_differences(mr, mhigh, wr, whigh, n_boot=10)
    assert h3.observed == pytest.approx(h1.observed + h2.observed, abs=1e-9)


def test_a_regime_with_too_few_periods_gives_no_answer():
    """Method B's in-sample case: one high-vol period cannot support a Sharpe."""
    idx = pd.date_range("2010-01-31", periods=120, freq="ME")
    r = pd.Series(np.linspace(0.01, 0.02, 120), index=idx)
    high = pd.Series(False, index=idx).astype("boolean")
    high.iloc[5] = True                          # exactly one high-vol period
    res = E.bootstrap_sharpe_gap(r, high, 12, "H1", favour="low", n_boot=50)
    assert np.isnan(res.observed)
    assert res.n_valid == 0


# --------------------------------------------------------------------------
# regime tables and Newey-West
# --------------------------------------------------------------------------


def test_regime_table_splits_every_period_exactly_once(two_regime_series):
    r, high = two_regime_series
    tab = E.regime_table(r, high, 12).set_index("regime")
    assert tab.loc["high-vol", "n_periods"] + tab.loc["low-vol", "n_periods"] == len(r)
    assert tab.loc["all", "n_periods"] == len(r)


def test_newey_west_recovers_a_known_mean_difference():
    idx = pd.date_range("2000-01-31", periods=300, freq="ME")
    high = pd.Series(np.tile([True, False, False, False], 75), index=idx)
    r = pd.Series(np.where(high, 0.03, 0.01), index=idx)
    rng = np.random.default_rng(5)
    r += rng.normal(0, 0.002, len(r))
    out = E.newey_west_dummy(r, high.astype("boolean"), "test")
    assert out["mean_low_vol_pct"] == pytest.approx(1.0, abs=0.05)
    assert out["diff_high_minus_low_pct"] == pytest.approx(2.0, abs=0.05)
    assert out["p_value_two_sided"] < 0.001


def test_newey_west_lag_rule():
    assert E.newey_west_lags(120) == 4
    assert E.newey_west_lags(521) == 5


def test_newey_west_declines_to_answer_with_one_regime():
    idx = pd.date_range("2000-01-31", periods=50, freq="ME")
    r = pd.Series(0.01, index=idx)
    high = pd.Series(False, index=idx).astype("boolean")
    out = E.newey_west_dummy(r, high, "test")
    assert "note" in out
