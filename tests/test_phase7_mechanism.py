"""Phase 7: the market-beta mechanism test.

Checked here:

  * regression A on synthetic data with a known regime-dependent beta gives
    that beta back, and regression B gives back a known bear-market up-beta;
  * the hedged series has mean a0 in calm periods and a0 + a1 in high-vol
    periods, as the pre-registration says;
  * B_t and the 126-day regime label use no data after their date: truncating
    the data at a date leaves every value up to that date unchanged;
  * the verdict bands are the pre-registered ones;
  * when the saved Phase 7 output exists, the raw H3 reproduces Phase 6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src import run_phase7_mechanism as P7


# --------------------------------------------------------------------------
# synthetic regressions
# --------------------------------------------------------------------------


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(11)
    n = 4000
    idx = pd.date_range("2000-01-31", periods=n, freq="D")
    hv = (rng.random(n) < 0.3).astype(float)
    m = rng.normal(0.005, 0.05, n)
    a0, a1, b0, b1 = 0.004, -0.010, 0.20, -0.60
    y = a0 + a1 * hv + b0 * m + b1 * hv * m + rng.normal(0, 0.01, n)
    return (pd.Series(y, idx), pd.Series(m, idx), pd.Series(hv.astype(bool), idx),
            dict(a0=a0, a1=a1, b0=b0, b1=b1))


def test_regression_a_recovers_known_beta(synthetic):
    y, m, hv, true = synthetic
    out = P7.regression_a(y, m, hv)
    assert out["b0"] == pytest.approx(true["b0"], abs=0.02)
    assert out["b1"] == pytest.approx(true["b1"], abs=0.03)
    assert out["beta_high"] == pytest.approx(true["b0"] + true["b1"], abs=0.03)
    assert out["a1"] == pytest.approx(true["a1"], abs=0.002)
    assert out["n"] == len(y)


def test_regression_a_raw_difference_is_the_mean_gap(synthetic):
    y, m, hv, _ = synthetic
    out = P7.regression_a(y, m, hv)
    assert out["raw_diff"] == pytest.approx(y[hv].mean() - y[~hv].mean(), abs=1e-12)


def test_hedged_series_means_are_a0_and_a0_plus_a1(synthetic):
    y, m, hv, _ = synthetic
    out = P7.regression_a(y, m, hv)
    h = P7.hedge(y.to_numpy(), hv.to_numpy(float), m.to_numpy())
    hvb = hv.to_numpy()
    assert h[~hvb].mean() == pytest.approx(out["a0"], abs=1e-12)
    assert h[hvb].mean() == pytest.approx(out["a0"] + out["a1"], abs=1e-12)


def test_hedging_removes_a_pure_beta_regime_effect():
    """If the regime effect is only beta times a market that rises in high vol,
    the raw high-vol gap is large and a1 is close to zero."""
    rng = np.random.default_rng(3)
    n = 5000
    idx = pd.date_range("2000-01-31", periods=n, freq="D")
    hv = rng.random(n) < 0.3
    m = np.where(hv, 0.03, 0.0) + rng.normal(0, 0.04, n)   # rebounds in high vol
    y = -0.8 * m + rng.normal(0, 0.005, n)                 # momentum-like: short beta
    out = P7.regression_a(pd.Series(y, idx), pd.Series(m, idx), pd.Series(hv, idx))
    assert out["raw_diff"] < -0.02
    assert abs(out["a1"]) < 0.002


def test_regression_b_recovers_bear_up_beta():
    rng = np.random.default_rng(5)
    n = 4000
    idx = pd.date_range("2000-01-31", periods=n, freq="D")
    b = (rng.random(n) < 0.25).astype(float)
    m = rng.normal(0.0, 0.05, n)
    u = (m > 0).astype(float)
    y = 0.002 + 0.001 * b + 0.1 * m - 0.3 * b * m - 0.5 * b * u * m + rng.normal(0, 0.01, n)
    out = P7.regression_b(pd.Series(y, idx), pd.Series(m, idx), pd.Series(b, idx), "monthly")
    assert out["bBU"] == pytest.approx(-0.5, abs=0.05)
    assert out["bB"] == pytest.approx(-0.3, abs=0.05)
    assert out["b0"] == pytest.approx(0.1, abs=0.02)
    assert out["note"] == ""


def test_regression_b_power_rule_flags_few_bear_months():
    idx = pd.date_range("2000-01-31", periods=120, freq="ME")
    rng = np.random.default_rng(1)
    m = pd.Series(rng.normal(0, 0.05, 120), idx)
    y = pd.Series(rng.normal(0, 0.03, 120), idx)
    b = pd.Series(0.0, idx)
    b.iloc[10:18] = 1.0                                     # 8 bear months < 12
    out = P7.regression_b(y, m, b, "monthly")
    assert out["n_bear"] == 8
    assert out["note"] == "too few bear periods to interpret"


# --------------------------------------------------------------------------
# no look-ahead
# --------------------------------------------------------------------------


@pytest.fixture
def index_level():
    rng = np.random.default_rng(9)
    idx = pd.bdate_range("2009-01-01", "2014-12-31")
    return pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.015, len(idx))), idx)


def test_bear_state_is_nan_without_24_months_of_history(index_level):
    b = P7.bear_state(index_level, pd.DatetimeIndex(["2010-12-31", "2011-01-31"]))
    assert np.isnan(b.iloc[0])
    assert np.isfinite(b.iloc[1])


def test_bear_state_matches_its_definition(index_level):
    d = pd.Timestamp("2013-06-28")
    past = index_level.loc[:d - pd.DateOffset(months=24)].iloc[-1]
    now = index_level.loc[:d].iloc[-1]
    assert P7.bear_state(index_level, [d]).iloc[0] == float(now / past - 1 < 0)


@pytest.mark.parametrize("cut", ["2012-03-30", "2013-09-13", "2014-06-30"])
def test_bear_state_no_lookahead(index_level, cut):
    dates = pd.bdate_range("2011-01-03", "2014-12-31", freq="BME")
    full = P7.bear_state(index_level, dates)
    trunc = P7.bear_state(index_level.loc[:cut], dates[dates <= cut])
    pd.testing.assert_series_equal(full.loc[:cut], trunc)


def test_bear_state_ignores_a_later_crash(index_level):
    """Changing every price after a date must not move B_t on or before it."""
    cut = pd.Timestamp("2012-12-31")
    shocked = index_level.copy()
    shocked.loc[shocked.index > cut] *= 0.1
    dates = pd.bdate_range("2011-01-03", cut, freq="BME")
    pd.testing.assert_series_equal(P7.bear_state(index_level, dates),
                                   P7.bear_state(shocked, dates))


@pytest.mark.parametrize("cut", ["2011-06-30", "2013-01-31"])
def test_slow_regime_no_lookahead(index_level, cut):
    r = index_level.pct_change()
    full = P7.slow_regime(r)
    trunc = P7.slow_regime(r.loc[:cut])
    pd.testing.assert_series_equal(full.loc[:cut], trunc)


def test_slow_regime_needs_126_days(index_level):
    flag = P7.slow_regime(index_level.pct_change())
    assert flag.iloc[:126].isna().all()       # first return is NaN, so 126 returns end at day 126
    assert flag.iloc[126:].notna().all()


# --------------------------------------------------------------------------
# verdict bands
# --------------------------------------------------------------------------


@pytest.mark.parametrize("raw,adj,lo,expected", [
    (2.0, 1.8, 0.2, "survives"),
    (2.0, 1.8, -0.1, "partly explained"),     # f >= 2/3 but interval includes zero
    (2.0, 1.0, 0.1, "partly explained"),
    (2.0, 0.5, -1.0, "mostly explained"),
    (2.0, -0.2, -2.0, "mostly explained"),
    (-0.5, 0.3, -1.0, "raw H3 not positive; not interpreted"),
])
def test_verdict_bands(raw, adj, lo, expected):
    assert P7.verdict(raw, adj, lo) == expected


# --------------------------------------------------------------------------
# reproduction of the published H3
# --------------------------------------------------------------------------


def test_saved_raw_h3_reproduces_phase6():
    path = cfg.RESULTS_DIR / "phase7_reproduction_check.csv"
    if not path.exists():
        pytest.skip("Phase 7 has not been run yet")
    rep = pd.read_csv(path)
    assert len(rep) == 20
    assert rep["identical"].all()
