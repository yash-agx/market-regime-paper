"""Demerger handling: no ex-date return may ever reach a signal raw.

A demerger cuts the parent's share price without costing the holder anything,
and Yahoo does not adjust for it. Left alone, Adani Enterprises' 2015 spin-off
reads as a 38.8% loss, and momentum duly shorted it for twelve months. These
tests pin down that such a return cannot reach a signal unexamined - either the
history is rescaled so the ex-date return becomes zero, or the return is
blanked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src import corporate_actions as CA
from src import data as D
from src import strategies as S
from src import windows as W


# --------------------------------------------------------------------------
# classifying corporate actions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("subject,expected", [
    ("Demerger", True),
    ("Scheme Of Arrangement", True),
    ("Scheme Of Arrangement - Bonus Debentures 6:1", True),
    ("Spin Off", True),
    ("Annual General Meeting / Dividend Rs.2/- Per Share / Scheme Of Arrangement", True),
    ("Dividend - Rs 1.40/- Per Share", False),
    ("Bonus 1:1", False),
    ("Face Value Split From Rs 10 To Rs 2", False),
    ("Annual General Meeting", False),
])
def test_only_price_splitting_actions_are_selected(subject, expected):
    assert CA.is_price_splitting(subject) is expected


# --------------------------------------------------------------------------
# the adjustment factor
# --------------------------------------------------------------------------


def _series(values, start="2015-05-25"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)))


def test_factor_is_the_ex_date_ratio():
    s = _series([100.0, 100.0, 60.0, 61.0])
    factor, how = CA.adjustment_factor(s, s.index[2])
    assert factor == pytest.approx(0.60)
    assert "previous close" in how


def test_factor_is_none_when_the_ex_date_cannot_be_measured():
    s = _series([100.0, 101.0, 102.0])
    factor, how = CA.adjustment_factor(s, pd.Timestamp("2020-01-01"))
    assert factor is None and how


# --------------------------------------------------------------------------
# applying the adjustment
# --------------------------------------------------------------------------


@pytest.fixture
def demerger_frame():
    idx = pd.bdate_range("2015-05-25", periods=8)
    prices = pd.DataFrame({"AAA.NS": [100.0, 102.0, 104.0, 62.0, 63.0, 64.0, 65.0, 66.0],
                           "BBB.NS": np.linspace(50, 57, 8)}, index=idx)
    events = pd.DataFrame([{"symbol": "AAA", "ex_date": idx[3].date(),
                            "subject": "Demerger", "source_url": "http://x"}])
    return prices, events, idx


def test_adjustment_makes_the_ex_date_return_zero(demerger_frame):
    prices, events, idx = demerger_frame
    adj, log, _ = CA.apply_demerger_adjustments(prices, events)
    r = adj["AAA.NS"].pct_change()
    assert r.loc[idx[3]] == pytest.approx(0.0, abs=1e-12)
    assert bool(log["applied"].iloc[0])


def test_adjustment_leaves_every_other_return_untouched(demerger_frame):
    prices, events, _ = demerger_frame
    adj, _, _ = CA.apply_demerger_adjustments(prices, events)
    before = prices["AAA.NS"].pct_change()
    after = adj["AAA.NS"].pct_change()
    other = after.index != after.index[3]
    pd.testing.assert_series_equal(before[other], after[other])


def test_other_tickers_are_not_touched(demerger_frame):
    prices, events, _ = demerger_frame
    adj, _, _ = CA.apply_demerger_adjustments(prices, events)
    pd.testing.assert_series_equal(prices["BBB.NS"], adj["BBB.NS"])


def test_a_price_rise_on_the_ex_date_is_never_scaled():
    """A demerger always cuts the parent's price - value leaves the company.
    Scaling on an ex-date where the price ROSE would erase a real return and
    invent a fake one. Adani Enterprises rose 15% and 21% on its two 2018
    scheme ex-dates."""
    idx = pd.bdate_range("2018-04-01", periods=6)
    prices = pd.DataFrame({"AAA.NS": [100.0, 100.0, 115.0, 116.0, 117.0, 118.0]},
                          index=idx)
    events = pd.DataFrame([{"symbol": "AAA", "ex_date": idx[2].date(),
                            "subject": "Scheme Of Arrangement", "source_url": "http://x"}])
    adj, log, neutralise = CA.apply_demerger_adjustments(prices, events)
    assert not bool(log["applied"].iloc[0])
    pd.testing.assert_series_equal(prices["AAA.NS"], adj["AAA.NS"])
    # not scaled, but still not trusted
    assert list(neutralise["ticker"]) == ["AAA.NS"]


def test_an_unmeasurable_event_is_neutralised_not_ignored():
    idx = pd.bdate_range("2015-01-01", periods=5)
    prices = pd.DataFrame({"AAA.NS": np.linspace(10, 14, 5)}, index=idx)
    events = pd.DataFrame([{"symbol": "AAA", "ex_date": pd.Timestamp("2030-01-01").date(),
                            "subject": "Demerger", "source_url": "http://x"}])
    _, log, neutralise = CA.apply_demerger_adjustments(prices, events)
    assert not bool(log["applied"].iloc[0])
    assert len(neutralise) == 1


def test_every_event_is_either_scaled_or_neutralised(demerger_frame):
    """No event may be silently dropped - that is how the artifacts survived."""
    prices, events, idx = demerger_frame
    extra = pd.DataFrame([{"symbol": "ZZZ", "ex_date": idx[3].date(),
                           "subject": "Demerger", "source_url": "http://x"}])
    all_events = pd.concat([events, extra], ignore_index=True)
    _, log, neutralise = CA.apply_demerger_adjustments(prices, all_events)
    assert len(log) == len(all_events)
    assert int(log["applied"].sum()) + len(neutralise) == len(all_events)


# --------------------------------------------------------------------------
# the real data: no demerger ex-date return reaches a signal raw
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cleaned():
    if not (cfg.PROC_DIR / D.PROC_FILES["prices"]).exists():
        pytest.skip("run `python -m src.run_phase1` first")
    return D.load_processed()


@pytest.fixture(scope="module")
def events():
    ev = D.load_corporate_actions()
    if ev.empty:
        pytest.skip("run `python -m src.run_corporate_actions` first")
    return ev


def test_no_demerger_ex_date_has_a_raw_return_in_the_data(cleaned, events):
    """Every ex-date return must be either ~0 (history rescaled) or NaN."""
    offenders = []
    for _, ev in events.iterrows():
        ticker = f"{ev['symbol']}.NS"
        ex = pd.Timestamp(ev["ex_date"])
        if ticker not in cleaned.returns.columns or ex not in cleaned.returns.index:
            continue
        val = cleaned.returns.at[ex, ticker]
        if pd.notna(val) and abs(val) > 1e-9:
            offenders.append((ticker, str(ev["ex_date"]), float(val)))
    assert not offenders, f"raw ex-date returns still present: {offenders}"


def test_the_known_artifacts_are_gone(cleaned):
    """Adani Enterprises 2015 and Tata Motors 2025, the two that reached the
    portfolios before this was fixed."""
    for ticker, date in (("ADANIENT.NS", "2015-06-03"), ("TMPV.NS", "2025-10-14")):
        val = cleaned.returns.at[pd.Timestamp(date), ticker]
        assert pd.isna(val) or abs(val) < 1e-9, f"{ticker} {date} is still {val}"


def test_no_momentum_signal_spans_an_unneutralised_ex_date(cleaned, events):
    """The signal is a price ratio across the window, so a scaled history is
    safe - but an ex-date left raw inside the window would corrupt it."""
    runs = D.flat_run_length(cleaned.prices)
    ex_by_ticker = {}
    for _, ev in events.iterrows():
        ex_by_ticker.setdefault(f"{ev['symbol']}.NS", []).append(
            pd.Timestamp(ev["ex_date"]))

    bad = []
    for win in W.momentum_windows(cleaned.prices.index):
        if not (pd.Timestamp(cfg.STUDY_START) <= win.rebalance_date
                <= pd.Timestamp(cfg.INSAMPLE_END)):
            continue
        allowed = D.eligible_at(cleaned, win.rebalance_date, win.formation_start,
                                win.formation_end, runs=runs)
        for ticker in allowed:
            for ex in ex_by_ticker.get(ticker, []):
                if win.formation_start <= ex <= win.formation_end:
                    val = cleaned.returns.at[ex, ticker] if ex in cleaned.returns.index else np.nan
                    if pd.notna(val) and abs(val) > 1e-9:
                        bad.append((ticker, str(ex.date()), str(win.rebalance_date.date())))
    assert not bad, f"signals spanning a raw ex-date return: {bad[:10]}"
