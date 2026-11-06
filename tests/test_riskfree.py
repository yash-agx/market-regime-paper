"""The risk-free rate loader: causal forward-fill, percent-not-fraction, clear failure."""

from __future__ import annotations

import pandas as pd
import pytest

from src import config as cfg
from src import data as D


@pytest.fixture
def tbill_file(tmp_path, monkeypatch):
    """A tiny fake weekly T-bill file, so the test does not need the real one."""
    path = tmp_path / "tbill_91d.csv"
    path.write_text(
        "date,yield_pct\n"
        "2009-12-30,4.00\n"
        "2010-01-06,5.00\n"
        "2010-01-13,6.00\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cfg, "TBILL_CSV", path)
    return path


def test_yield_is_converted_to_a_daily_rate(tbill_file):
    cal = pd.DatetimeIndex(["2009-12-31", "2010-01-06", "2010-01-07"])
    rf = D.load_riskfree(cal)
    expected = (1 + 0.04) ** (1 / cfg.TRADING_DAYS) - 1
    assert rf.loc["2009-12-31"] == pytest.approx(expected)
    # annualising the daily rate must give the original yield back
    assert (1 + rf.loc["2009-12-31"]) ** cfg.TRADING_DAYS - 1 == pytest.approx(0.04)


def test_forward_fill_never_uses_a_future_auction(tbill_file):
    """On 2010-01-07 only the 06 Jan auction is known, not the 13 Jan one."""
    cal = pd.DatetimeIndex(["2010-01-07", "2010-01-12", "2010-01-13"])
    rf = D.load_riskfree(cal)
    five = (1 + 0.05) ** (1 / cfg.TRADING_DAYS) - 1
    six = (1 + 0.06) ** (1 / cfg.TRADING_DAYS) - 1
    assert rf.loc["2010-01-07"] == pytest.approx(five)
    assert rf.loc["2010-01-12"] == pytest.approx(five)
    assert rf.loc["2010-01-13"] == pytest.approx(six)


def test_truncating_the_calendar_does_not_change_earlier_values(tbill_file):
    cal = pd.DatetimeIndex(["2010-01-07", "2010-01-12", "2010-01-13", "2010-01-20"])
    full = D.load_riskfree(cal)
    part = D.load_riskfree(cal[:2])
    pd.testing.assert_series_equal(part, full.iloc[:2])


def test_missing_file_raises_with_the_required_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "TBILL_CSV", tmp_path / "nope.csv")
    with pytest.raises(FileNotFoundError, match="yield_pct"):
        D.load_riskfree(pd.DatetimeIndex(["2010-01-01"]))


def test_wrong_columns_raise_with_the_required_columns(tmp_path, monkeypatch):
    path = tmp_path / "tbill_91d.csv"
    path.write_text("Date,Rate\n2010-01-01,5.0\n", encoding="utf-8")
    monkeypatch.setattr(cfg, "TBILL_CSV", path)
    with pytest.raises(ValueError, match="yield_pct"):
        D.load_riskfree(pd.DatetimeIndex(["2010-01-01"]))
