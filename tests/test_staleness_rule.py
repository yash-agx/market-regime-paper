"""The staleness rule: exclude a formation window hit by a run of >=10 identical closes.

Built on small hand-made price series, so the expected answer can be counted by
eye rather than taken from the pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src import data as D


def _frame(values: dict[str, list[float]], n: int | None = None) -> pd.DataFrame:
    n = n if n is not None else len(next(iter(values.values())))
    idx = pd.bdate_range("2010-01-04", periods=n)
    return pd.DataFrame(values, index=idx)


# --------------------------------------------------------------------------
# flat_run_length
# --------------------------------------------------------------------------


def test_run_length_counts_backwards_from_each_day():
    prices = _frame({"A": [10, 10, 10, 11, 11, 12]})
    runs = D.flat_run_length(prices)["A"].tolist()
    assert runs == [1, 2, 3, 1, 2, 1]


def test_run_length_is_causal_under_truncation():
    """A run in progress is reported at its length so far, not its eventual length."""
    prices = _frame({"A": [5, 5, 5, 5, 6]})
    full = D.flat_run_length(prices)["A"]
    part = D.flat_run_length(prices.iloc[:3])["A"]
    pd.testing.assert_series_equal(part, full.iloc[:3])
    assert full.iloc[2] == 3          # three days into the run, not four


def test_run_length_ignores_gaps_rather_than_resetting():
    """A missing day inside a flat stretch must not hide the staleness."""
    prices = _frame({"A": [7, 7, np.nan, 7, 7]})
    runs = D.flat_run_length(prices)["A"]
    assert runs.iloc[-1] == 4
    assert pd.isna(runs.iloc[2])


def test_run_length_handles_an_empty_column():
    prices = _frame({"A": [1.0, 2.0, 3.0], "B": [np.nan] * 3})
    runs = D.flat_run_length(prices)
    assert runs["B"].isna().all()


# --------------------------------------------------------------------------
# stale_in_window
# --------------------------------------------------------------------------


def test_window_with_a_long_flat_run_is_excluded():
    prices = _frame({"flat": [3.0] * 12 + [4.0] * 3, "moving": list(np.arange(15.0))})
    runs = D.flat_run_length(prices)
    stale = D.stale_in_window(runs, prices.index[0], prices.index[-1])
    assert stale["flat"]
    assert not stale["moving"]


def test_a_nine_day_run_is_not_excluded():
    """The threshold is 10, so 9 must pass - the boundary is worth pinning down."""
    prices = _frame({"A": [2.0] * 9 + [3.0, 4.0]})
    runs = D.flat_run_length(prices)
    assert not D.stale_in_window(runs, prices.index[0], prices.index[-1])["A"]


def test_a_ten_day_run_is_excluded():
    prices = _frame({"A": [2.0] * 10 + [3.0]})
    runs = D.flat_run_length(prices)
    assert D.stale_in_window(runs, prices.index[0], prices.index[-1])["A"]


def test_a_run_ending_inside_a_short_window_excludes_it():
    """The reversal case: a 5-day window cannot contain a 10-day run, but it can
    sit inside one. This is the overlap reading the spec records."""
    prices = _frame({"A": [1.0] * 20 + list(np.arange(2.0, 12.0))})
    runs = D.flat_run_length(prices)
    short_window = D.stale_in_window(runs, prices.index[15], prices.index[19])
    assert short_window["A"]
    # strict containment cannot see it, which is why the spec does not use it
    strict = D.stale_in_window(
        runs, prices.index[15], prices.index[19], strict_containment=True
    )
    assert not strict["A"]


def test_a_run_entirely_before_the_window_does_not_exclude_it():
    prices = _frame({"A": [1.0] * 12 + list(np.arange(2.0, 12.0))})
    runs = D.flat_run_length(prices)
    stale = D.stale_in_window(runs, prices.index[12], prices.index[-1])
    assert not stale["A"]


def test_staleness_verdict_is_unchanged_by_later_data():
    """Point-in-time: adding future prices cannot change a past window's verdict."""
    prices = _frame({"A": [1.0] * 11 + [2.0] * 4})
    runs_full = D.flat_run_length(prices)
    runs_part = D.flat_run_length(prices.iloc[:11])
    w0, w1 = prices.index[0], prices.index[10]
    assert (D.stale_in_window(runs_part, w0, w1) == D.stale_in_window(runs_full, w0, w1)).all()


# --------------------------------------------------------------------------
# the rule as wired into eligible_at, on the real data
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cleaned():
    if not (cfg.PROC_DIR / D.PROC_FILES["prices"]).exists():
        pytest.skip("run `python -m src.run_phase1` first")
    return D.load_processed()


def test_nestle_is_excluded_from_its_flat_period_and_returns_afterwards(cleaned):
    """NESTLEIND.NS is flat from 2009-01-02 to 2010-01-07; the rule must catch it."""
    runs = D.flat_run_length(cleaned.prices)
    hit = D.stale_in_window(runs, pd.Timestamp("2009-02-02"), pd.Timestamp("2009-12-31"))
    assert hit["NESTLEIND.NS"]
    # a formation window entirely after the flat run is clean again
    clean = D.stale_in_window(runs, pd.Timestamp("2010-02-01"), pd.Timestamp("2010-12-31"))
    assert not clean["NESTLEIND.NS"]


def test_only_nestle_is_ever_excluded_by_staleness(cleaned):
    """The rule must remove the vendor artefact and nothing else."""
    from src.run_staleness_report import audit

    combined = pd.concat(audit(cleaned).values(), ignore_index=True)
    assert set(combined["ticker"].unique()) <= {"NESTLEIND.NS"}


def test_eligible_at_applies_the_staleness_rule(cleaned):
    """A momentum window reaching into the flat run must drop Nestle."""
    runs = D.flat_run_length(cleaned.prices)
    excluded = D.eligible_at(
        cleaned,
        as_of=pd.Timestamp("2010-01-29"),
        formation_start=pd.Timestamp("2008-12-31"),
        formation_end=pd.Timestamp("2009-12-31"),
        runs=runs,
    )
    assert "NESTLEIND.NS" not in excluded
