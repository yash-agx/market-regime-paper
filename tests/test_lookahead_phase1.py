"""Look-ahead tests for Phase 1 (the golden rule, made executable).

The idea behind every test here: build the dataset twice - once from the full
download, once from a copy of the download that has been cut off at some date T -
and check that everything the pipeline says about days up to T is bit-for-bit
identical. If any step peeked at data after T, the two runs would disagree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src import data as D

TRUNCATION_DATES = ["2012-06-29", "2016-03-31", "2019-12-31"]


@pytest.fixture(scope="module")
def raw():
    """The cached raw download. Phase 1 must have been run at least once."""
    for name in (D.RAW_STOCKS, D.RAW_INDEX, D.RAW_VIX):
        if not (cfg.RAW_DIR / name).exists():
            pytest.skip(f"{name} not downloaded yet - run `python -m src.run_phase1 --force`")
    return D.download_raw(force=False)


@pytest.fixture(scope="module")
def full(raw):
    return D.clean(raw)


def _truncated(raw, cut):
    cut = pd.Timestamp(cut)
    return D.clean({k: v.loc[:cut] for k, v in raw.items()})


@pytest.mark.parametrize("cut", TRUNCATION_DATES)
def test_returns_unchanged_by_truncation(raw, full, cut):
    """Returns are what every signal is built from, so these must match exactly."""
    part = _truncated(raw, cut)
    cut = pd.Timestamp(cut)

    pd.testing.assert_frame_equal(part.returns, full.returns.loc[:cut])
    pd.testing.assert_series_equal(part.nsei, full.nsei.loc[:cut])
    pd.testing.assert_series_equal(part.nsei_returns, full.nsei_returns.loc[:cut])
    pd.testing.assert_series_equal(part.vix, full.vix.loc[:cut])


@pytest.mark.parametrize("cut", TRUNCATION_DATES)
def test_prices_change_only_by_a_constant_demerger_factor(raw, full, cut):
    """Price LEVELS may be restated by a demerger; ratios may not.

    When a stock demerges, the history before the ex-date is rescaled so the
    ex-date stops looking like a crash. Seen from before the ex-date that
    rescaling has not happened yet, so the two runs disagree on the level - by
    exactly one constant per ticker, which is what "adjusted close" has always
    meant and which cancels out of every return and every signal.

    What would be a real leak is a ratio changing, so that is what is tested:
    the full/truncated price ratio must be constant through time for every
    ticker, and only tickers with a demerger after the cut may differ at all.
    """
    part = _truncated(raw, cut)
    cut = pd.Timestamp(cut)
    ratio = (full.prices.loc[:cut] / part.prices).replace([np.inf, -np.inf], np.nan)

    spread = (ratio.max() - ratio.min()).dropna()
    drifting = spread[spread > 1e-9]
    assert drifting.empty, f"price ratio moves through time for {list(drifting.index)}"

    level = ratio.mean().dropna()
    rescaled = set(level[(level - 1).abs() > 1e-9].index)
    events = D.load_corporate_actions()
    future = set()
    if not events.empty:
        future = {f"{e.symbol}.NS" for e in events.itertuples()
                  if pd.Timestamp(e.ex_date) > cut}
    assert rescaled <= future, f"rescaled without a later demerger: {rescaled - future}"


@pytest.mark.parametrize("cut", TRUNCATION_DATES)
def test_eligibility_unchanged_by_truncation(raw, full, cut):
    """The 252-day history rule must be a running count, not a full-sample one."""
    part = _truncated(raw, cut)
    pd.testing.assert_frame_equal(part.eligible, full.eligible.loc[:pd.Timestamp(cut)])


@pytest.mark.parametrize("cut", TRUNCATION_DATES)
def test_formation_window_filter_unchanged_by_truncation(raw, full, cut):
    """The 5%-missing formation-window rule uses only days inside the window."""
    cut = pd.Timestamp(cut)
    part = _truncated(raw, cut)
    start = cut - pd.DateOffset(years=1)

    pd.testing.assert_series_equal(
        D.formation_missing_fraction(part.prices, start, cut),
        D.formation_missing_fraction(full.prices, start, cut),
    )
    assert list(D.eligible_at(part, cut, start)) == list(D.eligible_at(full, cut, start))


def test_eligibility_is_monotone_in_history(full):
    """Once a stock has 252 days of history it can never lose them again."""
    for col in full.eligible.columns:
        flags = full.eligible[col].to_numpy()
        # a 0 after a 1 would mean history was revoked
        assert not np.any(flags[:-1] & ~flags[1:]), f"{col} loses eligibility"


def test_no_stock_is_eligible_before_252_observations(full):
    obs = full.prices.notna().cumsum()
    assert not ((obs < cfg.MIN_HISTORY_DAYS) & full.eligible).to_numpy().any()


def test_returns_start_only_after_first_price(full):
    """A stock that lists mid-sample must not have a return on its first day."""
    for col in full.prices.columns:
        s = full.prices[col]
        if not s.notna().any():
            continue
        first = s.first_valid_index()
        assert full.returns[col].loc[:first].isna().all(), f"{col} has a pre-listing return"


def test_all_series_share_the_index_calendar(full):
    cal = full.nsei.index
    assert full.prices.index.equals(cal)
    assert full.returns.index.equals(cal)
    assert full.eligible.index.equals(cal)
    assert full.vix.index.equals(cal)
    assert cal.is_monotonic_increasing and not cal.has_duplicates


def test_extreme_returns_are_flagged_not_removed(full):
    """The spec says flag, never delete."""
    extreme = full.returns.abs() > cfg.EXTREME_RETURN
    assert len(full.flagged) == int(extreme.to_numpy().sum())
    for _, row in full.flagged.iterrows():
        assert full.returns.at[row["date"], row["ticker"]] == row["return"]


def test_universe_file_is_well_formed():
    uni = D.load_universe()
    assert len(uni) == 50, f"expected 50 Nifty 50 members, found {len(uni)}"
    assert uni["ticker"].str.endswith(".NS").all()
    assert uni["source_url"].notna().all(), "every ticker needs a documented source"
