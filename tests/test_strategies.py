"""Phase 3: portfolio construction, turnover and costs.

The look-ahead test for a strategy is specifically about *portfolio formation*:
the holding-period return obviously needs data after the rebalance day, but the
signal, the eligible universe and the chosen holdings must be identical whether
or not the future exists. That is what is tested here.

The arithmetic (quintiles, equal weights, weight drift, turnover, cost) is
tested on small synthetic price paths where the right answer can be worked out
by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src import data as D
from src import strategies as S
from src import windows as W


# --------------------------------------------------------------------------
# a synthetic dataset with known answers
# --------------------------------------------------------------------------


def make_cleaned(prices: pd.DataFrame) -> D.CleanedData:
    """Wrap a price frame as a CleanedData with every filter satisfied."""
    returns = prices.pct_change()
    eligible = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    nsei = pd.Series(100.0, index=prices.index)
    return D.CleanedData(
        prices=prices, returns=returns, eligible=eligible,
        nsei=nsei, nsei_returns=nsei.pct_change(),
        vix=pd.Series(15.0, index=prices.index),
        listing=pd.DataFrame(), flagged=pd.DataFrame(),
    )


@pytest.fixture
def ramp():
    """10 stocks on 300 business days, each compounding at a different rate.

    Stock i grows at (i+1) basis points a day, so the ordering by any trailing
    return is always stock 0 (worst) .. stock 9 (best). That makes the expected
    quintile membership obvious.
    """
    idx = pd.bdate_range("2009-01-01", periods=300)
    data = {f"S{i}": 100 * (1 + (i + 1) / 10_000) ** np.arange(len(idx))
            for i in range(10)}
    return make_cleaned(pd.DataFrame(data, index=idx))


def one_window(cal, k=1):
    """A simple formation/holding window over a synthetic calendar."""
    ends = W.month_end_dates(cal)
    return W.Window(rebalance_date=ends[-3], formation_start=ends[-5],
                    formation_end=ends[-3], holding_end=ends[-2])


# --------------------------------------------------------------------------
# signals and quintiles
# --------------------------------------------------------------------------


def test_window_return_reads_only_the_formation_window(ramp):
    win = one_window(ramp.prices.index)
    sig = S.window_return(ramp.prices, win)
    expected = (ramp.prices.loc[win.formation_end]
                / ramp.prices.loc[win.formation_start] - 1)
    pd.testing.assert_series_equal(sig, expected.rename("signal"))


def test_momentum_buys_winners_and_sells_losers(ramp):
    win = one_window(ramp.prices.index)
    res = S.run_strategy(ramp, [win], "top", "bottom", "m", "monthly")
    longs = set(res.holdings[res.holdings["side"] == "long"]["ticker"])
    shorts = set(res.holdings[res.holdings["side"] == "short"]["ticker"])
    assert longs == {"S8", "S9"}, longs       # top quintile of 10 = 2 names
    assert shorts == {"S0", "S1"}, shorts


def test_reversal_buys_losers_and_sells_winners(ramp):
    win = one_window(ramp.prices.index)
    res = S.run_strategy(ramp, [win], "bottom", "top", "r", "weekly")
    longs = set(res.holdings[res.holdings["side"] == "long"]["ticker"])
    shorts = set(res.holdings[res.holdings["side"] == "short"]["ticker"])
    assert longs == {"S0", "S1"}
    assert shorts == {"S8", "S9"}


def test_long_only_has_no_short_leg(ramp):
    win = one_window(ramp.prices.index)
    res = S.run_strategy(ramp, [win], "top", None, "m", "monthly")
    assert res.n_short.iloc[0] == 0
    assert res.gross.iloc[0] == pytest.approx(res.r_long_leg.iloc[0])


def test_benchmark_holds_every_eligible_stock(ramp):
    win = one_window(ramp.prices.index)
    res = S.run_strategy(ramp, [win], "top", None, "b", "monthly", equal_weight_all=True)
    assert res.n_long.iloc[0] == 10
    hold = S.holding_return(ramp.prices, win)
    assert res.gross.iloc[0] == pytest.approx(hold.mean())


def test_too_few_stocks_produces_an_empty_portfolio():
    idx = pd.bdate_range("2009-01-01", periods=200)
    prices = pd.DataFrame({f"S{i}": np.linspace(100, 110, len(idx)) for i in range(3)},
                          index=idx)
    cleaned = make_cleaned(prices)
    win = one_window(idx)
    res = S.run_strategy(cleaned, [win], "top", "bottom", "m", "monthly")
    assert res.n_long.iloc[0] == 0 and res.n_short.iloc[0] == 0
    assert res.gross.iloc[0] == 0.0


# --------------------------------------------------------------------------
# weights, drift, turnover, costs
# --------------------------------------------------------------------------


def test_weights_are_equal_within_each_leg(ramp):
    win = one_window(ramp.prices.index)
    res = S.run_strategy(ramp, [win], "top", "bottom", "m", "monthly")
    hold = S.holding_return(ramp.prices, win)
    expected = hold[["S8", "S9"]].mean() - hold[["S0", "S1"]].mean()
    assert res.gross.iloc[0] == pytest.approx(expected)


def test_first_rebalance_is_charged_for_building_the_book(ramp):
    """Opening $1 long and $1 short trades 2.0 of capital."""
    win = one_window(ramp.prices.index)
    res = S.run_strategy(ramp, [win], "top", "bottom", "m", "monthly")
    assert res.turnover.iloc[0] == pytest.approx(2.0)


def test_holding_the_same_names_costs_only_the_drift(ramp):
    """If the portfolio does not change, turnover is small - not 2.0 again."""
    ends = W.month_end_dates(ramp.prices.index)
    wins = [
        W.Window(ends[-4], ends[-6], ends[-4], ends[-3]),
        W.Window(ends[-3], ends[-5], ends[-3], ends[-2]),
    ]
    res = S.run_strategy(ramp, wins, "top", "bottom", "m", "monthly")
    # the ramp keeps the same ordering, so the same names are picked twice
    assert res.turnover.iloc[0] == pytest.approx(2.0)
    assert res.turnover.iloc[1] < 0.05


def test_full_replacement_trades_four(ramp):
    """Sell $2, buy $2. This is the number the 80bp monthly cost comes from."""
    new_long = pd.Series({"A": 0.5, "B": 0.5})
    old_long = pd.Series({"C": 0.5, "D": 0.5})
    assert S._traded(new_long, old_long) == pytest.approx(2.0)
    # both legs together
    assert (S._traded(new_long, old_long) + S._traded(new_long, old_long)
            == pytest.approx(4.0))


def test_drift_renormalises_to_one():
    w = pd.Series({"A": 0.5, "B": 0.5})
    r = pd.Series({"A": 1.0, "B": 0.0})          # A doubles, B flat
    drifted = S._drifted(w, r)
    assert drifted.sum() == pytest.approx(1.0)
    assert drifted["A"] == pytest.approx(2 / 3)


def test_cost_is_bps_times_turnover(ramp):
    win = one_window(ramp.prices.index)
    res = S.run_strategy(ramp, [win], "top", "bottom", "m", "monthly")
    for bps in S.COST_BPS:
        expected = res.gross.iloc[0] - (bps / 10_000) * res.turnover.iloc[0]
        assert res.net[bps].iloc[0] == pytest.approx(expected)


def test_higher_costs_never_help(ramp):
    win = one_window(ramp.prices.index)
    res = S.run_strategy(ramp, [win], "top", "bottom", "m", "monthly")
    assert res.gross.iloc[0] >= res.net[10].iloc[0] >= res.net[20].iloc[0] >= res.net[30].iloc[0]


# --------------------------------------------------------------------------
# holding to the end of the period
# --------------------------------------------------------------------------


def test_a_stock_that_stops_trading_is_held_at_its_last_price():
    """The spec says a position is held to the end of its holding period."""
    idx = pd.bdate_range("2009-01-01", periods=200)
    prices = pd.DataFrame({f"S{i}": np.linspace(100, 200, len(idx)) for i in range(10)},
                          index=idx)
    prices.loc[idx[-10:], "S9"] = np.nan        # stops trading near the end
    cleaned = make_cleaned(prices)
    win = W.Window(idx[-30], idx[-60], idx[-30], idx[-1])
    hold = S.holding_return(prices, win)
    last_traded = prices["S9"].dropna().iloc[-1]
    assert hold["S9"] == pytest.approx(last_traded / prices.loc[win.rebalance_date, "S9"] - 1)
    assert not np.isnan(hold["S9"])


# --------------------------------------------------------------------------
# look-ahead: portfolio formation on truncated data
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real():
    if not (cfg.PROC_DIR / D.PROC_FILES["prices"]).exists():
        pytest.skip("run `python -m src.run_phase1` first")
    return D.load_processed()


@pytest.mark.parametrize("cut", ["2013-06-28", "2017-09-29", "2019-12-31"])
def test_holdings_chosen_at_t_do_not_depend_on_data_after_t(real, cut):
    """Truncate the price history at the rebalance day and re-pick the portfolio."""
    cut = pd.Timestamp(cut)
    truncated = D.CleanedData(
        prices=real.prices.loc[:cut], returns=real.returns.loc[:cut],
        eligible=real.eligible.loc[:cut], nsei=real.nsei.loc[:cut],
        nsei_returns=real.nsei_returns.loc[:cut], vix=real.vix.loc[:cut],
        listing=pd.DataFrame(), flagged=pd.DataFrame(),
    )
    full_runs = D.flat_run_length(real.prices)
    part_runs = D.flat_run_length(truncated.prices)

    for builder in (W.momentum_windows, W.reversal_windows):
        wins = [w for w in builder(real.prices.index) if w.rebalance_date == cut]
        if not wins:
            continue
        win = wins[0]
        full_allowed = D.eligible_at(real, win.rebalance_date, win.formation_start,
                                     win.formation_end, runs=full_runs)
        part_allowed = D.eligible_at(truncated, win.rebalance_date, win.formation_start,
                                     win.formation_end, runs=part_runs)
        assert list(full_allowed) == list(part_allowed)

        full_sig = S.window_return(real.prices, win).reindex(full_allowed).dropna()
        part_sig = S.window_return(truncated.prices, win).reindex(part_allowed).dropna()
        pd.testing.assert_series_equal(full_sig, part_sig)


def test_long_short_is_zero_investment(real):
    """$1 long and $1 short: the net capital committed is zero by construction."""
    runs = D.flat_run_length(real.prices)
    wins = [w for w in W.momentum_windows(real.prices.index)
            if pd.Timestamp("2015-01-01") <= w.rebalance_date <= pd.Timestamp("2015-12-31")]
    res = S.run_strategy(real, wins, "top", "bottom", "m", "monthly", runs)
    assert (res.n_long == res.n_short).all()
    # the return is the spread of the two legs, with no risk-free component
    pd.testing.assert_series_equal(
        res.gross, (res.r_long_leg - res.r_short_leg).rename("gross"))


def test_quintiles_are_roughly_a_fifth_of_the_universe(real):
    runs = D.flat_run_length(real.prices)
    wins = [w for w in W.momentum_windows(real.prices.index)
            if pd.Timestamp("2015-01-01") <= w.rebalance_date <= pd.Timestamp("2015-12-31")]
    res = S.run_strategy(real, wins, "top", "bottom", "m", "monthly", runs)
    ratio = res.n_long / res.n_eligible
    assert ((ratio > 0.15) & (ratio < 0.25)).all()
