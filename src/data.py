"""Phase 1: download, clean and load the price data.

Plain-English summary of what this module does:

1.  Downloads adjusted daily closing prices for the current Nifty 50 members,
    the Nifty 50 index (^NSEI) and India VIX (^INDIAVIX) from Yahoo Finance.
2.  Puts every stock on one shared trading calendar - the days the index itself
    traded - so "missing" means the same thing for every stock.
3.  Works out, for each day, which stocks are allowed in the universe: a stock
    needs 252 trading days of history behind it before it can be used.
4.  Flags (never deletes) daily returns beyond +/-50% so they can be checked by
    hand against corporate-action records.
5.  Saves everything to data/processed/ and writes the Phase 1 reports to
    results/.

Every operation here is causal: a value computed for day t uses only data up to
and including day t. The one exception is the *choice of universe* (the current
index members), which is survivorship bias by construction and is the thing
Phase 5 fixes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as cfg

# --------------------------------------------------------------------------
# universe
# --------------------------------------------------------------------------


def load_universe() -> pd.DataFrame:
    """The Version-1 universe: the current Nifty 50 members, as published by NSE."""
    uni = pd.read_csv(cfg.UNIVERSE_CSV)
    if uni["ticker"].duplicated().any():
        raise ValueError("duplicate tickers in config/universe.csv")
    return uni


def universe_tickers() -> list[str]:
    return load_universe()["ticker"].tolist()


# --------------------------------------------------------------------------
# download (raw)
# --------------------------------------------------------------------------

RAW_STOCKS = "stocks_raw.csv"
RAW_INDEX = "nsei_raw.csv"
RAW_VIX = "indiavix_raw.csv"


def _yf_download(tickers: list[str], start: str, end: str, tries: int = 3) -> pd.DataFrame:
    """Adjusted closes from Yahoo Finance, as a date x ticker frame."""
    import yfinance as yf

    end_inclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    last_err: Exception | None = None
    for attempt in range(tries):
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end_inclusive,
                auto_adjust=True,      # adjusted closes (splits and dividends)
                progress=False,
                threads=True,
                actions=False,
            )
            if raw is None or len(raw) == 0:
                raise RuntimeError("empty frame returned")
            close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
            if isinstance(close, pd.Series):
                close = close.to_frame(tickers[0])
            close.columns = [str(c) for c in close.columns]
            close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
            return close.sort_index()
        except Exception as err:  # noqa: BLE001 - network flakiness, so retry
            last_err = err
            if attempt < tries - 1:
                time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"yfinance download failed after {tries} tries: {last_err}")


def download_raw(force: bool = False) -> dict[str, pd.DataFrame]:
    """Download and cache the raw series. Served from cache unless force=True."""
    cfg.ensure_dirs()
    paths = {
        "stocks": cfg.RAW_DIR / RAW_STOCKS,
        "index": cfg.RAW_DIR / RAW_INDEX,
        "vix": cfg.RAW_DIR / RAW_VIX,
    }

    if not force and all(p.exists() for p in paths.values()):
        return {k: _read_raw(p) for k, p in paths.items()}

    tickers = universe_tickers()
    print(
        f"Downloading {len(tickers)} stocks, {cfg.INDEX_TICKER} and {cfg.VIX_TICKER} "
        f"({cfg.DATA_START} to {cfg.DATA_END})..."
    )
    stocks = _yf_download(tickers, cfg.DATA_START, cfg.DATA_END)
    stocks = stocks.reindex(columns=tickers)          # stable, documented column order
    stocks = _repair_missing_columns(stocks)
    index = _yf_download([cfg.INDEX_TICKER], cfg.DATA_START, cfg.DATA_END)
    vix = _yf_download([cfg.VIX_TICKER], cfg.DATA_START, cfg.DATA_END)

    out = {"stocks": stocks, "index": index, "vix": vix}
    for key, frame in out.items():
        frame.to_csv(paths[key], index_label="date")
    return out


def _repair_missing_columns(stocks: pd.DataFrame) -> pd.DataFrame:
    """Retry, one ticker at a time, any column the bulk download returned empty.

    Yahoo occasionally drops a ticker from a batch request (its local cache can
    lock under parallel threads). That is a transport failure, not real missing
    data, so retrying it serially is not cherry-picking.
    """
    empty = [c for c in stocks.columns if stocks[c].notna().sum() == 0]
    for ticker in empty:
        print(f"  retrying {ticker} on its own (empty in the batch download)...")
        try:
            one = _yf_download([ticker], cfg.DATA_START, cfg.DATA_END, tries=2)
        except RuntimeError as err:
            print(f"    still failed: {err}")
            continue
        col = one.iloc[:, 0].reindex(stocks.index)
        if col.notna().sum() > 0:
            stocks[ticker] = col
            print(f"    recovered {int(col.notna().sum())} observations")
    return stocks


def _read_raw(path) -> pd.DataFrame:
    return pd.read_csv(path, index_col="date", parse_dates=["date"]).sort_index()


# --------------------------------------------------------------------------
# cleaning
# --------------------------------------------------------------------------


def load_corporate_actions() -> pd.DataFrame:
    """Demergers and schemes of arrangement, from NSE's corporate-action record.

    Empty if Phase 5's sweep has not been run - the pipeline still works, it
    just leaves the fake ex-date returns in, which is what it did before.
    """
    path = cfg.CONFIG_DIR / "corporate_actions.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["ex_date"] = pd.to_datetime(df["ex_date"]).dt.date
    return df


@dataclass
class CleanedData:
    prices: pd.DataFrame        # adjusted closes on the index calendar
    returns: pd.DataFrame       # simple daily returns
    eligible: pd.DataFrame      # bool: has >= 252 trading days of history by t
    nsei: pd.Series             # index level
    nsei_returns: pd.Series
    vix: pd.Series
    listing: pd.DataFrame       # per-stock coverage summary
    flagged: pd.DataFrame       # returns beyond +/-50%


def clean(raw: dict[str, pd.DataFrame]) -> CleanedData:
    """Turn the raw downloads into the frames every later phase reads."""
    nsei = raw["index"].iloc[:, 0].dropna()
    nsei = nsei.loc[cfg.DATA_START:cfg.DATA_END]
    calendar = nsei.index                                   # the shared trading calendar

    prices = raw["stocks"].reindex(calendar)
    # Yahoo sometimes returns an all-NaN column for a renamed or delisted ticker.
    all_missing = prices.columns[prices.notna().sum() == 0].tolist()
    if all_missing:
        print(f"  WARNING: no data at all for {all_missing} - kept as empty columns")

    # Demergers: Yahoo adjusts for splits and dividends but not for a business
    # being spun off, so the parent's ex-date drop is left in as a fake return.
    # Rescaling the pre-ex-date history by (ex-date close / previous close)
    # zeroes that return and changes no other return in the series, because a
    # constant factor cancels in a ratio.
    #
    # This stays causal: the factor can only be computed once the ex-date price
    # exists, so on truncated data an event in the future simply is not applied,
    # and every return up to the cut is identical either way.
    adjustment_log = pd.DataFrame()
    unadjusted_exdates = pd.DataFrame()
    events = load_corporate_actions()
    if not events.empty:
        from . import corporate_actions as CA

        prices, adjustment_log, unadjusted_exdates = CA.apply_demerger_adjustments(
            prices, events)

    returns = prices.pct_change()

    # An event that could not be measured leaves its ex-date return unusable.
    # It is blanked rather than left in: a fake -40% is far worse than a gap,
    # and the 5%-missing rule already knows what to do with a gap.
    for _, row in unadjusted_exdates.iterrows():
        ticker, ex = row["ticker"], pd.Timestamp(row["ex_date"])
        if ticker in returns.columns and ex in returns.index:
            returns.loc[ex, ticker] = np.nan
    # The first observation of a stock that starts mid-sample is not a return.
    first_obs = prices.notna().idxmax()
    for col in prices.columns:
        if prices[col].notna().any():
            returns.loc[: first_obs[col], col] = np.nan

    # 252 trading days of history, counted causally (observations up to and including t).
    obs_count = prices.notna().cumsum()
    eligible = obs_count >= cfg.MIN_HISTORY_DAYS

    vix = raw["vix"].iloc[:, 0].reindex(calendar)

    listing = _coverage_report(prices)
    flagged = _flag_extreme_returns(returns)

    return CleanedData(
        prices=prices,
        returns=returns,
        eligible=eligible,
        nsei=nsei,
        nsei_returns=nsei.pct_change(),
        vix=vix,
        listing=listing,
        flagged=flagged,
    )


def _coverage_report(prices: pd.DataFrame) -> pd.DataFrame:
    """Per stock: when its data starts and ends, and how much is missing inside that span.

    Days before a stock's first observation are "not listed yet", not "missing";
    counting them as missing would make every recent index addition look broken.
    """
    rows = []
    for col in prices.columns:
        s = prices[col]
        if not s.notna().any():
            rows.append({
                "ticker": col, "first_date": pd.NaT, "last_date": pd.NaT,
                "n_obs": 0, "n_calendar_days_in_span": 0, "n_missing_in_span": 0,
                "pct_missing_in_span": np.nan, "first_eligible_date": pd.NaT,
            })
            continue
        first, last = s.first_valid_index(), s.last_valid_index()
        span = s.loc[first:last]
        n_missing = int(span.isna().sum())
        elig = s.notna().cumsum() >= cfg.MIN_HISTORY_DAYS
        rows.append({
            "ticker": col,
            "first_date": first,
            "last_date": last,
            "n_obs": int(s.notna().sum()),
            "n_calendar_days_in_span": int(len(span)),
            "n_missing_in_span": n_missing,
            "pct_missing_in_span": round(100 * n_missing / len(span), 3),
            "first_eligible_date": elig.idxmax() if elig.any() else pd.NaT,
        })
    return pd.DataFrame(rows).set_index("ticker")


def _flag_extreme_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Daily moves beyond +/-50%: usually an unadjusted corporate action."""
    mask = returns.abs() > cfg.EXTREME_RETURN
    idx = np.argwhere(mask.to_numpy())
    rows = [
        {"date": returns.index[i], "ticker": returns.columns[j], "return": returns.iat[i, j]}
        for i, j in idx
    ]
    out = pd.DataFrame(rows, columns=["date", "ticker", "return"])
    if not out.empty:
        out = out.sort_values("return", key=abs, ascending=False).reset_index(drop=True)
        out["return_pct"] = (100 * out["return"]).round(2)
    return out


def staleness_report(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Per stock: how often the price simply does not move.

    This is a *diagnostic only* - nothing is filtered on it. The missing-data
    rules in the spec catch gaps (no number at all), but not a vendor serving a
    flat, carried-forward price for a stretch where it has no real history. A
    long flat run is a fake zero return, and would feed a fake zero signal into
    the momentum sort, so it needs to be visible even though the spec does not
    (yet) act on it.
    """
    rows = []
    for col in prices.columns:
        s = prices[col].dropna()
        r = returns[col].dropna()
        if len(s) < 2:
            rows.append({"ticker": col, "n_obs": len(s), "pct_zero_return_days": np.nan,
                         "longest_flat_run_days": 0, "longest_flat_run_start": pd.NaT,
                         "longest_flat_run_end": pd.NaT})
            continue
        # length of each run of identical consecutive prices
        new_run = s.ne(s.shift())
        run_id = new_run.cumsum()
        sizes = run_id.value_counts()
        worst_id = sizes.idxmax()
        worst = s.index[run_id == worst_id]
        rows.append({
            "ticker": col,
            "n_obs": int(len(s)),
            "pct_zero_return_days": round(100 * float((r == 0).mean()), 3),
            "longest_flat_run_days": int(sizes.max()),
            "longest_flat_run_start": worst.min(),
            "longest_flat_run_end": worst.max(),
        })
    out = pd.DataFrame(rows).set_index("ticker")
    return out.sort_values("longest_flat_run_days", ascending=False)


# --------------------------------------------------------------------------
# the 5%-missing formation-window rule
# --------------------------------------------------------------------------


def formation_missing_fraction(
    prices: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.Series:
    """Fraction of the formation window [start, end] each stock is missing.

    Uses only days inside the window, so it can never see the future. A stock
    whose data begins after `start` counts those earlier days as missing: it
    genuinely has no price for them, so its signal would be unreliable.
    """
    window = prices.loc[start:end]
    if len(window) == 0:
        return pd.Series(1.0, index=prices.columns)
    return 1.0 - window.notna().sum() / len(window)


# --------------------------------------------------------------------------
# the staleness rule (flat runs of >= 10 identical closes)
# --------------------------------------------------------------------------


def flat_run_length(prices: pd.DataFrame) -> pd.DataFrame:
    """For each stock and day d: the length of the run of identical closes ending at d.

    Causal by construction - the value at day d counts backwards from d and never
    forwards, so a run still in progress at d is reported at its length so far.

    Runs are counted over observed prices only, so a one-day gap in the middle of
    a flat stretch does not reset the count. That is the conservative choice: it
    catches more staleness, not less.
    """
    out = pd.DataFrame(index=prices.index, columns=prices.columns, dtype="float64")
    for col in prices.columns:
        s = prices[col].dropna()
        if s.empty:
            continue
        run_id = s.ne(s.shift()).cumsum()
        out[col] = s.groupby(run_id).cumcount().add(1).reindex(prices.index)
    return out


def stale_in_window(
    runs: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_run: int = cfg.MAX_FLAT_RUN_DAYS,
    strict_containment: bool = False,
) -> pd.Series:
    """True for each stock whose formation window [start, end] is hit by a flat run.

    `runs` comes from flat_run_length(). Taking the maximum run length over the
    days inside the window finds the longest run that *intersects* the window:
    for a run that ends inside the window that maximum is the run's full length,
    and for a run still going at `end` it is its length so far. Either way only
    data up to `end` is used.

    With strict_containment=True the run must fit entirely inside the window
    instead, which is reported alongside for comparison but is not the rule.
    """
    window = runs.loc[start:end]
    if window.empty:
        return pd.Series(False, index=runs.columns)
    if strict_containment:
        capped = window.clip(upper=len(window))
        return (capped.max() >= min_run).fillna(False)
    return (window.max() >= min_run).fillna(False)


def eligible_at(
    cleaned: CleanedData,
    as_of: pd.Timestamp,
    formation_start: pd.Timestamp,
    formation_end: pd.Timestamp | None = None,
    runs: pd.DataFrame | None = None,
) -> pd.Index:
    """Tickers usable for a portfolio formed at `as_of`.

    Four filters, all causal:
      * at least 252 trading days of history by `as_of`;
      * no more than 5% of the formation window missing;
      * no flat run of >= 10 identical closes intersecting the formation window;
      * a price exists on `as_of`, so the position can actually be entered.
    """
    as_of = pd.Timestamp(as_of)
    formation_start = pd.Timestamp(formation_start)
    formation_end = as_of if formation_end is None else pd.Timestamp(formation_end)

    has_history = cleaned.eligible.loc[:as_of].iloc[-1]
    missing = formation_missing_fraction(cleaned.prices, formation_start, formation_end)
    tradable = cleaned.prices.loc[:as_of].iloc[-1].notna()

    if runs is None:
        runs = flat_run_length(cleaned.prices.loc[:formation_end])
    stale = stale_in_window(runs, formation_start, formation_end)

    keep = has_history & (missing <= cfg.MAX_MISSING_FRACTION) & tradable & ~stale
    return cleaned.prices.columns[keep.to_numpy()]


# --------------------------------------------------------------------------
# risk-free rate
# --------------------------------------------------------------------------

TBILL_HELP = f"""
Missing file: {cfg.TBILL_CSV}

Please save the RBI 91-day Treasury bill yield there as a CSV with exactly
these two columns (header row required, this spelling):

    date,yield_pct

  * date      - YYYY-MM-DD (or DD-MM-YYYY), one row per observation.
                Weekly auction dates are fine; the series is forward-filled
                onto trading days. It must start on or before 2010-01-01.
  * yield_pct - the annualised yield in PERCENT per year, e.g. 6.57 for 6.57%.
                Not a decimal fraction, not a price.

Example:
    date,yield_pct
    2009-01-07,4.78
    2009-01-14,4.65
""".strip()


def load_riskfree(calendar: pd.DatetimeIndex) -> pd.Series:
    """Daily risk-free return on the trading calendar, from the 91-day T-bill yield.

    The annual yield y% becomes a daily rate as (1+y/100)**(1/252)-1, and is
    forward-filled: each day uses the most recent auction at or before that day,
    never a later one.
    """
    if not cfg.TBILL_CSV.exists():
        raise FileNotFoundError(TBILL_HELP)

    df = pd.read_csv(cfg.TBILL_CSV)
    cols = {c.strip().lower(): c for c in df.columns}
    if "date" not in cols or "yield_pct" not in cols:
        raise ValueError(
            f"{cfg.TBILL_CSV} must have columns 'date' and 'yield_pct'; "
            f"found {list(df.columns)}\n\n{TBILL_HELP}"
        )
    # Accept either YYYY-MM-DD or DD-MM-YYYY: try ISO first, fall back to day-first.
    raw_dates = df[cols["date"]].astype(str).str.strip()
    dates = pd.to_datetime(raw_dates, errors="coerce", format="ISO8601")
    if dates.isna().any():
        dates = dates.fillna(pd.to_datetime(raw_dates, errors="coerce", dayfirst=True))

    s = pd.Series(
        pd.to_numeric(df[cols["yield_pct"]], errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(dates),
    ).dropna().sort_index()
    if s.empty:
        raise ValueError(f"no usable rows in {cfg.TBILL_CSV}\n\n{TBILL_HELP}")

    daily = (1.0 + s / 100.0) ** (1.0 / cfg.TRADING_DAYS) - 1.0
    return daily.reindex(daily.index.union(calendar)).ffill().reindex(calendar)


# --------------------------------------------------------------------------
# save / load processed
# --------------------------------------------------------------------------

PROC_FILES = {
    "prices": "prices.csv",
    "returns": "returns.csv",
    "eligible": "eligible.csv",
    "nsei": "nsei.csv",
    "vix": "indiavix.csv",
}


def save_processed(cleaned: CleanedData) -> None:
    cfg.ensure_dirs()
    cleaned.prices.to_csv(cfg.PROC_DIR / PROC_FILES["prices"], index_label="date")
    cleaned.returns.to_csv(cfg.PROC_DIR / PROC_FILES["returns"], index_label="date")
    cleaned.eligible.astype(int).to_csv(
        cfg.PROC_DIR / PROC_FILES["eligible"], index_label="date"
    )
    pd.DataFrame({"nsei_close": cleaned.nsei, "nsei_return": cleaned.nsei_returns}).to_csv(
        cfg.PROC_DIR / PROC_FILES["nsei"], index_label="date"
    )
    cleaned.vix.rename("india_vix").to_csv(
        cfg.PROC_DIR / PROC_FILES["vix"], index_label="date"
    )


def load_processed() -> CleanedData:
    """Read back what Phase 1 saved, for use by Phases 2-5."""
    p = cfg.PROC_DIR
    prices = pd.read_csv(p / PROC_FILES["prices"], index_col="date", parse_dates=["date"])
    returns = pd.read_csv(p / PROC_FILES["returns"], index_col="date", parse_dates=["date"])
    eligible = pd.read_csv(
        p / PROC_FILES["eligible"], index_col="date", parse_dates=["date"]
    ).astype(bool)
    nsei_df = pd.read_csv(p / PROC_FILES["nsei"], index_col="date", parse_dates=["date"])
    vix = pd.read_csv(p / PROC_FILES["vix"], index_col="date", parse_dates=["date"]).iloc[:, 0]
    return CleanedData(
        prices=prices, returns=returns, eligible=eligible,
        nsei=nsei_df["nsei_close"], nsei_returns=nsei_df["nsei_return"], vix=vix,
        listing=_coverage_report(prices), flagged=_flag_extreme_returns(returns),
    )


def build(force_download: bool = False, truncate_at: str | None = None) -> CleanedData:
    """Full Phase 1 pipeline. `truncate_at` is what the look-ahead tests use."""
    raw = download_raw(force=force_download)
    if truncate_at is not None:
        cut = pd.Timestamp(truncate_at)
        raw = {k: v.loc[:cut] for k, v in raw.items()}
    return clean(raw)
