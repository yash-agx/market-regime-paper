"""Phase 5: demergers, spin-offs and schemes of arrangement, from NSE records.

Why this exists
---------------
A demerger splits a company in two. The parent's share price drops on the
ex-date by roughly the value of the business that left, but shareholders are
made whole with shares in the new entity, so no money was lost. Yahoo Finance
adjusts for splits and dividends; it does NOT adjust for these. The drop is
left in the price series and reads as a catastrophic one-day return.

Two of these sit in our data: Adani Enterprises falls 38.8% on 2015-06-03 and
Tata Motors (as TMPV) falls 40.2% on 2025-10-14. Both are fictional. Momentum
shorted Adani Enterprises for twelve consecutive months on the strength of one.

Searching for them by return size is the wrong way round - it finds only the
big ones, and cannot tell a demerger from a crash (Adani Enterprises also fell
28% on real news in 2023). So this module asks NSE's corporate-action record
for every ticker in either universe and takes the answer from there.

The source
----------
NSE's corporate-actions endpoint, one query per symbol:

    https://www.nseindia.com/api/corporates-corporateActions
        ?index=equities&symbol=<SYM>&from_date=<dd-mm-yyyy>&to_date=<dd-mm-yyyy>

It needs a browser-like session cookie, which is obtained from the homepage
first. Each row carries the ex-date and a free-text "subject"; the ones that
matter say "Scheme Of Arrangement", "Demerger" or "Spin Off".
"""

from __future__ import annotations

import datetime as dt
import json
import time
import urllib.error
import urllib.request
from http.cookiejar import CookieJar

import pandas as pd

from . import config as cfg

API = ("https://www.nseindia.com/api/corporates-corporateActions"
       "?index=equities&symbol={sym}&from_date={lo}&to_date={hi}")
HOME = "https://www.nseindia.com/"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

CORPORATE_ACTIONS_CSV = cfg.CONFIG_DIR / "corporate_actions.csv"
RAW_ACTIONS_CSV = cfg.RAW_DIR / "nse_corporate_actions_raw.csv"

# Subjects that move the price without losing the holder any money.
SPLIT_SUBJECTS = ("demerger", "scheme of arrangement", "spin off", "spin-off",
                  "arrangement", "restructuring", "composite scheme")


HTML_HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
}
JSON_HEADERS = {
    "User-Agent": UA,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-actions",
    "X-Requested-With": "XMLHttpRequest",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
}


def _opener():
    """A cookie-bearing opener. NSE rejects anything that does not look like a
    browser, so the session cookie is picked up from the homepage first with
    document-style headers before any JSON call is made."""
    jar = CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    op.open(urllib.request.Request(HOME, headers=HTML_HEADERS), timeout=30).read()
    return op


def fetch_actions(symbols: list[str], start: dt.date, end: dt.date,
                  pause: float = 0.4) -> pd.DataFrame:
    """Every corporate action NSE has on record for these symbols."""
    op = _opener()
    rows, failed = [], []
    lo, hi = start.strftime("%d-%m-%Y"), end.strftime("%d-%m-%Y")

    for i, sym in enumerate(symbols, 1):
        url = API.format(sym=sym, lo=lo, hi=hi)
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers=JSON_HEADERS)
                raw = op.open(req, timeout=45).read()
                data = json.loads(raw)
                if isinstance(data, dict):
                    data = data.get("data", [])
                for rec in data:
                    rec["query_symbol"] = sym
                    rec["source_url"] = url
                    rows.append(rec)
                break
            except Exception:             # noqa: BLE001 - refresh session, retry
                if attempt == 2:
                    failed.append(sym)
                else:
                    time.sleep(2 * (attempt + 1))
                    try:
                        op = _opener()
                    except Exception:     # noqa: BLE001
                        pass
        time.sleep(pause)
        if i % 20 == 0:
            print(f"  {i}/{len(symbols)} symbols, {len(rows)} actions so far")

    if failed:
        print(f"  WARNING: no response for {len(failed)} symbols: {failed}")
    return pd.DataFrame(rows)


def is_price_splitting(subject: str) -> bool:
    """True for actions that mechanically cut the price without a loss."""
    s = str(subject).lower()
    return any(k in s for k in SPLIT_SUBJECTS)


def select_demergers(raw: pd.DataFrame) -> pd.DataFrame:
    """The subset that needs a price adjustment, tidied up."""
    if raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    df["subject"] = df["subject"].astype(str)
    df = df[df["subject"].map(is_price_splitting)]
    if df.empty:
        return df
    df["ex_date"] = pd.to_datetime(df["exDate"], format="%d-%b-%Y",
                                   errors="coerce").dt.date
    df = df.dropna(subset=["ex_date"])
    out = df[["query_symbol", "comp", "ex_date", "subject", "isin",
              "source_url"]].rename(columns={"query_symbol": "symbol",
                                             "comp": "company_name"})
    return out.drop_duplicates(["symbol", "ex_date", "subject"]).sort_values(
        ["ex_date", "symbol"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# adjusting the price history
# --------------------------------------------------------------------------

ADJUSTMENT_LOG_CSV = cfg.RESULTS_DIR / "phase5_demerger_adjustments.csv"


def adjustment_factor(prices: pd.Series, ex_date, window: int = 5
                      ) -> tuple[float | None, str]:
    """The factor the pre-ex-date history must be multiplied by, and how it was got.

    NSE publishes an adjustment factor for F&O contracts but not in any bulk
    machine-readable form, so the factor is taken from the price itself: the
    ratio of the ex-date close to the previous close. On a clean demerger that
    ratio IS the proportion of value that stayed with the parent, which is
    exactly the factor needed to put the earlier history on the same footing.

    Returns (factor, method). A factor of None means the event could not be
    measured - no price either side - and the caller must fall back to
    neutralising the ex-date return instead of scaling anything.
    """
    ex = pd.Timestamp(ex_date)
    before = prices.loc[:ex - pd.Timedelta(days=1)].dropna()
    on_after = prices.loc[ex:].dropna()
    if before.empty or on_after.empty:
        return None, "no price either side of the ex-date"
    if (on_after.index[0] - ex).days > window or (ex - before.index[-1]).days > window:
        return None, f"no trading within {window} days of the ex-date"
    prev, first = float(before.iloc[-1]), float(on_after.iloc[0])
    if prev <= 0 or first <= 0:
        return None, "non-positive price"
    return first / prev, "ex-date close / previous close"


def apply_demerger_adjustments(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    min_drop: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rescale pre-ex-date prices so demerger ex-dates stop being fake returns.

    Multiplying everything before the ex-date by (ex-date close / previous
    close) makes the ex-date return zero and leaves every other return in the
    series untouched, because a constant scaling cancels in a ratio.

    Events with a drop too small to distinguish from ordinary movement are
    recorded and left alone - scaling on noise would corrupt real returns.
    Events that cannot be measured at all are returned in `unadjusted`, and
    their ex-date return must be neutralised by the caller instead.

    Returns (adjusted prices, log of what was done, mask of ex-dates to treat
    as missing for signal purposes).
    """
    adjusted = prices.copy()
    log_rows = []
    neutralise = []

    for _, ev in events.iterrows():
        ticker = f"{ev['symbol']}.NS"
        if ticker not in adjusted.columns:
            log_rows.append({**_ev(ev), "method": "no price series in this universe",
                             "factor": None, "applied": False})
            continue
        series = adjusted[ticker]
        factor, how = adjustment_factor(series, ev["ex_date"])
        if factor is None:
            log_rows.append({**_ev(ev), "method": f"{how} - ex-date return neutralised",
                             "factor": None, "applied": False})
            neutralise.append({"ticker": ticker, "ex_date": pd.Timestamp(ev["ex_date"])})
            continue
        drop = 1.0 - factor
        # Only a FALL is adjusted, and the sign matters more than it looks.
        # A demerger always cuts the parent's price, because value leaves the
        # company - the factor is below 1 by construction. A scheme of
        # arrangement whose ex-date price went UP (Adani Enterprises rose 15%
        # and 21% on its two 2018 ex-dates) did not strip value out; scaling on
        # that would erase a real return and invent a fake one in its place.
        if drop <= min_drop:
            # Nothing to scale. The ex-date return is still not trustworthy, so
            # it falls back to the conservative treatment: missing for signals,
            # zero for holding returns.
            log_rows.append({
                **_ev(ev), "factor": round(factor, 6),
                "method": f"{how}; price rose {-100 * drop:.2f}% on the ex-date, so "
                          f"no value left the company - ex-date return neutralised "
                          f"instead of scaled",
                "applied": False})
            neutralise.append({"ticker": ticker, "ex_date": pd.Timestamp(ev["ex_date"])})
            continue
        ex = pd.Timestamp(ev["ex_date"])
        adjusted.loc[adjusted.index < ex, ticker] = (
            adjusted.loc[adjusted.index < ex, ticker] * factor)
        neutralise.append({"ticker": ticker, "ex_date": ex})
        log_rows.append({**_ev(ev), "factor": round(factor, 6), "method": how,
                         "applied": True})

    return adjusted, pd.DataFrame(log_rows), pd.DataFrame(neutralise)


def _ev(ev) -> dict:
    return {"symbol": ev["symbol"], "ex_date": str(ev["ex_date"]),
            "subject": ev["subject"], "source_url": ev["source_url"]}
