"""Phase 5a: discover NSE press releases and build the Nifty 50 index-change list.

    python -m src.run_phase5a --discover    # scan for real PDFs (slow, polite)
    python -m src.run_phase5a               # parse the cached list

Writes config/press_release_index.csv (every real PDF found, with its size) and
config/index_changes.csv (one row per Nifty 50 inclusion or exclusion, each
carrying the URL it was read from and a quote from that document).

Nothing here is taken on trust: a URL is only recorded after its PDF has been
downloaded and the change has been read out of the text.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt

import pandas as pd

from . import config as cfg
from . import index_changes as IC

SCAN_START = dt.date(2009, 1, 1)


def discover(start: dt.date, end: dt.date, workers: int = 12) -> pd.DataFrame:
    """Find every date whose press-release URL serves an actual PDF."""
    days = [start + dt.timedelta(n) for n in range((end - start).days + 1)]
    print(f"Checking {len(days)} candidate dates "
          f"({start} to {end})...")

    def check(d: dt.date):
        url = IC.PRESS_RELEASE_URL.format(d)
        size = IC.is_real_pdf(url)
        return None if size is None else {"date": d.isoformat(), "url": url,
                                          "bytes": size}

    found = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, res in enumerate(ex.map(check, days)):
            if res:
                found.append(res)
            if i and i % 1000 == 0:
                print(f"  {i}/{len(days)} checked, {len(found)} real PDFs so far")
    df = pd.DataFrame(found)
    print(f"Found {len(df)} real press-release PDFs.")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 5a: index change list")
    ap.add_argument("--discover", action="store_true",
                    help="rescan niftyindices.com for press-release PDFs")
    ap.add_argument("--end", default=None, help="last date to scan (YYYY-MM-DD)")
    ap.add_argument("--discover-only", action="store_true",
                    help="stop after building the press-release index")
    args = ap.parse_args()
    if args.discover_only:
        args.discover = True

    cfg.ensure_dirs()
    end = dt.date.fromisoformat(args.end) if args.end else dt.date.today()

    if args.discover or not IC.RELEASE_INDEX_CSV.exists():
        index = discover(SCAN_START, end)
        index.to_csv(IC.RELEASE_INDEX_CSV, index=False)
    else:
        index = pd.read_csv(IC.RELEASE_INDEX_CSV)
    print(f"\nPress-release index: {len(index)} PDFs "
          f"-> {IC.RELEASE_INDEX_CSV}")
    if len(index):
        by_year = index.assign(year=index["date"].str[:4]).groupby("year").agg(
            n=("url", "size"), total_mb=("bytes", lambda s: round(s.sum() / 1e6, 1)))
        print(by_year.to_string())
        print(f"  total download: {index['bytes'].sum() / 1e6:.0f} MB")
    if args.discover_only:
        return

    # ---- download everything first, in parallel ---------------------------
    print(f"\nDownloading/caching {len(index)} PDFs...")

    def prefetch(url: str):
        try:
            IC.download_pdf(url)
        except Exception:                   # noqa: BLE001 - retried below
            pass

    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for i, _ in enumerate(ex.map(prefetch, index["url"].tolist()), 1):
            if i % 200 == 0:
                print(f"  {i}/{len(index)} fetched")

    # ---- parse every release ----------------------------------------------
    # Two passes: a cheap pypdf screen to find documents that mention the index
    # and a replacement at all, then the slower coordinate-based extraction on
    # those, which is the only one whose reading order can be trusted.
    all_changes, failed, with_changes, screened = [], [], 0, 0
    for i, row in enumerate(index.itertuples(), 1):
        try:
            quick = IC.fetch_pdf_text(row.url).lower()
        except Exception as err:            # noqa: BLE001 - record and continue
            failed.append({"url": row.url, "error": f"screen: {err}"})
            continue
        if "nifty" not in quick or ("being excluded" not in quick
                                    and "being included" not in quick):
            continue
        screened += 1
        try:
            text = IC.fetch_pdf_layout_text(row.url)
        except Exception as err:            # noqa: BLE001
            failed.append({"url": row.url, "error": f"layout: {err}"})
            continue
        changes = IC.parse_release(text, row.url)
        if changes:
            with_changes += 1
            all_changes.extend(changes)
        if screened % 25 == 0:
            print(f"  {i}/{len(index)} checked, {screened} candidates, "
                  f"{len(all_changes)} Nifty 50 changes so far")

    df = IC.changes_to_frame(all_changes)
    df.to_csv(IC.CHANGES_CSV, index=False)

    print(f"\n{'=' * 90}\nPHASE 5a: Nifty 50 index-change list\n{'=' * 90}")
    print(f"\n  Press releases checked:        {len(index)}")
    print(f"  Passed the keyword screen:      {screened}")
    print(f"  Releases with Nifty 50 changes: {with_changes}")
    print(f"  Download/parse failures:        {len(failed)}")
    print(f"  Individual changes recorded:    {len(df)}")
    if df.empty:
        print("\n  NOTHING PARSED - stopping, this needs looking at.")
        return

    print(f"  Effective dates span:           {df['effective_date'].min()} "
          f"to {df['effective_date'].max()}")
    print(f"  Every row has a source URL:     {bool(df['source_url'].notna().all())}")

    # ---- consistency checks ------------------------------------------------
    print(f"\n{'-' * 90}\nConsistency checks\n{'-' * 90}\n")
    per_date = df.pivot_table(index="effective_date", columns="action",
                              values="symbol", aggfunc="count").fillna(0).astype(int)
    for col in ("exclude", "include"):
        if col not in per_date:
            per_date[col] = 0
    unbalanced = per_date[per_date["exclude"] != per_date["include"]]
    print(f"  Dates where #excluded != #included: {len(unbalanced)}")
    if len(unbalanced):
        print("    (a replacement should be one-for-one; these need a look)")
        print(unbalanced.to_string())

    by_year = df.assign(year=df["effective_date"].str[:4]).groupby(
        ["year", "action"]).size().unstack(fill_value=0)
    print(f"\n  Changes by year and action:\n")
    print(by_year.to_string())

    gaps = sorted(set(df["effective_date"]))
    print(f"\n  {len(gaps)} distinct effective dates:")
    for d in gaps:
        sub = df[df["effective_date"] == d]
        out = ", ".join(sub[sub["action"] == "exclude"]["symbol"])
        inn = ", ".join(sub[sub["action"] == "include"]["symbol"])
        print(f"    {d}   out: {out or '-':40s} in: {inn or '-'}")

    print(f"\n  Changes -> {IC.CHANGES_CSV}")
    if failed:
        print(f"\n  {len(failed)} downloads failed:")
        for f in failed[:10]:
            print(f"    {f['url']}: {f['error']}")


if __name__ == "__main__":
    main()
