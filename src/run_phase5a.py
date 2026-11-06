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


def _probe(urls: list[tuple[str, str]], workers: int, label: str) -> list[dict]:
    """HEAD every (date, url) pair and keep the ones that serve a real PDF."""
    def check(item):
        date, url = item
        size = IC.is_real_pdf(url)
        return None if size is None else {"date": date, "url": url, "bytes": size}

    found = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, res in enumerate(ex.map(check, urls)):
            if res:
                found.append(res)
            if i and i % 1000 == 0:
                print(f"  [{label}] {i}/{len(urls)} checked, {len(found)} found")
    return found


def discover(start: dt.date, end: dt.date, workers: int = 12) -> pd.DataFrame:
    """Find every press-release PDF, including same-day revisions.

    NSE sometimes issues more than one release on a date, and the later one
    gets a "_1" suffix: ind_prs24022022_1.pdf. Those are not optional extras -
    the March 2022 Nifty 50 replacement (Indian Oil out, Apollo Hospitals in)
    exists ONLY in the suffixed file. Probing the bare filename alone misses
    that whole class of document, and the gap is invisible unless the
    reconstructed membership count is checked against 50.
    """
    days = [start + dt.timedelta(n) for n in range((end - start).days + 1)]
    base = [(d.isoformat(), IC.PRESS_RELEASE_URL.format(d)) for d in days]
    print(f"Checking {len(base)} candidate dates ({start} to {end})...")
    found = _probe(base, workers, "base")
    print(f"  {len(found)} unsuffixed PDFs")

    # Same-day revisions, chased until a suffix comes back empty.
    suffix, alive = 1, [d.isoformat() for d in days]
    while alive and suffix <= 4:
        cand = [(d, IC.PRESS_RELEASE_URL.format(dt.date.fromisoformat(d))
                 .replace(".pdf", f"_{suffix}.pdf")) for d in alive]
        got = _probe(cand, workers, f"_{suffix}")
        print(f"  {len(got)} PDFs with suffix _{suffix}")
        found.extend(got)
        alive = [g["date"] for g in got]
        suffix += 1

    df = pd.DataFrame(found).sort_values(["date", "url"]).reset_index(drop=True)
    print(f"Found {len(df)} real press-release PDFs in total.")
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

    # Rows from the earliest releases have no symbol (those tables have no
    # Symbol column), so fall back to the company name rather than show a dash.
    label = df["symbol"].fillna("").where(
        df["symbol"].fillna("").str.len() > 0, df["company_name"])
    shown = df.assign(label=label)

    n_no_symbol = int((df["symbol"].fillna("").str.len() == 0).sum())
    print(f"\n  Rows with no ticker in the source (name only): {n_no_symbol}"
          f" of {len(df)}")

    gaps = sorted(set(df["effective_date"]))
    print(f"\n  {len(gaps)} distinct effective dates:")
    for d in gaps:
        sub = shown[shown["effective_date"] == d]
        out = ", ".join(sub[sub["action"] == "exclude"]["label"])
        inn = ", ".join(sub[sub["action"] == "include"]["label"])
        print(f"    {d}")
        print(f"        out: {out or '-'}")
        print(f"         in: {inn or '-'}")

    print(f"\n  Changes -> {IC.CHANGES_CSV}")
    if failed:
        print(f"\n  {len(failed)} downloads failed:")
        for f in failed[:10]:
            print(f"    {f['url']}: {f['error']}")


if __name__ == "__main__":
    main()
