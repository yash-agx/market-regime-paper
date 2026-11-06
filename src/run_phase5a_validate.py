"""Phase 5a validation: supersede, resolve tickers, and audit completeness.

    python -m src.run_phase5a_validate

Takes the raw parsed changes and does four things:

  1.  Marks superseded announcements. The same replacement can be announced
      twice - YES Bank was scheduled out for 27 March 2020 in the February
      review, then pulled forward to 19 March after the RBI moratorium.
      Both documents are real and both stay in the file, but only the one that
      took effect is applied.

  2.  Resolves the pre-2012 rows, whose tables carry no Symbol column, to
      tickers using names and symbols read out of NSE's own documents.
      Anything that cannot be resolved from a source is listed, not guessed.

  3.  CHECK ONE - rebuilds membership backwards from today's constituents and
      verifies the count never leaves 50.

  4.  CHECK TWO - lists every semi-annual review window and says whether a
      Nifty 50 change was found in it. The count check cannot see a review
      where BOTH sides are missing, because the count stays at 50; this is the
      only check that can.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from . import config as cfg
from . import index_changes as IC

# Replacements announced twice: (effective_date, symbol) that did NOT happen.
# The YES Bank periodic review scheduled 27 March 2020 was overtaken by the
# accelerated removal on 19 March after the RBI moratorium.
SUPERSEDED_DATES = {"2020-03-27"}

RESOLVED_CSV = cfg.CONFIG_DIR / "index_changes_resolved.csv"
UNRESOLVED_CSV = cfg.RESULTS_DIR / "phase5a_unresolved_names.csv"
REVIEW_AUDIT_CSV = cfg.RESULTS_DIR / "phase5a_review_audit.csv"
MEMBERSHIP_CSV = cfg.RESULTS_DIR / "phase5a_membership_check.csv"


def mark_superseded(df: pd.DataFrame) -> pd.DataFrame:
    """Flag announcements that a later announcement overtook."""
    df = df.copy()
    df["superseded"] = df["effective_date"].isin(SUPERSEDED_DATES)
    return df


def resolve_tickers(df: pd.DataFrame, urls: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill in the missing symbols from NSE sources; report what is left over."""
    df = df.copy()
    df["symbol"] = df["symbol"].fillna("").astype(str).str.strip()

    print("Building a name -> ticker map from NSE documents...")
    from_releases = IC.build_name_symbol_map(urls)
    from_equity = IC.load_nse_equity_names()
    print(f"  {len(from_releases)} names from press-release tables, "
          f"{len(from_equity)} from the NSE equity list")

    sym, src = [], []
    for _, row in df.iterrows():
        if row["symbol"]:
            sym.append(row["symbol"])
            src.append("in the source document")
            continue
        key = IC.normalise_company(row["company_name"])
        hit = from_releases.get(key) or from_equity.get(key)
        sym.append(hit[0] if hit else "")
        src.append(hit[1] if hit else "")
    df["symbol_resolved"] = sym
    df["symbol_source"] = src

    unresolved = df[df["symbol_resolved"] == ""][
        ["effective_date", "action", "company_name", "source_url"]]
    return df, unresolved


# --------------------------------------------------------------------------
# CHECK ONE: does the membership count stay at 50?
# --------------------------------------------------------------------------

# Between these dates the index carried 51 securities: Tata Motors DVR was a
# second listed line of a company already in the index. Added 2016-04-01
# (three out, four in) and removed 2017-09-29 (four out, three in).
DVR_FROM, DVR_TO = "2016-04-01", "2017-09-29"


def membership_check(df: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    """Walk backwards from today's constituents, undoing each change."""
    members = set(pd.read_csv(cfg.UNIVERSE_CSV)["symbol"])
    rows, problems = [], []

    applied = df[~df["superseded"]]
    for d in sorted(applied["effective_date"].unique(), reverse=True):
        if d > dt.date.today().isoformat():
            continue
        sub = applied[applied["effective_date"] == d]
        inc = [aliases.get(s, s) for s in sub[sub.action == "include"]["symbol_resolved"] if s]
        exc = [aliases.get(s, s) for s in sub[sub.action == "exclude"]["symbol_resolved"] if s]
        for s in inc:                      # undo an inclusion by removing it
            if s in members:
                members.discard(s)
            else:
                problems.append(f"{d}: cannot remove {s} - was not a member")
        for s in exc:                      # undo an exclusion by adding it back
            if s in members:
                problems.append(f"{d}: {s} was already a member")
            else:
                members.add(s)
        expected = 51 if DVR_FROM <= d <= DVR_TO else 50
        rows.append({"before_date": d, "n_members": len(members),
                     "expected": expected, "ok": len(members) == expected,
                     "undid_out": len(exc), "undid_in": len(inc)})
    out = pd.DataFrame(rows)
    out.attrs["problems"] = problems
    return out


# --------------------------------------------------------------------------
# CHECK TWO: was every semi-annual review accounted for?
# --------------------------------------------------------------------------


def review_audit(df: pd.DataFrame) -> pd.DataFrame:
    """One row per semi-annual review window, 2009-2025.

    NSE reviews the index twice a year, with changes taking effect at the end
    of March and the end of September. A review that produced a change shows
    up in the changes file; a review that produced none leaves no trace there,
    and is indistinguishable from a review whose change we failed to parse.
    Both cases are listed so the difference has to be looked at rather than
    assumed.
    """
    applied = df[~df["superseded"]]
    eff = pd.to_datetime(applied["effective_date"])
    rows = []
    for year in range(2009, 2026):
        for label, lo, hi in (
            ("March review", f"{year}-02-15", f"{year}-05-31"),
            ("September review", f"{year}-08-15", f"{year}-11-30"),
        ):
            mask = (eff >= pd.Timestamp(lo)) & (eff <= pd.Timestamp(hi))
            hit = applied[mask.to_numpy()]
            rows.append({
                "year": year, "review": label,
                "window": f"{lo} to {hi}",
                "changes_found": len(hit),
                "effective_dates": ", ".join(sorted(hit["effective_date"].unique())),
                "out": ", ".join(hit[hit.action == "exclude"]["symbol_resolved"]),
                "in": ", ".join(hit[hit.action == "include"]["symbol_resolved"]),
            })
    return pd.DataFrame(rows)


def main() -> None:
    cfg.ensure_dirs()
    df = pd.read_csv(IC.CHANGES_CSV)
    index = pd.read_csv(IC.RELEASE_INDEX_CSV)

    df = mark_superseded(df)
    df, unresolved = resolve_tickers(df, index["url"].tolist())
    df.to_csv(RESOLVED_CSV, index=False)
    unresolved.to_csv(UNRESOLVED_CSV, index=False)

    aliases = {}
    if IC.TICKER_STATUS_CSV.exists():
        ts = pd.read_csv(IC.TICKER_STATUS_CSV)
        aliases = dict(zip(ts["old_symbol"], ts["new_symbol"]))

    print(f"\n{'=' * 92}\nPHASE 5a VALIDATION\n{'=' * 92}")
    print(f"\n  Changes in file:            {len(df)}")
    print(f"  Marked superseded:          {int(df['superseded'].sum())}")
    print(f"  Had a ticker in the source: {int((df['symbol'] != '').sum())}")
    print(f"  Ticker resolved from NSE:   "
          f"{int(((df['symbol'] == '') & (df['symbol_resolved'] != '')).sum())}")
    print(f"  Still unresolved:           {len(unresolved)}")
    if len(unresolved):
        print("\n  UNRESOLVED (listed, never guessed):")
        for _, r in unresolved.iterrows():
            print(f"    {r['effective_date']}  {r['action']:8s} {r['company_name']}")

    mem = membership_check(df, aliases)
    mem.to_csv(MEMBERSHIP_CSV, index=False)
    bad = mem[~mem["ok"]]
    print(f"\n{'-' * 92}\nCHECK 1 - membership count walking backwards from today"
          f"\n{'-' * 92}")
    print(f"\n  Steps checked:        {len(mem)}")
    print(f"  Steps off-count:      {len(bad)}")
    if len(bad):
        print(bad.to_string(index=False))
    for p in mem.attrs["problems"]:
        print(f"    PROBLEM {p}")
    if not len(bad) and not mem.attrs["problems"]:
        print("  The index holds exactly 50 members at every step "
              "(51 while Tata Motors DVR was listed).")

    audit = review_audit(df)
    audit.to_csv(REVIEW_AUDIT_CSV, index=False)
    missing = audit[audit["changes_found"] == 0]
    print(f"\n{'-' * 92}\nCHECK 2 - every semi-annual review, 2009-2025"
          f"\n{'-' * 92}\n")
    print(f"  Review windows:            {len(audit)}")
    print(f"  With a change found:       {len(audit) - len(missing)}")
    print(f"  With NO change found:      {len(missing)}")
    print("\n  Reviews with no Nifty 50 change in the file - each is either a")
    print("  review that genuinely changed nothing, or one we failed to find:\n")
    for _, r in missing.iterrows():
        print(f"    {r['year']}  {r['review']}")

    print(f"\n  Resolved changes -> {RESOLVED_CSV}")
    print(f"  Review audit     -> {REVIEW_AUDIT_CSV}")


if __name__ == "__main__":
    main()
