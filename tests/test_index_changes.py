"""Phase 5a: parsing Nifty 50 changes out of NSE press releases.

The parsing tests run on text laid out exactly like the real documents, so the
awkward cases - "Nifty 500" sitting next to "Nifty 50", the numbering style
changing between years, two Nifty 50 sections in one release - are pinned down
without needing the network.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import index_changes as IC

RELEASE_2024 = """
PRESS RELEASE
Mumbai, February 28, 2024
Replacements in indices

The Index Maintenance Sub -Committee (Equity) of NSE Indices Limited has decided to make the
replacement of stocks in various indices as part of its periodic review as listed hereunder. These
changes shall become effective from March 28, 2024 (close of March 27, 2024).

A. Replacements on account of semi-annual review of broad market indices:

1) Nifty 50

The following company is being excluded:

Sr. No. Company Name Symbol
1 UPL Ltd. UPL

The following company is being included:

Sr. No. Company Name Symbol
1 Shriram Finance Ltd. SHRIRAMFIN

The above replacement will also be applicable to Nifty50 Equal Weight index.

2) Nifty Next 50

The following companies are being excluded:

Sr. No. Company Name Symbol
1 Adani Wilmar Ltd. AWL
"""

RELEASE_2025 = """
PRESS RELEASE
Mumbai, February 21, 2025
Replacements in indices

changes shall become effective from March 28, 2025 (close of March 27, 2025).

1) Replacements on account of semi-annual review of broad market indices:

a) Nifty 50

The following companies are being excluded:

Sr. No. Company Name Symbol
1 Bharat Petroleum Corporation Ltd. BPCL
2 Britannia Industries Ltd. BRITANNIA

The following companies are being included:

Sr. No. Company Name Symbol
1 Jio Financial Services Ltd. JIOFIN
2 Zomato Ltd. ZOMATO

b) Nifty 500

The following companies are being excluded:

Sr. No. Company Name Symbol
1 Archean Chemical Industries Ltd. ACI
"""


RELEASE_2009 = """
IISL
INDIA INDEX SERVICES & PRODUCTS LIMITED
PRESS RELEASE

The Index Maintenance Sub-Committee has decided to make the following changes in various
indices during its review. These changes shall become effective from January 12, 2009.

(1) S&P CNX Nifty Index

The following company is being excluded:

Sr. No. Company Name
1 Satyam Computer Services Ltd.

The following company is being included:

Sr. No. Company Name
1 Reliance Capital Ltd.

(2) CNX Nifty Junior Index

The following company is being excluded:

Sr. No. Company Name
1 Reliance Capital Ltd.

The following company is being included:

Sr. No. Company Name
1 Glenmark Pharmaceuticals Ltd.

(3) CNX 100 Index

The following company is being excluded:

Sr. No. Company Name
1 Satyam Computer Services Ltd.

(4) S&P CNX 500 Index

The following company is being excluded:

Sr. No. Company Name
1 Some Other Co Ltd.

Place: Mumbai
Date: January 7, 2009.
"""


def test_parses_the_2009_s_and_p_cnx_nifty_name():
    """The index was called "S&P CNX Nifty" until late 2013. A parser that only
    knows "Nifty 50" silently finds nothing before 2014."""
    out = IC.parse_release(RELEASE_2009, "http://example/2009.pdf")
    assert len(out) == 2
    by = {c.action: c for c in out}
    assert by["exclude"].company_name == "Satyam Computer Services Ltd"
    assert by["include"].company_name == "Reliance Capital Ltd"
    assert by["exclude"].effective_date == "2009-01-12"
    assert by["exclude"].announcement_date == "2009-01-07"


def test_pre_2018_rows_have_no_symbol_and_say_so():
    out = IC.parse_release(RELEASE_2009, "http://example/2009.pdf")
    assert all(c.symbol == "" for c in out)


def test_cnx_nifty_junior_is_not_mistaken_for_the_nifty():
    """The nastiest near-miss in the old documents."""
    out = IC.parse_release(RELEASE_2009, "http://example/2009.pdf")
    names = {c.company_name for c in out}
    assert "Glenmark Pharmaceuticals Ltd" not in names


def test_cnx_100_and_s_and_p_cnx_500_are_not_the_nifty():
    out = IC.parse_release(RELEASE_2009, "http://example/2009.pdf")
    assert "Some Other Co Ltd" not in {c.company_name for c in out}


@pytest.mark.parametrize("heading,expected", [
    ("(1) S&P CNX Nifty Index", True),
    ("(2) CNX Nifty Index", True),
    ("1) Nifty 50", True),
    ("a) Nifty 50", True),
    ("b) Nifty 50 Index", True),
    ("(2) CNX Nifty Junior Index", False),
    ("2) Nifty Next 50", False),
    ("b) Nifty 500", False),
    ("(4) S&P CNX 500 Index", False),
    ("(3) CNX 100 Index", False),
    ("1) Nifty Bank", False),
    ("The above also applies to Nifty50 Equal Weight index.", False),
])
def test_index_heading_recognition(heading, expected):
    assert bool(IC.NIFTY50_HEAD_RE.match(heading)) is expected


RELEASE_DETACHED_SERIALS = """
PRESS RELEASE
Mumbai, August 23, 2024
effective from September 30, 2024 (close of September 27, 2024).

a) Nifty 50
The following companies are being excluded:
Sr. No. Company Name Symbol
Divi's Laboratories Ltd. DIVISLAB
1
2 LTIMindtree Ltd. LTIM
The following companies are being included:
Sr. No. Company Name Symbol
Bharat Electronics Ltd. BEL
1
2 Trent Ltd. TRENT
Note:
1. Trent Ltd. (average free-float market capitalization Rs. 99,364 crores) and Bharat Electronics Ltd.
have been included in Nifty 50 index as the 6-month average free-float market capitalization
b) Nifty Next 50
The following companies are being excluded:
Sr. No. Company Name Symbol
1 Berger Paints India Ltd. BERGEPAINT
"""


def test_a_serial_number_stranded_on_its_own_line_does_not_lose_the_row():
    """Coordinate extraction splits some rows, putting the serial number on its
    own line. Requiring it dropped the first row of the table, and because an
    unnumbered row looks like prose, it also truncated the rest."""
    out = IC.parse_release(RELEASE_DETACHED_SERIALS, "http://example/f.pdf")
    assert {c.symbol for c in out if c.action == "exclude"} == {"DIVISLAB", "LTIM"}
    assert {c.symbol for c in out if c.action == "include"} == {"BEL", "TRENT"}


def test_explanatory_notes_are_not_mistaken_for_table_rows():
    """The notes are prose that can end in capitals; they must not become rows."""
    out = IC.parse_release(RELEASE_DETACHED_SERIALS, "http://example/f.pdf")
    assert len(out) == 4
    assert "BERGEPAINT" not in {c.symbol for c in out}


@pytest.mark.parametrize("name,ok", [
    ("Trent Ltd", True),
    ("Divi's Laboratories Ltd.", True),
    ("Bharat Petroleum Corporation Ltd.", True),
    ("1. Trent Ltd. (average free-float market capitalization Rs. 99,364 crores) and", False),
    ("have been included in Nifty 50 index as the 6-month average free-float", False),
    ("", False),
])
def test_company_name_plausibility(name, ok):
    assert IC._plausible_company(name) is ok


def test_parses_the_2024_numbering_style():
    out = IC.parse_release(RELEASE_2024, "http://example/a.pdf")
    assert len(out) == 2
    by = {c.action: c for c in out}
    assert by["exclude"].symbol == "UPL"
    assert by["include"].symbol == "SHRIRAMFIN"
    assert by["include"].company_name == "Shriram Finance Ltd"


def test_parses_the_2025_lettered_numbering_style():
    """The heading went from '1) Nifty 50' to 'a) Nifty 50' between years."""
    out = IC.parse_release(RELEASE_2025, "http://example/b.pdf")
    assert {c.symbol for c in out} == {"BPCL", "BRITANNIA", "JIOFIN", "ZOMATO"}
    assert sum(c.action == "exclude" for c in out) == 2
    assert sum(c.action == "include" for c in out) == 2


def test_nifty_500_is_not_mistaken_for_nifty_50():
    """The single most dangerous confusion in this parser."""
    out = IC.parse_release(RELEASE_2025, "http://example/b.pdf")
    assert "ACI" not in {c.symbol for c in out}


def test_nifty_next_50_is_not_mistaken_for_nifty_50():
    out = IC.parse_release(RELEASE_2024, "http://example/a.pdf")
    assert "AWL" not in {c.symbol for c in out}


def test_effective_date_is_read_from_the_document_not_assumed():
    out = IC.parse_release(RELEASE_2024, "http://example/a.pdf")
    assert all(c.effective_date == "2024-03-28" for c in out)
    assert all(c.announcement_date == "2024-02-28" for c in out)


def test_every_change_carries_its_source_url_and_a_quote():
    out = IC.parse_release(RELEASE_2024, "http://example/a.pdf")
    for c in out:
        assert c.source_url == "http://example/a.pdf"
        assert c.source_quote and "effective from" in c.source_quote.lower()


def test_a_release_with_no_nifty_50_section_yields_nothing():
    text = RELEASE_2024.replace("1) Nifty 50", "1) Nifty Bank")
    assert IC.parse_release(text, "http://example/c.pdf") == []


def test_two_nifty_50_sections_are_both_read():
    """A release can carry a periodic review and an ad-hoc change separately."""
    text = RELEASE_2024 + """
B. Replacements on account of corporate action:

1) Nifty 50

The following company is being excluded:

Sr. No. Company Name Symbol
1 Some Merged Co Ltd. MERGED

The following company is being included:

Sr. No. Company Name Symbol
1 Some New Co Ltd. NEWCO
"""
    out = IC.parse_release(text, "http://example/d.pdf")
    assert {"UPL", "SHRIRAMFIN", "MERGED", "NEWCO"} == {c.symbol for c in out}


def test_duplicate_rows_are_not_double_counted():
    text = RELEASE_2024 + RELEASE_2024.split("A. Replacements")[1]
    out = IC.parse_release(text, "http://example/e.pdf")
    symbols = [c.symbol for c in out]
    assert len(symbols) == len(set(symbols))


def test_changes_to_frame_sorts_and_handles_empty():
    assert IC.changes_to_frame([]).empty
    out = IC.parse_release(RELEASE_2025, "http://example/b.pdf")
    df = IC.changes_to_frame(out)
    assert list(df["effective_date"].unique()) == ["2025-03-28"]
    assert df["action"].tolist() == sorted(df["action"].tolist())


# --------------------------------------------------------------------------
# the soft-404 trap
# --------------------------------------------------------------------------


def test_is_real_pdf_requires_a_pdf_content_type(monkeypatch):
    """niftyindices.com answers 200 for URLs that do not exist, including
    February 31st. Only the Content-Type tells the truth."""

    class FakeResp:
        def __init__(self, ctype, length="123"):
            self.headers = {"Content-Type": ctype, "Content-Length": length}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_open(req, timeout=20):
        if "real" in req.full_url:
            return FakeResp("application/pdf")
        return FakeResp("text/html; charset=utf-8")

    monkeypatch.setattr(IC.urllib.request, "urlopen", fake_open)
    assert IC.is_real_pdf("http://x/real.pdf") == 123
    assert IC.is_real_pdf("http://x/ind_prs31022012.pdf") is None


def test_is_real_pdf_survives_a_network_error(monkeypatch):
    def boom(req, timeout=20):
        raise OSError("connection reset")

    monkeypatch.setattr(IC.urllib.request, "urlopen", boom)
    assert IC.is_real_pdf("http://x/y.pdf") is None


RELEASE_WRAPPED_MARKER = """
PRESS RELEASE
Mumbai, August 18, 2010
These changes shall become effective from October 1, 2010.

(1) S&P CNX Nifty Index

The following companies are being excluded:

Sr. No. Company Name
1 ABB Ltd.
2 Idea Cellular Ltd.
3 Unitech Ltd.

included:
The following companies are being

Sr. No. Company Name
1 Bajaj Auto Ltd.
2 Dr. Reddy's Laboratories Ltd.
3 Sesa Goa Ltd.

(2) CNX Nifty Junior Index
"""


def test_a_marker_split_across_lines_is_still_recognised():
    """Coordinate grouping can split "The following companies are being
    included:" and put the tail ABOVE the rest. The marker is then on no single
    line, the table below it is never read, and the release reports a
    replacement in one direction only."""
    out = IC.parse_release(RELEASE_WRAPPED_MARKER, "http://example/g.pdf")
    excluded = {c.company_name for c in out if c.action == "exclude"}
    included = {c.company_name for c in out if c.action == "include"}
    assert excluded == {"ABB Ltd", "Idea Cellular Ltd", "Unitech Ltd"}
    assert included == {"Bajaj Auto Ltd", "Dr. Reddy's Laboratories Ltd",
                        "Sesa Goa Ltd"}


def test_rows_without_symbols_are_not_collapsed_together():
    """Pre-2012 rows all have an empty symbol. De-duplicating on the symbol
    alone threw away every row of a table after the first."""
    out = IC.parse_release(RELEASE_WRAPPED_MARKER, "http://example/g.pdf")
    assert len([c for c in out if c.action == "exclude"]) == 3
    assert all(c.symbol == "" for c in out)


# --------------------------------------------------------------------------
# the doubled-glyph PDFs, against the real cached document
# --------------------------------------------------------------------------


def test_doubled_glyph_pdf_is_readable():
    """Some 2012 releases fake bold by drawing every glyph twice, which
    extracts as "SS&&PP CCNNXX NNiiffttyy IInnddeexx" and matches nothing.
    Without character de-duplication the whole of 2012 goes missing."""
    url = "https://www.niftyindices.com/Press_Release/ind_prs14032012.pdf"
    cached = IC.PDF_CACHE / url.rsplit("/", 1)[-1]
    if not cached.exists():
        pytest.skip("press release not cached - run `python -m src.run_phase5a`")
    text = IC.fetch_pdf_layout_text(url)
    assert "S&P CNX Nifty Index" in text
    assert "SS&&PP" not in text
    out = IC.parse_release(text, url)
    assert {c.symbol for c in out if c.action == "exclude"} == {"RCOM", "RPOWER"}
    assert {c.symbol for c in out if c.action == "include"} == {"ASIANPAINT",
                                                                "BANKBARODA"}


# --------------------------------------------------------------------------
# the committed file, if it exists
# --------------------------------------------------------------------------


@pytest.fixture
def changes():
    if not IC.CHANGES_CSV.exists():
        pytest.skip("run `python -m src.run_phase5a` first")
    return pd.read_csv(IC.CHANGES_CSV)


def test_committed_changes_all_have_a_source_url(changes):
    assert changes["source_url"].notna().all()
    assert changes["source_url"].str.startswith("http").all()


def test_committed_changes_have_valid_actions_and_dates(changes):
    assert set(changes["action"]) <= {"include", "exclude"}
    assert pd.to_datetime(changes["effective_date"]).notna().all()
    # a change cannot take effect before it is announced
    assert (pd.to_datetime(changes["effective_date"])
            >= pd.to_datetime(changes["announcement_date"])).all()


def test_committed_changes_have_no_duplicate_rows(changes):
    # The company name is part of the key: pre-2012 rows have no symbol at all,
    # so keying on the symbol alone makes every name-only row in a table look
    # like a duplicate of the others.
    key = ["effective_date", "action", "symbol", "company_name"]
    assert not changes.duplicated(key).any()
