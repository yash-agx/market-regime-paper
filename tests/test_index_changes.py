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
    key = ["effective_date", "action", "symbol"]
    assert not changes.duplicated(key).any()
