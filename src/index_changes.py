"""Phase 5a: build the Nifty 50 index-change list from NSE press releases.

The rule for this phase is that every change must come from an NSE
announcement with a source URL, and no URL may be guessed. That is harder than
it sounds, for two reasons found while building this:

  * niftyindices.com serves HTTP 200 for every URL under /Press_Release/,
    including dates that do not exist (February 31st). A 200 therefore proves
    nothing. Only a response whose Content-Type is application/pdf is a real
    document, and that is the test used here.

  * The press releases are not listed anywhere machine-readable, so the set of
    real documents is discovered by checking each date and keeping the ones
    that return an actual PDF. A URL only enters the changes file after its PDF
    has been downloaded and the text supporting the change has been read out of
    it - so every row carries a quote from the document it came from.

The press releases are highly regular:

    These changes shall become effective from March 28, 2024 (close of ...).
    ...
    1) Nifty 50
    The following company is being excluded:
    Sr. No. Company Name Symbol
    1 UPL Ltd. UPL
    The following company is being included:
    Sr. No. Company Name Symbol
    1 Shriram Finance Ltd. SHRIRAMFIN

so the Nifty 50 section is located, the excluded and included lists are read,
and the effective date is taken from the document rather than assumed.
"""

from __future__ import annotations

import datetime as dt
import io
import re
import urllib.request
from dataclasses import dataclass, asdict

import pandas as pd

from . import config as cfg

PRESS_RELEASE_URL = "https://www.niftyindices.com/Press_Release/ind_prs{:%d%m%Y}.pdf"
PDF_CACHE = cfg.RAW_DIR / "press_releases"
RELEASE_INDEX_CSV = cfg.CONFIG_DIR / "press_release_index.csv"
CHANGES_CSV = cfg.CONFIG_DIR / "index_changes.csv"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0 Safari/537.36"


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------


def is_real_pdf(url: str, timeout: int = 20) -> int | None:
    """Content-Length if the URL serves an actual PDF, else None.

    Status code is deliberately ignored: this host answers 200 for everything.
    """
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if "pdf" in resp.headers.get("Content-Type", "").lower():
                return int(resp.headers.get("Content-Length", 0))
    except Exception:
        return None
    return None


def fetch_pdf_text(url: str) -> str:
    """Download (and cache) a press release, returning its text."""
    from pypdf import PdfReader

    PDF_CACHE.mkdir(parents=True, exist_ok=True)
    path = PDF_CACHE / url.rsplit("/", 1)[-1]
    if not path.exists():
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as resp:
            if "pdf" not in resp.headers.get("Content-Type", "").lower():
                raise ValueError(f"not a PDF: {url}")
            path.write_bytes(resp.read())
    reader = PdfReader(io.BytesIO(path.read_bytes()))
    return "\n".join((p.extract_text() or "") for p in reader.pages)


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

MONTHS = ("January February March April May June July August September "
          "October November December").split()
DATE_RE = re.compile(
    r"(" + "|".join(MONTHS) + r")\s+(\d{1,2})\s*,\s*(\d{4})", re.IGNORECASE)
EFFECTIVE_RE = re.compile(
    r"effective\s+from\s+(" + "|".join(MONTHS) + r")\s+(\d{1,2})\s*,?\s*(\d{4})",
    re.IGNORECASE)
# "1 Shriram Finance Ltd. SHRIRAMFIN" -> name, symbol
ROW_RE = re.compile(r"^\s*(\d{1,2})\s+(.+?)\s+([A-Z][A-Z0-9&\-]{1,19})\s*$")
# A heading such as "1) Nifty 50" or "a) Nifty 50" - the numbering style changes
# between years. The end-anchor is what keeps "Nifty 500" and "Nifty Next 50"
# from matching, so do not loosen it.
NIFTY50_HEAD_RE = re.compile(
    r"^\s*(?:\d+|[a-zA-Z])[).]\s*Nifty\s*50(?:\s+Index)?\s*$", re.IGNORECASE)
ANY_HEAD_RE = re.compile(r"^\s*(?:\d+|[a-zA-Z])[).]\s*\S")


def _parse_date(match: re.Match) -> dt.date:
    month = MONTHS.index(match.group(1).capitalize()) + 1
    return dt.date(int(match.group(3)), month, int(match.group(2)))


def _clean(text: str) -> list[str]:
    return [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]


def nifty50_sections(lines: list[str]) -> list[list[str]]:
    """Every Nifty 50 block in the release.

    Plural because a single release can carry both a periodic review and an
    ad-hoc change (a merger, a demerger) in separate sections. Taking only the
    first would silently drop the second.
    """
    sections = []
    for i, ln in enumerate(lines):
        if not NIFTY50_HEAD_RE.match(ln):
            continue
        block = []
        for nxt in lines[i + 1:]:
            if ANY_HEAD_RE.match(nxt):        # the next index's section begins
                break
            block.append(nxt)
        if block:
            sections.append(block)
    return sections


def _collect(section: list[str], marker: str) -> list[tuple[str, str]]:
    """Rows following 'being excluded:' / 'being included:' until the list ends."""
    out = []
    collecting = False
    for ln in section:
        low = ln.lower()
        if marker in low:
            collecting = True
            continue
        if not collecting:
            continue
        if not ln or low.startswith("sr. no") or low.startswith("sr no"):
            continue
        m = ROW_RE.match(ln)
        if m:
            name = m.group(2).strip(" .")
            out.append((name, m.group(3)))
            continue
        # a sentence or a new sub-heading means the table is over
        if len(ln) > 0 and ("being included" in low or "being excluded" in low
                            or ln[0].isalpha()):
            collecting = False
    return out


@dataclass
class Change:
    effective_date: str
    announcement_date: str
    action: str                 # "include" or "exclude"
    symbol: str
    company_name: str
    source_url: str
    source_quote: str


def parse_release(text: str, url: str) -> list[Change]:
    """Extract the Nifty 50 inclusions and exclusions from one press release."""
    lines = _clean(text)
    sections = nifty50_sections(lines)
    if not sections:
        return []

    head = "\n".join(lines[:60])
    eff = EFFECTIVE_RE.search(head)
    ann = DATE_RE.search(head)
    if eff is None or ann is None:
        return []
    effective, announced = _parse_date(eff), _parse_date(ann)

    quote = next((ln for ln in lines[:60] if "effective from" in ln.lower()), "")
    out, seen = [], set()
    for section in sections:
        for action, marker in (("exclude", "being excluded"),
                               ("include", "being included")):
            for name, symbol in _collect(section, marker):
                if (action, symbol) in seen:
                    continue
                seen.add((action, symbol))
                out.append(Change(
                    effective_date=effective.isoformat(),
                    announcement_date=announced.isoformat(),
                    action=action, symbol=symbol, company_name=name,
                    source_url=url,
                    source_quote=f"{name} ({symbol}) being {action}d from Nifty 50. {quote}",
                ))
    return out


def changes_to_frame(changes: list[Change]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(c) for c in changes])
    if df.empty:
        return df
    return df.sort_values(["effective_date", "action", "symbol"]).reset_index(drop=True)
