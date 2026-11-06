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


def download_pdf(url: str):
    """Download a press release to the local cache, returning its path."""
    PDF_CACHE.mkdir(parents=True, exist_ok=True)
    path = PDF_CACHE / url.rsplit("/", 1)[-1]
    if path.exists() and path.stat().st_size > 0:
        return path
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=90) as resp:
        if "pdf" not in resp.headers.get("Content-Type", "").lower():
            raise ValueError(f"not a PDF: {url}")
        data = resp.read()
    path.write_bytes(data)
    return path


def fetch_pdf_text(url: str) -> str:
    """Quick text extraction, used only as a cheap screen. Reading order is
    NOT reliable - see fetch_pdf_layout_text for the one that is."""
    from pypdf import PdfReader

    path = download_pdf(url)
    reader = PdfReader(io.BytesIO(path.read_bytes()))
    return "\n".join((p.extract_text() or "") for p in reader.pages)


def fetch_pdf_layout_text(url: str, line_tol: float = 3.0) -> str:
    """Text rebuilt from word coordinates, so lines come out in reading order.

    The pre-2018 press releases need this. Their headings and their tables are
    separate objects in the PDF, and a plain text extraction returns all the
    headings first and all the tables afterwards, detached from each other:

        (1) S&P CNX Nifty Index
        The following company is being excluded:
        ... (two more index headings) ...
        Sr. No. Company Name
        1 Satyam Computer Services Ltd.      <- which index is this under?

    Grouping words by their vertical position puts each table back under its
    own heading, which is the only way to attribute a change to the right index
    without guessing.
    """
    import pdfplumber

    path = download_pdf(url)
    out: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            # Several of the older releases fake bold text by drawing every
            # glyph twice at the same spot, which extracts as
            # "SS&&PP CCNNXX NNiiffttyy" and matches no pattern at all.
            # Dropping the duplicate characters is what makes 2012 readable.
            try:
                page = page.dedupe_chars(tolerance=1)
            except Exception:             # noqa: BLE001 - older pdfplumber
                pass
            rows: dict[float, list[tuple[float, str]]] = {}
            for w in page.extract_words():
                key = round(w["top"] / line_tol) * line_tol
                rows.setdefault(key, []).append((w["x0"], w["text"]))
            for key in sorted(rows):
                out.append(" ".join(t for _, t in sorted(rows[key])))
    return "\n".join(out)


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
# The announcement date sits in the header in modern releases ("Mumbai,
# February 28, 2024") but at the FOOT of the pre-2018 ones ("Date: January 7,
# 2009."). Looking only at the header picks up the effective date by mistake
# and silently reports a release as announced on the day it took effect.
DATED_RE = re.compile(
    r"Date\s*:\s*(" + "|".join(MONTHS) + r")\s+(\d{1,2})\s*,?\s*(\d{4})",
    re.IGNORECASE)
MUMBAI_RE = re.compile(
    r"Mumbai\s*,\s*(" + "|".join(MONTHS) + r")\s+(\d{1,2})\s*,?\s*(\d{4})",
    re.IGNORECASE)
# "1 Shriram Finance Ltd. SHRIRAMFIN" -> name, symbol (2018 onwards).
# The serial number is optional because coordinate-based extraction sometimes
# puts it on its own line:
#     Divi's Laboratories Ltd. DIVISLAB
#     1
#     2 LTIMindtree Ltd. LTIM
# Requiring it dropped the first row of many tables - and, because an
# unnumbered row also looks like prose, ended the table early and lost the
# rest of it too.
ROW_RE = re.compile(r"^\s*(?:(\d{1,2})\s+)?(.+?)\s+([A-Z][A-Z0-9&\-]{1,19})\s*$")
BARE_NUMBER_RE = re.compile(r"^\s*\d{1,2}\s*$")
# Lines that end a table rather than continue it.
TABLE_END_RE = re.compile(
    r"^\s*(note\s*:|the above|these changes|disclaimer|for further|place\s*:|date\s*:)",
    re.IGNORECASE)
# "1 Satyam Computer Services Ltd." -> name only. Before 2018 the tables have
# no Symbol column at all, so the ticker has to be resolved separately.
NAME_ONLY_ROW_RE = re.compile(r"^\s*(\d{1,2})\s+([A-Za-z][A-Za-z0-9&.,'()/\- ]{3,80})\s*$")

# The index has been renamed twice: "S&P CNX Nifty" until late 2013, then
# "CNX Nifty", then "Nifty 50" from 2015. A parser that only knows the current
# name silently finds nothing before 2014 - which is most of the study period.
#
# The end-anchor is what keeps the near-misses out, and they are everywhere in
# these documents: "CNX Nifty Junior", "Nifty Next 50", "S&P CNX 500",
# "Nifty 500", "CNX 100". Do not loosen it.
_PREFIX = r"^\s*\(?(?:\d+|[a-zA-Z])[).]\s*"
_NIFTY_NAMES = r"(?:S\s*&\s*P\s+)?CNX\s+Nifty|Nifty\s*50"
NIFTY50_HEAD_RE = re.compile(
    _PREFIX + r"(?:" + _NIFTY_NAMES + r")(?:\s+Index)?\s*$", re.IGNORECASE)
ANY_HEAD_RE = re.compile(_PREFIX + r"\S")


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


WRAPPED_MARKER_RE = re.compile(r"^(excluded|included)\s*:?\s*$", re.IGNORECASE)


def _repair_wrapped_markers(lines: list[str]) -> list[str]:
    """Rejoin "The following company is being" / "excluded:" split across lines.

    Coordinate grouping sometimes puts the last word of the marker sentence on
    its own line, and occasionally ABOVE the rest of the sentence rather than
    below it. The marker is then never present on any single line, the table
    that follows is never collected, and the release silently reports changes
    in one direction only - which is what produced one-sided replacement dates.
    """
    out = list(lines)
    for i, ln in enumerate(out):
        m = WRAPPED_MARKER_RE.match(ln.strip())
        if not m:
            continue
        word = m.group(1).lower()
        for j in (i - 1, i + 1):          # the fragment can sit either side
            if 0 <= j < len(out) and out[j].rstrip().lower().endswith("being"):
                out[j] = f"{out[j].rstrip()} {word}:"
                out[i] = ""
                break
    return out


def _plausible_company(name: str) -> bool:
    """Reject prose that happens to end in capitals (the explanatory notes do)."""
    if not name or len(name) > 70:
        return False
    low = name.lower()
    return not any(w in low for w in
                   ("crore", "market capitalization", "eligible", "pursuant",
                    "replacement", "index", "average"))


def _collect(section: list[str], marker: str) -> list[tuple[str, str]]:
    """Rows following 'being excluded:' / 'being included:' until the list ends."""
    out = []
    collecting = False
    for ln in _repair_wrapped_markers(section):
        low = ln.lower()
        if marker in low:
            collecting = True
            continue
        if not collecting:
            continue
        if not ln or low.startswith("sr. no") or low.startswith("sr no"):
            continue
        if BARE_NUMBER_RE.match(ln):
            continue                       # a serial number stranded on its own line
        if TABLE_END_RE.match(ln) or "being included" in low or "being excluded" in low:
            collecting = False
            continue

        m = ROW_RE.match(ln)
        if m and _plausible_company(m.group(2)):
            out.append((m.group(2).strip(" ."), m.group(3)))
            continue
        m = NAME_ONLY_ROW_RE.match(ln)
        if m and _plausible_company(m.group(2)):
            # Pre-2018 tables carry no Symbol column; the ticker is resolved later.
            out.append((m.group(2).strip(" ."), ""))
            continue
        # Anything else inside a table region is ignored rather than treated as
        # the end of it - an unrecognised line used to truncate the table.
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

    whole = "\n".join(lines)
    head = "\n".join(lines[:60])
    eff = EFFECTIVE_RE.search(whole)
    if eff is None:
        return []
    effective = _parse_date(eff)

    # Prefer an explicit "Mumbai, <date>" or "Date: <date>"; fall back to the
    # first date in the header that is not the effective date itself. An
    # announcement can never be later than the change it announces, so that is
    # checked rather than assumed.
    announced = None
    for pattern, scope in ((MUMBAI_RE, head), (DATED_RE, whole), (MUMBAI_RE, whole)):
        m = pattern.search(scope)
        if m:
            cand = _parse_date(m)
            if cand <= effective:
                announced = cand
                break
    if announced is None:
        for m in DATE_RE.finditer(head):
            cand = _parse_date(m)
            if cand <= effective and cand != effective:
                announced = cand
                break
    if announced is None:
        announced = effective

    quote = next((ln for ln in lines[:60] if "effective from" in ln.lower()), "")
    out, seen = [], set()
    for section in sections:
        for action, marker in (("exclude", "being excluded"),
                               ("include", "being included")):
            for name, symbol in _collect(section, marker):
                if (action, symbol, name) in seen:
                    continue
                seen.add((action, symbol, name))
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
