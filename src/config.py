"""Paths, dates and frozen constants shared by every phase.

Nothing here is a result. Changing a number here changes the experiment, so these
values mirror the spec in the README and should not be edited without updating it.
"""

from __future__ import annotations

from pathlib import Path

# --- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "results" / "figures"

UNIVERSE_CSV = CONFIG_DIR / "universe.csv"
TBILL_CSV = RAW_DIR / "tbill_91d.csv"

# --- sample ----------------------------------------------------------------
# 2009 is warm-up only: the 12-month momentum signal needs a year of history
# before the first in-sample month end. The study period starts in 2010.
DATA_START = "2009-01-01"
DATA_END = "2025-12-31"
STUDY_START = "2010-01-01"
STUDY_END = "2025-12-31"
INSAMPLE_END = "2019-12-31"   # 2020-2025 is held out
HOLDOUT_START = "2020-01-01"

# --- tickers ---------------------------------------------------------------
INDEX_TICKER = "^NSEI"        # Nifty 50 index
VIX_TICKER = "^INDIAVIX"      # India VIX

# --- cleaning rules --------------------------------------------------------
MIN_HISTORY_DAYS = 252        # a stock is eligible only after 252 trading days
MAX_MISSING_FRACTION = 0.05   # drop a stock for a period if >5% of its formation window is missing
MAX_FLAT_RUN_DAYS = 10        # drop a stock for a window hit by a run of >=10 identical closes
EXTREME_RETURN = 0.50         # flag (never delete) daily returns beyond +/-50%

# --- conventions -----------------------------------------------------------
TRADING_DAYS = 252
MONTHS_PER_YEAR = 12
WEEKS_PER_YEAR = 52
RANDOM_SEED = 20100101        # fixed seed everywhere (HMM, bootstrap)


def ensure_dirs() -> None:
    """Create the output directories if they do not exist."""
    for d in (RAW_DIR, PROC_DIR, RESULTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
