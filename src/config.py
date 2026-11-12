"""Paths, dates and frozen constants shared by every phase.

Nothing here is a result. Changing a number here changes the experiment, so these
values mirror the frozen spec described in the README and should not be edited
without updating it there too.
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
EXTREME_RETURN = 0.25         # flag (never delete) daily returns beyond +/-25%
# Lowered from 0.50 in Phase 5. Both unadjusted demergers in the data sit
# between 38% and 41% - a 50% threshold saw neither of them, and the Adani
# Enterprises one had already reached the in-sample portfolios.

# --- conventions -----------------------------------------------------------
TRADING_DAYS = 252

# --- HMM (Method B) --------------------------------------------------------
# The spec as frozen says only "2-state Gaussian HMM, refit every 12 months on
# an expanding window, fixed seed". These are the estimation details it leaves
# open. HMM_TRANSMAT_PRIOR = 2.0 is a weak Dirichlet prior worth one
# pseudo-observation of each transition. It was added in Phase 2, BEFORE any
# strategy results were seen, to fix a degenerate estimate: unregularised
# maximum likelihood put the low->high transition at exactly zero for the 2011
# and 2012 training windows, making the low-volatility state absorbing, so the
# filter could never leave it once it entered. See
# results/phase2_methodb_prior_sensitivity.csv for what the value costs -
# almost nothing on the well-identified later fits.
HMM_TRANSMAT_PRIOR = 2.0
HMM_RESTARTS = 10             # deterministic restarts; the best log-likelihood wins
HMM_MIN_TRAIN_DAYS = 200      # the 2009 warm-up year (~248 days) clears this
MONTHS_PER_YEAR = 12
WEEKS_PER_YEAR = 52
RANDOM_SEED = 20100101        # fixed seed everywhere (HMM, bootstrap)


def ensure_dirs() -> None:
    """Create the output directories if they do not exist."""
    for d in (RAW_DIR, PROC_DIR, RESULTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
