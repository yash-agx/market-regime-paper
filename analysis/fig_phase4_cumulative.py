"""Figure: cumulative returns of both strategies, with high-volatility periods shaded.

    python -m analysis.fig_phase4_cumulative              # in-sample 2010-2019
    python -m analysis.fig_phase4_cumulative --holdout    # once in-sample is final

Gross and net are drawn together because the gap between them IS the finding
for reversal: the signal works, the trading destroys it. Shading is Method A,
the pre-specified main regime definition.
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src import config as cfg
from src import data as D
from src import evaluation as E
from src import regimes as R
from src import strategies as S
from src import windows as W

GROSS = "#2a78d6"          # categorical slot 1, blue
NET = "#eb6834"            # categorical slot 2, orange
HIGH_VOL = "#e34948"       # categorical slot 8, red
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e5e4e0"
SURFACE = "#fcfcfb"


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true")
    args = ap.parse_args()
    if args.holdout:
        lo, hi, tag, title = (cfg.HOLDOUT_START, cfg.STUDY_END, "holdout",
                              "2020-2025 (held out)")
    else:
        lo, hi, tag, title = (cfg.STUDY_START, cfg.INSAMPLE_END, "insample",
                              "2010-2019 (in-sample)")

    cfg.ensure_dirs()
    cleaned = D.load_processed()
    reg = R.build(cleaned.nsei_returns, cleaned.vix)
    flag = reg.labels["high_vol_a"].loc[lo:hi]
    strats = S.build_all(cleaned)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                             gridspec_kw={"hspace": 0.2})
    fig.patch.set_facecolor(SURFACE)

    panels = [
        (axes[0], "momentum_ls", "Momentum, long-short (12-1 signal, monthly)"),
        (axes[1], "reversal_ls", "Reversal, long-short (1-week signal, weekly)"),
    ]
    for ax, key, heading in panels:
        sub = S.slice_period(strats[key], lo, hi)
        # Shade high-volatility stretches behind the curves.
        for _, row in R.regime_runs(flag).query("high_vol").iterrows():
            ax.axvspan(row["start"], row["end"], color=HIGH_VOL, alpha=0.20,
                       linewidth=0, zorder=0)

        for series, colour, label in ((sub.gross, GROSS, "before costs"),
                                      (sub.ret(20), NET, "after costs (20bp)")):
            curve = (1 + series).cumprod()
            curve = pd.concat([pd.Series([1.0], index=[series.index[0]]), curve])
            ax.plot(curve.index, curve.to_numpy(), color=colour, linewidth=1.8, zorder=3)
            ax.annotate(f"{curve.iloc[-1]:.2f}x", xy=(curve.index[-1], curve.iloc[-1]),
                        xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=9.5, color=colour, fontweight="bold")

        ax.set_yscale("log")
        ax.axhline(1.0, color=TEXT_SECONDARY, linewidth=0.9, linestyle=":", zorder=2)
        ax.set_title(heading, fontsize=11, color=TEXT_PRIMARY, loc="left", pad=8)
        ax.set_ylabel("growth of 1 rupee", fontsize=9, color=TEXT_SECONDARY)
        style(ax)

    band = plt.Rectangle((0, 0), 1, 1, color=HIGH_VOL, alpha=0.20)
    gross_line = plt.Line2D([], [], color=GROSS, linewidth=1.8)
    net_line = plt.Line2D([], [], color=NET, linewidth=1.8)
    axes[0].legend([gross_line, net_line, band],
                   ["before costs", "after costs (20bp)", "high-volatility regime"],
                   loc="upper left", fontsize=9, frameon=False,
                   labelcolor=TEXT_SECONDARY, ncol=3)

    fig.suptitle(f"Momentum and reversal through volatility regimes, {title}",
                 fontsize=13, color=TEXT_PRIMARY, x=0.125, ha="left", y=0.965)
    fig.text(0.125, 0.930,
             "Shaded = high volatility by Method A. Log scale, so equal vertical "
             "distances are equal percentage moves.",
             fontsize=9.5, color=TEXT_SECONDARY, ha="left")

    out = cfg.FIGURES_DIR / f"phase4_cumulative_{tag}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
