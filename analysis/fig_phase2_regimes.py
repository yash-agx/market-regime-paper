"""Figure: the Nifty 50 index with high-volatility periods shaded, Method A vs Method B.

    python -m analysis.fig_phase2_regimes

Two stacked panels sharing one x-axis rather than both shadings on one chart:
overlapping translucent bands would be unreadable, and the whole point of the
figure is to let the two methods be compared stretch by stretch. A third panel
shows the realised-volatility series against its expanding median, which is the
actual decision rule behind Method A.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src import config as cfg
from src import data as D
from src import regimes as R

# Validated two-colour palette (dataviz reference instance, light mode).
INDEX_LINE = "#2a78d6"     # categorical slot 1, blue
HIGH_VOL = "#e34948"       # categorical slot 8, red
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e5e4e0"
SURFACE = "#fcfcfb"


def shade_regimes(ax, flag: pd.Series, color: str = HIGH_VOL) -> None:
    """Shade every unbroken high-volatility stretch."""
    runs = R.regime_runs(flag)
    for _, row in runs[runs["high_vol"]].iterrows():
        ax.axvspan(row["start"], row["end"], color=color, alpha=0.22, linewidth=0, zorder=0)


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
    cfg.ensure_dirs()
    cleaned = D.load_processed()
    reg = R.build(cleaned.nsei_returns, cleaned.vix)
    labels = reg.labels.loc[cfg.STUDY_START:cfg.STUDY_END]
    nsei = cleaned.nsei.loc[cfg.STUDY_START:cfg.STUDY_END]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 0.9], "hspace": 0.22})
    fig.patch.set_facecolor(SURFACE)

    panels = [
        (axes[0], labels["high_vol_a"],
         "Method A (main): 21-day realised volatility above its expanding median"),
        (axes[1], labels["high_vol_b"],
         "Method B: 2-state Gaussian HMM, filtered P(high-volatility state) > 0.5"),
    ]
    for ax, flag, title in panels:
        shade_regimes(ax, flag)
        ax.plot(nsei.index, nsei.to_numpy(), color=INDEX_LINE, linewidth=1.4, zorder=2)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=10.5, color=TEXT_PRIMARY, loc="left", pad=8)
        ax.set_ylabel("Nifty 50 (log scale)", fontsize=9, color=TEXT_SECONDARY)
        pct = 100 * float(flag.mean())
        ax.annotate(f"high-volatility on {pct:.1f}% of days",
                    xy=(0.995, 0.06), xycoords="axes fraction", ha="right",
                    fontsize=9, color=TEXT_SECONDARY)
        style(ax)

    # Panel 3: the decision rule behind Method A, drawn explicitly.
    ax = axes[2]
    shade_regimes(ax, labels["high_vol_a"])
    ax.plot(labels.index, 100 * labels["realised_vol"].to_numpy(),
            color=INDEX_LINE, linewidth=1.2, zorder=2, label="21-day realised volatility")
    ax.plot(labels.index, 100 * reg.a["expanding_median"].reindex(labels.index).to_numpy(),
            color=TEXT_PRIMARY, linewidth=1.6, zorder=3, linestyle="--",
            label="expanding median (the threshold)")
    ax.set_title("The Method A decision rule: a day is high-volatility when the blue line "
                 "is above the dashed line", fontsize=10.5, color=TEXT_PRIMARY, loc="left",
                 pad=8)
    ax.set_ylabel("annualised volatility (%)", fontsize=9, color=TEXT_SECONDARY)
    style(ax)
    ax.legend(loc="upper right", fontsize=9, frameon=False, labelcolor=TEXT_SECONDARY)

    # One shared legend entry for the shading, since it means the same thing everywhere.
    band = plt.Rectangle((0, 0), 1, 1, color=HIGH_VOL, alpha=0.22)
    axes[0].legend([band], ["high-volatility regime"], loc="upper left", fontsize=9,
                   frameon=False, labelcolor=TEXT_SECONDARY)

    fig.suptitle("Volatility regimes in the Nifty 50, 2010-2025",
                 fontsize=13, color=TEXT_PRIMARY, x=0.125, ha="left", y=0.965)
    fig.text(0.125, 0.932,
             "Regime labels use only data available on the day they are assigned. "
             "Shaded = high volatility.",
             fontsize=9.5, color=TEXT_SECONDARY, ha="left")

    out = cfg.FIGURES_DIR / "phase2_regimes.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
