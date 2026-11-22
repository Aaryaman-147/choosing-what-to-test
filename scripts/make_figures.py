"""
SM03 — figures.

Panel A  cumulative active-discovery curves on the test pool, mean +/- sd over
         seeds 42/43/44, against the random floor and the greedy ceiling.
Panel B  the yield/diversity trade-off across the whole policy ladder on
         validation (10 seeds), which is the honest summary of what the
         exploration terms actually buy.

Rendered at 1920px wide for use in the submission video.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = ROOT / "outputs"

# Validated categorical slots 1-3 (light surface #fcfcfb): all six checks pass,
# worst all-pairs CVD dE 9.2, normal-vision 24.0. Aqua is sub-3:1 on this
# surface, so every series carries a direct text label, not colour alone.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"

SERIES = {
    "ucb_diverse_b05_l03": ("UCB + diversity  (frozen)", BLUE),
    "greedy": ("Greedy  (yield ceiling)", ORANGE),
    "random": ("Random  (uninformed floor)", AQUA),
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def panel_a(ax) -> None:
    curves = pd.read_csv(OUT / "test_curves.csv")
    manifest = pd.read_json(OUT / "test_run_manifest.json", typ="series")
    base_rate = float(manifest["base_rate"])

    for policy, (label, colour) in SERIES.items():
        sub = curves[curves.policy == policy]
        if sub.empty:
            continue
        grouped = sub.groupby("round")
        x = grouped["molecules_tested"].first().to_numpy()
        mean = grouped["cumulative_actives"].mean().to_numpy()
        sd = grouped["cumulative_actives"].std(ddof=1).fillna(0).to_numpy()

        ax.fill_between(x, mean - sd, mean + sd, color=colour, alpha=0.15, linewidth=0)
        ax.plot(x, mean, color=colour, linewidth=2, zorder=3)
        # 8px markers with a 2px surface ring so overlapping points stay legible.
        ax.plot(
            x, mean, "o", color=colour, markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=4,
        )
        # Direct label — identity never rests on colour alone.
        ax.annotate(
            label,
            xy=(x[-1], mean[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            color=colour,
            fontsize=11,
            fontweight="600",
            va="center",
        )

    # The base-rate line: what testing the same number at random would yield.
    xs = np.array([40, 140])
    ax.plot(xs, xs * base_rate, linestyle=(0, (4, 4)), color=MUTED, linewidth=1.5, zorder=1)
    # Anchored mid-line, below it, so it clears the Random series' end label.
    ax.annotate(
        f"random expectation ({base_rate:.0%} base rate)",
        xy=(104, 104 * base_rate),
        xytext=(0, -16),
        textcoords="offset points",
        color=MUTED,
        fontsize=9.5,
        ha="center",
        va="top",
        rotation=19,
        rotation_mode="anchor",
    )

    ax.set_title(
        "Actives found as the testing budget is spent",
        fontsize=13.5, fontweight="700", color=INK, pad=14, loc="left",
    )
    ax.set_xlabel("molecules tested  (40 seeded, then 5 batches of 20)", fontsize=10.5)
    ax.set_ylabel("cumulative actives discovered", fontsize=10.5)
    ax.set_xticks([40, 60, 80, 100, 120, 140])
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(32, 196)
    ax.legend(
        handles=[
            plt.Line2D([], [], color=c, linewidth=2, label=l)
            for l, c in SERIES.values()
        ],
        loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK2,
    )


def panel_b(ax) -> None:
    tie = pd.read_csv(OUT / "tiebreak_summary.csv")

    short = {
        "random": "random",
        "uncertainty": "uncertainty\nonly",
        "greedy": "greedy",
        "ucb_b05": "UCB",
        "greedy_diverse_l05": "greedy\n+ diversity",
        "ucb_diverse_b05_l03": "UCB + diversity\n(frozen)",
    }
    highlight = "ucb_diverse_b05_l03"
    # greedy and UCB sit almost on top of each other, so their labels are
    # placed on opposite sides rather than both below.
    offsets = {
        "greedy": (0, -30, "center"),
        "ucb_b05": (0, 16, "center"),
        "greedy_diverse_l05": (0, -34, "center"),
        "uncertainty": (0, -34, "center"),
        "random": (0, -26, "center"),
        highlight: (0, 16, "center"),
    }

    for _, row in tie.iterrows():
        name = row["policy"]
        is_hero = name == highlight
        colour = BLUE if is_hero else (ORANGE if name == "greedy" else MUTED)
        ax.errorbar(
            row["scaffolds_per_20"], row["acquired_mean"],
            yerr=row["acquired_sem"],
            fmt="o", markersize=11 if is_hero else 8,
            color=colour, ecolor=colour, elinewidth=1.5, capsize=3,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=3,
        )
        dx, dy, ha = offsets.get(name, (0, -26, "center"))
        ax.annotate(
            short.get(name, name),
            xy=(row["scaffolds_per_20"], row["acquired_mean"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha, fontsize=9.5,
            color=colour if is_hero or name == "greedy" else INK2,
            fontweight="700" if is_hero else "400",
        )

    ax.set_title(
        "What the exploration terms cost: yield vs batch diversity",
        fontsize=13.5, fontweight="700", color=INK, pad=14, loc="left",
    )
    ax.set_xlabel("distinct Bemis–Murcko scaffolds per 20-molecule batch", fontsize=10.5)
    ax.set_ylabel("actives acquired per 100 tested  (10 seeds, ±sem)", fontsize=10.5)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(15.2, 20.2)
    ax.set_ylim(30, 72)


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    panel_a(ax1)
    panel_b(ax2)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.80, bottom=0.13, wspace=0.26)
    fig.text(
        0.055, 0.945,
        "SM03 · BACE1 sequential selection under a fixed testing budget",
        fontsize=15.5, fontweight="700", color=INK,
    )
    fig.text(
        0.055, 0.893,
        "Retrospective simulation. Left: held-out test pool, seeds 42/43/44. "
        "Right: policy selection on validation, 10 seeds.",
        fontsize=10.5, color=INK2,
    )

    path = OUT / "sm03_figure.png"
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    print(f"wrote {path}")

    # Table view — the relief obligation for the sub-3:1 series, and the
    # numbers a reviewer needs without reading pixels.
    curves = pd.read_csv(OUT / "test_curves.csv")
    table = (
        curves.groupby(["policy", "molecules_tested"])["cumulative_actives"]
        .agg(["mean", "std"])
        .round(2)
        .reset_index()
    )
    table.to_csv(OUT / "figure_table_view.csv", index=False)
    print(f"wrote {OUT / 'figure_table_view.csv'}")


if __name__ == "__main__":
    main()
