"""
generate_standard_results_plot.py
----------------------------------
Clean scorecard plot for Slide 6 — Baseline Results: Standard Timeslots.
Shows minimum clashes achieved by dynamic-only across all 12 Toronto instances.

Output: outputs/reports/custom_visuals/standard_results.png
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "outputs" / "reports" / "custom_visuals"
OUT.mkdir(parents=True, exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────
BG       = "#F7F3EC"
PANEL_BG = "#EFEBE1"
NAVY     = "#1A2A4A"
GOLD     = "#C8A032"
MID      = "#6B7A8D"
RULE     = "#D8CFC0"
GREEN    = "#3A8A4A"
GREEN_LT = "#D4EDDA"
AMBER    = "#C87820"
AMBER_LT = "#FFF0D4"
RED      = "#A03020"
RED_LT   = "#FDE8E8"

# ── Data (sorted by n_exams = increasing difficulty) ──────────────────────
INSTANCES = [
    # (label,        min_clash, n_exams, timeslots)
    ("hec-s-92",      0,   81,  18),
    ("sta-f-83",      0,  139,  13),
    ("yor-f-83",      0,  181,  21),
    ("ute-s-92",      0,  184,  10),
    ("ear-f-83",      0,  190,  24),
    ("tre-s-92",      0,  261,  23),
    ("lse-f-91",      2,  381,  18),
    ("kfu-s-93",      0,  461,  20),
    ("rye-s-93",      0,  486,  23),
    ("car-f-92",     12,  543,  32),
    ("uta-s-92",      0,  622,  35),
    ("car-s-91",      0,  682,  35),
]

# reverse so largest is at the top
INSTANCES = INSTANCES[::-1]

N = len(INSTANCES)


def make_plot():
    fig = plt.figure(figsize=(11, 6.4), facecolor=BG)
    fig.patch.set_facecolor(BG)

    # ── layout: left scorecard  |  right stat panel ───────────────────
    ax  = fig.add_axes([0.01, 0.07, 0.70, 0.86])   # main chart
    axs = fig.add_axes([0.74, 0.07, 0.24, 0.86])   # stat panel
    ax.set_facecolor(BG)
    axs.set_facecolor(BG)
    axs.axis("off")

    # ── draw each instance row ─────────────────────────────────────────
    ROW_H   = 0.72
    BAR_H   = 0.36
    MAX_X   = 16          # x-axis max (clashes)

    for i, (name, clash, n_exams, slots) in enumerate(INSTANCES):
        y      = i * ROW_H
        solved = clash == 0

        # alternating row background
        row_bg = "#F0EBE0" if i % 2 == 0 else BG
        ax.fill_between([0, MAX_X], [y - BAR_H/2 - 0.06], [y + BAR_H/2 + 0.06],
                        color=row_bg, zorder=0)

        if solved:
            # green pill at x=0
            pill = FancyBboxPatch(
                (-0.05, y - BAR_H/2), 1.2, BAR_H,
                boxstyle="round,pad=0.05",
                facecolor=GREEN_LT, edgecolor=GREEN,
                linewidth=1.2, zorder=3
            )
            ax.add_patch(pill)
            ax.text(0.55, y, "0", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=GREEN,
                    fontfamily="DejaVu Sans", zorder=4)
            # checkmark
            ax.text(1.35, y, "✓", ha="left", va="center",
                    fontsize=13, color=GREEN,
                    fontfamily="DejaVu Sans", zorder=4)
        else:
            # amber/red bar extending rightward
            bar_col    = RED    if clash >= 10 else AMBER
            bar_col_lt = RED_LT if clash >= 10 else AMBER_LT

            bar = FancyBboxPatch(
                (-0.05, y - BAR_H/2), clash + 0.4, BAR_H,
                boxstyle="round,pad=0.04",
                facecolor=bar_col_lt, edgecolor=bar_col,
                linewidth=1.4, zorder=3
            )
            ax.add_patch(bar)
            ax.text(clash * 0.5, y, str(clash),
                    ha="center", va="center",
                    fontsize=11, fontweight="bold", color=bar_col,
                    fontfamily="DejaVu Sans", zorder=4)
            # X marker
            ax.text(clash + 0.65, y, "✗", ha="left", va="center",
                    fontsize=12, color=bar_col,
                    fontfamily="DejaVu Sans", zorder=4)

        # instance name label (left)
        ax.text(-0.3, y, name,
                ha="right", va="center", fontsize=10.5,
                color=NAVY, fontfamily="DejaVu Serif",
                fontweight="bold", zorder=4)

        # exam count (right side, subtle)
        ax.text(MAX_X * 0.98, y, f"{n_exams} exams",
                ha="right", va="center", fontsize=8.5,
                color=MID, fontfamily="DejaVu Sans",
                fontstyle="italic", zorder=4)

    # ── zero reference line ────────────────────────────────────────────
    ax.axvline(0, color=GREEN, lw=1.4, alpha=0.55, zorder=2, linestyle="-")

    # ── axis styling ───────────────────────────────────────────────────
    ax.set_xlim(-5.2, MAX_X)
    ax.set_ylim(-0.6, (N - 1) * ROW_H + 0.6)
    ax.set_yticks([])
    ax.set_xticks([0, 4, 8, 12])
    ax.set_xticklabels(["0", "4", "8", "12"],
                       fontsize=9, color=MID,
                       fontfamily="DejaVu Sans")
    ax.set_xlabel("Minimum clashes achieved  (dynamic-only method)",
                  fontsize=9.5, color=MID, fontfamily="DejaVu Sans",
                  labelpad=6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(left=False, bottom=True, colors=MID, length=3)
    ax.grid(axis="x", color=RULE, lw=0.6, alpha=0.7, zorder=0)

    # difficulty separator lines
    # easier instances (solved) and harder (unsolved) are interleaved,
    # but we mark lse-f-91 and car-f-92 rows with a subtle left accent
    unsolved_idx = [i for i, (_, c, _, _) in enumerate(INSTANCES) if c > 0]
    for ui in unsolved_idx:
        y = ui * ROW_H
        ax.plot([-5.2, -4.8], [y, y], color=RED, lw=3, solid_capstyle="round",
                zorder=5)

    # ── stat panel (right) ─────────────────────────────────────────────
    # big "10 / 12" number
    axs.text(0.5, 0.78, "10 / 12",
             ha="center", va="center", fontsize=38,
             fontweight="bold", color=NAVY,
             fontfamily="DejaVu Sans",
             transform=axs.transAxes)
    axs.text(0.5, 0.63, "instances solved\nto zero clashes",
             ha="center", va="center", fontsize=12,
             color=MID, fontfamily="DejaVu Sans",
             linespacing=1.5,
             transform=axs.transAxes)

    # thin divider
    axs.plot([0.08, 0.92], [0.55, 0.55], color=RULE, lw=1.2,
             transform=axs.transAxes)

    # legend chips
    for ypos, col_lt, col, txt in [
        (0.43, GREEN_LT, GREEN, "Zero clashes  ✓"),
        (0.30, AMBER_LT, AMBER, "Residual ≤ 12  ✗"),
    ]:
        chip = FancyBboxPatch((0.08, ypos - 0.055), 0.84, 0.11,
                              boxstyle="round,pad=0.03",
                              facecolor=col_lt, edgecolor=col,
                              linewidth=1.2,
                              transform=axs.transAxes, zorder=3)
        axs.add_patch(chip)
        axs.text(0.5, ypos, txt,
                 ha="center", va="center", fontsize=10,
                 fontweight="bold", color=col,
                 fontfamily="DejaVu Sans",
                 transform=axs.transAxes, zorder=4)

    axs.text(0.5, 0.12,
             "Standard timeslots\n(full university schedule)",
             ha="center", va="center", fontsize=8.5,
             color=MID, fontfamily="DejaVu Sans",
             fontstyle="italic", linespacing=1.5,
             transform=axs.transAxes)

    # right panel border
    for side in ["top", "bottom", "left", "right"]:
        axs.spines[side].set_visible(False)
    left_line = plt.Line2D([0.02, 0.02], [0.04, 0.96],
                           transform=axs.transAxes,
                           color=RULE, lw=1.2)
    axs.add_artist(left_line)

    plt.savefig(OUT / "standard_results.png",
                dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓  standard_results.png  →  {OUT}")


if __name__ == "__main__":
    make_plot()
