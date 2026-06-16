"""
generate_contrast_plot.py
--------------------------
Two-panel contrast plot for the revised Slide 6.

Left:  Standard timeslots — dynamic-only min clashes (mostly zero)
Right: Minimum timeslots  — hybrid improvement (Δ = dyn − hyb)
       green bar  = hybrid wins
       gray bar   = tie
       red bar    = dynamic wins

Story: "With room to breathe → dynamic alone suffices.
        When squeezed tight  → Hopfield pulls ahead."

Output: outputs/reports/custom_visuals/contrast_standard_vs_minimum.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

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
RED_COL  = "#A03020"
RED_LT   = "#FDE8E8"
GRAY     = "#8899AA"
GRAY_LT  = "#E8ECF0"
BLUE_DYN = "#4C72B0"

# ── Data (sorted by n_exams, smallest → largest = bottom → top on y-axis) ─
INSTANCES = [
    # name          std_dyn  min_dyn  min_hyb  winner       n_exams
    ("hec-s-92",       0,      2,       2,    "tie",          81),
    ("sta-f-83",       0,      0,       0,    "tie",         139),
    ("yor-f-83",       0,     50,      40,    "hybrid",      181),
    ("ute-s-92",       0,      0,       0,    "tie",         184),
    ("ear-f-83",       0,     16,       8,    "hybrid",      190),
    ("tre-s-92",       0,     42,      22,    "hybrid",      261),
    ("lse-f-91",       2,     12,      14,    "dynamic",     381),
    ("kfu-s-93",       0,     16,      16,    "tie",         461),
    ("rye-s-93",       0,     18,      18,    "tie",         486),
    ("car-f-92",      12,    120,     124,    "dynamic",     543),
    ("uta-s-92",       0,     92,      52,    "hybrid",      622),
    ("car-s-91",       0,    328,     186,    "hybrid",      682),
]
# reverse so largest exam count is at the top
INSTANCES = INSTANCES[::-1]
N = len(INSTANCES)

names    = [r[0] for r in INSTANCES]
std_dyn  = [r[1] for r in INSTANCES]
min_dyn  = [r[2] for r in INSTANCES]
min_hyb  = [r[3] for r in INSTANCES]
winners  = [r[4] for r in INSTANCES]
n_exams  = [r[5] for r in INSTANCES]
deltas   = [d - h for d, h in zip(min_dyn, min_hyb)]   # positive = hybrid wins


def make_plot():
    fig = plt.figure(figsize=(13, 6.8), facecolor=BG)
    fig.patch.set_facecolor(BG)

    # ── axes: left panel 38%, right panel 55%, gap 7% ─────────────────
    ax_l = fig.add_axes([0.14, 0.09, 0.30, 0.83])
    ax_r = fig.add_axes([0.54, 0.09, 0.44, 0.83])

    y_pos = np.arange(N)
    BAR_H = 0.52

    # ═══════════════════════════════════════════════════════════════════
    # LEFT PANEL — Standard timeslots, dynamic-only
    # ═══════════════════════════════════════════════════════════════════
    ax_l.set_facecolor(BG)

    for i, (name, clash, winner) in enumerate(zip(names, std_dyn, winners)):
        y = y_pos[i]

        # row stripe
        if i % 2 == 0:
            ax_l.axhspan(y - 0.46, y + 0.46, color="#F0EBE0", zorder=0)

        solved = clash == 0
        bar_color    = GREEN    if solved else (AMBER if clash < 10 else RED_COL)
        bar_color_lt = GREEN_LT if solved else (AMBER_LT if clash < 10 else RED_LT)
        bar_width    = max(clash, 0.6)

        rect = FancyBboxPatch(
            (0, y - BAR_H / 2), bar_width, BAR_H,
            boxstyle="round,pad=0.04",
            facecolor=bar_color_lt, edgecolor=bar_color,
            linewidth=1.3, zorder=3
        )
        ax_l.add_patch(rect)

        label = "0 ✓" if solved else str(clash)
        ax_l.text(bar_width / 2, y, label,
                  ha="center", va="center",
                  fontsize=10, fontweight="bold", color=bar_color,
                  fontfamily="DejaVu Sans", zorder=4)

    # axes styling
    ax_l.set_yticks(y_pos)
    ax_l.set_yticklabels(names, fontsize=10, color=NAVY,
                         fontfamily="DejaVu Serif", fontweight="bold")
    ax_l.set_xlim(-0.5, 15)
    ax_l.set_ylim(-0.65, N - 0.35)
    ax_l.set_xticks([0, 4, 8, 12])
    ax_l.tick_params(axis="x", colors=MID, labelsize=8.5)
    ax_l.tick_params(axis="y", length=0)
    ax_l.set_xlabel("Min clashes (dynamic-only)", fontsize=9,
                    color=MID, fontfamily="DejaVu Sans", labelpad=5)
    for sp in ax_l.spines.values():
        sp.set_visible(False)
    ax_l.axvline(0, color=GREEN, lw=1.2, alpha=0.5)
    ax_l.grid(axis="x", color=RULE, lw=0.5, alpha=0.7, zorder=0)

    # panel title
    ax_l.set_title("Standard Timeslots\n(full university schedule)",
                   fontsize=10.5, color=NAVY, fontweight="bold",
                   fontfamily="DejaVu Sans", pad=8, linespacing=1.5)

    # "10/12" callout inside left panel
    ax_l.text(13.2, N - 1.3,
              "10 / 12\nzero ✓",
              ha="right", va="top", fontsize=13,
              fontweight="bold", color=GREEN,
              fontfamily="DejaVu Sans", linespacing=1.4,
              bbox=dict(boxstyle="round,pad=0.4",
                        facecolor=GREEN_LT, edgecolor=GREEN,
                        linewidth=1.3, alpha=0.92),
              zorder=6)

    # ═══════════════════════════════════════════════════════════════════
    # RIGHT PANEL — Minimum timeslots, Δ = dyn − hyb
    # ═══════════════════════════════════════════════════════════════════
    ax_r.set_facecolor(BG)

    MAX_DELTA = max(deltas) * 1.12   # x-axis right limit

    for i, (delta, winner) in enumerate(zip(deltas, winners)):
        y = y_pos[i]

        if i % 2 == 0:
            ax_r.axhspan(y - 0.46, y + 0.46, color="#F0EBE0", zorder=0)

        if winner == "hybrid":
            fc, ec, tc = GREEN_LT, GREEN, GREEN
        elif winner == "dynamic":
            fc, ec, tc = RED_LT,   RED_COL, RED_COL
        else:
            fc, ec, tc = GRAY_LT,  GRAY,  GRAY

        bar_w = abs(delta) if abs(delta) > 0 else 0.0

        if bar_w > 0:
            rect = FancyBboxPatch(
                (0, y - BAR_H / 2), bar_w, BAR_H,
                boxstyle="round,pad=0.04",
                facecolor=fc, edgecolor=ec,
                linewidth=1.3, zorder=3
            )
            ax_r.add_patch(rect)

        # value label
        label_x = max(bar_w + 1.5, 3.0)
        label = (f"+{delta}" if delta > 0
                 else (f"−{abs(delta)}" if delta < 0 else "tie"))
        ax_r.text(label_x, y, label,
                  ha="left", va="center",
                  fontsize=9.5, fontweight="bold", color=tc,
                  fontfamily="DejaVu Sans", zorder=5)

    # zero reference
    ax_r.axvline(0, color=NAVY, lw=1.1, alpha=0.4, zorder=2)

    # axes styling
    ax_r.set_yticks(y_pos)
    ax_r.set_yticklabels([""] * N)  # labels already on left panel
    ax_r.set_xlim(-18, MAX_DELTA + 22)
    ax_r.set_ylim(-0.65, N - 0.35)
    ax_r.tick_params(axis="x", colors=MID, labelsize=8.5)
    ax_r.tick_params(axis="y", length=0)
    ax_r.set_xlabel("Hybrid improvement  Δ = (dynamic min clashes) − (hybrid min clashes)",
                    fontsize=9, color=MID, fontfamily="DejaVu Sans", labelpad=5)
    for sp in ax_r.spines.values():
        sp.set_visible(False)
    ax_r.grid(axis="x", color=RULE, lw=0.5, alpha=0.7, zorder=0)

    ax_r.set_title("Minimum Timeslots\n(tightest feasible schedule)",
                   fontsize=10.5, color=NAVY, fontweight="bold",
                   fontfamily="DejaVu Sans", pad=8, linespacing=1.5)

    # legend chips — right panel
    for xpos, fc, ec, txt in [
        (0.01, GREEN_LT, GREEN,   "Hybrid wins  (+Δ)"),
        (0.36, GRAY_LT,  GRAY,    "Tie  (Δ = 0)"),
        (0.62, RED_LT,   RED_COL, "Dynamic wins  (−Δ)"),
    ]:
        chip = FancyBboxPatch((xpos, -0.055), 0.30, 0.052,
                              boxstyle="round,pad=0.01",
                              facecolor=fc, edgecolor=ec,
                              linewidth=1.1,
                              transform=ax_r.transAxes, zorder=3)
        ax_r.add_patch(chip)
        ax_r.text(xpos + 0.015, -0.029, txt,
                  ha="left", va="center", fontsize=8.5,
                  fontweight="bold", color=ec,
                  fontfamily="DejaVu Sans",
                  transform=ax_r.transAxes, zorder=4)

    # "5/12 hybrid" callout inside right panel
    ax_r.text(MAX_DELTA * 0.98, N - 1.3,
              "5 / 12\nhybrid wins",
              ha="right", va="top", fontsize=12,
              fontweight="bold", color=GREEN,
              fontfamily="DejaVu Sans", linespacing=1.4,
              bbox=dict(boxstyle="round,pad=0.4",
                        facecolor=GREEN_LT, edgecolor=GREEN,
                        linewidth=1.3, alpha=0.92),
              zorder=6)

    # ── arrow + label bridging the two panels ─────────────────────────
    fig.text(0.455, 0.74,  "squeeze\ntimeslots",
             ha="center", va="center", fontsize=9,
             color=NAVY, fontfamily="DejaVu Sans",
             fontstyle="italic", linespacing=1.4)
    fig.text(0.455, 0.625, "→",
             ha="center", va="center", fontsize=22,
             color=GOLD, fontfamily="DejaVu Sans",
             fontweight="bold")

    plt.savefig(OUT / "contrast_standard_vs_minimum.png",
                dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓  contrast_standard_vs_minimum.png  →  {OUT}")


if __name__ == "__main__":
    make_plot()
