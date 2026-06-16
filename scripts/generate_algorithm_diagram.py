"""
generate_algorithm_diagram.py  (v2 — clean professional version)
-----------------------------------------------------------------
Minimalist flowchart for Slide 5 — The Algorithm.
Output: outputs/reports/custom_visuals/algorithm_diagram.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import FancyArrowPatch

OUT = Path(__file__).resolve().parent.parent / "outputs" / "reports" / "custom_visuals"
OUT.mkdir(parents=True, exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────
BG      = "#F7F3EC"
NAVY    = "#1A2A4A"
MID     = "#6B7A8D"
RULE    = "#D8CFC0"

BLUE_F  = "#D6E4F7";  BLUE_E  = "#4C72B0"   # E Operator
ORG_F   = "#FCE8D8";  ORG_E   = "#C8593A"   # Hopfield
GRN_F   = "#D4EDDA";  GRN_E   = "#3A8A4A"   # Done / Keep
RED_F   = "#FDE8E8";  RED_E   = "#A03020"   # Rollback
GRAY_F  = "#EDEAE3";  GRAY_E  = "#8899AA"   # Init / neutral


# ── Drawing helpers ────────────────────────────────────────────────────────

def rect(ax, cx, cy, w, h, label, note="",
         fc=GRAY_F, ec=GRAY_E, label_fs=12, note_fs=8.5):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.07",
        facecolor=fc, edgecolor=ec, linewidth=2.0, zorder=4))
    ly = cy + (0.13 if note else 0)
    ax.text(cx, ly, label, ha="center", va="center",
            fontsize=label_fs, fontweight="bold",
            color=NAVY, fontfamily="DejaVu Sans", zorder=5)
    if note:
        ax.text(cx, cy - 0.20, note, ha="center", va="center",
                fontsize=note_fs, color=MID, fontstyle="italic",
                fontfamily="DejaVu Sans", zorder=5)


def arr(ax, x0, y0, x1, y1, color=NAVY, lw=1.8,
        rad=0.0, label="", lside="right"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=16,
                                connectionstyle=f"arc3,rad={rad}"))
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ox = 0.22 if lside == "right" else -0.22
        ax.text(mx+ox, my, label, ha="left" if lside=="right" else "right",
                va="center", fontsize=9, color=color,
                fontstyle="italic", fontfamily="DejaVu Sans", zorder=5)


def make():
    W, H = 12.5, 10.2
    fig, ax = plt.subplots(figsize=(W, H), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_aspect("equal"); ax.axis("off")

    # ── layout constants ───────────────────────────────────────────────
    CX   = 5.5          # main column
    BW   = 3.8          # main box width
    BH   = 0.92         # main box height
    SBW  = 2.1          # small box width
    SBH  = 0.80         # small box height

    DONE_X = 10.0       # Done box column
    KEEP_X = 3.6        # Keep box column
    ROLL_X = 7.4        # Rollback box column
    LOOP_X = 0.85       # left loop-back rail x

    Y_INIT  = 9.50
    Y_ELOOP = 7.90
    Y_SEP   = 6.68
    Y_HOP   = 5.60
    Y_CHK   = 4.20
    Y_OUT   = 2.85

    GAP = 0.08          # small arrow gap

    # ── 1. Initialize ─────────────────────────────────────────────────
    rect(ax, CX, Y_INIT, BW, BH,
         "Random schedule",
         note="assign each exam a random timeslot",
         fc=GRAY_F, ec=GRAY_E)

    arr(ax, CX, Y_INIT - BH/2 - GAP,
            CX, Y_ELOOP + BH/2 + GAP, color=NAVY)

    # ── 2. E Operator ─────────────────────────────────────────────────
    rect(ax, CX, Y_ELOOP, BW + 0.2, BH,
         "E Operator  (dynamic walk)",
         note="sort by conflicts  ·  shift clashing exam +1 slot  ·  repeat",
         fc=BLUE_F, ec=BLUE_E)

    # → Done (zero clashes)
    arr(ax, CX + (BW+0.2)/2 + GAP, Y_ELOOP,
            DONE_X - SBW/2 - GAP, Y_ELOOP,
        color=GRN_E, lw=1.6)
    # label above the arrow
    ax.text((CX + (BW+0.2)/2 + DONE_X - SBW/2) / 2,
            Y_ELOOP + 0.22,
            "zero clashes", ha="center", va="bottom",
            fontsize=9, color=GRN_E, fontstyle="italic",
            fontfamily="DejaVu Sans")
    rect(ax, DONE_X, Y_ELOOP, SBW, SBH,
         "Done  ✓", fc=GRN_F, ec=GRN_E, label_fs=12)

    # ── dashed separator ──────────────────────────────────────────────
    ax.plot([0.5, W - 0.3], [Y_SEP, Y_SEP],
            color=RULE, lw=1.3, linestyle="--", zorder=2)
    ax.text(W - 0.35, Y_SEP + 0.08, "hybrid only  ↓",
            ha="right", va="bottom", fontsize=8.5,
            color=ORG_E, fontweight="bold", fontfamily="DejaVu Sans")

    # ↓ stuck
    arr(ax, CX, Y_ELOOP - BH/2 - GAP,
            CX, Y_HOP + BH/2 + GAP,
        color=MID, lw=1.5, label="stuck", lside="right")

    # ── 3. Hopfield Recall ────────────────────────────────────────────
    rect(ax, CX, Y_HOP, BW + 0.2, BH,
         "Hopfield Recall",
         note="pull clashing exams toward nearest stored low-clash snapshot",
         fc=ORG_F, ec=ORG_E)

    arr(ax, CX, Y_HOP - BH/2 - GAP,
            CX, Y_CHK + SBH/2 + GAP, color=NAVY)

    # ── 4. Decision box ───────────────────────────────────────────────
    rect(ax, CX, Y_CHK, BW, SBH * 0.88,
         "Clashes improved?",
         fc=GRAY_F, ec=GRAY_E, label_fs=11)

    # → Keep (yes, left)
    arr(ax, CX - BW/2 - GAP, Y_CHK,
            KEEP_X + SBW/2 + GAP, Y_OUT,
        color=GRN_E, lw=1.6, rad=-0.2,
        label="yes", lside="left")
    rect(ax, KEEP_X, Y_OUT, SBW, SBH,
         "Keep", fc=GRN_F, ec=GRN_E, label_fs=12)

    # → Rollback (no, right)
    arr(ax, CX + BW/2 + GAP, Y_CHK,
            ROLL_X - SBW/2 - GAP, Y_OUT,
        color=RED_E, lw=1.6, rad=0.2,
        label="no", lside="right")
    rect(ax, ROLL_X, Y_OUT, SBW, SBH,
         "Rollback", fc=RED_F, ec=RED_E, label_fs=12)

    # ── loop-back: both Keep & Rollback → left rail → E Operator ──────
    JOIN_Y = Y_OUT - SBH/2 - 0.40

    # Keep bottom → join point (straight down then left)
    ax.plot([KEEP_X, KEEP_X, LOOP_X],
            [Y_OUT - SBH/2, JOIN_Y, JOIN_Y],
            color=MID, lw=1.5, zorder=3,
            solid_capstyle="round", solid_joinstyle="round")

    # Rollback bottom → join point
    ax.plot([ROLL_X, ROLL_X, LOOP_X],
            [Y_OUT - SBH/2, JOIN_Y, JOIN_Y],
            color=MID, lw=1.5, zorder=3,
            solid_capstyle="round", solid_joinstyle="round")

    # vertical rail up to E Operator row
    ax.annotate("",
        xy=(LOOP_X, Y_ELOOP),
        xytext=(LOOP_X, JOIN_Y),
        arrowprops=dict(arrowstyle="-|>", color=MID, lw=1.5,
                        mutation_scale=14))

    # rail → E operator left edge (horizontal)
    ax.annotate("",
        xy=(CX - (BW+0.2)/2 - GAP, Y_ELOOP),
        xytext=(LOOP_X, Y_ELOOP),
        arrowprops=dict(arrowstyle="-|>", color=MID, lw=1.5,
                        mutation_scale=14))

    ax.text(LOOP_X - 0.1, (Y_ELOOP + JOIN_Y) / 2,
            "back to\nE Operator",
            ha="right", va="center", fontsize=8.5,
            color=MID, fontstyle="italic",
            fontfamily="DejaVu Sans", linespacing=1.5)

    # ── legend row ─────────────────────────────────────────────────────
    LY = 0.62
    chips = [
        (2.5,  BLUE_F, BLUE_E, "E Operator"),
        (5.1,  ORG_F,  ORG_E,  "Hopfield Recall"),
        (7.9,  RED_F,  RED_E,  "Rollback safeguard"),
        (10.7, GRN_F,  GRN_E,  "Solved  ✓"),
    ]
    for cx_c, fc, ec, lbl in chips:
        ax.add_patch(FancyBboxPatch(
            (cx_c - 1.0, LY - 0.22), 2.0, 0.44,
            boxstyle="round,pad=0.05",
            facecolor=fc, edgecolor=ec, linewidth=1.3, zorder=4))
        ax.text(cx_c, LY, lbl, ha="center", va="center",
                fontsize=8.8, fontweight="bold",
                color=ec, fontfamily="DejaVu Sans", zorder=5)

    plt.savefig(OUT / "algorithm_diagram.png",
                dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓  algorithm_diagram.png  →  {OUT}")


if __name__ == "__main__":
    make()
