"""
generate_algorithm_plot.py
--------------------------
Creates a presentation-quality algorithm comparison plot
for the "Algorithm" slide. Beige minimalist style matching
the custom visuals in the presentation.

Output: outputs/reports/custom_visuals/algorithm_comparison.png
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

ROOT  = Path(__file__).resolve().parent.parent
HIST  = ROOT / "outputs" / "min_timeslot_experiment" / "clash_dynamics" / "histories"
OUT   = ROOT / "outputs" / "reports" / "custom_visuals"
OUT.mkdir(parents=True, exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────
BG         = "#F7F3EC"
PANEL_BG   = "#EFEBE1"
NAVY       = "#1A2A4A"
GOLD       = "#C8A032"
MID        = "#6B7A8D"
RULE       = "#D8CFC0"
BLUE_DYN   = "#4C72B0"
ORG_HYB    = "#C8593A"
GREEN_MARK = "#2A7A3A"


def running_min(arr: np.ndarray) -> np.ndarray:
    """Cumulative minimum (best-so-far at each event)."""
    out = np.empty_like(arr, dtype=float)
    cur = arr[0]
    for i, v in enumerate(arr):
        if v < cur:
            cur = v
        out[i] = cur
    return out


def plot_instance(ax, inst: str, title: str, show_ylabel: bool,
                  show_xlabel: bool = True):
    dyn_raw = np.load(HIST / f"{inst}_dynamic_only.npy").astype(float)
    hyb_raw = np.load(HIST / f"{inst}_hybrid.npy").astype(float)

    # skip event 0 (random init spike: tens-of-thousands of clashes)
    dyn_raw = dyn_raw[1:]
    hyb_raw = hyb_raw[1:]

    xd = np.arange(1, len(dyn_raw) + 1)
    xh = np.arange(1, len(hyb_raw) + 1)

    dyn_rm = running_min(dyn_raw)
    hyb_rm = running_min(hyb_raw)

    dmin = int(dyn_rm[-1])
    hmin = int(hyb_rm[-1])
    xmax = max(len(xd), len(xh))

    # ── raw oscillations (very faint fill) ────────────────────────────
    ax.fill_between(xd, dyn_raw, dmin, alpha=0.07, color=BLUE_DYN, zorder=1)
    ax.fill_between(xh, hyb_raw, hmin, alpha=0.07, color=ORG_HYB,  zorder=1)
    ax.plot(xd, dyn_raw, color=BLUE_DYN, lw=0.5, alpha=0.22, zorder=2)
    ax.plot(xh, hyb_raw, color=ORG_HYB,  lw=0.5, alpha=0.22, zorder=2)

    # ── running-minimum curves (bold) ─────────────────────────────────
    ax.plot(xd, dyn_rm, color=BLUE_DYN, lw=2.5, alpha=0.95,
            zorder=4, solid_capstyle="round", label="Dynamic-only  (best so far)")
    ax.plot(xh, hyb_rm, color=ORG_HYB,  lw=2.5, alpha=0.95,
            zorder=4, solid_capstyle="round", label="Hybrid  (best so far)")

    # ── horizontal plateau lines ───────────────────────────────────────
    ax.axhline(dmin, color=BLUE_DYN, lw=1.1, ls="--", alpha=0.50, zorder=3)
    ax.axhline(hmin, color=ORG_HYB,  lw=1.1, ls="--", alpha=0.50, zorder=3)

    # right-side minimum labels
    margin = xmax * 0.05
    ax.text(xmax + margin, dmin, f"{dmin}", color=BLUE_DYN,
            va="center", fontsize=9, fontweight="bold",
            fontfamily="DejaVu Sans", zorder=6)
    ax.text(xmax + margin, hmin, f"{hmin}", color=ORG_HYB,
            va="center", fontsize=9, fontweight="bold",
            fontfamily="DejaVu Sans", zorder=6)

    # ── Δ bracket between final minima (log-scale safe) ───────────────
    delta = dmin - hmin
    if delta > 0:
        bx    = xmax * 0.78
        # geometric midpoint on log scale
        mid_y = np.exp((np.log(dmin) + np.log(max(hmin, 1))) / 2)
        ax.annotate("",
            xy=(bx, hmin * 1.02), xytext=(bx, dmin * 0.98),
            arrowprops=dict(arrowstyle="<->", color=GOLD,
                            lw=1.8, mutation_scale=11),
            zorder=5)
        ax.text(bx + xmax * 0.03, mid_y,
                f"Δ = {delta}",
                color=GOLD, fontsize=10, fontweight="bold",
                va="center", fontfamily="DejaVu Sans", zorder=7,
                bbox=dict(boxstyle="round,pad=0.18",
                          facecolor=BG, edgecolor=GOLD,
                          linewidth=1.0, alpha=0.9))

    # ── "Hopfield escape" annotation ─────────────────────────────────
    # mark the point where the hybrid running-min first drops below dynamic
    min_len = min(len(dyn_rm), len(hyb_rm))
    gap     = dyn_rm[:min_len] - hyb_rm[:min_len]
    escape  = np.where(gap > delta * 0.2)[0]
    if len(escape) > 0:
        ex = int(escape[0]) + 1      # +1 because we skipped event 0
        ey = hyb_rm[escape[0]]
        ax.axvline(ex, color=MID, lw=0.9, ls=":", alpha=0.55, zorder=1)
        ax.text(ex + xmax * 0.015, ey * 1.12,
                "Hopfield\nescape",
                color=MID, fontsize=7.8, va="bottom",
                fontfamily="DejaVu Sans", fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.22",
                          facecolor=BG, edgecolor=RULE,
                          linewidth=0.8, alpha=0.92),
                zorder=7)

    # ── axes style ─────────────────────────────────────────────────────
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values():
        sp.set_color(RULE); sp.set_linewidth(0.9)

    ax.set_yscale("log")

    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", color=NAVY,
                     pad=8, fontfamily="DejaVu Sans")
    if show_xlabel:
        ax.set_xlabel("Event index  (after initialisation)", fontsize=9,
                      color=MID, fontfamily="DejaVu Sans")
    if show_ylabel:
        ax.set_ylabel("Total clashes  (log scale)", fontsize=9, color=MID,
                      fontfamily="DejaVu Sans")
    ax.tick_params(colors=MID, labelsize=8.5, which="both")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5, integer=True))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}" if x >= 1 else ""))
    ax.grid(True, color=RULE, lw=0.6, alpha=0.65, zorder=0, which="major")
    ax.grid(True, color=RULE, lw=0.3, alpha=0.35, zorder=0, which="minor")

    ymax = max(dyn_raw.max(), hyb_raw.max())
    ax.set_xlim(0, xmax * 1.16)
    ax.set_ylim(max(1, hmin * 0.6), ymax * 2.2)


def make_plot():
    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 5.4),
        facecolor=BG,
        gridspec_kw={"wspace": 0.32},
    )
    fig.patch.set_facecolor(BG)

    # ── super-title ────────────────────────────────────────────────────
    fig.text(0.5, 0.97,
             "Dynamic-Only  vs.  Hybrid — Clash Reduction at Minimum Timeslots",
             ha="center", va="top", fontsize=13, fontweight="bold",
             color=NAVY, fontfamily="DejaVu Sans")
    fig.text(0.5, 0.913,
             "bold line = running minimum (best-so-far) · "
             "faint trace = raw oscillation · "
             "y-axis: log scale",
             ha="center", va="top", fontsize=8.5,
             color=MID, fontfamily="DejaVu Sans", fontstyle="italic")

    plot_instance(axes[0], "car-s-91",
                  title="car-s-91   (682 exams, 28 min-slots)",
                  show_ylabel=True)

    plot_instance(axes[1], "uta-s-92",
                  title="uta-s-92   (622 exams, 30 min-slots)",
                  show_ylabel=False)

    # ── shared legend ──────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], color=BLUE_DYN, lw=2.4, label="Dynamic-only"),
        Line2D([0], [0], color=ORG_HYB,  lw=2.4, label="Hybrid  (+Hopfield network)"),
        Line2D([0], [0], color=GOLD,     lw=1.6,
               linestyle="--", label="Δ = improvement from Hopfield"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=3, fontsize=9.5,
               frameon=True, framealpha=0.95,
               facecolor=BG, edgecolor=RULE,
               bbox_to_anchor=(0.5, -0.01))

    plt.savefig(OUT / "algorithm_comparison.png",
                dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓  algorithm_comparison.png  →  {OUT}")


def make_uta_solo():
    """Single clean panel for uta-s-92, no title/subtitle/legend text."""
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG)
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(left=0.13, right=0.88, top=0.96, bottom=0.13)

    plot_instance(ax, "uta-s-92", title="", show_ylabel=False,
                  show_xlabel=False)

    # clean legend inside the plot, minimal
    legend_elements = [
        Line2D([0], [0], color=BLUE_DYN, lw=2.4, label="Dynamic-only"),
        Line2D([0], [0], color=ORG_HYB,  lw=2.4, label="Hybrid  (+Hopfield)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9,
              frameon=True, framealpha=0.95,
              facecolor=BG, edgecolor=RULE,
              loc="upper right")

    plt.savefig(OUT / "uta_s92_solo.png",
                dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓  uta_s92_solo.png  →  {OUT}")


if __name__ == "__main__":
    make_plot()
    make_uta_solo()
