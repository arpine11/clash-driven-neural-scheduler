"""
generate_custom_visuals.py
--------------------------
Creates two bespoke presentation visuals in a light-beige minimalist style:

  1. conflict_graph_diagram.png
     Exams → Conflict Graph → Colored Schedule (three-panel flow)

  2. gcd_orbit_diagram.png
     gcd(m,S)=1 quasi-periodic orbit  vs.  gcd(m,S)>1 resonant (trapped) orbit

Output directory: outputs/reports/custom_visuals/
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "outputs" / "reports" / "custom_visuals"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared palette  (beige-minimalist)
# ---------------------------------------------------------------------------
BG       = "#F7F3EC"          # warm parchment
PANEL_BG = "#EFEBE1"          # slightly darker panel fill
NAVY     = "#1A2A4A"
GOLD     = "#C8A032"
MID      = "#6B7A8D"
LIGHT_RULE = "#D8CFC0"

# five soft slot colours used for graph colouring
SLOT_COLS = ["#5B8DB8", "#E07B54", "#6BAE75", "#C47EC0", "#E8C442"]

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def circle_pos(n: int, r: float, cx=0.0, cy=0.0, offset_deg=90.0):
    """Return (x,y) positions of n equally-spaced points on a circle."""
    angles = [math.radians(offset_deg - i * 360 / n) for i in range(n)]
    return [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in angles]


def draw_node(ax, x, y, color, label, r=0.09, fs=8, lw=1.6,
              edge_color=NAVY, text_color="white"):
    c = Circle((x, y), r, color=color, zorder=4, linewidth=lw,
               edgecolor=edge_color)
    ax.add_patch(c)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=text_color,
            zorder=5, fontfamily="DejaVu Sans")


def draw_edge(ax, p1, p2, color=MID, lw=1.2, alpha=0.5, zorder=2):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
            color=color, lw=lw, alpha=alpha, zorder=zorder,
            solid_capstyle="round")


def flow_arrow(ax, x0, x1, y, color=NAVY, lw=2.0):
    ax.annotate("",
        xy=(x1, y), xytext=(x0, y),
        arrowprops=dict(arrowstyle="-|>", color=color,
                        lw=lw, mutation_scale=18),
        zorder=6)


def panel_bg(ax, x, y, w, h, color=PANEL_BG):
    r = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.02",
                       facecolor=color, edgecolor=LIGHT_RULE,
                       linewidth=1.0, zorder=1)
    ax.add_patch(r)

# ===========================================================================
# VISUAL 1 — Conflict Graph Diagram
# ===========================================================================

def make_conflict_diagram():
    fig = plt.figure(figsize=(14, 5.2), facecolor=BG)
    ax  = fig.add_axes([0, 0, 1, 1], facecolor=BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── subtle title ──────────────────────────────────────────────────────
    ax.text(7, 4.95, "From Exams to a Conflict-Free Schedule",
            ha="center", va="top", fontsize=13, color=NAVY,
            fontfamily="DejaVu Sans", fontweight="bold", alpha=0.85)

    # ── Panel positions ───────────────────────────────────────────────────
    #   Panel 1: x 0.3–3.7    Panel 2: x 4.7–8.1    Panel 3: x 9.1–13.1
    P1X, P2X, P3X = 0.3, 4.7, 9.1
    PW, PH = 3.4, 3.9
    PY = 0.35

    for px in (P1X, P2X, P3X):
        panel_bg(ax, px, PY, PW, PH)

    # ── Panel labels ──────────────────────────────────────────────────────
    for px, lbl, sub in [
        (P1X, "EXAMS",          "students enrolled"),
        (P2X, "CONFLICT GRAPH", "edge = shared student"),
        (P3X, "VALID SCHEDULE", "color = timeslot"),
    ]:
        ax.text(px + PW/2, PY + PH + 0.1, lbl,
                ha="center", va="bottom", fontsize=9.5, color=NAVY,
                fontweight="bold", fontfamily="DejaVu Sans")
        ax.text(px + PW/2, PY + PH - 0.02, sub,
                ha="center", va="top", fontsize=7.8, color=MID,
                fontstyle="italic", fontfamily="DejaVu Sans")

    # ── Flow arrows between panels ────────────────────────────────────────
    mid_y = PY + PH / 2 + 0.1
    for x0, x1 in [(P1X+PW+0.06, P2X-0.06), (P2X+PW+0.06, P3X-0.06)]:
        flow_arrow(ax, x0, x1, mid_y)

    # ── PANEL 1: Exam list ────────────────────────────────────────────────
    exams = ["Calculus", "Physics", "History", "Biology", "Chemistry", "Algebra"]
    for i, name in enumerate(exams):
        ey = PY + PH - 0.42 - i * 0.56
        rect = FancyBboxPatch((P1X+0.18, ey - 0.19), PW - 0.36, 0.38,
                              boxstyle="round,pad=0.04",
                              facecolor="white", edgecolor=LIGHT_RULE,
                              linewidth=1.0, zorder=3)
        ax.add_patch(rect)
        ax.text(P1X + 0.38, ey,
                f"E{i+1}  {name}",
                ha="left", va="center", fontsize=8.2, color=NAVY,
                zorder=4, fontfamily="DejaVu Sans")

    # ── PANEL 2: Conflict graph (plain, no colours) ───────────────────────
    # 6 nodes in a hexagonal arrangement
    cx2, cy2 = P2X + PW/2, PY + PH/2 + 0.12
    pos2 = circle_pos(6, 1.1, cx2, cy2)

    # conflict edges (pairs that share students)
    edges2 = [(0,1),(0,2),(1,3),(2,3),(2,4),(3,5),(4,5),(1,5)]
    for i, j in edges2:
        draw_edge(ax, pos2[i], pos2[j], color="#8899AA", lw=1.5, alpha=0.7)

    for i, (x, y) in enumerate(pos2):
        draw_node(ax, x, y, NAVY, f"E{i+1}", r=0.2, fs=8,
                  edge_color=GOLD, text_color="white")

    # ── PANEL 3: Colored (solved) schedule ────────────────────────────────
    cx3, cy3 = P3X + PW/2, PY + PH/2 + 0.12
    pos3 = circle_pos(6, 1.1, cx3, cy3)

    # same edges
    for i, j in edges2:
        draw_edge(ax, pos3[i], pos3[j], color="#8899AA", lw=1.5, alpha=0.7)

    # valid colouring (hand-assigned so no two adjacent share a colour)
    # 0-Calc, 1-Phys, 2-Hist, 3-Bio, 4-Chem, 5-Alg
    # adjacency: 0-1,0-2,1-3,2-3,2-4,3-5,4-5,1-5
    slot_assign = [0, 1, 2, 0, 1, 2]   # slot indices
    slot_names  = ["Slot 1", "Slot 2", "Slot 3"]
    for i, (x, y) in enumerate(pos3):
        draw_node(ax, x, y, SLOT_COLS[slot_assign[i]], f"E{i+1}",
                  r=0.2, fs=8, edge_color=NAVY, text_color="white")

    # legend for timeslots
    for k, (sn, sc) in enumerate(zip(slot_names, SLOT_COLS[:3])):
        lx = P3X + 0.25 + k * 1.06
        ly = PY + 0.18
        c = Circle((lx, ly), 0.1, color=sc, zorder=4,
                   linewidth=1.2, edgecolor=NAVY)
        ax.add_patch(c)
        ax.text(lx + 0.15, ly, sn, ha="left", va="center",
                fontsize=7.5, color=NAVY, fontfamily="DejaVu Sans")

    # ── bottom note ────────────────────────────────────────────────────────
    ax.text(7, 0.12,
            "A valid schedule = proper graph colouring  —  NP-complete in general",
            ha="center", va="bottom", fontsize=8.5, color=MID,
            fontstyle="italic", fontfamily="DejaVu Sans")

    plt.savefig(OUT / "conflict_graph_diagram.png",
                dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓  conflict_graph_diagram.png")


# ===========================================================================
# VISUAL 2 — gcd Orbit Diagram
# ===========================================================================

def orbit_points(S: int, m: int, start: int = 0) -> list[int]:
    """Follow the orbit starting at `start` under repeated +m (mod S)."""
    visited, cur = [], start
    while cur not in visited:
        visited.append(cur)
        cur = (cur + m) % S
    return visited


def draw_orbit_panel(ax, cx, cy, S, m, title, subtitle,
                     node_r=0.11, ring_r=1.05, highlight_color=NAVY,
                     label_color=NAVY, arrow_color=GOLD, orbit_only=True):
    """Draw one circular orbit panel centred at (cx, cy)."""
    # faint ring
    ring = Circle((cx, cy), ring_r, fill=False,
                  edgecolor=LIGHT_RULE, linewidth=1.2, linestyle="--", zorder=1)
    ax.add_patch(ring)

    # all S slot positions
    all_pos = circle_pos(S, ring_r, cx, cy)

    # which slots are reachable?
    reachable = set(orbit_points(S, m, 0))

    # draw arrows along the orbit path
    orbit_list = orbit_points(S, m, 0)
    for k in range(len(orbit_list)):
        src = orbit_list[k]
        dst = orbit_list[(k+1) % len(orbit_list)]
        xs, ys = all_pos[src]
        xd, yd = all_pos[dst]
        # shorten arrow so it doesn't overlap nodes
        frac = 0.78
        xm = xs + frac * (xd - xs)
        ym = ys + frac * (yd - ys)
        ax.annotate("",
            xy=(xm, ym), xytext=(xs + 0.12*(xd-xs), ys + 0.12*(yd-ys)),
            arrowprops=dict(arrowstyle="-|>",
                            color=arrow_color, lw=1.3,
                            mutation_scale=10,
                            connectionstyle="arc3,rad=0.0"),
            zorder=3)

    # draw nodes
    for i, (x, y) in enumerate(all_pos):
        if i in reachable:
            col = highlight_color
            tc  = "white"
            ew  = 1.8
        else:
            col = "#DDDBD4"
            tc  = MID
            ew  = 1.0
        c = Circle((x, y), node_r,
                   color=col, zorder=4,
                   linewidth=ew, edgecolor=NAVY if i in reachable else LIGHT_RULE)
        ax.add_patch(c)
        ax.text(x, y, str(i), ha="center", va="center",
                fontsize=6.5, color=tc, fontweight="bold",
                zorder=5, fontfamily="DejaVu Sans")

    # centre label
    ax.text(cx, cy + 0.0, f"m = {m}\nS = {S}",
            ha="center", va="center", fontsize=9,
            color=NAVY, fontweight="bold",
            fontfamily="DejaVu Sans", linespacing=1.5)

    # title
    ax.text(cx, cy + ring_r + 0.3, title,
            ha="center", va="bottom", fontsize=11,
            color=NAVY, fontweight="bold",
            fontfamily="DejaVu Sans")
    ax.text(cx, cy + ring_r + 0.08, subtitle,
            ha="center", va="bottom", fontsize=8.5,
            color=MID, fontstyle="italic",
            fontfamily="DejaVu Sans")

    # reachable fraction tag
    n_reach = len(reachable)
    frac_str = f"{n_reach} / {S} slots reachable"
    col_tag  = "#2A7A3A" if n_reach == S else "#A03020"
    ax.text(cx, cy - ring_r - 0.25, frac_str,
            ha="center", va="top", fontsize=8.5,
            color=col_tag, fontweight="bold",
            fontfamily="DejaVu Sans")


def make_gcd_diagram():
    fig = plt.figure(figsize=(14, 8.2), facecolor=BG)
    ax  = fig.add_axes([0, 0, 1, 1], facecolor=BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── main title ────────────────────────────────────────────────────────
    ax.text(7, 8.08, "Mode m and the Orbit Structure",
            ha="center", va="top", fontsize=13, color=NAVY,
            fontweight="bold", fontfamily="DejaVu Sans")
    ax.text(7, 7.74, "The choice of m determines whether an exam can reach every timeslot",
            ha="center", va="top", fontsize=9, color=MID,
            fontstyle="italic", fontfamily="DejaVu Sans")

    # ── divider ───────────────────────────────────────────────────────────
    ax.plot([7, 7], [0.65, 7.35], color=LIGHT_RULE, lw=1.5,
            linestyle="--", zorder=1)

    # ── formula boxes  (top area) ─────────────────────────────────────────
    BOX_TOP = 6.55
    BOX_H   = 0.92

    lbox = FancyBboxPatch((0.45, BOX_TOP), 5.9, BOX_H,
                          boxstyle="round,pad=0.06",
                          facecolor="#E8F0FE", edgecolor=NAVY,
                          linewidth=1.8, zorder=3)
    ax.add_patch(lbox)
    ax.text(3.4, BOX_TOP + BOX_H * 0.63, "gcd(m, S) = 1",
            ha="center", va="center",
            fontsize=17, color=NAVY, fontweight="bold",
            fontfamily="DejaVu Serif")
    ax.text(3.4, BOX_TOP + BOX_H * 0.22, "quasi-periodic  —  ergodic motion",
            ha="center", va="center", fontsize=9.5, color="#2A5A9A",
            fontstyle="italic", fontfamily="DejaVu Sans")

    rbox = FancyBboxPatch((7.65, BOX_TOP), 5.9, BOX_H,
                          boxstyle="round,pad=0.06",
                          facecolor="#FDE8E8", edgecolor="#A03020",
                          linewidth=1.8, zorder=3)
    ax.add_patch(rbox)
    ax.text(10.6, BOX_TOP + BOX_H * 0.63, "gcd(m, S) > 1",
            ha="center", va="center",
            fontsize=17, color="#A03020", fontweight="bold",
            fontfamily="DejaVu Serif")
    ax.text(10.6, BOX_TOP + BOX_H * 0.22, "resonant  —  trapped in smaller orbit",
            ha="center", va="center", fontsize=9.5, color="#A03020",
            fontstyle="italic", fontfamily="DejaVu Sans")

    # ── orbit panels  (centre area) ───────────────────────────────────────
    # ring_r=1.6 → top of ring at cy+1.6, bottom at cy-1.6
    # reachable tag drawn at cy - ring_r - 0.25
    # We want reachable tag top at y≈3.05, description lines below at ≈2.65 and 2.30
    # bottom note at y=0.72
    # → cy - 1.6 - 0.25 = 3.05  →  cy = 4.9
    ORBIT_CY  = 4.9
    RING_R    = 1.6
    NODE_R    = 0.185

    draw_orbit_panel(ax, cx=3.4,  cy=ORBIT_CY,
                     S=8, m=3, title="", subtitle="",
                     highlight_color=NAVY, arrow_color=GOLD,
                     ring_r=RING_R, node_r=NODE_R)

    draw_orbit_panel(ax, cx=10.6, cy=ORBIT_CY,
                     S=8, m=4, title="", subtitle="",
                     highlight_color="#A03020", arrow_color="#E07B54",
                     ring_r=RING_R, node_r=NODE_R)

    # ── descriptive text — two lines, well below reachable tag ────────────
    # reachable tag is at cy - RING_R - 0.25 = 4.9 - 1.6 - 0.25 = 3.05
    # tag text height (va=bottom) sits ABOVE 3.05 → safe
    # our lines start at 2.72 → clear gap
    DESC_Y1 = 2.72
    DESC_Y2 = 2.38

    for cx, line1, line2, col in [
        (3.4,
         "Exam visits every timeslot in turn.",
         "Maximum search-space coverage.",
         "#2A7A3A"),
        (10.6,
         "Exam trapped in a small sub-cycle.",
         "Cannot escape — algorithm stalls.",
         "#A03020"),
    ]:
        ax.text(cx, DESC_Y1, line1, ha="center", va="top",
                fontsize=8.8, color=col, fontfamily="DejaVu Sans")
        ax.text(cx, DESC_Y2, line2, ha="center", va="top",
                fontsize=8.8, color=col, fontfamily="DejaVu Sans")

    # ── bottom note ────────────────────────────────────────────────────────
    ax.text(7, 0.76,
            "Choose m coprime with S to guarantee full ergodic coverage of the timeslot space",
            ha="center", va="bottom", fontsize=8.5,
            color=MID, fontstyle="italic", fontfamily="DejaVu Sans")

    plt.savefig(OUT / "gcd_orbit_diagram.png",
                dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓  gcd_orbit_diagram.png")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating custom presentation visuals...")
    make_conflict_diagram()
    make_gcd_diagram()
    print(f"\nAll outputs → {OUT}")
