"""
Corrected clash-dynamics visualization with rolling correlation curves.

For each dataset in the paper-minimum timeslot experiment, generates a
two-panel figure:
  top panel    – dynamic-only clash curve + hybrid clash curve over event index
  bottom panel – rolling Pearson correlation between the two curves over time

What changed vs the previous scatter approach
---------------------------------------------
Previously: one scatter plot per instance (dynamic clashes on x, hybrid
clashes on y, one point per event) — this showed *which* clash levels
co-occurred but lost all temporal information.

Now: the correlation is itself a *curve* over the event index.  At each
time t we take a sliding window of the two trajectories ending at t and
compute Pearson r.  This shows *how* the relationship between the two
methods evolves — whether they track each other at the start, diverge as
hybrid Hopfield calls kick in, converge again, etc.

Alignment strategy
------------------
Both histories are truncated to min(len_dyn, len_hyb).  This is the
most conservative choice: we compare only the events that both methods
have in common.  The truncation point is noted in the plot title.

Window size
-----------
W = min(20, max(5, n_common // 4))  — adaptive:
  • large histories (≥80 events): W = 20
  • ute-s-92 (common=72):          W = 18
  • sta-f-83 (common=9):           W = 5 (only 5 valid points; noted)
Constant windows (std == 0 in either slice) → r = NaN →
matplotlib leaves a gap in the line (no fake flat segments).

Outputs
-------
outputs/min_timeslot_experiment/clash_dynamics/rolling_plots/
  {instance}_rolling.png        two-panel figure per dataset
  rolling_correlation_summary.csv
  rolling_correlation_overview.png   average r bar chart across all instances

Previous scatter plots in clash_dynamics/correlation_plots/ are UNTOUCHED.

Usage:
    python3 scripts/run_clash_dynamics_rolling.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

PAPER_MIN_SLOTS: dict[str, int] = {
    "hec-s-92": 17, "sta-f-83": 13, "ute-s-92": 10,
    "lse-f-91": 17, "yor-f-83": 19, "ear-f-83": 22,
    "kfu-s-93": 19, "tre-s-92": 20, "rye-s-93": 21,
    "car-f-92": 28, "uta-s-92": 30, "car-s-91": 28,
}

COLOR_DYN = "#4C72B0"   # blue
COLOR_HYB = "#DD8452"   # orange
COLOR_COR = "#2ca02c"   # green

INSTANCE_ORDER = sorted(PAPER_MIN_SLOTS)


# ---------------------------------------------------------------------------
# Rolling correlation
# ---------------------------------------------------------------------------

def _pearsonr_window(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r for a single window; returns NaN if either side is constant."""
    sx, sy = float(np.std(x)), float(np.std(y))
    if sx == 0 or sy == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rolling_pearson(d: np.ndarray, h: np.ndarray, W: int) -> np.ndarray:
    """
    Compute rolling Pearson r between aligned arrays d and h.
    Result[i] = r over window [i-W+1 … i]; undefined (NaN) for i < W-1
    or when either window slice is constant.
    """
    n = len(d)
    r = np.full(n, np.nan)
    for i in range(W - 1, n):
        r[i] = _pearsonr_window(d[i - W + 1:i + 1], h[i - W + 1:i + 1])
    return r


def adaptive_window(n_common: int) -> int:
    return min(20, max(5, n_common // 4))


# ---------------------------------------------------------------------------
# Two-panel plot
# ---------------------------------------------------------------------------

def plot_rolling(
    inst_name: str,
    dyn_full: np.ndarray,
    hyb_full: np.ndarray,
    out_dir: Path,
) -> dict:
    """
    Create the two-panel figure and return a summary dict.
    """
    n = min(len(dyn_full), len(hyb_full))
    d = dyn_full[:n].astype(float)
    h = hyb_full[:n].astype(float)
    W = adaptive_window(n)
    roll_r = rolling_pearson(d, h, W)

    S       = PAPER_MIN_SLOTS[inst_name]
    x       = np.arange(n)
    x_valid = np.where(~np.isnan(roll_r))[0]

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(11, 6),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.08},
    )

    # ── top panel: clash curves ──────────────────────────────────────────
    nonzero = np.concatenate([d[d > 0], h[h > 0]])
    use_log = nonzero.size > 0 and nonzero.max() / max(1, nonzero.min()) > 100

    ax1.plot(x, d, color=COLOR_DYN, linewidth=1.0, alpha=0.9,
             label=f"Dynamic-only  (min={int(d.min())})")
    ax1.plot(x, h, color=COLOR_HYB, linewidth=1.0, alpha=0.9,
             label=f"Hybrid        (min={int(h.min())})")

    for arr, col, tag in [(d, COLOR_DYN, "Dyn"), (h, COLOR_HYB, "Hyb")]:
        zeros = np.where(arr == 0)[0]
        if zeros.size:
            t0 = zeros[0]
            ax1.axvline(t0, color=col, linestyle=":", linewidth=0.9, alpha=0.6)
            ax1.scatter([t0], [0], color=col, s=55, zorder=5, marker="*",
                        label=f"{tag} → zero at t={t0}")

    if use_log:
        ax1.set_yscale("log")
        ax1.set_ylabel("Total clashes (log scale)")
    else:
        ax1.set_ylabel("Total clashes")

    trunc_note = ""
    if len(dyn_full) != len(hyb_full):
        longer = "dynamic" if len(dyn_full) > len(hyb_full) else "hybrid"
        trunc_note = f"  [truncated to n={n}; {longer} had more events]"

    ax1.set_title(
        f"{inst_name}  —  clash dynamics  "
        f"(S={S}, paper-minimum timeslots{trunc_note})",
        fontsize=10,
    )
    ax1.legend(fontsize=7.5, loc="upper right")
    ax1.grid(True, alpha=0.22)

    # ── bottom panel: rolling correlation ────────────────────────────────
    # plot only defined points; NaN gaps appear naturally
    ax2.plot(x, roll_r, color=COLOR_COR, linewidth=1.1,
             label=f"Rolling Pearson r  (W={W})", zorder=3)
    ax2.axhline(0,    color="black",    linewidth=0.7, linestyle="--", alpha=0.5)
    ax2.axhline( 0.8, color="gray",     linewidth=0.5, linestyle=":",  alpha=0.4)
    ax2.axhline(-0.8, color="gray",     linewidth=0.5, linestyle=":",  alpha=0.4)
    ax2.fill_between(x, roll_r, 0,
                     where=~np.isnan(roll_r),
                     alpha=0.12, color=COLOR_COR)

    if x_valid.size:
        mean_r = float(np.nanmean(roll_r))
        ax2.axhline(mean_r, color=COLOR_COR, linewidth=0.8, linestyle="--",
                    alpha=0.7, label=f"mean r = {mean_r:.3f}")

    ax2.set_ylim(-1.15, 1.15)
    ax2.set_ylabel("Rolling r", fontsize=9)
    ax2.set_xlabel("Event index  (dynamic step or Hopfield call)")
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax2.legend(fontsize=7.5, loc="lower right")
    ax2.grid(True, alpha=0.22)

    out_path = out_dir / f"{inst_name}_rolling.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {inst_name:<12}  n={n:>4}  W={W:>2}  mean_r={np.nanmean(roll_r):+.3f}  "
          f"dyn_min={int(d.min()):>5}  hyb_min={int(h.min()):>5}  → {out_path.name}")

    # ── summary row ──────────────────────────────────────────────────────
    mean_r_val = float(np.nanmean(roll_r)) if x_valid.size else float("nan")
    pct_defined = round(100 * x_valid.size / max(1, n), 1)

    dyn_min, hyb_min = int(d.min()), int(h.min())
    dyn_fin, hyb_fin = int(d[-1]),   int(h[-1])

    if dyn_min == 0 and hyb_min == 0:
        interp = "both solve to zero"
    elif not np.isnan(mean_r_val):
        if mean_r_val > 0.8:
            corr_desc = "strongly correlated"
        elif mean_r_val > 0.4:
            corr_desc = "moderately correlated"
        elif mean_r_val > 0.0:
            corr_desc = "weakly correlated"
        else:
            corr_desc = "anti-correlated / diverging"

        if hyb_min < dyn_min:
            outcome = "hybrid improves"
        elif dyn_min < hyb_min:
            outcome = "dynamic-only better"
        else:
            outcome = "equal min-clash"

        interp = f"{corr_desc}; {outcome}"
    else:
        interp = "insufficient data for correlation"

    return {
        "instance_name":         inst_name,
        "min_slots":             S,
        "common_length":         n,
        "window_W":              W,
        "pct_windows_defined":   pct_defined,
        "mean_rolling_r":        round(mean_r_val, 4) if not np.isnan(mean_r_val) else "",
        "dynamic_min_clash":     dyn_min,
        "hybrid_min_clash":      hyb_min,
        "dynamic_final_clash":   dyn_fin,
        "hybrid_final_clash":    hyb_fin,
        "interpretation":        interp,
    }


# ---------------------------------------------------------------------------
# Overview bar chart of mean rolling r
# ---------------------------------------------------------------------------

def plot_overview(rows: list[dict], out_dir: Path) -> None:
    names  = [r["instance_name"]  for r in rows]
    mean_r = [float(r["mean_rolling_r"]) if r["mean_rolling_r"] != "" else 0.0
              for r in rows]

    colors = [
        "#2ca02c" if v > 0.4 else
        "#aec7e8" if v > 0    else
        "#d62728"
        for v in mean_r
    ]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(13, 4))
    bars = ax.bar(x, mean_r, color=colors, alpha=0.85)
    ax.axhline(0,   color="black", linewidth=0.7)
    ax.axhline(0.8, color="gray",  linewidth=0.5, linestyle=":", alpha=0.6,
               label="r = 0.8 reference")

    for bar, v in zip(bars, mean_r):
        ax.annotate(
            f"{v:+.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, v),
            xytext=(0, 4 if v >= 0 else -12),
            textcoords="offset points",
            ha="center", fontsize=7.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{n}\n(S={PAPER_MIN_SLOTS[n]})" for n in names],
        rotation=30, ha="right", fontsize=8,
    )
    ax.set_ylim(-1.1, 1.25)
    ax.set_ylabel("Mean rolling Pearson r  (over all defined windows)")
    ax.set_title("Average rolling correlation between dynamic-only and hybrid clash trajectories\n"
                 "(paper-minimum timeslots; green = correlated, red = diverging)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.22)
    fig.tight_layout()
    path = out_dir / "rolling_correlation_overview.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Overview → {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hist-dir",
        default=str(ROOT / "outputs" / "min_timeslot_experiment" / "clash_dynamics" / "histories"),
        help="Directory containing *_dynamic_only.npy and *_hybrid.npy files",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "outputs" / "min_timeslot_experiment" / "clash_dynamics" / "rolling_plots"),
        help="Directory to write rolling plots and summary CSV into",
    )
    args = parser.parse_args()

    hist_dir = Path(args.hist_dir)
    if not hist_dir.exists():
        print(f"ERROR: {hist_dir} not found.\n"
              "Run scripts/run_clash_dynamics.py first to generate history arrays.")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading histories from {hist_dir}")
    print(f"Writing plots to      {out_dir}\n")
    print(f"{'Instance':<12}  {'n':>4}  {'W':>2}  {'mean r':>8}  {'dyn_min':>8}  "
          f"{'hyb_min':>8}  output")
    print("-" * 80)

    rows: list[dict] = []
    for inst_name in INSTANCE_ORDER:
        dyn_path = hist_dir / f"{inst_name}_dynamic_only.npy"
        hyb_path = hist_dir / f"{inst_name}_hybrid.npy"
        if not dyn_path.exists() or not hyb_path.exists():
            print(f"  {inst_name}: history files not found, skipping")
            continue
        dyn_h = np.load(dyn_path)
        hyb_h = np.load(hyb_path)
        row = plot_rolling(inst_name, dyn_h, hyb_h, out_dir)
        rows.append(row)

    # ── save summary CSV ──────────────────────────────────────────────────
    csv_path = out_dir.parent / "rolling_correlation_summary.csv"
    COLS = [
        "instance_name", "min_slots", "common_length", "window_W",
        "pct_windows_defined", "mean_rolling_r",
        "dynamic_min_clash", "hybrid_min_clash",
        "dynamic_final_clash", "hybrid_final_clash",
        "interpretation",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Summary CSV → {csv_path.name}")

    # ── overview plot ─────────────────────────────────────────────────────
    print()
    plot_overview(rows, out_dir)

    # ── print table ───────────────────────────────────────────────────────
    print()
    print(f"{'Instance':<12}  {'S':>3}  {'n':>4}  {'W':>2}  {'mean r':>8}  "
          f"{'dyn min':>7}  {'hyb min':>7}  interpretation")
    print("-" * 90)
    for r in rows:
        print(f"{r['instance_name']:<12}  {r['min_slots']:>3}  "
              f"{r['common_length']:>4}  {r['window_W']:>2}  "
              f"{str(r['mean_rolling_r']):>8}  "
              f"{r['dynamic_min_clash']:>7}  {r['hybrid_min_clash']:>7}  "
              f"{r['interpretation']}")

    print(f"\nAll outputs → {out_dir}")


if __name__ == "__main__":
    main()
