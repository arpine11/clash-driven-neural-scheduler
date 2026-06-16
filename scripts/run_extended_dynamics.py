"""
Extended clash-dynamics experiment — 10,000-event budget.

Investigates whether increasing the event budget from ~350–1,000 to 10,000
leads to further clash reduction, new zero-clash solutions, or convergence
plateaus.

Dynamic-only  : 10,000 E-steps (up from 400 / 1,000)
Hybrid        : same (instance, m, seed, pattern) as best config from
                raw_results.csv, but with enough cycles to reach ~10,000
                total events:  n_cycles = (target − 1) // events_per_cycle

Memory        : 10,000 int32 values = 40 KB per run; 24 runs ≈ 1 MB total.

Outputs (all in outputs/min_timeslot_experiment/extended_dynamics/):
  histories/              .npy per (instance, method)
  plots/                  per-instance clash-vs-time PNG (both methods)
  rolling_plots/          two-panel rolling-correlation PNG per instance
  extended_summary.csv    per-(instance, method): min, final, zero, improvement
  rolling_summary.csv     rolling r stats per instance pair
  analysis.txt            plain-text findings section

Previous outputs in clash_dynamics/ are UNTOUCHED.

Usage:
    python3 scripts/run_extended_dynamics.py
    python3 scripts/run_extended_dynamics.py --workers 4
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from experiment_utils import (
    load_instance,
    run_dynamic_only,
    run_hybrid_schedule,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXTENDED_MAX_STEPS = 10_000
TARGET_EVENTS      = 10_000

PAPER_MIN_SLOTS: dict[str, int] = {
    "hec-s-92": 17, "sta-f-83": 13, "ute-s-92": 10,
    "lse-f-91": 17, "yor-f-83": 19, "ear-f-83": 22,
    "kfu-s-93": 19, "tre-s-92": 20, "rye-s-93": 21,
    "car-f-92": 28, "uta-s-92": 30, "car-s-91": 28,
}

PATTERNS = {
    "p_10_100": [10, 100],
    "p_50_100": [50, 100],
    "p_100":    [100],
}

ORDERING      = "largest-weighted-degree"
INSTANCE_ORDER = sorted(PAPER_MIN_SLOTS)

COLOR_DYN = "#4C72B0"
COLOR_HYB = "#DD8452"
COLOR_COR = "#2ca02c"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def scaled_cycles(pattern: list[int], target: int = TARGET_EVENTS) -> int:
    """Number of cycles to reach ≈target total events with this pattern."""
    if not pattern:
        return 3
    # each cycle: sum(segment_lengths) dynamic steps + len(pattern) Hopfield calls
    events_per_cycle = sum(pattern) + len(pattern)
    return max(3, (target - 1) // events_per_cycle)


def _safe_int(val, default: int = -1) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _pearsonr_window(x: np.ndarray, y: np.ndarray) -> float:
    sx, sy = float(np.std(x)), float(np.std(y))
    if sx == 0 or sy == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rolling_pearson(d: np.ndarray, h: np.ndarray, W: int) -> np.ndarray:
    n = len(d)
    r = np.full(n, np.nan)
    for i in range(W - 1, n):
        r[i] = _pearsonr_window(d[i - W + 1:i + 1], h[i - W + 1:i + 1])
    return r


def adaptive_window(n_common: int) -> int:
    return min(20, max(5, n_common // 4))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_best_configs(raw_csv: Path) -> dict[tuple[str, str], dict]:
    best: dict[tuple[str, str], dict] = {}
    with open(raw_csv, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["instance_name"], row["experiment"])
            mc = int(row["min_clash"])
            if key not in best or mc < best[key]["_mc"]:
                best[key] = dict(row)
                best[key]["_mc"] = mc
    return best


def load_original_summary(csv_path: Path) -> dict[tuple[str, str], dict]:
    lookup: dict[tuple[str, str], dict] = {}
    if not csv_path.exists():
        return lookup
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["instance_name"], row["experiment"])
            lookup[key] = dict(row)
    return lookup


# ---------------------------------------------------------------------------
# Worker (runs in child process)
# ---------------------------------------------------------------------------

def _run_extended_job(args: tuple) -> dict:
    inst_name, min_slots, experiment, m, pattern, seed = args

    inst = load_instance(inst_name)
    d = copy.copy(inst)
    d.timeslots = min_slots

    if experiment == "dynamic_only":
        res = run_dynamic_only(d, m, EXTENDED_MAX_STEPS, ORDERING, seed,
                               track_history=True)
    else:
        cycles = scaled_cycles(pattern)
        res = run_hybrid_schedule(d, m, pattern, cycles, ORDERING, seed,
                                  track_history=True)

    history = res["history"] or []
    h_arr   = np.array(history, dtype=np.int32)

    zeros         = np.where(h_arr == 0)[0]
    first_zero    = int(zeros[0]) if zeros.size else -1
    min_val       = int(res["min_clash"])
    min_indices   = np.where(h_arr == min_val)[0]
    first_min_idx = int(min_indices[0]) if min_indices.size else -1

    return {
        "instance_name":    inst_name,
        "experiment":       experiment,
        "m":                m,
        "seed":             seed,
        "history":          history,
        "min_clash":        min_val,
        "final_clash":      int(res["final_clash"]),
        "reached_zero":     bool(res["reached_zero"]),
        "n_events":         len(history),
        "first_zero_event": first_zero,
        "first_min_event":  first_min_idx,
    }


# ---------------------------------------------------------------------------
# Job building
# ---------------------------------------------------------------------------

def build_extended_jobs(best: dict[tuple[str, str], dict]) -> list[tuple]:
    jobs = []
    for (inst_name, experiment), row in best.items():
        if inst_name not in PAPER_MIN_SLOTS:
            continue
        m        = int(row["mode_m"])
        seed     = int(row["seed"])
        pat_name = row.get("pattern_name", "")
        pat_raw  = row.get("pattern", "[]")
        try:
            pattern = ast.literal_eval(pat_raw) if pat_raw.strip() else []
        except (ValueError, SyntaxError):
            pattern = PATTERNS.get(pat_name, [100])
        jobs.append((inst_name, PAPER_MIN_SLOTS[inst_name], experiment, m, pattern, seed))
    return jobs


# ---------------------------------------------------------------------------
# Plot: per-instance clash dynamics
# ---------------------------------------------------------------------------

def plot_extended_dynamics(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_inst: dict[str, dict[str, dict]] = {}
    for r in results:
        by_inst.setdefault(r["instance_name"], {})[r["experiment"]] = r

    for inst_name in sorted(by_inst):
        methods  = by_inst[inst_name]
        fig, ax  = plt.subplots(figsize=(13, 4))
        all_vals: list[np.ndarray] = []

        for method, r in sorted(methods.items()):
            h     = np.array(r["history"], dtype=float)
            x     = np.arange(len(h))
            color = COLOR_DYN if method == "dynamic_only" else COLOR_HYB
            lname = "Dynamic-only" if method == "dynamic_only" else "Hybrid"
            all_vals.append(h)
            ax.plot(x, h, color=color, linewidth=0.7, alpha=0.85,
                    label=f"{lname}  (min={r['min_clash']}, final={r['final_clash']})")
            zeros = np.where(h == 0)[0]
            if zeros.size:
                ax.axvline(zeros[0], color=color, linestyle="--",
                           linewidth=0.8, alpha=0.5)
                ax.scatter([zeros[0]], [0], color=color, s=50, zorder=5,
                           marker="*", label=f"{lname} → zero at t={zeros[0]}")
            else:
                fm = r["first_min_event"]
                if fm >= 0:
                    ax.scatter([fm], [r["min_clash"]], color=color,
                               s=40, zorder=4, marker="v", alpha=0.7,
                               label=f"{lname} min at t={fm}")

        combined = np.concatenate(all_vals)
        nonzero  = combined[combined > 0]
        if nonzero.size and nonzero.max() / max(1, nonzero.min()) > 100:
            ax.set_yscale("log")
            ax.set_ylabel("Total clashes (log scale)")
        else:
            ax.set_ylabel("Total clashes")

        S = PAPER_MIN_SLOTS[inst_name]
        ax.set_xlabel("Event index  (dynamic step or Hopfield call)")
        ax.set_title(f"{inst_name}  —  extended clash dynamics  "
                     f"(S={S}, 10,000-event budget)")
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(True, alpha=0.22)
        fig.tight_layout()
        path = out_dir / f"{inst_name}_extended.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {inst_name} → {path.name}")


# ---------------------------------------------------------------------------
# Plot: two-panel rolling correlation
# ---------------------------------------------------------------------------

def plot_rolling(
    inst_name: str,
    dyn_full: np.ndarray,
    hyb_full: np.ndarray,
    out_dir: Path,
) -> dict:
    n     = min(len(dyn_full), len(hyb_full))
    d     = dyn_full[:n].astype(float)
    h     = hyb_full[:n].astype(float)
    W     = adaptive_window(n)
    roll_r = rolling_pearson(d, h, W)

    S       = PAPER_MIN_SLOTS[inst_name]
    x       = np.arange(n)
    x_valid = np.where(~np.isnan(roll_r))[0]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 6), sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.08},
    )

    # ── top panel: clash curves ────────────────────────────────────────────
    nzv    = np.concatenate([d[d > 0], h[h > 0]])
    use_log = nzv.size > 0 and nzv.max() / max(1, nzv.min()) > 100

    ax1.plot(x, d, color=COLOR_DYN, linewidth=0.7, alpha=0.9,
             label=f"Dynamic-only  (min={int(d.min())})")
    ax1.plot(x, h, color=COLOR_HYB, linewidth=0.7, alpha=0.9,
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
        f"{inst_name}  —  extended clash dynamics  "
        f"(S={S}, 10,000-event budget{trunc_note})",
        fontsize=10,
    )
    ax1.legend(fontsize=7.5, loc="upper right")
    ax1.grid(True, alpha=0.22)

    # ── bottom panel: rolling correlation ─────────────────────────────────
    ax2.plot(x, roll_r, color=COLOR_COR, linewidth=1.1,
             label=f"Rolling Pearson r  (W={W})", zorder=3)
    ax2.axhline(0,    color="black", linewidth=0.7, linestyle="--", alpha=0.5)
    ax2.axhline( 0.8, color="gray",  linewidth=0.5, linestyle=":",  alpha=0.4)
    ax2.axhline(-0.8, color="gray",  linewidth=0.5, linestyle=":",  alpha=0.4)
    ax2.fill_between(x, roll_r, 0, where=~np.isnan(roll_r),
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

    path = out_dir / f"{inst_name}_extended_rolling.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    mean_r_val  = float(np.nanmean(roll_r)) if x_valid.size else float("nan")
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

    mr_str = f"{mean_r_val:+.3f}" if not np.isnan(mean_r_val) else "   nan"
    print(f"  {inst_name:<12}  n={n:>6}  W={W:>2}  mean_r={mr_str}  "
          f"dyn_min={dyn_min:>5}  hyb_min={hyb_min:>5}  → {path.name}")

    return {
        "instance_name":       inst_name,
        "min_slots":           S,
        "common_length":       n,
        "window_W":            W,
        "pct_windows_defined": pct_defined,
        "mean_rolling_r":      round(mean_r_val, 4) if not np.isnan(mean_r_val) else "",
        "dynamic_min_clash":   dyn_min,
        "hybrid_min_clash":    hyb_min,
        "dynamic_final_clash": dyn_fin,
        "hybrid_final_clash":  hyb_fin,
        "interpretation":      interp,
    }


# ---------------------------------------------------------------------------
# Plot: overview bar chart
# ---------------------------------------------------------------------------

def plot_overview(rows: list[dict], out_dir: Path) -> None:
    names  = [r["instance_name"] for r in rows]
    mean_r = [float(r["mean_rolling_r"]) if r["mean_rolling_r"] != "" else 0.0
              for r in rows]
    colors = [
        "#2ca02c" if v > 0.4 else "#aec7e8" if v > 0 else "#d62728"
        for v in mean_r
    ]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(13, 4))
    bars = ax.bar(x, mean_r, color=colors, alpha=0.85)
    ax.axhline(0,   color="black", linewidth=0.7)
    ax.axhline(0.8, color="gray",  linewidth=0.5, linestyle=":", alpha=0.6,
               label="r = 0.8 reference")
    for bar, v in zip(bars, mean_r):
        ax.annotate(f"{v:+.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, v),
                    xytext=(0, 4 if v >= 0 else -12),
                    textcoords="offset points", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n(S={PAPER_MIN_SLOTS[n]})" for n in names],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylim(-1.1, 1.25)
    ax.set_ylabel("Mean rolling Pearson r")
    ax.set_title("Rolling correlation — extended run (10,000 events)\n"
                 "(green = correlated, red = diverging)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.22)
    fig.tight_layout()
    path = out_dir / "extended_rolling_overview.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Overview → {path.name}")


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def compute_extended_summary(
    results: list[dict],
    orig_lookup: dict[tuple[str, str], dict],
) -> list[dict]:
    rows = []
    for r in results:
        key      = (r["instance_name"], r["experiment"])
        orig     = orig_lookup.get(key, {})
        orig_min = _safe_int(orig.get("min_clash", ""), default=-1)
        improv   = (orig_min - r["min_clash"]) if orig_min >= 0 else ""
        rows.append({
            "instance_name":           r["instance_name"],
            "experiment":              r["experiment"],
            "n_events":                r["n_events"],
            "min_clash":               r["min_clash"],
            "final_clash":             r["final_clash"],
            "reached_zero":            r["reached_zero"],
            "first_zero_event":        r["first_zero_event"] if r["first_zero_event"] >= 0 else "",
            "first_min_event":         r["first_min_event"]  if r["first_min_event"]  >= 0 else "",
            "original_min_clash":      orig_min if orig_min >= 0 else "",
            "improvement_vs_original": improv,
        })
    return sorted(rows, key=lambda x: (x["instance_name"], x["experiment"]))


# ---------------------------------------------------------------------------
# Analysis text
# ---------------------------------------------------------------------------

def write_analysis(
    results: list[dict],
    orig_lookup: dict[tuple[str, str], dict],
    out_path: Path,
) -> None:
    lines: list[str] = []
    A = lines.append

    A("=" * 72)
    A("EXTENDED DYNAMICS ANALYSIS  (10,000-event budget)")
    A("=" * 72)
    A("")

    by_inst: dict[str, dict[str, dict]] = {}
    for r in results:
        by_inst.setdefault(r["instance_name"], {})[r["experiment"]] = r

    # ── 1. newly solved ───────────────────────────────────────────────────
    A("1.  INSTANCES THAT NEWLY ACHIEVED 0 CLASHES")
    A("-" * 50)
    newly_zero: list[tuple[str, str, int]] = []
    for r in results:
        key      = (r["instance_name"], r["experiment"])
        orig     = orig_lookup.get(key, {})
        was_zero = str(orig.get("reached_zero", "False")).strip().lower() in ("true", "1")
        if r["reached_zero"] and not was_zero:
            newly_zero.append((r["instance_name"], r["experiment"],
                               r["first_zero_event"]))
    if newly_zero:
        for inst, exp, t in sorted(newly_zero):
            A(f"  {inst:<12}  {exp:<14}  first zero at event {t:,}")
    else:
        A("  None — no new instances solved with extended budget.")
    A("")

    # ── 2. min-clash comparison table ────────────────────────────────────
    A("2.  MIN-CLASH:  original (~350–1,000 events)  vs  extended (10,000 events)")
    A("-" * 74)
    A(f"  {'Instance':<12}  {'Dyn orig':>8}  {'Dyn ext':>7}  {'Δ dyn':>6}  "
      f"{'Hyb orig':>8}  {'Hyb ext':>7}  {'Δ hyb':>6}  Winner")
    A("  " + "-" * 72)
    for inst_name in INSTANCE_ORDER:
        methods  = by_inst.get(inst_name, {})
        dyn_r    = methods.get("dynamic_only", {})
        hyb_r    = methods.get("hybrid", {})
        dyn_ext  = dyn_r.get("min_clash", "?")
        hyb_ext  = hyb_r.get("min_clash", "?")
        dyn_orig = _safe_int(orig_lookup.get((inst_name, "dynamic_only"), {}).get("min_clash", ""))
        hyb_orig = _safe_int(orig_lookup.get((inst_name, "hybrid"),       {}).get("min_clash", ""))
        dyn_delta = (dyn_orig - dyn_ext) if dyn_orig >= 0 and isinstance(dyn_ext, int) else "?"
        hyb_delta = (hyb_orig - hyb_ext) if hyb_orig >= 0 and isinstance(hyb_ext, int) else "?"
        if isinstance(dyn_ext, int) and isinstance(hyb_ext, int):
            if dyn_ext == 0 and hyb_ext == 0:
                winner = "tie (both 0)"
            elif dyn_ext < hyb_ext:
                winner = "dyn"
            elif hyb_ext < dyn_ext:
                winner = "hyb"
            else:
                winner = "tie"
        else:
            winner = "?"
        do_str = f"{dyn_orig:>8}" if dyn_orig >= 0 else f"{'?':>8}"
        ho_str = f"{hyb_orig:>8}" if hyb_orig >= 0 else f"{'?':>8}"
        A(f"  {inst_name:<12}  {do_str}  {str(dyn_ext):>7}  {str(dyn_delta):>6}  "
          f"{ho_str}  {str(hyb_ext):>7}  {str(hyb_delta):>6}  {winner}")
    A("")

    # ── 3. plateau detection ──────────────────────────────────────────────
    A("3.  PLATEAU DETECTION — when was min-clash first reached?")
    A("-" * 64)
    A(f"  {'Instance':<12}  {'Experiment':<14}  {'first_min_t':>11}  "
      f"{'n_events':>9}  {'% remaining':>12}")
    A("  " + "-" * 64)
    for r in sorted(results, key=lambda x: (x["instance_name"], x["experiment"])):
        fm = r["first_min_event"]
        n  = r["n_events"]
        if fm >= 0 and n > 1:
            pct_rem = round(100.0 * (n - 1 - fm) / (n - 1), 1)
        else:
            pct_rem = "?"
        fm_str = str(fm) if fm >= 0 else "?"
        A(f"  {r['instance_name']:<12}  {r['experiment']:<14}  {fm_str:>11}  "
          f"{n:>9}  {str(pct_rem):>11}%")
    A("")

    # ── 4. which method benefits more ────────────────────────────────────
    A("4.  WHICH METHOD BENEFITS MORE FROM 10,000 EVENTS?")
    A("-" * 55)
    dyn_imps: list[int] = []
    hyb_imps: list[int] = []
    for r in results:
        key      = (r["instance_name"], r["experiment"])
        orig     = orig_lookup.get(key, {})
        om       = _safe_int(orig.get("min_clash", ""), default=-1)
        if om > 0:
            imp = om - r["min_clash"]
            (dyn_imps if r["experiment"] == "dynamic_only" else hyb_imps).append(imp)
    if dyn_imps:
        A(f"  Dynamic-only: avg improvement = {np.mean(dyn_imps):+.1f} clashes  "
          f"(max = {max(dyn_imps):+d})")
        A(f"    per-instance Δ: {[int(v) for v in dyn_imps]}")
    if hyb_imps:
        A(f"  Hybrid:       avg improvement = {np.mean(hyb_imps):+.1f} clashes  "
          f"(max = {max(hyb_imps):+d})")
        A(f"    per-instance Δ: {[int(v) for v in hyb_imps]}")
    A("")

    # ── 5. hard instances / diminishing returns ───────────────────────────
    A("5.  CONVERGENCE ASSESSMENT")
    A("-" * 55)
    plateau_runs: list[tuple[str, str, int, int]] = []
    for r in results:
        fm = r["first_min_event"]
        n  = r["n_events"]
        if fm >= 0 and n > 1 and r["min_clash"] > 0:
            frac = fm / (n - 1)
            if frac < 0.05:
                plateau_runs.append((r["instance_name"], r["experiment"], fm, n - 1))
    if plateau_runs:
        A("  Runs that reached min-clash in the FIRST 5% of their budget")
        A("  (remaining 95%+ of events produced no improvement):")
        for inst, exp, fm, total in sorted(plateau_runs):
            A(f"    {inst:<12} ({exp}): min at t={fm:,}/{total:,}  "
              f"= {100*fm/total:.1f}% into run")
    else:
        A("  No extreme early-plateau behaviour detected.")
    A("")

    # ── 6. summary verdict ────────────────────────────────────────────────
    A("6.  VERDICT — DO LONGER RUNS MEANINGFULLY HELP?")
    A("-" * 50)
    n_inst          = sum(1 for i in INSTANCE_ORDER if i in by_inst)
    n_dyn_improved  = sum(1 for v in dyn_imps if v > 0)
    n_hyb_improved  = sum(1 for v in hyb_imps if v > 0)
    A(f"  Dynamic-only improved min-clash in {n_dyn_improved}/{n_inst} instances")
    A(f"  Hybrid       improved min-clash in {n_hyb_improved}/{n_inst} instances")
    A(f"  New zero-clash solutions:          {len(newly_zero)} run(s)")

    hard = [
        inst_name for inst_name in INSTANCE_ORDER
        if inst_name in by_inst
        and by_inst[inst_name].get("dynamic_only", {}).get("min_clash", 1) > 0
        and by_inst[inst_name].get("hybrid",       {}).get("min_clash", 1) > 0
    ]
    if hard:
        A(f"  Fundamentally hard (both methods > 0 even at 10,000 events):")
        for inst_name in hard:
            dyn_min = by_inst[inst_name].get("dynamic_only", {}).get("min_clash", "?")
            hyb_min = by_inst[inst_name].get("hybrid",       {}).get("min_clash", "?")
            A(f"    {inst_name:<12}  dyn_min={dyn_min}  hyb_min={hyb_min}")
    A("")

    txt = "\n".join(lines)
    print(txt)
    out_path.write_text(txt, encoding="utf-8")
    print(f"  Analysis → {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extended 10,000-event clash-dynamics experiment")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel worker processes (default 4)")
    args = parser.parse_args()

    out_root = ROOT / "outputs" / "min_timeslot_experiment"
    raw_csv  = out_root / "raw_results.csv"
    orig_csv = out_root / "clash_dynamics" / "dynamics_summary.csv"

    if not raw_csv.exists():
        print(f"ERROR: {raw_csv} not found.\n"
              "Run scripts/run_min_timeslot_experiment.py first.")
        sys.exit(1)

    ext_dir  = out_root / "extended_dynamics"
    hist_dir = ext_dir / "histories"
    plot_dir = ext_dir / "plots"
    roll_dir = ext_dir / "rolling_plots"
    for d in (ext_dir, hist_dir, plot_dir, roll_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("Loading best configs from raw_results.csv …")
    best        = load_best_configs(raw_csv)
    orig_lookup = load_original_summary(orig_csv)
    jobs        = build_extended_jobs(best)

    print(f"  {len(jobs)} extended jobs  |  workers: {args.workers}")
    print(f"  Dynamic-only budget: {EXTENDED_MAX_STEPS:,} steps")
    print(f"  Hybrid target:       ~{TARGET_EVENTS:,} events (cycles auto-scaled)\n")

    # print cycle preview
    print("  Hybrid cycle preview:")
    for j in sorted(jobs, key=lambda x: x[0]):
        inst_name, _, experiment, m, pattern, seed = j
        if experiment == "hybrid":
            nc  = scaled_cycles(pattern)
            epc = sum(pattern) + len(pattern)
            est = 1 + nc * epc
            print(f"    {inst_name:<12}  pattern={pattern}  "
                  f"cycles={nc}  est_events≈{est:,}")
    print()

    # ── run ────────────────────────────────────────────────────────────────
    results: list[dict] = []
    t0   = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_extended_job, j): j for j in jobs}
        for fut in as_completed(futures):
            try:
                r = fut.result()
                results.append(r)
                done += 1
                zero_tag = (f"  ← 0 at t={r['first_zero_event']:,}"
                            if r["reached_zero"] else "")
                print(f"  {done:>2}/{len(jobs)}: {r['instance_name']:<12} "
                      f"{r['experiment']:<14}  min={r['min_clash']:>5}  "
                      f"events={r['n_events']:>6}{zero_tag}")
            except Exception as exc:
                print(f"  ERROR: {exc}")

    elapsed = time.perf_counter() - t0
    print(f"\n{len(results)} runs completed in {elapsed:.0f}s")

    # ── save histories ─────────────────────────────────────────────────────
    print(f"\nSaving {len(results)} history arrays …")
    for r in results:
        fname = f"{r['instance_name']}_{r['experiment']}.npy"
        np.save(hist_dir / fname, np.array(r["history"], dtype=np.int32))
    print(f"  → {hist_dir.relative_to(ROOT)}")

    # ── dynamics plots ─────────────────────────────────────────────────────
    print("\nGenerating clash-vs-time plots …")
    plot_extended_dynamics(results, plot_dir)

    # ── rolling correlation plots ──────────────────────────────────────────
    print("\nGenerating rolling correlation plots …")
    by_inst: dict[str, dict[str, dict]] = {}
    for r in results:
        by_inst.setdefault(r["instance_name"], {})[r["experiment"]] = r

    print(f"  {'Instance':<12}  {'n':>6}  {'W':>2}  {'mean r':>8}  "
          f"{'dyn_min':>7}  {'hyb_min':>7}")
    print("  " + "-" * 60)

    rolling_rows: list[dict] = []
    for inst_name in INSTANCE_ORDER:
        methods = by_inst.get(inst_name, {})
        if "dynamic_only" not in methods or "hybrid" not in methods:
            continue
        dyn_h = np.array(methods["dynamic_only"]["history"], dtype=np.int32)
        hyb_h = np.array(methods["hybrid"]["history"],       dtype=np.int32)
        row   = plot_rolling(inst_name, dyn_h, hyb_h, roll_dir)
        rolling_rows.append(row)

    print()
    plot_overview(rolling_rows, roll_dir)

    # ── summary CSVs ───────────────────────────────────────────────────────
    print("\nSaving summary CSVs …")
    ext_rows = compute_extended_summary(results, orig_lookup)

    ext_csv  = ext_dir / "extended_summary.csv"
    EXT_COLS = [
        "instance_name", "experiment", "n_events",
        "min_clash", "final_clash", "reached_zero",
        "first_zero_event", "first_min_event",
        "original_min_clash", "improvement_vs_original",
    ]
    with open(ext_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EXT_COLS)
        w.writeheader()
        w.writerows(ext_rows)
    print(f"  {ext_csv.name}")

    roll_csv  = ext_dir / "rolling_summary.csv"
    ROLL_COLS = [
        "instance_name", "min_slots", "common_length", "window_W",
        "pct_windows_defined", "mean_rolling_r",
        "dynamic_min_clash", "hybrid_min_clash",
        "dynamic_final_clash", "hybrid_final_clash",
        "interpretation",
    ]
    with open(roll_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROLL_COLS)
        w.writeheader()
        w.writerows(rolling_rows)
    print(f"  {roll_csv.name}")

    # ── analysis ────────────────────────────────────────────────────────────
    print()
    write_analysis(results, orig_lookup, ext_dir / "analysis.txt")

    print(f"\nAll outputs → {ext_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
