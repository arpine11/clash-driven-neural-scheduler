"""
Clash dynamics analysis for the paper-minimum timeslot experiment.

Reads raw_results.csv produced by run_min_timeslot_experiment.py, identifies
the best configuration (lowest min_clash) for each (instance, method) pair,
then re-runs those configurations with full clash-history tracking enabled.

Outputs saved to outputs/min_timeslot_experiment/clash_dynamics/
  histories/              .npy array per (instance, method)
  plots/                  per-instance clash-vs-time PNG
  correlation_plots/      per-instance scatter PNG
  correlation_overview.png   bar chart of r values across all instances
  correlation_summary.csv
  dynamics_summary.csv

This script is ADDITIVE — does not modify any existing files.

Usage:
    python3 scripts/run_clash_dynamics.py
    python3 scripts/run_clash_dynamics.py --workers 4
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
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from experiment_utils import (
    Instance,
    load_instance,
    run_dynamic_only,
    run_hybrid_schedule,
)

# ---------------------------------------------------------------------------
# Constants (mirror run_min_timeslot_experiment.py — do not change)
# ---------------------------------------------------------------------------

PAPER_MIN_SLOTS: dict[str, int] = {
    "hec-s-92": 17, "sta-f-83": 13, "ute-s-92": 10,
    "lse-f-91": 17, "yor-f-83": 19, "ear-f-83": 22,
    "kfu-s-93": 19, "tre-s-92": 20, "rye-s-93": 21,
    "car-f-92": 28, "uta-s-92": 30, "car-s-91": 28,
}

LARGE_INSTANCES = {"car-f-92", "car-s-91", "uta-s-92"}

PATTERNS = {
    "p_10_100": [10, 100],
    "p_50_100": [50, 100],
    "p_100":    [100],
}

ORDERING = "largest-weighted-degree"
CYCLES   = 3


def _duration(inst_name: str) -> int:
    return 400 if inst_name in LARGE_INSTANCES else 1000


# ---------------------------------------------------------------------------
# Statistics helpers (no scipy required)
# ---------------------------------------------------------------------------

def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks with tie handling (mirrors scipy.stats.rankdata)."""
    tmp = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[tmp] = np.arange(1, len(a) + 1, dtype=float)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[tmp[j]] == a[tmp[i]]:
            j += 1
        if j > i + 1:
            avg = float(i + 1 + j) / 2.0
            ranks[tmp[i:j]] = avg
        i = j
    return ranks


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x.astype(float), y.astype(float))[0, 1])


def _spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    return _pearsonr(_rankdata(x.astype(float)), _rankdata(y.astype(float)))


# ---------------------------------------------------------------------------
# Load best config per (instance, experiment) from raw_results.csv
# ---------------------------------------------------------------------------

def load_best_configs(raw_csv: Path) -> dict[tuple[str, str], dict]:
    """Return the CSV row with the lowest min_clash for each (instance, experiment)."""
    best: dict[tuple[str, str], dict] = {}
    with open(raw_csv, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["instance_name"], row["experiment"])
            mc = int(row["min_clash"])
            if key not in best or mc < best[key]["_mc"]:
                best[key] = dict(row)
                best[key]["_mc"] = mc
    return best


# ---------------------------------------------------------------------------
# Worker function executed in child processes
# ---------------------------------------------------------------------------

def _run_history_job(args: tuple) -> dict:
    inst_name, min_slots, experiment, m, pattern, cycles, seed = args

    inst = load_instance(inst_name)
    d = copy.copy(inst)
    d.timeslots = min_slots

    if experiment == "dynamic_only":
        res = run_dynamic_only(d, m, _duration(inst_name), ORDERING, seed,
                               track_history=True)
    else:
        res = run_hybrid_schedule(d, m, pattern, cycles, ORDERING, seed,
                                  track_history=True)

    return {
        "instance_name": inst_name,
        "experiment":    experiment,
        "m":             m,
        "seed":          seed,
        "history":       res["history"],
        "min_clash":     int(res["min_clash"]),
        "final_clash":   int(res["final_clash"]),
        "reached_zero":  bool(res["reached_zero"]),
        "n_steps":       len(res["history"]) - 1 if res["history"] else 0,
    }


# ---------------------------------------------------------------------------
# Build job list from best configs
# ---------------------------------------------------------------------------

def build_history_jobs(best: dict[tuple[str, str], dict]) -> list[tuple]:
    jobs = []
    for (inst_name, experiment), row in best.items():
        min_slots = PAPER_MIN_SLOTS[inst_name]
        m         = int(row["mode_m"])
        seed      = int(row["seed"])
        pat_name  = row.get("pattern_name", "")
        # parse stored pattern string e.g. "[10, 100]" → [10, 100]
        pat_raw   = row.get("pattern", "[]")
        try:
            pattern = ast.literal_eval(pat_raw) if pat_raw.strip() else []
        except (ValueError, SyntaxError):
            pattern = PATTERNS.get(pat_name, [100])
        cycles = int(row.get("cycles", CYCLES))
        jobs.append((inst_name, min_slots, experiment, m, pattern, cycles, seed))
    return jobs


# ---------------------------------------------------------------------------
# Save / load history arrays
# ---------------------------------------------------------------------------

def save_histories(results: list[dict], hist_dir: Path) -> None:
    hist_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        fname = f"{r['instance_name']}_{r['experiment']}.npy"
        np.save(hist_dir / fname, np.array(r["history"], dtype=np.int32))
        print(f"  Saved {fname}  (len={len(r['history'])})")


# ---------------------------------------------------------------------------
# Per-instance clash-vs-time plots
# ---------------------------------------------------------------------------

def plot_dynamics(results: list[dict], plot_dir: Path) -> None:
    """One PNG per instance showing both methods' clash trajectories."""
    plot_dir.mkdir(parents=True, exist_ok=True)

    by_inst: dict[str, dict[str, dict]] = {}
    for r in results:
        by_inst.setdefault(r["instance_name"], {})[r["experiment"]] = r

    for inst_name in sorted(by_inst):
        methods = by_inst[inst_name]
        fig, ax = plt.subplots(figsize=(11, 4))

        all_vals: list[np.ndarray] = []
        for method, r in methods.items():
            h = np.array(r["history"], dtype=float)
            all_vals.append(h)
            x = np.arange(len(h))
            color = "#4C72B0" if method == "dynamic_only" else "#DD8452"
            label = "Dynamic-only" if method == "dynamic_only" else "Hybrid"
            ax.plot(x, h, color=color, linewidth=0.9, alpha=0.85,
                    label=f"{label}  (min={r['min_clash']})")
            # mark first zero-clash event
            zeros = np.where(h == 0)[0]
            if zeros.size:
                ax.axvline(zeros[0], color=color, linestyle="--",
                           linewidth=0.8, alpha=0.55)
                ax.scatter([zeros[0]], [0], color=color, s=50,
                           zorder=5, marker="*", label=f"{label} → zero at t={zeros[0]}")

        # choose y scale
        combined = np.concatenate(all_vals)
        nonzero  = combined[combined > 0]
        if nonzero.size and nonzero.max() / max(1, nonzero.min()) > 100:
            ax.set_yscale("log")
            ylabel = "Total clashes (log scale)"
        else:
            ylabel = "Total clashes"

        ax.set_xlabel("Event index  (dynamic step or Hopfield call)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{inst_name}  —  clash dynamics  (S={PAPER_MIN_SLOTS[inst_name]}, paper-minimum)")
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        path = plot_dir / f"{inst_name}_dynamics.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Dynamics → {path.name}")


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def compute_correlations(results: list[dict]) -> list[dict]:
    """
    For each instance pair (dynamic_only, hybrid), truncate both histories to
    the shorter length and compute Pearson r and Spearman rho.
    Alignment strategy: truncate to min(len_dyn, len_hyb).
    """
    by_inst: dict[str, dict[str, dict]] = {}
    for r in results:
        by_inst.setdefault(r["instance_name"], {})[r["experiment"]] = r

    rows = []
    for inst_name in sorted(by_inst):
        methods = by_inst[inst_name]
        if "dynamic_only" not in methods or "hybrid" not in methods:
            continue

        dyn_r = methods["dynamic_only"]
        hyb_r = methods["hybrid"]
        dyn_h = np.array(dyn_r["history"], dtype=float)
        hyb_h = np.array(hyb_r["history"], dtype=float)

        n_common = min(len(dyn_h), len(hyb_h))
        d = dyn_h[:n_common]
        h = hyb_h[:n_common]

        pr = _pearsonr(d, h)
        sr = _spearmanr(d, h)

        dyn_min = dyn_r["min_clash"]
        hyb_min = hyb_r["min_clash"]
        if dyn_min == 0 and hyb_min == 0:
            winner = "tie (both 0)"
        elif dyn_min <= hyb_min:
            winner = "dynamic_only"
        else:
            winner = "hybrid"

        rows.append({
            "instance_name":      inst_name,
            "min_slots":          PAPER_MIN_SLOTS[inst_name],
            "n_dynamic_steps":    len(dyn_h) - 1,
            "n_hybrid_steps":     len(hyb_h) - 1,
            "aligned_length":     n_common,
            "pearson_r":          "" if (isinstance(pr, float) and pr != pr) else round(pr, 4),
            "spearman_r":         "" if (isinstance(sr, float) and sr != sr) else round(sr, 4),
            "dynamic_min_clash":  dyn_min,
            "hybrid_min_clash":   hyb_min,
            "dynamic_final_clash": dyn_r["final_clash"],
            "hybrid_final_clash":  hyb_r["final_clash"],
            "winner":             winner,
        })
    return rows


# ---------------------------------------------------------------------------
# Per-instance correlation scatter plots
# ---------------------------------------------------------------------------

def plot_correlations(results: list[dict], corr_rows: list[dict],
                      out_dir: Path) -> None:
    """Scatter of dynamic vs hybrid clashes (coloured by event index)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    by_inst: dict[str, dict[str, dict]] = {}
    for r in results:
        by_inst.setdefault(r["instance_name"], {})[r["experiment"]] = r

    for crow in corr_rows:
        inst_name = crow["instance_name"]
        methods   = by_inst.get(inst_name, {})
        if "dynamic_only" not in methods or "hybrid" not in methods:
            continue

        dyn_h = np.array(methods["dynamic_only"]["history"], dtype=float)
        hyb_h = np.array(methods["hybrid"]["history"], dtype=float)
        n = min(len(dyn_h), len(hyb_h))
        d, h = dyn_h[:n], hyb_h[:n]

        fig, ax = plt.subplots(figsize=(5, 5))
        sc = ax.scatter(d, h, c=np.arange(n), cmap="viridis",
                        s=6, alpha=0.55, linewidths=0)
        plt.colorbar(sc, ax=ax, label="Event index (time)")

        lo = min(d.min(), h.min())
        hi = max(d.max(), h.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.45, label="y = x")

        pr = crow["pearson_r"]
        sr = crow["spearman_r"]
        ax.set_xlabel("Dynamic-only clashes")
        ax.set_ylabel("Hybrid clashes")
        ax.set_title(f"{inst_name}\nPearson r = {pr},  Spearman ρ = {sr}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        path = out_dir / f"{inst_name}_correlation.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Correlation scatter → {path.name}")


# ---------------------------------------------------------------------------
# Overview bar chart of correlation coefficients
# ---------------------------------------------------------------------------

def plot_overview(corr_rows: list[dict], out_dir: Path) -> None:
    instances = [r["instance_name"] for r in corr_rows]
    pearson  = [float(r["pearson_r"])  if r["pearson_r"]  not in ("", None) else 0.0 for r in corr_rows]
    spearman = [float(r["spearman_r"]) if r["spearman_r"] not in ("", None) else 0.0 for r in corr_rows]

    x = np.arange(len(instances))
    w = 0.35
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.bar(x - w / 2, pearson,  w, label="Pearson r",   color="#4C72B0", alpha=0.85)
    ax.bar(x + w / 2, spearman, w, label="Spearman ρ",  color="#DD8452", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(instances, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Correlation coefficient")
    ax.set_title("Correlation between dynamic-only and hybrid clash trajectories\n"
                 "(paper-minimum timeslots, truncated to common length)")
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = out_dir / "correlation_overview.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Overview → {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel worker processes (default 4)")
    args = parser.parse_args()

    # ---- paths ----
    out_root  = ROOT / "outputs" / "min_timeslot_experiment"
    raw_csv   = out_root / "raw_results.csv"
    if not raw_csv.exists():
        print(f"ERROR: {raw_csv} not found.\n"
              "Run scripts/run_min_timeslot_experiment.py first.")
        sys.exit(1)

    dyn_dir      = out_root / "clash_dynamics"
    hist_dir     = dyn_dir / "histories"
    plot_dir     = dyn_dir / "plots"
    corr_dir     = dyn_dir / "correlation_plots"
    for d in (dyn_dir, hist_dir, plot_dir, corr_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ---- find best configs ----
    print("Loading best configs from raw_results.csv …")
    best = load_best_configs(raw_csv)
    print(f"  Found {len(best)} (instance, method) pairs")

    jobs = build_history_jobs(best)
    print(f"  History tracking jobs: {len(jobs)}  |  workers: {args.workers}")

    # ---- run history jobs ----
    results: list[dict] = []
    done = 0
    t0 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_history_job, j): j for j in jobs}
        for fut in as_completed(futures):
            try:
                r = fut.result()
                results.append(r)
                done += 1
                print(f"  {done:>2}/{len(jobs)}: {r['instance_name']:<12} "
                      f"{r['experiment']:<12}  min={r['min_clash']:>5}  "
                      f"steps={r['n_steps']:>5}  zero={r['reached_zero']}")
            except Exception as exc:
                print(f"  ERROR: {exc}")

    print(f"\nAll runs done in {time.perf_counter() - t0:.0f}s")

    # ---- save histories ----
    print("\nSaving history arrays …")
    save_histories(results, hist_dir)

    # ---- clash-vs-time plots ----
    print("\nGenerating clash-vs-time plots …")
    plot_dynamics(results, plot_dir)

    # ---- correlation analysis ----
    print("\nComputing correlations …")
    corr_rows = compute_correlations(results)

    # save correlation summary CSV
    corr_csv = dyn_dir / "correlation_summary.csv"
    CORR_COLS = [
        "instance_name", "min_slots",
        "n_dynamic_steps", "n_hybrid_steps", "aligned_length",
        "pearson_r", "spearman_r",
        "dynamic_min_clash", "hybrid_min_clash",
        "dynamic_final_clash", "hybrid_final_clash",
        "winner",
    ]
    with open(corr_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CORR_COLS)
        w.writeheader()
        w.writerows(corr_rows)
    print(f"  Correlation CSV → {corr_csv.name}")

    # save dynamics summary CSV
    dyn_csv = dyn_dir / "dynamics_summary.csv"
    DYN_COLS = [
        "instance_name", "experiment", "n_steps",
        "min_clash", "final_clash", "reached_zero", "m", "seed",
    ]
    dyn_rows = sorted(
        [{k: r.get(k, "") for k in DYN_COLS} for r in results],
        key=lambda x: (x["instance_name"], x["experiment"]),
    )
    with open(dyn_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DYN_COLS)
        w.writeheader()
        w.writerows(dyn_rows)
    print(f"  Dynamics CSV    → {dyn_csv.name}")

    # ---- correlation / scatter plots ----
    print("\nGenerating correlation scatter plots …")
    plot_correlations(results, corr_rows, corr_dir)
    plot_overview(corr_rows, dyn_dir)

    # ---- print summary table ----
    print()
    print(f"{'Instance':<12} {'S':>3}  {'Pearson r':>10}  {'Spearman ρ':>11}  "
          f"{'Dyn min':>8}  {'Hyb min':>8}  {'Winner'}")
    print("-" * 80)
    for r in corr_rows:
        print(f"{r['instance_name']:<12} {r['min_slots']:>3}  "
              f"{str(r['pearson_r']):>10}  {str(r['spearman_r']):>11}  "
              f"{r['dynamic_min_clash']:>8}  {r['hybrid_min_clash']:>8}  "
              f"{r['winner']}")

    print(f"\nAll outputs → {dyn_dir}")


if __name__ == "__main__":
    main()
