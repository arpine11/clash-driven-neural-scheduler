"""
Separate experiment: dynamic-only vs hybrid at paper-minimum timeslot counts.

Outputs go to outputs/min_timeslot_experiment/ — nothing in the existing
outputs/experiments/ tree is touched.

Usage:
    python scripts/run_min_timeslot_experiment.py
    python scripts/run_min_timeslot_experiment.py --workers 4
"""
from __future__ import annotations

import argparse
import copy
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import gcd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from experiment_utils import (
    load_instance,
    run_dynamic_only,
    run_hybrid_schedule,
    Instance,
)

# ---------------------------------------------------------------------------
# Paper-minimum timeslot counts (from the bi-PAT paper)
# ---------------------------------------------------------------------------
PAPER_MIN_SLOTS: dict[str, int] = {
    "hec-s-92": 17,
    "sta-f-83": 13,
    "ute-s-92": 10,
    "lse-f-91": 17,
    "yor-f-83": 19,
    "ear-f-83": 22,
    "kfu-s-93": 19,
    "tre-s-92": 20,
    "rye-s-93": 21,
    "car-f-92": 28,
    "uta-s-92": 30,   # paper name uta-s-93
    "car-s-91": 28,
}

# Standard timeslots used in existing experiments (for reference column)
STANDARD_SLOTS: dict[str, int] = {
    "hec-s-92": 18,
    "sta-f-83": 13,
    "ute-s-92": 10,
    "lse-f-91": 18,
    "yor-f-83": 21,
    "ear-f-83": 24,
    "kfu-s-93": 20,
    "tre-s-92": 23,
    "rye-s-93": 23,
    "car-f-92": 32,
    "uta-s-92": 35,
    "car-s-91": 35,
}

LARGE_INSTANCES = {"car-f-92", "car-s-91", "uta-s-92"}

PATTERNS = {
    "p_10_100": [10, 100],
    "p_50_100": [50, 100],
    "p_100":    [100],
}

SEEDS = [0, 1, 2]
CYCLES = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_modes(S: int, max_modes: int = 12) -> list[int]:
    """Return a representative set of modes for timeslot count S.

    Includes boundary quasi-periodic modes (S-1, S+1), small primes,
    and periodic-factor modes, capped at max_modes for runtime reasons.
    """
    # anchor modes: always include
    anchors = {1, S - 1, S + 1}
    # quasi-periodic: gcd(m, S) == 1
    quasi = [m for m in range(2, 2 * S) if gcd(m, S) == 1 and m not in anchors]
    # periodic factors
    factors = [m for m in range(2, S) if S % m == 0]

    # pick a spread of quasi-periodic modes (evenly spaced indices)
    slots_left = max_modes - len(anchors) - len(factors)
    if slots_left > 0 and quasi:
        step = max(1, len(quasi) // slots_left)
        selected_quasi = quasi[::step][:slots_left]
    else:
        selected_quasi = []

    candidates = anchors | set(selected_quasi) | set(factors)
    return sorted(m for m in candidates if 1 <= m < 2 * S)


def override_slots(inst: Instance, new_slots: int) -> Instance:
    """Return a shallow copy of inst with timeslots replaced."""
    d = copy.copy(inst)
    d.timeslots = new_slots
    return d


def duration_for(inst_name: str) -> int:
    return 400 if inst_name in LARGE_INSTANCES else 1000


# ---------------------------------------------------------------------------
# Single-job runner (called in worker processes)
# ---------------------------------------------------------------------------

def _run_job(args: tuple) -> dict:
    kind, inst_name, min_slots, m, duration, pattern_name, pattern, cycles, ordering, seed = args
    inst = load_instance(inst_name)
    inst = override_slots(inst, min_slots)

    t0 = time.perf_counter()
    if kind == "dynamic_only":
        res = run_dynamic_only(inst, m, duration, ordering, seed, track_history=False)
    else:
        res = run_hybrid_schedule(
            inst, m, pattern, cycles, ordering, seed, track_history=False
        )
    elapsed = time.perf_counter() - t0

    return {
        "instance_name": inst_name,
        "min_slots": min_slots,
        "standard_slots": STANDARD_SLOTS.get(inst_name, ""),
        "experiment": kind,
        "mode_m": m,
        "duration": duration,
        "pattern_name": pattern_name,
        "pattern": str(pattern),
        "cycles": cycles,
        "ordering": ordering,
        "seed": seed,
        "init_clash": res.get("init_clash", ""),
        "final_clash": res["final_clash"],
        "min_clash": res["min_clash"],
        "reached_zero": res["reached_zero"],
        "n_hopfield_calls": res.get("n_hopfield_calls", 0),
        "wall_time_s": round(elapsed, 4),
    }


# ---------------------------------------------------------------------------
# Build job list
# ---------------------------------------------------------------------------

def build_jobs() -> list[tuple]:
    jobs = []
    ordering = "largest-weighted-degree"

    for inst_name, min_slots in PAPER_MIN_SLOTS.items():
        dur = duration_for(inst_name)
        modes = build_modes(min_slots)

        # dynamic-only
        for m in modes:
            for seed in SEEDS:
                jobs.append((
                    "dynamic_only", inst_name, min_slots,
                    m, dur, "", [], 1, ordering, seed,
                ))

        # hybrid
        for pat_name, pat in PATTERNS.items():
            for m in modes:
                for seed in SEEDS:
                    jobs.append((
                        "hybrid", inst_name, min_slots,
                        m, dur, pat_name, pat, CYCLES, ordering, seed,
                    ))

    return jobs


# ---------------------------------------------------------------------------
# Aggregate results → summary
# ---------------------------------------------------------------------------

def summarise(rows: list[dict]) -> list[dict]:
    from collections import defaultdict
    best: dict[tuple, dict] = {}

    for r in rows:
        key = (r["instance_name"], r["experiment"])
        mc = int(r["min_clash"])
        if key not in best or mc < best[key]["best_min_clash"]:
            best[key] = {
                "instance_name": r["instance_name"],
                "experiment": r["experiment"],
                "min_slots": r["min_slots"],
                "standard_slots": r["standard_slots"],
                "best_min_clash": mc,
                "best_mode": r["mode_m"],
                "best_pattern": r["pattern_name"],
            }

    zero_counts: dict[tuple, int] = defaultdict(int)
    run_counts:  dict[tuple, int] = defaultdict(int)
    for r in rows:
        key = (r["instance_name"], r["experiment"])
        run_counts[key] += 1
        if r["reached_zero"] in (True, "True"):
            zero_counts[key] += 1

    out = []
    for key, b in best.items():
        b["zero_runs"] = zero_counts[key]
        b["total_runs"] = run_counts[key]
        b["zero_rate"] = round(zero_counts[key] / max(1, run_counts[key]), 4)
        out.append(b)

    out.sort(key=lambda x: (x["instance_name"], x["experiment"]))
    return out


def pivot_summary(summary: list[dict]) -> list[dict]:
    """One row per instance: dynamic vs hybrid side by side."""
    from collections import defaultdict
    by_inst: dict[str, dict] = defaultdict(dict)
    for row in summary:
        by_inst[row["instance_name"]][row["experiment"]] = row

    out = []
    for inst_name in sorted(by_inst):
        dyn = by_inst[inst_name].get("dynamic_only", {})
        hyb = by_inst[inst_name].get("hybrid", {})
        dyn_best = dyn.get("best_min_clash", "")
        hyb_best = hyb.get("best_min_clash", "")
        if isinstance(dyn_best, int) and isinstance(hyb_best, int):
            if dyn_best == 0 and hyb_best == 0:
                winner = "tie (both 0)"
            elif dyn_best <= hyb_best:
                winner = "dynamic_only"
            else:
                winner = "hybrid"
        else:
            winner = ""
        out.append({
            "instance_name": inst_name,
            "min_slots": dyn.get("min_slots", hyb.get("min_slots", "")),
            "standard_slots": dyn.get("standard_slots", hyb.get("standard_slots", "")),
            "dynamic_best_min_clash": dyn_best,
            "hybrid_best_min_clash": hyb_best,
            "dynamic_zero_rate": dyn.get("zero_rate", ""),
            "hybrid_zero_rate": hyb.get("zero_rate", ""),
            "winner": winner,
        })
    return out


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(pivot: list[dict], out_dir: Path) -> None:
    instances = [r["instance_name"] for r in pivot]
    dyn_vals  = [r["dynamic_best_min_clash"] if isinstance(r["dynamic_best_min_clash"], int) else 0 for r in pivot]
    hyb_vals  = [r["hybrid_best_min_clash"]  if isinstance(r["hybrid_best_min_clash"],  int) else 0 for r in pivot]

    x = np.arange(len(instances))
    w = 0.35

    fig, ax = plt.subplots(figsize=(13, 5))
    bars_d = ax.bar(x - w / 2, dyn_vals, w, label="Dynamic-only", color="#4C72B0", alpha=0.85)
    bars_h = ax.bar(x + w / 2, hyb_vals, w, label="Hybrid",        color="#DD8452", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(instances, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Best min clashes")
    ax.set_title("Dynamic-only vs Hybrid — paper-minimum timeslots")
    ax.legend()
    ax.set_ylim(bottom=0)

    # annotate bars with value (skip 0 to avoid clutter)
    for bar in list(bars_d) + list(bars_h):
        h = bar.get_height()
        if h > 0:
            ax.annotate(
                str(int(h)),
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=7,
            )

    # add min-slot labels on x-axis
    slots_label = {r["instance_name"]: r["min_slots"] for r in pivot}
    ax.set_xticklabels(
        [f"{n}\n(S={slots_label[n]})" for n in instances],
        rotation=30, ha="right", fontsize=8,
    )

    fig.tight_layout()
    path = out_dir / "dynamic_vs_hybrid_min_slots.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {path}")


def make_delta_plot(pivot: list[dict], out_dir: Path) -> None:
    """Improvement of hybrid over dynamic (negative = hybrid worse)."""
    instances = [r["instance_name"] for r in pivot]
    deltas = []
    for r in pivot:
        d = r["dynamic_best_min_clash"]
        h = r["hybrid_best_min_clash"]
        if isinstance(d, int) and isinstance(h, int):
            deltas.append(d - h)   # positive = hybrid better
        else:
            deltas.append(0)

    colors = ["#2ca02c" if v > 0 else "#d62728" if v < 0 else "#aaaaaa" for v in deltas]
    x = np.arange(len(instances))

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.bar(x, deltas, color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    slots_label = {r["instance_name"]: r["min_slots"] for r in pivot}
    ax.set_xticklabels(
        [f"{n}\n(S={slots_label[n]})" for n in instances],
        rotation=30, ha="right", fontsize=8,
    )
    ax.set_ylabel("dynamic_best − hybrid_best\n(green = hybrid better, red = dynamic better)")
    ax.set_title("Hybrid improvement over dynamic-only — paper-minimum timeslots")
    fig.tight_layout()
    path = out_dir / "hybrid_improvement_delta.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    out_dir = ROOT / "outputs" / "min_timeslot_experiment"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_csv   = out_dir / "raw_results.csv"
    summ_csv  = out_dir / "summary_by_experiment.csv"
    pivot_csv = out_dir / "comparison_dynamic_vs_hybrid.csv"

    jobs = build_jobs()
    print(f"Total jobs: {len(jobs)}  |  workers: {args.workers}")

    RAW_COLS = [
        "instance_name", "min_slots", "standard_slots",
        "experiment", "mode_m", "duration",
        "pattern_name", "pattern", "cycles",
        "ordering", "seed",
        "init_clash", "final_clash", "min_clash",
        "reached_zero", "n_hopfield_calls", "wall_time_s",
    ]

    rows: list[dict] = []
    done = 0
    t_start = time.perf_counter()

    with open(raw_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_COLS)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_job, j): j for j in jobs}
            for fut in as_completed(futures):
                try:
                    row = fut.result()
                    writer.writerow(row)
                    f.flush()
                    rows.append(row)
                    done += 1
                    if done % 50 == 0 or done == len(jobs):
                        elapsed = time.perf_counter() - t_start
                        print(f"  {done}/{len(jobs)} done  ({elapsed:.0f}s)")
                except Exception as exc:
                    print(f"  ERROR: {exc}")

    print(f"\nRaw results → {raw_csv}  ({len(rows)} rows)")

    # summary
    summary = summarise(rows)
    SUMM_COLS = [
        "instance_name", "experiment", "min_slots", "standard_slots",
        "best_min_clash", "best_mode", "best_pattern",
        "zero_runs", "total_runs", "zero_rate",
    ]
    with open(summ_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMM_COLS)
        w.writeheader()
        w.writerows(summary)
    print(f"Summary       → {summ_csv}")

    # pivot
    pivot = pivot_summary(summary)
    PIVOT_COLS = [
        "instance_name", "min_slots", "standard_slots",
        "dynamic_best_min_clash", "hybrid_best_min_clash",
        "dynamic_zero_rate", "hybrid_zero_rate", "winner",
    ]
    with open(pivot_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PIVOT_COLS)
        w.writeheader()
        w.writerows(pivot)
    print(f"Comparison    → {pivot_csv}")

    # print to terminal
    print()
    print(f"{'Instance':<12} {'S_min':>5} {'S_std':>5} {'Dyn best':>9} {'Hyb best':>9} {'Winner':<20}")
    print("-" * 65)
    for r in pivot:
        print(
            f"{r['instance_name']:<12} {r['min_slots']:>5} {r['standard_slots']:>5} "
            f"{str(r['dynamic_best_min_clash']):>9} {str(r['hybrid_best_min_clash']):>9} "
            f"{r['winner']:<20}"
        )

    # plots
    print()
    make_plot(pivot, out_dir)
    make_delta_plot(pivot, out_dir)

    # text summary
    n_dyn_wins  = sum(1 for r in pivot if r["winner"] == "dynamic_only")
    n_hyb_wins  = sum(1 for r in pivot if r["winner"] == "hybrid")
    n_ties      = sum(1 for r in pivot if "tie" in str(r["winner"]))
    print()
    print("=== Summary ===")
    print(f"  Dynamic-only wins : {n_dyn_wins}")
    print(f"  Hybrid wins       : {n_hyb_wins}")
    print(f"  Ties (both 0)     : {n_ties}")
    print(f"  Conclusion: {'Hybrid improves on dynamic-only for ' + str(n_hyb_wins) + ' instance(s).' if n_hyb_wins else 'Dynamic-only matches or beats hybrid on all instances.'}")


if __name__ == "__main__":
    main()
