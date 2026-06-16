from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from math import gcd
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from experiment_utils import (
    DATA_DIR,
    FIG_DIR,
    OUT_DIR,
    classify_mode,
    greedy_largest_degree,
    list_available_instances,
    load_instance,
    load_paper_baselines,
    load_paper_feasible,
    random_baseline,
    run_dynamic_only,
    run_hopfield_only,
    run_hybrid_schedule,
    serialize_pattern,
)


EXP_DIR = OUT_DIR / "experiments"
EXP_DIR.mkdir(parents=True, exist_ok=True)
HIST_DIR = EXP_DIR / "histories"
HIST_DIR.mkdir(parents=True, exist_ok=True)
RAW_CSV = EXP_DIR / "raw_results.csv"
ERROR_LOG = EXP_DIR / "errors.log"
CONFIG_JSON = EXP_DIR / "run_config.json"


QUICK_INSTANCES = ["hec-s-92", "sta-f-83", "ute-s-92", "ear-f-83", "yor-f-83", "lse-f-91"]
MEDIUM_INSTANCES = [
    "hec-s-92", "sta-f-83", "ute-s-92", "ear-f-83", "yor-f-83",
    "lse-f-91", "kfu-s-93", "tre-s-92", "rye-s-93",
    "car-f-92", "uta-s-92", "car-s-91",
]
FULL_INSTANCES = MEDIUM_INSTANCES + ["pur-s-93"]
LARGE_INSTANCES = {"car-f-92", "uta-s-92", "car-s-91", "pur-s-93"}
VERY_LARGE_INSTANCES = {"pur-s-93"}

DURATIONS_QUICK = [100, 500, 1000]
DURATIONS_MEDIUM = [500, 1000]
DURATIONS_FULL = [500, 1000, 3000]
DURATIONS_LARGE_MAX = 400
DURATIONS_VERY_LARGE_MAX = 200

PATTERNS = {
    "p_10_100": [10, 100],
    "p_25_100": [25, 100],
    "p_50_100": [50, 100],
    "p_10_50_100": [10, 50, 100],
    "p_10_25_50_100": [10, 25, 50, 100],
    "p_100": [100],
    "p_10": [10],
}

ORDERINGS_QUICK = ["natural", "largest-weighted-degree"]
ORDERINGS_MEDIUM = ["natural", "largest-weighted-degree"]
ORDERINGS_FULL = ["natural", "largest-weighted-degree", "saturation-like"]

SEEDS_QUICK = [0]
SEEDS_MEDIUM = [0, 1]
SEEDS_FULL = [0, 1, 2]


RAW_COLUMNS = [
    "run_id", "experiment", "instance_name", "paper_name",
    "n_exams", "n_students", "timeslots", "paper_max_clashes",
    "mode_m", "mode_type", "ordering", "pattern_name", "pattern",
    "duration", "cycles", "cumulative_steps",
    "init_clash", "final_clash", "min_clash", "max_clash", "mean_clash",
    "best_step", "reached_zero",
    "n_hopfield_calls",
    "hopfield_improved_calls", "hopfield_worsened_calls",
    "hopfield_unchanged_calls", "hopfield_rolledback_calls",
    "hopfield_total_delta",
    "wall_time_seconds", "seed",
    "history_file", "error",
]


@dataclass
class Job:
    job_kind: str
    instance: str
    mode_m: int = 0
    duration: int = 0
    pattern_name: str = ""
    pattern: tuple = ()
    cycles: int = 1
    ordering: str = "natural"
    seed: int = 0


def build_modes(timeslots: int, paper_feasible: dict, problem_name: str, profile: str) -> list[int]:
    S = timeslots
    candidates: set[int] = set()
    candidates.update({1, 2, S - 1, S + 1})
    for m in range(2, 2 * S):
        if gcd(m, S) == 1:
            candidates.add(m)
    for m in range(2, S):
        if S % m == 0:
            candidates.add(m)
    if profile == "full":
        candidates.update(range(1, S))
    if problem_name in paper_feasible:
        for mode in paper_feasible[problem_name]["modes"]:
            candidates.add(mode["m"])
    candidates = {m for m in candidates if 1 <= m < 2 * S}
    return sorted(candidates)


def hop_event_stats(events: list[dict]) -> dict:
    counts = {"improved": 0, "worsened": 0, "unchanged": 0, "rolled-back": 0, "noop": 0}
    total_delta = 0
    for e in events:
        counts[e.get("outcome", "unchanged")] = counts.get(e.get("outcome", "unchanged"), 0) + 1
        total_delta += e.get("delta", 0)
    return {
        "improved": counts.get("improved", 0),
        "worsened": counts.get("worsened", 0),
        "unchanged": counts.get("unchanged", 0) + counts.get("noop", 0),
        "rolledback": counts.get("rolled-back", 0),
        "total_delta": int(total_delta),
    }


def save_history(history: list[int] | None, run_id: str) -> str:
    if history is None or not history:
        return ""
    arr = np.array(history, dtype=np.int32)
    path = HIST_DIR / f"{run_id}.npy"
    np.save(path, arr)
    return path.name


def execute_job(job: Job) -> dict:
    run_id = uuid.uuid4().hex[:12]
    try:
        inst = load_instance(job.instance)
    except Exception:
        return {
            "run_id": run_id, "experiment": job.job_kind, "instance_name": job.instance,
            "error": traceback.format_exc(),
        }

    row = {c: "" for c in RAW_COLUMNS}
    row.update({
        "run_id": run_id,
        "experiment": job.job_kind,
        "instance_name": inst.name,
        "paper_name": inst.paper_name,
        "n_exams": inst.n_exams,
        "n_students": inst.n_students,
        "timeslots": inst.timeslots,
        "paper_max_clashes": inst.paper_max_clashes if inst.paper_max_clashes is not None else "",
        "mode_m": job.mode_m,
        "mode_type": classify_mode(job.mode_m, inst.timeslots) if job.mode_m else "",
        "ordering": job.ordering,
        "pattern_name": job.pattern_name,
        "pattern": serialize_pattern(list(job.pattern)) if job.pattern else "",
        "cycles": job.cycles,
        "duration": job.duration,
        "seed": job.seed,
    })

    try:
        if job.job_kind == "dynamic_only":
            res = run_dynamic_only(inst, job.mode_m, job.duration, job.ordering, job.seed, track_history=True)
            row["cumulative_steps"] = res["cumulative_steps"]
            row["init_clash"] = res["init_clash"]
            row["final_clash"] = res["final_clash"]
            row["min_clash"] = res["min_clash"]
            row["max_clash"] = res["max_clash"]
            row["mean_clash"] = res["mean_clash"]
            row["best_step"] = res["best_step"]
            row["reached_zero"] = bool(res["reached_zero"])
            row["wall_time_seconds"] = res["wall_time_s"]
            row["history_file"] = save_history(res.get("history"), run_id)
            row["n_hopfield_calls"] = 0
            row["hopfield_improved_calls"] = 0
            row["hopfield_worsened_calls"] = 0
            row["hopfield_unchanged_calls"] = 0
            row["hopfield_rolledback_calls"] = 0
            row["hopfield_total_delta"] = 0

        elif job.job_kind == "hopfield_only":
            res = run_hopfield_only(inst, job.seed, n_calls=max(1, job.duration))
            row["cumulative_steps"] = 0
            row["init_clash"] = res["history"][0] if res.get("history") else inst.initial_max_clashes
            row["final_clash"] = res["final_clash"]
            row["min_clash"] = res["min_clash"]
            row["max_clash"] = res["max_clash"]
            row["mean_clash"] = float(np.mean(res["history"])) if res.get("history") else 0.0
            row["best_step"] = res["best_step"]
            row["reached_zero"] = bool(res["reached_zero"])
            row["wall_time_seconds"] = res["wall_time_s"]
            row["history_file"] = save_history(res.get("history"), run_id)
            stats = hop_event_stats(res["hopfield_events"])
            row["n_hopfield_calls"] = res["n_hopfield_calls"]
            row["hopfield_improved_calls"] = stats["improved"]
            row["hopfield_worsened_calls"] = stats["worsened"]
            row["hopfield_unchanged_calls"] = stats["unchanged"]
            row["hopfield_rolledback_calls"] = stats["rolledback"]
            row["hopfield_total_delta"] = stats["total_delta"]

        elif job.job_kind == "hybrid":
            res = run_hybrid_schedule(inst, job.mode_m, list(job.pattern), job.cycles, job.ordering, job.seed, track_history=True, pretrain_hopfield=True)
            row["cumulative_steps"] = res["cumulative_steps"]
            row["init_clash"] = res["history"][0] if res.get("history") else inst.initial_max_clashes
            row["final_clash"] = res["final_clash"]
            row["min_clash"] = res["min_clash"]
            row["max_clash"] = res["max_clash"]
            row["mean_clash"] = float(np.mean(res["history"])) if res.get("history") else 0.0
            row["best_step"] = res["best_step"]
            row["reached_zero"] = bool(res["reached_zero"])
            row["wall_time_seconds"] = res["wall_time_s"]
            row["history_file"] = save_history(res.get("history"), run_id)
            stats = hop_event_stats(res["hopfield_events"])
            row["n_hopfield_calls"] = res["n_hopfield_calls"]
            row["hopfield_improved_calls"] = stats["improved"]
            row["hopfield_worsened_calls"] = stats["worsened"]
            row["hopfield_unchanged_calls"] = stats["unchanged"]
            row["hopfield_rolledback_calls"] = stats["rolledback"]
            row["hopfield_total_delta"] = stats["total_delta"]

        elif job.job_kind == "random":
            res = random_baseline(inst, job.seed, n_trials=max(1, job.duration))
            row["cumulative_steps"] = 0
            row["init_clash"] = inst.initial_max_clashes
            row["final_clash"] = res["min_clash"]
            row["min_clash"] = res["min_clash"]
            row["max_clash"] = res["max_clash"]
            row["mean_clash"] = res["mean_clash"]
            row["best_step"] = 0
            row["reached_zero"] = res["min_clash"] == 0
            row["wall_time_seconds"] = 0.0
            row["history_file"] = ""
            row["n_hopfield_calls"] = 0
            row["hopfield_improved_calls"] = 0
            row["hopfield_worsened_calls"] = 0
            row["hopfield_unchanged_calls"] = 0
            row["hopfield_rolledback_calls"] = 0
            row["hopfield_total_delta"] = 0

        elif job.job_kind == "greedy":
            res = greedy_largest_degree(inst)
            row["cumulative_steps"] = 0
            row["init_clash"] = inst.initial_max_clashes
            row["final_clash"] = res["min_clash"]
            row["min_clash"] = res["min_clash"]
            row["max_clash"] = res["min_clash"]
            row["mean_clash"] = res["min_clash"]
            row["best_step"] = 0
            row["reached_zero"] = res["min_clash"] == 0
            row["wall_time_seconds"] = 0.0
            row["history_file"] = ""
            row["n_hopfield_calls"] = 0
            row["hopfield_improved_calls"] = 0
            row["hopfield_worsened_calls"] = 0
            row["hopfield_unchanged_calls"] = 0
            row["hopfield_rolledback_calls"] = 0
            row["hopfield_total_delta"] = 0

        else:
            row["error"] = f"unknown job kind: {job.job_kind}"

    except Exception:
        row["error"] = traceback.format_exc()

    return row


def build_jobs(profile: str) -> list[Job]:
    paper_feasible = load_paper_feasible()
    jobs: list[Job] = []

    if profile == "quick":
        instances = QUICK_INSTANCES
        durations = DURATIONS_QUICK
        orderings = ORDERINGS_QUICK
        seeds = SEEDS_QUICK
        patterns = {k: PATTERNS[k] for k in ["p_10_100", "p_50_100", "p_100"]}
    elif profile == "medium":
        instances = MEDIUM_INSTANCES
        durations = DURATIONS_MEDIUM
        orderings = ORDERINGS_MEDIUM
        seeds = SEEDS_MEDIUM
        patterns = {k: PATTERNS[k] for k in ["p_10_100", "p_50_100", "p_100", "p_25_100"]}
    else:
        instances = FULL_INSTANCES
        durations = DURATIONS_FULL
        orderings = ORDERINGS_FULL
        seeds = SEEDS_FULL
        patterns = PATTERNS

    for inst_name in instances:
        try:
            inst = load_instance(inst_name)
        except Exception:
            continue
        modes = build_modes(inst.timeslots, paper_feasible, inst.paper_name, profile)

        for seed in seeds:
            jobs.append(Job(job_kind="random", instance=inst_name, duration=10, seed=seed))
            jobs.append(Job(job_kind="hopfield_only", instance=inst_name, duration=5, seed=seed))
        jobs.append(Job(job_kind="greedy", instance=inst_name, seed=0))

        for m in modes:
            for d in durations:
                d_use = d
                if inst_name in VERY_LARGE_INSTANCES:
                    d_use = min(d, DURATIONS_VERY_LARGE_MAX)
                elif inst_name in LARGE_INSTANCES:
                    d_use = min(d, DURATIONS_LARGE_MAX)
                for ordering in orderings:
                    for seed in seeds:
                        jobs.append(Job(
                            job_kind="dynamic_only",
                            instance=inst_name, mode_m=m, duration=d_use,
                            ordering=ordering, seed=seed,
                        ))

        if inst.paper_name in paper_feasible:
            for mode in paper_feasible[inst.paper_name]["modes"]:
                d_cap = min(mode["d"], 5000 if inst_name not in LARGE_INSTANCES else DURATIONS_LARGE_MAX)
                if d_cap < mode["d"]:
                    continue
                jobs.append(Job(
                    job_kind="dynamic_only",
                    instance=inst_name, mode_m=mode["m"], duration=mode["d"],
                    ordering="natural", seed=0,
                ))

        hybrid_modes = sorted({m for m in modes if classify_mode(m, inst.timeslots) in ("quasi-periodic", "mixed")})
        if not hybrid_modes:
            hybrid_modes = modes[:4]
        hybrid_modes = hybrid_modes[:5]
        for m in hybrid_modes:
            for pname, plist in patterns.items():
                cycles = 5 if inst_name not in LARGE_INSTANCES else 3
                for seed in seeds:
                    jobs.append(Job(
                        job_kind="hybrid",
                        instance=inst_name, mode_m=m,
                        pattern_name=pname, pattern=tuple(plist),
                        cycles=cycles, ordering="natural", seed=seed,
                    ))

    seen = set()
    deduped = []
    for j in jobs:
        key = (j.job_kind, j.instance, j.mode_m, j.duration, j.pattern_name, j.pattern, j.cycles, j.ordering, j.seed)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(j)
    return deduped


def append_row(row: dict) -> None:
    df = pd.DataFrame([{c: row.get(c, "") for c in RAW_COLUMNS}])
    header = not RAW_CSV.exists()
    df.to_csv(RAW_CSV, mode="a", header=header, index=False)


def write_config(profile: str, jobs: list[Job], parallel: int) -> None:
    cfg = {
        "profile": profile,
        "n_jobs": len(jobs),
        "parallel_workers": parallel,
        "patterns": PATTERNS,
        "quick_instances": QUICK_INSTANCES,
        "full_instances": FULL_INSTANCES,
        "large_instances": sorted(LARGE_INSTANCES),
        "durations_quick": DURATIONS_QUICK,
        "durations_full": DURATIONS_FULL,
        "durations_large_max": DURATIONS_LARGE_MAX,
        "orderings_quick": ORDERINGS_QUICK,
        "orderings_full": ORDERINGS_FULL,
        "seeds_quick": SEEDS_QUICK,
        "seeds_full": SEEDS_FULL,
    }
    CONFIG_JSON.write_text(json.dumps(cfg, indent=2))


def write_summaries(df: pd.DataFrame) -> None:
    df = df.copy()
    for col in ["min_clash", "final_clash", "init_clash", "max_clash",
                "cumulative_steps", "wall_time_seconds", "best_step",
                "n_hopfield_calls", "hopfield_total_delta", "paper_max_clashes"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["reached_zero"] = df["reached_zero"].astype(str).str.lower().isin(["true", "1"])

    by_instance = (
        df.groupby(["instance_name", "experiment"])
          .agg(min_clash=("min_clash", "min"),
               mean_min_clash=("min_clash", "mean"),
               reached_zero_rate=("reached_zero", "mean"),
               n_runs=("run_id", "count"),
               total_time=("wall_time_seconds", "sum"))
          .reset_index()
    )
    by_instance.to_csv(EXP_DIR / "summary_by_instance.csv", index=False)

    by_mode = (
        df[df["mode_m"] != ""]
          .groupby(["instance_name", "mode_m", "mode_type", "experiment"])
          .agg(min_clash=("min_clash", "min"),
               n_runs=("run_id", "count"),
               reached_zero=("reached_zero", "max"))
          .reset_index()
    )
    by_mode.to_csv(EXP_DIR / "summary_by_mode.csv", index=False)

    by_pattern = (
        df[df["pattern_name"] != ""]
          .groupby(["instance_name", "pattern_name"])
          .agg(min_clash=("min_clash", "min"),
               mean_min_clash=("min_clash", "mean"),
               reached_zero_rate=("reached_zero", "mean"),
               n_runs=("run_id", "count"),
               total_hopfield_calls=("n_hopfield_calls", "sum"))
          .reset_index()
    )
    by_pattern.to_csv(EXP_DIR / "summary_by_pattern.csv", index=False)

    idx = df.groupby(["instance_name", "experiment"])["min_clash"].idxmin()
    idx = idx.dropna()
    best = df.loc[idx].copy()
    best.to_csv(EXP_DIR / "best_results.csv", index=False)

    paper = load_paper_baselines()
    comp_rows = []
    for inst, group in df.groupby("instance_name"):
        try:
            inst_obj = load_instance(inst)
            paper_max = inst_obj.paper_max_clashes
            paper_name = inst_obj.paper_name
        except Exception:
            paper_max, paper_name = None, inst
        dyn = group[group["experiment"] == "dynamic_only"]
        hyb = group[group["experiment"] == "hybrid"]
        comp_rows.append({
            "instance_name": inst,
            "paper_name": paper_name,
            "paper_max_clashes": paper_max,
            "best_dynamic_only_min": int(dyn["min_clash"].min()) if len(dyn) else None,
            "best_hybrid_min": int(hyb["min_clash"].min()) if len(hyb) else None,
            "dynamic_zero_count": int((dyn["reached_zero"]).sum()) if len(dyn) else 0,
            "hybrid_zero_count": int((hyb["reached_zero"]).sum()) if len(hyb) else 0,
            "improvement_over_paper_max": (
                (paper_max - int(min(dyn["min_clash"].min() if len(dyn) else paper_max,
                                     hyb["min_clash"].min() if len(hyb) else paper_max))) / paper_max
                if paper_max else None
            ),
        })
    pd.DataFrame(comp_rows).to_csv(EXP_DIR / "comparison_with_paper.csv", index=False)

    try:
        df.to_parquet(EXP_DIR / "raw_results.parquet")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["quick", "medium", "full"], default="quick")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    if args.reset and RAW_CSV.exists():
        RAW_CSV.unlink()
    if args.reset and ERROR_LOG.exists():
        ERROR_LOG.unlink()

    jobs = build_jobs(args.profile)
    print(f"Profile: {args.profile} | Jobs: {len(jobs)} | Workers: {args.workers}")
    write_config(args.profile, jobs, args.workers)

    errors = 0
    if args.workers <= 1:
        for job in tqdm(jobs, desc="experiments"):
            row = execute_job(job)
            append_row(row)
            if row.get("error"):
                errors += 1
                with open(ERROR_LOG, "a") as f:
                    f.write(f"\n=== {job} ===\n{row['error']}\n")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(execute_job, j) for j in jobs]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="experiments"):
                try:
                    row = fut.result()
                except Exception:
                    row = {"run_id": uuid.uuid4().hex[:12], "experiment": "unknown",
                           "error": traceback.format_exc()}
                append_row(row)
                if row.get("error"):
                    errors += 1
                    with open(ERROR_LOG, "a") as f:
                        f.write(f"\n=== {row.get('run_id')} ===\n{row['error']}\n")

    df = pd.read_csv(RAW_CSV)
    write_summaries(df)
    print(f"\nDone. Errors: {errors}")
    print(f"Raw rows: {len(df)} -> {RAW_CSV}")
    print(f"Summaries in {EXP_DIR}")


if __name__ == "__main__":
    main()
