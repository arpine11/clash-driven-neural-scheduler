from __future__ import annotations

import json
import math
import random
import sys
import time
import traceback
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
FIG_DIR = ROOT / "figs"
PAPER_DIR = OUT_DIR / "paper_baselines"

for d in (OUT_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)


FILE_TO_PAPER = {
    "hec-s-92": "hec-s-92",
    "sta-f-83": "sta-f-83",
    "ute-s-92": "ute-s-92",
    "lse-f-91": "lse-f-91",
    "yor-f-83": "yor-f-83",
    "ear-f-83": "ear-f-83",
    "kfu-s-93": "kfu-s-93",
    "tre-s-92": "tre-s-92",
    "rye-s-93": "rye-s-93",
    "car-f-92": "car-f-92",
    "uta-s-92": "uta-s-93",
    "car-s-91": "car-s-91",
    "pur-s-93": "pur-s-93",
}


@dataclass
class Instance:
    name: str
    paper_name: str
    n_exams: int
    n_students: int
    timeslots: int
    paper_max_clashes: int | None
    adj: list[np.ndarray]
    edges: np.ndarray
    degrees: np.ndarray
    weighted_degrees: np.ndarray
    edge_weights: np.ndarray
    initial_max_clashes: int


def _read_stu(stu_path: Path) -> list[list[int]]:
    out = []
    with open(stu_path) as f:
        for line in f:
            toks = line.strip().split()
            if not toks:
                continue
            row = []
            for t in toks:
                try:
                    row.append(int(t))
                except ValueError:
                    continue
            if row:
                out.append(row)
    return out


def _read_crs(crs_path: Path) -> dict[int, int]:
    out = {}
    with open(crs_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    out[int(parts[0])] = int(parts[1])
                except ValueError:
                    continue
    return out


def load_paper_baselines() -> dict:
    p = PAPER_DIR / "toronto_instances.json"
    if not p.exists():
        return {}
    items = json.loads(p.read_text())
    return {it["problem"]: it for it in items}


def load_paper_feasible() -> dict:
    p = PAPER_DIR / "paper_feasible_modes.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def list_available_instances() -> list[str]:
    if not DATA_DIR.exists():
        return []
    names = []
    for crs in sorted(DATA_DIR.glob("*.crs")):
        stu = crs.with_suffix(".stu")
        if stu.exists():
            names.append(crs.stem)
    return names


def load_instance(name: str) -> Instance:
    crs_path = DATA_DIR / f"{name}.crs"
    stu_path = DATA_DIR / f"{name}.stu"
    if not stu_path.exists():
        raise FileNotFoundError(f"missing {stu_path}")

    enrollments = _read_crs(crs_path) if crs_path.exists() else {}
    students = _read_stu(stu_path)

    max_id = max(enrollments.keys()) if enrollments else 0
    for row in students:
        for e in row:
            if e > max_id:
                max_id = e
    N = max_id

    pair_weights: dict[tuple[int, int], int] = {}
    for row in students:
        unique = sorted({e for e in row if 1 <= e <= N})
        for i_idx in range(len(unique)):
            for j_idx in range(i_idx + 1, len(unique)):
                a, b = unique[i_idx], unique[j_idx]
                pair_weights[(a, b)] = pair_weights.get((a, b), 0) + 1

    edge_a, edge_b, edge_w = [], [], []
    for (a, b), w in pair_weights.items():
        edge_a.append(a - 1)
        edge_b.append(b - 1)
        edge_w.append(w)

    edges = np.array(list(zip(edge_a, edge_b)), dtype=np.int32) if edge_a else np.empty((0, 2), dtype=np.int32)
    edge_weights = np.array(edge_w, dtype=np.int32) if edge_w else np.empty((0,), dtype=np.int32)

    adj_lists: list[list[int]] = [[] for _ in range(N)]
    wadj: list[list[int]] = [[] for _ in range(N)]
    for (a, b), w in pair_weights.items():
        ai, bi = a - 1, b - 1
        adj_lists[ai].append(bi)
        wadj[ai].append(w)
        adj_lists[bi].append(ai)
        wadj[bi].append(w)
    adj = [np.array(lst, dtype=np.int32) for lst in adj_lists]
    degrees = np.array([len(a) for a in adj], dtype=np.int32)
    weighted_degrees = np.array([sum(ws) for ws in wadj], dtype=np.int32)

    paper = load_paper_baselines()
    paper_name = FILE_TO_PAPER.get(name, name)
    paper_row = paper.get(paper_name)
    timeslots = paper_row["timeslots"] if paper_row else max(1, N // 10)
    paper_max = paper_row["max_clashes"] if paper_row else None
    n_students = paper_row["students"] if paper_row else len(students)

    inst = Instance(
        name=name,
        paper_name=paper_name,
        n_exams=N,
        n_students=n_students,
        timeslots=timeslots,
        paper_max_clashes=paper_max,
        adj=adj,
        edges=edges,
        degrees=degrees,
        weighted_degrees=weighted_degrees,
        edge_weights=edge_weights,
        initial_max_clashes=int(degrees.sum()),
    )
    return inst


def total_clashes(slot: np.ndarray, edges: np.ndarray) -> int:
    if edges.shape[0] == 0:
        return 0
    return int(2 * (slot[edges[:, 0]] == slot[edges[:, 1]]).sum())


def per_exam_clashes(slot: np.ndarray, adj: list[np.ndarray]) -> np.ndarray:
    out = np.zeros(len(adj), dtype=np.int32)
    for k in range(len(adj)):
        nbrs = adj[k]
        if nbrs.size:
            out[k] = int((slot[nbrs] == slot[k]).sum())
    return out


def in_clash_mask(slot: np.ndarray, adj: list[np.ndarray]) -> np.ndarray:
    out = np.zeros(len(adj), dtype=bool)
    for k in range(len(adj)):
        nbrs = adj[k]
        if nbrs.size and (slot[nbrs] == slot[k]).any():
            out[k] = True
    return out


def classify_mode(m: int, S: int) -> str:
    if m == 0:
        return "trivial"
    if m % S == 0:
        return "identity"
    g = gcd(m % S if (m % S) else S, S)
    if g == 1:
        if m == 2:
            return "mixed"
        return "quasi-periodic"
    if S % m == 0:
        return "periodic-factor"
    return "periodic-resonant"


def get_ordering(name: str, instance: Instance, rng: random.Random) -> np.ndarray:
    N = instance.n_exams
    if name == "natural":
        return np.arange(N, dtype=np.int32)
    if name == "random":
        idx = list(range(N))
        rng.shuffle(idx)
        return np.array(idx, dtype=np.int32)
    if name == "largest-degree":
        return np.argsort(-instance.degrees).astype(np.int32)
    if name == "largest-weighted-degree":
        return np.argsort(-instance.weighted_degrees).astype(np.int32)
    if name == "saturation-like":
        return np.argsort(-(instance.degrees * 2 + instance.weighted_degrees)).astype(np.int32)
    return np.arange(N, dtype=np.int32)


def apply_E_step(slot: np.ndarray, adj: list[np.ndarray], m: int, S: int, order: np.ndarray) -> None:
    for k in order:
        k = int(k)
        nbrs = adj[k]
        if nbrs.size == 0:
            continue
        for _ in range(m):
            if not (slot[nbrs] == slot[k]).any():
                break
            slot[k] = (slot[k] + 1) % S


def run_dynamic_segment(
    slot: np.ndarray,
    instance: Instance,
    m: int,
    steps: int,
    ordering: str,
    rng: random.Random,
    track_history: bool = True,
) -> dict:
    S = instance.timeslots
    history = [] if track_history else None
    min_clash = total_clashes(slot, instance.edges)
    init_clash = min_clash
    best_step = 0
    sum_clash = 0
    max_clash = min_clash
    if history is not None:
        history.append(min_clash)
    t0 = time.perf_counter()
    last_step = 0
    for step in range(1, steps + 1):
        last_step = step
        order = get_ordering(ordering, instance, rng)
        apply_E_step(slot, instance.adj, m, S, order)
        c = total_clashes(slot, instance.edges)
        sum_clash += c
        if c < min_clash:
            min_clash = c
            best_step = step
        if c > max_clash:
            max_clash = c
        if history is not None:
            history.append(c)
        if c == 0:
            break
    elapsed = time.perf_counter() - t0
    return {
        "init_clash": init_clash,
        "final_clash": int(total_clashes(slot, instance.edges)),
        "min_clash": int(min_clash),
        "max_clash": int(max_clash),
        "mean_clash": float(sum_clash / max(1, last_step)),
        "best_step": int(best_step),
        "steps_run": int(last_step),
        "history": history,
        "wall_time_s": float(elapsed),
    }


class HopfieldRepair:
    def __init__(self, instance: Instance, capacity_factor: float = 0.14):
        self.inst = instance
        N = instance.n_exams
        self.weights = np.zeros((N, N), dtype=np.float32)
        self.capacity = max(1, int(capacity_factor * N))
        self.trained = 0

    def train_from_slot_pattern(self, slot: np.ndarray) -> int:
        S = self.inst.timeslots
        trained_now = 0
        for s in range(S):
            if self.trained >= self.capacity:
                break
            p = np.where(slot == s, 1.0, -1.0).astype(np.float32)
            outer = np.outer(p, p)
            np.fill_diagonal(outer, 0.0)
            self.weights += outer
            self.trained += 1
            trained_now += 1
        return trained_now

    def recall_and_repair(self, slot: np.ndarray, rng: random.Random) -> dict:
        S = self.inst.timeslots
        N = self.inst.n_exams
        before = total_clashes(slot, self.inst.edges)

        per_clash = per_exam_clashes(slot, self.inst.adj)
        clashing = np.where(per_clash > 0)[0]
        if clashing.size == 0:
            return {"before": before, "after": before, "delta": 0, "moved": 0, "outcome": "noop"}

        net_by_slot = np.zeros((N, S), dtype=np.float32)
        for s in range(S):
            p = np.where(slot == s, 1.0, -1.0).astype(np.float32)
            net_by_slot[:, s] = self.weights @ p

        order = list(clashing)
        rng.shuffle(order)
        moved = 0
        for k in order:
            nbrs = self.inst.adj[k]
            candidate_slots = np.argsort(-net_by_slot[k])
            current_slot = int(slot[k])
            cur_clash = int((slot[nbrs] == current_slot).sum()) if nbrs.size else 0
            best_slot = current_slot
            best_clash = cur_clash
            best_score = net_by_slot[k, current_slot]
            for s in candidate_slots[:max(3, S // 4)]:
                s_int = int(s)
                c = int((slot[nbrs] == s_int).sum()) if nbrs.size else 0
                if c < best_clash or (c == best_clash and net_by_slot[k, s_int] > best_score):
                    best_clash = c
                    best_slot = s_int
                    best_score = net_by_slot[k, s_int]
            if best_slot != current_slot:
                slot[k] = best_slot
                moved += 1

        after = total_clashes(slot, self.inst.edges)
        delta = after - before
        if delta < 0:
            outcome = "improved"
        elif delta > 0:
            outcome = "worsened"
        else:
            outcome = "unchanged"
        return {"before": int(before), "after": int(after), "delta": int(delta), "moved": int(moved), "outcome": outcome}


def hopfield_pretrain(
    instance: Instance,
    rng: random.Random,
    n_warmup_modes: int = 3,
    warmup_steps: int = 80,
    ordering: str = "natural",
) -> HopfieldRepair:
    h = HopfieldRepair(instance)
    S = instance.timeslots
    candidate_ms = [m for m in [S - 1, S - 3, S + 1, S - 5, 7, 11, 13] if 2 <= m <= 2 * S]
    used = []
    for m in candidate_ms:
        if len(used) >= n_warmup_modes:
            break
        slot = np.zeros(instance.n_exams, dtype=np.int32)
        run_dynamic_segment(slot, instance, m, warmup_steps, ordering, rng, track_history=False)
        h.train_from_slot_pattern(slot)
        used.append(m)
    return h


def run_hybrid_schedule(
    instance: Instance,
    m: int,
    pattern: list[int],
    cycles: int,
    ordering: str,
    seed: int,
    track_history: bool = True,
    pretrain_hopfield: bool = True,
) -> dict:
    rng = random.Random(seed)
    np.random.seed(seed)

    slot = np.zeros(instance.n_exams, dtype=np.int32)
    histories: list[int] = [total_clashes(slot, instance.edges)]
    hopfield_events: list[dict] = []
    segment_records: list[dict] = []

    if pretrain_hopfield:
        h = hopfield_pretrain(instance, rng)
    else:
        h = HopfieldRepair(instance)

    cumulative_steps = 0
    min_clash_so_far = histories[0]
    max_clash_so_far = histories[0]
    best_step_so_far = 0
    seg_idx = 0
    call_idx = 0
    t0 = time.perf_counter()

    for cycle in range(cycles):
        for seg_len in pattern:
            before_seg = total_clashes(slot, instance.edges)
            res = run_dynamic_segment(slot, instance, m, seg_len, ordering, rng, track_history=track_history)
            if track_history and res["history"] is not None:
                histories.extend(res["history"][1:])
            cumulative_steps += res["steps_run"]
            seg_record = {
                "segment_index": seg_idx,
                "cycle": cycle,
                "segment_length": seg_len,
                "before": before_seg,
                "after": res["final_clash"],
                "min": res["min_clash"],
                "max": res["max_clash"],
                "mean": res["mean_clash"],
                "best_step_in_segment": res["best_step"],
                "wall_time": res["wall_time_s"],
                "cumulative_steps": cumulative_steps,
            }
            segment_records.append(seg_record)
            if res["min_clash"] < min_clash_so_far:
                min_clash_so_far = res["min_clash"]
                best_step_so_far = cumulative_steps - (res["steps_run"] - res["best_step"])
            if res["max_clash"] > max_clash_so_far:
                max_clash_so_far = res["max_clash"]
            seg_idx += 1

            if res["final_clash"] == 0:
                break

            hop_t0 = time.perf_counter()
            saved = slot.copy()
            saved_clash = res["final_clash"]
            hop = h.recall_and_repair(slot, rng)
            h.train_from_slot_pattern(slot)
            c_after = total_clashes(slot, instance.edges)
            if c_after > saved_clash:
                slot[:] = saved
                c_after = saved_clash
                hop["after"] = saved_clash
                hop["delta"] = 0
                hop["outcome"] = "rolled-back"
            hop["call_index"] = call_idx
            hop["cumulative_steps_before"] = cumulative_steps
            hop["wall_time"] = time.perf_counter() - hop_t0
            hopfield_events.append(hop)
            call_idx += 1
            if track_history:
                histories.append(c_after)
            if c_after < min_clash_so_far:
                min_clash_so_far = c_after
                best_step_so_far = cumulative_steps

        if total_clashes(slot, instance.edges) == 0:
            break

    elapsed = time.perf_counter() - t0
    final_clash = total_clashes(slot, instance.edges)
    return {
        "history": histories if track_history else None,
        "hopfield_events": hopfield_events,
        "segments": segment_records,
        "cumulative_steps": cumulative_steps,
        "final_clash": int(final_clash),
        "min_clash": int(min_clash_so_far),
        "max_clash": int(max_clash_so_far),
        "best_step": int(best_step_so_far),
        "reached_zero": final_clash == 0,
        "wall_time_s": float(elapsed),
        "n_hopfield_calls": len(hopfield_events),
    }


def run_dynamic_only(
    instance: Instance,
    m: int,
    duration: int,
    ordering: str,
    seed: int,
    track_history: bool = True,
) -> dict:
    rng = random.Random(seed)
    np.random.seed(seed)
    slot = np.zeros(instance.n_exams, dtype=np.int32)
    res = run_dynamic_segment(slot, instance, m, duration, ordering, rng, track_history=track_history)
    res["reached_zero"] = res["final_clash"] == 0
    res["cumulative_steps"] = res["steps_run"]
    res["n_hopfield_calls"] = 0
    res["hopfield_events"] = []
    res["segments"] = []
    return res


def run_hopfield_only(
    instance: Instance,
    seed: int,
    n_calls: int = 5,
) -> dict:
    rng = random.Random(seed)
    np.random.seed(seed)
    slot = np.zeros(instance.n_exams, dtype=np.int32)
    h = hopfield_pretrain(instance, rng)
    histories = [total_clashes(slot, instance.edges)]
    events = []
    t0 = time.perf_counter()
    for i in range(n_calls):
        ev = h.recall_and_repair(slot, rng)
        events.append(ev)
        histories.append(total_clashes(slot, instance.edges))
    elapsed = time.perf_counter() - t0
    final = total_clashes(slot, instance.edges)
    return {
        "history": histories,
        "hopfield_events": events,
        "segments": [],
        "cumulative_steps": 0,
        "final_clash": int(final),
        "min_clash": int(min(histories)),
        "max_clash": int(max(histories)),
        "best_step": int(np.argmin(histories)),
        "reached_zero": final == 0,
        "wall_time_s": float(elapsed),
        "n_hopfield_calls": n_calls,
    }


def random_baseline(instance: Instance, seed: int, n_trials: int = 20) -> dict:
    rng = np.random.default_rng(seed)
    best = None
    bests = []
    for _ in range(n_trials):
        slot = rng.integers(0, instance.timeslots, size=instance.n_exams, dtype=np.int32)
        c = total_clashes(slot, instance.edges)
        bests.append(c)
        if best is None or c < best:
            best = c
    return {"min_clash": int(best), "mean_clash": float(np.mean(bests)), "max_clash": int(max(bests))}


def greedy_largest_degree(instance: Instance) -> dict:
    N = instance.n_exams
    S = instance.timeslots
    order = np.argsort(-instance.degrees)
    slot = -np.ones(N, dtype=np.int32)
    for k in order:
        k = int(k)
        nbr_slots = slot[instance.adj[k]]
        counts = np.zeros(S, dtype=np.int32)
        for ns in nbr_slots:
            if ns >= 0:
                counts[ns] += 1
        best_s = int(np.argmin(counts))
        slot[k] = best_s
    final = total_clashes(slot.astype(np.int32), instance.edges)
    return {"min_clash": int(final), "final_clash": int(final), "slot": slot.tolist()}


def serialize_pattern(pattern: list[int]) -> str:
    return "[" + ",".join(str(x) for x in pattern) + "]"


def deserialize_pattern(s: str) -> list[int]:
    s = s.strip().lstrip("[").rstrip("]")
    if not s:
        return []
    return [int(x) for x in s.split(",")]


def safe_run(func, *args, **kwargs):
    try:
        return func(*args, **kwargs), None
    except Exception:
        tb = traceback.format_exc()
        return None, tb
