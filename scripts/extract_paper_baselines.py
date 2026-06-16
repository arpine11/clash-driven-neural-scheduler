from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "paper_baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)


TORONTO = [
    {"rank": 1, "problem": "hec-s-92", "exams": 81, "students": 2823, "clash_density_pct": 20, "max_clashes": 2726, "timeslots": 18},
    {"rank": 2, "problem": "sta-f-83", "exams": 139, "students": 611, "clash_density_pct": 14, "max_clashes": 2762, "timeslots": 13},
    {"rank": 3, "problem": "ute-s-92", "exams": 184, "students": 2749, "clash_density_pct": 8, "max_clashes": 2860, "timeslots": 10},
    {"rank": 4, "problem": "lse-f-91", "exams": 381, "students": 2726, "clash_density_pct": 6, "max_clashes": 9062, "timeslots": 18},
    {"rank": 5, "problem": "yor-f-83", "exams": 181, "students": 941, "clash_density_pct": 27, "max_clashes": 9412, "timeslots": 21},
    {"rank": 6, "problem": "ear-f-83", "exams": 190, "students": 1125, "clash_density_pct": 29, "max_clashes": 9586, "timeslots": 24},
    {"rank": 7, "problem": "kfu-s-93", "exams": 461, "students": 5349, "clash_density_pct": 6, "max_clashes": 11786, "timeslots": 20},
    {"rank": 8, "problem": "tre-s-92", "exams": 261, "students": 4360, "clash_density_pct": 18, "max_clashes": 12262, "timeslots": 23},
    {"rank": 9, "problem": "rye-s-93", "exams": 486, "students": 11483, "clash_density_pct": 7, "max_clashes": 17744, "timeslots": 23},
    {"rank": 10, "problem": "car-f-92", "exams": 543, "students": 18419, "clash_density_pct": 14, "max_clashes": 40610, "timeslots": 32},
    {"rank": 11, "problem": "uta-s-93", "exams": 622, "students": 21266, "clash_density_pct": 13, "max_clashes": 48498, "timeslots": 35},
    {"rank": 12, "problem": "car-s-91", "exams": 682, "students": 16925, "clash_density_pct": 13, "max_clashes": 59628, "timeslots": 35},
    {"rank": 13, "problem": "pur-s-93", "exams": 2419, "students": 30029, "clash_density_pct": 3, "max_clashes": 164598, "timeslots": 42},
]


FILE_ALIAS = {
    "uta-s-92": "uta-s-93",
}


FEASIBLE_MODES = {
    "hec-s-92": {"timeslots": 18, "modes": [{"m": 11, "d": 96}, {"m": 17, "d": 100}]},
    "sta-f-83": {"timeslots": 13, "modes": [{"m": m, "d": 12} for m in range(1, 15)], "note": "all modes 1<=m<=14, d<=12"},
    "ute-s-92": {"timeslots": 10, "modes": [{"m": 7, "d": 22}, {"m": 9, "d": 30}, {"m": 11, "d": 90}]},
    "lse-f-91": {"timeslots": 18, "modes": [{"m": 17, "d": 40010}]},
    "yor-f-83": {"timeslots": 21, "modes": [{"m": 19, "d": 9223}, {"m": 20, "d": 13502}]},
    "ear-f-83": {"timeslots": 24, "modes": [
        {"m": 11, "d": 74002}, {"m": 13, "d": 3615}, {"m": 17, "d": 3106},
        {"m": 19, "d": 2915}, {"m": 23, "d": 2606}, {"m": 25, "d": 889},
    ]},
    "kfu-s-93": {"timeslots": 20, "modes": [{"m": 17, "d": 4658}, {"m": 19, "d": 58134}, {"m": 21, "d": 15774}]},
    "tre-s-92": {"timeslots": 23, "modes": [
        {"m": 17, "d": 85639}, {"m": 18, "d": 88369}, {"m": 19, "d": 10570},
        {"m": 20, "d": 22989}, {"m": 21, "d": 42432}, {"m": 22, "d": 77780}, {"m": 24, "d": 19729},
    ]},
    "rye-s-93": {"timeslots": 23, "modes": [
        {"m": 12, "d": 65518}, {"m": 13, "d": 8585}, {"m": 14, "d": 5066}, {"m": 15, "d": 10274},
        {"m": 16, "d": 836}, {"m": 17, "d": 1732}, {"m": 18, "d": 1799}, {"m": 19, "d": 9705},
        {"m": 20, "d": 3591}, {"m": 21, "d": 1007}, {"m": 22, "d": 797}, {"m": 24, "d": 10512},
    ]},
}


PAPER_NOTES = {
    "model": "Uncapacitated examination timetabling. Exams = interacting particles cyclically moving on S discrete timeslots.",
    "rule": "F_k(t_k) = (t_k + b) mod S where b = 1 if exam k clashes at t_k, else 0.",
    "generalized": "F_k^m(t_k) = (t_k + c) mod S, 0 <= c <= m. c = number of successive clashing slots starting from t_k.",
    "cumulative": "E(m,d) shifts each of N exams up to m mod S slots forward subject to clashes, repeated d cycles.",
    "initial_state": "All exams in first timeslot (slot 0) -> maximal clashes.",
    "mode_types": {
        "periodic": "Steady-state dominates; m is a factor of S => strictly periodic.",
        "quasi_periodic": "m and S coprime => quasi-periodic (except m=2 may be steady-state).",
        "mixed": "transient and steady-state phases comparable.",
    },
    "amplitudes": "A_periodic << C0_periodic; quasi-periodic amplitudes grow with d.",
    "bi_PAT_modal": "Try E(m,d) with m coprime with S, 2<m<2S; double d until feasible.",
    "bi_PAT_cumulative": "Apply E(1,d), then E(m,d) sequentially using min-clash state of previous as next initial.",
    "car_f_92_warning": "For car-f-92 the simple modal version is not sufficient even at d = 1,000,000.",
    "lse_f_91_observation": "Cumulative bi-PAT converges in 3105 iterations vs 40010 for the only feasible single mode m=17.",
    "operator_noncommutativity": "F_k operators do not commute; exam ordering matters.",
    "best_baseline_methods_paper_table3": ["LD", "SD", "WD", "LE", "RO", "SA", "GA", "AC", "bi-PAT"],
    "file_aliases": FILE_ALIAS,
}


def main() -> None:
    df = pd.DataFrame(TORONTO)
    df.to_csv(OUT_DIR / "toronto_instances.csv", index=False)
    (OUT_DIR / "toronto_instances.json").write_text(json.dumps(TORONTO, indent=2))

    rows = []
    for problem, payload in FEASIBLE_MODES.items():
        for mode in payload["modes"]:
            rows.append({
                "problem": problem,
                "timeslots": payload["timeslots"],
                "m": mode["m"],
                "d": mode["d"],
                "note": payload.get("note", ""),
            })
    fdf = pd.DataFrame(rows)
    fdf.to_csv(OUT_DIR / "paper_feasible_modes.csv", index=False)
    (OUT_DIR / "paper_feasible_modes.json").write_text(json.dumps(FEASIBLE_MODES, indent=2))

    (OUT_DIR / "paper_notes.json").write_text(json.dumps(PAPER_NOTES, indent=2))

    print(f"Wrote {len(df)} instances and {len(fdf)} feasible modes to {OUT_DIR}")
    for p in sorted(OUT_DIR.glob("*")):
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
