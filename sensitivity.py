import argparse

from engine import TimetableEngine


def hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def sweep(engine, h_values, d_values, shift):
    grid = {}
    for h in sorted(set(h_values)):
        for d in sorted(set(d_values)):
            r = engine.run(d, h, shift)
            grid[(h, d)] = r
    return grid


def finite_diffs(grid, h_values, d_values, key='min_clashes'):
    h_values = sorted(set(h_values))
    d_values = sorted(set(d_values))
    h_deltas = []
    for d in d_values:
        for i in range(len(h_values) - 1):
            h1, h2 = h_values[i], h_values[i + 1]
            if h2 == h1:
                continue
            slope = abs(grid[(h2, d)][key] - grid[(h1, d)][key]) / (h2 - h1)
            h_deltas.append(((h1, h2, d), slope))
    d_deltas = []
    for h in h_values:
        for i in range(len(d_values) - 1):
            d1, d2 = d_values[i], d_values[i + 1]
            if d2 == d1:
                continue
            slope = abs(grid[(h, d2)][key] - grid[(h, d1)][key]) / (d2 - d1)
            d_deltas.append(((h, d1, d2), slope))
    return h_deltas, d_deltas


def analyze(engine, h_values, d_values, shift, threshold=0.05):
    grid = sweep(engine, h_values, d_values, shift)
    h_deltas, d_deltas = finite_diffs(grid, h_values, d_values, 'min_clashes')
    clashes = [r['min_clashes'] for r in grid.values()]
    mean_c = sum(clashes) / len(clashes) if clashes else 0.0
    var_c = sum((c - mean_c) ** 2 for c in clashes) / len(clashes) if clashes else 0.0
    std_c = var_c ** 0.5

    slopes_h = [s for _, s in h_deltas]
    slopes_d = [s for _, s in d_deltas]
    mean_sh = sum(slopes_h) / len(slopes_h) if slopes_h else 0.0
    mean_sd = sum(slopes_d) / len(slopes_d) if slopes_d else 0.0
    max_sh = max(slopes_h) if slopes_h else 0.0
    max_sd = max(slopes_d) if slopes_d else 0.0
    max_slope = max(max_sh, max_sd)
    normalized = (max_slope / mean_c) if mean_c > 0 else 0.0

    assignments = [r['slot_assignment'] for r in grid.values()]
    pairwise_hamming = []
    for i in range(len(assignments)):
        for j in range(i + 1, len(assignments)):
            pairwise_hamming.append(hamming(assignments[i], assignments[j]))
    mean_hamming = sum(pairwise_hamming) / len(pairwise_hamming) if pairwise_hamming else 0.0
    n_courses = len(assignments[0]) if assignments else 1
    rel_hamming = mean_hamming / n_courses if n_courses else 0.0

    return {
        'grid': grid,
        'h_values': sorted(set(h_values)),
        'd_values': sorted(set(d_values)),
        'mean_clashes': mean_c,
        'std_clashes': std_c,
        'mean_sensitivity_h': mean_sh,
        'mean_sensitivity_d': mean_sd,
        'max_sensitivity_h': max_sh,
        'max_sensitivity_d': max_sd,
        'max_sensitivity': max_slope,
        'normalized_sensitivity': normalized,
        'threshold': threshold,
        'is_sensitive': normalized > threshold,
        'mean_hamming_distance': mean_hamming,
        'relative_hamming': rel_hamming,
    }


def print_report(report):
    grid = report['grid']
    h_values = report['h_values']
    d_values = report['d_values']

    print()
    print("=" * 72)
    print("Sensitivity report")
    print("=" * 72)
    header = "h \\ d"
    print(f"{header:>8}", end="")
    for d in d_values:
        print(f"{d:>10}", end="")
    print()
    for h in h_values:
        print(f"{h:>8}", end="")
        for d in d_values:
            print(f"{grid[(h, d)]['min_clashes']:>10}", end="")
        print()
    print()
    print(f"Mean min-clashes across grid:        {report['mean_clashes']:.2f}")
    print(f"Std  min-clashes across grid:        {report['std_clashes']:.2f}")
    print(f"Mean |dC/dh| (per Hopfield run):     {report['mean_sensitivity_h']:.4f}")
    print(f"Max  |dC/dh|:                        {report['max_sensitivity_h']:.4f}")
    print(f"Mean |dC/dd| (per dynamic step):     {report['mean_sensitivity_d']:.6f}")
    print(f"Max  |dC/dd|:                        {report['max_sensitivity_d']:.6f}")
    print(f"Max slope:                           {report['max_sensitivity']:.4f}")
    print(f"Normalized sensitivity (max/mean):   {report['normalized_sensitivity']:.4f}")
    print(f"Threshold:                           {report['threshold']:.4f}")
    print(f"Relative Hamming across runs:        {report['relative_hamming']:.4f}")
    print("-" * 72)
    verdict = "SENSITIVE" if report['is_sensitive'] else "STABLE"
    print(f"VERDICT: {verdict}")
    print("=" * 72)


def main():
    p = argparse.ArgumentParser(description="Sensitivity analysis of the timetable system over (Hopfield runs, dynamic steps).")
    p.add_argument("--slots", type=int, default=22)
    p.add_argument("--courses", type=int, default=190)
    p.add_argument("--clashes", default="./data/ear-f-83.stu")
    p.add_argument("--shift", type=int, default=21)
    p.add_argument("--hopfield-runs", nargs="+", type=int, default=[0, 2, 5, 10, 20])
    p.add_argument("--dynamic-steps", nargs="+", type=int, default=[100, 250, 500, 1000])
    p.add_argument("--train-count", type=int, default=None)
    p.add_argument("--threshold", type=float, default=0.05)
    args = p.parse_args()

    engine = TimetableEngine(args.courses, args.slots, args.clashes)
    trained = engine.train(args.train_count)
    print(f"Trained Hopfield on {trained} patterns.")
    report = analyze(engine, args.hopfield_runs, args.dynamic_steps, args.shift, args.threshold)
    print_report(report)


if __name__ == "__main__":
    main()
