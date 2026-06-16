from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "data_visuals"
OUT_DIR.mkdir(exist_ok=True)


def parse_crs(path):
    enrollments = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    enrollments[int(parts[0])] = int(parts[1])
                except ValueError:
                    continue
    return enrollments


def parse_stu(path):
    students = []
    with open(path) as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                try:
                    students.append([int(t) for t in tokens])
                except ValueError:
                    continue
    return students


def build_clash_matrix(students, num_courses):
    size = num_courses + 1
    matrix = np.zeros((size, size), dtype=np.int32)
    for sc in students:
        unique = list(set(sc))
        for i in unique:
            if i <= 0 or i >= size:
                continue
            for j in unique:
                if j <= 0 or j >= size or i == j:
                    continue
                matrix[i][j] += 1
    return matrix


def find_instances():
    crs_files = sorted(DATA_DIR.glob("*.crs"))
    instances = []
    for crs in crs_files:
        stu = crs.with_suffix(".stu")
        if stu.exists():
            instances.append((crs.stem, crs, stu))
    return instances


def per_instance_plots(name, crs_path, stu_path):
    enrollments = parse_crs(crs_path)
    students = parse_stu(stu_path)
    num_courses = max(enrollments.keys()) if enrollments else 0
    matrix = build_clash_matrix(students, num_courses)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Instance: {name}", fontsize=16, fontweight="bold")

    vals = list(enrollments.values())
    axes[0, 0].hist(vals, bins=30, color="steelblue", edgecolor="black")
    axes[0, 0].set_title(f"Course Enrollment Distribution (n={len(vals)})")
    axes[0, 0].set_xlabel("Students enrolled in course")
    axes[0, 0].set_ylabel("Number of courses")
    axes[0, 0].grid(True, alpha=0.3)

    cps = [len(s) for s in students]
    if cps:
        axes[0, 1].hist(cps, bins=range(1, max(cps) + 2), color="coral", edgecolor="black")
    axes[0, 1].set_title(f"Courses per Student (students={len(students)})")
    axes[0, 1].set_xlabel("Courses taken")
    axes[0, 1].set_ylabel("Number of students")
    axes[0, 1].grid(True, alpha=0.3)

    binary = (matrix > 0).astype(np.int8)
    axes[1, 0].imshow(binary[1:, 1:], cmap="Greys", aspect="auto", interpolation="nearest")
    axes[1, 0].set_title(f"Clash Graph (binary) {num_courses}x{num_courses}")
    axes[1, 0].set_xlabel("Course ID")
    axes[1, 0].set_ylabel("Course ID")

    degrees = (matrix > 0).sum(axis=1)[1:]
    axes[1, 1].hist(degrees, bins=30, color="seagreen", edgecolor="black")
    axes[1, 1].set_title(f"Clash Degree Distribution (avg={degrees.mean():.1f})")
    axes[1, 1].set_xlabel("Number of conflicting courses")
    axes[1, 1].set_ylabel("Number of courses")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT_DIR / f"{name}_overview.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    capped = np.minimum(matrix[1:, 1:], 20)
    im = ax2.imshow(capped, cmap="hot", aspect="auto", interpolation="nearest")
    ax2.set_title(f"{name}: Clash Weight Heatmap (shared students, capped at 20)")
    ax2.set_xlabel("Course ID")
    ax2.set_ylabel("Course ID")
    plt.colorbar(im, ax=ax2, label="Shared students")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}_clash_heatmap.png", dpi=100, bbox_inches="tight")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(12, 6))
    sorted_enroll = sorted(vals, reverse=True)
    ax3.bar(range(len(sorted_enroll)), sorted_enroll, color="darkblue")
    ax3.set_title(f"{name}: Course Enrollments (sorted descending)")
    ax3.set_xlabel("Course rank")
    ax3.set_ylabel("Students enrolled")
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}_enrollments_sorted.png", dpi=100, bbox_inches="tight")
    plt.close(fig3)

    total_edges = int((matrix > 0).sum() / 2)
    possible = num_courses * (num_courses - 1) / 2
    density = total_edges / possible if possible > 0 else 0

    return {
        "name": name,
        "num_courses": num_courses,
        "num_students": len(students),
        "avg_enrollment": float(np.mean(vals)) if vals else 0,
        "median_enrollment": float(np.median(vals)) if vals else 0,
        "max_enrollment": int(max(vals)) if vals else 0,
        "avg_courses_per_student": float(np.mean(cps)) if cps else 0,
        "max_courses_per_student": int(max(cps)) if cps else 0,
        "total_clash_edges": total_edges,
        "graph_density": density,
        "avg_degree": float(degrees.mean()) if len(degrees) > 0 else 0,
        "max_degree": int(degrees.max()) if len(degrees) > 0 else 0,
    }


def summary_plots(stats):
    names = [s["name"] for s in stats]
    x = np.arange(len(names))

    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    fig.suptitle("All Instances - Summary Comparison", fontsize=18, fontweight="bold")

    axes[0, 0].bar(x, [s["num_courses"] for s in stats], color="steelblue")
    axes[0, 0].set_title("Number of Courses")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(names, rotation=45, ha="right")
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    axes[0, 1].bar(x, [s["num_students"] for s in stats], color="coral")
    axes[0, 1].set_title("Number of Students")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(names, rotation=45, ha="right")
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    axes[1, 0].bar(x, [s["avg_courses_per_student"] for s in stats], color="seagreen")
    axes[1, 0].set_title("Avg Courses per Student")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(names, rotation=45, ha="right")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    axes[1, 1].bar(x, [s["avg_enrollment"] for s in stats], color="goldenrod")
    axes[1, 1].set_title("Avg Enrollment per Course")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(names, rotation=45, ha="right")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    axes[2, 0].bar(x, [s["graph_density"] for s in stats], color="purple")
    axes[2, 0].set_title("Clash Graph Density")
    axes[2, 0].set_xticks(x)
    axes[2, 0].set_xticklabels(names, rotation=45, ha="right")
    axes[2, 0].grid(True, alpha=0.3, axis="y")

    axes[2, 1].bar(x, [s["avg_degree"] for s in stats], color="teal")
    axes[2, 1].set_title("Avg Clash Degree")
    axes[2, 1].set_xticks(x)
    axes[2, 1].set_xticklabels(names, rotation=45, ha="right")
    axes[2, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT_DIR / "00_all_instances_summary.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(12, 8))
    ncourses = np.array([s["num_courses"] for s in stats])
    nstudents = np.array([s["num_students"] for s in stats])
    densities = np.array([s["graph_density"] for s in stats])
    sizes = 100 + densities / max(densities.max(), 1e-9) * 1500
    colors = plt.cm.tab20(np.linspace(0, 1, len(stats)))
    ax.scatter(ncourses, nstudents, s=sizes, c=colors, alpha=0.7, edgecolors="black", linewidths=1.5)
    for s, xv, yv in zip(stats, ncourses, nstudents):
        ax.annotate(s["name"], (xv, yv), fontsize=9, xytext=(7, 7), textcoords="offset points")
    ax.set_xlabel("Number of Courses")
    ax.set_ylabel("Number of Students")
    ax.set_title("Instances: Courses vs Students (bubble size = clash density)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "00_size_vs_density.png", dpi=100, bbox_inches="tight")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(14, 7))
    metrics = ["graph_density", "avg_degree"]
    width = 0.35
    ax3b = ax3.twinx()
    ax3.bar(x - width / 2, [s["graph_density"] for s in stats], width, color="purple", label="Density", alpha=0.8)
    ax3b.bar(x + width / 2, [s["avg_degree"] for s in stats], width, color="teal", label="Avg degree", alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=45, ha="right")
    ax3.set_ylabel("Clash graph density", color="purple")
    ax3b.set_ylabel("Average clash degree", color="teal")
    ax3.set_title("Clash Difficulty per Instance")
    ax3.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "00_clash_difficulty.png", dpi=100, bbox_inches="tight")
    plt.close(fig3)


def write_stats_table(stats):
    headers = [
        "instance", "num_courses", "num_students", "avg_enrollment",
        "median_enrollment", "max_enrollment", "avg_courses_per_student",
        "max_courses_per_student", "total_clash_edges", "graph_density",
        "avg_degree", "max_degree",
    ]
    lines = [",".join(headers)]
    for s in stats:
        lines.append(",".join([
            s["name"],
            str(s["num_courses"]),
            str(s["num_students"]),
            f"{s['avg_enrollment']:.2f}",
            f"{s['median_enrollment']:.2f}",
            str(s["max_enrollment"]),
            f"{s['avg_courses_per_student']:.2f}",
            str(s["max_courses_per_student"]),
            str(s["total_clash_edges"]),
            f"{s['graph_density']:.6f}",
            f"{s['avg_degree']:.2f}",
            str(s["max_degree"]),
        ]))
    (OUT_DIR / "stats.csv").write_text("\n".join(lines))


def main():
    instances = find_instances()
    print(f"Found {len(instances)} instances")
    stats = []
    for name, crs, stu in instances:
        print(f"Processing {name}...")
        s = per_instance_plots(name, crs, stu)
        stats.append(s)
        print(f"  courses={s['num_courses']}, students={s['num_students']}, "
              f"density={s['graph_density']:.4f}, avg_deg={s['avg_degree']:.1f}")
    summary_plots(stats)
    write_stats_table(stats)
    files = sorted(OUT_DIR.glob("*"))
    print(f"\nDone. Output dir: {OUT_DIR}")
    print(f"Files created: {len(files)}")
    for f in files:
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
