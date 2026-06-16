from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from experiment_utils import FIG_DIR, OUT_DIR, load_paper_baselines


EXP_DIR = OUT_DIR / "experiments"
HIST_DIR = EXP_DIR / "histories"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelweight": "bold",
        "legend.fontsize": 8.5,
        "legend.framealpha": 0.85,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


def save(fig, name):
    p_png = FIG_DIR / f"{name}.png"
    p_pdf = FIG_DIR / f"{name}.pdf"
    fig.savefig(p_png, bbox_inches="tight")
    try:
        fig.savefig(p_pdf, bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)


def load_history(fname):
    if not fname or not isinstance(fname, str):
        return None
    p = HIST_DIR / fname
    if not p.exists():
        return None
    try:
        return np.load(p)
    except Exception:
        return None


def _apply_smart_yscale(ax, series_list, linthresh: float = 1.0) -> bool:
    """Switch ax to symlog y-scale when dynamic range >= 20x; returns True if applied."""
    flat = [v for s in series_list
            for v in np.asarray(s).ravel()
            if np.isfinite(v) and v >= 0]
    if not flat:
        return False
    vmax = max(flat)
    vmin_pos = min((v for v in flat if v > 0), default=0)
    if vmax > 0 and vmax / max(vmin_pos, 1) >= 20:
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.set_ylim(bottom=0)  # clashes are non-negative; hide phantom negative axis
        return True
    return False


def _clean_pname(v) -> str:
    """Return empty string if v is NaN/None/empty; otherwise the stripped string."""
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none", "") else s


def coerce_df(df):
    for c in ["min_clash", "final_clash", "max_clash", "init_clash", "mean_clash",
              "cumulative_steps", "wall_time_seconds", "best_step",
              "n_hopfield_calls", "hopfield_total_delta",
              "hopfield_improved_calls", "hopfield_worsened_calls",
              "hopfield_unchanged_calls", "hopfield_rolledback_calls",
              "paper_max_clashes", "mode_m", "duration"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "reached_zero" in df.columns:
        df["reached_zero"] = df["reached_zero"].astype(str).str.lower().isin(["true", "1", "1.0"])
    return df


def plot_clash_curves_per_instance(df):
    for inst, g in df.groupby("instance_name"):
        g = g[g["history_file"].astype(str) != ""].copy()
        g = g.sort_values("min_clash").head(8)
        if g.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        histories_plotted = []
        for _, row in g.iterrows():
            h = load_history(row["history_file"])
            if h is None or len(h) == 0:
                continue
            pname = _clean_pname(row.get('pattern_name'))
            label = f"{row['experiment']}  m={int(row['mode_m'])}" + (f"  {pname}" if pname else "")
            ax.plot(h, lw=1.2, alpha=0.85, label=label[:52])
            histories_plotted.append(h)
        log_applied = _apply_smart_yscale(ax, histories_plotted)
        ax.set_title(f"{inst}: clashes over steps — top runs")
        ax.set_xlabel("step")
        ax.set_ylabel("clashes (symlog)" if log_applied else "clashes")
        ax.legend(loc="upper right", ncol=1, fontsize=8)
        save(fig, f"curves_{inst}")

        fig2, ax2 = plt.subplots(figsize=(9, 5))
        for _, row in g.iterrows():
            h = load_history(row["history_file"])
            if h is None or len(h) == 0:
                continue
            pname2 = _clean_pname(row.get('pattern_name'))
            label2 = f"{row['experiment']}  m={int(row['mode_m'])}" + (f"  {pname2}" if pname2 else "")
            yvals = np.where(h <= 0, 0.5, h)
            ax2.semilogy(yvals, lw=1.2, alpha=0.85, label=label2[:52])
        ax2.set_title(f"{inst}: clashes over steps (log y)")
        ax2.set_xlabel("step")
        ax2.set_ylabel("clashes (log)")
        ax2.legend(loc="upper right", frameon=False, fontsize=8)
        save(fig2, f"curves_log_{inst}")


def plot_exams_vs_clashes(df):
    g = (df.groupby(["instance_name", "n_exams"])["min_clash"].min().reset_index())
    g = g.dropna()
    fig, ax = plt.subplots(figsize=(11, 6))
    zero = g[g["min_clash"] == 0]
    nonzero = g[g["min_clash"] > 0]

    ax.scatter(zero["n_exams"], zero["min_clash"], s=110, alpha=0.9,
               edgecolors="darkgreen", color="#2ca02c", zorder=3,
               label=f"Conflict-free — 0 clashes achieved ({len(zero)} instances)")
    ax.scatter(nonzero["n_exams"], nonzero["min_clash"], s=110, alpha=0.85,
               edgecolors="black", color="#1f77b4", zorder=3,
               label="Residual clashes remain")

    # Non-zero points: few, label directly
    for _, row in nonzero.iterrows():
        ax.annotate(row["instance_name"], (row["n_exams"], row["min_clash"]),
                    fontsize=9, xytext=(8, 7), textcoords="offset points",
                    color="#1f77b4", fontweight="bold")

    # Zero-clash cluster: staggered offsets to avoid pile-up
    offsets = [(8, 12), (-8, 20), (12, -14), (-12, -14), (18, 5),
               (-18, 5), (8, 28), (-8, -24), (22, 16), (-22, 16),
               (0, 34), (14, -22)]
    for i, (_, row) in enumerate(zero.sort_values("n_exams").iterrows()):
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(row["instance_name"], (row["n_exams"], row["min_clash"]),
                    fontsize=8, xytext=(dx, dy), textcoords="offset points",
                    color="darkgreen",
                    arrowprops=dict(arrowstyle="-", color="darkgreen",
                                   lw=0.6, alpha=0.5))

    ax.set_xlabel("number of exams (log scale)")
    ax.set_ylabel("best achieved min clash count")
    ax.set_xscale("log")
    ax.set_yscale("symlog")
    ax.set_ylim(bottom=0)
    ax.set_title("Exams vs Best Min-Clashes Achieved")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "exams_vs_min_clashes")


def plot_paper_vs_best(df):
    paper = load_paper_baselines()
    rows = []
    for inst, g in df.groupby("instance_name"):
        try:
            paper_name = g["paper_name"].iloc[0]
        except Exception:
            paper_name = inst
        info = paper.get(paper_name) or paper.get(inst)
        if not info:
            continue
        rows.append({
            "instance": inst,
            "paper_max": info["max_clashes"],
            "best_dynamic": g[g["experiment"] == "dynamic_only"]["min_clash"].min(),
            "best_hybrid": g[g["experiment"] == "hybrid"]["min_clash"].min(),
        })
    if not rows:
        return
    pdf = pd.DataFrame(rows).fillna(0).sort_values("paper_max")
    x = np.arange(len(pdf))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w, pdf["paper_max"], w, label="paper max clashes (slot 0 init)", color="#888")
    ax.bar(x, pdf["best_dynamic"], w, label="our best dynamic-only min", color="#1f77b4")
    ax.bar(x + w, pdf["best_hybrid"], w, label="our best hybrid min", color="#d62728")
    # Annotate zero-clash bars — invisible at y=0 on symlog; mark them explicitly
    for i, (_, row) in enumerate(pdf.iterrows()):
        if row["best_dynamic"] == 0:
            ax.text(x[i], 0.45, "0", ha="center", va="bottom",
                    fontsize=8.5, color="#1f77b4", fontweight="bold")
        if row["best_hybrid"] == 0:
            ax.text(x[i] + w, 0.45, "0", ha="center", va="bottom",
                    fontsize=8.5, color="#d62728", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pdf["instance"], rotation=35, ha="right")
    ax.set_yscale("symlog")
    ax.set_ylabel("clashes")
    ax.set_title("Paper Max Clashes vs Our Best Min Clashes")
    ax.legend()
    save(fig, "paper_max_vs_best")

    fig2, ax2 = plt.subplots(figsize=(11, 5))
    rd = pdf.copy()
    rd["dyn_red"] = 1 - rd["best_dynamic"] / rd["paper_max"]
    rd["hyb_red"] = 1 - rd["best_hybrid"] / rd["paper_max"]
    ax2.bar(x - w / 2, rd["dyn_red"], w, label="dynamic-only", color="#1f77b4")
    ax2.bar(x + w / 2, rd["hyb_red"], w, label="hybrid", color="#d62728")
    ax2.set_xticks(x)
    ax2.set_xticklabels(rd["instance"], rotation=35, ha="right")
    ax2.set_ylabel("reduction ratio  1 − best / paper_max")
    ax2.set_ylim(0, 1.05)
    ax2.set_title(
        "Improvement Ratio Compared to Paper Max Clashes\n"
        "(1.0 = zero clashes achieved — fully conflict-free)",
        fontsize=11)
    ax2.legend()
    ax2.axhline(1.0, color="green", lw=1, ls="--", alpha=0.6)
    # Annotate bars that fell below 1.0 so the reader can see the exact shortfall
    for i, (_, row) in enumerate(rd.iterrows()):
        v_dyn = row["dyn_red"]
        v_hyb = row["hyb_red"]
        if abs(v_dyn - 1.0) > 1e-6:
            ax2.text(i - w / 2, v_dyn - 0.035, f"{v_dyn:.4f}", ha="center",
                     fontsize=7, color="white", fontweight="bold")
        if abs(v_hyb - 1.0) > 1e-6:
            ax2.text(i + w / 2, v_hyb - 0.035, f"{v_hyb:.4f}", ha="center",
                     fontsize=7, color="white", fontweight="bold")
    save(fig2, "improvement_ratio")


def plot_best_mode_heatmap(df):
    g = df[df["experiment"] == "dynamic_only"].copy()
    if g.empty:
        return
    pivot = g.groupby(["instance_name", "mode_m"])["min_clash"].min().unstack()
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    data = np.log1p(pivot.values.astype(float))
    im = ax.imshow(data, aspect="auto", cmap="viridis_r")
    n_cols = len(pivot.columns)
    step = max(1, n_cols // 20)
    tick_pos = list(range(0, n_cols, step))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([pivot.columns[i] for i in tick_pos], rotation=90, fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("mode m")
    ax.set_ylabel("instance")
    ax.set_title("Min clashes (log1p) per instance and mode (dynamic-only)")
    plt.colorbar(im, ax=ax, label="log(1 + min_clash)")
    save(fig, "best_mode_heatmap")


def plot_hopfield_effect(df):
    g = df[df["experiment"] == "hybrid"].copy()
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    g["pct_improved"] = g["hopfield_improved_calls"] / g["n_hopfield_calls"].replace(0, np.nan)
    g["pct_worsened"] = g["hopfield_worsened_calls"] / g["n_hopfield_calls"].replace(0, np.nan)
    grouped = g.groupby("instance_name").agg(
        improved=("pct_improved", "mean"),
        worsened=("pct_worsened", "mean"),
        rolledback=("hopfield_rolledback_calls", "sum"),
    ).reset_index().fillna(0)
    x = np.arange(len(grouped))
    ax.bar(x - 0.2, grouped["improved"], 0.4, label="improved", color="#2ca02c")
    ax.bar(x + 0.2, grouped["worsened"], 0.4, label="worsened (after rollback)", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["instance_name"], rotation=35, ha="right")
    ax.set_ylabel("fraction of Hopfield calls")
    ax.set_title("Hopfield Effect per Instance")
    ax.legend()
    save(fig, "hopfield_effect")


def plot_dyn_vs_hybrid(df):
    dyn = df[df["experiment"] == "dynamic_only"].groupby("instance_name")["min_clash"].min()
    hyb = df[df["experiment"] == "hybrid"].groupby("instance_name")["min_clash"].min()
    common = sorted(set(dyn.index) & set(hyb.index))
    if not common:
        return
    fig, ax = plt.subplots(figsize=(9, 7))
    xs = [dyn[c] for c in common]
    ys = [hyb[c] for c in common]

    # Color zero-clash instances (both methods at 0) distinctly
    colors = ["#2ca02c" if xi == 0 and yi == 0 else "#1f77b4"
              for xi, yi in zip(xs, ys)]
    ax.scatter(xs, ys, s=90, c=colors, edgecolors="black", alpha=0.85, zorder=3)

    # Stagger label offsets to avoid pile-up at (0, 0)
    zero_offsets = [(-8, 14), (10, -16), (-10, -22), (16, 16), (-20, 4),
                    (4, 26), (-20, -8), (20, -4), (0, 32), (-24, 16)]
    other_offsets = [(7, 7), (-7, 14), (7, -14), (-14, 7), (12, -6)]
    z_count = oth_count = 0
    for c_name, xi, yi in zip(common, xs, ys):
        if xi == 0 and yi == 0:
            dx, dy = zero_offsets[z_count % len(zero_offsets)]
            z_count += 1
        else:
            dx, dy = other_offsets[oth_count % len(other_offsets)]
            oth_count += 1
        ax.annotate(c_name, (xi, yi), fontsize=8, xytext=(dx, dy),
                    textcoords="offset points")

    mx = max(max(xs), max(ys), 1)
    ax.plot([0, mx], [0, mx], "k--", alpha=0.5, label="y=x (no advantage)")

    from matplotlib.lines import Line2D
    extra_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c",
               markeredgecolor="black", markersize=9,
               label="Both methods: 0 clashes (conflict-free)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4",
               markeredgecolor="black", markersize=9,
               label="Residual clashes remain"),
    ]
    ax.set_xlabel("dynamic-only best min clash")
    ax.set_ylabel("hybrid best min clash")
    ax.set_xscale("symlog")
    ax.set_yscale("symlog")
    ax.set_title("Dynamic-only vs Dynamic+Hopfield")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + extra_handles, loc="lower right", fontsize=8.5)
    save(fig, "dyn_vs_hybrid")


def plot_runtime_vs_quality(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    for exp, grp in df.groupby("experiment"):
        ax.scatter(grp["wall_time_seconds"], grp["min_clash"].clip(lower=0.5),
                   alpha=0.5, label=exp, s=18)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wall time (s, log)")
    ax.set_ylabel("min clash (log, clipped at 0.5)")
    ax.set_title("Runtime vs Quality")
    ax.legend()
    save(fig, "runtime_vs_quality")


def plot_pattern_distributions(df):
    g = df[df["experiment"] == "hybrid"].copy()
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    patterns = sorted(g["pattern_name"].dropna().unique())
    data = [g[g["pattern_name"] == p]["min_clash"].dropna().values for p in patterns]
    ax.boxplot(data, labels=patterns, showfliers=True)
    ax.set_ylabel("min clash")
    ax.set_title("Min Clash Distribution by Hybrid Pattern")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.set_yscale("symlog")
    save(fig, "pattern_min_clash_dist")

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    data2 = [g[g["pattern_name"] == p]["final_clash"].dropna().values for p in patterns]
    ax2.boxplot(data2, labels=patterns, showfliers=True)
    ax2.set_ylabel("final clash")
    ax2.set_title("Final Clash Distribution by Hybrid Pattern")
    plt.setp(ax2.get_xticklabels(), rotation=20, ha="right")
    ax2.set_yscale("symlog")
    save(fig2, "pattern_final_clash_dist")


def plot_hop_calls_vs_improvement(df):
    g = df[df["experiment"] == "hybrid"].copy()
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(g["n_hopfield_calls"], g["hopfield_total_delta"], alpha=0.5, s=20)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("number of Hopfield calls")
    ax.set_ylabel("sum of (after - before) deltas")
    ax.set_title("Hopfield Calls vs Net Clash Delta")
    save(fig, "hopfield_calls_vs_delta")


def plot_normalized_best(df):
    paper = load_paper_baselines()
    rows = []
    for inst, g in df.groupby("instance_name"):
        try:
            paper_name = g["paper_name"].iloc[0]
        except Exception:
            paper_name = inst
        info = paper.get(paper_name) or paper.get(inst)
        if not info or not info["max_clashes"]:
            continue
        best = g["min_clash"].min()
        rows.append({"instance": inst, "norm_best": best / info["max_clashes"]})
    if not rows:
        return
    rdf = pd.DataFrame(rows).sort_values("norm_best")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bar_colors = ["#2ca02c" if v == 0 else "#1f77b4" for v in rdf["norm_best"]]
    bars = ax.bar(rdf["instance"], rdf["norm_best"], color=bar_colors)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(range(len(rdf)))
    ax.set_xticklabels(list(rdf["instance"]), rotation=35, ha="right")
    ax.set_ylabel("best_min_clash / paper_max_clashes")
    ax.set_title(
        "Best Achieved Clash Count Normalized by Paper Max\n"
        "(0.0 = fully conflict-free — zero clashes remaining)")
    # Annotate zero-clash bars — they show no visible bar height but need clear labelling
    ymax = rdf["norm_best"].max()
    ann_y = ymax * 0.04 if ymax > 0 else 0.00002
    for bar, (_, row) in zip(bars, rdf.iterrows()):
        if row["norm_best"] == 0:
            ax.text(bar.get_x() + bar.get_width() / 2, ann_y,
                    "Conflict-\nfree", ha="center", va="bottom",
                    fontsize=7, color="#2ca02c", fontweight="bold", linespacing=1.1)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#2ca02c", label="0 clashes achieved (conflict-free)"),
        Patch(facecolor="#1f77b4", label="Residual clashes remain"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper left")
    save(fig, "best_normalized_by_paper")


def plot_mode_type_comparison(df):
    g = df[df["experiment"] == "dynamic_only"].copy()
    if g.empty or "mode_type" not in g.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    types = sorted(g["mode_type"].dropna().unique())
    data = [g[g["mode_type"] == t]["min_clash"].dropna().values for t in types]
    ax.boxplot(data, labels=types)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    ax.set_ylabel("min clash")
    ax.set_yscale("symlog")
    ax.set_title("Min Clash by Mode Type (dynamic-only)")
    save(fig, "mode_type_comparison")


def plot_convergence_per_instance(df):
    best_idx = df[df["history_file"].astype(str) != ""].groupby("instance_name")["min_clash"].idxmin()
    best_idx = best_idx.dropna()
    best = df.loc[best_idx]
    for _, row in best.iterrows():
        h = load_history(row["history_file"])
        if h is None or len(h) == 0:
            continue
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(h, lw=1.5, color="#1f77b4")
        ax.axhline(0, color="green", lw=0.9, ls="--", alpha=0.7, label="0 clashes")
        log_applied = _apply_smart_yscale(ax, [h])
        ax.set_xlabel("step")
        ax.set_ylabel("clashes (symlog)" if log_applied else "clashes")
        pname = _clean_pname(row.get('pattern_name'))
        title_extra = f"  {pname}" if pname else ""
        ax.set_title(
            f"Best convergence — {row['instance_name']}  "
            f"[{row['experiment']}  m={int(row['mode_m'])}{title_extra}]"
        )
        ax.legend(loc="upper right", frameon=False)
        # Emphasise if zero clashes were achieved
        h_arr = np.asarray(h)
        zero_steps = np.where(h_arr == 0)[0]
        if len(zero_steps) > 0:
            first_zero = int(zero_steps[0])
            ax.text(0.02, 0.06,
                    f"★  Conflict-free — 0 clashes reached at step {first_zero}",
                    transform=ax.transAxes, fontsize=9.5, color="#2ca02c",
                    fontweight="bold", va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="#f0fff0",
                              edgecolor="#2ca02c", alpha=0.9))
        save(fig, f"best_convergence_{row['instance_name']}")


def main():
    style()
    raw = EXP_DIR / "raw_results.csv"
    if not raw.exists():
        print(f"No raw results found at {raw}; run the experiments first.")
        return
    df = pd.read_csv(raw)
    df = coerce_df(df)
    df = df[df["error"].isna() | (df["error"].astype(str).str.len() == 0) | (df["error"].astype(str) == "nan")]
    print(f"Loaded {len(df)} rows. Generating figures into {FIG_DIR}")

    plot_clash_curves_per_instance(df)
    plot_exams_vs_clashes(df)
    plot_paper_vs_best(df)
    plot_best_mode_heatmap(df)
    plot_hopfield_effect(df)
    plot_dyn_vs_hybrid(df)
    plot_runtime_vs_quality(df)
    plot_pattern_distributions(df)
    plot_hop_calls_vs_improvement(df)
    plot_normalized_best(df)
    plot_mode_type_comparison(df)
    plot_convergence_per_instance(df)

    figs = sorted(FIG_DIR.glob("*.png"))
    print(f"Done. {len(figs)} PNG figures saved to {FIG_DIR}.")


if __name__ == "__main__":
    main()
