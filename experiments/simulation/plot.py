"""Render figures from saved CSVs without rerunning any transport solver."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_rows(path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    numeric = ("n", "d", "L", "sweep_value", "median_ms", "cost", "retention_ratio",
               "relative_gap", "exact_cost", "peak_traced_mib")
    for row in rows:
        for key in numeric:
            row[key] = float(row[key]) if row.get(key) not in (None, "", "None") else None
    return rows


def label(row):
    if row["method"] == "Sinkhorn":
        return f"Sinkhorn ε={float(row['epsilon']):g}"
    return f"{row['method']} L={int(row['L'])}"


def draw_curves(ax, rows, x_key, y_key):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get(x_key) is not None and row.get(y_key) is not None:
            groups[label(row)][row[x_key]].append(row[y_key])
    for name, points in groups.items():
        xs = sorted(points)
        means = np.array([np.mean(points[x]) for x in xs])
        std = np.array([np.std(points[x], ddof=1) if len(points[x]) > 1 else 0 for x in xs])
        line, = ax.plot(xs, means, "o-", label=name, markersize=4)
        ax.fill_between(xs, means-std, means+std, color=line.get_color(), alpha=.12)
    ax.grid(alpha=.2)
    if groups:
        ax.legend(fontsize=7)
    else:
        ax.text(.5, .5, "No converged results", transform=ax.transAxes, ha="center")


def plot_results(output_dir):
    out = Path(output_dir)
    all_rows = read_rows(out/"summary.csv")
    figures = out/"figures"
    figures.mkdir(exist_ok=True)
    combinations = sorted({(r["d"], r["weights"], r["projection_kind"], r["L"])
                           for r in all_rows if r["method"] == "LMOT"})
    for d, weights, kind, count in combinations:
        group = [r for r in all_rows if r["d"] == d and r["weights"] == weights and
                 (r["method"] == "Sinkhorn" or (r["projection_kind"] == kind and r["L"] == count))]
        valid = [r for r in group if r["status"] == "ok"]
        if not valid:
            continue
        reference_sizes = [r["n"] for r in valid if r["exact_cost"] is not None]
        n_quality = max(reference_sizes or [r["n"] for r in valid])
        values = sorted({r["sweep_value"] for r in valid})
        mid = min(values, key=lambda v: abs(v-.5))
        small = [r for r in valid if r["n"] == n_quality]
        scale = [r for r in valid if r["sweep_value"] == mid]
        scenario = group[0]["scenario"]
        x_label = "Requested support overlap" if scenario == "support" else "Weight change δ"
        fig, ax = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        draw_curves(ax[0, 0], small, "sweep_value", "cost")
        ax[0, 0].set(title=f"Ground transport cost (n={int(n_quality)})", xlabel=x_label, ylabel="Cost (no entropy)")
        draw_curves(ax[0, 1], scale, "n", "median_ms")
        ax[0, 1].set(title=f"Cost + map runtime (sweep={mid:g})", xlabel="Atoms per measure", ylabel="Median wall time (ms)", xscale="log", yscale="log")
        draw_curves(ax[1, 0], small, "sweep_value", "retention_ratio")
        ax[1, 0].set(title="Retained / available common mass", xlabel=x_label, ylabel="Retention ratio")
        # Same n and data setting; no mixing of different transport problems.
        pareto = [r for r in valid if r["n"] == n_quality and r["sweep_value"] == mid]
        metric = "relative_gap" if any(r["relative_gap"] is not None for r in pareto) else "cost"
        collected = defaultdict(list)
        for row in pareto:
            if row[metric] is not None:
                collected[label(row)].append(row)
        for name, points in collected.items():
            ax[1, 1].scatter(np.mean([p["median_ms"] for p in points]), np.mean([p[metric] for p in points]), label=name)
        ax[1, 1].set(title=f"Quality–runtime (n={int(n_quality)}, sweep={mid:g})",
                     xlabel="Median wall time (ms)", ylabel="Relative exact-OT gap" if metric == "relative_gap" else "Ground cost", xscale="log")
        ax[1, 1].grid(alpha=.2)
        if collected:
            ax[1, 1].legend(fontsize=7)
        excluded = sum(r["status"] != "ok" for r in group)
        fig.suptitle(f"d={int(d)} · {weights} · {kind} directions · L={int(count)}\n"
                     f"Valid/converged runs only; {excluded} rows excluded (see CSV). Bands: ±1 SD across seeds.", fontsize=11)
        stem = f"comparison_d{int(d)}_{weights}_{kind}_L{int(count)}"
        fig.savefig(figures/f"{stem}.png", dpi=160)
        fig.savefig(figures/f"{stem}.pdf")
        plt.close(fig)
    return figures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    plot_results(parser.parse_args().output_dir)
