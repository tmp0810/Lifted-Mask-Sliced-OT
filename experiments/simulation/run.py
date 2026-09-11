"""Run a simulation from repo root: python -m experiments.simulation.run."""
import argparse
import csv
from datetime import datetime, timezone
from functools import partial
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import numpy as np
import yaml
from threadpoolctl import threadpool_limits, threadpool_info
from lmot import solve_lmot, solve_est, solve_sinkhorn
from lmot.benchmark import benchmark_functions, evaluate_with_memory
from lmot.metrics import overlap_statistics, collision_statistics, exact_ot_cost, evaluate_quality
from lmot.projections import make_projections
from .data import make_pair

ROOT = Path(__file__).resolve().parents[2]


def save_table(path, rows, delimiter=","):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def metadata():
    packages = {}
    for name in ("numpy", "scipy", "matplotlib", "PyYAML", "threadpoolctl", "POT"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    def git(*args):
        try:
            result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
            return result.stdout.strip() if result.returncode == 0 else None
        except FileNotFoundError:
            return None
    files = sorted(list((ROOT/"src").rglob("*.py")) + list((ROOT/"experiments").rglob("*.py")))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return {"utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "platform": platform.platform(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "packages": packages,
            "threadpools": threadpool_info(), "git_commit": git("rev-parse", "HEAD"),
            "git_status": git("status", "--porcelain"), "source_sha256": digest.hexdigest(),
            "dtype": "float64", "device": "CPU", "output": "cost + barycentric map",
            "memory_metric": "untimed tracemalloc peak, not total RSS",
            "cost": "squared Euclidean, excluding entropy; bounding-box cost <= 1"}


def validate_config(cfg):
    if cfg["scenario"] not in ("support", "weights"):
        raise ValueError("unknown scenario")
    values = cfg["overlap_values" if cfg["scenario"] == "support" else "weight_change_values"]
    if not values or any(not 0 <= v <= 1 for v in values):
        raise ValueError("sweep values must be in [0,1]")
    for name in ("sizes", "dimensions", "projection_counts", "seeds", "projection_kinds", "weight_modes"):
        if not cfg[name] or len(set(cfg[name])) != len(cfg[name]):
            raise ValueError(f"{name} must be nonempty and contain no duplicates")
    if min(cfg["sizes"]) < 2 or min(cfg["dimensions"]) < 2 or min(cfg["projection_counts"]) < 1:
        raise ValueError("invalid simulation size/dimension/projection count")
    if cfg["repeats"] < 1 or cfg["warmups"] < 0 or cfg["threads"] < 1:
        raise ValueError("invalid timing settings")
    if cfg["sinkhorn"]["backend"] == "pot":
        import ot  # Fail early with a clear missing-dependency error.
        if cfg["sinkhorn"].get("max_seconds") is not None:
            raise ValueError("POT requires max_seconds: null; use max_iter as its budget")
    return values


def run(config_path, output_dir=None, *, make_plots=True):
    cfg = yaml.safe_load(Path(config_path).read_text())
    values = validate_config(cfg)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out = Path(output_dir) if output_dir else ROOT/"results"/"simulation"/f"{cfg['name']}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)  # Never silently replace an earlier run.
    (out/"projections").mkdir()
    if cfg["save_inputs"]:
        (out/"inputs").mkdir()
    (out/"config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    rows, raw = [], []
    projection_cache = {}
    with threadpool_limits(limits=cfg["threads"]):
        (out/"metadata.json").write_text(json.dumps(metadata(), indent=2))
        for n, d, weights, seed, value in itertools.product(
                cfg["sizes"], cfg["dimensions"], cfg["weight_modes"], cfg["seeds"], values):
            args, is_self = make_pair(n, d, value=value, seed=seed, weights=weights,
                                     scenario=cfg["scenario"], geometry=cfg["geometry"])
            x, alpha, y, beta = args
            case = f"n{n}_d{d}_{weights}_s{seed}_v{value:g}"
            if cfg["save_inputs"]:
                np.savez_compressed(out/"inputs"/f"{case}.npz", X=x, alpha=alpha, Y=y, beta=beta)
            i, j, overlap = overlap_statistics(*args)
            exact = exact_ot_cost(*args, max_entries=cfg["exact_max_entries"]) if n*n <= cfg["exact_max_entries"] else None
            tags = {"case": case, "scenario": cfg["scenario"], "geometry": cfg["geometry"],
                    "n": n, "m": len(y), "d": d, "weights": weights, "seed": seed,
                    "sweep_value": value, "is_self": is_self, **overlap}
            functions, specifications = {}, {}
            for kind, count in itertools.product(cfg["projection_kinds"], cfg["projection_counts"]):
                key = (d, seed, kind)
                if key not in projection_cache:
                    theta_seed = int(np.random.SeedSequence([cfg["projection_seed"], d, seed]).generate_state(1)[0])
                    projection_cache[key] = make_projections(d, max(cfg["projection_counts"]), kind=kind, seed=theta_seed)
                    np.save(out/"projections"/f"d{d}_s{seed}_{kind}.npy", projection_cache[key])
                theta = projection_cache[key][:count]
                collisions = collision_statistics(*args, theta, fiber_tol=cfg["fiber_tol"])
                for name, solver in (("LMOT", solve_lmot), ("EST", solve_est)):
                    label = f"{name}/{kind}/L{count}"
                    functions[label] = partial(solver, *args, projections=theta,
                                               fiber_tol=cfg["fiber_tol"], mass_tol=cfg["mass_tol"])
                    specifications[label] = {"method": name, "projection_kind": kind, "L": count,
                                             "epsilon": None, **collisions}
            sink = dict(cfg["sinkhorn"])
            epsilons = sink.pop("epsilons")
            for epsilon in epsilons:
                label = f"Sinkhorn/eps={epsilon:g}"
                spec = {"method": "Sinkhorn", "projection_kind": "none", "L": 0,
                        "epsilon": epsilon, "backend": sink["backend"]}
                if sink["max_entries"] is not None and n*len(y) > sink["max_entries"]:
                    rows.append({**tags, **spec, "method_id": label, "status": "skipped_size",
                                 "note": "configured dense allocation cap, not measured OOM"})
                else:
                    functions[label] = partial(solve_sinkhorn, *args, epsilon=epsilon, **sink)
                    specifications[label] = spec
            timing, measurements = benchmark_functions(functions, repeats=cfg["repeats"], warmups=cfg["warmups"])
            raw.extend({**tags, **entry} for entry in measurements)
            for label, fn in functions.items():
                result, peak = evaluate_with_memory(fn, enabled=cfg["measure_memory"])
                quality = evaluate_quality(result, *args, i, j, overlap["common_mass"], exact_cost=exact,
                                           is_self=is_self, tolerance=cfg["feasibility_tol"])
                status = result.diagnostics.get("status", "ok")
                statuses = [r["status"] for r in measurements if r["method"] == label]
                if status == "ok" and any(s != "ok" for s in statuses):
                    status = "mixed_timing_status"
                if status == "ok" and not quality["feasible"]:
                    status = "invalid_marginals"
                row = {**tags, **specifications[label], "method_id": label, **timing[label],
                       **quality, "status": status, "peak_traced_mib": peak,
                       "iterations": result.diagnostics.get("iterations")}
                rows.append(row)
                del result
            # Checkpoint after each dataset; Sinkhorn is measured once per dataset,
            # NOT repeated for every projection bank/count that it does not use.
            save_table(out/"summary.csv", rows)
            save_table(out/"summary.tsv", rows, "\t")
            save_table(out/"raw_timings.csv", raw)
            print(case + ": " + "; ".join(
                f"{label} {timing[label]['median_ms']:.2f} ms" for label in functions), flush=True)
    (out/"DONE.json").write_text(json.dumps({"rows": len(rows), "timed_calls": len(raw)}))
    if make_plots:
        from .plot import plot_results
        plot_results(out)
    print(f"Results: {out.resolve()}", flush=True)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"experiments/simulation/configs/smoke.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    run(args.config, args.output_dir, make_plots=not args.no_plots)


if __name__ == "__main__":
    main()
