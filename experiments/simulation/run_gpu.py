"""CUDA simulation: implicit cost/map benchmark, or full dense-plan benchmark.

Use paper_results_gpu for the usual dense runtime + plan RMSE table. This
module defaults to implicit outputs for scaling. CUDA is required unless
--device cpu is explicitly selected for parity/debugging. All transports,
reference plans, masks, sorting and RMSE use float64 torch tensors.

Inputs and directions are generated as before, then uploaded before timing.
Each timed call synchronizes CUDA before t0 AND before stopping the wall clock.
Output remains on the GPU. Input/output transfers, reference solves, metrics
and CSV I/O are excluded. Dense mode includes complete matrix construction;
both modes also include the existing cost/map evaluation. The two timing
protocols are labeled separately and must not be pooled in one comparison.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from functools import partial
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml
from threadpoolctl import threadpool_limits

from lmot.gpu import solve_lmot, solve_est, solve_sinkhorn, resolve_device, synchronize
from lmot.gpu.common import tensor
from lmot.gpu.metrics import (plan_rmse, overlap_statistics, collision_statistics,
                             identity_reference, inspect_plan, evaluate_result)
from lmot.gpu.plans import DensePlan
from lmot.projections import make_projections
from .data import make_pair
from .utils import ROOT, metadata, save_table, validate_config


GROUP_COLUMNS = ("scenario", "geometry", "n", "m", "d", "weights", "sweep_value",
                 "projection_kind", "L", "method", "epsilon", "reference_epsilon",
                 "device", "dtype", "output_mode", "backend")


def mean_std(values):
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else None


def aggregate(rows, *, identity=False, output_mode="dense"):
    groups = defaultdict(list)
    for row in rows:
        if not identity or row["is_self"]:
            groups[tuple(row[k] for k in GROUP_COLUMNS)].append(row)
    table = []
    metric = "identity_rmse" if identity else "plan_rmse" if output_mode == "dense" else "cost"
    status_key = "rmse_status" if not identity and output_mode == "dense" else "plan_status"
    for key, group in groups.items():
        valid = [r for r in group if r[status_key] == "ok" and r.get(metric) is not None]
        row = dict(zip(GROUP_COLUMNS, key))
        row.update(n_pairs=len(group), n_valid=len(valid), n_failed=len(group)-len(valid))
        row["runtime_ms_mean"], row["runtime_ms_std"] = mean_std([r["runtime_ms"] for r in valid])
        row[metric + "_mean"], row[metric + "_std"] = mean_std([r[metric] for r in valid])
        if identity:
            row["identity_max_error"] = max((r["identity_max_error"] for r in valid), default=None)
        table.append(row)
    return table


def checkpoint(out, rows, raw, output_mode):
    # Atomic replacement avoids a partially written CSV if a Colab run stops.
    def save(name, values, delimiter=","):
        if values:
            temporary = out / (name + ".tmp")
            save_table(temporary, values, delimiter)
            temporary.replace(out / name)
    save("per_pair.csv", rows)
    save("raw_timings.csv", raw)
    name = "paper_results" if output_mode == "dense" else "summary"
    table = aggregate(rows, output_mode=output_mode)
    save(name + ".csv", table)
    save(name + ".tsv", table, "\t")
    if output_mode == "dense":
        identities = aggregate(rows, identity=True)
        save("identity_results.csv", identities)
        save("identity_results.tsv", identities, "\t")


def prediction(solver, output_mode, max_entries, batch_size):
    result = solver()
    plan = None
    if output_mode == "dense":
        plan = result.plan.matrix if isinstance(result.plan, DensePlan) else result.plan.to_dense(
            max_entries=max_entries, batch_size=batch_size)
    return result, plan


def timed_prediction(solver, *, device, output_mode, max_entries, batch_size, measure_memory=False):
    """Synchronized end-to-end wall time for already device-resident inputs."""
    synchronize(device)
    baseline = None
    if measure_memory and device.type == "cuda":
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    result, plan = prediction(solver, output_mode, max_entries, batch_size)
    synchronize(device)
    elapsed = 1e3 * (time.perf_counter() - start)
    peak = ((torch.cuda.max_memory_allocated(device) - baseline) / 1024**2
            if baseline is not None else None)
    return result, plan, elapsed, peak


@torch.no_grad()
def run(config_path, output_dir=None, *, device="cuda", output_mode="implicit",
        reference_epsilon=1e-3, reference_tolerance=1e-9, reference_max_iter=50_000,
        max_entries=1_048_576, batch_size=128, repeats=None, warmups=None,
        sinkhorn_backend=None):
    device = resolve_device(device)  # Fail before producing mislabeled CPU results.
    if output_mode not in ("dense", "implicit"):
        raise ValueError("output_mode must be dense or implicit")
    cfg = yaml.safe_load(Path(config_path).read_text())
    if repeats is not None:
        cfg["repeats"] = repeats
    if warmups is not None:
        cfg["warmups"] = warmups
    if sinkhorn_backend is not None:
        cfg["sinkhorn"]["backend"] = sinkhorn_backend
    if cfg["sinkhorn"]["backend"] not in ("pot", "torch_log"):
        raise ValueError("GPU supports pot or torch_log; pass --sinkhorn-backend pot")
    values = validate_config(cfg)
    if (not np.isfinite([reference_epsilon, reference_tolerance]).all()
            or min(reference_epsilon, reference_tolerance) <= 0
            or min(reference_max_iter, max_entries, batch_size) < 1):
        raise ValueError("invalid reference/allocation settings")
    if not cfg["sinkhorn"]["epsilons"]:
        raise ValueError("at least one Sinkhorn epsilon is required")
    cap = min(max_entries, cfg["sinkhorn"].get("max_entries") or max_entries)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    prefix = "paper_gpu" if output_mode == "dense" else "implicit_gpu"
    out = Path(output_dir) if output_dir else ROOT / "results/simulation" / f"{prefix}_{cfg['name']}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "projections").mkdir()
    if cfg["save_inputs"]:
        (out / "inputs").mkdir()
    settings = dict(device=str(device), dtype="float64", output_mode=output_mode,
                    reference_epsilon=reference_epsilon, reference_tolerance=reference_tolerance,
                    reference_max_iter=reference_max_iter, max_entries=cap, batch_size=batch_size)
    (out / "config.yaml").write_text(yaml.safe_dump({**cfg, "gpu_evaluation": settings}, sort_keys=False))
    rows, raw, projection_cache = [], [], {}
    old_threads = torch.get_num_threads()
    torch.set_num_threads(cfg["threads"])
    try:
        with threadpool_limits(limits=cfg["threads"]):
            info = metadata()
            info.update(**settings, torch_version=torch.__version__, cuda_version=torch.version.cuda,
                        gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                        gpu_capability=list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None,
                        gpu_total_memory_bytes=torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None,
                        output="cost + barycentric map + " + ("full GPU matrix" if output_mode == "dense" else "implicit plan"),
                        memory_metric="peak extra torch CUDA allocated MiB inside prediction; not reserved memory or RSS",
                        reference="converged entropic Sinkhorn; unused in implicit mode",
                        timing="wall clock synchronized before/after; H2D, D2H, reference, metrics excluded",
                        aggregation="per-pair median runtime; mean/sample SD across seeds; single-seed SD blank",
                        rmse="sqrt(mean((P - P_gt)**2)) on the device; no row normalization")
            (out / "metadata.json").write_text(json.dumps(info, indent=2))
            for n, d, weights, seed, value in itertools.product(
                    cfg["sizes"], cfg["dimensions"], cfg["weight_modes"], cfg["seeds"], values):
                numpy_args, is_self = make_pair(n, d, value=value, seed=seed, weights=weights,
                                               scenario=cfg["scenario"], geometry=cfg["geometry"])
                args = tuple(tensor(v, device) for v in numpy_args)
                x, alpha, y, beta = args
                case = f"n{n}_d{d}_{weights}_s{seed}_v{value:g}"
                if cfg["save_inputs"]:
                    np.savez_compressed(out / "inputs" / f"{case}.npz",
                                        X=numpy_args[0], alpha=numpy_args[1], Y=numpy_args[2], beta=numpy_args[3])
                i, j, overlap = overlap_statistics(*args)
                tags = dict(case=case, scenario=cfg["scenario"], geometry=cfg["geometry"], n=n,
                            m=len(y), d=d, weights=weights, seed=seed, sweep_value=value, is_self=is_self,
                            reference_epsilon=reference_epsilon if output_mode == "dense" else None,
                            device=str(device), dtype="float64", output_mode=output_mode, **overlap)
                too_large = n * len(y) > cap
                functions, specs = {}, {}
                for kind, count in itertools.product(cfg["projection_kinds"], cfg["projection_counts"]):
                    key = (d, seed, kind)
                    if key not in projection_cache:
                        theta_seed = int(np.random.SeedSequence(
                            [cfg["projection_seed"], d, seed]).generate_state(1)[0])
                        bank = make_projections(d, max(cfg["projection_counts"]), kind=kind, seed=theta_seed)
                        np.save(out / "projections" / f"d{d}_s{seed}_{kind}.npy", bank)
                        projection_cache[key] = tensor(bank, device)
                    theta = projection_cache[key][:count]
                    collisions = {} if too_large and output_mode == "dense" else collision_statistics(
                        *args, theta, fiber_tol=cfg["fiber_tol"])
                    for name, solver in (("LMOT", solve_lmot), ("EST", solve_est)):
                        label = f"{name}/{kind}/L{count}"
                        functions[label] = partial(solver, *args, projections=theta, device=device,
                                                  fiber_tol=cfg["fiber_tol"], mass_tol=cfg["mass_tol"])
                        specs[label] = dict(method=name, projection_kind=kind, L=count,
                                            epsilon=None, backend="torch", **collisions)
                sink = dict(cfg["sinkhorn"])
                epsilons = sink.pop("epsilons")
                sink["max_entries"] = cap
                for epsilon in epsilons:
                    label = f"Sinkhorn/eps={epsilon:g}"
                    functions[label] = partial(solve_sinkhorn, *args, epsilon=epsilon, device=device, **sink)
                    specs[label] = dict(method="Sinkhorn", projection_kind="none", L=0,
                                        epsilon=epsilon, backend=sink["backend"])
                for label in list(functions):
                    if too_large and (output_mode == "dense" or specs[label]["method"] == "Sinkhorn"):
                        rows.append({**tags, **specs[label], "method_id": label, "plan_status": "skipped_size",
                                     "rmse_status": "skipped_size", "reference_status": "skipped_size"})
                        del functions[label]
                if not functions:
                    checkpoint(out, rows, raw, output_mode)
                    print(f"{case}: skipped_size ({n * len(y):,} entries > {cap:,})", flush=True)
                    continue

                gt, identity = None, None
                reference_ok = False
                tags["reference_status"] = "not_computed_implicit"
                if output_mode == "dense":
                    gt_result = solve_sinkhorn(*args, device=device, epsilon=reference_epsilon,
                        tolerance=reference_tolerance, max_iter=reference_max_iter, max_entries=cap,
                        backend=sink["backend"], check_every=sink.get("check_every", 10), max_seconds=None)
                    gt = gt_result.plan.matrix
                    gt_check = inspect_plan(gt, alpha, beta, gt_result.diagnostics, reference_tolerance)
                    reference_ok = gt_check["plan_status"] == "ok"
                    tags.update(reference_status=gt_check["plan_status"],
                                reference_iterations=gt_check["iterations"],
                                reference_row_l1=gt_check["row_l1"], reference_col_l1=gt_check["col_l1"])
                    del gt_result
                    if is_self:
                        identity = identity_reference(*args, i, j)
                for _ in range(cfg["warmups"]):
                    for fn in functions.values():
                        result, p = prediction(fn, output_mode, cap, batch_size)
                        synchronize(device)
                        del result, p

                measurements = defaultdict(list)
                labels = list(functions)
                for repeat in range(cfg["repeats"]):
                    offset = repeat % len(labels)
                    for label in labels[offset:] + labels[:offset]:
                        result, p, elapsed, peak = timed_prediction(functions[label], device=device,
                            output_mode=output_mode, max_entries=cap, batch_size=batch_size,
                            measure_memory=cfg["measure_memory"])
                        quality = evaluate_result(result, args, i, j, overlap["common_mass"], dense=p,
                                                  is_self=is_self, tolerance=cfg["feasibility_tol"])
                        valid = quality["plan_status"] == "ok"
                        rmse_status = quality["plan_status"] if not valid else (
                            "not_computed_implicit" if output_mode == "implicit" else
                            "ok" if reference_ok else "reference_failed")
                        entry = {**tags, **specs[label], "method_id": label, "repeat": repeat,
                                 "runtime_ms": elapsed, "peak_extra_cuda_mib": peak, **quality,
                                 "rmse_status": rmse_status,
                                 "plan_rmse": plan_rmse(p, gt) if rmse_status == "ok" else None,
                                 "identity_rmse": plan_rmse(p, identity) if valid and identity is not None else None,
                                 "identity_max_error": float((p - identity).abs().max())
                                 if valid and identity is not None else None}
                        measurements[label].append(entry)
                        raw.append(entry)
                        del result, p
                for label, calls in measurements.items():
                    row = {k: v for k, v in calls[0].items() if k != "repeat"}
                    row["runtime_ms"] = float(np.median([c["runtime_ms"] for c in calls]))
                    for status in ("plan_status", "rmse_status"):
                        states = {c[status] for c in calls}
                        row[status] = next(iter(states)) if len(states) == 1 else "mixed_status"
                    for metric in ("cost", "plan_rmse", "identity_rmse"):
                        available = [c[metric] for c in calls if c[metric] is not None]
                        row[metric] = float(np.mean(available)) if len(available) == len(calls) else None
                    for metric in ("identity_max_error", "row_l1", "col_l1", "row_relative",
                                   "self_map_error", "peak_extra_cuda_mib"):
                        row[metric] = max((c[metric] for c in calls if c[metric] is not None), default=None)
                    rows.append(row)
                checkpoint(out, rows, raw, output_mode)
                print(f"{case}: reference={tags['reference_status']}; " + "; ".join(
                    f"{label}: {np.median([c['runtime_ms'] for c in calls]):.2f} ms"
                    for label, calls in measurements.items()), flush=True)
                del gt, identity
    finally:
        torch.set_num_threads(old_threads)
    (out / "DONE.json").write_text(json.dumps({"pairs_methods": len(rows), "timed_calls": len(raw)}))
    print(f"Results: {out.resolve()}", flush=True)
    return out


def main(argv=None, *, default_mode="implicit"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/simulation/configs/smoke.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-mode", choices=("dense", "implicit"), default=default_mode)
    parser.add_argument("--reference-epsilon", type=float, default=1e-3)
    parser.add_argument("--reference-tolerance", type=float, default=1e-9)
    parser.add_argument("--reference-max-iter", type=int, default=50_000)
    parser.add_argument("--sinkhorn-backend", choices=("pot", "torch_log"))
    parser.add_argument("--max-entries", type=int, default=1_048_576)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--warmups", type=int)
    args = vars(parser.parse_args(argv))
    args["config_path"] = args.pop("config")
    run(**args)


if __name__ == "__main__":
    main()
