"""Dense-plan runtime and RMSE benchmark; run from the repository root.

    python -m experiments.simulation.paper_results \
        --config experiments/simulation/configs/smoke.yaml

Reuses the existing simulation configs and solvers without changing them.
Outputs paper_results.csv/.tsv (mean and sample SD across seeds),
identity_results.csv/.tsv (self pairs, coordinate-aligned identity reference),
per_pair.csv and raw_timings.csv (including failures). Each pair's runtime is
the median of its timed repetitions; RMSE is sqrt(mean((P - P_gt)**2)).

Timing includes the existing solver (cost/map included) AND dense-plan export.
Data, projections, reference, RMSE and marginal checks are outside the timer.
P_gt is independently converged Sinkhorn with ONE fixed reference epsilon.
The default reference epsilon is 0.01 on this simulation's cost scale (C <= 1);
use --reference-epsilon 0.001 for a smaller, potentially harder reference.
Baseline epsilons/budgets come from the config; they do not change P_gt.
Unconverged references are flagged and never used to score RMSE. Failed runs
remain in detailed outputs; aggregate tables report valid/total pair counts.
The dense allocation cap applies to this entire benchmark, not to LMOT's
implicit scalability. Raw RMSE should be compared at the same matrix size.
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
import yaml
from threadpoolctl import threadpool_limits

from lmot import solve_est, solve_lmot, solve_sinkhorn
from lmot.metrics import collision_statistics, overlap_statistics
from lmot.plans import DensePlan
from lmot.projections import make_projections
from .data import make_pair
from .run import ROOT, metadata, save_table, validate_config


GROUP_COLUMNS = (
    "scenario", "geometry", "n", "m", "d", "weights", "sweep_value",
    "projection_kind", "L", "method", "epsilon", "reference_epsilon",
)


def plan_rmse(plan, reference):
    """Entrywise RMSE of probability couplings, without row normalization."""
    if plan.shape != reference.shape:
        raise ValueError("plan and reference must have identical shapes")
    return float(np.sqrt(np.mean((plan - reference) ** 2)))


def dense_prediction(solver, *, max_entries, batch_size):
    """All operations in this function are inside the prediction timer."""
    result = solver()
    # Sinkhorn already produced an owned dense matrix: no extra copy needed.
    if isinstance(result.plan, DensePlan):
        plan = result.plan.matrix
    else:
        plan = result.plan.to_dense(max_entries=max_entries, batch_size=batch_size)
    return plan, result.diagnostics


def inspect_plan(plan, alpha, beta, diagnostics, tolerance):
    """Untimed checks on the actual exported matrix, not an implicit proxy."""
    if plan.shape != (len(alpha), len(beta)):
        raise ValueError("solver returned an incorrectly shaped plan")
    finite = bool(np.isfinite(plan).all())
    row_l1 = float(np.abs(plan.sum(1) - alpha).sum()) if finite else None
    col_l1 = float(np.abs(plan.sum(0) - beta).sum()) if finite else None
    status = diagnostics.get("status", "ok")
    if not finite or plan.min() < -1e-14:
        status = "invalid_plan"
    elif status == "ok" and max(row_l1, col_l1) > tolerance:
        status = "invalid_marginals"
    return {"plan_status": status, "row_l1": row_l1, "col_l1": col_l1,
            "iterations": diagnostics.get("iterations")}


def identity_reference(x, alpha, y, beta, i, j):
    """Self coupling aligned by atom equality, including permuted supports."""
    if len(i) != len(x) or len(j) != len(y) or not np.allclose(
            alpha[i], beta[j], rtol=0, atol=1e-12):
        raise ValueError("identity reference requires equal measures")
    plan = np.zeros((len(x), len(y)))
    plan[i, j] = alpha[i]
    return plan


def mean_std(values):
    if not values:
        return None, None
    # A single seed cannot estimate between-pair variability: leave SD blank.
    return float(np.mean(values)), (float(np.std(values, ddof=1)) if len(values) > 1 else None)


def aggregate(rows, *, identity=False):
    groups = defaultdict(list)
    for row in rows:
        if not identity or row["is_self"]:
            groups[tuple(row[k] for k in GROUP_COLUMNS)].append(row)
    output = []
    metric = "identity_rmse" if identity else "plan_rmse"
    status_key = "plan_status" if identity else "rmse_status"
    for key, group in groups.items():
        valid = [r for r in group if r[status_key] == "ok" and r.get(metric) is not None]
        time_mean, time_std = mean_std([r["runtime_ms"] for r in valid])
        error_mean, error_std = mean_std([r[metric] for r in valid])
        row = dict(zip(GROUP_COLUMNS, key))
        row.update(n_pairs=len(group), n_valid=len(valid),
                   n_failed=len(group) - len(valid),
                   runtime_ms_mean=time_mean, runtime_ms_std=time_std)
        row[metric + "_mean"], row[metric + "_std"] = error_mean, error_std
        if identity:
            row["identity_max_error"] = max(
                (r["identity_max_error"] for r in valid), default=None)
        output.append(row)
    return output


def checkpoint(out, rows, raw):
    save_table(out / "per_pair.csv", rows)
    save_table(out / "raw_timings.csv", raw)
    for name, table in (("paper_results", aggregate(rows)),
                        ("identity_results", aggregate(rows, identity=True))):
        if table:
            save_table(out / f"{name}.csv", table)
            save_table(out / f"{name}.tsv", table, "\t")


def run(config_path, output_dir=None, *, reference_epsilon=1e-2,
        reference_tolerance=1e-9, reference_max_iter=50_000,
        max_entries=1_048_576, batch_size=128, repeats=None, warmups=None):
    cfg = yaml.safe_load(Path(config_path).read_text())
    if repeats is not None:
        cfg["repeats"] = repeats
    if warmups is not None:
        cfg["warmups"] = warmups
    values = validate_config(cfg)
    if (not np.isfinite([reference_epsilon, reference_tolerance]).all()
            or min(reference_epsilon, reference_tolerance) <= 0
            or min(reference_max_iter, max_entries, batch_size) < 1):
        raise ValueError("reference parameters and allocation settings must be positive")
    if not cfg["sinkhorn"]["epsilons"]:
        raise ValueError("configure at least one Sinkhorn baseline epsilon")
    # Respect both the existing config budget and this script's dense cap.
    cap = min(max_entries, cfg["sinkhorn"].get("max_entries") or max_entries)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out = Path(output_dir) if output_dir else ROOT / "results/simulation" / f"paper_{cfg['name']}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "projections").mkdir()
    if cfg["save_inputs"]:
        (out / "inputs").mkdir()
    settings = dict(reference_epsilon=reference_epsilon,
                    reference_tolerance=reference_tolerance,
                    reference_max_iter=reference_max_iter, max_entries=cap,
                    batch_size=batch_size)
    (out / "config.yaml").write_text(yaml.safe_dump({**cfg, "paper_evaluation": settings}, sort_keys=False))
    rows, raw, projection_cache = [], [], {}

    with threadpool_limits(limits=cfg["threads"]):
        info = metadata()
        info.update(output="existing solver (cost + map) plus full dense probability plan",
                    memory_metric="not measured in this script",
                    reference="converged entropic Sinkhorn, not exact unregularized OT",
                    timing="CPU wall clock; generation, projections, reference and metrics excluded",
                    aggregation="per-pair median time; across-seed mean and sample SD (ddof=1)",
                    rmse="sqrt(mean((P - P_gt)**2)); unnormalized matrix entries",
                    **settings)
        (out / "metadata.json").write_text(json.dumps(info, indent=2))

        for n, d, weights, seed, value in itertools.product(
                cfg["sizes"], cfg["dimensions"], cfg["weight_modes"], cfg["seeds"], values):
            args, is_self = make_pair(n, d, value=value, seed=seed, weights=weights,
                                     scenario=cfg["scenario"], geometry=cfg["geometry"])
            x, alpha, y, beta = args
            case = f"n{n}_d{d}_{weights}_s{seed}_v{value:g}"
            if cfg["save_inputs"]:
                np.savez_compressed(out / "inputs" / f"{case}.npz", X=x, alpha=alpha, Y=y, beta=beta)
            i, j, overlap = overlap_statistics(*args)
            tags = dict(case=case, scenario=cfg["scenario"], geometry=cfg["geometry"],
                        n=n, m=len(y), d=d, weights=weights, seed=seed,
                        sweep_value=value, is_self=is_self,
                        reference_epsilon=reference_epsilon, **overlap)
            too_large = n * len(y) > cap
            functions, specs = {}, {}
            for kind, count in itertools.product(cfg["projection_kinds"], cfg["projection_counts"]):
                key = (d, seed, kind)
                if key not in projection_cache:
                    projection_seed = int(np.random.SeedSequence(
                        [cfg["projection_seed"], d, seed]).generate_state(1)[0])
                    projection_cache[key] = make_projections(
                        d, max(cfg["projection_counts"]), kind=kind, seed=projection_seed)
                    np.save(out / "projections" / f"d{d}_s{seed}_{kind}.npy", projection_cache[key])
                theta = projection_cache[key][:count]
                collisions = {} if too_large else collision_statistics(
                    *args, theta, fiber_tol=cfg["fiber_tol"])
                for name, solver in (("LMOT", solve_lmot), ("EST", solve_est)):
                    label = f"{name}/{kind}/L{count}"
                    functions[label] = partial(solver, *args, projections=theta,
                                              fiber_tol=cfg["fiber_tol"], mass_tol=cfg["mass_tol"])
                    specs[label] = dict(method=name, projection_kind=kind, L=count,
                                        epsilon=None, **collisions)
            sink = dict(cfg["sinkhorn"])
            epsilons = sink.pop("epsilons")
            sink["max_entries"] = cap
            for epsilon in epsilons:
                label = f"Sinkhorn/eps={epsilon:g}"
                functions[label] = partial(solve_sinkhorn, *args, epsilon=epsilon, **sink)
                specs[label] = dict(method="Sinkhorn", projection_kind="none", L=0,
                                    epsilon=epsilon, backend=sink["backend"])

            if too_large:
                for label, spec in specs.items():
                    rows.append({**tags, **spec, "method_id": label,
                                 "plan_status": "skipped_size", "rmse_status": "skipped_size",
                                 "reference_status": "skipped_size"})
                checkpoint(out, rows, raw)
                print(f"{case}: skipped_size ({n * len(y):,} entries > {cap:,})", flush=True)
                continue

            # Exactly one independent reference per pair, before any timing.
            gt_result = solve_sinkhorn(
                *args, epsilon=reference_epsilon, tolerance=reference_tolerance,
                max_iter=reference_max_iter, max_entries=cap, backend=sink["backend"],
                check_every=sink.get("check_every", 10), max_seconds=None)
            gt = gt_result.plan.matrix
            gt_check = inspect_plan(gt, alpha, beta, gt_result.diagnostics, reference_tolerance)
            reference_ok = gt_check["plan_status"] == "ok"
            tags.update(reference_status=gt_check["plan_status"],
                        reference_iterations=gt_check["iterations"],
                        reference_row_l1=gt_check["row_l1"], reference_col_l1=gt_check["col_l1"])
            identity = identity_reference(x, alpha, y, beta, i, j) if is_self else None

            for _ in range(cfg["warmups"]):
                for fn in functions.values():
                    p, diagnostics = dense_prediction(fn, max_entries=cap, batch_size=batch_size)
                    del p, diagnostics

            measurements = defaultdict(list)
            labels = list(functions)
            for repeat in range(cfg["repeats"]):
                # Rotate order to reduce systematic timing-order effects.
                offset = repeat % len(labels)
                for label in labels[offset:] + labels[:offset]:
                    start = time.perf_counter()
                    p, diagnostics = dense_prediction(
                        functions[label], max_entries=cap, batch_size=batch_size)
                    elapsed_ms = 1e3 * (time.perf_counter() - start)
                    check = inspect_plan(p, alpha, beta, diagnostics, cfg["feasibility_tol"])
                    valid = check["plan_status"] == "ok"
                    entry = {**tags, **specs[label], "method_id": label, "repeat": repeat,
                             "runtime_ms": elapsed_ms, **check,
                             "rmse_status": (check["plan_status"] if not valid else
                                             "ok" if reference_ok else "reference_failed"),
                             "plan_rmse": plan_rmse(p, gt) if valid and reference_ok else None,
                             "identity_rmse": plan_rmse(p, identity) if valid and is_self else None,
                             "identity_max_error": float(np.max(np.abs(p - identity)))
                             if valid and is_self else None}
                    measurements[label].append(entry)
                    raw.append(entry)
                    del p, diagnostics
            for label, calls in measurements.items():
                row = {k: v for k, v in calls[0].items() if k != "repeat"}
                row["runtime_ms"] = float(np.median([c["runtime_ms"] for c in calls]))
                for status in ("plan_status", "rmse_status"):
                    states = {c[status] for c in calls}
                    row[status] = next(iter(states)) if len(states) == 1 else "mixed_status"
                for metric in ("plan_rmse", "identity_rmse", "identity_max_error"):
                    available = [c[metric] for c in calls if c[metric] is not None]
                    row[metric] = float(np.mean(available)) if len(available) == len(calls) else None
                if row["identity_max_error"] is not None:
                    row["identity_max_error"] = max(c["identity_max_error"] for c in calls)
                row["row_l1"] = max((c["row_l1"] for c in calls if c["row_l1"] is not None), default=None)
                row["col_l1"] = max((c["col_l1"] for c in calls if c["col_l1"] is not None), default=None)
                rows.append(row)
            checkpoint(out, rows, raw)
            print(f"{case}: reference={tags['reference_status']}; " + "; ".join(
                f"{label}: {np.median([c['runtime_ms'] for c in calls]):.2f} ms"
                for label, calls in measurements.items()), flush=True)
            del gt, gt_result, identity

    (out / "DONE.json").write_text(json.dumps({"pairs_methods": len(rows), "timed_calls": len(raw)}))
    print(f"Results: {out.resolve()}", flush=True)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/simulation/configs/smoke.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reference-epsilon", type=float, default=1e-2)
    parser.add_argument("--reference-tolerance", type=float, default=1e-9)
    parser.add_argument("--reference-max-iter", type=int, default=50_000)
    parser.add_argument("--max-entries", type=int, default=1_048_576)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--warmups", type=int)
    args = parser.parse_args(argv)
    run(args.config, args.output_dir, reference_epsilon=args.reference_epsilon,
        reference_tolerance=args.reference_tolerance, reference_max_iter=args.reference_max_iter,
        max_entries=args.max_entries, batch_size=args.batch_size,
        repeats=args.repeats, warmups=args.warmups)


if __name__ == "__main__":
    main()
