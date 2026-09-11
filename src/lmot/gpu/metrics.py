"""Untimed GPU evaluation; only small final statistics leave the device."""
import torch
from .common import find_exact_overlap, make_fibers


def plan_rmse(plan, reference):
    if plan.shape != reference.shape or plan.device != reference.device:
        raise ValueError("plans must share shape and device")
    return float(torch.sqrt(torch.mean((plan - reference).square())))


def overlap_statistics(x, alpha, y, beta):
    i, j = find_exact_overlap(x, y)
    return i, j, {"overlap_count": len(i), "overlap_fraction": len(i) / min(len(x), len(y)),
                  "common_mass": float(torch.minimum(alpha[i], beta[j]).sum())}


def collision_statistics(x, alpha, y, beta, projections, *, fiber_tol=1e-12):
    weighted, fractions, largest = [], [], []
    for theta in projections:
        theta = theta / torch.linalg.vector_norm(theta)
        for points, weights in ((x, alpha), (y, beta)):
            f = make_fibers(points, weights, theta, fiber_tol)
            counts = f.starts[1:] - f.starts[:-1]
            hit = counts[f.groups] > 1
            weighted.append(weights[hit].sum())
            fractions.append(hit.to(points.dtype).mean())
            largest.append(counts.max())
    return {"collision_mass": float(torch.stack(weighted).mean()),
            "collision_fraction": float(torch.stack(fractions).mean()),
            "largest_fiber": int(torch.stack(largest).max())}


def identity_reference(x, alpha, y, beta, i, j):
    if len(i) != len(x) or len(j) != len(y) or not torch.allclose(alpha[i], beta[j], atol=1e-12, rtol=0):
        raise ValueError("identity reference requires equal measures")
    plan = x.new_zeros((len(x), len(y)))
    plan[i, j] = alpha[i]
    return plan


def inspect_plan(plan, alpha, beta, diagnostics, tolerance):
    if plan.shape != (len(alpha), len(beta)) or plan.device != alpha.device:
        raise ValueError("plan must have the correct shape and device")
    finite = bool(torch.isfinite(plan).all())
    row_l1 = float((plan.sum(1) - alpha).abs().sum()) if finite else None
    col_l1 = float((plan.sum(0) - beta).abs().sum()) if finite else None
    status = diagnostics.get("status", "ok")
    if not finite or bool(plan.min() < -1e-14):
        status = "invalid_plan"
    elif status == "ok" and max(row_l1, col_l1) > tolerance:
        status = "invalid_marginals"
    return {"plan_status": status, "row_l1": row_l1, "col_l1": col_l1,
            "iterations": diagnostics.get("iterations")}


def evaluate_result(result, args, i, j, common_mass, *, dense=None, is_self=False, tolerance=1e-8):
    x, alpha, y, beta = args
    if result.barycentric_map.device != x.device or result.squared_cost.device != x.device:
        raise RuntimeError("solver output is on the wrong device")
    if dense is not None:
        check = inspect_plan(dense, alpha, beta, result.diagnostics, tolerance)
        rows = dense.sum(1)
        retained = float(dense[i, j].sum())
    else:
        rows, cols = result.plan.marginals()
        a, b = float((rows - alpha).abs().sum()), float((cols - beta).abs().sum())
        status = result.diagnostics.get("status", "ok")
        if not bool(torch.isfinite(rows).all() & torch.isfinite(cols).all()):
            status = "invalid_plan"
        elif status == "ok" and max(a, b) > tolerance:
            status = "invalid_marginals"
        check = {"plan_status": status, "row_l1": a, "col_l1": b,
                 "iterations": result.diagnostics.get("iterations")}
        retained = float(result.plan.entries(i, j).sum())
    if not bool(torch.isfinite(result.squared_cost) & torch.isfinite(result.barycentric_map).all()):
        check["plan_status"] = "invalid_plan"
    return {**check, "cost": float(result.squared_cost),
            "row_relative": float(((rows - alpha).abs() / alpha).max()),
            "retained_mass": retained,
            "retention_ratio": retained / common_mass if common_mass > 0 else None,
            "self_cost": float(result.squared_cost) if is_self else None,
            "self_map_error": float(torch.linalg.vector_norm(result.barycentric_map - x, dim=1).max())
            if is_self else None}
