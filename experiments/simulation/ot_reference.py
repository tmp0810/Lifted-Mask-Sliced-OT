"""Untimed, unregularized OT reference for the simulation benchmark.

POT's ``ot.emd`` solves the transport linear program with a C++ network simplex
on CPU. Only the validated reference plan is copied to the prediction device.
This evaluator does not change any of the GPU transport methods.
"""
import math
import warnings

import numpy as np
import torch

from lmot.gpu.common import check_dense_size


@torch.no_grad()
def solve_ot_reference(x, alpha, y, beta, *, max_iter=1_000_000,
                       tolerance=1e-9, max_entries=4_000_000, num_threads=1):
    """Return (P_star, diagnostics), or (None, diagnostics) if LP validation fails.

    Inputs are the prepared torch measures used by the benchmark. ``tolerance``
    checks marginals and a primal/dual certificate; it is not a Sinkhorn stopThr.
    An LP may have multiple minimizers: RMSE refers to the one POT returns.
    """
    if not math.isfinite(tolerance) or tolerance <= 0 or min(max_iter, num_threads) < 1:
        raise ValueError("positive finite reference tolerance and positive LP budgets required")
    check_dense_size(len(x), len(y), max_entries)
    import ot

    X, a, Y, b = (np.ascontiguousarray(v.detach().cpu().numpy(), dtype=np.float64)
                  for v in (x, alpha, y, beta))
    # Match the GPU methods' squared-Euclidean cost, using direct differences.
    # Avoid an n*m*d temporary and cancellation in ||x||^2 + ||y||^2 - 2<x,y>.
    C = np.zeros((len(X), len(Y)), dtype=np.float64, order="C")
    for coordinate in range(X.shape[1]):
        difference = X[:, coordinate, None] - Y[None, :, coordinate]
        np.square(difference, out=difference)
        C += difference
    del difference

    diagnostics = {"reference_status": "solver_error", "reference_result_code": None,
                   "reference_warning": None, "reference_cost": None,
                   "reference_row_l1": None, "reference_col_l1": None,
                   "reference_dual_gap": None, "reference_dual_violation": None}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            P, log = ot.emd(a, b, C, numItermax=max_iter, numThreads=num_threads,
                            log=True, check_marginals=True)
    except (ValueError, RuntimeError) as error:
        diagnostics["reference_warning"] = str(error)
        return None, diagnostics

    code = int(log["result_code"])
    warning = log.get("warning")
    messages = [str(w.message) for w in caught]
    if warning:
        messages.append(str(warning))
    diagnostics.update(reference_result_code=code,
                       reference_warning="; ".join(dict.fromkeys(messages)) or None)
    if P.shape != C.shape or not np.isfinite(P).all() or np.min(P) < -1e-14:
        diagnostics["reference_status"] = "invalid_plan"
        return None, diagnostics

    row_error = float(np.abs(P.sum(axis=1) - a).sum())
    col_error = float(np.abs(P.sum(axis=0) - b).sum())
    primal = float(np.einsum("ij,ij->", P, C))
    diagnostics.update(reference_row_l1=row_error, reference_col_l1=col_error,
                       reference_cost=primal)
    # POT's network-simplex success code is 1. Feasible but unfinished plans
    # must never silently become ground truth.
    if code != 1 or warning:
        diagnostics["reference_status"] = "not_optimal"
        return None, diagnostics
    if max(row_error, col_error) > tolerance:
        diagnostics["reference_status"] = "invalid_marginals"
        return None, diagnostics

    u, v = np.asarray(log["u"]), np.asarray(log["v"])
    dual = float(a @ u + b @ v)
    gap = abs(primal - dual)
    violation = 0.0
    for start in range(0, len(X), 128):
        slack = u[start:start + 128, None] + v[None, :] - C[start:start + 128]
        violation = max(violation, float(slack.max()))
    diagnostics.update(reference_dual_gap=gap, reference_dual_violation=violation)
    cost_scale = max(1.0, float(np.max(np.abs(C))))
    objective_scale = max(1.0, abs(primal), abs(dual))
    if (not np.isfinite(u).all() or not np.isfinite(v).all()
            or not math.isfinite(gap) or gap > tolerance * objective_scale
            or violation > tolerance * cost_scale):
        diagnostics["reference_status"] = "invalid_optimality"
        return None, diagnostics

    diagnostics["reference_status"] = "ok"
    return torch.as_tensor(P, dtype=torch.float64, device=x.device), diagnostics
