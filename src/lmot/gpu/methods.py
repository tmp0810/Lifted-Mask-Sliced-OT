"""GPU solvers; import with `from lmot.gpu import solve_lmot, ...`."""
import math
import time
import torch

from .common import (prepare_measures, prepare_projections, find_exact_overlap,
                     squared_distances, synchronize)
from .plans import DensePlan, SlicedPlan, TransportResult


@torch.no_grad()
def solve_lmot(x, alpha, y, beta, *, projections, device="cuda", fiber_tol=1e-12, mass_tol=1e-12):
    x, alpha, y, beta = prepare_measures(x, alpha, y, beta, device=device)
    theta = prepare_projections(projections, x.shape[1], x.device)
    i, j = find_exact_overlap(x, y)  # Once per solve, inside the benchmark timer.
    return SlicedPlan(x, alpha, y, beta, theta, i, j, method="LMOT",
                      fiber_tol=fiber_tol, mass_tol=mass_tol).evaluate()


@torch.no_grad()
def solve_est(x, alpha, y, beta, *, projections, device="cuda", fiber_tol=1e-12, mass_tol=1e-12):
    x, alpha, y, beta = prepare_measures(x, alpha, y, beta, device=device)
    theta = prepare_projections(projections, x.shape[1], x.device)
    empty = torch.empty(0, dtype=torch.long, device=x.device)
    return SlicedPlan(x, alpha, y, beta, theta, empty, empty, method="EST",
                      fiber_tol=fiber_tol, mass_tol=mass_tol).evaluate()


@torch.no_grad()
def solve_sinkhorn(x, alpha, y, beta, *, epsilon=1e-3, device="cuda", backend="pot",
                   max_iter=5000, tolerance=1e-8, check_every=10,
                   max_entries=4_000_000, max_seconds=None):
    x, alpha, y, beta = prepare_measures(x, alpha, y, beta, device=device)
    if not all(map(math.isfinite, (epsilon, tolerance))) or min(epsilon, tolerance) <= 0:
        raise ValueError("epsilon and tolerance must be finite and positive")
    if min(max_iter, check_every) < 1:
        raise ValueError("iteration settings must be positive")
    if backend not in ("pot", "torch_log"):
        raise ValueError("GPU Sinkhorn backend must be pot or torch_log")
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be positive")
    if backend == "pot" and max_seconds is not None:
        raise ValueError("POT uses max_iter as its budget; set max_seconds=null")
    C = squared_distances(x, y, max_entries)
    timed_out = False
    if backend == "pot":
        import ot
        P, log = ot.sinkhorn(alpha, beta, C, epsilon, method="sinkhorn_log",
                             numItermax=max_iter, stopThr=tolerance / math.sqrt(max(len(x), len(y))),
                             log=True, warn=False)
        iterations = int(log.get("niter", max_iter - 1)) + 1
    else:
        log_kernel = -C / epsilon
        u, v = torch.zeros_like(alpha), torch.zeros_like(beta)
        log_alpha, log_beta = alpha.log(), beta.log()
        synchronize(x.device)
        started = time.perf_counter()
        for iteration in range(1, max_iter + 1):
            u = log_alpha - torch.logsumexp(log_kernel + v[None, :], dim=1)
            v = log_beta - torch.logsumexp(log_kernel + u[:, None], dim=0)
            if iteration % check_every == 0 or iteration == max_iter:
                P = (log_kernel + u[:, None] + v[None, :]).exp()
                residual = torch.maximum((P.sum(1)-alpha).abs().sum(), (P.sum(0)-beta).abs().sum())
                if bool(residual <= tolerance):
                    break
                if max_seconds is not None and time.perf_counter() - started >= max_seconds:
                    timed_out = True
                    break
        iterations = iteration
    if not isinstance(P, torch.Tensor) or P.device != x.device:
        raise RuntimeError("Sinkhorn returned a plan on the wrong device")
    row_error, col_error = (P.sum(1) - alpha).abs().sum(), (P.sum(0) - beta).abs().sum()
    converged = bool(torch.isfinite(P).all() & (torch.maximum(row_error, col_error) <= tolerance))
    origin = x[0]
    bary = (P @ (y - origin) + P.sum(1)[:, None] * origin) / alpha[:, None]
    return TransportResult((P * C).sum(), bary, DensePlan(P), {
        "method": "Sinkhorn", "backend": backend, "device": str(x.device),
        "epsilon": epsilon, "iterations": iterations, "converged": converged,
        "status": "ok" if converged else "timeout" if timed_out else "not_converged",
        "row_l1": float(row_error), "col_l1": float(col_error)})
