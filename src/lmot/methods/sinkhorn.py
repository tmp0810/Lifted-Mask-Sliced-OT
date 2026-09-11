"""Balanced entropic OT in log space, with explicit convergence diagnostics.

Default backend: POT ot.sinkhorn(method='sinkhorn_log'). An independent
backend='scipy_log' is also supplied. Both return the ground cost, excluding
entropy. This module never debiases the cost or forces a self-plan to identity.
"""
import time
import numpy as np
from scipy.special import logsumexp
from ..common import prepare_measures, squared_distances, check_dense_size
from ..plans import DensePlan, TransportResult


def solve_sinkhorn(x, alpha, y, beta, *, epsilon=1e-3, max_iter=5000,
                   tolerance=1e-8, check_every=10, max_entries=4_000_000,
                   max_seconds=None, backend="pot"):
    x, alpha, y, beta = prepare_measures(x, alpha, y, beta)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if max_iter < 1 or check_every < 1 or tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError("invalid iteration or tolerance settings")
    if max_seconds is not None and (max_seconds <= 0 or not np.isfinite(max_seconds)):
        raise ValueError("max_seconds must be finite and positive")
    if backend not in ("scipy_log", "pot"):
        raise ValueError("backend must be scipy_log or pot")
    check_dense_size(len(x), len(y), max_entries)
    start = time.perf_counter()
    C = squared_distances(x, y)
    timed_out = False
    if backend == "pot":
        if max_seconds is not None:
            raise ValueError("POT backend uses max_iter; set max_seconds=null")
        import ot
        P, log = ot.sinkhorn(alpha, beta, C, epsilon, method="sinkhorn_log",
                             numItermax=max_iter, stopThr=tolerance / np.sqrt(max(len(alpha), len(beta))),
                             log=True, warn=False)
        iterations = int(log.get("niter", max_iter-1)) + 1
    else:
        kernel = -C / epsilon
        log_alpha, log_beta = np.log(alpha), np.log(beta)
        u, v = np.zeros_like(alpha), np.zeros_like(beta)
        for iteration in range(1, max_iter + 1):
            u = log_alpha - logsumexp(kernel + v[None, :], axis=1)
            v = log_beta - logsumexp(kernel + u[:, None], axis=0)
            if iteration % check_every == 0 or iteration == max_iter:
                P = np.exp(kernel + u[:, None] + v[None, :])
                residual = max(np.abs(P.sum(1)-alpha).sum(), np.abs(P.sum(0)-beta).sum())
                if residual <= tolerance:
                    break
                if max_seconds is not None and time.perf_counter()-start >= max_seconds:
                    timed_out = True
                    break
        iterations = iteration
    row_error = float(np.abs(P.sum(1)-alpha).sum())
    col_error = float(np.abs(P.sum(0)-beta).sum())
    finite = bool(np.isfinite(P).all())
    converged = finite and max(row_error, col_error) <= tolerance
    status = "ok" if converged else ("timeout" if timed_out else "not_converged")
    origin = x[0]
    bary = (P @ (y-origin) + P.sum(1)[:, None]*origin) / alpha[:, None]
    return TransportResult(float(np.sum(P*C)), bary, DensePlan(P), {
        "method": "Sinkhorn", "backend": backend, "epsilon": epsilon,
        "iterations": iterations, "converged": converged, "status": status,
        "row_l1": row_error, "col_l1": col_error,
    })
