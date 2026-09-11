"""Untimed evaluation: overlap, collisions, feasibility and exact-OT gap."""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import kron, eye, csr_matrix, vstack
from .common import make_fibers, squared_distances, check_dense_size
from .methods.lmot import find_exact_overlap


def overlap_statistics(x, alpha, y, beta):
    i, j = find_exact_overlap(x, y)
    common = float(np.minimum(alpha[i], beta[j]).sum())
    return i, j, {"overlap_count": len(i), "overlap_fraction": len(i)/min(len(x), len(y)),
                  "common_mass": common}


def collision_statistics(x, alpha, y, beta, projections, *, fiber_tol=1e-12):
    weighted, count_ratios, largest = [], [], []
    for theta in projections:
        for points, weights in ((x, alpha), (y, beta)):
            f = make_fibers(points, weights, theta, fiber_tol)
            counts = np.diff(f.starts)
            hit = counts[f.groups] > 1
            weighted.append(float(weights[hit].sum()))
            count_ratios.append(float(hit.mean()))
            largest.append(int(counts.max()))
    return {"collision_mass": float(np.mean(weighted)),
            "collision_fraction": float(np.mean(count_ratios)),
            "largest_fiber": max(largest)}


def exact_ot_cost(x, alpha, y, beta, *, max_entries=262144):
    """Unregularized LP solved by SciPy HiGHS; only for small reference cases."""
    n, m = len(x), len(y)
    check_dense_size(n, m, max_entries)
    C = squared_distances(x, y)
    row = kron(eye(n, format="csr"), csr_matrix(np.ones((1, m))), format="csr")
    col = kron(csr_matrix(np.ones((1, n))), eye(m, format="csr"), format="csr")
    constraints = vstack((row, col[:-1]), format="csr")
    result = linprog(C.ravel(), A_eq=constraints, b_eq=np.r_[alpha, beta[:-1]],
                     bounds=(0, None), method="highs",
                     options={"dual_feasibility_tolerance": 1e-9,
                              "primal_feasibility_tolerance": 1e-9})
    if not result.success:
        raise RuntimeError(f"exact OT failed: {result.message}")
    return float(result.fun)


def evaluate_quality(result, x, alpha, y, beta, i, j, common_mass,
                     *, exact_cost=None, is_self=False, tolerance=1e-8):
    rows, cols = result.plan.marginals()
    row_l1, col_l1 = float(np.abs(rows-alpha).sum()), float(np.abs(cols-beta).sum())
    finite = bool(np.isfinite(result.squared_cost) and np.isfinite(result.barycentric_map).all()
                  and np.isfinite(rows).all() and np.isfinite(cols).all())
    feasible = finite and max(row_l1, col_l1) <= tolerance
    retained = float(result.plan.entries(i, j).sum()) if len(i) else 0.0
    cost = float(result.squared_cost)
    gap = cost - exact_cost if feasible and exact_cost is not None else None
    relative = gap / exact_cost if gap is not None and exact_cost > 1e-12 else None
    return {"cost": cost, "row_l1": row_l1, "col_l1": col_l1,
            "row_relative": float(np.max(np.abs(rows-alpha)/alpha)), "feasible": feasible,
            "retained_mass": retained,
            "retention_ratio": retained/common_mass if common_mass > 0 else None,
            "exact_cost": exact_cost, "absolute_gap": gap, "relative_gap": relative,
            "self_cost": cost if is_self else None,
            "self_map_error": float(np.max(np.linalg.norm(result.barycentric_map-x, axis=1))) if is_self else None}
