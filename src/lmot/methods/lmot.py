"""Exact zero-cost mask, found once by vectorized binary search."""
import numpy as np
from ..common import _records, prepare_measures, prepare_projections
from ..plans import SlicedPlan

def find_exact_overlap(x, y):
    """Find ground-space equality once: sort target tuples, then binary search.

    Input supports must contain distinct atoms. All coordinates are compared;
    equal scalar projections alone do not imply that two atoms are identical.
    Returns source/target indices of the matching pairs, not an n-by-m mask.
    """
    target_order = np.argsort(_records(y), kind="stable")
    sorted_target = _records(y)[target_order]
    positions = np.searchsorted(sorted_target, _records(x))
    source_i = np.flatnonzero(positions < len(y))
    target_j = target_order[positions[source_i]]
    equal = np.all(x[source_i] == y[target_j], axis=1)
    return source_i[equal], target_j[equal]


def solve_lmot(x, alpha, y, beta, *, projections, fiber_tol=1e-12, mass_tol=1e-12):
    """Return cost, barycentric map and an implicit maximal-common-mass plan."""
    x, alpha, y, beta = prepare_measures(x, alpha, y, beta)
    theta = prepare_projections(projections, x.shape[1])
    i, j = find_exact_overlap(x, y)
    return SlicedPlan(x, alpha, y, beta, theta, i, j, method="LMOT",
                      fiber_tol=fiber_tol, mass_tol=mass_tol).evaluate()
